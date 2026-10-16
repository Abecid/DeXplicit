"""Structure/contact supervision for the controlled long-horizon stages.

Research provenance
-------------------
* Stage 2 is motivated by ParticleFormer (arXiv:2506.23126), which reports a
  Transformer point-cloud world model with hybrid global/local point-cloud
  reconstruction supervision for rigid, deformable, and flexible materials.
  This file does NOT reproduce ParticleFormer's reported hybrid loss: no public
  official code repository was linked on its project page during the audit.
  PointWorld has persistent point correspondences, so we test a different,
  project-specific hypothesis: compare canonical-KNN edge lengths with the
  actual future GT edge lengths.  That penalizes accidental morphing while still
  permitting real GT deformation.

* Stage 3 borrows JointHOI's continuous, temporally varying, left/right dynamic
  distance-map representation (arXiv:2607.01768).  JointHOI jointly generates
  contact and motion with diffusion; here contact is a deterministic auxiliary
  prediction target.  The public JointHOI Code link did not resolve to an
  accessible implementation when audited.

KNN k, sampled-point counts, SmoothL1 beta, log-distance scaling, near-contact
weighting, and all loss coefficients are PointWorld pilot hyperparameters rather
than claimed paper defaults.  Full details are recorded in
``training/long_horizon_provenance.py``.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F

from training.long_horizon_common import (
    _gather_btn,
    _gather_btnd,
    _gather_btnh,
    _horizon_mask,
    _masked_sum_count,
    _split_hand_points,
    batched_min_hand_distance,
    contact_strength,
)


class LongHorizonStructureLossMixin:
    def _edge_terms(
        self,
        outputs: dict[str, torch.Tensor],
        data: dict[str, Any],
    ) -> tuple[
        torch.Tensor,
        dict[str, tuple[torch.Tensor, torch.Tensor]],
    ]:
        predicted = outputs["scene_flows"].float()
        target = data["gt_scene_flows"].to(predicted.device).float()
        object_valid = outputs["object_valid"].bool()
        canonical = data["point_canonical_xyz"].to(predicted.device).float()
        exists = (
            data["scene_exists"].to(predicted.device).squeeze(-1).bool()
        )
        supervised = (
            data["scene_supervised_mask"]
            .to(predicted.device)
            .squeeze(-1)
            .bool()
        )
        indices, selected_valid = self._selected_points(
            data, object_valid, self.long_horizon_edge_points
        )
        loss_sum = predicted.sum() * 0.0
        loss_count = torch.zeros(
            (), device=predicted.device, dtype=torch.long
        )
        metric_sums: dict[str, torch.Tensor] = {}
        metric_counts: dict[str, torch.Tensor] = {}

        for batch_index in range(predicted.shape[0]):
            for object_index in range(object_valid.shape[1]):
                count = int(
                    selected_valid[batch_index, object_index].sum()
                )
                if (
                    not bool(object_valid[batch_index, object_index])
                    or count < 3
                ):
                    continue
                point_index = indices[
                    batch_index, object_index, :count
                ]
                canonical_points = canonical[batch_index, point_index]
                distance = torch.cdist(
                    canonical_points, canonical_points
                )
                # Project-specific structure graph: persistent correspondences
                # let us use a fixed canonical KNN topology.  This is not
                # ParticleFormer's Chamfer/Hausdorff-style hybrid point-set loss.
                neighbors = distance.topk(
                    k=min(self.long_horizon_edge_k + 1, count),
                    largest=False,
                ).indices[:, 1:]
                source = (
                    torch.arange(count, device=predicted.device)[:, None]
                    .expand_as(neighbors)
                    .reshape(-1)
                )
                destination = neighbors.reshape(-1)
                pred_points = predicted[batch_index, :, point_index]
                gt_points = target[batch_index, :, point_index]
                point_valid = exists[
                    batch_index, :, point_index
                ] & supervised[batch_index, :, point_index]
                edge_valid = point_valid[:, source] & point_valid[
                    :, destination
                ]
                pred_length = torch.linalg.norm(
                    pred_points[:, source] - pred_points[:, destination],
                    dim=-1,
                )
                gt_length = torch.linalg.norm(
                    gt_points[:, source] - gt_points[:, destination],
                    dim=-1,
                )
                error = (pred_length - gt_length).abs()
                # beta=5 mm is a HOT3D pilot scale, not a cited default.
                smooth = F.smooth_l1_loss(
                    pred_length,
                    gt_length,
                    reduction="none",
                    beta=0.005,
                )
                future = edge_valid.clone()
                future[0] = False
                loss_sum = loss_sum + smooth[future].sum()
                loss_count = loss_count + future.sum()

                ranges = (
                    ("edge_mae_t01_t14", 1, 14),
                    (
                        "edge_mae_t15_t34",
                        15,
                        predicted.shape[1] - 1,
                    ),
                    (
                        "edge_mae_t01_t34",
                        1,
                        predicted.shape[1] - 1,
                    ),
                    (
                        "edge_mae_t34",
                        predicted.shape[1] - 1,
                        predicted.shape[1] - 1,
                    ),
                )
                for name, start, end in ranges:
                    horizon = _horizon_mask(
                        predicted.shape[1],
                        start,
                        end,
                        predicted.device,
                    )[:, None]
                    valid_metric = edge_valid & horizon
                    total, number = _masked_sum_count(
                        error, valid_metric
                    )
                    metric_sums[name] = metric_sums.get(
                        name, error.sum() * 0.0
                    ) + total
                    metric_counts[name] = metric_counts.get(
                        name,
                        torch.zeros(
                            (),
                            device=predicted.device,
                            dtype=torch.long,
                        ),
                    ) + number

        loss = loss_sum / loss_count.clamp_min(1).to(loss_sum.dtype)
        metrics = {
            name: (metric_sums[name], metric_counts[name])
            for name in metric_sums
        }
        return loss, metrics

    def _contact_targets(
        self,
        outputs: dict[str, torch.Tensor],
        data: dict[str, Any],
        *,
        use_predicted_geometry: bool,
        return_indices: bool = False,
    ):
        # JointHOI-inspired target representation: continuous distance from each
        # persistent object anchor to each hand, separately for left/right.
        # Computing it from future GT during training is ordinary supervision;
        # no future object state is supplied to the inference graph.
        object_valid = outputs["object_valid"].bool()
        indices, selected_valid = self._selected_points(
            data, object_valid, self.long_horizon_contact_points
        )
        source = (
            outputs["scene_flows"]
            if use_predicted_geometry
            else data["gt_scene_flows"].to(object_valid.device)
        ).float()
        points = _gather_btnd(source, indices)
        batch, timesteps, objects, selected_points, _ = points.shape
        if objects == 0:
            empty_distance = source.new_zeros(
                batch, timesteps, 0, selected_points, 2
            )
            empty_valid = torch.zeros_like(
                empty_distance, dtype=torch.bool
            )
            return (
                empty_distance,
                empty_valid,
                None,
                indices,
                selected_valid,
            )

        points_flat = points.reshape(
            batch, timesteps, objects * selected_points, 3
        )
        exists = (
            data["scene_exists"]
            .to(object_valid.device)
            .squeeze(-1)
            .bool()
        )
        supervised = (
            data["scene_supervised_mask"]
            .to(object_valid.device)
            .squeeze(-1)
            .bool()
        )
        valid = _gather_btn(exists & supervised, indices)
        valid = valid & selected_valid[:, None]
        valid_flat = valid.reshape(
            batch, timesteps, objects * selected_points
        )
        hand_xyz = data["robot_flows"].to(object_valid.device).float()
        hand_valid = data["robot_exists"].to(object_valid.device).bool()

        distances: list[torch.Tensor] = []
        valid_sides: list[torch.Tensor] = []
        nearest_sides: list[torch.Tensor] = []
        for side_points, side_valid in _split_hand_points(
            hand_xyz, hand_valid
        ):
            distance, valid_distance, nearest = batched_min_hand_distance(
                points_flat,
                valid_flat,
                side_points,
                side_valid,
                return_indices=return_indices,
            )
            distances.append(distance)
            valid_sides.append(valid_distance)
            if nearest is not None:
                nearest_sides.append(nearest)

        distance = torch.stack(distances, dim=-1).view(
            batch, timesteps, objects, selected_points, 2
        )
        valid_distance = torch.stack(valid_sides, dim=-1).view(
            batch, timesteps, objects, selected_points, 2
        )
        nearest = (
            torch.stack(nearest_sides, dim=-1).view(
                batch, timesteps, objects, selected_points, 2
            )
            if nearest_sides
            else None
        )
        return distance, valid_distance, nearest, indices, selected_valid

    def _contact_aux_terms(self, outputs, data):
        target, valid, _nearest, indices, selected_valid = (
            self._contact_targets(
                outputs, data, use_predicted_geometry=False
            )
        )
        predicted = _gather_btnh(
            outputs["long_horizon_contact_distance"], indices
        )
        valid = valid & selected_valid[:, None, :, :, None]
        valid[:, 0] = False
        tau = max(self.long_horizon_contact_tau, 1e-6)
        # Log distance gives the near-contact region substantially more
        # resolution.  This objective and its near-weight are our adaptation,
        # not asserted to match JointHOI's diffusion parameterization.
        pred_log = torch.log1p(predicted.float() / tau)
        target_log = torch.log1p(target.float() / tau)
        weight = 1.0 + self.long_horizon_contact_near_weight * (
            contact_strength(target, tau)
        )
        error = F.smooth_l1_loss(
            pred_log, target_log, reduction="none"
        ) * weight
        # DDP averages parameter gradients across ranks.  A rank-local mean
        # would therefore give a sparse-contact rank the same influence as a
        # dense-contact rank.  Build one globally count-weighted value while
        # retaining a differentiable local numerator; the world-size factor
        # exactly cancels DDP's subsequent gradient averaging.
        local_sum = error[valid].sum()
        local_count = valid.sum().to(dtype=torch.long)
        if dist.is_available() and dist.is_initialized():
            global_sum = local_sum.detach().clone()
            global_count = local_count.detach().clone()
            dist.all_reduce(global_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(global_count, op=dist.ReduceOp.SUM)
            if int(global_count.item()) == 0:
                return local_sum * 0.0
            denominator = global_count.to(local_sum.dtype)
            world_size = float(dist.get_world_size())
            global_value = global_sum / denominator
            return global_value + world_size * (
                local_sum - local_sum.detach()
            ) / denominator
        if int(local_count.item()) == 0:
            return local_sum * 0.0
        return local_sum / local_count.to(local_sum.dtype)
