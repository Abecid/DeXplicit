"""Particle regularization, physical probes, and contact-consistency losses.

Research provenance
-------------------
The contact-consistency idea is motivated by JointHOI's Contact Inner Guidance
(CIG, arXiv:2607.01768): compare generated contact with contact implied by the
generated geometry.  This implementation is NOT JointHOI CIG.  JointHOI applies
guidance during diffusion sampling; PointWorld is deterministic, so stage 8
uses a supervised training loss instead.  The nearest-hand velocity term is a
new, GT-gated approximation for persistent sticking contact, not a copied
stick/slip model from JointHOI.

The particle balance/entropy/compactness terms are also PointWorld-specific.
LPWM (arXiv:2603.04553; https://github.com/taldatech/lpwm) motivates persistent
particles and per-particle dynamics, but the official model learns DLP particle
attributes and stochastic latent actions rather than these supervised 3-D soft
clusters.

All thresholds and coefficients in this module are pilot hyperparameters that
must be tuned on the held-out HOT3D protocol.  See
``training/long_horizon_provenance.py``.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from training.long_horizon_common import (
    _gather_bnd,
    _gather_btnd,
    _split_hand_points,
    contact_strength,
)
from training.object_jepa_se3 import object_pose_state


class LongHorizonPhysicsLossMixin:
    def _particle_regularization(self, outputs, data):
        # Project-specific regularizers for the LPWM-inspired 3-D part
        # bottleneck.  These are not LPWM's ELBO, latent-action KL, or published
        # loss weights.
        assignment = outputs["long_horizon_particle_assignment"].float()
        object_valid = outputs["object_valid"].bool()
        ids = data["point_object_id"].to(assignment.device).long()
        canonical = data["point_canonical_xyz"].to(assignment.device).float()
        exists0 = (
            data["scene_exists"]
            .to(assignment.device)[:, 0]
            .squeeze(-1)
            .bool()
        )
        balance = assignment.sum() * 0.0
        entropy = assignment.sum() * 0.0
        compact = assignment.sum() * 0.0
        count = 0
        particle_count = int(assignment.shape[-1])
        for batch_index in range(assignment.shape[0]):
            for object_index in range(object_valid.shape[1]):
                mask = exists0[batch_index] & (
                    ids[batch_index] == object_index
                )
                if (
                    not bool(object_valid[batch_index, object_index])
                    or mask.sum() < 3
                ):
                    continue
                probabilities = assignment[batch_index, mask].clamp_min(1e-8)
                usage = probabilities.mean(dim=0)
                balance = balance + (
                    usage - 1.0 / particle_count
                ).square().mean()
                entropy = entropy + (
                    -(probabilities * probabilities.log()).sum(dim=-1)
                ).mean()
                denominator = probabilities.sum(dim=0).clamp_min(1e-6)
                center = torch.einsum(
                    "nk,nc->kc",
                    probabilities,
                    canonical[batch_index, mask],
                ) / denominator[:, None]
                distance = torch.cdist(
                    canonical[batch_index, mask], center
                ).square()
                compact = compact + (
                    probabilities * distance
                ).sum() / probabilities.sum().clamp_min(1)
                count += 1
        divisor = max(count, 1)
        return balance / divisor, entropy / divisor, compact / divisor

    def _physical_probe_loss(self, outputs, data):
        # V-JEPA 2 / DINO-WM motivate physically useful predictive latents, but
        # this supervised 21-D probe is our diagnostic auxiliary.  It does not
        # reproduce V-JEPA 2's EMA target encoder or DINO-WM's patch predictor.
        cache = self._object_jepa_gt_pose_cache
        if cache is None or int(outputs["object_valid"].shape[1]) == 0:
            return outputs["scene_flows"].sum() * 0.0
        rotation, translation, pose_valid = cache
        pose_target = object_pose_state(
            rotation.float(), translation.float()
        )[:, :, 1:]
        pose_target = pose_target.clone()
        pose_target[..., 6:9] *= 10.0
        pose_target[..., 15:18] *= 10.0

        (
            contact_target,
            contact_valid,
            _nearest,
            indices,
            selected_valid,
        ) = self._contact_targets(
            outputs, data, use_predicted_geometry=False
        )
        contact_valid = contact_valid & selected_valid[
            :, None, :, :, None
        ]
        contact_masked = contact_target.masked_fill(~contact_valid, 1.0)
        contact_min = (
            contact_masked.min(dim=3).values[:, 1:]
            .permute(0, 2, 1, 3)
        )
        contact_object_valid = (
            contact_valid.any(dim=3)[:, 1:].permute(0, 2, 1, 3)
        )
        contact_log = torch.log1p(
            contact_min / max(self.long_horizon_contact_tau, 1e-6)
        )
        contact_log = torch.where(
            contact_object_valid,
            contact_log,
            torch.zeros_like(contact_log),
        )

        gt = data["gt_scene_flows"].to(rotation.device).float()
        canonical = data["point_canonical_xyz"].to(rotation.device).float()
        points = _gather_btnd(gt, indices)
        canonical_points = _gather_bnd(canonical, indices)
        local = torch.einsum(
            "botij,botkj->botki",
            rotation.transpose(-1, -2),
            points.permute(0, 2, 1, 3, 4)
            - translation[:, :, :, None],
        )
        deformation = torch.linalg.norm(
            local - canonical_points[:, :, None], dim=-1
        )
        deformation_valid = selected_valid[:, :, None]
        deformation_rms = torch.sqrt(
            (
                deformation.square()
                * deformation_valid.to(deformation.dtype)
            ).sum(dim=-1)
            / deformation_valid.sum(dim=-1).clamp_min(1)
        )[:, :, 1:, None]
        target = torch.cat(
            [pose_target, contact_log, deformation_rms * 20.0], dim=-1
        )
        predicted = outputs["long_horizon_physical_probe"].float()
        valid = pose_valid[:, :, 1:] & outputs["object_valid"][:, :, None]
        error = F.smooth_l1_loss(
            predicted, target.detach(), reduction="none"
        ).mean(dim=-1)
        return error[valid].mean() if valid.any() else error.sum() * 0.0

    def _contact_physics_losses(self, outputs, data):
        # JointHOI-CIG-inspired consistency, translated from inference guidance
        # into a deterministic training objective. This is an adaptation, not a
        # reproduction of JointHOI's denoising-gradient implementation.
        (
            target,
            target_valid,
            nearest,
            indices,
            selected_valid,
        ) = self._contact_targets(
            outputs,
            data,
            use_predicted_geometry=False,
            return_indices=True,
        )
        (
            predicted_distance,
            predicted_valid,
            _unused,
            _indices,
            _selected,
        ) = self._contact_targets(
            outputs, data, use_predicted_geometry=True
        )
        valid = target_valid & predicted_valid & selected_valid[
            :, None, :, :, None
        ]
        valid[:, 0] = False
        near_weight = contact_strength(
            target, self.long_horizon_contact_tau
        )
        log_error = (
            torch.log1p(
                predicted_distance
                / max(self.long_horizon_contact_tau, 1e-6)
            )
            - torch.log1p(
                target / max(self.long_horizon_contact_tau, 1e-6)
            )
        ).abs()
        consistency = (
            (log_error * near_weight)[valid].mean()
            if valid.any()
            else log_error.sum() * 0.0
        )
        if nearest is None:
            return consistency, consistency * 0.0

        object_valid = outputs["object_valid"].bool()
        predicted_points = _gather_btnd(
            outputs["scene_flows"].float(), indices
        )
        gt_points = _gather_btnd(
            data["gt_scene_flows"]
            .to(predicted_points.device)
            .float(),
            indices,
        )
        batch, timesteps, objects, point_count, _ = predicted_points.shape
        hand_xyz = data["robot_flows"].to(predicted_points.device).float()
        hand_valid = data["robot_exists"].to(predicted_points.device).bool()
        stick_losses: list[torch.Tensor] = []
        for hand_index, (side_points, _side_valid) in enumerate(
            _split_hand_points(hand_xyz, hand_valid)
        ):
            nearest_side = nearest[..., hand_index].reshape(
                batch, timesteps, objects * point_count
            )
            previous_index = nearest_side[:, :-1]
            previous_hands = side_points[:, :-1, None].expand(
                -1, -1, objects * point_count, -1, -1
            )
            current_hands = side_points[:, 1:, None].expand(
                -1, -1, objects * point_count, -1, -1
            )
            gather_index = previous_index[..., None, None].expand(
                -1, -1, -1, 1, 3
            )
            hand_previous = torch.gather(
                previous_hands, 3, gather_index
            ).squeeze(3).view(
                batch,
                timesteps - 1,
                objects,
                point_count,
                3,
            )
            hand_current = torch.gather(
                current_hands, 3, gather_index
            ).squeeze(3).view(
                batch,
                timesteps - 1,
                objects,
                point_count,
                3,
            )
            gt_delta = gt_points[:, 1:] - gt_points[:, :-1]
            pred_delta = (
                predicted_points[:, 1:] - predicted_points[:, :-1]
            )
            hand_delta = hand_current - hand_previous
            relative_speed = torch.linalg.norm(
                gt_delta - hand_delta, dim=-1
            )
            contact_now = (
                target[:, 1:, ..., hand_index]
                < self.long_horizon_contact_threshold
            )
            contact_previous = (
                target[:, :-1, ..., hand_index]
                < self.long_horizon_contact_threshold
            )
            # Project-specific conservative stick gate.  It activates only when
            # GT shows consecutive near-contact and low object/hand relative
            # motion; otherwise sliding and releasing contacts remain legal.
            stick = (
                contact_now
                & contact_previous
                & (relative_speed < self.long_horizon_stick_velocity)
                & target_valid[:, 1:, ..., hand_index]
                & target_valid[:, :-1, ..., hand_index]
                & object_valid[:, None, :, None]
            )
            velocity_error = F.smooth_l1_loss(
                pred_delta,
                hand_delta,
                reduction="none",
                beta=0.005,
            ).mean(dim=-1)
            if stick.any():
                stick_losses.append(velocity_error[stick].mean())
        stick_loss = (
            torch.stack(stick_losses).mean()
            if stick_losses
            else consistency * 0.0
        )
        return consistency, stick_loss
