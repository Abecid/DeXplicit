"""Production model wrapper for the single-seed deformation-graph campaign."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from training.deformation_graph import (
    DeformationGraphTopology,
    arap_graph_loss,
    local_transform_magnitude,
    temporal_transform_acceleration,
)
from training.deformation_graph_predictor import (
    DEFORMATION_GRAPH_CANDIDATES,
    GRAPH_ENABLED_CANDIDATES,
    DeformationGraphStructuredPredictor,
)
from training.score_se3_model import ScoreSE3ExperimentalModel


class DeformationGraphExperimentalModel(ScoreSE3ExperimentalModel):
    """Selected metric_pose model plus an optional final canonical graph decoder."""

    def __init__(
        self,
        args: Any,
        data_info_dict: dict[str, int],
        rank: int = 0,
        cpu_pg=None,
    ) -> None:
        candidate = str(getattr(args, "deformation_graph_candidate", ""))
        if candidate not in DEFORMATION_GRAPH_CANDIDATES:
            raise ValueError(
                "deformation_graph_candidate must be dense_decoder, deformation_graph, gated_deformation_graph, or larger_deformation_graph"
            )
        if str(getattr(args, "score_se3_candidate", "")) != "metric_pose":
            raise ValueError("dense_decoder-larger_deformation_graph must all inherit the selected metric_pose stack")
        if bool(getattr(args, "score_se3_allow_oracle", False)):
            raise ValueError("deformation-graph campaign is fully deployable and forbids oracle input")
        self.deformation_graph_candidate = candidate
        self.deformation_graph_arap_weight = float(
            getattr(args, "deformation_graph_arap_weight", 0.25)
        )
        self.deformation_graph_temporal_weight = float(
            getattr(args, "deformation_graph_temporal_weight", 0.05)
        )
        self.deformation_graph_magnitude_weight = float(
            getattr(args, "deformation_graph_magnitude_weight", 0.01)
        )
        self.deformation_graph_gate_weight = float(
            getattr(args, "deformation_graph_gate_weight", 0.10)
        )
        self.deformation_graph_gate_rigid_floor_m = float(
            getattr(args, "deformation_graph_gate_rigid_floor_m", 0.002)
        )
        self.deformation_graph_gate_full_scale_m = float(
            getattr(args, "deformation_graph_gate_full_scale_m", 0.020)
        )
        for name, value in (
            ("arap_weight", self.deformation_graph_arap_weight),
            ("temporal_weight", self.deformation_graph_temporal_weight),
            ("magnitude_weight", self.deformation_graph_magnitude_weight),
            ("gate_weight", self.deformation_graph_gate_weight),
            ("gate_rigid_floor_m", self.deformation_graph_gate_rigid_floor_m),
            ("gate_full_scale_m", self.deformation_graph_gate_full_scale_m),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"deformation graph {name} must be finite and non-negative")
        if self.deformation_graph_gate_full_scale_m <= self.deformation_graph_gate_rigid_floor_m:
            raise ValueError("gate full scale must be greater than its rigid floor")

        for key, value in self._graph_argument_contract(args).items():
            setattr(args, key, value)
            original = getattr(args, "og_args", None)
            if original is not None:
                setattr(original, key, value)

        super().__init__(args, data_info_dict, rank=rank, cpu_pg=cpu_pg)
        if candidate in GRAPH_ENABLED_CANDIDATES:
            if self._object_jepa_hook is not None:
                self._object_jepa_hook.remove()
            self.dynamics_predictor = DeformationGraphStructuredPredictor(
                args,
                int(self.channels),
                int(self.T),
                str(self.action_mode),
                int(self.long_horizon_stage),
            )
            self._object_jepa_hook = self.dynamics_predictor.object_time.register_forward_hook(
                self._capture_object_time_hidden
            )
        self._write_deformation_graph_contract(args)

    def _graph_argument_contract(self, args: Any) -> dict[str, Any]:
        candidate = self.deformation_graph_candidate
        default_nodes = 16 if candidate == "larger_deformation_graph" else 8
        return {
            "deformation_graph_candidate": candidate,
            "deformation_graph_nodes": int(
                getattr(args, "deformation_graph_nodes", default_nodes)
            ),
            "deformation_graph_influences": int(
                getattr(args, "deformation_graph_influences", 4)
            ),
            "deformation_graph_edges": int(
                getattr(args, "deformation_graph_edges", 4)
            ),
            "deformation_graph_radius_scale": float(
                getattr(args, "deformation_graph_radius_scale", 1.5)
            ),
            "deformation_graph_minimum_radius_m": float(
                getattr(args, "deformation_graph_minimum_radius_m", 0.005)
            ),
            "deformation_graph_translation_bound_m": float(
                getattr(args, "deformation_graph_translation_bound_m", 0.03)
            ),
            "deformation_graph_rotation_bound_deg": float(
                getattr(args, "deformation_graph_rotation_bound_deg", 12.0)
            ),
            "deformation_graph_arap_weight": self.deformation_graph_arap_weight,
            "deformation_graph_temporal_weight": self.deformation_graph_temporal_weight,
            "deformation_graph_magnitude_weight": self.deformation_graph_magnitude_weight,
            "deformation_graph_gate_weight": self.deformation_graph_gate_weight,
            "deformation_graph_gate_rigid_floor_m": self.deformation_graph_gate_rigid_floor_m,
            "deformation_graph_gate_full_scale_m": self.deformation_graph_gate_full_scale_m,
        }

    def _write_deformation_graph_contract(self, args: Any) -> None:
        output_dir = getattr(args, "output_dir", None)
        if not output_dir or int(getattr(self, "rank", 0)) != 0:
            return
        path = Path(output_dir).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        candidate = self.deformation_graph_candidate
        payload = {
            "schema_version": 1,
            "candidate": candidate,
            "inherited_method": "selected score-SE3 metric_pose checkpoint",
            "graph_enabled": candidate in GRAPH_ENABLED_CANDIDATES,
            "learned_deformability_gate": candidate == "gated_deformation_graph",
            "final_pointwise_bypass": False if candidate in GRAPH_ENABLED_CANDIDATES else True,
            "global_pose_source": (
                "differentiable Kabsch projection of metric_pose dense votes"
                if candidate in GRAPH_ENABLED_CANDIDATES
                else "unchanged metric_pose output"
            ),
            "local_deformation": (
                "canonical embedded graph with shared local SE3 transforms"
                if candidate in GRAPH_ENABLED_CANDIDATES
                else "none"
            ),
            "future_ground_truth_inference_input": False,
            "deployable": True,
            "single_training_seed": int(getattr(args, "seed", 2026)),
            "hyperparameters": self._graph_argument_contract(args),
            "reference_fidelity": {
                "adopted": [
                    "persistent canonical graph nodes",
                    "distance-weighted skinning",
                    "shared local rigid transforms",
                    "ARAP graph-edge consistency",
                ],
                "not_claimed": [
                    "NeuralGraph RGB-D reconstruction pipeline",
                    "volumetric SDF decoder",
                    "upstream correspondence network",
                    "upstream training scale or hyperparameter reproduction",
                ],
            },
        }
        (path / "deformation_graph_model_contract.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    @staticmethod
    def _topology(outputs: dict[str, torch.Tensor]) -> DeformationGraphTopology:
        return DeformationGraphTopology(
            node_centers=outputs["deformation_graph_node_centers"],
            node_valid=outputs["deformation_graph_node_valid"],
            point_node_indices=outputs["deformation_graph_point_node_indices"],
            point_node_weights=outputs["deformation_graph_point_node_weights"],
            node_neighbors=outputs["deformation_graph_node_neighbors"],
            node_neighbor_valid=outputs["deformation_graph_node_neighbor_valid"],
            node_mass=outputs["deformation_graph_node_mass"],
        )

    @torch.no_grad()
    def _gt_deformability_target(
        self,
        outputs: dict[str, torch.Tensor],
        data: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        predicted = outputs["scene_flows"]
        target = data["gt_scene_flows"].to(predicted.device).float()
        canonical = data["point_canonical_xyz"].to(predicted.device).float()
        ids = data["point_object_id"].to(predicted.device).long()
        object_valid = outputs["object_valid"].bool()
        valid_tracks = (
            data["scene_exists"].to(predicted.device).squeeze(-1).bool()
            & data["scene_supervised_mask"].to(predicted.device).squeeze(-1).bool()
        )
        rotation, translation, pose_valid = self._gt_pose(
            target, canonical, ids, valid_tracks, int(object_valid.shape[1])
        )
        residual = target.new_zeros(
            target.shape[0], object_valid.shape[1], target.shape[1] - 1
        )
        residual_valid = torch.zeros_like(residual, dtype=torch.bool)
        for batch_index in range(target.shape[0]):
            for object_index in range(object_valid.shape[1]):
                for future_index in range(target.shape[1] - 1):
                    time_index = future_index + 1
                    mask = valid_tracks[batch_index, time_index] & (
                        ids[batch_index] == object_index
                    )
                    if not bool(pose_valid[batch_index, object_index, time_index]) or not mask.any():
                        continue
                    rigid = torch.einsum(
                        "ij,nj->ni",
                        rotation[batch_index, object_index, time_index],
                        canonical[batch_index, mask],
                    ) + translation[batch_index, object_index, time_index]
                    residual[batch_index, object_index, future_index] = torch.linalg.vector_norm(
                        rigid - target[batch_index, time_index, mask], dim=-1
                    ).mean()
                    residual_valid[batch_index, object_index, future_index] = True
        denominator = (
            self.deformation_graph_gate_full_scale_m
            - self.deformation_graph_gate_rigid_floor_m
        )
        deformability = (
            (residual - self.deformation_graph_gate_rigid_floor_m) / denominator
        ).clamp(0.0, 1.0)
        return deformability, residual, residual_valid & object_valid[:, :, None]

    def loss_fn(self, outputs, data, training=True):
        total, log = super().loss_fn(outputs, data, training=training)
        if self.deformation_graph_candidate not in GRAPH_ENABLED_CANDIDATES:
            return total, log
        topology = self._topology(outputs)
        arap = arap_graph_loss(
            topology,
            outputs["deformation_graph_local_rotation"],
            outputs["deformation_graph_local_translation"],
        )
        magnitude = local_transform_magnitude(
            outputs["deformation_graph_local_translation"],
            outputs["deformation_graph_local_rotation_vector"],
            outputs["deformation_graph_node_valid"],
        )
        temporal = temporal_transform_acceleration(
            outputs["deformation_graph_local_translation"],
            outputs["deformation_graph_local_rotation_vector"],
            outputs["deformation_graph_node_valid"],
        )
        graph_regularization = (
            self.deformation_graph_arap_weight * arap
            + self.deformation_graph_magnitude_weight * magnitude
            + self.deformation_graph_temporal_weight * temporal
        )
        total = total + graph_regularization
        gate_loss = total.new_zeros(())
        gt_residual_mean = total.new_zeros(())
        if self.deformation_graph_candidate == "gated_deformation_graph":
            (
                deformability_target,
                gt_rigid_residual,
                gate_valid,
            ) = self._gt_deformability_target(outputs, data)
            predicted_gate = outputs["deformation_graph_gate"].squeeze(-1)
            if gate_valid.any():
                gate_loss = F.smooth_l1_loss(
                    predicted_gate[gate_valid],
                    deformability_target[gate_valid],
                    beta=0.1,
                )
                gt_residual_mean = gt_rigid_residual[gate_valid].mean()
            total = total + self.deformation_graph_gate_weight * gate_loss
        with torch.no_grad():
            kabsch_valid = outputs["deformation_graph_kabsch_valid"]
            valid_fraction = (
                kabsch_valid.float().mean()
                if kabsch_valid.numel()
                else total.new_zeros(())
            )
            gate_mean = outputs["deformation_graph_gate"].mean()
        log.update(
            {
                "deformation_graph/arap_m": float(arap.detach().cpu()),
                "deformation_graph/arap_weighted": float(
                    (self.deformation_graph_arap_weight * arap).detach().cpu()
                ),
                "deformation_graph/magnitude_m": float(magnitude.detach().cpu()),
                "deformation_graph/magnitude_weighted": float(
                    (self.deformation_graph_magnitude_weight * magnitude).detach().cpu()
                ),
                "deformation_graph/temporal_m": float(temporal.detach().cpu()),
                "deformation_graph/temporal_weighted": float(
                    (self.deformation_graph_temporal_weight * temporal).detach().cpu()
                ),
                "deformation_graph/gate_loss": float(gate_loss.detach().cpu()),
                "deformation_graph/gate_weighted": float(
                    (self.deformation_graph_gate_weight * gate_loss).detach().cpu()
                ),
                "deformation_graph/gate_mean": float(gate_mean.detach().cpu()),
                "deformation_graph/gt_rigid_residual_m": float(
                    gt_residual_mean.detach().cpu()
                ),
                "deformation_graph/kabsch_valid_fraction": float(
                    valid_fraction.detach().cpu()
                ),
                "total_loss": float(total.detach().cpu()),
            }
        )
        return total, log

    @torch.no_grad()
    def long_horizon_eval_sums(self, batch, outputs):
        result = super().long_horizon_eval_sums(batch, outputs)
        if self.deformation_graph_candidate not in GRAPH_ENABLED_CANDIDATES:
            return result
        prediction = outputs["scene_flows"]
        target = batch["gt_scene_flows"].to(prediction.device)
        point_object_id = batch["point_object_id"].to(prediction.device).long()
        valid = (
            batch["scene_exists"].to(prediction.device).squeeze(-1).bool()
            & batch["scene_supervised_mask"].to(prediction.device)
            .squeeze(-1)
            .bool()
            & ~batch["scene_context_mask"].to(prediction.device)
            .squeeze(-1)
            .bool()
            & (point_object_id[:, None, :] >= 0)
        )
        moved = batch["scene_moved_mask"].to(prediction.device).squeeze(-1).bool()
        metric_mask = valid & moved
        metric_mask[:, :1] = False

        def add_point_error(name: str, value: torch.Tensor) -> None:
            error = torch.linalg.vector_norm(value - target, dim=-1)
            result[f"deformation_graph/{name}"] = (
                error[metric_mask].sum(),
                metric_mask.sum(),
            )

        add_point_error("final_l2_moved_t01_t34", prediction)
        add_point_error(
            "e2_votes_l2_moved_t01_t34",
            outputs["deformation_graph_e2_votes"],
        )
        add_point_error(
            "rigid_projection_l2_moved_t01_t34",
            outputs["deformation_graph_rigid_projection"],
        )
        valid_future = valid[:, 1:]
        graph_from_rigid = torch.linalg.vector_norm(
            prediction[:, 1:]
            - outputs["deformation_graph_rigid_projection"][:, 1:],
            dim=-1,
        )
        vote_from_rigid = torch.linalg.vector_norm(
            outputs["deformation_graph_e2_votes"][:, 1:]
            - outputs["deformation_graph_rigid_projection"][:, 1:],
            dim=-1,
        )
        result["deformation_graph/local_deformation_m"] = (
            graph_from_rigid[valid_future].sum(),
            valid_future.sum(),
        )
        result["deformation_graph/e2_nonrigid_vote_m"] = (
            vote_from_rigid[valid_future].sum(),
            valid_future.sum(),
        )
        kabsch_valid = outputs["deformation_graph_kabsch_valid"]
        result["deformation_graph/kabsch_valid_fraction"] = (
            kabsch_valid.float().sum(),
            torch.tensor(
                kabsch_valid.numel(),
                device=kabsch_valid.device,
                dtype=torch.long,
            ),
        )
        gate = outputs["deformation_graph_gate"].squeeze(-1)
        gate_valid = outputs["object_valid"][:, :, None].expand_as(gate)
        result["deformation_graph/gate_mean"] = (
            gate[gate_valid].sum(),
            gate_valid.sum(),
        )
        return result


__all__ = ["DeformationGraphExperimentalModel"]
