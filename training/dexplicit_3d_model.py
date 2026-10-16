"""DeXplicit 3D model: shared object motion, local deformation and metric refinement."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import torch

from training.carrier_gauge_core import so3_log
from training.carrier_gauge_metrics import (
    scalar_sum_count,
    trajectory_decomposition_sums,
)
from training.dexplicit_3d_predictor import (
    CARRIER_CANDIDATES,
    CARRIER_GAUGE_CANDIDATES,
    CARRIER_GAUGE_ENABLED,
    LOW_RANK_CANDIDATES,
    RELATIONAL_CANDIDATES,
    DeXplicit3DPredictor,
)
from training.carrier_graph_signals import edge_strain_loss
from training.deformation_graph_model import DeformationGraphExperimentalModel


CARRIER_GAUGE_CAUSAL_ARGS = (
    "carrier_gauge_candidate",
    "carrier_gauge_mode_rank",
    "carrier_gauge_temporal_controls",
    "carrier_gauge_relation_temperature_m",
    "carrier_gauge_carrier_translation_bound_m",
    "carrier_gauge_carrier_rotation_bound_deg",
    "carrier_gauge_edge_strain_weight",
    "carrier_gauge_coefficient_weight",
    "carrier_gauge_carrier_tv_weight",
    "carrier_gauge_carrier_strength_weight",
    "carrier_gauge_deformable_threshold_m",
)


def _argument_or_environment(
    args: Any,
    name: str,
    environment: str,
    default: Any,
    cast,
):
    value = getattr(args, name, None)
    if value is None:
        value = os.environ.get(environment, default)
    return cast(value)


def _set_argument(args: Any, name: str, value: Any) -> None:
    setattr(args, name, value)
    original = getattr(args, "og_args", None)
    if original is not None:
        setattr(original, name, value)


def _safe_mean(
    value: torch.Tensor, valid: torch.Tensor | None = None
) -> torch.Tensor:
    if valid is None:
        return value.mean() if value.numel() else value.sum() * 0.0
    valid = valid.bool().expand_as(value)
    return value[valid].mean() if valid.any() else value.sum() * 0.0


class DeXplicit3DModel(DeformationGraphExperimentalModel):
    """Selected metric_pose graph model plus identifiable low-rank carrier dynamics."""

    def __init__(
        self,
        args: Any,
        data_info_dict: dict[str, int],
        rank: int = 0,
        cpu_pg=None,
    ) -> None:
        candidate = str(
            getattr(args, "carrier_gauge_candidate", None)
            or os.environ.get("CARRIER_GAUGE_CANDIDATE", "")
        )
        if candidate not in CARRIER_GAUGE_CANDIDATES:
            raise ValueError(
                "carrier_gauge_candidate must be unfixed_graph, gauge_fixed_graph, low_rank_graph, hand_relational_graph, or dexplicit_3d"
            )
        if str(getattr(args, "score_se3_candidate", "")) != "metric_pose":
            raise ValueError("unfixed_graph--dexplicit_3d must inherit the selected metric_pose stack")
        graph_candidate = "deformation_graph" if candidate in {"unfixed_graph", "gauge_fixed_graph"} else "gated_deformation_graph"
        _set_argument(args, "carrier_gauge_candidate", candidate)
        _set_argument(args, "deformation_graph_candidate", graph_candidate)
        # The campaign compares mechanisms at fixed graph capacity. A separate
        # capacity campaign may change this only through an explicit argument.
        if getattr(args, "deformation_graph_nodes", None) is None:
            _set_argument(
                args,
                "deformation_graph_nodes",
                int(os.environ.get("CARRIER_GAUGE_GRAPH_NODES", "8")),
            )

        values = {
            "carrier_gauge_mode_rank": _argument_or_environment(
                args, "carrier_gauge_mode_rank", "CARRIER_GAUGE_MODE_RANK", 4, int
            ),
            "carrier_gauge_temporal_controls": _argument_or_environment(
                args,
                "carrier_gauge_temporal_controls",
                "CARRIER_GAUGE_TEMPORAL_CONTROLS",
                6,
                int,
            ),
            "carrier_gauge_relation_temperature_m": _argument_or_environment(
                args,
                "carrier_gauge_relation_temperature_m",
                "CARRIER_GAUGE_RELATION_TEMPERATURE_M",
                0.04,
                float,
            ),
            "carrier_gauge_carrier_translation_bound_m": _argument_or_environment(
                args,
                "carrier_gauge_carrier_translation_bound_m",
                "CARRIER_GAUGE_CARRIER_TRANSLATION_BOUND_M",
                0.05,
                float,
            ),
            "carrier_gauge_carrier_rotation_bound_deg": _argument_or_environment(
                args,
                "carrier_gauge_carrier_rotation_bound_deg",
                "CARRIER_GAUGE_CARRIER_ROTATION_BOUND_DEG",
                15.0,
                float,
            ),
            "carrier_gauge_edge_strain_weight": _argument_or_environment(
                args,
                "carrier_gauge_edge_strain_weight",
                "CARRIER_GAUGE_EDGE_STRAIN_WEIGHT",
                0.5,
                float,
            ),
            "carrier_gauge_coefficient_weight": _argument_or_environment(
                args,
                "carrier_gauge_coefficient_weight",
                "CARRIER_GAUGE_COEFFICIENT_WEIGHT",
                1e-4,
                float,
            ),
            "carrier_gauge_carrier_tv_weight": _argument_or_environment(
                args,
                "carrier_gauge_carrier_tv_weight",
                "CARRIER_GAUGE_CARRIER_TV_WEIGHT",
                1e-3,
                float,
            ),
            "carrier_gauge_carrier_strength_weight": _argument_or_environment(
                args,
                "carrier_gauge_carrier_strength_weight",
                "CARRIER_GAUGE_CARRIER_STRENGTH_WEIGHT",
                1e-4,
                float,
            ),
            "carrier_gauge_deformable_threshold_m": _argument_or_environment(
                args,
                "carrier_gauge_deformable_threshold_m",
                "CARRIER_GAUGE_DEFORMABLE_THRESHOLD_M",
                0.005,
                float,
            ),
        }
        if values["carrier_gauge_mode_rank"] <= 0:
            raise ValueError("carrier-gauge mode rank must be positive")
        if values["carrier_gauge_temporal_controls"] < 4:
            raise ValueError("carrier-gauge temporal controls must be at least four")
        for name, value in values.items():
            if (
                name.endswith("_weight")
                or name.endswith("_m")
                or name.endswith("_deg")
            ):
                if not math.isfinite(float(value)) or float(value) < 0:
                    raise ValueError(f"{name} must be finite and non-negative")
            _set_argument(args, name, value)
        if values["carrier_gauge_relation_temperature_m"] <= 0:
            raise ValueError("relation temperature must be positive")
        if values["carrier_gauge_carrier_translation_bound_m"] <= 0:
            raise ValueError("carrier translation bound must be positive")
        if values["carrier_gauge_carrier_rotation_bound_deg"] <= 0:
            raise ValueError("carrier rotation bound must be positive")
        if bool(getattr(args, "score_se3_strict_contract", False)) and int(
            getattr(args, "seed", 2026)
        ) != 2026:
            raise ValueError("strict carrier-gauge campaign uses exactly seed 2026")

        self.carrier_gauge_candidate = candidate
        for name, value in values.items():
            setattr(self, name, value)
        super().__init__(args, data_info_dict, rank=rank, cpu_pg=cpu_pg)
        if candidate in CARRIER_GAUGE_ENABLED:
            if self._object_jepa_hook is not None:
                self._object_jepa_hook.remove()
            self.dynamics_predictor = DeXplicit3DPredictor(
                args,
                int(self.channels),
                int(self.T),
                str(self.action_mode),
                int(self.long_horizon_stage),
            )
            self._object_jepa_hook = (
                self.dynamics_predictor.object_time.register_forward_hook(
                    self._capture_object_time_hidden
                )
            )
        self._write_carrier_gauge_contract(args)

    def _write_carrier_gauge_contract(self, args: Any) -> None:
        output_dir = getattr(args, "output_dir", None)
        if not output_dir or int(getattr(self, "rank", 0)) != 0:
            return
        path = Path(output_dir).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        candidate = self.carrier_gauge_candidate
        payload = {
            "schema_version": 1,
            "candidate": candidate,
            "inherited_score_method": "metric_pose",
            "inherited_graph_candidate": str(
                getattr(args, "deformation_graph_candidate")
            ),
            "exact_six_dof_gauge_fix": candidate in CARRIER_GAUGE_ENABLED,
            "low_rank_graph_time_decoder": candidate in LOW_RANK_CANDIDATES,
            "node_hand_relations": candidate in RELATIONAL_CANDIDATES,
            "carrier_conditioned_global_transport": candidate in CARRIER_CANDIDATES,
            "final_foreground_pointwise_bypass": False,
            "future_ground_truth_inference_input": False,
            "training_seed": int(getattr(args, "seed", 2026)),
            "hyperparameters": {
                name: getattr(args, name) for name in CARRIER_GAUGE_CAUSAL_ARGS
            },
            "shared_loss": {
                "gt_edge_strain_weight": self.carrier_gauge_edge_strain_weight,
                "purpose": (
                    "match GT future edge strain while allowing genuine deformation"
                ),
            },
            "fidelity_boundary": {
                "adopted": [
                    "embedded canonical graph skinning",
                    "proper weighted Kabsch gauge projection",
                    "graph-Laplacian low-frequency spatial modes",
                    "clamped cubic B-spline temporal basis",
                    "object-frame node/hand relational features",
                    "bounded carrier-frame pose residuals",
                ],
                "not_claimed": [
                    "reproduction of an upstream RGB-D reconstruction pipeline",
                    "reproduction of an upstream simulator or material model",
                    "paper-default PointWorld forecasting hyperparameters",
                ],
            },
        }
        (path / "dexplicit_3d_model_contract.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def loss_fn(self, outputs, data, training=True):
        total, log = super().loss_fn(outputs, data, training=training)
        topology = self._topology(outputs)
        predicted = outputs["scene_flows"].float()
        target = data["gt_scene_flows"].to(predicted.device).float()
        canonical = data["point_canonical_xyz"].to(predicted.device).float()
        ids = data["point_object_id"].to(predicted.device).long()
        point_valid = (
            data["scene_exists"].to(predicted.device).squeeze(-1).bool()
            & data["scene_supervised_mask"]
            .to(predicted.device)
            .squeeze(-1)
            .bool()
        )
        edge_strain = edge_strain_loss(
            predicted,
            target,
            canonical,
            ids,
            point_valid[:, 0],
            topology,
            future_valid=point_valid,
        )
        total = total + self.carrier_gauge_edge_strain_weight * edge_strain

        coefficient = total.new_zeros(())
        carrier_tv = total.new_zeros(())
        carrier_strength = total.new_zeros(())
        fixed_displacement = total.new_zeros(())
        fixed_acceleration = total.new_zeros(())
        gauge_translation = total.new_zeros(())
        gauge_rotation = total.new_zeros(())
        gauge_valid_fraction = total.new_zeros(())
        if self.carrier_gauge_candidate in CARRIER_GAUGE_ENABLED:
            fixed = outputs["carrier_gauge_fixed_local_points"].float()
            frame_valid = point_valid[:, 0] & (ids >= 0)
            displacement = torch.linalg.vector_norm(
                fixed - canonical[:, None], dim=-1
            )
            fixed_displacement = _safe_mean(
                displacement, frame_valid[:, None].expand_as(displacement)
            )
            if fixed.shape[1] >= 3:
                acceleration = torch.linalg.vector_norm(
                    fixed[:, 2:] - 2 * fixed[:, 1:-1] + fixed[:, :-2],
                    dim=-1,
                )
                fixed_acceleration = _safe_mean(
                    acceleration,
                    frame_valid[:, None].expand_as(acceleration),
                )
            gauge_valid = outputs["carrier_gauge_valid"].bool()
            gauge_translation = _safe_mean(
                torch.linalg.vector_norm(
                    outputs["carrier_gauge_translation"].float(), dim=-1
                ),
                gauge_valid,
            )
            gauge_rotation = _safe_mean(
                torch.linalg.vector_norm(
                    so3_log(outputs["carrier_gauge_rotation"].float()), dim=-1
                ),
                gauge_valid,
            )
            gauge_valid_fraction = _safe_mean(gauge_valid.float())

        if self.carrier_gauge_candidate in LOW_RANK_CANDIDATES:
            coefficients = outputs["carrier_gauge_temporal_coefficients"].float()
            object_valid = outputs["object_valid"].bool()
            coefficient = _safe_mean(
                coefficients.square(),
                object_valid[:, :, None, None, None].expand_as(coefficients),
            )
            total = total + self.carrier_gauge_coefficient_weight * coefficient

        if self.carrier_gauge_candidate in CARRIER_CANDIDATES:
            weights = outputs["carrier_gauge_carrier_weights"].float()
            strength = outputs["carrier_gauge_carrier_strength"].float()
            object_valid = outputs["object_valid"].bool()
            carrier_strength = _safe_mean(
                strength.square(),
                object_valid[:, :, None, None].expand_as(strength),
            )
            if weights.shape[2] >= 2:
                difference = (weights[:, :, 1:] - weights[:, :, :-1]).abs()
                carrier_tv = _safe_mean(
                    difference,
                    object_valid[:, :, None, None].expand_as(difference),
                )
            total = (
                total
                + self.carrier_gauge_carrier_tv_weight * carrier_tv
                + self.carrier_gauge_carrier_strength_weight * carrier_strength
            )

        log.update(
            {
                "carrier_gauge/edge_strain": float(edge_strain.detach().cpu()),
                "carrier_gauge/edge_strain_weighted": float(
                    (
                        self.carrier_gauge_edge_strain_weight * edge_strain
                    ).detach().cpu()
                ),
                "carrier_gauge/coefficient_l2": float(coefficient.detach().cpu()),
                "carrier_gauge/carrier_tv": float(carrier_tv.detach().cpu()),
                "carrier_gauge/carrier_strength_l2": float(
                    carrier_strength.detach().cpu()
                ),
                "carrier_gauge/fixed_local_displacement_m": float(
                    fixed_displacement.detach().cpu()
                ),
                "carrier_gauge/fixed_local_acceleration_m": float(
                    fixed_acceleration.detach().cpu()
                ),
                "carrier_gauge/gauge_translation_m": float(
                    gauge_translation.detach().cpu()
                ),
                "carrier_gauge/gauge_rotation_rad": float(
                    gauge_rotation.detach().cpu()
                ),
                "carrier_gauge/gauge_valid_fraction": float(
                    gauge_valid_fraction.detach().cpu()
                ),
                "total_loss": float(total.detach().cpu()),
            }
        )
        return total, log

    @torch.no_grad()
    def long_horizon_eval_sums(self, batch, outputs):
        result = super().long_horizon_eval_sums(batch, outputs)
        predicted = outputs["scene_flows"].float()
        target = batch["gt_scene_flows"].to(predicted.device).float()
        canonical = batch["point_canonical_xyz"].to(predicted.device).float()
        ids = batch["point_object_id"].to(predicted.device).long()
        point_valid = (
            batch["scene_exists"].to(predicted.device).squeeze(-1).bool()
            & batch["scene_supervised_mask"]
            .to(predicted.device)
            .squeeze(-1)
            .bool()
        )
        result.update(
            trajectory_decomposition_sums(
                predicted,
                target,
                canonical,
                ids,
                point_valid,
                outputs["object_valid"].bool(),
                deformable_threshold_m=(
                    self.carrier_gauge_deformable_threshold_m
                ),
            )
        )
        if self.carrier_gauge_candidate in CARRIER_GAUGE_ENABLED:
            gauge_valid = outputs["carrier_gauge_valid"].bool()
            result["carrier_gauge/gauge/translation_m"] = scalar_sum_count(
                torch.linalg.vector_norm(
                    outputs["carrier_gauge_translation"].float(), dim=-1
                ),
                gauge_valid,
            )
            result["carrier_gauge/gauge/rotation_rad"] = scalar_sum_count(
                torch.linalg.vector_norm(
                    so3_log(outputs["carrier_gauge_rotation"].float()), dim=-1
                ),
                gauge_valid,
            )
        if self.carrier_gauge_candidate in LOW_RANK_CANDIDATES:
            coefficients = outputs["carrier_gauge_temporal_coefficients"].float()
            valid = outputs["object_valid"].bool()[:, :, None, None, None]
            result["carrier_gauge/low_rank/coefficient_abs"] = scalar_sum_count(
                coefficients.abs(), valid
            )
        if self.carrier_gauge_candidate in CARRIER_CANDIDATES:
            weights = outputs["carrier_gauge_carrier_weights"].float()
            valid = (
                outputs["object_valid"].bool()[:, :, None, None]
                & outputs["carrier_gauge_carrier_valid"].bool()
            )
            normalized = weights.clamp_min(1e-12)
            entropy = -(normalized * normalized.log()).sum(-1)
            result["carrier_gauge/carrier/entropy"] = scalar_sum_count(
                entropy,
                outputs["object_valid"].bool()[:, :, None],
            )
            for index, name in enumerate(
                ("base", "left", "right", "bimanual", "world")
            ):
                result[f"carrier_gauge/carrier/mass_{name}"] = scalar_sum_count(
                    weights[..., index], valid[..., index]
                )
            strength = outputs[
                "carrier_gauge_carrier_strength"
            ].float().squeeze(-1)
            result["carrier_gauge/carrier/strength_abs"] = scalar_sum_count(
                strength.abs(),
                outputs["object_valid"].bool()[:, :, None],
            )
        return result


__all__ = [
    "CARRIER_GAUGE_CAUSAL_ARGS",
    "DeXplicit3DModel",
]
