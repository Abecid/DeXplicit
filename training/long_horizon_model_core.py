"""Core wiring for the long-horizon ablations.

Every run writes both its selected mechanisms and an audited research-provenance
record.  This prevents a paper-inspired PointWorld adaptation from later being
mistaken for a faithful port of an upstream architecture or hyperparameter
recipe.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch

from training.long_horizon_common import (
    _env_float,
    _env_int,
    normalize_track,
    stage_features,
    stage_name,
)
from training.long_horizon_predictor import LongHorizonStructuredPredictor
from training.long_horizon_provenance import provenance_contract
from training.object_jepa_se3 import deterministic_object_indices


class LongHorizonModelCoreMixin:
    def __init__(
        self,
        args: Any,
        data_info_dict: dict[str, int],
        rank=0,
        cpu_pg=None,
    ):
        self.long_horizon_stage = _env_int("LONG_HORIZON_STAGE", 0)
        self.long_horizon_stage_name = stage_name(self.long_horizon_stage)
        self.long_horizon_track = normalize_track()
        self.long_horizon_features = frozenset(
            stage_features(
                self.long_horizon_stage,
                self.long_horizon_track,
            )
        )
        os.environ.setdefault(
            "OBJECT_JEPA_LOSS_WEIGHT",
            "0.10"
            if "object_state_jepa" in self.long_horizon_features
            else "0.0",
        )
        os.environ["OBJECT_JEPA_FORCE_POSE_OUTPUT"] = "0"
        self.long_horizon_edge_weight = _env_float(
            "LONG_HORIZON_EDGE_LOSS_WEIGHT", 2.0
        )
        self.long_horizon_edge_points = _env_int(
            "LONG_HORIZON_EDGE_POINTS", 128
        )
        self.long_horizon_edge_k = _env_int("LONG_HORIZON_EDGE_K", 6)
        self.long_horizon_contact_weight = _env_float(
            "LONG_HORIZON_CONTACT_LOSS_WEIGHT", 0.1
        )
        self.long_horizon_contact_points = _env_int(
            "LONG_HORIZON_CONTACT_POINTS", 128
        )
        self.long_horizon_contact_tau = _env_float(
            "LONG_HORIZON_CONTACT_TAU_M", 0.02
        )
        self.long_horizon_contact_near_weight = _env_float(
            "LONG_HORIZON_CONTACT_NEAR_WEIGHT", 4.0
        )
        self.long_horizon_particle_balance_weight = _env_float(
            "LONG_HORIZON_PARTICLE_BALANCE_WEIGHT", 0.02
        )
        self.long_horizon_particle_entropy_weight = _env_float(
            "LONG_HORIZON_PARTICLE_ENTROPY_WEIGHT", 0.002
        )
        self.long_horizon_particle_compact_weight = _env_float(
            "LONG_HORIZON_PARTICLE_COMPACT_WEIGHT", 0.1
        )
        self.long_horizon_physical_weight = _env_float(
            "LONG_HORIZON_PHYSICAL_PROBE_WEIGHT", 0.05
        )
        self.long_horizon_consistency_weight = _env_float(
            "LONG_HORIZON_CONTACT_CONSISTENCY_WEIGHT", 0.1
        )
        self.long_horizon_stick_weight = _env_float(
            "LONG_HORIZON_STICK_LOSS_WEIGHT", 0.05
        )
        self.long_horizon_contact_threshold = _env_float(
            "LONG_HORIZON_CONTACT_THRESHOLD_M", 0.005
        )
        self.long_horizon_stick_velocity = _env_float(
            "LONG_HORIZON_STICK_VELOCITY_M", 0.01
        )

        super().__init__(args, data_info_dict, rank=rank, cpu_pg=cpu_pg)
        if self._object_jepa_hook is not None:
            self._object_jepa_hook.remove()
        self.dynamics_predictor = LongHorizonStructuredPredictor(
            args,
            int(self.channels),
            int(self.T),
            str(self.action_mode),
            self.long_horizon_stage,
        )
        self._object_jepa_hook = (
            self.dynamics_predictor.object_time.register_forward_hook(
                self._capture_object_time_hidden
            )
        )
        for key, value in {
            "long_horizon_stage": self.long_horizon_stage,
            "long_horizon_stage_name": self.long_horizon_stage_name,
            "long_horizon_features": sorted(self.long_horizon_features),
            "long_horizon_track": self.long_horizon_track,
        }.items():
            setattr(args, key, value)
            original = getattr(args, "og_args", None)
            if original is not None:
                setattr(original, key, value)
        self._write_long_horizon_contract(args)

    def _write_long_horizon_contract(self, args: Any) -> None:
        output_dir = getattr(args, "output_dir", None)
        if not output_dir or int(getattr(self, "rank", 0)) != 0:
            return
        path = Path(output_dir).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        score_candidate = str(getattr(args, "score_se3_candidate", "") or "")
        future_gt_oracle = score_candidate == "oracle_contact"
        inference_inputs = [
            "context 3D points and pointwise DINO features",
            "point object ids and canonical coordinates",
            "complete future hand/action trajectory",
        ]
        if future_gt_oracle:
            inference_inputs.append(
                "future GT object geometry for non-deployable contact oracle"
            )
        payload = {
            "stage": int(self.long_horizon_stage),
            "name": self.long_horizon_stage_name,
            "track": self.long_horizon_track,
            "features": sorted(self.long_horizon_features),
            "independent_init": True,
            "inference_inputs": inference_inputs,
            "future_gt_used_only_as_training_target": not future_gt_oracle,
            "future_gt_contact_oracle": future_gt_oracle,
            "output_mode": self.dynamics_predictor.long_horizon_output_mode,
            "pose_head_init": self.dynamics_predictor.long_horizon_pose_head_init,
            "future_visibility_condition": False,
            "weights": {
                "edge": self.long_horizon_edge_weight,
                "contact": self.long_horizon_contact_weight,
                "physical_probe": self.long_horizon_physical_weight,
                "contact_consistency": self.long_horizon_consistency_weight,
                "stick": self.long_horizon_stick_weight,
            },
            # Machine-readable record of the exact paper/code relationship.
            # In particular this records which stages are custom syntheses and
            # which upstream implementations were unavailable.
            "research_provenance": provenance_contract(
                self.long_horizon_stage
            ),
        }
        (path / "long_horizon_ablation_contract.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def forward(self, data_dict, training=True, encoded_scene_feat0=None):
        self.dynamics_predictor.set_long_horizon_context(data_dict)
        return super().forward(
            data_dict,
            training=training,
            encoded_scene_feat0=encoded_scene_feat0,
        )

    def _selected_points(
        self,
        data: dict[str, Any],
        object_valid: torch.Tensor,
        max_points: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ids = data["point_object_id"].to(object_valid.device).long()
        if int(object_valid.shape[1]) == 0:
            batch = int(ids.shape[0])
            return (
                torch.zeros(
                    batch,
                    0,
                    int(max_points),
                    dtype=torch.long,
                    device=ids.device,
                ),
                torch.zeros(
                    batch,
                    0,
                    int(max_points),
                    dtype=torch.bool,
                    device=ids.device,
                ),
            )
        exists0 = (
            data["scene_exists"]
            .to(object_valid.device)[:, 0]
            .squeeze(-1)
            .bool()
        )
        context = exists0 & (ids >= 0)
        return deterministic_object_indices(
            ids,
            context,
            int(object_valid.shape[1]),
            int(max_points),
        )
