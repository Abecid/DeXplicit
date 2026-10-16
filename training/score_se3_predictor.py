"""Residual-pose predictor used by score-SE3 conditions metric_pose--oracle_contact."""

from __future__ import annotations

import math
from typing import Any

import torch

from training.contact_se3_conditioning import (
    ContactConditioningResult,
    ContactSE3Conditioner,
    hand_side_validity,
)
from training.long_horizon_predictor import LongHorizonStructuredPredictor
from training.residual_se3_adapter import (
    ResidualSE3PoseAdapter,
    apply_object_local_residual_to_points,
)


EXPLICIT_ACTION_FEATURE_DIM = 17  # rel xyz + delta xyz + their valid flags + time


class ScoreSE3StructuredPredictor(LongHorizonStructuredPredictor):
    """Stage-0 predictor plus a capacity-matched object-level pose adapter."""

    def __init__(
        self,
        args: Any,
        dim: int,
        T: int,
        action_mode: str,
        stage: int,
    ) -> None:
        candidate = str(getattr(args, "score_se3_candidate", ""))
        if candidate not in {"metric_pose", "predicted_contact", "oracle_contact"}:
            raise ValueError("ScoreSE3StructuredPredictor is only for metric_pose/predicted_contact/oracle_contact")
        self.score_se3_candidate = candidate
        self.score_se3_allow_oracle = bool(
            getattr(args, "score_se3_allow_oracle", False)
        )
        if candidate == "oracle_contact" and not self.score_se3_allow_oracle:
            raise PermissionError("oracle_contact requires --score-se3-allow-oracle")
        if candidate != "oracle_contact" and self.score_se3_allow_oracle:
            raise ValueError("oracle access is forbidden for metric_pose/predicted_contact")

        super().__init__(args, dim, T, action_mode, stage)
        translation_bound = float(
            getattr(args, "score_se3_translation_bound_m", 0.05)
        )
        rotation_bound_degrees = float(
            getattr(args, "score_se3_rotation_bound_deg", 15.0)
        )
        contact_tau = float(getattr(args, "score_se3_contact_tau_m", 0.02))
        if not math.isclose(
            float(self.long_horizon_contact_tau),
            contact_tau,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "score-SE3 contact conditioner and supervised contact head "
                "must use the same tau: "
                f"{contact_tau} vs {self.long_horizon_contact_tau}"
            )
        self.score_se3_adapter = ResidualSE3PoseAdapter(
            int(dim) + EXPLICIT_ACTION_FEATURE_DIM,
            max_translation_m=translation_bound,
            max_rotation_radians=math.radians(rotation_bound_degrees),
        )
        self.score_se3_contact_conditioner = ContactSE3Conditioner(
            candidate,
            tau=contact_tau,
            detach_predicted=bool(
                getattr(args, "score_se3_contact_stop_gradient", True)
            ),
            allow_future_ground_truth=self.score_se3_allow_oracle,
        )
        self.score_se3_translation_bound_m = translation_bound
        self.score_se3_rotation_bound_deg = rotation_bound_degrees
        self.score_se3_contact_tau_m = contact_tau
        self._score_se3_oracle_xyz: torch.Tensor | None = None

    def set_long_horizon_context(self, data: dict[str, Any]) -> None:
        super().set_long_horizon_context(data)
        self._score_se3_oracle_xyz = None
        if self.score_se3_candidate != "oracle_contact":
            return
        # This is the only deployability boundary at which future object GT is
        # copied into predictor state. metric_pose/predicted_contact return above and cannot reach it.
        future = data.get("gt_scene_flows")
        if not isinstance(future, torch.Tensor):
            raise KeyError("oracle_contact oracle requires gt_scene_flows")
        if future.ndim != 4 or future.shape[-1] != 3:
            raise ValueError("gt_scene_flows must have shape (B,T,N,3)")
        self._score_se3_oracle_xyz = future

    @staticmethod
    def _canonical_centroids(
        canonical: torch.Tensor,
        ids: torch.Tensor,
        scene_valid: torch.Tensor,
        object_valid: torch.Tensor,
    ) -> torch.Tensor:
        batch, objects = object_valid.shape
        centroids = canonical.new_zeros(batch, objects, 3)
        for batch_index in range(batch):
            for object_index in range(objects):
                mask = scene_valid[batch_index].bool() & (
                    ids[batch_index] == object_index
                )
                if bool(object_valid[batch_index, object_index]) and mask.any():
                    centroids[batch_index, object_index] = canonical[
                        batch_index, mask
                    ].mean(dim=0)
        return centroids

    def _center_contact_anchors(
        self,
        result: ContactConditioningResult,
        canonical: torch.Tensor,
        ids: torch.Tensor,
        scene_valid: torch.Tensor,
        object_valid: torch.Tensor,
    ) -> torch.Tensor:
        features = result.features
        if self.score_se3_candidate == "metric_pose":
            # Preserve the literal-zero capacity control exactly.
            return features
        centroid = self._canonical_centroids(
            canonical, ids, scene_valid, object_valid
        )[:, :, None]
        centered = features.clone()
        for side, start in enumerate((0, 5)):
            anchor = centered[..., start : start + 3] - centroid
            centered[..., start : start + 3] = torch.where(
                result.anchor_valid[..., side, None],
                anchor,
                torch.zeros_like(anchor),
            )
        return centered

    @staticmethod
    def _explicit_action_features(
        encoded: torch.Tensor,
        hand_relative: torch.Tensor,
        hand_delta: torch.Tensor,
        hand_side_valid: torch.Tensor,
        initial_rotation: torch.Tensor,
    ) -> torch.Tensor:
        batch, objects, horizons, _ = encoded.shape
        if hand_relative.shape[:2] != (batch, horizons + 1) or (
            hand_relative.shape[2:] != (2, 3)
        ):
            raise ValueError("hand_relative must have shape (B,H+1,2,3)")
        if hand_delta.shape != hand_relative.shape:
            raise ValueError("hand_delta must match hand_relative")
        expected_side_valid = (batch, horizons + 1, 2)
        if tuple(hand_side_valid.shape) != expected_side_valid:
            raise ValueError(
                "hand_side_valid must have shape "
                f"{expected_side_valid}, got {tuple(hand_side_valid.shape)}"
            )
        if tuple(initial_rotation.shape) != (batch, objects, 3, 3):
            raise ValueError(
                "initial_rotation must have shape "
                f"{(batch, objects, 3, 3)}, got {tuple(initial_rotation.shape)}"
            )
        side_valid = hand_side_valid.to(
            device=hand_relative.device, dtype=torch.bool
        )
        current_valid = side_valid[:, 1:]
        # A frame-0-relative displacement is meaningful only when the same
        # hand is observed both now and at the reference frame.  Otherwise the
        # zero-filled invalid centroid would turn into a spurious world-scale
        # displacement.
        relative_valid = current_valid & side_valid[:, :1]
        relative_world = torch.where(
            relative_valid[..., None],
            hand_relative[:, 1:],
            torch.zeros_like(hand_relative[:, 1:]),
        )
        # Likewise, a one-step delta is defined only across two consecutive
        # valid observations.  This suppresses fake motion when a hand
        # disappears or reappears after an occlusion.
        delta_valid = current_valid & side_valid[:, :-1]
        delta_world = torch.where(
            delta_valid[..., None],
            hand_delta[:, 1:],
            torch.zeros_like(hand_delta[:, 1:]),
        )
        # The adapter emits an object-local twist, so its explicit action
        # vectors must use the same frame.  R0 maps canonical/object-local
        # coordinates to world; R0^T maps known hand motion back to the initial
        # object frame.  Each object's token therefore receives its own action
        # representation even though the world-frame hand trajectory is shared.
        world_to_object = initial_rotation.transpose(-1, -2).to(
            device=encoded.device, dtype=encoded.dtype
        )
        relative_local = torch.einsum(
            "boij,bthj->bothi",
            world_to_object,
            relative_world.to(device=encoded.device, dtype=encoded.dtype),
        ).reshape(batch, objects, horizons, 6)
        delta_local = torch.einsum(
            "boij,bthj->bothi",
            world_to_object,
            delta_world.to(device=encoded.device, dtype=encoded.dtype),
        ).reshape(batch, objects, horizons, 6)
        relative_flags = relative_valid.to(
            device=encoded.device, dtype=encoded.dtype
        )[:, None].expand(-1, objects, -1, -1)
        delta_flags = delta_valid.to(
            device=encoded.device, dtype=encoded.dtype
        )[:, None].expand(-1, objects, -1, -1)
        time = torch.arange(
            1,
            horizons + 1,
            device=encoded.device,
            dtype=encoded.dtype,
        ).view(1, horizons, 1) / float(max(horizons, 1))
        action = torch.cat(
            (
                relative_local,
                delta_local,
                relative_flags,
                delta_flags,
                time[:, None].expand(batch, objects, -1, -1),
            ),
            dim=-1,
        )
        return action

    def _contact_features(
        self,
        *,
        encoded: torch.Tensor,
        canonical: torch.Tensor,
        ids: torch.Tensor,
        scene_valid: torch.Tensor,
        object_valid: torch.Tensor,
        contact_distance: torch.Tensor,
        hand_xyz: torch.Tensor,
        hand_valid: torch.Tensor,
    ) -> ContactConditioningResult:
        horizons = int(encoded.shape[2])
        conditioner = self.score_se3_contact_conditioner
        if self.score_se3_candidate == "metric_pose":
            return conditioner.action_only(
                object_valid, horizons, reference=encoded
            )
        if self.score_se3_candidate == "predicted_contact":
            return conditioner.predicted(
                canonical_xyz=canonical,
                point_object_id=ids,
                point_valid=scene_valid,
                object_valid=object_valid,
                predicted_point_hand_distance=contact_distance[:, 1:],
                hand_valid=hand_valid[:, 1:],
            )
        if self._score_se3_oracle_xyz is None:
            raise RuntimeError("oracle_contact oracle context was not installed before forward")
        # oracle_contact changes only the source of the contact distance field.  It uses the
        # same frame-0 persistent point support as predicted_contact instead of future
        # visibility/supervision masks, which would otherwise leak a second
        # oracle signal through contact availability and mass.
        future_point_valid = scene_valid[:, None].expand(
            -1, int(encoded.shape[2]), -1
        )
        return conditioner.oracle(
            canonical_xyz=canonical,
            point_object_id=ids,
            object_valid=object_valid,
            future_object_xyz=self._score_se3_oracle_xyz.to(
                device=encoded.device, dtype=canonical.dtype
            )[:, 1:],
            future_point_valid=future_point_valid,
            hand_xyz=hand_xyz[:, 1:],
            hand_valid=hand_valid[:, 1:],
        )

    def _apply_score_se3_pose_adapter(
        self,
        *,
        encoded: torch.Tensor,
        rotation: torch.Tensor,
        translation: torch.Tensor,
        initial_rotation: torch.Tensor,
        object_valid: torch.Tensor,
        canonical: torch.Tensor,
        ids: torch.Tensor,
        scene_valid: torch.Tensor,
        contact_distance: torch.Tensor,
        hand_xyz: torch.Tensor,
        hand_valid: torch.Tensor,
        hand_relative: torch.Tensor,
        hand_delta: torch.Tensor,
        point_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        contact = self._contact_features(
            encoded=encoded,
            canonical=canonical,
            ids=ids,
            scene_valid=scene_valid,
            object_valid=object_valid,
            contact_distance=contact_distance,
            hand_xyz=hand_xyz,
            hand_valid=hand_valid,
        )
        conditioning = self._center_contact_anchors(
            contact, canonical, ids, scene_valid, object_valid
        )
        action = self._explicit_action_features(
            encoded,
            hand_relative,
            hand_delta,
            hand_side_validity(hand_valid),
            initial_rotation,
        )
        adapter_input = torch.cat((encoded, action), dim=-1)
        base_rotation = rotation
        base_translation = translation
        adapted = self.score_se3_adapter(
            adapter_input,
            conditioning,
            base_rotation,
            base_translation,
            object_valid,
        )
        outputs = {
            "score_se3_base_object_rotation": base_rotation,
            "score_se3_base_object_translation": base_translation,
            "score_se3_residual_twist": adapted.raw_twist,
            "score_se3_residual_translation_local": adapted.local_translation,
            "score_se3_residual_rotation_vector": adapted.local_rotation_vector,
            "score_se3_residual_rotation": adapted.residual_rotation,
            "score_se3_contact_conditioning": conditioning,
            "score_se3_contact_valid": contact.valid,
            "score_se3_adapted_object_rotation": adapted.rotation,
            "score_se3_adapted_object_translation": adapted.translation,
        }
        # Preserve the base pose here.  The residual is applied after the
        # existing direct/pose blend so it remains a full coherent correction
        # rather than being scaled by the legacy blend gate.
        return base_rotation, base_translation, outputs

    def _apply_score_se3_output_adapter(
        self,
        output_abs: torch.Tensor,
        *,
        ids: torch.Tensor,
        scene_valid: torch.Tensor,
        rotation: torch.Tensor,
        translation: torch.Tensor,
        object_valid: torch.Tensor,
        score_se3_outputs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        required = (
            "score_se3_residual_translation_local",
            "score_se3_residual_rotation",
        )
        if any(key not in score_se3_outputs for key in required):
            raise RuntimeError("score-SE3 adapter outputs are incomplete")
        corrected_future = apply_object_local_residual_to_points(
            output_abs[:, 1:],
            ids,
            scene_valid,
            rotation,
            translation,
            score_se3_outputs["score_se3_residual_translation_local"],
            score_se3_outputs["score_se3_residual_rotation"],
            object_valid,
        )
        return torch.cat((output_abs[:, :1], corrected_future), dim=1)
