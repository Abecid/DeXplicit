"""Carrier-conditioned, gauge-fixed canonical deformation-graph predictor."""
from __future__ import annotations

import math
from typing import Any

import torch

from training.carrier_gauge_core import (
    apply_pose_delta,
    candidate_deltas,
    carrier_pose_candidates,
    compose_pose,
    gauge_fix,
    hand_carriers,
    render_objects,
    so3_exp,
    warp_local_graph,
)
from training.carrier_gauge_head import LowRankCarrierGraphHead
from training.carrier_graph_signals import node_hand_features
from training.contact_se3_conditioning import hand_side_validity
from training.deformation_graph import (
    DeformationGraphTopology,
    build_deformation_graph,
    mass_center_node_twists,
    pool_point_features_to_nodes,
    weighted_kabsch_projection,
)
from training.deformation_graph_predictor import (
    DeformationGraphStructuredPredictor,
)
from training.score_se3_predictor import ScoreSE3StructuredPredictor


CARRIER_GAUGE_CANDIDATES = frozenset({"unfixed_graph", "gauge_fixed_graph", "low_rank_graph", "hand_relational_graph", "dexplicit_3d"})
CARRIER_GAUGE_ENABLED = frozenset({"gauge_fixed_graph", "low_rank_graph", "hand_relational_graph", "dexplicit_3d"})
LOW_RANK_CANDIDATES = frozenset({"low_rank_graph", "hand_relational_graph", "dexplicit_3d"})
RELATIONAL_CANDIDATES = frozenset({"hand_relational_graph", "dexplicit_3d"})
CARRIER_CANDIDATES = frozenset({"dexplicit_3d"})


class DeXplicit3DPredictor(DeformationGraphStructuredPredictor):
    """Deformation graph with exact gauge fixing and optional carrier transport."""

    def __init__(
        self,
        args: Any,
        dim: int,
        T: int,
        action_mode: str,
        stage: int,
    ) -> None:
        candidate = str(getattr(args, "carrier_gauge_candidate", ""))
        if candidate not in CARRIER_GAUGE_ENABLED:
            raise ValueError("predictor is restricted to gauge_fixed_graph--dexplicit_3d")
        expected_graph = "deformation_graph" if candidate == "gauge_fixed_graph" else "gated_deformation_graph"
        if str(getattr(args, "deformation_graph_candidate", "")) != expected_graph:
            raise ValueError(
                f"{candidate} requires deformation_graph_candidate={expected_graph}"
            )
        self.carrier_gauge_candidate = candidate
        super().__init__(args, dim, T, action_mode, stage)
        self.carrier_gauge_mode_rank = int(
            getattr(args, "carrier_gauge_mode_rank", 4)
        )
        self.carrier_gauge_temporal_controls = int(
            getattr(args, "carrier_gauge_temporal_controls", 6)
        )
        self.carrier_gauge_relation_temperature_m = float(
            getattr(args, "carrier_gauge_relation_temperature_m", 0.04)
        )
        self.carrier_gauge_carrier_translation_bound_m = float(
            getattr(args, "carrier_gauge_carrier_translation_bound_m", 0.05)
        )
        self.carrier_gauge_carrier_rotation_bound_deg = float(
            getattr(args, "carrier_gauge_carrier_rotation_bound_deg", 15.0)
        )
        if candidate in LOW_RANK_CANDIDATES:
            self.deformation_graph_head = LowRankCarrierGraphHead(
                int(dim),
                int(T) - 1,
                mode_rank=self.carrier_gauge_mode_rank,
                temporal_controls=self.carrier_gauge_temporal_controls,
                use_relations=candidate in RELATIONAL_CANDIDATES,
                use_carriers=candidate in CARRIER_CANDIDATES,
                translation_bound_m=self.deformation_graph_translation_bound_m,
                rotation_bound_rad=self.deformation_graph_rotation_bound_rad,
                carrier_translation_bound_m=(
                    self.carrier_gauge_carrier_translation_bound_m
                ),
                carrier_rotation_bound_rad=math.radians(
                    self.carrier_gauge_carrier_rotation_bound_deg
                ),
            )
        self._carrier_gauge_scene0: torch.Tensor | None = None
        self._carrier_gauge_identity_checked = False

    def set_long_horizon_context(self, data: dict[str, Any]) -> None:
        super().set_long_horizon_context(data)
        scene = data.get("scene_flows")
        if not isinstance(scene, torch.Tensor) or scene.ndim != 4:
            raise KeyError("carrier-gauge predictor requires scene_flows")
        self._carrier_gauge_scene0 = scene[:, 0]

    @staticmethod
    def _topology_outputs(
        topology: DeformationGraphTopology,
    ) -> dict[str, torch.Tensor]:
        return {
            "deformation_graph_node_centers": topology.node_centers,
            "deformation_graph_node_valid": topology.node_valid,
            "deformation_graph_point_node_indices": topology.point_node_indices,
            "deformation_graph_point_node_weights": topology.point_node_weights,
            "deformation_graph_node_neighbors": topology.node_neighbors,
            "deformation_graph_node_neighbor_valid": (
                topology.node_neighbor_valid
            ),
            "deformation_graph_node_mass": topology.node_mass,
        }

    @staticmethod
    def _initial_translation(
        canonical: torch.Tensor,
        current: torch.Tensor,
        ids: torch.Tensor,
        point_valid: torch.Tensor,
        object_valid: torch.Tensor,
        initial_rotation: torch.Tensor,
    ) -> torch.Tensor:
        translation = current.new_zeros(
            object_valid.shape[0], object_valid.shape[1], 3
        )
        for batch_index in range(current.shape[0]):
            for object_index in range(object_valid.shape[1]):
                mask = point_valid[batch_index].bool() & (
                    ids[batch_index].long() == object_index
                )
                if bool(object_valid[batch_index, object_index]) and mask.any():
                    rotated = torch.einsum(
                        "ij,nj->ni",
                        initial_rotation[batch_index, object_index],
                        canonical[batch_index, mask],
                    )
                    translation[batch_index, object_index] = (
                        current[batch_index, mask] - rotated
                    ).mean(0)
        return translation

    def _raw_graph_outputs(
        self,
        *,
        encoded: torch.Tensor,
        action: torch.Tensor,
        point_features: torch.Tensor,
        canonical: torch.Tensor,
        ids: torch.Tensor,
        scene_valid: torch.Tensor,
        object_valid: torch.Tensor,
        initial_rotation: torch.Tensor,
        initial_translation: torch.Tensor,
        hand_xyz: torch.Tensor,
        hand_valid: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        topology = build_deformation_graph(
            canonical,
            ids,
            scene_valid,
            object_valid,
            node_count=self.deformation_graph_node_count,
            influence_count=self.deformation_graph_influence_count,
            edge_count=self.deformation_graph_edge_count,
            radius_scale=self.deformation_graph_radius_scale,
            minimum_radius_m=self.deformation_graph_minimum_radius_m,
        )
        node_features = pool_point_features_to_nodes(
            point_features, ids, scene_valid, topology
        )
        descriptor = self._node_descriptor(topology)
        outputs = self._topology_outputs(topology)
        if self.carrier_gauge_candidate == "gauge_fixed_graph":
            raw_twist, gate = self.deformation_graph_head(
                encoded,
                action,
                node_features,
                descriptor,
                topology.node_valid,
            )
            bounded = torch.cat(
                (
                    self.deformation_graph_translation_bound_m
                    * torch.tanh(raw_twist[..., :3]),
                    self.deformation_graph_rotation_bound_rad
                    * torch.tanh(raw_twist[..., 3:]),
                ),
                dim=-1,
            )
            centered = mass_center_node_twists(
                bounded, topology.node_mass, topology.node_valid
            )
            translation = self._enforce_bound(
                centered[..., :3],
                self.deformation_graph_translation_bound_m,
            ) * gate[..., None]
            rotation_vector = self._enforce_bound(
                centered[..., 3:],
                self.deformation_graph_rotation_bound_rad,
            ) * gate[..., None]
            local_rotation = so3_exp(rotation_vector.float()).to(encoded.dtype)
            outputs.update(
                {
                    "deformation_graph_raw_twist": raw_twist,
                    "deformation_graph_local_translation": translation,
                    "deformation_graph_local_rotation_vector": rotation_vector,
                    "deformation_graph_local_rotation": local_rotation,
                    "deformation_graph_gate": gate,
                }
            )
            return outputs

        relation = None
        if self.carrier_gauge_candidate in RELATIONAL_CANDIDATES:
            relation = node_hand_features(
                topology.node_centers,
                topology.node_valid,
                initial_rotation,
                initial_translation,
                hand_xyz,
                hand_valid,
                temperature_m=self.carrier_gauge_relation_temperature_m,
            )
        low_rank = self.deformation_graph_head(
            encoded,
            action,
            node_features,
            descriptor,
            topology,
            relation_features=relation,
        )
        outputs.update(
            {
                "deformation_graph_raw_twist": low_rank.node_twist,
                "deformation_graph_local_translation": (
                    low_rank.local_translation
                ),
                "deformation_graph_local_rotation_vector": (
                    low_rank.local_rotation_vector
                ),
                "deformation_graph_local_rotation": low_rank.local_rotation,
                "deformation_graph_gate": low_rank.gate,
                "carrier_gauge_raw_mode_twist": low_rank.raw_mode_twist,
                "carrier_gauge_projected_mode_twist": (
                    low_rank.projected_mode_twist
                ),
                "carrier_gauge_temporal_coefficients": (
                    low_rank.temporal_coefficients
                ),
                "carrier_gauge_spatial_basis": low_rank.spatial_basis,
            }
        )
        if relation is not None:
            outputs["carrier_gauge_node_hand_features"] = relation
        return outputs

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
        point_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        rotation, translation, outputs = (
            ScoreSE3StructuredPredictor._apply_score_se3_pose_adapter(
                self,
                encoded=encoded,
                rotation=rotation,
                translation=translation,
                initial_rotation=initial_rotation,
                object_valid=object_valid,
                canonical=canonical,
                ids=ids,
                scene_valid=scene_valid,
                contact_distance=contact_distance,
                hand_xyz=hand_xyz,
                hand_valid=hand_valid,
                hand_relative=hand_relative,
                hand_delta=hand_delta,
                point_features=point_features,
            )
        )
        action = self._explicit_action_features(
            encoded,
            hand_relative,
            hand_delta,
            hand_side_validity(hand_valid),
            initial_rotation,
        )
        if self._carrier_gauge_scene0 is None:
            raise RuntimeError("set_long_horizon_context was not called")
        current = self._carrier_gauge_scene0.to(
            canonical.device, canonical.dtype
        )
        initial_translation = self._initial_translation(
            canonical,
            current,
            ids,
            scene_valid,
            object_valid,
            initial_rotation,
        )
        outputs.update(
            self._raw_graph_outputs(
                encoded=encoded,
                action=action,
                point_features=point_features,
                canonical=canonical,
                ids=ids,
                scene_valid=scene_valid,
                object_valid=object_valid,
                initial_rotation=initial_rotation,
                initial_translation=initial_translation,
                hand_xyz=hand_xyz,
                hand_valid=hand_valid,
            )
        )
        outputs.update(
            {
                "carrier_gauge_encoded": encoded,
                "carrier_gauge_action": action,
                "carrier_gauge_initial_rotation": initial_rotation,
                "carrier_gauge_initial_translation": initial_translation,
            }
        )
        return rotation, translation, outputs

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
        corrected = ScoreSE3StructuredPredictor._apply_score_se3_output_adapter(
            self,
            output_abs,
            ids=ids,
            scene_valid=scene_valid,
            rotation=rotation,
            translation=translation,
            object_valid=object_valid,
            score_se3_outputs=score_se3_outputs,
        )
        output_dtype = corrected.dtype
        with torch.autocast(device_type=corrected.device.type, enabled=False):
            return self._compose_graph_output(
                corrected.float(),
                ids=ids,
                scene_valid=scene_valid,
                object_valid=object_valid,
                score_se3_outputs=score_se3_outputs,
            ).to(output_dtype)

    def _compose_graph_output(
        self,
        corrected: torch.Tensor,
        *,
        ids: torch.Tensor,
        scene_valid: torch.Tensor,
        object_valid: torch.Tensor,
        score_se3_outputs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        # Graph geometry runs in float32 with autocast disabled.  Under the
        # trainer's bf16 autocast the skinning/render einsums round
        # metre-scale coordinates by ~2 mm, which breaks the exact identity
        # gate and is comparable to the metric itself (first GPU smoke).
        canonical = self.canonical_xyz.to(corrected.device, torch.float32)
        batch, objects = object_valid.shape
        horizons = corrected.shape[1] - 1
        global_rotation = score_se3_outputs[
            "score_se3_adapted_object_rotation"
        ].clone().float()
        global_translation = score_se3_outputs[
            "score_se3_adapted_object_translation"
        ].clone().float()
        fit_valid = torch.zeros(
            batch,
            objects,
            horizons,
            device=corrected.device,
            dtype=torch.bool,
        )
        rigid_projection = corrected[:, 1:].clone()
        for batch_index in range(batch):
            for object_index in range(objects):
                mask = scene_valid[batch_index].bool() & (
                    ids[batch_index].long() == object_index
                )
                if (
                    not bool(object_valid[batch_index, object_index])
                    or int(mask.sum()) < 3
                ):
                    continue
                fitted_rotation, fitted_translation, valid = (
                    weighted_kabsch_projection(
                        canonical[batch_index, mask],
                        corrected[batch_index, 1:, mask],
                        fallback_rotation=global_rotation[
                            batch_index, object_index
                        ],
                        fallback_translation=global_translation[
                            batch_index, object_index
                        ],
                    )
                )
                global_rotation[batch_index, object_index] = fitted_rotation
                global_translation[batch_index, object_index] = fitted_translation
                fit_valid[batch_index, object_index] = valid
                rigid_projection[batch_index, :, mask] = (
                    torch.einsum(
                        "tij,nj->tni",
                        fitted_rotation,
                        canonical[batch_index, mask],
                    )
                    + fitted_translation[:, None]
                )

        topology = self._topology_from_outputs(score_se3_outputs)
        raw_local = warp_local_graph(
            canonical,
            ids,
            scene_valid,
            topology,
            score_se3_outputs["deformation_graph_local_rotation"].float(),
            score_se3_outputs["deformation_graph_local_translation"].float(),
        )
        fixed_local, gauge_rotation, gauge_translation, gauge_valid = gauge_fix(
            canonical, raw_local, ids, scene_valid, object_valid
        )
        final_rotation, final_translation = compose_pose(
            global_rotation,
            global_translation,
            gauge_rotation,
            gauge_translation,
        )

        if self.carrier_gauge_candidate in CARRIER_CANDIDATES:
            if self._long_horizon_batch is None:
                raise RuntimeError("hand trajectory context is missing")
            hand_xyz = self._long_horizon_batch["robot_flows"].to(
                corrected.device, corrected.dtype
            )
            hand_valid = self._long_horizon_batch["robot_exists"].to(
                corrected.device
            )
            carriers = hand_carriers(
                hand_xyz, hand_valid,
                unlabelled=self._long_horizon_batch.get("action_is_unlabelled"),
            )
            candidate_rotation, candidate_translation, candidate_valid = (
                carrier_pose_candidates(
                    score_se3_outputs["carrier_gauge_initial_rotation"].float(),
                    score_se3_outputs["carrier_gauge_initial_translation"].float(),
                    final_rotation,
                    final_translation,
                    carriers,
                )
            )
            delta_translation, delta_rotation, candidate_features = (
                candidate_deltas(
                    final_rotation,
                    final_translation,
                    candidate_rotation,
                    candidate_translation,
                    candidate_valid,
                )
            )
            carrier = self.deformation_graph_head.mix_carriers(
                score_se3_outputs["carrier_gauge_encoded"].float(),
                score_se3_outputs["carrier_gauge_action"].float(),
                candidate_features,
                delta_translation,
                delta_rotation,
                candidate_valid,
            )
            object_mask = object_valid[:, :, None, None]
            carrier_translation = torch.where(
                object_mask,
                carrier.local_translation,
                torch.zeros_like(carrier.local_translation),
            )
            carrier_rotation = torch.where(
                object_mask,
                carrier.local_rotation_vector,
                torch.zeros_like(carrier.local_rotation_vector),
            )
            final_rotation, final_translation = apply_pose_delta(
                final_rotation,
                final_translation,
                carrier_translation,
                carrier_rotation,
            )
            score_se3_outputs.update(
                {
                    "carrier_gauge_carrier_weights": carrier.weights,
                    "carrier_gauge_carrier_logits": carrier.logits,
                    "carrier_gauge_carrier_strength": carrier.strength,
                    "carrier_gauge_carrier_translation": carrier_translation,
                    "carrier_gauge_carrier_rotation_vector": carrier_rotation,
                    "carrier_gauge_carrier_valid": candidate_valid,
                }
            )

        graph_future = render_objects(
            fixed_local,
            ids,
            scene_valid,
            object_valid,
            final_rotation,
            final_translation,
            fallback=corrected[:, 1:],
        )
        score_se3_outputs.update(
            {
                "deformation_graph_global_rotation": global_rotation,
                "deformation_graph_global_translation": global_translation,
                "deformation_graph_kabsch_valid": fit_valid,
                "deformation_graph_rigid_projection": torch.cat(
                    (corrected[:, :1], rigid_projection), dim=1
                ),
                "deformation_graph_e2_votes": corrected,
                "carrier_gauge_raw_local_points": raw_local,
                "carrier_gauge_fixed_local_points": fixed_local,
                "carrier_gauge_rotation": gauge_rotation,
                "carrier_gauge_translation": gauge_translation,
                "carrier_gauge_valid": gauge_valid,
                "carrier_gauge_final_rotation": final_rotation,
                "carrier_gauge_final_translation": final_translation,
            }
        )
        if not self._carrier_gauge_identity_checked:
            # The inherited checkpoint loader marks the parent graph identity
            # invariant satisfied for trained resume/eval checkpoints. Fresh
            # metric_pose initialization leaves that flag false, so only a genuinely
            # new zero-initialized extension is tested against the rigid
            # projection. This keeps the init proof strict without rejecting
            # legitimate trained CG checkpoints on their first eval forward.
            if self._deformation_graph_identity_checked:
                self._carrier_gauge_identity_checked = True
            else:
                maximum_twist = float(
                    score_se3_outputs["deformation_graph_raw_twist"]
                    .detach()
                    .abs()
                    .max()
                )
                maximum_difference = float(
                    (graph_future.detach() - rigid_projection.detach()).abs().max()
                )
                if maximum_twist > 1e-7 or maximum_difference > 3e-5:
                    raise RuntimeError(
                        "carrier-gauge graph violates its identity initialization: "
                        f"max_twist={maximum_twist:.3e}, "
                        f"max_projection_delta={maximum_difference:.3e}"
                    )
                self._carrier_gauge_identity_checked = True
                self._deformation_graph_identity_checked = True
        return torch.cat((corrected[:, :1], graph_future), dim=1)


__all__ = [
    "CARRIER_CANDIDATES",
    "CARRIER_GAUGE_CANDIDATES",
    "CARRIER_GAUGE_ENABLED",
    "DeXplicit3DPredictor",
    "LOW_RANK_CANDIDATES",
    "RELATIONAL_CANDIDATES",
]
