"""Score-metric_pose predictor whose final decoder is a canonical deformation graph."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

from training.contact_se3_conditioning import hand_side_validity
from training.deformation_graph import (
    DeformationGraphTopology,
    build_deformation_graph,
    mass_center_node_twists,
    pool_point_features_to_nodes,
    warp_points_with_deformation_graph,
    weighted_kabsch_projection,
)
from training.residual_se3_adapter import so3_exp_map
from training.score_se3_predictor import (
    EXPLICIT_ACTION_FEATURE_DIM,
    ScoreSE3StructuredPredictor,
)


DEFORMATION_GRAPH_CANDIDATES = frozenset({"dense_decoder", "deformation_graph", "gated_deformation_graph", "larger_deformation_graph"})
GRAPH_ENABLED_CANDIDATES = frozenset({"deformation_graph", "gated_deformation_graph", "larger_deformation_graph"})


def _zero_last_linear(module: nn.Module) -> None:
    linears = [child for child in module.modules() if isinstance(child, nn.Linear)]
    if not linears:
        raise ValueError("module has no linear projection to zero-initialize")
    with torch.no_grad():
        linears[-1].weight.zero_()
        if linears[-1].bias is not None:
            linears[-1].bias.zero_()


class CanonicalDeformationGraphHead(nn.Module):
    """Predict one bounded local twist for each persistent graph node/time."""

    def __init__(self, dim: int, *, learned_gate: bool) -> None:
        super().__init__()
        input_dim = 2 * int(dim) + EXPLICIT_ACTION_FEATURE_DIM + 4
        hidden = max(128, int(dim))
        self.twist = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 6),
        )
        _zero_last_linear(self.twist)
        self.gate = None
        if learned_gate:
            gate_input = int(dim) + EXPLICIT_ACTION_FEATURE_DIM
            self.gate = nn.Sequential(
                nn.LayerNorm(gate_input),
                nn.Linear(gate_input, max(64, int(dim) // 2)),
                nn.GELU(),
                nn.Linear(max(64, int(dim) // 2), 1),
            )
            with torch.no_grad():
                self.gate[-1].weight.zero_()
                # Start nearly rigid without saturating the gate's gradient.
                self.gate[-1].bias.fill_(math.log(0.05 / 0.95))

    def forward(
        self,
        encoded: torch.Tensor,
        action: torch.Tensor,
        node_features: torch.Tensor,
        node_descriptor: torch.Tensor,
        node_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, objects, horizons, channels = encoded.shape
        nodes = int(node_features.shape[2])
        if tuple(action.shape[:3]) != (batch, objects, horizons):
            raise ValueError("action features do not match object-time features")
        if tuple(node_features.shape) != (batch, objects, nodes, channels):
            raise ValueError("node_features must have shape (B,O,K,C)")
        if tuple(node_descriptor.shape) != (batch, objects, nodes, 4):
            raise ValueError("node_descriptor must have shape (B,O,K,4)")
        if tuple(node_valid.shape) != (batch, objects, nodes):
            raise ValueError("node_valid must have shape (B,O,K)")
        # Keep the mixed semantic/geometry token in the object-time dtype.
        # Canonical coordinates stay float32 for graph construction, while AMP
        # may place encoded semantic/action features in bf16/fp16.
        action = action.to(device=encoded.device, dtype=encoded.dtype)
        node_features = node_features.to(
            device=encoded.device, dtype=encoded.dtype
        )
        node_descriptor = node_descriptor.to(
            device=encoded.device, dtype=encoded.dtype
        )
        feature = torch.cat(
            (
                encoded[:, :, :, None].expand(-1, -1, -1, nodes, -1),
                action[:, :, :, None].expand(-1, -1, -1, nodes, -1),
                node_features[:, :, None].expand(-1, -1, horizons, -1, -1),
                node_descriptor[:, :, None].expand(-1, -1, horizons, -1, -1),
            ),
            dim=-1,
        )
        raw_twist = self.twist(feature)
        raw_twist = raw_twist * node_valid[:, :, None, :, None]
        if self.gate is None:
            gate = encoded.new_ones(batch, objects, horizons, 1)
        else:
            gate = torch.sigmoid(self.gate(torch.cat((encoded, action), dim=-1)))
        return raw_twist, gate


class DeformationGraphStructuredPredictor(ScoreSE3StructuredPredictor):
    """metric_pose dense votes projected through a coherent canonical graph decoder."""

    def __init__(
        self,
        args: Any,
        dim: int,
        T: int,
        action_mode: str,
        stage: int,
    ) -> None:
        candidate = str(getattr(args, "deformation_graph_candidate", ""))
        if candidate not in GRAPH_ENABLED_CANDIDATES:
            raise ValueError(
                "DeformationGraphStructuredPredictor is restricted to deformation_graph/gated_deformation_graph/larger_deformation_graph"
            )
        if str(getattr(args, "score_se3_candidate", "")) != "metric_pose":
            raise ValueError("deformation-graph candidates must inherit score-SE3 metric_pose")
        self.deformation_graph_candidate = candidate
        super().__init__(args, dim, T, action_mode, stage)
        self.deformation_graph_node_count = int(
            getattr(args, "deformation_graph_nodes", 16 if candidate == "larger_deformation_graph" else 8)
        )
        self.deformation_graph_influence_count = int(
            getattr(args, "deformation_graph_influences", 4)
        )
        self.deformation_graph_edge_count = int(
            getattr(args, "deformation_graph_edges", 4)
        )
        self.deformation_graph_radius_scale = float(
            getattr(args, "deformation_graph_radius_scale", 1.5)
        )
        self.deformation_graph_minimum_radius_m = float(
            getattr(args, "deformation_graph_minimum_radius_m", 0.005)
        )
        self.deformation_graph_translation_bound_m = float(
            getattr(args, "deformation_graph_translation_bound_m", 0.03)
        )
        self.deformation_graph_rotation_bound_rad = math.radians(
            float(getattr(args, "deformation_graph_rotation_bound_deg", 12.0))
        )
        if self.deformation_graph_node_count < 2:
            raise ValueError("deformation graph requires at least two nodes")
        if not 1 <= self.deformation_graph_influence_count <= self.deformation_graph_node_count:
            raise ValueError("graph influences must be in [1,node_count]")
        if not 1 <= self.deformation_graph_edge_count < self.deformation_graph_node_count:
            raise ValueError("graph edges must be in [1,node_count-1]")
        for name, value in (
            ("radius_scale", self.deformation_graph_radius_scale),
            ("minimum_radius_m", self.deformation_graph_minimum_radius_m),
            ("translation_bound_m", self.deformation_graph_translation_bound_m),
            ("rotation_bound_rad", self.deformation_graph_rotation_bound_rad),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"deformation graph {name} must be finite and positive")
        self.deformation_graph_head = CanonicalDeformationGraphHead(
            int(dim), learned_gate=candidate == "gated_deformation_graph"
        )
        # The graph is a structural projection rather than an additive no-op,
        # so the legacy branch-parity assertion is inapplicable.  We replace it
        # with the exact identity-local-transform check below.
        self._long_horizon_zero_init_checked = True
        self._deformation_graph_identity_checked = False

    @staticmethod
    def _node_descriptor(topology: DeformationGraphTopology) -> torch.Tensor:
        centers = topology.node_centers
        mass = topology.node_mass * topology.node_valid.to(dtype=centers.dtype)
        mass_sum = mass.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        centroid = (centers * mass[..., None]).sum(dim=2) / mass_sum
        relative = centers - centroid[:, :, None]
        scale = torch.sqrt(
            (relative.square().sum(dim=-1) * mass).sum(
                dim=-1, keepdim=True
            )
            / mass_sum
        ).clamp_min(1e-4)
        normalized = relative / scale[..., None]
        mass_fraction = (mass / mass_sum).unsqueeze(-1)
        descriptor = torch.cat((normalized, mass_fraction), dim=-1)
        return descriptor * topology.node_valid[..., None]

    @staticmethod
    def _enforce_bound(value: torch.Tensor, bound: float) -> torch.Tensor:
        maximum = value.abs().amax(dim=(-2, -1), keepdim=True)
        scale = (float(bound) / maximum.clamp_min(1e-12)).clamp(max=1.0)
        return value * scale

    def _graph_local_transforms(
        self,
        *,
        encoded: torch.Tensor,
        action: torch.Tensor,
        point_features: torch.Tensor,
        canonical: torch.Tensor,
        ids: torch.Tensor,
        scene_valid: torch.Tensor,
        object_valid: torch.Tensor,
    ) -> tuple[DeformationGraphTopology, dict[str, torch.Tensor]]:
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
        raw_twist, gate = self.deformation_graph_head(
            encoded,
            action,
            node_features,
            self._node_descriptor(topology),
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
            centered[..., :3], self.deformation_graph_translation_bound_m
        )
        rotation_vector = self._enforce_bound(
            centered[..., 3:], self.deformation_graph_rotation_bound_rad
        )
        translation = translation * gate[..., None]
        rotation_vector = rotation_vector * gate[..., None]
        local_rotation = so3_exp_map(rotation_vector.float()).to(encoded.dtype)
        outputs = {
            "deformation_graph_raw_twist": raw_twist,
            "deformation_graph_local_translation": translation,
            "deformation_graph_local_rotation_vector": rotation_vector,
            "deformation_graph_local_rotation": local_rotation,
            "deformation_graph_gate": gate,
            "deformation_graph_node_centers": topology.node_centers,
            "deformation_graph_node_valid": topology.node_valid,
            "deformation_graph_point_node_indices": topology.point_node_indices,
            "deformation_graph_point_node_weights": topology.point_node_weights,
            "deformation_graph_node_neighbors": topology.node_neighbors,
            "deformation_graph_node_neighbor_valid": topology.node_neighbor_valid,
            "deformation_graph_node_mass": topology.node_mass,
        }
        return topology, outputs

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
        rotation, translation, outputs = super()._apply_score_se3_pose_adapter(
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
        action = self._explicit_action_features(
            encoded,
            hand_relative,
            hand_delta,
            hand_side_validity(hand_valid),
            initial_rotation,
        )
        _topology, graph_outputs = self._graph_local_transforms(
            encoded=encoded,
            action=action,
            point_features=point_features,
            canonical=canonical,
            ids=ids,
            scene_valid=scene_valid,
            object_valid=object_valid,
        )
        outputs.update(graph_outputs)
        return rotation, translation, outputs

    @staticmethod
    def _topology_from_outputs(outputs: dict[str, torch.Tensor]) -> DeformationGraphTopology:
        return DeformationGraphTopology(
            node_centers=outputs["deformation_graph_node_centers"],
            node_valid=outputs["deformation_graph_node_valid"],
            point_node_indices=outputs["deformation_graph_point_node_indices"],
            point_node_weights=outputs["deformation_graph_point_node_weights"],
            node_neighbors=outputs["deformation_graph_node_neighbors"],
            node_neighbor_valid=outputs["deformation_graph_node_neighbor_valid"],
            node_mass=outputs["deformation_graph_node_mass"],
        )

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
        # First reproduce the trained metric_pose output exactly; it serves as a dense
        # set of pose votes.  The final returned points do not bypass the graph.
        corrected = super()._apply_score_se3_output_adapter(
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
        horizons = int(corrected.shape[1] - 1)
        global_rotation = score_se3_outputs[
            "score_se3_adapted_object_rotation"
        ].clone().float()
        global_translation = score_se3_outputs[
            "score_se3_adapted_object_translation"
        ].clone().float()
        fit_valid = torch.zeros(
            batch, objects, horizons, dtype=torch.bool, device=corrected.device
        )
        rigid_projection = corrected[:, 1:].clone()
        for batch_index in range(batch):
            for object_index in range(objects):
                mask = scene_valid[batch_index].bool() & (
                    ids[batch_index] == object_index
                )
                if not bool(object_valid[batch_index, object_index]) or mask.sum() < 3:
                    continue
                fitted_rotation, fitted_translation, valid = weighted_kabsch_projection(
                    canonical[batch_index, mask],
                    corrected[batch_index, 1:, mask],
                    # Clone the fallbacks before CopySlices updates below.
                    # This prevents an autograd version-counter conflict when
                    # Phase B differentiates the graph loss through Kabsch.
                    fallback_rotation=global_rotation[
                        batch_index, object_index
                    ].clone(),
                    fallback_translation=global_translation[
                        batch_index, object_index
                    ].clone(),
                )
                global_rotation[batch_index, object_index] = fitted_rotation
                global_translation[batch_index, object_index] = fitted_translation
                fit_valid[batch_index, object_index] = valid
                rigid_projection[batch_index, :, mask] = torch.einsum(
                    "tij,nj->tni", fitted_rotation, canonical[batch_index, mask]
                ) + fitted_translation[:, None]

        topology = self._topology_from_outputs(score_se3_outputs)
        graph_future = warp_points_with_deformation_graph(
            canonical,
            ids,
            scene_valid,
            topology,
            score_se3_outputs["deformation_graph_local_rotation"],
            score_se3_outputs["deformation_graph_local_translation"],
            global_rotation,
            global_translation,
            fallback_points=corrected[:, 1:],
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
            }
        )
        if not self._deformation_graph_identity_checked:
            maximum_twist = float(
                score_se3_outputs["deformation_graph_raw_twist"]
                .detach()
                .abs()
                .max()
            )
            maximum_difference = float(
                (graph_future.detach() - rigid_projection.detach()).abs().max()
            )
            if maximum_twist > 1e-7 or maximum_difference > 2e-5:
                raise RuntimeError(
                    "deformation graph is not an identity local warp at initialization: "
                    f"max_twist={maximum_twist:.3e}, "
                    f"max_projection_delta={maximum_difference:.3e}"
                )
            self._deformation_graph_identity_checked = True
        return torch.cat((corrected[:, :1], graph_future), dim=1)


__all__ = [
    "CanonicalDeformationGraphHead",
    "DEFORMATION_GRAPH_CANDIDATES",
    "DeformationGraphStructuredPredictor",
    "GRAPH_ENABLED_CANDIDATES",
]
