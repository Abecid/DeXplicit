"""Trainable low-rank deformation and carrier-mixture heads."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from training.carrier_gauge_core import so3_exp
from training.carrier_graph_signals import (
    clamped_bspline_basis,
    graph_laplacian_basis,
    temporal_project,
)


def _zero_last_linear(module: nn.Module) -> None:
    linears = [child for child in module.modules() if isinstance(child, nn.Linear)]
    if not linears:
        raise ValueError("module has no Linear layer")
    with torch.no_grad():
        linears[-1].weight.zero_()
        if linears[-1].bias is not None:
            linears[-1].bias.zero_()


@dataclass(frozen=True)
class LowRankGraphOutput:
    raw_mode_twist: torch.Tensor
    projected_mode_twist: torch.Tensor
    temporal_coefficients: torch.Tensor
    spatial_basis: torch.Tensor
    node_twist: torch.Tensor
    local_translation: torch.Tensor
    local_rotation_vector: torch.Tensor
    local_rotation: torch.Tensor
    gate: torch.Tensor


@dataclass(frozen=True)
class CarrierMixOutput:
    weights: torch.Tensor
    logits: torch.Tensor
    strength: torch.Tensor
    local_translation: torch.Tensor
    local_rotation_vector: torch.Tensor


class LowRankCarrierGraphHead(nn.Module):
    """Graph-spectral deformation with a B-spline temporal bottleneck."""

    def __init__(
        self,
        dim: int,
        horizons: int,
        *,
        mode_rank: int = 4,
        temporal_controls: int = 6,
        action_dim: int = 17,
        relation_dim: int = 18,
        use_relations: bool = False,
        use_carriers: bool = False,
        translation_bound_m: float = 0.03,
        rotation_bound_rad: float = math.radians(12.0),
        carrier_translation_bound_m: float = 0.05,
        carrier_rotation_bound_rad: float = math.radians(15.0),
    ) -> None:
        super().__init__()
        if mode_rank <= 0 or temporal_controls < 4 or horizons <= 0:
            raise ValueError("invalid low-rank graph dimensions")
        self.dim = int(dim)
        self.horizons = int(horizons)
        self.mode_rank = int(mode_rank)
        self.temporal_controls = int(temporal_controls)
        self.action_dim = int(action_dim)
        self.relation_dim = int(relation_dim)
        self.use_relations = bool(use_relations)
        self.use_carriers = bool(use_carriers)
        self.translation_bound_m = float(translation_bound_m)
        self.rotation_bound_rad = float(rotation_bound_rad)
        self.carrier_translation_bound_m = float(
            carrier_translation_bound_m
        )
        self.carrier_rotation_bound_rad = float(carrier_rotation_bound_rad)
        self.register_buffer(
            "temporal_basis",
            clamped_bspline_basis(
                self.horizons, self.temporal_controls, degree=3
            ),
        )

        mode_input = 2 * self.dim + self.action_dim + 4
        if self.use_relations:
            mode_input += self.relation_dim
        hidden = max(128, self.dim)
        self.mode_head = nn.Sequential(
            nn.LayerNorm(mode_input),
            nn.Linear(mode_input, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 6),
        )
        _zero_last_linear(self.mode_head)
        self.gate = nn.Sequential(
            nn.LayerNorm(self.dim + self.action_dim),
            nn.Linear(
                self.dim + self.action_dim, max(64, self.dim // 2)
            ),
            nn.GELU(),
            nn.Linear(max(64, self.dim // 2), 1),
        )
        with torch.no_grad():
            self.gate[-1].weight.zero_()
            self.gate[-1].bias.fill_(math.log(0.05 / 0.95))

        if self.use_carriers:
            carrier_input = self.dim + self.action_dim + 7
            carrier_hidden = max(96, self.dim // 2)
            self.carrier_score = nn.Sequential(
                nn.LayerNorm(carrier_input),
                nn.Linear(carrier_input, carrier_hidden),
                nn.GELU(),
                nn.Linear(carrier_hidden, 1),
            )
            self.carrier_strength = nn.Sequential(
                nn.LayerNorm(self.dim + self.action_dim),
                nn.Linear(
                    self.dim + self.action_dim, carrier_hidden
                ),
                nn.GELU(),
                nn.Linear(carrier_hidden, 1),
            )
            _zero_last_linear(self.carrier_strength)
        else:
            self.carrier_score = None
            self.carrier_strength = None

    def forward(
        self,
        encoded: torch.Tensor,
        action: torch.Tensor,
        node_features: torch.Tensor,
        node_descriptor: torch.Tensor,
        topology: Any,
        *,
        relation_features: torch.Tensor | None = None,
    ) -> LowRankGraphOutput:
        batch, objects, horizons, channels = encoded.shape
        nodes = topology.node_centers.shape[2]
        if horizons != self.horizons or channels != self.dim:
            raise ValueError("encoded shape does not match configured head")
        if action.shape != (batch, objects, horizons, self.action_dim):
            raise ValueError("invalid action feature shape")
        if node_features.shape != (batch, objects, nodes, self.dim):
            raise ValueError("invalid node feature shape")
        if node_descriptor.shape != (batch, objects, nodes, 4):
            raise ValueError("invalid node descriptor shape")

        spatial = graph_laplacian_basis(topology, self.mode_rank)
        node_context = torch.cat((node_features, node_descriptor), dim=-1)
        mode_context = torch.einsum(
            "bokr,bokd->bord", spatial, node_context
        )
        inputs = [
            encoded[:, :, :, None].expand(
                -1, -1, -1, self.mode_rank, -1
            ),
            action[:, :, :, None].expand(
                -1, -1, -1, self.mode_rank, -1
            ),
            mode_context[:, :, None].expand(
                -1, -1, horizons, -1, -1
            ),
        ]
        if self.use_relations:
            expected = (
                batch,
                objects,
                horizons,
                nodes,
                self.relation_dim,
            )
            if relation_features is None or relation_features.shape != expected:
                raise ValueError(
                    f"relation_features must have shape {expected}"
                )
            inputs.append(
                torch.einsum(
                    "bokr,botkf->botrf", spatial, relation_features
                )
            )

        raw_mode = self.mode_head(torch.cat(inputs, dim=-1))
        projected_mode, coefficients = temporal_project(
            raw_mode, self.temporal_basis
        )
        node_twist = torch.einsum(
            "bokr,botrc->botkc", spatial, projected_mode
        )
        valid = topology.node_valid[:, :, None, :, None]
        mass = (
            topology.node_mass.to(node_twist.dtype)
            * topology.node_valid.to(node_twist.dtype)
        )[:, :, None, :, None]
        mean = (node_twist * mass).sum(3, keepdim=True) / mass.sum(
            3, keepdim=True
        ).clamp_min(1e-12)
        node_twist = (node_twist - mean) * valid
        gate = torch.sigmoid(
            self.gate(torch.cat((encoded, action), dim=-1))
        )
        translation = self.translation_bound_m * torch.tanh(
            node_twist[..., :3]
        )
        rotation_vector = self.rotation_bound_rad * torch.tanh(
            node_twist[..., 3:]
        )
        translation = translation * gate[..., None] * valid
        rotation_vector = rotation_vector * gate[..., None] * valid
        return LowRankGraphOutput(
            raw_mode_twist=raw_mode,
            projected_mode_twist=projected_mode,
            temporal_coefficients=coefficients,
            spatial_basis=spatial,
            node_twist=node_twist,
            local_translation=translation,
            local_rotation_vector=rotation_vector,
            local_rotation=so3_exp(rotation_vector.float()).to(encoded.dtype),
            gate=gate,
        )

    def mix_carriers(
        self,
        encoded: torch.Tensor,
        action: torch.Tensor,
        candidate_features: torch.Tensor,
        candidate_translation: torch.Tensor,
        candidate_rotation_vector: torch.Tensor,
        candidate_valid: torch.Tensor,
    ) -> CarrierMixOutput:
        """Predict an exact-zero-initialized mixture of kinematic candidates."""
        if self.carrier_score is None or self.carrier_strength is None:
            raise RuntimeError("carrier mixer is disabled")
        candidates = candidate_features.shape[-2]
        repeated_encoded = encoded[..., None, :].expand(
            *encoded.shape[:-1], candidates, encoded.shape[-1]
        )
        repeated_action = action[..., None, :].expand(
            *action.shape[:-1], candidates, action.shape[-1]
        )
        logits = self.carrier_score(
            torch.cat(
                (
                    repeated_encoded,
                    repeated_action,
                    candidate_features,
                ),
                dim=-1,
            )
        ).squeeze(-1)
        logits = logits.masked_fill(~candidate_valid.bool(), -1e4)
        weights = torch.softmax(logits, dim=-1)
        strength = torch.tanh(
            self.carrier_strength(torch.cat((encoded, action), dim=-1))
        )
        bounded_translation = self.carrier_translation_bound_m * torch.tanh(
            candidate_translation / self.carrier_translation_bound_m
        )
        bounded_rotation = self.carrier_rotation_bound_rad * torch.tanh(
            candidate_rotation_vector / self.carrier_rotation_bound_rad
        )
        mixed_translation = (
            weights[..., None] * bounded_translation
        ).sum(-2)
        mixed_rotation = (weights[..., None] * bounded_rotation).sum(-2)
        return CarrierMixOutput(
            weights=weights,
            logits=logits,
            strength=strength,
            local_translation=strength * mixed_translation,
            local_rotation_vector=strength * mixed_rotation,
        )


__all__ = [
    "CarrierMixOutput",
    "LowRankCarrierGraphHead",
    "LowRankGraphOutput",
]
