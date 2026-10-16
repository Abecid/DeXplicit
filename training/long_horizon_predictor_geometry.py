"""PointWorld-specific contact and canonical-deformation mechanisms.

Research provenance
-------------------
Stage 3 borrows JointHOI's *representation principle*: temporally varying,
continuous distance-based contact maps with separate left/right channels
(arXiv:2607.01768).  It is not a port of JointHOI's joint diffusion model; the
public Code link did not resolve to accessible implementation code during the
2026-08-28 audit.

Stage 4 is our synthesis: JointHOI-style contact distances route the known hand
trajectory locally.  LPWM (arXiv:2603.04553; official code at
https://github.com/taldatech/lpwm) motivates entity/particle-specific transition
context, but LPWM does not implement this Gaussian PointWorld routing rule.

Stage 5 borrows only ContactFlow's paper-specified 7-vector
``[x,y,z, dx,dy,dz, confidence]`` (arXiv:2607.26579).  ContactFlow projects that
signal into a sparse 7-channel video condition for VACE/ControlNet and trains a
video latent flow-matching model.  This module instead decodes an object-frame
3D residual.  The official repository
https://github.com/rpl-bonn/contactflow stated that implementation/checkpoints
were still under construction when audited, so this is a paper-spec adaptation,
not an upstream code port.

The Gaussian distance scale, route floor, small MLPs, mean-free correction, and
metric residual bounds are explicitly project pilot choices.  See
``training/long_horizon_provenance.py`` for the full borrowed/changed/custom
ledger.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from training.long_horizon_common import (
    contact_strength,
    hand_side_trajectory_features,
)


class LongHorizonGeometryMixin:
    def _hand_tokens(self, device: torch.device, dtype: torch.dtype):
        if self._long_horizon_batch is None:
            raise RuntimeError("set_long_horizon_context must be called before forward")
        hand_xyz = self._long_horizon_batch["robot_flows"].to(
            device=device, dtype=torch.float32
        )
        hand_valid = self._long_horizon_batch["robot_exists"].to(device=device).bool()
        centroid, relative, delta, side_valid = hand_side_trajectory_features(
            hand_xyz, hand_valid
        )
        time = torch.linspace(0, 1, hand_xyz.shape[1], device=device).view(
            1, -1, 1, 1
        )
        time = time.expand(hand_xyz.shape[0], -1, 2, -1)
        features = torch.cat([centroid, relative, delta, time], dim=-1)
        with torch.autocast(device_type=device.type, enabled=False):
            token = self.long_horizon_hand_encoder(features.float())
        token = torch.where(
            side_valid.unsqueeze(-1), token, torch.zeros_like(token)
        )
        return (
            hand_xyz.to(dtype),
            hand_valid,
            centroid.to(dtype),
            relative.to(dtype),
            delta.to(dtype),
            side_valid,
            token.to(dtype),
        )

    def _predict_contact_distance(
        self,
        point_features: torch.Tensor,
        hand_token: torch.Tensor,
        side_valid: torch.Tensor,
    ) -> torch.Tensor:
        # JointHOI motivates a continuous dynamic distance map rather than a
        # thresholded contact bit.  The normalized dot-product head and
        # softplus/expm1 metric parameterization below are our deterministic
        # PointWorld adaptation; they are not copied from JointHOI code.
        point_query = F.normalize(
            self.long_horizon_contact_point(point_features.float()),
            dim=-1,
            eps=1e-6,
        )
        hand_key = F.normalize(
            self.long_horizon_contact_hand(hand_token.float()),
            dim=-1,
            eps=1e-6,
        )
        similarity = torch.einsum("bnr,bthr->btnh", point_query, hand_key)
        raw_log = F.softplus(
            self.long_horizon_contact_bias.float()
            - similarity * math.sqrt(float(self.long_horizon_contact_rank))
        )
        distance = self.long_horizon_contact_tau * torch.expm1(raw_log).clamp_min(
            0.0
        )
        return torch.where(
            side_valid[:, :, None], distance, torch.full_like(distance, 1.0)
        )

    @staticmethod
    def _mean_free(
        residual: torch.Tensor,
        ids: torch.Tensor,
        valid: torch.Tensor,
        object_valid: torch.Tensor,
    ) -> torch.Tensor:
        # Project-specific identifiability constraint: object translation must
        # remain in the SE(3) head rather than being duplicated by deformation.
        output = residual.clone()
        batch = residual.shape[0]
        for batch_index in range(batch):
            for object_index in range(object_valid.shape[1]):
                if not bool(object_valid[batch_index, object_index]):
                    continue
                mask = valid[batch_index].bool() & (
                    ids[batch_index] == object_index
                )
                if mask.any():
                    output[batch_index, :, mask] -= output[
                        batch_index, :, mask
                    ].mean(dim=1, keepdim=True)
        return output

    def _dense_local_residual(
        self,
        point_features: torch.Tensor,
        ids: torch.Tensor,
        valid: torch.Tensor,
        object_valid: torch.Tensor,
    ) -> torch.Tensor:
        # New PointWorld synthesis motivated by SPOT's coherent object SE(3)
        # state and ParticleFormer's global/local dynamics supervision.  No cited
        # work supplies this exact mean-free canonical residual decoder.
        batch, points, _ = point_features.shape
        raw = self.long_horizon_local_residual(point_features).view(
            batch, points, self.T - 1, 3
        )
        raw = raw.permute(0, 2, 1, 3)
        raw = torch.tanh(raw) * self.long_horizon_max_deformation
        raw = self._mean_free(raw, ids, valid, object_valid)
        return torch.cat([raw.new_zeros(batch, 1, points, 3), raw], dim=1)

    def _contact_route_and_flow(
        self,
        base_residual: torch.Tensor,
        contact_distance: torch.Tensor,
        canonical: torch.Tensor,
        ids: torch.Tensor,
        valid: torch.Tensor,
        object_valid: torch.Tensor,
        initial_rotation: torch.Tensor,
        hand_relative: torch.Tensor,
        hand_delta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # The Gaussian influence field is our metric contact-routing hypothesis,
        # not a published default.  LONG_HORIZON_CONTACT_TAU_M must be calibrated
        # on HOT3D rather than treated as a canonical JointHOI/LPWM value.
        strength = contact_strength(
            contact_distance, self.long_horizon_contact_tau
        )
        output = base_residual.clone()
        route_correction = torch.zeros_like(base_residual)
        flow_correction = torch.zeros_like(base_residual)
        for batch_index in range(base_residual.shape[0]):
            for object_index in range(object_valid.shape[1]):
                if not bool(object_valid[batch_index, object_index]):
                    continue
                mask = valid[batch_index].bool() & (
                    ids[batch_index] == object_index
                )
                if not mask.any():
                    continue
                rotation0_t = initial_rotation[
                    batch_index, object_index
                ].transpose(0, 1)
                relative_local = torch.einsum(
                    "ij,thj->thi", rotation0_t, hand_relative[batch_index]
                )
                delta_local = torch.einsum(
                    "ij,thj->thi", rotation0_t, hand_delta[batch_index]
                )
                point_strength = strength[batch_index, :, mask]

                if self._has("contact_gated_action_routing"):
                    # Project synthesis: route known hand displacement through
                    # the learned contact field. LPWM motivates per-entity action
                    # context but does not contain this 3-D linear route head.
                    routed = torch.tanh(
                        self.long_horizon_route_head(relative_local.float())
                    )
                    routed = (
                        routed.to(output.dtype)
                        * self.long_horizon_max_deformation
                    )
                    correction = torch.einsum(
                        "tnh,thc->tnc", point_strength, routed
                    )
                    gate = self.long_horizon_route_floor + (
                        1.0 - self.long_horizon_route_floor
                    ) * point_strength.max(dim=-1).values
                    output[batch_index, :, mask] *= gate.unsqueeze(-1)
                    output[batch_index, :, mask] += correction
                    route_correction[batch_index, :, mask] = correction.to(route_correction.dtype)

                if self._has("contact_flow_tokens"):
                    # ContactFlow paper-spec adaptation. The paper's seven
                    # values are [contact XYZ, contact displacement XYZ,
                    # confidence]. Here XYZ is a contact-weighted canonical
                    # anchor and displacement is the known object-frame hand
                    # delta. Unlike ContactFlow, this token is NOT rendered as a
                    # sparse 7-channel VACE/ControlNet video condition.
                    canonical_points = canonical[batch_index, mask].float()
                    denominator = point_strength.sum(dim=1).clamp_min(1e-6)
                    anchor = torch.einsum(
                        "tnh,nc->thc",
                        point_strength.float(),
                        canonical_points,
                    ) / denominator.unsqueeze(-1)
                    confidence = point_strength.mean(dim=1).unsqueeze(-1)
                    token = torch.cat(
                        [anchor, delta_local.float(), confidence.float()],
                        dim=-1,
                    )
                    flow_delta = torch.tanh(
                        self.long_horizon_contact_flow_head(token)
                    )
                    flow_delta = (
                        flow_delta.to(output.dtype)
                        * self.long_horizon_max_deformation
                    )
                    correction = torch.einsum(
                        "tnh,thc->tnc", point_strength, flow_delta
                    )
                    output[batch_index, :, mask] += correction
                    flow_correction[batch_index, :, mask] = correction.to(flow_correction.dtype)

        output = self._mean_free(output, ids, valid, object_valid)
        return output, route_correction, flow_correction
