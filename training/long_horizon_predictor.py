from __future__ import annotations

import math
import os
from typing import Any

import torch
import torch.nn as nn

from pointworld.base import MLP
from training.dynamics_experiment_v2 import (
    StructuredSE3Predictor, kabsch_transform, rotation_6d_to_matrix,
)
from training.long_horizon_common import (
    _env_bool, _env_float, _env_int, _inverse_softplus, _zero_last_linear,
    normalize_track, stage_features, stage_name,
)
from training.long_horizon_predictor_geometry import LongHorizonGeometryMixin
from training.long_horizon_predictor_particles import LongHorizonParticleMixin

class LongHorizonStructuredPredictor(LongHorizonGeometryMixin, LongHorizonParticleMixin, StructuredSE3Predictor):

    def __init__(self, args: Any, dim: int, T: int, action_mode: str, stage: int):
        super().__init__(args, dim, T, action_mode)
        self.long_horizon_stage = int(stage)
        stage_name(self.long_horizon_stage)
        self.long_horizon_track = normalize_track()
        self.long_horizon_features = frozenset(
            stage_features(self.long_horizon_stage, self.long_horizon_track)
        )
        self.long_horizon_max_deformation = _env_float('LONG_HORIZON_MAX_DEFORMATION_M', 0.08)
        self.long_horizon_contact_tau = _env_float('LONG_HORIZON_CONTACT_TAU_M', 0.02)
        self.long_horizon_contact_rank = _env_int('LONG_HORIZON_CONTACT_RANK', 32)
        self.long_horizon_route_floor = _env_float('LONG_HORIZON_ROUTE_FLOOR', 0.15)
        self.long_horizon_particle_count = _env_int('LONG_HORIZON_PARTICLES', 16)
        self.long_horizon_particle_temperature = _env_float('LONG_HORIZON_PARTICLE_TEMPERATURE', 0.5)
        self.long_horizon_local_residual = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, 3 * (T - 1)))
        _zero_last_linear(self.long_horizon_local_residual)
        self.long_horizon_hand_encoder = MLP(10, dim, dim)
        self.long_horizon_contact_point = nn.Linear(dim, self.long_horizon_contact_rank)
        self.long_horizon_contact_hand = nn.Linear(dim, self.long_horizon_contact_rank)
        target_log = math.log1p(0.1 / max(self.long_horizon_contact_tau, 1e-06))
        self.long_horizon_contact_bias = nn.Parameter(torch.tensor(_inverse_softplus(target_log), dtype=torch.float32))
        self.long_horizon_route_head = nn.Linear(3, 3, bias=False)
        with torch.no_grad():
            self.long_horizon_route_head.weight.zero_()
        self.long_horizon_contact_flow_head = nn.Sequential(nn.Linear(7, 64), nn.GELU(), nn.Linear(64, 3))
        _zero_last_linear(self.long_horizon_contact_flow_head)
        self.long_horizon_particle_assignment = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, self.long_horizon_particle_count))
        self.long_horizon_particle_contact = nn.Linear(dim, dim, bias=False)
        self.long_horizon_particle_motion = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, 3))
        _zero_last_linear(self.long_horizon_particle_motion)
        self.long_horizon_physical_probe = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, 21))
        self._long_horizon_batch: dict[str, torch.Tensor] | None = None
        self._long_horizon_zero_init_checked = False
        # FROM-SCRATCH FIX: the base predictor zero-inits the pose head (fine
        # for warm starts, where the backbone/pose already work and only NEW
        # branches must be silent). In these stages every supervised object
        # point is produced by the pose path alone, so a zero-init pose head
        # means zero upstream gradient at step 0 and a dead bootstrap when
        # training from scratch (verified empirically: 1,700 steps with the
        # eval frozen at the zero-motion baseline). With
        # LONG_HORIZON_POSE_HEAD_INIT=scaled the pose output layer gets the
        # same kaiming*scale init the v2 dense head uses; the 6d-rotation bias
        # keeps the identity offset so initial poses stay near-identity.
        # Method-branch zero-init (and paired attribution) is unaffected.
        pose_init = os.environ.get('LONG_HORIZON_POSE_HEAD_INIT', 'zero')
        if pose_init not in ('zero', 'scaled'):
            raise ValueError(f"LONG_HORIZON_POSE_HEAD_INIT must be 'zero' or 'scaled', got {pose_init!r}")
        self.long_horizon_pose_head_init = pose_init
        if pose_init == 'scaled':
            # NOTE: args.dynamics_head_init_scale is 0.0 for HOT3D domains, so
            # this uses its own scale (default 0.01 -> ~cm-level initial pose
            # noise) rather than inheriting a zero multiplier.
            with torch.no_grad():
                nn.init.kaiming_normal_(self.pose[-1].weight)
                self.pose[-1].weight.mul_(_env_float('LONG_HORIZON_POSE_HEAD_INIT_SCALE', 0.01))
        # Output composition is part of the ARCHITECTURE, not a runtime
        # preference: it is resolved once here (invalid values are fatal) and
        # persisted in the state dict via a buffer, so a checkpoint keeps its
        # semantics even when loaded in a process without the env var.
        output_mode = os.environ.get('LONG_HORIZON_OUTPUT_MODE', 'pose')
        if output_mode not in ('pose', 'blend'):
            raise ValueError(f"LONG_HORIZON_OUTPUT_MODE must be 'pose' or 'blend', got {output_mode!r}")
        self.register_buffer(
            'long_horizon_output_mode_flag',
            torch.tensor(1 if output_mode == 'blend' else 0, dtype=torch.long),
        )

    @property
    def long_horizon_output_mode(self) -> str:
        return 'blend' if int(self.long_horizon_output_mode_flag.item()) == 1 else 'pose'

    def _has(self, feature: str) -> bool:
        return feature in self.long_horizon_features

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
        """Extension hook; the continued_baseline/metric_loss predictor is an exact no-op."""

        return rotation, translation, {}

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
        """Extension hook applied after the legacy direct/pose blend."""

        return output_abs

    def set_long_horizon_context(self, data: dict[str, Any]) -> None:
        hand_xyz = data.get('robot_flows')
        hand_valid = data.get('robot_exists')
        if not isinstance(hand_xyz, torch.Tensor) or not isinstance(hand_valid, torch.Tensor):
            raise KeyError('Long-horizon ablations require robot_flows and robot_exists')
        from training.phystwin_controls import unlabelled_rows
        self._long_horizon_batch = {
            'robot_flows': hand_xyz, 'robot_exists': hand_valid,
            'action_is_unlabelled': unlabelled_rows(data, hand_xyz.shape[0], hand_xyz.device),
        }

    def forward(self, sx, sf, sv, hx, hf, hv, **_):
        if self.object_ids is None or self.canonical_xyz is None:
            raise KeyError('structured object context missing')
        ids = self.object_ids.to(sx.device).long()
        canonical = self.canonical_xyz.to(sx.device, sx.dtype)
        batch, point_count = sx.shape[:2]
        scene, action = self.backbone(sx, sf, sv, hx, hf, hv)
        point_features = scene + self.canonical(canonical.float())
        direct_raw = self.direct(scene).view(batch, point_count, self.T - 1, 3).permute(0, 2, 1, 3)
        direct_raw = torch.cat([direct_raw.new_zeros(batch, 1, point_count, 3), direct_raw], dim=1)
        direct_abs = sx[:, None] + direct_raw
        log_var = self.var(scene).permute(0, 2, 1).unsqueeze(-1)
        log_var = torch.cat([log_var.new_zeros(batch, 1, point_count, 1), log_var], dim=1)
        object_count = int(ids.max().item()) + 1 if (ids >= 0).any() else 0
        if object_count == 0:
            _hand_xyz, _hand_valid, hand_centroid, _hand_relative, _hand_delta, side_valid, hand_token = self._hand_tokens(sx.device, sx.dtype)
            contact_distance = self._predict_contact_distance(point_features, hand_token, side_valid)
            live = self._keep_new_parameters_live(scene)
            empty_object = torch.zeros(batch, 0, dtype=torch.bool, device=sx.device)
            return {'pred': direct_raw + live, 'log_var': log_var, 'direct_scene_flows': direct_abs, 'pose_scene_flows': direct_abs, 'long_horizon_scene_flows': direct_abs, 'long_horizon_local_residual': direct_abs.new_zeros(batch, self.T, point_count, 3), 'long_horizon_route_correction': direct_abs.new_zeros(batch, self.T, point_count, 3), 'long_horizon_contact_flow_correction': direct_abs.new_zeros(batch, self.T, point_count, 3), 'long_horizon_contact_distance': contact_distance, 'long_horizon_particle_assignment': direct_abs.new_zeros(batch, point_count, self.long_horizon_particle_count), 'long_horizon_particle_motion': direct_abs.new_zeros(batch, 0, self.T, self.long_horizon_particle_count, 3), 'long_horizon_physical_probe': direct_abs.new_zeros(batch, 0, self.T - 1, 21), 'long_horizon_object_hidden': direct_abs.new_zeros(batch, 0, self.T - 1, self.dim), 'long_horizon_hand_centroid': hand_centroid, 'long_horizon_hand_valid': side_valid, 'object_rotation': direct_abs.new_zeros(batch, 0, self.T, 3, 3), 'object_translation': direct_abs.new_zeros(batch, 0, self.T, 3), 'object_valid': empty_object, 'object_blend': direct_abs.new_zeros(batch, 0, self.T - 1)}
        object_features = scene.new_zeros(batch, object_count, self.dim)
        object_valid = torch.zeros(batch, object_count, dtype=torch.bool, device=sx.device)
        initial_rotation = torch.eye(3, device=sx.device, dtype=sx.dtype).view(1, 1, 3, 3).repeat(batch, object_count, 1, 1)
        initial_translation = sx.new_zeros(batch, object_count, 3)
        for batch_index in range(batch):
            for object_id in torch.unique(ids[batch_index][sv[batch_index].bool()]):
                object_index = int(object_id.item())
                mask = sv[batch_index].bool() & (ids[batch_index] == object_index)
                if object_index < 0 or mask.sum() < 3:
                    continue
                features = point_features[batch_index, mask]
                object_features[batch_index, object_index] = self.object_pool(torch.cat([features.mean(dim=0), features.max(dim=0).values], dim=-1))
                rotation0, translation0 = kabsch_transform(canonical[batch_index, mask].float(), sx[batch_index, mask].float())
                initial_rotation[batch_index, object_index] = rotation0.to(sx.dtype)
                initial_translation[batch_index, object_index] = translation0.to(sx.dtype)
                object_valid[batch_index, object_index] = True
        tokens = object_features[:, :, None] + action[:, None, 1:]
        flat = tokens.reshape(batch, object_count * (self.T - 1), self.dim)
        valid_tokens = object_valid[:, :, None].expand(-1, -1, self.T - 1).reshape(batch, -1)
        padding_mask = ~valid_tokens
        all_padding = padding_mask.all(dim=1)
        if all_padding.any():
            padding_mask = padding_mask.clone()
            padding_mask[all_padding] = False
        encoded_flat = self.object_time(flat, src_key_padding_mask=padding_mask)
        encoded_flat = torch.where(valid_tokens.unsqueeze(-1), encoded_flat, torch.zeros_like(encoded_flat))
        encoded = encoded_flat.view(batch, object_count, self.T - 1, self.dim)
        pose_parameters = self.pose(encoded)
        rotation = rotation_6d_to_matrix(pose_parameters[..., 3:9]) @ initial_rotation[:, :, None]
        translation = initial_translation[:, :, None] + pose_parameters[..., :3]
        gate = torch.tanh(self.blend(encoded).squeeze(-1))
        hand_xyz, hand_valid, hand_centroid, hand_relative, hand_delta, side_valid, hand_token = self._hand_tokens(sx.device, sx.dtype)
        contact_distance = self._predict_contact_distance(point_features, hand_token, side_valid)
        rotation, translation, score_se3_outputs = self._apply_score_se3_pose_adapter(
            encoded=encoded,
            rotation=rotation,
            translation=translation,
            initial_rotation=initial_rotation,
            object_valid=object_valid,
            canonical=canonical,
            ids=ids,
            scene_valid=sv,
            contact_distance=contact_distance,
            hand_xyz=hand_xyz,
            hand_valid=hand_valid,
            hand_relative=hand_relative,
            hand_delta=hand_delta,
            point_features=point_features,
        )
        pose_abs = direct_abs.clone()
        for batch_index in range(batch):
            for object_index in range(object_count):
                if not bool(object_valid[batch_index, object_index]):
                    continue
                mask = sv[batch_index].bool() & (ids[batch_index] == object_index)
                posed = torch.einsum('tij,nj->tni', rotation[batch_index, object_index], canonical[batch_index, mask]) + translation[batch_index, object_index, :, None]
                pose_abs[batch_index, 1:, mask] = posed
        if self._has('particle_deformation_bottleneck'):
            local_residual, assignments, particle_motion = self._particle_residual(point_features, encoded, contact_distance, hand_token, ids, sv, object_valid)
        elif self._has('object_local_residual'):
            local_residual = self._dense_local_residual(point_features, ids, sv, object_valid)
            assignments = pose_abs.new_zeros(batch, point_count, self.long_horizon_particle_count)
            particle_motion = pose_abs.new_zeros(batch, object_count, self.T, self.long_horizon_particle_count, 3)
        else:
            local_residual = pose_abs.new_zeros(batch, self.T, point_count, 3)
            assignments = pose_abs.new_zeros(batch, point_count, self.long_horizon_particle_count)
            particle_motion = pose_abs.new_zeros(batch, object_count, self.T, self.long_horizon_particle_count, 3)
        route_correction = torch.zeros_like(local_residual)
        flow_correction = torch.zeros_like(local_residual)
        if self._has('contact_gated_action_routing') or self._has('contact_flow_tokens'):
            local_residual, route_correction, flow_correction = self._contact_route_and_flow(local_residual, contact_distance, canonical, ids, sv, object_valid, initial_rotation, hand_relative, hand_delta)
        # Output composition. 'pose' (PR default) makes every object point a
        # pure SE(3)+residual reconstruction - correct for warm starts but
        # untrainable from scratch (the supervised loss never touches the
        # dense head, and the pose bottleneck cannot bootstrap: verified
        # empirically, evals frozen at the zero-motion baseline).
        # LONG_HORIZON_OUTPUT_MODE=blend restores the validated v2 SOTA
        # composition direct + gate*(pose_world - direct): the dense head
        # carries from-scratch learning while the zero-init gate brings the
        # object path in gradually. Stage differences (residual/contact/
        # particle branches) still enter only through the zero-init method
        # heads, so cross-stage init parity is preserved.
        blend_output = self.long_horizon_output_mode == 'blend'
        output_abs = direct_abs.clone() if blend_output else pose_abs.clone()
        # Zero-init parity reference: the SAME composition with every method
        # branch at zero (residual/route/flow excluded). From scratch this
        # equals direct_abs (gate zero-init); on a warm start with a trained
        # gate it equals exactly what the loaded checkpoint would predict.
        reference_abs = output_abs.clone()
        for batch_index in range(batch):
            for object_index in range(object_count):
                if not bool(object_valid[batch_index, object_index]):
                    continue
                mask = sv[batch_index].bool() & (ids[batch_index] == object_index)
                local = canonical[batch_index, mask][None] + local_residual[batch_index, 1:, mask]
                world = torch.einsum('tij,tnj->tni', rotation[batch_index, object_index], local) + translation[batch_index, object_index, :, None]
                if blend_output:
                    blend_gate = gate[batch_index, object_index][:, None, None]
                    output_abs[batch_index, 1:, mask] = direct_abs[batch_index, 1:, mask] + blend_gate * (world - direct_abs[batch_index, 1:, mask])
                    reference_abs[batch_index, 1:, mask] = direct_abs[batch_index, 1:, mask] + blend_gate * (pose_abs[batch_index, 1:, mask] - direct_abs[batch_index, 1:, mask])
                else:
                    output_abs[batch_index, 1:, mask] = world
                    reference_abs[batch_index, 1:, mask] = pose_abs[batch_index, 1:, mask]
        # The score-SE3 adapter acts on the final warm-started trajectory, not
        # inside the legacy pose blend.  This prevents its rigid correction
        # from being attenuated by (or linearly mixed through) the blend gate.
        output_abs = self._apply_score_se3_output_adapter(
            output_abs,
            ids=ids,
            scene_valid=sv,
            rotation=rotation,
            translation=translation,
            object_valid=object_valid,
            score_se3_outputs=score_se3_outputs,
        )
        physical_probe = self.long_horizon_physical_probe(encoded)
        if not self._long_horizon_zero_init_checked and _env_bool('LONG_HORIZON_ASSERT_ZERO_INIT', True):
            maximum = float((output_abs.detach() - reference_abs.detach()).abs().max())
            if maximum > 1e-6:
                raise RuntimeError(
                    'Long-horizon branch perturbs the stage control at initialization: '
                    f'max_abs={maximum:.3e}. The ablation is not clean.'
                )
            self._long_horizon_zero_init_checked = True
        output_abs = output_abs + self._keep_new_parameters_live(output_abs)
        result = {'pred': output_abs - sx[:, None], 'log_var': log_var, 'direct_scene_flows': direct_abs, 'pose_scene_flows': pose_abs, 'long_horizon_scene_flows': output_abs, 'long_horizon_local_residual': local_residual, 'long_horizon_route_correction': route_correction, 'long_horizon_contact_flow_correction': flow_correction, 'long_horizon_contact_distance': contact_distance, 'long_horizon_particle_assignment': assignments, 'long_horizon_particle_motion': particle_motion, 'long_horizon_physical_probe': physical_probe, 'long_horizon_object_hidden': encoded, 'long_horizon_hand_centroid': hand_centroid, 'long_horizon_hand_valid': side_valid, 'object_rotation': torch.cat([initial_rotation[:, :, None], rotation], dim=2), 'object_translation': torch.cat([initial_translation[:, :, None], translation], dim=2), 'object_valid': object_valid, 'object_blend': gate}
        result.update(score_se3_outputs)
        return result
