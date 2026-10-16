"""Leak-free HOT3D dynamics variants used only by train_hot3d_dynamics.py."""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from pointworld.base import BaseModel, MLP, SpatialEmbedding, TemporalEmbedding, build_ptv3

ROT_W, TRANS_W, VEL_W, ACC_W, DISP_W = 2.0, 20.0, 2.0, 2.0, 10.0


def rotation_6d_to_matrix(x: torch.Tensor) -> torch.Tensor:
    a, b = x[..., :3], x[..., 3:6]
    u = F.normalize(a, dim=-1, eps=1e-8)
    v = F.normalize(b - (u * b).sum(-1, keepdim=True) * u, dim=-1, eps=1e-8)
    return torch.stack((u, v, torch.cross(u, v, dim=-1)), dim=-1)


def rotation_geodesic_radians(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    r = a.transpose(-1, -2) @ b
    c = ((r.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5).clamp(-1 + 1e-6, 1 - 1e-6)
    return torch.acos(c)


def kabsch_transform(source: torch.Tensor, target: torch.Tensor):
    if source.shape != target.shape or source.ndim != 2 or source.shape[-1] != 3:
        raise ValueError("Kabsch inputs must match (N,3)")
    # cuSOLVER SVD/det are not implemented for bf16; always run in float32.
    src_dtype = source.dtype
    source = source.float()
    target = target.float()
    sc, tc = source.mean(0), target.mean(0)
    if source.shape[0] < 3:
        eye = torch.eye(3, device=source.device, dtype=src_dtype)
        return eye, (tc - sc).to(src_dtype)
    with torch.autocast(device_type=source.device.type, enabled=False):
        u, _, vh = torch.linalg.svd((source - sc).T @ (target - tc), full_matrices=False)
        r = vh.T @ u.T
        if torch.det(r) < 0:
            vh = vh.clone()
            vh[-1] *= -1
            r = vh.T @ u.T
    return r.to(src_dtype), (tc - sc @ r.T).to(src_dtype)


def compute_action_deltas(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 4 or x.shape[-1] != 3:
        raise ValueError(f"Expected hand points (B,T,N,3), got {tuple(x.shape)}")
    prev = torch.cat([x[:, :1], x[:, :-1]], 1)
    out = x - prev
    out[:, 0] = 0
    return out


def _masked_mean(x, mask, dim):
    w = mask.to(x.dtype).unsqueeze(-1)
    return (x * w).sum(dim) / w.sum(dim).clamp_min(1)


def _masked_max(x, mask, dim):
    y = x.masked_fill(~mask.unsqueeze(-1), torch.finfo(x.dtype).min).max(dim).values
    return torch.where(mask.any(dim).unsqueeze(-1), y, torch.zeros_like(y))


def _expand_mask(error, mask):
    return (mask.unsqueeze(-1) if error.ndim == mask.ndim + 1 else mask).expand_as(error)


def masked_mse(error, mask):
    m = _expand_mask(error, mask)
    return error[m].square().mean() if m.any() else error.sum() * 0


def horizon_weights(n, start, end, device, dtype):
    w = torch.linspace(start, end, n, device=device, dtype=dtype)
    return w / w.mean().clamp_min(torch.finfo(dtype).eps)


class TemporalActionBackbone(nn.Module):
    """Absolute PointWorld hand geometry plus optional H_t-H_(t-1) action tokens."""

    def __init__(self, args: Any, dim: int, mode: str):
        super().__init__()
        if mode not in {"absolute_tokens", "delta_tokens"}:
            raise ValueError(mode)
        self.mode, self.dim = mode, dim
        self.ptv3 = build_ptv3(dim, args)
        # Only used for delta_tokens; omit in absolute_tokens so DDP does not see
        # unused parameters (find_unused_parameters=False by default).
        self.delta_embed = SpatialEmbedding(dim) if mode == "delta_tokens" else None
        self.pool = MLP(2 * dim, dim, dim)
        self.time = TemporalEmbedding(dim)
        layer = nn.TransformerEncoderLayer(
            dim,
            int(getattr(args, "hand_action_encoder_heads", 4)),
            dim * 4,
            float(getattr(args, "hand_action_encoder_dropout", 0.1)),
            "gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(
            layer,
            int(getattr(args, "hand_action_encoder_layers", 2)),
            nn.LayerNorm(dim),
        )
        self.skip_g = nn.Parameter(torch.ones(1, 1, dim))
        self.skip_b = nn.Parameter(torch.zeros(1, 1, dim))
        self.act_g = nn.Parameter(torch.ones(1, 1, dim))
        self.act_b = nn.Parameter(torch.zeros(1, 1, dim))
        self.register_buffer("grid", torch.tensor([float(args.grid_size)]), persistent=False)

    def forward(self, scene_xyz, scene_feat, scene_valid, hand_xyz, hand_feat, hand_valid):
        B, N, _ = scene_xyz.shape
        T, R = hand_xyz.shape[1:3]
        device = scene_xyz.device
        with torch.autocast(device_type=device.type, enabled=False):
            hand_feat = hand_feat.float()
            if self.delta_embed is not None:
                hand_feat = hand_feat + self.delta_embed(compute_action_deltas(hand_xyz).float())
        coord = torch.cat([scene_xyz, hand_xyz.reshape(B, T * R, 3)], 1)
        feat = torch.cat([scene_feat, hand_feat.reshape(B, T * R, self.dim)], 1)
        exists = torch.cat([scene_valid, hand_valid.reshape(B, T * R)], 1)
        is_hand = torch.cat(
            [
                torch.zeros(B, N, dtype=torch.bool, device=device),
                torch.ones(B, T * R, dtype=torch.bool, device=device),
            ],
            1,
        )
        batch = torch.arange(B, device=device).repeat_interleave(N + T * R)
        keep, hand_flat = exists.reshape(-1), is_hand.reshape(-1)
        point = self.ptv3(
            {
                "coord": coord.reshape(-1, 3)[keep],
                "feat": feat.reshape(-1, self.dim)[keep],
                "batch": batch[keep],
                "is_robot": hand_flat[keep],
                "grid_size": self.grid,
            }
        )
        with torch.autocast(device_type=device.type, enabled=False):
            scene_out = point.feat.new_zeros(B, N, self.dim)
            scene_out[scene_valid] = point.feat[~hand_flat[keep]]
            hand_out = point.feat.new_zeros(B, T * R, self.dim)
            hand_out[hand_valid.reshape(B, T * R)] = point.feat[hand_flat[keep]]
            hand_out = hand_out.view(B, T, R, self.dim)
            token = self.pool(
                torch.cat(
                    [
                        _masked_mean(hand_out, hand_valid, 2),
                        _masked_max(hand_out, hand_valid, 2),
                    ],
                    -1,
                )
            )
            time_values = torch.linspace(0, 1, T, device=device).view(1, T)
            token = token + self.time(time_values)
            time_valid = hand_valid.any(2)
            padding_mask = ~time_valid
            all_padding = padding_mask.all(1)
            if all_padding.any():
                padding_mask = padding_mask.clone()
                padding_mask[all_padding] = False
            action = self.temporal(token, src_key_padding_mask=padding_mask)
            action = torch.where(time_valid.unsqueeze(-1), action, torch.zeros_like(action))
            global_action = _masked_max(action, time_valid, 1)[:, None]
            scene = (
                scene_out
                + scene_feat * self.skip_g
                + self.skip_b
                + global_action * self.act_g
                + self.act_b
            )
        return scene, action


class ActionDirectPredictor(nn.Module):
    def __init__(self, args, dim, T, action_mode):
        super().__init__()
        self.T = T
        self.backbone = TemporalActionBackbone(args, dim, action_mode)
        self.head = nn.Sequential(MLP(dim, 128, 128), nn.Linear(128, 3 * (T - 1)))
        self.var = nn.Sequential(MLP(dim, 128, 128), nn.Linear(128, T - 1))
        with torch.no_grad():
            nn.init.kaiming_normal_(self.head[-1].weight)
            self.head[-1].weight.mul_(float(args.dynamics_head_init_scale))
            self.head[-1].bias.zero_()
            self.var[-1].bias.fill_(math.log(0.005**2))

    def forward(self, sx, sf, sv, hx, hf, hv, **_):
        B, N = sx.shape[:2]
        scene, _ = self.backbone(sx, sf, sv, hx, hf, hv)
        prediction = self.head(scene).view(B, N, self.T - 1, 3).permute(0, 2, 1, 3)
        log_var = self.var(scene).permute(0, 2, 1).unsqueeze(-1)
        return {
            "pred": torch.cat([prediction.new_zeros(B, 1, N, 3), prediction], 1),
            "log_var": torch.cat([log_var.new_zeros(B, 1, N, 1), log_var], 1),
        }


class LocalIncrementPredictor(nn.Module):
    def __init__(self, args, dim, T, action_mode):
        super().__init__()
        self.T = T
        self.backbone = TemporalActionBackbone(args, dim, action_mode)
        self.head = nn.Sequential(MLP(dim, 128, 128), nn.Linear(128, 3))
        self.var = nn.Sequential(MLP(dim, 128, 128), nn.Linear(128, 1))
        with torch.no_grad():
            nn.init.kaiming_normal_(self.head[-1].weight)
            self.head[-1].weight.mul_(float(args.dynamics_head_init_scale))
            self.head[-1].bias.zero_()
            self.var[-1].bias.fill_(math.log(0.005**2))

    def forward(self, sx, sf, sv, hx, hf, hv, **_):
        B, N = sx.shape[:2]
        scene, action = self.backbone(sx, sf, sv, hx, hf, hv)
        temporal_features = scene[:, None] + action[:, 1:, None]
        local_delta = self.head(temporal_features)
        local_log_var = self.var(temporal_features)
        local_delta = torch.cat([local_delta.new_zeros(B, 1, N, 3), local_delta], 1)

        # The base PointWorld objective is on cumulative positions. Assuming
        # independent local errors, cumulative variance is the sum of local
        # variances, not the raw local variance relabeled at every horizon.
        local_var = torch.exp(local_log_var.float()).clamp_min(1e-12)
        cumulative_log_var = torch.log(torch.cumsum(local_var, dim=1)).to(local_log_var.dtype)
        cumulative_log_var = torch.cat(
            [cumulative_log_var.new_zeros(B, 1, N, 1), cumulative_log_var], 1
        )
        return {
            "pred": torch.cumsum(local_delta, 1),
            "log_var": cumulative_log_var,
            "local_delta_norm": local_delta,
            "local_log_var": torch.cat(
                [local_log_var.new_zeros(B, 1, N, 1), local_log_var], 1
            ),
        }


class StructuredSE3Predictor(nn.Module):
    """Flexible dense prediction plus an ObjectForesight-style SE(3) correction path."""

    def __init__(self, args, dim, T, action_mode):
        super().__init__()
        self.T, self.dim = T, dim
        self.backbone = TemporalActionBackbone(args, dim, action_mode)
        self.direct = nn.Sequential(MLP(dim, 128, 128), nn.Linear(128, 3 * (T - 1)))
        self.var = nn.Sequential(MLP(dim, 128, 128), nn.Linear(128, T - 1))
        self.canonical = SpatialEmbedding(dim)
        self.object_pool = MLP(2 * dim, dim, dim)
        layer = nn.TransformerEncoderLayer(
            dim, 8, dim * 4, 0.1, "gelu", batch_first=True, norm_first=True
        )
        self.object_time = nn.TransformerEncoder(layer, 8, nn.LayerNorm(dim))
        self.pose = nn.Sequential(MLP(dim, 128, 128), nn.Linear(128, 9))
        self.blend = nn.Linear(dim, 1)
        self.object_ids = None
        self.canonical_xyz = None
        with torch.no_grad():
            nn.init.kaiming_normal_(self.direct[-1].weight)
            self.direct[-1].weight.mul_(float(args.dynamics_head_init_scale))
            self.direct[-1].bias.zero_()
            self.var[-1].bias.fill_(math.log(0.005**2))
            self.pose[-1].weight.zero_()
            self.pose[-1].bias.zero_()
            self.pose[-1].bias[3:9] = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
            self.blend.weight.zero_()
            self.blend.bias.zero_()

    def set_object_context(self, ids, canonical):
        self.object_ids, self.canonical_xyz = ids, canonical

    def _keep_structured_params_live(self, scene, action):
        pooled = self.object_pool(torch.cat([scene.mean(1), scene.amax(1)], -1))
        tokens = pooled[:, None] + action[:, :1]
        encoded = self.object_time(tokens)
        return (pooled.sum() + encoded.sum() + self.pose(encoded).sum() + self.blend(encoded).sum()) * 0

    def forward(self, sx, sf, sv, hx, hf, hv, **_):
        if self.object_ids is None or self.canonical_xyz is None:
            raise KeyError("structured object context missing")
        ids = self.object_ids.to(sx.device)
        canonical = self.canonical_xyz.to(sx.device, sx.dtype)
        B, N = sx.shape[:2]
        scene, action = self.backbone(sx, sf, sv, hx, hf, hv)
        raw = self.direct(scene).view(B, N, self.T - 1, 3).permute(0, 2, 1, 3)
        raw = torch.cat([raw.new_zeros(B, 1, N, 3), raw], 1)
        direct_abs = sx[:, None] + raw
        log_var = self.var(scene).permute(0, 2, 1).unsqueeze(-1)
        log_var = torch.cat([log_var.new_zeros(B, 1, N, 1), log_var], 1)

        num_objects = int(ids.max().item()) + 1 if (ids >= 0).any() else 0
        if num_objects == 0:
            # Keep object heads in the autograd graph so DDP ranks stay aligned
            # when some ranks have no labelled objects this step.
            raw = raw + self._keep_structured_params_live(scene, action)
            return {"pred": raw, "log_var": log_var}
        object_features = scene.new_zeros(B, num_objects, self.dim)
        object_valid = torch.zeros(B, num_objects, dtype=torch.bool, device=sx.device)
        initial_rotation = (
            torch.eye(3, device=sx.device, dtype=sx.dtype)
            .view(1, 1, 3, 3)
            .repeat(B, num_objects, 1, 1)
        )
        initial_translation = sx.new_zeros(B, num_objects, 3)
        point_features = scene + self.canonical(canonical.float())
        for batch_index in range(B):
            for object_id in torch.unique(ids[batch_index][sv[batch_index]]):
                object_index = int(object_id.item())
                mask = sv[batch_index].bool() & (ids[batch_index] == object_index)
                if object_index < 0 or mask.sum() < 3:
                    continue
                features = point_features[batch_index, mask]
                object_features[batch_index, object_index] = self.object_pool(
                    torch.cat([features.mean(0), features.max(0).values], -1)
                )
                rotation, translation = kabsch_transform(
                    canonical[batch_index, mask].float(), sx[batch_index, mask].float()
                )
                initial_rotation[batch_index, object_index] = rotation.to(sx.dtype)
                initial_translation[batch_index, object_index] = translation.to(sx.dtype)
                object_valid[batch_index, object_index] = True

        live = scene.sum() * 0
        if not object_valid.any():
            live = self._keep_structured_params_live(scene, action)

        tokens = object_features[:, :, None] + action[:, None, 1:]
        flat = tokens.reshape(B, num_objects * (self.T - 1), self.dim)
        valid = object_valid[:, :, None].expand(-1, -1, self.T - 1).reshape(B, -1)
        padding_mask = ~valid
        all_padding = padding_mask.all(1)
        if all_padding.any():
            padding_mask = padding_mask.clone()
            padding_mask[all_padding] = False
        encoded = self.object_time(flat, src_key_padding_mask=padding_mask)
        encoded = torch.where(valid.unsqueeze(-1), encoded, torch.zeros_like(encoded))
        encoded = encoded.view(B, num_objects, self.T - 1, self.dim)
        pose_parameters = self.pose(encoded)
        rotation = rotation_6d_to_matrix(pose_parameters[..., 3:9]) @ initial_rotation[:, :, None]
        translation = initial_translation[:, :, None] + pose_parameters[..., :3]
        gate = torch.tanh(self.blend(encoded).squeeze(-1))

        pose_abs = direct_abs.clone()
        output = direct_abs.clone()
        for batch_index in range(B):
            for object_index in range(num_objects):
                if not object_valid[batch_index, object_index]:
                    continue
                mask = sv[batch_index].bool() & (ids[batch_index] == object_index)
                posed = (
                    torch.einsum(
                        "tij,nj->tni",
                        rotation[batch_index, object_index],
                        canonical[batch_index, mask],
                    )
                    + translation[batch_index, object_index, :, None, :]
                )
                pose_abs[batch_index, 1:, mask] = posed
                blend = gate[batch_index, object_index, :, None, None]
                output[batch_index, 1:, mask] = (
                    direct_abs[batch_index, 1:, mask]
                    + blend * (posed - direct_abs[batch_index, 1:, mask])
                )
        output = output + live
        return {
            "pred": output - sx[:, None],
            "log_var": log_var,
            "direct_scene_flows": direct_abs,
            "pose_scene_flows": pose_abs,
            "object_rotation": torch.cat([initial_rotation[:, :, None], rotation], 2),
            "object_translation": torch.cat([initial_translation[:, :, None], translation], 2),
            "object_valid": object_valid,
            "object_blend": gate,
        }


class V2ExperimentalBaseModel(BaseModel):
    def __init__(self, args, data_info_dict, rank=0, cpu_pg=None):
        super().__init__(args, data_info_dict, rank=rank, cpu_pg=cpu_pg)
        self.experiment_mode = str(getattr(args, "dynamics_experiment", "direct"))
        default = "pointworld" if self.experiment_mode == "direct" else "delta_tokens"
        self.action_mode = str(getattr(args, "action_conditioning_mode", default))
        if self.experiment_mode == "direct" and self.action_mode != "pointworld":
            self.dynamics_predictor = ActionDirectPredictor(
                args, self.channels, self.T, self.action_mode
            )
        elif self.experiment_mode == "local_increment":
            self.dynamics_predictor = LocalIncrementPredictor(
                args, self.channels, self.T, self.action_mode
            )
        elif self.experiment_mode == "structured":
            self.dynamics_predictor = StructuredSE3Predictor(
                args, self.channels, self.T, self.action_mode
            )

    def forward(self, data_dict, training=True, encoded_scene_feat0=None):
        if isinstance(self.dynamics_predictor, StructuredSE3Predictor):
            self.dynamics_predictor.set_object_context(
                data_dict.get("point_object_id"), data_dict.get("point_canonical_xyz")
            )
        outputs = BaseModel.forward(
            self,
            data_dict,
            training=training,
            encoded_scene_feat0=encoded_scene_feat0,
        )
        if "local_delta_norm" in outputs:
            outputs["scene_delta"] = outputs["local_delta_norm"]
        return outputs

    @staticmethod
    def _valid(data, device):
        return (
            ~data["scene_context_mask"].to(device).squeeze(-1).bool()
            & data["scene_exists"].to(device).squeeze(-1).bool()
            & data["scene_supervised_mask"].to(device).squeeze(-1).bool()
        )

    @torch.no_grad()
    def _gt_pose(self, gt, canonical, ids, valid, num_objects):
        B, T = gt.shape[:2]
        rotation = (
            torch.eye(3, device=gt.device, dtype=gt.dtype)
            .view(1, 1, 1, 3, 3)
            .repeat(B, num_objects, T, 1, 1)
        )
        translation = gt.new_zeros(B, num_objects, T, 3)
        valid_pose = torch.zeros(
            B, num_objects, T, dtype=torch.bool, device=gt.device
        )
        for batch_index in range(B):
            for object_index in range(num_objects):
                object_mask = ids[batch_index] == object_index
                for time_index in range(T):
                    mask = object_mask & valid[batch_index, time_index]
                    if mask.sum() < 3:
                        continue
                    current_rotation, current_translation = kabsch_transform(
                        canonical[batch_index, mask].float(),
                        gt[batch_index, time_index, mask].float(),
                    )
                    rotation[batch_index, object_index, time_index] = current_rotation.to(gt.dtype)
                    translation[batch_index, object_index, time_index] = current_translation.to(gt.dtype)
                    valid_pose[batch_index, object_index, time_index] = True
        return rotation, translation, valid_pose

    def loss_fn(self, outputs, data, training=True):
        loss, log = BaseModel.loss_fn(self, outputs, data, training=training)
        if self.experiment_mode == "direct":
            return loss, log

        predicted = outputs["scene_flows"]
        target = data["gt_scene_flows"].to(predicted.device)
        valid = self._valid(data, predicted.device)
        predicted_delta = torch.cat(
            [predicted[:, :1] * 0, predicted[:, 1:] - predicted[:, :-1]], 1
        )
        target_delta = torch.cat(
            [target[:, :1] * 0, target[:, 1:] - target[:, :-1]], 1
        )
        increment_valid = valid
        if "scene_delta_valid" in data:
            increment_valid = valid & data["scene_delta_valid"].to(predicted.device).squeeze(-1).bool()
        increment_mse = masked_mse(predicted_delta - target_delta, increment_valid)
        increment_weight = float(
            getattr(self.args, "local_increment_loss_weight", 0)
        )
        loss = loss + increment_weight * increment_mse
        log.update(
            {
                "local_increment_mse": float(increment_mse.detach().cpu()),
                "local_increment_weighted": float(
                    (increment_weight * increment_mse).detach().cpu()
                ),
            }
        )
        if self.experiment_mode != "structured":
            log["total_loss"] = float(loss.detach().cpu())
            return loss, log

        ids = data["point_object_id"].to(predicted.device)
        canonical = data["point_canonical_xyz"].to(predicted.device).float()
        num_objects = outputs["object_valid"].shape[1]
        valid_tracks = (
            data["scene_exists"].to(predicted.device).squeeze(-1).bool()
            & data["scene_supervised_mask"].to(predicted.device).squeeze(-1).bool()
        )
        gt_rotation, gt_translation, gt_valid = self._gt_pose(
            target, canonical, ids, valid_tracks, num_objects
        )
        predicted_rotation = outputs["object_rotation"]
        predicted_translation = outputs["object_translation"]
        pose_valid = gt_valid & outputs["object_valid"][:, :, None]
        future_valid = pose_valid[:, :, 1:]
        weights = horizon_weights(
            predicted.shape[1] - 1,
            float(self.args.horizon_weight_start),
            float(self.args.horizon_weight_end),
            predicted.device,
            predicted.dtype,
        )
        weighted_mask = future_valid.to(predicted.dtype) * weights.view(1, 1, -1)
        denominator = weighted_mask.sum().clamp_min(1)
        translation_loss = (
            torch.linalg.norm(
                predicted_translation[:, :, 1:] - gt_translation[:, :, 1:], dim=-1
            )
            * weighted_mask
        ).sum() / denominator
        rotation_loss = (
            rotation_geodesic_radians(
                predicted_rotation[:, :, 1:], gt_rotation[:, :, 1:]
            )
            * weighted_mask
        ).sum() / denominator
        pose_loss = TRANS_W * translation_loss + ROT_W * rotation_loss
        loss = loss + pose_loss

        predicted_translation_delta = (
            predicted_translation[:, :, 1:] - predicted_translation[:, :, :-1]
        )
        gt_translation_delta = gt_translation[:, :, 1:] - gt_translation[:, :, :-1]
        velocity_valid = pose_valid[:, :, 1:] & pose_valid[:, :, :-1]
        predicted_relative_rotation = (
            predicted_rotation[:, :, :-1].transpose(-1, -2)
            @ predicted_rotation[:, :, 1:]
        )
        gt_relative_rotation = (
            gt_rotation[:, :, :-1].transpose(-1, -2) @ gt_rotation[:, :, 1:]
        )
        velocity_rotation_error = rotation_geodesic_radians(
            predicted_relative_rotation, gt_relative_rotation
        )
        velocity_loss = masked_mse(
            predicted_translation_delta - gt_translation_delta, velocity_valid
        ) + (
            velocity_rotation_error[velocity_valid].square().mean()
            if velocity_valid.any()
            else velocity_rotation_error.sum() * 0
        )
        velocity_weight = float(
            getattr(self.args, "velocity_loss_weight", VEL_W)
        )
        loss = loss + velocity_weight * velocity_loss

        predicted_acceleration = (
            predicted_translation_delta[:, :, 1:]
            - predicted_translation_delta[:, :, :-1]
        )
        gt_acceleration = (
            gt_translation_delta[:, :, 1:] - gt_translation_delta[:, :, :-1]
        )
        acceleration_valid = velocity_valid[:, :, 1:] & velocity_valid[:, :, :-1]
        predicted_rotation_acceleration = (
            predicted_relative_rotation[:, :, :-1].transpose(-1, -2)
            @ predicted_relative_rotation[:, :, 1:]
        )
        gt_rotation_acceleration = (
            gt_relative_rotation[:, :, :-1].transpose(-1, -2)
            @ gt_relative_rotation[:, :, 1:]
        )
        acceleration_rotation_error = rotation_geodesic_radians(
            predicted_rotation_acceleration, gt_rotation_acceleration
        )
        acceleration_loss = masked_mse(
            predicted_acceleration - gt_acceleration, acceleration_valid
        ) + (
            acceleration_rotation_error[acceleration_valid].square().mean()
            if acceleration_valid.any()
            else acceleration_rotation_error.sum() * 0
        )
        acceleration_weight = float(
            getattr(self.args, "acceleration_loss_weight", ACC_W)
        )
        loss = loss + acceleration_weight * acceleration_loss

        endpoint_valid = pose_valid[:, :, 0] & pose_valid[:, :, -1]
        endpoint_error = torch.linalg.norm(
            (predicted_translation[:, :, -1] - predicted_translation[:, :, 0])
            - (gt_translation[:, :, -1] - gt_translation[:, :, 0]),
            dim=-1,
        )
        displacement_loss = (
            endpoint_error[endpoint_valid].mean()
            if endpoint_valid.any()
            else endpoint_error.sum() * 0
        )
        displacement_weight = float(
            getattr(self.args, "object_displacement_loss_weight", DISP_W)
        )
        loss = loss + displacement_weight * displacement_loss

        with torch.no_grad():
            moved = data["scene_moved_mask"].to(predicted.device).squeeze(-1).bool()
            moved_valid = valid & moved
            if moved_valid.any():
                log["structured/direct_l2_moved"] = float(
                    torch.linalg.norm(
                        outputs["direct_scene_flows"] - target, dim=-1
                    )[moved_valid]
                    .mean()
                    .cpu()
                )
                log["structured/pose_l2_moved"] = float(
                    torch.linalg.norm(
                        outputs["pose_scene_flows"] - target, dim=-1
                    )[moved_valid]
                    .mean()
                    .cpu()
                )
            blend_valid = outputs["object_valid"][:, :, None].expand_as(
                outputs["object_blend"]
            )
            if blend_valid.any():
                log["structured/blend_abs_mean"] = float(
                    outputs["object_blend"][blend_valid].abs().mean().cpu()
                )
        log.update(
            {
                "structured/translation_l2": float(translation_loss.detach().cpu()),
                "structured/rotation_rad": float(rotation_loss.detach().cpu()),
                "structured/pose_weighted": float(pose_loss.detach().cpu()),
                "structured/velocity": float(velocity_loss.detach().cpu()),
                "structured/acceleration": float(acceleration_loss.detach().cpu()),
                "structured/displacement": float(displacement_loss.detach().cpu()),
                "total_loss": float(loss.detach().cpu()),
            }
        )
        return loss, log
