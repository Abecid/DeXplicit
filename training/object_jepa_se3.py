"""Object-state JEPA regularization for 35-frame structured SE(3) PointWorld.

The proven action-conditioned PointWorld backbone and SE(3) pose decoder remain
unchanged.  This module adds one training-only prediction target per object and
future time step:

    persistent context pointwise-DINO semantics
    + GT object pose / velocity
    + GT object-local shape
        -> fixed, stop-gradient object-state latent

The existing action-conditioned ``object_time`` hidden state predicts that
latent.  The target encoder is deliberately *fixed* rather than a second learned
network: this avoids collapse/EMA plumbing and makes the experiment a controlled
one-variable auxiliary loss.  It never consumes future RGB or future DINO.
Future GT 3D is used only as a supervised target, just like the existing pose
and point-trajectory losses.

At inference the auxiliary predictor is unused.  The default experiment also
makes the empirically superior rigid ``pose_scene_flows`` the canonical output;
a paired weight-zero control isolates that output change from the JEPA loss.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from training.dynamics_experiment_v2 import (
    StructuredSE3Predictor,
    V2ExperimentalBaseModel,
)
from utils import _print


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """Encode the first two rotation columns in the repository convention."""
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"Expected (...,3,3), got {tuple(matrix.shape)}")
    return matrix[..., :3, :2].transpose(-1, -2).reshape(*matrix.shape[:-2], 6)


def object_pose_state(
    rotation: torch.Tensor,
    translation: torch.Tensor,
) -> torch.Tensor:
    """Return relative pose and one-step velocity as an 18-D object state.

    Args:
        rotation: ``(B,O,T,3,3)``.
        translation: ``(B,O,T,3)``.
    Returns:
        ``(B,O,T,18)`` containing relative rotation6d, relative translation,
        one-step rotation6d and one-step translation.
    """
    if rotation.ndim != 5 or rotation.shape[-2:] != (3, 3):
        raise ValueError("rotation must be (B,O,T,3,3)")
    if translation.shape != (*rotation.shape[:-2], 3):
        raise ValueError("translation must be (B,O,T,3)")

    rotation0 = rotation[:, :, :1]
    relative_rotation = rotation0.transpose(-1, -2) @ rotation
    relative_translation = translation - translation[:, :, :1]

    previous_rotation = torch.cat([rotation[:, :, :1], rotation[:, :, :-1]], dim=2)
    delta_rotation = previous_rotation.transpose(-1, -2) @ rotation
    previous_translation = torch.cat(
        [translation[:, :, :1], translation[:, :, :-1]], dim=2
    )
    delta_translation = translation - previous_translation

    identity = torch.eye(3, device=rotation.device, dtype=rotation.dtype)
    delta_rotation = delta_rotation.clone()
    delta_rotation[:, :, 0] = identity
    delta_translation = delta_translation.clone()
    delta_translation[:, :, 0] = 0

    return torch.cat(
        [
            matrix_to_rotation_6d(relative_rotation),
            relative_translation,
            matrix_to_rotation_6d(delta_rotation),
            delta_translation,
        ],
        dim=-1,
    )


def deterministic_object_indices(
    object_ids: torch.Tensor,
    context_valid: torch.Tensor,
    num_objects: int,
    max_points: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select an evenly spaced deterministic point subset for each object.

    Returns ``indices`` and ``valid`` with shape ``(B,O,K)``.  Stable selection
    prevents the latent target itself from jittering across workers or epochs.
    """
    if object_ids.ndim != 2 or context_valid.shape != object_ids.shape:
        raise ValueError("object_ids/context_valid must be matching (B,N) tensors")
    if num_objects < 1 or max_points < 1:
        raise ValueError("num_objects and max_points must be positive")

    batch = object_ids.shape[0]
    indices = torch.zeros(
        batch, num_objects, max_points, dtype=torch.long, device=object_ids.device
    )
    valid = torch.zeros(
        batch, num_objects, max_points, dtype=torch.bool, device=object_ids.device
    )
    for batch_index in range(batch):
        for object_index in range(num_objects):
            candidates = torch.nonzero(
                context_valid[batch_index]
                & (object_ids[batch_index] == object_index),
                as_tuple=False,
            ).flatten()
            if candidates.numel() == 0:
                continue
            if candidates.numel() > max_points:
                positions = torch.linspace(
                    0,
                    candidates.numel() - 1,
                    max_points,
                    device=candidates.device,
                ).round().long()
                candidates = candidates[positions]
            count = int(candidates.numel())
            indices[batch_index, object_index, :count] = candidates
            valid[batch_index, object_index, :count] = True
    return indices, valid


def _gather_bnd(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather ``(B,N,D)`` by ``(B,O,K)`` indices -> ``(B,O,K,D)``."""
    batch, _, channels = values.shape
    object_count, point_count = indices.shape[1:]
    expanded_values = values[:, None].expand(-1, object_count, -1, -1)
    expanded_indices = indices[..., None].expand(
        batch, object_count, point_count, channels
    )
    return torch.gather(expanded_values, 2, expanded_indices)


def _gather_btnd(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather ``(B,T,N,D)`` -> ``(B,T,O,K,D)``."""
    batch, timesteps, _, channels = values.shape
    object_count, point_count = indices.shape[1:]
    expanded_values = values[:, :, None].expand(-1, -1, object_count, -1, -1)
    expanded_indices = indices[:, None, :, :, None].expand(
        batch, timesteps, object_count, point_count, channels
    )
    return torch.gather(expanded_values, 3, expanded_indices)


def _gather_btn(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather ``(B,T,N)`` -> ``(B,T,O,K)``."""
    batch, timesteps, _ = values.shape
    object_count, point_count = indices.shape[1:]
    expanded_values = values[:, :, None].expand(-1, -1, object_count, -1)
    expanded_indices = indices[:, None].expand(
        batch, timesteps, object_count, point_count
    )
    return torch.gather(expanded_values, 3, expanded_indices)


def horizon_weighted_mean(
    values: torch.Tensor,
    valid: torch.Tensor,
    start: float,
    end: float,
) -> torch.Tensor:
    """Masked mean with linearly increasing, mean-normalized horizon weights."""
    if values.shape != valid.shape or values.ndim != 3:
        raise ValueError("values/valid must be matching (B,O,T) tensors")
    weights = torch.linspace(
        float(start),
        float(end),
        values.shape[-1],
        device=values.device,
        dtype=values.dtype,
    )
    weights = weights / weights.mean().clamp_min(torch.finfo(weights.dtype).eps)
    weighted_valid = valid.to(values.dtype) * weights.view(1, 1, -1)
    denominator = weighted_valid.sum()
    if denominator <= 0:
        return values.sum() * 0.0
    return (values * weighted_valid).sum() / denominator


def fixed_projection(input_dim: int, output_dim: int, seed: int) -> torch.Tensor:
    """Deterministic approximately orthonormal projection matrix.

    Fixed projections retain semantic/geometry distances without creating a
    learned target encoder that could collapse.  The matrix is saved in the
    checkpoint as a buffer, making the target contract fully reproducible.
    """
    if input_dim < 1 or output_dim < 1:
        raise ValueError("projection dimensions must be positive")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    matrix = torch.randn(input_dim, output_dim, generator=generator)
    if input_dim >= output_dim:
        matrix, _ = torch.linalg.qr(matrix, mode="reduced")
    else:
        matrix = F.normalize(matrix, dim=0)
    return matrix.float()


def _project_normalized(value: torch.Tensor, projection: torch.Tensor) -> torch.Tensor:
    value = F.layer_norm(value.float(), (value.shape[-1],))
    projected = value @ projection.to(value.device, value.dtype)
    return F.normalize(projected, dim=-1, eps=1e-6)


def _block_cosine_error(
    predicted: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    return 1.0 - F.cosine_similarity(
        predicted.float(), target.float(), dim=-1, eps=1e-6
    )


class _MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class ObjectStateJEPAV2ExperimentalBaseModel(V2ExperimentalBaseModel):
    """Structured PointWorld with a fixed-target object-state JEPA objective."""

    _JEPA_PREFIX = "object_jepa_"

    def __init__(self, args: Any, data_info_dict: dict[str, int], rank=0, cpu_pg=None):
        self.object_jepa_loss_weight = _env_float("OBJECT_JEPA_LOSS_WEIGHT", 0.1)
        self.object_jepa_semantic_weight = _env_float(
            "OBJECT_JEPA_SEMANTIC_WEIGHT", 0.25
        )
        self.object_jepa_pose_weight = _env_float("OBJECT_JEPA_POSE_WEIGHT", 1.0)
        self.object_jepa_shape_weight = _env_float("OBJECT_JEPA_SHAPE_WEIGHT", 0.5)
        self.object_jepa_delta_weight = _env_float("OBJECT_JEPA_DELTA_WEIGHT", 0.25)
        self.object_jepa_horizon_start = _env_float("OBJECT_JEPA_HORIZON_START", 1.0)
        self.object_jepa_horizon_end = _env_float("OBJECT_JEPA_HORIZON_END", 2.0)
        self.object_jepa_dim = _env_int("OBJECT_JEPA_DIM", 128)
        self.object_jepa_max_points = _env_int("OBJECT_JEPA_MAX_POINTS", 64)
        self.object_jepa_force_pose_output = _env_bool(
            "OBJECT_JEPA_FORCE_POSE_OUTPUT", True
        )
        self.object_jepa_enabled = self.object_jepa_loss_weight > 0

        for key, value in {
            "object_jepa_loss_weight": self.object_jepa_loss_weight,
            "object_jepa_semantic_weight": self.object_jepa_semantic_weight,
            "object_jepa_pose_weight": self.object_jepa_pose_weight,
            "object_jepa_shape_weight": self.object_jepa_shape_weight,
            "object_jepa_delta_weight": self.object_jepa_delta_weight,
            "object_jepa_horizon_start": self.object_jepa_horizon_start,
            "object_jepa_horizon_end": self.object_jepa_horizon_end,
            "object_jepa_dim": self.object_jepa_dim,
            "object_jepa_max_points": self.object_jepa_max_points,
            "object_jepa_force_pose_output": self.object_jepa_force_pose_output,
        }.items():
            setattr(args, key, value)
            original = getattr(args, "og_args", None)
            if original is not None:
                setattr(original, key, value)

        super().__init__(args, data_info_dict, rank=rank, cpu_pg=cpu_pg)
        if not isinstance(self.dynamics_predictor, StructuredSE3Predictor):
            raise TypeError(
                "Object-state JEPA requires --dynamics-experiment structured; got "
                f"{type(self.dynamics_predictor).__name__}."
            )
        if not bool(getattr(args, "identity_norm_stats", False)):
            raise ValueError(
                "Object-state JEPA canonical pose output currently requires "
                "--identity_norm_stats true."
            )
        if self.object_jepa_dim < 32:
            raise ValueError("OBJECT_JEPA_DIM must be at least 32")
        if self.object_jepa_max_points < 3:
            raise ValueError("OBJECT_JEPA_MAX_POINTS must be at least 3")
        for name in (
            "object_jepa_loss_weight",
            "object_jepa_semantic_weight",
            "object_jepa_pose_weight",
            "object_jepa_shape_weight",
            "object_jepa_delta_weight",
        ):
            if float(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be non-negative")

        self._object_jepa_hidden: torch.Tensor | None = None
        self._object_jepa_gt_pose_cache: tuple[torch.Tensor, ...] | None = None
        self._object_jepa_hook = self.dynamics_predictor.object_time.register_forward_hook(
            self._capture_object_time_hidden
        )

        self.object_jepa_semantic_dim = self.object_jepa_dim // 2
        self.object_jepa_pose_dim = (3 * self.object_jepa_dim) // 8
        self.object_jepa_shape_dim = (
            self.object_jepa_dim
            - self.object_jepa_semantic_dim
            - self.object_jepa_pose_dim
        )
        if self.object_jepa_shape_dim < 4:
            raise ValueError("OBJECT_JEPA_DIM leaves too few local-shape channels")

        if self.object_jepa_enabled:
            scene_dim = int(data_info_dict["scene_features_dim"])
            pose_fourier_dim = 18 * 4 * 2
            shape_input_dim = int(self.object_jepa_max_points) * 6
            self.register_buffer(
                "object_jepa_semantic_projection",
                fixed_projection(2 * scene_dim, self.object_jepa_semantic_dim, 202601),
            )
            self.register_buffer(
                "object_jepa_pose_projection",
                fixed_projection(pose_fourier_dim, self.object_jepa_pose_dim, 202602),
            )
            self.register_buffer(
                "object_jepa_shape_projection",
                fixed_projection(shape_input_dim, self.object_jepa_shape_dim, 202603),
            )
            self.object_jepa_predictor = _MLP(
                int(self.channels) + self.object_jepa_dim,
                2 * self.object_jepa_dim,
                self.object_jepa_dim,
            )

        self._write_object_jepa_contract(args)

    def _write_object_jepa_contract(self, args: Any) -> None:
        output_dir = getattr(args, "output_dir", None)
        if not output_dir or int(getattr(self, "rank", 0)) != 0:
            return
        path = Path(output_dir).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        payload = {
            "implementation": "object_state_jepa_v1_fixed_teacher",
            "enabled": bool(self.object_jepa_enabled),
            "force_pose_scene_flows": bool(self.object_jepa_force_pose_output),
            "loss_weight": float(self.object_jepa_loss_weight),
            "component_weights": {
                "semantic": float(self.object_jepa_semantic_weight),
                "pose": float(self.object_jepa_pose_weight),
                "local_shape": float(self.object_jepa_shape_weight),
                "temporal_delta": float(self.object_jepa_delta_weight),
            },
            "horizon_weights": [
                float(self.object_jepa_horizon_start),
                float(self.object_jepa_horizon_end),
            ],
            "latent_dim": int(self.object_jepa_dim),
            "latent_blocks": {
                "semantic": int(self.object_jepa_semantic_dim),
                "pose_velocity": int(self.object_jepa_pose_dim),
                "object_local_shape": int(self.object_jepa_shape_dim),
            },
            "max_points_per_object": int(self.object_jepa_max_points),
            "future_rgb_or_dino_target": False,
            "target_encoder_trainable": False,
            "target": (
                "fixed projection of persistent context pointwise-DINO semantics, "
                "GT SE(3) pose/velocity, and GT object-local point geometry"
            ),
        }
        (path / "object_jepa_contract.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )

    def _capture_object_time_hidden(self, _module, _inputs, output) -> None:
        self._object_jepa_hidden = output[0] if isinstance(output, tuple) else output

    @torch.no_grad()
    def _gt_pose(self, *args, **kwargs):
        result = super()._gt_pose(*args, **kwargs)
        self._object_jepa_gt_pose_cache = result
        return result

    def forward(self, data_dict, training=True, encoded_scene_feat0=None):
        self._object_jepa_hidden = None
        outputs = super().forward(
            data_dict,
            training=training,
            encoded_scene_feat0=encoded_scene_feat0,
        )
        if self.object_jepa_force_pose_output:
            pose = outputs.get("pose_scene_flows")
            if pose is None:
                raise RuntimeError("Structured model did not return pose_scene_flows")
            outputs["blended_scene_flows"] = outputs["scene_flows"]
            context = data_dict["scene_flows"][:, :1].to(pose.device, pose.dtype)
            relative = pose - context
            outputs["scene_flows"] = pose
            outputs["scene_relative"] = relative
            outputs["scene_relative_norm"] = self.normalize(relative)
            outputs["pred"] = outputs["scene_relative_norm"]
        return outputs

    def _all_jepa_parameters_live(self, reference: torch.Tensor) -> torch.Tensor:
        if not self.object_jepa_enabled:
            return reference.sum() * 0.0
        live = reference.sum() * 0.0
        for name, parameter in self.named_parameters():
            if name.startswith(self._JEPA_PREFIX):
                live = live + parameter.sum() * 0.0
        return live

    def _scene_semantics(self, data: dict[str, Any]) -> torch.Tensor:
        features = data.get("scene_features")
        if not isinstance(features, torch.Tensor):
            raise KeyError("Object-state JEPA requires batch['scene_features']")
        if features.ndim == 4:
            features = features[:, 0]
        if features.ndim != 3:
            raise ValueError(
                f"scene_features must be (B,T,N,D) or (B,N,D), got {tuple(features.shape)}"
            )
        return features.float()

    def _build_target_latents(
        self,
        data: dict[str, Any],
        object_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ids = data.get("point_object_id")
        canonical = data.get("point_canonical_xyz")
        if not isinstance(ids, torch.Tensor) or not isinstance(canonical, torch.Tensor):
            raise KeyError("Object-state JEPA requires object ids and canonical xyz")
        ids = ids.to(object_valid.device).long()
        canonical = canonical.to(object_valid.device).float()
        if ids.ndim != 2 or canonical.shape[:2] != ids.shape:
            raise ValueError("Object id/canonical tensors have inconsistent shapes")

        gt = data["gt_scene_flows"].to(object_valid.device).float()
        exists = data["scene_exists"].to(object_valid.device).squeeze(-1).bool()
        supervised = (
            data["scene_supervised_mask"].to(object_valid.device).squeeze(-1).bool()
        )
        context_valid = exists[:, 0] & (ids >= 0)
        object_count = int(object_valid.shape[1])
        indices, selected_valid = deterministic_object_indices(
            ids,
            context_valid,
            object_count,
            int(self.object_jepa_max_points),
        )

        semantics = _gather_bnd(
            self._scene_semantics(data).to(object_valid.device), indices
        )
        canonical_points = _gather_bnd(canonical, indices)
        gt_points = _gather_btnd(gt, indices)
        future_valid = _gather_btn(exists & supervised, indices)
        point_valid = future_valid & selected_valid[:, None]
        enough_points = point_valid.sum(dim=3).ge(3)

        semantic_mask = selected_valid[..., None]
        semantic_count = selected_valid.sum(dim=2, keepdim=True).clamp_min(1)
        semantic_mean = (semantics * semantic_mask).sum(dim=2) / semantic_count
        semantic_max_input = semantics.masked_fill(
            ~semantic_mask, torch.finfo(semantics.dtype).min
        )
        semantic_max = semantic_max_input.max(dim=2).values
        semantic_max = torch.where(
            selected_valid.any(dim=2, keepdim=True),
            semantic_max,
            torch.zeros_like(semantic_max),
        )
        semantic_descriptor = torch.cat([semantic_mean, semantic_max], dim=-1)
        semantic_latent = _project_normalized(
            semantic_descriptor, self.object_jepa_semantic_projection
        )

        pose_cache = self._object_jepa_gt_pose_cache
        if pose_cache is None:
            raise RuntimeError(
                "Base structured loss did not produce GT pose targets; keep the "
                "existing pose supervision enabled for object-state JEPA."
            )
        rotation, translation, pose_valid = pose_cache
        pose_state = object_pose_state(rotation.float(), translation.float())
        scaled_pose = pose_state.clone()
        scaled_pose[..., 6:9] *= 10.0
        scaled_pose[..., 15:18] *= 10.0
        frequencies = torch.tensor(
            [1.0, 2.0, 4.0, 8.0],
            device=scaled_pose.device,
            dtype=scaled_pose.dtype,
        )
        angles = math.pi * scaled_pose[..., None] * frequencies
        pose_fourier = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        pose_fourier = pose_fourier.flatten(start_dim=-2).permute(0, 2, 1, 3)
        pose_latent = _project_normalized(
            pose_fourier, self.object_jepa_pose_projection
        )

        rotation_bt = rotation.permute(0, 2, 1, 3, 4).float()
        translation_bt = translation.permute(0, 2, 1, 3).float()
        centered_world = gt_points - translation_bt[:, :, :, None]
        local_points = torch.einsum(
            "btoij,btokj->btoki",
            rotation_bt.transpose(-1, -2),
            centered_world,
        )
        canonical_bt = canonical_points[:, None].expand_as(local_points)
        local_residual = local_points - canonical_bt
        valid_xyz = point_valid[..., None]
        shape_values = torch.cat(
            [
                canonical_bt * valid_xyz * 10.0,
                local_residual * valid_xyz * 20.0,
            ],
            dim=-1,
        )
        shape_descriptor = shape_values.flatten(start_dim=3)
        shape_latent = _project_normalized(
            shape_descriptor, self.object_jepa_shape_projection
        )

        timesteps = int(gt.shape[1])
        semantic_latent = semantic_latent[:, None].expand(-1, timesteps, -1, -1)
        target = torch.cat([semantic_latent, pose_latent, shape_latent], dim=-1)
        target = F.normalize(target, dim=-1, eps=1e-6).detach()
        target_valid = (
            enough_points
            & pose_valid.permute(0, 2, 1)
            & object_valid[:, None]
        )
        return target, target_valid, selected_valid

    def _object_jepa_loss(
        self,
        outputs: dict[str, torch.Tensor],
        data: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        reference = outputs["scene_flows"]
        if not self.object_jepa_enabled:
            zero = reference.sum() * 0.0
            return zero, {"total": zero, "valid_tokens": zero.detach()}

        object_valid = outputs.get("object_valid")
        hidden = self._object_jepa_hidden
        if not isinstance(object_valid, torch.Tensor) or hidden is None:
            zero = self._all_jepa_parameters_live(reference)
            return zero, {"total": zero, "valid_tokens": zero.detach()}

        batch, object_count = object_valid.shape
        future_steps = int(reference.shape[1]) - 1
        expected_shape = (batch, object_count * future_steps, self.channels)
        if hidden.shape != expected_shape:
            raise RuntimeError(
                "Unexpected object-time hidden shape: "
                f"{tuple(hidden.shape)} vs {expected_shape}"
            )
        hidden = hidden.view(batch, object_count, future_steps, self.channels)

        target, target_valid, _selected_valid = self._build_target_latents(
            data, object_valid.bool()
        )
        context_latent = target[:, 0]
        target_future = target[:, 1:].permute(0, 2, 1, 3)
        valid_future = target_valid[:, 1:].permute(0, 2, 1)

        context_expand = context_latent[:, :, None].expand(-1, -1, future_steps, -1)
        predicted = self.object_jepa_predictor(
            torch.cat([hidden, context_expand], dim=-1)
        )
        predicted = F.normalize(predicted.float(), dim=-1, eps=1e-6)

        semantic_slice = slice(0, self.object_jepa_semantic_dim)
        pose_start = self.object_jepa_semantic_dim
        pose_slice = slice(pose_start, pose_start + self.object_jepa_pose_dim)
        shape_slice = slice(pose_start + self.object_jepa_pose_dim, self.object_jepa_dim)

        semantic_error = _block_cosine_error(
            predicted[..., semantic_slice], target_future[..., semantic_slice]
        )
        pose_error = _block_cosine_error(
            predicted[..., pose_slice], target_future[..., pose_slice]
        )
        shape_error = _block_cosine_error(
            predicted[..., shape_slice], target_future[..., shape_slice]
        )
        state_error = (
            self.object_jepa_semantic_weight * semantic_error
            + self.object_jepa_pose_weight * pose_error
            + self.object_jepa_shape_weight * shape_error
        )
        state_loss = horizon_weighted_mean(
            state_error,
            valid_future,
            self.object_jepa_horizon_start,
            self.object_jepa_horizon_end,
        )

        if future_steps > 1:
            dynamic_slice = slice(self.object_jepa_semantic_dim, self.object_jepa_dim)
            predicted_delta = (
                predicted[:, :, 1:, dynamic_slice]
                - predicted[:, :, :-1, dynamic_slice]
            )
            target_delta = (
                target_future[:, :, 1:, dynamic_slice]
                - target_future[:, :, :-1, dynamic_slice]
            )
            delta_error = F.smooth_l1_loss(
                predicted_delta,
                target_delta,
                reduction="none",
            ).mean(dim=-1)
            delta_valid = valid_future[:, :, 1:] & valid_future[:, :, :-1]
            delta_loss = horizon_weighted_mean(
                delta_error,
                delta_valid,
                self.object_jepa_horizon_start,
                self.object_jepa_horizon_end,
            )
        else:
            delta_loss = state_loss * 0.0

        raw_total = state_loss + self.object_jepa_delta_weight * delta_loss
        weighted = self.object_jepa_loss_weight * raw_total

        def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            return value[mask].mean() if mask.any() else value.sum() * 0.0

        early_end = min(14, future_steps)
        early_mask = valid_future[:, :, :early_end]
        late_mask = valid_future[:, :, early_end:]
        return weighted, {
            "total": raw_total,
            "state": state_loss,
            "semantic": horizon_weighted_mean(
                semantic_error,
                valid_future,
                self.object_jepa_horizon_start,
                self.object_jepa_horizon_end,
            ),
            "pose": horizon_weighted_mean(
                pose_error,
                valid_future,
                self.object_jepa_horizon_start,
                self.object_jepa_horizon_end,
            ),
            "shape": horizon_weighted_mean(
                shape_error,
                valid_future,
                self.object_jepa_horizon_start,
                self.object_jepa_horizon_end,
            ),
            "delta": delta_loss,
            "early_state_t01_t14": masked_mean(
                state_error[:, :, :early_end], early_mask
            ),
            "late_state_t15_t34": masked_mean(
                state_error[:, :, early_end:], late_mask
            ),
            "valid_tokens": valid_future.sum().to(reference.dtype),
        }

    def loss_fn(self, outputs, data, training=True):
        self._object_jepa_gt_pose_cache = None
        base_loss, log = super().loss_fn(outputs, data, training=training)
        jepa_weighted, components = self._object_jepa_loss(outputs, data)
        total = base_loss + jepa_weighted
        log["object_jepa/enabled"] = float(self.object_jepa_enabled)
        log["object_jepa/force_pose_output"] = float(
            self.object_jepa_force_pose_output
        )
        log["object_jepa/loss_weight"] = float(self.object_jepa_loss_weight)
        log["object_jepa/weighted_loss"] = float(
            jepa_weighted.detach().float().cpu()
        )
        for key, value in components.items():
            if torch.is_tensor(value):
                value = float(value.detach().float().cpu())
            log[f"object_jepa/{key}"] = float(value)
        return total, log


def install_object_jepa_checkpoint_loader() -> None:
    """Permit a strict warm start with only new ``object_jepa_*`` keys missing.

    Extra keys or any missing base-model key remain fatal.  Optimizer state must
    be reset because the parameter set changed.  This is intentionally narrower
    than general ``strict=False`` loading.
    """
    import training.checkpointing as checkpointing

    if getattr(checkpointing, "_object_jepa_loader_installed", False):
        return
    original = checkpointing.load_checkpoint

    def load_checkpoint(trainer, checkpoint):
        if not _env_bool("POINTWORLD_OBJECT_JEPA", False):
            return original(trainer, checkpoint)

        model = checkpointing._unwrap_model(trainer.model)
        expected = set(model.state_dict())
        supplied = set(checkpoint["model"])
        missing = sorted(expected - supplied)
        extra = sorted(supplied - expected)
        illegal_missing = [
            key
            for key in missing
            if not key.startswith(ObjectStateJEPAV2ExperimentalBaseModel._JEPA_PREFIX)
        ]
        if extra or illegal_missing:
            raise RuntimeError(
                "Object-JEPA warm start rejected an unexpected checkpoint mismatch. "
                f"Illegal missing={illegal_missing[:10]}, extra={extra[:10]}"
            )
        incompatible = model.load_state_dict(checkpoint["model"], strict=False)
        if sorted(incompatible.missing_keys) != missing or incompatible.unexpected_keys:
            raise RuntimeError(
                "Unexpected non-strict load result: "
                f"missing={incompatible.missing_keys[:10]} "
                f"unexpected={incompatible.unexpected_keys[:10]}"
            )
        if trainer.rank == 0:
            _print(
                f"Warm-started base model; initialized {len(missing)} new "
                "object-JEPA tensors/buffers from scratch."
            )

        if not trainer.inference_only:
            if not bool(getattr(trainer.args, "reset_optimizer_on_resume", False)):
                raise RuntimeError(
                    "Object-JEPA architecture extension requires "
                    "--reset_optimizer_on_resume true."
                )
            if trainer.rank == 0:
                _print("Resetting optimizer for object-JEPA architecture extension.")

        if _env_bool("OBJECT_JEPA_RESET_COUNTERS", True):
            trainer.batch_count = 0
            trainer.epoch_count = 0.0
            trainer.sample_count = 0
            trainer.best_checkpoint_metric = None
            trainer.best_checkpoint_metric_name = None
            trainer.best_checkpoint_filename = None
            if trainer.rank == 0:
                _print("Reset training counters for the controlled object-JEPA run.")
        else:
            for key in ("batch_count", "epoch_count", "sample_count"):
                if key not in checkpoint:
                    raise KeyError(f"Checkpoint missing required counter {key!r}")
            trainer.batch_count = checkpoint["batch_count"]
            trainer.epoch_count = checkpoint["epoch_count"]
            trainer.sample_count = checkpoint["sample_count"]
            trainer.best_checkpoint_metric = checkpoint.get("best_checkpoint_metric")
            trainer.best_checkpoint_metric_name = checkpoint.get(
                "best_checkpoint_metric_name"
            )
            trainer.best_checkpoint_filename = checkpoint.get(
                "best_checkpoint_filename"
            )

    checkpointing.load_checkpoint = load_checkpoint
    checkpointing._object_jepa_loader_installed = True
