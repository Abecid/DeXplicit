"""Contact-ready object-local residual :math:`SE(3)` pose adapter.

The adapter is deliberately agnostic to how its ten conditioning channels are
produced.  metric_pose supplies ten literal zero channels, predicted_contact supplies predicted-contact
channels, and oracle_contact supplies oracle-contact channels.  Keeping that width fixed is
what makes the three conditions parameter-count matched.

The six predicted values form an object-local tangent twist ``(v, omega)``.
They are mapped through the full :math:`SE(3)` exponential, whose local
translation is ``J(omega) @ v`` for the SO(3) left Jacobian ``J``.  For a base
pose ``(R, t)`` the resulting residual ``(R_delta, t_delta)`` is
right-composed as

``R_new = R @ R_delta`` and ``t_new = t + R @ t_delta``.

Thus translation is never mean-centred across points: one residual moves every
point of an object coherently.  ``apply_object_local_residual_to_points`` can
apply the same correction to an arbitrary base point trajectory.  In
particular, it can preserve the current blended SOTA trajectory exactly at
zero initialization instead of replacing it with the pose-only trajectory.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn


# Five values per hand: canonical anchor xyz, contact mass, and transformed
# minimum distance.  metric_pose/predicted_contact/oracle_contact must all retain this exact width.
CONTACT_CONDITIONING_DIM = 10


def zero_contact_conditioning(
    object_time_features: torch.Tensor,
    dimension: int = CONTACT_CONDITIONING_DIM,
) -> torch.Tensor:
    """Return metric_pose's literal-zero conditioning with the campaign-wide shape."""

    if object_time_features.ndim != 4:
        raise ValueError(
            "object_time_features must have shape (B,O,T,D), got "
            f"{tuple(object_time_features.shape)}"
        )
    if int(dimension) <= 0:
        raise ValueError("conditioning dimension must be positive")
    return object_time_features.new_zeros(
        *object_time_features.shape[:-1], int(dimension)
    )


def _skew_symmetric(vector: torch.Tensor) -> torch.Tensor:
    if vector.shape[-1] != 3:
        raise ValueError(
            f"rotation vectors must end in dimension 3, got {tuple(vector.shape)}"
        )
    x, y, z = vector.unbind(dim=-1)
    zero = torch.zeros_like(x)
    return torch.stack(
        (zero, -z, y, z, zero, -x, -y, x, zero), dim=-1
    ).reshape(*vector.shape[:-1], 3, 3)


def so3_exp_map(rotation_vector: torch.Tensor) -> torch.Tensor:
    """Stable Rodrigues exponential for axis-angle rotation vectors.

    Half/bfloat inputs are evaluated in float32.  Near zero, fourth-order
    Taylor expansions avoid the removable ``0/0`` singularities in
    ``sin(theta)/theta`` and ``(1-cos(theta))/theta**2``.  The zero vector maps
    to an exact identity matrix, which is important for initialization parity.
    """

    if not torch.is_floating_point(rotation_vector):
        raise TypeError("rotation_vector must be floating point")
    if rotation_vector.shape[-1] != 3:
        raise ValueError(
            "rotation_vector must have shape (...,3), got "
            f"{tuple(rotation_vector.shape)}"
        )

    output_dtype = rotation_vector.dtype
    work_dtype = (
        torch.float32
        if output_dtype in (torch.float16, torch.bfloat16)
        else output_dtype
    )
    omega = rotation_vector.to(dtype=work_dtype)
    theta_squared = (omega * omega).sum(dim=-1, keepdim=True)
    theta_fourth = theta_squared * theta_squared

    # Clamp only the denominator branch.  torch.where evaluates both inputs,
    # so unclamped zero denominators can poison backward even when the series
    # branch is selected.
    denominator_floor = torch.finfo(work_dtype).tiny
    safe_theta_squared = theta_squared.clamp_min(denominator_floor)
    safe_theta = torch.sqrt(safe_theta_squared)
    a_exact = torch.sin(safe_theta) / safe_theta
    b_exact = (1.0 - torch.cos(safe_theta)) / safe_theta_squared
    a_series = 1.0 - theta_squared / 6.0 + theta_fourth / 120.0
    b_series = 0.5 - theta_squared / 24.0 + theta_fourth / 720.0

    # A fixed angular threshold is preferable to dtype epsilon here: it keeps
    # the numerically stable branch consistent between fp32 and fp64 tests.
    # Through theta=0.1 the omitted sixth-order term is below fp32 precision,
    # while the exact ``1-cos(theta)`` form already loses useful fp32 digits.
    use_series = theta_squared <= 1e-2
    a = torch.where(use_series, a_series, a_exact)[..., None]
    b = torch.where(use_series, b_series, b_exact)[..., None]

    skew = _skew_symmetric(omega)
    identity = torch.eye(3, dtype=work_dtype, device=omega.device)
    identity = identity.expand(*omega.shape[:-1], 3, 3)
    rotation = identity + a * skew + b * (skew @ skew)
    return rotation.to(dtype=output_dtype)


def se3_exp_translation(
    tangent_translation: torch.Tensor,
    rotation_vector: torch.Tensor,
) -> torch.Tensor:
    """Map the translational part of a tangent twist through ``Exp_SE3``.

    For ``xi = (v, omega)``, the translational component of ``Exp(xi)`` is

    ``J(omega) v``, where

    ``J = I + (1-cos(theta))/theta^2 Omega``
    ``      + (theta-sin(theta))/theta^3 Omega^2``.

    Half and bfloat16 inputs are evaluated in float32.  Fourth-order series
    handle the removable singularities at zero, so ``omega == 0`` maps to
    *exactly* ``v`` and keeps the zero-initialized adapter an exact identity.
    """

    if not torch.is_floating_point(tangent_translation) or not torch.is_floating_point(
        rotation_vector
    ):
        raise TypeError("SE(3) tangent vectors must be floating point")
    if tangent_translation.shape != rotation_vector.shape or (
        tangent_translation.shape[-1] != 3
    ):
        raise ValueError(
            "tangent_translation and rotation_vector must share shape (...,3), "
            f"got {tuple(tangent_translation.shape)} and "
            f"{tuple(rotation_vector.shape)}"
        )
    if tangent_translation.device != rotation_vector.device:
        raise ValueError("SE(3) tangent vectors must share a device")

    output_dtype = tangent_translation.dtype
    work_dtype = torch.promote_types(
        tangent_translation.dtype, rotation_vector.dtype
    )
    if work_dtype in (torch.float16, torch.bfloat16):
        work_dtype = torch.float32
    tangent = tangent_translation.to(dtype=work_dtype)
    omega = rotation_vector.to(dtype=work_dtype)
    theta_squared = (omega * omega).sum(dim=-1, keepdim=True)
    theta_fourth = theta_squared * theta_squared

    denominator_floor = torch.finfo(work_dtype).tiny
    safe_theta_squared = theta_squared.clamp_min(denominator_floor)
    safe_theta = torch.sqrt(safe_theta_squared)
    b_exact = (1.0 - torch.cos(safe_theta)) / safe_theta_squared
    safe_theta_cubed = (safe_theta_squared * safe_theta).clamp_min(
        denominator_floor
    )
    c_exact = (safe_theta - torch.sin(safe_theta)) / safe_theta_cubed
    b_series = 0.5 - theta_squared / 24.0 + theta_fourth / 720.0
    c_series = 1.0 / 6.0 - theta_squared / 120.0 + theta_fourth / 5040.0
    use_series = theta_squared <= 1e-2
    b = torch.where(use_series, b_series, b_exact)
    c = torch.where(use_series, c_series, c_exact)

    # Jv = v + B (omega x v) + C (omega x (omega x v)).  This avoids
    # materializing a Jacobian and makes omega==0 return v bit-for-bit.
    omega_cross_v = torch.cross(omega, tangent, dim=-1)
    omega_cross_squared_v = torch.cross(omega, omega_cross_v, dim=-1)
    mapped = tangent + b * omega_cross_v + c * omega_cross_squared_v
    return mapped.to(dtype=output_dtype)


def _expanded_object_valid(
    object_valid: torch.Tensor,
    batch: int,
    objects: int,
    timesteps: int,
    device: torch.device,
) -> torch.Tensor:
    if not isinstance(object_valid, torch.Tensor):
        raise TypeError("object_valid must be a tensor")
    if object_valid.ndim == 2 and tuple(object_valid.shape) == (batch, objects):
        valid = object_valid[:, :, None].expand(batch, objects, timesteps)
    elif object_valid.ndim == 3 and tuple(object_valid.shape) == (
        batch,
        objects,
        timesteps,
    ):
        valid = object_valid
    else:
        raise ValueError(
            "object_valid must have shape (B,O) or (B,O,T), got "
            f"{tuple(object_valid.shape)} for B={batch}, O={objects}, T={timesteps}"
        )
    return valid.to(device=device, dtype=torch.bool)


def compose_object_local_residual(
    base_rotation: torch.Tensor,
    base_translation: torch.Tensor,
    local_translation: torch.Tensor,
    residual_rotation: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-compose an object-local rigid correction with base poses.

    Shapes are ``(...,3,3)`` for rotations and ``(...,3)`` for translations.
    The additive form for rotation is algebraically equivalent to
    ``base_rotation @ residual_rotation`` while preserving exact equality to
    the input tensor when ``residual_rotation`` is exactly identity.
    """

    expected_rotation = (*base_translation.shape[:-1], 3, 3)
    if tuple(base_rotation.shape) != expected_rotation:
        raise ValueError(
            "base_rotation/base_translation shapes disagree: "
            f"{tuple(base_rotation.shape)} vs {tuple(base_translation.shape)}"
        )
    if tuple(local_translation.shape) != tuple(base_translation.shape):
        raise ValueError(
            "local_translation must match base_translation, got "
            f"{tuple(local_translation.shape)} vs {tuple(base_translation.shape)}"
        )
    if tuple(residual_rotation.shape) != tuple(base_rotation.shape):
        raise ValueError(
            "residual_rotation must match base_rotation, got "
            f"{tuple(residual_rotation.shape)} vs {tuple(base_rotation.shape)}"
        )
    if base_rotation.device != base_translation.device:
        raise ValueError("base rotations and translations must share a device")

    local_translation = local_translation.to(
        device=base_translation.device, dtype=base_translation.dtype
    )
    residual_rotation = residual_rotation.to(
        device=base_rotation.device, dtype=base_rotation.dtype
    )
    identity = torch.eye(
        3, dtype=base_rotation.dtype, device=base_rotation.device
    )
    rotation_delta = base_rotation @ (residual_rotation - identity)
    corrected_rotation = base_rotation + rotation_delta
    corrected_translation = base_translation + torch.einsum(
        "...ij,...j->...i", base_rotation, local_translation
    )
    return corrected_rotation, corrected_translation


def apply_object_local_residual_to_points(
    base_points: torch.Tensor,
    point_object_ids: torch.Tensor,
    point_valid: torch.Tensor,
    base_rotation: torch.Tensor,
    base_translation: torch.Tensor,
    local_translation: torch.Tensor,
    residual_rotation: torch.Tensor,
    object_valid: torch.Tensor,
) -> torch.Tensor:
    """Apply object-local rigid residuals to a base point trajectory.

    ``base_points`` is ``(B,T,N,3)`` and object poses/residuals are
    ``(B,O,T,...)``.  A point with a negative/out-of-range object id, an invalid
    point mask, or an invalid object-time mask is returned unchanged.  The
    correction is expressed as a *delta* from ``base_points`` so a zero adapter
    is exactly silent even when the base trajectory is the direct/pose blend
    rather than a perfectly rigid canonical reconstruction.
    """

    if base_points.ndim != 4 or base_points.shape[-1] != 3:
        raise ValueError(
            f"base_points must have shape (B,T,N,3), got {tuple(base_points.shape)}"
        )
    batch, timesteps, point_count, _ = base_points.shape
    if point_object_ids.ndim != 2 or tuple(point_object_ids.shape) != (
        batch,
        point_count,
    ):
        raise ValueError(
            "point_object_ids must have shape (B,N), got "
            f"{tuple(point_object_ids.shape)}"
        )
    if point_object_ids.dtype.is_floating_point or point_object_ids.dtype == torch.bool:
        raise TypeError("point_object_ids must use an integer dtype")

    if point_valid.ndim == 2 and tuple(point_valid.shape) == (batch, point_count):
        expanded_point_valid = point_valid[:, None].expand(
            batch, timesteps, point_count
        )
    elif point_valid.ndim == 3 and tuple(point_valid.shape) == (
        batch,
        timesteps,
        point_count,
    ):
        expanded_point_valid = point_valid
    else:
        raise ValueError(
            "point_valid must have shape (B,N) or (B,T,N), got "
            f"{tuple(point_valid.shape)}"
        )
    expanded_point_valid = expanded_point_valid.to(
        device=base_points.device, dtype=torch.bool
    )

    if base_rotation.ndim != 5 or base_rotation.shape[-2:] != (3, 3):
        raise ValueError(
            "base_rotation must have shape (B,O,T,3,3), got "
            f"{tuple(base_rotation.shape)}"
        )
    objects = int(base_rotation.shape[1])
    expected_rotation = (batch, objects, timesteps, 3, 3)
    expected_translation = (batch, objects, timesteps, 3)
    if tuple(base_rotation.shape) != expected_rotation:
        raise ValueError(
            f"base_rotation must have shape {expected_rotation}, got {tuple(base_rotation.shape)}"
        )
    for name, tensor in (
        ("base_translation", base_translation),
        ("local_translation", local_translation),
    ):
        if tuple(tensor.shape) != expected_translation:
            raise ValueError(
                f"{name} must have shape {expected_translation}, got {tuple(tensor.shape)}"
            )
    if tuple(residual_rotation.shape) != expected_rotation:
        raise ValueError(
            "residual_rotation must match base_rotation, got "
            f"{tuple(residual_rotation.shape)}"
        )

    object_time_valid = _expanded_object_valid(
        object_valid, batch, objects, timesteps, base_points.device
    )
    # Keep adapter parameters in the autograd graph on ranks that receive no
    # valid objects.  This avoids a per-rank unused-parameter mismatch in DDP.
    graph_zero = (
        local_translation.sum() + residual_rotation.sum()
    ).to(base_points.dtype) * 0.0
    if objects == 0:
        return base_points + graph_zero

    ids = point_object_ids.to(device=base_points.device, dtype=torch.long)
    id_valid = (ids >= 0) & (ids < objects)
    safe_ids = ids.clamp(0, objects - 1)

    def gather_object_time(tensor: torch.Tensor) -> torch.Tensor:
        # (B,O,T,...) -> (B,T,N,...)
        ordered = tensor.permute(0, 2, 1, *range(3, tensor.ndim))
        trailing = tuple(tensor.shape[3:])
        # Python 3.10 compatible (PEP 646 star-in-subscript needs 3.11+).
        index = safe_ids[(slice(None), None, slice(None)) + (None,) * len(trailing)]
        index = index.expand(batch, timesteps, point_count, *trailing)
        return torch.gather(ordered, 2, index)

    point_base_rotation = gather_object_time(base_rotation).to(
        dtype=base_points.dtype
    )
    point_base_translation = gather_object_time(base_translation).to(
        dtype=base_points.dtype
    )
    point_local_translation = gather_object_time(local_translation).to(
        dtype=base_points.dtype
    )
    point_residual_rotation = gather_object_time(residual_rotation).to(
        dtype=base_points.dtype
    )
    gathered_object_valid = gather_object_time(
        object_time_valid[..., None]
    ).squeeze(-1)

    centered = base_points - point_base_translation
    local_point = torch.einsum(
        "btnji,btnj->btni", point_base_rotation, centered
    )
    identity = torch.eye(
        3, dtype=base_points.dtype, device=base_points.device
    )
    local_delta = torch.einsum(
        "btnij,btnj->btni",
        point_residual_rotation - identity,
        local_point,
    ) + point_local_translation
    world_delta = torch.einsum(
        "btnij,btnj->btni", point_base_rotation, local_delta
    )
    valid = (
        expanded_point_valid
        & id_valid[:, None]
        & gathered_object_valid
    )
    world_delta = torch.where(valid[..., None], world_delta, torch.zeros_like(world_delta))
    return base_points + world_delta + graph_zero


@dataclass(frozen=True)
class ResidualSE3AdapterOutput:
    """Typed output of :class:`ResidualSE3PoseAdapter`."""

    raw_twist: torch.Tensor
    local_tangent_translation: torch.Tensor
    local_translation: torch.Tensor
    local_rotation_vector: torch.Tensor
    residual_rotation: torch.Tensor
    rotation: torch.Tensor
    translation: torch.Tensor
    valid: torch.Tensor


class ResidualSE3PoseAdapter(nn.Module):
    """Predict a bounded object-local residual rigid pose for every horizon."""

    def __init__(
        self,
        object_feature_dim: int,
        *,
        conditioning_dim: int = CONTACT_CONDITIONING_DIM,
        hidden_dim: int | None = None,
        max_translation_m: float = 0.05,
        max_rotation_radians: float = math.radians(15.0),
    ) -> None:
        super().__init__()
        object_feature_dim = int(object_feature_dim)
        conditioning_dim = int(conditioning_dim)
        hidden_dim = int(hidden_dim or max(64, object_feature_dim))
        if object_feature_dim <= 0 or conditioning_dim <= 0 or hidden_dim <= 0:
            raise ValueError("feature, conditioning, and hidden dimensions must be positive")
        if not math.isfinite(max_translation_m) or max_translation_m <= 0.0:
            raise ValueError("max_translation_m must be finite and positive")
        if not math.isfinite(max_rotation_radians) or not (
            0.0 < max_rotation_radians < math.pi
        ):
            raise ValueError(
                "max_rotation_radians must be finite and strictly between 0 and pi"
            )

        self.object_feature_dim = object_feature_dim
        self.conditioning_dim = conditioning_dim
        self.hidden_dim = hidden_dim
        self.max_translation_m = float(max_translation_m)
        self.max_rotation_radians = float(max_rotation_radians)
        total_dim = object_feature_dim + conditioning_dim
        self.trunk = nn.Sequential(
            nn.LayerNorm(total_dim),
            nn.Linear(total_dim, hidden_dim),
            nn.GELU(),
        )
        self.final_projection = nn.Linear(hidden_dim, 6)
        # Exact zero output is the attribution contract.  The rest of the MLP
        # remains normally initialized so upstream gradients become useful as
        # soon as the final projection moves away from zero.
        with torch.no_grad():
            self.final_projection.weight.zero_()
            self.final_projection.bias.zero_()

    @property
    def input_dim(self) -> int:
        return self.object_feature_dim + self.conditioning_dim

    def forward(
        self,
        object_time_features: torch.Tensor,
        conditioning: torch.Tensor,
        base_rotation: torch.Tensor,
        base_translation: torch.Tensor,
        object_valid: torch.Tensor,
    ) -> ResidualSE3AdapterOutput:
        if object_time_features.ndim != 4:
            raise ValueError(
                "object_time_features must have shape (B,O,T,D), got "
                f"{tuple(object_time_features.shape)}"
            )
        batch, objects, timesteps, feature_dim = object_time_features.shape
        if int(feature_dim) != self.object_feature_dim:
            raise ValueError(
                f"expected object feature width {self.object_feature_dim}, got {feature_dim}"
            )
        expected_conditioning = (
            batch,
            objects,
            timesteps,
            self.conditioning_dim,
        )
        if tuple(conditioning.shape) != expected_conditioning:
            raise ValueError(
                "conditioning must have the fixed campaign shape "
                f"{expected_conditioning}, got {tuple(conditioning.shape)}"
            )
        expected_rotation = (batch, objects, timesteps, 3, 3)
        expected_translation = (batch, objects, timesteps, 3)
        if tuple(base_rotation.shape) != expected_rotation:
            raise ValueError(
                f"base_rotation must have shape {expected_rotation}, got {tuple(base_rotation.shape)}"
            )
        if tuple(base_translation.shape) != expected_translation:
            raise ValueError(
                "base_translation must have shape "
                f"{expected_translation}, got {tuple(base_translation.shape)}"
            )
        if object_time_features.device != base_rotation.device or (
            base_rotation.device != base_translation.device
        ):
            raise ValueError("features and base poses must share a device")

        valid = _expanded_object_valid(
            object_valid,
            batch,
            objects,
            timesteps,
            object_time_features.device,
        )
        conditioning = conditioning.to(
            device=object_time_features.device,
            dtype=object_time_features.dtype,
        )
        adapter_input = torch.cat((object_time_features, conditioning), dim=-1)
        raw_twist = self.final_projection(self.trunk(adapter_input))
        raw_twist = torch.where(
            valid[..., None], raw_twist, torch.zeros_like(raw_twist)
        )

        local_tangent_translation = (
            torch.tanh(raw_twist[..., :3]) * self.max_translation_m
        )
        local_rotation_vector = (
            torch.tanh(raw_twist[..., 3:]) * self.max_rotation_radians
        )
        residual_rotation = so3_exp_map(local_rotation_vector)
        # This is the translational component of the full object-local SE(3)
        # exponential, not a decoupled additive translation.  Preserve the
        # base pose dtypes only after the stable exponential computation.
        local_translation = se3_exp_translation(
            local_tangent_translation,
            local_rotation_vector,
        ).to(dtype=base_translation.dtype)
        local_rotation_vector = local_rotation_vector.to(dtype=base_rotation.dtype)
        residual_rotation = residual_rotation.to(dtype=base_rotation.dtype)
        corrected_rotation, corrected_translation = compose_object_local_residual(
            base_rotation,
            base_translation,
            local_translation,
            residual_rotation,
        )
        return ResidualSE3AdapterOutput(
            raw_twist=raw_twist,
            local_tangent_translation=local_tangent_translation.to(
                dtype=base_translation.dtype
            ),
            local_translation=local_translation,
            local_rotation_vector=local_rotation_vector,
            residual_rotation=residual_rotation,
            rotation=corrected_rotation,
            translation=corrected_translation,
            valid=valid,
        )
