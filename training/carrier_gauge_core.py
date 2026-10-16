"""Rigid-gauge and carrier geometry for canonical deformation graphs.

Column-vector convention: ``(R, t)`` maps ``p`` to ``R @ p + t``.
All routines are differentiable with respect to floating-point inputs except
for validity masks and the discrete graph/carrier topology.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import contextlib
import functools

import torch


def autocast_disabled(function):
    """Run ``function`` with autocast off on every device.

    The gauge/graph geometry relies on SVD, determinants, eigendecomposition,
    and pseudo-inverses.  Under bf16 autocast the surrounding matmuls emit
    bf16 tensors and CUDA has no bf16 factorizations, so the whole routine
    runs in its float32 working precision and casts back on return.
    """

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        with contextlib.ExitStack() as stack:
            stack.enter_context(torch.autocast(device_type="cpu", enabled=False))
            if torch.cuda.is_available():
                stack.enter_context(
                    torch.autocast(device_type="cuda", enabled=False)
                )
            return function(*args, **kwargs)

    return wrapper


def _work_dtype(*tensors: torch.Tensor) -> torch.dtype:
    dtype = tensors[0].dtype
    for tensor in tensors[1:]:
        dtype = torch.promote_types(dtype, tensor.dtype)
    return torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype


def so3_exp(rotation_vector: torch.Tensor) -> torch.Tensor:
    """Rodrigues exponential for axis-angle vectors ``(..., 3)``."""
    if rotation_vector.shape[-1:] != (3,) or not torch.is_floating_point(rotation_vector):
        raise ValueError("rotation_vector must be floating point with shape (...,3)")
    dtype = rotation_vector.dtype
    omega = rotation_vector.to(_work_dtype(rotation_vector))
    theta2 = (omega * omega).sum(-1, keepdim=True)
    theta4 = theta2.square()
    safe = theta2.clamp_min(torch.finfo(omega.dtype).tiny)
    theta = safe.sqrt()
    a = torch.where(
        theta2 <= 1e-4,
        1 - theta2 / 6 + theta4 / 120,
        torch.sin(theta) / theta,
    )
    b = torch.where(
        theta2 <= 1e-4,
        0.5 - theta2 / 24 + theta4 / 720,
        (1 - torch.cos(theta)) / safe,
    )
    x, y, z = omega.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack(
        (zero, -z, y, z, zero, -x, -y, x, zero), -1
    ).reshape(*omega.shape[:-1], 3, 3)
    eye = torch.eye(3, device=omega.device, dtype=omega.dtype).expand(
        *omega.shape[:-1], 3, 3
    )
    return (eye + a[..., None] * skew + b[..., None] * (skew @ skew)).to(dtype)


def so3_log(rotation: torch.Tensor) -> torch.Tensor:
    """Principal logarithm for proper rotations ``(..., 3, 3)``."""
    if rotation.shape[-2:] != (3, 3) or not torch.is_floating_point(rotation):
        raise ValueError("rotation must be floating point with shape (...,3,3)")
    dtype = rotation.dtype
    matrix = rotation.to(_work_dtype(rotation))
    trace = matrix.diagonal(dim1=-2, dim2=-1).sum(-1)
    cosine = ((trace - 1) / 2).clamp(-1, 1)
    vee = torch.stack(
        (
            matrix[..., 2, 1] - matrix[..., 1, 2],
            matrix[..., 0, 2] - matrix[..., 2, 0],
            matrix[..., 1, 0] - matrix[..., 0, 1],
        ),
        -1,
    )
    sine = 0.5 * torch.linalg.vector_norm(vee, dim=-1)
    theta = torch.atan2(sine, cosine)
    ordinary = theta[..., None] * vee / (2 * sine.clamp_min(1e-8))[..., None]
    result = torch.where(
        (theta.abs() < 1e-4)[..., None], 0.5 * vee, ordinary
    )
    return result.to(dtype)


@autocast_disabled
def fit_rigid_trajectory(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    weights: torch.Tensor | None = None,
    valid: torch.Tensor | None = None,
    rank_epsilon: float = 1e-7,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Weighted proper Kabsch fit ``source (N,3) -> target (T,N,3)``."""
    if (
        source.ndim != 2
        or source.shape[-1] != 3
        or target.ndim != 3
        or target.shape[1:] != source.shape
    ):
        raise ValueError("expected source (N,3) and target (T,N,3)")
    frames, points = target.shape[:2]
    weights = source.new_ones(points) if weights is None else weights
    valid = (
        torch.ones(frames, points, device=target.device, dtype=torch.bool)
        if valid is None
        else valid.bool()
    )
    if weights.shape != (points,) or valid.shape != (frames, points):
        raise ValueError("invalid Kabsch weights/validity shape")
    dtype = _work_dtype(source, target)
    source_f = source.to(dtype)
    target_f = target.to(dtype)
    base_weight = weights.to(dtype).clamp_min(0)
    rotations = torch.eye(
        3, device=target.device, dtype=target.dtype
    ).expand(frames, 3, 3).clone()
    translations = target.new_zeros(frames, 3)
    fit_valid = torch.zeros(frames, device=target.device, dtype=torch.bool)
    for time in range(frames):
        weight = base_weight * valid[time].to(dtype)
        if int((weight > 0).sum()) < 3:
            continue
        total = weight.sum()
        source_center = (source_f * weight[:, None]).sum(0) / total
        target_center = (target_f[time] * weight[:, None]).sum(0) / total
        source_zero = source_f - source_center
        target_zero = target_f[time] - target_center
        covariance = torch.einsum(
            "n,ni,nj->ij", weight, source_zero, target_zero
        )
        if not torch.isfinite(covariance).all():
            continue
        u, singular, vh = torch.linalg.svd(covariance, full_matrices=False)
        if (
            not torch.isfinite(singular).all()
            or not bool(singular[1] > rank_epsilon)
        ):
            continue
        correction = torch.eye(3, device=target.device, dtype=dtype)
        if bool(
            torch.linalg.det(vh.transpose(-1, -2) @ u.transpose(-1, -2)) < 0
        ):
            correction[-1, -1] = -1
        rotation = (
            vh.transpose(-1, -2) @ correction @ u.transpose(-1, -2)
        )
        translation = target_center - rotation @ source_center
        rotations[time] = rotation.to(target.dtype)
        translations[time] = translation.to(target.dtype)
        fit_valid[time] = True
    return rotations, translations, fit_valid


def warp_local_graph(
    canonical: torch.Tensor,
    object_ids: torch.Tensor,
    point_valid: torch.Tensor,
    topology: Any,
    local_rotation: torch.Tensor,
    local_translation: torch.Tensor,
    *,
    fallback: torch.Tensor | None = None,
) -> torch.Tensor:
    """Embedded-deformation warp in canonical space, without global pose."""
    batch, points = canonical.shape[:2]
    objects, horizons, nodes = local_translation.shape[1:4]
    if (
        canonical.shape != (batch, points, 3)
        or local_rotation.shape
        != (batch, objects, horizons, nodes, 3, 3)
    ):
        raise ValueError("invalid graph warp shapes")
    output = (
        canonical[:, None].expand(batch, horizons, points, 3).clone()
        if fallback is None
        else fallback.clone()
    )
    for b in range(batch):
        for o in range(objects):
            mask = point_valid[b].bool() & (object_ids[b].long() == o)
            if not mask.any():
                continue
            point = canonical[b, mask]
            indices = topology.point_node_indices[b, mask].long()
            weight = topology.point_node_weights[b, mask]
            centers = topology.node_centers[b, o][indices]
            rotation = local_rotation[b, o][:, indices]
            translation = local_translation[b, o][:, indices]
            transformed = torch.matmul(
                rotation,
                (point[:, None] - centers)[None, ..., None],
            ).squeeze(-1)
            output[b, :, mask] = (
                (transformed + centers[None] + translation)
                * weight[None, ..., None]
            ).sum(2)
    return output


def gauge_fix(
    canonical: torch.Tensor,
    raw_local: torch.Tensor,
    object_ids: torch.Tensor,
    point_valid: torch.Tensor,
    object_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Remove each local field's best finite rigid transform exactly."""
    batch, horizons, _ = raw_local.shape[:3]
    objects = object_valid.shape[1]
    fixed = raw_local.clone()
    gauge_r = torch.eye(
        3, device=raw_local.device, dtype=raw_local.dtype
    ).expand(batch, objects, horizons, 3, 3).clone()
    gauge_t = raw_local.new_zeros(batch, objects, horizons, 3)
    gauge_valid = torch.zeros(
        batch,
        objects,
        horizons,
        device=raw_local.device,
        dtype=torch.bool,
    )
    for b in range(batch):
        for o in range(objects):
            mask = point_valid[b].bool() & (object_ids[b].long() == o)
            if not bool(object_valid[b, o]) or int(mask.sum()) < 3:
                continue
            rotation, translation, valid = fit_rigid_trajectory(
                canonical[b, mask], raw_local[b, :, mask]
            )
            gauge_r[b, o] = rotation
            gauge_t[b, o] = translation
            gauge_valid[b, o] = valid
            normalized = torch.einsum(
                "tij,tnj->tni",
                rotation.transpose(-1, -2),
                raw_local[b, :, mask] - translation[:, None],
            )
            fixed[b, :, mask] = torch.where(
                valid[:, None, None], normalized, raw_local[b, :, mask]
            )
    return fixed, gauge_r, gauge_t, gauge_valid


def compose_pose(
    outer_rotation: torch.Tensor,
    outer_translation: torch.Tensor,
    inner_rotation: torch.Tensor,
    inner_translation: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compose poses: ``outer(inner(p))``."""
    if (
        outer_rotation.shape != inner_rotation.shape
        or outer_translation.shape != inner_translation.shape
    ):
        raise ValueError("pose shapes must match")
    return (
        outer_rotation @ inner_rotation,
        outer_translation
        + torch.einsum("...ij,...j->...i", outer_rotation, inner_translation),
    )


def render_objects(
    local_points: torch.Tensor,
    object_ids: torch.Tensor,
    point_valid: torch.Tensor,
    object_valid: torch.Tensor,
    rotation: torch.Tensor,
    translation: torch.Tensor,
    *,
    fallback: torch.Tensor,
) -> torch.Tensor:
    """Render canonical-local points through one pose per object/time."""
    output = fallback.clone()
    for b in range(local_points.shape[0]):
        for o in range(object_valid.shape[1]):
            mask = point_valid[b].bool() & (object_ids[b].long() == o)
            if bool(object_valid[b, o]) and mask.any():
                output[b, :, mask] = (
                    torch.einsum(
                        "tij,tnj->tni", rotation[b, o], local_points[b, :, mask]
                    )
                    + translation[b, o, :, None]
                )
    return output


@dataclass(frozen=True)
class CarrierTransforms:
    rotation: torch.Tensor
    translation: torch.Tensor
    valid: torch.Tensor


def hand_carriers(
    hand_xyz: torch.Tensor, hand_valid: torch.Tensor,
    *, unlabelled: torch.Tensor | None = None,
) -> CarrierTransforms:
    """Fit frame-0-to-future left, right, and bimanual hand transforms."""
    if hand_valid.ndim == hand_xyz.ndim:
        hand_valid = hand_valid.squeeze(-1)
    batch, timesteps, hand_points = hand_xyz.shape[:3]
    if hand_points % 2 or timesteps < 2:
        raise ValueError(
            "expected two equal hand point sets and at least two frames"
        )
    horizons, half = timesteps - 1, hand_points // 2
    rotation = torch.eye(
        3, device=hand_xyz.device, dtype=hand_xyz.dtype
    ).expand(batch, horizons, 3, 3, 3).clone()
    translation = hand_xyz.new_zeros(batch, horizons, 3, 3)
    valid_out = torch.zeros(
        batch, horizons, 3, device=hand_xyz.device, dtype=torch.bool
    )
    ranges = ((0, half), (half, hand_points), (0, hand_points))
    for b in range(batch):
        for carrier, (start, end) in enumerate(ranges):
            if unlabelled is not None and bool(unlabelled[b]) and carrier < 2:
                continue  # no fabricated left/right carrier: use aggregate fit
            common = (
                hand_valid[b, 1:, start:end].bool()
                & hand_valid[b, 0, start:end].bool()[None]
            )
            fitted = fit_rigid_trajectory(
                hand_xyz[b, 0, start:end],
                hand_xyz[b, 1:, start:end],
                valid=common,
            )
            rotation[b, :, carrier] = fitted[0]
            translation[b, :, carrier] = fitted[1]
            valid_out[b, :, carrier] = fitted[2]
    return CarrierTransforms(rotation, translation, valid_out)


def carrier_pose_candidates(
    initial_rotation: torch.Tensor,
    initial_translation: torch.Tensor,
    pose_rotation: torch.Tensor,
    pose_translation: torch.Tensor,
    carriers: CarrierTransforms,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pose/left/right/bimanual/world candidates, pose candidate first."""
    batch, objects, horizons = pose_translation.shape[:3]
    rotations = pose_rotation[:, :, :, None].expand(
        -1, -1, -1, 5, -1, -1
    ).clone()
    translations = pose_translation[:, :, :, None].expand(
        -1, -1, -1, 5, -1
    ).clone()
    valid = torch.ones(
        batch,
        objects,
        horizons,
        5,
        device=pose_rotation.device,
        dtype=torch.bool,
    )
    for index in range(3):
        carrier_r = carriers.rotation[:, :, index]
        carrier_t = carriers.translation[:, :, index]
        rotations[:, :, :, index + 1] = torch.einsum(
            "btij,bojk->botik", carrier_r, initial_rotation
        )
        translations[:, :, :, index + 1] = (
            torch.einsum(
                "btij,boj->boti", carrier_r, initial_translation
            )
            + carrier_t[:, None]
        )
        valid[:, :, :, index + 1] = carriers.valid[:, :, index][:, None]
    rotations[:, :, :, 4] = initial_rotation[:, :, None]
    translations[:, :, :, 4] = initial_translation[:, :, None]
    return rotations, translations, valid


def candidate_deltas(
    base_rotation: torch.Tensor,
    base_translation: torch.Tensor,
    candidate_rotation: torch.Tensor,
    candidate_translation: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Express carrier candidates as base-local translation/rotation deltas."""
    relative_r = torch.einsum(
        "...ji,...cjk->...cik", base_rotation, candidate_rotation
    )
    rotation_vector = so3_log(relative_r)
    translation = torch.einsum(
        "...ji,...cj->...ci",
        base_rotation,
        candidate_translation - base_translation[..., None, :],
    )
    features = torch.cat(
        (translation, rotation_vector, valid[..., None].to(translation.dtype)),
        -1,
    )
    features = torch.where(
        valid[..., None], features, torch.zeros_like(features)
    )
    return translation, rotation_vector, features


def apply_pose_delta(
    rotation: torch.Tensor,
    translation: torch.Tensor,
    local_translation: torch.Tensor,
    local_rotation_vector: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-compose a bounded local pose correction."""
    return (
        rotation @ so3_exp(local_rotation_vector),
        translation
        + torch.einsum("...ij,...j->...i", rotation, local_translation),
    )


__all__ = [
    "CarrierTransforms",
    "apply_pose_delta",
    "candidate_deltas",
    "carrier_pose_candidates",
    "compose_pose",
    "fit_rigid_trajectory",
    "gauge_fix",
    "hand_carriers",
    "render_objects",
    "so3_exp",
    "so3_log",
    "warp_local_graph",
]
