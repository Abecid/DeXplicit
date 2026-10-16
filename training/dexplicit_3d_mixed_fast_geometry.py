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

from training.batched_kabsch import (
    batched_weighted_kabsch,
    gather_object_pose_per_point,
)


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
    """Weighted proper Kabsch fit ``source (N,3) -> target (T,N,3)``.

    All frames are fitted in one batched SVD (see ``training/batched_kabsch``);
    the validity rules and outputs match the original per-frame loop.
    """
    if (
        source.ndim != 2
        or source.shape[-1] != 3
        or target.ndim != 3
        or target.shape[1:] != source.shape
    ):
        raise ValueError("expected source (N,3) and target (T,N,3)")
    frames, points = target.shape[:2]
    if weights is not None and tuple(weights.shape) != (points,):
        raise ValueError("invalid Kabsch weights/validity shape")
    if valid is not None and tuple(valid.shape) != (frames, points):
        raise ValueError("invalid Kabsch weights/validity shape")
    rotation, translation, fit_valid = batched_weighted_kabsch(
        source, target, weights=weights, valid=valid, rank_epsilon=rank_epsilon
    )
    return rotation.to(target.dtype), translation.to(target.dtype), fit_valid


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
    """Embedded-deformation warp in canonical space, without global pose.

    Vectorised over batch and objects: every point gathers its object's node
    transforms through its skinning indices, exactly as the per-object loop did.
    """
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
    ids = object_ids.long()
    member = point_valid.bool() & (ids >= 0) & (ids < objects)
    if not bool(member.any()):
        return output
    oid = ids.clamp(0, objects - 1)
    indices = topology.point_node_indices.long()  # (B,N,J)
    weight = topology.point_node_weights.to(canonical.dtype)  # (B,N,J)
    batch_index = torch.arange(batch, device=canonical.device)[:, None, None]
    centers = topology.node_centers.to(canonical.dtype)[
        batch_index, oid[:, :, None], indices
    ]  # (B,N,J,3)
    rotation = local_rotation.to(canonical.dtype)[
        batch_index, oid[:, :, None], :, indices
    ]  # (B,N,J,T,3,3)
    translation = local_translation.to(canonical.dtype)[
        batch_index, oid[:, :, None], :, indices
    ]  # (B,N,J,T,3)
    relative = canonical[:, :, None] - centers  # (B,N,J,3)
    transformed = torch.einsum("bnjtik,bnjk->btnji", rotation, relative)
    warped = (
        (transformed + centers[:, None] + translation.permute(0, 3, 1, 2, 4))
        * weight[:, None, :, :, None]
    ).sum(3)  # (B,T,N,3)
    return torch.where(member[:, None, :, None], warped, output)


def gauge_fix(
    canonical: torch.Tensor,
    raw_local: torch.Tensor,
    object_ids: torch.Tensor,
    point_valid: torch.Tensor,
    object_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Remove each local field's best finite rigid transform exactly.

    One batched Kabsch over (batch, object, horizon); points of objects that
    are invalid or have fewer than three members keep their raw local field.
    """
    batch, horizons, points = raw_local.shape[:3]
    objects = object_valid.shape[1]
    ids = object_ids.long()
    member = point_valid.bool()[:, None, :] & (
        ids[:, None, :]
        == torch.arange(objects, device=ids.device)[None, :, None]
    )  # (B,O,N)
    active = object_valid.bool() & (member.sum(-1) >= 3)  # (B,O)
    rotation, translation, fit_valid = batched_weighted_kabsch(
        canonical[:, None].expand(batch, objects, points, 3),
        raw_local[:, None].expand(batch, objects, horizons, points, 3),
        weights=member.to(torch.float32),
    )
    gauge_valid = fit_valid & active[:, :, None]  # (B,O,T)
    eye = torch.eye(3, device=raw_local.device, dtype=rotation.dtype)
    rotation = torch.where(gauge_valid[..., None, None], rotation, eye)
    translation = torch.where(
        gauge_valid[..., None], translation, torch.zeros_like(translation)
    )
    point_rotation, point_translation = gather_object_pose_per_point(
        rotation, translation, ids
    )  # (B,T,N,3,3), (B,T,N,3)
    apply = (
        gauge_valid.permute(0, 2, 1)[
            torch.arange(batch, device=ids.device)[:, None, None],
            torch.arange(horizons, device=ids.device)[None, :, None],
            ids.clamp(0, objects - 1)[:, None, :],
        ]
        & member.any(1)[:, None, :]
    )  # (B,T,N)
    normalized = torch.einsum(
        "btnji,btnj->btni",
        point_rotation.to(raw_local.dtype),
        raw_local - point_translation.to(raw_local.dtype),
    )
    fixed = torch.where(apply[..., None], normalized, raw_local)
    return (
        fixed,
        rotation.to(raw_local.dtype),
        translation.to(raw_local.dtype),
        gauge_valid,
    )


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
    objects = object_valid.shape[1]
    ids = object_ids.long()
    in_range = (ids >= 0) & (ids < objects)
    oid = ids.clamp(0, objects - 1)
    batch_index = torch.arange(ids.shape[0], device=ids.device)[:, None]
    member = point_valid.bool() & in_range & object_valid.bool()[batch_index, oid]
    point_rotation, point_translation = gather_object_pose_per_point(
        rotation, translation, ids
    )
    world = torch.einsum(
        "btnij,btnj->btni", point_rotation.to(local_points.dtype), local_points
    ) + point_translation.to(local_points.dtype)
    return torch.where(member[:, None, :, None], world, fallback)


@dataclass(frozen=True)
class CarrierTransforms:
    rotation: torch.Tensor
    translation: torch.Tensor
    valid: torch.Tensor


def hand_carriers(
    hand_xyz: torch.Tensor, hand_valid: torch.Tensor
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
    device = hand_xyz.device
    carrier_weight = torch.zeros(3, hand_points, device=device, dtype=torch.float32)
    carrier_weight[0, :half] = 1.0
    carrier_weight[1, half:] = 1.0
    carrier_weight[2, :] = 1.0
    common = hand_valid[:, 1:].bool() & hand_valid[:, :1].bool()  # (B,H,P)
    rotation, translation, valid = batched_weighted_kabsch(
        hand_xyz[:, 0][:, None].expand(batch, 3, hand_points, 3),
        hand_xyz[:, 1:][:, None].expand(batch, 3, horizons, hand_points, 3),
        weights=carrier_weight[None].expand(batch, 3, hand_points),
        valid=common[:, None].expand(batch, 3, horizons, hand_points),
    )
    return CarrierTransforms(
        rotation.permute(0, 2, 1, 3, 4).to(hand_xyz.dtype),
        translation.permute(0, 2, 1, 3).to(hand_xyz.dtype),
        valid.permute(0, 2, 1),
    )


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
