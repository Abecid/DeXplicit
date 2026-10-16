from __future__ import annotations

import math
import os

import torch
import torch.nn as nn

STAGE_NAMES: dict[int, str] = {
    0: "pose_control",
    1: "local_residual",
    2: "structure_edge",
    3: "contact_aux",
    4: "contact_routing",
    5: "contact_flow",
    6: "latent_particles",
    7: "object_jepa_physical",
    8: "contact_physics",
}

# Minimal prerequisite sets: use these for the first scientific pass. Every
# experiment starts from the same SOTA checkpoint and adds only the requested
# method (plus the smallest mechanism needed to make that method meaningful).
ISOLATED_FEATURES_BY_STAGE: dict[int, tuple[str, ...]] = {
    0: ("pose_canonical",),
    1: ("pose_canonical", "object_local_residual"),
    2: (
        "pose_canonical",
        "object_local_residual",
        "gt_relative_edge_loss",
    ),
    3: ("pose_canonical", "per_hand_contact_distance"),
    4: (
        "pose_canonical",
        "per_hand_contact_distance",
        "contact_gated_action_routing",
    ),
    5: (
        "pose_canonical",
        "per_hand_contact_distance",
        "contact_flow_tokens",
    ),
    6: ("pose_canonical", "particle_deformation_bottleneck"),
    7: ("pose_canonical", "object_state_jepa", "physical_probe"),
    8: (
        "pose_canonical",
        "per_hand_contact_distance",
        "contact_consistency",
        "stick_contact",
    ),
}

# Winner-stacking ladder: run only after the isolated probes identify useful
# mechanisms. Stage n adds one row from the requested method table to stage n-1.
CUMULATIVE_FEATURES_BY_STAGE: dict[int, tuple[str, ...]] = {
    0: ("pose_canonical",),
    1: ("pose_canonical", "object_local_residual"),
    2: (
        "pose_canonical",
        "object_local_residual",
        "gt_relative_edge_loss",
    ),
    3: (
        "pose_canonical",
        "object_local_residual",
        "gt_relative_edge_loss",
        "per_hand_contact_distance",
    ),
    4: (
        "pose_canonical",
        "object_local_residual",
        "gt_relative_edge_loss",
        "per_hand_contact_distance",
        "contact_gated_action_routing",
    ),
    5: (
        "pose_canonical",
        "object_local_residual",
        "gt_relative_edge_loss",
        "per_hand_contact_distance",
        "contact_gated_action_routing",
        "contact_flow_tokens",
    ),
    6: (
        "pose_canonical",
        "particle_deformation_bottleneck",
        "gt_relative_edge_loss",
        "per_hand_contact_distance",
        "contact_gated_action_routing",
        "contact_flow_tokens",
    ),
    7: (
        "pose_canonical",
        "particle_deformation_bottleneck",
        "gt_relative_edge_loss",
        "per_hand_contact_distance",
        "contact_gated_action_routing",
        "contact_flow_tokens",
        "object_state_jepa",
        "physical_probe",
    ),
    8: (
        "pose_canonical",
        "particle_deformation_bottleneck",
        "gt_relative_edge_loss",
        "per_hand_contact_distance",
        "contact_gated_action_routing",
        "contact_flow_tokens",
        "object_state_jepa",
        "physical_probe",
        "contact_consistency",
        "stick_contact",
    ),
}

# Backward-compatible name for prior PR tests/docs.
FEATURES_BY_STAGE = CUMULATIVE_FEATURES_BY_STAGE


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def stage_name(stage: int) -> str:
    if int(stage) not in STAGE_NAMES:
        raise ValueError(f"LONG_HORIZON_STAGE must be 0..8, got {stage}")
    return STAGE_NAMES[int(stage)]


def normalize_track(track: str | None = None) -> str:
    value = (track or os.environ.get("LONG_HORIZON_TRACK", "isolated")).strip().lower()
    if value not in {"isolated", "cumulative"}:
        raise ValueError(f"LONG_HORIZON_TRACK must be isolated or cumulative, got {value!r}")
    return value


def stage_features(stage: int, track: str | None = None) -> tuple[str, ...]:
    stage_name(stage)
    selected = normalize_track(track)
    table = (
        ISOLATED_FEATURES_BY_STAGE
        if selected == "isolated"
        else CUMULATIVE_FEATURES_BY_STAGE
    )
    return table[int(stage)]


def feature_enabled(stage: int, feature: str, track: str | None = None) -> bool:
    return feature in stage_features(stage, track)


def _masked_mean(value: torch.Tensor, valid: torch.Tensor, dim: int) -> torch.Tensor:
    weight = valid.to(value.dtype).unsqueeze(-1)
    return (value * weight).sum(dim=dim) / weight.sum(dim=dim).clamp_min(1.0)


def _zero_last_linear(module: nn.Module) -> None:
    linears = [child for child in module.modules() if isinstance(child, nn.Linear)]
    if not linears:
        raise ValueError("Expected at least one Linear layer")
    with torch.no_grad():
        linears[-1].weight.zero_()
        if linears[-1].bias is not None:
            linears[-1].bias.zero_()


def _inverse_softplus(value: float) -> float:
    value = max(float(value), 1e-8)
    return math.log(math.expm1(value))


def _split_hand_points(
    hand_xyz: torch.Tensor,
    hand_valid: torch.Tensor,
) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    """Split raw HOT3D hand-surface points into left/right halves."""
    if hand_xyz.ndim != 4 or hand_xyz.shape[-1] != 3:
        raise ValueError(f"hand_xyz must be (B,T,R,3), got {tuple(hand_xyz.shape)}")
    if hand_valid.shape != hand_xyz.shape[:-1]:
        raise ValueError("hand_valid must match hand_xyz without the XYZ channel")
    count = int(hand_xyz.shape[2])
    if count < 2:
        raise ValueError("Need at least two hand points to form left/right groups")
    left_count = count // 2
    return (
        (hand_xyz[:, :, :left_count], hand_valid[:, :, :left_count]),
        (hand_xyz[:, :, left_count:], hand_valid[:, :, left_count:]),
    )


def hand_side_trajectory_features(
    hand_xyz: torch.Tensor,
    hand_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return side centroids, relative motion, one-step motion and validity."""
    groups = _split_hand_points(hand_xyz.float(), hand_valid.bool())
    centroids: list[torch.Tensor] = []
    valid_sides: list[torch.Tensor] = []
    for points, valid in groups:
        centroids.append(_masked_mean(points, valid, dim=2))
        valid_sides.append(valid.any(dim=2))
    centroid = torch.stack(centroids, dim=2)
    side_valid = torch.stack(valid_sides, dim=2)
    centroid = torch.where(side_valid.unsqueeze(-1), centroid, torch.zeros_like(centroid))
    relative = centroid - centroid[:, :1]
    previous = torch.cat([centroid[:, :1], centroid[:, :-1]], dim=1)
    delta = centroid - previous
    delta[:, 0] = 0
    return centroid, relative, delta, side_valid


def _gather_bnd(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    batch, _, channels = values.shape
    objects, points = indices.shape[1:]
    expanded = values[:, None].expand(-1, objects, -1, -1)
    gather_index = indices[..., None].expand(batch, objects, points, channels)
    return torch.gather(expanded, 2, gather_index)


def _gather_btnd(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    batch, timesteps, _, channels = values.shape
    objects, points = indices.shape[1:]
    expanded = values[:, :, None].expand(-1, -1, objects, -1, -1)
    gather_index = indices[:, None, :, :, None].expand(
        batch, timesteps, objects, points, channels
    )
    return torch.gather(expanded, 3, gather_index)


def _gather_btn(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    batch, timesteps, _ = values.shape
    objects, points = indices.shape[1:]
    expanded = values[:, :, None].expand(-1, -1, objects, -1)
    gather_index = indices[:, None].expand(batch, timesteps, objects, points)
    return torch.gather(expanded, 3, gather_index)


def _gather_btnh(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    batch, timesteps, _, hands = values.shape
    objects, points = indices.shape[1:]
    expanded = values[:, :, None].expand(-1, -1, objects, -1, -1)
    gather_index = indices[:, None, :, :, None].expand(
        batch, timesteps, objects, points, hands
    )
    return torch.gather(expanded, 3, gather_index)


def batched_min_hand_distance(
    points: torch.Tensor,
    point_valid: torch.Tensor,
    hand_points: torch.Tensor,
    hand_valid: torch.Tensor,
    *,
    chunk_bt: int = 32,
    return_indices: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Minimum point-to-hand distance with bounded temporary memory."""
    if points.ndim != 4 or hand_points.ndim != 4:
        raise ValueError("points and hand_points must be (B,T,N,3)")
    if points.shape[:2] != hand_points.shape[:2]:
        raise ValueError("points and hands must share B,T")
    if point_valid.shape != points.shape[:-1] or hand_valid.shape != hand_points.shape[:-1]:
        raise ValueError("valid masks do not match point tensors")

    batch, timesteps, point_count, _ = points.shape
    hand_count = int(hand_points.shape[2])
    flat_points = points.float().reshape(batch * timesteps, point_count, 3)
    flat_point_valid = point_valid.bool().reshape(batch * timesteps, point_count)
    flat_hands = hand_points.float().reshape(batch * timesteps, hand_count, 3)
    flat_hand_valid = hand_valid.bool().reshape(batch * timesteps, hand_count)
    output_distance = flat_points.new_zeros(batch * timesteps, point_count)
    output_valid = torch.zeros_like(flat_point_valid)
    output_indices = (
        torch.zeros(
            batch * timesteps,
            point_count,
            device=points.device,
            dtype=torch.long,
        )
        if return_indices
        else None
    )
    for start in range(0, batch * timesteps, int(chunk_bt)):
        end = min(start + int(chunk_bt), batch * timesteps)
        distance = torch.cdist(flat_points[start:end], flat_hands[start:end])
        distance = distance.masked_fill(
            ~flat_hand_valid[start:end, None], torch.finfo(distance.dtype).max
        )
        minimum, nearest = distance.min(dim=-1)
        valid = flat_point_valid[start:end] & flat_hand_valid[
            start:end
        ].any(dim=-1, keepdim=True)
        output_distance[start:end] = torch.where(
            valid, minimum, torch.zeros_like(minimum)
        )
        output_valid[start:end] = valid
        if output_indices is not None:
            output_indices[start:end] = nearest
    distance = output_distance.view(batch, timesteps, point_count)
    valid = output_valid.view(batch, timesteps, point_count)
    nearest = (
        output_indices.view(batch, timesteps, point_count)
        if output_indices is not None
        else None
    )
    return distance, valid, nearest


def contact_strength(distance: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.exp(-0.5 * (distance.float() / max(float(tau), 1e-6)).square())


def _horizon_mask(
    timesteps: int,
    start: int,
    end: int,
    device: torch.device,
) -> torch.Tensor:
    index = torch.arange(timesteps, device=device)
    return (index >= int(start)) & (index <= int(end))


def _masked_sum_count(
    values: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if values.shape != valid.shape:
        raise ValueError("values and valid must have the same shape")
    count = valid.sum()
    total = values[valid].sum() if count > 0 else values.sum() * 0.0
    return total, count
