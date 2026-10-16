"""Count-weighted diagnostics for global pose versus canonical deformation."""
from __future__ import annotations

import torch

from training.carrier_gauge_core import fit_rigid_trajectory


def _ranges(timesteps: int) -> tuple[tuple[str, int, int], ...]:
    last = timesteps - 1
    if last < 1:
        return ()
    return (
        ("t01_t14", 1, min(14, last)),
        ("t15_t34", min(15, last), last),
        ("t01_t34", 1, last),
        ("t34", last, last),
    )


def _accumulate(
    result: dict[str, tuple[torch.Tensor, torch.Tensor]],
    key: str,
    value: torch.Tensor,
    valid: torch.Tensor,
) -> None:
    valid = valid.bool()
    total = value[valid].sum() if valid.any() else value.sum() * 0.0
    count = valid.sum().to(dtype=torch.long)
    if key in result:
        old_total, old_count = result[key]
        result[key] = (old_total + total, old_count + count)
    else:
        result[key] = (total, count)


@torch.no_grad()
def trajectory_decomposition_sums(
    predicted: torch.Tensor,
    target: torch.Tensor,
    canonical: torch.Tensor,
    object_ids: torch.Tensor,
    point_valid: torch.Tensor,
    object_valid: torch.Tensor,
    *,
    deformable_threshold_m: float = 0.005,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Split point error into fitted rigid-pose and object-frame deformation.

    ``predicted`` and ``target`` are absolute trajectories ``(B,T,N,3)``.
    Every returned value is an unnormalized sum paired with its exact count, so
    distributed evaluation can reduce the statistics without mean-of-means bias.
    """
    if predicted.shape != target.shape or predicted.ndim != 4:
        raise ValueError("predicted/target must share shape (B,T,N,3)")
    batch, timesteps, points, coordinates = predicted.shape
    if coordinates != 3 or canonical.shape != (batch, points, 3):
        raise ValueError("canonical geometry shape does not match trajectories")
    if object_ids.shape != (batch, points):
        raise ValueError("object_ids must have shape (B,N)")
    if point_valid.shape != (batch, timesteps, points):
        raise ValueError("point_valid must have shape (B,T,N)")
    if object_valid.ndim != 2 or object_valid.shape[0] != batch:
        raise ValueError("object_valid must have shape (B,O)")
    if deformable_threshold_m < 0:
        raise ValueError("deformable threshold must be non-negative")

    result: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for batch_index in range(batch):
        for object_index in range(object_valid.shape[1]):
            object_mask = object_ids[batch_index].long() == object_index
            selected = object_mask.nonzero(as_tuple=False).squeeze(-1)
            if (
                not bool(object_valid[batch_index, object_index])
                or int(selected.numel()) < 3
            ):
                continue
            valid = point_valid[batch_index, :, selected].bool()
            pred_rotation, pred_translation, pred_fit_valid = fit_rigid_trajectory(
                canonical[batch_index, selected],
                predicted[batch_index, :, selected],
                valid=valid,
            )
            gt_rotation, gt_translation, gt_fit_valid = fit_rigid_trajectory(
                canonical[batch_index, selected],
                target[batch_index, :, selected],
                valid=valid,
            )
            fit_valid = pred_fit_valid & gt_fit_valid
            pred_rigid = (
                torch.einsum(
                    "tij,nj->tni",
                    pred_rotation,
                    canonical[batch_index, selected],
                )
                + pred_translation[:, None]
            )
            gt_rigid = (
                torch.einsum(
                    "tij,nj->tni",
                    gt_rotation,
                    canonical[batch_index, selected],
                )
                + gt_translation[:, None]
            )
            pred_local = torch.einsum(
                "tji,tnj->tni",
                pred_rotation,
                predicted[batch_index, :, selected]
                - pred_translation[:, None],
            )
            gt_local = torch.einsum(
                "tji,tnj->tni",
                gt_rotation,
                target[batch_index, :, selected] - gt_translation[:, None],
            )
            point_fit_valid = valid & fit_valid[:, None]
            reconstruction = torch.linalg.vector_norm(
                predicted[batch_index, :, selected]
                - target[batch_index, :, selected],
                dim=-1,
            )
            pose_error = torch.linalg.vector_norm(
                pred_rigid - gt_rigid, dim=-1
            )
            deformation_error = torch.linalg.vector_norm(
                pred_local - gt_local, dim=-1
            )
            pred_nonrigidity = torch.linalg.vector_norm(
                predicted[batch_index, :, selected] - pred_rigid,
                dim=-1,
            )
            gt_nonrigidity = torch.linalg.vector_norm(
                target[batch_index, :, selected] - gt_rigid,
                dim=-1,
            )
            artificial_excess = (
                pred_nonrigidity - gt_nonrigidity
            ).clamp_min(0)
            gt_object_nonrigidity = (
                (gt_nonrigidity * point_fit_valid).sum(-1)
                / point_fit_valid.sum(-1).clamp_min(1)
            )
            deformable = gt_object_nonrigidity > deformable_threshold_m
            subsets = {
                "all": torch.ones_like(deformable, dtype=torch.bool),
                "rigid": ~deformable,
                "deformable": deformable,
            }
            metrics = {
                "reconstruction_l2_m": reconstruction,
                "rigid_pose_l2_m": pose_error,
                "object_frame_deformation_l2_m": deformation_error,
                "pred_nonrigidity_m": pred_nonrigidity,
                "gt_nonrigidity_m": gt_nonrigidity,
                "artificial_nonrigidity_excess_m": artificial_excess,
            }
            for range_name, start, end in _ranges(timesteps):
                if start > end:
                    continue
                horizon = torch.zeros(
                    timesteps, device=predicted.device, dtype=torch.bool
                )
                horizon[start : end + 1] = True
                for subset_name, subset in subsets.items():
                    selection = (
                        point_fit_valid
                        & horizon[:, None]
                        & subset[:, None]
                    )
                    for metric_name, values in metrics.items():
                        _accumulate(
                            result,
                            (
                                "carrier_gauge/decomposition/"
                                f"{subset_name}/{metric_name}_{range_name}"
                            ),
                            values,
                            selection,
                        )
    return result


def scalar_sum_count(
    value: torch.Tensor, valid: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert an arbitrary diagnostic tensor to a sum/count pair."""
    if valid is None:
        valid = torch.ones_like(value, dtype=torch.bool)
    else:
        valid = valid.bool().expand_as(value)
    total = value[valid].sum() if valid.any() else value.sum() * 0.0
    return total, valid.sum().to(dtype=torch.long)


__all__ = ["scalar_sum_count", "trajectory_decomposition_sums"]
