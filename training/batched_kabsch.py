"""Batched weighted proper Kabsch fits with exact loop semantics.

Replaces the per-frame / per-object Python loops that dominated dexplicit_3d training
time (profiled 2026-09-05: ``fit_rigid_trajectory`` 35% and the inherited
``_gt_pose`` 22% of wall clock).  The math is identical to the loop versions
kept in ``tests/support/reference_graph_loops.py``; only the batching differs,
so results agree to float32 rounding and the validity masks agree exactly.
"""
from __future__ import annotations

import contextlib

import torch


def autocast_off() -> contextlib.ExitStack:
    stack = contextlib.ExitStack()
    stack.enter_context(torch.autocast(device_type="cpu", enabled=False))
    if torch.cuda.is_available():
        stack.enter_context(torch.autocast(device_type="cuda", enabled=False))
    return stack


def batched_weighted_kabsch(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    weights: torch.Tensor | None = None,
    valid: torch.Tensor | None = None,
    rank_epsilon: float | None = 1e-7,
    min_points: int = 3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fit one proper rigid transform per frame, batched over leading dims.

    ``source`` is ``(..., N, 3)``, ``target`` is ``(..., T, N, 3)``,
    ``weights`` is ``(..., N)`` (non-negative) and ``valid`` is ``(..., T, N)``.
    Returns ``rotation (..., T, 3, 3)``, ``translation (..., T, 3)`` in the
    working precision (float64 stays float64, everything else float32) and
    ``fit_valid (..., T)``.  A frame is valid when at least ``min_points``
    points carry positive weight, the covariance and singular values are
    finite, the fitted transform is finite and, when ``rank_epsilon`` is not
    ``None``, the second singular value exceeds it.  Invalid frames return the
    identity rotation and zero translation.
    """
    if source.shape[-1] != 3 or target.shape[-1] != 3 or target.ndim != source.ndim + 1:
        raise ValueError("expected source (...,N,3) and target (...,T,N,3)")
    lead = tuple(target.shape[:-3])
    frames, points = int(target.shape[-3]), int(target.shape[-2])
    if tuple(source.shape[:-2]) != lead or int(source.shape[-2]) != points:
        raise ValueError("source/target leading or point dimensions differ")
    with autocast_off():
        device = target.device
        # Same working-precision rule as the loop implementation: keep float64
        # inputs in float64, promote half/bfloat16 to float32.
        work = torch.promote_types(source.dtype, target.dtype)
        if work in (torch.float16, torch.bfloat16) or not work.is_floating_point:
            work = torch.float32
        src = source.to(work)
        tgt = target.to(work)
        if weights is None:
            weight = torch.ones(*lead, points, device=device, dtype=work)
        else:
            if tuple(weights.shape) != (*lead, points):
                raise ValueError("weights must have shape (...,N)")
            weight = weights.to(work).clamp_min(0)
        if valid is None:
            valid_mask = torch.ones(*lead, frames, points, device=device, dtype=torch.bool)
        else:
            if tuple(valid.shape) != (*lead, frames, points):
                raise ValueError("valid must have shape (...,T,N)")
            valid_mask = valid.bool()
        frame_weight = weight[..., None, :] * valid_mask.to(work)  # (...,T,N)
        count = (frame_weight > 0).sum(dim=-1)  # (...,T)
        total = frame_weight.sum(dim=-1)  # (...,T)
        safe_total = total.clamp_min(1e-12)[..., None]
        source_center = (frame_weight[..., None] * src[..., None, :, :]).sum(dim=-2) / safe_total
        target_center = (frame_weight[..., None] * tgt).sum(dim=-2) / safe_total
        source_zero = src[..., None, :, :] - source_center[..., None, :]
        target_zero = tgt - target_center[..., None, :]
        covariance = torch.einsum(
            "...tn,...tni,...tnj->...tij", frame_weight, source_zero, target_zero
        )
        eye = torch.eye(3, device=device, dtype=work)
        # Validity is decided with a no-grad probe; the differentiable SVD then
        # runs only on valid slots.  Empty or rank-deficient covariances (which
        # the loop version skipped) would otherwise produce infinite SVD
        # gradients that leak through torch.where into the model.
        with torch.no_grad():
            finite_covariance = torch.isfinite(covariance).all(dim=-1).all(dim=-1)
            probe = torch.where(finite_covariance[..., None, None], covariance, eye)
            probe_singular = torch.linalg.svdvals(probe)
            fit_valid = (count >= int(min_points)) & finite_covariance
            fit_valid = fit_valid & torch.isfinite(probe_singular).all(dim=-1)
            if rank_epsilon is not None:
                fit_valid = fit_valid & (probe_singular[..., 1] > float(rank_epsilon))
        slots = fit_valid.reshape(-1)
        selected = slots.nonzero(as_tuple=False).squeeze(-1)
        flat_rotation = eye.expand(slots.numel(), 3, 3).clone()
        flat_translation = torch.zeros(slots.numel(), 3, device=device, dtype=work)
        if int(selected.numel()) > 0:
            cov_sel = covariance.reshape(-1, 3, 3)[selected]
            u, singular, vh = torch.linalg.svd(cov_sel, full_matrices=False)
            vu = vh.transpose(-1, -2) @ u.transpose(-1, -2)
            determinant = torch.linalg.det(vu)
            correction = torch.ones(int(selected.numel()), 3, device=device, dtype=work)
            correction[:, -1] = torch.where(
                determinant < 0, -torch.ones_like(determinant), torch.ones_like(determinant)
            )
            rotation_sel = (
                vh.transpose(-1, -2) @ torch.diag_embed(correction) @ u.transpose(-1, -2)
            )
            source_center_sel = source_center.reshape(-1, 3)[selected]
            target_center_sel = target_center.reshape(-1, 3)[selected]
            translation_sel = target_center_sel - torch.einsum(
                "nij,nj->ni", rotation_sel, source_center_sel
            )
            with torch.no_grad():
                finite_sel = (
                    torch.isfinite(rotation_sel).all(dim=-1).all(dim=-1)
                    & torch.isfinite(translation_sel).all(dim=-1)
                )
            rotation_sel = torch.where(finite_sel[:, None, None], rotation_sel, eye)
            translation_sel = torch.where(
                finite_sel[:, None], translation_sel, torch.zeros_like(translation_sel)
            )
            flat_rotation = flat_rotation.index_put((selected,), rotation_sel)
            flat_translation = flat_translation.index_put((selected,), translation_sel)
            slot_valid = slots.clone()
            slot_valid[selected] = finite_sel
            fit_valid = slot_valid.reshape(fit_valid.shape)
        rotation = flat_rotation.reshape(*lead, frames, 3, 3)
        translation = flat_translation.reshape(*lead, frames, 3)
        return rotation, translation, fit_valid


def gather_object_pose_per_point(
    rotation: torch.Tensor,
    translation: torch.Tensor,
    object_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Broadcast ``(B,O,T,·)`` poses to ``(B,T,N,·)`` by each point's object id.

    Ids outside ``[0, O)`` are clamped; callers mask those points separately.
    """
    batch, objects, frames = rotation.shape[:3]
    ids = object_ids.long().clamp(0, objects - 1)
    batch_index = torch.arange(batch, device=rotation.device)[:, None, None]
    time_index = torch.arange(frames, device=rotation.device)[None, :, None]
    rotation_bt = rotation.permute(0, 2, 1, 3, 4)  # (B,T,O,3,3)
    translation_bt = translation.permute(0, 2, 1, 3)  # (B,T,O,3)
    return (
        rotation_bt[batch_index, time_index, ids[:, None, :]],
        translation_bt[batch_index, time_index, ids[:, None, :]],
    )


__all__ = ["autocast_off", "batched_weighted_kabsch", "gather_object_pose_per_point"]
