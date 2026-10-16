"""Eval-only time-sliced L2 helpers. Does not change the model or training loss."""

from __future__ import annotations

import torch


def time_range_mask(
    valid: torch.Tensor, start: int, end_inclusive: int
) -> torch.Tensor:
    """Boolean mask over ``valid``'s time axis, inclusive of both ends."""
    mask = torch.zeros_like(valid)
    t_end = min(int(end_inclusive), int(valid.shape[1]) - 1)
    t_start = max(int(start), 0)
    if t_start <= t_end:
        mask[:, t_start : t_end + 1] = True
    return mask
