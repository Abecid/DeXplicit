"""Explicit per-row action-representation flag; never infer it from zero joints."""
from __future__ import annotations
import torch


def unlabelled_rows(data, batch: int, device) -> torch.Tensor:
    value = data.get("action_is_unlabelled")
    if value is None:
        return torch.zeros(batch, dtype=torch.bool, device=device)
    value = torch.as_tensor(value, device=device)
    if value.dtype != torch.bool or value.shape != (batch,):
        raise ValueError("action_is_unlabelled must be one boolean per batch row")
    return value
