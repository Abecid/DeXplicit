"""Remove the direct context-target shortcut from object-state JEPA.

Research provenance
-------------------
V-JEPA 2 (arXiv:2506.09985; official code:
https://github.com/facebookresearch/vjepa2) motivates predicting masked future
representations from an online context encoder while training a frozen EMA target
encoder.  DINO-WM (arXiv:2411.04983; official code:
https://github.com/gaoyuezhou/dino_wm) motivates using pretrained DINO features
as a predictive state rather than reconstructing pixels.

This experiment is deliberately smaller and is NOT a faithful implementation of
either codebase.  ``object_jepa_se3.py`` uses fixed seeded projections of known
object-level semantics, supervised GT pose/velocity, and supervised object-local
shape.  It has no learned/EMA target encoder, no V-JEPA masking curriculum, and
no DINO-WM patch-transition/planning stack.  The fixed-target design was chosen
to isolate whether an auxiliary future object-state target helps PointWorld; its
latent size and loss weights are project pilot values.

The base experiment's predictor receives ``[object_time_hidden, context_target]``.
For the controlled hypothesis we want the action-conditioned object-time state
itself to retain semantics, pose trend and local shape.  This subclass replaces
only the predictor: it accepts the existing concatenated call contract for
backward compatibility but discards the target-derived suffix before prediction.
See ``training/long_horizon_provenance.py`` for the complete audit ledger.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from training.object_jepa_se3 import ObjectStateJEPAV2ExperimentalBaseModel


class HiddenOnlyObjectJEPAPredictor(nn.Module):
    """Predict the fixed PointWorld object-state target from ``h[o,t]`` only."""

    def __init__(self, hidden_dim: int, target_dim: int):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, 2 * int(target_dim)),
            nn.GELU(),
            nn.Linear(2 * int(target_dim), int(target_dim)),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.shape[-1] < self.hidden_dim:
            raise ValueError(
                f"Object-JEPA predictor expected at least {self.hidden_dim} channels, "
                f"got {value.shape[-1]}"
            )
        return self.net(value[..., : self.hidden_dim])


class HiddenOnlyObjectStateJEPAV2ExperimentalBaseModel(
    ObjectStateJEPAV2ExperimentalBaseModel
):
    """Fixed-target JEPA-inspired PointWorld auxiliary with no target bypass."""

    def __init__(self, args, data_info_dict, rank=0, cpu_pg=None):
        super().__init__(args, data_info_dict, rank=rank, cpu_pg=cpu_pg)
        if self.object_jepa_enabled:
            self.object_jepa_predictor = HiddenOnlyObjectJEPAPredictor(
                int(self.channels), int(self.object_jepa_dim)
            )
