from __future__ import annotations

import math
from typing import Iterable, Mapping

import torch
import torch.distributed as dist
import torch.nn as nn

from training.score_se3_metrics import (
    COMPOSITE_SCORE,
    COMPONENT_KEYS,
    ScoreMetricSpec,
    score_aligned_loss,
)


class ScoreAlignedLossController(nn.Module):
    """Apply and optionally gradient-calibrate the metric-matched raw geometry loss."""

    def __init__(
        self,
        *,
        enabled: bool,
        explicit_weight: float,
        gradient_ratio: float = 0.0,
        max_calibrated_weight: float = 100.0,
        spec: ScoreMetricSpec | None = None,
    ) -> None:
        super().__init__()
        if explicit_weight < 0:
            raise ValueError("explicit_weight must be non-negative")
        if gradient_ratio < 0:
            raise ValueError("gradient_ratio must be non-negative")
        if max_calibrated_weight <= 0:
            raise ValueError("max_calibrated_weight must be positive")
        self.enabled = bool(enabled)
        self.explicit_weight = float(explicit_weight)
        self.gradient_ratio = float(gradient_ratio)
        self.max_calibrated_weight = float(max_calibrated_weight)
        self.spec = spec or ScoreMetricSpec()
        self.register_buffer(
            "effective_weight",
            torch.tensor(
                self.explicit_weight if self.enabled else 0.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "calibrated",
            torch.tensor(self.gradient_ratio == 0.0 or not self.enabled),
        )
        self.register_buffer(
            "base_gradient_norm_at_calibration",
            torch.tensor(float("nan"), dtype=torch.float32),
        )
        self.register_buffer(
            "score_gradient_norm_at_calibration",
            torch.tensor(float("nan"), dtype=torch.float32),
        )

    @staticmethod
    def _gradient_squared_norm(
        loss: torch.Tensor,
        parameters: tuple[nn.Parameter, ...],
    ) -> torch.Tensor:
        gradients = torch.autograd.grad(
            loss,
            parameters,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )
        total = loss.new_zeros((), dtype=torch.float32)
        for gradient in gradients:
            if gradient is not None:
                total = total + gradient.detach().float().square().sum()
        return total

    def _calibrate_once(
        self,
        base_loss: torch.Tensor,
        score_loss: torch.Tensor,
        parameters: Iterable[nn.Parameter],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if bool(self.calibrated.item()):
            return (
                self.base_gradient_norm_at_calibration,
                self.score_gradient_norm_at_calibration,
            )
        reference = tuple(parameter for parameter in parameters if parameter.requires_grad)
        if not reference:
            raise RuntimeError(
                "score-SE3 gradient calibration has no trainable reference parameters"
            )
        base_squared = self._gradient_squared_norm(base_loss, reference)
        score_squared = self._gradient_squared_norm(score_loss, reference)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(base_squared, op=dist.ReduceOp.SUM)
            dist.all_reduce(score_squared, op=dist.ReduceOp.SUM)
        base_norm = torch.sqrt(base_squared.clamp_min(0.0))
        score_norm = torch.sqrt(score_squared.clamp_min(0.0))
        if not torch.isfinite(base_norm) or not torch.isfinite(score_norm):
            raise RuntimeError(
                "non-finite gradient norm during score-SE3 loss calibration"
            )
        if float(score_norm.item()) <= 0.0:
            raise RuntimeError(
                "score-SE3 loss has zero gradient on the calibration parameters"
            )
        calibrated = (
            self.explicit_weight
            * self.gradient_ratio
            * float(base_norm.item())
            / float(score_norm.item())
        )
        calibrated = min(calibrated, self.max_calibrated_weight)
        with torch.no_grad():
            self.effective_weight.fill_(float(calibrated))
            self.calibrated.fill_(True)
            self.base_gradient_norm_at_calibration.copy_(base_norm.float())
            self.score_gradient_norm_at_calibration.copy_(score_norm.float())
        return base_norm, score_norm

    def forward(
        self,
        base_loss: torch.Tensor,
        prediction: torch.Tensor,
        target: torch.Tensor,
        batch: Mapping[str, object],
        *,
        training: bool,
        calibration_parameters: Iterable[nn.Parameter] = (),
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        zero = base_loss.detach() * 0.0
        if not self.enabled:
            diagnostics = {
                "enabled": zero,
                "explicit_weight": zero,
                "effective_weight": zero,
                "calibrated": zero + 1.0,
                "raw_composite": zero,
                "weighted_composite": zero,
            }
            for key in COMPONENT_KEYS:
                diagnostics[f"component/{key}"] = zero
                diagnostics[f"count/{key}"] = zero
            return base_loss, diagnostics

        score_loss, score_diagnostics = score_aligned_loss(
            prediction,
            target,
            batch,
            spec=self.spec,
        )
        if training and not bool(self.calibrated.item()):
            self._calibrate_once(
                base_loss,
                score_loss,
                calibration_parameters,
            )
        weight = self.effective_weight.to(score_loss.device, score_loss.dtype)
        weighted = weight * score_loss
        total = base_loss + weighted
        diagnostics = {
            "enabled": score_loss.detach() * 0.0 + 1.0,
            "explicit_weight": score_loss.detach() * 0.0
            + self.explicit_weight,
            "effective_weight": weight.detach(),
            "calibrated": score_loss.detach() * 0.0
            + float(self.calibrated.item()),
            "raw_composite": score_loss.detach(),
            "weighted_composite": weighted.detach(),
            "calibration/base_gradient_norm": self.base_gradient_norm_at_calibration.detach(),
            "calibration/score_gradient_norm": self.score_gradient_norm_at_calibration.detach(),
        }
        for key in COMPONENT_KEYS:
            diagnostics[f"component/{key}"] = score_diagnostics[key].detach()
            diagnostics[f"count/{key}"] = score_diagnostics[f"{key}/count"].detach()
        if not math.isfinite(float(weight.detach().cpu().item())):
            raise RuntimeError("score-SE3 effective loss weight is not finite")
        if COMPOSITE_SCORE not in score_diagnostics:
            raise RuntimeError("score-SE3 metric loss omitted its composite diagnostic")
        return total, diagnostics
