from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn

from training.long_horizon_model import LongHorizonAblationModel
from training.score_se3_eval import score_metric_spec_from_args
from training.score_se3_loss import ScoreAlignedLossController
from training.score_se3_metrics import score_aligned_loss
from training.score_se3_predictor import ScoreSE3StructuredPredictor


class ScoreSE3ExperimentalModel(LongHorizonAblationModel):
    """Stage-0 SOTA model plus the controlled continued_baseline-oracle_contact score-SE3 additions."""

    def __init__(
        self,
        args: Any,
        data_info_dict: dict[str, int],
        rank: int = 0,
        cpu_pg=None,
    ) -> None:
        candidate = str(getattr(args, "score_se3_candidate", ""))
        if candidate not in {"continued_baseline", "metric_loss", "metric_pose", "predicted_contact", "oracle_contact"}:
            raise ValueError(f"invalid score-SE3 candidate {candidate!r}")
        stage = int(os.environ.get("LONG_HORIZON_STAGE", "0"))
        track = os.environ.get("LONG_HORIZON_TRACK", "isolated").strip().lower()
        if stage != 0 or track != "isolated":
            raise ValueError(
                "score-SE3 continued_baseline-oracle_contact must use LONG_HORIZON_STAGE=0 and "
                f"LONG_HORIZON_TRACK=isolated, got stage={stage}, track={track!r}"
            )
        output_mode = os.environ.get("LONG_HORIZON_OUTPUT_MODE", "blend")
        if output_mode != "blend":
            raise ValueError(
                "score-SE3 campaign requires LONG_HORIZON_OUTPUT_MODE=blend"
            )
        object_jepa_weight = float(os.environ.get("OBJECT_JEPA_LOSS_WEIGHT", "0"))
        if object_jepa_weight != 0.0:
            raise ValueError(
                "score-SE3 continued_baseline-oracle_contact require OBJECT_JEPA_LOSS_WEIGHT=0 to preserve stage-0 control"
            )

        loss_weight = float(getattr(args, "score_se3_loss_weight", 0.0))
        gradient_ratio = float(
            getattr(args, "score_se3_gradient_ratio", 0.0)
        )
        max_calibrated_weight = float(
            getattr(args, "score_se3_max_calibrated_weight", 100.0)
        )
        calibration_only = bool(
            getattr(args, "score_se3_calibration_only", False)
        )
        require_calibration_artifact = bool(
            getattr(args, "score_se3_require_calibration_artifact", False)
        )
        calibration_artifact_sha256 = getattr(
            args, "score_se3_calibration_artifact_sha256", None
        )
        calibration_formula_version = getattr(
            args, "score_se3_calibration_formula_version", None
        )
        calibration_sample_sequence_sha256 = getattr(
            args, "score_se3_calibration_sample_sequence_sha256", None
        )
        calibration_warm_checkpoint_sha256 = getattr(
            args, "score_se3_calibration_warm_checkpoint_sha256", None
        )
        calibration_batches = int(
            getattr(args, "score_se3_calibration_batches", 16)
        )
        if not 16 <= calibration_batches <= 32:
            raise ValueError("score-SE3 calibration batches must be in [16,32]")
        if candidate == "continued_baseline" and (
            loss_weight != 0.0 or gradient_ratio != 0.0
        ):
            raise ValueError(
                "continued_baseline is the unchanged continued-training control and requires "
                "score_se3_loss_weight=0 and score_se3_gradient_ratio=0"
            )
        if calibration_only and candidate != "metric_loss":
            raise ValueError("metric-loss calibration-only mode is restricted to metric_loss")
        if calibration_only and (loss_weight != 0.0 or gradient_ratio != 0.0):
            raise ValueError(
                "metric-loss calibration-only mode requires zero configured loss "
                "weight and zero in-model gradient calibration"
            )
        if calibration_only and require_calibration_artifact:
            raise ValueError(
                "metric-loss calibration-only creates and cannot consume an artifact"
            )
        if calibration_only and not getattr(
            args, "score_se3_calibration_artifact", None
        ):
            raise ValueError(
                "metric-loss calibration-only requires an artifact destination"
            )
        if candidate == "continued_baseline" and (
            require_calibration_artifact
            or getattr(args, "score_se3_calibration_artifact", None)
            or calibration_artifact_sha256
        ):
            raise ValueError("continued_baseline forbids metric-loss calibration artifacts")
        if candidate != "continued_baseline" and not calibration_only and loss_weight <= 0.0:
            raise ValueError(f"{candidate} requires score_se3_loss_weight > 0")
        if require_calibration_artifact and candidate != "continued_baseline" and not calibration_only:
            from training.score_se3_calibration import (
                CALIBRATION_FORMULA_VERSION,
            )

            digest = str(calibration_artifact_sha256 or "")
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError(
                    f"{candidate} requires a validated metric-loss calibration artifact"
                )
            for name, value in (
                (
                    "score_se3_calibration_sample_sequence_sha256",
                    calibration_sample_sequence_sha256,
                ),
                (
                    "score_se3_calibration_warm_checkpoint_sha256",
                    calibration_warm_checkpoint_sha256,
                ),
            ):
                value = str(value or "")
                if len(value) != 64 or any(
                    character not in "0123456789abcdef" for character in value
                ):
                    raise ValueError(
                        f"{candidate} requires validated {name}"
                    )
            if calibration_formula_version != CALIBRATION_FORMULA_VERSION:
                raise ValueError(
                    f"{candidate} requires the frozen calibration formula version"
                )
            if gradient_ratio != 0.0:
                raise ValueError(
                    "artifact-calibrated production runs forbid in-model one-batch "
                    "gradient calibration"
                )
        allow_oracle = bool(getattr(args, "score_se3_allow_oracle", False))
        if candidate == "oracle_contact" and not allow_oracle:
            raise PermissionError("oracle_contact requires --score-se3-allow-oracle")
        if candidate != "oracle_contact" and allow_oracle:
            raise ValueError("oracle access is forbidden for deployable candidates")

        self.score_se3_candidate = candidate
        self.score_se3_loss_weight = loss_weight
        self.score_se3_gradient_ratio = gradient_ratio
        self.score_se3_max_calibrated_weight = max_calibrated_weight
        self.score_se3_calibration_only = calibration_only
        self.score_se3_require_calibration_artifact = require_calibration_artifact
        self.score_se3_calibration_artifact_sha256 = calibration_artifact_sha256
        self.score_se3_calibration_formula_version = calibration_formula_version
        self.score_se3_calibration_sample_sequence_sha256 = (
            calibration_sample_sequence_sha256
        )
        self.score_se3_calibration_warm_checkpoint_sha256 = (
            calibration_warm_checkpoint_sha256
        )
        self.score_se3_calibration_batches = calibration_batches
        self.score_se3_allow_oracle = allow_oracle
        self.score_se3_contact_loss_weight = float(
            getattr(args, "score_se3_contact_loss_weight", 0.1)
        )
        self.score_se3_contact_stop_gradient = bool(
            getattr(args, "score_se3_contact_stop_gradient", True)
        )
        if not self.score_se3_contact_stop_gradient:
            raise ValueError(
                "the frozen continued_baseline-oracle_contact campaign requires predicted-contact stop-gradient"
            )
        for key, value in {
            "score_se3_candidate": candidate,
            "score_se3_loss_weight": loss_weight,
            "score_se3_gradient_ratio": gradient_ratio,
            "score_se3_max_calibrated_weight": max_calibrated_weight,
            "score_se3_calibration_only": calibration_only,
            "score_se3_require_calibration_artifact": require_calibration_artifact,
            "score_se3_calibration_artifact_sha256": calibration_artifact_sha256,
            "score_se3_calibration_formula_version": calibration_formula_version,
            "score_se3_calibration_sample_sequence_sha256": (
                calibration_sample_sequence_sha256
            ),
            "score_se3_calibration_warm_checkpoint_sha256": (
                calibration_warm_checkpoint_sha256
            ),
            "score_se3_calibration_batches": calibration_batches,
            "score_se3_allow_oracle": allow_oracle,
            "score_se3_contact_loss_weight": self.score_se3_contact_loss_weight,
            "score_se3_contact_stop_gradient": self.score_se3_contact_stop_gradient,
        }.items():
            setattr(args, key, value)
            original = getattr(args, "og_args", None)
            if original is not None:
                setattr(original, key, value)

        super().__init__(args, data_info_dict, rank=rank, cpu_pg=cpu_pg)
        if candidate in {"metric_pose", "predicted_contact", "oracle_contact"}:
            if self._object_jepa_hook is not None:
                self._object_jepa_hook.remove()
            self.dynamics_predictor = ScoreSE3StructuredPredictor(
                args,
                int(self.channels),
                int(self.T),
                str(self.action_mode),
                int(self.long_horizon_stage),
            )
            self._object_jepa_hook = (
                self.dynamics_predictor.object_time.register_forward_hook(
                    self._capture_object_time_hidden
                )
            )
        self.score_se3_loss = ScoreAlignedLossController(
            enabled=candidate != "continued_baseline" and not calibration_only,
            explicit_weight=loss_weight,
            gradient_ratio=gradient_ratio,
            max_calibrated_weight=max_calibrated_weight,
            spec=score_metric_spec_from_args(args),
        )
        self._write_score_se3_model_contract(args)

    def _write_score_se3_model_contract(self, args: Any) -> None:
        output_dir = getattr(args, "output_dir", None)
        if not output_dir or int(getattr(self, "rank", 0)) != 0:
            return
        path = Path(output_dir).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "candidate": self.score_se3_candidate,
            "base_model": "LongHorizonAblationModel stage-0 isolated blend",
            "metric_matched_loss_enabled": (
                self.score_se3_candidate != "continued_baseline"
                and not self.score_se3_calibration_only
            ),
            "metric_matched_loss_weight": self.score_se3_loss_weight,
            "gradient_ratio": self.score_se3_gradient_ratio,
            "max_calibrated_weight": self.score_se3_max_calibrated_weight,
            "metric_loss_calibration_only": self.score_se3_calibration_only,
            "metric_loss_calibration_artifact_sha256": (
                self.score_se3_calibration_artifact_sha256
            ),
            "metric_loss_calibration_formula_version": getattr(
                args, "score_se3_calibration_formula_version", None
            ),
            "metric_loss_calibration_required": (
                self.score_se3_require_calibration_artifact
            ),
            "metric_loss_calibration_batches": self.score_se3_calibration_batches,
            "metric_loss_calibration_sample_sequence_sha256": (
                self.score_se3_calibration_sample_sequence_sha256
            ),
            "metric_loss_calibration_warm_checkpoint_sha256": (
                self.score_se3_calibration_warm_checkpoint_sha256
            ),
            "residual_se3_adapter_enabled": self.score_se3_candidate in {"metric_pose", "predicted_contact", "oracle_contact"},
            "contact_conditioning": {
                "continued_baseline": "none",
                "metric_loss": "none",
                "metric_pose": "ten_literal_zero_channels",
                "predicted_contact": "predicted_continuous_contact",
                "oracle_contact": "future_gt_contact_oracle",
            }[self.score_se3_candidate],
            "contact_aux_loss_weight": self.score_se3_contact_loss_weight,
            "predicted_contact_stop_gradient": self.score_se3_contact_stop_gradient,
            "contact_aux_points": int(self.long_horizon_contact_points),
            "contact_aux_near_weight": float(
                self.long_horizon_contact_near_weight
            ),
            "contact_aux_tau_m": float(self.long_horizon_contact_tau),
            "oracle": self.score_se3_candidate == "oracle_contact",
            "deployable": self.score_se3_candidate != "oracle_contact",
        }
        (path / "score_se3_model_contract.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if self.score_se3_candidate == "oracle_contact":
            (path / "ORACLE_NON_DEPLOYABLE").write_text(
                "oracle_contact consumes future ground-truth object geometry to derive contact.\n",
                encoding="utf-8",
            )

    def _score_calibration_parameters(self) -> Iterable[nn.Parameter]:
        predictor = self.dynamics_predictor
        pose = getattr(predictor, "pose", None)
        if not isinstance(pose, nn.Module):
            raise RuntimeError("score-SE3 calibration requires the object pose head")
        return pose.parameters()

    def score_se3_calibration_losses(self, outputs, data):
        """Return the unmodified current loss and raw metric loss for metric_loss.

        This bypasses :class:`ScoreAlignedLossController` deliberately: the
        calibration job measures both gradients independently and performs no
        training update.
        """

        if not self.score_se3_calibration_only or self.score_se3_candidate != "metric_loss":
            raise RuntimeError(
                "score_se3_calibration_losses is available only in metric_loss "
                "calibration-only mode"
            )
        current_loss, _ = super().loss_fn(outputs, data, training=True)
        prediction = outputs.get("scene_flows")
        target = data.get("gt_scene_flows")
        if not isinstance(prediction, torch.Tensor) or not isinstance(
            target, torch.Tensor
        ):
            raise KeyError(
                "metric-loss calibration requires scene_flows and gt_scene_flows"
            )
        score_loss, _ = score_aligned_loss(
            prediction,
            target.to(prediction.device),
            data,
            spec=self.score_se3_loss.spec,
        )
        return current_loss, score_loss

    def loss_fn(self, outputs, data, training=True):
        if bool(getattr(self, "score_se3_calibration_only", False)):
            raise RuntimeError(
                "calibration-only metric_loss must use the no-update calibration runner, "
                "not the training loss path"
            )
        base_loss, log = super().loss_fn(outputs, data, training=training)
        prediction = outputs.get("scene_flows")
        target = data.get("gt_scene_flows")
        if not isinstance(prediction, torch.Tensor) or not isinstance(
            target, torch.Tensor
        ):
            raise KeyError(
                "score-SE3 loss requires outputs['scene_flows'] and "
                "data['gt_scene_flows']"
            )
        total, diagnostics = self.score_se3_loss(
            base_loss,
            prediction,
            target.to(prediction.device),
            data,
            training=bool(training),
            calibration_parameters=self._score_calibration_parameters(),
        )
        if self.score_se3_candidate == "predicted_contact" and self.score_se3_contact_loss_weight > 0:
            contact_loss = self._contact_aux_terms(outputs, data)
            weighted_contact = self.score_se3_contact_loss_weight * contact_loss
            total = total + weighted_contact
            log["score_se3/contact_aux"] = float(contact_loss.detach().cpu())
            log["score_se3/contact_aux_weighted"] = float(
                weighted_contact.detach().cpu()
            )
        for key, value in diagnostics.items():
            log[f"score_se3/{key}"] = float(
                value.detach().float().cpu().item()
            )
        log["total_loss"] = float(total.detach().float().cpu().item())
        return total, log
