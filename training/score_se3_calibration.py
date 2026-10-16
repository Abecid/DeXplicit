"""Immutable pre-campaign calibration for the metric-aligned loss weight.

The production campaign does not tune ``lambda_score`` independently for each
condition.  A separate, no-update metric_loss job measures pose-head gradient norms on
the first deterministic training batches for a seed and writes one immutable
artifact.  metric_loss--oracle_contact for that seed must all consume that exact artifact.

The frozen formula is::

    lambda_score = 0.3 * mean_batch(g_current) / mean_batch(g_score)

where both norms are measured on ``dynamics_predictor.pose.*`` and ``g_score``
is the variance-independent metric-aligned raw geometry loss.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn as nn


CALIBRATION_ARTIFACT_TYPE = "score_se3_metric_loss_gradient_calibration"
CALIBRATION_SCHEMA_VERSION = 1
CALIBRATION_FORMULA_VERSION = "pose_path_mean_gradient_norm_ratio_v1"
CALIBRATION_FORMULA = (
    "lambda_score = 0.3 * mean(g_current) / mean(g_score)"
)
CALIBRATION_TARGET_RATIO = 0.3
CALIBRATION_PARAMETER_SCOPE = "dynamics_predictor.pose.*"
MIN_CALIBRATION_BATCHES = 16
MAX_CALIBRATION_BATCHES = 32
DEFAULT_CALIBRATION_BATCHES = 16


def sha256_file(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"cannot hash missing file: {resolved}")
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        while True:
            chunk = handle.read(int(chunk_size))
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _git_sha(repo_root: str | Path) -> str:
    root = Path(repo_root).expanduser().resolve()
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            f"metric-loss calibration cannot resolve git SHA at {root}"
        ) from exc
    commit = completed.stdout.strip()
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise RuntimeError(f"invalid git commit returned for calibration: {commit!r}")
    return commit


def _positive_finite(value: Any, name: str, *, allow_zero: bool = False) -> float:
    value = float(value)
    valid_sign = value >= 0.0 if allow_zero else value > 0.0
    if not math.isfinite(value) or not valid_sign:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {qualifier}, got {value!r}")
    return value


def _validate_batch_count(value: Any) -> int:
    if isinstance(value, bool) or int(value) != value:
        raise ValueError("calibration batch_count must be an integer")
    value = int(value)
    if not MIN_CALIBRATION_BATCHES <= value <= MAX_CALIBRATION_BATCHES:
        raise ValueError(
            "calibration batch_count must be in "
            f"[{MIN_CALIBRATION_BATCHES},{MAX_CALIBRATION_BATCHES}], got {value}"
        )
    return value


def _validate_hex_digest(value: Any, name: str, length: int = 64) -> str:
    value = str(value)
    if len(value) != int(length) or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase {length}-character hex digest")
    return value


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def calibration_sample_identity(batch: Mapping[str, Any]) -> list[dict[str, str]]:
    """Return the ordered, stable identities for one calibration batch."""

    keys = batch.get("__key__")
    if keys is None or isinstance(keys, torch.Tensor):
        raise KeyError(
            "metric-loss calibration requires batch['__key__'] sample identities"
        )
    datasets = batch.get("__dataset__")
    if datasets is None:
        datasets = ["unknown"] * len(keys)
    if isinstance(datasets, torch.Tensor) or len(datasets) != len(keys):
        raise ValueError(
            "batch['__dataset__'] must contain one value per calibration sample"
        )
    if not keys:
        raise ValueError("calibration batch cannot be empty")
    return [
        {"dataset": str(dataset), "sample_key": str(key)}
        for dataset, key in zip(datasets, keys)
    ]


def sample_sequence_sha256(
    batch_identities: Sequence[Sequence[Mapping[str, str]]],
) -> str:
    canonical = [
        [
            {
                "dataset": str(sample["dataset"]),
                "sample_key": str(sample["sample_key"]),
            }
            for sample in batch
        ]
        for batch in batch_identities
    ]
    encoded = json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def pose_path_parameters(model: nn.Module) -> tuple[nn.Parameter, ...]:
    model = model.module if hasattr(model, "module") else model
    predictor = getattr(model, "dynamics_predictor", None)
    pose = getattr(predictor, "pose", None)
    if not isinstance(pose, nn.Module):
        raise RuntimeError(
            "metric-loss calibration requires dynamics_predictor.pose"
        )
    parameters = tuple(
        parameter for parameter in pose.parameters() if parameter.requires_grad
    )
    if not parameters:
        raise RuntimeError("metric-loss calibration pose path has no trainable parameters")
    return parameters


def gradient_l2_norm(
    loss: torch.Tensor,
    parameters: Iterable[nn.Parameter],
    *,
    retain_graph: bool,
) -> torch.Tensor:
    """Return the raw L2 norm without populating ``Parameter.grad``."""

    reference = tuple(parameters)
    if not reference:
        raise RuntimeError("gradient norm requires at least one reference parameter")
    gradients = torch.autograd.grad(
        loss,
        reference,
        retain_graph=bool(retain_graph),
        create_graph=False,
        allow_unused=True,
    )
    squared = loss.new_zeros((), dtype=torch.float32)
    used = 0
    for gradient in gradients:
        if gradient is None:
            continue
        used += 1
        squared = squared + gradient.detach().float().square().sum()
    if used == 0:
        raise RuntimeError("calibration loss has no gradient on the pose path")
    norm = torch.sqrt(squared.clamp_min(0.0))
    if not bool(torch.isfinite(norm).item()):
        raise RuntimeError("non-finite pose-path gradient norm during calibration")
    return norm


@dataclass
class GradientCalibrationAccumulator:
    """Collect per-batch norms and finalize the frozen formula."""

    expected_batches: int = DEFAULT_CALIBRATION_BATCHES

    def __post_init__(self) -> None:
        self.expected_batches = _validate_batch_count(self.expected_batches)
        self.current_gradient_norms: list[float] = []
        self.score_gradient_norms: list[float] = []
        self.batch_identities: list[list[dict[str, str]]] = []

    def add(
        self,
        current_loss: torch.Tensor,
        score_loss: torch.Tensor,
        parameters: Iterable[nn.Parameter],
        batch: Mapping[str, Any],
    ) -> tuple[float, float]:
        if len(self.current_gradient_norms) >= self.expected_batches:
            raise RuntimeError("calibration received more batches than declared")
        reference = tuple(parameters)
        current_norm = gradient_l2_norm(
            current_loss, reference, retain_graph=True
        )
        score_norm = gradient_l2_norm(score_loss, reference, retain_graph=False)
        current_value = _positive_finite(
            current_norm.detach().cpu().item(),
            "g_current",
            allow_zero=True,
        )
        score_value = _positive_finite(
            score_norm.detach().cpu().item(), "g_score"
        )
        self.current_gradient_norms.append(current_value)
        self.score_gradient_norms.append(score_value)
        self.batch_identities.append(calibration_sample_identity(batch))
        return current_value, score_value

    def measurements(self) -> dict[str, Any]:
        observed = len(self.current_gradient_norms)
        if observed != self.expected_batches:
            raise RuntimeError(
                "calibration batch count mismatch: "
                f"expected={self.expected_batches}, observed={observed}"
            )
        mean_current = sum(self.current_gradient_norms) / observed
        mean_score = sum(self.score_gradient_norms) / observed
        _positive_finite(mean_current, "mean_g_current", allow_zero=True)
        _positive_finite(mean_score, "mean_g_score")
        loss_weight = (
            CALIBRATION_TARGET_RATIO * mean_current / mean_score
        )
        _positive_finite(loss_weight, "lambda_score")
        return {
            "batch_count": observed,
            "sample_count": sum(len(batch) for batch in self.batch_identities),
            "sample_sequence_sha256": sample_sequence_sha256(
                self.batch_identities
            ),
            "g_current_per_batch": list(self.current_gradient_norms),
            "g_score_per_batch": list(self.score_gradient_norms),
            "mean_g_current": mean_current,
            "mean_g_score": mean_score,
            "lambda_score": loss_weight,
        }


def calibration_provenance_from_args(
    args: Any,
    *,
    repo_root: str | Path,
    require_warm_checkpoint: bool,
) -> dict[str, Any]:
    """Resolve the causal provenance that every consumer must match."""

    manifest_path = getattr(args, "score_se3_manifest_path", None)
    if not manifest_path:
        raise ValueError("metric-loss calibration requires --score-se3-manifest-path")
    warm_checkpoint_sha256 = None
    if require_warm_checkpoint:
        checkpoint_path = getattr(args, "model_path", None)
        if not checkpoint_path:
            raise ValueError("metric-loss calibration requires the warm checkpoint")
        warm_checkpoint_sha256 = sha256_file(checkpoint_path)
    return {
        "source_candidate": "metric_loss",
        "seed": int(getattr(args, "seed")),
        "warm_checkpoint_sha256": warm_checkpoint_sha256,
        "split_manifest_sha256": sha256_file(manifest_path),
        "git_sha": _git_sha(repo_root),
        "batch_count": _validate_batch_count(
            getattr(
                args,
                "score_se3_calibration_batches",
                DEFAULT_CALIBRATION_BATCHES,
            )
        ),
        "batch_selection": "first_n_seeded_full_coverage_training_batches",
        "model_mode": "eval_with_gradients",
        "optimizer_updates": 0,
        "campaign_steps": 0,
    }


def build_calibration_artifact(
    *,
    provenance: Mapping[str, Any],
    measurements: Mapping[str, Any],
) -> dict[str, Any]:
    batch_count = _validate_batch_count(provenance.get("batch_count"))
    if int(measurements.get("batch_count", -1)) != batch_count:
        raise ValueError("measurement/provenance calibration batch counts disagree")
    artifact = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "artifact_type": CALIBRATION_ARTIFACT_TYPE,
        "formula": {
            "version": CALIBRATION_FORMULA_VERSION,
            "expression": CALIBRATION_FORMULA,
            "target_gradient_ratio": CALIBRATION_TARGET_RATIO,
            "parameter_scope": CALIBRATION_PARAMETER_SCOPE,
            "batch_reduction": "arithmetic_mean_of_per_batch_l2_norms",
            "uncertainty_weighting_applied_to_score": False,
            "clipped_or_capped": False,
        },
        "provenance": dict(provenance),
        "measurements": dict(measurements),
        "consumer_contract": {
            "allowed_candidates": ["metric_loss", "metric_pose", "predicted_contact", "oracle_contact"],
            "same_artifact_required_within_seed": True,
            "E0_weight": 0.0,
        },
    }
    validate_calibration_artifact(artifact)
    return artifact


def validate_calibration_artifact(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("calibration artifact must be a mapping")
    if int(payload.get("schema_version", -1)) != CALIBRATION_SCHEMA_VERSION:
        raise ValueError("unsupported metric-loss calibration schema")
    if payload.get("artifact_type") != CALIBRATION_ARTIFACT_TYPE:
        raise ValueError("unexpected metric-loss calibration artifact type")
    formula = payload.get("formula")
    provenance = payload.get("provenance")
    measurements = payload.get("measurements")
    consumers = payload.get("consumer_contract")
    if not all(
        isinstance(value, Mapping)
        for value in (formula, provenance, measurements, consumers)
    ):
        raise ValueError("calibration artifact sections must be mappings")
    expected_formula = {
        "version": CALIBRATION_FORMULA_VERSION,
        "expression": CALIBRATION_FORMULA,
        "target_gradient_ratio": CALIBRATION_TARGET_RATIO,
        "parameter_scope": CALIBRATION_PARAMETER_SCOPE,
        "batch_reduction": "arithmetic_mean_of_per_batch_l2_norms",
        "uncertainty_weighting_applied_to_score": False,
        "clipped_or_capped": False,
    }
    if dict(formula) != expected_formula:
        raise ValueError("metric-loss calibration formula/version changed")
    if provenance.get("source_candidate") != "metric_loss":
        raise ValueError("metric-loss calibration must be generated by metric_loss")
    if isinstance(provenance.get("seed"), bool):
        raise ValueError("metric-loss calibration seed must be an integer")
    int(provenance.get("seed"))
    _validate_hex_digest(
        provenance.get("warm_checkpoint_sha256"), "warm_checkpoint_sha256"
    )
    _validate_hex_digest(
        provenance.get("split_manifest_sha256"), "split_manifest_sha256"
    )
    _validate_hex_digest(provenance.get("git_sha"), "git_sha", length=40)
    if provenance.get("batch_selection") != (
        "first_n_seeded_full_coverage_training_batches"
    ):
        raise ValueError("metric-loss calibration batch-selection contract changed")
    if provenance.get("model_mode") != "eval_with_gradients":
        raise ValueError("metric-loss calibration model mode changed")
    batch_count = _validate_batch_count(provenance.get("batch_count"))
    if int(provenance.get("optimizer_updates", -1)) != 0 or int(
        provenance.get("campaign_steps", -1)
    ) != 0:
        raise ValueError("metric-loss calibration artifact is not no-update")
    if int(measurements.get("batch_count", -1)) != batch_count:
        raise ValueError("metric-loss calibration measurement batch count changed")
    sample_count = measurements.get("sample_count")
    if isinstance(sample_count, bool) or int(sample_count) != sample_count or int(
        sample_count
    ) <= 0:
        raise ValueError("calibration sample_count must be a positive integer")
    current = measurements.get("g_current_per_batch")
    score = measurements.get("g_score_per_batch")
    if not isinstance(current, list) or not isinstance(score, list):
        raise ValueError("calibration per-batch norms must be lists")
    if len(current) != batch_count or len(score) != batch_count:
        raise ValueError("calibration per-batch norm lengths disagree with batch count")
    current_values = [
        _positive_finite(value, "g_current", allow_zero=True) for value in current
    ]
    score_values = [_positive_finite(value, "g_score") for value in score]
    mean_current = sum(current_values) / batch_count
    mean_score = sum(score_values) / batch_count
    expected_lambda = CALIBRATION_TARGET_RATIO * mean_current / mean_score
    for name, observed, expected in (
        ("mean_g_current", measurements.get("mean_g_current"), mean_current),
        ("mean_g_score", measurements.get("mean_g_score"), mean_score),
        ("lambda_score", measurements.get("lambda_score"), expected_lambda),
    ):
        observed_value = _positive_finite(
            observed, name, allow_zero=name == "mean_g_current"
        )
        if not math.isclose(
            observed_value, expected, rel_tol=1e-12, abs_tol=1e-15
        ):
            raise ValueError(f"calibration artifact {name} does not match raw norms")
    _validate_hex_digest(
        measurements.get("sample_sequence_sha256"),
        "sample_sequence_sha256",
    )
    expected_consumers = {
        "allowed_candidates": ["metric_loss", "metric_pose", "predicted_contact", "oracle_contact"],
        "same_artifact_required_within_seed": True,
        "E0_weight": 0.0,
    }
    if dict(consumers) != expected_consumers:
        raise ValueError("metric-loss calibration consumer contract changed")
    return dict(payload)


def load_calibration_artifact(path: str | Path) -> tuple[dict[str, Any], str]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"missing metric-loss calibration artifact: {resolved}")
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid metric-loss calibration artifact: {resolved}") from exc
    return validate_calibration_artifact(payload), sha256_file(resolved)


def write_immutable_calibration_artifact(
    path: str | Path,
    payload: Mapping[str, Any],
) -> tuple[Path, str]:
    """Create once, or accept an exactly byte-identical idempotent rerun."""

    validate_calibration_artifact(payload)
    resolved = Path(path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    encoded = canonical_json_bytes(payload)
    try:
        descriptor = os.open(
            resolved,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o444,
        )
    except FileExistsError:
        observed = resolved.read_bytes()
        if observed != encoded:
            raise RuntimeError(
                "immutable metric-loss calibration artifact already exists with "
                f"different content: {resolved}"
            )
    else:
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
        except Exception:
            # A partial exclusive file is evidence of an interrupted write.
            # Leave it in place so a later run fails closed rather than
            # silently replacing provenance.
            raise
    return resolved, sha256_file(resolved)


def validate_artifact_for_run(
    payload: Mapping[str, Any],
    *,
    candidate: str,
    expected_provenance: Mapping[str, Any],
    require_warm_checkpoint_match: bool,
) -> float:
    artifact = validate_calibration_artifact(payload)
    candidate = str(candidate)
    if candidate not in {"metric_loss", "metric_pose", "predicted_contact", "oracle_contact"}:
        raise ValueError("only metric_loss-oracle_contact may consume a calibration artifact")
    provenance = artifact["provenance"]
    compared_fields = (
        "seed",
        "split_manifest_sha256",
        "git_sha",
        "batch_count",
        "batch_selection",
        "model_mode",
        "optimizer_updates",
        "campaign_steps",
    )
    mismatches = {
        field: {
            "artifact": provenance.get(field),
            "run": expected_provenance.get(field),
        }
        for field in compared_fields
        if provenance.get(field) != expected_provenance.get(field)
    }
    if require_warm_checkpoint_match and provenance.get(
        "warm_checkpoint_sha256"
    ) != expected_provenance.get("warm_checkpoint_sha256"):
        mismatches["warm_checkpoint_sha256"] = {
            "artifact": provenance.get("warm_checkpoint_sha256"),
            "run": expected_provenance.get("warm_checkpoint_sha256"),
        }
    if mismatches:
        raise RuntimeError(
            "metric-loss calibration provenance mismatch: "
            f"{json.dumps(mismatches, sort_keys=True)}"
        )
    return _positive_finite(
        artifact["measurements"]["lambda_score"], "lambda_score"
    )


def apply_calibration_artifact_to_args(
    args: Any,
    *,
    repo_root: str | Path,
) -> dict[str, Any]:
    """Load, provenance-check, and install the explicit score-loss weight."""

    candidate = str(getattr(args, "score_se3_candidate", ""))
    path = getattr(args, "score_se3_calibration_artifact", None)
    if candidate == "continued_baseline":
        if path:
            raise ValueError("continued_baseline forbids a metric-loss calibration artifact")
        args.score_se3_loss_weight = 0.0
        args.score_se3_calibration_artifact_sha256 = None
        args.score_se3_calibration_formula_version = CALIBRATION_FORMULA_VERSION
        args.score_se3_calibration_sample_sequence_sha256 = None
        return {}
    if candidate not in {"metric_loss", "metric_pose", "predicted_contact", "oracle_contact"}:
        raise ValueError(f"invalid score-SE3 candidate {candidate!r}")
    if not path:
        raise ValueError(
            f"{candidate} requires --score-se3-calibration-artifact"
        )
    payload, artifact_sha256 = load_calibration_artifact(path)
    role = str(getattr(args, "score_se3_checkpoint_role", "init"))
    expected = calibration_provenance_from_args(
        args,
        repo_root=repo_root,
        require_warm_checkpoint=role == "init",
    )
    # Held-out (checkpoint_role=eval) processes may run at a HEAD advanced
    # past the campaign SHA by post-selection tooling fixes.  The operator
    # accepts exactly one stale SHA via ALLOW_STALE_SELECTION_GIT_SHA (the
    # same override the final-eval launchers honor); it must equal the
    # artifact's recorded SHA verbatim.  Training/init/resume roles never
    # honor the override.
    stale_sha = os.environ.get("ALLOW_STALE_SELECTION_GIT_SHA", "").strip()
    if (
        role == "eval"
        and len(stale_sha) == 40
        and all(c in "0123456789abcdef" for c in stale_sha)
        and payload.get("provenance", {}).get("git_sha") == stale_sha
    ):
        print(
            "[score-se3-calibration] WARNING: accepting stale calibration "
            f"git SHA {stale_sha} in eval-role process at {expected['git_sha']} "
            "(operator override)"
        )
        expected["git_sha"] = stale_sha
    weight = validate_artifact_for_run(
        payload,
        candidate=candidate,
        expected_provenance=expected,
        require_warm_checkpoint_match=role == "init",
    )
    maximum_weight = _positive_finite(
        getattr(args, "score_se3_max_calibrated_weight", 100.0),
        "score_se3_max_calibrated_weight",
    )
    if weight > maximum_weight:
        raise RuntimeError(
            "metric-loss calibration produced an unsafe loss weight: "
            f"lambda_score={weight:.9g} exceeds configured fail-closed maximum "
            f"{maximum_weight:.9g}; the artifact is never silently clipped"
        )
    args.score_se3_loss_weight = float(weight)
    args.score_se3_gradient_ratio = 0.0
    args.score_se3_calibration_artifact = str(Path(path).expanduser().resolve())
    args.score_se3_calibration_artifact_sha256 = artifact_sha256
    args.score_se3_calibration_formula_version = CALIBRATION_FORMULA_VERSION
    args.score_se3_calibration_sample_sequence_sha256 = payload["measurements"][
        "sample_sequence_sha256"
    ]
    args.score_se3_calibration_warm_checkpoint_sha256 = payload["provenance"][
        "warm_checkpoint_sha256"
    ]
    return payload


def run_metric_loss_calibration(
    trainer: Any,
    *,
    repo_root: str | Path,
) -> tuple[Path, str, dict[str, Any]]:
    """Execute metric_loss's deterministic, no-update pre-campaign calibration."""

    args = trainer.args
    if not bool(getattr(args, "score_se3_calibration_only", False)):
        raise RuntimeError("calibration runner requires calibration-only mode")
    if str(getattr(args, "score_se3_candidate", "")) != "metric_loss":
        raise RuntimeError("calibration runner is restricted to metric_loss")
    if bool(getattr(args, "distributed", False)) or int(
        getattr(trainer, "world_size", 1)
    ) != 1:
        raise RuntimeError("metric-loss calibration must run on exactly one process/GPU")
    if int(getattr(trainer, "optimizer_step", 0)) != 0 or int(
        getattr(trainer, "score_se3_campaign_step", 0)
    ) != 0:
        raise RuntimeError("metric-loss calibration must start before campaign step zero")
    optimizer = getattr(trainer, "optimizer", None)
    if optimizer is None:
        raise RuntimeError("metric-loss calibration requires the declared optimizer")
    if optimizer.state:
        raise RuntimeError("metric-loss calibration optimizer already has update state")
    destination = getattr(args, "score_se3_calibration_artifact", None)
    if not destination:
        raise ValueError("calibration-only mode requires an artifact destination")

    warm_cache = getattr(trainer, "_warmup_dino_cache", None)
    if callable(warm_cache):
        warm_cache()
    model = trainer.model
    core_model = model.module if hasattr(model, "module") else model
    calibration_losses = getattr(core_model, "score_se3_calibration_losses", None)
    if not callable(calibration_losses):
        raise RuntimeError("score-SE3 model lacks calibration loss access")
    reference = pose_path_parameters(core_model)
    batch_count = _validate_batch_count(
        getattr(args, "score_se3_calibration_batches", DEFAULT_CALIBRATION_BATCHES)
    )
    accumulator = GradientCalibrationAccumulator(batch_count)
    iterator = getattr(trainer, "train_iter", None)
    if iterator is None:
        iterator = iter(trainer.train_dataloader)
    was_training = bool(model.training)
    model.eval()
    model.zero_grad(set_to_none=True)
    try:
        for batch_index in range(batch_count):
            try:
                batch = next(iterator)
            except StopIteration as exc:
                raise RuntimeError(
                    "training loader exhausted before the declared calibration "
                    f"batch count ({batch_index}/{batch_count})"
                ) from exc
            if not isinstance(batch, Mapping):
                raise TypeError("calibration training batch must be a mapping")
            device_batch = {
                key: (
                    value.to(trainer.device, non_blocking=True)
                    if isinstance(value, torch.Tensor)
                    else value
                )
                for key, value in batch.items()
            }
            if getattr(trainer.device, "type", str(trainer.device)) == "cuda":
                autocast_context = torch.autocast(
                    device_type="cuda", dtype=trainer.amp_dtype
                )
            else:
                autocast_context = nullcontext()
            with autocast_context:
                outputs = model(device_batch, training=True)
                current_loss, score_loss = calibration_losses(
                    outputs, device_batch
                )
            accumulator.add(
                current_loss,
                score_loss,
                reference,
                batch,
            )
            model.zero_grad(set_to_none=True)
    finally:
        model.train(was_training)
        model.zero_grad(set_to_none=True)

    if optimizer.state:
        raise RuntimeError("metric-loss calibration created optimizer state")
    if any(parameter.grad is not None for parameter in core_model.parameters()):
        raise RuntimeError("metric-loss calibration populated persistent gradients")
    if int(getattr(trainer, "optimizer_step", 0)) != 0 or int(
        getattr(trainer, "score_se3_campaign_step", 0)
    ) != 0:
        raise RuntimeError("metric-loss calibration mutated training counters")

    provenance = calibration_provenance_from_args(
        args,
        repo_root=repo_root,
        require_warm_checkpoint=True,
    )
    measurements = accumulator.measurements()
    maximum_weight = _positive_finite(
        getattr(args, "score_se3_max_calibrated_weight", 100.0),
        "score_se3_max_calibrated_weight",
    )
    if float(measurements["lambda_score"]) > maximum_weight:
        raise RuntimeError(
            "metric-loss calibration lambda exceeds the configured fail-closed "
            f"maximum: lambda_score={measurements['lambda_score']:.9g}, "
            f"maximum={maximum_weight:.9g}; no artifact was written"
        )
    artifact = build_calibration_artifact(
        provenance=provenance,
        measurements=measurements,
    )
    path, digest = write_immutable_calibration_artifact(destination, artifact)
    return path, digest, artifact


__all__ = [
    "CALIBRATION_ARTIFACT_TYPE",
    "CALIBRATION_FORMULA",
    "CALIBRATION_FORMULA_VERSION",
    "CALIBRATION_PARAMETER_SCOPE",
    "CALIBRATION_SCHEMA_VERSION",
    "CALIBRATION_TARGET_RATIO",
    "DEFAULT_CALIBRATION_BATCHES",
    "GradientCalibrationAccumulator",
    "MAX_CALIBRATION_BATCHES",
    "MIN_CALIBRATION_BATCHES",
    "apply_calibration_artifact_to_args",
    "build_calibration_artifact",
    "calibration_provenance_from_args",
    "calibration_sample_identity",
    "gradient_l2_norm",
    "load_calibration_artifact",
    "pose_path_parameters",
    "run_metric_loss_calibration",
    "sample_sequence_sha256",
    "sha256_file",
    "validate_artifact_for_run",
    "validate_calibration_artifact",
    "write_immutable_calibration_artifact",
]
