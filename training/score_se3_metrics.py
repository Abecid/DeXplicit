from __future__ import annotations

import math
import json
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Mapping

import torch
import torch.distributed as dist


MOVED_T01_T34 = "l2_moved/mean_t01_t34"
MOVED_T01_T14 = "l2_moved/mean_t01_t14"
EVER_MOVED_T34 = "horizon_exact/l2_ever_moved_t34"
COMPOSITE_SCORE = "score_se3/composite"

COMPONENT_KEYS = (MOVED_T01_T34, MOVED_T01_T14, EVER_MOVED_T34)
DEFAULT_COMPONENT_WEIGHTS: dict[str, float] = {
    MOVED_T01_T34: 0.45,
    MOVED_T01_T14: 0.20,
    EVER_MOVED_T34: 0.35,
}
SOURCE_CLIP_METADATA_KEYS = (
    "source_clip_id",
    "clip_id",
    "clip_uid",
    "source_id",
    "sequence_id",
)


def _normalized_source_clip(value: Any) -> str | None:
    if value is None or isinstance(value, bool):
        return None
    normalized = str(value).strip()
    return normalized or None


_AUDITED_CLIP_PATTERN = re.compile(r"^(clip-\d{6})(?:$|[-._])")


@lru_cache(maxsize=4)
def _audited_manifest_clips(manifest_path: str) -> frozenset[str] | None:
    """Audited clip universe from the split manifest, or None if unusable.

    The manifest is only consulted when it explicitly records the
    ``allow_filename_source_inference`` audit exception made at split-build
    time; its SHA is bound by the campaign's split verification receipt.
    """
    try:
        payload = json.loads(open(manifest_path, encoding="utf-8").read())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, Mapping):
        return None
    if not bool(payload.get("allow_filename_source_inference")):
        return None
    clips: set[str] = set()
    subsets = payload.get("subsets")
    if not isinstance(subsets, Mapping):
        return None
    for subset in subsets.values():
        if isinstance(subset, Mapping):
            for clip in subset.get("source_clips", []) or []:
                normalized = _normalized_source_clip(clip)
                if normalized is not None:
                    clips.add(normalized)
    return frozenset(clips) if clips else None


def _source_clip_from_audited_manifest(sample_key: Any) -> str | None:
    """Closed-world identity fallback validated against the audited manifest.

    Only active when SCORE_SE3_SPLIT_MANIFEST records the filename-inference
    audit exception. The derived id must be a member of the manifest's clip
    universe; anything else stays unidentified (and the caller raises).
    """
    manifest_path = os.environ.get("SCORE_SE3_SPLIT_MANIFEST", "").strip()
    if not manifest_path:
        return None
    clips = _audited_manifest_clips(manifest_path)
    if not clips:
        return None
    stem = os.path.basename(str(sample_key).strip())
    match = _AUDITED_CLIP_PATTERN.match(stem)
    if match is None:
        return None
    candidate = match.group(1)
    return candidate if candidate in clips else None


def _source_clip_from_metadata_json(value: Any) -> str | None:
    if value is None:
        return None
    try:
        payload = json.loads(str(value))
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(payload, Mapping):
        return None
    for key in SOURCE_CLIP_METADATA_KEYS:
        normalized = _normalized_source_clip(payload.get(key))
        if normalized is not None:
            return normalized
    return None


@dataclass(frozen=True)
class ScoreMetricSpec:
    """Immutable definition shared by score-aligned training and evaluation."""

    ever_moved_threshold_m: float = 0.005
    smooth_epsilon_m: float = 0.001
    early_end_index: int = 14
    final_index: int = 34
    moved_t01_t34_weight: float = 0.45
    moved_t01_t14_weight: float = 0.20
    ever_moved_t34_weight: float = 0.35

    def __post_init__(self) -> None:
        if self.ever_moved_threshold_m < 0:
            raise ValueError("ever_moved_threshold_m must be non-negative")
        if self.smooth_epsilon_m < 0:
            raise ValueError("smooth_epsilon_m must be non-negative")
        if self.early_end_index < 1:
            raise ValueError("early_end_index must be at least 1")
        if self.final_index < self.early_end_index:
            raise ValueError("final_index must be >= early_end_index")
        weights = self.component_weights
        if any(weight < 0 for weight in weights.values()):
            raise ValueError("score component weights must be non-negative")
        if not math.isclose(sum(weights.values()), 1.0, rel_tol=0.0, abs_tol=1e-8):
            raise ValueError("score component weights must sum to 1")

    @property
    def component_weights(self) -> dict[str, float]:
        return {
            MOVED_T01_T34: float(self.moved_t01_t34_weight),
            MOVED_T01_T14: float(self.moved_t01_t14_weight),
            EVER_MOVED_T34: float(self.ever_moved_t34_weight),
        }


def _bool_btn(
    value: torch.Tensor,
    *,
    name: str,
    shape: tuple[int, int, int],
) -> torch.Tensor:
    if value.ndim == 4 and value.shape[-1] == 1:
        value = value.squeeze(-1)
    if tuple(value.shape) != shape:
        raise ValueError(
            f"{name} must have shape {shape} or {shape + (1,)}, "
            f"got {tuple(value.shape)}"
        )
    return value.bool()


def validate_score_inputs(
    prediction: torch.Tensor,
    target: torch.Tensor,
    batch: Mapping[str, Any],
    spec: ScoreMetricSpec,
) -> tuple[int, int, int]:
    if prediction.shape != target.shape:
        raise ValueError(
            "prediction and target must have identical shapes, got "
            f"{tuple(prediction.shape)} and {tuple(target.shape)}"
        )
    if prediction.ndim != 4 or prediction.shape[-1] != 3:
        raise ValueError(
            "prediction and target must have shape (B,T,N,3), got "
            f"{tuple(prediction.shape)}"
        )
    batch_size, timesteps, points, _ = prediction.shape
    if timesteps <= spec.final_index:
        raise ValueError(
            f"score metric final_index={spec.final_index} requires at least "
            f"{spec.final_index + 1} timesteps, got {timesteps}"
        )
    required = (
        "scene_context_mask",
        "scene_exists",
        "scene_supervised_mask",
        "scene_moved_mask",
    )
    missing = [name for name in required if name not in batch]
    if missing:
        raise KeyError(f"score metric batch is missing required keys: {missing}")
    shape = (batch_size, timesteps, points)
    for name in required:
        value = batch[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"batch[{name!r}] must be a torch.Tensor")
        _bool_btn(value, name=name, shape=shape)
    return batch_size, timesteps, points


def score_masks(
    target: torch.Tensor,
    batch: Mapping[str, Any],
    spec: ScoreMetricSpec | None = None,
) -> dict[str, torch.Tensor]:
    """Build the exact masks used by the three declared LH35 score components."""

    spec = spec or ScoreMetricSpec()
    validate_score_inputs(target, target, batch, spec)
    shape = tuple(target.shape[:3])
    device = target.device

    context = _bool_btn(
        batch["scene_context_mask"].to(device),
        name="scene_context_mask",
        shape=shape,
    )
    exists = _bool_btn(
        batch["scene_exists"].to(device),
        name="scene_exists",
        shape=shape,
    )
    supervised = _bool_btn(
        batch["scene_supervised_mask"].to(device),
        name="scene_supervised_mask",
        shape=shape,
    )
    moved = _bool_btn(
        batch["scene_moved_mask"].to(device),
        name="scene_moved_mask",
        shape=shape,
    )
    valid = (~context) & exists & supervised

    displacement = torch.linalg.norm(target - target[:, :1], dim=-1)
    ever_moved = displacement.max(dim=1).values > spec.ever_moved_threshold_m

    time = torch.arange(target.shape[1], device=device)
    through_final = (time >= 1) & (time <= spec.final_index)
    through_early = (time >= 1) & (time <= spec.early_end_index)

    return {
        MOVED_T01_T34: valid & moved & through_final.view(1, -1, 1),
        MOVED_T01_T14: valid & moved & through_early.view(1, -1, 1),
        EVER_MOVED_T34: valid[:, spec.final_index] & ever_moved,
        "valid": valid,
        "ever_moved": ever_moved,
    }


def euclidean_error(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    smooth_epsilon_m: float = 0.0,
) -> torch.Tensor:
    """Pointwise Euclidean error, optionally smoothed while remaining zero at zero."""

    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have identical shapes")
    squared = (prediction.float() - target.float()).square().sum(dim=-1)
    epsilon = float(smooth_epsilon_m)
    if epsilon < 0:
        raise ValueError("smooth_epsilon_m must be non-negative")
    if epsilon == 0:
        return torch.sqrt(squared)
    return torch.sqrt(squared + epsilon * epsilon) - epsilon


def score_component_sums(
    prediction: torch.Tensor,
    target: torch.Tensor,
    batch: Mapping[str, Any],
    *,
    spec: ScoreMetricSpec | None = None,
    smooth_for_loss: bool = False,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Return count-weighted numerator/denominator pairs for all score components."""

    spec = spec or ScoreMetricSpec()
    validate_score_inputs(prediction, target, batch, spec)
    masks = score_masks(target, batch, spec)
    error = euclidean_error(
        prediction,
        target,
        smooth_epsilon_m=spec.smooth_epsilon_m if smooth_for_loss else 0.0,
    )
    components: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for key in (MOVED_T01_T34, MOVED_T01_T14):
        mask = masks[key]
        count = mask.sum()
        total = error[mask].sum() if bool(count > 0) else error.sum() * 0.0
        components[key] = (total, count)

    final_mask = masks[EVER_MOVED_T34]
    final_error = error[:, spec.final_index]
    final_count = final_mask.sum()
    final_total = (
        final_error[final_mask].sum()
        if bool(final_count > 0)
        else final_error.sum() * 0.0
    )
    components[EVER_MOVED_T34] = (final_total, final_count)
    return components


def component_means(
    components: Mapping[str, tuple[torch.Tensor, torch.Tensor]],
    *,
    empty_value: float = float("nan"),
) -> dict[str, torch.Tensor]:
    means: dict[str, torch.Tensor] = {}
    for key in COMPONENT_KEYS:
        if key not in components:
            raise KeyError(f"missing score component {key!r}")
        total, count = components[key]
        if bool(count > 0):
            means[key] = total / count.to(total.dtype)
        else:
            means[key] = total * 0.0 + float(empty_value)
    return means


def composite_from_means(
    means: Mapping[str, torch.Tensor | float],
    *,
    spec: ScoreMetricSpec | None = None,
) -> torch.Tensor:
    spec = spec or ScoreMetricSpec()
    result: torch.Tensor | None = None
    for key, weight in spec.component_weights.items():
        if key not in means:
            raise KeyError(f"missing score component mean {key!r}")
        value = means[key]
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(float(value), dtype=torch.float64)
        weighted = value * float(weight)
        result = weighted if result is None else result + weighted
    assert result is not None
    return result


def score_aligned_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    batch: Mapping[str, Any],
    *,
    spec: ScoreMetricSpec | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Differentiable metric-matched loss, independent of aleatoric variance."""

    spec = spec or ScoreMetricSpec()
    components = score_component_sums(
        prediction,
        target,
        batch,
        spec=spec,
        smooth_for_loss=True,
    )
    # DDP averages gradients across ranks.  Averaging independently normalized
    # per-rank means is biased whenever ranks contain different numbers of
    # moved points.  Scale each local numerator by world_size/global_count so
    # DDP's gradient average is exactly the global count-weighted derivative.
    distributed = dist.is_available() and dist.is_initialized()
    world_size = dist.get_world_size() if distributed else 1
    means: dict[str, torch.Tensor] = {}
    global_counts: dict[str, torch.Tensor] = {}
    for key in COMPONENT_KEYS:
        local_total, local_count = components[key]
        global_count = local_count.detach().clone()
        if distributed:
            dist.all_reduce(global_count, op=dist.ReduceOp.SUM)
        global_counts[key] = global_count
        if bool(global_count > 0):
            means[key] = local_total * float(world_size) / global_count.to(
                local_total.dtype
            )
        else:
            means[key] = local_total * 0.0
    composite = composite_from_means(means, spec=spec)
    diagnostics = dict(means)
    diagnostics[COMPOSITE_SCORE] = composite
    for key, count in global_counts.items():
        diagnostics[f"{key}/count"] = count
    return composite, diagnostics


def accumulate_component_sums(
    metrics: dict[str, torch.Tensor],
    counts: dict[str, torch.Tensor],
    components: Mapping[str, tuple[torch.Tensor, torch.Tensor]],
) -> None:
    """Add component numerators/counts to trainer-style distributed accumulators."""

    for key in COMPONENT_KEYS:
        total, count = components[key]
        metrics[key] += total.detach()
        counts[key] += count.detach()


def composite_from_log_dict(
    log_dict: Mapping[str, float],
    *,
    prefix: str,
    spec: ScoreMetricSpec | None = None,
) -> float:
    spec = spec or ScoreMetricSpec()
    means = {}
    for key in COMPONENT_KEYS:
        full_key = f"{prefix}/{key}" if prefix else key
        if full_key not in log_dict:
            raise KeyError(f"missing globally reduced score metric {full_key!r}")
        value = float(log_dict[full_key])
        if not math.isfinite(value):
            raise ValueError(f"score metric {full_key!r} is not finite: {value}")
        means[key] = value
    return float(composite_from_means(means, spec=spec).item())


def per_window_score_records(
    prediction: torch.Tensor,
    target: torch.Tensor,
    batch: Mapping[str, Any],
    *,
    spec: ScoreMetricSpec | None = None,
) -> list[dict[str, Any]]:
    """Return deterministic JSON-serializable numerator/count records per sample."""

    spec = spec or ScoreMetricSpec()
    batch_size, _, _ = validate_score_inputs(prediction, target, batch, spec)
    keys = batch.get("__key__")
    if keys is None:
        raise KeyError("per-window score export requires batch['__key__']")
    if isinstance(keys, torch.Tensor) or len(keys) != batch_size:
        raise ValueError("batch['__key__'] must be a sequence with one value per sample")
    datasets = batch.get("__dataset__", ["unknown"] * batch_size)
    if isinstance(datasets, torch.Tensor) or len(datasets) != batch_size:
        raise ValueError(
            "batch['__dataset__'] must be a sequence with one value per sample"
        )
    source_clips = batch.get("source_clip_id")
    if source_clips is not None and (
        isinstance(source_clips, torch.Tensor) or len(source_clips) != batch_size
    ):
        raise ValueError(
            "batch['source_clip_id'] must be a sequence with one value per sample"
        )
    metadata_values = batch.get("metadata_json")
    if metadata_values is not None and (
        isinstance(metadata_values, torch.Tensor) or len(metadata_values) != batch_size
    ):
        raise ValueError(
            "batch['metadata_json'] must be a sequence with one value per sample"
        )

    records: list[dict[str, Any]] = []
    for index in range(batch_size):
        sample_batch: dict[str, Any] = {}
        for name in (
            "scene_context_mask",
            "scene_exists",
            "scene_supervised_mask",
            "scene_moved_mask",
        ):
            sample_batch[name] = batch[name][index : index + 1]
        components = score_component_sums(
            prediction[index : index + 1],
            target[index : index + 1],
            sample_batch,
            spec=spec,
            smooth_for_loss=False,
        )
        explicit_source = (
            _normalized_source_clip(source_clips[index])
            if source_clips is not None
            else None
        )
        metadata_source = (
            _source_clip_from_metadata_json(metadata_values[index])
            if metadata_values is not None
            else None
        )
        if (
            explicit_source is not None
            and metadata_source is not None
            and explicit_source != metadata_source
        ):
            raise ValueError(
                "batch source_clip_id disagrees with metadata_json for "
                f"sample {keys[index]!r}: {explicit_source!r} vs {metadata_source!r}"
            )
        source_clip_id = explicit_source or metadata_source
        if source_clip_id is None:
            source_clip_id = _source_clip_from_audited_manifest(keys[index])
        if source_clip_id is None:
            raise ValueError(
                "per-window score export requires authoritative source_clip_id "
                "from the dataset, metadata_json, or the audited split manifest "
                "(SCORE_SE3_SPLIT_MANIFEST with a recorded filename-inference "
                "exception); raw filename heuristics are forbidden"
            )
        record: dict[str, Any] = {
            "sample_key": str(keys[index]),
            "dataset": str(datasets[index]),
            "source_clip_id": source_clip_id,
        }
        finite_means: dict[str, float] = {}
        for key, (total, count) in components.items():
            total_value = float(total.detach().cpu().item())
            count_value = int(count.detach().cpu().item())
            record[f"{key}/sum"] = total_value
            record[f"{key}/count"] = count_value
            record[key] = total_value / count_value if count_value > 0 else None
            if count_value > 0:
                finite_means[key] = total_value / count_value
        record[COMPOSITE_SCORE] = (
            float(composite_from_means(finite_means, spec=spec).item())
            if len(finite_means) == len(COMPONENT_KEYS)
            else None
        )
        records.append(record)
    return records


def _records_equal(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    atol: float,
) -> bool:
    if set(left) != set(right):
        return False
    for key in left:
        left_value = left[key]
        right_value = right[key]
        if isinstance(left_value, (int, float)) and isinstance(
            right_value, (int, float)
        ):
            if not math.isclose(
                float(left_value),
                float(right_value),
                rel_tol=1e-7,
                abs_tol=float(atol),
            ):
                return False
        elif left_value != right_value:
            return False
    return True


def deduplicate_score_records(
    records: list[dict[str, Any]],
    *,
    atol: float = 1e-8,
) -> list[dict[str, Any]]:
    """Deduplicate DistributedSampler padding and fail on conflicting repeats."""

    unique: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        if "sample_key" not in record or "dataset" not in record:
            raise KeyError("score record requires sample_key and dataset")
        identity = (str(record["dataset"]), str(record["sample_key"]))
        normalized = dict(record)
        if identity in unique:
            if not _records_equal(unique[identity], normalized, atol=atol):
                raise ValueError(
                    "conflicting score records for repeated eval sample "
                    f"dataset={identity[0]!r}, sample_key={identity[1]!r}"
                )
            continue
        unique[identity] = normalized
    return [unique[key] for key in sorted(unique)]


def aggregate_score_records(
    records: list[dict[str, Any]],
    *,
    spec: ScoreMetricSpec | None = None,
) -> dict[str, float | int]:
    """Recompute unique-window global metrics from exported numerator/count pairs."""

    spec = spec or ScoreMetricSpec()
    unique = deduplicate_score_records(records)
    result: dict[str, float | int] = {"score_se3/num_windows": len(unique)}
    means: dict[str, float] = {}
    for key in COMPONENT_KEYS:
        sum_key = f"{key}/sum"
        count_key = f"{key}/count"
        missing = [
            record["sample_key"]
            for record in unique
            if sum_key not in record or count_key not in record
        ]
        if missing:
            raise KeyError(
                f"score records missing {sum_key!r}/{count_key!r}: {missing[:5]}"
            )
        total = math.fsum(float(record[sum_key]) for record in unique)
        count = sum(int(record[count_key]) for record in unique)
        result[sum_key] = total
        result[count_key] = count
        mean = total / count if count > 0 else float("nan")
        result[key] = mean
        means[key] = mean
    if all(math.isfinite(value) for value in means.values()):
        result[COMPOSITE_SCORE] = float(
            composite_from_means(means, spec=spec).item()
        )
    else:
        result[COMPOSITE_SCORE] = float("nan")
    return result
