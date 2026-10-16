from __future__ import annotations

import copy
import json
import math
import os
import re
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from training.horizon_eval_metrics import time_range_mask
from training.score_se3_metrics import (
    COMPOSITE_SCORE,
    COMPONENT_KEYS,
    ScoreMetricSpec,
    accumulate_component_sums,
    aggregate_score_records,
    composite_from_log_dict,
    deduplicate_score_records,
    per_window_score_records,
    score_component_sums,
)


def score_se3_dataset_seed(
    base_seed: int,
    *,
    mode: str,
    rank: int,
    enabled: bool,
    eval_data_seed: int | None = None,
) -> int:
    """Return rank-invariant eval seed and rank-specific training seed."""

    if enabled and mode == "test":
        seed = int(base_seed) if eval_data_seed is None else int(eval_data_seed)
        if seed < 0:
            raise ValueError("score-SE3 eval data seed must be non-negative")
        return seed
    return int(base_seed) + int(rank)


def install_rank_invariant_hoi_eval_seed() -> None:
    """Patch score-SE3 HOI eval so a repeated sample has identical point sampling."""

    import dataset_components.dataloader as dataloader_module

    original = dataloader_module.build_hoi_dataloader

    def build_hoi_dataloader(args, mode: str, rank: int = 0, world_size: int = 1):
        enabled = bool(getattr(args, "score_se3_eval", False))
        if (
            enabled
            and mode == "test"
            and getattr(args, "dataset_type", None) == "hoi_npz"
        ):
            # The repository loader internally constructs config.seed as
            # ``args.seed + rank``. Offset a shallow copy so every rank resolves
            # to the same base seed without mutating the run contract or train path.
            adjusted = copy.copy(args)
            desired = score_se3_dataset_seed(
                int(args.seed),
                mode=mode,
                rank=rank,
                enabled=True,
                eval_data_seed=int(
                    getattr(args, "score_se3_eval_data_seed", 2026)
                ),
            )
            adjusted.seed = desired - int(rank)
            return original(
                adjusted,
                mode=mode,
                rank=rank,
                world_size=world_size,
            )
        return original(args, mode=mode, rank=rank, world_size=world_size)

    dataloader_module.build_hoi_dataloader = build_hoi_dataloader


def score_metric_spec_from_args(args: Any) -> ScoreMetricSpec:
    return ScoreMetricSpec(
        ever_moved_threshold_m=float(
            getattr(args, "ever_moved_threshold", 0.005)
        )
    )


def _add(
    metrics: dict[str, torch.Tensor],
    counts: dict[str, torch.Tensor],
    key: str,
    values: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    count = mask.sum()
    if bool(count > 0):
        metrics[key] += values[mask].sum()
        counts[key] += count


def _safe_token(value: Any) -> str:
    token = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(value).strip())
    return token.strip("_") or "unknown"


def _atomic_write_text(path: Path, text: str, *, immutable: bool = False) -> None:
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if immutable:
            if existing != text:
                raise RuntimeError(
                    "score-SE3 evaluation artifact already exists with different "
                    f"content: {path}"
                )
            # Retrying a completed cadence is idempotent, but a run may never
            # replace the scientific record for an existing candidate/step.
            return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_score_record_bundle(
    output_dir: str | Path,
    *,
    prefix: str,
    optimizer_step: int,
    campaign_step: int | None = None,
    records: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> tuple[Path, Path]:
    unique = deduplicate_score_records(records)
    directory = Path(output_dir).expanduser()
    if campaign_step is None:
        campaign_step = int(optimizer_step)
    stem = (
        f"{_safe_token(prefix)}__campaign-step={int(campaign_step):08d}"
        f"__optimizer-step={int(optimizer_step):08d}"
        f"__candidate={_safe_token(metadata.get('candidate', 'unknown'))}"
    )
    records_path = directory / f"{stem}.jsonl"
    metadata_path = directory / f"{stem}.meta.json"
    lines = "".join(
        json.dumps(record, sort_keys=True, allow_nan=False) + "\n"
        for record in unique
    )
    payload = dict(metadata)
    payload["records_file"] = records_path.name
    payload["num_unique_windows"] = len(unique)
    payload["num_unique_source_clips"] = len(
        {
            (str(record["dataset"]), str(record["source_clip_id"]))
            for record in unique
        }
    )
    _atomic_write_text(records_path, lines, immutable=True)
    _atomic_write_text(
        metadata_path,
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        immutable=True,
    )
    return records_path, metadata_path


def write_pending_score_record_bundle(
    trainer,
    *,
    checkpoint_path: str | Path,
    checkpoint_sha256: str,
) -> tuple[Path, Path] | None:
    """Bind a completed evaluation to the exact checkpoint it measured."""

    if trainer.rank != 0 or not bool(
        getattr(trainer.args, "score_se3_export_per_window", False)
    ):
        return None
    pending = getattr(trainer, "_score_se3_pending_bundle", None)
    if not isinstance(pending, dict):
        raise RuntimeError(
            "score-SE3 bundle export requested without a completed test evaluation"
        )
    digest = str(checkpoint_sha256).lower()
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("checkpoint_sha256 must be 64 lowercase hex characters")
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"score-SE3 evaluated checkpoint is missing: {path}")
    from training.score_se3_contract import sha256_file

    observed_digest = sha256_file(path)
    if observed_digest != digest:
        raise RuntimeError(
            "score-SE3 evaluated checkpoint hash changed: "
            f"declared={digest}, observed={observed_digest}, path={path}"
        )

    spec = score_metric_spec_from_args(trainer.args)
    candidate = str(
        getattr(trainer.args, "deformation_graph_candidate", None)
        or getattr(trainer.args, "score_se3_candidate", "unknown")
    )
    campaign_step = int(getattr(trainer, "score_se3_campaign_step", 0))
    optimizer_step = int(getattr(trainer, "optimizer_step", 0))
    metadata = {
        "schema_version": 2,
        "candidate": candidate,
        "oracle": candidate == "oracle_contact",
        "prefix": str(pending["prefix"]),
        "campaign_step": campaign_step,
        "optimizer_step": optimizer_step,
        "split_role": str(
            getattr(trainer.args, "score_se3_split_role", "validation")
        ),
        "checkpoint_path": str(path),
        "checkpoint_sha256": digest,
        "epoch_count": float(trainer.epoch_count),
        "seed": int(getattr(trainer.args, "seed", 0)),
        "eval_data_seed": int(
            getattr(trainer.args, "score_se3_eval_data_seed", 2026)
        ),
        "num_raw_records": int(pending["num_raw_records"]),
        "expected_windows": pending["expected_windows"],
        "full_eval_required": bool(
            getattr(trainer.args, "score_se3_require_full_eval", False)
        ),
        "metric_spec": {
            "ever_moved_threshold_m": spec.ever_moved_threshold_m,
            "smooth_epsilon_m": spec.smooth_epsilon_m,
            "early_end_index": spec.early_end_index,
            "final_index": spec.final_index,
            "component_weights": spec.component_weights,
        },
        "aggregate": pending["aggregate"],
        "checkpoint_metric": str(
            getattr(trainer.args, "checkpoint_metric", "")
        ),
    }
    result = write_score_record_bundle(
        Path(trainer.save_dir) / "score_se3_eval",
        prefix=str(pending["prefix"]),
        campaign_step=campaign_step,
        optimizer_step=optimizer_step,
        records=list(pending["records"]),
        metadata=metadata,
    )
    del trainer._score_se3_pending_bundle
    return result


def finalize_score_records(
    log_dict: dict[str, float],
    *,
    prefix: str,
    records: list[dict[str, Any]],
    spec: ScoreMetricSpec,
    expected_windows: int | None,
    require_full_eval: bool,
) -> tuple[dict[str, float], list[dict[str, Any]], dict[str, float | int]]:
    unique = deduplicate_score_records(records)
    aggregate = aggregate_score_records(unique, spec=spec)
    if (
        require_full_eval
        and expected_windows is not None
        and len(unique) != expected_windows
    ):
        raise RuntimeError(
            "score-SE3 evaluation did not cover the complete unique validation set: "
            f"observed={len(unique)}, expected={expected_windows}"
        )
    for key in COMPONENT_KEYS:
        value = float(aggregate[key])
        if not math.isfinite(value):
            raise RuntimeError(f"score-SE3 aggregate {key!r} is not finite")
        log_dict[f"{prefix}/{key}"] = value
    composite = float(aggregate[COMPOSITE_SCORE])
    if not math.isfinite(composite):
        raise RuntimeError("score-SE3 composite is not finite")
    log_dict[f"{prefix}/{COMPOSITE_SCORE}"] = composite
    log_dict[f"{prefix}/score_se3/num_windows"] = float(len(unique))
    return log_dict, unique, aggregate


def install_horizon_and_score_eval(trainer_module) -> None:
    """Install exact horizon metrics plus optional unique-window score evaluation."""

    trainer_cls = trainer_module.Trainer
    original_accumulate = trainer_cls._accumulate_hoi_dataset_eval_metrics

    def accumulate(self, metrics, counts, batch, outputs):
        original_accumulate(self, metrics, counts, batch, outputs)
        if getattr(self.args, "dataset_type", None) != "hoi_npz":
            return

        prediction = outputs["scene_flows"]
        target = batch["gt_scene_flows"].to(prediction.device)
        error = torch.linalg.norm(
            prediction.float() - target.float(),
            dim=-1,
        )
        context = (
            batch["scene_context_mask"].to(prediction.device).squeeze(-1).bool()
        )
        exists = batch["scene_exists"].to(prediction.device).squeeze(-1).bool()
        supervised = (
            batch["scene_supervised_mask"]
            .to(prediction.device)
            .squeeze(-1)
            .bool()
        )
        valid = (~context) & exists & supervised
        threshold = float(getattr(self.args, "ever_moved_threshold", 0.005))
        displacement = torch.linalg.norm(
            target.float() - target[:, :1].float(),
            dim=-1,
        )
        ever_moved = displacement.max(dim=1).values > threshold
        ever_moved_valid = valid & ever_moved[:, None, :]

        _add(
            metrics,
            counts,
            "horizon/ade_ever_moved",
            error,
            ever_moved_valid,
        )
        _add(
            metrics,
            counts,
            "horizon/fde_ever_moved",
            error[:, -1],
            valid[:, -1] & ever_moved,
        )

        moved = (
            batch["scene_moved_mask"].to(prediction.device).squeeze(-1).bool()
        )
        last_time = int(error.shape[1]) - 1
        score_enabled = bool(getattr(self.args, "score_se3_eval", False))
        spec = score_metric_spec_from_args(self.args) if score_enabled else None
        if score_enabled:
            if int(prediction.shape[1]) != spec.final_index + 1:
                raise ValueError(
                    "score-SE3 evaluation requires exactly "
                    f"{spec.final_index + 1} timesteps, got {prediction.shape[1]}"
                )
            components = score_component_sums(
                prediction,
                target,
                batch,
                spec=spec,
                smooth_for_loss=False,
            )
            accumulate_component_sums(metrics, counts, components)
            current_prefix = getattr(self, "_score_se3_eval_prefix", None)
            current_records = getattr(self, "_score_se3_eval_records", None)
            if current_prefix == "test" and isinstance(current_records, list):
                campaign_step = int(
                    getattr(self, "score_se3_campaign_step", 0)
                )
                optimizer_step = int(getattr(self, "optimizer_step", 0))
                candidate = str(
                    getattr(self.args, "deformation_graph_candidate", None)
                    or getattr(self.args, "score_se3_candidate", "unknown")
                )
                records = per_window_score_records(
                    prediction,
                    target,
                    batch,
                    spec=spec,
                )
                for record in records:
                    record.update(
                        {
                            "candidate": candidate,
                            "campaign_step": campaign_step,
                            "optimizer_step": optimizer_step,
                            "seed": int(getattr(self.args, "seed", 0)),
                        }
                    )
                current_records.extend(records)
        else:
            _add(
                metrics,
                counts,
                "l2_moved/mean_t01_t14",
                error,
                valid & moved & time_range_mask(valid, 1, 14),
            )
            _add(
                metrics,
                counts,
                "l2_moved/mean_t01_t34",
                error,
                valid & moved & time_range_mask(valid, 1, last_time),
            )

        _add(
            metrics,
            counts,
            "horizon/ade_ever_moved_t01_t14",
            error,
            ever_moved_valid & time_range_mask(valid, 1, 14),
        )
        if last_time >= 14:
            _add(
                metrics,
                counts,
                "horizon/fde_ever_moved_t14",
                error[:, 14],
                valid[:, 14] & ever_moved,
            )
        for time_index in range(1, prediction.shape[1]):
            _add(
                metrics,
                counts,
                f"horizon_exact/l2_all_t{time_index:02d}",
                error[:, time_index],
                valid[:, time_index],
            )
            if not (
                score_enabled
                and spec is not None
                and time_index == spec.final_index
            ):
                _add(
                    metrics,
                    counts,
                    f"horizon_exact/l2_ever_moved_t{time_index:02d}",
                    error[:, time_index],
                    valid[:, time_index] & ever_moved,
                )

    trainer_cls._accumulate_hoi_dataset_eval_metrics = accumulate

    original_eval_step = trainer_cls.eval_step

    def eval_step(self, dataloader, data_iter, prefix, log_dict=None):
        score_enabled = bool(getattr(self.args, "score_se3_eval", False))
        if not score_enabled:
            return original_eval_step(
                self,
                dataloader,
                data_iter,
                prefix,
                log_dict=log_dict,
            )

        if hasattr(self, "_score_se3_eval_records"):
            raise RuntimeError("nested score-SE3 evaluation is not supported")
        self._score_se3_eval_prefix = str(prefix)
        self._score_se3_eval_records: list[dict[str, Any]] = []
        try:
            result, data_iter = original_eval_step(
                self,
                dataloader,
                data_iter,
                prefix,
                log_dict=log_dict,
            )
            records = self._score_se3_eval_records
        finally:
            del self._score_se3_eval_records
            del self._score_se3_eval_prefix

        spec = score_metric_spec_from_args(self.args)
        if prefix != "test":
            result[f"{prefix}/{COMPOSITE_SCORE}"] = composite_from_log_dict(
                result,
                prefix=prefix,
                spec=spec,
            )
            return result, data_iter

        if self.args.distributed:
            gathered: list[list[dict[str, Any]] | None] = [None] * self.world_size
            dist.all_gather_object(gathered, records, group=self.cpu_pg)
            records = [
                record
                for rank_records in gathered
                for record in (rank_records or [])
            ]

        expected_windows = None
        dataset = getattr(dataloader, "dataset", None)
        if dataset is not None:
            try:
                expected_windows = len(dataset)
            except TypeError:
                expected_windows = None
        result, unique, aggregate = finalize_score_records(
            result,
            prefix=prefix,
            records=records,
            spec=spec,
            expected_windows=expected_windows,
            require_full_eval=bool(
                getattr(self.args, "score_se3_require_full_eval", False)
            ),
        )

        if bool(getattr(self.args, "score_se3_export_per_window", False)):
            if hasattr(self, "_score_se3_pending_bundle"):
                raise RuntimeError(
                    "previous score-SE3 test bundle was not bound to a checkpoint"
                )
            self._score_se3_pending_bundle = {
                "prefix": str(prefix),
                "records": unique,
                "aggregate": aggregate,
                "num_raw_records": len(records),
                "expected_windows": expected_windows,
            }
        return result, data_iter

    trainer_cls.eval_step = eval_step
