"""Restart-safe training-data cursor for the score-SE3 campaign.

The score-SE3 launcher uses a deterministic, finite, full-coverage HOI loader.
PyTorch does not serialize a ``DataLoader`` iterator, so restoring only model
and optimizer state would silently replay the beginning of the current epoch.
This module makes the *next* training batch a derived, audited part of campaign
state:

``sampler_epoch, batch_offset = divmod(campaign_step, len(train_loader))``.

The checkpoint also records the loader/sampler layout used to derive that
cursor.  A resume fails closed when the current loader has a different length,
world size, batch size, dataset size, sampler type, seed, or padding layout.
"""
from __future__ import annotations

import random
import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist


DATA_CURSOR_SCHEMA_VERSION = 1
RNG_STATE_SCHEMA_VERSION = 1


def score_se3_cadence_anchor(
    campaign_step: int,
    frequency: int,
    *,
    checkpoint_role: str,
    evaluate_initialization: bool,
) -> int:
    """Return the last completed cadence boundary for a fresh/resumed loop."""

    campaign_step = _require_int(campaign_step, "cadence.campaign_step")
    frequency = int(frequency)
    if frequency <= 0:
        return -1
    if (
        campaign_step == 0
        and checkpoint_role in ("init", "finetune")
        and evaluate_initialization
    ):
        return -frequency
    return (campaign_step // frequency) * frequency


def _qualified_name(value: Any) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _dataset_inventory_fingerprint(trainer: Any, dataset: Any) -> str:
    cached = getattr(trainer, "score_se3_dataset_inventory_fingerprint", None)
    if isinstance(cached, str) and len(cached) == 64:
        return cached
    override = getattr(dataset, "score_se3_inventory_records", None)
    if override is not None:
        records = list(override)
    else:
        samples = getattr(dataset, "samples", None)
        if not isinstance(samples, (list, tuple)) or not samples:
            raise RuntimeError(
                "score-SE3 dataset must expose its ordered sample inventory"
            )
        records = []
        for index, sample in enumerate(samples):
            lexical = Path(os.path.abspath(Path(sample).expanduser()))
            try:
                resolved = lexical.resolve(strict=True)
                stat = resolved.stat()
            except OSError as exc:
                raise RuntimeError(
                    f"score-SE3 dataset inventory entry is unavailable: {lexical}"
                ) from exc
            records.append(
                {
                    "index": index,
                    "lexical_path": str(lexical),
                    "resolved_path": str(resolved),
                    "size_bytes": int(stat.st_size),
                    "mtime_ns": int(stat.st_mtime_ns),
                }
            )
    canonical = json.dumps(
        records,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    digest = hashlib.sha256(canonical).hexdigest()
    trainer.score_se3_dataset_inventory_fingerprint = digest
    trainer.args.score_se3_dataset_inventory_fingerprint = digest
    return digest


def _require_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise RuntimeError(f"score-SE3 data cursor {name} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"score-SE3 data cursor {name} must be an integer"
        ) from exc
    if result != value or result < minimum:
        raise RuntimeError(
            f"score-SE3 data cursor {name} must be >= {minimum}, got {value!r}"
        )
    return result


def _score_training_enabled(trainer: Any) -> bool:
    return bool(getattr(trainer.args, "score_se3_eval", False)) and not bool(
        getattr(trainer, "inference_only", False)
    ) and not bool(getattr(trainer, "evaluation_only", False))


def _validate_runtime_loader(trainer: Any) -> tuple[Any, Any, int, dict[str, Any]]:
    if not _score_training_enabled(trainer):
        raise RuntimeError("score-SE3 training data cursor requested outside training")

    args = trainer.args
    requirements = {
        "deterministic_train": bool(getattr(args, "deterministic_train", False)),
        "full_train_coverage": bool(getattr(args, "full_train_coverage", False)),
        "eval_test_only": bool(getattr(args, "eval_test_only", False)),
    }
    disabled = sorted(name for name, enabled in requirements.items() if not enabled)
    if disabled:
        raise RuntimeError(
            "restart-safe score-SE3 training requires deterministic, full-coverage, "
            f"test-only evaluation data flow; disabled={disabled}"
        )

    loader = getattr(trainer, "train_dataloader", None)
    if loader is None:
        raise RuntimeError("score-SE3 training has no train_dataloader")
    loader_length = int(len(loader))
    if loader_length <= 0:
        raise RuntimeError("score-SE3 train_dataloader must contain at least one batch")
    if int(getattr(loader, "num_workers", 0)) != 0:
        raise RuntimeError(
            "restart-safe score-SE3 training requires train num_workers=0 so no "
            "prefetched samples exist outside the audited iterator cursor"
        )

    sampler = getattr(loader, "sampler", None)
    if sampler is None or not callable(getattr(sampler, "set_epoch", None)):
        raise RuntimeError(
            "restart-safe score-SE3 training requires an epoch-aware sampler"
        )
    plan = getattr(sampler, "plan", None)
    if not isinstance(plan, Mapping):
        raise RuntimeError(
            "restart-safe score-SE3 training requires the full-coverage sampler plan"
        )
    info = getattr(trainer, "train_dataloader_info", None)
    if not isinstance(info, Mapping) or not bool(info.get("full_coverage", False)):
        raise RuntimeError(
            "score-SE3 train loader is missing authoritative full-coverage metadata"
        )
    if _require_int(info.get("batches_per_rank"), "batches_per_rank", minimum=1) != loader_length:
        raise RuntimeError(
            "score-SE3 train loader length disagrees with full-coverage metadata: "
            f"len={loader_length}, metadata={info.get('batches_per_rank')!r}"
        )
    expected_epoch_samples = (
        loader_length
        * _require_int(getattr(args, "batch_size", None), "batch_size", minimum=1)
        * _require_int(getattr(trainer, "world_size", None), "world_size", minimum=1)
    )
    if _require_int(
        getattr(trainer, "train_total_samples", None),
        "train_total_samples",
        minimum=1,
    ) != expected_epoch_samples:
        raise RuntimeError(
            "score-SE3 padded epoch size is inconsistent with loader length, "
            "batch size, and world size"
        )
    if _require_int(plan.get("batches_per_rank"), "plan.batches_per_rank", minimum=1) != loader_length:
        raise RuntimeError("full-coverage sampler plan disagrees with loader length")
    dataset = getattr(loader, "dataset", None)
    if dataset is None:
        raise RuntimeError("score-SE3 train loader has no dataset")
    dataset_size = _require_int(len(dataset), "dataset_size", minimum=1)
    unique_samples = _require_int(
        info.get("unique_samples"), "unique_samples", minimum=1
    )
    if dataset_size != unique_samples:
        raise RuntimeError(
            "score-SE3 dataset size disagrees with full-coverage metadata: "
            f"dataset={dataset_size}, metadata={unique_samples}"
        )
    batch_size = _require_int(
        getattr(args, "batch_size", None), "batch_size", minimum=1
    )
    if _require_int(len(sampler), "sampler_samples_per_rank", minimum=1) != (
        loader_length * batch_size
    ):
        raise RuntimeError(
            "score-SE3 sampler length does not equal full batches per rank"
        )
    if _require_int(
        plan.get("sampler_total_samples"),
        "plan.sampler_total_samples",
        minimum=1,
    ) != expected_epoch_samples:
        raise RuntimeError(
            "score-SE3 sampler plan global size disagrees with padded epoch size"
        )
    if _require_int(
        plan.get("padded_duplicate_samples"),
        "plan.padded_duplicate_samples",
    ) != _require_int(
        info.get("padded_duplicate_samples"),
        "padded_duplicate_samples",
    ):
        raise RuntimeError(
            "score-SE3 sampler padding disagrees with full-coverage metadata"
        )
    if bool(getattr(sampler, "shuffle", False)):
        raise RuntimeError(
            "deterministic score-SE3 training requires sampler shuffle=False"
        )
    if not bool(getattr(loader, "drop_last", False)):
        raise RuntimeError(
            "score-SE3 full-coverage loader must enforce full per-rank batches"
        )
    return loader, sampler, loader_length, dict(info)


def score_se3_data_cursor_metadata(
    trainer: Any,
    *,
    require_committed: bool = True,
) -> dict[str, Any]:
    """Return immutable checkpoint metadata for the next training batch."""

    loader, sampler, loader_length, info = _validate_runtime_loader(trainer)
    campaign_step = _require_int(
        getattr(trainer, "score_se3_campaign_step", None),
        "campaign_step",
    )
    committed = _require_int(
        getattr(trainer, "score_se3_data_batches_committed", campaign_step),
        "committed_batches",
    )
    if require_committed and bool(
        getattr(trainer, "score_se3_data_batch_pending", False)
    ):
        raise RuntimeError(
            "score-SE3 cannot checkpoint with a fetched but uncommitted training "
            "batch; use the preceding cadence checkpoint"
        )
    if require_committed and committed != campaign_step:
        raise RuntimeError(
            "score-SE3 data cursor diverged from campaign_step: "
            f"committed_batches={committed}, campaign_step={campaign_step}"
        )
    epoch, offset = divmod(campaign_step, loader_length)
    plan = dict(getattr(sampler, "plan"))
    dataset = getattr(loader, "dataset", None)
    dataset_size = len(dataset) if dataset is not None else None
    if dataset_size is None:
        raise RuntimeError("score-SE3 train loader dataset has no finite length")
    sampler_size = len(sampler)
    batch_size = _require_int(
        getattr(trainer.args, "batch_size", None), "batch_size", minimum=1
    )
    world_size = _require_int(
        getattr(trainer, "world_size", None), "world_size", minimum=1
    )
    sample_count = _require_int(
        getattr(trainer, "sample_count", None), "sample_count"
    )
    batch_count = _require_int(
        getattr(trainer, "batch_count", None), "batch_count"
    )
    expected_sample_count = campaign_step * batch_size * world_size
    expected_batch_count = campaign_step * world_size
    if sample_count != expected_sample_count or batch_count != expected_batch_count:
        raise RuntimeError(
            "score-SE3 checkpoint counters do not describe one consumed global "
            "batch per campaign step: "
            f"campaign_step={campaign_step}, sample_count={sample_count} "
            f"(expected {expected_sample_count}), batch_count={batch_count} "
            f"(expected {expected_batch_count})"
        )
    return {
        "schema_version": DATA_CURSOR_SCHEMA_VERSION,
        "cursor_semantics": "next_training_batch",
        "campaign_step": campaign_step,
        "sampler_epoch": epoch,
        "batch_offset_within_epoch": offset,
        "batches_per_rank": loader_length,
        "per_rank_batch_size": batch_size,
        "world_size": world_size,
        "global_batch_size": batch_size * world_size,
        "sample_count": sample_count,
        "batch_count": batch_count,
        "padded_epoch_samples": _require_int(
            getattr(trainer, "train_total_samples", None),
            "train_total_samples",
            minimum=1,
        ),
        "unique_samples": _require_int(
            info.get("unique_samples"), "unique_samples", minimum=1
        ),
        "padded_duplicate_samples": _require_int(
            info.get("padded_duplicate_samples"),
            "padded_duplicate_samples",
        ),
        "sampler_samples_per_rank": _require_int(
            sampler_size, "sampler_samples_per_rank", minimum=1
        ),
        "sampler_class": _qualified_name(sampler),
        "sampler_seed": _require_int(getattr(sampler, "seed", None), "sampler_seed"),
        "sampler_shuffle": bool(getattr(sampler, "shuffle", False)),
        "loader_drop_last": bool(getattr(loader, "drop_last", False)),
        "loader_num_workers": int(getattr(loader, "num_workers", -1)),
        "dataset_size": _require_int(dataset_size, "dataset_size", minimum=1),
        "dataset_inventory_sha256": _dataset_inventory_fingerprint(
            trainer, dataset
        ),
        "sampler_total_samples": _require_int(
            plan.get("sampler_total_samples"),
            "plan.sampler_total_samples",
            minimum=1,
        ),
    }


def _validate_saved_cursor(
    saved: Any,
    expected: Mapping[str, Any],
) -> None:
    if not isinstance(saved, Mapping):
        raise RuntimeError(
            "score-SE3 resume checkpoint is missing score_se3_data_cursor metadata"
        )
    required = set(expected)
    missing = sorted(required - set(saved))
    extra = sorted(set(saved) - required)
    if missing or extra:
        raise RuntimeError(
            "score-SE3 data cursor schema mismatch: "
            f"missing={missing}, extra={extra}"
        )
    differences = {
        key: {"checkpoint": saved[key], "current": expected[key]}
        for key in sorted(required)
        if saved[key] != expected[key]
    }
    if differences:
        raise RuntimeError(
            "score-SE3 training-data layout/cursor changed on resume: "
            f"{differences}"
        )


def restore_score_se3_train_iterator(
    trainer: Any,
    checkpoint: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Restore the sampler epoch and exact within-epoch batch position.

    Initialization checkpoints are warm starts and intentionally begin at
    campaign step zero.  Exact resume checkpoints must carry matching cursor
    metadata.  Held-out/evaluation-only processes do not own a training cursor.
    """

    if not _score_training_enabled(trainer):
        return None
    expected = score_se3_data_cursor_metadata(trainer)
    role = str(getattr(trainer.args, "score_se3_checkpoint_role", "init"))
    if role == "resume":
        if checkpoint is None:
            raise RuntimeError("score-SE3 resume requires a checkpoint")
        _validate_saved_cursor(checkpoint.get("score_se3_data_cursor"), expected)
    elif role in ("init", "finetune"):
        if expected["campaign_step"] != 0:
            raise RuntimeError("score-SE3 init data cursor must start at campaign step zero")
    else:
        raise RuntimeError(
            f"training data cursor cannot be restored for checkpoint role {role!r}"
        )

    loader = trainer.train_dataloader
    sampler = loader.sampler
    epoch = int(expected["sampler_epoch"])
    offset = int(expected["batch_offset_within_epoch"])
    sampler.set_epoch(epoch)
    iterator = iter(loader)
    for skipped in range(offset):
        try:
            next(iterator)
        except StopIteration as exc:
            raise RuntimeError(
                "score-SE3 train loader exhausted while restoring within-epoch "
                f"offset {offset}; stopped after {skipped} batches"
            ) from exc
    trainer.train_iter = iterator
    trainer.score_se3_data_batches_committed = int(expected["campaign_step"])
    trainer.score_se3_data_batch_pending = False
    trainer.score_se3_data_cursor = dict(expected)
    return dict(expected)


def ensure_score_se3_train_batch_fetch_allowed(trainer: Any) -> None:
    """Fail before advancing the iterator when the prior batch was not committed."""

    if not _score_training_enabled(trainer):
        return
    if bool(getattr(trainer, "score_se3_data_batch_pending", False)):
        raise RuntimeError(
            "previous score-SE3 training batch was fetched but never committed; "
            "refusing to consume another batch and corrupt the resume cursor"
        )
    campaign_step = int(getattr(trainer, "score_se3_campaign_step", -1))
    committed = int(getattr(trainer, "score_se3_data_batches_committed", -1))
    if committed != campaign_step:
        raise RuntimeError(
            "score-SE3 data cursor is inconsistent before batch fetch: "
            f"committed={committed}, campaign_step={campaign_step}"
        )


def mark_score_se3_train_batch_fetched(trainer: Any) -> None:
    """Mark one fetched batch, before evaluation/forward/backward.

    Cadence checkpointing is ordered before this function.  Checkpoint creation
    rejects pending batches because fetch/forward may already have consumed RNG
    that cannot be paired with replaying the same data cursor.
    """

    if not _score_training_enabled(trainer):
        return
    ensure_score_se3_train_batch_fetch_allowed(trainer)
    trainer.score_se3_data_batch_pending = True


def commit_score_se3_train_batch(trainer: Any) -> None:
    """Commit the fetched batch after campaign_step advances exactly once."""

    if not _score_training_enabled(trainer):
        return
    if not bool(getattr(trainer, "score_se3_data_batch_pending", False)):
        raise RuntimeError("score-SE3 campaign advanced without a fetched training batch")
    committed = int(getattr(trainer, "score_se3_data_batches_committed", -1))
    campaign_step = int(getattr(trainer, "score_se3_campaign_step", -1))
    if campaign_step != committed + 1:
        raise RuntimeError(
            "each score-SE3 training batch must advance campaign_step exactly once: "
            f"before={committed}, after={campaign_step}"
        )
    trainer.score_se3_data_batches_committed = campaign_step
    trainer.score_se3_data_batch_pending = False
    # Recompute now so any dynamic loader mutation fails at the first step,
    # rather than much later when a checkpoint is written.
    trainer.score_se3_data_cursor = score_se3_data_cursor_metadata(trainer)


def _local_score_se3_rng_state(trainer: Any) -> dict[str, Any]:
    rank = _require_int(getattr(trainer, "rank", 0), "rng.rank")
    world_size = _require_int(
        getattr(trainer, "world_size", 1), "rng.world_size", minimum=1
    )
    device = torch.device(getattr(trainer, "device", "cpu"))
    cuda_state = None
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "score-SE3 trainer uses a CUDA device but CUDA is unavailable while "
                "capturing RNG state"
            )
        cuda_state = torch.cuda.get_rng_state(device).cpu()
    return {
        "rank": rank,
        "world_size": world_size,
        "device_type": device.type,
        "device_index": device.index,
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_cpu_rng_state": torch.get_rng_state().cpu(),
        "torch_cuda_rng_state": cuda_state,
    }


def _validate_rng_bundle(bundle: Any, trainer: Any) -> dict[str, Any]:
    if not isinstance(bundle, Mapping):
        raise RuntimeError(
            "score-SE3 resume checkpoint is missing score_se3_rng_state"
        )
    if bundle.get("schema_version") != RNG_STATE_SCHEMA_VERSION:
        raise RuntimeError(
            "unsupported score-SE3 RNG-state schema: "
            f"{bundle.get('schema_version')!r}"
        )
    world_size = _require_int(
        getattr(trainer, "world_size", 1), "rng.world_size", minimum=1
    )
    rank = _require_int(getattr(trainer, "rank", 0), "rng.rank")
    if _require_int(bundle.get("world_size"), "rng.saved_world_size", minimum=1) != world_size:
        raise RuntimeError(
            "score-SE3 RNG topology changed on resume: "
            f"checkpoint_world_size={bundle.get('world_size')!r}, "
            f"current_world_size={world_size}"
        )
    states = bundle.get("rank_states")
    if not isinstance(states, (list, tuple)) or len(states) != world_size:
        raise RuntimeError(
            "score-SE3 RNG checkpoint must contain exactly one state per rank"
        )
    observed_ranks: list[int] = []
    for index, state in enumerate(states):
        if not isinstance(state, Mapping):
            raise RuntimeError(f"score-SE3 RNG rank state {index} is malformed")
        observed_ranks.append(
            _require_int(state.get("rank"), f"rng.rank_states[{index}].rank")
        )
        if _require_int(
            state.get("world_size"),
            f"rng.rank_states[{index}].world_size",
            minimum=1,
        ) != world_size:
            raise RuntimeError(
                f"score-SE3 RNG rank state {index} has inconsistent world_size"
            )
    if observed_ranks != list(range(world_size)):
        raise RuntimeError(
            "score-SE3 RNG states must be ordered one-per-rank: "
            f"observed={observed_ranks}, expected={list(range(world_size))}"
        )
    return dict(states[rank])


def collect_score_se3_rng_states(trainer: Any) -> dict[str, Any] | None:
    """Collect one restart RNG snapshot per rank before rank zero writes.

    This collective must be called by every rank at the same checkpoint
    boundary.  It intentionally captures only the current CUDA device on each
    rank, matching DDP's one-process/one-device execution model.
    """

    if not _score_training_enabled(trainer) or not bool(
        getattr(trainer.args, "full_train_coverage", False)
    ):
        return None
    local = _local_score_se3_rng_state(trainer)
    world_size = int(getattr(trainer, "world_size", 1))
    distributed = bool(getattr(trainer.args, "distributed", False))
    if distributed:
        if world_size <= 1:
            raise RuntimeError(
                "score-SE3 RNG capture has distributed=true but world_size <= 1"
            )
        gathered: list[Any] = [None] * world_size
        dist.all_gather_object(
            gathered,
            local,
            group=getattr(trainer, "cpu_pg", None),
        )
    else:
        if world_size != 1:
            raise RuntimeError(
                "score-SE3 RNG capture has distributed=false but world_size != 1"
            )
        gathered = [local]
    bundle = {
        "schema_version": RNG_STATE_SCHEMA_VERSION,
        "world_size": world_size,
        "rank_states": gathered,
    }
    _validate_rng_bundle(bundle, trainer)
    return bundle


def restore_score_se3_rng_state(
    trainer: Any,
    checkpoint: Mapping[str, Any] | None,
) -> None:
    """Restore this rank's RNG streams after iterator reconstruction.

    ``iter(DataLoader)`` may consume CPU Torch RNG even with no worker
    processes.  The Trainer therefore calls this only after setting the sampler
    epoch and skipping the saved within-epoch batch offset.
    """

    if not _score_training_enabled(trainer):
        return
    role = str(getattr(trainer.args, "score_se3_checkpoint_role", "init"))
    if role in ("init", "finetune"):
        seed = _require_int(getattr(trainer.args, "seed", None), "seed") + _require_int(
            getattr(trainer, "rank", 0), "rank"
        )
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        device = torch.device(getattr(trainer, "device", "cpu"))
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "cannot seed score-SE3 CUDA RNG because CUDA is unavailable"
                )
            with torch.cuda.device(device):
                torch.cuda.manual_seed(seed)
        return
    if role != "resume":
        raise RuntimeError(f"cannot restore training RNG for checkpoint role {role!r}")
    if checkpoint is None:
        raise RuntimeError("score-SE3 resume requires a checkpoint RNG state")
    state = _validate_rng_bundle(checkpoint.get("score_se3_rng_state"), trainer)
    required = {
        "python_random_state",
        "numpy_random_state",
        "torch_cpu_rng_state",
        "torch_cuda_rng_state",
        "device_type",
        "device_index",
    }
    missing = sorted(required - set(state))
    if missing:
        raise RuntimeError(f"score-SE3 RNG rank state is missing fields: {missing}")

    device = torch.device(getattr(trainer, "device", "cpu"))
    if state["device_type"] != device.type or state["device_index"] != device.index:
        raise RuntimeError(
            "score-SE3 RNG device topology changed on resume: "
            f"checkpoint={state['device_type']}:{state['device_index']}, "
            f"current={device.type}:{device.index}"
        )
    cpu_state = state["torch_cpu_rng_state"]
    if not torch.is_tensor(cpu_state) or cpu_state.dtype != torch.uint8:
        raise RuntimeError("score-SE3 checkpoint has invalid Torch CPU RNG state")
    cuda_state = state["torch_cuda_rng_state"]
    if device.type == "cuda":
        if not torch.is_tensor(cuda_state) or cuda_state.dtype != torch.uint8:
            raise RuntimeError("score-SE3 checkpoint has invalid CUDA RNG state")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA RNG state cannot be restored because CUDA is unavailable")
    elif cuda_state is not None:
        raise RuntimeError("CPU score-SE3 checkpoint unexpectedly contains CUDA RNG state")

    try:
        random.setstate(state["python_random_state"])
        np.random.set_state(state["numpy_random_state"])
        torch.set_rng_state(cpu_state.cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state(cuda_state.cpu(), device=device)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("score-SE3 checkpoint contains malformed RNG state") from exc
