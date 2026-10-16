# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import math
import shutil
import torch
import torch.distributed as dist
from utils import _print
from pointworld.checkpoint_contract import attach_checkpoint_contract
from training.score_se3_contract import (
    score_se3_optimizer_runtime_contract,
    sha256_file,
)


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _checkpoint_metric_value(trainer, log_dict):
    metric_name = str(getattr(trainer.args, "checkpoint_metric", "test/l2_moved/mean"))
    if not log_dict or metric_name not in log_dict:
        return None
    value = log_dict[metric_name]
    if torch.is_tensor(value):
        if value.numel() != 1:
            return None
        value = float(value.detach().cpu().item())
    elif isinstance(value, (int, float)):
        value = float(value)
    else:
        return None
    return value if math.isfinite(value) else None


def _metric_improved(trainer, current_value: float) -> bool:
    best_value = getattr(trainer, "best_checkpoint_metric", None)
    if best_value is None:
        return True
    mode = str(getattr(trainer.args, "checkpoint_metric_mode", "min")).lower()
    if mode == "max":
        return current_value > float(best_value)
    return current_value < float(best_value)


def _safe_filename_token(value) -> str:
    token = str(value).replace("/", "_").replace("\\", "_").replace(" ", "")
    return "".join(ch if ch.isalnum() or ch in "._=-+" else "_" for ch in token)


def _checkpoint_filename(trainer, log_dict) -> str:
    epoch = float(getattr(trainer, "epoch_count", 0.0))
    batch = int(getattr(trainer, "batch_count", 0))
    sample = int(getattr(trainer, "sample_count", 0))
    pieces = [
        "model",
        f"epoch={epoch:09.4f}",
        f"batch={batch:012d}",
        f"samples={sample:012d}",
    ]
    metric_name = str(getattr(trainer.args, "checkpoint_metric", "test/l2_moved/mean"))
    metric_value = _checkpoint_metric_value(trainer, log_dict)
    if metric_value is not None:
        pieces.append(f"{_safe_filename_token(metric_name)}={metric_value:.6g}")
    return "__".join(pieces) + ".pt"


def _link_or_copy(src: str, dst: str) -> None:
    tmp_dst = f"{dst}.tmp"
    if os.path.exists(tmp_dst):
        os.remove(tmp_dst)
    relative_src = os.path.relpath(src, start=os.path.dirname(dst))
    try:
        os.symlink(relative_src, tmp_dst)
    except OSError:
        try:
            os.link(src, tmp_dst)
        except OSError:
            shutil.copy2(src, tmp_dst)
    os.replace(tmp_dst, dst)


def _atomic_torch_save(payload, path: str) -> None:
    """Write a checkpoint without exposing a partially serialized target."""

    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    temporary = os.path.join(
        directory,
        f".{os.path.basename(path)}.tmp-{os.getpid()}",
    )
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def _remove_previous_best_checkpoint(trainer, checkpoint_dir: str, new_basename: str) -> None:
    previous_basename = getattr(trainer, "best_checkpoint_filename", None)
    if not previous_basename or previous_basename == new_basename:
        return
    previous_path = os.path.join(checkpoint_dir, os.path.basename(str(previous_basename)))
    if os.path.isfile(previous_path):
        os.remove(previous_path)
        _print(f"Removed previous best checkpoint: {previous_path}")


def _should_write_checkpoint(trainer, log_dict, final_local_path: str, force: bool) -> bool:
    if force or not getattr(trainer.args, "save_best_only", False):
        return True

    metric_name = str(getattr(trainer.args, "checkpoint_metric", "test/l2_moved/mean"))
    current_value = _checkpoint_metric_value(trainer, log_dict)
    if current_value is None:
        if not os.path.exists(final_local_path) and getattr(trainer, "best_checkpoint_metric", None) is None:
            _print(
                f"No finite checkpoint metric '{metric_name}' was available; "
                "saving initial checkpoint because model-last.pt does not exist yet."
            )
            return True
        _print(f"Skipping checkpoint save: metric '{metric_name}' was not available in the latest eval logs.")
        return False

    if not _metric_improved(trainer, current_value):
        _print(
            f"Skipping checkpoint save: {metric_name}={current_value:.6g} did not improve "
            f"over best={float(trainer.best_checkpoint_metric):.6g}."
        )
        return False

    trainer.best_checkpoint_metric = current_value
    trainer.best_checkpoint_metric_name = metric_name
    _print(f"New best checkpoint metric: {metric_name}={current_value:.6g}; saving model-last.pt.")
    return True


def _build_checkpoint_dict(
    trainer,
    log_dict=None,
    *,
    score_se3_rng_state=None,
) -> dict:
    scheduler = getattr(trainer, "lr_scheduler", None)
    grad_scaler = getattr(trainer, "grad_scaler", None)
    save_dict = {
        "model": _unwrap_model(trainer.model).state_dict(),
        "optimizer": trainer.optimizer.state_dict(),
        "args": trainer.args,
        "exp_name": trainer.exp_name,
        "wandb_id": trainer.wandb_id,
        "batch_count": trainer.batch_count,
        "epoch_count": trainer.epoch_count,
        "sample_count": trainer.sample_count,
        # optimizer_step counts successful AdamW updates.  campaign_step also
        # counts the synchronized continued_baseline/metric_loss Phase-A no-op; it is authoritative for
        # phase transitions, cadence, and the 850-step campaign budget.
        "optimizer_step": int(getattr(trainer, "optimizer_step", 0)),
        "score_se3_campaign_step": int(
            getattr(trainer, "score_se3_campaign_step", 0)
        ),
        "optimizer_phase": str(getattr(trainer, "optimizer_phase", "single_phase")),
        "lr_scheduler": None if scheduler is None else scheduler.state_dict(),
        "lr_scheduler_class": (
            None
            if scheduler is None
            else f"{type(scheduler).__module__}.{type(scheduler).__qualname__}"
        ),
        "grad_scaler": None if grad_scaler is None else grad_scaler.state_dict(),
        "grad_scaler_class": (
            None
            if grad_scaler is None
            else f"{type(grad_scaler).__module__}.{type(grad_scaler).__qualname__}"
        ),
        "best_checkpoint_metric": getattr(trainer, "best_checkpoint_metric", None),
        "best_checkpoint_metric_name": getattr(trainer, "best_checkpoint_metric_name", None),
        "best_checkpoint_filename": getattr(trainer, "best_checkpoint_filename", None),
    }
    if bool(getattr(trainer.args, "score_se3_eval", False)):
        save_dict["score_se3_optimizer_runtime"] = (
            score_se3_optimizer_runtime_contract(trainer)
        )
        save_dict["score_se3_optimizer_plan"] = getattr(
            trainer, "score_se3_optimizer_plan", None
        )
        if (
            bool(getattr(trainer.args, "full_train_coverage", False))
            and not bool(getattr(trainer, "inference_only", False))
            and not bool(getattr(trainer, "evaluation_only", False))
        ):
            if grad_scaler is None:
                raise RuntimeError(
                    "score-SE3 training checkpoint requires causal GradScaler state"
                )
            from training.score_se3_data_resume import (
                score_se3_data_cursor_metadata,
            )

            save_dict["score_se3_data_cursor"] = (
                score_se3_data_cursor_metadata(trainer)
            )
            if score_se3_rng_state is None:
                if bool(getattr(trainer.args, "distributed", False)) or int(
                    getattr(trainer, "world_size", 1)
                ) != 1:
                    raise RuntimeError(
                        "distributed score-SE3 checkpoint construction requires "
                        "collective per-rank RNG state"
                    )
                from training.score_se3_data_resume import (
                    collect_score_se3_rng_states,
                )

                score_se3_rng_state = collect_score_se3_rng_states(trainer)
            save_dict["score_se3_rng_state"] = score_se3_rng_state
    attach_checkpoint_contract(save_dict, args=trainer.args, context="training save checkpoint")
    return save_dict


def save_latest_checkpoint_now(trainer, log_dict=None) -> None:
    """Always write model-latest.pt for resume, independent of save_best_only."""
    if trainer.args.distributed:
        dist.barrier()
    from training.score_se3_data_resume import collect_score_se3_rng_states

    score_se3_rng_state = collect_score_se3_rng_states(trainer)
    rank_zero_error = None
    if trainer.rank == 0:
        try:
            save_dict = _build_checkpoint_dict(
                trainer,
                log_dict,
                score_se3_rng_state=score_se3_rng_state,
            )
            checkpoint_dir = os.path.join(trainer.save_dir, "checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)
            latest_basename = _checkpoint_filename(trainer, log_dict or {})
            latest_named_path = os.path.join(checkpoint_dir, latest_basename)
            _atomic_torch_save(save_dict, latest_named_path)
            latest_alias = os.path.join(trainer.save_dir, "model-latest.pt")
            _link_or_copy(latest_named_path, latest_alias)
            _print(
                f"Latest checkpoint saved: epoch={trainer.epoch_count:.4f} "
                f"batch={trainer.batch_count} -> {latest_alias}"
            )
        except Exception as exc:
            if not trainer.args.distributed:
                raise
            rank_zero_error = f"{type(exc).__name__}: {exc}"
    if trainer.args.distributed:
        holder = [rank_zero_error]
        dist.broadcast_object_list(
            holder,
            src=0,
            group=getattr(trainer, "cpu_pg", None),
        )
        dist.barrier()
        if holder[0] is not None:
            raise RuntimeError(
                "rank zero failed to save the latest resume checkpoint: "
                f"{holder[0]}"
            )


def _score_se3_eval_checkpoint_basename(trainer) -> str:
    candidate = _safe_filename_token(
        getattr(trainer.args, "score_se3_candidate", "unknown")
    )
    campaign_step = int(getattr(trainer, "score_se3_campaign_step", 0))
    optimizer_step = int(getattr(trainer, "optimizer_step", 0))
    if campaign_step < 0 or optimizer_step < 0 or optimizer_step > campaign_step:
        raise RuntimeError(
            "invalid score-SE3 checkpoint counters: "
            f"campaign_step={campaign_step}, optimizer_step={optimizer_step}"
        )
    return (
        f"eval__candidate={candidate}"
        f"__campaign-step={campaign_step:08d}"
        f"__optimizer-step={optimizer_step:08d}.pt"
    )


def save_score_se3_eval_checkpoint(trainer, log_dict=None):
    """Retain one immutable checkpoint for every validation cadence.

    Unlike ``model-last.pt`` this checkpoint is never pruned by best-model
    selection.  Its stable name carries both campaign and actual AdamW update
    counters, and its sidecar returns/records a SHA256 for an auditable selected
    checkpoint manifest.
    """

    if not bool(getattr(trainer.args, "score_se3_eval", False)):
        raise ValueError("score-SE3 eval checkpoint requested for a non-score run")
    if getattr(trainer, "optimizer", None) is None:
        raise RuntimeError("cannot save a training eval checkpoint without an optimizer")
    # Validate rank-invariant inputs before entering collectives.  In
    # particular, a missing selection metric must fail every rank rather than
    # raising only on rank zero while peers wait forever in broadcast.
    basename = _score_se3_eval_checkpoint_basename(trainer)
    metric_name = str(
        getattr(trainer.args, "checkpoint_metric", "test/score_se3/composite")
    )
    metric_value = _checkpoint_metric_value(trainer, log_dict or {})
    if metric_value is None:
        raise RuntimeError(
            "retained score-SE3 eval checkpoint requires a finite selection "
            f"metric {metric_name!r} from the completed evaluation"
        )
    if trainer.args.distributed:
        dist.barrier()
    from training.score_se3_data_resume import collect_score_se3_rng_states

    score_se3_rng_state = collect_score_se3_rng_states(trainer)
    record = None
    rank_zero_error = None
    if trainer.rank == 0:
        try:
            directory = os.path.join(trainer.save_dir, "eval_checkpoints")
            os.makedirs(directory, exist_ok=True)
            path = os.path.abspath(os.path.join(directory, basename))
            metadata_path = os.path.splitext(path)[0] + ".json"

            if not os.path.exists(path):
                payload = _build_checkpoint_dict(
                    trainer,
                    log_dict or {},
                    score_se3_rng_state=score_se3_rng_state,
                )
                _atomic_torch_save(payload, path)
            digest = sha256_file(path)
            record = {
                "schema_version": 1,
                "checkpoint_type": "score_se3_validation_eval",
                "candidate": str(getattr(trainer.args, "score_se3_candidate", "")),
                "campaign_step": int(getattr(trainer, "score_se3_campaign_step", 0)),
                "optimizer_step": int(getattr(trainer, "optimizer_step", 0)),
                "metric_name": metric_name,
                "metric_value": metric_value,
                "path": path,
                "sha256": digest,
                "oracle": str(getattr(trainer.args, "score_se3_candidate", "")) == "oracle_contact",
                "deployable": str(getattr(trainer.args, "score_se3_candidate", "")) != "oracle_contact",
            }
            serialized = json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
            if os.path.exists(metadata_path):
                with open(metadata_path, "r", encoding="utf-8") as handle:
                    existing = handle.read()
                if existing != serialized:
                    raise RuntimeError(
                        "retained score-SE3 eval checkpoint metadata changed at an "
                        f"existing campaign step: {metadata_path}"
                    )
            else:
                temporary_metadata = f"{metadata_path}.tmp-{os.getpid()}"
                try:
                    with open(temporary_metadata, "w", encoding="utf-8") as handle:
                        handle.write(serialized)
                    os.replace(temporary_metadata, metadata_path)
                finally:
                    if os.path.exists(temporary_metadata):
                        os.remove(temporary_metadata)
            _print(
                "Retained score-SE3 eval checkpoint: "
                f"campaign_step={record['campaign_step']} "
                f"optimizer_step={record['optimizer_step']} sha256={digest} -> {path}"
            )
        except Exception as exc:
            if not trainer.args.distributed:
                raise
            rank_zero_error = f"{type(exc).__name__}: {exc}"
    if trainer.args.distributed:
        holder = [{"record": record, "error": rank_zero_error}]
        dist.broadcast_object_list(
            holder,
            src=0,
            group=getattr(trainer, "cpu_pg", None),
        )
        dist.barrier()
        payload = holder[0]
        if payload["error"] is not None:
            raise RuntimeError(
                "rank zero failed to retain score-SE3 eval checkpoint: "
                f"{payload['error']}"
            )
        record = payload["record"]
    return record


def save_checkpoint_now(trainer, adjusted_batch_count, log_dict=None, force=False):
    if trainer.args.distributed:
        dist.barrier()
    from training.score_se3_data_resume import collect_score_se3_rng_states

    score_se3_rng_state = collect_score_se3_rng_states(trainer)
    rank_zero_error = None
    if trainer.rank == 0:
        try:
            _print("Saving checkpoint")
            save_checkpoint(
                trainer,
                log_dict or {},
                force=force,
                score_se3_rng_state=score_se3_rng_state,
            )
        except Exception as exc:
            if not trainer.args.distributed:
                raise
            rank_zero_error = f"{type(exc).__name__}: {exc}"
    if trainer.args.distributed:
        holder = [rank_zero_error]
        dist.broadcast_object_list(
            holder,
            src=0,
            group=getattr(trainer, "cpu_pg", None),
        )
        dist.barrier()
        if holder[0] is not None:
            raise RuntimeError(
                "rank zero failed to save the selected checkpoint: "
                f"{holder[0]}"
            )


def save_checkpoint(
    trainer,
    log_dict=None,
    force=False,
    *,
    score_se3_rng_state=None,
):
    if trainer.rank != 0:
        return

    final_local_path = os.path.join(trainer.save_dir, "model-last.pt")
    if not _should_write_checkpoint(trainer, log_dict or {}, final_local_path, force=force):
        return trainer.save_dir

    save_dict = _build_checkpoint_dict(
        trainer,
        log_dict,
        score_se3_rng_state=score_se3_rng_state,
    )

    checkpoint_dir = os.path.join(trainer.save_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_basename = _checkpoint_filename(trainer, log_dict or {})
    named_local_path = os.path.join(checkpoint_dir, checkpoint_basename)
    _atomic_torch_save(save_dict, named_local_path)
    _link_or_copy(named_local_path, final_local_path)
    if getattr(trainer.args, "save_best_only", False) and not force:
        _remove_previous_best_checkpoint(trainer, checkpoint_dir, checkpoint_basename)
    trainer.best_checkpoint_filename = checkpoint_basename
    _print(f"Checkpoint saved: {named_local_path}")
    _print(f"Checkpoint alias: {final_local_path}")

    return trainer.save_dir


def load_checkpoint_from_path(trainer, model_path=None):
    if model_path is None:
        if trainer.rank == 0:
            _print("No checkpoint path provided, starting from scratch")
        return None
    assert os.path.exists(model_path), f"Checkpoint not found: {model_path}"
    if trainer.rank == 0:
        _print(f"Loading checkpoint from {model_path}")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    if "wandb_id" in checkpoint:
        checkpoint.pop("wandb_id")
    if "exp_name" in checkpoint:
        checkpoint.pop("exp_name")
    return checkpoint


def _scheduler_class_name(scheduler) -> str | None:
    if scheduler is None:
        return None
    return f"{type(scheduler).__module__}.{type(scheduler).__qualname__}"


def _restore_grad_scaler_state(trainer, checkpoint, *, score_resume: bool) -> None:
    scaler = getattr(trainer, "grad_scaler", None)
    saved_state = checkpoint.get("grad_scaler")
    saved_class = checkpoint.get("grad_scaler_class")
    current_class = _scheduler_class_name(scaler)
    strict = score_resume and bool(
        getattr(trainer.args, "full_train_coverage", False)
    )
    if saved_state is None or saved_class is None:
        if strict:
            raise RuntimeError(
                "score-SE3 resume checkpoint is missing causal GradScaler state"
            )
        return
    if scaler is None:
        raise RuntimeError(
            "checkpoint records GradScaler state but current training has no scaler"
        )
    if saved_class != current_class:
        raise RuntimeError(
            "GradScaler class mismatch on resume: "
            f"checkpoint={saved_class!r}, current={current_class!r}"
        )
    if not isinstance(saved_state, dict):
        raise RuntimeError("checkpoint GradScaler state must be a dictionary")
    try:
        scaler.load_state_dict(saved_state)
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("checkpoint GradScaler state is incompatible") from exc


def _optimizer_topology(runtime: dict) -> dict:
    """Select resume-critical topology while excluding time-varying values."""

    groups = []
    for group in runtime.get("parameter_groups", []):
        groups.append(
            {
                "index": int(group["index"]),
                "name": str(group["name"]),
                "parameter_names": list(group["parameter_names"]),
                "weight_decay": float(group["weight_decay"]),
            }
        )
    return {
        "optimizer_class": runtime.get("optimizer_class"),
        "scheduler_class": runtime.get("scheduler_class"),
        "trainable_parameter_names": list(runtime.get("trainable_parameter_names", [])),
        "parameter_groups": groups,
    }


def _validate_score_se3_optimizer_topology(trainer, checkpoint) -> None:
    if not bool(getattr(trainer.args, "score_se3_eval", False)):
        return
    if str(getattr(trainer.args, "score_se3_checkpoint_role", "init")) != "resume":
        return
    saved = checkpoint.get("score_se3_optimizer_runtime")
    if not isinstance(saved, dict):
        raise RuntimeError(
            "score-SE3 resume checkpoint is missing score_se3_optimizer_runtime; "
            "an initialization checkpoint must use checkpoint_role=init"
        )
    current = score_se3_optimizer_runtime_contract(trainer)
    saved_topology = _optimizer_topology(saved)
    current_topology = _optimizer_topology(current)
    if current_topology != saved_topology:
        raise RuntimeError(
            "score-SE3 optimizer topology mismatch on resume. "
            f"saved={saved_topology} current={current_topology}"
        )


def _restore_scheduler_state(trainer, checkpoint) -> None:
    scheduler = getattr(trainer, "lr_scheduler", None)
    saved_state = checkpoint.get("lr_scheduler")
    saved_class = checkpoint.get("lr_scheduler_class")
    current_class = _scheduler_class_name(scheduler)
    score_resume = bool(getattr(trainer.args, "score_se3_eval", False)) and str(
        getattr(trainer.args, "score_se3_checkpoint_role", "init")
    ) == "resume"

    if scheduler is None and saved_state is None:
        if score_resume and saved_class is not None:
            raise RuntimeError(
                "score-SE3 resume checkpoint records a scheduler but the current run has none"
            )
        return
    if scheduler is None or saved_state is None:
        message = (
            "LR scheduler state mismatch on resume: "
            f"checkpoint_class={saved_class!r}, current_class={current_class!r}"
        )
        if getattr(trainer.args, "allow_optimizer_reset", False) and not score_resume:
            _print(message + "; scheduler reset was explicitly allowed.")
            return
        raise RuntimeError(message)
    if saved_class != current_class:
        raise RuntimeError(
            "LR scheduler class mismatch on resume: "
            f"checkpoint={saved_class!r}, current={current_class!r}"
        )
    try:
        if bool(getattr(trainer.args, "score_se3_eval", False)) and hasattr(
            scheduler, "apply_learning_rates"
        ):
            scheduler.load_state_dict(
                saved_state, optimizer=trainer.optimizer
            )
        else:
            scheduler.load_state_dict(saved_state)
        apply_learning_rates = getattr(scheduler, "apply_learning_rates", None)
        if callable(apply_learning_rates):
            apply_learning_rates(trainer.optimizer)
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("LR scheduler state is incompatible with the current run") from exc


def load_checkpoint_state(trainer, checkpoint, *, restore_optimizer=None):
    """Restore an architecture-identical checkpoint, optimizer, and schedule.

    Experimental warm-start wrappers may replace ``load_checkpoint``.  A
    score-SE3 resume is not a warm start and calls this stable core directly so
    the long-horizon init wrapper cannot reset its optimizer/counters.  Eval-
    only callers pass ``restore_optimizer=False`` while retaining exact model
    and authoritative counter restoration.
    """
    model_state_dict = checkpoint["model"]
    expected_keys = set(_unwrap_model(trainer.model).state_dict().keys())
    ckpt_keys = set(model_state_dict.keys())
    if ckpt_keys != expected_keys:
        missing = sorted(expected_keys - ckpt_keys)
        extra = sorted(ckpt_keys - expected_keys)
        missing_suffix = "..." if len(missing) > 10 else ""
        extra_suffix = "..." if len(extra) > 10 else ""
        raise RuntimeError(
            "Checkpoint model keys do not match the current model state_dict. "
            "Ensure the distributed/Non-DDP setting and model config match. "
            f"Missing keys: {missing[:10]}{missing_suffix}. "
            f"Extra keys: {extra[:10]}{extra_suffix}."
        )
    _unwrap_model(trainer.model).load_state_dict(model_state_dict)

    if restore_optimizer is None:
        restore_optimizer = not trainer.inference_only
    if restore_optimizer:
        score_resume = bool(getattr(trainer.args, "score_se3_eval", False)) and str(
            getattr(trainer.args, "score_se3_checkpoint_role", "init")
        ) == "resume"
        if score_resume and bool(
            getattr(trainer.args, "allow_optimizer_reset", False)
        ):
            raise RuntimeError(
                "score-SE3 resume forbids allow_optimizer_reset; optimizer and "
                "scheduler state must be restored exactly"
            )
        optimizer_restored = False
        if getattr(trainer.args, "reset_optimizer_on_resume", False):
            lr_values = [
                group.get("lr")
                for group in getattr(trainer.optimizer, "param_groups", [])
            ]
            lr_msg = f" lr={lr_values[0]:.6g}" if len(lr_values) == 1 else f" lrs={lr_values}"
            _print(f"Resetting optimizer on resume; using fresh optimizer state with{lr_msg}.")
        elif "optimizer" not in checkpoint:
            if getattr(trainer.args, "allow_optimizer_reset", False) and not score_resume:
                _print("Checkpoint missing optimizer state; resetting optimizer as requested.")
            else:
                raise RuntimeError(
                    "Checkpoint missing optimizer state. "
                    "Rerun with --allow_optimizer_reset=true to skip optimizer restore."
                )
        else:
            _validate_score_se3_optimizer_topology(trainer, checkpoint)
            try:
                trainer.optimizer.load_state_dict(checkpoint["optimizer"])
                optimizer_restored = True
            except ValueError as exc:
                if getattr(trainer.args, "allow_optimizer_reset", False) and not score_resume:
                    _print(
                        "Optimizer state incompatible with current optimizer; "
                        "resetting optimizer state as requested."
                    )
                else:
                    raise RuntimeError(
                        "Optimizer state incompatible with current optimizer. "
                        "Rerun with --allow_optimizer_reset=true to skip optimizer restore."
                    ) from exc
        if optimizer_restored:
            _restore_scheduler_state(trainer, checkpoint)
            _restore_grad_scaler_state(
                trainer,
                checkpoint,
                score_resume=score_resume,
            )
    assert "batch_count" in checkpoint, "Checkpoint missing batch_count."
    assert "epoch_count" in checkpoint, "Checkpoint missing epoch_count."
    assert "sample_count" in checkpoint, "Checkpoint missing sample_count."
    trainer.batch_count = checkpoint["batch_count"]
    trainer.epoch_count = checkpoint["epoch_count"]
    trainer.sample_count = checkpoint["sample_count"]
    if "optimizer_step" in checkpoint:
        optimizer_step = int(checkpoint["optimizer_step"])
        if optimizer_step < 0:
            raise RuntimeError("Checkpoint optimizer_step must be non-negative")
    else:
        score_resume = bool(getattr(trainer.args, "score_se3_eval", False)) and str(
            getattr(trainer.args, "score_se3_checkpoint_role", "init")
        ) == "resume"
        if score_resume:
            raise RuntimeError(
                "score-SE3 resume requires a checkpoint with authoritative optimizer_step"
            )
        optimizer_step = int(checkpoint["batch_count"]) // max(
            int(getattr(trainer, "world_size", 1)), 1
        )
        _print(
            "Checkpoint predates optimizer_step; inferred a legacy value "
            f"of {optimizer_step} from batch_count/current world size."
        )
    trainer.optimizer_step = optimizer_step
    if bool(getattr(trainer.args, "score_se3_eval", False)):
        if "score_se3_campaign_step" not in checkpoint:
            raise RuntimeError(
                "score-SE3 resume requires a checkpoint with authoritative "
                "score_se3_campaign_step"
            )
        campaign_step = int(checkpoint["score_se3_campaign_step"])
        if campaign_step < optimizer_step:
            raise RuntimeError(
                "score_se3_campaign_step cannot be smaller than optimizer_step"
            )
        trainer.score_se3_campaign_step = campaign_step
    trainer.optimizer_phase = str(
        checkpoint.get("optimizer_phase", getattr(trainer, "optimizer_phase", "single_phase"))
    )
    runtime = checkpoint.get("score_se3_optimizer_runtime")
    if isinstance(runtime, dict) and int(runtime.get("optimizer_step", optimizer_step)) != optimizer_step:
        raise RuntimeError(
            "Checkpoint optimizer_step disagrees with score_se3_optimizer_runtime"
        )
    if isinstance(runtime, dict) and bool(getattr(trainer.args, "score_se3_eval", False)):
        if int(runtime.get("campaign_step", campaign_step)) != campaign_step:
            raise RuntimeError(
                "Checkpoint score_se3_campaign_step disagrees with "
                "score_se3_optimizer_runtime"
            )
    trainer.best_checkpoint_metric = checkpoint.get("best_checkpoint_metric")
    trainer.best_checkpoint_metric_name = checkpoint.get("best_checkpoint_metric_name")
    trainer.best_checkpoint_filename = checkpoint.get("best_checkpoint_filename")


def load_checkpoint(trainer, checkpoint):
    return load_checkpoint_state(trainer, checkpoint)
