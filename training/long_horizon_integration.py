from __future__ import annotations

import json
import os
from pathlib import Path

from training.long_horizon_common import _env_bool, normalize_track, stage_name
from utils import _print


def _allowed_new_key(key: str) -> bool:
    return key.startswith("object_jepa_") or any(
        component.startswith("long_horizon_")
        for component in key.split(".")
    )


def install_long_horizon_checkpoint_loader() -> None:
    """Strict warm-start loader allowing only the campaign's new tensors."""
    import training.checkpointing as checkpointing

    if getattr(checkpointing, "_long_horizon_loader_installed", False):
        return
    original = checkpointing.load_checkpoint

    def load_checkpoint(trainer, checkpoint):
        if not _env_bool("POINTWORLD_LONG_HORIZON_ABLATION", False):
            return original(trainer, checkpoint)
        model = checkpointing._unwrap_model(trainer.model)
        expected = set(model.state_dict())
        supplied = set(checkpoint["model"])
        missing = sorted(expected - supplied)
        extra = sorted(supplied - expected)
        illegal_missing = [key for key in missing if not _allowed_new_key(key)]
        if extra or illegal_missing:
            raise RuntimeError(
                "Long-horizon warm start rejected checkpoint mismatch. "
                f"illegal_missing={illegal_missing[:20]} extra={extra[:20]}"
            )
        incompatible = model.load_state_dict(checkpoint["model"], strict=False)
        if (
            sorted(incompatible.missing_keys) != missing
            or incompatible.unexpected_keys
        ):
            raise RuntimeError(
                "Unexpected checkpoint load result: "
                f"missing={incompatible.missing_keys[:20]} "
                f"unexpected={incompatible.unexpected_keys[:20]}"
            )
        if trainer.rank == 0:
            _print(
                "Warm-started immutable SOTA checkpoint; initialized "
                f"{len(missing)} long-horizon tensors/buffers from scratch."
            )
        if not trainer.inference_only and not bool(
            getattr(trainer.args, "reset_optimizer_on_resume", False)
        ):
            raise RuntimeError(
                "Long-horizon architecture extension requires "
                "--reset_optimizer_on_resume true."
            )
        trainer.batch_count = 0
        trainer.epoch_count = 0.0
        trainer.sample_count = 0
        trainer.best_checkpoint_metric = None
        trainer.best_checkpoint_metric_name = None
        trainer.best_checkpoint_filename = None

    checkpointing.load_checkpoint = load_checkpoint
    checkpointing._long_horizon_loader_installed = True


def install_long_horizon_eval_metrics() -> None:
    """Attach count-weighted structural/contact metrics and persist eval JSON."""
    import training.trainer as trainer_module

    if getattr(trainer_module, "_long_horizon_eval_installed", False):
        return
    original_accumulate = (
        trainer_module.Trainer._accumulate_hoi_dataset_eval_metrics
    )
    original_eval_step = trainer_module.Trainer.eval_step

    def accumulate(self, metrics, counts, batch, outputs):
        original_accumulate(self, metrics, counts, batch, outputs)
        model = self.model.module if self.args.distributed else self.model
        if not hasattr(model, "long_horizon_eval_sums"):
            return
        for key, (total, count) in model.long_horizon_eval_sums(
            batch, outputs
        ).items():
            metrics[key] += total.to(metrics[key].dtype)
            counts[key] += count.to(counts[key].dtype)

    def eval_step(self, dataloader, data_iter, prefix, log_dict=None):
        result, iterator = original_eval_step(
            self,
            dataloader,
            data_iter,
            prefix,
            log_dict=log_dict,
        )
        if prefix == "test" and self.rank == 0:
            output_dir = Path(
                str(getattr(self.args, "output_dir", self.save_dir))
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            stage = int(os.environ.get("LONG_HORIZON_STAGE", 0))
            payload = {
                "stage": stage,
                "stage_name": stage_name(stage),
                "track": normalize_track(),
                "metrics": {
                    key: value
                    for key, value in result.items()
                    if key.startswith("test/")
                },
            }
            (output_dir / "latest_test_metrics.json").write_text(
                json.dumps(payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        return result, iterator

    trainer_module.Trainer._accumulate_hoi_dataset_eval_metrics = accumulate
    trainer_module.Trainer.eval_step = eval_step
    trainer_module._long_horizon_eval_installed = True
