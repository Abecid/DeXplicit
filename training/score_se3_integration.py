from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


SCORE_SE3_RESUME_CAUSAL_ARGS = (
    "score_se3_loss_weight",
    "score_se3_gradient_ratio",
    "score_se3_max_calibrated_weight",
    "score_se3_calibration_only",
    "score_se3_require_calibration_artifact",
    "score_se3_calibration_batches",
    "score_se3_calibration_artifact_sha256",
    "score_se3_calibration_formula_version",
    "score_se3_calibration_sample_sequence_sha256",
    "score_se3_calibration_warm_checkpoint_sha256",
    "score_se3_contact_loss_weight",
    "score_se3_contact_stop_gradient",
    "score_se3_translation_bound_m",
    "score_se3_rotation_bound_deg",
    "score_se3_contact_tau_m",
    "score_se3_eval_data_seed",
    "checkpoint_metric",
    "checkpoint_metric_mode",
    "eval_freq",
    "save_freq",
    "save_best_only",
    "max_train_steps",
    "allow_optimizer_reset",
    # These define the deterministic full-coverage iterator whose exact
    # within-epoch cursor is restored from campaign_step.
    "deterministic_train",
    "full_train_coverage",
    "eval_test_only",
    "num_workers",
    "batch_size",
    "seed",
    "grad_clip_max_norm",
    "deterministic_algorithms",
    "local_increment_loss_weight",
    "horizon_position_loss_weight",
    "horizon_weight_start",
    "horizon_weight_end",
    "velocity_loss_weight",
    "acceleration_loss_weight",
    "object_displacement_loss_weight",
    "local_isometry_loss_weight",
    "local_isometry_k",
    "local_isometry_decay",
    "huber_delta",
    "min_motion_loss_weight",
    "balanced_moved_static_loss",
    "moved_loss_weight",
    "static_loss_weight",
    "static_displacement_loss_weight",
    "static_displacement_tau",
    "use_move_prediction_head",
    "move_prediction_loss_weight",
    "move_prediction_tau",
    "move_prediction_temp_scale",
)

SCORE_SE3_INFERENCE_CAUSAL_ARGS = (
    "score_se3_loss_weight",
    "score_se3_gradient_ratio",
    "score_se3_calibration_only",
    "score_se3_require_calibration_artifact",
    "score_se3_calibration_batches",
    "score_se3_calibration_artifact_sha256",
    "score_se3_calibration_formula_version",
    "score_se3_calibration_sample_sequence_sha256",
    "score_se3_calibration_warm_checkpoint_sha256",
    "score_se3_eval_data_seed",
    "score_se3_translation_bound_m",
    "score_se3_rotation_bound_deg",
    "score_se3_contact_tau_m",
    "score_se3_allow_oracle",
    "dynamics_experiment",
    "action_conditioning_mode",
    "hand_condition_encoder",
    "hand_condition_fusion",
    "hand_action_token_encoder",
    "hoi_target_horizon",
)

DEFORMATION_GRAPH_CAUSAL_ARGS = (
    "deformation_graph_candidate",
    "deformation_graph_nodes",
    "deformation_graph_influences",
    "deformation_graph_edges",
    "deformation_graph_radius_scale",
    "deformation_graph_minimum_radius_m",
    "deformation_graph_translation_bound_m",
    "deformation_graph_rotation_bound_deg",
    "deformation_graph_arap_weight",
    "deformation_graph_temporal_weight",
    "deformation_graph_magnitude_weight",
    "deformation_graph_gate_weight",
    "deformation_graph_gate_rigid_floor_m",
    "deformation_graph_gate_full_scale_m",
)
SCORE_SE3_RESUME_CAUSAL_ARGS = (
    SCORE_SE3_RESUME_CAUSAL_ARGS + DEFORMATION_GRAPH_CAUSAL_ARGS
)
SCORE_SE3_INFERENCE_CAUSAL_ARGS = (
    SCORE_SE3_INFERENCE_CAUSAL_ARGS + DEFORMATION_GRAPH_CAUSAL_ARGS
)


def _is_score_key(key: str) -> bool:
    return any(component.startswith("score_se3_") for component in key.split("."))


def _is_deformation_graph_key(key: str) -> bool:
    return any(
        component.startswith("deformation_graph_") for component in key.split(".")
    )


def _is_allowed_base_extension_key(key: str) -> bool:
    return _is_score_key(key) or key.startswith("object_jepa_") or any(
        component.startswith("long_horizon_") for component in key.split(".")
    )


def prepare_score_se3_initial_state(
    expected_keys: set[str],
    supplied_state: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Filter prior-candidate state and validate an init-only architecture extension."""

    filtered_state = {
        key: value
        for key, value in supplied_state.items()
        if not _is_score_key(key)
    }
    supplied = set(filtered_state)
    missing = sorted(expected_keys - supplied)
    extra = sorted(supplied - expected_keys)
    illegal_missing = [
        key for key in missing if not _is_allowed_base_extension_key(key)
    ]
    if extra or illegal_missing:
        raise RuntimeError(
            "score-SE3 init checkpoint mismatch. "
            f"illegal_missing={illegal_missing[:20]} extra={extra[:20]}"
        )
    return filtered_state, missing


def prepare_deformation_graph_initial_state(
    expected_keys: set[str],
    supplied_state: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Retain the complete trained metric_pose state and initialize only graph tensors."""

    filtered_state = dict(supplied_state)
    supplied = set(filtered_state)
    missing = sorted(expected_keys - supplied)
    extra = sorted(supplied - expected_keys)
    illegal_missing = [key for key in missing if not _is_deformation_graph_key(key)]
    if extra or illegal_missing:
        raise RuntimeError(
            "deformation-graph metric_pose init checkpoint mismatch. "
            f"illegal_missing={illegal_missing[:20]} extra={extra[:20]}"
        )
    return filtered_state, missing


def _saved_candidate(checkpoint: dict[str, Any]) -> str | None:
    saved_args = checkpoint.get("args")
    if saved_args is None:
        return None
    if isinstance(saved_args, dict):
        value = saved_args.get("score_se3_candidate")
    else:
        value = getattr(saved_args, "score_se3_candidate", None)
    return None if value is None else str(value)


def _saved_deformation_graph_candidate(
    checkpoint: dict[str, Any],
) -> str | None:
    saved_args = checkpoint.get("args")
    if saved_args is None:
        return None
    if isinstance(saved_args, dict):
        value = saved_args.get("deformation_graph_candidate")
    else:
        value = getattr(saved_args, "deformation_graph_candidate", None)
    return None if value is None else str(value)


def _argument_value(args: Any, name: str) -> Any:
    if isinstance(args, Mapping):
        return args.get(name)
    return getattr(args, name, None)


def score_se3_resume_argument_differences(
    saved_args: Any,
    current_args: Any,
) -> dict[str, dict[str, Any]]:
    differences: dict[str, dict[str, Any]] = {}
    for name in SCORE_SE3_RESUME_CAUSAL_ARGS:
        saved = _argument_value(saved_args, name)
        current = _argument_value(current_args, name)
        if saved != current:
            differences[name] = {"checkpoint": saved, "current": current}
    return differences


INHERITED_E2_CAUSAL_ARGS = (
    "score_se3_loss_weight",
    "score_se3_translation_bound_m",
    "score_se3_rotation_bound_deg",
    "score_se3_contact_tau_m",
    "dynamics_experiment",
    "action_conditioning_mode",
    "hand_condition_encoder",
    "hand_condition_fusion",
    "hand_action_token_encoder",
    "hoi_target_horizon",
)


def inherited_e2_argument_differences(
    saved_args: Any,
    current_args: Any,
) -> dict[str, dict[str, Any]]:
    differences: dict[str, dict[str, Any]] = {}
    for name in INHERITED_E2_CAUSAL_ARGS:
        saved = _argument_value(saved_args, name)
        current = _argument_value(current_args, name)
        if saved != current:
            differences[name] = {"checkpoint": saved, "current": current}
    return differences


def score_se3_inference_argument_differences(
    saved_args: Any,
    current_args: Any,
) -> dict[str, dict[str, Any]]:
    differences: dict[str, dict[str, Any]] = {}
    for name in SCORE_SE3_INFERENCE_CAUSAL_ARGS:
        saved = _argument_value(saved_args, name)
        current = _argument_value(current_args, name)
        if saved != current:
            differences[name] = {"checkpoint": saved, "current": current}
    return differences


def _prepare_optimizer_for_score_se3_resume(trainer: Any, checkpoint: dict[str, Any]) -> None:
    saved_plan = checkpoint.get("score_se3_optimizer_plan")
    current_plan = getattr(trainer, "score_se3_optimizer_plan", None)
    if saved_plan != current_plan:
        raise RuntimeError(
            "score-SE3 optimizer phase plan changed on resume: "
            f"checkpoint={saved_plan!r}, current={current_plan!r}"
        )
    saved_phase = str(checkpoint.get("optimizer_phase", ""))
    if not saved_phase:
        raise RuntimeError("score-SE3 resume checkpoint does not record optimizer_phase")
    current_phase = str(getattr(trainer, "optimizer_phase", ""))
    if current_phase == saved_phase:
        return
    if isinstance(saved_plan, Mapping) and bool(
        saved_plan.get("static_optimizer_topology", False)
    ):
        # The staged controller keeps identical named AdamW groups for both
        # phases. Its serialized scheduler state restores counters/rates after
        # optimizer.load_state_dict, so no topology rebuild is necessary.
        return

    # A staged implementation must rebuild the exact saved groups before
    # Optimizer.load_state_dict.  The hook is deliberately explicit so a Phase
    # B checkpoint can never be loaded into Phase A parameter ordering.
    prepare = getattr(trainer, "prepare_score_se3_optimizer_for_resume", None)
    if not callable(prepare):
        raise RuntimeError(
            "score-SE3 resume needs optimizer phase "
            f"{saved_phase!r}, current phase is {current_phase!r}, and trainer "
            "does not implement prepare_score_se3_optimizer_for_resume(checkpoint)"
        )
    prepare(checkpoint)
    if str(getattr(trainer, "optimizer_phase", "")) != saved_phase:
        raise RuntimeError(
            "prepare_score_se3_optimizer_for_resume did not install the saved phase: "
            f"expected={saved_phase!r}, current={getattr(trainer, 'optimizer_phase', None)!r}"
        )


def _assert_loaded_score_se3_model_contract(model: Any) -> None:
    """Reject persisted buffers that silently override campaign architecture."""

    predictor = getattr(model, "dynamics_predictor", None)
    if predictor is None:
        raise RuntimeError("score-SE3 model has no dynamics_predictor after load")
    observed = {
        "stage": int(getattr(predictor, "long_horizon_stage", -1)),
        "track": str(getattr(predictor, "long_horizon_track", "")),
        "output_mode": str(
            getattr(predictor, "long_horizon_output_mode", "")
        ),
    }
    expected = {"stage": 0, "track": "isolated", "output_mode": "blend"}
    if observed != expected:
        raise RuntimeError(
            "loaded score-SE3 predictor violates the fixed stage-0 isolated "
            f"blend contract: observed={observed}, expected={expected}"
        )


def _assert_restored_score_se3_controller_state(trainer: Any) -> None:
    """Cross-check redundant staged-schedule counters after resume."""

    controller = getattr(trainer, "score_se3_controller", None)
    if controller is None:
        raise RuntimeError(
            "score-SE3 resume did not construct its staged optimizer controller"
        )
    expected = {
        "campaign_step": int(getattr(trainer, "score_se3_campaign_step", -1)),
        "optimizer_step": int(getattr(trainer, "optimizer_step", -1)),
        "optimizer_phase": str(getattr(trainer, "optimizer_phase", "")),
    }
    observed = {
        "campaign_step": int(getattr(controller, "campaign_step", -1)),
        "optimizer_step": int(getattr(controller, "optimizer_step", -1)),
        "optimizer_phase": str(getattr(controller, "optimizer_phase", "")),
    }
    if observed != expected:
        raise RuntimeError(
            "score-SE3 staged controller disagrees with restored checkpoint "
            f"counters: controller={observed}, checkpoint={expected}"
        )


def write_trainer_score_se3_contract(trainer: Any) -> tuple[Path, dict[str, Any]] | None:
    """Persist the runtime-aware immutable run contract on rank zero."""

    if not bool(getattr(trainer.args, "score_se3_eval", False)):
        return None
    if int(getattr(trainer, "rank", 0)) != 0:
        return None
    if not getattr(trainer.args, "output_dir", None):
        if bool(getattr(trainer.args, "score_se3_strict_contract", False)):
            raise ValueError("score-SE3 strict contract requires --output-dir")
        trainer.args.output_dir = str(getattr(trainer, "save_dir"))
    from training.score_se3_contract import write_score_se3_contract

    repo_root = Path(__file__).resolve().parents[1]
    return write_score_se3_contract(
        trainer.args,
        repo_root=repo_root,
        trainer=trainer,
    )


def _dump_init_state_hashes(
    trainer: Any, model: Any, checkpoint_keys: "set[str]"
) -> None:
    """Write an exact per-tensor fingerprint of the constructed+loaded model.

    GPU evaluation of the PTv3 backbone is not bit-reproducible run-to-run
    (custom CUDA kernels outside torch.use_deterministic_algorithms), so
    initialization parity is proven on the WEIGHTS.  Each entry records
    ``from_checkpoint``: whether the tensor's value came from the frozen base
    checkpoint (these must hash identically across all candidates) or is an
    extension tensor constructed per-candidate (loss buffers, contact heads,
    adapter trunk); extension randomness is candidate machinery and does not
    perturb the warm prediction because the adapter final projection is
    exactly zero — that zero is the other hard gate.
    """
    import hashlib
    import json as _json
    from pathlib import Path as _Path

    output_dir = getattr(trainer.args, "output_dir", None)
    if not output_dir:
        return
    entries: dict[str, dict[str, Any]] = {}
    for key, tensor in model.state_dict().items():
        data = tensor.detach().cpu().contiguous()
        digest = hashlib.sha256(data.numpy().tobytes()).hexdigest()
        entries[key] = {
            "sha256": digest,
            "zero": bool((data == 0).all().item()),
            "shape": list(data.shape),
            "from_checkpoint": key in checkpoint_keys,
        }
    path = _Path(output_dir) / "init_state_hashes.json"
    payload = {
        "schema_version": 2,
        "candidate": str(getattr(trainer.args, "score_se3_candidate", "")),
        "entries": entries,
    }
    tmp = path.with_suffix(".tmp")
    tmp.write_text(_json.dumps(payload, indent=1, sort_keys=True))
    tmp.replace(path)


def install_score_se3_checkpoint_loader() -> None:
    """Install explicit init, resume, and eval checkpoint semantics for continued_baseline-oracle_contact."""

    import training.checkpointing as checkpointing

    if getattr(checkpointing, "_score_se3_loader_installed", False):
        return
    original = checkpointing.load_checkpoint

    def load_checkpoint(trainer, checkpoint):
        if not bool(getattr(trainer.args, "score_se3_eval", False)):
            return original(trainer, checkpoint)
        role = str(
            getattr(trainer.args, "score_se3_checkpoint_role", "init")
        )
        if role not in {"init", "resume", "eval", "finetune"}:
            raise ValueError(
                "score_se3_checkpoint_role must be init, resume, eval, or finetune, "
                f"got {role!r}"
            )
        current_candidate = str(
            getattr(trainer.args, "score_se3_candidate", "")
        )
        current_graph_candidate = str(
            getattr(trainer.args, "deformation_graph_candidate", "") or ""
        )
        model = checkpointing._unwrap_model(trainer.model)

        if role in {"resume", "eval", "finetune"}:
            saved_candidate = _saved_candidate(checkpoint)
            if saved_candidate is None:
                raise RuntimeError(
                    f"score-SE3 {role} checkpoint does not record its candidate"
                )
            if saved_candidate != current_candidate:
                raise RuntimeError(
                    f"score-SE3 {role} candidate mismatch: "
                    f"checkpoint={saved_candidate}, current={current_candidate}"
                )
            if current_graph_candidate:
                saved_graph_candidate = _saved_deformation_graph_candidate(checkpoint)
                if saved_graph_candidate != current_graph_candidate:
                    raise RuntimeError(
                        f"deformation-graph {role} candidate mismatch: "
                        f"checkpoint={saved_graph_candidate!r}, "
                        f"current={current_graph_candidate!r}"
                    )
        if role == "eval":
            differences = score_se3_inference_argument_differences(
                checkpoint.get("args"),
                trainer.args,
            )
            if differences:
                raise RuntimeError(
                    "score-SE3 inference arguments changed for eval: "
                    f"{differences}"
                )
            saved_plan = checkpoint.get("score_se3_optimizer_plan")
            # The held-out process is inference-only: it must not construct or
            # validate a training optimizer topology.  Retained production
            # checkpoints carry the historical plan, so preserve it as
            # provenance when present, but older/audited model checkpoints are
            # still valid evaluation inputs without it.
            if isinstance(saved_plan, Mapping):
                trainer.score_se3_optimizer_plan = dict(saved_plan)
                trainer.args.score_se3_optimizer_plan = dict(saved_plan)
                trainer.args.score_se3_trainable_parameter_names = list(
                    saved_plan.get("trainable_parameter_names", ())
                )
            else:
                trainer.score_se3_optimizer_plan = None
                trainer.args.score_se3_optimizer_plan = None
                trainer.args.score_se3_trainable_parameter_names = []
            result = checkpointing.load_checkpoint_state(
                trainer,
                checkpoint,
                restore_optimizer=False,
            )
            trainer.curr_run_batch_count = int(
                trainer.score_se3_campaign_step
            ) * int(getattr(trainer, "world_size", 1))
            # Make accidental optimizer use/checkpoint mutation impossible in
            # the held-out evaluation process.
            trainer.optimizer = None
            trainer.lr_scheduler = None
            trainer.score_se3_eval_only_loaded = True
            # The long-horizon zero-init cleanliness assertion compares the
            # first forward against a reference that excludes the score-SE3
            # adapter; it is an INIT-time invariant.  Held-out evaluation
            # loads TRAINED checkpoints whose adapters are legitimately
            # non-zero, so the assertion would reject every trained metric_pose-oracle_contact
            # checkpoint.  Mark it satisfied for this inference-only process.
            predictor = getattr(
                getattr(model, "module", model), "dynamics_predictor", None
            )
            if predictor is not None and hasattr(
                predictor, "_long_horizon_zero_init_checked"
            ):
                predictor._long_horizon_zero_init_checked = True
            if predictor is not None and hasattr(
                predictor, "_deformation_graph_identity_checked"
            ):
                predictor._deformation_graph_identity_checked = True
            _assert_loaded_score_se3_model_contract(model)
            return result

        if role == "finetune":
            # Warm-start every parameter of a TRAINED candidate checkpoint
            # (e.g. the selected full dexplicit_3d) for task fine-tuning on a new corpus
            # (PhysTwin): strict full load, fresh optimizer/schedule counters
            # (unlike resume), and training enabled (unlike eval). The
            # zero-init/identity assertions are init-time invariants that a
            # trained graph/carrier extension legitimately violates.
            state = checkpoint["model"]
            if any(key.startswith("module.") for key in state):
                state = {key[len("module."):] if key.startswith("module.") else key: value
                         for key, value in state.items()}
            incompatible = model.load_state_dict(state, strict=False)
            if incompatible.missing_keys or incompatible.unexpected_keys:
                raise RuntimeError(
                    "score-SE3 finetune requires an exact parameter match: "
                    f"missing={incompatible.missing_keys[:20]} "
                    f"unexpected={incompatible.unexpected_keys[:20]}"
                )
            predictor = getattr(model, "dynamics_predictor", None)
            for flag in ("_long_horizon_zero_init_checked", "_deformation_graph_identity_checked"):
                if predictor is not None and hasattr(predictor, flag):
                    setattr(predictor, flag, True)
            trainer.batch_count = 0
            trainer.optimizer_step = 0
            trainer.score_se3_campaign_step = 0
            trainer.curr_run_batch_count = 0
            trainer.epoch_count = 0.0
            trainer.sample_count = 0
            trainer.best_checkpoint_metric = None
            trainer.best_checkpoint_metric_name = None
            trainer.best_checkpoint_filename = None
            if trainer.rank == 0:
                from utils import _print

                _print(
                    f"Fine-tune warm start: loaded all {len(state)} tensors of the trained "
                    f"{current_candidate}/{current_graph_candidate or '-'} checkpoint with fresh counters."
                )
            _assert_loaded_score_se3_model_contract(model)
            return None

        if role == "resume":
            if bool(getattr(trainer.args, "reset_optimizer_on_resume", False)):
                raise RuntimeError(
                    "score-SE3 checkpoint_role=resume requires "
                    "reset_optimizer_on_resume=false"
                )
            if bool(getattr(trainer.args, "allow_optimizer_reset", False)):
                raise RuntimeError(
                    "score-SE3 checkpoint_role=resume forbids "
                    "allow_optimizer_reset; optimizer and scheduler state are "
                    "causal campaign state"
                )
            saved_args = checkpoint.get("args")
            differences = score_se3_resume_argument_differences(
                saved_args,
                trainer.args,
            )
            if differences:
                raise RuntimeError(
                    "score-SE3 causal arguments changed on resume: "
                    f"{differences}"
                )
            _prepare_optimizer_for_score_se3_resume(trainer, checkpoint)
            result = checkpointing.load_checkpoint_state(trainer, checkpoint)
            _assert_restored_score_se3_controller_state(trainer)
            predictor = getattr(model, "dynamics_predictor", None)
            if predictor is not None and hasattr(
                predictor, "_deformation_graph_identity_checked"
            ):
                predictor._deformation_graph_identity_checked = True
            # Legacy cadence fields mirror the restart-safe campaign counter.
            # It includes continued_baseline/metric_loss's matched Phase-A no-op steps, unlike the
            # successful-update-only optimizer_step.
            trainer.curr_run_batch_count = int(
                trainer.score_se3_campaign_step
            ) * int(
                getattr(trainer, "world_size", 1)
            )
            _assert_loaded_score_se3_model_contract(model)
            return result

        if not bool(getattr(trainer.args, "reset_optimizer_on_resume", False)):
            raise RuntimeError(
                "score-SE3 checkpoint_role=init requires "
                "reset_optimizer_on_resume=true"
            )

        expected = set(model.state_dict())
        if current_graph_candidate:
            saved_candidate = _saved_candidate(checkpoint)
            if saved_candidate != "metric_pose":
                raise RuntimeError(
                    "deformation-graph initialization requires a trained metric_pose "
                    f"checkpoint, got {saved_candidate!r}"
                )
            inherited_differences = inherited_e2_argument_differences(
                checkpoint.get("args"), trainer.args
            )
            if inherited_differences:
                raise RuntimeError(
                    "deformation-graph initialization changed inherited metric_pose "
                    f"arguments: {inherited_differences}"
                )
            filtered_state, missing = prepare_deformation_graph_initial_state(
                expected, checkpoint["model"]
            )
        else:
            filtered_state, missing = prepare_score_se3_initial_state(
                expected,
                checkpoint["model"],
            )
        incompatible = model.load_state_dict(filtered_state, strict=False)
        if (
            sorted(incompatible.missing_keys) != missing
            or incompatible.unexpected_keys
        ):
            raise RuntimeError(
                "unexpected score-SE3 checkpoint load result: "
                f"missing={incompatible.missing_keys[:20]} "
                f"unexpected={incompatible.unexpected_keys[:20]}"
            )
        trainer.batch_count = 0
        trainer.optimizer_step = 0
        trainer.score_se3_campaign_step = 0
        trainer.curr_run_batch_count = 0
        trainer.epoch_count = 0.0
        trainer.sample_count = 0
        trainer.best_checkpoint_metric = None
        trainer.best_checkpoint_metric_name = None
        trainer.best_checkpoint_filename = None
        if trainer.rank == 0:
            from utils import _print

            _print(
                "Initialized score-SE3 candidate "
                f"{current_candidate} from base checkpoint; retained "
                f"{len(missing)} audited extension tensors at configured init."
            )
            _dump_init_state_hashes(trainer, model, set(filtered_state))
        _assert_loaded_score_se3_model_contract(model)

    checkpointing.load_checkpoint = load_checkpoint
    checkpointing._score_se3_loader_installed = True
