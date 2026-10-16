from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping

from training.score_se3_eval import score_metric_spec_from_args


CONTRACT_FILENAME = "score_se3_run_contract.json"


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


def _run_git(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _path_is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _unquoted_git_path(value: str) -> str:
    """Decode Git's quoted porcelain path when possible.

    Git uses a C-style quoted string for unusual path names.  JSON's string
    decoder handles the quoting/escaping used for the paths relevant here; if
    Git emits something outside that subset, fail closed by returning the raw
    value so it remains a dirty entry.
    """

    value = value.strip()
    if not value.startswith('"'):
        return value
    try:
        decoded = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return value
    return decoded if isinstance(decoded, str) else value


def _is_allowed_generated_entry(
    entry: str,
    *,
    repo_root: Path,
    allowed_untracked_paths: tuple[Path, ...],
) -> bool:
    # Never hide modifications to tracked files.  Only newly generated,
    # untracked artifacts beneath the exact run directory are exempted.
    if not entry.startswith("?? "):
        return False
    relative = _unquoted_git_path(entry[3:])
    candidate = (repo_root / relative).resolve()
    return any(_path_is_within(candidate, allowed) for allowed in allowed_untracked_paths)


def git_provenance(
    repo_root: str | Path,
    *,
    allowed_untracked_paths: Iterable[str | Path] = (),
) -> dict[str, Any]:
    root = Path(repo_root).expanduser().resolve()
    allowed = tuple(
        resolved
        for path in allowed_untracked_paths
        for resolved in (Path(path).expanduser().resolve(),)
        if resolved != root and _path_is_within(resolved, root)
    )
    try:
        commit = _run_git(root, "rev-parse", "HEAD")
        branch = _run_git(root, "rev-parse", "--abbrev-ref", "HEAD")
        status = _run_git(root, "status", "--porcelain", "--untracked-files=all")
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        return {
            "available": False,
            "repo_root": str(root),
            "error": str(exc),
            "commit": None,
            "branch": None,
            "dirty": None,
            "dirty_entries": [],
        }
    all_entries = [line for line in status.splitlines() if line]
    ignored_generated_entries = [
        line
        for line in all_entries
        if _is_allowed_generated_entry(
            line,
            repo_root=root,
            allowed_untracked_paths=allowed,
        )
    ]
    ignored = set(ignored_generated_entries)
    dirty_entries = [line for line in all_entries if line not in ignored]
    return {
        "available": True,
        "repo_root": str(root),
        "commit": commit,
        "branch": branch,
        "dirty": bool(dirty_entries),
        "dirty_entries": dirty_entries,
    }


def _json_safe(value: Any, *, path: str = "value") -> Any:
    """Return a deterministic JSON representation or reject opaque values."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not (float("-inf") < value < float("inf")):
            raise ValueError(f"{path} contains a non-finite float")
        return value
    if isinstance(value, Path):
        return str(value.expanduser().resolve())
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item, path=f"{path}.{key}")
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [
            _json_safe(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, (set, frozenset)):
        return sorted(_json_safe(item, path=f"{path}[]") for item in value)
    raise TypeError(f"{path} contains non-JSON value {type(value).__name__}")


def _unwrap_model(model: Any) -> Any:
    return model.module if hasattr(model, "module") else model


def _parameter_name_map(model: Any) -> dict[int, str]:
    if model is None or not hasattr(model, "named_parameters"):
        return {}
    return {id(parameter): name for name, parameter in _unwrap_model(model).named_parameters()}


def _optimizer_group_contract(model: Any, optimizer: Any) -> list[dict[str, Any]]:
    if optimizer is None:
        return []
    names = _parameter_name_map(model)
    groups: list[dict[str, Any]] = []
    for index, group in enumerate(getattr(optimizer, "param_groups", ())):
        parameter_names: list[str] = []
        for parameter in group.get("params", ()):
            name = names.get(id(parameter))
            if name is None:
                raise RuntimeError(
                    "optimizer contains a parameter that is not present in model.named_parameters()"
                )
            parameter_names.append(name)
        groups.append(
            {
                "index": index,
                "name": str(group.get("name", group.get("group_name", f"group_{index}"))),
                "parameter_names": parameter_names,
                "lr": float(group.get("lr", 0.0)),
                "initial_lr": float(group.get("initial_lr", group.get("lr", 0.0))),
                "weight_decay": float(group.get("weight_decay", 0.0)),
            }
        )
    return groups


def score_se3_optimizer_runtime_contract(trainer: Any) -> dict[str, Any]:
    """Describe the exact optimizer state topology saved in a checkpoint.

    This payload deliberately records parameter *names* rather than only the
    integer parameter IDs used by ``Optimizer.state_dict``.  It lets resume
    fail before silently attaching moment estimates to the wrong staged group.
    """

    model = getattr(trainer, "model", None)
    optimizer = getattr(trainer, "optimizer", None)
    scheduler = getattr(trainer, "lr_scheduler", None)
    trainable = []
    if model is not None and hasattr(_unwrap_model(model), "named_parameters"):
        trainable = sorted(
            name
            for name, parameter in _unwrap_model(model).named_parameters()
            if bool(parameter.requires_grad)
        )
    return {
        "schema_version": 1,
        "optimizer_step": int(getattr(trainer, "optimizer_step", 0)),
        "campaign_step": int(getattr(trainer, "score_se3_campaign_step", 0)),
        "optimizer_phase": str(getattr(trainer, "optimizer_phase", "single_phase")),
        "optimizer_class": (
            None
            if optimizer is None
            else f"{type(optimizer).__module__}.{type(optimizer).__qualname__}"
        ),
        "scheduler_class": (
            None
            if scheduler is None
            else f"{type(scheduler).__module__}.{type(scheduler).__qualname__}"
        ),
        "trainable_parameter_names": trainable,
        "parameter_groups": _optimizer_group_contract(model, optimizer),
    }


def _configured_optimizer_contract(args: Any, trainer: Any | None) -> dict[str, Any]:
    plan = getattr(trainer, "score_se3_optimizer_plan", None) if trainer is not None else None
    if plan is None:
        plan = getattr(args, "score_se3_optimizer_plan", None)

    if trainer is not None:
        runtime = score_se3_optimizer_runtime_contract(trainer)
        trainable_names = runtime["trainable_parameter_names"]
        runtime_groups = runtime["parameter_groups"]
        optimizer_class = runtime["optimizer_class"]
    else:
        trainable_names = sorted(
            str(name)
            for name in (getattr(args, "score_se3_trainable_parameter_names", None) or [])
        )
        runtime_groups = _json_safe(
            getattr(args, "score_se3_optimizer_groups", None) or [],
            path="score_se3_optimizer_groups",
        )
        optimizer_class = str(getattr(args, "score_se3_optimizer_class", "torch.optim.AdamW"))

    initial_phase = str(
        getattr(args, "score_se3_initial_optimizer_phase", "phase_a")
    )
    if isinstance(plan, Mapping) and plan.get("initial_phase") is not None:
        initial_phase = str(plan["initial_phase"])
    if isinstance(plan, Mapping):
        optimizer_class = str(plan.get("optimizer_class", optimizer_class))
        phases = plan.get("phases")
        phase_contract = None
        if isinstance(phases, Mapping):
            phase_contract = phases.get(initial_phase)
        elif isinstance(phases, (list, tuple)):
            phase_contract = next(
                (
                    item
                    for item in phases
                    if isinstance(item, Mapping)
                    and str(item.get("name", item.get("phase", ""))) == initial_phase
                ),
                None,
            )
        if isinstance(phase_contract, Mapping):
            planned_groups = phase_contract.get(
                "parameter_groups", phase_contract.get("groups")
            )
            if planned_groups is not None:
                runtime_groups = _json_safe(
                    planned_groups,
                    path=f"score_se3_optimizer_plan.phases.{initial_phase}.groups",
                )

        planned_names = plan.get("trainable_parameter_names")
        if planned_names is not None:
            trainable_names = sorted(str(name) for name in planned_names)
    return {
        "class": optimizer_class,
        "initial_phase": initial_phase,
        "trainable_parameter_names": trainable_names,
        "initial_parameter_groups": runtime_groups,
        "phase_plan": _json_safe(plan, path="score_se3_optimizer_plan"),
    }


def _evaluation_only_optimizer_contract(args: Any, trainer: Any | None) -> dict[str, Any]:
    saved_plan = (
        getattr(trainer, "score_se3_optimizer_plan", None)
        if trainer is not None
        else getattr(args, "score_se3_optimizer_plan", None)
    )
    checkpoint_role = str(getattr(args, "score_se3_checkpoint_role", "init"))
    return {
        "class": None,
        "initial_phase": None,
        "trainable_parameter_names": [],
        "initial_parameter_groups": [],
        "phase_plan": _json_safe(saved_plan, path="score_se3_optimizer_plan"),
        "optimizer_constructed": False,
        "optimizer_restored": False,
        "checkpoint_training_counters_restored": checkpoint_role == "eval",
    }


def _file_contract(
    path: str | Path | None,
    *,
    role: str,
    required: bool,
) -> dict[str, Any] | None:
    if path is None or str(path).strip() == "":
        if required:
            raise ValueError(f"score-SE3 strict contract requires {role}")
        return None
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        if required:
            raise FileNotFoundError(f"score-SE3 {role} does not exist: {resolved}")
        return {"path": str(resolved), "exists": False, "sha256": None}
    return {
        "path": str(resolved),
        "exists": True,
        "size_bytes": int(resolved.stat().st_size),
        "sha256": sha256_file(resolved),
    }


def _selected_environment() -> dict[str, str]:
    prefixes = ("SCORE_SE3_", "LONG_HORIZON_")
    exact = {
        "LR_SCHEDULE",
        "LR_SCHEDULE_TOTAL_STEPS",
        "WORLD_SIZE",
    }
    return {
        key: value
        for key, value in sorted(os.environ.items())
        if key.startswith(prefixes) or key in exact
    }


def _metric_loss_calibration_contract(
    args: Any,
    *,
    candidate: str,
    strict: bool,
) -> dict[str, Any]:
    """Describe either the immutable calibration input or its destination.

    Producer contracts intentionally do not inspect destination existence: the
    contract is written before calibration, and an idempotent rerun must remain
    byte-identical after the artifact has been created.
    """

    from training.score_se3_calibration import (
        CALIBRATION_FORMULA_VERSION,
        MAX_CALIBRATION_BATCHES,
        MIN_CALIBRATION_BATCHES,
    )

    calibration_only = bool(
        getattr(args, "score_se3_calibration_only", False)
    )
    required = bool(
        getattr(args, "score_se3_require_calibration_artifact", False)
    )
    path_value = getattr(args, "score_se3_calibration_artifact", None)
    batch_count = int(getattr(args, "score_se3_calibration_batches", 16))
    if not MIN_CALIBRATION_BATCHES <= batch_count <= MAX_CALIBRATION_BATCHES:
        raise ValueError(
            "score-SE3 calibration batch count must be in "
            f"[{MIN_CALIBRATION_BATCHES},{MAX_CALIBRATION_BATCHES}]"
        )
    if candidate == "continued_baseline":
        if calibration_only or required or path_value:
            raise ValueError("continued_baseline forbids metric-loss calibration artifacts")
        return {
            "mode": "forbidden_control",
            "required": False,
            "calibration_only": False,
            "path": None,
            "exists": None,
            "size_bytes": None,
            "sha256": None,
            "formula_version": CALIBRATION_FORMULA_VERSION,
            "sample_sequence_sha256": None,
            "warm_checkpoint_sha256": None,
            "batch_count": batch_count,
        }
    if calibration_only:
        if candidate != "metric_loss" or not path_value or required:
            raise ValueError(
                "calibration-only contracts require metric_loss, an artifact destination, "
                "and require_calibration_artifact=false"
            )
        return {
            "mode": "no_update_producer",
            "required": False,
            "calibration_only": True,
            "path": str(Path(path_value).expanduser().resolve()),
            "exists": None,
            "size_bytes": None,
            "sha256": None,
            "formula_version": CALIBRATION_FORMULA_VERSION,
            "sample_sequence_sha256": None,
            "warm_checkpoint_sha256": None,
            "batch_count": batch_count,
        }
    if strict and not required:
        raise ValueError(
            f"strict {candidate} requires a metric-loss calibration artifact"
        )
    if required and not path_value:
        raise ValueError(
            f"{candidate} requires a metric-loss calibration artifact path"
        )
    if not path_value:
        return {
            "mode": "development_explicit_weight",
            "required": False,
            "calibration_only": False,
            "path": None,
            "exists": None,
            "size_bytes": None,
            "sha256": None,
            "formula_version": None,
            "sample_sequence_sha256": None,
            "warm_checkpoint_sha256": None,
            "batch_count": batch_count,
        }

    artifact_file = _file_contract(
        path_value,
        role="metric-loss calibration artifact",
        required=required or strict,
    )
    if artifact_file is None:
        raise RuntimeError("calibration artifact file contract unexpectedly absent")
    observed_sha = artifact_file.get("sha256")
    expected_sha = getattr(
        args, "score_se3_calibration_artifact_sha256", None
    )
    if observed_sha != expected_sha:
        raise RuntimeError(
            "metric-loss calibration artifact hash disagrees with the validated "
            f"CLI state: file={observed_sha!r}, args={expected_sha!r}"
        )
    formula_version = getattr(
        args, "score_se3_calibration_formula_version", None
    )
    if formula_version != CALIBRATION_FORMULA_VERSION:
        raise RuntimeError("metric-loss calibration formula version is not validated")
    return {
        "mode": "immutable_consumer",
        "required": required,
        "calibration_only": False,
        **artifact_file,
        "formula_version": formula_version,
        "sample_sequence_sha256": getattr(
            args, "score_se3_calibration_sample_sequence_sha256", None
        ),
        "warm_checkpoint_sha256": getattr(
            args, "score_se3_calibration_warm_checkpoint_sha256", None
        ),
        "batch_count": batch_count,
    }
def build_score_se3_contract(
    args: Any,
    *,
    repo_root: str | Path,
    trainer: Any | None = None,
    initialization_checkpoint_override: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not bool(getattr(args, "score_se3_eval", False)):
        raise ValueError("score-SE3 contract requires score_se3_eval=true")
    candidate = str(getattr(args, "score_se3_candidate", ""))
    if candidate not in {"continued_baseline", "metric_loss", "metric_pose", "predicted_contact", "oracle_contact"}:
        raise ValueError(f"invalid score-SE3 candidate {candidate!r}")
    oracle_authorized = bool(getattr(args, "score_se3_allow_oracle", False))
    if candidate == "oracle_contact" and not oracle_authorized:
        raise PermissionError("oracle_contact contract requires explicit oracle authorization")
    if candidate != "oracle_contact" and oracle_authorized:
        raise ValueError("oracle authorization is forbidden for continued_baseline-predicted_contact contracts")
    graph_candidate = str(
        getattr(args, "deformation_graph_candidate", "") or ""
    )
    if graph_candidate and graph_candidate not in {"dense_decoder", "deformation_graph", "gated_deformation_graph", "larger_deformation_graph"}:
        raise ValueError(f"invalid deformation-graph candidate {graph_candidate!r}")
    if graph_candidate and candidate != "metric_pose":
        raise ValueError("deformation-graph contracts must inherit score-SE3 metric_pose")
    if graph_candidate and int(getattr(args, "seed", -1)) != 2026:
        raise ValueError("the frozen deformation-graph campaign uses only seed 2026")
    strict = bool(getattr(args, "score_se3_strict_contract", False))
    calibration_only = bool(
        getattr(args, "score_se3_calibration_only", False)
    )
    output_dir = getattr(args, "output_dir", None)
    provenance = git_provenance(
        repo_root,
        # Trainer/model setup legitimately creates run metadata before the
        # final runtime-aware contract is written.  Exempt only untracked
        # files under that exact directory; tracked modifications still fail.
        allowed_untracked_paths=(() if not output_dir else (output_dir,)),
    )
    if strict:
        if not provenance["available"]:
            raise RuntimeError("score-SE3 strict contract could not resolve git provenance")
        if provenance["dirty"]:
            raise RuntimeError(
                "score-SE3 strict contract refuses a dirty checkout: "
                f"{provenance['dirty_entries'][:10]}"
            )

    spec = score_metric_spec_from_args(args)
    if int(getattr(args, "score_se3_eval_data_seed", 2026)) < 0:
        raise ValueError("score-SE3 eval data seed must be non-negative")
    if strict and not math.isclose(
        float(spec.ever_moved_threshold_m),
        0.005,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "score-SE3 strict contract requires the frozen 0.005 m "
            "ever-moved threshold"
        )
    manifest = _file_contract(
        getattr(args, "score_se3_manifest_path", None),
        role="dataset manifest",
        required=strict,
    )
    checkpoint = (
        _json_safe(
            initialization_checkpoint_override,
            path="initialization_checkpoint_override",
        )
        if initialization_checkpoint_override is not None
        else _file_contract(
            getattr(args, "model_path", None),
            role="initialization checkpoint",
            required=strict,
        )
    )
    calibration = _metric_loss_calibration_contract(
        args,
        candidate=candidate,
        # dense_decoder-larger_deformation_graph continue the already calibrated metric_pose checkpoint and preserve its
        # exact serialized score-loss coefficient.  Recalibrating at the metric_pose
        # endpoint would change the objective and confound the decoder test.
        strict=strict and not bool(graph_candidate),
    )
    if (
        graph_candidate
        and calibration.get("mode") == "development_explicit_weight"
    ):
        calibration = {
            **calibration,
            "mode": "inherited_selected_e2_weight",
            "configured_weight": float(
                getattr(args, "score_se3_loss_weight", 0.0)
            ),
            "source_checkpoint_sha256": (
                None if checkpoint is None else checkpoint.get("sha256")
            ),
        }
    if strict and not calibration_only and not bool(
        getattr(args, "score_se3_require_full_eval", False)
    ):
        raise ValueError("score-SE3 strict contract requires full unique-window eval")
    if strict and not calibration_only and not bool(
        getattr(args, "score_se3_export_per_window", False)
    ):
        raise ValueError("score-SE3 strict contract requires per-window exports")

    campaign_step_limit = int(getattr(args, "max_train_steps", -1))
    evaluation_only = bool(getattr(args, "score_se3_eval_only", False))
    checkpoint_role = str(getattr(args, "score_se3_checkpoint_role", "init"))
    if checkpoint_role not in {"init", "resume", "eval", "finetune"}:
        raise ValueError(
            "score_se3_checkpoint_role must be init, resume, eval, or finetune, "
            f"got {checkpoint_role!r}"
        )
    optimizer_contract = (
        _evaluation_only_optimizer_contract(args, trainer)
        if evaluation_only
        else _configured_optimizer_contract(args, trainer)
    )
    split_role = str(getattr(args, "score_se3_split_role", "validation"))
    if split_role not in {"validation", "final_test"}:
        raise ValueError(
            "score_se3_split_role must be validation or final_test, "
            f"got {split_role!r}"
        )
    if calibration_only and evaluation_only:
        raise ValueError("score-SE3 calibration-only and evaluation-only are disjoint")
    if calibration_only:
        if candidate != "metric_loss" or checkpoint_role != "init" or split_role != "validation":
            raise ValueError(
                "score-SE3 calibration-only requires metric_loss/init/validation"
            )
        if float(getattr(args, "score_se3_loss_weight", 0.0)) != 0.0 or float(
            getattr(args, "score_se3_gradient_ratio", 0.0)
        ) != 0.0:
            raise ValueError(
                "score-SE3 calibration-only requires zero configured loss weights"
            )
        if bool(getattr(args, "distributed", False)) or int(
            os.environ.get("WORLD_SIZE", "1")
        ) != 1:
            raise ValueError("score-SE3 calibration-only must be single-process")
        if not bool(getattr(args, "deterministic_train", False)) or not bool(
            getattr(args, "deterministic_algorithms", False)
        ) or not bool(
            getattr(args, "full_train_coverage", False)
        ) or not bool(
            getattr(args, "eval_test_only", False)
        ):
            raise ValueError(
                "score-SE3 calibration-only requires deterministic full train coverage"
            )
        if int(getattr(args, "num_workers", -1)) != 0:
            raise ValueError("score-SE3 calibration-only requires num_workers=0")
    if evaluation_only:
        if checkpoint_role == "resume":
            raise ValueError(
                "score-SE3 evaluation-only runs cannot use checkpoint_role=resume"
            )
        expected_split = "final_test" if checkpoint_role == "eval" else "validation"
        if split_role != expected_split:
            raise ValueError(
                f"checkpoint_role={checkpoint_role} evaluation requires "
                f"split_role={expected_split}"
            )
    elif checkpoint_role == "eval" or split_role == "final_test":
        raise ValueError(
            "score-SE3 final-test checkpoint evaluation must be evaluation-only"
        )
    if strict and not evaluation_only:
        restart_safe = {
            "deterministic_train": bool(
                getattr(args, "deterministic_train", False)
            ),
            "deterministic_algorithms": bool(
                getattr(args, "deterministic_algorithms", False)
            ),
            "full_train_coverage": bool(
                getattr(args, "full_train_coverage", False)
            ),
            "eval_test_only": bool(getattr(args, "eval_test_only", False)),
        }
        disabled = sorted(
            name for name, enabled in restart_safe.items() if not enabled
        )
        if disabled or int(getattr(args, "num_workers", -1)) != 0:
            raise ValueError(
                "score-SE3 strict contract requires restart-safe training data "
                f"flow: disabled={disabled}, num_workers="
                f"{getattr(args, 'num_workers', None)!r}"
            )
    if strict and not calibration_only:
        checkpoint_metric = str(getattr(args, "checkpoint_metric", ""))
        checkpoint_mode = str(getattr(args, "checkpoint_metric_mode", "")).lower()
        if checkpoint_metric != "test/score_se3/composite":
            raise ValueError(
                "score-SE3 strict contract requires checkpoint_metric="
                "test/score_se3/composite"
            )
        if checkpoint_mode != "min":
            raise ValueError("score-SE3 strict contract requires checkpoint_metric_mode=min")
        if not evaluation_only:
            if int(getattr(args, "eval_freq", -1)) <= 0:
                raise ValueError("score-SE3 strict contract requires a positive eval cadence")
            if int(getattr(args, "save_freq", -1)) <= 0:
                raise ValueError("score-SE3 strict contract requires a positive save cadence")
            if campaign_step_limit <= 0:
                raise ValueError(
                    "score-SE3 strict contract requires a positive campaign step budget"
                )
            for cadence_name in ("eval_freq", "save_freq"):
                cadence = int(getattr(args, cadence_name))
                if campaign_step_limit % cadence != 0:
                    raise ValueError(
                        "score-SE3 strict contract requires the final campaign step "
                        f"to land on {cadence_name}: budget={campaign_step_limit}, "
                        f"cadence={cadence}"
                    )
            if not bool(getattr(args, "save_best_only", False)):
                raise ValueError("score-SE3 strict contract requires save_best_only=true")
            gradient_ratio = float(getattr(args, "score_se3_gradient_ratio", 0.0))
            distributed = bool(getattr(args, "distributed", False)) or int(
                os.environ.get("WORLD_SIZE", "1")
            ) > 1
            if distributed and gradient_ratio != 0.0:
                raise ValueError(
                    "score-SE3 distributed production runs require "
                    "score_se3_gradient_ratio=0; autograd.grad calibration is not "
                    "supported through DistributedDataParallel"
                )
        if trainer is not None and not evaluation_only:
            if optimizer_contract["phase_plan"] is None:
                raise ValueError(
                    "score-SE3 strict contract requires trainer.score_se3_optimizer_plan"
                )
            if not optimizer_contract["trainable_parameter_names"]:
                raise ValueError(
                    "score-SE3 strict contract requires audited trainable parameter names"
                )
            if not optimizer_contract["initial_parameter_groups"]:
                raise ValueError(
                    "score-SE3 strict contract requires explicit optimizer parameter groups"
                )
            if int(getattr(trainer, "score_se3_campaign_step", 0)) == 0 and str(
                getattr(trainer, "optimizer_phase", "")
            ) != str(optimizer_contract["initial_phase"]):
                raise ValueError(
                    "score-SE3 trainer phase disagrees with optimizer plan at step zero: "
                    f"trainer={getattr(trainer, 'optimizer_phase', None)!r}, "
                    f"plan={optimizer_contract['initial_phase']!r}"
                )
    contract = {
        "schema_version": 2,
        "campaign": (
            "lh35_canonical_deformation_graph"
            if graph_candidate
            else "lh35_metric_aligned_residual_se3"
        ),
        "candidate": candidate,
        "oracle": candidate == "oracle_contact",
        "deployable": candidate != "oracle_contact",
        "git": provenance,
        "files": {
            "root_checkpoint_role": (
                "init" if checkpoint_role == "resume" else checkpoint_role
            ),
            "initialization_checkpoint": checkpoint,
            "dataset_manifest": manifest,
            "metric_loss_calibration": calibration,
        },
        "data": {
            "train_dirs": list(getattr(args, "hoi_train_dirs", None) or []),
            "validation_dirs": list(getattr(args, "hoi_val_dirs", None) or []),
            "dataset_type": str(getattr(args, "dataset_type", "")),
            "horizon": int(getattr(args, "hoi_target_horizon", -1)),
            "full_train_coverage": bool(
                getattr(args, "full_train_coverage", False)
            ),
            "eval_test_only": bool(getattr(args, "eval_test_only", False)),
            "train_num_workers": int(getattr(args, "num_workers", -1)),
        },
        "evaluation": {
            "split_role": split_role,
            "data_seed": int(
                getattr(args, "score_se3_eval_data_seed", 2026)
            ),
            "metric_spec": {
                "ever_moved_threshold_m": spec.ever_moved_threshold_m,
                "smooth_epsilon_m": spec.smooth_epsilon_m,
                "early_end_index": spec.early_end_index,
                "final_index": spec.final_index,
                "component_weights": spec.component_weights,
            },
            "checkpoint_metric": str(getattr(args, "checkpoint_metric", "")),
            "checkpoint_metric_mode": str(
                getattr(args, "checkpoint_metric_mode", "min")
            ),
            "save_best_only": bool(getattr(args, "save_best_only", False)),
            "eval_every_campaign_steps": int(getattr(args, "eval_freq", -1)),
            "save_every_campaign_steps": int(getattr(args, "save_freq", -1)),
            "num_eval_batches_per_rank": int(
                getattr(args, "num_eval_batches", -1)
            ),
            "require_full_unique_windows": bool(
                getattr(args, "score_se3_require_full_eval", False)
            ),
            "export_per_window": bool(
                getattr(args, "score_se3_export_per_window", False)
            ),
        },
        "optimization": {
            "execution_mode": (
                "metric_loss_calibration_only"
                if calibration_only
                else ("evaluation_only" if evaluation_only else "training")
            ),
            "seed": int(getattr(args, "seed", 0)),
            "batch_size_per_rank": int(getattr(args, "batch_size", -1)),
            "world_size_env": int(os.environ.get("WORLD_SIZE", "1")),
            "base_lr": float(getattr(args, "base_lr", 0.0)),
            "weight_decay": float(getattr(args, "weight_decay", 0.0)),
            # Evaluation-only processes execute no training loop.  Preserve
            # the legacy parser value for provenance, but never describe -1
            # (its "unset" sentinel) as a negative campaign budget.
            "raw_max_train_steps": campaign_step_limit,
            "max_train_steps": (
                0 if evaluation_only or calibration_only else campaign_step_limit
            ),
            "campaign_step_budget": (
                0 if evaluation_only or calibration_only else campaign_step_limit
            ),
            "counter_semantics": {
                "campaign_step": (
                    "synchronized consumed training batches, including matched no-op"
                ),
                "optimizer_step": "successful optimizer updates only",
                "cadence_counter": "campaign_step",
            },
            "reset_optimizer_on_resume": bool(
                getattr(args, "reset_optimizer_on_resume", False)
            ),
            "allow_optimizer_reset": bool(
                getattr(args, "allow_optimizer_reset", False)
            ),
            "lr_schedule": os.environ.get("LR_SCHEDULE", "none"),
            "lr_schedule_total_steps": int(
                os.environ.get(
                    "LR_SCHEDULE_TOTAL_STEPS",
                    str(max(campaign_step_limit, 0)),
                )
            ),
            "metric_aligned_objective": {
                "active": not evaluation_only
                and not calibration_only
                and float(getattr(args, "score_se3_loss_weight", 0.0)) != 0.0,
                "configured_weight": float(
                    getattr(args, "score_se3_loss_weight", 0.0)
                ),
                "gradient_ratio": float(
                    getattr(args, "score_se3_gradient_ratio", 0.0)
                ),
                "max_calibrated_weight": float(
                    getattr(args, "score_se3_max_calibrated_weight", 100.0)
                ),
                "variance_independent": True,
                "calibration": calibration,
            },
            "contact_objective": {
                "active": not evaluation_only
                and candidate == "predicted_contact"
                and float(getattr(args, "score_se3_contact_loss_weight", 0.1))
                != 0.0,
                "weight": float(
                    getattr(args, "score_se3_contact_loss_weight", 0.1)
                ),
                "predicted_contact_stop_gradient": bool(
                    getattr(args, "score_se3_contact_stop_gradient", True)
                ),
                "tau_m": float(
                    os.environ.get(
                        "LONG_HORIZON_CONTACT_TAU_M",
                        getattr(args, "score_se3_contact_tau_m", 0.02),
                    )
                ),
                "sampled_object_points": int(
                    os.environ.get("LONG_HORIZON_CONTACT_POINTS", "128")
                ),
                "near_contact_weight": float(
                    os.environ.get("LONG_HORIZON_CONTACT_NEAR_WEIGHT", "4.0")
                ),
            },
            "current_objective": {
                name: _json_safe(getattr(args, name, None), path=name)
                for name in (
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
            },
            "numerics": {
                "autocast_dtype": (
                    None
                    if trainer is None or getattr(trainer, "amp_dtype", None) is None
                    else str(getattr(trainer, "amp_dtype"))
                ),
                "grad_clip_max_norm": float(
                    getattr(args, "grad_clip_max_norm", 5.0)
                ),
                "deterministic_train": bool(
                    getattr(args, "deterministic_train", False)
                ),
                "deterministic_algorithms": bool(
                    getattr(args, "deterministic_algorithms", False)
                ),
                "full_train_coverage": bool(
                    getattr(args, "full_train_coverage", False)
                ),
                "eval_test_only": bool(
                    getattr(args, "eval_test_only", False)
                ),
                "train_num_workers": int(getattr(args, "num_workers", -1)),
            },
            "optimizer": optimizer_contract,
        },
        "model": {
            "dynamics_experiment": str(
                getattr(args, "dynamics_experiment", "")
            ),
            "action_conditioning_mode": str(
                getattr(args, "action_conditioning_mode", "")
            ),
            "hand_condition_encoder": str(
                getattr(args, "hand_condition_encoder", "")
            ),
            "hand_condition_fusion": str(
                getattr(args, "hand_condition_fusion", "")
            ),
            "hand_action_token_encoder": str(
                getattr(args, "hand_action_token_encoder", "")
            ),
            "long_horizon_output_mode": os.environ.get(
                "LONG_HORIZON_OUTPUT_MODE"
            ),
            "long_horizon_pose_head_init": os.environ.get(
                "LONG_HORIZON_POSE_HEAD_INIT"
            ),
            "residual_se3": {
                "enabled": candidate in {"metric_pose", "predicted_contact", "oracle_contact"},
                "translation_bound_m": float(
                    getattr(args, "score_se3_translation_bound_m", 0.05)
                ),
                "rotation_bound_deg": float(
                    getattr(args, "score_se3_rotation_bound_deg", 15.0)
                ),
                "contact_tau_m": float(
                    getattr(args, "score_se3_contact_tau_m", 0.02)
                ),
                "contact_conditioning": {
                    "continued_baseline": "none",
                    "metric_loss": "none",
                    "metric_pose": "literal_zero_channels",
                    "predicted_contact": "predicted_continuous_contact",
                    "oracle_contact": "future_gt_contact_oracle",
                }[candidate],
            },
            "deformation_graph": {
                "enabled": graph_candidate in {"deformation_graph", "gated_deformation_graph", "larger_deformation_graph"},
                "candidate": graph_candidate or None,
                "node_count": int(
                    getattr(args, "deformation_graph_nodes", 0) or 0
                ),
                "influence_count": int(
                    getattr(args, "deformation_graph_influences", 0) or 0
                ),
                "edge_count": int(
                    getattr(args, "deformation_graph_edges", 0) or 0
                ),
                "radius_scale": float(
                    getattr(args, "deformation_graph_radius_scale", 0.0) or 0.0
                ),
                "minimum_radius_m": float(
                    getattr(args, "deformation_graph_minimum_radius_m", 0.0)
                    or 0.0
                ),
                "translation_bound_m": float(
                    getattr(
                        args, "deformation_graph_translation_bound_m", 0.0
                    )
                    or 0.0
                ),
                "rotation_bound_deg": float(
                    getattr(args, "deformation_graph_rotation_bound_deg", 0.0)
                    or 0.0
                ),
                "learned_deformability_gate": graph_candidate == "gated_deformation_graph",
                "final_pointwise_bypass": graph_candidate in {"", "dense_decoder"},
                "loss_weights": {
                    "arap": float(
                        getattr(args, "deformation_graph_arap_weight", 0.0)
                        or 0.0
                    ),
                    "temporal": float(
                        getattr(args, "deformation_graph_temporal_weight", 0.0)
                        or 0.0
                    ),
                    "magnitude": float(
                        getattr(
                            args, "deformation_graph_magnitude_weight", 0.0
                        )
                        or 0.0
                    ),
                    "gate": float(
                        getattr(args, "deformation_graph_gate_weight", 0.0)
                        or 0.0
                    ),
                },
            },
        },
        "inference_contract": {
            "future_object_ground_truth_is_input": candidate == "oracle_contact",
            "oracle_explicitly_authorized": bool(
                oracle_authorized
            ),
            "future_object_ground_truth_role": (
                "oracle_contact_features_only"
                if candidate == "oracle_contact"
                else "training_targets_only"
            ),
            "export_or_deployment_allowed": candidate != "oracle_contact",
        },
        "environment": _selected_environment(),
    }
    return contract


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def write_score_se3_contract(
    args: Any,
    *,
    repo_root: str | Path,
    trainer: Any | None = None,
) -> tuple[Path, dict[str, Any]]:
    output_dir = getattr(args, "output_dir", None)
    if not output_dir:
        raise ValueError("score-SE3 contract requires args.output_dir")
    path = Path(output_dir).expanduser() / CONTRACT_FILENAME
    existing = None
    existing_contract = None
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        existing_contract = json.loads(existing)
        if not isinstance(existing_contract, dict):
            raise RuntimeError(f"score-SE3 run contract is not a JSON object: {path}")

    role = str(getattr(args, "score_se3_checkpoint_role", "init"))
    if role == "resume" and existing_contract is None:
        raise RuntimeError(
            "score-SE3 resume requires the immutable run contract created by "
            f"the initialization launch: {path}"
        )
    initialization_override = None
    if role == "resume" and existing_contract is not None:
        try:
            initialization_override = existing_contract["files"][
                "initialization_checkpoint"
            ]
        except (KeyError, TypeError) as exc:
            raise RuntimeError(
                f"cannot resume from malformed score-SE3 run contract: {path}"
            ) from exc
    contract = build_score_se3_contract(
        args,
        repo_root=repo_root,
        trainer=trainer,
        initialization_checkpoint_override=initialization_override,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if existing is not None:

        # A resume launch points model_path at model-latest.pt and necessarily
        # flips reset_optimizer_on_resume.  Those are resume mechanics, not a
        # change to the immutable run that began from the recorded SOTA
        # initialization checkpoint.  Preserve the original root fields while
        # checking every causal training/evaluation setting below them.
        if role == "resume":
            try:
                contract["optimization"]["reset_optimizer_on_resume"] = (
                    existing_contract["optimization"]["reset_optimizer_on_resume"]
                )
            except (KeyError, TypeError) as exc:
                raise RuntimeError(
                    f"cannot resume from malformed score-SE3 run contract: {path}"
                ) from exc

        serialized = _canonical_json(contract)
        if existing != serialized:
            raise RuntimeError(
                "score-SE3 run contract changed inside an existing output directory: "
                f"{path}"
            )
        return path, contract
    serialized = _canonical_json(contract)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(serialized, encoding="utf-8")
    os.replace(temporary, path)
    return path, contract
