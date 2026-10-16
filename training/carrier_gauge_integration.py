"""Entrypoint and fail-closed runtime wiring for carrier-gauge campaigns."""
from __future__ import annotations

import argparse
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Sequence

from training.carrier_gauge_contract import CARRIER_GAUGE_CAUSAL_ARGS


CANDIDATES = ("unfixed_graph", "gauge_fixed_graph", "low_rank_graph", "hand_relational_graph", "dexplicit_3d")


@dataclass(frozen=True)
class CarrierGaugeOptions:
    candidate: str
    mode_rank: int
    temporal_controls: int
    relation_temperature_m: float
    carrier_translation_bound_m: float
    carrier_rotation_bound_deg: float
    edge_strain_weight: float
    coefficient_weight: float
    carrier_tv_weight: float
    carrier_strength_weight: float
    deformable_threshold_m: float

    @property
    def graph_candidate(self) -> str:
        return "deformation_graph" if self.candidate in {"unfixed_graph", "gauge_fixed_graph"} else "gated_deformation_graph"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--carrier-gauge-candidate",
        choices=CANDIDATES,
        default=os.environ.get("CARRIER_GAUGE_CANDIDATE"),
    )
    parser.add_argument("--carrier-gauge-mode-rank", type=int, default=4)
    parser.add_argument("--carrier-gauge-temporal-controls", type=int, default=6)
    parser.add_argument(
        "--carrier-gauge-relation-temperature-m", type=float, default=0.04
    )
    parser.add_argument(
        "--carrier-gauge-carrier-translation-bound-m", type=float, default=0.05
    )
    parser.add_argument(
        "--carrier-gauge-carrier-rotation-bound-deg", type=float, default=15.0
    )
    parser.add_argument(
        "--carrier-gauge-edge-strain-weight", type=float, default=0.5
    )
    parser.add_argument(
        "--carrier-gauge-coefficient-weight", type=float, default=1e-4
    )
    parser.add_argument(
        "--carrier-gauge-carrier-tv-weight", type=float, default=1e-3
    )
    parser.add_argument(
        "--carrier-gauge-carrier-strength-weight", type=float, default=1e-4
    )
    parser.add_argument(
        "--carrier-gauge-deformable-threshold-m", type=float, default=0.005
    )
    return parser


def parse_carrier_gauge_options(
    argv: Sequence[str],
) -> tuple[CarrierGaugeOptions, list[str]]:
    parser = _build_parser()
    if any(token in {"-h", "--help"} for token in argv):
        print("Carrier-conditioned gauge-fixed deformation-graph options:")
        print(parser.format_help())
    namespace, remaining = parser.parse_known_args(list(argv))
    if not namespace.carrier_gauge_candidate:
        parser.error(
            "--carrier-gauge-candidate unfixed_graph|gauge_fixed_graph|low_rank_graph|hand_relational_graph|dexplicit_3d is required"
        )
    options = CarrierGaugeOptions(
        candidate=str(namespace.carrier_gauge_candidate),
        mode_rank=int(namespace.carrier_gauge_mode_rank),
        temporal_controls=int(namespace.carrier_gauge_temporal_controls),
        relation_temperature_m=float(
            namespace.carrier_gauge_relation_temperature_m
        ),
        carrier_translation_bound_m=float(
            namespace.carrier_gauge_carrier_translation_bound_m
        ),
        carrier_rotation_bound_deg=float(
            namespace.carrier_gauge_carrier_rotation_bound_deg
        ),
        edge_strain_weight=float(namespace.carrier_gauge_edge_strain_weight),
        coefficient_weight=float(namespace.carrier_gauge_coefficient_weight),
        carrier_tv_weight=float(namespace.carrier_gauge_carrier_tv_weight),
        carrier_strength_weight=float(
            namespace.carrier_gauge_carrier_strength_weight
        ),
        deformable_threshold_m=float(
            namespace.carrier_gauge_deformable_threshold_m
        ),
    )
    _validate(options)
    return options, remaining


def _validate(options: CarrierGaugeOptions) -> None:
    import math

    if options.mode_rank <= 0:
        raise ValueError("carrier-gauge mode rank must be positive")
    if options.temporal_controls < 4:
        raise ValueError("carrier-gauge temporal controls must be at least four")
    positive = {
        "relation_temperature_m": options.relation_temperature_m,
        "carrier_translation_bound_m": options.carrier_translation_bound_m,
        "carrier_rotation_bound_deg": options.carrier_rotation_bound_deg,
        "deformable_threshold_m": options.deformable_threshold_m,
    }
    nonnegative = {
        "edge_strain_weight": options.edge_strain_weight,
        "coefficient_weight": options.coefficient_weight,
        "carrier_tv_weight": options.carrier_tv_weight,
        "carrier_strength_weight": options.carrier_strength_weight,
    }
    for name, value in positive.items():
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    for name, value in nonnegative.items():
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")


def _option_value(argv: Sequence[str], name: str) -> str | None:
    for index, token in enumerate(argv):
        if token == name:
            if index + 1 >= len(argv):
                raise ValueError(f"{name} requires a value")
            return str(argv[index + 1])
        if token.startswith(name + "="):
            return token.split("=", 1)[1]
    return None


def prepare_base_argv(
    options: CarrierGaugeOptions, remaining: Sequence[str]
) -> list[str]:
    """Inject the exact inherited metric_pose/graph identity, rejecting contradictions."""
    argv = list(remaining)
    if "--no-score-se3-eval" in argv:
        raise ValueError("carrier-gauge campaigns require --score-se3-eval")
    if "--score-se3-eval" not in argv:
        argv.append("--score-se3-eval")

    score_candidate = _option_value(argv, "--score-se3-candidate")
    if score_candidate is None:
        argv.extend(("--score-se3-candidate", "metric_pose"))
    elif score_candidate != "metric_pose":
        raise ValueError(
            "carrier-gauge campaigns must inherit --score-se3-candidate metric_pose"
        )

    graph_candidate = _option_value(argv, "--deformation-graph-candidate")
    if graph_candidate is None:
        argv.extend(
            ("--deformation-graph-candidate", options.graph_candidate)
        )
    elif graph_candidate != options.graph_candidate:
        raise ValueError(
            f"{options.candidate} requires --deformation-graph-candidate "
            f"{options.graph_candidate}, got {graph_candidate}"
        )
    return argv


def export_environment(options: CarrierGaugeOptions) -> None:
    values = {
        "CARRIER_GAUGE_CANDIDATE": options.candidate,
        "CARRIER_GAUGE_MODE_RANK": options.mode_rank,
        "CARRIER_GAUGE_TEMPORAL_CONTROLS": options.temporal_controls,
        "CARRIER_GAUGE_RELATION_TEMPERATURE_M": options.relation_temperature_m,
        "CARRIER_GAUGE_CARRIER_TRANSLATION_BOUND_M": (
            options.carrier_translation_bound_m
        ),
        "CARRIER_GAUGE_CARRIER_ROTATION_BOUND_DEG": (
            options.carrier_rotation_bound_deg
        ),
        "CARRIER_GAUGE_EDGE_STRAIN_WEIGHT": options.edge_strain_weight,
        "CARRIER_GAUGE_COEFFICIENT_WEIGHT": options.coefficient_weight,
        "CARRIER_GAUGE_CARRIER_TV_WEIGHT": options.carrier_tv_weight,
        "CARRIER_GAUGE_CARRIER_STRENGTH_WEIGHT": (
            options.carrier_strength_weight
        ),
        "CARRIER_GAUGE_DEFORMABLE_THRESHOLD_M": (
            options.deformable_threshold_m
        ),
    }
    for name, value in values.items():
        os.environ[name] = str(value)


def _extend_tuple(module: Any, name: str) -> None:
    existing = tuple(getattr(module, name))
    setattr(
        module,
        name,
        tuple(dict.fromkeys(existing + CARRIER_GAUGE_CAUSAL_ARGS)),
    )


def artifact_candidate(args: Any) -> str:
    """Return the most specific scientific candidate represented by ``args``."""
    candidate = str(getattr(args, "carrier_gauge_candidate", "") or "")
    if candidate:
        if candidate not in CANDIDATES:
            raise ValueError(f"invalid carrier-gauge candidate {candidate!r}")
        return candidate
    candidate = str(getattr(args, "deformation_graph_candidate", "") or "")
    if candidate:
        return candidate
    return str(getattr(args, "score_se3_candidate", "unknown"))


@contextmanager
def _score_artifact_identity(args: Any) -> Iterator[None]:
    """Expose the CG identity only while inherited score records are created.

    The checkpoint's causal arguments must remain ``score_se3_candidate=metric_pose`` and
    ``deformation_graph_candidate=deformation_graph|gated_deformation_graph``.  The inherited score exporter, however,
    resolves its artifact label through ``deformation_graph_candidate``.  This
    scoped override keeps the stored model contract untouched while ensuring
    every score record and bundle is named unfixed_graph--dexplicit_3d.
    """
    candidate = artifact_candidate(args)
    if candidate not in CANDIDATES:
        yield
        return
    had_value = hasattr(args, "deformation_graph_candidate")
    previous = getattr(args, "deformation_graph_candidate", None)
    setattr(args, "deformation_graph_candidate", candidate)
    try:
        yield
    finally:
        if had_value:
            setattr(args, "deformation_graph_candidate", previous)
        else:
            delattr(args, "deformation_graph_candidate")


def _carrier_contract(args: Any) -> dict[str, Any]:
    candidate = str(getattr(args, "carrier_gauge_candidate", ""))
    return {
        "candidate": candidate,
        "exact_six_dof_gauge_fix": candidate in {
            "gauge_fixed_graph",
            "low_rank_graph",
            "hand_relational_graph",
            "dexplicit_3d",
        },
        "low_rank_graph_time_decoder": candidate in {"low_rank_graph", "hand_relational_graph", "dexplicit_3d"},
        "node_hand_relations": candidate in {"hand_relational_graph", "dexplicit_3d"},
        "carrier_conditioned_global_transport": candidate == "dexplicit_3d",
        "final_foreground_pointwise_bypass": False,
        "future_ground_truth_inference_input": False,
        "hyperparameters": {
            name: getattr(args, name, None)
            for name in CARRIER_GAUGE_CAUSAL_ARGS
        },
    }


def _install_score_artifact_identity() -> None:
    """Make inherited score bundles and per-window rows use unfixed_graph--dexplicit_3d."""
    import training.score_se3_eval as score_eval

    if getattr(score_eval, "_carrier_gauge_identity_installed", False):
        return

    original_install = score_eval.install_horizon_and_score_eval

    def install_horizon_and_score_eval(trainer_module) -> None:
        original_install(trainer_module)
        trainer_cls = trainer_module.Trainer
        if getattr(
            trainer_cls, "_carrier_gauge_accumulator_installed", False
        ):
            return
        original_accumulate = trainer_cls._accumulate_hoi_dataset_eval_metrics

        def accumulate(self, metrics, counts, batch, outputs):
            with _score_artifact_identity(self.args):
                return original_accumulate(
                    self, metrics, counts, batch, outputs
                )

        trainer_cls._accumulate_hoi_dataset_eval_metrics = accumulate
        trainer_cls._carrier_gauge_accumulator_installed = True

    original_pending = score_eval.write_pending_score_record_bundle

    def write_pending_score_record_bundle(trainer, **kwargs):
        with _score_artifact_identity(trainer.args):
            return original_pending(trainer, **kwargs)

    score_eval.install_horizon_and_score_eval = install_horizon_and_score_eval
    score_eval.write_pending_score_record_bundle = (
        write_pending_score_record_bundle
    )
    score_eval._carrier_gauge_identity_installed = True


def _install_checkpoint_artifact_identity() -> None:
    """Retain checkpoints with CG sidecars without mutating saved metric_pose arguments."""
    import json
    import re

    import training.checkpointing as checkpointing

    if getattr(
        checkpointing, "_carrier_gauge_checkpoint_identity_installed", False
    ):
        return

    def save_score_se3_eval_checkpoint(trainer, log_dict=None):
        if not bool(getattr(trainer.args, "score_se3_eval", False)):
            raise ValueError(
                "score-SE3 eval checkpoint requested for a non-score run"
            )
        if getattr(trainer, "optimizer", None) is None:
            raise RuntimeError(
                "cannot save a training eval checkpoint without an optimizer"
            )
        candidate = artifact_candidate(trainer.args)
        if candidate not in CANDIDATES:
            raise RuntimeError(
                "carrier-gauge checkpoint writer received a non-CG candidate"
            )
        campaign_step = int(
            getattr(trainer, "score_se3_campaign_step", 0)
        )
        optimizer_step = int(getattr(trainer, "optimizer_step", 0))
        if (
            campaign_step < 0
            or optimizer_step < 0
            or optimizer_step > campaign_step
        ):
            raise RuntimeError(
                "invalid score-SE3 checkpoint counters: "
                f"campaign_step={campaign_step}, "
                f"optimizer_step={optimizer_step}"
            )
        token = checkpointing._safe_filename_token(candidate)
        basename = (
            f"eval__candidate={token}"
            f"__campaign-step={campaign_step:08d}"
            f"__optimizer-step={optimizer_step:08d}.pt"
        )
        metric_name = str(
            getattr(
                trainer.args,
                "checkpoint_metric",
                "test/score_se3/composite",
            )
        )
        metric_value = checkpointing._checkpoint_metric_value(
            trainer, log_dict or {}
        )
        if metric_value is None:
            raise RuntimeError(
                "retained score-SE3 eval checkpoint requires a finite "
                f"selection metric {metric_name!r}"
            )
        distributed = bool(getattr(trainer.args, "distributed", False))
        if distributed:
            checkpointing.dist.barrier()
        from training.score_se3_data_resume import (
            collect_score_se3_rng_states,
        )

        rng_state = collect_score_se3_rng_states(trainer)
        record = None
        rank_zero_error = None
        if int(getattr(trainer, "rank", 0)) == 0:
            try:
                directory = os.path.join(
                    trainer.save_dir, "eval_checkpoints"
                )
                os.makedirs(directory, exist_ok=True)
                path = os.path.abspath(os.path.join(directory, basename))
                metadata_path = os.path.splitext(path)[0] + ".json"
                if not os.path.exists(path):
                    # Do not temporarily rewrite score_se3_candidate.  The
                    # serialized causal checkpoint must continue to record metric_pose,
                    # deformation_graph/gated_deformation_graph, and CG* as three separate hierarchy levels.
                    payload = checkpointing._build_checkpoint_dict(
                        trainer,
                        log_dict or {},
                        score_se3_rng_state=rng_state,
                    )
                    checkpointing._atomic_torch_save(payload, path)
                digest = checkpointing.sha256_file(path)
                record = {
                    "schema_version": 1,
                    "checkpoint_type": "score_se3_validation_eval",
                    "candidate": candidate,
                    "score_se3_parent_candidate": str(
                        getattr(trainer.args, "score_se3_candidate", "")
                    ),
                    "deformation_graph_parent_candidate": str(
                        getattr(
                            trainer.args,
                            "deformation_graph_candidate",
                            "",
                        )
                    ),
                    "campaign_step": campaign_step,
                    "optimizer_step": optimizer_step,
                    "metric_name": metric_name,
                    "metric_value": metric_value,
                    "path": path,
                    "sha256": digest,
                    "oracle": False,
                    "deployable": True,
                }
                serialized = (
                    json.dumps(
                        record,
                        indent=2,
                        sort_keys=True,
                        allow_nan=False,
                    )
                    + "\n"
                )
                if os.path.exists(metadata_path):
                    with open(
                        metadata_path, "r", encoding="utf-8"
                    ) as handle:
                        existing = handle.read()
                    if existing != serialized:
                        raise RuntimeError(
                            "retained carrier-gauge checkpoint metadata "
                            f"changed at an existing step: {metadata_path}"
                        )
                else:
                    temporary = f"{metadata_path}.tmp-{os.getpid()}"
                    try:
                        with open(
                            temporary, "w", encoding="utf-8"
                        ) as handle:
                            handle.write(serialized)
                        os.replace(temporary, metadata_path)
                    finally:
                        if os.path.exists(temporary):
                            os.remove(temporary)
                checkpointing._print(
                    "Retained carrier-gauge eval checkpoint: "
                    f"candidate={candidate} campaign_step={campaign_step} "
                    f"optimizer_step={optimizer_step} "
                    f"sha256={digest} -> {path}"
                )
            except Exception as exc:
                if not distributed:
                    raise
                rank_zero_error = f"{type(exc).__name__}: {exc}"
        if distributed:
            holder = [{"record": record, "error": rank_zero_error}]
            checkpointing.dist.broadcast_object_list(
                holder,
                src=0,
                group=getattr(trainer, "cpu_pg", None),
            )
            checkpointing.dist.barrier()
            payload = holder[0]
            if payload["error"] is not None:
                raise RuntimeError(
                    "rank zero failed to retain carrier-gauge checkpoint: "
                    f"{payload['error']}"
                )
            record = payload["record"]
        if not isinstance(record, dict) or not re.fullmatch(
            r"[0-9a-f]{64}", str(record.get("sha256", ""))
        ):
            raise RuntimeError(
                "carrier-gauge checkpoint writer returned invalid metadata"
            )
        return record

    checkpointing.save_score_se3_eval_checkpoint = (
        save_score_se3_eval_checkpoint
    )
    checkpointing._carrier_gauge_checkpoint_identity_installed = True


def install_carrier_gauge_runtime(options: CarrierGaugeOptions) -> None:
    """Patch the existing graph entrypoint without duplicating train/eval code."""
    export_environment(options)

    import training.deformation_graph_model as graph_model
    import training.score_se3_contract as score_contract
    import training.score_se3_integration as score_integration
    from training.dexplicit_3d_model import DeXplicit3DModel

    graph_model.DeformationGraphExperimentalModel = (
        DeXplicit3DModel
    )
    _extend_tuple(score_integration, "SCORE_SE3_RESUME_CAUSAL_ARGS")
    _extend_tuple(score_integration, "SCORE_SE3_INFERENCE_CAUSAL_ARGS")
    _install_score_artifact_identity()
    _install_checkpoint_artifact_identity()

    if not getattr(score_contract, "_carrier_gauge_contract_installed", False):
        original = score_contract.build_score_se3_contract

        def build_score_se3_contract(*args, **kwargs):
            contract = original(*args, **kwargs)
            contract["campaign"] = "lh35_carrier_gauge_graph"
            current_args = args[0] if args else kwargs.get("args")
            candidate = artifact_candidate(current_args)
            contract["carrier_gauge_candidate"] = candidate
            contract.setdefault("model", {})["carrier_gauge"] = (
                _carrier_contract(current_args)
            )
            return contract

        score_contract.build_score_se3_contract = build_score_se3_contract
        score_contract._carrier_gauge_contract_installed = True


__all__ = [
    "CANDIDATES",
    "CarrierGaugeOptions",
    "artifact_candidate",
    "export_environment",
    "install_carrier_gauge_runtime",
    "parse_carrier_gauge_options",
    "prepare_base_argv",
]
