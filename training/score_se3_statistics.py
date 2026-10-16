from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from training.score_se3_metrics import (
    COMPOSITE_SCORE,
    COMPONENT_KEYS,
    EVER_MOVED_T34,
    MOVED_T01_T14,
    MOVED_T01_T34,
    ScoreMetricSpec,
    aggregate_score_records,
    composite_from_means,
    deduplicate_score_records,
)

ALL_SCORE_KEYS = (*COMPONENT_KEYS, COMPOSITE_SCORE)
# Metric-aligned SE(3) ladder (continued_baseline-oracle_contact), canonical deformation-graph ladder
# (dense_decoder-larger_deformation_graph), and carrier-conditioned gauge-fixed graph ladder (unfixed_graph-dexplicit_3d).  The
# graph campaigns reuse this loader for selection, verification, and paired
# comparisons; the first GPU smoke of the CG ladder failed here because only
# continued_baseline-oracle_contact were listed.
KNOWN_CANDIDATES = frozenset(
    {
        "continued_baseline", "metric_loss", "metric_pose", "predicted_contact", "oracle_contact",
        "dense_decoder", "deformation_graph", "gated_deformation_graph", "larger_deformation_graph",
        "unfixed_graph", "gauge_fixed_graph", "low_rank_graph", "hand_relational_graph", "dexplicit_3d",
    }
)
KNOWN_SPLIT_ROLES = frozenset({"validation", "final_test"})
SELECTION_POLICIES = frozenset(
    {
        "per_condition_mean_validation_best",
        "e0_common_step_sensitivity",
        "g0_common_step_sensitivity",
        "unfixed_graph_common_step_sensitivity",
    }
)
EXPECTED_SELECTION_SPLIT = "validation"
EXPECTED_CHECKPOINT_METRIC = "test/score_se3/composite"
EXPECTED_CHECKPOINT_METRIC_MODE = "min"
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


@dataclass(frozen=True)
class ScoreBundle:
    """One candidate/seed/step unique-window evaluation bundle."""

    metadata_path: Path
    records_path: Path
    schema_version: int
    candidate: str
    seed: int
    campaign_step: int
    optimizer_step: int
    split_role: str
    checkpoint_sha256: str
    oracle: bool
    metric_spec: ScoreMetricSpec
    records: tuple[dict[str, Any], ...]
    aggregate: Mapping[str, float | int]
    # Schema-v2 campaigns must evaluate every training seed on the same
    # deterministic point subsample.  A default keeps programmatic test/legacy
    # construction concise; on-disk schema-v2 bundles are validated strictly.
    eval_data_seed: int = 2026


@dataclass(frozen=True)
class CheckpointSelection:
    candidate: str
    seed: int
    campaign_step: int
    optimizer_step: int
    checkpoint_path: str
    checkpoint_sha256: str
    metric_loss_calibration: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class SelectionManifest:
    path: Path
    manifest_sha256: str
    schema_version: int
    policy: str
    selection_split: str
    checkpoint_metric: str
    checkpoint_metric_mode: str
    selections: tuple[CheckpointSelection, ...]
    git_sha: str | None = None
    warm_initialization_checkpoint_sha256: str | None = None
    validation_split_manifest_sha256: str | None = None
    eval_data_seed: int | None = None


def load_selection_manifest(path: str | Path) -> SelectionManifest:
    """Load the sole authorization for comparing condition-specific steps."""

    manifest_path = Path(path).expanduser().resolve()
    raw = manifest_path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, Mapping):
        raise ValueError("selection manifest root must be an object")
    schema_version = int(payload.get("schema_version", -1))
    if schema_version not in {1, 2}:
        raise ValueError("selection manifest schema_version must be 1 or 2")
    policy = payload.get("policy")
    if policy not in SELECTION_POLICIES:
        raise ValueError(
            f"selection manifest policy must be one of {sorted(SELECTION_POLICIES)}"
        )
    selection_split = payload.get("selection_split")
    if selection_split != EXPECTED_SELECTION_SPLIT:
        raise ValueError(
            f"selection_split must be {EXPECTED_SELECTION_SPLIT!r}, got "
            f"{selection_split!r}"
        )
    checkpoint_metric = payload.get("checkpoint_metric")
    if checkpoint_metric != EXPECTED_CHECKPOINT_METRIC:
        raise ValueError(
            f"checkpoint_metric must be {EXPECTED_CHECKPOINT_METRIC!r}, got "
            f"{checkpoint_metric!r}"
        )
    checkpoint_metric_mode = payload.get("checkpoint_metric_mode")
    if checkpoint_metric_mode != EXPECTED_CHECKPOINT_METRIC_MODE:
        raise ValueError(
            "checkpoint_metric_mode must be "
            f"{EXPECTED_CHECKPOINT_METRIC_MODE!r}, got {checkpoint_metric_mode!r}"
        )
    git_sha = payload.get("git_sha")
    warm_initialization_checkpoint_sha256 = payload.get(
        "warm_initialization_checkpoint_sha256"
    )
    validation_split_manifest_sha256 = payload.get(
        "validation_split_manifest_sha256"
    )
    eval_data_seed = payload.get("eval_data_seed")
    if schema_version == 2:
        if not isinstance(git_sha, str) or not re.fullmatch(
            r"[0-9a-f]{40}", git_sha
        ):
            raise ValueError("schema-v2 selection manifest requires lowercase git_sha")
        for name, value in (
            (
                "warm_initialization_checkpoint_sha256",
                warm_initialization_checkpoint_sha256,
            ),
            (
                "validation_split_manifest_sha256",
                validation_split_manifest_sha256,
            ),
        ):
            if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
                raise ValueError(
                    f"schema-v2 selection manifest requires {name}"
                )
        if isinstance(eval_data_seed, bool) or not isinstance(eval_data_seed, int):
            raise ValueError(
                "schema-v2 selection manifest eval_data_seed must be an integer"
            )
        if int(eval_data_seed) < 0:
            raise ValueError("selection manifest eval_data_seed must be non-negative")
    rows = payload.get("selections")
    if not isinstance(rows, list) or not rows:
        raise ValueError("selection manifest requires a non-empty selections list")

    selections: list[CheckpointSelection] = []
    seen: set[tuple[str, int]] = set()
    required = {
        "candidate",
        "seed",
        "optimizer_step",
        "checkpoint_path",
        "checkpoint_sha256",
    }
    if schema_version == 2:
        required.add("campaign_step")
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"selection row {index} must be an object")
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(f"selection row {index} is missing fields {missing}")
        candidate = str(row["candidate"])
        if candidate not in KNOWN_CANDIDATES:
            raise ValueError(
                f"selection row {index} has unknown candidate {candidate!r}"
            )
        if isinstance(row["seed"], bool) or not isinstance(row["seed"], int):
            raise ValueError(f"selection row {index} seed must be an integer")
        if isinstance(row["optimizer_step"], bool) or not isinstance(
            row["optimizer_step"], int
        ):
            raise ValueError(
                f"selection row {index} optimizer_step must be an integer"
            )
        seed = int(row["seed"])
        if schema_version == 2:
            if isinstance(row["campaign_step"], bool) or not isinstance(
                row["campaign_step"], int
            ):
                raise ValueError(
                    f"selection row {index} campaign_step must be an integer"
                )
            campaign_step = int(row["campaign_step"])
        else:
            # Explicit schema-v1 compatibility: legacy artifacts used one
            # optimizer_step field for both campaign alignment and updates.
            campaign_step = int(row["optimizer_step"])
        optimizer_step = int(row["optimizer_step"])
        if campaign_step < 0 or optimizer_step < 0:
            raise ValueError(
                f"selection row {index} campaign_step and optimizer_step "
                "must be non-negative"
            )
        checkpoint_path = row["checkpoint_path"]
        if not isinstance(checkpoint_path, str) or not checkpoint_path.strip():
            raise ValueError(
                f"selection row {index} checkpoint_path must be non-empty"
            )
        checkpoint_sha256 = row["checkpoint_sha256"]
        if not isinstance(checkpoint_sha256, str) or not _SHA256_RE.fullmatch(
            checkpoint_sha256
        ):
            raise ValueError(
                f"selection row {index} checkpoint_sha256 must be 64 hex characters"
            )
        calibration = row.get("metric_loss_calibration")
        if schema_version == 2 and (
            candidate == "continued_baseline" or not candidate.startswith("E")
        ):
            # continued_baseline trains without the metric loss; the G*/CG* graph ladders
            # inherit the selected metric_pose weight and bind no calibration artifact.
            if calibration not in (None, {}):
                raise ValueError(
                    f"{candidate} selection must not bind metric-loss calibration"
                )
            calibration = None
        elif schema_version == 2:
            if not isinstance(calibration, Mapping):
                raise ValueError(
                    f"{candidate} selection requires metric_loss_calibration"
                )
            required_calibration = {
                "path",
                "sha256",
                "seed",
                "batch_count",
                "formula_version",
                "sample_sequence_sha256",
                "warm_checkpoint_sha256",
                "split_manifest_sha256",
                "git_sha",
                "lambda_score",
            }
            missing_calibration = sorted(required_calibration - set(calibration))
            if missing_calibration:
                raise ValueError(
                    "selection calibration is missing fields "
                    f"{missing_calibration}"
                )
            if not isinstance(calibration["path"], str) or not calibration[
                "path"
            ].strip():
                raise ValueError("selection calibration path must be non-empty")
            for name in ("sha256", "sample_sequence_sha256"):
                if not isinstance(calibration[name], str) or not _SHA256_RE.fullmatch(
                    calibration[name]
                ):
                    raise ValueError(f"selection calibration {name} must be SHA-256")
            if int(calibration["seed"]) != seed:
                raise ValueError("selection calibration seed differs from row seed")
            batch_count = int(calibration["batch_count"])
            if not 16 <= batch_count <= 32:
                raise ValueError("selection calibration batch_count must be in [16,32]")
            if not isinstance(calibration["formula_version"], str) or not calibration[
                "formula_version"
            ]:
                raise ValueError("selection calibration formula_version is missing")
            if calibration["warm_checkpoint_sha256"] != (
                warm_initialization_checkpoint_sha256
            ):
                raise ValueError(
                    "selection calibration warm checkpoint differs from manifest"
                )
            if calibration["split_manifest_sha256"] != (
                validation_split_manifest_sha256
            ):
                raise ValueError(
                    "selection calibration split manifest differs from selection"
                )
            if calibration["git_sha"] != git_sha:
                raise ValueError("selection calibration git SHA differs from manifest")
            lambda_score = float(calibration["lambda_score"])
            if not math.isfinite(lambda_score) or lambda_score <= 0.0:
                raise ValueError("selection calibration lambda_score must be positive")
            calibration = dict(calibration)
        identity = (candidate, seed)
        if identity in seen:
            raise ValueError(
                f"duplicate selection row for candidate={candidate}, seed={seed}"
            )
        seen.add(identity)
        selections.append(
            CheckpointSelection(
                candidate=candidate,
                seed=seed,
                campaign_step=campaign_step,
                optimizer_step=optimizer_step,
                checkpoint_path=checkpoint_path.strip(),
                checkpoint_sha256=checkpoint_sha256.lower(),
                metric_loss_calibration=calibration,
            )
        )

    if policy == "per_condition_mean_validation_best":
        for candidate in sorted({row.candidate for row in selections}):
            candidate_steps = {
                row.campaign_step
                for row in selections
                if row.candidate == candidate
            }
            if len(candidate_steps) != 1:
                raise ValueError(
                    "per_condition_mean_validation_best requires one common "
                    f"campaign_step across seeds for candidate {candidate}"
                )
    elif policy == "e0_common_step_sensitivity":
        steps = {row.campaign_step for row in selections}
        if len(steps) != 1:
            raise ValueError(
                "e0_common_step_sensitivity selections must all use one "
                "campaign_step"
            )
    return SelectionManifest(
        path=manifest_path,
        manifest_sha256=hashlib.sha256(raw).hexdigest(),
        schema_version=schema_version,
        policy=str(policy),
        selection_split=str(selection_split),
        checkpoint_metric=str(checkpoint_metric),
        checkpoint_metric_mode=str(checkpoint_metric_mode),
        selections=tuple(selections),
        git_sha=str(git_sha) if git_sha is not None else None,
        warm_initialization_checkpoint_sha256=(
            str(warm_initialization_checkpoint_sha256)
            if warm_initialization_checkpoint_sha256 is not None
            else None
        ),
        validation_split_manifest_sha256=(
            str(validation_split_manifest_sha256)
            if validation_split_manifest_sha256 is not None
            else None
        ),
        eval_data_seed=(
            int(eval_data_seed) if eval_data_seed is not None else None
        ),
    )


def _manifest_selection(
    manifest: SelectionManifest,
    candidate: str,
    seed: int,
) -> CheckpointSelection:
    matches = [
        row
        for row in manifest.selections
        if row.candidate == candidate and row.seed == int(seed)
    ]
    if len(matches) != 1:
        raise ValueError(
            "selection manifest must contain exactly one row for "
            f"candidate={candidate}, seed={seed}; found {len(matches)}"
        )
    return matches[0]


def _authorize_bundle(
    bundle: ScoreBundle,
    manifest: SelectionManifest,
) -> CheckpointSelection:
    row = _manifest_selection(manifest, bundle.candidate, bundle.seed)
    if (
        row.campaign_step != bundle.campaign_step
        or row.optimizer_step != bundle.optimizer_step
        or row.checkpoint_sha256 != bundle.checkpoint_sha256
    ):
        raise ValueError(
            "score bundle metadata disagrees with selection manifest: "
            f"candidate={bundle.candidate}, seed={bundle.seed}, "
            f"bundle_campaign_step={bundle.campaign_step}, "
            f"selected_campaign_step={row.campaign_step}, "
            f"bundle_optimizer_step={bundle.optimizer_step}, "
            f"selected_optimizer_step={row.optimizer_step}, "
            f"bundle_checkpoint_sha256={bundle.checkpoint_sha256!r}, "
            f"selected_checkpoint_sha256={row.checkpoint_sha256!r}"
        )
    return row


def _spec_from_metadata(payload: Mapping[str, Any]) -> ScoreMetricSpec:
    spec = payload.get("metric_spec")
    if not isinstance(spec, Mapping):
        raise ValueError("score bundle metadata is missing metric_spec")
    weights = spec.get("component_weights")
    if not isinstance(weights, Mapping):
        raise ValueError("score bundle metric_spec is missing component_weights")
    return ScoreMetricSpec(
        ever_moved_threshold_m=float(spec["ever_moved_threshold_m"]),
        smooth_epsilon_m=float(spec["smooth_epsilon_m"]),
        early_end_index=int(spec["early_end_index"]),
        final_index=int(spec["final_index"]),
        moved_t01_t34_weight=float(weights[MOVED_T01_T34]),
        moved_t01_t14_weight=float(weights[MOVED_T01_T14]),
        ever_moved_t34_weight=float(weights[EVER_MOVED_T34]),
    )


def _identity(record: Mapping[str, Any]) -> tuple[str, str]:
    return str(record["dataset"]), str(record["sample_key"])


def _source_clip_identity(record: Mapping[str, Any]) -> tuple[str, str]:
    """Return the independent evaluation unit for a window record.

    ``sample_key`` identifies an overlapping evaluation window.  It is not an
    independent bootstrap unit.  ``source_clip_id`` must identify the source
    recording before window extraction, so every window from one recording is
    kept together in a bootstrap replicate.
    """

    if "source_clip_id" not in record:
        raise ValueError(
            "score record is missing source_clip_id; clip-level paired "
            "statistics cannot be computed safely"
        )
    source_clip_id = record["source_clip_id"]
    if source_clip_id is None or isinstance(source_clip_id, bool):
        raise ValueError("score record source_clip_id must be a non-empty value")
    normalized = str(source_clip_id).strip()
    if not normalized:
        raise ValueError("score record source_clip_id must be a non-empty value")
    return str(record["dataset"]), normalized


def _window_to_source_clip(
    records: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], tuple[str, str]]:
    return {_identity(record): _source_clip_identity(record) for record in records}


def _finite_close(left: Any, right: Any, *, atol: float = 1e-8) -> bool:
    try:
        left_value = float(left)
        right_value = float(right)
    except (TypeError, ValueError):
        return left == right
    if not math.isfinite(left_value) or not math.isfinite(right_value):
        return left_value == right_value
    return math.isclose(left_value, right_value, rel_tol=1e-7, abs_tol=atol)


def load_score_bundle(metadata_path: str | Path) -> ScoreBundle:
    metadata_path = Path(metadata_path).expanduser().resolve()
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    schema_version = int(payload.get("schema_version", -1))
    if schema_version not in {1, 2}:
        raise ValueError(
            f"unsupported score bundle schema in {metadata_path}: "
            f"{payload.get('schema_version')!r}"
        )
    candidate = str(payload.get("candidate", ""))
    if candidate not in KNOWN_CANDIDATES:
        raise ValueError(f"invalid candidate {candidate!r} in {metadata_path}")
    oracle = bool(payload.get("oracle", False))
    if oracle != (candidate == "oracle_contact"):
        raise ValueError(
            f"oracle flag/candidate mismatch in {metadata_path}: "
            f"candidate={candidate}, oracle={oracle}"
    )
    seed = int(payload["seed"])
    if "optimizer_step" not in payload:
        raise ValueError(f"score bundle is missing optimizer_step: {metadata_path}")
    optimizer_step = int(payload["optimizer_step"])
    if schema_version == 2:
        if "campaign_step" not in payload:
            raise ValueError(
                f"schema-v2 score bundle is missing campaign_step: {metadata_path}"
            )
        campaign_step = int(payload["campaign_step"])
    else:
        # Explicit schema-v1 compatibility: the legacy optimizer_step was also
        # the only campaign/checkpoint alignment label.
        campaign_step = optimizer_step
    if campaign_step < 0 or optimizer_step < 0:
        raise ValueError("campaign_step and optimizer_step must be non-negative")
    if schema_version == 2:
        split_role = payload.get("split_role")
        if split_role not in KNOWN_SPLIT_ROLES:
            raise ValueError(
                "schema-v2 score bundle split_role must be one of "
                f"{sorted(KNOWN_SPLIT_ROLES)}, got {split_role!r}"
            )
        checkpoint_sha256 = payload.get("checkpoint_sha256")
        if not isinstance(checkpoint_sha256, str) or not _SHA256_RE.fullmatch(
            checkpoint_sha256
        ):
            raise ValueError(
                "schema-v2 score bundle checkpoint_sha256 must be 64 hex "
                f"characters: {metadata_path}"
            )
        checkpoint_sha256 = checkpoint_sha256.lower()
        eval_data_seed = payload.get("eval_data_seed")
        if isinstance(eval_data_seed, bool) or not isinstance(eval_data_seed, int):
            raise ValueError(
                "schema-v2 score bundle eval_data_seed must be an integer: "
                f"{metadata_path}"
            )
        eval_data_seed = int(eval_data_seed)
        if eval_data_seed < 0:
            raise ValueError("score bundle eval_data_seed must be non-negative")
    else:
        split_role = "legacy_unspecified"
        checkpoint_sha256 = ""
        eval_data_seed = -1
    records_file = payload.get("records_file")
    if not isinstance(records_file, str) or not records_file:
        raise ValueError(f"metadata {metadata_path} is missing records_file")
    records_path = (metadata_path.parent / records_file).resolve()
    if not records_path.is_file():
        raise FileNotFoundError(f"score record file not found: {records_path}")
    records = [
        json.loads(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    records = deduplicate_score_records(records)
    if not records:
        raise ValueError(f"score record file is empty: {records_path}")
    for record in records:
        _source_clip_identity(record)
        if str(record.get("candidate", candidate)) != candidate:
            raise ValueError(
                f"record candidate disagrees with metadata in {records_path}"
            )
        if int(record.get("seed", seed)) != seed:
            raise ValueError(f"record seed disagrees with metadata in {records_path}")
        if int(record.get("optimizer_step", optimizer_step)) != optimizer_step:
            raise ValueError(
                f"record optimizer_step disagrees with metadata in {records_path}"
            )
        if int(record.get("campaign_step", campaign_step)) != campaign_step:
            raise ValueError(
                f"record campaign_step disagrees with metadata in {records_path}"
            )
    metric_spec = _spec_from_metadata(payload)
    aggregate = aggregate_score_records(records, spec=metric_spec)
    declared = payload.get("aggregate")
    if isinstance(declared, Mapping):
        for key in ALL_SCORE_KEYS:
            if key in declared and not _finite_close(declared[key], aggregate[key]):
                raise ValueError(
                    f"metadata aggregate {key!r} disagrees with records in "
                    f"{metadata_path}: {declared[key]} vs {aggregate[key]}"
                )
    expected = payload.get("num_unique_windows")
    if expected is not None and int(expected) != len(records):
        raise ValueError(
            f"metadata unique-window count disagrees with records in {metadata_path}"
        )
    unique_source_clips = {_source_clip_identity(record) for record in records}
    expected_source_clips = payload.get("num_unique_source_clips")
    if (
        expected_source_clips is not None
        and int(expected_source_clips) != len(unique_source_clips)
    ):
        raise ValueError(
            "metadata unique-source-clip count disagrees with records in "
            f"{metadata_path}"
        )
    return ScoreBundle(
        metadata_path=metadata_path,
        records_path=records_path,
        schema_version=schema_version,
        candidate=candidate,
        seed=seed,
        campaign_step=campaign_step,
        optimizer_step=optimizer_step,
        split_role=str(split_role),
        checkpoint_sha256=checkpoint_sha256,
        oracle=oracle,
        metric_spec=metric_spec,
        records=tuple(records),
        aggregate=aggregate,
        eval_data_seed=eval_data_seed,
    )


def index_bundles_by_seed(
    bundles: Sequence[ScoreBundle],
    *,
    expected_candidate: str,
    campaign_step: int | None = None,
) -> dict[int, ScoreBundle]:
    selected = [bundle for bundle in bundles if bundle.candidate == expected_candidate]
    if not selected:
        raise ValueError(f"no score bundles found for candidate {expected_candidate}")
    available_steps = sorted({bundle.campaign_step for bundle in selected})
    if campaign_step is None:
        if len(available_steps) != 1:
            raise ValueError(
                f"candidate {expected_candidate} has multiple campaign steps "
                f"{available_steps}; select one explicitly"
            )
        campaign_step = available_steps[0]
    selected = [
        bundle for bundle in selected if bundle.campaign_step == int(campaign_step)
    ]
    if not selected:
        raise ValueError(
            f"candidate {expected_candidate} has no bundle at campaign step "
            f"{campaign_step}"
        )
    by_seed: dict[int, ScoreBundle] = {}
    for bundle in selected:
        if bundle.seed in by_seed:
            raise ValueError(
                f"duplicate candidate={expected_candidate}, seed={bundle.seed}, "
                f"campaign_step={campaign_step} bundles"
            )
        by_seed[bundle.seed] = bundle
    return by_seed


def index_bundles_by_selection(
    bundles: Sequence[ScoreBundle],
    *,
    expected_candidate: str,
    selection_manifest: SelectionManifest,
) -> dict[int, ScoreBundle]:
    """Select exactly the candidate/seed/step rows authorized by a manifest."""

    rows = [
        row
        for row in selection_manifest.selections
        if row.candidate == expected_candidate
    ]
    if not rows:
        raise ValueError(
            f"selection manifest has no rows for candidate {expected_candidate}"
        )
    available: dict[tuple[int, int, int], ScoreBundle] = {}
    for bundle in bundles:
        if bundle.candidate != expected_candidate:
            continue
        identity = (bundle.seed, bundle.campaign_step, bundle.optimizer_step)
        if identity in available:
            raise ValueError(
                f"duplicate candidate={expected_candidate}, seed={bundle.seed}, "
                f"campaign_step={bundle.campaign_step}, "
                f"optimizer_step={bundle.optimizer_step} bundles"
            )
        available[identity] = bundle
    selected: dict[int, ScoreBundle] = {}
    for row in rows:
        identity = (row.seed, row.campaign_step, row.optimizer_step)
        bundle = available.get(identity)
        if bundle is None:
            raise ValueError(
                "no score bundle matches selection manifest row: "
                f"candidate={row.candidate}, seed={row.seed}, "
                f"campaign_step={row.campaign_step}, "
                f"optimizer_step={row.optimizer_step}"
            )
        _authorize_bundle(bundle, selection_manifest)
        selected[row.seed] = bundle
    return selected


def _allow_condition_specific_campaign_steps(
    selection_manifest: SelectionManifest | None,
) -> bool:
    return bool(
        selection_manifest is not None
        and selection_manifest.policy == "per_condition_mean_validation_best"
    )


def _assert_pair_compatible(
    control: ScoreBundle,
    candidate: ScoreBundle,
    *,
    allow_different_campaign_steps: bool = False,
) -> None:
    if control.seed != candidate.seed:
        raise ValueError("paired score bundles must use the same seed")
    if (
        not allow_different_campaign_steps
        and control.campaign_step != candidate.campaign_step
    ):
        raise ValueError("paired score bundles must use the same campaign step")
    if control.metric_spec != candidate.metric_spec:
        raise ValueError("paired score bundles use different metric specifications")
    if control.split_role != candidate.split_role:
        raise ValueError(
            "paired score bundles use different split roles: "
            f"control={control.split_role!r}, candidate={candidate.split_role!r}"
        )
    if control.eval_data_seed != candidate.eval_data_seed:
        raise ValueError(
            "paired score bundles use different held-out sampling seeds: "
            f"control={control.eval_data_seed}, candidate={candidate.eval_data_seed}"
        )
    control_ids = {_identity(record) for record in control.records}
    candidate_ids = {_identity(record) for record in candidate.records}
    if control_ids != candidate_ids:
        missing = sorted(control_ids - candidate_ids)
        extra = sorted(candidate_ids - control_ids)
        raise ValueError(
            "paired score bundles cover different windows: "
            f"missing_from_candidate={missing[:5]}, extra_in_candidate={extra[:5]}"
        )
    control_sources = _window_to_source_clip(control.records)
    candidate_sources = _window_to_source_clip(candidate.records)
    mismatched_assignments = sorted(
        identity
        for identity in control_ids
        if control_sources[identity] != candidate_sources[identity]
    )
    if mismatched_assignments:
        preview = [
            {
                "window": identity,
                "control_source_clip": control_sources[identity],
                "candidate_source_clip": candidate_sources[identity],
            }
            for identity in mismatched_assignments[:5]
        ]
        raise ValueError(
            "paired score bundles assign windows to different source clips: "
            f"{preview}"
        )
    control_clip_ids = set(control_sources.values())
    candidate_clip_ids = set(candidate_sources.values())
    if control_clip_ids != candidate_clip_ids:
        raise ValueError("paired score bundles cover different source clips")


def pair_bundle_groups(
    control: Mapping[int, ScoreBundle],
    candidate: Mapping[int, ScoreBundle],
    *,
    selection_manifest: SelectionManifest | None = None,
) -> list[tuple[ScoreBundle, ScoreBundle]]:
    control_seeds = set(control)
    candidate_seeds = set(candidate)
    if control_seeds != candidate_seeds:
        raise ValueError(
            "control/candidate seed sets differ: "
            f"control={sorted(control_seeds)}, candidate={sorted(candidate_seeds)}"
        )
    control_candidates = {bundle.candidate for bundle in control.values()}
    candidate_candidates = {bundle.candidate for bundle in candidate.values()}
    if len(control_candidates) != 1 or len(candidate_candidates) != 1:
        raise ValueError(
            "each paired bundle group must contain exactly one candidate identity"
        )
    pairs = [(control[seed], candidate[seed]) for seed in sorted(control_seeds)]
    allow_different_campaign_steps = _allow_condition_specific_campaign_steps(
        selection_manifest
    )
    for control_bundle, candidate_bundle in pairs:
        if selection_manifest is not None:
            _authorize_bundle(control_bundle, selection_manifest)
            _authorize_bundle(candidate_bundle, selection_manifest)
        _assert_pair_compatible(
            control_bundle,
            candidate_bundle,
            allow_different_campaign_steps=allow_different_campaign_steps,
        )
    steps = {
        bundle.campaign_step
        for pair in pairs
        for bundle in pair
    }
    if not allow_different_campaign_steps and len(steps) != 1:
        raise ValueError(
            f"paired seeds do not share one campaign step: {sorted(steps)}"
        )
    specs = {control_bundle.metric_spec for control_bundle, _ in pairs}
    if len(specs) != 1:
        raise ValueError("paired seeds do not share one metric specification")
    split_roles = {
        bundle.split_role
        for pair in pairs
        for bundle in pair
    }
    if len(split_roles) != 1:
        raise ValueError("paired seeds do not share one evaluation split role")
    eval_data_seeds = {
        bundle.eval_data_seed for pair in pairs for bundle in pair
    }
    if len(eval_data_seeds) != 1:
        raise ValueError(
            "paired seeds do not share one held-out sampling seed"
        )

    # The common sampling seed is necessary but not sufficient evidence that
    # every training seed was scored on the same held-out population.  Bind
    # all seed pairs to the exact window -> source-clip map established by the
    # first pair so inventory drift cannot be hidden by the nested bootstrap.
    reference_map = _window_to_source_clip(pairs[0][0].records)
    for control_bundle, candidate_bundle in pairs[1:]:
        for bundle in (control_bundle, candidate_bundle):
            observed_map = _window_to_source_clip(bundle.records)
            if observed_map != reference_map:
                reference_ids = set(reference_map)
                observed_ids = set(observed_map)
                changed = sorted(
                    identity
                    for identity in reference_ids & observed_ids
                    if reference_map[identity] != observed_map[identity]
                )
                raise ValueError(
                    "paired seeds do not share one held-out window/source-clip "
                    "identity map: "
                    f"seed={bundle.seed}, "
                    f"missing={sorted(reference_ids - observed_ids)[:5]}, "
                    f"extra={sorted(observed_ids - reference_ids)[:5]}, "
                    f"reassigned={changed[:5]}"
                )
    return pairs


def _ordered_source_clip_arrays(
    bundle: ScoreBundle,
) -> tuple[list[tuple[str, str]], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Aggregate all windows from one source clip before resampling clips."""

    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for record in bundle.records:
        grouped.setdefault(_source_clip_identity(record), []).append(record)
    clip_identities = sorted(grouped)
    sums = {
        key: np.asarray(
            [
                sum(float(record[f"{key}/sum"]) for record in grouped[clip_id])
                for clip_id in clip_identities
            ],
            dtype=np.float64,
        )
        for key in COMPONENT_KEYS
    }
    counts = {
        key: np.asarray(
            [
                sum(int(record[f"{key}/count"]) for record in grouped[clip_id])
                for clip_id in clip_identities
            ],
            dtype=np.int64,
        )
        for key in COMPONENT_KEYS
    }
    return clip_identities, sums, counts


def _bootstrap_pair(
    control: ScoreBundle,
    candidate: ScoreBundle,
    *,
    num_samples: int,
    rng: np.random.Generator,
    allow_different_campaign_steps: bool = False,
) -> dict[str, dict[str, np.ndarray]]:
    _assert_pair_compatible(
        control,
        candidate,
        allow_different_campaign_steps=allow_different_campaign_steps,
    )
    control_ids, control_sums, control_counts = _ordered_source_clip_arrays(control)
    candidate_ids, candidate_sums, candidate_counts = _ordered_source_clip_arrays(
        candidate
    )
    if control_ids != candidate_ids:
        raise RuntimeError("internal paired-source-clip ordering mismatch")
    source_clip_count = len(control_ids)
    sample_index = rng.integers(
        0,
        source_clip_count,
        size=(int(num_samples), source_clip_count),
        endpoint=False,
    )
    result = {
        "control": {},
        "candidate": {},
        "delta": {},
    }
    for key in COMPONENT_KEYS:
        control_total = control_sums[key][sample_index].sum(axis=1)
        control_count = control_counts[key][sample_index].sum(axis=1)
        candidate_total = candidate_sums[key][sample_index].sum(axis=1)
        candidate_count = candidate_counts[key][sample_index].sum(axis=1)
        control_mean = np.divide(
            control_total,
            control_count,
            out=np.full_like(control_total, np.nan, dtype=np.float64),
            where=control_count > 0,
        )
        candidate_mean = np.divide(
            candidate_total,
            candidate_count,
            out=np.full_like(candidate_total, np.nan, dtype=np.float64),
            where=candidate_count > 0,
        )
        result["control"][key] = control_mean
        result["candidate"][key] = candidate_mean
        result["delta"][key] = candidate_mean - control_mean

    weights = control.metric_spec.component_weights
    for side in ("control", "candidate"):
        result[side][COMPOSITE_SCORE] = sum(
            float(weights[key]) * result[side][key] for key in COMPONENT_KEYS
        )
    result["delta"][COMPOSITE_SCORE] = (
        result["candidate"][COMPOSITE_SCORE]
        - result["control"][COMPOSITE_SCORE]
    )
    return result


def _observed_bundle_metrics(bundle: ScoreBundle) -> dict[str, float]:
    return {key: float(bundle.aggregate[key]) for key in ALL_SCORE_KEYS}


def compare_bundle_groups(
    pairs: Sequence[tuple[ScoreBundle, ScoreBundle]],
    *,
    num_bootstrap: int = 10_000,
    bootstrap_seed: int = 2026,
    confidence: float = 0.95,
    max_component_regression: float = 0.01,
    min_practical_improvement: float = 0.02,
    minimum_seeds: int = 3,
    selection_manifest: SelectionManifest | None = None,
) -> dict[str, Any]:
    """Hierarchical paired bootstrap over training seeds and source clips.

    Windows are first aggregated within ``source_clip_id``.  Bootstrap draws
    then resample whole source clips, preserving the correlation among
    overlapping windows from the same original recording.
    """

    if not pairs:
        raise ValueError("at least one paired seed is required")
    if num_bootstrap < 100:
        raise ValueError("num_bootstrap must be at least 100")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    if max_component_regression < 0 or min_practical_improvement < 0:
        raise ValueError("gate tolerances must be non-negative")
    if minimum_seeds < 1:
        raise ValueError("minimum_seeds must be at least 1")
    allow_different_campaign_steps = _allow_condition_specific_campaign_steps(
        selection_manifest
    )
    for control, candidate in pairs:
        if selection_manifest is not None:
            _authorize_bundle(control, selection_manifest)
            _authorize_bundle(candidate, selection_manifest)
        _assert_pair_compatible(
            control,
            candidate,
            allow_different_campaign_steps=allow_different_campaign_steps,
        )
    control_candidates = {control.candidate for control, _ in pairs}
    candidate_candidates = {candidate.candidate for _, candidate in pairs}
    if len(control_candidates) != 1 or len(candidate_candidates) != 1:
        raise ValueError(
            "paired comparisons must contain one control and one candidate identity"
        )
    pair_seeds = [control.seed for control, _ in pairs]
    if len(set(pair_seeds)) != len(pair_seeds):
        raise ValueError("paired comparisons contain duplicate training seeds")
    all_pair_campaign_steps = {
        bundle.campaign_step
        for pair in pairs
        for bundle in pair
    }
    if (
        not allow_different_campaign_steps
        and len(all_pair_campaign_steps) != 1
    ):
        raise ValueError(
            "paired comparisons span multiple campaign steps without a "
            "per-condition selection manifest"
        )
    pair_split_roles = {
        bundle.split_role
        for pair in pairs
        for bundle in pair
    }
    if len(pair_split_roles) != 1:
        raise ValueError("paired comparisons span multiple evaluation split roles")
    pair_specs = {bundle.metric_spec for pair in pairs for bundle in pair}
    if len(pair_specs) != 1:
        raise ValueError("paired comparisons span multiple metric specifications")
    pair_eval_data_seeds = {
        bundle.eval_data_seed for pair in pairs for bundle in pair
    }
    if len(pair_eval_data_seeds) != 1:
        raise ValueError(
            "paired comparisons span multiple held-out sampling seeds"
        )

    rng = np.random.default_rng(int(bootstrap_seed))
    per_seed_bootstrap = [
        _bootstrap_pair(
            control,
            candidate,
            num_samples=int(num_bootstrap),
            rng=rng,
            allow_different_campaign_steps=allow_different_campaign_steps,
        )
        for control, candidate in pairs
    ]
    seed_count = len(pairs)
    seed_index = rng.integers(
        0,
        seed_count,
        size=(int(num_bootstrap), seed_count),
        endpoint=False,
    )
    # Each resampled seed occurrence gets an independent within-seed clip
    # bootstrap draw. If one seed is selected twice in a replicate, reusing one
    # clip draw for both copies would spuriously correlate the nested samples.
    within_seed_bootstrap_index = rng.integers(
        0,
        int(num_bootstrap),
        size=(int(num_bootstrap), seed_count),
        endpoint=False,
    )
    alpha = (1.0 - float(confidence)) / 2.0

    control_observed_by_seed = [
        _observed_bundle_metrics(control) for control, _ in pairs
    ]
    candidate_observed_by_seed = [
        _observed_bundle_metrics(candidate) for _, candidate in pairs
    ]
    metric_results: dict[str, Any] = {}
    for key in ALL_SCORE_KEYS:
        control_seed_values = np.asarray(
            [values[key] for values in control_observed_by_seed], dtype=np.float64
        )
        candidate_seed_values = np.asarray(
            [values[key] for values in candidate_observed_by_seed], dtype=np.float64
        )
        control_mean = float(control_seed_values.mean())
        candidate_mean = float(candidate_seed_values.mean())
        delta = candidate_mean - control_mean
        relative_improvement = (
            (control_mean - candidate_mean) / control_mean
            if control_mean != 0
            else float("nan")
        )

        control_bootstrap = np.stack(
            [values["control"][key] for values in per_seed_bootstrap], axis=0
        )
        candidate_bootstrap = np.stack(
            [values["candidate"][key] for values in per_seed_bootstrap], axis=0
        )
        control_hierarchical = control_bootstrap[
            seed_index, within_seed_bootstrap_index
        ].mean(axis=1)
        candidate_hierarchical = candidate_bootstrap[
            seed_index, within_seed_bootstrap_index
        ].mean(axis=1)
        delta_hierarchical = candidate_hierarchical - control_hierarchical
        finite = np.isfinite(delta_hierarchical)
        finite_count = int(finite.sum())
        if finite_count < max(100, int(0.9 * num_bootstrap)):
            raise RuntimeError(
                f"too few finite bootstrap replicates for {key}: "
                f"{finite_count}/{num_bootstrap}"
            )
        finite_delta = delta_hierarchical[finite]
        ci_low, ci_high = np.quantile(finite_delta, [alpha, 1.0 - alpha])
        metric_results[key] = {
            "control_mean": control_mean,
            "candidate_mean": candidate_mean,
            "delta_candidate_minus_control": float(delta),
            "relative_improvement": float(relative_improvement),
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
            "probability_candidate_better": float((finite_delta < 0).mean()),
            "finite_bootstrap_replicates": finite_count,
        }

    component_regressions = {
        key: metric_results[key]["relative_improvement"]
        for key in COMPONENT_KEYS
        if metric_results[key]["relative_improvement"]
        < -float(max_component_regression)
    }
    statistical_pass = (
        metric_results[COMPOSITE_SCORE]["ci_high"] < 0.0
    )
    practical_pass = (
        metric_results[MOVED_T01_T34]["relative_improvement"]
        >= float(min_practical_improvement)
        and metric_results[EVER_MOVED_T34]["relative_improvement"]
        >= float(min_practical_improvement)
    )
    component_guard_pass = not component_regressions
    seed_count_pass = seed_count >= int(minimum_seeds)
    promotion_pass = (
        statistical_pass
        and practical_pass
        and component_guard_pass
        and seed_count_pass
    )

    per_seed = []
    for (control, candidate), control_values, candidate_values in zip(
        pairs,
        control_observed_by_seed,
        candidate_observed_by_seed,
    ):
        row: dict[str, Any] = {
            "seed": int(control.seed),
            "split_role": control.split_role,
            "control_campaign_step": int(control.campaign_step),
            "candidate_campaign_step": int(candidate.campaign_step),
            "control_optimizer_step": int(control.optimizer_step),
            "candidate_optimizer_step": int(candidate.optimizer_step),
            "control_metadata": str(control.metadata_path),
            "candidate_metadata": str(candidate.metadata_path),
            "control_bundle_checkpoint_sha256": control.checkpoint_sha256,
            "candidate_bundle_checkpoint_sha256": candidate.checkpoint_sha256,
            "metrics": {
                key: {
                    "control": control_values[key],
                    "candidate": candidate_values[key],
                    "delta": candidate_values[key] - control_values[key],
                }
                for key in ALL_SCORE_KEYS
            },
        }
        if control.campaign_step == candidate.campaign_step:
            row["campaign_step"] = int(control.campaign_step)
        if control.optimizer_step == candidate.optimizer_step:
            row["optimizer_step"] = int(control.optimizer_step)
        if selection_manifest is not None:
            control_selection = _authorize_bundle(control, selection_manifest)
            candidate_selection = _authorize_bundle(candidate, selection_manifest)
            row["control_checkpoint"] = {
                "path": control_selection.checkpoint_path,
                "sha256": control_selection.checkpoint_sha256,
                "metric_loss_calibration": (
                    dict(control_selection.metric_loss_calibration)
                    if control_selection.metric_loss_calibration is not None
                    else None
                ),
            }
            row["candidate_checkpoint"] = {
                "path": candidate_selection.checkpoint_path,
                "sha256": candidate_selection.checkpoint_sha256,
                "metric_loss_calibration": (
                    dict(candidate_selection.metric_loss_calibration)
                    if candidate_selection.metric_loss_calibration is not None
                    else None
                ),
            }
        per_seed.append(row)

    control_candidate = pairs[0][0].candidate
    candidate_name = pairs[0][1].candidate
    all_steps = {
        bundle.optimizer_step
        for pair in pairs
        for bundle in pair
    }
    all_campaign_steps = {
        bundle.campaign_step
        for pair in pairs
        for bundle in pair
    }
    report: dict[str, Any] = {
        "schema_version": 2,
        "control_candidate": control_candidate,
        "candidate": candidate_name,
        "oracle_candidate": bool(pairs[0][1].oracle),
        "split_role": pairs[0][0].split_role,
        "eval_data_seed": int(pairs[0][0].eval_data_seed),
        "campaign_step": (
            int(next(iter(all_campaign_steps)))
            if len(all_campaign_steps) == 1
            else None
        ),
        "campaign_steps": {
            control_candidate: {
                str(control.seed): int(control.campaign_step)
                for control, _ in pairs
            },
            candidate_name: {
                str(candidate.seed): int(candidate.campaign_step)
                for _, candidate in pairs
            },
        },
        "optimizer_step": int(next(iter(all_steps))) if len(all_steps) == 1 else None,
        "optimizer_steps": {
            control_candidate: {
                str(control.seed): int(control.optimizer_step)
                for control, _ in pairs
            },
            candidate_name: {
                str(candidate.seed): int(candidate.optimizer_step)
                for _, candidate in pairs
            },
        },
        "seeds": [int(control.seed) for control, _ in pairs],
        "num_seeds": seed_count,
        "num_unique_windows_per_seed": [len(control.records) for control, _ in pairs],
        "num_unique_source_clips_per_seed": [
            len({_source_clip_identity(record) for record in control.records})
            for control, _ in pairs
        ],
        "metric_spec": {
            "ever_moved_threshold_m": pairs[0][0].metric_spec.ever_moved_threshold_m,
            "smooth_epsilon_m": pairs[0][0].metric_spec.smooth_epsilon_m,
            "early_end_index": pairs[0][0].metric_spec.early_end_index,
            "final_index": pairs[0][0].metric_spec.final_index,
            "component_weights": pairs[0][0].metric_spec.component_weights,
        },
        "bootstrap": {
            "method": (
                "paired hierarchical percentile bootstrap: seeds then source clips; "
                "windows aggregated within source_clip_id"
            ),
            "replicates": int(num_bootstrap),
            "seed": int(bootstrap_seed),
            "confidence": float(confidence),
        },
        "metrics": metric_results,
        "per_seed": per_seed,
        "gates": {
            "statistical_composite_ci_upper_below_zero": statistical_pass,
            "component_guard_pass": component_guard_pass,
            "component_regressions_beyond_limit": component_regressions,
            "max_component_regression": float(max_component_regression),
            "practical_full_and_final_pass": practical_pass,
            "min_practical_improvement": float(min_practical_improvement),
            "minimum_seeds": int(minimum_seeds),
            "seed_count_pass": bool(seed_count_pass),
            "promotion_pass": promotion_pass,
        },
    }
    if selection_manifest is not None:
        report["selection_manifest"] = {
            "path": str(selection_manifest.path),
            "sha256": selection_manifest.manifest_sha256,
            "schema_version": selection_manifest.schema_version,
            "policy": selection_manifest.policy,
            "selection_split": selection_manifest.selection_split,
            "checkpoint_metric": selection_manifest.checkpoint_metric,
            "checkpoint_metric_mode": selection_manifest.checkpoint_metric_mode,
            "git_sha": selection_manifest.git_sha,
            "warm_initialization_checkpoint_sha256": (
                selection_manifest.warm_initialization_checkpoint_sha256
            ),
            "validation_split_manifest_sha256": (
                selection_manifest.validation_split_manifest_sha256
            ),
            "eval_data_seed": selection_manifest.eval_data_seed,
        }
        report["selected_checkpoints"] = {
            control_candidate: {
                str(control.seed): {
                    "campaign_step": int(control.campaign_step),
                    "optimizer_step": int(control.optimizer_step),
                    "path": _authorize_bundle(
                        control, selection_manifest
                    ).checkpoint_path,
                    "sha256": _authorize_bundle(
                        control, selection_manifest
                    ).checkpoint_sha256,
                    "metric_loss_calibration": (
                        dict(
                            _authorize_bundle(
                                control, selection_manifest
                            ).metric_loss_calibration
                        )
                        if _authorize_bundle(
                            control, selection_manifest
                        ).metric_loss_calibration is not None
                        else None
                    ),
                }
                for control, _ in pairs
            },
            candidate_name: {
                str(candidate.seed): {
                    "campaign_step": int(candidate.campaign_step),
                    "optimizer_step": int(candidate.optimizer_step),
                    "path": _authorize_bundle(
                        candidate, selection_manifest
                    ).checkpoint_path,
                    "sha256": _authorize_bundle(
                        candidate, selection_manifest
                    ).checkpoint_sha256,
                    "metric_loss_calibration": (
                        dict(
                            _authorize_bundle(
                                candidate, selection_manifest
                            ).metric_loss_calibration
                        )
                        if _authorize_bundle(
                            candidate, selection_manifest
                        ).metric_loss_calibration is not None
                        else None
                    ),
                }
                for _, candidate in pairs
            },
        }
    return report


def oracle_recovery_report(
    e2: Mapping[int, ScoreBundle],
    e3: Mapping[int, ScoreBundle],
    e4: Mapping[int, ScoreBundle],
    *,
    minimum_recovery: float = 0.50,
    minimum_score_improvement: float = 0.02,
    minimum_exact_t34_improvement: float = 0.03,
    num_bootstrap: int = 10_000,
    bootstrap_seed: int = 2026,
    confidence: float = 0.95,
    minimum_seeds: int = 3,
    selection_manifest: SelectionManifest | None = None,
) -> dict[str, Any]:
    """Gate contact work on a significant, practically useful oracle gap.

    oracle_contact must beat metric_pose by the configured relative margins on both the unified
    score and exact-t34, with paired clip-bootstrap confidence intervals wholly
    below zero.  Only then can predicted_contact pass by recovering the configured fraction of
    the oracle_contact-vs-metric_pose unified-score gain.
    """

    seeds = set(e2)
    if seeds != set(e3) or seeds != set(e4):
        raise ValueError("metric_pose/predicted_contact/oracle_contact oracle recovery requires identical seed sets")
    if not seeds:
        raise ValueError("oracle recovery requires at least one seed")
    if (
        minimum_recovery < 0
        or minimum_score_improvement < 0
        or minimum_exact_t34_improvement < 0
    ):
        raise ValueError("oracle gate thresholds must be non-negative")
    rows = []
    allow_different_campaign_steps = _allow_condition_specific_campaign_steps(
        selection_manifest
    )
    for seed in sorted(seeds):
        if selection_manifest is not None:
            _authorize_bundle(e2[seed], selection_manifest)
            _authorize_bundle(e3[seed], selection_manifest)
            _authorize_bundle(e4[seed], selection_manifest)
        _assert_pair_compatible(
            e2[seed],
            e3[seed],
            allow_different_campaign_steps=allow_different_campaign_steps,
        )
        _assert_pair_compatible(
            e2[seed],
            e4[seed],
            allow_different_campaign_steps=allow_different_campaign_steps,
        )
        if e2[seed].candidate != "metric_pose" or e3[seed].candidate != "predicted_contact":
            raise ValueError("oracle recovery inputs must be metric_pose, predicted_contact, and oracle_contact")
        if e4[seed].candidate != "oracle_contact" or not e4[seed].oracle:
            raise ValueError("oracle_contact bundle must be marked oracle")
        rows.append(
            {
                key: (
                    float(e2[seed].aggregate[key]),
                    float(e3[seed].aggregate[key]),
                    float(e4[seed].aggregate[key]),
                )
                for key in ALL_SCORE_KEYS
            }
        )

    oracle_comparison = compare_bundle_groups(
        pair_bundle_groups(
            e2,
            e4,
            selection_manifest=selection_manifest,
        ),
        num_bootstrap=num_bootstrap,
        bootstrap_seed=bootstrap_seed,
        confidence=confidence,
        # The oracle-specific gates below intentionally use separate score and
        # exact-t34 margins.  The generic promotion gates are retained in the
        # embedded comparison only as diagnostics.
        max_component_regression=1.0,
        min_practical_improvement=0.0,
        minimum_seeds=minimum_seeds,
        selection_manifest=selection_manifest,
    )

    metrics = {}
    for key in ALL_SCORE_KEYS:
        e2_mean = float(np.mean([row[key][0] for row in rows]))
        e3_mean = float(np.mean([row[key][1] for row in rows]))
        e4_mean = float(np.mean([row[key][2] for row in rows]))
        oracle_gain = e2_mean - e4_mean
        predicted_gain = e2_mean - e3_mean
        recovery = predicted_gain / oracle_gain if oracle_gain > 0 else float("nan")
        metrics[key] = {
            "e2_action_only": e2_mean,
            "e3_predicted_contact": e3_mean,
            "e4_gt_contact_oracle": e4_mean,
            "oracle_gain": oracle_gain,
            "predicted_contact_gain": predicted_gain,
            "recovery_fraction": recovery,
        }
    composite_recovery = metrics[COMPOSITE_SCORE]
    score_comparison = oracle_comparison["metrics"][COMPOSITE_SCORE]
    exact_t34_comparison = oracle_comparison["metrics"][EVER_MOVED_T34]
    score_practical_pass = (
        score_comparison["relative_improvement"]
        >= float(minimum_score_improvement)
    )
    exact_t34_practical_pass = (
        exact_t34_comparison["relative_improvement"]
        >= float(minimum_exact_t34_improvement)
    )
    score_significance_pass = score_comparison["ci_high"] < 0.0
    exact_t34_significance_pass = exact_t34_comparison["ci_high"] < 0.0
    predicted_contact_recovery_pass = (
        composite_recovery["oracle_gain"] > 0
        and composite_recovery["predicted_contact_gain"] > 0
        and math.isfinite(composite_recovery["recovery_fraction"])
        and composite_recovery["recovery_fraction"] >= float(minimum_recovery)
    )
    seed_count_pass = len(seeds) >= int(minimum_seeds)
    pass_gate = (
        score_practical_pass
        and exact_t34_practical_pass
        and score_significance_pass
        and exact_t34_significance_pass
        and predicted_contact_recovery_pass
        and seed_count_pass
    )
    all_steps = {
        bundle.optimizer_step
        for group in (e2, e3, e4)
        for bundle in group.values()
    }
    all_campaign_steps = {
        bundle.campaign_step
        for group in (e2, e3, e4)
        for bundle in group.values()
    }
    report: dict[str, Any] = {
        "schema_version": 2,
        "seeds": sorted(int(seed) for seed in seeds),
        "split_role": e2[next(iter(seeds))].split_role,
        "campaign_step": (
            int(next(iter(all_campaign_steps)))
            if len(all_campaign_steps) == 1
            else None
        ),
        "campaign_steps": {
            candidate: {
                str(seed): int(bundle.campaign_step)
                for seed, bundle in sorted(group.items())
            }
            for candidate, group in (("metric_pose", e2), ("predicted_contact", e3), ("oracle_contact", e4))
        },
        "optimizer_step": int(next(iter(all_steps))) if len(all_steps) == 1 else None,
        "optimizer_steps": {
            candidate: {
                str(seed): int(bundle.optimizer_step)
                for seed, bundle in sorted(group.items())
            }
            for candidate, group in (("metric_pose", e2), ("predicted_contact", e3), ("oracle_contact", e4))
        },
        "metrics": metrics,
        "minimum_recovery": float(minimum_recovery),
        "minimum_score_improvement": float(minimum_score_improvement),
        "minimum_exact_t34_improvement": float(minimum_exact_t34_improvement),
        "minimum_seeds": int(minimum_seeds),
        "bootstrap": oracle_comparison["bootstrap"],
        "oracle_vs_action_comparison": oracle_comparison,
        "gates": {
            "score_practical_pass": bool(score_practical_pass),
            "exact_t34_practical_pass": bool(exact_t34_practical_pass),
            "score_ci_upper_below_zero": bool(score_significance_pass),
            "exact_t34_ci_upper_below_zero": bool(exact_t34_significance_pass),
            "predicted_contact_recovery_pass": bool(
                predicted_contact_recovery_pass
            ),
            "seed_count_pass": bool(seed_count_pass),
        },
        "continue_contact_investment": bool(pass_gate),
    }
    if selection_manifest is not None:
        report["selection_manifest"] = oracle_comparison["selection_manifest"]
        report["selected_checkpoints"] = {
            candidate: {
                str(seed): {
                    "campaign_step": int(bundle.campaign_step),
                    "optimizer_step": int(bundle.optimizer_step),
                    "path": _authorize_bundle(
                        bundle, selection_manifest
                    ).checkpoint_path,
                    "sha256": _authorize_bundle(
                        bundle, selection_manifest
                    ).checkpoint_sha256,
                }
                for seed, bundle in sorted(group.items())
            }
            for candidate, group in (("metric_pose", e2), ("predicted_contact", e3), ("oracle_contact", e4))
        }
    return report
