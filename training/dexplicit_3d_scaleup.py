"""dexplicit_3d budget and raw-L2 policy; the frozen 850-step campaign is unchanged."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import shlex
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

METRIC = "l2_moved/mean_t01_t34"
CHECKPOINT_METRIC = "test/" + METRIC
POLICY = "dexplicit_3d_raw_moved_l2_t01_t34_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_json(path: Path, value: Mapping[str, Any]) -> None:
    """Publish a complete immutable receipt atomically, including concurrent calls."""
    text = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".receipt-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)  # Atomic create-if-absent, not overwrite.
        except FileExistsError:
            if path.read_text(encoding="utf-8") != text:
                raise ValueError(f"immutable receipt changed: {path}")
    finally:
        os.unlink(temporary)


def budget(windows: int, epochs: int = 20) -> dict[str, int]:
    if (type(windows) is not int or type(epochs) is not int
            or windows < 16 or epochs <= 2):
        raise ValueError("require integer >=16 training windows and >2 total epochs")
    steps = (windows + 15) // 16
    cadence = steps // 2 if steps % 2 == 0 else steps
    return dict(epochs=epochs, windows=windows, batch_size=16,
                steps_per_epoch=steps, total_steps=epochs * steps,
                phase_a_steps=2 * steps, warmup_steps=min(20, 2 * steps),
                cadence=cadence, padded_windows_per_epoch=steps * 16 - windows)


def smoke_budget(plan: Mapping[str, Any]) -> dict[str, int]:
    return {**plan["budget"], "total_steps": 2, "phase_a_steps": 1,
            "warmup_steps": 1, "cadence": 1}


def replace_option(argv: Sequence[str], name: str, value: Any) -> list[str]:
    result = list(argv)
    matches = [i for i, item in enumerate(result)
               if item == name or item.startswith(name + "=")]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {name}, found {len(matches)}")
    index = matches[0]
    if result[index] == name:
        if index + 1 >= len(result) or result[index + 1].startswith("--"):
            raise ValueError(f"missing value for {name}")
        result[index + 1] = str(value)
    else:
        result[index] = name + "=" + str(value)
    return result


def template_command(text: str) -> list[str]:
    prefix = "[deformation-graph] command:"
    lines = [line[len(prefix):].strip() for line in text.splitlines()
             if line.startswith(prefix)]
    if len(lines) != 1 or "$'" in lines[0]:
        raise ValueError("expected one ASCII shell-escaped inherited command")
    argv = shlex.split(lines[0])
    if argv[:4] != ["torchrun", "--standalone", "--nproc_per_node=1",
                    "train_hot3d_carrier_gauge.py"]:
        raise ValueError("unexpected inherited single-GPU entrypoint")
    argv[3] = "train_hot3d_dexplicit_3d_scaleup.py"
    argv = replace_option(argv, "--checkpoint_metric", CHECKPOINT_METRIC)
    return replace_option(argv, "--checkpoint_metric_mode", "min")


def raw_mean(records: Sequence[Mapping[str, Any]]) -> float:
    total, count = 0.0, 0.0
    for row in records:
        numerator = float(row[METRIC + "/sum"])
        denominator = float(row[METRIC + "/count"])
        if (not math.isfinite(numerator) or not math.isfinite(denominator)
                or numerator < 0 or denominator < 0 or denominator != int(denominator)
                or (denominator == 0 and numerator != 0)):
            raise ValueError("invalid moved-point L2 sum/count")
        total += numerator
        count += denominator
    if not math.isfinite(total) or not math.isfinite(count) or count <= 0:
        raise ValueError("raw L2 requires a finite nonempty supervised population")
    return total / count


def population(records: Sequence[Mapping[str, Any]]) -> list[tuple]:
    rows = sorted((str(r["dataset"]), str(r["sample_key"]),
                   str(r["source_clip_id"]), float(r[METRIC + "/count"]))
                  for r in records)
    if not rows or len({r[:2] for r in rows}) != len(rows):
        raise ValueError("empty or duplicate unique-window identities")
    if any(not all(r[:3]) or r[2] == "None" for r in rows):
        raise ValueError("missing source/window identity")
    return rows


def choose_bundle(bundles: Sequence[Any], total_steps: int, cadence: int) -> Any:
    """Accept only a complete, matched validation curve, including step zero."""
    if cadence <= 0 or total_steps <= 0 or total_steps % cadence:
        raise ValueError("invalid evaluation cadence/budget")
    expected = set(range(0, total_steps + 1, cadence))
    if len(bundles) != len(expected):
        raise ValueError("incomplete validation curve")
    seen, reference, reference_spec = set(), None, None
    for b in bundles:
        if (b.candidate != "dexplicit_3d" or b.seed != 2026 or b.eval_data_seed != 2026
                or b.split_role != "validation" or b.schema_version != 2 or b.oracle):
            raise ValueError("selection requires deployable dexplicit_3d/2026 validation")
        if b.campaign_step in seen:
            raise ValueError("duplicate evaluation step")
        seen.add(b.campaign_step)
        identities = population(b.records)
        if reference is None:
            reference, reference_spec = identities, b.metric_spec
        elif identities != reference or b.metric_spec != reference_spec:
            raise ValueError("validation windows, source clips, masks, or metric changed")
        raw_mean(b.records)
    if seen != expected:
        raise ValueError("missing or unexpected evaluation steps")
    return min(bundles, key=lambda b: (raw_mean(b.records), b.campaign_step))


def select_checkpoint(run_dir: Path, plan: Mapping[str, Any]) -> dict[str, Any]:
    from training.score_se3_statistics import load_score_bundle

    bundles = [load_score_bundle(p) for p in sorted(run_dir.rglob("*.meta.json"))]
    schedule = plan["budget"]
    best = choose_bundle(bundles, schedule["total_steps"], schedule["cadence"])
    manifest_path = Path(plan["split_manifest"])
    if sha256(manifest_path) != plan["split_sha256"]:
        raise ValueError("frozen split manifest changed")
    manifest = json.loads(manifest_path.read_text())
    subset = manifest["subsets"]["validation"]
    allowed = set(subset["source_clips"])
    if {str(r["source_clip_id"]) for r in best.records} != allowed:
        raise ValueError("score population does not cover frozen validation clips")
    if len(best.records) != subset["num_windows"]:
        raise ValueError("validation file/unique-window counts differ; reconcile before selecting")
    matches = []
    for path in run_dir.glob("**/eval_checkpoints/*.json"):
        row = json.loads(path.read_text())
        if (row.get("candidate") == "dexplicit_3d"
                and row.get("campaign_step") == best.campaign_step
                and row.get("optimizer_step") == best.optimizer_step
                and row.get("sha256") == best.checkpoint_sha256):
            matches.append(row)
    if len(matches) != 1:
        raise ValueError("selected score bundle must match exactly one checkpoint")
    row = matches[0]
    checkpoint = Path(row["path"]).resolve()
    checkpoint.relative_to(run_dir.resolve())
    value = raw_mean(best.records)
    if (row["metric_name"] != CHECKPOINT_METRIC
            or not math.isclose(float(row["metric_value"]), value, rel_tol=1e-7, abs_tol=1e-9)
            or sha256(checkpoint) != best.checkpoint_sha256):
        raise ValueError("checkpoint metric or content does not match raw-L2 selection")
    result = dict(schema_version=1, policy=POLICY, selection_split="validation",
                  candidate="dexplicit_3d", seed=2026, eval_data_seed=2026,
                  checkpoint_metric=CHECKPOINT_METRIC, checkpoint_metric_mode="min",
                  plan_sha256=sha256(run_dir.parent / "plan.json"),
                  checkpoint_path=str(checkpoint), checkpoint_sha256=best.checkpoint_sha256,
                  campaign_step=best.campaign_step, optimizer_step=best.optimizer_step,
                  metric_value=value, metadata_path=str(best.metadata_path),
                  metadata_sha256=sha256(best.metadata_path), records_sha256=sha256(best.records_path),
                  unique_windows=len(best.records), unique_source_clips=len(allowed),
                  population_sha256=hashlib.sha256(json.dumps(population(best.records)).encode()).hexdigest())
    freeze_json(run_dir.parent / "selected_raw_l2.json", result)
    return result


def build_raw_contract(original, args, **kwargs):
    """Version the legacy policy; retain every other inherited strict check.

    The parent only accepts composite. Validate its checks on a shallow copy,
    then serialize the actual raw-L2 policy. Model, loss, trainer and checkpoint
    arguments are never mutated; the historical entrypoint remains unchanged.
    """
    if (args.checkpoint_metric != CHECKPOINT_METRIC
            or args.checkpoint_metric_mode != "min" or args.score_se3_candidate != "metric_pose"
            or args.deformation_graph_candidate != "gated_deformation_graph" or args.carrier_gauge_candidate != "dexplicit_3d"
            or not args.score_se3_strict_contract):
        raise ValueError("dexplicit_3d scaleup requires strict dexplicit_3d/gated_deformation_graph/metric_pose and raw-L2/min policy")
    shadow = copy.copy(args)
    shadow.checkpoint_metric = "test/score_se3/composite"
    contract = original(shadow, **kwargs)
    contract["evaluation"]["checkpoint_metric"] = CHECKPOINT_METRIC
    contract["evaluation"]["selection_policy"] = POLICY
    return contract


def paired_raw_bootstrap(reference, candidate, repeats: int = 10000) -> dict[str, Any]:
    """Paired source-clip resampling of sums/counts, conditional on one training seed."""
    import numpy as np

    if (reference.split_role != "final_test" or candidate.split_role != "final_test"
            or reference.schema_version != 2 or candidate.schema_version != 2
            or reference.oracle or candidate.oracle
            or reference.seed != candidate.seed or reference.eval_data_seed != candidate.eval_data_seed
            or reference.metric_spec != candidate.metric_spec
            or population(reference.records) != population(candidate.records)):
        raise ValueError("comparison requires identical deployable final-test populations/protocols")
    if repeats < 100:
        raise ValueError("at least 100 bootstrap replicates required")
    grouped = {}
    for side, bundle in enumerate((reference, candidate)):
        raw_mean(bundle.records)
        for r in bundle.records:
            key = (str(r["dataset"]), str(r["source_clip_id"]))
            row = grouped.setdefault(key, np.zeros(4, dtype=np.float64))
            row[2 * side] += float(r[METRIC + "/sum"])
            row[2 * side + 1] += float(r[METRIC + "/count"])
    # Zero-supervision clips contain no contribution to this estimand.
    values = np.array([v for _, v in sorted(grouped.items()) if v[1] > 0])
    if len(values) < 2:
        raise ValueError("need at least two supervised source clips")
    rng, deltas = np.random.default_rng(2026), []
    for start in range(0, repeats, 256):
        ids = rng.integers(len(values), size=(min(256, repeats - start), len(values)))
        sums = values[ids].sum(axis=1)
        deltas.extend((sums[:, 2] / sums[:, 3] - sums[:, 0] / sums[:, 1]).tolist())
    ref, new = raw_mean(reference.records), raw_mean(candidate.records)
    return dict(metric=METRIC, reference=ref, candidate=new, delta_m=new - ref,
                relative_improvement=(ref - new) / ref if ref > 0 else None,
                delta_ci95_m=np.quantile(deltas, [0.025, 0.975]).tolist(),
                bootstrap_seed=2026, bootstrap_replicates=repeats,
                supervised_source_clips=len(values), zero_supervision_clips=len(grouped) - len(values),
                uncertainty_scope="source-clip resampling conditional on fixed trained checkpoints; not training-seed robustness")
