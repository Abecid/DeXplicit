# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from dataset_components.constants import (
    DEFAULT_SOFT_SELECTOR_TAU,
    DEFAULT_SOFT_SELECTOR_TEMP_SCALE,
    RELEASE_CONTEXT_HORIZON,
    RELEASE_MAX_RELATIVE_MOVEMENT,
    RELEASE_PRED_HORIZON,
    RELEASE_WEIGHT_GAMMA,
)
from dataset_components.mano_sidecar import (
    load_mano_sidecar_arrays,
    merge_mano_sidecar_arrays,
    resolve_mano_sidecar_path,
)
from utils import make_soft_selector_labels


POINTWORLD_HORIZON = RELEASE_CONTEXT_HORIZON + RELEASE_PRED_HORIZON
TEMPORAL_ARRAY_KEYS = {
    "scene_flows",
    "scene_colors",
    "scene_normals",
    "scene_visibility",
    "scene_depth_valid_mask",
    "scene_transition_valid",
    "robot_flows",
    "robot_colors",
    "robot_normals",
    "left_gripper_open",
    "right_gripper_open",
    "hand_joints_world",
    "hand_joint_valid",
    "hand_action_ids",
    "mano_pose_coeffs",
    "mano_hand_pose",
    "mano_thetas",
    "mano_betas",
    "mano_wrist_xform",
    "mano_wrist_transform",
    "mano_global_orient",
    "mano_translation",
    "mano_transl",
    "mano_translation_xyz",
    "mano_rot6d",
    "mano_valid",
    "robot_features",
    "robot_valid",
    "gt_scene_flows",
    "gt_scene_flows_relative",
    "point_weights",
}
OPTIONAL_MANO_ARRAY_KEYS = {
    "mano_pose_coeffs",
    "mano_hand_pose",
    "mano_thetas",
    "mano_betas",
    "mano_wrist_xform",
    "mano_wrist_transform",
    "mano_global_orient",
    "mano_translation",
    "mano_transl",
    "mano_translation_xyz",
    "mano_rot6d",
    "mano_valid",
}
SOURCE_CLIP_METADATA_KEYS = (
    "source_clip_id",
    "clip_id",
    "clip_uid",
    "source_id",
    "sequence_id",
)

FINGER_DENSE_RATIO = 0.5
FINGER_REGION_PERCENTILE = 45.0
FINGERTIP_SEED_RATIO = 0.25
FINGER_CHAINS_21 = (
    (1, 2, 3, 4),
    (5, 6, 7, 8),
    (9, 10, 11, 12),
    (13, 14, 15, 16),
    (17, 18, 19, 20),
)
FINGER_CHAINS_16 = (
    (1, 2, 3),
    (4, 5, 6),
    (7, 8, 9),
    (10, 11, 12),
    (13, 14, 15),
)


@dataclass(frozen=True)
class HOIDatasetConfig:
    max_scene_points: int
    max_robot_points: int
    target_horizon: int = POINTWORLD_HORIZON
    deterministic: bool = False
    require_dino_features: bool = True
    seed: int = 0
    domain: str = "hand_world"
    mano_sidecar_root: str | Path | None = None


def _as_path_list(roots: Iterable[str | Path]) -> list[Path]:
    return [Path(root).expanduser() for root in roots]


def _discover_npz_samples(roots: Iterable[str | Path]) -> list[Path]:
    samples: list[Path] = []
    for root in _as_path_list(roots):
        if root.is_file() and root.suffix == ".npz":
            samples.append(root)
        elif root.is_dir():
            samples.extend(sorted(p for p in root.glob("*.npz") if p.is_file()))
        else:
            continue
    return sorted(samples)


def _lexical_absolute(path: Path) -> Path:
    """Absolute path without resolving a campaign inventory symlink."""

    return Path(os.path.abspath(path.expanduser()))


def _source_clip_from_metadata(metadata_json: str) -> str | None:
    try:
        metadata = json.loads(metadata_json)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(metadata, dict):
        return None
    for key in SOURCE_CLIP_METADATA_KEYS:
        value = metadata.get(key)
        if value is None or isinstance(value, bool):
            continue
        normalized = str(value).strip()
        if normalized:
            return normalized
    return None


def _campaign_source_clip_map(roots: Iterable[str | Path]) -> dict[Path, str]:
    """Load authoritative window-to-source mappings from campaign manifests.

    The split builder writes lexical symlink paths into ``split_manifest.json``.
    Resolving those symlinks would lose which validation/final inventory a
    window came from, so mappings intentionally use non-resolved absolute paths.
    """

    manifests: set[Path] = set()
    for root in _as_path_list(roots):
        directory = root if root.is_dir() else root.parent
        for candidate in (directory / "split_manifest.json", directory.parent / "split_manifest.json"):
            if candidate.is_file():
                manifests.add(candidate.resolve())
    mapping: dict[Path, str] = {}
    for manifest_path in sorted(manifests):
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("split_kind") != "source_clip_grouped_validation_final":
            continue
        subsets = payload.get("subsets")
        if not isinstance(subsets, dict):
            raise ValueError(f"campaign split manifest has invalid subsets: {manifest_path}")
        for subset in subsets.values():
            if not isinstance(subset, dict):
                raise ValueError(f"campaign split manifest has invalid subset: {manifest_path}")
            for entry in subset.get("windows", []):
                if not isinstance(entry, dict):
                    raise ValueError(f"campaign split manifest has invalid window: {manifest_path}")
                source_clip = entry.get("source_clip")
                if source_clip is None or isinstance(source_clip, bool) or not str(source_clip).strip():
                    raise ValueError(f"campaign window has no source_clip: {manifest_path}")
                window = _lexical_absolute(manifest_path.parent / str(entry.get("path", "")))
                normalized = str(source_clip).strip()
                previous = mapping.get(window)
                if previous is not None and previous != normalized:
                    raise ValueError(
                        f"campaign manifests disagree for {window}: {previous!r} vs {normalized!r}"
                    )
                mapping[window] = normalized
    return mapping


def _rng_for_sample(config: HOIDatasetConfig, index: int, salt: int) -> np.random.Generator:
    if config.deterministic:
        return np.random.default_rng(int(config.seed) + index * 1009 + salt)
    return np.random.default_rng()


def _time_indices(total: int, target: int, rng: np.random.Generator, deterministic: bool) -> np.ndarray:
    if total <= 0:
        raise ValueError("HOI samples must contain at least one timestep.")
    if total >= target:
        return np.arange(0, target, dtype=np.int64)
    pad = np.full(target - total, total - 1, dtype=np.int64)
    return np.concatenate([np.arange(total, dtype=np.int64), pad], axis=0)


def _point_indices(total: int, target: int, rng: np.random.Generator) -> np.ndarray:
    if total <= target:
        return np.arange(total, dtype=np.int64)
    return np.sort(rng.choice(total, size=target, replace=False)).astype(np.int64)


def _hand_point_budget(total: int) -> tuple[int, int]:
    total = max(int(total), 0)
    left = total // 2
    return left, total - left


def _finger_chains_for_joint_count(joint_count: int) -> tuple[tuple[int, ...], ...]:
    if joint_count >= 21:
        return FINGER_CHAINS_21
    if joint_count >= 16:
        return FINGER_CHAINS_16
    raise ValueError(f"Expected at least 16 MANO joints for finger-dense sampling, got {joint_count}")


def _point_to_segment_distance(points: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    segment = end - start
    denom = float(np.dot(segment, segment))
    if denom <= 1e-10:
        return np.linalg.norm(points - start[None, :], axis=1)
    t = np.clip(((points - start[None, :]) @ segment) / denom, 0.0, 1.0)
    closest = start[None, :] + t[:, None] * segment[None, :]
    return np.linalg.norm(points - closest, axis=1)


def _min_distance_to_segments(
    points: np.ndarray,
    joints: np.ndarray,
    segments: list[tuple[int, int]],
) -> np.ndarray:
    distances = [_point_to_segment_distance(points, joints[start], joints[end]) for start, end in segments]
    return np.min(np.stack(distances, axis=1), axis=1)


def _farthest_point_sample(points: np.ndarray, target: int, seeds: np.ndarray | None = None) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    num_points = int(points.shape[0])
    target = int(target)
    if target <= 0 or num_points == 0:
        return np.zeros((0,), dtype=np.int64)
    if target >= num_points:
        return np.arange(num_points, dtype=np.int64)

    selected: list[int] = []
    seen: set[int] = set()
    if seeds is not None:
        for idx in np.asarray(seeds, dtype=np.int64).reshape(-1):
            idx_int = int(idx)
            if 0 <= idx_int < num_points and idx_int not in seen:
                selected.append(idx_int)
                seen.add(idx_int)
                if len(selected) == target:
                    return np.asarray(selected, dtype=np.int64)

    if not selected:
        centroid = points.mean(axis=0)
        first = int(np.argmax(np.linalg.norm(points - centroid[None, :], axis=-1)))
        selected.append(first)
        seen.add(first)

    min_dist = np.full((num_points,), np.inf, dtype=np.float32)
    for idx in selected:
        min_dist = np.minimum(min_dist, np.linalg.norm(points - points[idx], axis=-1))
        min_dist[idx] = -1.0

    while len(selected) < target:
        idx = int(np.argmax(min_dist))
        if idx in seen:
            break
        selected.append(idx)
        seen.add(idx)
        min_dist = np.minimum(min_dist, np.linalg.norm(points - points[idx], axis=-1))
        min_dist[idx] = -1.0
    return np.asarray(selected, dtype=np.int64)


def _sample_candidate_indices(
    points: np.ndarray,
    candidates: np.ndarray,
    target: int,
    seed_indices: np.ndarray | None = None,
) -> np.ndarray:
    candidates = np.asarray(candidates, dtype=np.int64).reshape(-1)
    target = int(target)
    if target <= 0 or candidates.size == 0:
        return np.zeros((0,), dtype=np.int64)
    if candidates.size <= target:
        return candidates.astype(np.int64)

    seed_local = None
    if seed_indices is not None and seed_indices.size:
        local_lookup = {int(global_idx): local_idx for local_idx, global_idx in enumerate(candidates.tolist())}
        seed_local = np.asarray(
            [local_lookup[int(global_idx)] for global_idx in seed_indices.tolist() if int(global_idx) in local_lookup],
            dtype=np.int64,
        )
    sampled_local = _farthest_point_sample(points[candidates], target, seeds=seed_local)
    return candidates[sampled_local].astype(np.int64)


def _valid_reference_frame(points: np.ndarray, valid: np.ndarray | None) -> int:
    if points.shape[0] == 0:
        return 0
    nonzero = np.linalg.norm(points, axis=-1) > 1e-8
    if valid is not None and valid.shape == points.shape[:2]:
        usable = (valid > 0.5) | nonzero
    else:
        usable = nonzero
    counts = usable.sum(axis=1)
    return int(np.argmax(counts))


def _finger_dense_subsample_slots(
    hand_points: np.ndarray,
    hand_joints: np.ndarray | None,
    hand_valid: np.ndarray | None,
    target: int,
) -> np.ndarray:
    target = int(target)
    source = int(hand_points.shape[1])
    if target <= 0:
        return np.zeros((0,), dtype=np.int64)
    if source <= target:
        return np.arange(source, dtype=np.int64)

    ref_t = _valid_reference_frame(hand_points, hand_valid)
    ref_points = np.asarray(hand_points[ref_t], dtype=np.float32)
    usable = np.linalg.norm(ref_points, axis=-1) > 1e-8
    if hand_valid is not None and hand_valid.shape == hand_points.shape[:2]:
        usable = usable | (hand_valid[ref_t] > 0.5)
    if not usable.any():
        return _farthest_point_sample(ref_points, target)

    usable_indices = np.flatnonzero(usable).astype(np.int64)
    if usable_indices.size <= target:
        selected = usable_indices.tolist()
        selected_set = set(selected)
        selected.extend(int(idx) for idx in range(source) if idx not in selected_set)
        return np.asarray(selected[:target], dtype=np.int64)

    if hand_joints is None or hand_joints.ndim != 3:
        return _sample_candidate_indices(ref_points, usable_indices, target)

    joints = np.asarray(hand_joints[ref_t], dtype=np.float32)
    try:
        finger_chains = _finger_chains_for_joint_count(joints.shape[0])
        finger_segments = [(start, end) for chain in finger_chains for start, end in zip(chain[:-1], chain[1:])]
        distal_segments = [(chain[-2], chain[-1]) for chain in finger_chains]
        tip_joints = tuple(chain[-1] for chain in finger_chains)

        finger_distance = _min_distance_to_segments(ref_points, joints, finger_segments)
        finite_usable = usable & np.isfinite(finger_distance)
        if not finite_usable.any():
            return _sample_candidate_indices(ref_points, usable_indices, target)
        finger_threshold = np.percentile(finger_distance[finite_usable], FINGER_REGION_PERCENTILE)
        finger_candidates = np.flatnonzero(finite_usable & (finger_distance <= finger_threshold)).astype(np.int64)
        min_finger_pool = min(
            int(usable_indices.size),
            max(int(round(target * FINGER_DENSE_RATIO)), int(round(usable_indices.size * FINGER_DENSE_RATIO))),
        )
        if finger_candidates.size < min_finger_pool:
            usable_by_finger_distance = usable_indices[np.argsort(finger_distance[usable_indices])]
            finger_candidates = usable_by_finger_distance[:min_finger_pool].astype(np.int64)
        rest_candidates = np.setdiff1d(usable_indices, finger_candidates, assume_unique=True)
    except (ValueError, IndexError, FloatingPointError):
        return _sample_candidate_indices(ref_points, usable_indices, target)

    finger_target = min(int(round(target * FINGER_DENSE_RATIO)), int(finger_candidates.size))
    rest_target = target - finger_target
    if rest_candidates.size < rest_target:
        finger_target = min(target - int(rest_candidates.size), int(finger_candidates.size))
        rest_target = target - finger_target

    distal_distance = _min_distance_to_segments(ref_points, joints, distal_segments)
    tip_distance = np.min(
        np.linalg.norm(ref_points[:, None, :] - joints[np.asarray(tip_joints, dtype=np.int64)][None, :, :], axis=2),
        axis=1,
    )
    fingertip_score = distal_distance + tip_distance
    seed_count = min(
        finger_target,
        max(len(tip_joints) * 4, int(round(finger_target * FINGERTIP_SEED_RATIO))),
        int(finger_candidates.size),
    )
    seed_indices = finger_candidates[np.argsort(fingertip_score[finger_candidates])[:seed_count]]
    selected = np.concatenate(
        [
            _sample_candidate_indices(ref_points, finger_candidates, finger_target, seed_indices=seed_indices),
            _sample_candidate_indices(ref_points, rest_candidates, rest_target),
        ],
        axis=0,
    )
    if selected.size < target:
        remaining = np.setdiff1d(usable_indices, selected, assume_unique=False)
        selected = np.concatenate(
            [selected, _sample_candidate_indices(ref_points, remaining, target - int(selected.size))],
            axis=0,
        )
    return selected[:target].astype(np.int64)


def _robot_point_indices(arrays: dict[str, np.ndarray], target: int, rng: np.random.Generator) -> np.ndarray:
    robot_flows = np.asarray(arrays["robot_flows"])
    total = int(robot_flows.shape[1])
    target = int(target)
    if total <= target:
        return np.arange(total, dtype=np.int64)
    if target <= 0:
        return np.zeros((0,), dtype=np.int64)

    if bool(np.asarray(arrays.get("action_is_unlabelled", False)).item()):
        # PhysTwin: first group is an aggregate control set, second is padding.
        # Do not apply a MANO finger-biased sampler to non-anatomical tracks.
        if total % 2 or target < 2:
            raise ValueError("tracked controls require two storage slots")
        half = total // 2
        count = min(half, target // 2)
        left = _farthest_point_sample(robot_flows[0, :half], count)
        right = np.arange(min(half, target-count), dtype=np.int64) + half
        return np.concatenate((left, right)).astype(np.int64)

    hand_joints = arrays.get("hand_joints_world")
    if hand_joints is None:
        return _point_indices(total, target, rng)
    hand_joints = np.asarray(hand_joints)
    if hand_joints.ndim != 4 or hand_joints.shape[1] < 2:
        return _point_indices(total, target, rng)

    source_left_count, source_right_count = _hand_point_budget(total)
    target_left_count, target_right_count = _hand_point_budget(target)
    robot_valid = arrays.get("robot_valid")
    robot_valid = np.asarray(robot_valid) if isinstance(robot_valid, np.ndarray) else None

    left_points = robot_flows[:, :source_left_count]
    right_points = robot_flows[:, source_left_count : source_left_count + source_right_count]
    left_valid = robot_valid[:, :source_left_count] if robot_valid is not None and robot_valid.shape == robot_flows.shape[:2] else None
    right_valid = (
        robot_valid[:, source_left_count : source_left_count + source_right_count]
        if robot_valid is not None and robot_valid.shape == robot_flows.shape[:2]
        else None
    )

    left_local = _finger_dense_subsample_slots(left_points, hand_joints[:, 0], left_valid, target_left_count)
    right_local = _finger_dense_subsample_slots(right_points, hand_joints[:, 1], right_valid, target_right_count)
    return np.concatenate([left_local, source_left_count + right_local], axis=0).astype(np.int64)


def _select_time(array: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return np.asarray(array)[indices]


def _is_temporal_array(key: str, value: np.ndarray, original_timesteps: int) -> bool:
    if value.ndim < 1 or value.shape[0] != original_timesteps:
        return False
    if key == "scene_features":
        return value.ndim == 3
    if key == "scene_dino_features":
        return value.ndim == 3
    return key in TEMPORAL_ARRAY_KEYS


def _select_or_pad_time_points(array: np.ndarray, target: int, indices: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    if array.shape[1] >= target:
        return array[:, indices]
    out_shape = (array.shape[0], target) + array.shape[2:]
    out = np.zeros(out_shape, dtype=array.dtype)
    out[:, : array.shape[1], ...] = array
    return out


def _select_or_pad_robot_time_points(
    array: np.ndarray,
    target: int,
    indices: np.ndarray,
) -> np.ndarray:
    """Select robot points while preserving the left/right half boundary.

    HOT3D robot surface points are stored as ``left || right`` and downstream
    code splits the fixed-width tensor at its midpoint.  Generic end-padding
    would shift part of the raw right hand into the padded left half whenever
    the raw point count is below ``target``.  Pad each hand inside its own
    target half instead.
    """

    array = np.asarray(array)
    target = int(target)
    if target < 0:
        raise ValueError("robot point target must be non-negative")
    source_count = int(array.shape[1])
    if source_count >= target:
        return array[:, indices]
    expected = np.arange(source_count, dtype=np.int64)
    if not np.array_equal(np.asarray(indices, dtype=np.int64), expected):
        raise ValueError(
            "undersized robot tensors require the complete ordered point inventory"
        )

    source_left, source_right = _hand_point_budget(source_count)
    target_left, target_right = _hand_point_budget(target)
    if source_left > target_left or source_right > target_right:
        raise ValueError("robot half cannot fit inside its target hand budget")
    out_shape = (array.shape[0], target) + array.shape[2:]
    out = np.zeros(out_shape, dtype=array.dtype)
    out[:, :source_left, ...] = array[:, :source_left, ...]
    right_source = array[:, source_left : source_left + source_right, ...]
    out[:, target_left : target_left + source_right, ...] = right_source
    return out


def _select_or_pad_points(array: np.ndarray, target: int, indices: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    if array.shape[0] >= target:
        return array[indices]
    out_shape = (target,) + array.shape[1:]
    out = np.zeros(out_shape, dtype=array.dtype)
    out[: array.shape[0], ...] = array
    return out


def _scene_feature_tensor(arrays: dict[str, np.ndarray], scene_idx: np.ndarray, config: HOIDatasetConfig) -> np.ndarray:
    if "scene_features" in arrays:
        scene_features = arrays["scene_features"].astype(np.float32)
        if scene_features.ndim == 2:
            scene_features = scene_features[None, ...]
        if scene_features.ndim != 3:
            raise ValueError(f"scene_features must have shape (N,D) or (T,N,D), got {scene_features.shape}")
        first_features = scene_features[:1]
        return _select_or_pad_time_points(first_features, config.max_scene_points, scene_idx)

    scene_visibility = arrays["scene_visibility"].astype(np.float32)
    parts = [
        arrays["scene_flows"][0].astype(np.float32),
        arrays["scene_colors"][0].astype(np.float32),
        arrays["scene_normals"][0].astype(np.float32),
        scene_visibility[0, :, None],
    ]
    if "scene_dino_features" in arrays:
        dino = arrays["scene_dino_features"].astype(np.float32)
        if dino.ndim == 3:
            dino = dino[0]
        if dino.ndim != 2:
            raise ValueError(f"scene_dino_features must have shape (N,D) or (T,N,D), got {dino.shape}")
        parts.append(dino)
    elif config.require_dino_features:
        raise ValueError("HOI sample is missing required scene_dino_features.")

    feature = np.concatenate(parts, axis=-1)
    feature = _select_or_pad_points(feature, config.max_scene_points, scene_idx)
    return feature[None, ...].astype(np.float32)


def _robot_feature_tensor(arrays: dict[str, np.ndarray], robot_idx: np.ndarray, config: HOIDatasetConfig) -> np.ndarray:
    if "robot_features" in arrays:
        robot_features = arrays["robot_features"].astype(np.float32)
        if robot_features.ndim != 3:
            raise ValueError(f"robot_features must have shape (T,N,D), got {robot_features.shape}")
        if robot_features.shape[0] != arrays["robot_flows"].shape[0]:
            raise ValueError(
                f"robot_features timesteps ({robot_features.shape[0]}) must match robot_flows "
                f"({arrays['robot_flows'].shape[0]})"
            )
        return _select_or_pad_robot_time_points(
            robot_features, config.max_robot_points, robot_idx
        )

    robot_flows = arrays["robot_flows"].astype(np.float32)
    robot_colors = arrays.get("robot_colors", np.zeros_like(robot_flows)).astype(np.float32)
    robot_normals = arrays.get("robot_normals", np.zeros_like(robot_flows)).astype(np.float32)
    T = robot_flows.shape[0]
    left_open = arrays.get("left_gripper_open", np.zeros((T, 1), dtype=np.float32)).astype(np.float32)
    right_open = arrays.get("right_gripper_open", np.zeros((T, 1), dtype=np.float32)).astype(np.float32)
    robot_open = np.concatenate([left_open.reshape(T, -1)[:, :1], right_open.reshape(T, -1)[:, :1]], axis=-1)
    robot_open = np.repeat(robot_open[:, None, :], robot_flows.shape[1], axis=1)

    feature = np.concatenate([robot_flows, robot_colors, robot_normals, robot_open], axis=-1)
    return _select_or_pad_robot_time_points(
        feature, config.max_robot_points, robot_idx
    ).astype(np.float32)


def _metadata_json(path: Path) -> str:
    meta_path = path.with_suffix(".json")
    if meta_path.is_file():
        return meta_path.read_text()
    return json.dumps({"source_npz": str(path)})


def _dataset_label_from_metadata(metadata_json: str, path: Path) -> str:
    try:
        metadata = json.loads(metadata_json)
    except json.JSONDecodeError:
        metadata = {}
    label = metadata.get("dataset")
    if isinstance(label, str) and label.strip():
        return label.strip()

    text = str(path).lower()
    for candidate in ("hot3d", "h2o", "hocap", "dexycb", "oakink2", "trumans"):
        if candidate in text:
            return candidate
    return path.parent.name or "unknown"


def _compute_helper_variables(sample: dict, max_relative_movement: float) -> dict:
    gt_scene_flows = sample["gt_scene_flows"]
    gt_scene_flows_delta = gt_scene_flows[1:] - gt_scene_flows[:-1]
    zeros_first_frame = np.zeros_like(gt_scene_flows[:1])
    gt_scene_flows_delta = np.concatenate([zeros_first_frame, gt_scene_flows_delta], axis=0)
    gt_scene_flows_relative = gt_scene_flows - gt_scene_flows[0:1]

    if "scene_transition_valid" in sample:
        pair = np.zeros(gt_scene_flows.shape[:2], dtype=bool)
        position = sample["scene_visibility"].astype(bool) & sample["scene_depth_valid_mask"].astype(bool)
        pair[1:] = position[:-1] & position[1:] & sample["scene_transition_valid"][:-1].astype(bool)
        gt_scene_flows_delta = np.where(pair[..., None], gt_scene_flows_delta, 0)
        sample["scene_delta_valid"] = pair
    if "scene_delta_valid" not in sample:
        good = sample["scene_visibility"].astype(bool) & sample["scene_depth_valid_mask"].astype(bool)
        pair = np.zeros_like(good, dtype=bool)
        pair[1:] = good[1:] & good[:-1]
        sample["scene_delta_valid"] = pair
    delta_norm = np.linalg.norm(gt_scene_flows_delta, axis=-1)
    scene_selector_gt = make_soft_selector_labels(
        delta_norm,
        DEFAULT_SOFT_SELECTOR_TAU,
        DEFAULT_SOFT_SELECTOR_TEMP_SCALE,
    )
    moved_mask = scene_selector_gt > 0.5
    sample["gt_scene_flows_delta"] = gt_scene_flows_delta
    sample["gt_scene_flows_delta_norm"] = delta_norm
    sample["gt_scene_flows_relative"] = np.clip(
        gt_scene_flows_relative,
        -max_relative_movement,
        max_relative_movement,
    )
    sample["scene_selector_gt"] = scene_selector_gt
    sample["scene_moved_mask"] = moved_mask
    sample["scene_static_mask"] = ~moved_mask
    return sample


class HandObjectInteractionDataset(Dataset):
    """Direct NPZ loader for hand-object interaction samples.

    The returned tensors follow the batch contract consumed by the released
    PointWorld trainer. Samples may be raw HOI exports or materialized tensors
    from `hoi-point-world/scripts/process/materialize_pointworld_training_samples.py`.
    """

    def __init__(self, roots: Iterable[str | Path], config: HOIDatasetConfig):
        self.config = config
        roots = list(roots)
        self.samples = _discover_npz_samples(roots)
        self.source_clip_by_path = _campaign_source_clip_map(roots)
        if not self.samples:
            raise FileNotFoundError(f"No .npz samples found under: {[str(p) for p in _as_path_list(roots)]}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        path = self.samples[index]
        # ``scene_dino_features`` is ~98% of every npz and a single DEFLATE stream
        # spanning all timesteps, but only the context frame is ever consumed. Load
        # all other (tiny) arrays here and fetch the one needed DINO frame lazily via
        # an uncompressed sidecar below, avoiding a full ~170MB decompress per sample.
        npz = np.load(path, allow_pickle=False)
        try:
            member_names = set(npz.files)
            arrays = {k: npz[k] for k in member_names if k != "scene_dino_features"}

            required = {"scene_flows", "robot_flows"}
            missing = sorted(required - arrays.keys())
            if missing:
                raise ValueError(f"{path} is missing required HOI arrays: {missing}")
            if arrays["scene_flows"].shape[0] != arrays["robot_flows"].shape[0]:
                raise ValueError(
                    f"{path} has mismatched scene/robot timesteps: "
                    f"{arrays['scene_flows'].shape[0]} vs {arrays['robot_flows'].shape[0]}"
                )

            if bool(np.asarray(arrays.get("action_is_unlabelled", False)).item()) and self.config.deterministic:
                # Caches cannot depend on which CSV split/subset is loaded.
                import hashlib
                index = int.from_bytes(hashlib.sha256(path.stem.encode()).digest()[:4], "little")
            time_rng = _rng_for_sample(self.config, index, 11)
            scene_rng = _rng_for_sample(self.config, index, 23)
            robot_rng = _rng_for_sample(self.config, index, 37)
            original_timesteps = arrays["scene_flows"].shape[0]
            if bool(np.asarray(arrays.get("action_is_unlabelled", False)).item()):
                if original_timesteps != self.config.target_horizon:
                    raise ValueError("PhysTwin export horizon must match checkpoint horizon; reprocess instead of resampling masks")
            mano_sidecar_path = resolve_mano_sidecar_path(path, self.config.mano_sidecar_root)
            if mano_sidecar_path is not None and mano_sidecar_path.is_file():
                mano_arrays = load_mano_sidecar_arrays(mano_sidecar_path)
                arrays = merge_mano_sidecar_arrays(
                    arrays,
                    mano_arrays,
                    expected_timesteps=original_timesteps,
                )
            time_idx = _time_indices(
                original_timesteps,
                self.config.target_horizon,
                time_rng,
                self.config.deterministic,
            )

            time_valid = np.ones((self.config.target_horizon,), dtype=bool)
            if original_timesteps < self.config.target_horizon:
                time_valid[original_timesteps:] = False
            for key, value in list(arrays.items()):
                if isinstance(value, np.ndarray) and _is_temporal_array(key, value, original_timesteps):
                    arrays[key] = _select_time(value, time_idx)

            # Inject only the consumed DINO context frame (original frame ``time_idx[0]``)
            # as a single (N, D) array, matching the prior ``dino[time_idx][0]`` semantics.
            if "scene_dino_features" in member_names:
                arrays["scene_dino_features"] = self._load_dino_context_frame(
                    path, npz, original_timesteps, int(time_idx[0])
                )

            return self._assemble_sample(
                path=path,
                arrays=arrays,
                scene_rng=scene_rng,
                robot_rng=robot_rng,
                time_valid=time_valid,
            )
        finally:
            npz.close()

    def _assemble_sample(self, path, arrays, scene_rng, robot_rng, time_valid) -> dict:
        scene_count = arrays["scene_flows"].shape[1]
        robot_count = arrays["robot_flows"].shape[1]
        scene_idx = _point_indices(scene_count, self.config.max_scene_points, scene_rng)
        robot_idx = _robot_point_indices(arrays, self.config.max_robot_points, robot_rng)

        scene_flows = _select_or_pad_time_points(
            arrays["scene_flows"].astype(np.float32),
            self.config.max_scene_points,
            scene_idx,
        )
        robot_flows = _select_or_pad_robot_time_points(
            arrays["robot_flows"].astype(np.float32),
            self.config.max_robot_points,
            robot_idx,
        )
        if "scene_visibility" in arrays:
            scene_visibility = _select_or_pad_time_points(
                arrays["scene_visibility"].astype(np.float32),
                self.config.max_scene_points,
                scene_idx,
            )
        else:
            scene_visibility = np.ones(scene_flows.shape[:2], dtype=np.float32)
            if scene_count < self.config.max_scene_points:
                scene_visibility[:, scene_count:] = 0.0
        scene_visibility[~time_valid] = 0.0

        if "scene_depth_valid_mask" in arrays:
            scene_depth_valid_mask = _select_or_pad_time_points(
                arrays["scene_depth_valid_mask"].astype(np.float32),
                self.config.max_scene_points,
                scene_idx,
            )
        else:
            scene_depth_valid_mask = scene_visibility.copy()
        scene_depth_valid_mask[~time_valid] = 0.0

        if "robot_valid" in arrays and arrays["robot_valid"].shape == arrays["robot_flows"].shape[:2]:
            robot_exists = _select_or_pad_robot_time_points(
                arrays["robot_valid"].astype(np.float32),
                self.config.max_robot_points,
                robot_idx,
            ).astype(bool)
        else:
            raw_exists = np.ones(
                arrays["robot_flows"].shape[:2], dtype=np.float32
            )
            robot_exists = _select_or_pad_robot_time_points(
                raw_exists,
                self.config.max_robot_points,
                robot_idx,
            ).astype(bool)
        robot_exists[~time_valid] = False

        metadata_json = _metadata_json(path)
        metadata_source_clip = _source_clip_from_metadata(metadata_json)
        manifest_source_clip = self.source_clip_by_path.get(_lexical_absolute(path))
        if (
            metadata_source_clip is not None
            and manifest_source_clip is not None
            and metadata_source_clip != manifest_source_clip
        ):
            raise ValueError(
                f"sample metadata and campaign manifest disagree on source clip for {path}: "
                f"{metadata_source_clip!r} vs {manifest_source_clip!r}"
            )
        source_clip_id = metadata_source_clip or manifest_source_clip
        sample = {
            "__key__": path.stem,
            "__domain__": self.config.domain,
            "__dataset__": _dataset_label_from_metadata(metadata_json, path),
            "scene_flows": scene_flows,
            "scene_visibility": scene_visibility,
            "scene_depth_valid_mask": scene_depth_valid_mask,
            "scene_features": _scene_feature_tensor(arrays, scene_idx, self.config),
            "robot_flows": robot_flows,
            "robot_features": _robot_feature_tensor(arrays, robot_idx, self.config),
            "gt_scene_flows": scene_flows.copy(),
            "scene_exists": scene_visibility.astype(bool),
            "robot_exists": robot_exists,
            "metadata_json": metadata_json,
        }
        if source_clip_id is not None:
            sample["source_clip_id"] = source_clip_id

        sample["action_is_unlabelled"] = np.asarray(
            bool(np.asarray(arrays.get("action_is_unlabelled", False)).item()), dtype=bool
        )
        if "scene_transition_valid" in arrays:
            sample["scene_transition_valid"] = _select_or_pad_time_points(
                arrays["scene_transition_valid"].astype(bool), self.config.max_scene_points, scene_idx
            )
        if sample["action_is_unlabelled"]:
            if robot_count % 2 or arrays["robot_valid"][:, robot_count//2:].any():
                raise ValueError("unlabelled controls require aggregate prefix + invalid second group")
            select_robot = globals().get("_select_or_pad_robot_time_points", _select_or_pad_time_points)
            for key in ("robot_point_id", "robot_hand_side"):
                selected = select_robot(np.asarray(arrays[key])[None], self.config.max_robot_points, robot_idx)[0]
                sample[key] = np.where(robot_exists.any(0), selected, -1).astype(np.int64)

        sample["scene_context_mask"] = np.ones((*scene_flows.shape[:2], 1), dtype=np.float32)
        sample["scene_context_mask"][RELEASE_CONTEXT_HORIZON:] = 0.0
        sample = _compute_helper_variables(
            sample,
            max_relative_movement=RELEASE_MAX_RELATIVE_MOVEMENT,
        )
        sample["scene_supervised_mask"] = (
            sample["scene_visibility"].astype(bool) & sample["scene_depth_valid_mask"].astype(bool)
        )

        if "scene_point_id" in arrays:
            point_ids = np.full((self.config.max_scene_points,), -1, dtype=np.int64)
            point_ids[:len(scene_idx)] = np.asarray(arrays["scene_point_id"], dtype=np.int64)[scene_idx]
            sample["point_track_id"] = point_ids

        if "hand_joints_world" in arrays:
            sample["hand_joints_world"] = arrays["hand_joints_world"].astype(np.float32)
        if "hand_joint_valid" in arrays:
            sample["hand_joint_valid"] = arrays["hand_joint_valid"].astype(np.float32)
        if "hand_action_ids" in arrays:
            sample["hand_action_ids"] = arrays["hand_action_ids"].astype(np.int64)
        # Optional MANO parameter arrays for articulation-aware hand encoders.
        # Server-side data agents can add any of these keys to processed NPZs; older
        # datasets remain valid because the model fills missing MANO params with zeros.
        for key in (
            "mano_pose_coeffs",
            "mano_hand_pose",
            "mano_thetas",
            "mano_betas",
            "mano_wrist_xform",
            "mano_wrist_transform",
            "mano_global_orient",
            "mano_translation",
            "mano_transl",
            "mano_translation_xyz",
            "mano_rot6d",
            "mano_valid",
        ):
            if key in arrays:
                sample[key] = arrays[key].astype(np.float32)

        return {
            key: torch.from_numpy(value) if isinstance(value, np.ndarray) else value
            for key, value in sample.items()
        }

    def _dino_sidecar_path(self, path: Path) -> Path:
        return dino_sidecar_path(path, self.config.target_horizon)

    def _load_dino_context_frame(
        self, path: Path, npz, original_timesteps: int, ctx_frame: int
    ) -> np.ndarray:
        """Return the single consumed DINO context frame ``(N, D)``.

        HOI training is anchored on the initial observation, so the context frame
        is always source frame 0. The sidecar therefore stores only that DINO row.
        """
        side = self._dino_sidecar_path(path)

        cached = None
        if side.is_file():
            try:
                mm = np.load(side, mmap_mode="r")
                if mm.ndim == 3 and mm.shape[0] >= 1:
                    cached = mm
            except Exception:
                cached = None  # corrupt/partial sidecar: fall back to rebuild

        if cached is None:
            dino = np.asarray(npz["scene_dino_features"], dtype=np.float32)
            if dino.ndim == 2:
                dino = dino[None, ...]
            if dino.ndim != 3:
                raise ValueError(
                    f"scene_dino_features must have shape (N,D) or (T,N,D), got {dino.shape}"
                )
            cached = np.ascontiguousarray(dino[:1])
            _atomic_write_npy(side, cached)

        # Copy out of the (possibly memory-mapped) array so only this frame is paged in.
        return np.array(cached[0], dtype=np.float32)


def dino_sidecar_path(npz_path: Path, target_horizon: int) -> Path:
    # Horizon-specific name preserves compatibility with existing training cache
    # warmup and avoids silently reusing caches across different CLI settings.
    npz_path = Path(npz_path)
    return npz_path.with_name(f"{npz_path.stem}.dinoctx_h{target_horizon}.npy")


def _atomic_write_npy(dest: Path, array: np.ndarray) -> None:
    """Write ``array`` to ``dest`` as an uncompressed .npy via a unique temp file
    plus atomic rename, so concurrent writers can never observe or produce a
    partially written sidecar. Caching failures are non-fatal."""
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(dest.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as handle:
                # File-handle form: numpy does not append ".npy" and does not compress.
                np.save(handle, array)
            os.replace(tmp, dest)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
    except Exception:
        # e.g. read-only filesystem: training continues with in-memory frames.
        pass


def build_dino_sidecar(npz_path: Path, target_horizon: int) -> str:
    """Build the uncompressed DINO context-frame sidecar for one clip if needed.

    Idempotent and safe to call concurrently. Returns one of:
    ``"built"``, ``"skipped"`` (already valid), or ``"no_dino"`` (clip has none).
    """
    npz_path = Path(npz_path)
    side = dino_sidecar_path(npz_path, target_horizon)
    npz = np.load(npz_path, allow_pickle=False)
    try:
        if "scene_dino_features" not in npz.files:
            return "no_dino"
        if side.is_file():
            try:
                mm = np.load(side, mmap_mode="r")
                if mm.ndim == 3 and mm.shape[0] >= 1:
                    return "skipped"
            except Exception:
                pass  # rebuild corrupt/partial sidecar
        dino = np.asarray(npz["scene_dino_features"], dtype=np.float32)
        if dino.ndim == 2:
            dino = dino[None, ...]
        keep = np.ascontiguousarray(dino[:1])
        _atomic_write_npy(side, keep)
        return "built"
    finally:
        npz.close()


def _pad_time_points(tensor: torch.Tensor, target: int) -> torch.Tensor:
    if tensor.shape[1] == target:
        return tensor
    out_shape = (tensor.shape[0], target) + tuple(tensor.shape[2:])
    out = torch.zeros(out_shape, dtype=tensor.dtype, device=tensor.device)
    out[:, : tensor.shape[1], ...] = tensor
    return out


def hoi_collate_fn(batch: list[dict], args) -> dict:
    max_scene = max(sample["scene_flows"].shape[1] for sample in batch)
    max_robot = max(sample["robot_flows"].shape[1] for sample in batch)
    collated: dict = {}
    keys = sorted(set().union(*(sample.keys() for sample in batch)))

    for key in keys:
        values = [sample.get(key) for sample in batch]
        if any(value is None for value in values):
            if key not in OPTIONAL_MANO_ARRAY_KEYS:
                continue
            template = next((value for value in values if isinstance(value, torch.Tensor)), None)
            if template is None:
                continue
            fill_value = 1.0 if key == "mano_valid" else 0.0
            values = [
                value if isinstance(value, torch.Tensor) else torch.full_like(template, fill_value)
                for value in values
            ]
        first = values[0]
        if not isinstance(first, torch.Tensor):
            collated[key] = values
            continue

        padded: list[torch.Tensor] = []
        for value in values:
            assert isinstance(value, torch.Tensor)
            if key.startswith("scene_") or key.startswith("gt_scene_") or key == "point_weights":
                if value.ndim >= 2:
                    padded.append(_pad_time_points(value, max_scene))
                else:
                    padded.append(value)
            elif key.startswith("robot_"):
                if value.ndim >= 2:
                    padded.append(_pad_time_points(value, max_robot))
                else:
                    padded.append(value)
            else:
                padded.append(value)
        try:
            collated[key] = torch.stack(padded, dim=0)
        except RuntimeError as exc:
            raise RuntimeError(f"Could not stack HOI batch key '{key}'. Check per-sample shapes.") from exc

    selector_gt = collated["scene_selector_gt"].float()
    context = collated["scene_context_mask"].bool().squeeze(-1)
    exists = collated["scene_exists"].bool().squeeze(-1)
    supervised = collated["scene_supervised_mask"].bool().squeeze(-1)
    pred_exists_supervised = (~context) & exists & supervised

    weights = selector_gt.pow(RELEASE_WEIGHT_GAMMA)
    min_motion_loss_weight = getattr(args, "min_motion_loss_weight", None)
    if min_motion_loss_weight is not None:
        weights = weights.clamp_min(float(min_motion_loss_weight))
    weights = torch.where(pred_exists_supervised, weights, torch.zeros_like(weights))
    weights_norm = weights.sum().clamp(min=1.0)
    collated["point_weights"] = weights / weights_norm
    collated["weights_norm"] = weights_norm
    return collated


def build_hoi_dataloader(args, mode: str, rank: int = 0, world_size: int = 1):
    if mode not in ("train", "test"):
        raise ValueError(f"Unsupported HOI dataloader mode: {mode}")

    roots = args.hoi_train_dirs if mode == "train" else args.hoi_val_dirs
    if not roots and mode == "test":
        roots = args.hoi_train_dirs
        if rank == 0:
            print("[hoi_npz] --hoi_val_dirs not provided; using train dirs for test/eval.")
    if not roots:
        return None, {}

    deterministic = bool(args.deterministic_train if mode == "train" else True)
    dataset = HandObjectInteractionDataset(
        roots,
        HOIDatasetConfig(
            max_scene_points=args.max_scene_points,
            max_robot_points=args.max_robot_points,
            target_horizon=args.hoi_target_horizon,
            deterministic=deterministic,
            require_dino_features=args.hoi_require_dino_features,
            seed=args.seed + rank,
            domain=args.hoi_domain,
            mano_sidecar_root=getattr(args, "hoi_mano_sidecar_root", None),
        ),
    )
    drop_last = (mode == "train") and (world_size > 1)
    sampler = None
    shuffle = (mode == "train") and not args.deterministic_train
    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            drop_last=drop_last,
        )
        shuffle = False

    if sampler is None:
        batches_per_rank = int(math.ceil(len(dataset) / args.batch_size))
    elif drop_last:
        samples_per_rank = len(dataset) // world_size
        batches_per_rank = samples_per_rank // args.batch_size
    else:
        samples_per_rank = int(math.ceil(len(dataset) / world_size))
        batches_per_rank = int(math.ceil(samples_per_rank / args.batch_size))
    batches_per_rank = max(1, batches_per_rank)

    num_workers = args.num_workers if mode == "train" else min(args.num_workers, args.eval_num_workers)
    if mode == "train" and args.deterministic_train:
        num_workers = 0
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        drop_last=drop_last,
        collate_fn=partial(hoi_collate_fn, args=args),
    )
    info = {
        "total_samples": int(len(dataset)),
        "total_batches_needed": int(math.ceil(len(dataset) / args.batch_size)),
        "batches_per_rank": int(batches_per_rank),
        "world_size": int(world_size),
    }
    if rank == 0:
        print(
            f"[hoi_npz:{mode}] samples={len(dataset)} batches_per_rank={batches_per_rank} "
            f"roots={[str(Path(r)) for r in roots]}"
        )
    return dataloader, info
