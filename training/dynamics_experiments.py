"""Shared CLI and dataset contract for HOT3D dynamics experiments.

Architectures live in ``training.dynamics_experiment_v2``. This module only
parses experiment flags and patches the HOI dataset inside the experimental
entrypoint; the released ``train.py`` path is untouched.
"""
from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

OBJECTFORESIGHT_VELOCITY_WEIGHT = 2.0
OBJECTFORESIGHT_ACCELERATION_WEIGHT = 2.0
OBJECTFORESIGHT_DISPLACEMENT_WEIGHT = 10.0
OBJECTFORESIGHT_HORIZON_START = 1.0
OBJECTFORESIGHT_HORIZON_END = 3.0
DYNAMIC_3D_GAUSSIANS_K = 20
DYNAMIC_3D_GAUSSIANS_DISTANCE_DECAY = 2000.0


@dataclass(frozen=True)
class ExperimentOptions:
    mode: str
    require_pointwise_dino: bool
    local_increment_weight: float
    horizon_position_weight: float
    horizon_weight_start: float
    horizon_weight_end: float
    velocity_weight: float
    acceleration_weight: float
    object_displacement_weight: float
    local_isometry_weight: float
    local_isometry_k: int
    local_isometry_decay: float


def build_experiment_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--dynamics-experiment",
        choices=("direct", "local_increment", "structured"),
        default="direct",
    )
    parser.add_argument(
        "--require-pointwise-dino",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reject legacy samples whose same frame-global DINO vector was repeated over points.",
    )
    parser.add_argument("--local-increment-loss-weight", type=float, default=None)
    parser.add_argument("--horizon-position-loss-weight", type=float, default=0.0)
    parser.add_argument("--horizon-weight-start", type=float, default=OBJECTFORESIGHT_HORIZON_START)
    parser.add_argument("--horizon-weight-end", type=float, default=OBJECTFORESIGHT_HORIZON_END)
    parser.add_argument("--velocity-loss-weight", type=float, default=None)
    parser.add_argument("--acceleration-loss-weight", type=float, default=None)
    parser.add_argument("--object-displacement-loss-weight", type=float, default=None)
    parser.add_argument("--local-isometry-loss-weight", type=float, default=0.0)
    parser.add_argument("--local-isometry-k", type=int, default=DYNAMIC_3D_GAUSSIANS_K)
    parser.add_argument("--local-isometry-decay", type=float, default=DYNAMIC_3D_GAUSSIANS_DISTANCE_DECAY)
    return parser


def parse_experiment_options(argv: list[str]) -> tuple[ExperimentOptions, list[str]]:
    parsed, remaining = build_experiment_parser().parse_known_args(argv)
    mode = str(parsed.dynamics_experiment)
    local_weight = parsed.local_increment_loss_weight
    if local_weight is None:
        local_weight = OBJECTFORESIGHT_VELOCITY_WEIGHT if mode == "local_increment" else 0.0
    velocity_weight = parsed.velocity_loss_weight
    if velocity_weight is None:
        velocity_weight = OBJECTFORESIGHT_VELOCITY_WEIGHT if mode == "structured" else 0.0
    acceleration_weight = parsed.acceleration_loss_weight
    if acceleration_weight is None:
        acceleration_weight = OBJECTFORESIGHT_ACCELERATION_WEIGHT if mode == "structured" else 0.0
    displacement_weight = parsed.object_displacement_loss_weight
    if displacement_weight is None:
        displacement_weight = OBJECTFORESIGHT_DISPLACEMENT_WEIGHT if mode == "structured" else 0.0

    options = ExperimentOptions(
        mode=mode,
        require_pointwise_dino=bool(parsed.require_pointwise_dino),
        local_increment_weight=float(local_weight),
        horizon_position_weight=float(parsed.horizon_position_loss_weight),
        horizon_weight_start=float(parsed.horizon_weight_start),
        horizon_weight_end=float(parsed.horizon_weight_end),
        velocity_weight=float(velocity_weight),
        acceleration_weight=float(acceleration_weight),
        object_displacement_weight=float(displacement_weight),
        local_isometry_weight=float(parsed.local_isometry_loss_weight),
        local_isometry_k=int(parsed.local_isometry_k),
        local_isometry_decay=float(parsed.local_isometry_decay),
    )
    _validate_options(options)
    return options, remaining


def _validate_options(options: ExperimentOptions) -> None:
    if options.local_isometry_k < 1:
        raise ValueError("--local-isometry-k must be >= 1")
    if options.local_isometry_decay < 0:
        raise ValueError("--local-isometry-decay must be non-negative")
    for name in (
        "local_increment_weight",
        "horizon_position_weight",
        "velocity_weight",
        "acceleration_weight",
        "object_displacement_weight",
        "local_isometry_weight",
    ):
        if getattr(options, name) < 0:
            raise ValueError(f"{name} must be non-negative")


def attach_experiment_options(args: Any, options: ExperimentOptions) -> Any:
    args.dynamics_experiment = options.mode
    args.require_pointwise_dino = options.require_pointwise_dino
    args.local_increment_loss_weight = options.local_increment_weight
    args.horizon_position_loss_weight = options.horizon_position_weight
    args.horizon_weight_start = options.horizon_weight_start
    args.horizon_weight_end = options.horizon_weight_end
    args.velocity_loss_weight = options.velocity_weight
    args.acceleration_loss_weight = options.acceleration_weight
    args.object_displacement_loss_weight = options.object_displacement_weight
    args.local_isometry_loss_weight = options.local_isometry_weight
    args.local_isometry_k = options.local_isometry_k
    args.local_isometry_decay = options.local_isometry_decay
    if options.mode != "direct" and not bool(getattr(args, "identity_norm_stats", False)):
        raise ValueError(
            "local_increment/structured modes require --identity_norm_stats true; "
            "cumulative-displacement norm statistics are not valid for local increments."
        )
    return args


def _metadata_for_path(path: Path) -> dict[str, Any]:
    metadata_path = path.with_suffix(".json")
    if not metadata_path.is_file():
        return {}
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _infer_object_structure(
    arrays: dict[str, np.ndarray], metadata: dict[str, Any], *, require_exact: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    points0 = np.asarray(arrays["scene_flows"][0], dtype=np.float32)
    count = int(points0.shape[0])
    has_ids = "scene_object_id" in arrays
    has_canonical = "scene_canonical_xyz" in arrays
    if require_exact and not (has_ids and has_canonical):
        raise ValueError(
            "structured V2 requires scene_object_id and exact scene_canonical_xyz. "
            "Rebuild with scripts/process/prepare_hot3d_long_horizon_data.sh."
        )

    if has_ids:
        object_ids = np.asarray(arrays["scene_object_id"])
        if object_ids.ndim == 2:
            object_ids = object_ids[0]
        object_ids = object_ids.reshape(-1).astype(np.int64)
        if object_ids.shape[0] != count:
            raise ValueError("scene_object_id length does not match scene points")
    else:
        names = metadata.get("object_ids")
        num_objects = len(names) if isinstance(names, list) and names else 1
        if count % num_objects != 0:
            raise ValueError(f"Cannot infer object blocks: N={count}, objects={num_objects}")
        object_ids = np.repeat(np.arange(num_objects, dtype=np.int64), count // num_objects)

    if has_canonical:
        canonical = np.asarray(arrays["scene_canonical_xyz"], dtype=np.float32)
        if canonical.ndim == 3:
            canonical = canonical[0]
        if canonical.shape != points0.shape:
            raise ValueError("scene_canonical_xyz must have shape (N,3)")
    else:
        canonical = np.zeros_like(points0)
        for object_id in np.unique(object_ids):
            mask = object_ids == object_id
            canonical[mask] = points0[mask] - points0[mask].mean(0, keepdims=True)
    return object_ids, canonical


def _assert_pointwise_dino(arrays: dict[str, np.ndarray], path: Path) -> None:
    if "scene_dino_features" not in arrays:
        raise ValueError(f"{path} is missing scene_dino_features")
    features = np.asarray(arrays["scene_dino_features"], dtype=np.float32)
    if features.ndim == 3:
        features = features[0]
    if features.ndim != 2:
        raise ValueError(f"{path}: expected pointwise DINO (N,D), got {features.shape}")

    valid = np.ones((features.shape[0],), dtype=bool)
    if "scene_dino_valid_mask" in arrays:
        mask = np.asarray(arrays["scene_dino_valid_mask"])
        if mask.ndim == 2:
            mask = mask[0]
        if mask.shape[0] == features.shape[0]:
            valid &= mask.reshape(-1) > 0.5
    if "scene_visibility" in arrays:
        visibility = np.asarray(arrays["scene_visibility"])
        if visibility.ndim == 2 and visibility.shape[1] == features.shape[0]:
            valid &= visibility[0] > 0.5

    selected = features[valid]
    if selected.shape[0] < 2:
        raise ValueError(f"{path}: fewer than two valid pointwise DINO features")
    across_point_std = float(np.mean(np.std(selected, axis=0)))
    max_pair_delta = float(np.max(np.abs(selected - selected[:1])))
    if across_point_std <= 1e-7 or max_pair_delta <= 1e-6:
        raise ValueError(
            f"{path}: DINO is identical across scene points. Rebuild the context-only dataset."
        )


def install_dataset_patch(options: ExperimentOptions) -> None:
    """Patch only the experimental entrypoint's HOI dataset class."""
    import dataset_components.hoi_dataset as hoi

    base_class = hoi.HandObjectInteractionDataset
    point_indices = hoi._point_indices

    class ExperimentalHOIDataset(base_class):
        def _assemble_sample(self, path, arrays, scene_rng, robot_rng, time_valid):
            if options.require_pointwise_dino:
                _assert_pointwise_dino(arrays, Path(path))

            clone = np.random.default_rng()
            clone.bit_generator.state = copy.deepcopy(scene_rng.bit_generator.state)
            scene_idx = point_indices(
                int(arrays["scene_flows"].shape[1]), self.config.max_scene_points, clone
            )
            sample = super()._assemble_sample(path, arrays, scene_rng, robot_rng, time_valid)

            full_ids, full_canonical = _infer_object_structure(
                arrays,
                _metadata_for_path(Path(path)),
                require_exact=options.mode == "structured",
            )
            target = int(self.config.max_scene_points)
            selected_ids = full_ids[scene_idx]
            selected_canonical = full_canonical[scene_idx]
            object_ids = np.full((target,), -1, dtype=np.int64)
            canonical = np.zeros((target, 3), dtype=np.float32)
            object_ids[: selected_ids.shape[0]] = selected_ids
            canonical[: selected_canonical.shape[0]] = selected_canonical
            sample["point_object_id"] = torch.from_numpy(object_ids)
            sample["point_canonical_xyz"] = torch.from_numpy(canonical)
            return sample

    hoi.HandObjectInteractionDataset = ExperimentalHOIDataset
