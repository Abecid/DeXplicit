# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np


MANO_SIDECAR_KEYS = {
    "hand_joints_world",
    "hand_joint_valid",
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


def resolve_mano_sidecar_path(sample_path: str | Path, sidecar_root: str | Path | None) -> Path | None:
    if sidecar_root is None:
        return None
    sample = Path(sample_path)
    root = Path(sidecar_root).expanduser()
    candidates = [
        root / sample.parent.name / sample.name,
        root / sample.name,
        root / sample.with_suffix(".mano.npz").name,
        root / f"{sample.stem}.mano.npz",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def load_mano_sidecar_arrays(path: str | Path) -> dict[str, np.ndarray]:
    sidecar_path = Path(path).expanduser()
    with np.load(sidecar_path, allow_pickle=False) as arrays:
        return {
            key: np.asarray(arrays[key])
            for key in arrays.files
            if key in MANO_SIDECAR_KEYS and np.asarray(arrays[key]).dtype != object
        }


def merge_mano_sidecar_arrays(
    arrays: Mapping[str, np.ndarray],
    mano_arrays: Mapping[str, np.ndarray],
    *,
    expected_timesteps: int,
) -> dict[str, np.ndarray]:
    merged = dict(arrays)
    for key, value in mano_arrays.items():
        value = np.asarray(value)
        if value.ndim > 1 and value.shape[0] not in {2, int(expected_timesteps)}:
            raise ValueError(
                f"MANO sidecar key {key!r} has incompatible first dimension {value.shape[0]}; "
                f"expected {expected_timesteps} timesteps or 2 hands."
            )
        merged[key] = value
    return merged
