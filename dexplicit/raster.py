# SPDX-License-Identifier: Apache-2.0
"""Native RGB-cache, depth-tested point splatting used by the selected renderer."""
from __future__ import annotations
import math
import numpy as np
from typing import Any

def _normal_to_rgb(normals: np.ndarray) -> np.ndarray:
    return np.clip((np.asarray(normals, dtype=np.float32) + 1.0) * 0.5, 0.0, 1.0)

def _depth_to_gray(depth: np.ndarray, near: float, far: float) -> np.ndarray:
    denom = max(float(far) - float(near), 1e-6)
    normalized = 1.0 - np.clip((depth.astype(np.float32) - near) / denom, 0.0, 1.0)
    return normalized[:, None].repeat(3, axis=1).astype(np.float32)

def _robust_depth_range(depth: np.ndarray) -> tuple[float, float]:
    valid = depth[np.isfinite(depth) & (depth > 1e-6)]
    if valid.size == 0:
        return 0.0, 1.0
    near = float(np.percentile(valid, 2.0))
    far = float(np.percentile(valid, 98.0))
    if not math.isfinite(near) or not math.isfinite(far) or far <= near + 1e-6:
        near = float(valid.min())
        far = float(valid.max() + 1e-3)
    return near, far

def _valid_pixel_mask(pixels: np.ndarray, depth: np.ndarray, valid: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    h, w = image_shape
    return (
        np.asarray(valid, dtype=bool)
        & np.isfinite(pixels).all(axis=1)
        & np.isfinite(depth)
        & (depth > 1e-6)
        & (pixels[:, 0] >= 0)
        & (pixels[:, 0] < w)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < h)
    )

def _sample_rgb_at_pixels(rgb: np.ndarray, pixels: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rgb = np.asarray(rgb, dtype=np.uint8)
    h, w = rgb.shape[:2]
    sampled = np.zeros((pixels.shape[0], 3), dtype=np.float32)
    valid_px = (
        np.asarray(valid, dtype=bool)
        & np.isfinite(pixels).all(axis=1)
        & (pixels[:, 0] >= 0)
        & (pixels[:, 0] < w)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < h)
    )
    if valid_px.any():
        xy = np.rint(pixels[valid_px]).astype(np.int64)
        xy[:, 0] = np.clip(xy[:, 0], 0, w - 1)
        xy[:, 1] = np.clip(xy[:, 1], 0, h - 1)
        sampled[valid_px] = rgb[xy[:, 1], xy[:, 0]].astype(np.float32) / 255.0
    return sampled, valid_px

def _array_valid(arrays: dict[str, np.ndarray], key: str, valid_key: str, t: int) -> np.ndarray:
    count = int(arrays[key].shape[1])
    valid = np.ones((count,), dtype=bool)
    if valid_key in arrays:
        candidate = np.asarray(arrays[valid_key][t]).reshape(-1)
        if candidate.shape[0] == count:
            valid &= candidate > 0.5
    elif key == "robot_flows" and "hand_joint_valid" in arrays:
        candidate = np.asarray(arrays["hand_joint_valid"][t]).reshape(-1)
        if candidate.shape[0] == count:
            valid &= candidate > 0.5
    pts = np.asarray(arrays[key][t], dtype=np.float32)
    valid &= np.isfinite(pts).all(axis=1)
    valid &= np.linalg.norm(pts, axis=1) > 1e-8
    return valid

def _semantic_colors(arrays: dict[str, np.ndarray], key: str, t: int, depth: np.ndarray, near: float, far: float) -> np.ndarray:
    depth_rgb = _depth_to_gray(depth, near, far)
    colors = np.asarray(arrays.get(key.replace("flows", "colors"), np.zeros_like(arrays[key]))[t], dtype=np.float32)
    normals = _normal_to_rgb(np.asarray(arrays.get(key.replace("flows", "normals"), np.zeros_like(arrays[key]))[t], dtype=np.float32))
    if key == "scene_flows":
        return np.clip(0.55 * depth_rgb + 0.25 * normals + 0.20 * colors, 0.0, 1.0)
    return np.clip(0.40 * depth_rgb + 0.45 * colors + 0.15 * normals, 0.0, 1.0)

def _build_rgb_cache(payload: SamplePayload, reference_t: int, project: Any) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    rgb = payload.rgb_frames[reference_t]
    projected: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    zbuf = np.full(rgb.shape[:2], np.inf, dtype=np.float32)

    for key, valid_key in (("scene_flows", "scene_visibility"), ("robot_flows", "robot_valid")):
        if key not in payload.arrays:
            continue
        pixels, depth, valid = project(payload.arrays[key][reference_t])
        valid = _valid_pixel_mask(pixels, depth, valid, rgb.shape[:2])
        valid &= _array_valid(payload.arrays, key, valid_key, reference_t)
        projected[key] = (pixels, depth, valid)
        indices = np.flatnonzero(valid)
        if indices.size:
            xy = np.rint(pixels[indices]).astype(np.int64)
            xy[:, 0] = np.clip(xy[:, 0], 0, rgb.shape[1] - 1)
            xy[:, 1] = np.clip(xy[:, 1], 0, rgb.shape[0] - 1)
            np.minimum.at(zbuf, (xy[:, 1], xy[:, 0]), depth[indices])

    cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for key, (pixels, depth, valid) in projected.items():
        indices = np.flatnonzero(valid)
        source_visible = np.zeros_like(valid)
        if indices.size:
            xy = np.rint(pixels[indices]).astype(np.int64)
            xy[:, 0] = np.clip(xy[:, 0], 0, rgb.shape[1] - 1)
            xy[:, 1] = np.clip(xy[:, 1], 0, rgb.shape[0] - 1)
            source_visible[indices] = depth[indices] <= zbuf[xy[:, 1], xy[:, 0]] + 1e-5
        cache[key] = _sample_rgb_at_pixels(rgb, pixels, valid & source_visible)
    return cache

def _splat(
    image: np.ndarray,
    mask: np.ndarray,
    zbuf: np.ndarray,
    pixels: np.ndarray,
    depth: np.ndarray,
    valid: np.ndarray,
    colors: np.ndarray,
    radius: int,
    bias: float = 0.0,
) -> None:
    h, w = image.shape[:2]
    radius = max(0, int(radius))
    valid = _valid_pixel_mask(pixels, depth, valid, (h, w))
    indices = np.flatnonzero(valid)
    if indices.size == 0:
        return
    order = indices[np.argsort(depth[indices])[::-1]]
    color_u8 = np.clip(colors * 255.0, 0.0, 255.0).astype(np.uint8)
    for idx in order.tolist():
        cx = int(round(float(pixels[idx, 0])))
        cy = int(round(float(pixels[idx, 1])))
        z = float(depth[idx]) - float(bias)
        x0, x1 = max(0, cx - radius), min(w, cx + radius + 1)
        y0, y1 = max(0, cy - radius), min(h, cy + radius + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        patch = zbuf[y0:y1, x0:x1]
        update = z <= patch
        if not np.any(update):
            continue
        patch[update] = z
        image[y0:y1, x0:x1][update] = color_u8[idx]
        mask[y0:y1, x0:x1][update] = 255

def _render_frame(
    arrays: dict[str, np.ndarray],
    t: int,
    image_shape: tuple[int, int],
    radius: int,
    project: Any,
    condition_render_mode: str,
    rgb_cache: dict[str, tuple[np.ndarray, np.ndarray]] | None,
) -> tuple[np.ndarray, np.ndarray]:
    if condition_render_mode not in {"semantic", "rgb_cache"}:
        raise ValueError(f"Unsupported condition_render_mode={condition_render_mode!r}; expected semantic or rgb_cache.")
    h, w = image_shape
    image = np.zeros((h, w, 3), dtype=np.uint8)
    mask = np.zeros((h, w, 3), dtype=np.uint8)
    zbuf = np.full((h, w), np.inf, dtype=np.float32)
    projections: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    depth_pool: list[np.ndarray] = []
    for key in ("scene_flows", "robot_flows"):
        if key not in arrays:
            continue
        pixels, depth, valid = project(arrays[key][t])
        projections[key] = (pixels, depth, valid)
        depth_pool.append(depth[valid])
    near, far = _robust_depth_range(np.concatenate(depth_pool) if depth_pool else np.zeros((0,), dtype=np.float32))
    for key, valid_key, bias, point_radius in (
        ("scene_flows", "scene_visibility", 0.0, radius),
        ("robot_flows", "robot_valid", 1e-4, max(radius + 1, 1)),
    ):
        if key not in projections:
            continue
        pixels, depth, valid = projections[key]
        valid = valid & _array_valid(arrays, key, valid_key, t)
        if condition_render_mode == "rgb_cache":
            if rgb_cache is None or key not in rgb_cache:
                continue
            colors, cache_valid = rgb_cache[key]
            if colors.shape[0] != arrays[key].shape[1] or cache_valid.shape[0] != arrays[key].shape[1]:
                raise ValueError(f"RGB cache for {key} has inconsistent point count.")
            valid = valid & cache_valid
        else:
            colors = _semantic_colors(arrays, key, t, depth, near, far)
        _splat(image, mask, zbuf, pixels, depth, valid, colors, point_radius, bias)
    return image, mask
