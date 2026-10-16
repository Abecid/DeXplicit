# SPDX-License-Identifier: Apache-2.0
"""Render the predictor's unchanged metric trajectories into RGB-cache controls."""
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
from safetensors.numpy import load_file
from .camera import Camera
from .image_utils import resize_rgb, resize_mask
from .raster import _build_rgb_cache, _render_frame


def render(forecast, geometry_input, initial_image, camera_path):
    inputs = load_file(str(geometry_input))
    with np.load(forecast, allow_pickle=False) as z:
        scene = z['scene_flows']
    if scene.ndim != 4 or scene.shape[0] != 1 or scene.shape[1] < 33:
        raise ValueError('Expected a single forecast [1,T>=33,N,3]')
    if not np.isfinite(scene).all(): raise ValueError('Nonfinite prediction')
    initial = inputs['scene_flows'][0, 0]
    if not np.allclose(scene[0, 0], initial, atol=1e-5):
        raise ValueError('Forecast and initial input do not match')
    valid0 = inputs['scene_exists'][0, 0].reshape(-1).astype(bool)
    if valid0.shape != scene.shape[2:3]: raise ValueError('Point identity mismatch')
    hands = inputs['robot_flows'][0, :33]
    hand_valid = inputs['robot_exists'][0, :33].reshape(hands.shape[:2])
    if hands.shape[0] != 33 or not np.isfinite(hands[hand_valid.astype(bool)]).all():
        raise ValueError('Expected 33 finite supplied hand-action states')
    # Selection removes padded points only. No smoothing, resampling or GT repair.
    arrays = {'scene_flows': scene[0, :33][:, valid0],
              'scene_visibility': np.ones((33, int(valid0.sum())), dtype=bool),
              'robot_flows': hands, 'robot_valid': hand_valid}
    camera = Camera(camera_path)
    rgb = np.asarray(Image.open(initial_image).convert('RGB'))
    if rgb.shape[:2] != tuple(camera.hw[0]):
        raise ValueError('Initial image size must equal camera source_hw; use the calibrated source image')
    payload = SimpleNamespace(rgb_frames=[rgb], arrays=arrays)
    cache = _build_rgb_cache(payload, 0, lambda p: camera.project(p, 0))
    if not cache['scene_flows'][1].any():
        raise ValueError('No initial object points project onto the image; check calibration and coordinate frames')
    frames, masks = [], []
    for t in range(33):
        frame, mask = _render_frame(arrays, t, tuple(camera.hw[t]), 2,
                                  lambda p: camera.project(p, t), 'rgb_cache', cache)
        if camera.rotate:
            frame, mask = np.rot90(frame, -1), np.rot90(mask, -1)
        frames.append(resize_rgb(frame, 832, 480))
        masks.append(resize_mask(mask, 832, 480))
    reference = resize_rgb(np.rot90(rgb, -1) if camera.rotate else rgb, 832, 480)
    return np.stack(frames), np.stack(masks), reference, {
        'geometry_states': list(range(33)), 'source_visible_points': {k: int(v[1].sum()) for k, v in cache.items()},
        'coverage_pixels': [int((m[..., 0] > 0).sum()) for m in masks],
        'camera_model': camera.model, 'future_scene_or_rgb_input': False,
        'point_positions_modified': False}


def save_controls(forecast, geometry_input, initial_image, camera_path, output):
    frames, masks, reference, receipt = render(forecast, geometry_input, initial_image, camera_path)
    np.savez_compressed(output, rgb=frames, coverage=masks, reference=reference)
    return receipt


def main():
    import argparse
    import json
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trajectories', required=True, help='NPZ with scene_flows [1,T>=33,N,3], preserving input point IDs')
    p.add_argument('--scene-source', required=True, choices=['predicted', 'measured'],
                   help='Measured trajectories are training/evaluation supervision, never a prediction')
    p.add_argument('--geometry-input', required=True); p.add_argument('--initial-image', required=True)
    p.add_argument('--camera', required=True); p.add_argument('--output', required=True)
    a = p.parse_args()
    output = Path(a.output)
    if output.exists(): p.error('Choose a new output path')
    output.parent.mkdir(parents=True, exist_ok=True)
    receipt = save_controls(a.trajectories, a.geometry_input, a.initial_image, a.camera, output)
    receipt['scene_source'] = a.scene_source
    receipt['future_scene_or_rgb_input'] = a.scene_source == 'measured'
    receipt['future_rgb_input'] = False
    output.with_suffix('.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt))

if __name__ == '__main__': main()
