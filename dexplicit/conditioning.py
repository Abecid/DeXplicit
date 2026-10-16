# SPDX-License-Identifier: Apache-2.0
"""Shared predicted-3D conditioning for DeXplicit inference and renderer training."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

MODEL_3D_NAME = 'DeXplicit 3D model'
RENDERER_NAME = 'DeXplicit renderer'
TRAINING_FORMAT = 'dexplicit-renderer-training-v2'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def predict_controls(*, geometry_input, initial_image, camera, weights, output_dir,
                     geometry_python=sys.executable, geometry_config=None):
    """Run the frozen 3D model, then project its unchanged forecast.

    No target-video or future-object-geometry argument enters this boundary.
    Both inference and renderer training call this same function.
    """
    paths = {key: Path(value).resolve(strict=True) for key, value in {
        'geometry_input': geometry_input, 'initial_image': initial_image,
        'camera': camera, 'geometry_weights': weights}.items()}
    out = Path(output_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use a new empty directory for predicted 3D conditions')
    out.mkdir(parents=True, exist_ok=True)
    forecast, controls = out/'forecast.npz', out/'controls.npz'
    command = [str(geometry_python), '-m', 'dexplicit.geometry', 'infer',
               '--weights', str(paths['geometry_weights']),
               '--input', str(paths['geometry_input']), '--output', str(forecast)]
    if geometry_config:
        command += ['--config', str(Path(geometry_config).resolve(strict=True))]
    # Exit before renderer/VAE allocation, releasing the 3D model's GPU memory.
    subprocess.run(command, check=True)
    from .controls import save_controls
    projection = save_controls(forecast, paths['geometry_input'], paths['initial_image'],
                               paths['camera'], controls)
    receipt = {**projection, 'schema': 'dexplicit-predicted-controls-v1',
               'model_name': MODEL_3D_NAME, 'scene_source': 'predicted',
               'producer': 'dexplicit.geometry.infer',
               'inputs': {key: {'sha256': sha256(path)} for key, path in paths.items()},
               'forecast_sha256': sha256(forecast), 'controls_sha256': sha256(controls),
               'forecast_file': forecast.name, 'controls_file': controls.name}
    if receipt['future_scene_or_rgb_input'] or receipt['point_positions_modified']:
        raise ValueError('Conditions must preserve the forecast and initial-only input boundary')
    (out/'controls.json').write_text(json.dumps(receipt, indent=2)+'\n')
    return controls, receipt


def validate_training_row(row, root):
    """Reject precomputed/measured controls before any GPU work."""
    required = {'geometry_input', 'initial_image', 'camera', 'video', 'prompt'}
    allowed = required | {'source_id', 'split'}
    if not isinstance(row, dict) or not required <= row.keys() or row.keys() - allowed:
        raise ValueError('Renderer training requires initial-only geometry_input, initial_image, '
                         'camera, target video and prompt; precomputed controls/trajectories are not accepted')
    if row.get('split', 'train') != 'train':
        raise ValueError('Only the training split may enter renderer training')
    if not isinstance(row['prompt'], str) or not row['prompt'].strip():
        raise ValueError('A nonempty training prompt is required')
    result = dict(row)
    for key in required - {'prompt'}:
        result[key] = (Path(root)/row[key]).resolve(strict=True)
    return result


def validate_training_bundle(path):
    """Require a bundle linked to preserved predicted-control artifacts."""
    from safetensors import safe_open
    path = Path(path).resolve(strict=True)
    with safe_open(str(path), framework='np') as bundle:
        meta = bundle.metadata() or {}
        if meta.get('format') != TRAINING_FORMAT or meta.get('conditioning_source') != 'predicted_3d':
            raise ValueError('DeXplicit renderer training requires predicted-3D bundles from prepare_renderer; '
                             'measured or unlabelled conditioning is rejected')
        shapes = {'context': [1, 512, 4096], 'vace_context': [1, 96, 10, 60, 104],
                  'input_latents': [1, 16, 10, 60, 104]}
        if set(bundle.keys()) != set(shapes):
            raise ValueError('Unexpected renderer training bundle fields')
        for key, shape in shapes.items():
            if bundle.get_slice(key).get_shape() != shape:
                raise ValueError(f'Unexpected renderer training shape for {key}')
    receipt_path = (path.parent/meta['conditioning_receipt']).resolve(strict=True)
    if not receipt_path.is_relative_to(path.parent):
        raise ValueError('Conditioning receipts must accompany their training bundle')
    if sha256(receipt_path) != meta.get('conditioning_receipt_sha256'):
        raise ValueError('Conditioning receipt checksum mismatch')
    receipt = json.loads(receipt_path.read_text())
    if (receipt.get('schema') != 'dexplicit-predicted-controls-v1'
            or receipt.get('scene_source') != 'predicted'
            or receipt.get('producer') != 'dexplicit.geometry.infer'
            or receipt.get('future_scene_or_rgb_input') is not False
            or receipt.get('point_positions_modified') is not False):
        raise ValueError('Training controls must come from the unchanged DeXplicit 3D model forecast')
    for filename_key, digest_key in [('forecast_file', 'forecast_sha256'),
                                      ('controls_file', 'controls_sha256')]:
        artifact = (receipt_path.parent/receipt[filename_key]).resolve(strict=True)
        if not artifact.is_relative_to(receipt_path.parent) or sha256(artifact) != receipt[digest_key]:
            raise ValueError('Predicted conditioning artifact checksum mismatch')
    if meta.get('geometry_weights_sha256') != receipt['inputs']['geometry_weights']['sha256']:
        raise ValueError('3D model checkpoint identity mismatch')
    return {'bundle': str(path), 'sha256': sha256(path),
            'source_id': meta.get('source_id'), 'conditioning_source': 'predicted_3d',
            'geometry_weights_sha256': meta['geometry_weights_sha256'],
            'conditioning_receipt_sha256': meta['conditioning_receipt_sha256']}
