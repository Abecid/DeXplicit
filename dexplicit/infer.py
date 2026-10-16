# SPDX-License-Identifier: Apache-2.0
"""Initial scene + supplied actions -> predicted 3D -> calibrated controls -> video."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from .conditioning import MODEL_3D_NAME, RENDERER_NAME, predict_controls, sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True, help='Initial-only geometry Safetensors plus supplied actions')
    p.add_argument('--initial-image', required=True); p.add_argument('--camera', required=True)
    p.add_argument('--weights-dir', default='weights'); p.add_argument('--output-dir', required=True)
    p.add_argument('--geometry-python', default=sys.executable, help='Python with geometry CUDA extensions installed')
    p.add_argument('--geometry-config'); p.add_argument('--backend', required=True)
    p.add_argument('--base-model-dir'); p.add_argument('--tokenizer')
    p.add_argument('--prompt', default='realistic egocentric hand object interaction video')
    p.add_argument('--seed', type=int, default=17); p.add_argument('--steps', type=int, default=30)
    p.add_argument('--cfg', type=float, default=5.)
    a = p.parse_args()
    out, weights = Path(a.output_dir).resolve(), Path(a.weights_dir).resolve()
    if out.exists() and any(out.iterdir()): p.error('Use a new empty output directory')
    out.mkdir(parents=True, exist_ok=True)
    gw = weights / 'dexplicit-3d-model.safetensors'
    high, low = [weights / f'dexplicit-renderer-{phase}.safetensors' for phase in ['high', 'low']]
    receipt = {'inputs': {name: {'sha256': sha256(path)} for name, path in {
        'geometry_input': a.input, 'initial_image': a.initial_image, 'camera': a.camera,
        'geometry_weights': gw, 'high_weights': high, 'low_weights': low}.items()},
        'sampler': {'seed': a.seed, 'steps': a.steps, 'cfg': a.cfg, 'expert_boundary': .875, 'sigma_shift': 5.},
        'models': {'3d': MODEL_3D_NAME, 'renderer': RENDERER_NAME},
        'conditioning_source': 'predicted_3d', 'prompt': a.prompt, 'status': 'started'}
    (out/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    control_path, receipt['controls'] = predict_controls(
        geometry_input=a.input, initial_image=a.initial_image, camera=a.camera,
        weights=gw, output_dir=out/'predicted-controls', geometry_python=a.geometry_python,
        geometry_config=a.geometry_config)
    cmd = [sys.executable, '-m', 'dexplicit.renderer', '--high-weights', str(high), '--low-weights', str(low),
           '--controls', str(control_path), '--output', str(out/'video.mp4'),
           '--rgb-output', str(out/'rgb.npz'), '--backend', a.backend,
           '--prompt', a.prompt, '--seed', str(a.seed), '--steps', str(a.steps), '--cfg', str(a.cfg)]
    for option in ['base_model_dir', 'tokenizer']:
        if getattr(a, option): cmd += ['--'+option.replace('_', '-'), getattr(a, option)]
    subprocess.run(cmd, check=True)
    receipt['outputs'] = {name: sha256(out/name) for name in [
        'predicted-controls/forecast.npz', 'predicted-controls/controls.npz',
        'predicted-controls/controls.json', 'rgb.npz', 'video.mp4']}
    receipt['status'] = 'complete'
    (out/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps({'output': str(out/'video.mp4'), 'receipt': str(out/'receipt.json')}))

if __name__ == '__main__': main()
