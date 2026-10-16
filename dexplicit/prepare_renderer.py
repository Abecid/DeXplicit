# SPDX-License-Identifier: Apache-2.0
"""Prepare DeXplicit renderer training with predicted 3D and recorded RGB targets."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file
from .renderer import backend, base_files, load_controls, add_model_args
from .conditioning import (MODEL_3D_NAME, RENDERER_NAME, TRAINING_FORMAT, predict_controls,
                           sha256, validate_training_row)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True,
                   help='JSON records: geometry_input, initial_image, camera, target video and prompt')
    p.add_argument('--geometry-weights', required=True, help='DeXplicit 3D model Safetensors')
    p.add_argument('--geometry-python', default=sys.executable)
    p.add_argument('--geometry-config')
    p.add_argument('--output-dir', required=True)
    add_model_args(p)
    a = p.parse_args()
    source = Path(a.manifest).resolve(strict=True)
    rows = json.loads(source.read_text())
    if not isinstance(rows, list) or not rows:
        raise ValueError('Expected a nonempty training-record list')
    rows = [validate_training_row(row, source.parent) for row in rows]
    out = Path(a.output_dir).resolve()
    if out.exists() and any(out.iterdir()):
        p.error('Choose a new empty output directory')
    out.mkdir(parents=True, exist_ok=True)
    prepared = []
    # Forecast before loading the video encoder. Target RGB remains unopened and
    # never enters the frozen DeXplicit 3D model or the conditioning projector.
    for index, row in enumerate(rows):
        control_dir = out/'predicted-controls'/f'sample-{index:06d}'
        controls, receipt = predict_controls(
            geometry_input=row['geometry_input'], initial_image=row['initial_image'],
            camera=row['camera'], weights=a.geometry_weights, output_dir=control_dir,
            geometry_python=a.geometry_python, geometry_config=a.geometry_config)
        prepared.append((row, controls, control_dir/'controls.json', receipt))
    Pipeline, Config = backend(a.backend)
    paths = base_files(a.base_model_dir, a.tokenizer)
    pipe = Pipeline.from_pretrained(torch_dtype=torch.bfloat16, device='cuda',
        model_configs=[Config(path=str(paths[k])) for k in ['text', 'vae']],
        tokenizer_config=Config(path=str(paths['tokenizer'])), redirect_common_files=False)
    pipe.requires_grad_(False)
    from diffsynth.pipelines.wan_video import (
        WanVideoUnit_PromptEmbedder, WanVideoUnit_VACE, WanVideoUnit_InputVideoEmbedder)
    import imageio.v2 as imageio
    bundles = []
    with torch.no_grad():
        for index, (row, control_path, receipt_path, receipt) in enumerate(prepared):
            controls, reference = load_controls(control_path)
            # Future RGB is supervision, separate from predicted conditions.
            rgb = np.stack(imageio.mimread(str(row['video'])))
            if rgb.shape != (33, 480, 832, 3):
                raise ValueError('Target video must contain 33 RGB frames at 832x480')
            frames = [Image.fromarray(x) for x in rgb]
            pipe.scheduler.set_timesteps(1000, training=True, shift=5.)
            context = WanVideoUnit_PromptEmbedder().process(pipe, row['prompt'], True)['context']
            vace = WanVideoUnit_VACE().process(
                pipe, controls, None, reference, 1., 480, 832, 33, False,
                (30, 52), (15, 26))['vace_context']
            target = WanVideoUnit_InputVideoEmbedder().process(
                pipe, frames, torch.zeros((1, 16, 10, 60, 104), device='cuda', dtype=torch.bfloat16),
                False, (30, 52), (15, 26), reference, False, None, None)['input_latents']
            name = f'sample-{index:06d}.safetensors'
            save_file({key: value.cpu().contiguous() for key, value in {
                'context': context, 'vace_context': vace, 'input_latents': target}.items()},
                str(out/name), metadata={
                    'format': TRAINING_FORMAT, 'model_name': RENDERER_NAME,
                    'geometry_model_name': MODEL_3D_NAME, 'conditioning_source': 'predicted_3d',
                    'source_id': str(row.get('source_id', index)),
                    'conditioning_receipt': str(receipt_path.relative_to(out)),
                    'conditioning_receipt_sha256': sha256(receipt_path),
                    'geometry_weights_sha256': receipt['inputs']['geometry_weights']['sha256'],
                    'target_video_sha256': sha256(row['video'])})
            bundles.append(name)
            print(json.dumps({'encoded': name, 'conditioning_source': 'predicted_3d'}), flush=True)
    (out/'manifest.json').write_text(json.dumps(bundles, indent=2)+'\n')


if __name__ == '__main__':
    main()
