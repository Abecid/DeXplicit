# SPDX-License-Identifier: Apache-2.0
"""The DeXplicit renderer, using the native dual-expert Wan/VACE backbone."""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file

BACKEND_COMMIT = 'ce9f4541d0f4ae47a138337e59066dd633207aa7'
BASE_REPO = 'alibaba-pai/Wan2.2-VACE-Fun-A14B'
BASE_REVISION = '4438caabbfdd7437bbb0283d86cb200d1b7223ba'
NEGATIVE_PROMPT = ('overexposed, static, blurred details, subtitles, painting, still image, '
                   'gray cast, worst quality, low quality, JPEG artifacts, ugly, deformed, '
                   'bad hands, bad face, malformed limbs, fused fingers, cluttered background')
TARGETS = ['q', 'k', 'v', 'o', 'ffn.0', 'ffn.2']


def backend(path):
    path = Path(path).resolve()
    actual = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != BACKEND_COMMIT:
        raise ValueError(f'Expected DiffSynth revision {BACKEND_COMMIT}, got {actual}')
    if subprocess.check_output(['git', '-C', str(path), 'status', '--porcelain', '--untracked-files=no'], text=True).strip():
        raise ValueError('DiffSynth has tracked modifications; use a clean pinned checkout')
    sys.path.insert(0, str(path))
    from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
    from diffsynth.configs import VRAM_MANAGEMENT_MODULE_MAPS
    VRAM_MANAGEMENT_MODULE_MAPS['diffsynth.models.wan_video_vace.VaceWanModel'][
        'diffsynth.models.wan_video_dit.DiTBlock'] = 'diffsynth.core.vram.layers.AutoWrappedNonRecurseModule'
    torch.set_num_threads(4)
    return WanVideoPipeline, ModelConfig


def base_files(base_dir=None, tokenizer=None):
    if base_dir is None:
        from huggingface_hub import snapshot_download
        base_dir = snapshot_download(BASE_REPO, revision=BASE_REVISION, allow_patterns=[
            'high_noise_model/diffusion_pytorch_model.safetensors',
            'low_noise_model/diffusion_pytorch_model.safetensors',
            'models_t5_umt5-xxl-enc-bf16.pth', 'Wan2.1_VAE.pth', 'google/umt5-xxl/*'])
    base = Path(base_dir).resolve()
    paths = {name: base / filename for name, filename in {
        'high': 'high_noise_model/diffusion_pytorch_model.safetensors',
        'low': 'low_noise_model/diffusion_pytorch_model.safetensors',
        'text': 'models_t5_umt5-xxl-enc-bf16.pth', 'vae': 'Wan2.1_VAE.pth'}.items()}
    paths['tokenizer'] = Path(tokenizer).resolve() if tokenizer else base / 'google/umt5-xxl'
    return paths


def adapter_state(weights):
    state = load_file(str(weights))
    state = {k.removeprefix('pipe.vace.').removeprefix('vace.'): v for k, v in state.items()}
    schema = json.loads((Path(__file__).parent/'resources/renderer-adapter-schema.json').read_text())
    if state.keys() != schema.keys() or any(list(v.shape) != schema[k] for k, v in state.items()):
        raise ValueError('Adapter must match the complete selected A14B VACE LoRA schema')
    if not state or any('.lora_A.' not in k and '.lora_B.' not in k for k in state):
        raise ValueError('Expected rank-32 VACE LoRA tensors only')
    if any(not torch.isfinite(v).all() or v.ndim != 2 or 32 not in v.shape for v in state.values()):
        raise ValueError('Invalid adapter tensor')
    return state


def load_renderer(high, low, *, backend_path, base_dir=None, tokenizer=None):
    Pipeline, Config = backend(backend_path)
    paths = base_files(base_dir, tokenizer)
    offload = dict(offload_dtype=torch.bfloat16, offload_device='cpu',
        onload_dtype=torch.bfloat16, onload_device='cpu', preparing_dtype=torch.bfloat16,
        preparing_device='cuda', computation_dtype=torch.bfloat16, computation_device='cuda')
    pipe = Pipeline.from_pretrained(torch_dtype=torch.bfloat16, device='cuda',
        model_configs=[Config(path=str(paths[k]), **offload) for k in ['high', 'low', 'text', 'vae']],
        tokenizer_config=Config(path=str(paths['tokenizer'])), redirect_common_files=False,
        vram_limit=torch.cuda.get_device_properties(0).total_memory / 2**30 - 2)
    for dit, vace in [(pipe.dit, pipe.vace), (pipe.dit2, pipe.vace2)]:
        if dit is None or vace is None or dit.dim != 5120 or len(dit.blocks) != 40 or vace.vace_in_dim != 96:
            raise ValueError('Expected the two native A14B experts')
    for module, weights in [(pipe.vace, high), (pipe.vace2, low)]:
        # The native loader merges exactly these A/B adapters into each expert.
        pipe.load_lora(module, state_dict=adapter_state(weights), alpha=1)
    pipe.requires_grad_(False)
    return pipe


def load_controls(path):
    with np.load(path, allow_pickle=False) as z:
        frames, reference = z['rgb'], z['reference']
    if frames.shape != (33, 480, 832, 3) or reference.shape != (480, 832, 3):
        raise ValueError('Expected 33 RGB control frames and one reference at 832x480')
    if frames.dtype != np.uint8 or reference.dtype != np.uint8:
        raise ValueError('Controls and reference must be uint8 RGB')
    return [Image.fromarray(x) for x in frames], Image.fromarray(reference)


@torch.inference_mode()
def generate(pipe, controls, *, prompt, seed=17, steps=30, cfg=5.):
    frames, reference = load_controls(controls)
    result = pipe(prompt=prompt, negative_prompt=NEGATIVE_PROMPT,
        vace_video=frames, vace_reference_image=reference, num_frames=33,
        height=480, width=832, num_inference_steps=steps, cfg_scale=cfg,
        cfg_merge=False, seed=seed, rand_device='cpu', sigma_shift=5.,
        switch_DiT_boundary=.875, tiled=True, tile_size=(30, 52), tile_stride=(15, 26))
    rgb = np.stack([np.asarray(frame) for frame in result])
    if rgb.shape != (33, 480, 832, 3) or rgb.dtype != np.uint8:
        raise ValueError(f'Invalid decoded video: {rgb.shape} {rgb.dtype}')
    return rgb


def write_video(path, rgb, fps=15):
    import imageio.v2 as imageio
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(str(path), rgb, fps=fps, codec='libx264', quality=8, macro_block_size=None)
    decoded = np.stack(imageio.mimread(str(path)))
    if decoded.shape != rgb.shape: raise ValueError('Encoded video failed round-trip validation')


def add_model_args(p):
    p.add_argument('--backend', required=True)
    p.add_argument('--base-model-dir')
    p.add_argument('--tokenizer', help='Optional local tokenizer directory for offline use')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--high-weights', required=True); p.add_argument('--low-weights', required=True)
    p.add_argument('--controls', required=True); p.add_argument('--output', required=True)
    p.add_argument('--prompt', default='realistic egocentric hand object interaction video')
    p.add_argument('--seed', type=int, default=17); p.add_argument('--steps', type=int, default=30)
    p.add_argument('--cfg', type=float, default=5.); p.add_argument('--rgb-output')
    add_model_args(p); a = p.parse_args()
    if a.steps < 1 or a.cfg < 0: p.error('Invalid sampler settings')
    pipe = load_renderer(a.high_weights, a.low_weights, backend_path=a.backend,
                         base_dir=a.base_model_dir, tokenizer=a.tokenizer)
    rgb = generate(pipe, a.controls, prompt=a.prompt, seed=a.seed, steps=a.steps, cfg=a.cfg)
    write_video(a.output, rgb)
    if a.rgb_output: np.savez_compressed(a.rgb_output, rgb=rgb)
    print(json.dumps({'output': a.output, 'shape': list(rgb.shape), 'steps': a.steps, 'seed': a.seed}), flush=True)

if __name__ == '__main__': main()
