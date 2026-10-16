# SPDX-License-Identifier: Apache-2.0
"""Train the DeXplicit renderer on verified predicted-3D conditions."""
import argparse
import json
from pathlib import Path
import torch
from safetensors.torch import load_file, save_file
from .renderer import backend, base_files, adapter_state, TARGETS
from .conditioning import RENDERER_NAME, sha256, validate_training_bundle


def load_expert(phase, *, backend_path, base_dir=None, weights=None):
    Pipeline, Config = backend(backend_path)
    paths = base_files(base_dir)
    pipe = Pipeline.from_pretrained(device='cpu', torch_dtype=torch.bfloat16,
        model_configs=[Config(path=str(paths[phase]))], tokenizer_config=None, redirect_common_files=False)
    if pipe.dit.dim != 5120 or len(pipe.dit.blocks) != 40 or pipe.vace.vace_in_dim != 96 or pipe.dit2 is not None:
        raise ValueError('Expected one native A14B expert')
    pipe.requires_grad_(False)
    from diffsynth.diffusion.training_module import DiffusionTrainingModule
    helper = DiffusionTrainingModule()
    pipe.vace = helper.add_lora_to_model(pipe.vace, TARGETS, 32, upcast_dtype=torch.float32)
    parameters = {k: v for k, v in pipe.vace.named_parameters() if v.requires_grad}
    if weights:
        state = helper.mapping_lora_state_dict(adapter_state(weights))
        if state.keys() != parameters.keys():
            raise ValueError(f'LoRA schema mismatch: {sorted(state.keys() ^ parameters.keys())}')
        with torch.no_grad():
            for k, v in parameters.items():
                if v.shape != state[k].shape: raise ValueError(f'Wrong LoRA shape: {k}')
                v.copy_(state[k])
    pipe.device = torch.device('cuda')
    pipe.dit.cuda().eval(); pipe.vace.cuda().train()
    pipe.scheduler.set_timesteps(1000, training=True, shift=5.)
    return pipe


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=['high', 'low'], required=True)
    p.add_argument('--weights', help='Omit to initialize a fresh rank-32 adapter on the pretrained backbone')
    p.add_argument('--manifest', required=True); p.add_argument('--output-dir', required=True)
    p.add_argument('--backend', required=True); p.add_argument('--base-model-dir')
    p.add_argument('--steps', type=int, default=3264); p.add_argument('--save-every', type=int, default=816)
    p.add_argument('--learning-rate', type=float, default=1e-4); p.add_argument('--seed', type=int, default=17)
    p.add_argument('--wandb-project')
    a = p.parse_args()
    if min(a.steps, a.save_every) < 1: p.error('Counts must be positive')
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()): p.error('Use a new empty output directory')
    source = Path(a.manifest).resolve(); entries = json.loads(source.read_text())
    if not isinstance(entries, list) or not entries: raise ValueError('Expected a nonempty bundle path list')
    paths = [source.parent / x for x in entries]
    provenance = [validate_training_bundle(path) for path in paths]
    torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    pipe = load_expert(a.phase, backend_path=a.backend, base_dir=a.base_model_dir, weights=a.weights)
    parameters = [v for v in pipe.vace.parameters() if v.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=a.learning_rate, weight_decay=.01, foreach=False)
    from diffsynth.diffusion.loss import FlowMatchSFTLoss
    bounds = (0., .358) if a.phase == 'high' else (.358, 1.)
    config = {**vars(a), 'model_name': RENDERER_NAME,
              'conditioning_source': 'predicted_3d', 'training_data': provenance,
              'timestep_index_bounds': bounds, 'sigma_shift': 5., 'lora_rank': 32}
    (out/'training_config.json').write_text(json.dumps(config, indent=2)+'\n')
    run = None
    if a.wandb_project:
        import wandb
        run = wandb.init(project=a.wandb_project, config=config, mode='online')
    generator = torch.Generator().manual_seed(a.seed); order = []
    try:
        with (out/'training.jsonl').open('w') as log:
            for step in range(1, a.steps+1):
                if not order: order = torch.randperm(len(paths), generator=generator).tolist()
                index = order.pop()
                if sha256(paths[index]) != provenance[index]['sha256']:
                    raise ValueError('Encoded training data changed after validation')
                data = load_file(str(paths[index]), device='cuda')
                if set(data) != {'context', 'vace_context', 'input_latents'}:
                    raise ValueError('Unexpected training bundle fields')
                if data['input_latents'].shape != (1, 16, 10, 60, 104) or data['vace_context'].shape != (1, 96, 10, 60, 104):
                    raise ValueError('Expected a 33-frame sample with a separate reference latent')
                data = {k: v.to(torch.bfloat16) for k, v in data.items()}
                if any(not torch.isfinite(v).all() for v in data.values()): raise ValueError('Nonfinite training input')
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    loss = FlowMatchSFTLoss(pipe, **data, vace_scale=1., min_timestep_boundary=bounds[0],
                        max_timestep_boundary=bounds[1], use_gradient_checkpointing=True,
                        use_gradient_checkpointing_offload=True)
                    if not torch.isfinite(loss): raise FloatingPointError('Nonfinite loss')
                    loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
                optimizer.step()
                row = {'step': step, 'loss': float(loss.detach()), 'gradient_norm': float(norm)}
                log.write(json.dumps(row)+'\n'); log.flush(); print(json.dumps(row), flush=True)
                if run: run.log(row, step=step)
                if step % a.save_every == 0 or step == a.steps:
                    state = {k: v.detach().cpu().clone() for k, v in pipe.vace.named_parameters() if v.requires_grad}
                    save_file(state, str(out/f'dexplicit-renderer-{a.phase}-step-{step:06d}.safetensors'),
                              metadata={'format': 'dexplicit-renderer-adapter-v1',
                                        'model_name': RENDERER_NAME,
                                        'conditioning_source': 'predicted_3d',
                                        'source_manifest_sha256': sha256(source)})
                    torch.save({'optimizer': optimizer.state_dict(), 'step': step}, out/f'optimizer-step-{step:06d}.pt')
    finally:
        if run: run.finish()

if __name__ == '__main__': main()
