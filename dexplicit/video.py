# SPDX-License-Identifier: Apache-2.0
"""Compact Wan 1.3B renderer with explicit hands and appearance transport."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import numpy as np
import torch
from safetensors.torch import load_file

BACKEND_COMMIT = 'ce9f4541d0f4ae47a138337e59066dd633207aa7'
BASE_REPO = 'Wan-AI/Wan2.1-VACE-1.3B'
BASE_REVISION = '574e6a744642ce3bee319afc31496b88bde8aac4'
HAND_KEYS = {'camera_xyz','reference_xyz','valid','uv','in_view','relative_camera'}


def backend(path=None):
    if path:
        sys.path.insert(0,str(Path(path).resolve()))
    from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
    return WanVideoPipeline, ModelConfig


def base_path(filename, override=None):
    if override:
        return str(Path(override).resolve())
    from huggingface_hub import hf_hub_download
    return hf_hub_download(BASE_REPO, filename, revision=BASE_REVISION)


def runtime():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.set_float32_matmul_precision('highest')


def load_renderer(weights, *, model_path=None, backend_path=None, train=False):
    runtime()
    Pipeline,Config=backend(backend_path)
    pipe=Pipeline.from_pretrained(device='cpu',torch_dtype=torch.bfloat16,
        model_configs=[Config(path=base_path('diffusion_pytorch_model.safetensors',model_path))],
        tokenizer_config=None,redirect_common_files=False)
    if pipe.dit.dim!=1536 or len(pipe.dit.blocks)!=30 or pipe.vace.vace_in_dim!=96:
        raise ValueError('Expected the pinned Wan2.1-VACE-1.3B backbone')
    pipe.requires_grad_(False)
    from diffsynth.diffusion.training_module import DiffusionTrainingModule
    pipe.vace=DiffusionTrainingModule().add_lora_to_model(pipe.vace,['q','k','v','o','ffn.0','ffn.2'],32,upcast_dtype=torch.float32)
    from .video_modules.hand import HandVace
    pipe.vace=HandVace(pipe.vace)
    tensors=load_file(str(weights))
    parameters={k:v for k,v in pipe.vace.named_parameters() if v.requires_grad}
    if tensors.keys()!=parameters.keys():
        raise ValueError(f'Adapter schema mismatch; missing={sorted(parameters.keys()-tensors.keys())}, extra={sorted(tensors.keys()-parameters.keys())}')
    with torch.no_grad():
        for k,v in parameters.items():
            if v.shape!=tensors[k].shape or not torch.isfinite(tensors[k]).all():
                raise ValueError(f'Invalid adapter tensor: {k}')
            v.copy_(tensors[k])
    pipe.device=torch.device('cuda')
    pipe.dit.cuda().eval()
    pipe.vace.cuda().train(train)
    pipe.scheduler.set_timesteps(1000,training=True,shift=5.)
    return pipe


def load_bundle(path, *, training=False):
    x=load_file(str(path),device='cuda')
    required={'reference','reactive','inactive','context','negative_context',
              'transport.point_features','transport.point_indices','transport.source_visible',
              *('hand.'+k for k in HAND_KEYS)}
    if training:required.add('target_latents')
    if set(x)!=required:raise ValueError(f'Input schema mismatch: {sorted(set(x)^required)}')
    from .video_modules.transport import materialize
    transport={k.removeprefix('transport.'):v for k,v in x.items() if k.startswith('transport.')}
    hands={k:x['hand.'+k] for k in HAND_KEYS}
    hands['feature_transport']=materialize(transport)[None]
    reference=x['reference'].to(torch.bfloat16)
    reactive=x['reactive'].to(torch.bfloat16)
    inactive=x['inactive'].to(torch.bfloat16)
    if reference.shape!=(1,16,1,60,104) or reactive.shape!=(1,16,9,60,104):
        raise ValueError('Expected one 33-frame 480x832 sample with a separate reference slot')
    ref=torch.cat((reference,torch.zeros_like(reference)),1)
    video=torch.cat((inactive,reactive),1)
    mask=torch.ones((1,64,10,60,104),device='cuda',dtype=torch.bfloat16);mask[:,:,:1]=0
    control=torch.cat((torch.cat((ref,video),2),mask),1)
    return {'vace_context':control,'hand_condition':hands,'context':x['context'].to(torch.bfloat16),
            'negative_context':x['negative_context'].to(torch.bfloat16),
            **({'input_latents':x['target_latents'].to(torch.bfloat16)} if training else {})}


@torch.no_grad()
def sample(pipe,bundle,*,seed=17,steps=30,cfg=5.):
    pipe.vace.eval()
    z=pipe.generate_noise((1,16,10,60,104),seed=seed,rand_device='cpu')
    x=torch.cat((z[:,:,-1:],z[:,:,:-1]),2).cuda().to(torch.bfloat16)
    pipe.scheduler.set_timesteps(steps,shift=5.)
    with pipe.vace.condition(bundle['hand_condition']),torch.autocast('cuda',dtype=torch.bfloat16):
        for timepoint in pipe.scheduler.timesteps:
            t=timepoint.reshape(1).to(device='cuda',dtype=torch.bfloat16)
            def prediction(context):
                return pipe.model_fn(dit=pipe.dit,vace=pipe.vace,latents=x,timestep=t,context=context,
                    vace_context=bundle['vace_context'],vace_scale=1.,use_gradient_checkpointing=False,
                    use_gradient_checkpointing_offload=False)
            pos=prediction(bundle['context']);neg=prediction(bundle['negative_context'])
            x=pipe.scheduler.step(neg+cfg*(pos-neg),timepoint,x)
            if not torch.isfinite(x).all():raise FloatingPointError('Nonfinite generated latent')
    return x[:,:,1:].detach()


@torch.no_grad()
def decode(latents, *, vae_path=None,backend_path=None):
    Pipeline,Config=backend(backend_path)
    codec=Pipeline.from_pretrained(device='cpu',torch_dtype=torch.bfloat16,
        model_configs=[Config(path=base_path('Wan2.1_VAE.pth',vae_path))],
        tokenizer_config=None,redirect_common_files=False)
    codec.device=torch.device('cuda');codec.vae.cuda().eval().requires_grad_(False)
    rgb=codec.vae.decode(latents,device='cuda',tiled=False)
    frames=((rgb[0].float().permute(1,2,3,0).clamp(-1,1)+1)*127.5).round().byte().cpu().numpy()
    if frames.shape!=(33,480,832,3):raise ValueError(f'Unexpected decoded shape: {frames.shape}')
    return frames


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights',required=True);p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    p.add_argument('--base-model');p.add_argument('--vae');p.add_argument('--backend')
    p.add_argument('--seed',type=int,default=17);p.add_argument('--steps',type=int,default=30);p.add_argument('--cfg',type=float,default=5.)
    p.add_argument('--latents-output');p.add_argument('--rgb-output')
    a=p.parse_args()
    pipe=load_renderer(a.weights,model_path=a.base_model,backend_path=a.backend)
    latent=sample(pipe,load_bundle(a.input),seed=a.seed,steps=a.steps,cfg=a.cfg)
    if a.latents_output:
        from safetensors.torch import save_file
        save_file({'latents':latent.cpu().contiguous()},a.latents_output)
    frames=decode(latent,vae_path=a.vae,backend_path=a.backend)
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    import imageio.v2 as imageio
    imageio.mimwrite(a.output,frames,fps=15,codec='libx264',quality=8,macro_block_size=None)
    if a.rgb_output:np.savez_compressed(a.rgb_output,rgb=frames)
    print(json.dumps({'output':a.output,'shape':list(frames.shape),'seed':a.seed,'steps':a.steps,'cfg':a.cfg}))


if __name__=='__main__':main()
