# SPDX-License-Identifier: Apache-2.0
"""Fine-tune the compact renderer with the original native flow-matching loss."""
import argparse,json
from pathlib import Path
import torch
from safetensors.torch import save_file
from .video import load_renderer,load_bundle


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights',required=True);p.add_argument('--manifest',required=True);p.add_argument('--output-dir',required=True)
    p.add_argument('--base-model');p.add_argument('--backend');p.add_argument('--steps',type=int,default=1000)
    p.add_argument('--learning-rate',type=float,default=1e-4);p.add_argument('--save-every',type=int,default=100)
    p.add_argument('--seed',type=int,default=17);p.add_argument('--wandb-project')
    a=p.parse_args()
    if a.steps<1 or a.save_every<1:p.error('Counts must be positive')
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    if (out/'training.jsonl').exists():p.error('Choose a fresh output directory')
    paths=json.loads(Path(a.manifest).read_text())
    if not isinstance(paths,list) or not paths:raise ValueError('Manifest must be a nonempty list of training bundle paths')
    paths=[str((Path(a.manifest).parent/Path(x)).resolve()) for x in paths]
    pipe=load_renderer(a.weights,model_path=a.base_model,backend_path=a.backend,train=True)
    torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
    parameters=[v for v in pipe.vace.parameters() if v.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=a.learning_rate,weight_decay=.01,foreach=False)
    from diffsynth.diffusion.loss import FlowMatchSFTLoss
    run=None
    if a.wandb_project:
        import wandb
        run=wandb.init(project=a.wandb_project,config=vars(a),mode='online')
    generator=torch.Generator().manual_seed(a.seed);order=[]
    try:
        with (out/'training.jsonl').open('w') as log:
            for step in range(1,a.steps+1):
                if not order:order=torch.randperm(len(paths),generator=generator).tolist()
                b=load_bundle(paths[order.pop()],training=True)
                optimizer.zero_grad(set_to_none=True)
                pipe.scheduler.set_timesteps(1000,training=True,shift=5.)
                with pipe.vace.condition(b['hand_condition']),torch.autocast('cuda',dtype=torch.bfloat16):
                    loss=FlowMatchSFTLoss(pipe,input_latents=b['input_latents'],context=b['context'],
                        vace_context=b['vace_context'],vace_scale=1.,min_timestep_boundary=0.,max_timestep_boundary=1.,
                        use_gradient_checkpointing=True,use_gradient_checkpointing_offload=True)
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
                    loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
                optimizer.step()
                row={'step':step,'loss':float(loss.detach()),'gradient_norm':float(norm)}
                log.write(json.dumps(row)+'\n');log.flush();print(json.dumps(row),flush=True)
                if run:run.log(row,step=step)
                if step%a.save_every==0 or step==a.steps:
                    state={k:v.detach().cpu().clone() for k,v in pipe.vace.named_parameters() if v.requires_grad}
                    save_file(state,str(out/f'video-step-{step:06d}.safetensors'))
                    torch.save({'optimizer':optimizer.state_dict(),'step':step},out/f'optimizer-step-{step:06d}.pt')
    finally:
        if run:run.finish()


if __name__=='__main__':main()
