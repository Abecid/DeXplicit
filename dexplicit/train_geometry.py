# SPDX-License-Identifier: Apache-2.0
"""Fine-tune the DeXplicit 3D model with its native structured and metric losses."""
from __future__ import annotations
import argparse
from functools import partial
import json
from pathlib import Path
import torch
from safetensors.torch import save_file
from .geometry import CONFIG, dataset, load_model, seed_everything


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights',required=True)
    p.add_argument('--data-dir',action='append',required=True)
    p.add_argument('--output-dir',required=True)
    p.add_argument('--config',default=str(CONFIG))
    p.add_argument('--steps',type=int,default=1000)
    p.add_argument('--batch-size',type=int,default=1)
    p.add_argument('--learning-rate',type=float,default=1e-4)
    p.add_argument('--save-every',type=int,default=100)
    p.add_argument('--wandb-project',default=None)
    p.add_argument('--device',default='cuda')
    a=p.parse_args()
    if a.steps<1 or a.batch_size<1 or a.save_every<1: p.error('Counts must be positive')
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    if (out/'training.jsonl').exists():p.error('Choose a fresh output directory to preserve prior runs')
    torch.set_num_threads(4)
    model,args=load_model(a.weights,a.config,a.device,train=True)
    seed_everything(args.seed)
    ds=dataset(a.data_dir,args,deterministic=False)
    from dataset_components.hoi_dataset import hoi_collate_fn
    loader=torch.utils.data.DataLoader(ds,batch_size=a.batch_size,shuffle=True,num_workers=0,
                                      collate_fn=partial(hoi_collate_fn,args=args),drop_last=False)
    optimizer=torch.optim.AdamW(model.parameters(),lr=a.learning_rate,weight_decay=args.weight_decay)
    run=None
    if a.wandb_project:
        import wandb
        run=wandb.init(project=a.wandb_project,config=vars(a),mode='online')
    iterator=iter(loader)
    try:
        with (out/'training.jsonl').open('w') as log:
            for step in range(1,a.steps+1):
                try:batch=next(iterator)
                except StopIteration:iterator=iter(loader);batch=next(iterator)
                batch={k:v.to(a.device) if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16,enabled=a.device.startswith('cuda')):
                    prediction=model(batch,training=True)
                    loss,metrics=model.loss_fn(prediction,batch,training=True)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
                loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),args.grad_clip_max_norm,error_if_nonfinite=True)
                optimizer.step()
                row={'step':step,'loss':float(loss.detach()),'gradient_norm':float(norm)}
                log.write(json.dumps(row)+'\n');log.flush();print(json.dumps(row),flush=True)
                if run:run.log(row,step=step)
                if step%a.save_every==0 or step==a.steps:
                    target=out/f'geometry-step-{step:06d}.safetensors'
                    save_file({k:v.detach().cpu().contiguous().clone() for k,v in model.state_dict().items()},str(target))
                    torch.save({'optimizer':optimizer.state_dict(),'step':step},out/f'optimizer-step-{step:06d}.pt')
    finally:
        if run:run.finish()


if __name__=='__main__':main()
