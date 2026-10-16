"""Pack documented renderer tensors from NPZ into a safe, portable input bundle."""
import argparse
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import save_file
from .video import HAND_KEYS


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    p.add_argument('--training',action='store_true',help='Require and include target_latents for training')
    a=p.parse_args()
    required={'reference','reactive','inactive','context','negative_context',
        'transport.point_features','transport.point_indices','transport.source_visible',*('hand.'+k for k in HAND_KEYS)}
    if a.training:required.add('target_latents')
    with np.load(a.input,allow_pickle=False) as f:
        if set(f.files)!=required:raise ValueError(f'Schema mismatch: {sorted(set(f.files)^required)}')
        tensors={k:torch.from_numpy(f[k].copy()).contiguous() for k in f.files}
    shapes={'reference':(1,16,1,60,104),'reactive':(1,16,9,60,104),'inactive':(1,16,9,60,104),
            'hand.camera_xyz':(1,33,2,21,3),'hand.reference_xyz':(1,33,2,21,3),
            'hand.valid':(1,33,2,21),'hand.uv':(1,33,2,21,2),'hand.in_view':(1,33,2,21),
            'hand.relative_camera':(1,33,3,4),'transport.point_indices':(33,60,104),
            'target_latents':(1,16,10,60,104)}
    for k,t in tensors.items():
        if k in shapes and tuple(t.shape)!=shapes[k]:raise ValueError(f'{k}: expected {shapes[k]}, got {tuple(t.shape)}')
        if t.dtype.is_floating_point and not torch.isfinite(t).all():raise ValueError(f'Nonfinite field: {k}')
    for key in ['context','negative_context']:
        if tensors[key].shape != (1,512,4096):
            raise ValueError(f'{key} must be [1,512,4096]')
    features=tensors['transport.point_features']
    if features.ndim!=2 or features.shape[0]==0 or features.shape[1]!=16:
        raise ValueError('transport.point_features must be [N,16] with N > 0')
    n=features.shape[0]
    if tensors['transport.source_visible'].shape!=(n,):
        raise ValueError('transport.source_visible must match the number of points')
    for key in ['hand.valid','hand.in_view','transport.source_visible']:
        if tensors[key].dtype!=torch.bool:raise ValueError(f'{key} must be boolean')
    ids=tensors['transport.point_indices']
    if ids.dtype not in (torch.int32,torch.int64) or (ids < -1).any() or (ids >= n).any():
        raise ValueError('Point indices must be integers in [-1,N-1]')
    known=ids>=0
    if not tensors['transport.source_visible'][ids[known].long()].all():
        raise ValueError('A point index refers to an unavailable source descriptor')
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    save_file(tensors,a.output,metadata={'format':'dexplicit-video-training-v1' if a.training else 'dexplicit-video-input-v1'})
    print(a.output)

if __name__=='__main__':main()
