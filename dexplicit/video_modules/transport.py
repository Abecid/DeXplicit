"""Initial-image VAE appearance transported through persistent GT object points.

E3C (2605.26316), Appendix B.2 motivates the 256/128/128 causal encoder,
two temporal reductions, no spatial reduction and 96-channel VACE addition.
Wan-Move 80c58a7d, wan/modules/trajectory.py:replace_feature motivates carrying
source descriptors along point identities. Unlike its in-place replacement,
we reject invalid source tracks and resolve collisions by geometric depth.
This adaptation uses one permitted image and moving object-local points.
"""
from __future__ import annotations
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def nearest_points(uv, depth, valid, height, width, image_hw=(480,832)):
    """Deterministic closest positive-depth point per image cell, -1 if unknown.

    All valid geometry must participate, including points unobserved at t=0.
    Input is CPU numpy [T,N,...]; source point index breaks exact depth ties.
    """
    uv=np.asarray(uv);depth=np.asarray(depth);valid=np.asarray(valid,bool)
    ih,iw=image_hw
    good=valid & np.isfinite(uv).all(-1) & np.isfinite(depth) & (depth>0)
    good &= (uv[...,0]>=0)&(uv[...,0]<=iw-1)&(uv[...,1]>=0)&(uv[...,1]<=ih-1)
    safe=np.where(good[...,None],uv,0)
    x=np.floor((safe[...,0]+.5)*width/iw).astype(np.int64)
    y=np.floor((safe[...,1]+.5)*height/ih).astype(np.int64)
    result=np.full((len(uv),height,width),-1,np.int32)
    for t in range(len(uv)):
        ids=np.flatnonzero(good[t]);cells=y[t,ids]*width+x[t,ids]
        order=np.lexsort((ids,depth[t,ids],cells))
        cells=cells[order];ids=ids[order]
        take=np.r_[True,cells[1:]!=cells[:-1]] if len(cells) else np.zeros(0,bool)
        result[t].reshape(-1)[cells[take]]=ids[take]
    return result


def build_transport(features, uv, depth, valid, *, source_allowed=None, image_hw=(480,832)):
    """One initial 16-channel feature image + declared geometry only.

    Conservative visibility: retain only the nearest object point in each
    source latent cell. This avoids assigning the same observed cell to hidden
    surfaces. Target occlusion is solved using ALL points before checking
    source visibility. Unknown/newly revealed appearance stays zero + mask=0.
    Sparse geometry cannot certify visibility against unrepresented occluders.
    """
    if features.ndim!=3 or features.shape[0]!=16:raise ValueError('Initial VAE feature image required')
    if not torch.isfinite(features).all():raise ValueError('Nonfinite source features')
    h,w=features.shape[-2:];n=uv.shape[1]
    winners=nearest_points(uv,depth,valid,h,w,image_hw)
    source_visible=np.zeros(n,bool)
    ids=winners[0][winners[0]>=0];source_visible[ids]=True
    if source_allowed is not None:source_visible &= source_allowed
    source_uv=np.asarray(uv[0]).copy()
    source_uv[~source_visible]=0
    grid=torch.from_numpy(source_uv).float()
    ih,iw=image_hw
    grid[...,0]=(grid[...,0]+.5)*2/iw-1
    grid[...,1]=(grid[...,1]+.5)*2/ih-1
    sampled=F.grid_sample(features.float()[None],grid[None,None],mode='bilinear',
        padding_mode='border',align_corners=False)[0,:,0].T.contiguous()
    sampled[~torch.from_numpy(source_visible)]=0
    known=(winners>=0)&source_visible[np.maximum(winners,0)]
    winners=np.where(known,winners,-1).astype(np.int32)
    return dict(point_features=sampled,point_indices=torch.from_numpy(winners),
                source_visible=torch.from_numpy(source_visible))


def materialize(c):
    ids=c['point_indices'].long();mask=ids>=0
    values=c['point_features'][ids.clamp_min(0)]*mask[...,None]
    return torch.cat((values.permute(3,0,1,2),mask.float()[None]),0).contiguous()


class FeatureEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        from diffsynth.models.wan_video_vae import CausalConv3d,ResidualBlock,RMS_norm
        self.input=CausalConv3d(17,256,3,padding=1)
        self.blocks=nn.ModuleList([ResidualBlock(256,256),ResidualBlock(256,128),ResidualBlock(128,128)])
        self.downs=nn.ModuleList([CausalConv3d(256,256,(3,1,1),stride=(2,1,1),padding=(1,0,0)),
                                 CausalConv3d(128,128,(3,1,1),stride=(2,1,1),padding=(1,0,0))])
        self.norm=RMS_norm(128,images=False)
        self.output=CausalConv3d(128,96,3,padding=1)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)

    def encode(self,x):
        x=self.input(x)
        for i,block in enumerate(self.blocks):
            x=block(x)[0]
            if i<2:x=self.downs[i](x)
        return self.output(F.silu(self.norm(x)))

    def forward(self,x):
        if x.ndim!=5 or x.shape[1:]!=(17,33,60,104):raise ValueError('Expected 16 appearance channels + coverage, all 33 frames')
        # Per-sample encoder checkpointing bounds activation/workspace memory.
        # Native VACE inference caches this result across all denoising steps.
        fn=lambda u:checkpoint(self.encode,u,use_reentrant=False) if torch.is_grad_enabled() else self.encode(u)
        return torch.cat([fn(u[None]) for u in x],0)
