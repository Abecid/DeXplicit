"""Identity-preserving 2D heatmaps and metric 3D hand actions for native VACE.

Inspired by Generated Reality's hybrid image/3D token addition and DexSIM's
42-channel Gaussian heatmaps. This is a VACE adaptation, not either paper's
exact architecture. All 33 action frames are controls; future RGB is forbidden.
"""
from __future__ import annotations
from contextlib import contextmanager
import torch
from torch import nn

JOINTS = 42
ACTION_DIM = JOINTS * 15 + 12


def pack_frames(x):
    """[B,4n+1,C,...] -> [B,n+1,4C,...], preserving every action frame."""
    if x.ndim < 3 or (x.shape[1] - 1) % 4:
        raise ValueError('Require 4n+1 action frames')
    y = torch.cat((x[:, :1].expand(-1, 4, *x.shape[2:]), x[:, 1:]), 1)
    return y.reshape(x.shape[0], (x.shape[1]+3)//4, 4*x.shape[2], *x.shape[3:])


def action_features(c):
    cam, ref, valid = c['camera_xyz'].float(), c['reference_xyz'].float(), c['valid'].bool()
    if cam.shape[2:] != (2, 21, 3) or ref.shape != cam.shape or valid.shape != cam.shape[:-1]:
        raise ValueError('Require ordered LEFT/RIGHT 21-joint coordinates')
    cam = torch.where(valid[..., None], cam, 0)
    ref = torch.where(valid[..., None], ref, 0)
    local_valid = valid & valid[:, :, :, :1]
    local = torch.where(local_valid[..., None], cam-cam[:, :, :, :1], 0)
    motion_valid = torch.cat((torch.zeros_like(valid[:, :1]), valid[:, 1:] & valid[:, :-1]), 1)
    velocity = torch.cat((torch.zeros_like(ref[:, :1]), ref[:, 1:]-ref[:, :-1]), 1)
    velocity = torch.where(motion_valid[..., None], velocity, 0)
    per_joint = torch.cat((cam, ref, local, velocity, valid[..., None],
                           local_valid[..., None], motion_valid[..., None]), -1)
    out = torch.cat((per_joint.flatten(2), c['relative_camera'].float().flatten(2)), -1)
    if out.shape[-1] != ACTION_DIM or not torch.isfinite(out).all():
        raise ValueError('Invalid metric action features')
    return out


def heatmaps(c, height=60, width=104, sigma=1.0):
    """One channel per anatomical joint; no clipping invalid points to borders.

    Coordinates refer to 480x832 image pixel centers; heatmap cells correspond
    to their 8x8 image regions. In-view is not an occlusion visibility label.
    """
    uv = c['uv'].float().flatten(2, 3)
    good = (c['in_view'] & c['valid']).flatten(2, 3)
    uv = torch.where(good[..., None], uv, 0)
    x = (torch.arange(width, device=uv.device, dtype=torch.float32)+.5)*8-.5
    y = (torch.arange(height, device=uv.device, dtype=torch.float32)+.5)*8-.5
    dx = (x[None, None, None, None, :] - uv[..., 0, None, None])/8
    dy = (y[None, None, None, :, None] - uv[..., 1, None, None])/8
    return torch.exp(-(dx.square()+dy.square())/(2*sigma*sigma)) * good[..., None, None]


class HandConditioner(nn.Module):
    def __init__(self, dim=1536, width=256):
        super().__init__()
        self.spatial = nn.Sequential(nn.Conv3d(4*JOINTS, 128, (1,2,2), stride=(1,2,2)),
                                     nn.SiLU(), nn.Conv3d(128, dim, 1))
        self.action = nn.Sequential(nn.Linear(4*ACTION_DIM, width), nn.SiLU(), nn.Linear(width, dim))
        for layer in (self.spatial[-1], self.action[-1]):
            nn.init.zeros_(layer.weight); nn.init.zeros_(layer.bias)

    def forward(self, c):
        hm = pack_frames(heatmaps(c)).permute(0,2,1,3,4)
        spatial = self.spatial(hm)
        action = self.action(pack_frames(action_features(c))).transpose(1,2)[...,None,None]
        return spatial + action


class HandVace(nn.Module):
    """Keep native attention/noisy-video/timestep path and add hand token input.

    Explicit batch fusion avoids per-sample forward-hook broadcasting bugs.
    The scope encloses forward AND backward; checkpoint recomputation sees the
    same immutable inputs. Native reference tokens receive no new residual.
    """
    def __init__(self, native, dim=1536):
        super().__init__()
        self.native = native
        self.hand = HandConditioner(dim)
        from .transport import FeatureEncoder
        self.transport = FeatureEncoder()
        self.transport_cached = None
        self.vace_layers_mapping = native.vace_layers_mapping
        self.vace_layers = native.vace_layers
        self.vace_in_dim = native.vace_in_dim
        self.bound = None
        self.cached = None

    @contextmanager
    def condition(self, c, enabled=True):
        if self.bound is not None:
            raise RuntimeError('Nested hand condition scopes are forbidden')
        required = {'feature_transport','camera_xyz','reference_xyz','valid','uv','in_view','relative_camera'}
        if set(c) != required:
            raise ValueError('Only declared GT pose/camera controls may enter the hand branch')
        self.bound = (c, enabled)
        try:
            yield
        finally:
            self.bound = self.cached = self.transport_cached = None

    def forward(self, x, vace_context, context, t_mod, freqs,
                use_gradient_checkpointing=False, use_gradient_checkpointing_offload=False):
        from diffsynth.core.gradient import gradient_checkpoint_forward
        if self.bound is None:
            raise RuntimeError('Explicit hand action is mandatory for this experiment')
        values, enabled = self.bound
        # E3C feature-context addition: RGB/native reference and hand paths remain.
        if values['feature_transport'] is not None:
            extra_context = self.transport_cached
            if extra_context is None:
                extra_context = self.transport(values['feature_transport'])
                if not torch.is_grad_enabled(): self.transport_cached = extra_context
            if extra_context.shape != (len(vace_context),96,9,60,104):
                raise ValueError('Transport temporal/spatial alignment mismatch')
            residual = torch.cat((torch.zeros_like(extra_context[:,:,:1]),extra_context),2)
            vace_context = vace_context + residual.to(vace_context.dtype)
        # Preserve native per-sample convolution order, then fuse the full batch.
        c = torch.cat([self.native.vace_patch_embedding(u.unsqueeze(0)) for u in vace_context])
        if enabled:
            extra = self.cached
            if extra is None:
                extra = self.hand(values)
                if not torch.is_grad_enabled(): self.cached = extra
            if extra.shape != (c.shape[0], c.shape[1], c.shape[2]-1, *c.shape[3:]):
                raise ValueError('Hand/reference/video batch or time alignment mismatch')
            c = c + torch.cat((torch.zeros_like(c[:,:,:1]), extra.to(c.dtype)), 2)
        c = c.flatten(2).transpose(1,2)
        c = torch.cat((c, c.new_zeros(c.shape[0], x.shape[1]-c.shape[1], c.shape[2])),1)
        for block in self.native.vace_blocks:
            c = gradient_checkpoint_forward(block, use_gradient_checkpointing,
                use_gradient_checkpointing_offload, c, x, context, t_mod, freqs)
        return torch.unbind(c)[:-1]
