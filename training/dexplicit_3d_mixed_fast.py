"""Batched rigid geometry with node-level deformation controls.

Only this entry point installs the vectorized geometry operations.
Model parameter names and losses remain unchanged.
"""
from __future__ import annotations
import torch


def gt_pose(self,gt,canonical,ids,valid,num_objects):
    from training.batched_kabsch import batched_weighted_kabsch
    b,t,n,_=gt.shape
    member=ids[:,None]==torch.arange(num_objects,device=gt.device)[None,:,None]
    r,p,v=batched_weighted_kabsch(canonical[:,None].expand(b,num_objects,n,3),
        gt[:,None].expand(b,num_objects,t,n,3),weights=member.float(),
        valid=valid.bool()[:,None].expand(b,num_objects,t,n),rank_epsilon=None)
    return r.to(gt.dtype),p.to(gt.dtype),v


@torch.no_grad()
def deformability(self,outputs,data):
    predicted=outputs['scene_flows']
    target=data['gt_scene_flows'].to(predicted.device).float()
    canonical=data['point_canonical_xyz'].to(predicted.device).float()
    ids=data['point_object_id'].to(predicted.device).long()
    object_valid=outputs['object_valid'].bool()
    valid=(data['scene_exists'].to(predicted.device).squeeze(-1).bool()
           & data['scene_supervised_mask'].to(predicted.device).squeeze(-1).bool())
    r,p,good=self._gt_pose(target,canonical,ids,valid,object_valid.shape[1])
    member=ids[:,None]==torch.arange(object_valid.shape[1],device=ids.device)[None,:,None]
    rigid=torch.einsum('botij,bnj->botni',r[:,:,1:],canonical)+p[:,:,1:,None]
    error=torch.linalg.vector_norm(rigid-target[:,None,1:],dim=-1)
    mask=valid[:,None,1:]&member[:,:,None]
    count=mask.sum(-1)
    residual=(error*mask).sum(-1)/count.clamp_min(1)
    residual_valid=good[:,:,1:]&(count>0)
    residual=torch.where(residual_valid,residual,0.)
    probability=((residual-self.deformation_graph_gate_rigid_floor_m)/
        (self.deformation_graph_gate_full_scale_m-self.deformation_graph_gate_rigid_floor_m)).clamp(0,1)
    return probability,residual,residual_valid&object_valid[:,:,None]


def hand_carriers(hand_xyz,hand_valid,*,unlabelled=None):
    from training import dexplicit_3d_mixed_fast_geometry as fast
    carriers=fast.hand_carriers(hand_xyz,hand_valid)
    if unlabelled is None:return carriers
    unlabeled=unlabelled.to(hand_xyz.device).bool()
    if unlabeled.shape!=(hand_xyz.shape[0],):raise ValueError('Unlabelled-control mask must be B')
    rotation,translation,valid=carriers.rotation,carriers.translation,carriers.valid
    if rotation.shape[:3]!=(hand_xyz.shape[0],hand_xyz.shape[1]-1,3):
        raise ValueError(f'Unexpected carrier layout {tuple(rotation.shape)}; expected (B,H,3,3,3)')
    # Unknown identity permits aggregate transport, not fake named hands: reset the
    # left/right carriers (index 0,1) of unlabelled rows with a broadcast mask
    # (layout-safe; advanced-index assignment failed at runtime on GPU batches).
    carrier_reset=torch.tensor([True,True,False],device=rotation.device)
    reset=(unlabeled[:,None,None]&carrier_reset[None,None,:])           # (B,1,3)
    eye=torch.eye(3,device=rotation.device,dtype=rotation.dtype)
    rotation=torch.where(reset[...,None,None],eye.expand_as(rotation),rotation)
    translation=torch.where(reset[...,None],torch.zeros_like(translation),translation)
    valid=valid&~reset
    return fast.CarrierTransforms(rotation,translation,valid)


def install():
    # Precede the predictor import, whose from-import binds these functions.
    import training.carrier_gauge_core as core
    from training import dexplicit_3d_mixed_fast_geometry as fast
    for name in ('fit_rigid_trajectory','gauge_fix','warp_local_graph','render_objects'):
        setattr(core,name,getattr(fast,name))
    core.hand_carriers=hand_carriers
    from training.dynamics_experiment_v2 import V2ExperimentalBaseModel
    from training.deformation_graph_model import DeformationGraphExperimentalModel
    V2ExperimentalBaseModel._gt_pose=gt_pose
    DeformationGraphExperimentalModel._gt_deformability_target=deformability
