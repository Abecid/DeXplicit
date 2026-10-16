"""Action-bound keys for frozen PhysTwin predictions (no target-side bytes)."""
from __future__ import annotations
import hashlib
import numpy as np
import torch

KEY_PROTOCOL = 'phystwin_inputs_v1'


def inference_inputs(batch):
    names=('__domain__','scene_flows','scene_features','scene_exists','robot_flows',
           'robot_features','robot_exists','action_is_unlabelled','point_object_id','point_canonical_xyz',
           'hand_joints_world','hand_joint_valid','mano_valid')
    result={k:batch[k] for k in names if k in batch}
    result.update(batch.get('pointworld_object_context',{}))
    for key in ('scene_flows','scene_features','scene_exists','robot_features'):
        result[key]=result[key][:,:1].expand_as(result[key])
    return result


def prediction_key(batch, row=0):
    clean=inference_inputs(batch)
    if not bool(clean['action_is_unlabelled'][row]):
        raise ValueError('PhysTwin cache key requires explicit unlabelled controls')
    h=hashlib.sha256(KEY_PROTOCOL.encode())
    for key in sorted(clean):
        value=clean[key]
        if key=='__domain__':
            h.update(key.encode()+str(value[row]).encode());continue
        x=value[row]
        if key in ('scene_flows','scene_features','scene_exists','robot_features'):
            x=x[:1]
        if not isinstance(x,torch.Tensor):raise TypeError(f'{key} must be a tensor')
        if x.is_floating_point():x=x.float()
        array=np.ascontiguousarray(x.detach().cpu().numpy())
        h.update(key.encode()+str(array.shape).encode()+str(array.dtype).encode()+array.tobytes())
    return h.hexdigest()
