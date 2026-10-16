"""Training-only physical-regime masking; FULL's forward/loss remain unchanged.

Contexts are local to a loss call, not global material labels. They do not alter
model parameters, state dicts, target coordinates or inference inputs. In
particular, GT rigid fits used by the deformability gate must NOT be masked.
"""
from __future__ import annotations
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace
from functools import wraps
from typing import Mapping
import torch

_RIGID = ContextVar('mixed_representation_rigid', default=None)
_POSE_ONLY = ContextVar('mixed_representation_pose_only', default=False)
DEFAULT_REGIMES = {'hot3d': 'rigid', 'oakink2': 'rigid', 'phystwin': 'deformable'}


def rigid_objects(data: Mapping, object_valid: torch.Tensor, regimes: Mapping) -> torch.Tensor:
    """Metadata selects training priors, not rendered motion or inferred material.

    Optional source overrides: "dataset:source_clip_id" -> regime. 'unknown'
    removes rigid priors too. No labels are inferred from a quiet trajectory.
    OakInk2's 'rigid' refers to its supplied independently posed instances, NOT
    a guessed single rigid transform for an articulated semantic assembly.
    """
    labels = data.get('__dataset__')
    if not isinstance(labels, (tuple, list)) or len(labels) != object_valid.shape[0]:
        raise ValueError('HYBRID loss needs the frozen dataset labels, one per row')
    sources = data.get('source_clip_id', [''] * len(labels))
    if len(sources) != len(labels):
        raise ValueError('source identity batch length mismatch')
    flags = []
    for dataset, source in zip(labels, sources):
        regime = regimes.get(f'{dataset}:{source}', regimes.get(str(dataset)))
        if regime not in {'rigid', 'deformable', 'unknown'}:
            raise ValueError(f'No declared physical training regime for {dataset}:{source}')
        flags.append(regime == 'rigid')
    mask = torch.tensor(flags, dtype=torch.bool, device=object_valid.device)
    return object_valid.bool() & mask[:, None]


@contextmanager
def hybrid_regularization(rigid: torch.Tensor):
    if rigid.ndim != 2 or rigid.dtype != torch.bool:
        raise ValueError('rigid prior mask must be boolean B,O')
    token = _RIGID.set(rigid)
    try:
        yield
    finally:
        _RIGID.reset(token)


def mask_pose_valid(valid: torch.Tensor) -> torch.Tensor:
    mask = _RIGID.get()
    if not _POSE_ONLY.get() or mask is None:
        return valid
    if valid.shape[:2] != mask.shape:
        raise ValueError('rigid mask/GT pose axes disagree')
    return valid & mask[..., None]


def install_loss_scopes():
    """Opt-in wrappers around three named inherited operations, no source edits.

    Installed once after the mixed50 fast fitter. FULL bypasses each wrapper.
    The V2 scope ends BEFORE the graph's GT-deformability target is built.
    """
    import training.dynamics_experiment_v2 as base
    import training.deformation_graph_model as graph
    cls = base.V2ExperimentalBaseModel
    if getattr(cls, '_mixed_representation_loss_scopes', False):
        return
    original_loss, original_pose = cls.loss_fn, cls._gt_pose
    original_arap, original_magnitude = graph.arap_graph_loss, graph.local_transform_magnitude

    @wraps(original_pose)
    def pose(self, *args, **kwargs):
        r, b, valid = original_pose(self, *args, **kwargs)
        return r, b, mask_pose_valid(valid)

    @wraps(original_loss)
    def loss(self, outputs, data, training=True):
        if _RIGID.get() is None:
            return original_loss(self, outputs, data, training=training)
        token = _POSE_ONLY.set(True)
        try:
            return original_loss(self, outputs, data, training=training)
        finally:
            _POSE_ONLY.reset(token)

    @wraps(original_arap)
    def arap(topology, rotation, translation, **kwargs):
        mask = _RIGID.get()
        if mask is not None:
            if topology.node_valid.shape[:2] != mask.shape:
                raise ValueError('rigid mask/graph axes disagree')
            topology = replace(topology,
                node_valid=topology.node_valid & mask[..., None],
                node_neighbor_valid=topology.node_neighbor_valid & mask[..., None, None])
        return original_arap(topology, rotation, translation, **kwargs)

    @wraps(original_magnitude)
    def magnitude(translation, rotation, valid, **kwargs):
        mask = _RIGID.get()
        if mask is not None:
            valid = valid & mask[..., None]
        return original_magnitude(translation, rotation, valid, **kwargs)

    cls._gt_pose, cls.loss_fn = pose, loss
    graph.arap_graph_loss, graph.local_transform_magnitude = arap, magnitude
    cls._mixed_representation_loss_scopes = True
