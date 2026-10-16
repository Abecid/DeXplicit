"""Three explicit mixed-data alternatives; no AR, affinity or new classifier.

FULL: native dexplicit_3d. HYBRID: identical forward with material-aware loss priors.
POINT: native H35 point/action path, with no object/pose/graph forward or loss.
All learned tensor keys/shapes stay compatible with the same trained dexplicit_3d file.
"""
from __future__ import annotations
import json
from pathlib import Path
import types
import torch
from pointworld.base import BaseModel, MANO_PARAM_KEYS
from training.dexplicit_3d_model import DeXplicit3DModel
from training.mixed_representation_losses import rigid_objects, hybrid_regularization

ARMS = ('FULL', 'POINT', 'HYBRID')
INPUT_KEYS = {'scene_flows', 'scene_features', 'scene_exists', 'robot_flows',
              'robot_features', 'robot_exists', '__domain__', 'action_is_unlabelled',
              'hand_joints_world', 'hand_joint_valid', *MANO_PARAM_KEYS}


def point_inputs(data):
    # Future scene geometry, targets, object IDs and canonical parts are not inputs.
    result = {k: v for k, v in data.items() if k in INPUT_KEYS}
    for key in ('scene_flows', 'scene_features', 'scene_exists'):
        result[key] = result[key][:, :1]
    return result


def point_forward(self, sx, sf, sv, hx, hf, hv, *, normalize_fn, **kwargs):
    """Execute all 34 output rows with native full-window action timing."""
    scene, _ = self.backbone(sx, sf, sv, hx, hf, hv)
    batch, points, _ = scene.shape
    future = self.direct(scene).view(batch, points, self.T - 1, 3).permute(0, 2, 1, 3)
    delta = torch.cat((future.new_zeros(batch, 1, points, 3), future), 1)
    variance = self.var(scene).permute(0, 2, 1).unsqueeze(-1)
    variance = torch.cat((torch.zeros_like(variance[:, :1]), variance), 1)
    return {'pred': normalize_fn(delta), 'log_var': variance}


class MixedRepresentationModel(DeXplicit3DModel):
    def __init__(self, args, *a, **kw):
        arm = args.mixed_representation_arm
        if arm not in ARMS:
            raise ValueError(f'Expected one of {ARMS}, not {arm}')
        self.mixed_representation_arm = arm
        self.mixed_representation_regimes = json.loads(args.mixed_representation_regimes)
        super().__init__(args, *a, **kw)
        if arm == 'POINT':
            self.dynamics_predictor.forward = types.MethodType(point_forward, self.dynamics_predictor)
        # No new random initialization, trainable head or state-dict key.
        self.mixed_representation_contract = {
            'version': 1, 'arm': arm, 'regimes': self.mixed_representation_regimes,
            'forward': 'native direct point H35' if arm == 'POINT' else 'unchanged full dexplicit_3d H35',
            'requires_object_canonical_inputs': arm != 'POINT',
            'requires_material_labels_at_inference': False,
            'pointworld_reproduction': False,
            'plan_sha256': args.mixed_representation_plan_sha256,
        }
        if int(getattr(self, 'rank', 0)) == 0:
            root = Path(args.output_dir); root.mkdir(parents=True, exist_ok=True)
            (root / 'mixed_representation_model_contract.json').write_text(
                json.dumps(self.mixed_representation_contract, sort_keys=True, indent=2) + '\n')

    def forward(self, data, training=True, encoded_scene_feat0=None):
        if self.mixed_representation_arm == 'POINT':
            return BaseModel.forward(self, point_inputs(data), training=training,
                                     encoded_scene_feat0=encoded_scene_feat0)
        return super().forward(data, training=training, encoded_scene_feat0=encoded_scene_feat0)

    def loss_fn(self, outputs, data, training=True):
        arm = self.mixed_representation_arm
        if arm == 'FULL':
            return super().loss_fn(outputs, data, training=training)
        if arm == 'POINT':
            # score_se3_loss is an inherited point-error auxiliary; despite its
            # name it is not a rigid-pose objective and is kept for comparability.
            total, log = BaseModel.loss_fn(self, outputs, data, training=training)
            total, diagnostics = self.score_se3_loss(total, outputs['scene_flows'],
                data['gt_scene_flows'].to(outputs['scene_flows'].device), data,
                training=bool(training), calibration_parameters=())
            log.update({'score_se3/' + k: float(v.detach()) for k, v in diagnostics.items()})
            log['total_loss'] = float(total.detach())
            return total, log
        valid = outputs['object_valid'].bool()
        rigid = rigid_objects(data, valid, self.mixed_representation_regimes)
        if torch.equal(rigid, valid):
            # Bitwise original loss on all-rigid batches: no alternate reduction.
            total, log = super().loss_fn(outputs, data, training=training)
        else:
            # Mask coefficient shrinkage for soft objects only in the LOSS COPY;
            # forward geometry and genuine GT deformation remain untouched.
            local = dict(outputs)
            coeff = outputs['carrier_gauge_temporal_coefficients']
            mask = rigid.view(*rigid.shape, *([1] * (coeff.ndim - 2)))
            local['carrier_gauge_temporal_coefficients'] = torch.where(mask, coeff, 0.)
            with hybrid_regularization(rigid):
                total, log = super().loss_fn(local, data, training=training)
        log['representation/rigid_prior_objects'] = int(rigid.sum())
        log['representation/relaxed_objects'] = int((valid & ~rigid).sum())
        return total, log

    @torch.no_grad()
    def long_horizon_eval_sums(self, batch, outputs):
        if self.mixed_representation_arm == 'POINT':
            return {}  # Primary point metrics use the native shared evaluator.
        return super().long_horizon_eval_sums(batch, outputs)


def load_mixed_representation_model(path, *, device='cuda'):
    """Strict loading of the actual scientific arm, not its legacy dexplicit_3d alias."""
    import argparse
    import copy
    import tempfile
    saved = torch.load(Path(path).expanduser().resolve(strict=True), map_location='cpu', weights_only=False)
    args = copy.deepcopy(saved['args'])
    if isinstance(args, dict): args = argparse.Namespace(**args)
    if getattr(args, 'mixed_representation_version', None) != 1 or getattr(args, 'mixed_representation_arm', None) not in ARMS:
        raise ValueError('Not a version1 mixed-representation checkpoint')
    state = {k.removeprefix('module.'): v for k, v in saved['model'].items()}
    args.device = torch.device(device); args.distributed = False; args.disable_compile = True
    features = {'scene_features_dim': int(state['scene_feature_encoder.scene_raw_feat_proj.weight'].shape[1]),
                'robot_features_dim': int(state['robot_proj.fc1.weight'].shape[1])}
    with tempfile.TemporaryDirectory(prefix='representation-load-') as root:
        args.output_dir = root
        model = MixedRepresentationModel(args, features, rank=0)
        model.load_state_dict(state, strict=True)
    predictor = model.dynamics_predictor
    for flag in ('_long_horizon_zero_init_checked', '_deformation_graph_identity_checked', '_carrier_gauge_identity_checked'):
        if hasattr(predictor, flag): setattr(predictor, flag, True)
    return model.to(args.device).eval()
