"""Reuse mixed50's data, eval and checkpoint machinery; select only the method."""
from __future__ import annotations
import json
from pathlib import Path
from training.mixed_representation_losses import install_loss_scopes

FIELDS = ('mixed_representation_arm', 'mixed_representation_version',
          'mixed_representation_regimes', 'mixed_representation_plan_sha256')


def point_parameter_plan(model, candidate):
    from training.score_se3_finetune import (ScoreSE3ParameterPlan, ScoreSE3ParameterGroupPlan,
                                            ADAPTER_GROUP, BASE_POSE_GROUP)
    names = dict(model.named_parameters())
    direct = sorted(n for n in names if n.startswith('dynamics_predictor.direct.'))
    action = sorted(n for n in names if n.startswith('dynamics_predictor.backbone.')
                    and not n.startswith(('dynamics_predictor.backbone.ptv3.',
                                          'dynamics_predictor.backbone.delta_embed.')))
    if not direct or not action:
        raise ValueError('Point-output trainable parameter names changed')
    active = set(direct + action)
    for name, parameter in names.items():
        parameter.requires_grad_(name in active)
    return ScoreSE3ParameterPlan(candidate=candidate, groups=(
        ScoreSE3ParameterGroupPlan(ADAPTER_GROUP, tuple(direct)),
        ScoreSE3ParameterGroupPlan(BASE_POSE_GROUP, tuple(action))),
        frozen_parameter_names=tuple(sorted(set(names) - active)))


def install(plan):
    from training import dexplicit_3d_mixed_runtime as shared
    from training.dexplicit_3d_mixed_manifest import digest
    shared.install(plan)
    install_loss_scopes()
    from training.mixed_representation_model import MixedRepresentationModel, ARMS
    import training.deformation_graph_model as factory
    import training.score_se3_finetune as fine
    import training.score_se3_integration as loading
    import training.score_se3_contract as contracts
    config = plan['representation']
    if config['arm'] not in ARMS or config['version'] != 1:
        raise ValueError('Unknown representation protocol')
    sha = digest(Path(plan['root']) / 'plan.json')
    fields = dict(mixed_representation_arm=config['arm'], mixed_representation_version=1,
        mixed_representation_regimes=json.dumps(config['regimes'], sort_keys=True),
        mixed_representation_plan_sha256=sha)

    class ConfiguredRepresentation(MixedRepresentationModel):
        def __init__(self, args, *a, **kw):
            # Preserve both the shared mixed50 and the new scientific identity.
            for key, value in {**fields, 'dexplicit_3d_mixed50_plan_sha256': sha}.items():
                setattr(args, key, value)
                if getattr(args, 'og_args', None) is not None:
                    setattr(args.og_args, key, value)
            super().__init__(args, *a, **kw)

    factory.DeformationGraphExperimentalModel = ConfiguredRepresentation
    for name in ('SCORE_SE3_RESUME_CAUSAL_ARGS', 'SCORE_SE3_INFERENCE_CAUSAL_ARGS'):
        setattr(loading, name, tuple(dict.fromkeys((*getattr(loading, name), *FIELDS))))
    native_parameters = fine.build_score_se3_parameter_plan

    def parameters(model, candidate):
        if getattr(model, 'mixed_representation_arm', '') == 'POINT':
            return point_parameter_plan(model, candidate)
        return native_parameters(model, candidate)

    fine.build_score_se3_parameter_plan = parameters
    native_contract = contracts.build_score_se3_contract

    def build(args, **kwargs):
        result = native_contract(args, **kwargs)
        if any(getattr(args, key, None) != value for key, value in fields.items()):
            raise ValueError('Runtime representation identity differs from the frozen plan')
        result['mixed_representation'] = {**config, 'plan_sha256': sha}
        result['dexplicit_3d_mixed50']['model'] = config['forward']
        return result

    contracts.build_score_se3_contract = build
