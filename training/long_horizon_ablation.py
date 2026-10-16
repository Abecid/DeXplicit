"""Public API for the controlled 35-frame long-horizon ablation ladder.

Before modifying or interpreting a stage, read
``training.long_horizon_provenance``.  It records whether a mechanism is an
existing PointWorld baseline, an adaptation informed by released official code,
a paper-spec implementation without released code, or a project-specific
synthesis.  No stage should be described generically as a faithful paper
reimplementation unless that registry explicitly says so.
"""

from training.long_horizon_common import (
    CUMULATIVE_FEATURES_BY_STAGE,
    FEATURES_BY_STAGE,
    ISOLATED_FEATURES_BY_STAGE,
    STAGE_NAMES,
    batched_min_hand_distance,
    contact_strength,
    hand_side_trajectory_features,
    normalize_track,
    stage_features,
    stage_name,
)
from training.long_horizon_integration import (
    install_long_horizon_checkpoint_loader,
    install_long_horizon_eval_metrics,
)
from training.long_horizon_model import LongHorizonAblationModel
from training.long_horizon_predictor import LongHorizonStructuredPredictor
from training.long_horizon_provenance import (
    METHOD_PROVENANCE,
    MethodProvenance,
    method_provenance,
    provenance_contract,
)

__all__ = [
    "CUMULATIVE_FEATURES_BY_STAGE",
    "FEATURES_BY_STAGE",
    "ISOLATED_FEATURES_BY_STAGE",
    "METHOD_PROVENANCE",
    "STAGE_NAMES",
    "normalize_track",
    "LongHorizonAblationModel",
    "LongHorizonStructuredPredictor",
    "MethodProvenance",
    "batched_min_hand_distance",
    "contact_strength",
    "hand_side_trajectory_features",
    "install_long_horizon_checkpoint_loader",
    "install_long_horizon_eval_metrics",
    "method_provenance",
    "provenance_contract",
    "stage_features",
    "stage_name",
]
