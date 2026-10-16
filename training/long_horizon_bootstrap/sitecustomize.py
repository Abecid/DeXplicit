"""Opt-in bootstrap for the controlled 35-frame long-horizon ladder."""
from __future__ import annotations

import os
import sys


def _enabled() -> bool:
    value = os.environ.get("POINTWORLD_LONG_HORIZON_ABLATION", "0").strip().lower()
    if value in {"0", "false", "no", "off", ""}:
        return False
    return any("train_hot3d_dynamics.py" in argument for argument in sys.argv)


if _enabled():
    import training.dynamics_experiment_v2 as dynamics_v2
    from training.long_horizon_ablation import (
        LongHorizonAblationModel,
        install_long_horizon_checkpoint_loader,
        install_long_horizon_eval_metrics,
    )

    dynamics_v2.V2ExperimentalBaseModel = LongHorizonAblationModel
    install_long_horizon_checkpoint_loader()
    install_long_horizon_eval_metrics()
