"""Opt-in bootstrap for the isolated object-state JEPA experiment.

The normal PointWorld entrypoint is untouched.  The experiment config prepends
this directory to PYTHONPATH and sets POINTWORLD_OBJECT_JEPA=1.  Python imports
sitecustomize before executing train_hot3d_dynamics.py, so the entrypoint sees
our strict subclass as its V2 model class.
"""
from __future__ import annotations

import os
import sys


def _enabled_for_this_process() -> bool:
    enabled = os.environ.get("POINTWORLD_OBJECT_JEPA", "0").strip().lower()
    if enabled in {"0", "false", "no", "off", ""}:
        return False
    return any("train_hot3d_dynamics.py" in argument for argument in sys.argv)


if _enabled_for_this_process():
    import training.dynamics_experiment_v2 as dynamics_v2
    from training.object_jepa_hidden_only import (
        HiddenOnlyObjectStateJEPAV2ExperimentalBaseModel,
    )
    from training.object_jepa_se3 import install_object_jepa_checkpoint_loader

    dynamics_v2.V2ExperimentalBaseModel = (
        HiddenOnlyObjectStateJEPAV2ExperimentalBaseModel
    )
    install_object_jepa_checkpoint_loader()
