"""Opt-in deterministic CLI seed for the long-horizon ablation processes.

Python imports ``usercustomize`` after ``sitecustomize``. The production stage
launcher already prepends this directory to ``PYTHONPATH``; therefore this hook
can pin the otherwise time-derived PointWorld seed without modifying the shared
trainer. It is inert for every unrelated process.
"""
from __future__ import annotations

import os
import sys


def _enabled() -> bool:
    value = os.environ.get("POINTWORLD_LONG_HORIZON_ABLATION", "0").strip().lower()
    if value in {"", "0", "false", "no", "off"}:
        return False
    return any("train_hot3d_dynamics.py" in argument for argument in sys.argv)


def _has_seed_argument() -> bool:
    for argument in sys.argv[1:]:
        if argument in {"--seed", "-s"}:
            return True
        if argument.startswith("--seed=") or argument.startswith("-s="):
            return True
    return False


if _enabled() and not _has_seed_argument():
    seed = int(os.environ.get("LONG_HORIZON_SEED", "2026"))
    sys.argv.extend(["--seed", str(seed)])
