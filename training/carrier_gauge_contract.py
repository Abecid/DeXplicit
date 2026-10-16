"""Shared immutable argument names for carrier-gauge checkpoint contracts."""
from __future__ import annotations

CARRIER_GAUGE_CAUSAL_ARGS = (
    "carrier_gauge_candidate",
    "carrier_gauge_mode_rank",
    "carrier_gauge_temporal_controls",
    "carrier_gauge_relation_temperature_m",
    "carrier_gauge_carrier_translation_bound_m",
    "carrier_gauge_carrier_rotation_bound_deg",
    "carrier_gauge_edge_strain_weight",
    "carrier_gauge_coefficient_weight",
    "carrier_gauge_carrier_tv_weight",
    "carrier_gauge_carrier_strength_weight",
    "carrier_gauge_deformable_threshold_m",
)

__all__ = ["CARRIER_GAUGE_CAUSAL_ARGS"]
