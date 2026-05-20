"""Evaluation metrics and rollout API for the Fusion World Model.

Provides quantitative evaluation of predicted vs ground-truth frame
sequences (reconstruction quality + physics-derived metrics) and the
forward-rollout stub that drives the demo pipeline once the WHAM model
is in place.

Related plans:
- ``plans/demo.md`` §4 (metric definitions and acceptance thresholds)
- ``plans/world-model-v0.md`` §5 (training-time eval hooks)
- ``plans/world-model-v0.md`` §7 (rollout algorithm)
"""

from __future__ import annotations

from imas_ambix.eval.metrics import (
    centroid_mse,
    chord_integrated_emission,
    chord_nrmse,
    compute_all_metrics,
    edge_displacement,
    frame_centroid,
    lpips,
    psnr,
    rfid,
)
from imas_ambix.eval.rollout import RolloutConfig, rollout

__all__ = [
    "centroid_mse",
    "chord_integrated_emission",
    "chord_nrmse",
    "compute_all_metrics",
    "edge_displacement",
    "frame_centroid",
    "lpips",
    "psnr",
    "rfid",
    "RolloutConfig",
    "rollout",
]
