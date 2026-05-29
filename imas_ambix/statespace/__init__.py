"""State-space modelling package for IMAS Ambix.

Provides the family inventory, dataset, splits, alignment helpers, and
calibration harness for the plasma-state-space-v0 stage-1 work.

Public sub-modules
------------------
inventory     -- per-shot family group inventory across all level-1 shots
families      -- diagnostic-family classification + leakage audit
align         -- time-alignment helpers (wraps tokenizer.alignment)
splits        -- train / held-out-shots / held-out-regime split generation
calibration   -- probabilistic calibration harness (coverage, CRPS, ECE, …)
dataset       -- multi-family torch-compatible dataset wrapper
camera_boundary -- camera-derived visible-emission-edge prototype (rbb)

Reserved (do NOT create here — later stages):
  baseline.py   -- S7.2
  engine.py     -- S7.3
  filter.py     -- S7.3
"""

from __future__ import annotations
