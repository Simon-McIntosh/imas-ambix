"""Scoring the GS-readout topology against the firewalled EFIT referee (gate 2).

The gate asks: from raw magnetics, does the topology READ from the model's
solved ψ (:mod:`imas_ambix.latent.topology`) match the absolute-magnetics
oracle (~0.5–0.7 skill on axis / X-point / boundary)?  This module provides the
scoring core — per-quantity RMSE-skill vs a train-mean baseline (the oracle's
formula) and the permutation-invariant X-point-set match (the X-point is an
order-invariant null set, so its error must not depend on slot order).

EFIT is a **firewalled referee** here: its axis / X-point / LCFS reconstruction
only *scores* the model's readout, it is never a training label
(``firewall-definition = code-outputs-only``).  The scoring functions take the
referee target as data; the caller reads it inside the referee's
``evaluator_context()``.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# The order-invariant X-point slot permutations for a ≤2-null set.
_XPT_PERMS = (((0, 0), (1, 1)), ((0, 1), (1, 0)))


def matched_xpoint_error(pred_set: np.ndarray, ref_set: np.ndarray) -> float:
    """Permutation-invariant RMS distance between two ≤2 X-point null sets.

    ``pred_set`` / ``ref_set`` : ``(2, 2)`` (R, Z) with NaN for an absent slot.
    Matches the predicted slots to the reference slots by the permutation that
    minimises the RMS Euclidean distance over pairs where BOTH slots are
    present; returns that RMS distance (metres).  NaN if no present pair can be
    matched under any permutation.
    """
    pred = np.asarray(pred_set, dtype=np.float64).reshape(2, 2)
    ref = np.asarray(ref_set, dtype=np.float64).reshape(2, 2)
    best = np.inf
    for perm in _XPT_PERMS:
        errs = []
        for pi, ri in perm:
            p, r = pred[pi], ref[ri]
            if np.isfinite(p).all() and np.isfinite(r).all():
                errs.append(float(np.hypot(p[0] - r[0], p[1] - r[1])))
        if errs:
            cost = float(np.sqrt(np.mean(np.square(errs))))
            best = min(best, cost)
    return best if np.isfinite(best) else np.nan


def per_quantity_skill(
    model: np.ndarray,
    ref: np.ndarray,
    baseline: np.ndarray,
    names: list[str],
) -> dict[str, float]:
    """Per-component RMSE-skill ``1 − RMSE_model / RMSE_baseline`` vs the referee.

    ``model`` / ``ref`` / ``baseline`` : ``(N, D)`` arrays (metres); ``ref`` is
    the firewalled EFIT target (NaN where undefined).  Skill is computed only
    over samples where the reference is finite; a component with no finite
    reference (or a degenerate zero-RMSE baseline) yields NaN skill.  This is
    the same skill definition the absolute-magnetics oracle reports, so the two
    are directly comparable (the ~0.5–0.7 bar).
    """
    model = np.asarray(model, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    baseline = np.asarray(baseline, dtype=np.float64)
    out: dict[str, float] = {}
    for d, name in enumerate(names):
        m = (
            np.isfinite(ref[:, d])
            & np.isfinite(model[:, d])
            & np.isfinite(baseline[:, d])
        )
        if not m.any():
            out[name] = np.nan
            continue
        rmse_m = float(np.sqrt(np.mean((model[m, d] - ref[m, d]) ** 2)))
        rmse_b = float(np.sqrt(np.mean((baseline[m, d] - ref[m, d]) ** 2)))
        out[name] = np.nan if rmse_b == 0.0 else 1.0 - rmse_m / rmse_b
    return out


def headline_skill(skill: dict[str, float], components: list[str]) -> float:
    """Mean skill over the headline components (axis + X-point), ignoring NaN."""
    vals = [skill[c] for c in components if c in skill and np.isfinite(skill[c])]
    return float(np.mean(vals)) if vals else np.nan


__all__ = ["matched_xpoint_error", "per_quantity_skill", "headline_skill"]
