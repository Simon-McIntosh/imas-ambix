#!/usr/bin/env python
"""Gate-2 evaluation with the variational patch-current inverse (no Picard).

The powered rematch of the training-free gate-2 protocol: per held-out slice,
invert the RAW measured magnetics for the patch-current vector (whitened sensor
misfit + Rogowski Ip anchor + λ·structure-residual, per-slice Adam under one of
the weight-policy arms), read axis / X-point set / LCFS radii from the
assembled ψ at evaluation time, and score against the firewalled EFIT referee
with the same skill formulas, train-mean baseline, shot list, and slice
selection as ``gs_solve_gate_eval.py`` — the numbers are directly comparable.

There is no inner solve and hence no convergence masking: every candidate
slice is scored (the corrected Picard chain scores 12/160).

Additionally recovers the closures the fit implies: per-slice p′(ψ) and
FF′(ψ)/μ0 from the per-bin regression coefficients with their uncertainties,
and the integrated p(ψ), F²(ψ) with the F² ≥ 0 integrability check
(F_vac from the measured TF current where available).

Artifacts:  imas_ambix/latent/artifacts/patch_gate/
Figures:    docs/figures/patch-current-force-balance/
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch

from imas_ambix.gs.geometry import GEOMETRY_TABLE_VERSION, build_table_for_shot
from imas_ambix.gs.operator import COIL_MODEL_VERSION, build_operator
from imas_ambix.latent.data import (
    CHANNEL_SCALE_KIND_FLOOR_REL,
    feature_schema,
    load_shot_windows,
    read_split_shot_lists,
    robust_channel_scale,
)
from imas_ambix.latent.evaluate import (
    headline_skill,
    matched_xpoint_error,
    per_quantity_skill,
)
from imas_ambix.latent.gs_solve import (
    EquilibriumGrid,
    _read_axis,
    _read_boundary_psi_robust,
)
from imas_ambix.latent.patch_basis import PatchBasis
from imas_ambix.latent.patch_inverse import (
    InverseConfig,
    SliceInversion,
    SlicePayload,
    _lambda_schedule,
    invert_slices,
)
from imas_ambix.latent.structure_residual import (
    fit_flux_functions,
    integrate_closures,
    structure_residual,
)
from imas_ambix.latent.topology import (
    _inside_polygon,
    find_critical_points,
    lcfs_radii,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("patch_gate_eval")

ARTIFACTS = Path("imas_ambix/latent/artifacts/patch_gate")
FIGURES = Path("docs/figures/patch-current-force-balance")

TARGET_NAMES = [
    "axis_R",
    "axis_Z",
    "xpt0_R",
    "xpt0_Z",
    "xpt1_R",
    "xpt1_Z",
    *[f"lcfs_r_{k}" for k in range(8)],
]

POLICY_COLOR = {
    "fixed": "#2166ac",
    "warm-start": "#1b7837",
    "discrepancy": "#d95f02",
}


def geometry_target(
    psi2d: np.ndarray,
    grid: EquilibriumGrid,
    *,
    smooth_sigma: float = 0.0,
    min_axis_dist: float = 0.0,
) -> tuple[np.ndarray, float, float]:
    """Oracle-shaped 14-D geometry read of an assembled ψ field.

    Mirrors ``gs_solve_gate_eval.equilibrium_target`` but reads everything from
    the ψ field alone (no EquilibriumResult): sign-aware conductor-clear axis,
    innermost in-polygon X-point / limiter-contact boundary flux, LCFS radii.
    ``smooth_sigma`` / ``min_axis_dist`` are the opt-in LCFS boundary-read
    robustifications (:func:`imas_ambix.latent.gs_solve._read_boundary_psi_robust`
    — measured lever A4); both default to 0.0, reproducing the original
    innermost-ψ / limiter-contact read exactly.  Returns
    (target, psi_axis, psi_boundary).
    """
    target = np.full(14, np.nan)
    # plasma current here is positive-Ip MAST convention: axis = max of ψ; the
    # sign-aware read picks the sign from the field itself via both attempts
    ax_pos, psi_pos = _read_axis(psi2d, grid, +1.0)
    ax_neg, psi_neg = _read_axis(psi2d, grid, -1.0)
    # choose the sign whose extremum deviates more from the field median —
    # the plasma well dominates the interior either way
    med = float(np.median(psi2d))
    if abs(psi_pos - med) >= abs(psi_neg - med):
        axis, axis_psi = ax_pos, psi_pos
    else:
        axis, axis_psi = ax_neg, psi_neg
    target[0], target[1] = axis
    boundary_psi = _read_boundary_psi_robust(
        psi2d,
        grid,
        tuple(axis),
        axis_psi,
        smooth_sigma=smooth_sigma,
        min_axis_dist=min_axis_dist,
    )
    cp = find_critical_points(psi2d, grid.rg, grid.zg)
    if cp.x_points.shape[0]:
        ins = _inside_polygon(
            cp.x_points[:, 0], cp.x_points[:, 1], grid.limiter_r, grid.limiter_z
        ) & grid.clear_of_conductors(cp.x_points[:, 0], cp.x_points[:, 1])
        pts = cp.x_points[ins]
        xpsi = cp.x_psi[ins]
        if pts.shape[0]:
            order = np.argsort(np.abs(xpsi - boundary_psi))
            for slot in range(min(2, pts.shape[0])):
                target[2 + 2 * slot] = pts[order[slot], 0]
                target[3 + 2 * slot] = pts[order[slot], 1]
    target[6:] = lcfs_radii(psi2d, grid.rg, grid.zg, tuple(axis), boundary_psi)
    return target, float(axis_psi), float(boundary_psi)


def shot_payloads(
    shot: int,
    *,
    nr,
    nz,
    max_slices,
    min_ip_ka,
    split="eval",
    scale_floor_rel: float = CHANNEL_SCALE_KIND_FLOOR_REL,
):
    """Per-shot geometry + slice payloads, identical selection to the Picard gate.

    ``scale_floor_rel`` is the ``rel_floor`` passed straight through to
    :func:`robust_channel_scale` — the default reproduces the training
    convention exactly (0.05); the F floor-sensitivity sweep is the only
    caller that varies it (see ``run_floor_sensitivity``).
    """
    table = build_table_for_shot(int(shot))
    fwd = build_operator(table)
    grid = EquilibriumGrid.from_table(table, nr=nr, nz=nz)
    basis = PatchBasis.from_table(table, nr=nr, nz=nz)
    g_sens, channels = grid.sensor_greens(table)

    w = load_shot_windows(int(shot), fwd, split, feature_schema(), with_referee=True)
    if w is None or w.ref_target is None:
        return None
    row_of = {ch: i for i, ch in enumerate(fwd.sensor_channels)}
    ch_rows = np.array([row_of.get(ch, -1) for ch in channels])
    present = ch_rows >= 0
    valid = [
        t
        for t in range(w.times.size)
        if np.isfinite(w.ref_target[t, :2]).all() and abs(w.anchored[t, 0]) > min_ip_ka
    ]
    if len(valid) > max_slices:
        valid = valid[:: max(1, len(valid) // max_slices)][:max_slices]
    if not valid:
        return None
    scale = robust_channel_scale(
        np.nanstd(w.raw_mag, axis=0), fwd.sensor_channels, rel_floor=scale_floor_rel
    )
    scale_ch = np.where(present, scale[np.clip(ch_rows, 0, None)], 1.0)

    payloads, refs = [], []
    for t in valid:
        vac = fwd.vacuum_prediction(w.i_pf[t])
        payloads.append(
            SlicePayload(
                measured=np.where(
                    present, w.raw_mag[t][np.clip(ch_rows, 0, None)], np.nan
                ),
                vacuum=np.where(present, vac[np.clip(ch_rows, 0, None)], 0.0),
                mask=present & w.mag_mask[t][np.clip(ch_rows, 0, None)],
                scale=scale_ch,
                i_pf=w.i_pf[t],
                ip_amperes=float(abs(w.anchored[t, 0])) * 1e3,
                shot=int(shot),
                t_index=int(t),
                time_s=float(w.times[t]),
            )
        )
        refs.append(w.ref_target[t])
    return {
        "table": table,
        "grid": grid,
        "basis": basis,
        "payloads": payloads,
        "refs": np.array(refs),
    }


def train_mean_baseline(n_train, n_baseline_shots, min_ip_ka):
    schema = feature_schema()
    train_shots, _ = read_split_shot_lists(n_train, 8)
    rows = []
    for s in train_shots[:n_baseline_shots]:
        try:
            fwd = build_operator(build_table_for_shot(int(s)))
        except Exception:  # noqa: BLE001
            continue
        wtr = load_shot_windows(int(s), fwd, "train", schema, with_referee=True)
        if wtr is not None and wtr.ref_target is not None:
            on = np.abs(wtr.anchored[:, 0]) > min_ip_ka
            rows.append(wtr.ref_target[on])
    return (
        np.nanmean(np.concatenate(rows, axis=0), axis=0)
        if rows
        else np.full(14, np.nan)
    )


def _xpoint_set_skill(
    model: np.ndarray, ref: np.ndarray, baseline: np.ndarray
) -> float:
    """The permutation-invariant X-point-set RMSE skill (shared by the point
    estimate and every bootstrap resample of it)."""
    xm = np.array(
        [
            matched_xpoint_error(model[i, 2:6].reshape(2, 2), ref[i, 2:6].reshape(2, 2))
            for i in range(len(model))
        ]
    )
    xb = np.array(
        [
            matched_xpoint_error(
                baseline[i, 2:6].reshape(2, 2), ref[i, 2:6].reshape(2, 2)
            )
            for i in range(len(model))
        ]
    )
    finite = np.isfinite(xm) & np.isfinite(xb)
    if not finite.any():
        return np.nan
    return float(
        1.0
        - np.sqrt(np.nanmean(xm[finite] ** 2)) / np.sqrt(np.nanmean(xb[finite] ** 2))
    )


def _bootstrap_skill_draws(
    model: np.ndarray,
    ref: np.ndarray,
    baseline: np.ndarray,
    shot_ids: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Paired-bootstrap resamples of every headline + per-quantity skill,
    resampling SHOTS (not slices) with replacement so a shot's slices always
    move together — the correct resampling unit given within-shot slice
    correlation.  Each of ``n_boot`` draws recomputes model-vs-baseline
    skill on the pooled slices of the drawn shots, paired against the SAME
    referee target ``ref`` used at the point estimate.

    Returns ``(axis_draws, lcfs_draws, xpt_draws, per_quantity_draws)`` with
    shapes ``(n_boot,)`` ×3 and ``(n_boot, len(TARGET_NAMES))``.
    NaN-filled if fewer than 2 unique shots (no meaningful resampling unit).
    """
    n_names = len(TARGET_NAMES)
    axis_draws = np.full(n_boot, np.nan)
    lcfs_draws = np.full(n_boot, np.nan)
    xpt_draws = np.full(n_boot, np.nan)
    per_q_draws = np.full((n_boot, n_names), np.nan)

    shot_ids = np.asarray(shot_ids)
    unique_shots = np.unique(shot_ids)
    if unique_shots.size < 2:
        return axis_draws, lcfs_draws, xpt_draws, per_q_draws

    rng = np.random.default_rng(seed)
    by_shot = {s: np.flatnonzero(shot_ids == s) for s in unique_shots}
    for b in range(n_boot):
        draw = rng.choice(unique_shots, size=unique_shots.size, replace=True)
        idx = np.concatenate([by_shot[s] for s in draw])
        m, r, base = model[idx], ref[idx], baseline[idx]
        sk = per_quantity_skill(m, r, base, TARGET_NAMES)
        per_q_draws[b] = [sk[name] for name in TARGET_NAMES]
        axis_draws[b] = headline_skill(sk, ["axis_R", "axis_Z"])
        lcfs_draws[b] = headline_skill(sk, [f"lcfs_r_{k}" for k in range(8)])
        xpt_draws[b] = _xpoint_set_skill(m, r, base)
    return axis_draws, lcfs_draws, xpt_draws, per_q_draws


def _percentile_ci(draws: np.ndarray) -> list[float | None]:
    if not np.isfinite(draws).any():
        return [None, None]
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return [float(lo), float(hi)]


def score(
    model, ref, baseline_vec, *, shot_ids=None, n_boot: int = 2000, ci_seed: int = 0
):
    """Same skill computation as the Picard gate (per-quantity RMSE skill).

    When ``shot_ids`` (one shot number per row of ``model``/``ref``) is
    supplied, every skill also carries a 95% paired-bootstrap CI over shots
    (``<metric>_ci`` = ``[lo, hi]``, ``ci_n_boot``, ``ci_seed`` — see
    :func:`_bootstrap_skill_draws`).  Without ``shot_ids`` the CI fields are
    omitted (unchanged behaviour for callers that have not yet threaded shot
    identity through, e.g. quick smoke arms).
    """
    baseline = np.tile(baseline_vec, (len(model), 1))
    skill = per_quantity_skill(model, ref, baseline, TARGET_NAMES)
    xpt_skill = _xpoint_set_skill(model, ref, baseline)
    axis_err = np.hypot(model[:, 0] - ref[:, 0], model[:, 1] - ref[:, 1])
    out = {
        "per_quantity_skill": {
            k: (None if not np.isfinite(v) else float(v)) for k, v in skill.items()
        },
        "axis_skill": headline_skill(skill, ["axis_R", "axis_Z"]),
        "xpoint_set_skill": None if not np.isfinite(xpt_skill) else float(xpt_skill),
        "lcfs_skill": headline_skill(skill, [f"lcfs_r_{k}" for k in range(8)]),
        "axis_error_mean_m": float(np.nanmean(axis_err)),
        "axis_error_median_m": float(np.nanmedian(axis_err)),
        "axis_errors": axis_err,
    }
    if shot_ids is not None:
        axis_draws, lcfs_draws, xpt_draws, per_q_draws = _bootstrap_skill_draws(
            model, ref, baseline, shot_ids, n_boot=n_boot, seed=ci_seed
        )
        out["axis_skill_ci"] = _percentile_ci(axis_draws)
        out["lcfs_skill_ci"] = _percentile_ci(lcfs_draws)
        out["xpoint_set_skill_ci"] = _percentile_ci(xpt_draws)
        out["per_quantity_skill_ci"] = {
            name: _percentile_ci(per_q_draws[:, i])
            for i, name in enumerate(TARGET_NAMES)
        }
        out["ci_n_boot"] = n_boot
        out["ci_seed"] = ci_seed
    return out


def count_saddles(psi2d: np.ndarray, grid: EquilibriumGrid) -> int:
    """In-limiter saddle count of ``psi2d`` — the raw count a saddle-excess
    metric subtracts the referee's own X-point count from.  Mirrors
    ``scripts.boundary_moment_gate_eval._count_saddles`` exactly (same
    inside-limiter-only test, no conductor-clearance filter) so the
    free-current and current-moment arms report a directly comparable
    saddle definition."""
    cp = find_critical_points(psi2d, grid.rg, grid.zg)
    if not cp.x_points.shape[0]:
        return 0
    ins = _inside_polygon(
        cp.x_points[:, 0], cp.x_points[:, 1], grid.limiter_r, grid.limiter_z
    )
    return int(np.count_nonzero(ins))


def saddle_excess_stats(saddle_counts, ref: np.ndarray) -> dict:
    """Saddle count IN EXCESS of the referee's own X-point count.

    MAST is routinely double-null: a naive ``saddles <= 1`` "saddle-free"
    label mislabels two genuine X-points as a spurious read.  The referee's
    per-slice X-point count (how many of ``xpt0``/``xpt1`` are finite in
    ``ref``) is eval-side only — it is read here purely to score, never fed
    into any fit path (firewall: code-outputs-only).  ``excess = saddles -
    n_referee_xpoints`` isolates genuine over-counting; ``saddle_clean_fraction``
    is the double-null-correct replacement for the old ``saddle_free_fraction``.
    """
    ref = np.asarray(ref, dtype=np.float64)
    saddle_counts = np.asarray(saddle_counts, dtype=np.float64)
    if saddle_counts.size == 0:
        return {
            "saddle_excess_mean": None,
            "saddle_excess_median": None,
            "saddle_clean_fraction": None,
            "referee_xpoint_count_mean": None,
        }
    ref_xpt_present = np.isfinite(ref[:, 2:6].reshape(len(ref), 2, 2)).all(axis=2)
    ref_xpt_count = ref_xpt_present.sum(axis=1).astype(np.float64)
    excess = saddle_counts - ref_xpt_count
    return {
        "saddle_excess_mean": float(np.mean(excess)),
        "saddle_excess_median": float(np.median(excess)),
        "saddle_clean_fraction": float(np.mean(excess <= 0)),
        "referee_xpoint_count_mean": float(np.mean(ref_xpt_count)),
    }


# ---------------------------------------------------------------------------
# Lever A4 — LCFS boundary-read robustification (measured; kept only if
# load-bearing).  ARM 2 / ARM 3 (saddle-distance guard / ψ-smoothing) are
# wired through geometry_target's opt-in kwargs above.  ARM 1
# (current-smoothness soft rung) is prototyped HERE, not in
# imas_ambix/latent/patch_inverse.py, until measured: it duplicates
# invert_slices with one added term so the shared inverse module is only
# touched if this arm wins (patch-equilibrium-wm-integration §3, A4).
# ---------------------------------------------------------------------------


def _grid_neighbor_pairs(basis: PatchBasis) -> torch.Tensor:
    """(P, 2) int64 tensor of 4-connected adjacent candidate-cell index pairs
    on the (R, Z) lattice — the adjacency the current-smoothness penalty
    (ARM 1) differences over.  Built once per campaign from the fixed
    ``r_cells`` / ``z_cells`` / ``grid_r`` / ``grid_z`` geometry (all cells,
    not just conductor-clear ones, share the same regular raster spacing).
    """
    r_c = basis.r_cells.detach().cpu().numpy()
    z_c = basis.z_cells.detach().cpu().numpy()
    grid_r = basis.grid_r.detach().cpu().numpy()
    grid_z = basis.grid_z.detach().cpu().numpy()
    dr = float(grid_r[1] - grid_r[0])
    dz = float(grid_z[1] - grid_z[0])
    j_idx = np.rint((r_c - grid_r[0]) / dr).astype(int)
    i_idx = np.rint((z_c - grid_z[0]) / dz).astype(int)
    pos = {
        (int(i), int(j)): k for k, (i, j) in enumerate(zip(i_idx, j_idx, strict=True))
    }
    pairs = [
        (k, pos[(i + di, j + dj)])
        for (i, j), k in pos.items()
        for di, dj in ((1, 0), (0, 1))
        if (i + di, j + dj) in pos
    ]
    return (
        torch.tensor(pairs, dtype=torch.long)
        if pairs
        else torch.zeros((0, 2), dtype=torch.long)
    )


def invert_slices_smooth(
    basis: PatchBasis,
    payloads: list[SlicePayload],
    cfg: InverseConfig,
    pairs: torch.Tensor,
    smooth_lambda: float,
    *,
    device: str | torch.device = "cpu",
) -> list[SliceInversion]:
    """ARM 1: :func:`invert_slices` plus a spatial-Laplacian current-smoothness
    penalty ``smooth_lambda * mean((x_i - x_j)^2)`` over 4-connected candidate
    grid-cell pairs on the dimensionless current shape ``x`` — the classic
    tomography smoothness regulariser, measured as a lever independent of the
    force-balance structure residual.  ``smooth_lambda=0`` reproduces
    ``invert_slices`` (same seed, same optimiser steps, zero extra loss term).
    """
    dev = torch.device(device)
    dt = cfg.dtype
    n = int(basis.r_cells.shape[0])
    b = len(payloads)

    m_sens = basis.m_sens.to(device=dev, dtype=dt)
    g_cc = basis.g_cc.to(device=dev, dtype=dt)
    r_c = basis.r_cells.to(device=dev, dtype=dt)
    z_c = basis.z_cells.to(device=dev, dtype=dt)
    candidate = basis.candidate_mask.to(device=dev, dtype=dt)
    cell_area = float(basis.cell_area)
    pairs = pairs.to(dev)
    has_pairs = pairs.numel() > 0

    meas = torch.stack(
        [torch.as_tensor(np.nan_to_num(p.measured), dtype=dt) for p in payloads]
    ).to(dev)
    vac = torch.stack([torch.as_tensor(p.vacuum, dtype=dt) for p in payloads]).to(dev)
    mask = torch.stack(
        [torch.as_tensor(p.mask.astype(np.float64), dtype=dt) for p in payloads]
    ).to(dev)
    scale = torch.stack([torch.as_tensor(p.scale, dtype=dt) for p in payloads]).to(dev)
    ip = torch.tensor([p.ip_amperes for p in payloads], dtype=dt, device=dev)
    psi_coil = torch.stack(
        [
            basis.psi_coil_cells_for(np.asarray(p.i_pf, dtype=np.float64))
            for p in payloads
        ]
    ).to(device=dev, dtype=dt)

    seed = torch.exp(
        -(((r_c - basis.r0) / cfg.seed_width_r) ** 2 + (z_c / cfg.seed_width_z) ** 2)
    )
    seed = seed / seed.sum() * n
    x = seed.expand(b, n).clone().requires_grad_(True)
    opt = torch.optim.Adam([x], lr=cfg.lr)

    lam = torch.zeros(b, dtype=dt, device=dev)
    target = torch.full((b,), float("inf"), dtype=dt, device=dev)
    warmup_end = int(cfg.warmup_fraction * cfg.iters)

    misfit = torch.zeros(b, dtype=dt, device=dev)
    fb = torch.zeros(b, dtype=dt, device=dev)
    for step in range(cfg.iters):
        with torch.no_grad():
            lam = _lambda_schedule(cfg, step, lam, misfit.detach(), target)
        opt.zero_grad()
        i_eff = x * candidate * (ip[:, None] / n)
        pred = vac + i_eff @ m_sens.T
        misfit = (mask * ((pred - meas) / scale) ** 2).sum(-1) / mask.sum(-1).clamp_min(
            1.0
        )
        ip_pen = ((i_eff.sum(-1) - ip) / ip) ** 2
        psi_c = i_eff @ g_cc.T + psi_coil
        fb_rows = [
            structure_residual(
                psi_c[k],
                r_c,
                i_eff[k] / cell_area,
                n_bins=cfg.n_bins,
                form=cfg.form,
                z_c=z_c,
                connectivity=cfg.connectivity,
                locality_scale=cfg.locality_scale,
            )
            for k in range(b)
        ]
        fb = torch.stack(fb_rows)
        if has_pairs and smooth_lambda > 0.0:
            diff = x[:, pairs[:, 0]] - x[:, pairs[:, 1]]
            smooth_pen = (diff * diff).mean(-1)
        else:
            smooth_pen = torch.zeros(b, dtype=dt, device=dev)
        loss = (
            misfit + cfg.ip_weight * ip_pen + lam * fb + smooth_lambda * smooth_pen
        ).sum()
        loss.backward()
        opt.step()
        if cfg.policy == "discrepancy" and step == max(warmup_end - 1, 0):
            target = cfg.misfit_ratio * misfit.detach().clone()

    out: list[SliceInversion] = []
    with torch.no_grad():
        i_fin = (x * candidate * (ip[:, None] / n)).cpu().numpy()
        for k, p in enumerate(payloads):
            out.append(
                SliceInversion(
                    i_cell=i_fin[k],
                    misfit=float(misfit[k]),
                    structure=float(fb[k]),
                    lambda_final=float(lam[k]),
                    ip_rel_err=float(abs(i_fin[k].sum() - p.ip_amperes) / p.ip_amperes),
                    shot=p.shot,
                    t_index=p.t_index,
                    time_s=p.time_s,
                )
            )
    return out


def lcfs_offset_cm_stats(
    model: np.ndarray, ref: np.ndarray, flattop_mask: np.ndarray
) -> dict:
    """Median LCFS radial offset in cm, overall and flat-top-only.

    NOTE units caveat: this is the median-of-per-slice-medians over the SAME
    8 fixed poloidal angles the gate's own oracle target/skill use
    (``target[6:14]``, matching :data:`TARGET_NAMES`'s ``lcfs_r_0..7``), not
    the 240-angle continuous contour comparison
    ``scripts/patch_flux_map_report.py`` used for the reported 31.3 cm
    flat-top baseline — the two are the same QUANTITY (median radial LCFS
    offset in cm) at different angular sampling density, comparable in scale
    but not bit-identical.  ``flattop_mask`` selects, per shot, the single
    scored slice with the largest |Ip| (mirrors
    ``patch_flux_map_report.select_slices``'s flat-top pick).
    """
    offset_cm = np.abs(model[:, 6:14] - ref[:, 6:14]) * 100.0  # (N, 8) [cm]
    per_slice_median = np.nanmedian(offset_cm, axis=1)  # (N,)
    flattop_vals = per_slice_median[flattop_mask]
    return {
        "lcfs_offset_median_cm_all": float(np.nanmedian(per_slice_median))
        if per_slice_median.size
        else None,
        "lcfs_offset_median_cm_flattop": float(np.nanmedian(flattop_vals))
        if flattop_vals.size
        else None,
        "n_flattop_slices": int(flattop_mask.sum()),
    }


#: The frozen P3-winner inverse config (patch-current-force-balance gate-2):
#: discrepancy policy, λ0=3, misfit_ratio=1.5, λmax=100 — axis skill +0.019,
#: 2.8 cm median, 160/160 scored.  A4 measures ONLY the readout (or, for
#: 'current-smooth', the inverse loss) against this frozen base.
P3_WINNER_KW = {
    "policy": "discrepancy",
    "lambda_fb": 3.0,
    "misfit_ratio": 1.5,
    "lambda_max": 100.0,
}


def run_boundary_arm(args) -> int:
    """Measure ONE A4 candidate arm against the frozen P3-winner inverse.

    Loads the tuning cohort (``--split train``) or the 160-slice held-out gate
    (``--split eval``, matching the P3 protocol exactly), inverts every slice
    once with the frozen P3-winner config, then reads geometry either with the
    baseline readout, the ``smooth_sigma`` / ``min_axis_dist`` robustification
    (ARM 2 / ARM 3, composable), or — for ``--boundary-arm current-smooth`` —
    re-inverts with the current-smoothness penalty (ARM 1, via
    :func:`invert_slices_smooth`) at ``--current-smooth-lambda`` before reading
    geometry with the same (optionally also-robustified) boundary read.
    Writes ``imas_ambix/latent/artifacts/patch_gate/boundary_read_<tag>.json``.
    """
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    device = (
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else args.device
    )
    connectivity = None if args.connectivity in ("", "none") else args.connectivity
    logger.info(
        "boundary-arm=%s split=%s smooth_sigma=%s min_axis_dist=%s "
        "current_smooth_lambda=%s device=%s",
        args.boundary_arm,
        args.split,
        args.smooth_sigma,
        args.min_axis_dist,
        args.current_smooth_lambda,
        device,
    )

    train_shots, held_shots = read_split_shot_lists(args.n_train, args.n_heldout)
    baseline_vec = train_mean_baseline(
        args.n_train, args.n_baseline_shots, args.min_ip_ka
    )
    eval_shots = (
        train_shots[args.n_baseline_shots : args.n_baseline_shots + args.n_tune_shots]
        if args.split == "train"
        else held_shots
    )

    shots = []
    for s in eval_shots:
        try:
            payload = shot_payloads(
                s,
                nr=args.nr,
                nz=args.nz,
                max_slices=args.max_slices_per_shot,
                min_ip_ka=args.min_ip_ka,
                split=args.split,
                scale_floor_rel=args.scale_floor_rel,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("shot %s failed to load: %s", s, exc)
            continue
        if payload is not None:
            shots.append(payload)

    prior_kw = {}
    if args.sign_prior != "none":
        prior_kw.update(sign_prior=args.sign_prior, sign_weight=args.sign_weight)
    if args.support_prior:
        prior_kw.update(
            support_prior=True,
            support_weight=args.support_weight,
            halo_budget=args.halo_budget,
        )
    if shots and (args.support_prior or args.report_outside):
        # the campaign geometry is shared across shots: one limiter serves all
        g0 = shots[0]["grid"]
        prior_kw.update(limiter_r=g0.limiter_r, limiter_z=g0.limiter_z)
    winner_cfg = InverseConfig(
        iters=args.iters,
        lr=args.lr,
        n_bins=args.n_bins,
        connectivity=connectivity,
        **P3_WINNER_KW,
        **prior_kw,
    )

    model_rows, ref_rows, flattop_flags, shot_rows, saddle_rows = [], [], [], [], []
    neg_rows, out_rows, misfit_rows = [], [], []
    t0 = time.perf_counter()
    for payload in shots:
        grid, basis = payload["grid"], payload["basis"]
        # flat-top proxy (matches patch_flux_map_report.select_slices' pick):
        # the single highest-|Ip| candidate slice for this shot
        ips = np.abs([p.ip_amperes for p in payload["payloads"]])
        flattop_idx = int(np.argmax(ips)) if ips.size else -1
        if args.boundary_arm == "current-smooth":
            pairs = _grid_neighbor_pairs(basis)
            inv = invert_slices_smooth(
                basis,
                payload["payloads"],
                winner_cfg,
                pairs,
                args.current_smooth_lambda,
                device=device,
            )
        else:
            inv = invert_slices(basis, payload["payloads"], winner_cfg, device=device)
        for k, r in enumerate(inv):
            psi2d = basis.psi_grid_2d_np(r.i_cell, payload["payloads"][k].i_pf)
            target, _, _ = geometry_target(
                psi2d,
                grid,
                smooth_sigma=args.smooth_sigma,
                min_axis_dist=args.min_axis_dist,
            )
            model_rows.append(target)
            ref_rows.append(payload["refs"][k])
            flattop_flags.append(k == flattop_idx)
            shot_rows.append(r.shot)
            saddle_rows.append(count_saddles(psi2d, grid))
            neg_rows.append(r.negative_fraction)
            out_rows.append(r.outside_fraction)
            misfit_rows.append(r.misfit)
    dt = time.perf_counter() - t0

    model = np.array(model_rows)
    ref = np.array(ref_rows)
    flattop_mask = np.array(flattop_flags, dtype=bool)
    shot_ids = np.array(shot_rows)
    sc = score(model, ref, baseline_vec, shot_ids=shot_ids)
    axis_errors = sc.pop("axis_errors")
    lcfs_cm = lcfs_offset_cm_stats(model, ref, flattop_mask)
    saddle_stats = saddle_excess_stats(saddle_rows, ref)

    tag_bits = [args.boundary_arm or "baseline"]
    if args.smooth_sigma:
        tag_bits.append(f"sigma{args.smooth_sigma:g}")
    if args.min_axis_dist:
        tag_bits.append(f"dist{args.min_axis_dist:g}")
    if args.current_smooth_lambda:
        tag_bits.append(f"lam{args.current_smooth_lambda:g}")
    if args.sign_prior != "none":
        tag_bits.append(f"sign-{args.sign_prior}")
        if args.sign_prior == "penalty":
            tag_bits.append(f"sw{args.sign_weight:g}")
    if args.support_prior:
        tag_bits.append(f"support-hb{args.halo_budget:g}-sw{args.support_weight:g}")
    if args.split == "train":
        tag_bits.append("tune")
    tag = "-".join(tag_bits)

    neg = np.asarray(neg_rows, dtype=np.float64)
    out_frac = np.asarray(out_rows, dtype=np.float64)
    prior_stats = {
        "sign_prior": args.sign_prior,
        "sign_weight": args.sign_weight if args.sign_prior == "penalty" else None,
        "support_prior": bool(args.support_prior),
        "support_weight": args.support_weight if args.support_prior else None,
        "halo_budget": args.halo_budget if args.support_prior else None,
        "negative_fraction_median": float(np.nanmedian(neg)) if neg.size else None,
        "negative_fraction_mean": float(np.nanmean(neg)) if neg.size else None,
        "misfit_median": (
            float(np.nanmedian(misfit_rows)) if misfit_rows else None
        ),
        "outside_fraction_median": (
            float(np.nanmedian(out_frac))
            if out_frac.size and np.isfinite(out_frac).any()
            else None
        ),
        "outside_fraction_mean": (
            float(np.nanmean(out_frac))
            if out_frac.size and np.isfinite(out_frac).any()
            else None
        ),
    }

    result = {
        "arm": args.boundary_arm or "baseline",
        "split": args.split,
        "smooth_sigma": args.smooth_sigma,
        "min_axis_dist": args.min_axis_dist,
        "current_smooth_lambda": args.current_smooth_lambda,
        "winner_config": {**P3_WINNER_KW, "iters": args.iters},
        "coil_model_version": COIL_MODEL_VERSION,
        "geometry_table_version": GEOMETRY_TABLE_VERSION,
        "n_scored": int(len(model)),
        "n_candidate": int(len(model)),
        "scored_fraction": 1.0,
        "wall_s": dt,
        **sc,
        **lcfs_cm,
        **saddle_stats,
        **prior_stats,
    }
    (ARTIFACTS / f"boundary_read_{tag}.json").write_text(json.dumps(result, indent=2))
    np.savez(
        ARTIFACTS / f"boundary_read_{tag}_arrays.npz",
        model=model,
        ref=ref,
        baseline=np.tile(baseline_vec, (len(model), 1)),
        axis_errors=axis_errors,
        flattop_mask=flattop_mask,
        saddles=np.asarray(saddle_rows),
        shot_ids=shot_ids,
        negative_fraction=neg,
        outside_fraction=out_frac,
        misfit=np.asarray(misfit_rows, dtype=np.float64),
    )
    logger.info(
        "[boundary-arm %s] scored %d/%d axis_skill=%.3f lcfs_skill=%s median %.3f m "
        "lcfs_offset_cm(all/flattop)=%s/%s (%.0f s)",
        tag,
        len(model),
        len(model),
        sc["axis_skill"],
        sc["lcfs_skill"],
        sc["axis_error_median_m"],
        lcfs_cm["lcfs_offset_median_cm_all"],
        lcfs_cm["lcfs_offset_median_cm_flattop"],
        dt,
    )
    return 0


def _invert_shots_once(
    shots: list[dict], winner_cfg: InverseConfig, device: str | torch.device
) -> tuple[list[tuple[dict, list[SliceInversion]]], np.ndarray]:
    """Invert every shot's candidate slices ONCE with the frozen P3-winner
    config; the (payload, inversion) pairs are reused across every boundary-
    read grid point.

    The Adam inverse is a chaotically-sensitive optimisation over a highly
    underdetermined current basis: re-running it per hyperparameter (as the
    single-arm ``run_boundary_arm`` path does for every ``--min-axis-dist`` /
    ``--smooth-sigma`` value) lands in a DIFFERENT local optimum run to run —
    confirmed by comparing two identical re-runs of the P3-winner config,
    which gave axis_skill -1.28 vs -1.41 on the same shots with NOTHING
    changed.  Since neither ``smooth_sigma`` nor ``min_axis_dist`` touch the
    inverse at all (only the readout downstream of the converged ψ), a valid
    A/B test of the boundary read must hold the currents fixed and vary only
    the read — this function is the fix.
    """
    cache: list[tuple[dict, list[SliceInversion]]] = []
    flattop_flags: list[bool] = []
    for payload in shots:
        basis = payload["basis"]
        ips = np.abs([p.ip_amperes for p in payload["payloads"]])
        flattop_idx = int(np.argmax(ips)) if ips.size else -1
        inv = invert_slices(basis, payload["payloads"], winner_cfg, device=device)
        cache.append((payload, inv))
        flattop_flags.extend(k == flattop_idx for k in range(len(inv)))
    return cache, np.array(flattop_flags, dtype=bool)


def _score_grid_point(
    cache: list[tuple[dict, list[SliceInversion]]],
    baseline_vec: np.ndarray,
    flattop_mask: np.ndarray,
    *,
    smooth_sigma: float,
    min_axis_dist: float,
) -> tuple[np.ndarray, np.ndarray, dict, dict, np.ndarray]:
    """Read geometry from the SAME cached currents at one (sigma, dist) point."""
    model_rows, ref_rows, shot_rows = [], [], []
    for payload, inv in cache:
        grid = payload["grid"]
        for k, r in enumerate(inv):
            psi2d = payload["basis"].psi_grid_2d_np(
                r.i_cell, payload["payloads"][k].i_pf
            )
            target, _, _ = geometry_target(
                psi2d, grid, smooth_sigma=smooth_sigma, min_axis_dist=min_axis_dist
            )
            model_rows.append(target)
            ref_rows.append(payload["refs"][k])
            shot_rows.append(r.shot)
    model = np.array(model_rows)
    ref = np.array(ref_rows)
    shot_ids = np.array(shot_rows)
    sc = score(model, ref, baseline_vec, shot_ids=shot_ids)
    axis_errors = sc.pop("axis_errors")
    lcfs_cm = lcfs_offset_cm_stats(model, ref, flattop_mask)
    return model, ref, sc, lcfs_cm, axis_errors


def _parse_grid_spec(spec: str) -> list[tuple[str, float, float]]:
    """Parse ``--grid`` into ``[(label, smooth_sigma, min_axis_dist), ...]``.

    Tokens are comma-separated; each is ``baseline``, ``sigma=<v>``,
    ``dist=<v>``, or a ``+``-joined combo ``sigma=<v>+dist=<v>``.
    """
    points = []
    for tok in spec.split(","):
        tok = tok.strip()
        sigma, dist = 0.0, 0.0
        if tok and tok != "baseline":
            for part in tok.split("+"):
                key, _, val = part.partition("=")
                if key == "sigma":
                    sigma = float(val)
                elif key == "dist":
                    dist = float(val)
                else:
                    raise ValueError(f"unknown grid token: {part!r} (in {tok!r})")
        points.append((tok or "baseline", sigma, dist))
    return points


def run_boundary_arm_grid(args) -> int:
    """Measure MANY (smooth_sigma, min_axis_dist) boundary-read points from
    ONE frozen-config inversion per shot — the leakage-free, nondeterminism-
    free A4 measurement (see :func:`_invert_shots_once`).  ``--grid`` is a
    comma list of points (``_parse_grid_spec``); writes ONE
    ``boundary_read_grid_<split>.json`` with all points' results.
    """
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    device = (
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else args.device
    )
    connectivity = None if args.connectivity in ("", "none") else args.connectivity
    points = _parse_grid_spec(args.grid)
    logger.info(
        "boundary-arm-grid split=%s points=%s device=%s", args.split, points, device
    )

    train_shots, held_shots = read_split_shot_lists(args.n_train, args.n_heldout)
    baseline_vec = train_mean_baseline(
        args.n_train, args.n_baseline_shots, args.min_ip_ka
    )
    eval_shots = (
        train_shots[args.n_baseline_shots : args.n_baseline_shots + args.n_tune_shots]
        if args.split == "train"
        else held_shots
    )

    shots = []
    for s in eval_shots:
        try:
            payload = shot_payloads(
                s,
                nr=args.nr,
                nz=args.nz,
                max_slices=args.max_slices_per_shot,
                min_ip_ka=args.min_ip_ka,
                split=args.split,
                scale_floor_rel=args.scale_floor_rel,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("shot %s failed to load: %s", s, exc)
            continue
        if payload is not None:
            shots.append(payload)

    winner_cfg = InverseConfig(
        iters=args.iters,
        lr=args.lr,
        n_bins=args.n_bins,
        connectivity=connectivity,
        **P3_WINNER_KW,
    )
    t0 = time.perf_counter()
    cache, flattop_mask = _invert_shots_once(shots, winner_cfg, device)
    invert_wall_s = time.perf_counter() - t0
    logger.info("inverted %d shots once in %.0f s", len(shots), invert_wall_s)

    results = []
    for label, sigma, dist in points:
        t0 = time.perf_counter()
        model, ref, sc, lcfs_cm, axis_errors = _score_grid_point(
            cache, baseline_vec, flattop_mask, smooth_sigma=sigma, min_axis_dist=dist
        )
        dt = time.perf_counter() - t0
        tag = f"{label}-{args.split}" if args.split == "train" else label
        np.savez(
            ARTIFACTS / f"boundary_read_grid_{tag}_arrays.npz",
            model=model,
            ref=ref,
            baseline=np.tile(baseline_vec, (len(model), 1)),
            axis_errors=axis_errors,
            flattop_mask=flattop_mask,
        )
        point_result = {
            "label": label,
            "smooth_sigma": sigma,
            "min_axis_dist": dist,
            "n_scored": int(len(model)),
            "n_candidate": int(len(model)),
            "scored_fraction": 1.0,
            "readout_wall_s": dt,
            **sc,
            **lcfs_cm,
        }
        results.append(point_result)
        logger.info(
            "[grid %s] axis_skill=%.3f lcfs_skill=%s median %.3f m "
            "lcfs_offset_cm(all/flattop)=%s/%s",
            label,
            sc["axis_skill"],
            sc["lcfs_skill"],
            sc["axis_error_median_m"],
            lcfs_cm["lcfs_offset_median_cm_all"],
            lcfs_cm["lcfs_offset_median_cm_flattop"],
        )

    out = {
        "split": args.split,
        "winner_config": {**P3_WINNER_KW, "iters": args.iters},
        "coil_model_version": COIL_MODEL_VERSION,
        "geometry_table_version": GEOMETRY_TABLE_VERSION,
        "invert_wall_s": invert_wall_s,
        "points": results,
    }
    grid_tag = "tune" if args.split == "train" else "eval"
    (ARTIFACTS / f"boundary_read_grid_{grid_tag}.json").write_text(
        json.dumps(out, indent=2)
    )
    logger.info("grid results written to boundary_read_grid_%s.json", grid_tag)
    return 0


# ---------------------------------------------------------------------------
# F — whitening-floor rel sensitivity of the per-slice inverse.  Unlike A4's
# boundary-read levers, the floor is baked into SlicePayload.scale at
# shot_payloads() load time and enters the whitened-misfit term of the
# inverse's objective directly, so each rel value needs a FRESH
# shot_payloads() load AND a fresh re-inversion — no cross-value current
# caching is possible (see _invert_shots_once's docstring for why caching
# matters when it IS valid).
# ---------------------------------------------------------------------------


def run_floor_sensitivity(args) -> int:
    """Sweep ``--floor-rel-grid``'s rel_floor values against the frozen
    P3-winner inverse, scoring axis/LCFS/misfit at each — the measurement
    that answers whether the training-motivated whitening floor (commit
    19820ad) costs the per-slice inverse axis-placement skill by
    over-deweighting quiet-but-precise flux loops (patch-equilibrium-wm-
    integration flux-map re-run, commit 0e5fba3).

    Cohort: the SAME train-shot tuning selection ``run_boundary_arm``/
    ``run_boundary_arm_grid`` use (``--split train``'s
    ``train_shots[n_baseline_shots : n_baseline_shots + n_tune_shots]``,
    default 4 shots) — leakage-free (referee labels here never touch the
    held-out gate).  Per shot, only the rampup-proxy (earliest valid) and
    flattop (highest-|Ip|) slices are scored, giving ~8 slices total at the
    default ``n_tune_shots=4`` — enough to see the effect while keeping each
    rel value's re-inversion cheap.  ``--split eval`` is also honoured
    (all held-out slices, no rampup/flattop subselection) for the ONE frozen-
    rel held-out verification pass; it must not be used for value selection.

    Writes ``floor_sensitivity_tune.json`` (``--split train``) or
    ``floor_sensitivity_heldout.json`` (``--split eval``).
    """
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    device = (
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else args.device
    )
    connectivity = None if args.connectivity in ("", "none") else args.connectivity
    rels = [float(v) for v in args.floor_rel_grid.split(",") if v.strip() != ""]
    logger.info("floor-rel-grid=%s split=%s device=%s", rels, args.split, device)

    train_shots, held_shots = read_split_shot_lists(args.n_train, args.n_heldout)
    baseline_vec = train_mean_baseline(
        args.n_train, args.n_baseline_shots, args.min_ip_ka
    )
    eval_shots = (
        train_shots[args.n_baseline_shots : args.n_baseline_shots + args.n_tune_shots]
        if args.split == "train"
        else held_shots
    )

    winner_cfg = InverseConfig(
        iters=args.iters,
        lr=args.lr,
        n_bins=args.n_bins,
        connectivity=connectivity,
        **P3_WINNER_KW,
    )

    results = []
    for rel in rels:
        t0 = time.perf_counter()
        shots = []
        for s in eval_shots:
            try:
                payload = shot_payloads(
                    s,
                    nr=args.nr,
                    nz=args.nz,
                    max_slices=args.max_slices_per_shot,
                    min_ip_ka=args.min_ip_ka,
                    split=args.split,
                    scale_floor_rel=rel,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("shot %s failed to load at rel=%s: %s", s, rel, exc)
                continue
            if payload is not None:
                shots.append(payload)

        model_rows, ref_rows, misfits = [], [], []
        flattop_flags: list[bool] = []
        for payload in shots:
            grid, basis = payload["grid"], payload["basis"]
            all_payloads = payload["payloads"]
            ips = np.abs([p.ip_amperes for p in all_payloads])
            flattop_idx = int(np.argmax(ips)) if ips.size else -1
            sel = (
                sorted({0, flattop_idx})
                if args.split == "train"
                else list(range(len(all_payloads)))
            )
            sub_payloads = [all_payloads[k] for k in sel]
            sub_refs = payload["refs"][sel]
            inv = invert_slices(basis, sub_payloads, winner_cfg, device=device)
            for k, r in enumerate(inv):
                psi2d = basis.psi_grid_2d_np(r.i_cell, sub_payloads[k].i_pf)
                target, _, _ = geometry_target(psi2d, grid)
                model_rows.append(target)
                ref_rows.append(sub_refs[k])
                misfits.append(r.misfit)
                flattop_flags.append(sel[k] == flattop_idx)
        dt = time.perf_counter() - t0

        model = np.array(model_rows)
        ref = np.array(ref_rows)
        flattop_mask = np.array(flattop_flags, dtype=bool)
        sc = score(model, ref, baseline_vec)
        sc.pop("axis_errors")
        lcfs_cm = lcfs_offset_cm_stats(model, ref, flattop_mask)
        n_finite_misfit = int(np.sum(np.isfinite(misfits)))
        point = {
            "scale_floor_rel": rel,
            "n_scored": int(len(model)),
            "n_shots": len(shots),
            "misfit_median": float(np.nanmedian(misfits)) if misfits else None,
            "misfit_n_nonfinite": len(misfits) - n_finite_misfit,
            "wall_s": dt,
            **sc,
            **lcfs_cm,
        }
        results.append(point)
        logger.info(
            "[floor rel=%.3f] scored %d axis_skill=%.3f axis_median=%.3fm "
            "lcfs_offset_cm(all/flattop)=%s/%s misfit_median=%s (%.0f s)",
            rel,
            len(model),
            sc["axis_skill"],
            sc["axis_error_median_m"],
            lcfs_cm["lcfs_offset_median_cm_all"],
            lcfs_cm["lcfs_offset_median_cm_flattop"],
            point["misfit_median"],
            dt,
        )

    out = {
        "split": args.split,
        "winner_config": {**P3_WINNER_KW, "iters": args.iters},
        "coil_model_version": COIL_MODEL_VERSION,
        "geometry_table_version": GEOMETRY_TABLE_VERSION,
        "n_tune_shots": args.n_tune_shots if args.split == "train" else None,
        "cohort": "rampup+flattop per shot"
        if args.split == "train"
        else "all candidate slices",
        "points": results,
    }
    tag = "tune" if args.split == "train" else "heldout"
    out_path = ARTIFACTS / f"floor_sensitivity_{tag}.json"
    out_path.write_text(json.dumps(out, indent=2))
    logger.info("floor sensitivity results written to %s", out_path)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-train", type=int, default=40)
    ap.add_argument("--n-heldout", type=int, default=8)
    ap.add_argument("--nr", type=int, default=65)
    ap.add_argument("--nz", type=int, default=97)
    ap.add_argument("--max-slices-per-shot", type=int, default=20)
    ap.add_argument("--min-ip-ka", type=float, default=300.0)
    ap.add_argument("--n-baseline-shots", type=int, default=10)
    ap.add_argument("--policies", type=str, default="fixed,warm-start,discrepancy")
    ap.add_argument("--lambda-fb", type=float, default=10.0)
    ap.add_argument(
        "--arms",
        type=str,
        default="",
        help=(
            "explicit arm spec overriding --policies/--lambda-fb: comma list of "
            "policy:lambda[:misfit_ratio[:lambda_max]] tokens, e.g. "
            "'fixed:0,fixed:3,fixed:10,warm-start:10,discrepancy:10:1.3:30'"
        ),
    )
    ap.add_argument(
        "--split",
        type=str,
        default="eval",
        choices=("eval", "train"),
        help=(
            "eval = the held-out gate; train = TRAIN-shot slices for "
            "leakage-free policy/lambda selection (shots after the baseline block)"
        ),
    )
    ap.add_argument("--n-tune-shots", type=int, default=4)
    ap.add_argument("--out-tag", type=str, default="")
    ap.add_argument("--iters", type=int, default=800)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--n-bins", type=int, default=24)
    ap.add_argument("--form", type=str, default="affine-r2")
    ap.add_argument("--connectivity", type=str, default="locality")
    ap.add_argument(
        "--sign-prior",
        type=str,
        default="none",
        choices=["none", "softplus", "penalty"],
        help="unidirectional-current prior: hard reparametrisation or soft penalty",
    )
    ap.add_argument("--sign-weight", type=float, default=10.0)
    ap.add_argument(
        "--support-prior",
        action="store_true",
        help="soft penalty on current outside the LCFS of its own psi",
    )
    ap.add_argument("--support-weight", type=float, default=20.0)
    ap.add_argument("--halo-budget", type=float, default=0.03)
    ap.add_argument(
        "--report-outside",
        action="store_true",
        help="report the outside-LCFS current fraction with the prior off",
    )
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--throughput-bench", action="store_true")
    ap.add_argument(
        "--boundary-arm",
        type=str,
        default="",
        choices=("", "baseline", "current-smooth"),
        help=(
            "lever A4 measurement mode: if set, run_boundary_arm() replaces "
            "the normal policy sweep entirely (writes boundary_read_<tag>.json "
            "against the frozen P3-winner inverse). '' = normal gate (default, "
            "unaffected). 'baseline' = P3-winner inverse + geometry_target read "
            "with --smooth-sigma/--min-axis-dist (0/0 reproduces the P3 gate "
            "numbers exactly). 'current-smooth' = re-invert with the ARM-1 "
            "current-smoothness penalty at --current-smooth-lambda."
        ),
    )
    ap.add_argument(
        "--smooth-sigma",
        type=float,
        default=0.0,
        help="ARM 3: Gaussian-smooth psi (grid cells) before the boundary read",
    )
    ap.add_argument(
        "--min-axis-dist",
        type=float,
        default=0.0,
        help="ARM 2: reject candidate X-points closer than this [m] to the axis",
    )
    ap.add_argument(
        "--current-smooth-lambda",
        type=float,
        default=0.0,
        help="ARM 1: weight of the current spatial-smoothness penalty",
    )
    ap.add_argument(
        "--grid",
        type=str,
        default="",
        help=(
            "lever A4 grid-measurement mode (preferred over --boundary-arm for "
            "ARM 2/3): comma list of 'baseline' / 'sigma=<v>' / 'dist=<v>' / "
            "'sigma=<v>+dist=<v>' points, evaluated from ONE frozen-config "
            "inversion per shot (invert once, vary only the readout — avoids "
            "the run-to-run Adam nondeterminism confound of re-inverting per "
            "point). Writes ONE boundary_read_grid_<split>.json."
        ),
    )
    ap.add_argument(
        "--scale-floor-rel",
        type=float,
        default=CHANNEL_SCALE_KIND_FLOOR_REL,
        help=(
            "rel_floor passed to robust_channel_scale for shot_payloads' "
            "sensor whitening scale (default = the training convention, "
            "0.05). Applies to every mode (normal gate, --boundary-arm, "
            "--grid, --floor-rel-grid)."
        ),
    )
    ap.add_argument(
        "--floor-rel-grid",
        type=str,
        default="",
        help=(
            "F: whitening-floor sensitivity sweep. Comma list of rel_floor "
            "values (e.g. '0.0,0.01,0.02,0.05'); each value re-loads "
            "shot_payloads AND re-inverts (the floor is baked into the "
            "whitened-misfit objective, unlike --grid's boundary-read "
            "levers, so currents cannot be cached across values). Runs on "
            "the frozen P3-winner config, --split train's tuning cohort by "
            "default (leakage-free). Writes floor_sensitivity_<tag>.json."
        ),
    )
    args = ap.parse_args()

    if args.floor_rel_grid:
        return run_floor_sensitivity(args)
    if args.grid:
        return run_boundary_arm_grid(args)
    if args.boundary_arm:
        return run_boundary_arm(args)

    device = (
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else args.device
    )
    connectivity = None if args.connectivity in ("", "none") else args.connectivity
    logger.info("device=%s connectivity=%s", device, connectivity)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    train_shots, held_shots = read_split_shot_lists(args.n_train, args.n_heldout)
    baseline_vec = train_mean_baseline(
        args.n_train, args.n_baseline_shots, args.min_ip_ka
    )
    logger.info(
        "baseline (train-mean) axis: (%.3f, %.3f)", baseline_vec[0], baseline_vec[1]
    )

    if args.split == "train":
        # tuning cohort: TRAIN shots after the baseline block — selection on
        # these referee labels never touches the held-out gate
        eval_shots = train_shots[
            args.n_baseline_shots : args.n_baseline_shots + args.n_tune_shots
        ]
    else:
        eval_shots = held_shots
    shots = []
    for s in eval_shots:
        try:
            payload = shot_payloads(
                s,
                nr=args.nr,
                nz=args.nz,
                max_slices=args.max_slices_per_shot,
                min_ip_ka=args.min_ip_ka,
                split=args.split,
                scale_floor_rel=args.scale_floor_rel,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("shot %s failed to load: %s", s, exc)
            continue
        if payload is not None:
            shots.append(payload)
            logger.info(
                "shot %d: %d candidate slices",
                payload["payloads"][0].shot,
                len(payload["payloads"]),
            )

    # ---- P1-gate throughput bench (batched forward on this device) ----------
    if args.throughput_bench and shots:
        basis = shots[0]["basis"]
        rates = []
        for batch in (64, 1024, 4096):
            rate = basis.throughput(batch=batch, n_iter=20, device=device)
            logger.info(
                "throughput: %d-slice batched forward on %s → %.0f slices/s",
                batch,
                device,
                rate,
            )
            rates.append({"batch": batch, "slices_per_s": rate})
        (ARTIFACTS / "throughput.json").write_text(
            json.dumps({"device": device, "rates": rates}, indent=2)
        )

    # ---- inverse per policy arm ---------------------------------------------
    per_policy: dict[str, dict] = {}
    tag = args.out_tag or ("_tune" if args.split == "train" else "")
    if args.arms:
        arm_specs = []
        for tok in args.arms.split(","):
            parts = tok.split(":")
            kw: dict = {"policy": parts[0]}
            if len(parts) > 1:
                kw["lambda_fb"] = float(parts[1])
            if len(parts) > 2:
                kw["misfit_ratio"] = float(parts[2])
            if len(parts) > 3:
                kw["lambda_max"] = float(parts[3])
            arm_specs.append((tok, kw))
    else:
        arm_specs = [
            (p, {"policy": p, "lambda_fb": args.lambda_fb})
            for p in args.policies.split(",")
        ]
    for policy, arm_kw in arm_specs:
        cfg = InverseConfig(
            iters=args.iters,
            lr=args.lr,
            n_bins=args.n_bins,
            connectivity=connectivity,
            **arm_kw,
        )
        model_rows, ref_rows, diag_rows, shot_rows, saddle_rows = [], [], [], [], []
        psi_reads = []  # (psi_axis, psi_boundary) per scored slice, for closures
        inversions_all = []
        t0 = time.perf_counter()
        for payload in shots:
            grid, basis = payload["grid"], payload["basis"]
            inv = invert_slices(basis, payload["payloads"], cfg, device=device)
            inversions_all.append((payload, inv))
            for k, r in enumerate(inv):
                psi2d = basis.psi_grid_2d_np(r.i_cell, payload["payloads"][k].i_pf)
                target, psi_ax, psi_b = geometry_target(psi2d, grid)
                model_rows.append(target)
                ref_rows.append(payload["refs"][k])
                psi_reads.append((psi_ax, psi_b))
                shot_rows.append(r.shot)
                saddle_rows.append(count_saddles(psi2d, grid))
                diag_rows.append(
                    {
                        "shot": r.shot,
                        "t_index": r.t_index,
                        "time_s": r.time_s,
                        "misfit": r.misfit,
                        "structure": r.structure,
                        "lambda_final": r.lambda_final,
                        "ip_rel_err": r.ip_rel_err,
                    }
                )
        dt = time.perf_counter() - t0
        model = np.array(model_rows)
        ref = np.array(ref_rows)
        shot_ids = np.array(shot_rows)
        sc = score(model, ref, baseline_vec, shot_ids=shot_ids)
        axis_errors = sc.pop("axis_errors")
        per_policy[policy] = {
            **sc,
            **saddle_excess_stats(saddle_rows, ref),
            "n_scored": int(len(model)),
            "n_candidate": int(len(model)),
            "scored_fraction": 1.0,
            "wall_s": dt,
            "diag": diag_rows,
        }
        np.savez(
            ARTIFACTS / f"gate_arrays_{policy.replace(':', '-')}{tag}.npz",
            model=model,
            ref=ref,
            baseline=np.tile(baseline_vec, (len(model), 1)),
            axis_errors=axis_errors,
        )
        logger.info(
            "[%s] scored %d/%d axis_skill=%.3f lcfs_skill=%s median %.3f m (%.0f s)",
            policy,
            len(model),
            len(model),
            sc["axis_skill"],
            sc["lcfs_skill"],
            sc["axis_error_median_m"],
            dt,
        )

        # ---- closures for this arm (recovered p', FF'/mu0 per slice) --------
        if args.split == "train":
            per_policy[policy]["closures"] = []
            continue  # tuning run: skills only, closures belong to the gate
        closure_rows = []
        idx = 0
        for payload, inv in inversions_all:
            basis = payload["basis"]
            r_c = basis.r_cells.to(torch.float64)
            for k, r in enumerate(inv):
                p = payload["payloads"][k]
                psi_c = basis.psi_cells_np(r.i_cell, p.i_pf)
                jphi = r.i_cell / float(basis.cell_area)
                fit = fit_flux_functions(
                    torch.as_tensor(psi_c, dtype=torch.float64),
                    r_c,
                    torch.as_tensor(jphi, dtype=torch.float64),
                    n_bins=args.n_bins,
                    form=args.form,
                )
                psi_ax, psi_b = psi_reads[idx]
                integ = integrate_closures(
                    fit, psi_axis=psi_ax, psi_boundary=psi_b, f_vac=0.85 * 0.55
                )
                closure_rows.append(
                    {
                        "shot": r.shot,
                        "t_index": r.t_index,
                        "psi_bins": np.asarray(fit.psi_centers).tolist(),
                        "a_k": np.asarray(fit.a_k).tolist(),
                        "b_k": np.asarray(fit.b_k).tolist(),
                        "a_err": np.asarray(fit.a_err).tolist(),
                        "b_err": np.asarray(fit.b_err).tolist(),
                        "weight_mass": np.asarray(fit.weight_mass).tolist(),
                        "psi_axis": psi_ax,
                        "psi_boundary": psi_b,
                        "f2_min": float(np.min(integ["f_squared"])),
                        "p_axis": float(np.max(np.abs(integ["p"]))),
                    }
                )
                idx += 1
        per_policy[policy]["closures"] = closure_rows

    # ---- artifacts + report --------------------------------------------------
    picard_ref = None
    picard_path = Path("imas_ambix/latent/artifacts/gs_solve_gate_eval.json")
    if picard_path.exists():
        picard_ref = json.loads(picard_path.read_text())
    result = {
        "config": {k: v for k, v in vars(args).items()},
        "device": device,
        "baseline_axis": [float(baseline_vec[0]), float(baseline_vec[1])],
        "picard_reference": picard_ref
        and {
            "n_scored": picard_ref["n_scored"],
            "n_candidate": picard_ref["n_candidate"],
            "axis_error_median_m": picard_ref["axis_error_median_m"],
            "axis_skill": picard_ref["axis_skill"],
            "lcfs_skill": picard_ref["lcfs_skill"],
        },
        "per_policy": {
            k: {kk: vv for kk, vv in v.items() if kk not in ("diag", "closures")}
            for k, v in per_policy.items()
        },
    }
    (ARTIFACTS / f"patch_gate_eval{tag}.json").write_text(json.dumps(result, indent=2))
    (ARTIFACTS / f"patch_gate_diag{tag}.json").write_text(
        json.dumps(
            {
                k: {"diag": v["diag"], "closures": v["closures"]}
                for k, v in per_policy.items()
            },
            indent=2,
        )
    )
    logger.info("gate artifacts written to %s", ARTIFACTS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
