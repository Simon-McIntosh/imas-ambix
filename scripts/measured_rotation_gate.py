#!/usr/bin/env python
"""Measured-rotation forward test: a fixed centrifugal γ(ψ_N) through the gate.

Stage A found the toroidal-rotation R⁴ column *visible but not load-bearing* in
the held-out GS residuals (it absorbs ~10%/6% of the within-surface residual,
c(ψ) one-signed, |c|-vs-CXRS Spearman 0.54 on the sign-constrained arm).  This
script asks the falsifiable forward question with ZERO new degrees of freedom:
if we build the centrifugal factor from MEASURED CXRS rotation and temperature
per slice and multiply it onto the winner closure config's pressure-drive basis
columns, does the held-out fit/topology improve on the rotating shots?

Physics.  The centrifugal enhancement of the pressure drive is
``exp[γ(ψ_N)·(R² − R₀²)]`` with

    γ(ψ_N) = m_D · Ω²(ψ_N) / (2 · T_i(ψ_N))     [m⁻²],

Ω = v_tor/R the toroidal angular frequency (a flux function under rigid
rotation), m_D the deuteron mass.  Ω comes from the CXRS carbon toroidal
velocity (``act`` group, ``ss``/``sw`` beam systems — see the system-selection
note below), T_i from the CXRS carbon temperature (``act`` … ``_temperature``,
units eV) as the ion-temperature proxy (T_i,C ≈ T_i,D in the collisional core —
the stated caveat; MAST Thomson ``ayc`` is absent on these shots, so the CXRS
carbon temperature is the only available T).

Two-pass mapping (honest — never uses EFIT).  Per slice we first solve the
winner config WITHOUT rotation (arm a), read its outboard-midplane ψ_N(R) from
the force-balanced ψ, and use it to convert the measured (R, v_tor, T_i) points
to γ(ψ_N); then re-solve WITH ``centrifugal_gamma`` (arm b).  The mapping uses
the static solve, not an external equilibrium.

System selection.  ``ss`` and ``sw`` are both "SS beam — Carbon Velocity" and
agree to ~3% over the full R range on the covered shots; ``pla`` reads ~20×
lower over identical radii (a different projection / geometry, not a clean
v_tor), so it is EXCLUDED.  Points are gated physically (finite, SNR>1,
|v|<300 km/s, 20 eV<T<5 keV) and restricted to the outboard branch
(R ≥ R_axis) where the monotone ψ_N(R) map is single-valued.

Rotation data exists on 18502/18503/18505 only; on the other 5 held-out shots
γ ≡ 0, so the two arms are byte-identical there — the null check.  The powered
comparison is the paired arm-b − arm-a delta on the ≤3 covered shots (LOW
POWER, n_shots ≤ 3).  Leakage-free: zero DOF, nothing to tune; EFIT enters only
the referee/scoring path.

Artifacts:  imas_ambix/latent/artifacts/patch_gate/measured_rotation_gate.json
            imas_ambix/latent/artifacts/patch_gate/measured_rotation_gate_arrays.npz
Figures:    docs/figures/force-balance-spine/fig-measured-rotation-deltas.png
            docs/figures/force-balance-spine/fig-measured-rotation-gamma.png
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np

from imas_ambix.latent.data import read_split_shot_lists
from imas_ambix.latent.evaluate import (
    headline_skill,
    matched_xpoint_error,
    per_quantity_skill,
)
from imas_ambix.latent.gs_solve import fit_profile_ladder

# import-only reuse of the hardened harness (identical cohort, calibration,
# sidecar, geometry readout, scoring) — never mutate these modules
from scripts.closure_gate_eval import _apply_calibration, _shot_passive_sidecar
from scripts.patch_gate_eval import (
    TARGET_NAMES,
    count_saddles,
    geometry_target,
    shot_payloads,
    train_mean_baseline,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("measured_rotation_gate")

ARTIFACTS = Path("imas_ambix/latent/artifacts/patch_gate")
FIGURES = Path("docs/figures/force-balance-spine")
CALIBRATION = str(ARTIFACTS / "static_calibration_offset_only.json")

# winner closure config (Stage-A precedent): n_p=n_f=1 ladder + rank-8 passive
N_P = 1
N_F = 1
PASSIVE_K = 8
CONV_LIMIT = 5e-3  # scoring-inclusion criterion, identical to closure_gate_eval

# physical constants
M_D = 3.343583719e-27  # deuteron mass [kg]
EV_TO_J = 1.602176634e-19

# CXRS physical gates (act carbon v_tor + T_i)
CXRS_SYSTEMS = ("ss", "sw")  # pla excluded — see module docstring
V_CEIL_MS = 3.0e5  # 300 km/s toroidal-rotation ceiling
T_FLOOR_EV = 20.0
T_CEIL_EV = 5.0e3
ROT_TIME_TOL_S = 0.015  # nearest act frame within this window of the slice time

# γ(ψ_N) construction
GAMMA_N_BINS = 8
GAMMA_MIN_POINTS = 6  # minimum mapped outboard points to build a profile


# ---------------------------------------------------------------------------
# CXRS loader (firewall-clean: a measured input, never a solver output)
# ---------------------------------------------------------------------------
def load_cxrs(shot: int) -> dict | None:
    """Per-shot CXRS carbon v_tor + T_i on the shared ``majorradius`` grid.

    Returns ``{'time', 'R', 'sys': {name: (v, ve, T, Te)}}`` with the arrays
    padded to the 160-chord union (each system populates its own chords; the
    rest are NaN and drop out under the physical gates).  None when the shot
    carries no usable ``act`` for the selected systems.
    """
    import zarr  # noqa: PLC0415

    from imas_ambix.statespace.baseline import _LEVEL1_DIR  # noqa: PLC0415

    p = Path(_LEVEL1_DIR) / f"{shot}.zarr"
    if not p.exists():
        return None
    try:
        st = zarr.open_group(str(p), mode="r")
    except Exception:  # noqa: BLE001
        return None
    if "act" not in st:
        return None
    grp = st["act"]
    if "time" not in grp or "majorradius" not in grp:
        return None
    t = np.asarray(grp["time"], dtype=np.float64)
    r = np.asarray(grp["majorradius"], dtype=np.float64)
    sysd: dict[str, tuple] = {}
    for name in CXRS_SYSTEMS:
        vk, ek = f"{name}_velocity", f"{name}_velocity_error"
        tk, tek = f"{name}_temperature", f"{name}_temperature_error"
        if vk not in grp or tk not in grp:
            continue
        v = np.asarray(grp[vk], dtype=np.float64)
        if v.ndim != 2 or v.shape[0] != t.size or v.shape[1] != r.size:
            continue
        ve = (
            np.asarray(grp[ek], dtype=np.float64)
            if ek in grp
            else np.full_like(v, np.nan)
        )
        temp = np.asarray(grp[tk], dtype=np.float64)
        temp_err = (
            np.asarray(grp[tek], dtype=np.float64)
            if tek in grp
            else np.full_like(temp, np.nan)
        )
        sysd[name] = (v, ve, temp, temp_err)
    if not sysd:
        return None
    return {"time": t, "R": r, "sys": sysd}


def frame_points(
    cxrs: dict | None, time_s: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Pooled physical (R, |v_tor|, T_i) points at the act frame nearest
    ``time_s`` (within ``ROT_TIME_TOL_S``); None if no frame in range or the
    frame carries no physical points."""
    if cxrs is None:
        return None
    t, r = cxrs["time"], cxrs["R"]
    it = int(np.argmin(np.abs(t - time_s)))
    if abs(float(t[it]) - time_s) > ROT_TIME_TOL_S:
        return None
    r_pts, v_pts, t_pts = [], [], []
    for vel, vel_err, temp, temp_err in cxrs["sys"].values():
        vslice, verr, tslice, terr = vel[it], vel_err[it], temp[it], temp_err[it]
        phys = (
            np.isfinite(vslice)
            & np.isfinite(verr)
            & (verr > 0)
            & (np.abs(vslice) < V_CEIL_MS)
            & (verr < np.abs(vslice))  # SNR > 1 on the velocity
            & np.isfinite(tslice)
            & (tslice > T_FLOOR_EV)
            & (tslice < T_CEIL_EV)
            & np.isfinite(terr)
            & (terr > 0)
            & (terr < tslice)  # SNR > 1 on the temperature
        )
        if phys.any():
            r_pts.append(r[phys])
            v_pts.append(np.abs(vslice[phys]))
            t_pts.append(tslice[phys])
    if not r_pts:
        return None
    return (
        np.concatenate(r_pts),
        np.concatenate(v_pts),
        np.concatenate(t_pts),
    )


# ---------------------------------------------------------------------------
# ψ_N(R) outboard-midplane map from a static solve (two-pass, honest)
# ---------------------------------------------------------------------------
def outboard_psi_n_of_r(res, grid):
    """Monotone outboard-midplane ψ_N(R) sampled from the force-balanced ψ.

    Evaluates ψ along Z = Z_axis for R ≥ R_axis by bilinear interpolation of the
    (nz, nr) grid, normalises to (ψ − ψ_axis)/(ψ_boundary − ψ_axis), and returns
    a forward callable R → ψ_N (np.interp on the increasing R samples).  Returns
    ``(callable, R_axis, R_samples, psi_n_samples)`` or None if the axis span is
    degenerate.
    """
    span = res.boundary_psi - res.axis_psi
    if abs(span) < 1e-12:
        return None
    r_ax, z_ax = float(res.axis[0]), float(res.axis[1])
    rg, zg = grid.rg, grid.zg
    r_out = rg[rg >= r_ax]
    if r_out.size < 3:
        return None
    # bilinear sample of ψ(R, Z_axis) along the outboard midplane
    jz = np.clip(np.searchsorted(zg, z_ax) - 1, 0, zg.size - 2)
    wz = (z_ax - zg[jz]) / (zg[jz + 1] - zg[jz])
    psi_row = (1.0 - wz) * res.psi[jz, :] + wz * res.psi[jz + 1, :]
    psi_line = np.interp(r_out, rg, psi_row)
    psi_n_line = (psi_line - res.axis_psi) / span

    def fn(r_query: np.ndarray) -> np.ndarray:
        return np.interp(
            np.asarray(r_query), r_out, psi_n_line, left=np.nan, right=np.nan
        )

    return fn, r_ax, r_out, psi_n_line


def build_gamma(
    pts: tuple[np.ndarray, np.ndarray, np.ndarray],
    psi_n_fn,
    r_axis: float,
) -> tuple | None:
    """Build the γ(ψ_N) callable from mapped CXRS points.

    Maps the outboard (R ≥ R_axis) measured points onto ψ_N, forms
    γ = m_D Ω²/(2 T_i) per point, aggregates by robust median into ψ_N bins,
    and returns a shape-preserving (PCHIP) interpolant that is zero outside the
    measured ψ_N coverage and clipped ≥ 0.  Returns
    ``(gamma_callable, diag)`` or None when fewer than ``GAMMA_MIN_POINTS``
    outboard points map into [0, 1].
    """
    from scipy.interpolate import PchipInterpolator  # noqa: PLC0415

    r_pts, v_pts, t_pts = pts
    out = r_pts >= r_axis
    r_pts, v_pts, t_pts = r_pts[out], v_pts[out], t_pts[out]
    if r_pts.size < GAMMA_MIN_POINTS:
        return None
    psi_n = psi_n_fn(r_pts)
    keep = np.isfinite(psi_n) & (psi_n >= 0.0) & (psi_n <= 1.0)
    if keep.sum() < GAMMA_MIN_POINTS:
        return None
    psi_n = psi_n[keep]
    omega = v_pts[keep] / r_pts[keep]  # rad/s
    gamma = M_D * omega**2 / (2.0 * t_pts[keep] * EV_TO_J)  # [m^-2]
    gamma = np.clip(gamma, 0.0, None)

    lo, hi = float(psi_n.min()), float(psi_n.max())
    if hi - lo < 1e-3:
        return None
    edges = np.linspace(lo, hi, GAMMA_N_BINS + 1)
    idx = np.clip(np.digitize(psi_n, edges) - 1, 0, GAMMA_N_BINS - 1)
    centres, gam_bin = [], []
    for b in range(GAMMA_N_BINS):
        m = idx == b
        if m.any():
            centres.append(float(np.median(psi_n[m])))
            gam_bin.append(float(np.median(gamma[m])))
    centres = np.asarray(centres)
    gam_bin = np.clip(np.asarray(gam_bin), 0.0, None)
    order = np.argsort(centres)
    centres, gam_bin = centres[order], gam_bin[order]
    # de-duplicate ψ_N (PCHIP needs strictly increasing x)
    uniq = np.concatenate(([True], np.diff(centres) > 1e-6))
    centres, gam_bin = centres[uniq], gam_bin[uniq]
    if centres.size < 2:
        return None
    c_lo, c_hi = float(centres[0]), float(centres[-1])
    pchip = PchipInterpolator(centres, gam_bin, extrapolate=False)

    def gamma_fn(psn: np.ndarray) -> np.ndarray:
        psn = np.asarray(psn, dtype=np.float64)
        vals = pchip(np.clip(psn, c_lo, c_hi))
        vals = np.where((psn >= c_lo) & (psn <= c_hi), vals, 0.0)
        return np.clip(np.nan_to_num(vals, nan=0.0), 0.0, None)

    diag = {
        "n_points": int(psi_n.size),
        "psi_n_lo": c_lo,
        "psi_n_hi": c_hi,
        "gamma_bin_centres": centres.tolist(),
        "gamma_bin_values": gam_bin.tolist(),
        "gamma_peak": float(gam_bin.max()),
        "v_tor_line_avg_ms": float(np.mean(v_pts[keep])),
        "v_tor_peak_ms": float(np.max(v_pts[keep])),
        "ti_median_ev": float(np.median(t_pts[keep])),
    }
    return gamma_fn, diag


# ---------------------------------------------------------------------------
# per-slice worker (fork pool; grid inherited copy-on-write)
# ---------------------------------------------------------------------------
_WORKER: dict = {}


def _init_worker(state: dict) -> None:
    _WORKER.update(state)


def _solve(grid, table, payload, gamma):
    """One ladder solve at the winner config; ``gamma`` None = static arm."""
    lf = fit_profile_ladder(
        grid,
        table,
        i_pf=payload.i_pf,
        ip_amperes=payload.ip_amperes,
        measured=payload.measured,
        vacuum_prediction=payload.vacuum,
        sensor_scale=payload.scale,
        sensor_mask=payload.mask,
        n_p=N_P,
        n_f=N_F,
        passive=_WORKER["sidecar"],
        passive_ridge=1.0,
        centrifugal_gamma=gamma,
    )
    res = lf.result
    scored = bool(res.converged or res.residual <= CONV_LIMIT)
    rec = {
        "converged": scored,
        "residual": float(res.residual),
        "cost": float(lf.cost),
    }
    if scored:
        target, _, _ = geometry_target(res.psi, grid)
        rec["target"] = target.tolist()
        rec["saddles"] = int(count_saddles(res.psi, grid))
    return rec, res


def analyze_slice(k: int) -> dict:
    grid = _WORKER["grid"]
    table = _WORKER["table"]
    payload = _WORKER["payloads"][k]

    out: dict = {
        "shot": int(payload.shot),
        "t_index": int(payload.t_index),
        "time_s": float(payload.time_s),
        "ip_amperes": float(payload.ip_amperes),
    }
    # arm a: static (no centrifugal term)
    rec_a, res_a = _solve(grid, table, payload, None)
    out["static"] = rec_a

    # build γ from the static solve's outboard ψ_N(R) map + CXRS at this time
    gamma_fn = None
    out["gamma"] = None
    out["covered"] = False
    if rec_a["converged"]:
        pts = frame_points(_WORKER["cxrs"], payload.time_s)
        mp = outboard_psi_n_of_r(res_a, grid)
        if pts is not None and mp is not None:
            fn, r_ax, _r_out, _psn = mp
            built = build_gamma(pts, fn, r_ax)
            if built is not None:
                gamma_fn, diag = built
                out["gamma"] = diag
                out["covered"] = True

    # arm b: with the fixed measured γ.  Uncovered slices get an explicit
    # zero callable so the null check exercises the SAME code path — exp(0)=1
    # makes the basis (hence the whole solve) byte-identical to the static arm.
    def zero_gamma(psn):
        return np.zeros_like(np.asarray(psn, dtype=np.float64))

    rec_b, _res_b = _solve(grid, table, payload, gamma_fn or zero_gamma)
    out["rotation"] = rec_b
    return out


def run_shot(shot: int, workers: int) -> list[dict]:
    payload = shot_payloads(
        shot, nr=65, nz=97, max_slices=20, min_ip_ka=300.0, split="eval"
    )
    if payload is None:
        logger.warning("shot %s: no payload", shot)
        return []
    _apply_calibration(payload, CALIBRATION)
    sidecar = _shot_passive_sidecar(payload, PASSIVE_K)
    cxrs = load_cxrs(shot)
    state = {
        "grid": payload["grid"],
        "table": payload["table"],
        "payloads": payload["payloads"],
        "sidecar": sidecar,
        "cxrs": cxrs,
        "r0": float(payload["grid"].r0),
        "refs": payload["refs"],
    }
    _init_worker(state)
    idx = list(range(len(payload["payloads"])))
    if workers > 1 and len(idx) > 1:
        ctx = multiprocessing.get_context("fork")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
            results = list(pool.map(analyze_slice, idx))
    else:
        results = [analyze_slice(k) for k in idx]
    refs = payload["refs"]
    for k, r in enumerate(results):
        r["ref"] = refs[k].tolist()
    n_cov = sum(r["covered"] for r in results)
    logger.info(
        "shot %s: %d slices, %d covered (CXRS γ built), cxrs=%s",
        shot,
        len(results),
        n_cov,
        cxrs is not None,
    )
    return results


# ---------------------------------------------------------------------------
# paired statistics
# ---------------------------------------------------------------------------
def _lcfs_offset_cm(target: np.ndarray, ref: np.ndarray) -> float:
    """Median radial LCFS offset [cm] over the 8 fixed angles (target[6:14])."""
    return float(np.nanmedian(np.abs(target[6:14] - ref[6:14]) * 100.0))


def _axis_err_m(target: np.ndarray, ref: np.ndarray) -> float:
    return float(np.hypot(target[0] - ref[0], target[1] - ref[1]))


def _xpt_err(target: np.ndarray, ref: np.ndarray) -> float:
    return float(
        matched_xpoint_error(target[2:6].reshape(2, 2), ref[2:6].reshape(2, 2))
    )


def paired_bootstrap(
    delta: np.ndarray, shot_ids: np.ndarray, n_boot: int = 2000, seed: int = 0
) -> dict:
    """Mean of a per-slice paired ``delta`` with a bootstrap-over-shots CI."""
    delta = np.asarray(delta, dtype=np.float64)
    shot_ids = np.asarray(shot_ids)
    keep = np.isfinite(delta)
    delta, shot_ids = delta[keep], shot_ids[keep]
    if delta.size == 0:
        return {"mean": None, "ci": None, "n": 0, "n_shots": 0}
    shots = np.unique(shot_ids)
    by_shot = {s: delta[shot_ids == s] for s in shots}
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(shots, size=shots.size, replace=True)
        means[b] = np.concatenate([by_shot[s] for s in pick]).mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {
        "mean": float(delta.mean()),
        "median": float(np.median(delta)),
        "ci": [float(lo), float(hi)],
        "n": int(delta.size),
        "n_shots": int(shots.size),
    }


def _headline(model, ref, baseline_vec, names):
    skill = per_quantity_skill(
        model, ref, np.tile(baseline_vec, (len(model), 1)), TARGET_NAMES
    )
    return headline_skill(skill, names)


def paired_skill_delta(
    model_a: np.ndarray,
    model_b: np.ndarray,
    ref: np.ndarray,
    baseline_vec: np.ndarray,
    shot_ids: np.ndarray,
    names: list[str],
    n_boot: int = 2000,
    seed: int = 0,
) -> dict:
    """Set-level skill(arm b) − skill(arm a) with a paired bootstrap over shots
    (same resample drives both arms — the honest paired skill delta)."""
    sa = _headline(model_a, ref, baseline_vec, names)
    sb = _headline(model_b, ref, baseline_vec, names)
    shots = np.unique(shot_ids)
    idx_by_shot = {s: np.flatnonzero(shot_ids == s) for s in shots}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        pick = rng.choice(shots, size=shots.size, replace=True)
        ii = np.concatenate([idx_by_shot[s] for s in pick])
        da = _headline(model_a[ii], ref[ii], baseline_vec, names)
        db = _headline(model_b[ii], ref[ii], baseline_vec, names)
        if np.isfinite(da) and np.isfinite(db):
            draws.append(db - da)
    ci = (
        [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]
        if len(draws) > 50
        else None
    )
    return {
        "skill_static": None if not np.isfinite(sa) else float(sa),
        "skill_rotation": None if not np.isfinite(sb) else float(sb),
        "skill_delta": (
            None if not (np.isfinite(sa) and np.isfinite(sb)) else float(sb - sa)
        ),
        "skill_delta_ci": ci,
    }


# ---------------------------------------------------------------------------
def run(args) -> int:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    _, held = read_split_shot_lists(40, 8)
    held = list(held)
    if args.shots:
        want = {int(s) for s in args.shots.split(",") if s.strip()}
        held = [s for s in held if int(s) in want]
    logger.info("held-out shots: %s", held)

    baseline_vec = train_mean_baseline(40, 10, 300.0)
    logger.info("baseline axis: (%.3f, %.3f)", baseline_vec[0], baseline_vec[1])

    all_slices: list[dict] = []
    for shot in held:
        all_slices.extend(run_shot(int(shot), args.workers))

    # -- null check on uncovered slices: arm b (zero γ) must equal arm a --
    null_max_target_diff = 0.0
    null_max_cost_diff = 0.0
    n_null = 0
    for s in all_slices:
        if s["covered"]:
            continue
        if not (s["static"]["converged"] and s["rotation"]["converged"]):
            continue
        n_null += 1
        ta = np.asarray(s["static"]["target"])
        tb = np.asarray(s["rotation"]["target"])
        null_max_target_diff = max(
            null_max_target_diff, float(np.nanmax(np.abs(tb - ta)))
        )
        null_max_cost_diff = max(
            null_max_cost_diff,
            abs(s["rotation"]["cost"] - s["static"]["cost"]),
        )
    logger.info(
        "null check (uncovered, n=%d): max|Δtarget|=%.3e  max|Δcost|=%.3e",
        n_null,
        null_max_target_diff,
        null_max_cost_diff,
    )

    # -- paired analysis on covered slices where BOTH arms scored --
    cov = [
        s
        for s in all_slices
        if s["covered"] and s["static"]["converged"] and s["rotation"]["converged"]
    ]
    shot_ids = np.array([s["shot"] for s in cov])
    model_a = (
        np.array([s["static"]["target"] for s in cov]) if cov else np.zeros((0, 14))
    )
    model_b = (
        np.array([s["rotation"]["target"] for s in cov]) if cov else np.zeros((0, 14))
    )
    ref = np.array([s["ref"] for s in cov]) if cov else np.zeros((0, 14))

    d_cost = np.array(
        [s["rotation"]["cost"] - s["static"]["cost"] for s in cov], dtype=np.float64
    )
    d_axis = np.array(
        [
            _axis_err_m(model_b[i], ref[i]) - _axis_err_m(model_a[i], ref[i])
            for i in range(len(cov))
        ],
        dtype=np.float64,
    )
    d_lcfs = np.array(
        [
            _lcfs_offset_cm(model_b[i], ref[i]) - _lcfs_offset_cm(model_a[i], ref[i])
            for i in range(len(cov))
        ],
        dtype=np.float64,
    )
    d_xpt = np.array(
        [
            _xpt_err(model_b[i], ref[i]) - _xpt_err(model_a[i], ref[i])
            for i in range(len(cov))
        ],
        dtype=np.float64,
    )
    v_line = np.array([s["gamma"]["v_tor_line_avg_ms"] for s in cov], dtype=np.float64)
    gamma_peak = np.array([s["gamma"]["gamma_peak"] for s in cov], dtype=np.float64)

    report: dict = {
        "arm_config": {
            "n_p": N_P,
            "n_f": N_F,
            "passive_k": PASSIVE_K,
            "conv_limit": CONV_LIMIT,
            "calibration": CALIBRATION,
            "cxrs_systems": list(CXRS_SYSTEMS),
            "v_ceil_ms": V_CEIL_MS,
            "t_floor_ev": T_FLOOR_EV,
            "t_ceil_ev": T_CEIL_EV,
            "rot_time_tol_s": ROT_TIME_TOL_S,
            "gamma_n_bins": GAMMA_N_BINS,
            "gamma_min_points": GAMMA_MIN_POINTS,
            "m_ion": "deuteron",
            "t_proxy": "cxrs carbon temperature (Ti,C ~= Ti,D core caveat)",
        },
        "cohort": {
            "n_shots_heldout": len(held),
            "shots_heldout": held,
            "n_slices_total": len(all_slices),
        },
        "coverage": {
            "n_slices_covered": int(sum(s["covered"] for s in all_slices)),
            "shots_covered": sorted({s["shot"] for s in all_slices if s["covered"]}),
            "n_slices_paired": len(cov),
            "shots_paired": sorted(set(shot_ids.tolist())),
        },
        "null_check": {
            "n_uncovered_paired": n_null,
            "max_abs_target_diff": null_max_target_diff,
            "max_abs_cost_diff": null_max_cost_diff,
            "byte_identical": bool(
                null_max_target_diff == 0.0 and null_max_cost_diff == 0.0
            ),
        },
    }

    if cov:
        report["paired_deltas"] = {
            "whitened_cost": paired_bootstrap(d_cost, shot_ids),
            "axis_error_m": paired_bootstrap(d_axis, shot_ids),
            "lcfs_offset_cm": paired_bootstrap(d_lcfs, shot_ids),
            "xpoint_error_m": paired_bootstrap(d_xpt, shot_ids),
        }
        report["skill_deltas"] = {
            "axis": paired_skill_delta(
                model_a, model_b, ref, baseline_vec, shot_ids, ["axis_R", "axis_Z"]
            ),
            "lcfs": paired_skill_delta(
                model_a,
                model_b,
                ref,
                baseline_vec,
                shot_ids,
                [f"lcfs_r_{k}" for k in range(8)],
            ),
        }
        # rotation strata (median split over covered slices)
        strata = {}
        if v_line.size >= 4 and np.isfinite(v_line).sum() >= 4:
            med = float(np.median(v_line))
            hi = v_line >= med
            lo = v_line < med
            strata = {
                "median_v_line_ms": med,
                "high_cost": paired_bootstrap(d_cost[hi], shot_ids[hi]),
                "low_cost": paired_bootstrap(d_cost[lo], shot_ids[lo]),
                "high_axis_m": paired_bootstrap(d_axis[hi], shot_ids[hi]),
                "low_axis_m": paired_bootstrap(d_axis[lo], shot_ids[lo]),
            }
        report["rotation_strata"] = strata
        report["gamma_summary"] = {
            "gamma_peak_median_m2": float(np.median(gamma_peak)),
            "gamma_peak_max_m2": float(np.max(gamma_peak)),
            "v_line_avg_median_ms": float(np.median(v_line)),
            "v_line_avg_max_ms": float(np.max(v_line)),
            # sanity: γ·(R_out² − R₀²) ~ M₀² should be ≲ 1
            "centrifugal_exponent_median": float(
                np.median(gamma_peak * (1.4**2 - 0.85**2))
            ),
            "centrifugal_exponent_max": float(np.max(gamma_peak * (1.4**2 - 0.85**2))),
        }

    (ARTIFACTS / "measured_rotation_gate.json").write_text(json.dumps(report, indent=2))

    # -- npz arrays --
    npz: dict[str, np.ndarray] = {
        "covered_shot": shot_ids,
        "d_cost": d_cost,
        "d_axis_m": d_axis,
        "d_lcfs_cm": d_lcfs,
        "d_xpt_m": d_xpt,
        "v_line_avg_ms": v_line,
        "gamma_peak_m2": gamma_peak,
    }
    if cov:
        npz["model_static"] = model_a
        npz["model_rotation"] = model_b
        npz["ref"] = ref
        # γ(ψ_N) profiles actually used, on a common grid
        common = np.linspace(0.0, 1.0, 40)
        prof = np.full((len(cov), common.size), np.nan)
        for i, s in enumerate(cov):
            g = s["gamma"]
            c = np.asarray(g["gamma_bin_centres"])
            gv = np.asarray(g["gamma_bin_values"])
            if c.size >= 2:
                inside = (common >= c[0]) & (common <= c[-1])
                prof[i, inside] = np.interp(common[inside], c, gv)
                prof[i, ~inside] = 0.0
        npz["gamma_psi_n_grid"] = common
        npz["gamma_profiles"] = prof
    np.savez(ARTIFACTS / "measured_rotation_gate_arrays.npz", **npz)
    logger.info("wrote artifact + arrays")

    _make_figures(report, npz, cov)
    _print_verdict(report)
    return 0


def _make_figures(report: dict, npz: dict, cov: list[dict]) -> None:
    import matplotlib.pyplot as plt

    shot_ids = npz["covered_shot"]
    shots = sorted(set(shot_ids.tolist()))
    colours = dict(
        zip(shots, plt.cm.tab10(np.linspace(0, 1, max(len(shots), 1))), strict=False)
    )

    # ---- figure 1: paired per-slice deltas vs rotation ----
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8))
    pd = report.get("paired_deltas", {})
    panels = [
        ("d_cost", "whitened_cost", "Δ whitened magnetics cost", axes[0]),
        ("d_axis_m", "axis_error_m", "Δ axis error [m]", axes[1]),
        ("d_lcfs_cm", "lcfs_offset_cm", "Δ LCFS offset [cm]", axes[2]),
    ]
    v = npz["v_line_avg_ms"] / 1e3
    for key, pkey, ylabel, ax in panels:
        d = npz[key]
        for s in shots:
            m = shot_ids == s
            ax.scatter(v[m], d[m], s=28, color=colours[s], label=str(s), alpha=0.8)
        ax.axhline(0.0, color="k", lw=0.8, ls=":")
        stat = pd.get(pkey, {})
        if stat.get("mean") is not None:
            ax.axhline(stat["mean"], color="#b2182b", lw=1.5)
            if stat.get("ci"):
                ax.axhspan(stat["ci"][0], stat["ci"][1], color="#b2182b", alpha=0.12)
            ax.set_title(
                f"{ylabel}\nmean={stat['mean']:.3g} "
                f"CI[{stat['ci'][0]:.3g},{stat['ci'][1]:.3g}] "
                f"(n={stat['n']}, {stat['n_shots']} shots)",
                fontsize=9,
            )
        ax.set_xlabel("line-avg |v_tor| [km/s]")
        ax.set_ylabel(ylabel)
    axes[0].legend(fontsize=7, title="shot", loc="best")
    fig.suptitle(
        "Measured-rotation forward test: arm(rotation) − arm(static) paired deltas "
        "(negative = improvement)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = FIGURES / "fig-measured-rotation-deltas.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    logger.info("wrote %s", out)

    # ---- figure 2: the γ(ψ_N) profiles actually used ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 5.0))
    if "gamma_profiles" in npz:
        grid = npz["gamma_psi_n_grid"]
        prof = npz["gamma_profiles"]
        v = npz["v_line_avg_ms"] / 1e3
        vmax = float(np.nanpercentile(v, 95)) if v.size else 1.0
        for i in range(prof.shape[0]):
            col = plt.cm.viridis(min(v[i] / max(vmax, 1.0), 1.0))
            ax1.plot(grid, prof[i], color=col, alpha=0.6, lw=1.0)
        ax1.set_xlabel("ψ_N")
        ax1.set_ylabel("γ(ψ_N)  = m_D Ω²/(2 T_i)   [m⁻²]")
        ax1.set_title("Measured γ(ψ_N) profiles (per covered slice)")
        sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0.0, vmax))
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax1)
        cb.set_label("line-avg |v_tor| [km/s]")
        # centrifugal exponent γ·(R²−R₀²) at the outboard edge — the M₀² sanity
        r_out2, r0sq = 1.4**2, 0.85**2
        ax2.plot(grid, prof.T * (r_out2 - r0sq), color="0.6", alpha=0.5, lw=0.8)
        ax2.axhline(
            1.0, color="#b2182b", lw=1.0, ls="--", label="M₀²=1 (sanity ceiling)"
        )
        ax2.set_xlabel("ψ_N")
        ax2.set_ylabel("γ·(R_out² − R₀²)  ≈ M₀²  (outboard edge)")
        ax2.set_title("Centrifugal exponent at the outboard edge")
        ax2.legend(fontsize=8)
    fig.suptitle(
        "Fixed measured centrifugal factor built from CXRS carbon v_tor + T_i "
        "(ss+sw beams)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = FIGURES / "fig-measured-rotation-gamma.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    logger.info("wrote %s", out)


def _print_verdict(report: dict) -> None:
    logger.info("=" * 72)
    logger.info("MEASURED-ROTATION FORWARD TEST — VERDICT")
    cv = report["coverage"]
    logger.info(
        "coverage: %d covered slices on shots %s; %d paired",
        cv["n_slices_covered"],
        cv["shots_covered"],
        cv["n_slices_paired"],
    )
    nc = report["null_check"]
    logger.info(
        "null check (uncovered n=%d): byte_identical=%s (max|Δtarget|=%.2e)",
        nc["n_uncovered_paired"],
        nc["byte_identical"],
        nc["max_abs_target_diff"],
    )
    if "paired_deltas" in report:
        for k, d in report["paired_deltas"].items():
            logger.info(
                "  Δ%-16s mean=%s CI=%s (n=%d, %d shots)",
                k,
                None if d["mean"] is None else f"{d['mean']:.4g}",
                d["ci"],
                d["n"],
                d["n_shots"],
            )
        for k, d in report["skill_deltas"].items():
            logger.info(
                "  skill[%s] static=%.4g rotation=%.4g Δ=%.4g CI=%s",
                k,
                d["skill_static"] or float("nan"),
                d["skill_rotation"] or float("nan"),
                d["skill_delta"] or float("nan"),
                d["skill_delta_ci"],
            )
        gs = report["gamma_summary"]
        logger.info(
            "  γ_peak median=%.4g max=%.4g m⁻²; centrifugal exponent (M₀²) "
            "median=%.3f max=%.3f",
            gs["gamma_peak_median_m2"],
            gs["gamma_peak_max_m2"],
            gs["centrifugal_exponent_median"],
            gs["centrifugal_exponent_max"],
        )
    logger.info("=" * 72)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--shots", type=str, default="")
    args = ap.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
