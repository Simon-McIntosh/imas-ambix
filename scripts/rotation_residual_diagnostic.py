#!/usr/bin/env python
"""Is the toroidal-rotation R⁴ column visible in our held-out GS residuals?

Solver-free re-analysis of the closure arm's converged current distributions.
For each held-out slice the profile-parametrised free-boundary GS fixed point is
re-solved on the calibrated raw magnetics (the recorded winner config:
``n_p=n_f=1``, rank-8 passive sidecar) in TWO arms — the free-sign winner and
the sign-constrained arm whose within-surface residual the force-balance study
found structured — and the per-flux-bin structure residual is evaluated with
BOTH the two-column affine relation ``R·jφ = a·R² + b`` and its
rigid-rotation extension ``R·jφ = a·R² + b + c·R⁴``.  The R⁴ column is the
exact leading-order centrifugal signature, ``c(ψ) ≈ ½ p₀ (m_iΩ²/T)′``.

The question this answers: does adding the R⁴ column measurably reduce the
unexplained current-weighted variance, is the recovered ``c(ψ)`` coherent with
the physics (one-signed where large, correlated with the measured CXRS
rotation), and does the sign-constrained arm's larger residual get absorbed by
it?  A clean negative — "rotation is not visible at the current error budget" —
is a first-class result.

Rotation stratification uses the RAW CXRS toroidal velocity in the ``act`` zarr
group (firewall-clean: a measured input, never a solver output), with honest
coverage reporting (older campaigns carry no CXRS).

Artifacts:  imas_ambix/latent/artifacts/patch_gate/rotation_residual_diagnostic.json
            imas_ambix/latent/artifacts/patch_gate/rotation_residual_diagnostic_arrays.npz
Figures:    docs/figures/force-balance-spine/fig-rotation-residual-delta.png
            docs/figures/force-balance-spine/fig-rotation-residual-cpsi.png
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
import torch

from imas_ambix.latent.data import read_split_shot_lists
from imas_ambix.latent.gs_solve import fit_profile_ladder
from imas_ambix.latent.structure_residual import (
    _binned_weights,
    _design,
    _solve_bins,
    fit_flux_functions,
    structure_residual,
)

# import-only reuse of the hardened harness (same cohort, calibration, sidecar)
from scripts.closure_gate_eval import _apply_calibration, _shot_passive_sidecar
from scripts.patch_gate_eval import shot_payloads

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("rotation_residual_diagnostic")

ARTIFACTS = Path("imas_ambix/latent/artifacts/patch_gate")
FIGURES = Path("docs/figures/force-balance-spine")
CALIBRATION = str(ARTIFACTS / "static_calibration_offset_only.json")

# frozen structure-residual knobs (module defaults) + the winner closure config
N_BINS = 24
BANDWIDTH_BINS = 1.0
CONNECTIVITY = "locality"  # honest topology-aware form (needs z_c)
CONV_LIMIT = 5e-3  # the gate's scoring-inclusion criterion
PASSIVE_K = 8
ARMS = (("free", False), ("nonneg", True))

# physical toroidal-rotation ceiling: MAST NBI drives up to ~250 km/s; values
# far above this in the raw CXRS fit output are bad fits / fill values
V_CEIL_MS = 3.0e5
V_ERR_CEIL_MS = 1.0e5
ROT_TIME_TOL_S = 0.015  # nearest act frame within this window of the slice time


# ---------------------------------------------------------------------------
# rotation proxy (raw CXRS, firewall-clean)
# ---------------------------------------------------------------------------
def load_rotation_proxy(shot: int) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Per-act-frame (time, line-average |v_tor|, peak |v_tor|) in m/s.

    Combines the physical-range chords of every available CXRS system
    (``pla`` / ``ss`` / ``sw``) at each frame; a frame needs ≥ 2 physical
    chords to yield a proxy.  Returns None when the shot has no usable ``act``.
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
    if "time" not in grp:
        return None
    t = np.asarray(grp["time"], dtype=np.float64)
    if t.size < 2:
        return None
    per_frame: list[list[float]] = [[] for _ in range(t.size)]
    found = False
    for sysname in ("pla", "ss", "sw"):
        vk, ek = f"{sysname}_velocity", f"{sysname}_velocity_error"
        if vk not in grp:
            continue
        v = np.asarray(grp[vk], dtype=np.float64)
        if v.ndim != 2 or v.shape[0] != t.size:
            continue
        e = (
            np.asarray(grp[ek], dtype=np.float64)
            if ek in grp
            else np.full_like(v, np.nan)
        )
        phys = (
            np.isfinite(v)
            & (np.abs(v) < V_CEIL_MS)
            & np.isfinite(e)
            & (e > 0)
            & (e < V_ERR_CEIL_MS)
        )
        if not phys.any():
            continue
        found = True
        for it in range(t.size):
            vals = np.abs(v[it][phys[it]])
            if vals.size:
                per_frame[it].extend(vals.tolist())
    if not found:
        return None
    line_avg = np.full(t.size, np.nan)
    peak = np.full(t.size, np.nan)
    for it in range(t.size):
        if len(per_frame[it]) >= 2:
            arr = np.asarray(per_frame[it])
            line_avg[it] = float(arr.mean())
            peak[it] = float(arr.max())
    if not np.isfinite(line_avg).any():
        return None
    return t, line_avg, peak


def slice_rotation(
    rot: tuple[np.ndarray, np.ndarray, np.ndarray] | None, time_s: float
) -> tuple[float, float]:
    """Nearest-frame (line-avg, peak) |v_tor| for a slice time; NaN if none in range."""
    if rot is None:
        return float("nan"), float("nan")
    t, la, pk = rot
    good = np.isfinite(la)
    if not good.any():
        return float("nan"), float("nan")
    tg, lag, pkg = t[good], la[good], pk[good]
    k = int(np.argmin(np.abs(tg - time_s)))
    if abs(float(tg[k]) - time_s) > ROT_TIME_TOL_S:
        return float("nan"), float("nan")
    return float(lag[k]), float(pkg[k])


# ---------------------------------------------------------------------------
# per-bin unexplained fraction (the structure_residual machinery, per bin)
# ---------------------------------------------------------------------------
def per_bin_unexplained(
    psi_c: torch.Tensor,
    r_c: torch.Tensor,
    z_c: torch.Tensor,
    jphi_c: torch.Tensor,
    form: str,
    bin_grid,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (mu, per-bin unexplained fraction, per-bin weight mass).

    Same weighted-LSQ as :func:`structure_residual` but the numerator/
    denominator are kept per ψ-bin instead of summed, so the R⁴ improvement can
    be localised in ψ.  ``bin_grid`` is shared across forms so the 2-col/3-col
    comparison uses identical bins.
    """
    w, mu = _binned_weights(
        psi_c,
        r_c,
        jphi_c,
        n_bins=N_BINS,
        bandwidth_bins=BANDWIDTH_BINS,
        z_c=z_c,
        connectivity=CONNECTIVITY,
        locality_scale=None,
        component_labels=None,
        bin_grid=bin_grid,
    )
    if w is None:
        z = np.full(N_BINS, np.nan)
        return z, z, np.zeros(N_BINS)
    x, y = _design(form, r_c, jphi_c)
    _, fit = _solve_bins(w, x, y)
    num = (w * (y[None, :] - fit) ** 2).sum(-1)
    den = (w * (y[None, :] ** 2)).sum(-1).clamp_min(1e-30)
    return (
        mu.detach().cpu().numpy(),
        (num / den).detach().cpu().numpy(),
        w.sum(-1).detach().cpu().numpy(),
    )


def _bin_grid_for(psi_c: torch.Tensor, jphi_c: torch.Tensor):
    """The detached ψ-bin centres/bandwidth (form-independent) to freeze bins."""
    from imas_ambix.latent.structure_residual import _bin_grid  # noqa: PLC0415

    w_amp = jphi_c * jphi_c
    total = w_amp.sum()
    if float(total) <= 0.0:
        return None
    return _bin_grid(psi_c, w_amp / total, N_BINS, BANDWIDTH_BINS)


# ---------------------------------------------------------------------------
# per-slice analysis (both arms) — runs in a fork worker (grid inherited COW)
# ---------------------------------------------------------------------------
_WORKER: dict = {}


def _init_worker(state: dict) -> None:
    _WORKER.update(state)


def analyze_slice(k: int) -> dict:
    grid = _WORKER["grid"]
    table = _WORKER["table"]
    payload = _WORKER["payloads"][k]
    sidecar = _WORKER["sidecar"]
    cells = grid.cells
    r_np = grid.flat_r[cells]
    z_np = grid.flat_z[cells]
    r_c = torch.as_tensor(r_np, dtype=torch.float64)
    z_c = torch.as_tensor(z_np, dtype=torch.float64)

    out: dict = {
        "shot": int(payload.shot),
        "t_index": int(payload.t_index),
        "time_s": float(payload.time_s),
        "ip_amperes": float(payload.ip_amperes),
    }
    for arm_name, nonneg in ARMS:
        lf = fit_profile_ladder(
            grid,
            table,
            i_pf=payload.i_pf,
            ip_amperes=payload.ip_amperes,
            measured=payload.measured,
            vacuum_prediction=payload.vacuum,
            sensor_scale=payload.scale,
            sensor_mask=payload.mask,
            n_p=1,
            n_f=1,
            nonneg=nonneg,
            passive=sidecar,
            passive_ridge=1.0,
        )
        res = lf.result
        converged = bool(res.converged or res.residual <= CONV_LIMIT)
        rec: dict = {
            "converged": converged,
            "residual": float(res.residual),
            "cost": float(lf.cost),
            "axis_psi": float(res.axis_psi),
            "boundary_psi": float(res.boundary_psi),
        }
        if converged:
            psi_c = torch.as_tensor(res.psi.ravel()[cells], dtype=torch.float64)
            jphi_c = torch.as_tensor(res.jphi.ravel()[cells], dtype=torch.float64)
            bin_grid = _bin_grid_for(psi_c, jphi_c)
            r2 = structure_residual(
                psi_c,
                r_c,
                jphi_c,
                form="affine-r2",
                z_c=z_c,
                connectivity=CONNECTIVITY,
                bin_grid=bin_grid,
            )
            r4 = structure_residual(
                psi_c,
                r_c,
                jphi_c,
                form="affine-r2-rotation",
                z_c=z_c,
                connectivity=CONNECTIVITY,
                bin_grid=bin_grid,
            )
            mu, frac2, mass = per_bin_unexplained(
                psi_c, r_c, z_c, jphi_c, "affine-r2", bin_grid
            )
            _, frac3, _ = per_bin_unexplained(
                psi_c, r_c, z_c, jphi_c, "affine-r2-rotation", bin_grid
            )
            cf = fit_flux_functions(
                psi_c,
                r_c,
                jphi_c,
                n_bins=N_BINS,
                bandwidth_bins=BANDWIDTH_BINS,
                form="affine-r2-rotation",
                z_c=z_c,
                connectivity=CONNECTIVITY,
            )
            span = res.boundary_psi - res.axis_psi
            span = span if abs(span) > 1e-12 else 1e-12
            psi_n = (mu - res.axis_psi) / span
            rec |= {
                "resid_2col": float(r2),
                "resid_3col": float(r4),
                "delta": float(r2) - float(r4),
                "psi_n": psi_n.tolist(),
                "frac2": frac2.tolist(),
                "frac3": frac3.tolist(),
                "mass": mass.tolist(),
                "c_k": np.asarray(cf.c_k).tolist(),
                "c_err": np.asarray(cf.c_err).tolist(),
                "a_k": np.asarray(cf.a_k).tolist(),
            }
        out[arm_name] = rec
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
    rot = load_rotation_proxy(shot)
    state = {
        "grid": payload["grid"],
        "table": payload["table"],
        "payloads": payload["payloads"],
        "sidecar": sidecar,
    }
    _init_worker(state)
    idx = list(range(len(payload["payloads"])))
    if workers > 1 and len(idx) > 1:
        ctx = multiprocessing.get_context("fork")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
            results = list(pool.map(analyze_slice, idx))
    else:
        results = [analyze_slice(k) for k in idx]
    for r in results:
        la, pk = slice_rotation(rot, r["time_s"])
        r["rot_line_avg_ms"] = la
        r["rot_peak_ms"] = pk
        r["rot_available"] = bool(np.isfinite(la))
    n_free = sum(r["free"]["converged"] for r in results)
    n_rot = sum(r["rot_available"] for r in results)
    logger.info(
        "shot %s: %d slices, %d free-arm converged, %d with rotation proxy",
        shot,
        len(results),
        n_free,
        n_rot,
    )
    return results


# ---------------------------------------------------------------------------
# aggregation + statistics
# ---------------------------------------------------------------------------
def bootstrap_over_shots(
    values: np.ndarray, shot_ids: np.ndarray, n_boot: int = 2000, seed: int = 0
) -> dict:
    """Mean of ``values`` with a bootstrap-over-shots 95% CI (resample shots,
    pool their slices, take the pooled mean)."""
    values = np.asarray(values, dtype=np.float64)
    shot_ids = np.asarray(shot_ids)
    keep = np.isfinite(values)
    values, shot_ids = values[keep], shot_ids[keep]
    if values.size == 0:
        return {"mean": None, "ci": None, "n": 0, "n_shots": 0}
    shots = np.unique(shot_ids)
    by_shot = {s: values[shot_ids == s] for s in shots}
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(shots, size=shots.size, replace=True)
        pooled = np.concatenate([by_shot[s] for s in pick])
        means[b] = pooled.mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {
        "mean": float(values.mean()),
        "ci": [float(lo), float(hi)],
        "n": int(values.size),
        "n_shots": int(shots.size),
    }


def spearman_with_ci(
    x: np.ndarray,
    y: np.ndarray,
    shot_ids: np.ndarray,
    n_boot: int = 2000,
    seed: int = 1,
) -> dict:
    """Spearman ρ(x, y) with a bootstrap-over-shots 95% CI."""
    from scipy.stats import spearmanr  # noqa: PLC0415

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    shot_ids = np.asarray(shot_ids)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y, shot_ids = x[keep], y[keep], shot_ids[keep]
    if x.size < 4 or np.unique(shot_ids).size < 2:
        return {"rho": None, "ci": None, "n": int(x.size)}
    rho = float(spearmanr(x, y).statistic)
    shots = np.unique(shot_ids)
    idx_by_shot = {s: np.flatnonzero(shot_ids == s) for s in shots}
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        pick = rng.choice(shots, size=shots.size, replace=True)
        ii = np.concatenate([idx_by_shot[s] for s in pick])
        if np.unique(shot_ids[ii]).size < 2 or ii.size < 4:
            continue
        r = spearmanr(x[ii], y[ii]).statistic
        if np.isfinite(r):
            boot.append(r)
    ci = (
        [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
        if len(boot) > 50
        else None
    )
    return {"rho": rho, "ci": ci, "n": int(x.size), "n_shots": int(shots.size)}


def resample_to_grid(
    psi_n: np.ndarray, vals: np.ndarray, mass: np.ndarray, grid: np.ndarray
) -> np.ndarray:
    """Interpolate a slice's per-bin ``vals(psi_n)`` onto a common ψ_N grid,
    using only well-populated bins (mass > 1e-3·max)."""
    keep = np.isfinite(psi_n) & np.isfinite(vals) & (mass > 1e-3 * (mass.max() or 1.0))
    if keep.sum() < 2:
        return np.full(grid.size, np.nan)
    order = np.argsort(psi_n[keep])
    xp, fp = psi_n[keep][order], vals[keep][order]
    out = np.interp(grid, xp, fp, left=np.nan, right=np.nan)
    return out


def slice_c_summary(rec: dict) -> tuple[float, float, float]:
    """Mass-weighted mean |c|, peak |c|, and dominant-sign fraction for a slice."""
    c = np.asarray(rec["c_k"], dtype=np.float64)
    mass = np.asarray(rec["mass"], dtype=np.float64)
    keep = np.isfinite(c) & (mass > 1e-3 * (mass.max() or 1.0))
    if keep.sum() < 1:
        return float("nan"), float("nan"), float("nan")
    cc, mm = c[keep], mass[keep]
    w = mm / mm.sum()
    mean_abs = float((w * np.abs(cc)).sum())
    peak_abs = float(np.abs(cc).max())
    # sign consistency among the high-|c| bins (top-half by |c|·mass)
    imp = np.abs(cc) * mm
    hi = imp >= np.median(imp)
    signs = np.sign(cc[hi])
    dom = (
        float(max((signs > 0).mean(), (signs < 0).mean())) if hi.any() else float("nan")
    )
    return mean_abs, peak_abs, dom


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--shots", type=str, default="")
    args = ap.parse_args()

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    _, held = read_split_shot_lists(40, 8)
    held = list(held)
    if args.shots:
        want = {int(s) for s in args.shots.split(",") if s.strip()}
        held = [s for s in held if int(s) in want]
    logger.info("held-out shots: %s", held)

    all_slices: list[dict] = []
    for shot in held:
        all_slices.extend(run_shot(int(shot), args.workers))

    # -- assemble per-arm slice tables --
    common = np.linspace(0.05, 0.95, 12)
    report: dict = {
        "cohort": {"n_shots": len(held), "shots": held, "n_slices": len(all_slices)},
        "config": {
            "n_p": 1,
            "n_f": 1,
            "passive_k": PASSIVE_K,
            "connectivity": CONNECTIVITY,
            "n_bins": N_BINS,
            "bandwidth_bins": BANDWIDTH_BINS,
            "conv_limit": CONV_LIMIT,
            "calibration": CALIBRATION,
            "v_ceil_ms": V_CEIL_MS,
        },
        "rotation_coverage": {},
        "arms": {},
    }

    # rotation coverage
    rot_avail = np.array([s["rot_available"] for s in all_slices])
    shots_with_rot = sorted({s["shot"] for s in all_slices if s["rot_available"]})
    report["rotation_coverage"] = {
        "n_slices_with_proxy": int(rot_avail.sum()),
        "n_slices_total": len(all_slices),
        "shots_with_proxy": shots_with_rot,
        "shots_without_proxy": sorted(set(held) - set(shots_with_rot)),
    }

    npz: dict[str, np.ndarray] = {}
    fig_data: dict[str, dict] = {}
    for arm_name, _ in ARMS:
        conv = np.array([s[arm_name]["converged"] for s in all_slices])
        idx = np.flatnonzero(conv)
        sl = [all_slices[i] for i in idx]
        shot_ids = np.array([s["shot"] for s in sl])
        r2 = np.array([s[arm_name]["resid_2col"] for s in sl])
        r3 = np.array([s[arm_name]["resid_3col"] for s in sl])
        delta = r2 - r3
        rot_la = np.array([s["rot_line_avg_ms"] for s in sl])
        rot_pk = np.array([s["rot_peak_ms"] for s in sl])
        has_rot = np.isfinite(rot_la)

        # rotation strata (median split over the covered slices)
        strata: dict = {}
        if has_rot.sum() >= 4:
            med = float(np.median(rot_la[has_rot]))
            hi = has_rot & (rot_la >= med)
            lo = has_rot & (rot_la < med)
            strata = {
                "median_line_avg_ms": med,
                "high": bootstrap_over_shots(delta[hi], shot_ids[hi]),
                "low": bootstrap_over_shots(delta[lo], shot_ids[lo]),
            }

        # c(psi) coherence
        c_mean_abs, c_peak_abs, c_dom = (
            np.array([slice_c_summary(s[arm_name]) for s in sl]).T
            if sl
            else (np.array([]), np.array([]), np.array([]))
        )
        spearman_lineavg = spearman_with_ci(c_mean_abs, rot_la, shot_ids)
        spearman_peak = spearman_with_ci(c_peak_abs, rot_pk, shot_ids)

        report["arms"][arm_name] = {
            "n_converged": int(conv.sum()),
            "n_candidate": len(all_slices),
            "resid_2col_median": float(np.median(r2)) if r2.size else None,
            "resid_3col_median": float(np.median(r3)) if r3.size else None,
            "delta_overall": bootstrap_over_shots(delta, shot_ids),
            "delta_relative_median": (
                float(np.median(delta / np.clip(r2, 1e-9, None))) if r2.size else None
            ),
            "delta_by_rotation_stratum": strata,
            "c_one_signed_high_bins_median": (
                float(np.nanmedian(c_dom)) if c_dom.size else None
            ),
            "c_abs_vs_rotation_lineavg_spearman": spearman_lineavg,
            "c_peak_vs_rotation_peak_spearman": spearman_peak,
        }

        # per-bin resampled curves for the figure (+ bootstrap CI over shots)
        f2 = np.array(
            [
                resample_to_grid(
                    np.asarray(s[arm_name]["psi_n"]),
                    np.asarray(s[arm_name]["frac2"]),
                    np.asarray(s[arm_name]["mass"]),
                    common,
                )
                for s in sl
            ]
        )
        f3 = np.array(
            [
                resample_to_grid(
                    np.asarray(s[arm_name]["psi_n"]),
                    np.asarray(s[arm_name]["frac3"]),
                    np.asarray(s[arm_name]["mass"]),
                    common,
                )
                for s in sl
            ]
        )
        fig_data[arm_name] = {
            "shot_ids": shot_ids,
            "f2": f2,
            "f3": f3,
            "rot_la": rot_la,
            "has_rot": has_rot,
            "delta": delta,
            "sl": sl,
        }
        # npz arrays
        npz[f"{arm_name}_shot"] = shot_ids
        npz[f"{arm_name}_resid_2col"] = r2
        npz[f"{arm_name}_resid_3col"] = r3
        npz[f"{arm_name}_delta"] = delta
        npz[f"{arm_name}_rot_line_avg_ms"] = rot_la
        npz[f"{arm_name}_rot_peak_ms"] = rot_pk
        npz[f"{arm_name}_c_mean_abs"] = np.asarray(c_mean_abs, dtype=np.float64)
        npz[f"{arm_name}_c_peak_abs"] = np.asarray(c_peak_abs, dtype=np.float64)
        if f2.size:
            npz[f"{arm_name}_perbin_psi_n_grid"] = common
            npz[f"{arm_name}_perbin_frac2"] = f2
            npz[f"{arm_name}_perbin_frac3"] = f3

    # W1 connection: how much of the (larger) sign-constrained 2-col residual
    # does the R⁴ column absorb, vs the free-sign arm?
    for arm_name, _ in ARMS:
        a = report["arms"][arm_name]
        d = a["delta_overall"]["mean"]
        r2m = a["resid_2col_median"]
        a["r4_absorbed_fraction_of_2col"] = (
            float(d / r2m) if (d is not None and r2m) else None
        )

    (ARTIFACTS / "rotation_residual_diagnostic.json").write_text(
        json.dumps(report, indent=2)
    )
    np.savez(ARTIFACTS / "rotation_residual_diagnostic_arrays.npz", **npz)
    logger.info("wrote artifact + arrays")

    _make_figures(fig_data, common, report)
    _print_verdict(report)
    return 0


def _make_figures(fig_data: dict, common: np.ndarray, report: dict) -> None:
    import matplotlib.pyplot as plt

    # ---- figure 1: per-bin unexplained fraction, 2-col vs 3-col ----
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.0))
    arm_titles = {"free": "free-sign (winner)", "nonneg": "sign-constrained"}
    for row, (arm_name, _) in enumerate(ARMS):
        fd = fig_data[arm_name]
        f2, f3 = fd["f2"], fd["f3"]
        shot_ids = fd["shot_ids"]
        ax = axes[row, 0]
        if f2.size:
            _band(ax, common, f2, shot_ids, "#2166ac", "2-col  R·jφ=aR²+b")
            _band(ax, common, f3, shot_ids, "#d95f02", "3-col  +cR⁴")
        ax.set_title(f"{arm_titles[arm_name]}: per-bin unexplained fraction")
        ax.set_xlabel("ψ_N")
        ax.set_ylabel("unexplained current-weighted fraction")
        ax.legend(fontsize=8)
        ax.set_ylim(bottom=0.0)

        ax2 = axes[row, 1]
        has_rot = fd["has_rot"]
        if has_rot.sum() >= 4 and f2.size:
            med = float(np.median(fd["rot_la"][has_rot]))
            hi = has_rot & (fd["rot_la"] >= med)
            lo = has_rot & (fd["rot_la"] < med)
            d_hi = (f2 - f3)[hi]
            d_lo = (f2 - f3)[lo]
            if d_hi.size:
                ax2.plot(
                    common,
                    np.nanmean(d_hi, axis=0),
                    color="#b2182b",
                    lw=2,
                    label=f"high rot (≥{med / 1e3:.0f} km/s, n={hi.sum()})",
                )
            if d_lo.size:
                ax2.plot(
                    common,
                    np.nanmean(d_lo, axis=0),
                    color="#4393c3",
                    lw=2,
                    label=f"low rot (n={lo.sum()})",
                )
            ax2.axhline(0.0, color="k", lw=0.6, ls=":")
            ax2.legend(fontsize=8)
        else:
            ax2.text(
                0.5,
                0.5,
                "insufficient rotation coverage\nfor a stratified split",
                ha="center",
                va="center",
                transform=ax2.transAxes,
                fontsize=10,
            )
        ax2.set_title(f"{arm_titles[arm_name]}: R⁴ per-bin Δ by rotation stratum")
        ax2.set_xlabel("ψ_N")
        ax2.set_ylabel("Δ = frac(2-col) − frac(3-col)")
    fig.suptitle(
        "Rotation R⁴ column: does it reduce the unexplained GS structure residual?",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = FIGURES / "fig-rotation-residual-delta.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    logger.info("wrote %s", out)

    # ---- figure 2: recovered c(ψ_N) spaghetti coloured by rotation ----
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2))
    rot_all = np.concatenate(
        [fig_data[a]["rot_la"][np.isfinite(fig_data[a]["rot_la"])] for a, _ in ARMS]
        or [np.array([np.nan])]
    )
    vmax = float(np.nanpercentile(rot_all, 95)) if np.isfinite(rot_all).any() else 1.0
    for col, (arm_name, _) in enumerate(ARMS):
        ax = axes[col]
        fd = fig_data[arm_name]
        sm = None
        for s in fd["sl"]:
            rec = s[arm_name]
            psi_n = np.asarray(rec["psi_n"], dtype=np.float64)
            c = np.asarray(rec["c_k"], dtype=np.float64)
            mass = np.asarray(rec["mass"], dtype=np.float64)
            keep = (
                np.isfinite(psi_n)
                & np.isfinite(c)
                & (mass > 1e-3 * (mass.max() or 1.0))
            )
            if keep.sum() < 2:
                continue
            order = np.argsort(psi_n[keep])
            xp, cp = psi_n[keep][order], c[keep][order]
            la = s["rot_line_avg_ms"]
            if np.isfinite(la):
                color = plt.cm.viridis(min(la / max(vmax, 1.0), 1.0))
                ax.plot(xp, cp, color=color, alpha=0.7, lw=1.0)
                sm = True
            else:
                ax.plot(xp, cp, color="0.75", alpha=0.4, lw=0.8)
        ax.axhline(0.0, color="k", lw=0.6, ls=":")
        ax.set_title(f"{arm_titles[arm_name]}: recovered c(ψ_N)  [R⁴ coefficient]")
        ax.set_xlabel("ψ_N")
        ax.set_ylabel("c(ψ_N)  ≈ ½ p₀ (m_iΩ²/T)′")
        if sm:
            mappable = plt.cm.ScalarMappable(
                cmap="viridis", norm=plt.Normalize(0.0, vmax / 1e3)
            )
            mappable.set_array([])
            cb = fig.colorbar(mappable, ax=ax)
            cb.set_label("line-avg |v_tor| [km/s]")
    fig.suptitle(
        "Recovered centrifugal R⁴ coefficient c(ψ_N), coloured by measured CXRS rotation",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = FIGURES / "fig-rotation-residual-cpsi.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    logger.info("wrote %s", out)


def _band(ax, x, arr, shot_ids, color, label):
    """Mean line + bootstrap-over-shots 95% band for a (n_slice, n_x) array."""
    mean = np.nanmean(arr, axis=0)
    shots = np.unique(shot_ids)
    by_shot = {s: arr[shot_ids == s] for s in shots}
    rng = np.random.default_rng(0)
    boot = np.empty((400, x.size))
    for b in range(400):
        pick = rng.choice(shots, size=shots.size, replace=True)
        pooled = np.concatenate([by_shot[s] for s in pick], axis=0)
        boot[b] = np.nanmean(pooled, axis=0)
    lo = np.nanpercentile(boot, 2.5, axis=0)
    hi = np.nanpercentile(boot, 97.5, axis=0)
    ax.plot(x, mean, color=color, lw=2, label=label)
    ax.fill_between(x, lo, hi, color=color, alpha=0.18)


def _print_verdict(report: dict) -> None:
    logger.info("=" * 72)
    logger.info("ROTATION RESIDUAL DIAGNOSTIC — VERDICT")
    rc = report["rotation_coverage"]
    logger.info(
        "rotation coverage: %d/%d slices, shots with proxy=%s",
        rc["n_slices_with_proxy"],
        rc["n_slices_total"],
        rc["shots_with_proxy"],
    )
    for arm in ("free", "nonneg"):
        a = report["arms"][arm]
        d = a["delta_overall"]
        logger.info(
            "[%s] n=%d  2col_med=%.4f 3col_med=%.4f  Δ=%.5f CI=%s  absorbed=%.1f%%",
            arm,
            a["n_converged"],
            a["resid_2col_median"] or float("nan"),
            a["resid_3col_median"] or float("nan"),
            d["mean"] or float("nan"),
            d["ci"],
            100.0 * (a["r4_absorbed_fraction_of_2col"] or 0.0),
        )
        sp = a["c_abs_vs_rotation_lineavg_spearman"]
        logger.info(
            "     c-coherence: one-signed-high-bins=%s  Spearman(|c|,rot)=%s CI=%s",
            a["c_one_signed_high_bins_median"],
            sp["rho"],
            sp.get("ci"),
        )
    logger.info("=" * 72)


if __name__ == "__main__":
    raise SystemExit(main())
