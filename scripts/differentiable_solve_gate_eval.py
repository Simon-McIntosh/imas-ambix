"""Gate D-A for the continuous-topology (smooth-map) free-boundary solve.

The hard per-sweep topology read (critical points + labelled core mask) makes
the free-boundary fixed-point map non-differentiable: on under-determined
slices the residual limit-cycles ~1e-3 as the discrete core mask flips, and no
per-solve accelerator can help.  The opt-in ``topology_read='connectivity'``
solve swaps in the continuous read (connectivity binding + sub-grid stencil
axis + smooth core-membership weight).  This gate records the two pre-declared
D-A verdicts on held-out slices:

  D-A1 — reproduction: the smooth-map solve reproduces the hard-map converged
         equilibria on the well-posed population (both arms converged) within
         the grid-free substrate tolerances: median axis ≤ 2.0 cm, median LCFS
         radius ≤ 3.0 cm, median jφ(ρ̂) profile RMS ≤ 0.10.  The smooth read
         must not be lossy.
  D-A2 — smoothness: on the slices where the HARD solve limit-cycles (residual
         above tolerance at the sweep cap — the under-determined class), the
         smooth-map residual under plain relaxed Picard decreases monotonically
         (tail monotone fraction ≥ 0.9, or the solve converges outright) —
         the discrete mask-flip signature is gone.

Both arms run the grid-free ``greens-matvec`` substrate with plain Picard, the
physical disc seed, and the current-centroid soft prior — exactly the frozen
label engine's configuration with the magnetics mask OFF.  Firewall: GS force
balance + physics + measured channels only; no EFIT.

Usage:
    uv run python -m scripts.differentiable_solve_gate_eval --n-shots 5 --max-slices 6
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from imas_ambix.latent.gs_solve import SUBSTRATE_GREENS, EquilibriumGrid
from scripts.greens_filament_gate_eval import (
    AXIS_TOL_CM,
    CONFINED_AXIS_R_MAX,
    LCFS_TOL_CM,
    PROFILE_RMS_TOL,
    _axis_cm,
    _lcfs_cm,
    _profile,
    _profile_rms,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("differentiable_solve_gate")

FIG_DIR = Path("docs/figures/differentiable-solve-accelerator")

#: sweeps to skip before scoring residual monotonicity (fixed-shape warmup +
#: the LSQ-engagement transient — the same window the Anderson mixer skips)
TRACE_SKIP = 20
#: tail monotone fraction at/above which the residual is called monotone
MONOTONE_FRAC_GATE = 0.9
#: sweep cap for the trace leg (long enough to expose a limit cycle)
TRACE_ITERATIONS = 260


def _fit_slice(
    grid,
    table,
    basis,
    p,
    *,
    n_p,
    n_f,
    nonneg,
    smoothness,
    boundary_read,
    centroid,
    warm,
    sigma,
    topology_read="hard",
    iteration_trace=None,
    retry_max_iterations=160,
    accelerator="picard",
):
    """One frozen-spine ladder solve on the grid-free substrate."""
    from scripts.closure_gate_eval import fit_and_read_slice

    off = np.zeros_like(p.mask, dtype=bool)
    return fit_and_read_slice(
        grid,
        table,
        dataclasses.replace(p, mask=off),
        beta0_grid=(0.5,),
        alpha_grid=(1.0,),
        cost_limit=float("inf"),
        convergence_limit=5e-3,
        retry_max_iterations=retry_max_iterations,
        fit_mode="ladder",
        n_p=n_p,
        n_f=n_f,
        nonneg=nonneg,
        smoothness=smoothness,
        warm_jphi=warm,
        centroid_constraint=(centroid[0], centroid[1], sigma),
        reseed_axis_r_max=None,
        keep_psi=True,
        keep_jphi=True,
        basis=basis,
        meta={},
        boundary_read=boundary_read,
        substrate=SUBSTRATE_GREENS,
        accelerator=accelerator,
        topology_read=topology_read,
        iteration_trace=iteration_trace,
    )


def _solve_arm(grid, topology_read, p, centroid, disc_seed, table, basis, **spine_kw):
    """K=2 position scaffold → rich ladder under one topology read."""
    f_k2 = _fit_slice(
        grid,
        table,
        basis,
        p,
        n_p=1,
        n_f=1,
        nonneg=False,
        smoothness=spine_kw["smoothness"],
        boundary_read=spine_kw["boundary_read"],
        centroid=centroid,
        warm=disc_seed,
        sigma=spine_kw["sigma"],
        topology_read=topology_read,
    )
    k2_ok = (
        f_k2.scored
        and f_k2.jphi_flat is not None
        and np.isfinite(f_k2.target[0])
        and f_k2.target[0] <= CONFINED_AXIS_R_MAX
    )
    seed = f_k2.jphi_flat if k2_ok else disc_seed
    t0 = time.perf_counter()
    f = _fit_slice(
        grid,
        table,
        basis,
        p,
        n_p=spine_kw["n_p"],
        n_f=spine_kw["n_f"],
        nonneg=spine_kw["nonneg"],
        smoothness=spine_kw["smoothness"],
        boundary_read=spine_kw["boundary_read"],
        centroid=centroid,
        warm=seed,
        sigma=spine_kw["sigma"],
        topology_read=topology_read,
    )
    return f, seed, float(time.perf_counter() - t0)


def _trace_arm(grid, topology_read, p, centroid, seed, table, basis, **spine_kw):
    """Long plain-Picard rich solve with the per-sweep residual trace kept."""
    trace: list[dict] = []
    _fit_slice(
        grid,
        table,
        basis,
        p,
        n_p=spine_kw["n_p"],
        n_f=spine_kw["n_f"],
        nonneg=spine_kw["nonneg"],
        smoothness=spine_kw["smoothness"],
        boundary_read=spine_kw["boundary_read"],
        centroid=centroid,
        warm=seed,
        sigma=spine_kw["sigma"],
        topology_read=topology_read,
        iteration_trace=trace,
        retry_max_iterations=TRACE_ITERATIONS,
    )
    res = [
        t["residual"]
        for t in trace
        if t["residual"] is not None and np.isfinite(t["residual"])
    ]
    return np.asarray(res, dtype=np.float64)


def _monotone_frac(residuals: np.ndarray, skip: int = TRACE_SKIP) -> float:
    """Fraction of non-increasing residual steps in the post-transient tail."""
    tail = residuals[skip:]
    if tail.size < 10:
        return float("nan")
    d = np.diff(tail)
    return float(np.mean(d <= 1e-6 * np.abs(tail[:-1])))


def run_shot(
    shot: int, *, nr: int, nz: int, max_slices: int, min_ip_ka: float, sigma: float
) -> dict:
    """Hard vs smooth topology-read solve over one shot's held-out slices."""
    from imas_ambix.latent.boundary_disc import disc_read
    from scripts.heldout_mse_gate_eval import _campaign_table, shot_bt0
    from scripts.position_controlled_solve_gate import _disc_seed_flat
    from scripts.spine_label_factory import (
        factory_shot_payloads,
        frozen_spine_config,
    )

    spine, spine_sha = frozen_spine_config()
    iso = spine["interior_solve"]
    n_p, n_f = int(iso["n_p"]), int(iso["n_f"])
    spine_kw = dict(
        n_p=n_p,
        n_f=n_f,
        nonneg=iso["profile_kind"] == "monomial-nonneg",
        smoothness=float(iso["smoothness"]),
        boundary_read=iso["boundary_read_scoring"],
        sigma=sigma,
    )

    table = _campaign_table(shot)
    if table is None:
        return {"shot": shot, "rows": [], "reason": "no campaign table"}
    payload = factory_shot_payloads(
        shot, nr=nr, nz=nz, max_slices=max_slices, min_ip_ka=min_ip_ka, table=table
    )
    if payload is None:
        return {"shot": shot, "rows": [], "reason": "no payloads"}
    tbl, basis = payload["table"], payload["basis"]
    grid = EquilibriumGrid.from_table(tbl, nr=nr, nz=nz)
    bt0 = shot_bt0(shot)
    order = np.argsort([p.time_s for p in payload["payloads"]])

    rows: list[dict] = []
    traces: list[dict] = []
    for k in order:
        p = payload["payloads"][int(k)]
        inv = disc_read(p, grid, tbl, basis)
        if inv is None or inv.ring is None:
            continue
        centroid = (float(inv.centroid_r), float(inv.centroid_z))
        disc_seed = _disc_seed_flat(grid, inv)

        f_h, seed_h, dt_h = _solve_arm(
            grid, "hard", p, centroid, disc_seed, tbl, basis, **spine_kw
        )
        f_s, seed_s, dt_s = _solve_arm(
            grid, "connectivity", p, centroid, disc_seed, tbl, basis, **spine_kw
        )
        both = f_h.scored and f_s.scored
        j_h = _profile(f_h, grid, bt0, n_p=n_p, n_f=n_f, nonneg=spine_kw["nonneg"])
        j_s = _profile(f_s, grid, bt0, n_p=n_p, n_f=n_f, nonneg=spine_kw["nonneg"])
        # the §4 limit cycle sits ~1e-3: above the 3e-4 solve tolerance but
        # under the 5e-3 scoring limit — classify by the SOLVE tolerance
        hard_res = float(f_h.residual) if f_h.residual is not None else None
        hard_limit_cycles = hard_res is not None and hard_res > 3e-4
        rows.append(
            {
                "shot": shot,
                "k": int(k),
                "time_s": float(p.time_s),
                "ip_a": float(abs(p.ip_amperes)),
                "hard_scored": bool(f_h.scored),
                "smooth_scored": bool(f_s.scored),
                "axis_cm": _axis_cm(f_h.target, f_s.target) if both else None,
                "lcfs_cm": _lcfs_cm(f_h.target, f_s.target) if both else None,
                "profile_rms": _profile_rms(j_h, j_s),
                "hard_residual": hard_res,
                "smooth_residual": float(f_s.residual)
                if f_s.residual is not None
                else None,
                "hard_dt_s": dt_h,
                "smooth_dt_s": dt_s,
                "hard_limit_cycles": bool(hard_limit_cycles),
            }
        )
        # trace leg — only the under-determined (hard limit-cycling) class
        if hard_limit_cycles:
            r_h = _trace_arm(grid, "hard", p, centroid, seed_h, tbl, basis, **spine_kw)
            r_s = _trace_arm(
                grid, "connectivity", p, centroid, seed_s, tbl, basis, **spine_kw
            )
            traces.append(
                {
                    "shot": shot,
                    "k": int(k),
                    "hard_monotone_frac": _monotone_frac(r_h),
                    "smooth_monotone_frac": _monotone_frac(r_s),
                    "hard_final": float(r_h[-1]) if r_h.size else None,
                    "smooth_final": float(r_s[-1]) if r_s.size else None,
                    "hard_trace": r_h.tolist(),
                    "smooth_trace": r_s.tolist(),
                }
            )
    return {"shot": shot, "spine_sha": spine_sha, "rows": rows, "traces": traces}


def _verdicts(results: list[dict]) -> dict:
    rows = [r for res in results for r in res.get("rows", [])]
    traces = [t for res in results for t in res.get("traces", [])]
    paired = [r for r in rows if r["axis_cm"] is not None]
    wellposed = [
        r
        for r in paired
        if not r["hard_limit_cycles"]
        and r["smooth_residual"] is not None
        and r["smooth_residual"] <= 1e-3
    ]

    def _med(key, pool):
        v = [r[key] for r in pool if r[key] is not None and np.isfinite(r[key])]
        return float(np.median(v)) if v else float("nan")

    axis_med = _med("axis_cm", wellposed)
    lcfs_med = _med("lcfs_cm", wellposed)
    prof_med = _med("profile_rms", wellposed)
    da1 = (
        np.isfinite(axis_med)
        and axis_med <= AXIS_TOL_CM
        and np.isfinite(lcfs_med)
        and lcfs_med <= LCFS_TOL_CM
        and np.isfinite(prof_med)
        and prof_med <= PROFILE_RMS_TOL
    )
    smooth_ok = [
        t
        for t in traces
        if (
            t["smooth_monotone_frac"] is not None
            and np.isfinite(t["smooth_monotone_frac"])
            and t["smooth_monotone_frac"] >= MONOTONE_FRAC_GATE
        )
        or (t["smooth_final"] is not None and t["smooth_final"] <= 3e-4)
    ]
    da2 = bool(traces) and len(smooth_ok) == len(traces)
    return {
        "n_rows": len(rows),
        "n_paired": len(paired),
        "n_wellposed": len(wellposed),
        "n_limit_cycling": len(traces),
        "axis_cm_median": axis_med,
        "lcfs_cm_median": lcfs_med,
        "profile_rms_median": prof_med,
        "hard_monotone_fracs": [t["hard_monotone_frac"] for t in traces],
        "smooth_monotone_fracs": [t["smooth_monotone_frac"] for t in traces],
        "smooth_dt_over_hard_dt_median": float(
            np.median(
                [
                    r["smooth_dt_s"] / r["hard_dt_s"]
                    for r in rows
                    if r["hard_dt_s"] and r["smooth_dt_s"]
                ]
            )
        )
        if rows
        else float("nan"),
        "D_A1_reproduction": bool(da1),
        "D_A2_monotone_smooth_map": bool(da2),
        "tolerances": {
            "axis_cm": AXIS_TOL_CM,
            "lcfs_cm": LCFS_TOL_CM,
            "profile_rms": PROFILE_RMS_TOL,
            "monotone_frac": MONOTONE_FRAC_GATE,
        },
    }


def _figures(results: list[dict], verdicts: dict) -> str | None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rows = [
        r for res in results for r in res.get("rows", []) if r["axis_cm"] is not None
    ]
    traces = [t for res in results for t in res.get("traces", [])]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))

    a0 = axes[0]
    if traces:
        worst = max(traces, key=lambda t: t["hard_final"] or 0.0)
        a0.semilogy(worst["hard_trace"], color="#a33", lw=1.2, label="hard read")
        a0.semilogy(
            worst["smooth_trace"], color="#2a7", lw=1.2, label="connectivity read"
        )
        a0.axhline(3e-4, color="k", ls="--", lw=0.8, label="tolerance")
        a0.set_title(f"under-determined slice {worst['shot']}/k{worst['k']}: residual")
        a0.set_xlabel("Picard sweep")
        a0.set_ylabel("map residual")
        a0.legend(fontsize=8)
    else:
        a0.text(0.5, 0.5, "no limit-cycling slices found", ha="center")

    a1 = axes[1]
    vals = [r["axis_cm"] for r in rows if np.isfinite(r["axis_cm"])]
    if vals:
        a1.hist(vals, bins=24, color="#268", alpha=0.8)
        a1.axvline(
            verdicts["axis_cm_median"],
            color="k",
            lw=1.2,
            label=f"median {verdicts['axis_cm_median']:.2f} cm",
        )
        a1.axvline(
            AXIS_TOL_CM, color="k", ls="--", lw=1, label=f"tol {AXIS_TOL_CM:.0f} cm"
        )
        a1.legend(fontsize=8)
    a1.set_title("axis agreement hard vs smooth (paired slices)")
    a1.set_xlabel("axis distance [cm]")

    a2 = axes[2]
    if traces:
        x = np.arange(len(traces))
        hf = [t["hard_monotone_frac"] for t in traces]
        sf = [t["smooth_monotone_frac"] for t in traces]
        a2.bar(x - 0.18, hf, width=0.36, color="#a33", label="hard")
        a2.bar(x + 0.18, sf, width=0.36, color="#2a7", label="connectivity")
        a2.axhline(MONOTONE_FRAC_GATE, color="k", ls="--", lw=0.8, label="gate 0.9")
        a2.set_xticks(x)
        a2.set_xticklabels([f"{t['shot']}\nk{t['k']}" for t in traces], fontsize=7)
        a2.set_ylim(0, 1.05)
        a2.legend(fontsize=8)
    a2.set_title("residual tail monotone fraction")

    fig.tight_layout()
    path = FIG_DIR / "fig-s2-smooth-map-gate.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("wrote %s", path)
    return str(path)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shots", type=str, default="", help="explicit comma list")
    ap.add_argument("--n-shots", type=int, default=5, help="cap held-out shots")
    ap.add_argument("--max-slices", type=int, default=6)
    ap.add_argument("--min-ip-ka", type=float, default=200.0)
    ap.add_argument("--nr", type=int, default=65)
    ap.add_argument("--nz", type=int, default=97)
    ap.add_argument("--sigma", type=float, default=0.02)
    ap.add_argument(
        "--out",
        type=str,
        default="imas_ambix/latent/artifacts/patch_gate/differentiable_solve_gate-v0.json",
    )
    ap.add_argument("--no-figures", action="store_true")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    from imas_ambix.eval import prediction_bar as pbar

    if args.shots:
        shots = [int(s) for s in args.shots.split(",") if s.strip()]
    else:
        manifest = pbar.load_locked_manifest()
        shots = list(pbar.held_out_shot_ids(manifest))
        if args.n_shots > 0:
            shots = shots[: args.n_shots]
    logger.info("smooth-map gate over %d held-out shots: %s", len(shots), shots)

    results = []
    for s in shots:
        logger.info("shot %d ...", s)
        try:
            res = run_shot(
                int(s),
                nr=args.nr,
                nz=args.nz,
                max_slices=args.max_slices,
                min_ip_ka=args.min_ip_ka,
                sigma=args.sigma,
            )
        except Exception as exc:  # noqa: BLE001 — record, don't abort the sweep
            logger.warning("  shot %d failed: %s", s, exc)
            res = {"shot": int(s), "rows": [], "traces": [], "reason": f"error: {exc}"}
        logger.info(
            "  %d slices, %d limit-cycling",
            len(res.get("rows", [])),
            len(res.get("traces", [])),
        )
        results.append(res)

    verdicts = _verdicts(results)
    logger.info("VERDICTS: %s", json.dumps(verdicts, indent=2))

    fig = None
    if not args.no_figures:
        fig = _figures(results, verdicts)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "schema": "differentiable-solve-gate-v0",
                "topology_reads_compared": ["hard", "connectivity"],
                "shots": shots,
                "verdicts": verdicts,
                "results": results,
                "figure": fig,
            },
            indent=2,
        )
    )
    logger.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
