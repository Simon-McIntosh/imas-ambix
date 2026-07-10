#!/usr/bin/env python
"""Low-DOF closure arm: EFIT-style profile-parametrised free-boundary inverse.

The free ~4,787-DOF patch inverse leaves a ~4,700-dimensional null space that
only soft priors span, so nothing shapes how current distributes within its
support.  EFIT's boundary is "trivial" instead because it parametrises
jφ(R,ψ_N) = R·p′(ψ_N) + FF′(ψ_N)/(μ₀R) with ~10 DOF, solves the free-boundary
fixed point each Picard sweep, and confines support by construction.  This
arm scores that fixed point on the hardened gate harness: per held-out slice
fit the two-parameter jφ(ψ_N; β0, α) profile against the RAW magnetics
(measured − vacuum, whitened, masked) through the in-tree free-boundary GS
solve (:mod:`imas_ambix.latent.gs_solve`), read axis / X-point set / LCFS from
the force-balanced ψ with the SAME readout the other arms use
(:func:`scripts.patch_gate_eval.geometry_target`), and score against the
firewalled EFIT referee with the paired-bootstrap CIs, saddle-excess metric,
cohort, and split the patch arm uses.

Coverage is reported honestly (n_scored / n_candidate): a slice whose best fit
does not meet the relaxed-Picard convergence criterion, or whose whitened
magnetics cost exceeds the honesty ceiling, is REPORTED as masked with its
reason — never scored with a fabricated readout, never silently dropped.

Protocol: leakage-free.  The profile grid, cost ceiling, and Picard knobs are
the frozen physics-motivated defaults, CONFIRMED on ``--split tune`` (the same
4 train-shot tune cohort the patch arm uses) and then applied UNCHANGED to the
single ``--split eval`` held-out run (160 slices).

Firewall: EFIT enters only the referee/scoring/plotting path (inside the
referee's evaluator context), never the fit.  Raw magnetics only; conventions
total flux Φ = 2πR·A_φ [Wb], μ0 explicit, MAST psi_axis > psi_boundary.

Artifacts:  imas_ambix/latent/artifacts/patch_gate/closure_gate_eval[-tune].json
Figures:    docs/figures/plasma-current-priors-hardening/fig-closure-*.png
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np

from imas_ambix.gs.geometry import GEOMETRY_TABLE_VERSION, build_table_for_shot
from imas_ambix.gs.operator import COIL_MODEL_VERSION, build_operator
from imas_ambix.latent.data import feature_schema, read_split_shot_lists
from imas_ambix.latent.gs_solve import (
    EquilibriumGrid,
    fit_profile,
    profile_jphi_shape,
)
from imas_ambix.latent.patch_inverse import SlicePayload

# reuse the hardened harness verbatim — identical cohort, readout, and scoring
from scripts.patch_gate_eval import (
    count_saddles,
    geometry_target,
    lcfs_offset_cm_stats,
    saddle_excess_stats,
    score,
    shot_payloads,
    train_mean_baseline,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("closure_gate_eval")

ARTIFACTS = Path("imas_ambix/latent/artifacts/patch_gate")
FIGURES = Path("docs/figures/plasma-current-priors-hardening")


@dataclass
class ClosureSliceFit:
    """One slice's closure fit: geometry read + fit provenance, or an honest
    masked record (``scored=False``, ``target=None``, ``reason`` set)."""

    shot: int
    t_index: int
    time_s: float
    ip_amperes: float
    scored: bool
    reason: str  # "scored" | "no-converged-candidate" | "cost-exceeds-limit"
    beta0: float | None = None
    alpha: float | None = None
    cost: float | None = None
    converged: bool | None = None
    residual: float | None = None
    target: np.ndarray | None = None  # 14-D geometry read (None if masked)
    saddles: int | None = None


def fit_and_read_slice(
    grid: EquilibriumGrid,
    table,
    payload: SlicePayload,
    *,
    beta0_grid: tuple[float, ...],
    alpha_grid: tuple[float, ...],
    cost_limit: float,
    convergence_limit: float,
    retry_max_iterations: int | None = None,
) -> ClosureSliceFit:
    """Fit (β0, α) against ``payload``'s raw magnetics through the GS fixed
    point and read the hardened 14-D geometry from the force-balanced ψ.

    ``retry_max_iterations``: if the default-Picard grid search masks the slice
    (no candidate meets ``convergence_limit``), retry ONCE with a longer Picard
    budget — a bounded, honest coverage lever (no fallback current is ever
    fabricated).  Returns a :class:`ClosureSliceFit`; masked slices carry
    ``scored=False`` and a ``reason``, never a fabricated readout.

    ``cost_limit`` is an OPTIONAL honesty ceiling on the whitened magnetics
    misfit; the default (``inf``) scores every CONVERGED equilibrium and
    carries the cost as a diagnostic.  This mirrors the free patch arm's
    contract (score all, report misfit) so the two arms' coverage is
    comparable, and it does not conflate "no force-balanced equilibrium"
    (the only thing that masks) with "converged but doesn't fit the magnetics"
    (the expected static coil/calibration misfit — a finding, not a mask).
    """
    kw = dict(
        i_pf=payload.i_pf,
        ip_amperes=payload.ip_amperes,
        measured=payload.measured,
        vacuum_prediction=payload.vacuum,
        sensor_scale=payload.scale,
        sensor_mask=payload.mask,
        beta0_grid=beta0_grid,
        alpha_grid=alpha_grid,
        convergence_limit=convergence_limit,
    )
    fit = fit_profile(grid, table, **kw)
    if fit is None and retry_max_iterations:
        fit = fit_profile(grid, table, max_iterations=retry_max_iterations, **kw)

    base = dict(
        shot=payload.shot,
        t_index=payload.t_index,
        time_s=payload.time_s,
        ip_amperes=payload.ip_amperes,
    )
    if fit is None:
        return ClosureSliceFit(
            **base, scored=False, reason="no-converged-candidate"
        )
    if fit.cost > cost_limit:
        return ClosureSliceFit(
            **base,
            scored=False,
            reason="cost-exceeds-limit",
            beta0=fit.beta0,
            alpha=fit.alpha,
            cost=fit.cost,
            converged=bool(fit.result.converged),
            residual=float(fit.result.residual),
        )
    target, _, _ = geometry_target(fit.result.psi, grid)
    return ClosureSliceFit(
        **base,
        scored=True,
        reason="scored",
        beta0=fit.beta0,
        alpha=fit.alpha,
        cost=fit.cost,
        converged=bool(fit.result.converged),
        residual=float(fit.result.residual),
        target=target,
        saddles=count_saddles(fit.result.psi, grid),
    )


# ---------------------------------------------------------------------------
# per-shot fork pool: the SuperLU-holding grid is inherited copy-on-write by
# the children (unpicklable), so worker state is populated in the PARENT and
# the pool only ships the tiny per-slice index + config (mirrors
# gs_solve_gate_eval's fork pattern; the default start method is fork here).
# ---------------------------------------------------------------------------
_WORKER: dict = {}


def _init_worker(state: dict) -> None:
    _WORKER.update(state)


def _fit_slice_worker(k: int) -> ClosureSliceFit:
    return fit_and_read_slice(
        _WORKER["grid"],
        _WORKER["table"],
        _WORKER["payloads"][k],
        beta0_grid=_WORKER["beta0_grid"],
        alpha_grid=_WORKER["alpha_grid"],
        cost_limit=_WORKER["cost_limit"],
        convergence_limit=_WORKER["convergence_limit"],
        retry_max_iterations=_WORKER["retry_max_iterations"],
    )


def fit_shot(payload: dict, cfg: dict, workers: int) -> list[ClosureSliceFit]:
    """Fit every candidate slice of one shot (fork pool over slices)."""
    payloads = payload["payloads"]
    state = {
        "grid": payload["grid"],
        "table": payload["table"],
        "payloads": payloads,
        **cfg,
    }
    _init_worker(state)
    idx = list(range(len(payloads)))
    if workers > 1 and len(idx) > 1:
        ctx = multiprocessing.get_context("fork")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
            return list(pool.map(_fit_slice_worker, idx))
    return [_fit_slice_worker(k) for k in idx]


def _grids_for(split: str, args) -> tuple[tuple[float, ...], tuple[float, ...]]:
    b = tuple(float(v) for v in args.beta0_grid.split(",") if v.strip())
    a = tuple(float(v) for v in args.alpha_grid.split(",") if v.strip())
    return b, a


def run_gate(args) -> int:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    beta0_grid, alpha_grid = _grids_for(args.split, args)
    cfg = {
        "beta0_grid": beta0_grid,
        "alpha_grid": alpha_grid,
        "cost_limit": args.cost_limit,
        "convergence_limit": args.convergence_limit,
        "retry_max_iterations": args.retry_max_iterations,
    }
    logger.info("closure gate split=%s cfg=%s", args.split, cfg)

    train_shots, held_shots = read_split_shot_lists(args.n_train, args.n_heldout)
    baseline_vec = train_mean_baseline(
        args.n_train, args.n_baseline_shots, args.min_ip_ka
    )
    logger.info(
        "baseline (train-mean) axis: (%.3f, %.3f)", baseline_vec[0], baseline_vec[1]
    )
    if args.split == "train":
        eval_shots = train_shots[
            args.n_baseline_shots : args.n_baseline_shots + args.n_tune_shots
        ]
    else:
        eval_shots = held_shots
    if args.shots:
        want = {int(s) for s in args.shots.split(",") if s.strip()}
        eval_shots = [s for s in eval_shots if int(s) in want]
    logger.info("shots: %s", list(eval_shots))

    all_fits: list[ClosureSliceFit] = []
    refs_by_shot: dict[int, np.ndarray] = {}
    t0 = time.perf_counter()
    for s in eval_shots:
        try:
            payload = shot_payloads(
                s,
                nr=args.nr,
                nz=args.nz,
                max_slices=args.max_slices_per_shot,
                min_ip_ka=args.min_ip_ka,
                split=args.split,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("shot %s failed to load: %s", s, exc)
            continue
        if payload is None:
            continue
        refs_by_shot[int(s)] = payload["refs"]
        fits = fit_shot(payload, cfg, args.workers)
        # attach the per-shot referee row index for scored slices
        for k, f in enumerate(fits):
            f._ref = payload["refs"][k]  # type: ignore[attr-defined]
        n_scored = sum(f.scored for f in fits)
        logger.info(
            "shot %s: %d/%d scored (%d masked)",
            s,
            n_scored,
            len(fits),
            len(fits) - n_scored,
        )
        all_fits.extend(fits)
    wall_s = time.perf_counter() - t0

    n_candidate = len(all_fits)
    scored = [f for f in all_fits if f.scored]
    n_scored = len(scored)
    reasons: dict[str, int] = {}
    for f in all_fits:
        reasons[f.reason] = reasons.get(f.reason, 0) + 1
    tag = "-tune" if args.split == "train" else ""
    if n_scored == 0:
        # write the diagnostic record anyway (coverage + reasons) — a zero-
        # coverage cohort is a reportable finding, not a silent abort
        logger.error(
            "no slices scored on split=%s (coverage 0/%d, reasons=%s)",
            args.split,
            n_candidate,
            reasons,
        )
        (ARTIFACTS / f"closure_gate_eval{tag}.json").write_text(
            json.dumps(
                {
                    "arm": "closure",
                    "split": args.split,
                    "config": cfg,
                    "n_scored": 0,
                    "n_candidate": n_candidate,
                    "scored_fraction": 0.0,
                    "mask_reasons": reasons,
                },
                indent=2,
            )
        )
        return 0

    model = np.array([f.target for f in scored])
    ref = np.array([f._ref for f in scored])  # type: ignore[attr-defined]
    shot_ids = np.array([f.shot for f in scored])
    saddles = np.array([f.saddles for f in scored])
    # flat-top proxy: per shot, the single scored slice with the largest |Ip|
    flattop_mask = np.zeros(n_scored, dtype=bool)
    for s in np.unique(shot_ids):
        idx = np.flatnonzero(shot_ids == s)
        ips = np.array([scored[i].ip_amperes for i in idx])
        flattop_mask[idx[int(np.argmax(ips))]] = True

    sc = score(model, ref, baseline_vec, shot_ids=shot_ids)
    axis_errors = sc.pop("axis_errors")
    lcfs_cm = lcfs_offset_cm_stats(model, ref, flattop_mask)
    saddle_stats = saddle_excess_stats(saddles, ref)

    beta0_fit = np.array([f.beta0 for f in scored], dtype=np.float64)
    alpha_fit = np.array([f.alpha for f in scored], dtype=np.float64)
    cost_fit = np.array([f.cost for f in scored], dtype=np.float64)
    conv_frac = float(np.mean([bool(f.converged) for f in scored]))

    result = {
        "arm": "closure",
        "split": args.split,
        "config": cfg
        | {"n_train": args.n_train, "n_heldout": args.n_heldout, "nr": args.nr,
           "nz": args.nz, "min_ip_ka": args.min_ip_ka},
        "coil_model_version": COIL_MODEL_VERSION,
        "geometry_table_version": GEOMETRY_TABLE_VERSION,
        "n_scored": n_scored,
        "n_candidate": n_candidate,
        "scored_fraction": float(n_scored / max(n_candidate, 1)),
        "mask_reasons": reasons,
        "strict_converged_fraction_of_scored": conv_frac,
        "beta0_median": float(np.median(beta0_fit)),
        "alpha_median": float(np.median(alpha_fit)),
        "cost_median_scored": float(np.median(cost_fit)),
        "cost_le_3_fraction": float(np.mean(cost_fit <= 3.0)),
        "wall_s": wall_s,
        **sc,
        **lcfs_cm,
        **saddle_stats,
    }
    tag = "-tune" if args.split == "train" else ""
    (ARTIFACTS / f"closure_gate_eval{tag}.json").write_text(json.dumps(result, indent=2))
    np.savez(
        ARTIFACTS / f"closure_gate_eval{tag}_arrays.npz",
        model=model,
        ref=ref,
        baseline=np.tile(baseline_vec, (n_scored, 1)),
        axis_errors=axis_errors,
        shot_ids=shot_ids,
        flattop_mask=flattop_mask,
        saddles=saddles,
        beta0=beta0_fit,
        alpha=alpha_fit,
        cost=np.array([f.cost for f in scored], dtype=np.float64),
    )
    logger.info(
        "[closure %s] scored %d/%d (%.0f%%) axis_skill=%.3f ci=%s lcfs_skill=%s "
        "xpt_skill=%s axis_median=%.3f m lcfs_cm(all/flat)=%s/%s saddle_excess_med=%s "
        "(%.0f s)",
        args.split,
        n_scored,
        n_candidate,
        100.0 * n_scored / max(n_candidate, 1),
        sc["axis_skill"],
        sc.get("axis_skill_ci"),
        sc["lcfs_skill"],
        sc["xpoint_set_skill"],
        sc["axis_error_median_m"],
        lcfs_cm["lcfs_offset_median_cm_all"],
        lcfs_cm["lcfs_offset_median_cm_flattop"],
        saddle_stats["saddle_excess_median"],
        wall_s,
    )
    return 0


# ---------------------------------------------------------------------------
# Figures (imas-ink): closure-arm ψ vs EFIT for the held-out shots + the
# fitted jφ(ψ_N) profile family.
# ---------------------------------------------------------------------------
def run_figures(args) -> int:
    import matplotlib.pyplot as plt

    from imas_ink.figures import equilibrium_figure_mpl
    from scripts.patch_flux_map_report import (
        _closed_contour_about,
        _efit_slice,
        _fig_to_rgba,
        _machine_geometry,
        _our_slice,
        select_slices,
    )

    FIGURES.mkdir(parents=True, exist_ok=True)
    beta0_grid, alpha_grid = _grids_for("eval", args)
    _, held_shots = read_split_shot_lists(args.n_train, args.n_heldout)
    logger.info("figures: held-out shots %s", list(held_shots))

    flux_panels = {"rampup": {}, "flattop": {}}
    profile_rows = []  # (regime, beta0, alpha, psi_n grid, shape)
    for shot in held_shots:
        try:
            payload = shot_payloads(
                shot, nr=args.nr, nz=args.nz, max_slices=args.max_slices_per_shot,
                min_ip_ka=args.min_ip_ka, split="eval",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("shot %s failed to load: %s", shot, exc)
            continue
        if payload is None:
            continue
        grid, table, payloads = payload["grid"], payload["table"], payload["payloads"]
        picks = select_slices(payloads, shot)
        if not picks:
            continue
        geom = _machine_geometry(grid, table)
        for kind, k, efit in picks:
            p = payloads[k]
            fit = fit_and_read_slice(
                grid, table, p,
                beta0_grid=beta0_grid, alpha_grid=alpha_grid,
                cost_limit=args.cost_limit, convergence_limit=args.convergence_limit,
                retry_max_iterations=args.retry_max_iterations,
            )
            if not fit.scored:
                logger.info("%d %-7s masked (%s) — skip panel", shot, kind, fit.reason)
                continue
            # re-solve at the fitted (β0, α) for the ψ field to plot
            from imas_ambix.latent.gs_solve import solve_equilibrium_bootstrapped

            res = solve_equilibrium_bootstrapped(
                grid, p.i_pf, p.ip_amperes, beta0=fit.beta0, alpha=fit.alpha
            )
            psi2d = res.psi
            target, psi_ax, psi_b = geometry_target(psi2d, grid)
            axis_rz = (float(target[0]), float(target[1]))
            our_lcfs = _closed_contour_about(grid.rg, grid.zg, psi2d, psi_b, *axis_rz)
            sl = _our_slice(
                psi2d, grid, target, psi_ax, psi_b, p.ip_amperes, p.time_s, our_lcfs
            )
            fig, _ax = equilibrium_figure_mpl(
                sl, geom, reference_slice=_efit_slice(efit), reference_name="EFIT",
                figsize=(4.6, 6.0), show_probes=False, show_flux_loops=False,
            )
            fig.suptitle(
                f"{shot} t={p.time_s:.3f}s ({kind})  β0={fit.beta0:.2f} α={fit.alpha:.1f}",
                fontsize=9,
            )
            flux_panels[kind][int(shot)] = _fig_to_rgba(fig)
            plt.close(fig)
            psi_n = np.linspace(0.0, 1.0, 60)
            shape = profile_jphi_shape(
                psi_n, np.full_like(psi_n, grid.r0), r0=grid.r0,
                beta0=fit.beta0, alpha=fit.alpha,
            )
            profile_rows.append((kind, fit.beta0, fit.alpha, psi_n, shape))
            logger.info("%d %-7s β0=%.2f α=%.1f rendered", shot, kind, fit.beta0, fit.alpha)

    fig_paths = []
    for regime in ("rampup", "flattop"):
        shots = sorted(flux_panels[regime])
        if not shots:
            continue
        ncol = min(4, len(shots))
        nrow = int(np.ceil(len(shots) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 5.4 * nrow))
        axes = np.atleast_1d(axes).ravel()
        for ax in axes:
            ax.axis("off")
        for ax, s in zip(axes, shots, strict=False):
            ax.imshow(flux_panels[regime][s])
        fig.suptitle(
            f"Closure-arm ψ(R,Z) vs EFIT — {regime} "
            f"(primary: profile-parametrised GS solve, faint: EFIT)",
            fontsize=13,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        out = FIGURES / f"fig-closure-flux-maps-{regime}.png"
        fig.savefig(out, dpi=110)
        plt.close(fig)
        fig_paths.append(str(out))
        logger.info("wrote %s (%d panels)", out, len(shots))

    # fitted jφ(ψ_N) profile family + β0/α distribution
    if profile_rows:
        fig, (axc, axd) = plt.subplots(1, 2, figsize=(11.0, 4.6))
        cmap = {"rampup": "#1b7837", "flattop": "#d95f02"}
        for kind, b0, al, psi_n, shape in profile_rows:
            axc.plot(psi_n, shape, color=cmap[kind], alpha=0.5, lw=1.2)
        axc.set_xlabel("ψ_N")
        axc.set_ylabel("jφ shape at R=R0  (β0 + (1−β0))·(1−ψ_N)^α")
        axc.set_title("Fitted jφ(ψ_N) profiles (per held-out slice)")
        for kind in ("rampup", "flattop"):
            axc.plot([], [], color=cmap[kind], label=kind)
        axc.legend(fontsize=8)
        for kind in ("rampup", "flattop"):
            b0 = [r[1] for r in profile_rows if r[0] == kind]
            al = [r[2] for r in profile_rows if r[0] == kind]
            axd.scatter(b0, al, color=cmap[kind], label=kind, s=40, alpha=0.7)
        axd.set_xlabel("β0 (pressure/FF′ split)")
        axd.set_ylabel("α (peakedness)")
        axd.set_title("Fitted (β0, α) distribution")
        axd.legend(fontsize=8)
        fig.tight_layout()
        out = FIGURES / "fig-closure-profile-fits.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        fig_paths.append(str(out))
        logger.info("wrote %s (%d profiles)", out, len(profile_rows))

    (ARTIFACTS / "closure_figures.json").write_text(
        json.dumps({"figures": fig_paths, "n_profiles": len(profile_rows)}, indent=2)
    )
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
    ap.add_argument("--n-tune-shots", type=int, default=4)
    ap.add_argument(
        "--split", type=str, default="eval", choices=("eval", "train"),
        help="eval = held-out gate; train = the 4-shot leakage-free tune cohort",
    )
    ap.add_argument(
        "--shots", type=str, default="",
        help="comma list to restrict the run (sbatch-array sharding); '' = all",
    )
    # frozen physics-motivated config (confirmed on --split train, applied
    # unchanged to --split eval)
    ap.add_argument("--beta0-grid", type=str, default="0.1,0.3,0.5,0.7,0.9")
    ap.add_argument("--alpha-grid", type=str, default="1.0,1.5,2.0")
    ap.add_argument(
        "--cost-limit", type=float, default=float("inf"),
        help="optional whitened-misfit ceiling; default inf scores every "
        "converged equilibrium (cost carried as a diagnostic)",
    )
    ap.add_argument("--convergence-limit", type=float, default=5e-3)
    ap.add_argument("--retry-max-iterations", type=int, default=160)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--figures", action="store_true", help="render figures only")
    args = ap.parse_args()

    if args.figures:
        return run_figures(args)
    return run_gate(args)


if __name__ == "__main__":
    raise SystemExit(main())
