"""Fixed-point accelerator A/B on the continuous-topology (smooth) map.

With the hard per-sweep topology read the free-boundary map is
non-differentiable: finite-differencing the residual across a discrete
core-mask flip corrupts Newton–Krylov's Jacobian-free GMRES, the mask-flip
noise corrupts Anderson's least-squares history, and under-determined slices
limit-cycle.  The continuous read (``topology_read='connectivity'``) removes
the flips, so the accelerators are retested on the map they are meant for.

Two legs, both on the grid-free ``greens-matvec`` substrate:

* **LSQ-ladder leg** — the production solve.  Per slice, four arms:
  {hard, connectivity} × {plain relaxed Picard, safeguarded Anderson}, each
  with the same seed and a 260-sweep budget.  Reports sweeps-to-tolerance,
  wall, final residual, and the axis agreement between the accelerated and
  the Picard fixed point (byte-comparability).
* **Fixed-shape NK leg** — the scheme the prior study measured as erratic on
  the hard map (divergent on 2/5 slices).  Per slice, Jacobian-free
  Newton–Krylov under {hard, connectivity}, scored against the converged
  fixed-shape Picard reference: final residual, wall, axis distance.

Verdict D-B (pre-declared): at least one accelerator converges the
previously-limit-cycling slices cleanly and materially faster than relaxed
Picard on the smooth map, byte-comparably — or the honest negative localises
the residual bottleneck to batching.

Usage:
    uv run python -m scripts.smooth_map_accelerator_eval --n-shots 5 --max-slices 4
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from imas_ambix.latent.gs_solve import (
    SUBSTRATE_GREENS,
    EquilibriumGrid,
    solve_equilibrium,
    solve_equilibrium_nk,
)
from scripts.differentiable_solve_gate_eval import _fit_slice

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("smooth_map_accelerator")

FIG_DIR = Path("docs/figures/differentiable-solve-accelerator")

SWEEP_BUDGET = 260
TOLERANCE = 3e-4


def _ladder_arm(
    grid, p, centroid, seed, table, basis, *, topology_read, accelerator, **spine_kw
):
    """One rich ladder solve; returns metrics for the A/B table."""
    trace: list[dict] = []
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
        accelerator=accelerator,
        iteration_trace=trace,
        retry_max_iterations=SWEEP_BUDGET,
    )
    wall = float(time.perf_counter() - t0)
    res = [
        t["residual"]
        for t in trace
        if t["residual"] is not None and np.isfinite(t["residual"])
    ]
    res = np.asarray(res, dtype=np.float64)
    sweeps_to_tol = (
        int(np.argmax(res <= TOLERANCE) + 1) if np.any(res <= TOLERANCE) else None
    )
    return {
        "scored": bool(f.scored),
        "residual": float(f.residual) if f.residual is not None else None,
        "converged": bool(res.size and res[-1] <= TOLERANCE)
        or (f.residual is not None and f.residual <= TOLERANCE),
        "sweeps_to_tol": sweeps_to_tol,
        "n_sweeps": int(res.size),
        "wall_s": wall,
        "axis": [float(f.target[0]), float(f.target[1])] if f.scored else None,
    }


def run_shot(
    shot: int,
    *,
    nr: int,
    nz: int,
    max_slices: int,
    min_ip_ka: float,
    sigma: float,
    legs: str = "both",
) -> dict:
    from imas_ambix.latent.boundary_disc import disc_read
    from scripts.heldout_mse_gate_eval import _campaign_table
    from scripts.position_controlled_solve_gate import _disc_seed_flat
    from scripts.spine_label_factory import (
        factory_shot_payloads,
        frozen_spine_config,
    )

    spine, spine_sha = frozen_spine_config()
    iso = spine["interior_solve"]
    spine_kw = dict(
        n_p=int(iso["n_p"]),
        n_f=int(iso["n_f"]),
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
    order = np.argsort([p.time_s for p in payload["payloads"]])

    rows: list[dict] = []
    nk_rows: list[dict] = []
    for k in order:
        p = payload["payloads"][int(k)]
        inv = disc_read(p, grid, tbl, basis)
        if inv is None or inv.ring is None:
            continue
        centroid = (float(inv.centroid_r), float(inv.centroid_z))
        disc_seed = _disc_seed_flat(grid, inv)

        if legs in ("both", "ladder"):
            arms = {}
            for topo in ("hard", "connectivity"):
                for acc in ("picard", "anderson"):
                    arms[f"{topo}:{acc}"] = _ladder_arm(
                        grid,
                        p,
                        centroid,
                        disc_seed,
                        tbl,
                        basis,
                        topology_read=topo,
                        accelerator=acc,
                        **spine_kw,
                    )
            base = arms["hard:picard"]
            rows.append(
                {
                    "shot": shot,
                    "k": int(k),
                    "time_s": float(p.time_s),
                    "ip_a": float(abs(p.ip_amperes)),
                    "limit_cycling_hard": bool(
                        base["residual"] is not None and base["residual"] > TOLERANCE
                    ),
                    "arms": arms,
                }
            )
        if legs not in ("both", "nk"):
            continue

        # fixed-shape NK leg (the prior study's erratic scheme), hard vs smooth
        ref = solve_equilibrium(
            grid,
            p.i_pf,
            p.ip_amperes,
            beta0=0.5,
            alpha=1.0,
            max_iterations=200,
            coil_field_mode="analytic-add",
            substrate=SUBSTRATE_GREENS,
        )
        nk = {}
        for topo in ("hard", "connectivity"):
            t0 = time.perf_counter()
            try:
                r = solve_equilibrium_nk(
                    grid,
                    p.i_pf,
                    p.ip_amperes,
                    substrate=SUBSTRATE_GREENS,
                    topology_read=topo,
                    initial_jphi=disc_seed,
                    maxiter=40,
                )
                nk[topo] = {
                    "residual": float(r.residual),
                    "converged": bool(r.converged),
                    "axis_cm_vs_picard": float(
                        100.0
                        * np.hypot(r.axis[0] - ref.axis[0], r.axis[1] - ref.axis[1])
                    ),
                    "wall_s": float(time.perf_counter() - t0),
                }
            except Exception as exc:  # noqa: BLE001 — an NK blow-up IS a result
                nk[topo] = {"error": str(exc)}
        nk_rows.append(
            {
                "shot": shot,
                "k": int(k),
                "picard_ref_converged": bool(ref.converged),
                "nk": nk,
            }
        )
    return {"shot": shot, "spine_sha": spine_sha, "rows": rows, "nk_rows": nk_rows}


def _verdicts(results: list[dict]) -> dict:
    rows = [r for res in results for r in res.get("rows", [])]
    nk_rows = [r for res in results for r in res.get("nk_rows", [])]
    cyc = [r for r in rows if r["limit_cycling_hard"]]
    well = [r for r in rows if not r["limit_cycling_hard"]]

    def _speedup(pool, arm, base="connectivity:picard"):
        """median sweeps(base)/sweeps(arm) over slices where both converged."""
        ratios = []
        for r in pool:
            a, b = r["arms"].get(arm), r["arms"].get(base)
            if a and b and a["sweeps_to_tol"] and b["sweeps_to_tol"]:
                ratios.append(b["sweeps_to_tol"] / a["sweeps_to_tol"])
        return float(np.median(ratios)) if ratios else float("nan")

    def _conv_count(pool, arm):
        return sum(1 for r in pool if r["arms"].get(arm, {}).get("converged"))

    def _axis_gap_cm(pool, arm, base="connectivity:picard"):
        gaps = []
        for r in pool:
            a, b = r["arms"].get(arm), r["arms"].get(base)
            if a and b and a["axis"] and b["axis"]:
                gaps.append(
                    100.0
                    * float(
                        np.hypot(
                            a["axis"][0] - b["axis"][0], a["axis"][1] - b["axis"][1]
                        )
                    )
                )
        return float(np.median(gaps)) if gaps else float("nan")

    arms = [
        "hard:picard",
        "hard:anderson",
        "connectivity:picard",
        "connectivity:anderson",
    ]
    summary = {
        arm: {
            "converged_wellposed": _conv_count(well, arm),
            "converged_limit_cycling": _conv_count(cyc, arm),
            "sweep_speedup_vs_smooth_picard_wellposed": _speedup(well, arm),
            "sweep_speedup_vs_smooth_picard_cycling": _speedup(cyc, arm),
            "axis_gap_cm_vs_smooth_picard": _axis_gap_cm(well, arm),
        }
        for arm in arms
    }
    nk_ok = {
        topo: sum(
            1
            for r in nk_rows
            if r["nk"].get(topo, {}).get("converged")
            and r["nk"][topo].get("axis_cm_vs_picard", 1e9) < 5.0
        )
        for topo in ("hard", "connectivity")
    }
    nk_diverged = {
        topo: sum(
            1
            for r in nk_rows
            if "error" in r["nk"].get(topo, {})
            or r["nk"].get(topo, {}).get("axis_cm_vs_picard", 0.0) > 20.0
        )
        for topo in ("hard", "connectivity")
    }
    # D-B: on the smooth map, an accelerator converges the previously-cycling
    # slices AND is materially faster than smooth-map Picard (byte-comparable)
    and_arm = summary["connectivity:anderson"]
    db = bool(
        cyc
        and and_arm["converged_limit_cycling"] > 0
        and (
            and_arm["sweep_speedup_vs_smooth_picard_cycling"] >= 1.5
            or and_arm["converged_limit_cycling"]
            > summary["connectivity:picard"]["converged_limit_cycling"]
        )
        and (
            not np.isfinite(and_arm["axis_gap_cm_vs_smooth_picard"])
            or and_arm["axis_gap_cm_vs_smooth_picard"] <= 2.0
        )
    )
    return {
        "n_rows": len(rows),
        "n_limit_cycling_hard": len(cyc),
        "arms": summary,
        "nk_converged_close": nk_ok,
        "nk_diverged": nk_diverged,
        "n_nk_rows": len(nk_rows),
        "D_B_accelerator_on_smooth_map": db,
    }


def _figures(results: list[dict], verdicts: dict) -> str | None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rows = [r for res in results for r in res.get("rows", [])]
    if not rows:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))

    a0 = axes[0]
    arms = [
        ("hard:picard", "#a33", "o"),
        ("hard:anderson", "#c73", "s"),
        ("connectivity:picard", "#268", "o"),
        ("connectivity:anderson", "#2a7", "s"),
    ]
    for i, r in enumerate(rows):
        for j, (arm, color, mk) in enumerate(arms):
            a = r["arms"].get(arm)
            if a:
                s = a["sweeps_to_tol"] if a["sweeps_to_tol"] else SWEEP_BUDGET + 20
                a0.plot(i + (j - 1.5) * 0.12, s, mk, color=color, ms=4)
    for arm, color, _mk in arms:
        a0.plot([], [], "o", color=color, label=arm)
    a0.axhline(SWEEP_BUDGET, color="k", ls="--", lw=0.8, label="budget (no conv)")
    a0.set_xlabel("slice")
    a0.set_ylabel("sweeps to tolerance")
    a0.set_title("sweeps to 3e-4 per arm")
    a0.legend(fontsize=7)

    a1 = axes[1]
    labels = [a for a, _c, _m in arms]
    conv_cyc = [verdicts["arms"][arm]["converged_limit_cycling"] for arm in labels]
    conv_wp = [verdicts["arms"][arm]["converged_wellposed"] for arm in labels]
    x = np.arange(len(labels))
    a1.bar(x - 0.18, conv_wp, width=0.36, color="#268", label="well-posed")
    a1.bar(x + 0.18, conv_cyc, width=0.36, color="#a33", label="prev. limit-cycling")
    a1.set_xticks(x)
    a1.set_xticklabels(labels, rotation=20, fontsize=7)
    a1.set_ylabel("slices converged")
    a1.set_title("convergence by arm")
    a1.legend(fontsize=8)

    a2 = axes[2]
    nk_rows = [r for res in results for r in res.get("nk_rows", [])]
    for topo, color in (("hard", "#a33"), ("connectivity", "#2a7")):
        vals = [
            r["nk"][topo].get("axis_cm_vs_picard")
            for r in nk_rows
            if topo in r["nk"] and "axis_cm_vs_picard" in r["nk"][topo]
        ]
        if vals:
            a2.plot(vals, "o", color=color, label=f"NK ({topo} read)", ms=4)
    a2.axhline(5.0, color="k", ls="--", lw=0.8, label="5 cm")
    a2.set_yscale("log")
    a2.set_xlabel("slice")
    a2.set_ylabel("NK axis dist. to Picard ref [cm]")
    a2.set_title("fixed-shape Newton–Krylov")
    a2.legend(fontsize=8)

    fig.tight_layout()
    path = FIG_DIR / "fig-s3-accelerator-retest.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("wrote %s", path)
    return str(path)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shots", type=str, default="", help="explicit comma list")
    ap.add_argument("--n-shots", type=int, default=5)
    ap.add_argument("--max-slices", type=int, default=4)
    ap.add_argument("--min-ip-ka", type=float, default=200.0)
    ap.add_argument("--nr", type=int, default=65)
    ap.add_argument("--nz", type=int, default=97)
    ap.add_argument("--sigma", type=float, default=0.02)
    ap.add_argument(
        "--legs",
        type=str,
        default="both",
        choices=("both", "ladder", "nk"),
        help="run the LSQ-ladder leg, the fixed-shape NK leg, or both",
    )
    ap.add_argument(
        "--out",
        type=str,
        default="imas_ambix/latent/artifacts/patch_gate/smooth_map_accelerator-v0.json",
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
    logger.info("accelerator A/B over %d held-out shots: %s", len(shots), shots)

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
                legs=args.legs,
            )
        except Exception as exc:  # noqa: BLE001 — record, don't abort the sweep
            logger.warning("  shot %d failed: %s", s, exc)
            res = {"shot": int(s), "rows": [], "nk_rows": [], "reason": f"error: {exc}"}
        logger.info("  %d slices", len(res.get("rows", [])))
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
                "schema": "smooth-map-accelerator-v0",
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
