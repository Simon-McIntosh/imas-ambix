#!/usr/bin/env python
"""Integrated flux-consumption ledger: back out the internal flux from
external measurements + the resistive model.

The shot-integrated identity — the surface flux swing (externally gauged:
flux loops + known coil/eddy fields through the whitened fit) splits into
inductive storage and resistive consumption:

    ΔΨ_bdry(t)  =  ΔΨ_internal(t)  +  ∫ V_res dt .

With the calibrated η(ψ_N) supplying V_res, the difference BACKS OUT the
internal flux content Ψ_int = ψ_bdry − ψ_axis — an interior quantity — from
external data.  This is the shot-integrated route to the interior: the
per-interval coefficient prior transfers only O(dt/τ_res) per step, while
the integrated resistive signal grows with the window (≈ 0.5–1 Wb per shot
against ≈ 0.1 Wb of read noise).  Ψ_int is li up to the Wpol integral (the
normalised internal inductance li3 = 4·Wpol/(μ0·Ip²·R0) is also computed
from the 1D state), and li combined with the magnetics moment βp + li/2
separates βp — the p′-family amplitude — from external data alone.

Validation (this report): on the tune shots, chain the budget-inferred
Ψ_int(t) from each shot's first slice and compare against every later
slice's OWN fitted flux content (each per-slice equilibrium provides
ψ_bdry − ψ_axis independently).  Agreement = the ledger closes and the li
readout is real; disagreement measures the η/model error the soft-prior
wiring would inherit.

Artifacts: imas_ambix/latent/artifacts/patch_gate/
           current_diffusion_flux_ledger[-tag].json
Figures:   docs/figures/temporal-physics-spine/
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

from imas_ambix.latent.current_diffusion import MU0, EtaProfile
from imas_ambix.latent.data import read_split_shot_lists
from scripts.current_diffusion_gate_eval import eval_shot, predict_interval
from scripts.spine_label_factory import frozen_spine_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("current_diffusion_flux_ledger")

ARTIFACTS = Path("imas_ambix/latent/artifacts/patch_gate")
FIGURES = Path("docs/figures/temporal-physics-spine")


def _wpol_li3(geo) -> float:
    """li3 = 4·Wpol/(μ0·Ip²·R0) from the 1D state (TORAX Wpol identity)."""
    dpsi = np.gradient(geo.psi_face, geo.rho_face, edge_order=2)
    vpr = np.clip(geo.vpr_face, 1e-30, None)
    bpol2 = (dpsi / (2.0 * np.pi)) ** 2 * geo.g2_face / vpr**2
    bpol2[0] = 0.0
    wpol = float(np.trapezoid(bpol2 * geo.vpr_face, geo.rho_face)) / (2.0 * MU0)
    return 4.0 * wpol / (MU0 * geo.ip_amperes**2 * geo.r0)


def shot_ledger(
    shot: int, args_d: dict, eta: EtaProfile, materials: dict | None = None
) -> dict | None:
    """Chain the budget-inferred internal flux along one shot."""
    if materials is None:
        out = eval_shot((int(shot), args_d))
        if out is None:
            return None
        m = out["materials"]
    else:
        m = materials
    geos = m["geos"]
    times = m["times"]
    spine, _ = frozen_spine_config()
    isolve = spine["interior_solve"]
    n_p, n_f = int(isolve["n_p"]), int(isolve["n_f"])
    nonneg = isolve["profile_kind"] == "monomial-nonneg"

    rows = []
    psi_int_budget = None
    for j in range(len(times)):
        geo = geos[j]
        if geo is None:
            continue
        span_fit = abs(geo.boundary_psi - geo.axis_psi)  # the slice's OWN read
        li3 = _wpol_li3(geo)
        if psi_int_budget is None:
            psi_int_budget = span_fit  # gauge the chain at the first slice
            v_res = 0.0
            d_bdry_meas = 0.0
        else:
            prev = rows[-1]
            pred = predict_interval(
                geos[prev["j"]],
                eta,
                t_start=times[prev["j"]],
                t_end=times[j],
                raw_times=m["raw_times"],
                ip_raw_amp=m["ip_raw_amp"],
                n_p=n_p,
                n_f=n_f,
                nonneg=nonneg,
                n_sub=int(args_d["n_sub_steps"]),
                par_weight=float(args_d["par_weight"]),
            )
            if pred is None:
                continue
            b = pred["budget"]
            # measured surface swing between the two slices' own fits
            d_bdry_meas = geo.boundary_psi - geos[prev["j"]].boundary_psi
            # resistive consumption over the interval (axis drift, model)
            v_res = b["d_psi_axis"]
            # ledger: dΨ_int = dΨ_bdry(meas) − dΨ_res(model), in flux content
            # (|span| convention: consumption REDUCES both; the sign algebra
            # collapses to the difference of the native deltas)
            psi_int_budget = psi_int_budget + (d_bdry_meas - v_res) * geo.flux_sign
        rows.append(
            {
                "j": j,
                "time_s": float(times[j]),
                "span_fit_wb": float(span_fit),
                "span_budget_wb": float(psi_int_budget),
                "li3_model": float(li3),
                "d_bdry_meas_wb": float(d_bdry_meas),
                "d_res_model_wb": float(v_res),
            }
        )
    if len(rows) < 3:
        return None
    fit = np.array([r["span_fit_wb"] for r in rows])
    bud = np.array([r["span_budget_wb"] for r in rows])
    return {
        "shot": int(shot),
        "rows": rows,
        "closure_rms_wb": float(np.sqrt(np.mean((fit - bud) ** 2))),
        "closure_corr": float(np.corrcoef(fit, bud)[0, 1]) if fit.size > 2 else None,
        "span_drift_fit_wb": float(fit[-1] - fit[0]),
        "span_drift_budget_wb": float(bud[-1] - bud[0]),
        "li3_median": float(np.median([r["li3_model"] for r in rows])),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=("tune", "eval"), default="tune")
    ap.add_argument("--eta-params", type=str, default="-6.3348,0.0,4.3827")
    ap.add_argument("--n-sub-steps", type=int, default=24)
    ap.add_argument("--par-weight", type=float, default=1.0)
    ap.add_argument("--b-phi0", type=float, default=0.55)
    ap.add_argument("--n-rho", type=int, default=24)
    ap.add_argument("--n-train", type=int, default=40)
    ap.add_argument("--n-heldout", type=int, default=8)
    ap.add_argument("--n-baseline-shots", type=int, default=10)
    ap.add_argument("--n-tune-shots", type=int, default=4)
    ap.add_argument("--nr", type=int, default=65)
    ap.add_argument("--nz", type=int, default=97)
    ap.add_argument("--max-slices-per-shot", type=int, default=16)
    ap.add_argument("--min-ip-ka", type=float, default=300.0)
    ap.add_argument("--convergence-limit", type=float, default=5e-3)
    ap.add_argument("--retry-max-iterations", type=int, default=160)
    ap.add_argument("--prior-weight", type=float, default=0.0)
    ap.add_argument(
        "--fit-eta",
        action="store_true",
        help="fit eta(psi_N) to CLOSE the ledger (pooled closure RMS over the "
        "split's shots) before reporting — the time-series identification "
        "that does not conflate per-slice fit jitter with profile relaxation",
    )
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out-suffix", type=str, default="")
    args = ap.parse_args()

    train_shots, held_shots = read_split_shot_lists(args.n_train, args.n_heldout)
    shots = (
        train_shots[args.n_baseline_shots : args.n_baseline_shots + args.n_tune_shots]
        if args.split == "tune"
        else held_shots
    )
    eta_vec = [float(v) for v in args.eta_params.split(",")]
    eta = EtaProfile.from_vector(np.asarray(eta_vec))
    args_d = vars(args) | {
        "mode": "eta-fit",  # pass-1 materials only
        "split": "train" if args.split == "tune" else "eval",
        "eta_vector": eta_vec,
    }
    ctx = multiprocessing.get_context("fork")
    # pass 1 chains once per shot (the expensive part); ledgers re-run cheaply
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
        mats = {
            int(s): r["materials"]
            for s, r in zip(
                shots,
                pool.map(eval_shot, [(int(s), args_d) for s in shots]),
                strict=True,
            )
            if r is not None
        }
    if not mats:
        raise SystemExit("no shots produced pass-1 materials")

    eta_fit_record = None
    if args.fit_eta:
        from scipy import optimize  # noqa: PLC0415

        def closure_objective(x: np.ndarray) -> float:
            cand = EtaProfile.from_vector(x)
            errs = []
            for s, m in mats.items():
                led = shot_ledger(s, args_d, cand, materials=m)
                if led is None:
                    errs.append(1.0)
                    continue
                errs.append(led["closure_rms_wb"] ** 2)
            return float(np.mean(errs))

        x0 = np.asarray(eta_vec, dtype=np.float64)
        res = optimize.minimize(
            closure_objective,
            x0,
            method="Nelder-Mead",
            options={"maxfev": 100, "xatol": 2e-2},
        )
        eta = EtaProfile.from_vector(res.x)
        eta_fit_record = {
            "eta_params": [float(v) for v in res.x],
            "eta0_ohm_m": eta.eta0,
            "contrast": eta.contrast,
            "shape": eta.shape,
            "closure_rms_pooled_wb": float(np.sqrt(res.fun)),
            "closure_rms_pooled_wb_at_x0": float(np.sqrt(closure_objective(x0))),
        }
        logger.info("closure-fit eta: %s", eta_fit_record)

    results = [
        r
        for r in (shot_ledger(s, args_d, eta, materials=m) for s, m in mats.items())
        if r is not None
    ]
    if not results:
        raise SystemExit("no shots produced a ledger")

    summary = {
        "arm": "current-diffusion-flux-ledger",
        "split": args.split,
        "eta_params": eta_vec,
        "shots": [r["shot"] for r in results],
        "closure_rms_wb": {str(r["shot"]): r["closure_rms_wb"] for r in results},
        "closure_corr": {str(r["shot"]): r["closure_corr"] for r in results},
        "span_drift_fit_wb": {str(r["shot"]): r["span_drift_fit_wb"] for r in results},
        "span_drift_budget_wb": {
            str(r["shot"]): r["span_drift_budget_wb"] for r in results
        },
        "li3_median": {str(r["shot"]): r["li3_median"] for r in results},
        "eta_closure_fit": eta_fit_record,
        "per_shot": {str(r["shot"]): r["rows"] for r in results},
    }
    tag = f"-{args.out_suffix}" if args.out_suffix else ""
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    out_json = ARTIFACTS / f"current_diffusion_flux_ledger{tag}.json"
    out_json.write_text(json.dumps(summary, indent=2))

    import matplotlib.pyplot as plt

    FIGURES.mkdir(parents=True, exist_ok=True)
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.8), squeeze=False)
    for k, r in enumerate(results):
        ax = axes[0][k]
        t = [row["time_s"] for row in r["rows"]]
        ax.plot(
            t,
            [row["span_fit_wb"] for row in r["rows"]],
            "o-",
            color="#4477aa",
            label="per-slice fit (measured-anchored)",
        )
        ax.plot(
            t,
            [row["span_budget_wb"] for row in r["rows"]],
            "s--",
            color="#cc6677",
            label="budget-inferred (ΔΨ_bdry − ∫V_res)",
        )
        ax.set_title(
            f"{r['shot']}: rms {r['closure_rms_wb'] * 1e3:.0f} mWb, "
            f"li3~{r['li3_median']:.2f}"
        )
        ax.set_xlabel("t [s]")
        ax.set_ylabel("Ψ_int = |ψ_bdry − ψ_axis| [Wb]")
        if k == 0:
            ax.legend(fontsize=7)
    fig.suptitle("Integrated flux ledger: internal flux backed out of external data")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out_fig = FIGURES / f"fig-current-diffusion-flux-ledger{tag}.png"
    fig.savefig(out_fig, dpi=120)
    plt.close(fig)
    logger.info(
        "flux ledger: closure rms %s mWb | %s %s",
        {k: round(v * 1e3) for k, v in summary["closure_rms_wb"].items()},
        out_json,
        out_fig,
    )
    return 0


def _worker(job):
    shot, args_d, eta = job
    try:
        return shot_ledger(shot, args_d, eta)
    except Exception as exc:  # noqa: BLE001 — one bad shot must not kill the report
        logger.warning("shot %s ledger failed: %s", shot, exc)
        return None


if __name__ == "__main__":
    raise SystemExit(main())
