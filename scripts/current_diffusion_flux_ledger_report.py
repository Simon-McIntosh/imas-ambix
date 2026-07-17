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

from imas_ambix.latent.current_diffusion import EtaProfile, wpol_li3
from imas_ambix.latent.data import read_split_shot_lists
from scripts.current_diffusion_gate_eval import eval_shot, predict_interval
from scripts.spine_label_factory import frozen_spine_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("current_diffusion_flux_ledger")

ARTIFACTS = Path("imas_ambix/latent/artifacts/patch_gate")
FIGURES = Path("docs/figures/temporal-physics-spine")


_wpol_li3 = wpol_li3  # shared library form; the local alias keeps call sites


def shot_ledger(
    shot: int,
    args_d: dict,
    eta: EtaProfile,
    materials: dict | None = None,
    *,
    swing: str = "fit",
    li3_sane_max: float | None = None,
    f_ni: float = 0.0,
) -> dict | None:
    """Chain the budget-inferred internal flux along one shot.

    ``swing`` selects the measured surface-swing anchor:

    * ``"fit"``   — the per-slice fits' own boundary flux (the original form:
      swing errors in the fit enter the budget AND the validation trace);
    * ``"floop"`` — the fit swing corrected by the drift of the fit's own
      flux-loop residual (measured − predicted on the fl_* channels), i.e.
      the swing is re-anchored to what the MEASURED loops actually did; the
      per-slice fit only supplies the (smooth, external-field-dominated)
      loop-to-boundary flux gap.

    ``li3_sane_max`` gates the GAUGE slice and the sane-subset metric: slices
    whose own 1D state reads an unphysical internal inductance (early-ramp
    fit pathologies read li3 of 2-3 against a physical 0.5-1.1) cannot gauge
    the chain or judge its closure.  All retained rows are still reported.

    ``f_ni`` scales the modelled resistive consumption by (1 − f_ni) — the
    bounded non-inductive-drive fraction (bootstrap/NBI carry part of the
    current, consuming no flux).
    """
    if materials is None:
        out = eval_shot((int(shot), args_d))
        if out is None:
            return None
        m = out["materials"]
    else:
        m = materials
    geos = m["geos"]
    times = m["times"]
    fl_resid = m.get("fl_resid_wb") or [None] * len(times)
    spine, _ = frozen_spine_config()
    isolve = spine["interior_solve"]
    n_p, n_f = int(isolve["n_p"]), int(isolve["n_f"])
    nonneg = isolve["profile_kind"] == "monomial-nonneg"

    def _sane(j: int) -> bool:
        if li3_sane_max is None:
            return True
        return _wpol_li3(geos[j]) <= li3_sane_max

    rows = []
    psi_int_budget = None
    n_anchor_missing = 0
    for j in range(len(times)):
        geo = geos[j]
        if geo is None:
            continue
        span_fit = abs(geo.boundary_psi - geo.axis_psi)  # the slice's OWN read
        li3 = _wpol_li3(geo)
        if psi_int_budget is None:
            if not _sane(j):
                continue  # an insane slice cannot gauge the chain
            psi_int_budget = span_fit  # gauge the chain at the first sane slice
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
                budget_only=True,
            )
            if pred is None:
                continue
            b = pred["budget"]
            # measured surface swing between the two slices' own fits
            d_bdry_meas = geo.boundary_psi - geos[prev["j"]].boundary_psi
            if swing == "floop":
                r0f, r1f = fl_resid[prev["j"]], fl_resid[j]
                if r0f is not None and r1f is not None:
                    # fit swing minus the fit's loop-residual drift = the
                    # measured-loop swing plus the fit's loop→boundary gap
                    d_bdry_meas = d_bdry_meas - (float(r1f) - float(r0f))
                else:
                    n_anchor_missing += 1
            # resistive consumption over the interval (axis drift, model),
            # scaled by the non-inductive fraction carrying no flux cost
            v_res = (1.0 - f_ni) * b["d_psi_axis"]
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
                "sane": bool(li3_sane_max is None or li3 <= li3_sane_max),
                "fl_resid_wb": None if fl_resid[j] is None else float(fl_resid[j]),
            }
        )
    if len(rows) < 3:
        return None
    fit = np.array([r["span_fit_wb"] for r in rows])
    bud = np.array([r["span_budget_wb"] for r in rows])
    sane = np.array([r["sane"] for r in rows], dtype=bool)
    rms_sane = (
        float(np.sqrt(np.mean((fit[sane] - bud[sane]) ** 2)))
        if sane.sum() >= 3
        else None
    )
    return {
        "shot": int(shot),
        "rows": rows,
        "closure_rms_wb": float(np.sqrt(np.mean((fit - bud) ** 2))),
        "closure_rms_sane_wb": rms_sane,
        "n_sane": int(sane.sum()),
        "n_rows": len(rows),
        "n_anchor_missing": int(n_anchor_missing),
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
    ap.add_argument(
        "--swing",
        choices=("fit", "floop"),
        default="fit",
        help="measured surface-swing anchor: the fits' own boundary flux, or "
        "the flux-loop re-anchored form (fit swing minus the fit's fl-residual "
        "drift)",
    )
    ap.add_argument(
        "--li3-sane-max",
        type=float,
        default=None,
        help="gauge/metric sanity gate on the per-slice li3 read (unphysical "
        "early-ramp fits read 2-3); None keeps every slice (original form)",
    )
    ap.add_argument(
        "--fit-nonind",
        action="store_true",
        help="fit a bounded non-inductive fraction f_ni (resistive consumption "
        "scaled by 1-f_ni) jointly with eta in the closure fit",
    )
    ap.add_argument(
        "--attribution",
        action="store_true",
        help="run the residual-attribution arm suite (baseline / sane-gauge / "
        "floop-anchor / +nonind), each with its own closure-fitted eta, and "
        "write the comparison artifact + figure",
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

    def closure_fit(
        swing: str, li3_sane_max: float | None, fit_nonind: bool
    ) -> tuple[EtaProfile, float, dict]:
        """Closure-fit η (+ optional f_ni) for one arm, multi-start on shape."""
        from scipy import optimize  # noqa: PLC0415

        def objective(x: np.ndarray) -> float:
            cand = EtaProfile.from_vector(x[:3])
            fni = float(np.clip(x[3], 0.0, 0.6)) if fit_nonind else 0.0
            errs = []
            for s, m in mats.items():
                led = shot_ledger(
                    s,
                    args_d,
                    cand,
                    materials=m,
                    swing=swing,
                    li3_sane_max=li3_sane_max,
                    f_ni=fni,
                )
                if led is None:
                    errs.append(1.0)
                    continue
                rms = (
                    led["closure_rms_sane_wb"]
                    if li3_sane_max is not None and led["closure_rms_sane_wb"]
                    else led["closure_rms_wb"]
                )
                errs.append(rms**2)
            return float(np.mean(errs))

        x0 = np.asarray(eta_vec, dtype=np.float64)
        starts = [x0, np.array([x0[0], 2.0, 2.0]), np.array([x0[0], 4.0, 1.0])]
        best = None
        for xs in starts:
            xs_full = np.concatenate([xs, [0.15]]) if fit_nonind else xs
            res = optimize.minimize(
                objective,
                xs_full,
                method="Nelder-Mead",
                options={"maxfev": 140, "xatol": 2e-2},
            )
            if best is None or res.fun < best.fun:
                best = res
        eta_best = EtaProfile.from_vector(best.x[:3])
        fni_best = float(np.clip(best.x[3], 0.0, 0.6)) if fit_nonind else 0.0
        record = {
            "eta_params": [float(v) for v in best.x[:3]],
            "eta0_ohm_m": eta_best.eta0,
            "contrast": eta_best.contrast,
            "shape": eta_best.shape,
            "f_ni": fni_best,
            "closure_rms_pooled_wb": float(np.sqrt(best.fun)),
            "closure_rms_pooled_wb_at_x0": float(
                np.sqrt(objective(np.concatenate([x0, [0.0]]) if fit_nonind else x0))
            ),
        }
        return eta_best, fni_best, record

    def arm_ledgers(
        cand: EtaProfile, swing: str, li3_sane_max: float | None, fni: float
    ) -> list[dict]:
        return [
            r
            for r in (
                shot_ledger(
                    s,
                    args_d,
                    cand,
                    materials=m,
                    swing=swing,
                    li3_sane_max=li3_sane_max,
                    f_ni=fni,
                )
                for s, m in mats.items()
            )
            if r is not None
        ]

    if args.attribution:
        return run_attribution(args, args_d, mats, eta_vec, closure_fit, arm_ledgers)

    eta_fit_record = None
    f_ni = 0.0
    if args.fit_eta:
        eta, f_ni, eta_fit_record = closure_fit(
            args.swing, args.li3_sane_max, args.fit_nonind
        )
        logger.info("closure-fit eta: %s", eta_fit_record)

    results = arm_ledgers(eta, args.swing, args.li3_sane_max, f_ni)
    if not results:
        raise SystemExit("no shots produced a ledger")

    summary = {
        "arm": "current-diffusion-flux-ledger",
        "split": args.split,
        "eta_params": eta_vec,
        "swing": args.swing,
        "li3_sane_max": args.li3_sane_max,
        "f_ni": f_ni,
        "shots": [r["shot"] for r in results],
        "closure_rms_wb": {str(r["shot"]): r["closure_rms_wb"] for r in results},
        "closure_rms_sane_wb": {
            str(r["shot"]): r["closure_rms_sane_wb"] for r in results
        },
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


def run_attribution(args, args_d, mats, eta_vec, closure_fit, arm_ledgers) -> int:
    """Residual-attribution arm suite: which candidate closes the ledger?

    Arms (each with its own closure-fitted η, shape multi-started so the
    flat-η restriction is genuinely released):

    * ``baseline``     — the landed form (fit swing, first-slice gauge);
    * ``sane-gauge``   — the gauge and the metric exclude slices whose own
      li3 read is unphysical (early-ramp fit pathologies);
    * ``floop-anchor`` — the surface swing re-anchored to the measured flux
      loops (candidate: per-slice boundary-flux read drift);
    * ``+nonind``      — floop-anchor plus the bounded non-inductive
      fraction (candidate: neglected non-inductive drive).

    The direct diagnostic (independent of any arm): the drift of each fit's
    flux-loop residual over the shot — if the fits' surface swing disagreed
    with the measured loops, this trace carries the disagreement in Wb.
    """
    arms = [
        ("baseline", dict(swing="fit", li3_sane_max=None, fit_nonind=False)),
        ("sane-gauge", dict(swing="fit", li3_sane_max=1.5, fit_nonind=False)),
        ("floop-anchor", dict(swing="floop", li3_sane_max=1.5, fit_nonind=False)),
        ("+nonind", dict(swing="floop", li3_sane_max=1.5, fit_nonind=True)),
    ]
    arm_out = {}
    ledgers_by_arm = {}
    for name, cfg in arms:
        eta_a, fni_a, rec = closure_fit(
            cfg["swing"], cfg["li3_sane_max"], cfg["fit_nonind"]
        )
        led = arm_ledgers(eta_a, cfg["swing"], cfg["li3_sane_max"], fni_a)
        ledgers_by_arm[name] = led
        arm_out[name] = {
            "config": cfg,
            "eta_closure_fit": rec,
            "closure_rms_wb": {str(r["shot"]): r["closure_rms_wb"] for r in led},
            "closure_rms_sane_wb": {
                str(r["shot"]): r["closure_rms_sane_wb"] for r in led
            },
            "closure_corr": {str(r["shot"]): r["closure_corr"] for r in led},
            "n_anchor_missing": {str(r["shot"]): r["n_anchor_missing"] for r in led},
        }
        logger.info(
            "arm %-13s pooled %s mWb | per-shot %s",
            name,
            round(rec["closure_rms_pooled_wb"] * 1e3),
            {k: round(v * 1e3) for k, v in arm_out[name]["closure_rms_wb"].items()},
        )

    # direct diagnostic: per-shot fl-residual drift (from the baseline rows)
    fl_drift = {}
    for r in ledgers_by_arm["baseline"]:
        res = [
            row["fl_resid_wb"] for row in r["rows"] if row["fl_resid_wb"] is not None
        ]
        bud_minus_fit = [
            row["span_budget_wb"] - row["span_fit_wb"] for row in r["rows"]
        ]
        drift = None
        corr = None
        if len(res) >= 3:
            res_arr = np.asarray(res)
            drift = float(res_arr.max() - res_arr.min())
            paired = [
                (row["span_budget_wb"] - row["span_fit_wb"], row["fl_resid_wb"])
                for row in r["rows"]
                if row["fl_resid_wb"] is not None
            ]
            if len(paired) >= 3:
                bm, fr = np.asarray(paired).T
                corr = float(np.corrcoef(bm, fr)[0, 1])
        fl_drift[str(r["shot"])] = {
            "fl_resid_drift_range_wb": drift,
            "corr_budget_minus_fit_vs_fl_resid": corr,
            "budget_minus_fit_final_wb": float(bud_minus_fit[-1]),
        }

    summary = {
        "arm": "current-diffusion-flux-ledger-attribution",
        "split": args.split,
        "eta_params_x0": eta_vec,
        "shots": sorted({r["shot"] for r in ledgers_by_arm["baseline"]}),
        "arms": arm_out,
        "fl_residual_diagnostic": fl_drift,
        "per_shot_rows": {
            name: {str(r["shot"]): r["rows"] for r in led}
            for name, led in ledgers_by_arm.items()
        },
    }
    tag = f"-{args.out_suffix}" if args.out_suffix else ""
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    out_json = ARTIFACTS / f"current_diffusion_flux_ledger_attribution{tag}.json"
    out_json.write_text(json.dumps(summary, indent=2))

    import matplotlib.pyplot as plt

    FIGURES.mkdir(parents=True, exist_ok=True)
    shots = summary["shots"]
    n = len(shots)
    fig, axes = plt.subplots(2, n, figsize=(4.4 * n, 7.2), squeeze=False)
    for k, shot in enumerate(shots):
        ax = axes[0][k]
        base = next(r for r in ledgers_by_arm["baseline"] if r["shot"] == shot)
        t0 = [row["time_s"] for row in base["rows"]]
        ax.plot(
            t0,
            [row["span_fit_wb"] for row in base["rows"]],
            "o-",
            color="#4477aa",
            label="per-slice fit",
        )
        ax.plot(
            t0,
            [row["span_budget_wb"] for row in base["rows"]],
            "s--",
            color="#cc6677",
            label="budget (baseline)",
        )
        anch = next(
            (r for r in ledgers_by_arm["floop-anchor"] if r["shot"] == shot), None
        )
        if anch is not None:
            ax.plot(
                [row["time_s"] for row in anch["rows"]],
                [row["span_budget_wb"] for row in anch["rows"]],
                "d-.",
                color="#228833",
                label="budget (fl-anchored)",
            )
        ax.set_title(
            f"{shot}: rms {base['closure_rms_wb'] * 1e3:.0f} → "
            f"{(anch['closure_rms_sane_wb'] or anch['closure_rms_wb']) * 1e3:.0f} mWb"
            if anch
            else f"{shot}"
        )
        ax.set_xlabel("t [s]")
        ax.set_ylabel("Ψ_int [Wb]")
        if k == 0:
            ax.legend(fontsize=7)
        ax2 = axes[1][k]
        tt = [row["time_s"] for row in base["rows"] if row["fl_resid_wb"] is not None]
        rr = [
            row["fl_resid_wb"] for row in base["rows"] if row["fl_resid_wb"] is not None
        ]
        ax2.plot(tt, rr, "o-", color="#aa3377", label="fit fl residual (pred−meas)")
        ax2.plot(
            t0,
            [row["span_budget_wb"] - row["span_fit_wb"] for row in base["rows"]],
            "s--",
            color="#66ccee",
            label="budget − fit (baseline)",
        )
        ax2.axhline(0.0, color="0.6", lw=0.8)
        ax2.set_xlabel("t [s]")
        ax2.set_ylabel("[Wb]")
        if k == 0:
            ax2.legend(fontsize=7)
    fig.suptitle(
        "Ledger residual attribution: swing anchor, gauge sanity, non-inductive arm"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_fig = FIGURES / f"fig-current-diffusion-ledger-attribution{tag}.png"
    fig.savefig(out_fig, dpi=120)
    plt.close(fig)
    logger.info("attribution: %s %s", out_json, out_fig)
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
