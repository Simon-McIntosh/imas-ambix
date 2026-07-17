#!/usr/bin/env python
"""Current-diffusion temporal prior gate: diffusion-chained profile fits vs
the frozen spine.

The per-slice static fit treats successive profile-coefficient fits as
independent, discarding the resistive flux-diffusion physics whose boundary
conditions are MEASURED (Rogowski Ip; the surface flux swing).  This gate
promotes that physics to a soft temporal-consistency prior:

* pass 1 — the frozen classical spine, slice by slice, byte-exact (the
  paired baseline; supplies each slice's converged ψ and coefficients);
* per-interval evolution — each scored slice's 1D flux-surface geometry is
  extracted from its own equilibrium (``flux_surface_geometry``); the ψ
  diffusion (``diffuse_psi``) integrates over the interval to the next slice
  at sub-label cadence with the measured Ip as the edge BC and η(ψ_N) the
  bounded cross-shot unknown; the evolved (j_tot, ⟨J·B⟩) profiles project
  onto the ladder basis (``project_coefficients``) — the coefficient
  prediction for the NEXT slice;
* pass 2 — every slice (except each shot's first) refit with its
  coefficients soft-centred on the diffusion prediction
  (``SoftPriors.coeff_prior_*``); weight swept on the tune split only.

Flux-consumption ledger (per shot, reported alongside the gate): the
predicted surface swing splits into RESISTIVE (axis flux drift — Ohm's law
at the axis) and INDUCTIVE (internal flux storage) channels; the measured
surface swing comes from the spine fits' own boundary flux (anchored to the
73 measured magnetics, flux loops included).  The windowed Ejima coefficient
C_E = |ΔΨ_res|/(μ0·R0·|ΔIp|) over each shot's ramp is the byproduct.

η(ψ_N) calibration (``--eta-fit``, tune split only): the 3 bounded
parameters minimise the pooled coefficient-prediction error against the
spine's own next-slice fits plus the surface-flux budget mismatch — the
measured V_loop anchor.  The eval split consumes the frozen parameters.

Gate P2 (real-data leg): flat-top LCFS non-inferiority (−0.05 skill margin)
AND axis median not degraded, n ≥ 128 held-out paired evaluation, byte-exact
spine reproduction.  Firewall: EFIT referee-only; all drives measured;
hybrid_greens only.

Artifacts: imas_ambix/latent/artifacts/patch_gate/current_diffusion_gate_eval[-tag].json
Figures:   docs/figures/temporal-physics-spine/
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np

from imas_ambix.gs.geometry import GEOMETRY_TABLE_VERSION
from imas_ambix.gs.operator import COIL_MODEL_VERSION
from imas_ambix.latent.current_diffusion import (
    EtaProfile,
    basis_projection_images,
    diffuse_psi,
    ejima_coefficient,
    flux_budget,
    flux_surface_geometry,
    predicted_current,
    project_coefficients,
)
from imas_ambix.latent.data import (
    anchored_columns,
    feature_schema,
    load_shot_slices_raw,
    read_split_shot_lists,
    schema_group_offsets,
)
from scripts.closure_gate_eval import _shot_passive_sidecar, fit_and_read_slice
from scripts.patch_gate_eval import (
    _bootstrap_skill_draws,
    _percentile_ci,
    score,
    shot_payloads,
    train_mean_baseline,
)
from scripts.spine_label_factory import frozen_spine_config
from scripts.temporal_operator_gate_eval import lcfs_rms_cm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("current_diffusion_gate_eval")

ARTIFACTS = Path("imas_ambix/latent/artifacts/patch_gate")
FIGURES = Path("docs/figures/temporal-physics-spine")


def raw_ip_stream(shot: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Full-shot 1 kHz measured plasma current [same units as the channel].

    Interior NaN samples are interpolated so a dropout never fabricates a
    current step; the caller rescales to amperes against the slice payloads
    (the channel is kA by convention — the rescale is measured, not assumed).
    """
    schema = feature_schema()
    loaded = load_shot_slices_raw(shot, schema)
    if loaded is None:
        return None
    x, times, _plasma_on = loaded
    ip_col, _ne = anchored_columns(schema)
    ip = np.abs(np.asarray(x[:, ip_col], dtype=np.float64))
    ok = np.isfinite(ip)
    if not ok.any():
        return None
    if not ok.all():
        ip = np.interp(times, times[ok], ip[ok])
    return np.asarray(times, dtype=np.float64), ip


def raw_flux_loop_vloop(shot: int, smooth_ms: float = 25.0) -> dict | None:
    """Diagnostic raw-cadence flux-loop loop-voltage trace (median over loops).

    dΨ_loop/dt per measured ``fl_*`` channel, box-smoothed, medianed across
    loops — the UNCORRECTED wall-side V_loop (coil + eddy + plasma swing).
    Reported as a figure trace only; the budget's measured anchor is the
    spine fits' boundary flux (which consumes the same loops through the
    whitened magnetics).
    """
    schema = feature_schema()
    loaded = load_shot_slices_raw(shot, schema)
    if loaded is None:
        return None
    x, times, _ = loaded
    off = schema_group_offsets(schema)["amb"]
    names = schema["amb"]
    dt = float(np.median(np.diff(times)))
    n_box = max(3, int(round(smooth_ms * 1e-3 / dt)))
    kernel = np.ones(n_box) / n_box
    v_list = []
    for j, name in enumerate(names):
        if not str(name).lower().startswith("fl"):
            continue
        sig = np.asarray(x[:, off + j], dtype=np.float64)
        ok = np.isfinite(sig)
        if ok.sum() < 10 * n_box:
            continue
        sig = np.interp(times, times[ok], sig[ok])
        sig = np.convolve(sig, kernel, mode="same")
        v_list.append(np.gradient(sig, times))
    if not v_list:
        return None
    return {"times": times, "v_loop_median": np.median(np.stack(v_list), axis=0)}


def _slice_geometry(fit, grid, isolve, b_phi0: float, n_rho: int):
    """Flux-surface geometry of one scored pass-1 slice (None = skip prior)."""
    if fit.psi is None or fit.coeffs is None:
        return None
    return flux_surface_geometry(
        fit.psi,
        grid,
        coeffs=np.asarray(fit.coeffs, dtype=np.float64),
        ip_amperes=abs(float(fit.ip_amperes)),
        n_p=int(isolve["n_p"]),
        n_f=int(isolve["n_f"]),
        nonneg=isolve["profile_kind"] == "monomial-nonneg",
        b_phi0=b_phi0,
        n_rho=n_rho,
    )


def predict_interval(
    geo,
    eta: EtaProfile,
    *,
    t_start: float,
    t_end: float,
    raw_times: np.ndarray,
    ip_raw_amp: np.ndarray,
    n_p: int,
    n_f: int,
    nonneg: bool,
    n_sub: int,
    par_weight: float,
) -> dict | None:
    """Evolve one label interval and project the next-slice coefficients."""
    t_sub = np.linspace(t_start, t_end, max(2, n_sub))
    ip_sub = np.interp(t_sub, raw_times, ip_raw_amp)
    step = diffuse_psi(geo, eta, t_grid=t_sub, ip_of_t=ip_sub)
    pred = predicted_current(geo, step["psi_face"][-1], step["psidot_face"], eta)
    images = basis_projection_images(geo, geo.s_k, n_p=n_p, n_f=n_f, nonneg=nonneg)
    c_pred = project_coefficients(
        geo,
        images,
        pred["j_tor"],
        pred["j_par_b"],
        nonneg=nonneg,
        par_weight=par_weight,
    )
    if c_pred is None:
        return None
    return {"c_pred": c_pred, "budget": flux_budget(step, geo), "step": step}


def eval_shot(job: tuple) -> dict | None:
    """One shot: byte-exact spine pass + diffusion-chained refit pass."""
    shot, args_d = job
    spine, config_sha = frozen_spine_config()
    isolve = spine["interior_solve"]
    spc = dict(spine["soft_priors"])

    payload = shot_payloads(
        shot,
        nr=args_d["nr"],
        nz=args_d["nz"],
        max_slices=args_d["max_slices_per_shot"],
        min_ip_ka=args_d["min_ip_ka"],
        split=args_d["split"],
    )
    if payload is None:
        return None
    grid, table, basis = payload["grid"], payload["table"], payload["basis"]
    sidecar = _shot_passive_sidecar(payload, int(isolve["passive_k"]))

    raw = raw_ip_stream(shot)
    if raw is None:
        logger.warning("shot %s: no raw Ip stream — skipped", shot)
        return None
    raw_times, ip_raw = raw

    fit_kw = dict(
        beta0_grid=(0.5,),
        alpha_grid=(1.0,),
        cost_limit=float("inf"),
        convergence_limit=args_d["convergence_limit"],
        retry_max_iterations=args_d["retry_max_iterations"],
        fit_mode="ladder",
        n_p=int(isolve["n_p"]),
        n_f=int(isolve["n_f"]),
        smoothness=float(isolve["smoothness"]),
        nonneg=isolve["profile_kind"] == "monomial-nonneg",
        passive=sidecar,
        passive_ridge=1.0,
        reseed_axis_r_max=float(isolve["reseed_axis_r_max"]),
        basis=basis,
        meta={},
        soft_prior_cfg=spc,
        boundary_read=isolve["boundary_read_scoring"],
    )
    n_p, n_f = int(isolve["n_p"]), int(isolve["n_f"])
    nonneg = isolve["profile_kind"] == "monomial-nonneg"

    # ---- pass 1: frozen spine chain, byte-exact (the paired baseline) ----
    t0 = time.perf_counter()
    slices: list[dict] = []
    n_masked = 0
    warm_jphi = None
    order = np.argsort([p.time_s for p in payload["payloads"]])
    for k in order:
        p = payload["payloads"][int(k)]
        f = fit_and_read_slice(
            grid, table, p, warm_jphi=warm_jphi, keep_jphi=True, keep_psi=True, **fit_kw
        )
        if not f.scored:
            n_masked += 1
            continue
        if f.converged:  # frozen-chain semantics: strict-converged only
            warm_jphi = f.jphi_flat
        slices.append(
            {
                "k": int(k),
                "t_index": p.t_index,
                "time_s": p.time_s,
                "ip": p.ip_amperes,
                "payload": p,
                "fit": f,
                "ref": payload["refs"][int(k)],
            }
        )
    if len(slices) < 2:
        return None
    wall_pass1 = time.perf_counter() - t0

    # measured raw-Ip → amperes rescale (the channel convention is kA; fit it)
    lab_ip = np.array([abs(s["ip"]) for s in slices])
    raw_at_lab = np.interp([s["time_s"] for s in slices], raw_times, ip_raw)
    good = raw_at_lab > 0
    ip_scale = float(np.median(lab_ip[good] / raw_at_lab[good])) if good.any() else 1e3
    ip_raw_amp = ip_raw * ip_scale

    eta = EtaProfile.from_vector(np.asarray(args_d["eta_vector"], dtype=np.float64))
    n_sub = int(args_d["n_sub_steps"])
    par_weight = float(args_d["par_weight"])
    b_phi0 = float(args_d["b_phi0"])
    n_rho = int(args_d["n_rho"])
    prior_weight = float(args_d["prior_weight"])

    # ---- per-interval geometry + diffusion predictions (from pass 1) ----
    geos = [_slice_geometry(s["fit"], grid, isolve, b_phi0, n_rho) for s in slices]

    if args_d.get("mode") == "eta-fit":
        # return the pass-1 materials — the η optimisation re-runs the cheap
        # diffusion predictions per candidate in the parent process
        return {
            "shot": int(shot),
            "materials": {
                "geos": geos,
                "times": [s["time_s"] for s in slices],
                "coeffs": [list(map(float, s["fit"].coeffs or [])) for s in slices],
                "raw_times": raw_times,
                "ip_raw_amp": ip_raw_amp,
                "boundary_psi": [
                    None if g is None else float(g.boundary_psi) for g in geos
                ],
            },
            "config_sha": config_sha,
            "wall_s": time.perf_counter() - t0,
        }
    preds: list[dict | None] = [None] * len(slices)
    budgets = []
    for j in range(len(slices) - 1):
        if geos[j] is None:
            continue
        out = predict_interval(
            geos[j],
            eta,
            t_start=slices[j]["time_s"],
            t_end=slices[j + 1]["time_s"],
            raw_times=raw_times,
            ip_raw_amp=ip_raw_amp,
            n_p=n_p,
            n_f=n_f,
            nonneg=nonneg,
            n_sub=n_sub,
            par_weight=par_weight,
        )
        if out is None:
            continue
        preds[j + 1] = out
        # measured surface swing: the spine fits' own boundary flux
        meas_dpsi = (
            float(geos[j + 1].boundary_psi - geos[j].boundary_psi)
            if geos[j + 1] is not None
            else float("nan")
        )
        budgets.append(
            out["budget"]
            | {
                "t0": slices[j]["time_s"],
                "t1": slices[j + 1]["time_s"],
                "d_ip": float(abs(slices[j + 1]["ip"]) - abs(slices[j]["ip"])),
                "d_psi_bdry_meas": meas_dpsi,
                "c_pred": [float(c) for c in out["c_pred"]],
                "c_spine_next": [float(c) for c in (slices[j + 1]["fit"].coeffs or [])],
            }
        )

    # ---- pass 2: refit with the diffusion coefficient prior ----
    rows: list[dict] = []
    for j, s in enumerate(slices):
        f1 = s["fit"]
        if preds[j] is None or prior_weight <= 0.0:
            # first slice / no prediction: the spine fit IS the dynamic arm
            f2 = f1
            c_pred = None
        else:
            c_pred = preds[j]["c_pred"]
            f2 = fit_and_read_slice(
                grid,
                table,
                s["payload"],
                warm_jphi=f1.jphi_flat,
                keep_jphi=True,
                coeff_prior=(c_pred, prior_weight),
                **fit_kw,
            )
        if not f2.scored:
            rows.append({"scored": False, "t_index": s["t_index"], "reason": f2.reason})
            continue
        rows.append(
            {
                "scored": True,
                "t_index": s["t_index"],
                "time_s": s["time_s"],
                "ip": s["ip"],
                "target_spine": f1.target,
                "target_dyn": f2.target,
                "ref": s["ref"],
                "cost_spine": float(f1.cost),
                "cost_dyn": float(f2.cost),
                "c_pred": None if c_pred is None else [float(c) for c in c_pred],
                "c_spine": [float(c) for c in (f1.coeffs or [])],
                "c_dyn": [float(c) for c in (f2.coeffs or [])],
                "reseeded_spine": f1.reason == "scored-reseeded",
                "reseeded_dyn": f2.reason == "scored-reseeded",
            }
        )

    # ---- per-shot flux ledger + windowed Ejima over the covered ramp ----
    ramp = [b for b in budgets if b["d_ip"] > 0]
    d_psi_res = sum(b["d_psi_axis"] for b in ramp)
    d_ip_ramp = sum(b["d_ip"] for b in ramp)
    ip_max = max(abs(s["ip"]) for s in slices)
    span_s = sum(b["t1"] - b["t0"] for b in budgets)
    # the Ejima normalisation is only meaningful when the covered window
    # actually spans a ramp — on a flat-top window ΔIp is fluctuation-scale
    # while the resistive consumption keeps integrating, and the ratio
    # diverges; report the resistive loop voltage instead there
    ramp_covered = d_ip_ramp >= 0.2 * ip_max
    ledger = {
        "n_intervals": len(budgets),
        "d_psi_res_ramp": float(d_psi_res),
        "d_psi_internal_ramp": float(sum(b["d_psi_internal"] for b in ramp)),
        "d_psi_bdry_pred_ramp": float(sum(b["d_psi_bdry"] for b in ramp)),
        "d_psi_bdry_meas_ramp": float(
            np.nansum([b["d_psi_bdry_meas"] for b in ramp]) if ramp else 0.0
        ),
        "d_ip_ramp": float(d_ip_ramp),
        "ip_max": float(ip_max),
        "ramp_covered": bool(ramp_covered),
        "v_res_mean": (
            float(abs(sum(b["d_psi_axis"] for b in budgets)) / span_s)
            if span_s > 0
            else None
        ),
        "ejima_windowed": (
            ejima_coefficient(d_psi_res, d_ip_ramp, grid.r0) if ramp_covered else None
        ),
    }
    return {
        "shot": int(shot),
        "rows": rows,
        "budgets": budgets,
        "ledger": ledger,
        "ip_scale": ip_scale,
        "n_masked_spine": n_masked,
        "n_geo_missing": int(sum(g is None for g in geos)),
        "config_sha": config_sha,
        "wall_s": time.perf_counter() - t0,
        "wall_pass1_s": wall_pass1,
    }


def fit_eta_on_pool(shot_results: list[dict], budget_weight: float) -> dict:
    """Cross-shot η(ψ_N) fit — REPORTING helper on precomputed budgets.

    The gate script's η fit re-runs `predict_interval` per candidate, which
    needs the geometries; that lives in `--eta-fit` mode inside main() via
    the worker pool.  This helper only summarises prediction quality at the
    frozen η for the artifact.
    """
    err = []
    bud = []
    for sr in shot_results:
        for b in sr["budgets"]:
            c_pred = np.asarray(b["c_pred"])
            c_next = np.asarray(b["c_spine_next"])
            if c_pred.size and c_next.size == c_pred.size:
                err.append(float(np.linalg.norm(c_pred - c_next)))
            if np.isfinite(b["d_psi_bdry_meas"]):
                bud.append(float(b["d_psi_bdry"] - b["d_psi_bdry_meas"]))
    return {
        "coeff_pred_err_median": float(np.median(err)) if err else None,
        "budget_mismatch_wb_median": float(np.median(np.abs(bud))) if bud else None,
        "budget_weight": budget_weight,
    }


def run_eta_fit(shot_results: list[dict], args) -> dict:
    """Cross-shot bounded η(ψ_N) fit on the pooled tune intervals.

    Objective: mean squared coefficient-prediction error against the spine's
    own next-slice fits + ``--eta-budget-weight`` × the normalised surface-
    flux budget mismatch (predicted vs fit-anchored measured ΔΨ_bdry — the
    measured V_loop anchor).  Nelder–Mead over the 3 bounded parameters.
    """
    from scipy import optimize  # noqa: PLC0415

    intervals = []
    for sr in shot_results:
        m = sr["materials"]
        for j in range(len(m["times"]) - 1):
            geo = m["geos"][j]
            if geo is None or not m["coeffs"][j + 1]:
                continue
            meas = (
                m["boundary_psi"][j + 1] - m["boundary_psi"][j]
                if m["boundary_psi"][j + 1] is not None
                and m["boundary_psi"][j] is not None
                else float("nan")
            )
            intervals.append(
                {
                    "shot": int(sr["shot"]),
                    "geo": geo,
                    "t0": m["times"][j],
                    "t1": m["times"][j + 1],
                    "raw_times": m["raw_times"],
                    "ip_raw_amp": m["ip_raw_amp"],
                    "c_next": np.asarray(m["coeffs"][j + 1], dtype=np.float64),
                    "d_psi_meas": meas,
                }
            )
    if not intervals:
        raise SystemExit("no usable intervals for the eta fit")
    # the measured surface-flux anchor works at the PER-SHOT SPAN: single-
    # interval boundary-flux differences at ~20 ms label cadence are read-
    # noise-dominated (the fit-read flux jitters slice to slice), and letting
    # them into the objective rails eta at the relaxed-attractor corner; the
    # span-integrated swing rises above the read noise.
    span_meas: dict[int, float] = {}
    for iv in intervals:
        if np.isfinite(iv["d_psi_meas"]):
            span_meas[iv["shot"]] = span_meas.get(iv["shot"], 0.0) + iv["d_psi_meas"]
    meas_scale = (
        float(np.median([abs(v) for v in span_meas.values()])) if span_meas else 1.0
    )
    if not np.isfinite(meas_scale) or meas_scale <= 0:
        meas_scale = 1.0
    spine, _sha = frozen_spine_config()
    isolve = spine["interior_solve"]
    n_p, n_f = int(isolve["n_p"]), int(isolve["n_f"])
    nonneg = isolve["profile_kind"] == "monomial-nonneg"
    bw = float(args.eta_budget_weight)

    trace = []

    def objective(x: np.ndarray) -> float:
        eta = EtaProfile.from_vector(x)
        err_c = []
        span_pred: dict[int, float] = {}
        for iv in intervals:
            out = predict_interval(
                iv["geo"],
                eta,
                t_start=iv["t0"],
                t_end=iv["t1"],
                raw_times=iv["raw_times"],
                ip_raw_amp=iv["ip_raw_amp"],
                n_p=n_p,
                n_f=n_f,
                nonneg=nonneg,
                n_sub=int(args.n_sub_steps),
                par_weight=float(args.par_weight),
            )
            if out is None:
                err_c.append(4.0)  # a failed prediction is a bad candidate
                continue
            c_pred = out["c_pred"]
            if c_pred.size == iv["c_next"].size:
                err_c.append(float(np.mean((c_pred - iv["c_next"]) ** 2)))
            if np.isfinite(iv["d_psi_meas"]):
                span_pred[iv["shot"]] = (
                    span_pred.get(iv["shot"], 0.0) + out["budget"]["d_psi_bdry"]
                )
        err_b = [
            ((span_pred[k] - span_meas[k]) / meas_scale) ** 2
            for k in span_pred
            if k in span_meas
        ]
        val = float(np.mean(err_c)) + bw * (float(np.mean(err_b)) if err_b else 0.0)
        trace.append({"x": [float(v) for v in x], "objective": val})
        return val

    x0 = np.asarray([float(v) for v in args.eta_params.split(",")], dtype=np.float64)
    res = optimize.minimize(
        objective, x0, method="Nelder-Mead", options={"maxfev": 120, "xatol": 2e-2}
    )
    eta_best = EtaProfile.from_vector(res.x)
    out = {
        "eta_params": [float(v) for v in res.x],
        "eta0_ohm_m": eta_best.eta0,
        "contrast": eta_best.contrast,
        "shape": eta_best.shape,
        "objective": float(res.fun),
        "objective_at_x0": float(trace[0]["objective"]) if trace else None,
        "n_intervals": len(intervals),
        "n_evals": len(trace),
        "budget_weight": bw,
        "budget_scale_wb": meas_scale,
        "pool_shots": [int(sr["shot"]) for sr in shot_results],
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS / "current_diffusion_eta_calibration.json"
    path.write_text(json.dumps(out, indent=2))
    logger.info(
        "eta fit: %s -> %s (obj %.4g -> %.4g, %d intervals) -> %s",
        [float(v) for v in x0],
        out["eta_params"],
        out["objective_at_x0"] or float("nan"),
        out["objective"],
        len(intervals),
        path,
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("gate", "eta-fit"), default="gate")
    ap.add_argument("--split", choices=("tune", "eval"), default="eval")
    ap.add_argument("--eta-budget-weight", type=float, default=0.3)
    ap.add_argument("--prior-weight", type=float, default=0.3)
    ap.add_argument(
        "--eta-params",
        type=str,
        default="-7.3,2.0,2.0",
        help="log10(eta0 [Ohm.m]), contrast, shape — cross-shot frozen "
        "(fit on tune with scripts/current_diffusion_eta_fit.py)",
    )
    ap.add_argument("--b-phi0", type=float, default=0.55)
    ap.add_argument("--n-rho", type=int, default=24)
    ap.add_argument("--n-sub-steps", type=int, default=24)
    ap.add_argument("--par-weight", type=float, default=1.0)
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
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--axis-median-tol-cm", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--shots", type=str, default="")
    ap.add_argument("--out-suffix", type=str, default="")
    args = ap.parse_args()

    train_shots, held_shots = read_split_shot_lists(args.n_train, args.n_heldout)
    if args.split == "tune":
        eval_shots = train_shots[
            args.n_baseline_shots : args.n_baseline_shots + args.n_tune_shots
        ]
        payload_split = "train"
    else:
        eval_shots = held_shots
        payload_split = "eval"
    if args.shots:
        want = {int(s) for s in args.shots.split(",") if s.strip()}
        eval_shots = [s for s in eval_shots if int(s) in want]
    baseline_vec = train_mean_baseline(
        args.n_train, args.n_baseline_shots, args.min_ip_ka
    )
    eta_vec = [float(v) for v in args.eta_params.split(",")]
    args_d = vars(args) | {"split": payload_split, "eta_vector": eta_vec}
    jobs = [(int(s), args_d) for s in eval_shots]
    logger.info(
        "current-diffusion %s: split=%s shots=%s w=%.3g eta=%s b_phi0=%.3g",
        args.mode,
        args.split,
        list(eval_shots),
        args.prior_weight,
        eta_vec,
        args.b_phi0,
    )
    ctx = multiprocessing.get_context("fork")
    if args.workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
            shot_results = [r for r in pool.map(eval_shot, jobs) if r is not None]
    else:
        shot_results = [r for r in map(eval_shot, jobs) if r is not None]

    if args.mode == "eta-fit":
        if args.split != "tune":
            raise SystemExit("eta-fit is a TUNE-split calibration — never eval")
        run_eta_fit(shot_results, args)
        return 0

    scored = [
        r | {"shot": sr["shot"]}
        for sr in shot_results
        for r in sr["rows"]
        if r.get("scored")
    ]
    n_dyn_masked = sum(
        1 for sr in shot_results for r in sr["rows"] if not r.get("scored")
    )
    n_scored = len(scored)
    logger.info("scored %d (dyn-masked %d)", n_scored, n_dyn_masked)
    if n_scored == 0:
        raise SystemExit("no slices scored — cannot gate")

    model_spine = np.array([r["target_spine"] for r in scored])
    model_dyn = np.array([r["target_dyn"] for r in scored])
    ref = np.array([r["ref"] for r in scored])
    shot_ids = np.array([r["shot"] for r in scored])

    flattop = np.zeros(n_scored, dtype=bool)
    rampup = np.zeros(n_scored, dtype=bool)
    for s in np.unique(shot_ids):
        idx = np.flatnonzero(shot_ids == s)
        ips = [scored[i]["ip"] for i in idx]
        ts = [scored[i]["time_s"] for i in idx]
        flattop[idx[int(np.argmax(ips))]] = True
        rampup[idx[int(np.argmin(ts))]] = True

    sc_spine = score(model_spine, ref, baseline_vec, shot_ids=shot_ids)
    sc_dyn = score(model_dyn, ref, baseline_vec, shot_ids=shot_ids)
    for sc in (sc_spine, sc_dyn):
        sc.pop("axis_errors", None)

    # byte-exact spine reproduction vs the frozen D2 eval arrays
    drift = None
    frozen_npz = ARTIFACTS / "closure_gate_eval-D2_arrays.npz"
    if args.split == "eval" and frozen_npz.exists():
        fz = np.load(frozen_npz)
        if fz["model"].shape == model_spine.shape and bool(
            (fz["shot_ids"] == shot_ids).all()
        ):
            d = np.abs(model_spine - fz["model"])
            drift = {
                "max_abs_target_diff": float(np.nanmax(d)),
                "median_abs_target_diff": float(np.nanmedian(d)),
            }
        else:
            drift = {"note": "cohort shape/order differs from frozen arrays"}
        logger.info("spine drift vs frozen eval: %s", drift)

    seed, n_boot = 0, 2000
    base_tile = np.tile(baseline_vec, (n_scored, 1))
    ax_s, lc_s, xp_s, _ = _bootstrap_skill_draws(
        model_spine, ref, base_tile, shot_ids, n_boot=n_boot, seed=seed
    )
    ax_d, lc_d, xp_d, _ = _bootstrap_skill_draws(
        model_dyn, ref, base_tile, shot_ids, n_boot=n_boot, seed=seed
    )
    delta = {
        "axis_skill_delta": float(sc_dyn["axis_skill"] - sc_spine["axis_skill"]),
        "axis_skill_delta_ci": _percentile_ci(ax_d - ax_s),
        "lcfs_skill_delta": float(sc_dyn["lcfs_skill"] - sc_spine["lcfs_skill"]),
        "lcfs_skill_delta_ci": _percentile_ci(lc_d - lc_s),
        "xpoint_set_skill_delta": (
            None
            if sc_dyn["xpoint_set_skill"] is None
            or sc_spine["xpoint_set_skill"] is None
            else float(sc_dyn["xpoint_set_skill"] - sc_spine["xpoint_set_skill"])
        ),
        "xpoint_set_skill_delta_ci": _percentile_ci(xp_d - xp_s),
    }

    lcfs_cm_dyn = lcfs_rms_cm(model_dyn, ref)
    lcfs_cm_spine = lcfs_rms_cm(model_spine, ref)
    axis_err_dyn = (
        np.hypot(model_dyn[:, 0] - ref[:, 0], model_dyn[:, 1] - ref[:, 1]) * 100.0
    )
    axis_err_spine = (
        np.hypot(model_spine[:, 0] - ref[:, 0], model_spine[:, 1] - ref[:, 1]) * 100.0
    )
    axis_med_dyn = float(np.median(axis_err_dyn))
    axis_med_spine = float(np.median(axis_err_spine))

    # ramp-up diagnostic (P1's gate leg — reported, not gated here)
    ru_diff = lcfs_cm_dyn[rampup] - lcfs_cm_spine[rampup]
    rng = np.random.default_rng(seed)
    boot = np.array(
        [
            np.mean(rng.choice(ru_diff, size=ru_diff.size, replace=True))
            for _ in range(n_boot)
        ]
    )
    ru_ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]

    # ---- gate P2 (real-data leg) ----
    # the n >= 128 power requirement applies to the HELD-OUT eval (the tune
    # split has 64 slices by construction — its verdict is selection-only)
    gate_flattop_noninf = bool(
        (args.split == "tune" or n_scored >= 128)
        and delta["lcfs_skill_delta_ci"][0] is not None
        and delta["lcfs_skill_delta_ci"][0] >= -args.margin
    )
    gate_axis_not_degraded = bool(
        axis_med_dyn <= axis_med_spine + args.axis_median_tol_cm
    )
    reseed_spine = float(np.mean([r["reseeded_spine"] for r in scored]))
    reseed_dyn = float(np.mean([r["reseeded_dyn"] for r in scored]))
    gate_pass = bool(gate_flattop_noninf and gate_axis_not_degraded)

    # coefficient prediction pull diagnostics
    pulls = []
    for r in scored:
        if r["c_pred"] is not None and r["c_dyn"]:
            pulls.append(
                float(np.linalg.norm(np.asarray(r["c_dyn"]) - np.asarray(r["c_pred"])))
            )
    ledgers = {str(sr["shot"]): sr["ledger"] for sr in shot_results}
    ejimas = [
        sr["ledger"]["ejima_windowed"]
        for sr in shot_results
        if sr["ledger"]["ejima_windowed"] is not None
    ]

    result = {
        "arm": "current-diffusion-vs-frozen-spine",
        "split": args.split,
        "prior_weight": args.prior_weight,
        "eta_params": eta_vec,
        "b_phi0": args.b_phi0,
        "n_rho": args.n_rho,
        "n_sub_steps": args.n_sub_steps,
        "par_weight": args.par_weight,
        "spine_config_sha256": shot_results[0]["config_sha"],
        "coil_model_version": COIL_MODEL_VERSION,
        "geometry_table_version": GEOMETRY_TABLE_VERSION,
        "n_scored": n_scored,
        "n_dyn_masked": n_dyn_masked,
        "n_geo_missing": int(sum(sr["n_geo_missing"] for sr in shot_results)),
        "gate_pass": gate_pass,
        "gate": {
            "flattop_noninferior": gate_flattop_noninf,
            "flattop_cm_dyn": float(np.nanmedian(lcfs_cm_dyn[flattop])),
            "flattop_cm_spine": float(np.nanmedian(lcfs_cm_spine[flattop])),
            "axis_not_degraded": gate_axis_not_degraded,
            "axis_median_cm_dyn": axis_med_dyn,
            "axis_median_cm_spine": axis_med_spine,
            "rampup_diff_cm_mean": float(np.mean(ru_diff)),
            "rampup_diff_cm_ci": ru_ci,
            "rampup_cm_dyn": float(np.nanmedian(lcfs_cm_dyn[rampup])),
            "rampup_cm_spine": float(np.nanmedian(lcfs_cm_spine[rampup])),
            "reseed_fraction_dyn": reseed_dyn,
            "reseed_fraction_spine": reseed_spine,
        },
        "delta_vs_spine": delta,
        "spine_drift_vs_frozen": drift,
        "spine": sc_spine,
        "dynamic": sc_dyn,
        "cost_median_spine": float(np.median([r["cost_spine"] for r in scored])),
        "cost_median_dyn": float(np.median([r["cost_dyn"] for r in scored])),
        "coeff_pull_median": float(np.median(pulls)) if pulls else None,
        "eta_prediction_quality": fit_eta_on_pool(shot_results, 0.0),
        "flux_ledgers": ledgers,
        "ejima_windowed_median": float(np.median(ejimas)) if ejimas else None,
        "ejima_windowed_all": [float(e) for e in ejimas],
        "wall_s_total": float(sum(sr["wall_s"] for sr in shot_results)),
    }
    tag = f"-{args.out_suffix}" if args.out_suffix else ""
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    out_json = ARTIFACTS / f"current_diffusion_gate_eval{tag}.json"
    out_json.write_text(json.dumps(result, indent=2))
    np.savez(
        ARTIFACTS / f"current_diffusion_gate_eval{tag}_arrays.npz",
        model_spine=model_spine,
        model_dyn=model_dyn,
        ref=ref,
        shot_ids=shot_ids,
        flattop_mask=flattop,
        rampup_mask=rampup,
        cost_spine=np.array([r["cost_spine"] for r in scored]),
        cost_dyn=np.array([r["cost_dyn"] for r in scored]),
        times=np.array([r["time_s"] for r in scored]),
    )

    # ---- figures ----
    import matplotlib.pyplot as plt

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))
    shots_sorted = np.unique(shot_ids)
    x = np.arange(shots_sorted.size)
    ft_sp = [
        float(np.nanmean(lcfs_cm_spine[(shot_ids == s) & flattop]))
        for s in shots_sorted
    ]
    ft_dy = [
        float(np.nanmean(lcfs_cm_dyn[(shot_ids == s) & flattop])) for s in shots_sorted
    ]
    for off, vals, col, lab in (
        (-0.15, ft_sp, "#4477aa", "frozen spine"),
        (0.15, ft_dy, "#228833", "diffusion chain"),
    ):
        axes[0].bar(x + off, vals, width=0.28, color=col, label=lab)
    axes[0].set_xticks(x, [str(s) for s in shots_sorted], rotation=45, fontsize=7)
    axes[0].set_ylabel("flat-top LCFS RMS offset [cm]")
    axes[0].set_title(
        f"flat-top: ΔLCFS skill {delta['lcfs_skill_delta']:+.3f} "
        f"CI {delta['lcfs_skill_delta_ci']} → "
        f"{'non-inferior' if gate_flattop_noninf else 'FAIL'}"
    )
    axes[0].legend(fontsize=8)

    ax_sp = [float(np.nanmedian(axis_err_spine[shot_ids == s])) for s in shots_sorted]
    ax_dy = [float(np.nanmedian(axis_err_dyn[shot_ids == s])) for s in shots_sorted]
    for off, vals, col, lab in (
        (-0.15, ax_sp, "#4477aa", "frozen spine"),
        (0.15, ax_dy, "#228833", "diffusion chain"),
    ):
        axes[1].bar(x + off, vals, width=0.28, color=col, label=lab)
    axes[1].set_xticks(x, [str(s) for s in shots_sorted], rotation=45, fontsize=7)
    axes[1].set_ylabel("axis offset median [cm]")
    axes[1].set_title(
        f"axis: {axis_med_spine:.2f} → {axis_med_dyn:.2f} cm → "
        f"{'ok' if gate_axis_not_degraded else 'DEGRADED'}"
    )
    axes[1].legend(fontsize=8)

    # flux-consumption ledger per shot (ramp window)
    led_res = [ledgers[str(s)]["d_psi_res_ramp"] for s in shots_sorted]
    led_ind = [ledgers[str(s)]["d_psi_internal_ramp"] for s in shots_sorted]
    led_meas = [ledgers[str(s)]["d_psi_bdry_meas_ramp"] for s in shots_sorted]
    axes[2].bar(
        x - 0.15,
        np.abs(led_res),
        width=0.28,
        color="#cc6677",
        label="resistive |ΔΨ_axis|",
    )
    axes[2].bar(
        x - 0.15,
        np.abs(led_ind),
        width=0.28,
        bottom=np.abs(led_res),
        color="#ddcc77",
        label="inductive |ΔΨ_int|",
    )
    axes[2].bar(
        x + 0.15,
        np.abs(led_meas),
        width=0.28,
        color="#4477aa",
        label="measured |ΔΨ_bdry|",
    )
    ej_txt = (
        f"median C_E={result['ejima_windowed_median']:.2f}"
        if result["ejima_windowed_median"] is not None
        else "C_E n/a"
    )
    axes[2].set_xticks(x, [str(s) for s in shots_sorted], rotation=45, fontsize=7)
    axes[2].set_ylabel("ramp-window flux consumption [Wb]")
    axes[2].set_title(f"flux ledger ({ej_txt})")
    axes[2].legend(fontsize=8)

    fig.suptitle(
        f"Current-diffusion gate — {args.split} n={n_scored}, "
        f"gate {'PASS' if gate_pass else 'FAIL'} (w={args.prior_weight:g})",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out_fig = FIGURES / f"fig-current-diffusion-gate{tag}.png"
    fig.savefig(out_fig, dpi=120)
    plt.close(fig)

    logger.info(
        "gate %s: flattop non-inf %s (dyn %.2f vs %.2f cm) | axis %s "
        "(%.2f vs %.2f cm) | ramp %+0.3f cm CI %s | reseed %.3f vs %.3f | "
        "C_E median %s | n=%d | %s %s",
        "PASS" if gate_pass else "FAIL",
        gate_flattop_noninf,
        result["gate"]["flattop_cm_dyn"],
        result["gate"]["flattop_cm_spine"],
        gate_axis_not_degraded,
        axis_med_dyn,
        axis_med_spine,
        float(np.mean(ru_diff)),
        ru_ci,
        reseed_dyn,
        reseed_spine,
        result["ejima_windowed_median"],
        n_scored,
        out_json,
        out_fig,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
