#!/usr/bin/env python
"""Dynamic passive spine gate: circuit-constrained eddies vs the frozen spine.

The per-slice static passive fit is under-determined by construction — the
vessel-eddy transients obey the passive circuit equation with MEASURED drives
(coil currents at the raw 1 kHz cadence, the plasma current history from the
spine's own fits), so the eddy amplitudes are integrable trajectories, not
per-slice unknowns.  This gate promotes the L/R eigenmode reduced model from
an ML feature to a constraint inside the classical fit:

* pass 1 — the frozen classical spine, slice by slice, byte-exact (the paired
  baseline arm; also supplies the plasma-current history);
* raw-cadence integration — the mode ODE ``da/dt + a/τ = −dΨ/dt`` integrated
  on the full-shot 1 kHz drive streams from the raw stream start (solenoid
  precharge included; removes the label-cadence ``a[0] = 0`` approximation),
  with the plasma flux entering as ``d(M(t)·I_p(t))/dt`` through the full
  time-varying cell-current distribution (changing shape / position / li);
* pass 2 — every slice refit with the passive amplitudes soft-pinned to the
  trajectory (sidecar whitened coordinates; the locked soft-prior-tight
  form), optionally re-iterated once when the plasma history shifts
  materially.

Free DOF, deliberately few and physical: a bounded UNIFORM resistance scale
(``--tau-scale``; the eigenbasis is built at nominal steel resistivity —
calibrated cross-shot on the tune split, never per-slice).  The initial mode
state is NOT free: integration starts at the raw stream start where the
machine is quiescent.

Scoring: the standard harness — n≥128 held-out paired evaluation, push-out
reader, paired-bootstrap Δskill vs the byte-exact spine arm (drift-checked
against the frozen D2 arrays), per-regime designators (flat-top = per-shot
max-|Ip| slice, ramp-up = per-shot earliest slice).  Gate P1: ramp-up LCFS
beats the frozen spine with the paired CI clear of zero; flat-top
non-inferiority (−0.05 skill margin); reseed rate not degraded.

Firewall: EFIT enters referee/scoring only; all drives are raw measurements;
couplings thick-cylinder ``hybrid_greens`` only.

Artifacts: imas_ambix/latent/artifacts/patch_gate/dynamic_passive_gate_eval[-tag].json
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
from imas_ambix.gs.operator import COIL_MODEL_VERSION, build_operator
from imas_ambix.latent.data import (
    anchored_columns,
    feature_schema,
    load_shot_slices_raw,
    read_split_shot_lists,
    schema_group_offsets,
)
from imas_ambix.latent.temporal_operator import (
    build_passive_eigenbasis,
    load_eigenbasis,
    raw_eddy_trajectory,
    save_eigenbasis,
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
logger = logging.getLogger("dynamic_passive_gate_eval")

ARTIFACTS = Path("imas_ambix/latent/artifacts/patch_gate")
EIGEN_DIR = Path("imas_ambix/latent/artifacts/temporal_operator")
FIGURES = Path("docs/figures/temporal-physics-spine")


def raw_drive_streams(
    shot: int, fwd
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Full-shot 1 kHz measured drives for the circuit ODE.

    Returns ``(times, i_pf_raw (T, C), ip_raw (T,))`` — coil channel currents
    assembled exactly as the slice payloads assemble them (same channel order
    as the eigenbasis ``m_coil`` columns), and the measured plasma current for
    the pre-label amplitude-following of the plasma flux pattern.  Interior
    NaN samples are interpolated per channel so a dropped sample never
    fabricates a flux step; channels absent for the whole shot contribute
    zero (as in the slice payloads).
    """
    schema = feature_schema()
    loaded = load_shot_slices_raw(shot, schema)
    if loaded is None:
        return None
    x, times, _plasma_on = loaded
    amc_names = schema["amc"]
    off = schema_group_offsets(schema)["amc"]
    amc = np.array(x[:, off : off + len(amc_names)], dtype=np.float64)
    for j in range(amc.shape[1]):
        ok = np.isfinite(amc[:, j])
        if ok.any() and not ok.all():
            amc[:, j] = np.interp(times, times[ok], amc[ok, j])
    n = times.size
    i_pf_raw = np.zeros((n, len(fwd.pf_amc_channels)))
    for t in range(n):
        vals = {
            ch: float(amc[t, j])
            for j, ch in enumerate(amc_names)
            if np.isfinite(amc[t, j])
        }
        i_pf_raw[t] = fwd.assemble_pf_currents(vals)
    ip_col, _ne_col = anchored_columns(schema)
    ip_raw = np.abs(np.nan_to_num(np.asarray(x[:, ip_col], dtype=np.float64)))
    return np.asarray(times, dtype=np.float64), i_pf_raw, ip_raw


def shot_eigenbasis_sectionavg(payload, campaign: str, k: int):
    """The section-averaged-linkage eigenbasis, cached per campaign.

    Distinct cache from the centroid-linked basis the (closed) temporal
    operator rung used — the L matrix here integrates the flux linkage over
    both source and observer sections.
    """
    n_ch = int(payload["payloads"][0].measured.size)
    cache = EIGEN_DIR / f"eigenbasis-sectionavg-{campaign}-k{k}.npz"
    if cache.exists():
        basis = load_eigenbasis(cache)
        if basis.a_sens.shape[0] == n_ch:
            return basis
        logger.warning(
            "eigenbasis %s cache rows %d != shot channels %d — per-shot rebuild",
            campaign,
            basis.a_sens.shape[0],
            n_ch,
        )
    basis = build_passive_eigenbasis(
        payload["table"],
        payload["grid"],
        sensor_scale=payload["payloads"][0].scale,
        k=k,
    )
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        save_eigenbasis(cache, basis)
        logger.info("section-averaged eigenbasis %s built -> %s", campaign, cache)
    return basis


def _trajectory_centers(
    eigen,
    modes: np.ndarray,
    raw: tuple[np.ndarray, np.ndarray, np.ndarray],
    label_times: np.ndarray,
    i_cell_labels: np.ndarray,
    tau_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Sidecar-coordinate trajectory centers at the labelled slices.

    Integrates the mode ODE at raw cadence, maps the L-orthonormal mode state
    to circuit-space currents (``v @ a``) and projects onto the sidecar's
    whitened observable modes (least squares — components invisible to the
    magnetics are dropped, exactly the DOF the fit does not carry).
    Returns ``(centers (T_lab, kp), i_circ (T_lab, n_passive))``.
    """
    raw_times, i_pf_raw, ip_raw = raw
    a_lab, _a_raw = raw_eddy_trajectory(
        eigen,
        raw_times,
        i_pf_raw,
        label_times,
        i_cell_labels,
        ip_raw=ip_raw,
        tau_scale=tau_scale,
    )
    i_circ = a_lab @ eigen.v.T  # (T_lab, n_passive) circuit-space currents [A]
    centers, *_ = np.linalg.lstsq(modes, i_circ.T, rcond=None)
    return centers.T, i_circ


def eval_shot(job: tuple) -> dict | None:
    """One shot: byte-exact spine pass + circuit-constrained refit pass."""
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
    campaign = table.signature.key
    sidecar = _shot_passive_sidecar(payload, int(isolve["passive_k"]))
    modes = np.asarray(sidecar["modes"], dtype=np.float64)
    cell_area = grid.dr * grid.dz

    fwd = build_operator(table)
    raw = raw_drive_streams(shot, fwd)
    if raw is None:
        logger.warning("shot %s: no raw drive streams — skipped", shot)
        return None
    eigen = shot_eigenbasis_sectionavg(payload, campaign, int(args_d["n_modes"]))

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

    # ---- pass 1: frozen spine chain, byte-exact (the paired baseline) ----
    t0 = time.perf_counter()
    slices: list[dict] = []
    n_masked = 0
    warm_jphi = None
    order = np.argsort([p.time_s for p in payload["payloads"]])
    for k in order:
        p = payload["payloads"][int(k)]
        f = fit_and_read_slice(
            grid, table, p, warm_jphi=warm_jphi, keep_jphi=True, **fit_kw
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
                "i_cell": f.jphi_flat[grid.cells] * cell_area,
                "ref": payload["refs"][int(k)],
            }
        )
    if not slices:
        return None
    wall_pass1 = time.perf_counter() - t0

    label_times = np.array([s["time_s"] for s in slices])
    prior_weight = float(args_d["prior_weight"])
    tau_scale = float(args_d["tau_scale"])

    # ---- pass 2 (+ optional single iteration on a material history shift) --
    def constrained_pass(i_cell_seq: np.ndarray) -> tuple[list[dict], np.ndarray]:
        centers, i_circ = _trajectory_centers(
            eigen, modes, raw, label_times, i_cell_seq, tau_scale
        )
        rows = []
        for j, s in enumerate(slices):
            f2 = fit_and_read_slice(
                grid,
                table,
                s["payload"],
                warm_jphi=s["fit"].jphi_flat,  # refit from the slice's own spine
                keep_jphi=True,
                passive_prior=(centers[j], prior_weight),
                **fit_kw,
            )
            rows.append({"fit": f2, "center": centers[j]})
        return rows, i_circ

    i_cell_seq = np.stack([s["i_cell"] for s in slices])
    dyn_rows, i_circ_1 = constrained_pass(i_cell_seq)
    n_iterated = 0
    if int(args_d["iterate"]):
        i_cell_2 = np.stack(
            [
                (
                    r["fit"].jphi_flat[grid.cells] * cell_area
                    if r["fit"].scored and r["fit"].jphi_flat is not None
                    else s["i_cell"]
                )
                for r, s in zip(dyn_rows, slices, strict=True)
            ]
        )
        psi_1 = i_cell_seq @ eigen.m_cells.T
        psi_2 = i_cell_2 @ eigen.m_cells.T
        span = max(float(np.ptp(psi_1)), 1e-30)
        shift = float(np.abs(psi_2 - psi_1).max() / span)
        if shift > float(args_d["iterate_threshold"]):
            logger.info(
                "shot %s: plasma-history shift %.3f — re-iterating", shot, shift
            )
            dyn_rows, _ = constrained_pass(i_cell_2)
            n_iterated = 1

    rows: list[dict] = []
    for s, r in zip(slices, dyn_rows, strict=True):
        f1, f2 = s["fit"], r["fit"]
        if not f2.scored:
            # honest: the constrained refit failed where the spine scored —
            # carried as a masked dynamic row (paired scoring drops it)
            rows.append(
                {
                    "scored": False,
                    "t_index": s["t_index"],
                    "reason": f2.reason,
                }
            )
            continue
        a_fit = (
            np.linalg.lstsq(modes, np.asarray(f2.passive_amp), rcond=None)[0]
            if f2.passive_amp is not None
            else np.full(modes.shape[1], np.nan)
        )
        rows.append(
            {
                "scored": True,
                "t_index": s["t_index"],
                "time_s": s["time_s"],
                "ip": s["ip"],
                "target_spine": s["fit"].target,
                "target_dyn": f2.target,
                "ref": s["ref"],
                "cost_spine": float(f1.cost),
                "cost_dyn": float(f2.cost),
                "center": r["center"],
                "a_fit": a_fit,
                "reseeded_spine": f1.reason == "scored-reseeded",
                "reseeded_dyn": f2.reason == "scored-reseeded",
            }
        )
    return {
        "shot": int(shot),
        "rows": rows,
        "n_masked_spine": n_masked,
        "n_iterated": n_iterated,
        "config_sha": config_sha,
        "i_circ_gross_max": float(np.abs(i_circ_1).sum(axis=1).max()),
        "wall_s": time.perf_counter() - t0,
        "wall_pass1_s": wall_pass1,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=("tune", "eval"), default="eval")
    ap.add_argument("--prior-weight", type=float, default=10.0)
    ap.add_argument("--tau-scale", type=float, default=1.0)
    ap.add_argument("--iterate", type=int, default=1)
    ap.add_argument("--iterate-threshold", type=float, default=0.01)
    ap.add_argument("--n-modes", type=int, default=12)
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
    args_d = vars(args) | {"split": payload_split}
    jobs = [(int(s), args_d) for s in eval_shots]
    logger.info(
        "dynamic passive gate: split=%s shots=%s w=%.3g tau_scale=%.3g",
        args.split,
        list(eval_shots),
        args.prior_weight,
        args.tau_scale,
    )
    ctx = multiprocessing.get_context("fork")
    if args.workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
            shot_results = [r for r in pool.map(eval_shot, jobs) if r is not None]
    else:
        shot_results = [r for r in map(eval_shot, jobs) if r is not None]

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

    # paired Δskill (identical seed → paired draws), R1 protocol
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

    # ---- gate 1: ramp-up LCFS beats the spine, paired CI clear of zero ----
    ru_diff = lcfs_cm_dyn[rampup] - lcfs_cm_spine[rampup]  # one per shot
    rng = np.random.default_rng(seed)
    boot = np.array(
        [
            np.mean(rng.choice(ru_diff, size=ru_diff.size, replace=True))
            for _ in range(n_boot)
        ]
    )
    ru_ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
    gate_rampup = bool(ru_diff.size >= 6 and ru_ci[1] < 0.0)

    # ---- gate 2: flat-top non-inferiority (paired LCFS skill delta) ----
    gate_flattop_noninf = bool(
        n_scored >= 128
        and delta["lcfs_skill_delta_ci"][0] is not None
        and delta["lcfs_skill_delta_ci"][0] >= -args.margin
        and delta["axis_skill_delta_ci"][0] >= -args.margin
    )

    # ---- gate 3: reseed rate not degraded ----
    reseed_spine = float(np.mean([r["reseeded_spine"] for r in scored]))
    reseed_dyn = float(np.mean([r["reseeded_dyn"] for r in scored]))
    gate_reseed = bool(reseed_dyn <= reseed_spine + 1e-12)

    gate_pass = bool(gate_rampup and gate_flattop_noninf and gate_reseed)

    center_norm = np.array([np.linalg.norm(r["center"]) for r in scored])
    a_fit_all = np.array([r["a_fit"] for r in scored])
    center_all = np.array([r["center"] for r in scored])
    pull = np.linalg.norm(a_fit_all - center_all, axis=1)

    result = {
        "arm": "dynamic-passive-vs-frozen-spine",
        "split": args.split,
        "prior_weight": args.prior_weight,
        "tau_scale": args.tau_scale,
        "iterate": args.iterate,
        "n_modes": args.n_modes,
        "spine_config_sha256": shot_results[0]["config_sha"],
        "coil_model_version": COIL_MODEL_VERSION,
        "geometry_table_version": GEOMETRY_TABLE_VERSION,
        "n_scored": n_scored,
        "n_dyn_masked": n_dyn_masked,
        "n_shots_iterated": int(sum(sr["n_iterated"] for sr in shot_results)),
        "gate_pass": gate_pass,
        "gate": {
            "rampup_beats_spine_ci_clear": gate_rampup,
            "rampup_diff_cm_mean": float(np.mean(ru_diff)),
            "rampup_diff_cm_ci": ru_ci,
            "rampup_cm_dyn": float(np.nanmedian(lcfs_cm_dyn[rampup])),
            "rampup_cm_spine": float(np.nanmedian(lcfs_cm_spine[rampup])),
            "flattop_noninferior": gate_flattop_noninf,
            "flattop_cm_dyn": float(np.nanmedian(lcfs_cm_dyn[flattop])),
            "flattop_cm_spine": float(np.nanmedian(lcfs_cm_spine[flattop])),
            "reseed_not_degraded": gate_reseed,
            "reseed_fraction_dyn": reseed_dyn,
            "reseed_fraction_spine": reseed_spine,
        },
        "delta_vs_spine": delta,
        "spine_drift_vs_frozen": drift,
        "spine": sc_spine,
        "dynamic": sc_dyn,
        "axis_median_cm_dyn": float(np.median(axis_err_dyn)),
        "axis_median_cm_spine": float(np.median(axis_err_spine)),
        "cost_median_spine": float(np.median([r["cost_spine"] for r in scored])),
        "cost_median_dyn": float(np.median([r["cost_dyn"] for r in scored])),
        "trajectory_center_norm_median": float(np.median(center_norm)),
        "fit_minus_center_norm_median": float(np.median(pull[np.isfinite(pull)])),
        "i_circ_gross_max": float(max(sr["i_circ_gross_max"] for sr in shot_results)),
        "wall_s_total": float(sum(sr["wall_s"] for sr in shot_results)),
    }
    tag = f"-{args.out_suffix}" if args.out_suffix else ""
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    out_json = ARTIFACTS / f"dynamic_passive_gate_eval{tag}.json"
    out_json.write_text(json.dumps(result, indent=2))
    np.savez(
        ARTIFACTS / f"dynamic_passive_gate_eval{tag}_arrays.npz",
        model_spine=model_spine,
        model_dyn=model_dyn,
        ref=ref,
        shot_ids=shot_ids,
        flattop_mask=flattop,
        rampup_mask=rampup,
        centers=center_all,
        a_fit=a_fit_all,
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
    ru_sp = [
        float(np.nanmean(lcfs_cm_spine[(shot_ids == s) & rampup])) for s in shots_sorted
    ]
    ru_dy = [
        float(np.nanmean(lcfs_cm_dyn[(shot_ids == s) & rampup])) for s in shots_sorted
    ]
    for off, vals, col, lab in (
        (-0.15, ru_sp, "#4477aa", "frozen spine"),
        (0.15, ru_dy, "#228833", "dynamic passive"),
    ):
        axes[0].bar(x + off, vals, width=0.28, color=col, label=lab)
    axes[0].set_xticks(x, [str(s) for s in shots_sorted], rotation=45, fontsize=7)
    axes[0].set_ylabel("ramp-up LCFS RMS offset [cm]")
    axes[0].set_title(
        f"ramp-up: dyn−spine {np.mean(ru_diff):+.2f} cm "
        f"CI [{ru_ci[0]:+.2f}, {ru_ci[1]:+.2f}] → "
        f"{'PASS' if gate_rampup else 'FAIL'}"
    )
    axes[0].legend(fontsize=8)

    ft_sp = [
        float(np.nanmean(lcfs_cm_spine[(shot_ids == s) & flattop]))
        for s in shots_sorted
    ]
    ft_dy = [
        float(np.nanmean(lcfs_cm_dyn[(shot_ids == s) & flattop])) for s in shots_sorted
    ]
    for off, vals, col, lab in (
        (-0.15, ft_sp, "#4477aa", "frozen spine"),
        (0.15, ft_dy, "#228833", "dynamic passive"),
    ):
        axes[1].bar(x + off, vals, width=0.28, color=col, label=lab)
    axes[1].set_xticks(x, [str(s) for s in shots_sorted], rotation=45, fontsize=7)
    axes[1].set_ylabel("flat-top LCFS RMS offset [cm]")
    axes[1].set_title(
        f"flat-top: ΔLCFS skill {delta['lcfs_skill_delta']:+.3f} "
        f"CI {delta['lcfs_skill_delta_ci']} → "
        f"{'non-inferior' if gate_flattop_noninf else 'FAIL'}"
    )
    axes[1].legend(fontsize=8)

    finite = np.isfinite(pull)
    axes[2].scatter(center_norm[finite], pull[finite], s=14, alpha=0.6, color="#228833")
    axes[2].set_xlabel("‖trajectory center‖ (whitened)")
    axes[2].set_ylabel("‖fit − center‖ (whitened)")
    axes[2].set_title(
        f"constraint pull @ w={args.prior_weight:g} "
        f"(median center {np.median(center_norm):.2f}, "
        f"median pull {np.median(pull[finite]):.2f})"
    )
    fig.suptitle(
        f"Dynamic passive gate — {args.split} n={n_scored}, "
        f"gate {'PASS' if gate_pass else 'FAIL'}",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out_fig = FIGURES / f"fig-dynamic-passive-gate{tag}.png"
    fig.savefig(out_fig, dpi=120)
    plt.close(fig)

    logger.info(
        "gate %s: rampup %s (dyn %.2f vs spine %.2f cm, CI %s) | "
        "flattop non-inf %s (dyn %.2f vs %.2f cm) | reseed %s (%.3f vs %.3f) | "
        "n=%d | %s %s",
        "PASS" if gate_pass else "FAIL",
        gate_rampup,
        result["gate"]["rampup_cm_dyn"],
        result["gate"]["rampup_cm_spine"],
        ru_ci,
        gate_flattop_noninf,
        result["gate"]["flattop_cm_dyn"],
        result["gate"]["flattop_cm_spine"],
        gate_reseed,
        reseed_dyn,
        reseed_spine,
        n_scored,
        out_json,
        out_fig,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
