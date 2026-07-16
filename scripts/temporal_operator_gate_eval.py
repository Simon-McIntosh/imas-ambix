#!/usr/bin/env python
"""Temporal-operator gate: the time-aware rung vs its frozen classical baselines.

Held-out paired evaluation on the closure gate cohort (n≥128 slices, 8 shots).
Per shot the frozen classical spine is fit slice-by-slice exactly as frozen
(warm chain, strict-converged only), the staged-disc boundary read is scored
as its own arm, and the trained temporal operator then corrects the spine
sequence causally — profile-DOF corrections through the exact profile Green's
columns plus L/R-eigenmode eddy amplitudes entering as external currents
through the exact passive Green's columns.  Every arm is read with the SAME
push-out geometry reader and scored against the firewalled EFIT referee with
the SAME paired-bootstrap harness.

The headline gate (three conditions, all required):

1. flat-top LCFS: the operator beats the staged-disc read on the per-shot
   flat-top slices with the paired 95% CI clear of zero;
2. ramp-up LCFS: median over the per-shot ramp-up proxy slices (earliest
   valid — the convention the 5–10 cm disc baseline was measured with)
   below 5 cm;
3. axis: median axis error over all scored slices below 5.1 cm (the D1
   classical axis baseline).

Non-inferiority vs the D2 spine (the R1 protocol) is reported as context.
Firewall: EFIT enters referee/scoring only; the operator saw only spine-
manufactured labels from disjoint train-split shots.

Artifacts: imas_ambix/latent/artifacts/patch_gate/temporal_operator_gate_eval.json
Figures:   docs/figures/learned-equilibrium-operator/
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
import torch

from imas_ambix.gs.geometry import GEOMETRY_TABLE_VERSION
from imas_ambix.gs.operator import COIL_MODEL_VERSION, build_operator
from imas_ambix.latent.boundary_disc import (
    DiscReadConfig,
    disc_read,
    sensor_signature_arrays,
)
from imas_ambix.latent.data import (
    feature_schema,
    load_shot_windows,
    read_split_shot_lists,
)
from imas_ambix.latent.profile_greens_decoder import ProfileGreensDecoder
from imas_ambix.latent.residual_operator import slice_globals, slice_tokens
from imas_ambix.latent.temporal_operator import (
    build_passive_eigenbasis,
    load_checkpoint,
    load_eigenbasis,
    physical_eddy_history,
    save_eigenbasis,
)
from scripts.closure_gate_eval import (
    _shot_passive_sidecar,
    fit_and_read_slice,
    geometry_target_pushout,
)
from scripts.patch_gate_eval import (
    _bootstrap_skill_draws,
    _percentile_ci,
    lcfs_offset_cm_stats,
    score,
    shot_payloads,
    train_mean_baseline,
)
from scripts.spine_label_factory import frozen_spine_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("temporal_operator_gate_eval")

ARTIFACTS = Path("imas_ambix/latent/artifacts/patch_gate")
EIGEN_DIR = Path("imas_ambix/latent/artifacts/temporal_operator")
FIGURES = Path("docs/figures/learned-equilibrium-operator")


def lcfs_rms_cm(model: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Per-slice LCFS radial RMS offset [cm] from the 8-radius target block."""
    return np.linalg.norm(model[:, 6:] - ref[:, 6:], axis=1) / np.sqrt(8.0) * 100.0


def shot_eigenbasis(payload, campaign: str, k: int):
    """Load the campaign eigenbasis from the training cache, or build.

    A cached basis whose sensor rows disagree with THIS shot's channel set
    (per-shot sensor availability varies within a campaign) is rebuilt for
    the shot without touching the cache.
    """
    n_ch = int(payload["payloads"][0].measured.size)
    cache = EIGEN_DIR / f"eigenbasis-{campaign}-k{k}.npz"
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
        logger.info("eigenbasis %s built at eval time -> %s", campaign, cache)
    return basis


def eval_shot(job: tuple) -> dict | None:
    """One shot: spine chain + disc read + causal temporal pass (fork worker)."""
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
        split="eval",
    )
    if payload is None:
        return None
    grid, table, basis = payload["grid"], payload["table"], payload["basis"]
    campaign = table.signature.key
    sidecar = _shot_passive_sidecar(payload, int(isolve["passive_k"]))
    modes = np.asarray(sidecar["modes"], dtype=np.float64)
    cell_area = grid.dr * grid.dz

    fwd = build_operator(table)
    w = load_shot_windows(
        shot, fwd, table.signature.key, feature_schema(), with_referee=False
    )
    ne_rows = None if w is None else w.anchored[:, 1]

    model, ckpt = load_checkpoint(args_d["checkpoint"])
    arm = str(ckpt.get("arm", "residual"))
    c_med = ckpt.get("c_med")
    eigen = shot_eigenbasis(payload, campaign, int(ckpt["n_modes"]))
    tau_init = np.asarray(ckpt.get("tau_init"), dtype=np.float64)
    tau_drift = float(
        np.max(np.abs(np.log(eigen.tau / tau_init)))
        if tau_init.shape == eigen.tau.shape
        else np.inf
    )

    from imas_ambix.latent.patch_basis import PatchBasis  # noqa: PLC0415

    basis64 = PatchBasis.from_table(
        table, nr=args_d["nr"], nz=args_d["nz"], dtype=torch.float64
    )
    dec = ProfileGreensDecoder(
        basis64,
        n_p=int(isolve["n_p"]),
        n_f=int(isolve["n_f"]),
        kind=str(isolve["profile_kind"]),
    )
    m_sens_np = np.asarray(dec.basis.m_sens.numpy(), dtype=np.float64)
    sr, sz, sang, is_flux = sensor_signature_arrays(table)
    disc_cfg = DiscReadConfig()

    # ---- pass 1: frozen spine chain + disc read, slice by slice ----
    slices: list[dict] = []
    n_masked = 0
    warm_jphi = None
    order = np.argsort([p.time_s for p in payload["payloads"]])
    t0 = time.perf_counter()
    for k in order:
        p = payload["payloads"][int(k)]
        f = fit_and_read_slice(
            grid,
            table,
            p,
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
            warm_jphi=warm_jphi,
            reseed_axis_r_max=float(isolve["reseed_axis_r_max"]),
            keep_psi=True,
            keep_jphi=True,
            basis=basis,
            meta={},
            soft_prior_cfg=spc,
            boundary_read=isolve["boundary_read_scoring"],
        )
        if not f.scored:
            n_masked += 1
            continue
        if f.converged:  # match the frozen harness chain: strict-converged only
            warm_jphi = f.jphi_flat
        i_cell0 = f.jphi_flat[grid.cells] * cell_area
        _t, psi_ax, psi_b = geometry_target_pushout(f.psi, grid)
        psi_n = np.clip(
            (f.psi.ravel()[grid.cells] - psi_ax) / (psi_b - psi_ax), 0.0, 1.5
        )
        if f.passive_amp is not None:
            a_pass, *_ = np.linalg.lstsq(modes, np.asarray(f.passive_amp), rcond=None)
            sens_passive = np.asarray(sidecar["g_cols"], dtype=np.float64) @ a_pass
        else:
            sens_passive = np.zeros(p.measured.size)
        spine_pred = m_sens_np @ i_cell0 + p.vacuum + sens_passive

        inv = disc_read(p, grid, table, basis, disc_cfg)
        target_disc = None
        if inv is not None and inv.ring is not None:
            target_disc, _, _ = geometry_target_pushout(inv.psi_tot, grid)

        n_e = float(ne_rows[p.t_index]) if ne_rows is not None else float("nan")
        slices.append(
            {
                "k": int(k),
                "t_index": p.t_index,
                "time_s": p.time_s,
                "ip": p.ip_amperes,
                "i_pf": p.i_pf,
                "n_e": n_e,
                "payload": p,
                "fit": f,
                "i_cell0": i_cell0,
                "psi_n": psi_n,
                "spine_pred": spine_pred,
                "sens_passive": sens_passive,
                "target_spine": f.target,
                "target_disc": target_disc,
                "ref": payload["refs"][int(k)],
                "reseeded": f.reason == "scored-reseeded",
            }
        )
    if not slices:
        return None

    # ---- pass 2: causal temporal pass over the scored sequence ----
    times = np.array([s["time_s"] for s in slices])
    i_pf_seq = np.stack([s["i_pf"] for s in slices])
    i_cell_seq = np.stack([s["i_cell0"] for s in slices])
    a_phys, u_drive = physical_eddy_history(eigen, times, i_pf_seq, i_cell_seq)
    dt = np.diff(times, prepend=times[0])
    dt[0] = float(np.median(dt[1:])) if len(slices) > 1 else 1e-2

    tokens, masks, gl = [], [], []
    for s in slices:
        p = s["payload"]
        tk, m = slice_tokens(
            p.measured,
            p.vacuum,
            s["spine_pred"],
            p.scale,
            p.mask,
            sr,
            sz,
            sang,
            is_flux,
        )
        tokens.append(tk)
        masks.append(m)
        gl.append(slice_globals(p.ip_amperes, s["n_e"]))
    with torch.no_grad():
        dc_seq, da_seq = model(
            torch.tensor(np.stack(tokens)).unsqueeze(0),
            torch.tensor(np.stack(masks)).unsqueeze(0),
            torch.tensor(np.stack(gl)).unsqueeze(0),
            torch.tensor(dt, dtype=torch.float32).unsqueeze(0),
            torch.tensor(a_phys, dtype=torch.float32).unsqueeze(0),
            torch.tensor(u_drive, dtype=torch.float32).unsqueeze(0),
        )
        dc_seq = dc_seq[0].double()
        da_seq = da_seq[0].double()

    rows: list[dict] = []
    for j, s in enumerate(slices):
        p, f = s["payload"], s["fit"]
        with torch.no_grad():
            cols = dec.profile_columns(
                torch.tensor(s["psi_n"], dtype=torch.float64).unsqueeze(0),
                torch.tensor([p.ip_amperes], dtype=torch.float64),
            )
            i_cell0_t = torch.tensor(s["i_cell0"], dtype=torch.float64).unsqueeze(0)
            if arm == "direct":
                # direct-DOF ablation: absolute coefficients about the
                # corpus-median column-unit profile, no classical warm start
                base = torch.zeros_like(i_cell0_t)
                eff = (torch.tensor(c_med, dtype=torch.float64) + dc_seq[j]).unsqueeze(
                    0
                )
            else:
                base = i_cell0_t
                eff = dc_seq[j].unsqueeze(0)
            i_new = dec.cell_currents(
                base,
                eff,
                cols,
                torch.tensor([p.ip_amperes], dtype=torch.float64),
            )[0].numpy()
        da_j = da_seq[j].numpy()
        dpsi = (
            dec.basis._g_pg_np @ (i_new - s["i_cell0"]) + eigen.g_grid @ da_j
        ).reshape(grid.nz, grid.nr)
        target_op, _, _ = geometry_target_pushout(f.psi + dpsi, grid)

        keep = p.mask & np.isfinite(p.measured)
        n_keep = max(int(keep.sum()), 1)
        scale = np.clip(p.scale, 1e-12, None)
        meas = np.nan_to_num(p.measured)
        op_pred = (
            s["spine_pred"] + m_sens_np @ (i_new - s["i_cell0"]) + eigen.a_sens @ da_j
        )
        rows.append(
            {
                "scored": True,
                "t_index": s["t_index"],
                "time_s": s["time_s"],
                "ip": s["ip"],
                "target_spine": s["target_spine"],
                "target_op": target_op,
                "target_disc": s["target_disc"],
                "ref": s["ref"],
                "cost_spine": float(
                    np.sum(np.where(keep, (s["spine_pred"] - meas) / scale, 0.0) ** 2)
                    / n_keep
                ),
                "cost_op": float(
                    np.sum(np.where(keep, (op_pred - meas) / scale, 0.0) ** 2) / n_keep
                ),
                "dc": dc_seq[j].numpy(),
                "da": da_j,
                "a_phys": a_phys[j],
                "reseeded": s["reseeded"],
                "dpsi_max": float(np.abs(dpsi).max()),
            }
        )
    return {
        "shot": int(shot),
        "wall_s": time.perf_counter() - t0,
        "rows": rows,
        "n_masked": n_masked,
        "config_sha": config_sha,
        "tau_drift_log": tau_drift,
        "operator_arm": arm,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--checkpoint",
        type=str,
        default="imas_ambix/latent/artifacts/temporal_operator/temporal_operator.pt",
    )
    ap.add_argument("--n-train", type=int, default=40)
    ap.add_argument("--n-heldout", type=int, default=8)
    ap.add_argument("--n-baseline-shots", type=int, default=10)
    ap.add_argument("--nr", type=int, default=65)
    ap.add_argument("--nz", type=int, default=97)
    ap.add_argument("--max-slices-per-shot", type=int, default=16)
    ap.add_argument("--min-ip-ka", type=float, default=300.0)
    ap.add_argument("--convergence-limit", type=float, default=5e-3)
    ap.add_argument("--retry-max-iterations", type=int, default=160)
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--rampup-cm-bar", type=float, default=5.0)
    ap.add_argument("--axis-cm-bar", type=float, default=5.1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out-suffix", type=str, default="")
    args = ap.parse_args()

    _train, held_shots = read_split_shot_lists(args.n_train, args.n_heldout)
    baseline_vec = train_mean_baseline(
        args.n_train, args.n_baseline_shots, args.min_ip_ka
    )
    args_d = vars(args)
    jobs = [(int(s), args_d) for s in held_shots]
    logger.info("gate eval: shots %s", held_shots)
    ctx = multiprocessing.get_context("fork")
    if args.workers > 1:
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
    n_candidate = sum(len(sr["rows"]) + sr["n_masked"] for sr in shot_results)
    n_scored = len(scored)
    logger.info("scored %d/%d", n_scored, n_candidate)
    if n_scored == 0:
        raise SystemExit("no slices scored — cannot gate")

    model_spine = np.array([r["target_spine"] for r in scored])
    model_op = np.array([r["target_op"] for r in scored])
    ref = np.array([r["ref"] for r in scored])
    shot_ids = np.array([r["shot"] for r in scored])
    disc_ok = np.array([r["target_disc"] is not None for r in scored])
    model_disc = np.array(
        [
            r["target_disc"] if r["target_disc"] is not None else np.full(14, np.nan)
            for r in scored
        ]
    )

    # regime designators (the conventions the frozen baselines were measured
    # with): flat-top = per-shot max-|Ip| slice; ramp-up = per-shot earliest
    flattop = np.zeros(n_scored, dtype=bool)
    rampup = np.zeros(n_scored, dtype=bool)
    for s in np.unique(shot_ids):
        idx = np.flatnonzero(shot_ids == s)
        ips = [scored[i]["ip"] for i in idx]
        ts = [scored[i]["time_s"] for i in idx]
        flattop[idx[int(np.argmax(ips))]] = True
        rampup[idx[int(np.argmin(ts))]] = True

    sc_spine = score(model_spine, ref, baseline_vec, shot_ids=shot_ids)
    sc_op = score(model_op, ref, baseline_vec, shot_ids=shot_ids)
    axis_err_op = np.hypot(model_op[:, 0] - ref[:, 0], model_op[:, 1] - ref[:, 1])
    axis_err_spine = np.hypot(
        model_spine[:, 0] - ref[:, 0], model_spine[:, 1] - ref[:, 1]
    )
    for sc in (sc_spine, sc_op):
        sc.pop("axis_errors", None)

    # drift check vs the frozen D2 eval arrays — byte-exact spine reproduction
    drift = None
    frozen_npz = ARTIFACTS / "closure_gate_eval-D2_arrays.npz"
    if frozen_npz.exists():
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

    # paired Δskill vs the spine (identical seed → paired draws), R1 protocol
    seed, n_boot = 0, 2000
    ax_s, lc_s, xp_s, _ = _bootstrap_skill_draws(
        model_spine,
        ref,
        np.tile(baseline_vec, (n_scored, 1)),
        shot_ids,
        n_boot=n_boot,
        seed=seed,
    )
    ax_o, lc_o, xp_o, _ = _bootstrap_skill_draws(
        model_op,
        ref,
        np.tile(baseline_vec, (n_scored, 1)),
        shot_ids,
        n_boot=n_boot,
        seed=seed,
    )
    d_axis, d_lcfs, d_xpt = ax_o - ax_s, lc_o - lc_s, xp_o - xp_s
    delta = {
        "axis_skill_delta": float(sc_op["axis_skill"] - sc_spine["axis_skill"]),
        "axis_skill_delta_ci": _percentile_ci(d_axis),
        "lcfs_skill_delta": float(sc_op["lcfs_skill"] - sc_spine["lcfs_skill"]),
        "lcfs_skill_delta_ci": _percentile_ci(d_lcfs),
        "xpoint_set_skill_delta": (
            None
            if sc_op["xpoint_set_skill"] is None
            else float(sc_op["xpoint_set_skill"] - sc_spine["xpoint_set_skill"])
        ),
        "xpoint_set_skill_delta_ci": _percentile_ci(d_xpt),
    }
    non_inferior = (
        n_scored >= 128
        and delta["axis_skill_delta_ci"][0] is not None
        and delta["axis_skill_delta_ci"][0] >= -args.margin
        and delta["lcfs_skill_delta_ci"][0] >= -args.margin
    )

    # ---- G2 condition 1: beat the disc read on flat-top slices, CI-clear ----
    lcfs_cm_op = lcfs_rms_cm(model_op, ref)
    lcfs_cm_spine = lcfs_rms_cm(model_spine, ref)
    lcfs_cm_disc = np.where(disc_ok, lcfs_rms_cm(model_disc, ref), np.nan)
    ft = flattop & disc_ok
    ft_diff = lcfs_cm_op[ft] - lcfs_cm_disc[ft]  # one per shot
    rng = np.random.default_rng(seed)
    boot = np.array(
        [
            np.mean(rng.choice(ft_diff, size=ft_diff.size, replace=True))
            for _ in range(n_boot)
        ]
    )
    ft_ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
    g2_flattop = bool(ft_diff.size >= 6 and ft_ci[1] < 0.0)

    # ---- G2 condition 2: ramp-up median < 5 cm ----
    ramp_med_op = float(np.nanmedian(lcfs_cm_op[rampup]))
    ramp_med_disc = float(np.nanmedian(lcfs_cm_disc[rampup & disc_ok]))
    ramp_med_spine = float(np.nanmedian(lcfs_cm_spine[rampup]))
    g2_rampup = bool(ramp_med_op < args.rampup_cm_bar)

    # ---- G2 condition 3: axis median < 5.1 cm ----
    axis_med_op = float(np.median(axis_err_op)) * 100.0
    axis_med_spine = float(np.median(axis_err_spine)) * 100.0
    g2_axis = bool(axis_med_op < args.axis_cm_bar)

    gate_pass = bool(g2_flattop and g2_rampup and g2_axis)

    lcfs_stats_spine = lcfs_offset_cm_stats(model_spine, ref, flattop)
    lcfs_stats_op = lcfs_offset_cm_stats(model_op, ref, flattop)
    dc_all = np.array([r["dc"] for r in scored])
    da_all = np.array([r["da"] for r in scored])
    a_phys_all = np.array([r["a_phys"] for r in scored])

    # p′/FF′ split on synthetic truth — folded in from the synthetic harness
    synth_path = EIGEN_DIR / "synthetic_pretrain_report.json"
    synth = json.loads(synth_path.read_text()) if synth_path.exists() else None

    result = {
        "arm": "temporal-operator-vs-baselines",
        "operator_arm": shot_results[0].get("operator_arm", "residual"),
        "checkpoint": args.checkpoint,
        "spine_config_sha256": shot_results[0]["config_sha"],
        "coil_model_version": COIL_MODEL_VERSION,
        "geometry_table_version": GEOMETRY_TABLE_VERSION,
        "n_scored": n_scored,
        "n_candidate": n_candidate,
        "gate_pass": gate_pass,
        "g2": {
            "flattop_beat_disc_ci_clear": g2_flattop,
            "flattop_diff_cm_mean": float(np.mean(ft_diff)),
            "flattop_diff_cm_ci": ft_ci,
            "flattop_cm_op": float(np.nanmedian(lcfs_cm_op[flattop])),
            "flattop_cm_disc": float(np.nanmedian(lcfs_cm_disc[ft])),
            "flattop_cm_spine": float(np.nanmedian(lcfs_cm_spine[flattop])),
            "rampup_below_bar": g2_rampup,
            "rampup_cm_bar": args.rampup_cm_bar,
            "rampup_cm_op": ramp_med_op,
            "rampup_cm_disc": ramp_med_disc,
            "rampup_cm_spine": ramp_med_spine,
            "axis_below_bar": g2_axis,
            "axis_cm_bar": args.axis_cm_bar,
            "axis_median_cm_op": axis_med_op,
            "axis_median_cm_spine": axis_med_spine,
            "synthetic_split": (None if synth is None else synth.get("split_recovery")),
        },
        "non_inferiority_vs_spine": {
            "margin": args.margin,
            "pass": bool(non_inferior),
            "delta": delta,
        },
        "spine_drift_vs_frozen": drift,
        "tau_drift_log_max": float(max(sr["tau_drift_log"] for sr in shot_results)),
        "spine": {**sc_spine, **{f"lcfs_{k}": v for k, v in lcfs_stats_spine.items()}},
        "operator": {**sc_op, **{f"lcfs_{k}": v for k, v in lcfs_stats_op.items()}},
        "cost_median_spine": float(np.median([r["cost_spine"] for r in scored])),
        "cost_median_operator": float(np.median([r["cost_op"] for r in scored])),
        "dc_abs_median": float(np.median(np.abs(dc_all))),
        "dc_abs_max": float(np.max(np.abs(dc_all))),
        "da_abs_median": float(np.median(np.abs(da_all))),
        "da_abs_max": float(np.max(np.abs(da_all))),
        "reseed_fraction": float(np.mean([r["reseeded"] for r in scored])),
        "wall_s_total": float(sum(sr["wall_s"] for sr in shot_results)),
    }
    tag = f"-{args.out_suffix}" if args.out_suffix else ""
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / f"temporal_operator_gate_eval{tag}.json").write_text(
        json.dumps(result, indent=2)
    )
    np.savez(
        ARTIFACTS / f"temporal_operator_gate_eval{tag}_arrays.npz",
        model_spine=model_spine,
        model_op=model_op,
        model_disc=model_disc,
        ref=ref,
        shot_ids=shot_ids,
        flattop_mask=flattop,
        rampup_mask=rampup,
        dc=dc_all,
        da=da_all,
        a_phys=a_phys_all,
        cost_spine=np.array([r["cost_spine"] for r in scored]),
        cost_op=np.array([r["cost_op"] for r in scored]),
        delta_axis_draws=d_axis,
        delta_lcfs_draws=d_lcfs,
        times=np.array([r["time_s"] for r in scored]),
    )

    # ---- figures ----
    import matplotlib.pyplot as plt

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))
    shots_sorted = np.unique(shot_ids)
    x = np.arange(shots_sorted.size)
    ft_op = [float(lcfs_cm_op[(shot_ids == s) & flattop][0]) for s in shots_sorted]
    ft_disc = [
        float(np.nanmean(lcfs_cm_disc[(shot_ids == s) & flattop])) for s in shots_sorted
    ]
    ft_sp = [float(lcfs_cm_spine[(shot_ids == s) & flattop][0]) for s in shots_sorted]
    for off, vals, col, lab in (
        (-0.25, ft_disc, "#999999", "disc read"),
        (0.0, ft_sp, "#4477aa", "D2 spine"),
        (0.25, ft_op, "#228833", "temporal op"),
    ):
        axes[0].bar(x + off, vals, width=0.24, color=col, label=lab)
    axes[0].set_xticks(x, [str(s) for s in shots_sorted], rotation=45, fontsize=7)
    axes[0].set_ylabel("flat-top LCFS RMS offset [cm]")
    axes[0].set_title(
        f"flat-top: op−disc = {np.mean(ft_diff):+.2f} cm "
        f"CI [{ft_ci[0]:+.2f}, {ft_ci[1]:+.2f}] → "
        f"{'PASS' if g2_flattop else 'FAIL'}"
    )
    axes[0].legend(fontsize=8)

    ru_op = [
        float(np.nanmean(lcfs_cm_op[(shot_ids == s) & rampup])) for s in shots_sorted
    ]
    ru_disc = [
        float(np.nanmean(lcfs_cm_disc[(shot_ids == s) & rampup])) for s in shots_sorted
    ]
    ru_sp = [
        float(np.nanmean(lcfs_cm_spine[(shot_ids == s) & rampup])) for s in shots_sorted
    ]
    for off, vals, col, lab in (
        (-0.25, ru_disc, "#999999", "disc read"),
        (0.0, ru_sp, "#4477aa", "D2 spine"),
        (0.25, ru_op, "#228833", "temporal op"),
    ):
        axes[1].bar(x + off, vals, width=0.24, color=col, label=lab)
    axes[1].axhline(
        args.rampup_cm_bar, color="#bb5566", ls="--", lw=1.5, label="G2 bar 5 cm"
    )
    axes[1].set_xticks(x, [str(s) for s in shots_sorted], rotation=45, fontsize=7)
    axes[1].set_ylabel("ramp-up LCFS RMS offset [cm]")
    axes[1].set_title(
        f"ramp-up: median op {ramp_med_op:.2f} cm "
        f"(disc {ramp_med_disc:.2f}) → {'PASS' if g2_rampup else 'FAIL'}"
    )
    axes[1].legend(fontsize=8)

    axes[2].hist(
        axis_err_spine * 100, bins=40, alpha=0.55, color="#4477aa", label="D2 spine"
    )
    axes[2].hist(
        axis_err_op * 100, bins=40, alpha=0.55, color="#228833", label="temporal op"
    )
    axes[2].axvline(
        args.axis_cm_bar, color="#bb5566", ls="--", lw=1.5, label="G2 bar 5.1 cm"
    )
    axes[2].axvline(axis_med_op, color="#228833", lw=1.8)
    axes[2].axvline(axis_med_spine, color="#4477aa", lw=1.8)
    axes[2].set_xlabel("axis error [cm]")
    axes[2].set_title(
        f"axis: median op {axis_med_op:.1f} cm "
        f"(spine {axis_med_spine:.1f}) → {'PASS' if g2_axis else 'FAIL'}"
    )
    axes[2].legend(fontsize=8)
    fig.suptitle(
        f"Temporal operator G2 — held-out n={n_scored}, "
        f"gate {'PASS' if gate_pass else 'FAIL'}",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = FIGURES / f"fig-temporal-operator-g2{tag}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    for ax, draws, d0, name in (
        (axes[0], d_axis, delta["axis_skill_delta"], "axis skill"),
        (axes[1], d_lcfs, delta["lcfs_skill_delta"], "LCFS skill"),
    ):
        ax.hist(draws[np.isfinite(draws)], bins=60, color="#4477aa", alpha=0.85)
        ax.axvline(-args.margin, color="#bb5566", ls="--", lw=1.6)
        ax.axvline(0.0, color="#555", lw=1.0)
        ax.axvline(d0, color="#222", lw=1.8, label=f"Δ = {d0:+.3f}")
        lo, hi = np.nanpercentile(draws, [2.5, 97.5])
        ax.set_title(f"paired Δ{name} (op − spine)\n95% CI [{lo:+.3f}, {hi:+.3f}]")
        ax.legend(fontsize=8)
    fig.tight_layout()
    out2 = FIGURES / f"fig-temporal-operator-delta{tag}.png"
    fig.savefig(out2, dpi=120)
    plt.close(fig)

    logger.info(
        "G2 %s: flattop %s (op-disc %+.2f cm CI %s) | rampup %s (%.2f cm) | "
        "axis %s (%.1f cm) | non-inferiority vs spine %s | n=%d | figs %s %s",
        "PASS" if gate_pass else "FAIL",
        g2_flattop,
        float(np.mean(ft_diff)),
        ft_ci,
        g2_rampup,
        ramp_med_op,
        g2_axis,
        axis_med_op,
        non_inferior,
        n_scored,
        out,
        out2,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
