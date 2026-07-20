#!/usr/bin/env python
"""Per-slice profile diagnostics: our reanalysis engine vs EnKF vs MSE truth.

For a handful of held-out shots this dumps, at rampup-start / mid / flat-top
slices, three comparisons on the SHARED representation:

* j(rho): our flux-surface-averaged toroidal current density jphi(rho_hat) vs
  the EnKF/TORAX analysis-arm j_total(rho_norm) (both a current profile on a
  normalised minor-radius coordinate — the kind='j' head-to-head lock);
* pitch(R): both predictors' MSE pitch at the sightlines against the held-out
  MSE truth (points + measured error);
* the Ejima resistive flux-consumption C_E(t) waveform backed out of our engine's
  own flux ledger (resistive axis loop voltage + windowed C_E over the ramp).

Firewall unchanged: our engine consumes only Ip + centroid + measured drives;
eta(psi_N) frozen from the tune-split measured-drive calibration; MSE is the
held-out validator only.  EnKF is the classical comparator (measured non-MSE
inputs + Te(rho) -> neoclassical sigma; MSE never on its input side either).

Figures: docs/figures/mse-gated-reanalysis/fig-compare-*.png
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("mse_profile_compare")

FIGURES = Path("docs/figures/mse-gated-reanalysis")
MU0 = 4.0e-7 * np.pi


# ---------------------------------------------------------------------------
# our engine — per-slice jphi(rho_hat), pitch, and the flux ledger for Ejima
# ---------------------------------------------------------------------------


def run_ours(shot: int, *, nr=65, nz=97, max_slices=16, min_ip_ka=60.0,
             sigma=0.02, prior_weight=0.3, n_sub=24, par_weight=1.0, n_rho=24) -> dict:
    from imas_ambix.latent.boundary_disc import disc_read  # noqa: PLC0415
    from imas_ambix.latent.current_diffusion import (  # noqa: PLC0415
        EtaProfile,
        flux_surface_geometry,
        predicted_current,
    )
    from imas_ambix.statespace.mse_eval import (  # noqa: PLC0415
        pitch_from_current_profile,
    )
    from scripts.closure_gate_eval import fit_and_read_slice  # noqa: PLC0415
    from scripts.current_diffusion_gate_eval import (  # noqa: PLC0415
        predict_interval,
        raw_ip_stream,
    )
    from scripts.heldout_mse_gate_eval import (  # noqa: PLC0415
        A_MINOR_M,
        R0_M,
        _axis,
        _campaign_table,
        _confined,
        frozen_eta_params,
        shot_bt0,
    )
    from scripts.position_controlled_solve_gate import _disc_seed_flat  # noqa: PLC0415
    from scripts.spine_label_factory import (  # noqa: PLC0415
        factory_shot_payloads,
        frozen_spine_config,
    )

    spine, _sha = frozen_spine_config()
    iso = spine["interior_solve"]
    n_p, n_f = int(iso["n_p"]), int(iso["n_f"])
    nonneg = iso["profile_kind"] == "monomial-nonneg"
    smoothness = float(iso["smoothness"])
    boundary_read = iso["boundary_read_scoring"]
    eta = EtaProfile.from_vector(np.asarray(frozen_eta_params(), dtype=np.float64))

    table = _campaign_table(shot)
    payload = factory_shot_payloads(shot, nr=nr, nz=nz, max_slices=max_slices,
                                    min_ip_ka=min_ip_ka, table=table)
    if payload is None:
        return {"shot": shot, "slices": []}
    grid, tab, basis = payload["grid"], payload["table"], payload["basis"]
    raw = raw_ip_stream(shot)
    raw_times, ip_raw = raw
    off = np.zeros_like(payload["payloads"][0].mask, dtype=bool)
    order = np.argsort([p.time_s for p in payload["payloads"]])
    bt0 = shot_bt0(shot)
    rpos = [float(r) for r in payload["channels"]] if False else None  # from manifest

    def _fit(p, *, n_p_, n_f_, nonneg_, warm, centroid, coeff_prior=None):
        return fit_and_read_slice(
            grid, tab, dataclasses.replace(p, mask=off), beta0_grid=(0.5,),
            alpha_grid=(1.0,), cost_limit=float("inf"), convergence_limit=5e-3,
            retry_max_iterations=160, fit_mode="ladder", n_p=n_p_, n_f=n_f_,
            nonneg=nonneg_, smoothness=smoothness, warm_jphi=warm,
            centroid_constraint=(centroid[0], centroid[1], sigma),
            coeff_prior=coeff_prior, reseed_axis_r_max=None, keep_psi=True,
            keep_jphi=True, basis=basis, meta={}, boundary_read=boundary_read)

    slices = []
    warm_k2 = None
    for k in order:
        p = payload["payloads"][int(k)]
        inv = disc_read(p, grid, tab, basis)
        if inv is None or inv.ring is None:
            continue
        centroid = (float(inv.centroid_r), float(inv.centroid_z))
        seed = _disc_seed_flat(grid, inv)
        f_k2 = _fit(p, n_p_=1, n_f_=1, nonneg_=False,
                    warm=warm_k2 if warm_k2 is not None else seed, centroid=centroid)
        if not f_k2.scored:
            continue
        conf = f_k2.jphi_flat is not None and _confined(_axis(f_k2)[0])
        if conf:
            warm_k2 = f_k2.jphi_flat
        slices.append({"p": p, "centroid": centroid, "k2": f_k2.jphi_flat if conf else seed})
    if len(slices) < 2:
        return {"shot": shot, "slices": []}

    lab_ip = np.array([abs(s["p"].ip_amperes) for s in slices])
    raw_at = np.interp([s["p"].time_s for s in slices], raw_times, ip_raw)
    good = raw_at > 0
    ip_scale = float(np.median(lab_ip[good] / raw_at[good])) if good.any() else 1e3
    ip_raw_amp = ip_raw * ip_scale

    # rich uncoupled + geometry
    f_uncs, geos = [], []
    for s in slices:
        fu = _fit(s["p"], n_p_=n_p, n_f_=n_f, nonneg_=nonneg, warm=s["k2"], centroid=s["centroid"])
        f_uncs.append(fu)
        if fu.scored and fu.psi is not None and fu.coeffs is not None and _confined(_axis(fu)[0]):
            geos.append(flux_surface_geometry(
                fu.psi, grid, coeffs=np.asarray(fu.coeffs, float), ip_amperes=abs(float(fu.ip_amperes)),
                n_p=n_p, n_f=n_f, nonneg=nonneg, b_phi0=bt0, n_rho=n_rho))
        else:
            geos.append(None)

    preds = [None] * len(slices)
    budgets = [None] * len(slices)
    for j in range(len(slices) - 1):
        if geos[j] is None:
            continue
        out = predict_interval(geos[j], eta, t_start=slices[j]["p"].time_s,
                               t_end=slices[j + 1]["p"].time_s, raw_times=raw_times,
                               ip_raw_amp=ip_raw_amp, n_p=n_p, n_f=n_f, nonneg=nonneg,
                               n_sub=n_sub, par_weight=par_weight)
        if out is not None:
            preds[j + 1] = out
            budgets[j] = out["budget"] | {"t0": slices[j]["p"].time_s,
                                          "t1": slices[j + 1]["p"].time_s,
                                          "ip0": abs(slices[j]["p"].ip_amperes),
                                          "ip1": abs(slices[j + 1]["p"].ip_amperes)}

    rows = []
    for j, s in enumerate(slices):
        p = s["p"]
        c_pred = preds[j]["c_pred"] if preds[j] is not None else None
        if c_pred is not None:
            fc = _fit(p, n_p_=n_p, n_f_=n_f, nonneg_=nonneg, warm=s["k2"],
                      centroid=s["centroid"], coeff_prior=(c_pred, prior_weight))
        else:
            fc = f_uncs[j]
        cr = _axis(fc)[0]
        rho = jtor = None
        if fc.scored and fc.psi is not None and fc.coeffs is not None and _confined(cr):
            geo = flux_surface_geometry(fc.psi, grid, coeffs=np.asarray(fc.coeffs, float),
                                        ip_amperes=abs(float(fc.ip_amperes)), n_p=n_p, n_f=n_f,
                                        nonneg=nonneg, b_phi0=bt0, n_rho=n_rho)
            if geo is not None:
                jtor = np.asarray(predicted_current(geo, geo.psi_face, np.zeros_like(geo.psi_face), eta)["j_tor"], float)
                rho = np.asarray(geo.rho_cell, float)
        rows.append({"time_s": float(p.time_s), "ip_a": float(abs(p.ip_amperes)),
                     "confined": bool(_confined(cr)),
                     "rho": None if rho is None else rho.tolist(),
                     "jtor": None if jtor is None else jtor.tolist(),
                     "fit": fc, "grid": grid, "bt0": bt0})
    return {"shot": shot, "bt0": bt0, "a_minor": A_MINOR_M, "r0": R0_M,
            "rows": rows, "budgets": [b for b in budgets if b is not None],
            "_pitch_fn": pitch_from_current_profile}


def ours_pitch(row, rpos) -> np.ndarray:
    """Our pitch (C,) at the sightlines for a captured slice (kind='j')."""
    from scripts.heldout_mse_gate_eval import A_MINOR_M, R0_M  # noqa: PLC0415
    from imas_ambix.statespace.mse_eval import pitch_from_current_profile  # noqa: PLC0415

    if row["rho"] is None:
        return np.full(len(rpos), np.nan)
    rho_m = np.asarray(row["rho"], float) * A_MINOR_M
    return pitch_from_current_profile(np.asarray(row["jtor"], float), rho_m,
                                      np.asarray(rpos, float), R0_M, row["bt0"], kind="j")


# ---------------------------------------------------------------------------
# EnKF (live TORAX) — analysis-arm mean pitch + j(rho) on the manifest grid
# ---------------------------------------------------------------------------


def run_enkf(shots, manifest) -> dict:
    import torax  # noqa: PLC0415

    torax.set_jax_precision()
    from imas_ambix.statespace import enkf_baseline as enkf  # noqa: PLC0415

    cfg = enkf.EnKFConfig(n_ensemble=48)
    grid = {}
    for sid in shots:
        e = manifest["shots"].get(str(int(sid)))
        if e and e.get("partition") == "held_out":
            grid[int(sid)] = {"t": np.asarray(e["beam_on_slice_times"], float),
                              "rpos": np.asarray(e["active_channel_rpos"], float)}
    _preds, results = enkf.predict_shots(shots, cfg, arm="analysis",
                                         return_results=True, manifest_grid=grid)
    out = {}
    for sid, res in results.items():
        pm = np.nanmean(res.pitch_samples, axis=2)  # (K,C)
        out[int(sid)] = {"t": np.asarray(res.slice_t, float), "pitch": pm,
                         "j": None if res.j_analysis is None else np.asarray(res.j_analysis, float),
                         "rho": None if res.rho_analysis is None else np.asarray(res.rho_analysis, float)}
    return out


# ---------------------------------------------------------------------------
# slice selection + figures
# ---------------------------------------------------------------------------


def pick_times(entry, ours) -> list[tuple[str, float, int]]:
    """(label, time, manifest-slice-index) at rampup / mid / flat-top.

    Indices are chosen from the PITCH-VALID beam-on slices inside our engine's
    confined coverage window, so the MSE truth (and the EnKF prediction, both on
    the manifest grid) exist at every selected slice.  Flat-top = our max-Ip
    confined slice, snapped to the nearest pitch-valid index.
    """
    t_beam = np.asarray(entry["beam_on_slice_times"], float)
    pv = np.asarray(entry["pitch_valid_mask"], bool)
    conf = [r for r in ours["rows"] if r["confined"] and r["rho"] is not None]
    if not conf:
        return []
    tmin = min(r["time_s"] for r in conf)
    tmax = max(r["time_s"] for r in conf)
    idx = np.where(pv & (t_beam >= tmin) & (t_beam <= tmax))[0]
    if idx.size < 3:
        idx = np.where(pv)[0]
    if idx.size < 3:
        return []
    ip = np.array([r["ip_a"] for r in conf])
    t_ft = conf[int(np.argmax(ip))]["time_s"]
    k_ft = int(idx[int(np.argmin(np.abs(t_beam[idx] - t_ft)))])
    k_ru = int(idx[0])
    k_mid = int(idx[idx.size // 2])
    return [("ramp-up", float(t_beam[k_ru]), k_ru),
            ("mid", float(t_beam[k_mid]), k_mid),
            ("flat-top", float(t_beam[k_ft]), k_ft)]


def _nearest_row(ours, t):
    cand = [r for r in ours["rows"] if r["confined"] and r["rho"] is not None]
    return min(cand, key=lambda r: abs(r["time_s"] - t)) if cand else None


def fig_j_compare(data, path):
    shots = list(data)
    ncol = 3
    fig, axes = plt.subplots(len(shots), ncol, figsize=(4.0 * ncol, 3.3 * len(shots)),
                             squeeze=False)
    for i, sid in enumerate(shots):
        d = data[sid]
        for c, (lab, t, k) in enumerate(d["times"]):
            ax = axes[i][c]
            row = _nearest_row(d["ours"], t)
            if row is not None:
                ax.plot(np.asarray(row["rho"]), np.asarray(row["jtor"]) / 1e6, "-",
                        color="#268", lw=1.8, label="ours jφ(ρ̂)")
            ek = d["enkf"].get(int(sid))
            if ek and ek["j"] is not None and k < ek["j"].shape[0]:
                ax.plot(ek["rho"], ek["j"][k] / 1e6, "--", color="#c66", lw=1.6,
                        label="EnKF j(ρ)")
            ax.set_title(f"{sid}  {lab}  t={t:.3f}s", fontsize=8)
            ax.set_xlabel("ρ (norm. minor radius)"); ax.set_xlim(0, 1)
            if c == 0:
                ax.set_ylabel("j$_φ$ [MA/m²]")
            if i == 0 and c == 0:
                ax.legend(fontsize=7)
    fig.suptitle("Toroidal current-density profile: our reanalysis engine vs EnKF/TORAX", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=130); plt.close(fig)


def fig_pitch_compare(data, truth, path):
    shots = list(data)
    ncol = 3
    fig, axes = plt.subplots(len(shots), ncol, figsize=(4.0 * ncol, 3.3 * len(shots)),
                             squeeze=False)
    for i, sid in enumerate(shots):
        d = data[sid]
        entry = d["entry"]
        rpos = np.asarray(entry["active_channel_rpos"], float)
        tr = truth.get(int(sid))
        for c, (lab, t, k) in enumerate(d["times"]):
            ax = axes[i][c]
            # truth indexed by the manifest slice k (tr.time == beam_on_slice_times)
            if tr is not None and k < np.asarray(tr.pitch).shape[0]:
                from imas_ambix.statespace.mse_split import pitch_point_gate  # noqa: PLC0415
                g = pitch_point_gate(tr.pitch[k:k + 1], tr.pitch_error[k:k + 1])[0]
                pe = np.where(np.isfinite(tr.pitch_error[k]) & (tr.pitch_error[k] > 0),
                              tr.pitch_error[k], np.nan)
                ax.errorbar(rpos[g], tr.pitch[k][g], yerr=pe[g], fmt="o", ms=3.5,
                            color="k", ecolor="#999", capsize=2, lw=1, label="MSE truth", zorder=5)
            row = _nearest_row(d["ours"], t)
            if row is not None:
                ax.plot(rpos, ours_pitch(row, rpos), "-", color="#268", lw=1.8, label="ours")
            ek = d["enkf"].get(int(sid))
            if ek is not None and k < ek["pitch"].shape[0]:
                ax.plot(rpos, ek["pitch"][k], "--", color="#c66", lw=1.6, label="EnKF")
            ax.axvline(d["ours"]["r0"], color="k", ls=":", lw=0.7)
            ax.set_title(f"{sid}  {lab}  t={t:.3f}s", fontsize=8)
            ax.set_xlabel("sightline R [m]")
            if c == 0:
                ax.set_ylabel("MSE pitch [rad]")
            if i == 0 and c == 0:
                ax.legend(fontsize=7)
    fig.suptitle("MSE pitch: our engine and EnKF vs the held-out MSE truth", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=130); plt.close(fig)


def fig_ejima(data, path):
    shots = list(data)
    fig, axes = plt.subplots(1, len(shots), figsize=(4.6 * len(shots), 3.6), squeeze=False)
    for i, sid in enumerate(shots):
        d = data[sid]
        r0 = d["ours"]["r0"]
        buds = d["ours"]["budgets"]
        tm, ce, vres, ramp = [], [], [], []
        for b in buds:
            t_mid = 0.5 * (b["t0"] + b["t1"])
            dip = b["ip1"] - b["ip0"]
            dt = max(b["t1"] - b["t0"], 1e-9)
            tm.append(t_mid)
            vres.append(abs(b["d_psi_axis"]) / dt)  # resistive axis loop voltage [V]
            is_ramp = dip > 0.02 * b["ip1"]
            ramp.append(is_ramp)
            ce.append(abs(b["d_psi_axis"]) / (MU0 * r0 * abs(dip)) if is_ramp and abs(dip) > 0 else np.nan)
        tm = np.array(tm); ce = np.array(ce); vres = np.array(vres); ramp = np.array(ramp)
        ax = axes[0][i]
        ax.plot(tm, vres, "o-", color="#268", lw=1.4, ms=3, label="resistive V$_{axis}$ [V]")
        ax2 = ax.twinx()
        ax2.plot(tm[ramp], ce[ramp], "s--", color="#c66", lw=1.4, ms=4, label="Ejima C$_E$ (ramp)")
        ax.set_title(f"shot {sid} — flux consumption", fontsize=9)
        ax.set_xlabel("time [s]"); ax.set_ylabel("resistive axis loop voltage [V]", color="#268")
        ax2.set_ylabel("incremental Ejima C$_E$", color="#c66")
        lines = ax.get_lines() + ax2.get_lines()
        ax.legend(lines, [ln.get_label() for ln in lines], fontsize=7, loc="best")
    fig.suptitle("Ejima resistive flux-consumption backed out of our engine's flux ledger "
                 "(EnKF/TORAX C$_E$ not exposed by the baseline)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=130); plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shots", type=str, default="21983,21985")
    ap.add_argument("--skip-enkf", action="store_true")
    args = ap.parse_args()
    shots = [int(s) for s in args.shots.split(",") if s.strip()]

    from imas_ambix.eval import prediction_bar as pbar  # noqa: PLC0415
    from imas_ambix.statespace import mse_eval  # noqa: PLC0415
    from imas_ambix.data.paths import LEVEL1_DIR  # noqa: PLC0415

    manifest = pbar.load_locked_manifest()
    truth = mse_eval.MseTruth(level1_dir=LEVEL1_DIR)

    enkf = {} if args.skip_enkf else run_enkf(shots, manifest)

    data = {}
    for sid in shots:
        logger.info("our engine: shot %d", sid)
        ours = run_ours(sid)
        entry = manifest["shots"].get(str(int(sid)))
        times = pick_times(entry, ours) if ours["rows"] else []
        data[sid] = {"ours": ours, "enkf": enkf, "entry": entry, "times": times}
        logger.info("shot %d: %d confined slices, times=%s", sid,
                    sum(1 for r in ours["rows"] if r["confined"]), times)

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig_j_compare(data, FIGURES / "fig-compare-jrho.png")
    fig_pitch_compare(data, truth, FIGURES / "fig-compare-pitch.png")
    fig_ejima(data, FIGURES / "fig-compare-ejima.png")
    logger.info("wrote figures to %s", FIGURES)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
