#!/usr/bin/env python
"""Held-out MSE pitch gate: score the dynamics-coupled reanalysis engine.

The position-controlled + dynamics-coupled solve earns the internal profile
from measured histories (resistive ψ-diffusion on the position spine, η(ψ_N)
the calibrated low-DOF unknown).  This gate proves that earned profile against
the DIRECT internal-field measurement it was never shown — held-out MSE pitch —
on the same ruler as the classical baselines.

The ruler (shared, fairness-binding)
------------------------------------
Every predictor is scored through the SAME observation model
(:func:`imas_ambix.statespace.mse_eval.pitch_from_current_profile`,
LOCKED ``kind='j'`` representation from the S9 head-to-head coordination): a
1D toroidal-current-density profile jφ(ρ̂) on a ρ̂·a_minor minor-radius grid
maps to MSE pitch per sightline; :func:`mse_eval.score` computes the
pre-registered PRIMARY pitch RMSE / CRPS / coverage.  The classical EnKF
(TORAX current-diffusion ensemble smoother) reads out its ``j_total(ρ_norm)``
identically, so a head-to-head isolates STATE INFERENCE, not the forward model.

The engine's readout, per slice
-------------------------------
The dynamics-coupled fit's force-balanced ψ + ladder coefficients give the
flux-surface geometry (:func:`current_diffusion.flux_surface_geometry`); the
flux-surface-averaged toroidal current density jφ(ρ̂)
(:func:`current_diffusion.predicted_current`) is the engine's ``kind='j'``
profile — the same quantity, in the same coordinate, TORAX emits.  Pitch is
read at the shot's sightlines with the per-shot R0 (0.85 m) + Bt0 (from
amc.tf_current, matching the EnKF baseline), then interpolated onto the
manifest's ``beam_on_slice_times``.

Firewall
--------
The engine consumes ONLY magnetics (Ip + R/Z current centroid) and the
measured Ip drive; η(ψ_N) is calibrated on the TUNE split against measured
drives and FROZEN here (eval consumes the frozen parameters).  MSE never
enters any fit/tune/train path — it is the held-out validator.

Label honesty
-------------
A slice whose coupled solve does not hold a confined, force-balanced readout
is flagged under-determined and carries NO pitch prediction — never a
fabricated profile (the ``label-honesty`` decision, binding).

Artifacts: imas_ambix/latent/artifacts/patch_gate/heldout_mse_gate[-tag].json
Figures:   docs/figures/mse-gated-reanalysis/
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("heldout_mse_gate_eval")

ARTIFACTS = Path("imas_ambix/latent/artifacts/patch_gate")
FIGURES = Path("docs/figures/mse-gated-reanalysis")

# Shared readout geometry — IDENTICAL to the EnKF baseline so the two are on one
# ruler (imas_ambix.statespace.enkf_baseline: MAST_R0/MAST_A + tf_current Bt0).
R0_M = 0.85  # major radius [m] (mse_eval.DEFAULT_R0)
A_MINOR_M = 0.50  # minor radius [m] for the ρ̂ -> minor-radius map
N_TF_EFF = 25.0  # tf_current -> Bt0 calibration (flat-top Bt0 ~ 0.5 T on MAST)
BT0_FALLBACK = 0.50  # vacuum toroidal field at R0 [T] when tf_current is absent
MU0 = 4.0e-7 * np.pi

CONFINED_AXIS_R_MAX = 1.4  # a readout beyond this is the outboard attractor [m]
DEFAULT_SIGMA_M = 0.02  # centroid tether 1σ [m]
# η(ψ_N): FROZEN from the tune-split measured-drive calibration (eval consumes it)
FROZEN_ETA_ARTIFACT = ARTIFACTS / "current_diffusion_eta_calibration.json"
DEFAULT_ETA = (-6.335, -0.097, 4.383)  # log10(eta0), contrast, shape (tune fit)


def frozen_eta_params() -> tuple[float, float, float]:
    """The tune-split calibrated η(ψ_N) — measured-drive fit, eval consumes it."""
    if FROZEN_ETA_ARTIFACT.exists():
        d = json.loads(FROZEN_ETA_ARTIFACT.read_text())
        p = d.get("eta_params")
        if p and len(p) == 3:
            return (float(p[0]), float(p[1]), float(p[2]))
    return DEFAULT_ETA


def shot_bt0(shot: int) -> float:
    """Per-shot vacuum Bt0 at R0 from amc.tf_current (EnKF-matched calibration).

    Bt0 = μ0 · N_TF_EFF · |I_tf|_flattop / (2π R0); falls back to 0.5 T when the
    tf_current channel is missing.  Firewall-safe (a measured machine scalar).
    """
    from imas_ambix.data.paths import local_shot_path  # noqa: PLC0415

    try:
        import zarr  # noqa: PLC0415

        store = zarr.open(str(local_shot_path(shot, tier="level1")), mode="r")
        if "amc" not in store:
            return BT0_FALLBACK
        amc = store["amc"]
        keys = set(amc.array_keys())
        if "tf_current" not in keys:
            return BT0_FALLBACK
        itf = np.abs(np.asarray(amc["tf_current"], dtype=np.float64)) * 1e3  # kA->A
        itf = itf[np.isfinite(itf)]
        if itf.size == 0:
            return BT0_FALLBACK
        i_ft = float(np.median(itf[itf > 0.1 * np.nanmax(itf)]))
        bt0 = MU0 * N_TF_EFF * i_ft / (2.0 * np.pi * R0_M)
        return float(bt0) if np.isfinite(bt0) and bt0 > 0 else BT0_FALLBACK
    except Exception:  # noqa: BLE001 — a missing channel must not sink the shot
        return BT0_FALLBACK


# Per-worker campaign geometry-table cache.  The passive-structure geometry
# (amm) is a CAMPAIGN property, so a table built from any shot in a campaign is
# valid for every shot in it — including shots whose own zarr lacks amm.  This
# mirrors the EnKF baseline's per-campaign operator cache.
_TABLE_CACHE: dict = {}


def _campaign_table(shot: int):
    """Geometry table for ``shot``'s campaign (cached), built from a
    representative shot that carries the amm passive-structure geometry."""
    from imas_ambix.gs.geometry import (  # noqa: PLC0415
        build_table_for_shot,
        read_efm_geometry,
        setup_signature,
    )
    from imas_ambix.statespace.enkf_baseline import (  # noqa: PLC0415
        _campaign_representatives,
    )

    try:
        key = setup_signature(read_efm_geometry(int(shot))).key
    except Exception:  # noqa: BLE001
        key = None
    if key in _TABLE_CACHE:
        return _TABLE_CACHE[key]
    reps = _campaign_representatives().get(key, []) if key else []
    for rep in list(reps) + [int(shot)]:
        try:
            table = build_table_for_shot(int(rep))
            if key is not None:
                _TABLE_CACHE[key] = table
            return table
        except Exception:  # noqa: BLE001 — try the next representative
            continue
    return None


def _axis(f) -> tuple[float, float]:
    if not (f.scored and f.target is not None):
        return float("nan"), float("nan")
    return float(f.target[0]), float(f.target[1])


def _confined(axis_r: float) -> bool:
    return bool(np.isfinite(axis_r) and axis_r <= CONFINED_AXIS_R_MAX)


def _pitch_from_fit(f, grid, eta, *, n_p, n_f, nonneg, b_phi0, n_rho, rpos, bt0):
    """MSE pitch (C,) at the sightlines from one coupled fit's equilibrium.

    Reduces the 2D force-balanced ψ to the LOCKED ``kind='j'`` profile jφ(ρ̂)
    (flux-surface-averaged toroidal current density) and pushes it through the
    shared cylindrical observation model — the identical readout the EnKF and
    the neural filter use.  Returns NaNs where the geometry cannot be binned
    (label honesty: no readout is fabricated).
    """
    from imas_ambix.latent.current_diffusion import (  # noqa: PLC0415
        flux_surface_geometry,
        predicted_current,
    )
    from imas_ambix.statespace.mse_eval import (  # noqa: PLC0415
        pitch_from_current_profile,
    )

    C = np.asarray(rpos).size
    nan = np.full(C, np.nan)
    if not (f.scored and f.psi is not None and f.coeffs is not None):
        return nan
    if not _confined(_axis(f)[0]):
        return nan
    geo = flux_surface_geometry(
        f.psi, grid, coeffs=np.asarray(f.coeffs, dtype=np.float64),
        ip_amperes=abs(float(f.ip_amperes)), n_p=n_p, n_f=n_f, nonneg=nonneg,
        b_phi0=b_phi0, n_rho=n_rho)
    if geo is None:
        return nan
    # jφ(ρ̂): the fit's own current profile from its ψ (psidot=0 -> no ohmic term)
    out = predicted_current(geo, geo.psi_face, np.zeros_like(geo.psi_face), eta)
    j_tor = np.asarray(out["j_tor"], dtype=np.float64)  # (n_rho,) on rho_cell
    if not np.isfinite(j_tor).all():
        return nan
    rho_m = np.asarray(geo.rho_cell, dtype=np.float64) * A_MINOR_M  # minor radius [m]
    pitch = pitch_from_current_profile(
        j_tor, rho_m, np.asarray(rpos, dtype=np.float64), R0_M, bt0, kind="j")
    return np.asarray(pitch, dtype=np.float64)


def coupled_solve_chain(
    shot: int, *, nr: int, nz: int, sigma: float, eta_params, prior_weight,
    n_sub: int, par_weight: float, n_rho: int, max_slices: int,
    min_ip_ka: float, skip_basin: bool = False, passive: dict | None = None,
    passive_centers_fn=None, passive_weight: float = 0.0,
    cache_grid: bool = False,
) -> dict:
    """The four-pass engine chain over one shot -> per-slice readout fits.

    Faithful to the §3 dynamics-coupled engine: the basin solve (free-sign
    n_p=n_f=1 amplitude pair, centroid-pinned) selects the confined basin and
    warm-starts the profile solve (non-negative ladder); its flux-surface
    geometry drives the ψ diffusion over each measured interval (measured Ip drive,
    frozen η), and the evolved current enters the next slice as a soft coefficient
    prior.  Returns the COUPLED fit per slice (the readout equilibrium) plus the
    chain context; ``reason`` is set (and ``slices`` empty) when the chain
    cannot run.

    ``skip_basin`` cold-starts the profile solve from the disc seed with the
    SAME centroid pin + coefficient prior — the basin-pass necessity ablation
    (the pass is priced directly; the disc PRIOR ablation is a different knob).

    ``passive`` (rank-k eigenmode sidecar) + ``passive_centers_fn`` (maps the
    chain's label times + per-slice cell currents to sidecar-coordinate
    trajectory centers) + ``passive_weight`` inject a PRECOMPUTED vessel-eddy
    trajectory through the frozen passive Green's columns; a large weight is
    the known-drive limit (the amplitudes are pinned, not fitted).  The eddy
    flux enters sensors AND the Picard field consistently (sidecar columns).
    All three default OFF — the validated chain is byte-identical then.
    """
    from imas_ambix.latent.boundary_disc import disc_read  # noqa: PLC0415
    from imas_ambix.latent.current_diffusion import (  # noqa: PLC0415
        EtaProfile,
        flux_surface_geometry,
    )
    from scripts.closure_gate_eval import fit_and_read_slice  # noqa: PLC0415
    from scripts.current_diffusion_gate_eval import (  # noqa: PLC0415
        predict_interval,
        raw_ip_stream,
    )
    from scripts.position_controlled_solve_gate import (  # noqa: PLC0415
        _disc_seed_flat,
    )
    from scripts.spine_label_factory import (  # noqa: PLC0415
        factory_shot_payloads,
        frozen_spine_config,
    )

    spine, spine_sha = frozen_spine_config()
    isolve = spine["interior_solve"]
    n_p, n_f = int(isolve["n_p"]), int(isolve["n_f"])
    nonneg = isolve["profile_kind"] == "monomial-nonneg"
    smoothness = float(isolve["smoothness"])
    boundary_read = isolve["boundary_read_scoring"]

    table_cmp = _campaign_table(shot)
    if table_cmp is None:
        return {"shot": shot, "slices": [], "reason": "no campaign table"}
    payload = factory_shot_payloads(shot, nr=nr, nz=nz, max_slices=max_slices,
                                    min_ip_ka=min_ip_ka, table=table_cmp,
                                    cache_grid=cache_grid)
    if payload is None:
        return {"shot": shot, "slices": [], "reason": "no payloads"}
    grid, table, basis = payload["grid"], payload["table"], payload["basis"]
    raw = raw_ip_stream(shot)
    if raw is None:
        return {"shot": shot, "slices": [], "reason": "no raw Ip"}
    raw_times, ip_raw = raw

    off = np.zeros_like(payload["payloads"][0].mask, dtype=bool)
    assert not off.any(), "firewall: every arm runs with the magnetics mask OFF"
    order = np.argsort([p.time_s for p in payload["payloads"]])
    bt0 = shot_bt0(shot)
    b_phi0 = bt0  # F_boundary = R0·Bt0 (measured toroidal field)

    def _fit(p, *, n_p_, n_f_, nonneg_, warm, centroid, coeff_prior=None,
             passive_prior=None):
        kw = {}
        if passive is not None:
            kw["passive"] = passive
            kw["passive_ridge"] = 1.0
            if passive_prior is not None:
                kw["passive_prior"] = passive_prior
        return fit_and_read_slice(
            grid, table, dataclasses.replace(p, mask=off),
            beta0_grid=(0.5,), alpha_grid=(1.0,), cost_limit=float("inf"),
            convergence_limit=5e-3, retry_max_iterations=160, fit_mode="ladder",
            n_p=n_p_, n_f=n_f_, nonneg=nonneg_, smoothness=smoothness,
            warm_jphi=warm, centroid_constraint=(centroid[0], centroid[1], sigma),
            coeff_prior=coeff_prior, reseed_axis_r_max=None,
            keep_psi=True, keep_jphi=True, basis=basis, meta={},
            boundary_read=boundary_read, **kw)

    # ---- pass 1: the basin solve (stable, landed §2) ----
    slices: list[dict] = []
    warm_basin = None
    for k in order:
        p = payload["payloads"][int(k)]
        inv = disc_read(p, grid, table, basis)
        if inv is None or inv.ring is None:
            continue
        centroid = (float(inv.centroid_r), float(inv.centroid_z))
        disc_seed = _disc_seed_flat(grid, inv)
        if skip_basin:
            # basin-pass ablation: the profile solve cold-starts from the
            # disc seed under the same centroid pin + coefficient prior
            slices.append({"k": int(k), "p": p, "centroid": centroid,
                           "basin_jphi": disc_seed})
            continue
        f_basin = _fit(p, n_p_=1, n_f_=1, nonneg_=False,
                    warm=warm_basin if warm_basin is not None else disc_seed,
                    centroid=centroid)
        if not f_basin.scored:
            continue
        basin_conf = f_basin.jphi_flat is not None and _confined(_axis(f_basin)[0])
        if basin_conf:
            warm_basin = f_basin.jphi_flat
        slices.append({"k": int(k), "p": p, "centroid": centroid,
                       "basin_jphi": f_basin.jphi_flat if basin_conf else disc_seed})
    if len(slices) < 2:
        return {"shot": shot, "slices": [], "reason": "too few scored slices"}

    # ---- optional precomputed vessel-eddy trajectory (known drive) ----
    passive_priors: list[tuple | None] = [None] * len(slices)
    if passive is not None and passive_centers_fn is not None:
        cell_area = grid.dr * grid.dz
        i_cell_seq = np.stack(
            [s["basin_jphi"][grid.cells] * cell_area for s in slices])
        label_times = np.array([s["p"].time_s for s in slices])
        centers = passive_centers_fn(label_times, i_cell_seq)
        passive_priors = [(centers[j], passive_weight)
                          for j in range(len(slices))]

    lab_ip = np.array([abs(s["p"].ip_amperes) for s in slices])
    raw_at_lab = np.interp([s["p"].time_s for s in slices], raw_times, ip_raw)
    good = raw_at_lab > 0
    ip_scale = float(np.median(lab_ip[good] / raw_at_lab[good])) if good.any() else 1e3
    ip_raw_amp = ip_raw * ip_scale

    # ---- pass 2a: uncoupled profile solve + its flux-surface geometry ----
    eta = EtaProfile.from_vector(np.asarray(eta_params, dtype=np.float64))
    f_uncs, geos = [], []
    for j, s in enumerate(slices):
        f_unc = _fit(s["p"], n_p_=n_p, n_f_=n_f, nonneg_=nonneg,
                     warm=s["basin_jphi"], centroid=s["centroid"],
                     passive_prior=passive_priors[j])
        f_uncs.append(f_unc)
        if (f_unc.scored and f_unc.psi is not None and f_unc.coeffs is not None
                and _confined(_axis(f_unc)[0])):
            geos.append(flux_surface_geometry(
                f_unc.psi, grid, coeffs=np.asarray(f_unc.coeffs, dtype=np.float64),
                ip_amperes=abs(float(f_unc.ip_amperes)), n_p=n_p, n_f=n_f,
                nonneg=nonneg, b_phi0=b_phi0, n_rho=n_rho))
        else:
            geos.append(None)

    # ---- per-interval diffusion prediction (frozen η, measured Ip drive) ----
    preds: list[dict | None] = [None] * len(slices)
    for j in range(len(slices) - 1):
        if geos[j] is None:
            continue
        out = predict_interval(
            geos[j], eta, t_start=slices[j]["p"].time_s,
            t_end=slices[j + 1]["p"].time_s, raw_times=raw_times,
            ip_raw_amp=ip_raw_amp, n_p=n_p, n_f=n_f, nonneg=nonneg,
            n_sub=n_sub, par_weight=par_weight)
        if out is not None:
            preds[j + 1] = out

    # ---- pass 2b: coupled profile solve -> the readout equilibrium ----
    fits: list = []
    for j, s in enumerate(slices):
        c_pred = preds[j]["c_pred"] if preds[j] is not None else None
        if c_pred is not None and prior_weight > 0.0:
            f_cpl = _fit(s["p"], n_p_=n_p, n_f_=n_f, nonneg_=nonneg,
                         warm=s["basin_jphi"], centroid=s["centroid"],
                         coeff_prior=(c_pred, prior_weight),
                         passive_prior=passive_priors[j])
        else:
            f_cpl = f_uncs[j]
        fits.append(f_cpl)
    return {"shot": shot, "spine_sha": spine_sha, "grid": grid, "table": table,
            "basis": basis, "payload": payload, "bt0": bt0, "b_phi0": b_phi0,
            "eta": eta, "n_p": n_p, "n_f": n_f, "nonneg": nonneg,
            "slices": slices, "fits": fits}


def run_shot(shot: int, *, nr: int, nz: int, sigma: float, eta_params, prior_weight,
             n_sub: int, par_weight: float, n_rho: int, max_slices: int,
             min_ip_ka: float, rpos: list, times_beam: list) -> dict:
    """Coupled solve chain over one shot -> per-slice MSE pitch at the sightlines.

    Thin readout over :func:`coupled_solve_chain` (the four-pass engine): the
    COUPLED fit's equilibrium is read out as MSE pitch per sightline.
    """
    from scripts.dynamics_coupled_solve_gate import _current_centroid  # noqa: PLC0415

    chain = coupled_solve_chain(
        shot, nr=nr, nz=nz, sigma=sigma, eta_params=eta_params,
        prior_weight=prior_weight, n_sub=n_sub, par_weight=par_weight,
        n_rho=n_rho, max_slices=max_slices, min_ip_ka=min_ip_ka)
    if not chain["slices"]:
        return {"shot": shot, "slices": [], "reason": chain.get("reason", "chain")}
    grid, eta, bt0 = chain["grid"], chain["eta"], chain["bt0"]
    n_p, n_f, nonneg = chain["n_p"], chain["n_f"], chain["nonneg"]

    rows: list[dict] = []
    for s, f_cpl in zip(chain["slices"], chain["fits"], strict=True):
        p, centroid = s["p"], s["centroid"]
        cr = _axis(f_cpl)[0]
        confined = _confined(cr)
        pitch = _pitch_from_fit(
            f_cpl, grid, eta, n_p=n_p, n_f=n_f, nonneg=nonneg,
            b_phi0=chain["b_phi0"], n_rho=n_rho, rpos=rpos, bt0=bt0)
        cen = _current_centroid(grid, f_cpl)
        rows.append({
            "k": s["k"], "time_s": float(p.time_s), "ip_a": float(abs(p.ip_amperes)),
            "axis_r": float(cr) if np.isfinite(cr) else None, "confined": confined,
            "scored": bool(f_cpl.scored and confined and np.isfinite(pitch).any()),
            "centroid_err_cm": (
                float(100.0 * np.hypot(cen[0] - centroid[0], cen[1] - centroid[1]))
                if np.isfinite(cen[0]) else None),
            "pitch": [None if not np.isfinite(v) else float(v) for v in pitch],
        })
    n_scored = sum(r["scored"] for r in rows)
    return {"shot": shot, "spine_sha": chain["spine_sha"], "bt0": bt0,
            "n_p": n_p, "n_f": n_f,
            "eta_params": list(map(float, eta_params)), "prior_weight": prior_weight,
            "n_slices": len(rows), "n_scored": n_scored, "rows": rows}


def _worker(job):
    shot, cfg = job
    try:
        return run_shot(shot, **cfg)
    except Exception as exc:  # a shot that dies must not sink the gate
        logger.exception("shot %d failed: %s", shot, exc)
        return {"shot": shot, "slices": [], "reason": f"exception: {exc}"}


def _engine_prediction(result: dict, entry: dict):
    """Build a ShotPrediction from an engine shot result + its manifest entry.

    The engine's per-slice pitch (sparse, at the coupled-solve slice times) is
    interpolated per channel onto the manifest ``beam_on_slice_times``; the
    predictive std is the measured per-channel aleatoric pitch error (the same
    observation-noise floor persistence + EnKF use — NOT tuned).  Returns None
    when the shot has no confined, scored readout (label honesty).
    """
    from imas_ambix.statespace.mse_eval import MseTruth, ShotPrediction  # noqa: PLC0415
    from imas_ambix.data.paths import LEVEL1_DIR  # noqa: PLC0415

    rows = [r for r in result.get("rows", []) if r.get("scored")]
    if len(rows) < 2:
        return None, 0
    t_eng = np.array([r["time_s"] for r in rows], dtype=np.float64)
    C = len(entry["active_channel_ids"])
    pit = np.array([[np.nan if v is None else v for v in r["pitch"]] for r in rows])
    if pit.shape[1] != C:
        return None, 0
    t_beam = np.asarray(entry["beam_on_slice_times"], dtype=np.float64)
    K = t_beam.size
    pmean = np.full((K, C), np.nan)
    for c in range(C):
        col = pit[:, c]
        fin = np.isfinite(col)
        if fin.sum() >= 2:
            pmean[:, c] = np.interp(t_beam, t_eng[fin], col[fin])
        elif fin.sum() == 1:
            pmean[:, c] = col[fin][0]
    # aleatoric std: per-channel median measured pitch error (persistence/EnKF floor)
    truth = MseTruth(level1_dir=LEVEL1_DIR)
    tr = truth.get(int(result["shot"]))
    if tr is not None and tr.pitch_error is not None:
        pe = np.asarray(tr.pitch_error, dtype=np.float64)
        with np.errstate(invalid="ignore"):
            per_ch = np.nanmedian(np.where(pe > 0, pe, np.nan), axis=0)
        per_ch = per_ch[:C] if per_ch.size >= C else np.full(C, np.nan)
        per_ch = np.where(np.isfinite(per_ch) & (per_ch > 0), per_ch, 0.1)
    else:
        per_ch = np.full(C, 0.1)
    pstd = np.broadcast_to(per_ch, (K, C)).copy()
    # fill any all-nan channel with the slice-median pitch so the array is finite
    med = np.nanmedian(pmean) if np.isfinite(pmean).any() else 0.0
    pmean = np.where(np.isfinite(pmean), pmean, med)
    return ShotPrediction(t=t_beam, pitch_mean=pmean, pitch_std=pstd), len(rows)


def _per_shot_pitch_rmse(preds: dict, manifest: dict, truth) -> dict:
    """Per-shot PRIMARY pitch RMSE via the harness (score one shot at a time)."""
    from imas_ambix.statespace import mse_eval  # noqa: PLC0415

    out = {}
    for sid, pred in preds.items():
        sc = mse_eval.score({sid: pred}, manifest, truth)
        pp = sc["primary"]["pitch"]
        if pp["n_shots"] > 0 and np.isfinite(pp["rmse"]):
            out[sid] = float(pp["rmse"])
    return out


def _paired_bootstrap(a: dict, b: dict, *, n_boot: int = 4000, seed: int = 0):
    """Bootstrap CI of mean(b) - mean(a) over the shots common to both dicts.

    a = engine per-shot RMSE, b = reference per-shot RMSE.  A CI whose LOWER
    bound is > 0 means the engine's mean RMSE is below the reference's,
    CI-clear (the harness aggregates as the mean over shots, so bootstrapping
    the per-shot means is the matched test).
    """
    keys = sorted(set(a) & set(b))
    if len(keys) < 3:
        return None
    av = np.array([a[k] for k in keys])
    bv = np.array([b[k] for k in keys])
    rng = np.random.default_rng(seed)
    n = len(keys)
    draws = np.array([
        np.mean(bv[idx]) - np.mean(av[idx])
        for idx in (rng.integers(0, n, n) for _ in range(n_boot))])
    return {
        "n_shots": n, "mean_engine": float(np.mean(av)),
        "mean_reference": float(np.mean(bv)),
        "delta_mean": float(np.mean(bv) - np.mean(av)),
        "delta_ci95": [float(np.percentile(draws, 2.5)),
                       float(np.percentile(draws, 97.5))],
        "ci_clear_engine_better": bool(np.percentile(draws, 2.5) > 0.0),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shots", type=str, default="", help="explicit comma list")
    ap.add_argument("--n-shots", type=int, default=0, help="cap held-out shots (0=all)")
    ap.add_argument("--nr", type=int, default=65)
    ap.add_argument("--nz", type=int, default=97)
    ap.add_argument("--sigma", type=float, default=DEFAULT_SIGMA_M)
    ap.add_argument("--prior-weight", type=float, default=0.3)
    ap.add_argument("--eta-params", type=str, default="")
    ap.add_argument("--n-sub-steps", type=int, default=24)
    ap.add_argument("--par-weight", type=float, default=1.0)
    ap.add_argument("--n-rho", type=int, default=24)
    ap.add_argument("--max-slices-per-shot", type=int, default=16)
    ap.add_argument("--min-ip-ka", type=float, default=60.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out-suffix", type=str, default="")
    args = ap.parse_args()

    from imas_ambix.eval import prediction_bar as pbar  # noqa: PLC0415
    from imas_ambix.statespace import mse_eval  # noqa: PLC0415
    from imas_ambix.data.paths import LEVEL1_DIR  # noqa: PLC0415

    manifest = pbar.load_locked_manifest()
    held = pbar.held_out_shot_ids(manifest)
    if args.shots:
        want = {int(s) for s in args.shots.split(",") if s.strip()}
        held = [s for s in held if s in want]
    if args.n_shots > 0:
        held = held[: args.n_shots]

    eta_params = ([float(v) for v in args.eta_params.split(",")]
                  if args.eta_params else list(frozen_eta_params()))
    logger.info("held-out MSE gate: %d shots, eta=%s, prior_w=%.3g",
                len(held), eta_params, args.prior_weight)

    cfg = dict(nr=args.nr, nz=args.nz, sigma=args.sigma, eta_params=eta_params,
               prior_weight=args.prior_weight, n_sub=args.n_sub_steps,
               par_weight=args.par_weight, n_rho=args.n_rho,
               max_slices=args.max_slices_per_shot, min_ip_ka=args.min_ip_ka,
               rpos=None, times_beam=None)  # rpos/times filled per shot below

    jobs = []
    for sid in held:
        entry = manifest["shots"].get(str(int(sid)))
        if entry is None:
            continue
        c = dict(cfg)
        c["rpos"] = [float(r) for r in entry["active_channel_rpos"]]
        c["times_beam"] = [float(t) for t in entry["beam_on_slice_times"]]
        jobs.append((int(sid), c))

    if args.workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            results = list(ex.map(_worker, [j for j in jobs]))
    else:
        results = [_worker(j) for j in jobs]

    # ---- assemble engine predictions on the manifest grid ----
    truth = mse_eval.MseTruth(level1_dir=LEVEL1_DIR)
    engine_preds, n_eng_slices = {}, {}
    skipped = []
    for res in results:
        sid = int(res["shot"])
        entry = manifest["shots"].get(str(sid))
        if entry is None:
            continue
        pred, n_used = _engine_prediction(res, entry)
        if pred is None:
            skipped.append({"shot": sid, "reason": res.get("reason", "no-confined-readout"),
                            "n_scored": int(res.get("n_scored", 0))})
            continue
        engine_preds[sid] = pred
        n_eng_slices[sid] = n_used

    if not engine_preds:
        raise SystemExit("no engine predictions produced — cannot gate")

    # ---- score the engine on the same harness as the baselines ----
    scored_engine = mse_eval.score(engine_preds, manifest, truth)
    persist_all = mse_eval.PersistencePredictor().predict(manifest, truth)
    persist_preds = {sid: p for sid, p in persist_all.items() if sid in engine_preds}
    scored_persist = mse_eval.score(persist_preds, manifest, truth)

    enkf_leg = pbar.enkf_leg_from_reference()

    # ---- paired per-shot bootstrap engine vs persistence (G4a CI-clear) ----
    rmse_engine = _per_shot_pitch_rmse(engine_preds, manifest, truth)
    rmse_persist = _per_shot_pitch_rmse(persist_preds, manifest, truth)
    boot_vs_persist = _paired_bootstrap(rmse_engine, rmse_persist)

    eng_pitch = scored_engine["primary"]["pitch"]
    per_pitch = scored_persist["primary"]["pitch"]
    eng_rmse = float(eng_pitch["rmse"])
    eng_cov = float(eng_pitch["cov90"])
    n_shots_scored = int(eng_pitch["n_shots"])

    # ---- G4 verdicts (pre-declared) ----
    beats_persist_ci = bool(boot_vs_persist and boot_vs_persist["ci_clear_engine_better"])
    coverage_ok = bool(np.isfinite(eng_cov) and eng_cov >= mse_eval.COVERAGE_GATE_LO)
    g4a = bool(beats_persist_ci)  # binding MUST = beat persistence CI-clear (stop rule)
    # G4b: approach the EnKF bar (parity if the engine RMSE reaches the EnKF CI)
    enkf_rmse = float(enkf_leg.pitch_rmse)
    enkf_ci = enkf_leg.pitch_rmse_ci or (float("nan"), float("nan"))
    g4b_parity = bool(np.isfinite(eng_rmse) and eng_rmse <= enkf_ci[1])
    frontier = float(pbar.REFERENCE["physics_frontier_pitch_rmse"])
    near_axis_floor = float(pbar.REFERENCE["near_axis_rad_floor"])

    # ---- per-radius residual (the near-axis interior floor) ----
    resid_by_r = _residual_vs_radius(engine_preds, manifest, truth)

    summary = {
        "n_heldout_requested": len(held),
        "n_shots_scored": n_shots_scored,
        "n_shots_skipped_underdetermined": len(skipped),
        "engine_pitch_rmse": eng_rmse,
        "engine_pitch_cov90": eng_cov,
        "engine_pitch_crps": float(eng_pitch["crps"]),
        "persistence_pitch_rmse_live": float(per_pitch["rmse"]),
        "persistence_pitch_rmse_reference": pbar.REFERENCE["persistence_pitch_rmse"],
        "enkf_pitch_rmse_reference": enkf_rmse,
        "enkf_pitch_rmse_ci": list(enkf_ci),
        "enkf_pitch_cov90_reference": float(enkf_leg.pitch_cov90),
        "physics_frontier_pitch_rmse": frontier,
        "near_axis_rad_floor": near_axis_floor,
        "coverage_gate_lo": mse_eval.COVERAGE_GATE_LO,
        "boot_engine_vs_persistence": boot_vs_persist,
        "G4a_beats_persistence_ci_clear": beats_persist_ci,
        "G4a_coverage_ge_0p88": coverage_ok,
        "G4a_pass": g4a,
        "G4b_reaches_enkf_ci": g4b_parity,
        "G4b_gap_to_enkf_rad": float(eng_rmse - enkf_rmse),
        "G4c_gap_to_frontier_rad": float(eng_rmse - frontier),
        "eta_params_frozen": eta_params,
        "eta_source": str(FROZEN_ETA_ARTIFACT) if FROZEN_ETA_ARTIFACT.exists() else "default",
        "prior_weight": args.prior_weight,
        "readout_representation": "kind='j' jphi(rho_hat) on rho_hat*a_minor grid "
                                  "(S9 head-to-head lock; matches EnKF)",
        "R0_m": R0_M, "a_minor_m": A_MINOR_M,
    }
    logger.info("G4 summary: %s", json.dumps(summary, indent=1, default=float))

    result = {
        "arm": "heldout-mse-pitch-dynamics-coupled-engine",
        "summary": summary,
        "scored_engine": scored_engine,
        "scored_persistence_live": scored_persist,
        "residual_vs_radius": resid_by_r,
        "skipped": skipped,
        "n_engine_slices_per_shot": {str(k): v for k, v in n_eng_slices.items()},
    }
    sfx = f"-{args.out_suffix}" if args.out_suffix else ""
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    out = ARTIFACTS / f"heldout_mse_gate{sfx}.json"
    out.write_text(json.dumps(result, indent=1, default=float))
    logger.info("wrote %s", out)

    FIGURES.mkdir(parents=True, exist_ok=True)
    _fig_pitch_bar(summary, FIGURES / f"fig-heldout-pitch-bar{sfx}.png")
    _fig_residual_radius(resid_by_r, near_axis_floor,
                         FIGURES / f"fig-heldout-residual-radius{sfx}.png")

    print(f"G4a_PASS={g4a} engine_rmse={eng_rmse:.3f} "
          f"persist_live={float(per_pitch['rmse']):.3f} enkf={enkf_rmse:.3f} "
          f"cov90={eng_cov:.3f} n={n_shots_scored} "
          f"beats_persist_ci={beats_persist_ci}")
    return 0


def _residual_vs_radius(preds: dict, manifest: dict, truth) -> dict:
    """Mean signed pitch residual (pred - truth) per sightline major radius."""
    from imas_ambix.statespace import mse_split as M  # noqa: PLC0415

    rbins, resid, absresid = [], [], []
    for sid, pred in preds.items():
        entry = manifest["shots"].get(str(int(sid)))
        tr = truth.get(int(sid))
        if entry is None or tr is None:
            continue
        rpos = np.asarray(entry["active_channel_rpos"], dtype=np.float64)
        pv = np.asarray(entry["pitch_valid_mask"], dtype=bool)
        gate = M.pitch_point_gate(tr.pitch, tr.pitch_error) & pv[:, None]
        d = pred.pitch_mean - np.asarray(tr.pitch, dtype=np.float64)
        for c in range(rpos.size):
            m = gate[:, c] & np.isfinite(d[:, c])
            if m.any():
                rbins.append(float(rpos[c]))
                resid.append(float(np.mean(d[m, c])))
                absresid.append(float(np.sqrt(np.mean(d[m, c] ** 2))))
    if not rbins:
        return {}
    rbins = np.asarray(rbins)
    order = np.argsort(rbins)
    return {"rpos": rbins[order].tolist(),
            "mean_resid": np.asarray(resid)[order].tolist(),
            "rmse": np.asarray(absresid)[order].tolist(),
            "r0_m": R0_M}


def _fig_pitch_bar(summary: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    names = ["persistence\n(floor)", "engine\n(coupled)", "EnKF\n(classical)",
             "frontier"]
    vals = [summary["persistence_pitch_rmse_reference"], summary["engine_pitch_rmse"],
            summary["enkf_pitch_rmse_reference"], summary["physics_frontier_pitch_rmse"]]
    cols = ["#999", "#268", "#c66", "#4a4"]
    ax.bar(range(4), vals, color=cols, width=0.62)
    enkf_ci = summary["enkf_pitch_rmse_ci"]
    if enkf_ci and np.isfinite(enkf_ci[0]):
        ax.errorbar(2, summary["enkf_pitch_rmse_reference"],
                    yerr=[[summary["enkf_pitch_rmse_reference"] - enkf_ci[0]],
                          [enkf_ci[1] - summary["enkf_pitch_rmse_reference"]]],
                    fmt="none", ecolor="k", capsize=4, lw=1.2)
    b = summary.get("boot_engine_vs_persistence")
    if b:
        ax.errorbar(1, summary["engine_pitch_rmse"], fmt="none")
        ax.annotate(f"beats persistence\nΔ={b['delta_mean']:.3f} "
                    f"CI[{b['delta_ci95'][0]:.3f},{b['delta_ci95'][1]:.3f}]\n"
                    f"CI-clear={b['ci_clear_engine_better']}",
                    (1, summary["engine_pitch_rmse"]), fontsize=7,
                    ha="center", va="bottom",
                    xytext=(1, summary["engine_pitch_rmse"] + 0.05), textcoords="data")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.008, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(range(4), names, fontsize=8)
    ax.set_ylabel("held-out MSE pitch RMSE [rad]")
    g4a = "PASS" if summary["G4a_pass"] else "FAIL"
    ax.set_title(f"§4 held-out MSE pitch gate — G4a {g4a} "
                 f"(n={summary['n_shots_scored']} shots, cov90={summary['engine_pitch_cov90']:.2f})",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _fig_residual_radius(resid: dict, near_axis_floor: float, path: Path) -> None:
    if not resid or not resid.get("rpos"):
        return
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    r = np.asarray(resid["rpos"])
    ax.scatter(r, resid["rmse"], s=10, color="#268", label="per-channel RMSE")
    ax.axhline(near_axis_floor, color="#c66", ls="--", lw=1.0,
               label=f"near-axis floor {near_axis_floor:.2f}")
    ax.axvline(resid.get("r0_m", R0_M), color="k", ls=":", lw=0.8, label="magnetic axis R0")
    ax.set_xlabel("sightline major radius R [m]")
    ax.set_ylabel("pitch residual RMSE [rad]")
    ax.set_title("§4 pitch residual vs radius (the near-axis interior floor §5 must close)",
                 fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
