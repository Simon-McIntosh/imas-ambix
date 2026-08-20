#!/usr/bin/env python
"""Vacuum-trained passive-resistance extraction with case-current holdback.

On a coil-only interval the measured magnetics minus the static coil
prediction is PURE eddy signal: no plasma, measured drives, geometry-exact L
— the only unknowns are a few bounded resistance multipliers.  This driver
fits them on the vetted coil-only pool (fleet −2.5 s preludes carrying the
full CS precharge swing + dedicated vacuum shots running whole coil programs)
and validates on held-out vacuum shots the fit never sees.

Case-current holdback (binding): the measured ``*_case_current`` channels are
NEVER inputs.  The circuit system is built with the measured-case circuits
moved INTO the passive set (``hold_back_cases``): their currents are
predicted from the coil drives through the mutual couplings, and the measured
case currents supervise the fit purely as held-back per-circuit targets —
reproducing a measured case transient from independent drives validates both
L and R at the per-circuit level.

Identifiability ladder (cross-shot, never per-shot): global scale →
vessel/case → vessel regions + case pairs → per-case, each tier accepted only
if the held-out combined loss improves.  Supervision: whitened magnetics eddy
residual (per-channel offset nuisance removed) + whitened held-back case
currents, equal weight.

The static coil calibration deliberately fit dI/dt-QUIET intervals (gains);
the transient decays that encode τ = L/R are untouched information.  Only the
LEADING coil-only run of each shot is used — post-plasma coil-only tails
carry vessel currents of plasma origin the coil-driven model must not chase.

Artifacts:
  imas_ambix/latent/artifacts/patch_gate/vacuum_passive_resistance_fit.json
  imas_ambix/latent/artifacts/patch_gate/passive_resistance_calibration.json
Figures:  docs/figures/temporal-physics-spine/fig-vacuum-resistance-*.png
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from imas_ambix.data.description_reader import read_geometry_table
from imas_ambix.gs.machine_geometry import MachineGeometryService

from imas_ambix.gs.operator import COIL_MODEL_VERSION, build_operator
from imas_ambix.latent.data import (
    align_sensor_columns,
    anchored_columns,
    feature_schema,
    load_shot_slices_raw,
    robust_channel_scale,
    schema_group_offsets,
)
from imas_ambix.latent.passive_resistance import (
    LADDER_LEVELS,
    MULTIPLIER_BOUNDS,
    ResistanceCalibration,
    VacuumShotData,
    campaign_mode_maps,
    resistance_group_labels,
    save_calibration,
    shot_loss_terms,
    zoh_mode_response,
)
from imas_ambix.latent.temporal_operator import (
    build_passive_circuit_system,
    load_circuit_system,
    save_circuit_system,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("vacuum_passive_resistance_fit")

ARTIFACTS = Path("imas_ambix/latent/artifacts/patch_gate")
SYSTEM_DIR = Path("imas_ambix/latent/artifacts/temporal_operator")
FIGURES = Path("docs/figures/temporal-physics-spine")

#: coil-only guard, as the vacuum coil-response audit
IP_VACUUM_KA = 20.0

#: minimum usable leading coil-only run [samples at 1 kHz]
MIN_RUN_SAMPLES = 200


def _campaign_system(table, grid):
    """The case-holdback circuit system, cached per campaign (atomic publish)."""
    key = table.signature.key
    cache = SYSTEM_DIR / f"circuit-system-holdback-{key}.npz"
    if cache.exists():
        return load_circuit_system(cache)
    system = build_passive_circuit_system(table, grid, hold_back_cases=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(f".pid{os.getpid()}.npz")
    save_circuit_system(tmp, system)
    tmp.replace(cache)
    logger.info("holdback circuit system %s built -> %s", key, cache)
    return system


def prep_shot(job: tuple) -> dict | None:
    """Prepare one shot's coil-only fit arrays (worker process)."""
    shot, stratum, nr, nz = job
    from imas_ambix.latent.gs_solve import EquilibriumGrid  # noqa: PLC0415

    schema = feature_schema()
    try:
        table = read_geometry_table(int(shot))
        fwd = build_operator(table)
        grid = EquilibriumGrid.from_table(table, nr=nr, nz=nz)
        system = _campaign_system(table, grid)
    except Exception as exc:  # noqa: BLE001
        logger.warning("shot %s: build failed (%s)", shot, exc)
        return None
    loaded = load_shot_slices_raw(int(shot), schema)
    if loaded is None:
        return None
    x, times, plasma_on = loaded
    x = np.asarray(x, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    all_vacuum = not bool(np.any(plasma_on))

    offsets = schema_group_offsets(schema)
    amb_names = schema["amb"]
    amc_names = schema["amc"]

    # σ from the plasma-on interval (fleet preludes) or whole shot (vacuum) —
    # the audit's convention, so whitening is frozen-comparable
    op_rows, x_cols = align_sensor_columns(fwd.sensor_channels, amb_names)
    n_sensor = len(fwd.sensor_channels)
    raw_mag_full = np.full((x.shape[0], n_sensor), np.nan)
    if op_rows.size:
        raw_mag_full[:, op_rows] = x[:, offsets["amb"] + x_cols]
    sigma_src = raw_mag_full[plasma_on] if not all_vacuum else raw_mag_full
    with np.errstate(all="ignore"):
        sigma = robust_channel_scale(np.nanstd(sigma_src, axis=0), fwd.sensor_channels)

    # leading coil-only run (post-plasma tails carry plasma-origin eddies);
    # a NaN Rogowski sample with the plasma-on flag False is still vacuum —
    # the flag (50 kA / 0.2·peak threshold) already guards breakdown edges
    ip_col, _ = anchored_columns(schema)
    ip_ka = np.abs(x[:, ip_col])
    coil_only = (~plasma_on) & (~np.isfinite(ip_ka) | (ip_ka < IP_VACUUM_KA))
    if not coil_only[0]:
        logger.warning("shot %s: stream does not start coil-only — skipped", shot)
        return None
    t_end = int(np.argmin(coil_only)) if not coil_only.all() else coil_only.size
    if t_end < MIN_RUN_SAMPLES:
        logger.warning("shot %s: leading run %d samples — skipped", shot, t_end)
        return None
    x = x[:t_end]
    times = times[:t_end]
    raw_mag = raw_mag_full[:t_end]

    dts = np.diff(times)
    if dts.size == 0 or np.ptp(dts) > 1e-9:
        logger.warning("shot %s: non-uniform raw grid — skipped", shot)
        return None
    dt = float(dts[0])

    # amc block, interior-NaN interpolated so a dropout never fakes a step
    amc = np.array(x[:, offsets["amc"] : offsets["amc"] + len(amc_names)])
    for j in range(amc.shape[1]):
        ok = np.isfinite(amc[:, j])
        if ok.any() and not ok.all():
            amc[:, j] = np.interp(times, times[ok], amc[ok, j])

    # drives: the system's own (non-case) channel list, assembled per sample
    i_pf_all = np.zeros((t_end, len(fwd.pf_amc_channels)))
    for t in range(t_end):
        vals = {
            ch: float(amc[t, j])
            for j, ch in enumerate(amc_names)
            if np.isfinite(amc[t, j])
        }
        i_pf_all[t] = fwd.assemble_pf_currents(vals)
    col_of = {ch: c for c, ch in enumerate(fwd.pf_amc_channels)}
    drive_cols = [col_of.get(ch, -1) for ch in system.coil_channels]
    i_drive = np.zeros((t_end, len(system.coil_channels)))
    for k, c in enumerate(drive_cols):
        if c >= 0:
            i_drive[:, k] = i_pf_all[:, c]
    psi_circ = i_drive @ system.m_coil_circ.T

    # static prediction from the SAME non-case drives (g_pf case columns are
    # never applied here — the cases are dynamic state, not drives)
    g_cols = np.zeros((n_sensor, len(system.coil_channels)))
    for k, c in enumerate(drive_cols):
        if c >= 0:
            g_cols[:, k] = fwd.g_pf[:, c]
    meas_resid = raw_mag - i_drive @ g_cols.T

    # held-back measured case currents on the system's sorted case rows
    case_channels = sorted(system.case_channel_row)
    amc_idx = {ch: j for j, ch in enumerate(amc_names)}
    case_meas = np.full((t_end, len(case_channels)), np.nan)
    for k, ch in enumerate(case_channels):
        j = amc_idx.get(ch, -1)
        if j >= 0 and np.isfinite(amc[:, j]).any():
            case_meas[:, k] = amc[:, j] * 1000.0  # kA·turn → A (turns = 1)

    return {
        "data": VacuumShotData(
            shot=int(shot),
            campaign=table.signature.key,
            stratum=stratum,
            dt=dt,
            psi_circ=psi_circ,
            meas_resid=meas_resid,
            sigma=sigma,
            case_meas=case_meas,
            i_drive=i_drive,  # already amperes (assemble_pf_currents converts)
        ),
        "n_samples": t_end,
        "sibling_audit": _sibling_identity_audit(amc, amc_names),
    }


def _sibling_identity_audit(amc: np.ndarray, amc_names: list[str]) -> dict:
    """Exact-identity residuals of the unconsumed amc siblings, one shot.

    Tier A of the structure discovery: are the ``*_feed_current`` and plain
    ``*_current`` siblings NEW measured drives, or linear combinations of
    channels already consumed?  Tested per coil on the coil-only run:
    ``coil = N·feed`` (turns-scaled duplicate) and ``plain = Σ coils + case``
    (the case current metered inside the coil's supply circuit — a plain
    channel therefore LEAKS the held-back case measurement and is
    inadmissible as a discovery input).  Relative RMS residuals ≈ 0 confirm
    redundancy; the identities themselves are the wiring evidence tier B
    consumes.
    """
    cols = {ch: amc[:, j] for j, ch in enumerate(amc_names)}

    def _rel(target: np.ndarray, pred: np.ndarray) -> float | None:
        ok = np.isfinite(target) & np.isfinite(pred)
        if ok.sum() < 200:
            return None
        t, p = target[ok], pred[ok]
        denom = float(np.std(t))
        if denom < 1e-9:
            return None
        return float(np.sqrt(np.mean(((t - t.mean()) - (p - p.mean())) ** 2)) / denom)

    out: dict[str, dict] = {"feed": {}, "plain": {}}
    for ch in amc_names:
        if not ch.endswith("_feed_current"):
            continue
        base = ch[: -len("_feed_current")]
        coil = cols.get(f"{base}_coil_current")
        feed = cols[ch]
        if coil is None:
            continue
        ok = np.isfinite(coil) & np.isfinite(feed) & (np.abs(feed) > 1e-6)
        if ok.sum() < 200:
            continue
        turns = float(np.round(np.median(coil[ok] / feed[ok])))
        rel = _rel(coil, turns * feed)
        if rel is not None:
            out["feed"][base] = {"turns": turns, "rel_resid": rel}
    for ch in amc_names:
        if not ch.endswith("_case_current"):
            continue
        base = ch[: -len("_case_current")]
        plain = cols.get(f"{base}_current")
        if plain is None:
            continue
        family, pos = base[:-1], base[-1]
        coil_sum = np.zeros(amc.shape[0])
        n_parents = 0
        for cc in amc_names:
            if not cc.endswith("_coil_current"):
                continue
            b = cc.split("_")[0]
            if b.startswith(family) and b.endswith(pos):
                coil_sum = coil_sum + np.nan_to_num(cols[cc])
                n_parents += 1
        if n_parents == 0:
            continue
        rel = _rel(plain, coil_sum + np.nan_to_num(cols[ch]))
        if rel is not None:
            out["plain"][base] = {"n_parent_coils": n_parents, "rel_resid": rel}
    return out


# --- parallel objective ------------------------------------------------

_POOL_SHOTS: list[VacuumShotData] = []
_POOL_SYSTEMS: dict = {}
_POOL_SIGMA: dict = {}
_POOL_SIGMA_CASE: dict = {}


def _pool_init(shots, systems, sigma_med, sigma_case):
    global _POOL_SHOTS, _POOL_SYSTEMS, _POOL_SIGMA, _POOL_SIGMA_CASE
    _POOL_SHOTS = shots
    _POOL_SYSTEMS = systems
    _POOL_SIGMA = sigma_med
    _POOL_SIGMA_CASE = sigma_case


def _shot_terms_by_index(job: tuple) -> tuple:
    idx, theta_by_campaign = job
    d = _POOL_SHOTS[idx]
    maps = campaign_mode_maps(_POOL_SYSTEMS[d.campaign], theta_by_campaign[d.campaign])
    return shot_loss_terms(
        d, maps, _POOL_SIGMA[d.campaign], _POOL_SIGMA_CASE[d.campaign]
    )


class ParallelObjective:
    """Combined mean whitened square over a shot pool, Pool-parallel."""

    def __init__(self, shots, systems, sigma_med, sigma_case, group_index, workers):
        self.shots = shots
        self.systems = systems
        self.sigma_med = sigma_med
        self.sigma_case = sigma_case
        self.group_index = group_index
        ctx = multiprocessing.get_context("fork")
        self.pool = (
            ctx.Pool(
                workers,
                initializer=_pool_init,
                initargs=(shots, systems, sigma_med, sigma_case),
            )
            if workers > 1
            else None
        )
        self.n_eval = 0

    def components(self, theta: np.ndarray) -> dict[str, float]:
        theta_by_campaign = {
            key: np.asarray(theta)[self.group_index[key]] for key in self.systems
        }
        jobs = [(i, theta_by_campaign) for i in range(len(self.shots))]
        if self.pool is not None:
            terms = self.pool.map(_shot_terms_by_index, jobs)
        else:
            _pool_init(self.shots, self.systems, self.sigma_med, self.sigma_case)
            terms = [_shot_terms_by_index(j) for j in jobs]
        ss_mag = sum(t[0] for t in terms)
        n_mag = sum(t[1] for t in terms)
        ss_case = sum(t[2] for t in terms)
        n_case = sum(t[3] for t in terms)
        mag = ss_mag / max(n_mag, 1)
        case = ss_case / max(n_case, 1)
        self.n_eval += 1
        return {"combined": mag + case, "mag": mag, "case": case}

    def __call__(self, log_theta: np.ndarray) -> float:
        return self.components(np.exp(log_theta))["combined"]

    def close(self):
        if self.pool is not None:
            self.pool.close()
            self.pool.join()


def _case_reproduction(shots, systems, theta_by_campaign) -> dict:
    """Held-back case-current reproduction per channel: relative RMS + corr."""
    per_chan: dict[str, dict[str, list[float]]] = {}
    for d in shots:
        system = systems[d.campaign]
        maps = campaign_mode_maps(system, theta_by_campaign[d.campaign])
        a = zoh_mode_response(maps.tau, d.dt, d.psi_circ @ maps.v)
        pred = a @ maps.case_v.T
        for k, ch in enumerate(sorted(system.case_channel_row)):
            meas = d.case_meas[:, k]
            good = np.isfinite(meas)
            if good.sum() < 100:
                continue
            m = meas[good] - np.mean(meas[good])
            p = pred[good, k] - np.mean(pred[good, k])
            denom = float(np.sqrt(np.mean(m**2)))
            if denom < 1.0:  # dead/flat channel this shot
                continue
            entry = per_chan.setdefault(ch, {"rel_rms": [], "corr": []})
            entry["rel_rms"].append(float(np.sqrt(np.mean((m - p) ** 2)) / denom))
            cc = np.corrcoef(m, p)[0, 1] if m.size > 2 else np.nan
            entry["corr"].append(float(cc))
    return {
        ch: {
            "rel_rms_median": float(np.median(v["rel_rms"])),
            "corr_median": float(np.median(v["corr"])),
            "n_shots": len(v["rel_rms"]),
        }
        for ch, v in per_chan.items()
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--pool-artifact",
        type=str,
        default=str(ARTIFACTS / "vacuum_coil_response_audit-solv4-vac.json"),
        help="audit artifact supplying the vetted coil-only shot pool",
    )
    ap.add_argument("--holdout-stride", type=int, default=5)
    ap.add_argument("--nr", type=int, default=65)
    ap.add_argument("--nz", type=int, default=97)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--max-shots", type=int, default=0, help="debug cap (0 = all)")
    ap.add_argument("--out-suffix", type=str, default="")
    args = ap.parse_args()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    pool_meta = json.loads(Path(args.pool_artifact).read_text())
    shots = [int(s) for s in pool_meta["shots_used"]]
    # stratum bookkeeping: the audit pooled fleet preludes first, then the
    # dedicated vacuum stratum — recover per-shot strata from its counts
    n_fleet = int(pool_meta["strata"]["fleet"]["n_shots"])
    strata = ["fleet"] * n_fleet + ["dedicated_vacuum"] * (len(shots) - n_fleet)
    if args.max_shots > 0:
        shots, strata = shots[: args.max_shots], strata[: args.max_shots]

    # ---- pre-build the per-campaign circuit systems (sequential, cached) —
    # otherwise every pool worker races on the minutes-long kernel build ----
    from imas_ambix.latent.gs_solve import EquilibriumGrid  # noqa: PLC0415

    t0 = time.perf_counter()
    seen_campaigns: set[str] = set()
    for s in shots:
        try:
            table = read_geometry_table(int(s))
        except Exception:  # noqa: BLE001 — the prep worker logs the skip
            continue
        if table.signature.key in seen_campaigns:
            continue
        seen_campaigns.add(table.signature.key)
        grid = EquilibriumGrid.from_table(table, nr=args.nr, nz=args.nz)
        _campaign_system(table, grid)
    logger.info(
        "campaign systems ready: %s (%.0f s)",
        sorted(seen_campaigns),
        time.perf_counter() - t0,
    )

    # ---- prepare shots (parallel) ----
    t0 = time.perf_counter()
    jobs = [(s, st, args.nr, args.nz) for s, st in zip(shots, strata, strict=True)]
    ctx = multiprocessing.get_context("fork")
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
            prepped = [r for r in pool.map(prep_shot, jobs) if r is not None]
    else:
        prepped = [r for r in map(prep_shot, jobs) if r is not None]
    data = [r["data"] for r in prepped]
    if len(data) < 10:
        logger.error("too few prepared shots (%d)", len(data))
        return 1
    logger.info(
        "prepared %d/%d shots (%.0f s), %d total samples",
        len(data),
        len(shots),
        time.perf_counter() - t0,
        sum(d.n_samples for d in data),
    )

    campaigns = sorted({d.campaign for d in data})
    systems = {}
    for key in campaigns:
        systems[key] = load_circuit_system(
            SYSTEM_DIR / f"circuit-system-holdback-{key}.npz"
        )
        logger.info(
            "campaign %s: %d passive circuits (%d held-back cases), %d drives",
            key,
            systems[key].n_circuits,
            len(systems[key].case_channel_row),
            len(systems[key].coil_channels),
        )

    # ---- held-out split: every Nth shot per stratum ----
    train, held = [], []
    for stratum in ("fleet", "dedicated_vacuum"):
        grp = [d for d in data if d.stratum == stratum]
        for i, d in enumerate(grp):
            is_held = i % args.holdout_stride == args.holdout_stride - 1
            (held if is_held else train).append(d)
    logger.info("split: %d train / %d held-out shots", len(train), len(held))

    # ---- pooled whitening scales ----
    sigma_med, sigma_case = {}, {}
    for key in campaigns:
        sig = np.stack([d.sigma for d in data if d.campaign == key])
        with np.errstate(all="ignore"):
            sigma_med[key] = np.nanmedian(sig, axis=0)
        case_stack = np.concatenate(
            [d.case_meas for d in train if d.campaign == key], axis=0
        )
        with np.errstate(all="ignore"):
            sc = np.nanstd(case_stack, axis=0)
        sigma_case[key] = np.where(np.isfinite(sc) & (sc > 1.0), sc, 1.0)

    # ---- the identifiability ladder ----
    from scipy.optimize import minimize  # noqa: PLC0415

    lb, ub = np.log(MULTIPLIER_BOUNDS[0]), np.log(MULTIPLIER_BOUNDS[1])
    ladder_records = []
    # the nominal model is the incumbent every rung must beat on held-out —
    # if none does, the honest result is an identity calibration
    nominal_index = {
        key: np.zeros(systems[key].n_circuits, dtype=np.int64) for key in campaigns
    }
    held_nom = ParallelObjective(held, systems, sigma_med, sigma_case, nominal_index, 1)
    nom_comp = held_nom.components(np.ones(1))
    held_nom.close()
    logger.info(
        "nominal R held-out: combined %.5f (mag %.5f case %.5f)",
        nom_comp["combined"],
        nom_comp["mag"],
        nom_comp["case"],
    )
    incumbent = ("nominal", ["all"], np.ones(1), nom_comp)
    labels_by_level: dict[str, dict[str, list[str]]] = {
        "nominal": {key: ["all"] * systems[key].n_circuits for key in campaigns}
    }
    for level in LADDER_LEVELS:
        labels = {
            key: resistance_group_labels(
                systems[key].circuits,
                systems[key].centroid_r,
                systems[key].centroid_z,
                level,
            )
            for key in campaigns
        }
        labels_by_level[level] = labels
        group_names = sorted({lb_ for labs in labels.values() for lb_ in labs})
        group_index = {
            key: np.array([group_names.index(lb_) for lb_ in labels[key]])
            for key in campaigns
        }
        obj = ParallelObjective(
            train, systems, sigma_med, sigma_case, group_index, args.workers
        )
        held_obj = ParallelObjective(
            held, systems, sigma_med, sigma_case, group_index, 1
        )
        # warm start from the incumbent (each group inherits its parent value)
        x0 = np.zeros(len(group_names))
        if incumbent is not None:
            prev_level, prev_names, prev_theta, _ = incumbent
            for gi, gname in enumerate(group_names):
                # map through any campaign circuit carrying this group
                for key in campaigns:
                    rows = [r for r, lb_ in enumerate(labels[key]) if lb_ == gname]
                    if rows:
                        parent = labels_by_level[prev_level][key][rows[0]]
                        x0[gi] = np.log(prev_theta[prev_names.index(parent)])
                        break
        t1 = time.perf_counter()
        res = minimize(
            obj,
            x0,
            method="L-BFGS-B",
            bounds=[(lb, ub)] * len(group_names),
            options={"maxiter": 60, "ftol": 1e-8},
        )
        theta = np.exp(res.x)
        train_comp = obj.components(theta)
        held_comp = held_obj.components(theta)
        obj.close()
        held_obj.close()
        rec = {
            "level": level,
            "n_dof": len(group_names),
            "group_names": group_names,
            "theta": {g: float(t) for g, t in zip(group_names, theta, strict=True)},
            "train": train_comp,
            "held_out": held_comp,
            "n_obj_evals": obj.n_eval,
            "wall_s": time.perf_counter() - t1,
        }
        ladder_records.append(rec)
        logger.info(
            "ladder %s (%d DOF): train %.5f held %.5f (mag %.5f case %.5f) "
            "theta %s [%.0f s]",
            level,
            len(group_names),
            train_comp["combined"],
            held_comp["combined"],
            held_comp["mag"],
            held_comp["case"],
            {g: round(float(t), 3) for g, t in zip(group_names, theta, strict=True)},
            rec["wall_s"],
        )
        if incumbent is None or held_comp["combined"] < incumbent[3]["combined"]:
            incumbent = (level, group_names, theta, held_comp)
        else:
            logger.info("ladder %s: held-out did not improve — stopping", level)
            break

    level, group_names, theta, held_comp = incumbent
    if level == "nominal":
        # no rung beat nominal R on held-out: the honest artifact is an
        # identity calibration in the global-level vocabulary
        level, group_names, theta = "global", ["all"], np.ones(1)
    logger.info(
        "chosen level: %s  theta=%s", level, dict(zip(group_names, theta, strict=True))
    )

    # ---- before/after reporting ----
    theta_nom = {key: np.ones(systems[key].n_circuits) for key in campaigns}
    labels = labels_by_level[level]
    theta_cal = {
        key: np.array([theta[group_names.index(lb_)] for lb_ in labels[key]])
        for key in campaigns
    }
    group_index = {
        key: np.array([group_names.index(lb_) for lb_ in labels[key]])
        for key in campaigns
    }

    def _pool_components(shots_, theta_by_campaign):
        _pool_init(shots_, systems, sigma_med, sigma_case)
        terms = [
            shot_loss_terms(
                d,
                campaign_mode_maps(systems[d.campaign], theta_by_campaign[d.campaign]),
                sigma_med[d.campaign],
                sigma_case[d.campaign],
            )
            for d in shots_
        ]
        n_mag = sum(t[1] for t in terms)
        n_case = sum(t[3] for t in terms)
        return {
            "mag": sum(t[0] for t in terms) / max(n_mag, 1),
            "case": sum(t[2] for t in terms) / max(n_case, 1),
        }

    before_held = _pool_components(held, theta_nom)
    after_held = _pool_components(held, theta_cal)
    case_before = _case_reproduction(held, systems, theta_nom)
    case_after = _case_reproduction(held, systems, theta_cal)

    calibration = ResistanceCalibration(
        level=level,
        group_multipliers={
            g: float(t) for g, t in zip(group_names, theta, strict=True)
        },
        provenance={
            "fitted": "2026-07-17",
            "coil_model_version": COIL_MODEL_VERSION,
            "geometry_table_version": MachineGeometryService().identity(11766).derivation_id,
            "pool_artifact": str(args.pool_artifact),
            "n_train_shots": len(train),
            "n_held_out_shots": len(held),
            "held_out_shots": sorted(d.shot for d in held),
            "supervision": "whitened eddy magnetics + held-back case currents",
            "case_holdback": True,
        },
    )
    tag = f"-{args.out_suffix}" if args.out_suffix else ""
    cal_path = ARTIFACTS / f"passive_resistance_calibration{tag}.json"
    save_calibration(cal_path, calibration)
    logger.info("wrote %s", cal_path)

    out = {
        "kind": "vacuum-passive-resistance-fit",
        "coil_model_version": COIL_MODEL_VERSION,
        "geometry_table_version": MachineGeometryService().identity(11766).derivation_id,
        "leakage_free": True,
        "case_holdback": True,
        "ip_vacuum_ka": IP_VACUUM_KA,
        "pool": {
            "artifact": str(args.pool_artifact),
            "n_shots_prepared": len(data),
            "n_train": len(train),
            "n_held_out": len(held),
            "held_out_shots": sorted(d.shot for d in held),
            "n_samples_total": int(sum(d.n_samples for d in data)),
            "campaigns": campaigns,
        },
        "ladder": ladder_records,
        "chosen_level": level,
        "group_multipliers": calibration.group_multipliers,
        "held_out_loss": {"before": before_held, "after": after_held},
        "case_reproduction_held_out": {
            "before": case_before,
            "after": case_after,
        },
        "calibration_artifact": str(cal_path),
    }
    out_path = ARTIFACTS / f"vacuum_passive_resistance_fit{tag}.json"
    out_path.write_text(json.dumps(out, indent=2))
    logger.info("wrote %s", out_path)

    _figures(out, held, systems, theta_nom, theta_cal, sigma_med, tag)
    return 0


def _figures(out, held, systems, theta_nom, theta_cal, sigma_med, tag=""):
    # --- fig 1: ladder + chosen multipliers ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    recs = out["ladder"]
    x = np.arange(len(recs))
    axes[0].plot(x, [r["train"]["combined"] for r in recs], "o-", label="train")
    axes[0].plot(x, [r["held_out"]["combined"] for r in recs], "s-", label="held-out")
    axes[0].set_xticks(
        x, [f"{r['level']}\n({r['n_dof']} DOF)" for r in recs], fontsize=8
    )
    axes[0].set_ylabel("combined mean whitened square")
    axes[0].set_title(f"identifiability ladder — chosen: {out['chosen_level']}")
    axes[0].legend(fontsize=8)
    gm = out["group_multipliers"]
    names = sorted(gm)
    axes[1].barh(np.arange(len(names)), [gm[n] for n in names], color="#228833")
    axes[1].set_yticks(np.arange(len(names)), names, fontsize=8)
    axes[1].axvline(1.0, color="k", lw=1.0)
    axes[1].set_xlabel("resistance multiplier (nominal = 1)")
    axes[1].set_title("calibrated multipliers")
    fig.tight_layout()
    fig.savefig(FIGURES / f"fig-vacuum-resistance-ladder{tag}.png", dpi=130)
    plt.close(fig)

    # --- fig 2: held-back case-current reproduction, representative shots ---
    vac = [d for d in held if d.stratum == "dedicated_vacuum"] or held
    show = vac[:2]
    n_ch = 4
    fig, axes = plt.subplots(
        len(show), n_ch, figsize=(15, 3.2 * len(show)), squeeze=False
    )
    for r, d in enumerate(show):
        system = systems[d.campaign]
        chans = sorted(system.case_channel_row)
        for arm, color, lab in (
            (theta_nom, "#bb5566", "nominal R"),
            (theta_cal, "#228833", "calibrated R"),
        ):
            maps = campaign_mode_maps(system, arm[d.campaign])
            a = zoh_mode_response(maps.tau, d.dt, d.psi_circ @ maps.v)
            pred = a @ maps.case_v.T
            t = np.arange(d.n_samples) * d.dt
            for c in range(min(n_ch, len(chans))):
                ax = axes[r][c]
                if lab == "nominal R":
                    ax.plot(
                        t,
                        d.case_meas[:, c] / 1e3,
                        color="#222",
                        lw=1.0,
                        label="measured (held back)",
                    )
                ax.plot(t, pred[:, c] / 1e3, color=color, lw=0.9, label=lab)
                ax.set_title(f"{d.shot} {chans[c]}", fontsize=8)
                if r == len(show) - 1:
                    ax.set_xlabel("t from stream start [s]")
                if c == 0:
                    ax.set_ylabel("case current [kA]")
    axes[0][0].legend(fontsize=7)
    fig.suptitle(
        "Held-back case currents: measured vs predicted from coil drives only",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIGURES / f"fig-vacuum-case-reproduction{tag}.png", dpi=130)
    plt.close(fig)

    # --- fig 3: whitened eddy residual per channel, before/after ---
    def _chan_rms(theta_by_campaign):
        acc, cnt = None, None
        for d in held:
            maps = campaign_mode_maps(
                systems[d.campaign], theta_by_campaign[d.campaign]
            )
            a = zoh_mode_response(maps.tau, d.dt, d.psi_circ @ maps.v)
            resid = d.meas_resid - a @ maps.a_sens_modes.T
            with np.errstate(invalid="ignore"):
                resid = resid - np.nanmean(resid, axis=0, keepdims=True)
            w = resid / sigma_med[d.campaign]
            fin = np.isfinite(w)
            ss = np.where(fin, w, 0.0) ** 2
            if acc is None:
                acc, cnt = ss.sum(axis=0), fin.sum(axis=0)
            else:
                acc += ss.sum(axis=0)
                cnt += fin.sum(axis=0)
        return np.sqrt(acc / np.maximum(cnt, 1))

    rms_b = _chan_rms(theta_nom)
    rms_a = _chan_rms(theta_cal)
    fig, ax = plt.subplots(figsize=(14, 4))
    xs = np.arange(rms_b.size)
    ax.plot(xs, rms_b, "o", ms=3.5, color="#bb5566", label="nominal R")
    ax.plot(xs, rms_a, "o", ms=3.5, color="#228833", label="calibrated R")
    ax.set_xlabel("sensor channel index")
    ax.set_ylabel("whitened eddy-residual RMS")
    ax.set_title("Held-out coil-only eddy residual per channel")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / f"fig-vacuum-eddy-residual{tag}.png", dpi=130)
    plt.close(fig)
    logger.info("figures written to %s", FIGURES)


if __name__ == "__main__":
    raise SystemExit(main())
