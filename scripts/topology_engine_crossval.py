"""Cross-validate the ENGINE against EFIT, per topology class.

The census (:mod:`scripts.topology_census`) stratifies the corpus by EFIT-read
topology class; here the engine runs on the selected slices — measured
magnetics only, firewall intact, no EFIT inputs — and its connectivity
boundary read is scored against EFIT's reported reconstruction per class.
Two chains share the harness:

* ``--chain bare`` — the Pass-1 disc-read initialiser alone (the seed-quality
  baseline; reproduces the documented unpinned outboard failure).
* ``--chain full`` — the validated four-pass pipeline
  (:func:`scripts.heldout_mse_gate_eval.coupled_solve_chain`: disc read →
  centroid-pinned basin solve → coupled profile solve with the ψ-diffusion
  coefficient prior, frozen η).  Arms:
    ``plain``   the validated chain as gated;
    ``nobasin`` Pass 1 dropped — the profile solve cold-starts from the disc
                seed under the SAME pin + prior (prices the basin pass;
                report-only, never gates);
    ``eddy``    the per-shot vessel-eddy trajectory precomputed from the
                measured drives and injected as a KNOWN drive through the
                frozen passive Green's columns (pinned amplitudes, sensor +
                Picard field consistent).

Verdict keys (module glossary — code below is named by mechanism):
  T-F3   bare-chain characterisation (landed; kept for reproduction).
  G-E1   full-engine completion: scored fraction of ATTEMPTED valid slices
         (engine-attributable failures count against; harness non-attempts —
         thinning, missing geometry/sensors, degenerate referee polygons —
         are accounted separately) reported against the >99% bar with every
         failure individually named.
  G-E2   full-engine per-class boundary quality.  Tolerances PRE-DECLARED
         before scoring, from the engine's validated envelope (flat-top LCFS
         1.5–2.1 cm on curated cohorts, ×~1.5 headroom for the uncurated
         stratified draw): per-class FLAT-TOP median LCFS-radii residual
         ≤ 3.0 cm and diverted/limited class agreement ≥ 0.80.  Class-
         conditional failures are surfaced, never averaged away.
  G-E3   ramp eddy ablation: the known-drive injection must not regress any
         class (flat-top median within +0.25 cm) and its early-time
         (t < 0.2 s) effect is quantified per class.

Scoring frame: the engine boundary is read on the ENGINE grid (its own solve
domain); EFIT's LCFS polygon is rendered to radii about the ENGINE's read axis
so both sides describe the boundary from the same origin.

Usage:
    uv run python -m scripts.topology_engine_crossval --chain full --classes limited
    uv run python -m scripts.topology_engine_crossval --aggregate
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("topology_engine_crossval")

from scripts.topology_census import LEVEL2_SHOTS  # noqa: E402
from scripts.topology_efit_read_eval import (  # noqa: E402
    SUPPORT_FLOOR,
    _agg,
    _xset_match_cm,
    polygon_ray_radii,
    stratified_selection,
)

PASS_CLASSES = ("limited", "sn-lower", "connected-dn")
FINDING_CLASSES = ("sn-upper", "marginal-dn", "snowflake-candidate")

ENGINE_LCFS_TOL_CM = 5.0
ENGINE_CLASS_ACC_TOL = 0.80
TIME_MATCH_S = 0.02  # census slice ↔ engine payload time pairing window
MAX_PAYLOAD_SLICES = 200  # per-shot payload build ceiling (dense time cover)
MIN_IP_KA = 100.0

# ---- full-chain (four-pass engine) configuration + PRE-DECLARED tolerances --
# Chain wiring identical to the held-out MSE gate (sigma, prior weight, frozen
# η, sub-stepping); only the slice budget is denser for ramp-phase coverage.
CHAIN_SIGMA_M = 0.02  # centroid tether 1σ [m] (the gate's DEFAULT_SIGMA_M)
CHAIN_PRIOR_WEIGHT = 0.3  # ψ-diffusion coefficient-prior weight (gate value)
CHAIN_N_SUB = 24
CHAIN_PAR_WEIGHT = 1.0
CHAIN_N_RHO = 24
CHAIN_MAX_SLICES = 24  # dense enough to cover ramp + flat-top
CHAIN_MIN_IP_KA = 60.0  # the engine's operating floor (gate value)

RAMP_END_S = 0.2  # early-phase bin boundary [s]

# Declared BEFORE any full-chain scoring ran (validated flat-top envelope
# 1.5–2.1 cm on curated cohorts; ×~1.5 headroom for the uncurated census draw).
FLATTOP_LCFS_TOL_CM = 3.0  # per-class flat-top median LCFS-radii residual
FULL_CLASS_ACC_TOL = 0.80  # diverted/limited agreement per class
COMPLETION_BAR = 0.99  # scored / attempted valid slices (the lead's bar)
EDDY_REGRESSION_BAND_CM = 0.25  # flat-top per-class median may not worsen more

# eddy known-drive limit: amplitudes pinned to the precomputed trajectory
EDDY_KNOWN_DRIVE_WEIGHT = 100.0
EDDY_N_MODES = 12  # eigenbasis rank (the dynamic-passive rung's value)

# drop reasons the ENGINE owns (count against completion); everything else is
# harness / referee accounting (reported, not attempted)
ENGINE_FAILURE_REASONS = frozenset({
    "solve-error", "solve-no-ring", "chain-failed", "fit-not-scored",
    "unconfined-axis", "seed-in-wall", "read-not-found",
})


def engine_rows_for_shot(
    shot: int, recs, *, nr: int, nz: int
) -> tuple[list[dict], list[dict]]:
    """Run the disc engine on one shot and score its read at the census times.

    Returns ``(rows, drops)`` — every requested slice lands in exactly one of
    the two lists; ``drops`` carries a reason code so no failure mode is
    silently averaged away (world-model label production needs the accounting,
    not just the survivors).
    """
    import zarr  # noqa: PLC0415

    from imas_ambix.latent.boundary_disc import disc_read  # noqa: PLC0415
    from imas_ambix.latent.connectivity_boundary import boundary_read  # noqa: PLC0415
    from imas_ambix.worldmodel.equilibrium_labels import LCFS_ANGLES  # noqa: PLC0415
    from scripts.heldout_mse_gate_eval import _campaign_table  # noqa: PLC0415
    from scripts.spine_label_factory import factory_shot_payloads  # noqa: PLC0415

    def _drops(reason: str, detail: str = "") -> list[dict]:
        return [
            {
                "shot": int(shot),
                "k": int(r["k"]),
                "reason": reason,
                "detail": detail,
            }
            for r in recs
        ]

    table = _campaign_table(int(shot))
    if table is None:
        return [], _drops("no-campaign-geometry")
    payload = factory_shot_payloads(
        int(shot),
        nr=nr,
        nz=nz,
        max_slices=MAX_PAYLOAD_SLICES,
        min_ip_ka=MIN_IP_KA,
        table=table,
        cache_grid=True,  # campaign-scope reuse: one grid build per campaign
    )
    if payload is None:
        return [], _drops("no-sensor-windows")
    grid, tbl, basis = payload["grid"], payload["table"], payload["basis"]
    times = np.array([float(p.time_s) for p in payload["payloads"]])

    g = zarr.open_group(str(LEVEL2_SHOTS / f"{shot}.zarr"), mode="r")
    eq = g["equilibrium"]

    rows, drops = [], []

    def _drop(rec, reason: str, detail: str = "") -> None:
        drops.append(
            {
                "shot": int(shot),
                "k": int(rec["k"]),
                "reason": reason,
                "detail": detail,
            }
        )

    for rec in recs:
        t_ref = float(rec["time_s"])
        j = int(np.argmin(np.abs(times - t_ref)))
        if abs(times[j] - t_ref) > TIME_MATCH_S:
            _drop(rec, "no-payload-at-time", f"nearest {abs(times[j] - t_ref):.3f} s")
            continue
        p = payload["payloads"][j]
        try:
            inv = disc_read(p, grid, tbl, basis)
        except Exception as exc:  # noqa: BLE001 — sweep on, record the cause
            _drop(rec, "solve-error", f"{type(exc).__name__}: {exc}"[:160])
            continue
        if inv is None or inv.ring is None:
            _drop(rec, "solve-no-ring")
            continue
        psi = np.asarray(inv.psi_tot, dtype=np.float64)
        centroid = (float(inv.centroid_r), float(inv.centroid_z))
        if not (np.isfinite(centroid[0]) and centroid[0] <= 1.4):
            _drop(rec, "unconfined-centroid", f"R={centroid[0]:.3f}")
            continue
        try:
            eng = boundary_read(psi, grid, centroid, lcfs_norm=1.0)
        except ValueError as exc:
            _drop(rec, "seed-in-wall", str(exc)[:120])
            continue
        if not eng.found:
            _drop(rec, "read-not-found")
            continue
        k = int(rec["k"])
        lcfs = np.c_[
            np.asarray(eq["lcfs_r"][:, k], dtype=np.float64),
            np.asarray(eq["lcfs_z"][:, k], dtype=np.float64),
        ]
        lcfs = lcfs[np.isfinite(lcfs).all(axis=1) & (lcfs[:, 0] > 0)]
        if lcfs.shape[0] < 8:
            _drop(rec, "efit-lcfs-degenerate")
            continue
        efit_radii = polygon_ray_radii(lcfs, eng.axis, LCFS_ANGLES)
        ok = np.isfinite(eng.radii) & np.isfinite(efit_radii)
        if not ok.any():
            _drop(rec, "no-finite-radii-pair")
            continue
        dr_cm = 100.0 * np.abs(eng.radii[ok] - efit_radii[ok])
        efit_axis = (float(eq["magnetic_axis_r"][k]), float(eq["magnetic_axis_z"][k]))
        u_x = np.array(
            [
                (rec["u_x_lo"], rec["x_lo_r"], rec["x_lo_z"]),
                (rec["u_x_hi"], rec["x_hi_r"], rec["x_hi_z"]),
            ]
        )
        from scripts.topology_census import X_BIND_U  # noqa: PLC0415

        binding = np.abs(u_x[:, 0] - 1.0) <= X_BIND_U
        rows.append(
            {
                "shot": int(shot),
                "k": k,
                "time_s": t_ref,
                "ip_ka": float(rec["ip_ka"]),
                "radii_dmed_cm": float(np.median(dr_cm)),
                "radii_dmax_cm": float(np.max(dr_cm)),
                "axis_d_cm": 100.0
                * float(
                    np.hypot(eng.axis[0] - efit_axis[0], eng.axis[1] - efit_axis[1])
                ),
                "xset_d_cm": _xset_match_cm(eng.xset, u_x[binding][:, 1:]),
                "dev_is_diverted": bool(eng.is_diverted),
                "class_margin": float(np.clip(eng.class_margin, -1.0, 1.0)),
            }
        )
    return rows, drops


def _eddy_center_builder(shot: int, payload: dict):
    """Precompute the vessel-eddy known-drive trajectory hook for one shot.

    Returns ``(sidecar, centers_fn)`` or ``(None, None)`` when the raw drive
    streams / eigenbasis are unavailable.  ``centers_fn(label_times,
    i_cell_seq)`` maps the chain's Pass-1 plasma history to sidecar-coordinate
    trajectory centers — the measured coil drives + the plasma's own flux
    swing integrated on the vessel L/R eigenmodes from the quiescent start.
    """
    from imas_ambix.gs.operator import build_operator  # noqa: PLC0415
    from scripts.closure_gate_eval import _shot_passive_sidecar  # noqa: PLC0415
    from scripts.dynamic_passive_gate_eval import (  # noqa: PLC0415
        _trajectory_centers,
        raw_drive_streams,
        shot_eigenbasis_sectionavg,
    )
    from scripts.spine_label_factory import frozen_spine_config  # noqa: PLC0415

    spine, _sha = frozen_spine_config()
    sidecar = _shot_passive_sidecar(payload, int(spine["interior_solve"]["passive_k"]))
    modes = np.asarray(sidecar["modes"], dtype=np.float64)
    fwd = build_operator(payload["table"])
    raw = raw_drive_streams(shot, fwd)
    if raw is None:
        return None, None
    eigen = shot_eigenbasis_sectionavg(
        payload, payload["table"].signature.key, EDDY_N_MODES)

    def centers_fn(label_times, i_cell_seq):
        centers, _i_circ = _trajectory_centers(
            eigen, modes, raw, np.asarray(label_times, dtype=np.float64),
            np.asarray(i_cell_seq, dtype=np.float64), 1.0)
        return centers

    return sidecar, centers_fn


def full_engine_rows_for_shot(
    shot: int, recs, *, nr: int, nz: int, arm: str
) -> tuple[list[dict], list[dict]]:
    """Run the four-pass engine on one shot and score its boundary read.

    ``recs`` are ALL of this shot's valid census rows (any class) — the chain
    is temporal, so one solve services every requested slice.  Returns
    ``(rows, drops)``; every rec lands in exactly one list, with drop reasons
    split into engine-attributable vs harness accounting (see module glossary).
    """
    import zarr  # noqa: PLC0415

    from imas_ambix.latent.connectivity_boundary import boundary_read  # noqa: PLC0415
    from imas_ambix.worldmodel.equilibrium_labels import LCFS_ANGLES  # noqa: PLC0415
    from scripts.heldout_mse_gate_eval import (  # noqa: PLC0415
        _campaign_table,
        coupled_solve_chain,
        frozen_eta_params,
    )
    from scripts.spine_label_factory import factory_shot_payloads  # noqa: PLC0415
    from scripts.topology_census import X_BIND_U  # noqa: PLC0415

    def _all(reason: str, detail: str = "") -> tuple[list[dict], list[dict]]:
        # one shot-level entry; n_slices bounds the attempted count honestly
        # (the chain solves at most CHAIN_MAX_SLICES of the shot's census rows)
        return [], [
            {"shot": int(shot), "reason": reason, "detail": detail,
             "n_slices": int(min(len(recs), CHAIN_MAX_SLICES))}
        ]

    table = _campaign_table(int(shot))
    if table is None:
        return _all("chain-unavailable", "no-campaign-geometry")
    payload = factory_shot_payloads(
        int(shot), nr=nr, nz=nz, max_slices=CHAIN_MAX_SLICES,
        min_ip_ka=CHAIN_MIN_IP_KA, table=table, cache_grid=True)
    if payload is None:
        return _all("chain-unavailable", "no-sensor-windows")

    passive = centers_fn = None
    if arm == "eddy":
        try:
            passive, centers_fn = _eddy_center_builder(int(shot), payload)
        except Exception as exc:  # noqa: BLE001 — surfaced per shot, sweep on
            return _all("chain-unavailable", f"eddy-setup: {exc}"[:160])
        if passive is None:
            return _all("chain-unavailable", "eddy-no-raw-drives")

    try:
        chain = coupled_solve_chain(
            int(shot), nr=nr, nz=nz, sigma=CHAIN_SIGMA_M,
            eta_params=list(frozen_eta_params()),
            prior_weight=CHAIN_PRIOR_WEIGHT, n_sub=CHAIN_N_SUB,
            par_weight=CHAIN_PAR_WEIGHT, n_rho=CHAIN_N_RHO,
            max_slices=CHAIN_MAX_SLICES, min_ip_ka=CHAIN_MIN_IP_KA,
            skip_basin=(arm == "nobasin"), passive=passive,
            passive_centers_fn=centers_fn,
            passive_weight=EDDY_KNOWN_DRIVE_WEIGHT, cache_grid=True)
    except Exception as exc:  # noqa: BLE001 — a dead chain is named, not fatal
        return _all("chain-failed", f"{type(exc).__name__}: {exc}"[:160])
    if not chain["slices"]:
        reason = chain.get("reason", "")
        if reason == "too few scored slices":
            return _all("chain-failed", reason)  # engine-side attrition
        return _all("chain-unavailable", reason)

    grid = chain["grid"]
    payloads = chain["payload"]["payloads"]
    # engine attempt = one payload slice.  Completion is measured over the
    # payload slices the engine tried (each counted ONCE) that carry a valid
    # census reference — never per census rec, which over-counts a dropped
    # payload slice every time a nearby census rec maps to it.
    fit_by_payload = {int(s["k"]): (s, chain["fits"][j])
                      for j, s in enumerate(chain["slices"])}
    rec_times = np.array([float(r["time_s"]) for r in recs])

    g = zarr.open_group(str(LEVEL2_SHOTS / f"{shot}.zarr"), mode="r")
    eq = g["equilibrium"]

    rows, drops = [], []
    used_rec: set[int] = set()

    def _drop(rec, reason: str, detail: str = "") -> None:
        drops.append({"shot": int(shot), "k": int(rec["k"]),
                      "cls": _rec_class(rec), "time_s": float(rec["time_s"]),
                      "reason": reason, "detail": detail})

    for p_idx in range(len(payloads)):
        t_p = float(payloads[p_idx].time_s)
        ir = int(np.argmin(np.abs(rec_times - t_p)))
        if abs(rec_times[ir] - t_p) > TIME_MATCH_S:
            continue  # this attempted slice has no census-valid reference
        used_rec.add(ir)
        rec = recs[ir]
        t_ref = float(rec["time_s"])
        if p_idx not in fit_by_payload:
            _drop(rec, "fit-not-scored", "pass-1 attrition")  # disc/basin drop
            continue
        s, f = fit_by_payload[p_idx]
        if not (f.scored and f.psi is not None):
            _drop(rec, "fit-not-scored")
            continue
        axis_r, axis_z = _fit_axis(f)
        if not (np.isfinite(axis_r) and axis_r <= 1.4):
            _drop(rec, "unconfined-axis", f"R={axis_r:.3f}")
            continue
        psi = np.asarray(f.psi, dtype=np.float64)
        seed = ((axis_r, axis_z) if np.isfinite(axis_z) else
                (float(s["centroid"][0]), float(s["centroid"][1])))
        try:
            eng = boundary_read(psi, grid, seed, lcfs_norm=1.0)
        except ValueError as exc:
            _drop(rec, "seed-in-wall", str(exc)[:120])
            continue
        if not eng.found:
            _drop(rec, "read-not-found")
            continue
        k = int(rec["k"])
        lcfs = np.c_[
            np.asarray(eq["lcfs_r"][:, k], dtype=np.float64),
            np.asarray(eq["lcfs_z"][:, k], dtype=np.float64),
        ]
        lcfs = lcfs[np.isfinite(lcfs).all(axis=1) & (lcfs[:, 0] > 0)]
        if lcfs.shape[0] < 8:
            _drop(rec, "efit-lcfs-degenerate")
            continue
        efit_radii = polygon_ray_radii(lcfs, eng.axis, LCFS_ANGLES)
        ok = np.isfinite(eng.radii) & np.isfinite(efit_radii)
        if not ok.any():
            _drop(rec, "no-finite-radii-pair")
            continue
        dr_cm = 100.0 * np.abs(eng.radii[ok] - efit_radii[ok])
        efit_axis = (float(eq["magnetic_axis_r"][k]), float(eq["magnetic_axis_z"][k]))
        u_x = np.array([
            (rec["u_x_lo"], rec["x_lo_r"], rec["x_lo_z"]),
            (rec["u_x_hi"], rec["x_hi_r"], rec["x_hi_z"]),
        ])
        binding = np.abs(u_x[:, 0] - 1.0) <= X_BIND_U
        rows.append({
            "shot": int(shot), "k": k, "time_s": t_ref,
            "ip_ka": float(rec["ip_ka"]), "cls": _rec_class(rec),
            "phase": "ramp" if t_ref < RAMP_END_S else "flattop",
            "radii_dmed_cm": float(np.median(dr_cm)),
            "radii_dmax_cm": float(np.max(dr_cm)),
            "axis_d_cm": 100.0 * float(
                np.hypot(eng.axis[0] - efit_axis[0], eng.axis[1] - efit_axis[1])),
            "xset_d_cm": _xset_match_cm(eng.xset, u_x[binding][:, 1:]),
            "dev_is_diverted": bool(eng.is_diverted),
            "class_margin": float(np.clip(eng.class_margin, -1.0, 1.0)),
        })
    n_thinned = int(len(recs) - len(used_rec))  # census slices past the budget
    if n_thinned:
        drops.append({"shot": int(shot), "reason": "not-attempted-thinning",
                      "detail": "payload slice budget", "n_slices": n_thinned})
    return rows, drops


def _rec_class(rec) -> str:
    from scripts.topology_census import CLASSES  # noqa: PLC0415

    name = CLASSES[int(rec["cls"])]
    return "snowflake-candidate" if bool(rec["snowflake"]) else name


def _fit_axis(f) -> tuple[float, float]:
    if not (f.scored and f.target is not None):
        return float("nan"), float("nan")
    return float(f.target[0]), float(f.target[1])


def score_class(
    cname: str, recs: np.ndarray, *, nr: int, nz: int, checkpoint: Path | None = None
) -> dict:
    rows: list[dict] = []
    drops: list[dict] = []
    done: set[int] = set()
    if checkpoint is not None and checkpoint.exists():
        state = json.loads(checkpoint.read_text())
        rows = state["rows"]
        # a scored shot is done; a shot whose slices ALL dropped reruns so the
        # instrumented reason codes replace a silent gap
        done = {int(r["shot"]) for r in rows}
        logger.info("  %s: resuming — %d rows from checkpoint", cname, len(rows))
    by_shot: dict[int, list] = {}
    for rec in recs:
        if int(rec["shot"]) not in done:
            by_shot.setdefault(int(rec["shot"]), []).append(rec)
    for i, (shot, srecs) in enumerate(sorted(by_shot.items())):
        try:
            srows, sdrops = engine_rows_for_shot(shot, srecs, nr=nr, nz=nz)
            rows += srows
            drops += sdrops
        except Exception as exc:  # noqa: BLE001 — engine attrition is reported
            logger.warning("  %s shot %d failed: %s", cname, shot, exc)
            drops += [
                {
                    "shot": int(shot),
                    "k": int(r["k"]),
                    "reason": "shot-error",
                    "detail": f"{type(exc).__name__}: {exc}"[:160],
                }
                for r in srecs
            ]
        if (i + 1) % 10 == 0:
            logger.info(
                "  %s: %d/%d shots, %d rows", cname, i + 1, len(by_shot), len(rows)
            )
        if checkpoint is not None:
            # scored rows survive a wall-clock kill; the artifact is rebuilt from
            # the last checkpoint on rerun
            checkpoint.write_text(
                json.dumps({"class": cname, "rows": rows, "drops": drops})
            )
    diverted_expected = cname != "limited"
    class_hits = [r["dev_is_diverted"] == diverted_expected for r in rows]
    drop_counts: dict[str, int] = {}
    for d in drops:
        drop_counts[d["reason"]] = drop_counts.get(d["reason"], 0) + 1
    return {
        "class": cname,
        "n_selected": int(recs.size),
        "n_scored": len(rows),
        "n_dropped": len(drops),
        "drop_counts": dict(sorted(drop_counts.items(), key=lambda kv: -kv[1])),
        "radii_dmed_cm": _agg([r["radii_dmed_cm"] for r in rows]),
        "axis_d_cm": _agg([r["axis_d_cm"] for r in rows]),
        "xset_d_cm": _agg([r["xset_d_cm"] for r in rows]),
        "class_agreement": float(np.mean(class_hits)) if class_hits else None,
        "rows": rows,
        "drops": drops,
    }


def class_verdict(res: dict) -> dict:
    supported = res["n_scored"] >= SUPPORT_FLOOR
    if res["class"] not in PASS_CLASSES:
        return {"supported": supported, "gated": False, "pass": None}
    if not supported:
        return {"supported": False, "gated": True, "pass": None}
    checks = {
        "radii": res["radii_dmed_cm"]["med"] is not None
        and res["radii_dmed_cm"]["med"] <= ENGINE_LCFS_TOL_CM,
        "class": res["class_agreement"] is not None
        and res["class_agreement"] >= ENGINE_CLASS_ACC_TOL,
    }
    return {
        "supported": True,
        "gated": True,
        "checks": checks,
        "pass": all(checks.values()),
    }


def _full_worker(job: tuple) -> tuple[int, list[dict], list[dict]]:
    shot, recs, nr, nz, arm = job
    try:
        rows, drops = full_engine_rows_for_shot(shot, recs, nr=nr, nz=nz, arm=arm)
    except Exception as exc:  # noqa: BLE001 — a dead shot is named, not fatal
        rows, drops = [], [
            {"shot": int(shot), "reason": "chain-failed",
             "detail": f"{type(exc).__name__}: {exc}"[:160],
             "n_slices": int(min(len(recs), CHAIN_MAX_SLICES))}
        ]
    return int(shot), rows, drops


def run_full_chain_class(
    cname: str, shots: list[int], census_rows: np.ndarray, *,
    nr: int, nz: int, arm: str, workers: int, checkpoint: Path | None
) -> dict:
    """Four-pass engine over one class's stratified shots, all census recs.

    Each selected shot contributes EVERY valid census slice it carries (any
    class) that lands on a chain slice — the per-phase tables come from the
    chain's own time cover.  Rows are tagged with the REC's class, so the
    aggregation regroups honestly across arms.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: PLC0415

    from scripts.topology_census import CLASSES  # noqa: PLC0415

    invalid_ci = CLASSES.index("invalid")
    valid = (census_rows["cls"] != invalid_ci) & (
        np.abs(census_rows["ip_ka"]) >= CHAIN_MIN_IP_KA)

    rows: list[dict] = []
    drops: list[dict] = []
    done: set[int] = set()
    if checkpoint is not None and checkpoint.exists():
        state = json.loads(checkpoint.read_text())
        rows, drops = state["rows"], state["drops"]
        done = set(state["done_shots"])
        logger.info("  %s/%s: resuming — %d shots done", cname, arm, len(done))

    jobs = []
    for shot in shots:
        if int(shot) in done:
            continue
        recs = census_rows[valid & (census_rows["shot"] == int(shot))]
        recs = recs[np.argsort(recs["time_s"])]
        if recs.size == 0:
            done.add(int(shot))
            continue
        jobs.append((int(shot), recs, nr, nz, arm))

    def _flush() -> None:
        if checkpoint is not None:
            checkpoint.write_text(json.dumps(
                {"class": cname, "arm": arm, "rows": rows, "drops": drops,
                 "done_shots": sorted(done)}))

    if workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_full_worker, j) for j in jobs]
            for i, fut in enumerate(as_completed(futs)):
                shot, srows, sdrops = fut.result()
                rows += srows
                drops += sdrops
                done.add(shot)
                if (i + 1) % 4 == 0 or (i + 1) == len(futs):
                    logger.info("  %s/%s: %d/%d shots, %d rows",
                                cname, arm, len(done), len(shots), len(rows))
                _flush()
    else:
        for j in jobs:
            shot, srows, sdrops = _full_worker(j)
            rows += srows
            drops += sdrops
            done.add(shot)
            _flush()
    _flush()
    return {"class": cname, "arm": arm, "n_shots": len(shots),
            "rows": rows, "drops": drops}


ARTIFACT_DIR = Path("imas_ambix/latent/artifacts/patch_gate")
FIGURE_DIR = Path("docs/figures/connectivity-topology-reader")
FULL_ARMS = ("plain", "nobasin", "eddy")
BARE_BASELINE_ARTIFACT = ARTIFACT_DIR / "topology_engine_crossval-v0.json"
VERDICT_ARTIFACT = ARTIFACT_DIR / "topology_full_engine_verdicts-v0.json"

ALL_CLASSES = ("limited", "sn-lower", "sn-upper", "connected-dn", "marginal-dn",
               "snowflake-candidate")


def _full_class_artifact(arm: str, cname: str) -> Path:
    """Per-class arm artifact (per-class files merge cleanly across jobs)."""
    return ARTIFACT_DIR / f"topology_full_engine_crossval-{arm}-{cname}.json"


def _load_arm_rows(arm: str) -> dict | None:
    """Concatenate an arm's per-class artifacts into one rows/drops dict."""
    rows: list[dict] = []
    drops: list[dict] = []
    found = False
    for path in sorted(ARTIFACT_DIR.glob(
            f"topology_full_engine_crossval-{arm}-*.json")):
        found = True
        d = json.loads(path.read_text())
        rows += d.get("rows", [])
        drops += d.get("drops", [])
    if not found:
        return None
    return {"rows": _dedup_rows(rows), "drops": drops}


def _dedup_rows(rows: list[dict]) -> list[dict]:
    seen, out = set(), []
    for r in rows:
        key = (r["shot"], r["k"])
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def _class_phase_table(rows: list[dict]) -> dict:
    """Per class × phase medians + support + class agreement."""
    out: dict = {}
    for cname in ALL_CLASSES:
        crows = [r for r in rows if r["cls"] == cname]
        entry: dict = {"n": len(crows)}
        for phase in ("flattop", "ramp", "all"):
            prows = (crows if phase == "all"
                     else [r for r in crows if r["phase"] == phase])
            entry[phase] = {
                "n": len(prows),
                "radii_dmed_cm": _agg([r["radii_dmed_cm"] for r in prows]),
                "axis_d_cm": _agg([r["axis_d_cm"] for r in prows]),
                "xset_d_cm": _agg([r["xset_d_cm"] for r in prows]),
            }
        expected_div = cname != "limited"
        hits = [r["dev_is_diverted"] == expected_div for r in crows]
        entry["class_agreement"] = float(np.mean(hits)) if hits else None
        hits_ft = [r["dev_is_diverted"] == expected_div
                   for r in crows if r["phase"] == "flattop"]
        entry["class_agreement_flattop"] = (
            float(np.mean(hits_ft)) if hits_ft else None)
        out[cname] = entry
    return out


def _completion(rows: list[dict], drops: list[dict]) -> dict:
    """Scored / attempted valid slices; engine failures individually named."""
    n_scored = len(rows)
    failures = []
    n_failed = 0
    for d in drops:
        if d["reason"] in ENGINE_FAILURE_REASONS:
            n = int(d.get("n_slices", 1))
            n_failed += n
            failures.append(d)
    attempted = n_scored + n_failed
    return {
        "n_scored": n_scored,
        "n_engine_failed": n_failed,
        "n_attempted": attempted,
        "completion": (n_scored / attempted) if attempted else None,
        "bar": COMPLETION_BAR,
        "failures_named": failures,
        "harness_accounting": _drop_counts(
            [d for d in drops if d["reason"] not in ENGINE_FAILURE_REASONS]),
    }


def _drop_counts(drops: list[dict]) -> dict:
    out: dict[str, int] = {}
    for d in drops:
        out[d["reason"]] = out.get(d["reason"], 0) + int(d.get("n_slices", 1))
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def _paired_class_delta(
    rows_a: list[dict], rows_b: list[dict], phase: str | None = None
) -> dict:
    """Per-class median of paired (b − a) radii residuals on common slices."""
    a_by = {(r["shot"], r["k"]): r for r in rows_a}
    out: dict = {}
    for cname in ALL_CLASSES:
        deltas = []
        for r in rows_b:
            if r["cls"] != cname or (phase and r["phase"] != phase):
                continue
            ra = a_by.get((r["shot"], r["k"]))
            if ra is not None:
                deltas.append(r["radii_dmed_cm"] - ra["radii_dmed_cm"])
        out[cname] = {
            "n_paired": len(deltas),
            "delta_med_cm": float(np.median(deltas)) if deltas else None,
        }
    return out


def aggregate_verdicts(baseline_path: Path = BARE_BASELINE_ARTIFACT) -> dict:
    """Fold the arm artifacts into the gate verdicts + figures."""
    arms: dict[str, dict] = {}
    for arm in FULL_ARMS:
        loaded = _load_arm_rows(arm)
        if loaded is not None:
            arms[arm] = loaded
    if "plain" not in arms:
        raise SystemExit("no plain-arm artifact — run --chain full first")
    plain = arms["plain"]

    table = _class_phase_table(plain["rows"])
    completion = _completion(plain["rows"], plain["drops"])

    # per-class boundary-quality checks (pre-declared tolerances)
    quality: dict = {}
    for cname in ALL_CLASSES:
        e = table[cname]
        med = e["flattop"]["radii_dmed_cm"]["med"]
        acc = e["class_agreement_flattop"]
        supported = e["flattop"]["n"] >= SUPPORT_FLOOR
        quality[cname] = {
            "supported": supported,
            "flattop_radii_med_cm": med,
            "flattop_class_agreement": acc,
            "radii_ok": (med is not None and med <= FLATTOP_LCFS_TOL_CM),
            "class_ok": (acc is not None and acc >= FULL_CLASS_ACC_TOL),
        }
        quality[cname]["pass"] = (
            bool(quality[cname]["radii_ok"] and quality[cname]["class_ok"])
            if supported else None)
    supported_q = [q for q in quality.values() if q["supported"]]
    quality_pass = bool(supported_q) and all(q["pass"] for q in supported_q)

    # eddy known-drive ablation (paired on common slices)
    eddy_verdict = None
    if "eddy" in arms:
        eddy_rows = arms["eddy"]["rows"]
        flattop_delta = _paired_class_delta(plain["rows"], eddy_rows, "flattop")
        ramp_delta = _paired_class_delta(plain["rows"], eddy_rows, "ramp")
        regressions = {
            c: d for c, d in flattop_delta.items()
            if d["delta_med_cm"] is not None
            and d["delta_med_cm"] > EDDY_REGRESSION_BAND_CM}
        eddy_verdict = {
            "table": _class_phase_table(eddy_rows),
            "flattop_delta_vs_plain_cm": flattop_delta,
            "ramp_delta_vs_plain_cm": ramp_delta,
            "regression_band_cm": EDDY_REGRESSION_BAND_CM,
            "regressions": regressions,
            "no_class_regression": not regressions,
        }

    # basin-pass ablation (report-only; pre-declared decision rule)
    basin_verdict = None
    if "nobasin" in arms:
        nb_rows = arms["nobasin"]["rows"]
        nb_table = _class_phase_table(nb_rows)
        nb_completion = _completion(nb_rows, arms["nobasin"]["drops"])
        per_class = {}
        for cname in ALL_CLASSES:
            e = nb_table[cname]
            med = e["flattop"]["radii_dmed_cm"]["med"]
            acc = e["class_agreement_flattop"]
            if e["flattop"]["n"] == 0:
                per_class[cname] = {"holds": None, "n": 0}
                continue
            per_class[cname] = {
                "n": e["flattop"]["n"],
                "flattop_radii_med_cm": med,
                "flattop_class_agreement": acc,
                "holds": bool(
                    med is not None and med <= FLATTOP_LCFS_TOL_CM
                    and acc is not None and acc >= FULL_CLASS_ACC_TOL),
            }
        evaluated = [v for v in per_class.values() if v["holds"] is not None]
        basin_verdict = {
            "table": nb_table,
            "completion": nb_completion,
            "flattop_delta_vs_plain_cm": _paired_class_delta(
                plain["rows"], nb_rows, "flattop"),
            "per_class_within_tolerances": per_class,
            "basin_pass_retirable": (bool(evaluated)
                                     and all(v["holds"] for v in evaluated)),
        }

    # bare-initialiser baseline comparison (the measured value of the chain)
    baseline_cmp = None
    if baseline_path.exists():
        base = json.loads(baseline_path.read_text())
        baseline_cmp = {}
        for cname, cls_res in base.get("classes", {}).items():
            full_med = table.get(cname, {}).get("all", {}).get(
                "radii_dmed_cm", {}).get("med") if cname in table else None
            full_ft = table.get(cname, {}).get("flattop", {}).get(
                "radii_dmed_cm", {}).get("med") if cname in table else None
            baseline_cmp[cname] = {
                "bare_radii_med_cm": cls_res["radii_dmed_cm"]["med"],
                "bare_axis_med_cm": cls_res["axis_d_cm"]["med"],
                "full_radii_med_cm": full_med,
                "full_radii_med_flattop_cm": full_ft,
                "full_axis_med_cm": table.get(cname, {}).get("all", {}).get(
                    "axis_d_cm", {}).get("med") if cname in table else None,
            }

    verdict = {
        "verdict_keys": {
            "G-E1": (completion["completion"] is not None
                     and completion["completion"] >= COMPLETION_BAR),
            "G-E2": quality_pass,
            "G-E3": (eddy_verdict["no_class_regression"]
                     if eddy_verdict else None),
        },
        "tolerances": {
            "flattop_lcfs_med_cm": FLATTOP_LCFS_TOL_CM,
            "class_acc": FULL_CLASS_ACC_TOL,
            "completion_bar": COMPLETION_BAR,
            "eddy_regression_band_cm": EDDY_REGRESSION_BAND_CM,
            "support_floor": SUPPORT_FLOOR,
        },
        "completion": completion,
        "class_phase_table": table,
        "per_class_quality": quality,
        "eddy_ablation": eddy_verdict,
        "basin_ablation": basin_verdict,
        "baseline_comparison": baseline_cmp,
    }
    VERDICT_ARTIFACT.write_text(json.dumps(verdict, indent=2, default=float))
    logger.info("verdicts -> %s", VERDICT_ARTIFACT)
    _aggregate_figures(verdict, arms)
    return verdict


def _aggregate_figures(verdict: dict, arms: dict) -> None:
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    classes = [c for c in ALL_CLASSES
               if verdict["class_phase_table"][c]["n"] > 0]
    x = np.arange(len(classes))

    # ---- full engine vs bare-initialiser baseline ----
    cmp_ = verdict.get("baseline_comparison") or {}
    if cmp_:
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        bare = [cmp_.get(c, {}).get("bare_radii_med_cm") or np.nan
                for c in classes]
        full_ft = [cmp_.get(c, {}).get("full_radii_med_flattop_cm") or np.nan
                   for c in classes]
        ax.bar(x - 0.2, bare, width=0.38, color="#c66",
               label="bare disc-read initialiser (seed baseline)")
        ax.bar(x + 0.2, full_ft, width=0.38, color="#268",
               label="four-pass engine (flat-top)")
        ax.axhline(FLATTOP_LCFS_TOL_CM, color="k", ls="--", lw=0.9,
                   label=f"pre-declared tolerance {FLATTOP_LCFS_TOL_CM} cm")
        ax.set_yscale("log")
        ax.set_xticks(x, classes, fontsize=8, rotation=12)
        ax.set_ylabel("median LCFS-radii residual vs EFIT [cm]")
        ax.set_title("per-class boundary residual — pinned+coupled chain vs "
                     "bare seed", fontsize=9)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / "fig-full-engine-vs-baseline.png", dpi=130)
        plt.close(fig)

    # ---- phase split ----
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for off, phase, col in ((-0.2, "flattop", "#268"), (0.2, "ramp", "#e90")):
        med = [verdict["class_phase_table"][c][phase]["radii_dmed_cm"]["med"]
               or np.nan for c in classes]
        n = [verdict["class_phase_table"][c][phase]["n"] for c in classes]
        ax.bar(x + off, med, width=0.38, color=col, label=phase)
        for xi, (m, ni) in zip(x + off, zip(med, n, strict=True), strict=True):
            if np.isfinite(m):
                ax.text(xi, m, f"n={ni}", ha="center", va="bottom", fontsize=6)
    ax.axhline(FLATTOP_LCFS_TOL_CM, color="k", ls="--", lw=0.9)
    ax.set_xticks(x, classes, fontsize=8, rotation=12)
    ax.set_ylabel("median LCFS-radii residual [cm]")
    ax.set_title(f"four-pass engine per phase (ramp = t < {RAMP_END_S} s)",
                 fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig-full-engine-phase.png", dpi=130)
    plt.close(fig)

    # ---- eddy known-drive ablation ----
    ev = verdict.get("eddy_ablation")
    if ev:
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        for off, key, col in ((-0.2, "ramp_delta_vs_plain_cm", "#e90"),
                              (0.2, "flattop_delta_vs_plain_cm", "#268")):
            d = [ev[key][c]["delta_med_cm"] if ev[key][c]["delta_med_cm"]
                 is not None else np.nan for c in classes]
            ax.bar(x + off, d, width=0.38, color=col,
                   label=key.split("_")[0])
        ax.axhline(0, color="k", lw=0.8)
        ax.axhline(EDDY_REGRESSION_BAND_CM, color="#c66", ls="--", lw=0.9,
                   label="regression band")
        ax.set_xticks(x, classes, fontsize=8, rotation=12)
        ax.set_ylabel("Δ median residual (eddy − plain) [cm]")
        ax.set_title("vessel-eddy known-drive ablation (negative = eddy helps)",
                     fontsize=9)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / "fig-eddy-ablation.png", dpi=130)
        plt.close(fig)

    # ---- basin-pass ablation ----
    bv = verdict.get("basin_ablation")
    if bv:
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        full_ft = [verdict["class_phase_table"][c]["flattop"]
                   ["radii_dmed_cm"]["med"] or np.nan for c in classes]
        nb_ft = [bv["table"][c]["flattop"]["radii_dmed_cm"]["med"] or np.nan
                 for c in classes]
        ax.bar(x - 0.2, full_ft, width=0.38, color="#268",
               label="two-pass (basin + profile)")
        ax.bar(x + 0.2, nb_ft, width=0.38, color="#9b6",
               label="profile-only (disc cold start)")
        ax.axhline(FLATTOP_LCFS_TOL_CM, color="k", ls="--", lw=0.9)
        ax.set_xticks(x, classes, fontsize=8, rotation=12)
        ax.set_ylabel("flat-top median LCFS-radii residual [cm]")
        ax.set_title("basin-pass necessity ablation (report-only)", fontsize=9)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / "fig-basin-ablation.png", dpi=130)
        plt.close(fig)
    logger.info("figures -> %s", FIGURE_DIR)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--census",
        type=Path,
        default=Path("imas_ambix/latent/artifacts/patch_gate/topology_census-v0.npz"),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="artifact path (defaults per chain/arm)",
    )
    ap.add_argument("--nr", type=int, default=65)
    ap.add_argument("--nz", type=int, default=97)
    ap.add_argument("--chain", choices=("bare", "full"), default="full")
    ap.add_argument("--arm", choices=("plain", "nobasin", "eddy"),
                    default="plain")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--classes", nargs="*", default=None)
    ap.add_argument("--per-class", type=int, default=None, help="cap shots per class")
    ap.add_argument("--aggregate", action="store_true",
                    help="fold arm artifacts into gate verdicts + figures")
    args = ap.parse_args()

    if args.aggregate:
        aggregate_verdicts()
        return

    if args.chain == "full":
        census_rows = np.load(args.census)["rows"]
        sel = stratified_selection(census_rows)
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        config = {
            "sigma_m": CHAIN_SIGMA_M, "prior_weight": CHAIN_PRIOR_WEIGHT,
            "n_sub": CHAIN_N_SUB, "n_rho": CHAIN_N_RHO,
            "max_slices": CHAIN_MAX_SLICES, "min_ip_ka": CHAIN_MIN_IP_KA,
            "eddy_weight": EDDY_KNOWN_DRIVE_WEIGHT, "eddy_n_modes": EDDY_N_MODES,
        }
        for cname, recs in sel.items():
            if args.classes and cname not in args.classes:
                continue
            shots = sorted({int(s) for s in np.unique(recs["shot"])})
            if args.per_class and len(shots) > args.per_class:
                idx = np.linspace(0, len(shots) - 1, args.per_class).astype(int)
                shots = [shots[i] for i in idx]
            logger.info("class %s (%s): %d shots", cname, args.arm, len(shots))
            out = args.out or _full_class_artifact(args.arm, cname)
            ckpt = out.with_name(out.stem + "_checkpoint.json")
            res = run_full_chain_class(
                cname, shots, census_rows, nr=args.nr, nz=args.nz,
                arm=args.arm, workers=args.workers, checkpoint=ckpt)
            rows_d = _dedup_rows(res["rows"])
            artifact = {
                "chain": "full", "arm": args.arm, "class": cname,
                "config": config,
                "n_shots": len(shots), "n_rows": len(rows_d),
                "class_phase_table": _class_phase_table(rows_d),
                "completion": _completion(rows_d, res["drops"]),
                "rows": res["rows"], "drops": res["drops"],
            }
            out.write_text(json.dumps(artifact, indent=1, default=float))
            logger.info("%s/%s -> %s (%d rows)",
                        args.arm, cname, out, len(rows_d))
        return

    rows = np.load(args.census)["rows"]
    sel = stratified_selection(rows)
    out = args.out or Path(
        "imas_ambix/latent/artifacts/patch_gate/topology_engine_crossval-v0.json")
    args.out = out
    results, verdicts = {}, {}
    for cname, recs in sel.items():
        if args.classes and cname not in args.classes:
            continue
        if args.per_class and recs.size > args.per_class:
            idx = np.linspace(0, recs.size - 1, args.per_class).astype(int)
            recs = recs[idx]
        logger.info("class %s: %d selected", cname, recs.size)
        ckpt = args.out.with_name(args.out.stem + f"_{cname}_checkpoint.json")
        res = score_class(cname, recs, nr=args.nr, nz=args.nz, checkpoint=ckpt)
        results[cname] = res
        verdicts[cname] = class_verdict(res)
        logger.info(
            "  scored %d — radii med %s cm, axis med %s cm, xset med %s cm, acc %s",
            res["n_scored"],
            res["radii_dmed_cm"]["med"],
            res["axis_d_cm"]["med"],
            res["xset_d_cm"]["med"],
            res["class_agreement"],
        )
    gated = [v for v in verdicts.values() if v.get("gated")]
    gate_pass = (
        bool(gated)
        and all(v["pass"] for v in gated if v["supported"])
        and any(v["supported"] for v in gated)
    )
    artifact = {
        "verdict_keys": {"T-F3": bool(gate_pass)},
        "tolerances": {
            "pass_classes": list(PASS_CLASSES),
            "lcfs_med_cm": ENGINE_LCFS_TOL_CM,
            "class_acc": ENGINE_CLASS_ACC_TOL,
            "support_floor": SUPPORT_FLOOR,
        },
        "verdicts": verdicts,
        "classes": {
            c: {k: v for k, v in r.items() if k not in ("rows", "drops")}
            for c, r in results.items()
        },
        "rows": {c: r["rows"] for c, r in results.items()},
        "drops": {c: r["drops"] for c, r in results.items()},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2))
    logger.info("T-F3 %s → %s", "PASS" if gate_pass else "FAIL", args.out)


if __name__ == "__main__":
    main()
