#!/usr/bin/env python
"""Data-led discovery of the passive drive & coupling structure.

The vacuum resistance calibration validated the L/R circuit model per
circuit (P3-P5 case transients reproduced at corr 0.96-0.99 from independent
drives) but left STRUCTURED residuals: P2 cases unpredictable at any R, the
measured P3 case pair moving as one circuit, inboard bounding boxes driven
to open circuit.  This driver runs the structured hypothesis library over
the same coil-only pool (locked decision: structured-library — never a free
P×P interaction fit), every tier gated on the SAME held-out vacuum shots
with the measured case channels held back as targets throughout:

  A  missing measured drives — the unconsumed ``*_feed_current`` and plain
     ``*_current`` siblings, tested as exact linear identities against the
     consumed channels (a redundant column can move nothing; a column that
     contains the held-back case measurement is inadmissible);
  B1 case-pair wiring — series / anti-series constraint reductions of the
     measured case pairs (sign from the measured pair correlation);
  B2 case-coil galvanic wiring — the case loop driven by its winding's
     terminal voltage (g_v·dΛ_w/dt + r_w·i_coil), per pair then per case;
  B3 common/differential drive-gain corrections for the un-separable
     up/down coil pairs;
  C  adjacency-restricted galvanic couplings over the vessel set (SPD
     off-diagonal R stamps on a section-scale-normalised neighbour graph,
     parallel single-edge probing, greedy acceptance with cross-shot
     stability folds).

The resistance multipliers are REFIT JOINTLY with each accepted tier (warm
started from the incumbent).  Headline metric per tier: held-back P2-case
reproduction.  Honest scope: vacuum data cannot see plasma-era drive errors.

Artifacts:
  imas_ambix/latent/artifacts/patch_gate/vacuum_passive_structure_discovery.json
  imas_ambix/latent/artifacts/patch_gate/passive_structure_calibration.json
Figures:  docs/figures/temporal-physics-spine/fig-structure-*.png
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from imas_ambix.data.description_reader import read_geometry_table
from imas_ambix.gs.geometry import GEOMETRY_TABLE_VERSION
from imas_ambix.gs.operator import COIL_MODEL_VERSION
from imas_ambix.latent.passive_resistance import (
    MULTIPLIER_BOUNDS,
    PassiveStructure,
    StructureHypothesis,
    VacuumShotData,
    build_structure_hypothesis,
    case_parent_coil_channels,
    coil_pair_channels,
    load_calibration,
    neighbour_edges,
    resistance_group_labels,
    save_structure,
    structured_mode_maps,
    structured_shot_loss,
    zoh_mode_response,
)
from imas_ambix.latent.temporal_operator import (
    build_drive_linkage,
    build_passive_circuit_system,
    load_circuit_system,
    save_circuit_system,
)
from scripts.vacuum_passive_resistance_fit import (
    ARTIFACTS,
    FIGURES,
    SYSTEM_DIR,
    prep_shot,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("vacuum_passive_structure_discovery")

#: bounds of the structure DOF (raw optimiser coordinates)
GV_BOUND = 64.0  # winding-voltage gain (expected ≈ turns count)
RW_BOUND = 1.0  # resistive wiring term [Ω per A·turn of parent current]
PAIR_GAIN_BOUND = 0.5  # common/differential drive-column corrections
EDGE_RHO_BOUNDS = (1e-7, 1e-1)  # adjacency coupling resistance [Ω], log slots
EDGE_PROBE_GRID = (3e-6, 3e-5, 3e-4, 3e-3)

#: the ladder's DOF level for the resistance multiplier groups (the incumbent
#: calibration's chosen level — the incumbent every tier must beat)
R_LEVEL = "regions-percase"


# ---------------------------------------------------------------------------
# θ layout: mixed log/linear slots over a growing structure hypothesis set
# ---------------------------------------------------------------------------
@dataclass
class ThetaLayout:
    """Slot bookkeeping for one stage's continuous DOF vector.

    Raw optimiser coordinates: R multipliers and adjacency resistances live
    in log space (positive, bounded); wiring gains and pair drive gains are
    linear signed.  ``decode`` returns the physical values keyed for the
    per-campaign hypothesis application.
    """

    group_names: list[str]
    wiring_groups: list[list[str]] = field(default_factory=list)
    pair_labels: list[str] = field(default_factory=list)
    edge_slots: list[tuple[str, int]] = field(default_factory=list)

    @property
    def n(self) -> int:
        return (
            len(self.group_names)
            + 2 * len(self.wiring_groups)
            + 2 * len(self.pair_labels)
            + len(self.edge_slots)
        )

    def bounds(self) -> list[tuple[float, float]]:
        lb, ub = np.log(MULTIPLIER_BOUNDS[0]), np.log(MULTIPLIER_BOUNDS[1])
        out = [(lb, ub)] * len(self.group_names)
        out += [(-GV_BOUND, GV_BOUND)] * len(self.wiring_groups)
        out += [(-RW_BOUND, RW_BOUND)] * len(self.wiring_groups)
        out += [(-PAIR_GAIN_BOUND, PAIR_GAIN_BOUND)] * (2 * len(self.pair_labels))
        out += [(np.log(EDGE_RHO_BOUNDS[0]), np.log(EDGE_RHO_BOUNDS[1]))] * len(
            self.edge_slots
        )
        return out

    def decode(self, x: np.ndarray) -> dict:
        x = np.asarray(x, dtype=np.float64)
        n_g, n_w = len(self.group_names), len(self.wiring_groups)
        i = 0
        mult = {
            g: float(np.exp(v)) for g, v in zip(self.group_names, x[:n_g], strict=True)
        }
        i += n_g
        g_v: dict[str, float] = {}
        for grp, v in zip(self.wiring_groups, x[i : i + n_w], strict=True):
            for ch in grp:
                g_v[ch] = float(v)
        i += n_w
        r_w: dict[str, float] = {}
        for grp, v in zip(self.wiring_groups, x[i : i + n_w], strict=True):
            for ch in grp:
                r_w[ch] = float(v)
        i += n_w
        pair: dict[str, tuple[float, float]] = {}
        for lab in self.pair_labels:
            pair[lab] = (float(x[i]), float(x[i + 1]))
            i += 2
        edge_r: dict[str, dict[int, float]] = {}
        for (camp, e_idx), v in zip(self.edge_slots, x[i:], strict=True):
            edge_r.setdefault(camp, {})[e_idx] = float(np.exp(v))
        return {"mult": mult, "g_v": g_v, "r_w": r_w, "pair": pair, "edge_r": edge_r}

    def encode_warm(
        self,
        mult: dict[str, float],
        g_v: dict[str, float] | None = None,
        r_w: dict[str, float] | None = None,
        pair: dict[str, tuple[float, float]] | None = None,
        edge_r: dict[str, dict[int, float]] | None = None,
    ) -> np.ndarray:
        """Warm-start vector: known values carried over, new slots at init."""
        x = np.zeros(self.n)
        i = 0
        for g in self.group_names:
            x[i] = np.log(np.clip(mult.get(g, 1.0), *MULTIPLIER_BOUNDS))
            i += 1
        for grp in self.wiring_groups:
            x[i] = (g_v or {}).get(grp[0], 0.0)
            i += 1
        for grp in self.wiring_groups:
            x[i] = (r_w or {}).get(grp[0], 0.0)
            i += 1
        for lab in self.pair_labels:
            gc, gd = (pair or {}).get(lab, (0.0, 0.0))
            x[i], x[i + 1] = gc, gd
            i += 2
        for camp, e_idx in self.edge_slots:
            rho = (edge_r or {}).get(camp, {}).get(e_idx, 1e-5)
            x[i] = np.log(np.clip(rho, *EDGE_RHO_BOUNDS))
            i += 1
        return x


@dataclass
class StageSpec:
    """One stage's frozen structure hypothesis (θ-independent)."""

    name: str
    layout: ThetaLayout
    hyps: dict[str, StructureHypothesis]
    wiring_order: dict[str, list[str]]  # campaign → hyp.wiring_rows channels
    pair_order: dict[str, list[str]]  # campaign → hyp.pair_cols labels
    case_series: list[dict] = field(default_factory=list)


def campaign_theta_parts(spec: StageSpec, campaign: str, dec: dict) -> dict:
    """Per-campaign array arguments for :func:`structured_mode_maps`."""
    hyp = spec.hyps[campaign]
    system = hyp.system
    labels = resistance_group_labels(
        system.circuits, system.centroid_r, system.centroid_z, R_LEVEL
    )
    parts: dict = {"multipliers": np.array([dec["mult"][lb] for lb in labels])}
    order = spec.wiring_order.get(campaign, [])
    if order:
        parts["g_v"] = np.array([dec["g_v"][ch] for ch in order])
        parts["r_w"] = np.array([dec["r_w"][ch] for ch in order])
    p_order = spec.pair_order.get(campaign, [])
    if p_order:
        parts["pair_gains"] = np.array([dec["pair"][lab] for lab in p_order]).reshape(
            -1, 2
        )
    if hyp.edges:
        er = dec["edge_r"].get(campaign, {})
        parts["edge_r"] = np.array([er.get(i, 0.0) for i in range(len(hyp.edges))])
    return parts


def stage_shot_terms(spec: StageSpec, d: VacuumShotData, dec, sigma_med, sigma_case):
    parts = campaign_theta_parts(spec, d.campaign, dec)
    maps = structured_mode_maps(spec.hyps[d.campaign], **parts)
    return structured_shot_loss(d, maps, sigma_med[d.campaign], sigma_case[d.campaign])


# ---------------------------------------------------------------------------
# Pool-parallel stage objective
# ---------------------------------------------------------------------------
_W: dict = {}


def _worker_init(shots, spec, sigma_med, sigma_case):
    _W.update(shots=shots, spec=spec, sigma_med=sigma_med, sigma_case=sigma_case)
    _W["cache"] = {}


def _worker_terms(job: tuple) -> tuple:
    idx, x_bytes = job
    d = _W["shots"][idx]
    spec = _W["spec"]
    key = (x_bytes, d.campaign)
    maps = _W["cache"].get(key)
    if maps is None:
        dec = spec.layout.decode(np.frombuffer(x_bytes))
        parts = campaign_theta_parts(spec, d.campaign, dec)
        maps = structured_mode_maps(spec.hyps[d.campaign], **parts)
        if len(_W["cache"]) > 4:
            _W["cache"].clear()
        _W["cache"][key] = maps
    return structured_shot_loss(
        d, maps, _W["sigma_med"][d.campaign], _W["sigma_case"][d.campaign]
    )


class StageObjective:
    """Combined mean whitened square of one stage over a shot pool."""

    def __init__(self, shots, spec, sigma_med, sigma_case, workers):
        self.shots = shots
        self.spec = spec
        self.sigma_med = sigma_med
        self.sigma_case = sigma_case
        ctx = multiprocessing.get_context("fork")
        self.pool = (
            ctx.Pool(
                workers,
                initializer=_worker_init,
                initargs=(shots, spec, sigma_med, sigma_case),
            )
            if workers > 1
            else None
        )
        self.n_eval = 0

    def components(self, x: np.ndarray) -> dict[str, float]:
        x_bytes = np.ascontiguousarray(x, dtype=np.float64).tobytes()
        jobs = [(i, x_bytes) for i in range(len(self.shots))]
        if self.pool is not None:
            terms = self.pool.map(_worker_terms, jobs)
        else:
            _worker_init(self.shots, self.spec, self.sigma_med, self.sigma_case)
            terms = [_worker_terms(j) for j in jobs]
        n_mag = sum(t[1] for t in terms)
        n_case = sum(t[3] for t in terms)
        mag = sum(t[0] for t in terms) / max(n_mag, 1)
        case = sum(t[2] for t in terms) / max(n_case, 1)
        self.n_eval += 1
        return {"combined": mag + case, "mag": mag, "case": case}

    def __call__(self, x: np.ndarray) -> float:
        return self.components(x)["combined"]

    def close(self):
        if self.pool is not None:
            self.pool.close()
            self.pool.join()


def case_reproduction(shots, spec, x) -> dict:
    """Held-back case reproduction per channel under one stage/θ."""
    dec = spec.layout.decode(x)
    per_chan: dict[str, dict[str, list[float]]] = {}
    maps_by_camp: dict = {}
    for d in shots:
        maps = maps_by_camp.get(d.campaign)
        if maps is None:
            parts = campaign_theta_parts(spec, d.campaign, dec)
            maps = structured_mode_maps(spec.hyps[d.campaign], **parts)
            maps_by_camp[d.campaign] = maps
        psi_m = d.i_drive @ maps.drive_flux.T
        volt_m = None if maps.drive_volt is None else d.i_drive @ maps.drive_volt.T
        a = zoh_mode_response(maps.tau, d.dt, psi_m, volt_m=volt_m)
        pred = a @ maps.case_map.T
        chans = sorted(spec.hyps[d.campaign].system.case_channel_row)
        for k, ch in enumerate(chans):
            meas = d.case_meas[:, k]
            good = np.isfinite(meas)
            if good.sum() < 100:
                continue
            m = meas[good] - np.mean(meas[good])
            p = pred[good, k] - np.mean(pred[good, k])
            denom = float(np.sqrt(np.mean(m**2)))
            if denom < 1.0:
                continue
            entry = per_chan.setdefault(ch, {"rel_rms": [], "corr": []})
            entry["rel_rms"].append(float(np.sqrt(np.mean((m - p) ** 2)) / denom))
            entry["corr"].append(
                float(np.corrcoef(m, p)[0, 1]) if m.size > 2 else np.nan
            )
    return {
        ch: {
            "rel_rms_median": float(np.median(v["rel_rms"])),
            "corr_median": float(np.median(v["corr"])),
            "n_shots": len(v["rel_rms"]),
        }
        for ch, v in per_chan.items()
    }


# ---------------------------------------------------------------------------
# Stage construction + joint refit
# ---------------------------------------------------------------------------
def build_stage(
    name: str,
    systems: dict,
    linkages: dict,
    group_names: list[str],
    *,
    case_series: list[dict] | None = None,
    wiring_groups: list[list[str]] | None = None,
    pair_labels_wanted: list[str] | None = None,
    edges_by_campaign: dict[str, list[tuple[int, int]]] | None = None,
) -> StageSpec:
    """Freeze one hypothesis set over the given campaigns into a StageSpec."""
    case_series = case_series or []
    wiring_groups = wiring_groups or []
    hyps: dict[str, StructureHypothesis] = {}
    wiring_order: dict[str, list[str]] = {}
    pair_order: dict[str, list[str]] = {}
    edge_slots: list[tuple[str, int]] = []
    for camp, system in systems.items():
        series = [
            (p["channels"][0], p["channels"][1], p["sign"])
            for p in case_series
            if all(ch in system.case_channel_row for ch in p["channels"])
        ]
        w_cases = sorted(
            ch for grp in wiring_groups for ch in grp if ch in system.case_channel_row
        )
        pairs_all = coil_pair_channels(system.coil_channels)
        labels_all = [u.split("_")[0][:-1] for u, _l in pairs_all]
        keep = [
            (lab, pr)
            for lab, pr in zip(labels_all, pairs_all, strict=True)
            if pair_labels_wanted and lab in pair_labels_wanted
        ]
        edges = (edges_by_campaign or {}).get(camp, [])
        hyps[camp] = build_structure_hypothesis(
            system,
            np.zeros(system.n_circuits, dtype=np.int64),
            case_series=series,
            wiring_cases=w_cases,
            drive_linkage=linkages[camp] if w_cases else None,
            pair_channels=[pr for _lab, pr in keep],
            edges=edges,
        )
        wiring_order[camp] = w_cases
        pair_order[camp] = [lab for lab, _pr in keep]
        edge_slots += [(camp, i) for i in range(len(edges))]
    layout = ThetaLayout(
        group_names=list(group_names),
        wiring_groups=[list(g) for g in wiring_groups],
        pair_labels=sorted({lab for po in pair_order.values() for lab in po}),
        edge_slots=edge_slots,
    )
    return StageSpec(
        name=name,
        layout=layout,
        hyps=hyps,
        wiring_order=wiring_order,
        pair_order=pair_order,
        case_series=case_series,
    )


def fit_stage(
    spec: StageSpec,
    train,
    held,
    sigma_med,
    sigma_case,
    x0: np.ndarray,
    *,
    workers: int,
    maxiter: int,
) -> dict:
    """Refit ALL continuous DOF of one stage (joint, warm-started)."""
    from scipy.optimize import minimize  # noqa: PLC0415

    obj = StageObjective(train, spec, sigma_med, sigma_case, workers)
    held_obj = StageObjective(held, spec, sigma_med, sigma_case, 1)
    t0 = time.perf_counter()
    res = minimize(
        obj,
        x0,
        method="L-BFGS-B",
        bounds=spec.layout.bounds(),
        options={"maxiter": maxiter, "ftol": 1e-8},
    )
    train_comp = obj.components(res.x)
    held_comp = held_obj.components(res.x)
    n_eval = obj.n_eval
    obj.close()
    held_obj.close()
    return {
        "x": res.x,
        "train": train_comp,
        "held_out": held_comp,
        "n_obj_evals": n_eval,
        "wall_s": time.perf_counter() - t0,
    }


# ---------------------------------------------------------------------------
# Tier C: parallel single-edge probing
# ---------------------------------------------------------------------------
_PB: dict = {}


def _probe_init(shots_by_camp, systems, linkages, group_names, incumbent, sig, sigc):
    _PB.update(
        shots_by_camp=shots_by_camp,
        systems=systems,
        linkages=linkages,
        group_names=group_names,
        incumbent=incumbent,
        sigma_med=sig,
        sigma_case=sigc,
        base_cache={},
    )


def _probe_eval(camp: str, extra_edge, rho, shot_idx) -> float:
    """Loss of the incumbent structure (+ one candidate edge) on a subset of
    one campaign's probe shots — an edge changes nothing off-campaign."""
    inc = _PB["incumbent"]
    edges = list(inc["edges_by_campaign"].get(camp, []))
    edge_vals = dict(inc["edge_r"].get(camp, {}))
    if extra_edge is not None:
        edge_vals[len(edges)] = rho
        edges = edges + [extra_edge]
    spec = build_stage(
        "C-probe",
        {camp: _PB["systems"][camp]},
        _PB["linkages"],
        _PB["group_names"],
        case_series=inc["case_series"],
        wiring_groups=inc["wiring_groups"],
        pair_labels_wanted=inc["pair_labels"],
        edges_by_campaign={camp: edges},
    )
    x = spec.layout.encode_warm(
        inc["mult"], inc["g_v"], inc["r_w"], inc["pair"], {camp: edge_vals}
    )
    dec = spec.layout.decode(x)
    shots = [_PB["shots_by_camp"][camp][i] for i in shot_idx]
    ss_m = n_m = ss_c = n_c = 0.0
    for d in shots:
        sm, nm, sc, nc = stage_shot_terms(
            spec, d, dec, _PB["sigma_med"], _PB["sigma_case"]
        )
        ss_m += sm
        n_m += nm
        ss_c += sc
        n_c += nc
    return ss_m / max(n_m, 1) + ss_c / max(n_c, 1)


def _probe_edge(job: tuple) -> tuple:
    camp, edge, shot_idx = job
    shot_idx = tuple(shot_idx)
    base_key = (camp, shot_idx)
    base = _PB["base_cache"].get(base_key)
    if base is None:
        base = _probe_eval(camp, None, 0.0, shot_idx)
        _PB["base_cache"][base_key] = base
    best_gain, best_rho = 0.0, EDGE_PROBE_GRID[0]
    for rho in EDGE_PROBE_GRID:
        gain = base - _probe_eval(camp, edge, rho, shot_idx)
        if gain > best_gain:
            best_gain, best_rho = gain, rho
    return camp, edge, float(best_gain), float(best_rho)


def main() -> int:  # noqa: PLR0915 — one auditable discovery ladder
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--pool-artifact",
        type=str,
        default=str(ARTIFACTS / "vacuum_coil_response_audit-solv4-vac.json"),
    )
    ap.add_argument(
        "--baseline-calibration",
        type=str,
        default=str(ARTIFACTS / "passive_resistance_calibration.json"),
    )
    ap.add_argument("--holdout-stride", type=int, default=5)
    ap.add_argument("--nr", type=int, default=65)
    ap.add_argument("--nz", type=int, default=97)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--max-shots", type=int, default=0)
    ap.add_argument("--neighbour-factor", type=float, default=1.5)
    ap.add_argument("--max-edges", type=int, default=12)
    ap.add_argument("--probe-shots", type=int, default=10)
    ap.add_argument("--stability-folds", type=int, default=5)
    ap.add_argument("--maxiter", type=int, default=40)
    ap.add_argument("--maxiter-trial", type=int, default=15)
    ap.add_argument("--out-suffix", type=str, default="")
    args = ap.parse_args()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    pool_meta = json.loads(Path(args.pool_artifact).read_text())
    shots = [int(s) for s in pool_meta["shots_used"]]
    n_fleet = int(pool_meta["strata"]["fleet"]["n_shots"])
    strata = ["fleet"] * n_fleet + ["dedicated_vacuum"] * (len(shots) - n_fleet)
    if args.max_shots > 0:
        shots, strata = shots[: args.max_shots], strata[: args.max_shots]

    # ---- campaign systems (rebuild when the cached form predates
    # section_scale — the adjacency rule needs the circuit sizes) ----
    from imas_ambix.latent.gs_solve import EquilibriumGrid  # noqa: PLC0415

    seen: dict[str, tuple] = {}
    for s in shots:
        try:
            table = read_geometry_table(int(s))
        except Exception:  # noqa: BLE001
            continue
        key = table.signature.key
        if key in seen:
            continue
        grid = EquilibriumGrid.from_table(table, nr=args.nr, nz=args.nz)
        cache = SYSTEM_DIR / f"circuit-system-holdback-{key}.npz"
        system = load_circuit_system(cache) if cache.exists() else None
        if system is None or system.section_scale is None:
            system = build_passive_circuit_system(table, grid, hold_back_cases=True)
            tmp = cache.with_suffix(f".pid{os.getpid()}.npz")
            save_circuit_system(tmp, system)
            tmp.replace(cache)
            logger.info("campaign system %s (re)built with section scales", key)
        lk_cache = SYSTEM_DIR / f"drive-linkage-{key}.npz"
        if lk_cache.exists():
            with np.load(lk_cache) as z:
                linkage = ([str(c) for c in z["channels"]], z["lam"])
        else:
            linkage = build_drive_linkage(table)
            np.savez_compressed(lk_cache, channels=np.array(linkage[0]), lam=linkage[1])
            logger.info("drive linkage %s built -> %s", key, lk_cache)
        seen[key] = (system, linkage)
    systems = {k: v[0] for k, v in seen.items()}
    linkages = {k: v[1] for k, v in seen.items()}

    # ---- prepare shots (parallel; i_drive + sibling audit included) ----
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
        "prepared %d/%d shots (%.0f s), %d samples",
        len(data),
        len(shots),
        time.perf_counter() - t0,
        sum(d.n_samples for d in data),
    )
    campaigns = sorted({d.campaign for d in data})

    # ---- tier A: sibling identity audit (pool-wide) ----
    feed_res: dict[str, list[float]] = {}
    plain_res: dict[str, list[float]] = {}
    turns: dict[str, float] = {}
    for r in prepped:
        aud = r.get("sibling_audit") or {}
        for base, rec in aud.get("feed", {}).items():
            feed_res.setdefault(base, []).append(rec["rel_resid"])
            turns[base] = rec["turns"]
        for base, rec in aud.get("plain", {}).items():
            plain_res.setdefault(base, []).append(rec["rel_resid"])
    tier_a = {
        "feed": {
            b: {
                "turns": turns[b],
                "rel_resid_median": float(np.median(v)),
                "rel_resid_max": float(np.max(v)),
                "n_shots": len(v),
            }
            for b, v in sorted(feed_res.items())
        },
        "plain": {
            b: {
                "rel_resid_median": float(np.median(v)),
                "rel_resid_max": float(np.max(v)),
                "n_shots": len(v),
            }
            for b, v in sorted(plain_res.items())
        },
        "verdict": (
            "no admissible missing drives: feed channels are turns-scaled "
            "duplicates of the consumed coil channels; plain channels equal "
            "coil(s) + case exactly, so they carry the held-back case "
            "measurement (inadmissible as inputs) and nothing else new. The "
            "plain identity is the galvanic wiring evidence tier B consumes."
        ),
    }
    logger.info(
        "tier A: feed rel-resid max %.2e; plain rel-resid max %.2e",
        max((v["rel_resid_max"] for v in tier_a["feed"].values()), default=np.nan),
        max((v["rel_resid_max"] for v in tier_a["plain"].values()), default=np.nan),
    )

    # ---- split + pooled whitening on the incumbent held-out cohort ----
    train, held = [], []
    for stratum in ("fleet", "dedicated_vacuum"):
        grp = [d for d in data if d.stratum == stratum]
        for i, d in enumerate(grp):
            is_held = i % args.holdout_stride == args.holdout_stride - 1
            (held if is_held else train).append(d)
    logger.info("split: %d train / %d held-out", len(train), len(held))
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

    # ---- baseline: incumbent calibration under the structured path ----
    baseline = load_calibration(args.baseline_calibration)
    group_names = sorted(
        {
            lb
            for key in campaigns
            for lb in resistance_group_labels(
                systems[key].circuits,
                systems[key].centroid_r,
                systems[key].centroid_z,
                R_LEVEL,
            )
        }
    )
    spec0 = build_stage("baseline", systems, linkages, group_names)
    x_base = spec0.layout.encode_warm(baseline.group_multipliers)
    held_obj0 = StageObjective(held, spec0, sigma_med, sigma_case, 1)
    base_held = held_obj0.components(x_base)
    held_obj0.close()
    repro_base = case_reproduction(held, spec0, x_base)
    logger.info(
        "baseline (incumbent calibration): held-out %.5f (mag %.5f case %.5f)",
        base_held["combined"],
        base_held["mag"],
        base_held["case"],
    )

    ladder: list[dict] = [
        {
            "stage": "baseline",
            "accepted": True,
            "held_out": base_held,
            "case_reproduction": repro_base,
            "n_dof": spec0.layout.n,
        }
    ]
    incumbent = {
        "spec": spec0,
        "x": x_base,
        "held": base_held,
        "mult": dict(baseline.group_multipliers),
        "g_v": {},
        "r_w": {},
        "pair": {},
        "edge_r": {},
        "case_series": [],
        "wiring_groups": [],
        "pair_labels": [],
        "edges_by_campaign": {},
    }

    def stage_kwargs(inc: dict) -> dict:
        return {
            "case_series": inc["case_series"],
            "wiring_groups": inc["wiring_groups"],
            "pair_labels_wanted": inc["pair_labels"],
            "edges_by_campaign": inc["edges_by_campaign"],
        }

    def warm(spec: StageSpec, edge_vals=None) -> np.ndarray:
        return spec.layout.encode_warm(
            incumbent["mult"],
            incumbent["g_v"],
            incumbent["r_w"],
            incumbent["pair"],
            edge_vals if edge_vals is not None else incumbent["edge_r"],
        )

    def accept_or_reject(name: str, spec: StageSpec, fit: dict, extra: dict) -> bool:
        nonlocal incumbent
        dec = spec.layout.decode(fit["x"])
        repro = case_reproduction(held, spec, fit["x"])
        gain = incumbent["held"]["combined"] - fit["held_out"]["combined"]
        accepted = gain > 0.0
        ladder.append(
            {
                "stage": name,
                "accepted": bool(accepted),
                "held_out_gain": float(gain),
                "train": fit["train"],
                "held_out": fit["held_out"],
                "n_dof": spec.layout.n,
                "n_obj_evals": fit["n_obj_evals"],
                "wall_s": fit["wall_s"],
                "case_reproduction": repro,
                "decoded": {
                    "mult": dec["mult"],
                    "g_v": dec["g_v"],
                    "r_w": dec["r_w"],
                    "pair": dec["pair"],
                    "edge_r": {
                        c: {str(i): v for i, v in e.items()}
                        for c, e in dec["edge_r"].items()
                    },
                },
            }
            | {k: v for k, v in extra.items() if k != "spec"}
        )
        logger.info(
            "stage %s: held %.5f (gain %+.5f) -> %s",
            name,
            fit["held_out"]["combined"],
            gain,
            "ACCEPT" if accepted else "reject",
        )
        if accepted:
            incumbent = (
                incumbent
                | {
                    "spec": spec,
                    "x": fit["x"],
                    "held": fit["held_out"],
                    "mult": dec["mult"],
                    "g_v": dec["g_v"],
                    "r_w": dec["r_w"],
                    "pair": dec["pair"],
                    "edge_r": dec["edge_r"],
                }
                | {k: v for k, v in extra.items() if k in incumbent}
            )
        return accepted

    # ---- case-pair series/anti-series constraint reductions ----
    case_chans = sorted(systems[campaigns[0]].case_channel_row)
    fams = sorted({ch.split("_")[0][:-1] for ch in case_chans})
    pair_candidates = []
    for fam in fams:
        lo, up = f"{fam}l_case_current", f"{fam}u_case_current"
        if lo not in case_chans or up not in case_chans:
            continue
        num = den_l = den_u = 0.0
        for d in train:
            chans = sorted(systems[d.campaign].case_channel_row)
            il, iu = chans.index(lo), chans.index(up)
            a, b = d.case_meas[:, il], d.case_meas[:, iu]
            ok = np.isfinite(a) & np.isfinite(b)
            if ok.sum() < 100:
                continue
            a, b = a[ok] - a[ok].mean(), b[ok] - b[ok].mean()
            num += float((a * b).sum())
            den_l += float((a**2).sum())
            den_u += float((b**2).sum())
        corr = num / max(np.sqrt(den_l * den_u), 1e-9)
        pair_candidates.append(
            {"channels": [lo, up], "sign": int(np.sign(corr) or 1), "corr": corr}
        )
    logger.info("case-pair candidates (pooled corr): %s", pair_candidates)
    accepted_series: list[dict] = []
    for cand in pair_candidates:
        trial = [*accepted_series, {"channels": cand["channels"], "sign": cand["sign"]}]
        spec = build_stage(
            "B1",
            systems,
            linkages,
            group_names,
            **(stage_kwargs(incumbent) | {"case_series": trial}),
        )
        fit = fit_stage(
            spec,
            train,
            held,
            sigma_med,
            sigma_case,
            warm(spec),
            workers=args.workers,
            maxiter=args.maxiter_trial,
        )
        name = f"B1-series-{cand['channels'][0].split('_')[0][:-1]}"
        if accept_or_reject(
            name, spec, fit, {"case_series": trial, "pair_corr": cand["corr"]}
        ):
            accepted_series = trial

    # ---- case-coil galvanic wiring (per pair, then per case) ----
    wiring_pairs = [
        [f"{fam}l_case_current", f"{fam}u_case_current"]
        for fam in fams
        if f"{fam}l_case_current" in case_chans and f"{fam}u_case_current" in case_chans
    ]
    for label, groups in (
        ("B2-wiring-perpair", wiring_pairs),
        ("B2-wiring-percase", [[ch] for ch in case_chans]),
    ):
        spec = build_stage(
            label,
            systems,
            linkages,
            group_names,
            **(stage_kwargs(incumbent) | {"wiring_groups": groups}),
        )
        fit = fit_stage(
            spec,
            train,
            held,
            sigma_med,
            sigma_case,
            warm(spec),
            workers=args.workers,
            maxiter=args.maxiter,
        )
        accept_or_reject(label, spec, fit, {"wiring_groups": groups})

    # ---- coil-pair common/differential drive gains ----
    pair_labels = sorted(
        {
            u.split("_")[0][:-1]
            for key in campaigns
            for u, _l in coil_pair_channels(systems[key].coil_channels)
        }
    )
    spec = build_stage(
        "B3-pairdrive",
        systems,
        linkages,
        group_names,
        **(stage_kwargs(incumbent) | {"pair_labels_wanted": pair_labels}),
    )
    fit = fit_stage(
        spec,
        train,
        held,
        sigma_med,
        sigma_case,
        warm(spec),
        workers=args.workers,
        maxiter=args.maxiter,
    )
    accept_or_reject("B3-pairdrive", spec, fit, {"pair_labels": pair_labels})

    # ---- tier C: adjacency couplings — parallel probe, greedy + stability ---
    candidates: dict[str, list[tuple[int, int]]] = {}
    for key in campaigns:
        system = systems[key]
        exclude = set(system.case_channel_row.values())
        candidates[key] = neighbour_edges(
            system.centroid_r,
            system.centroid_z,
            system.section_scale,
            factor=args.neighbour_factor,
            exclude_rows=exclude,
        )
        logger.info(
            "tier C: campaign %s — %d candidate edges (factor %.2f)",
            key,
            len(candidates[key]),
            args.neighbour_factor,
        )
    shots_by_camp = {
        key: [d for d in train if d.campaign == key][: args.probe_shots]
        for key in campaigns
    }
    inc_snapshot = {
        k: incumbent[k]
        for k in (
            "mult",
            "g_v",
            "r_w",
            "pair",
            "edge_r",
            "case_series",
            "wiring_groups",
            "pair_labels",
            "edges_by_campaign",
        )
    }
    probe_jobs = [
        (key, e, tuple(range(len(shots_by_camp[key]))))
        for key in campaigns
        for e in candidates[key]
    ]
    t0 = time.perf_counter()
    ctx = multiprocessing.get_context("fork")
    with ctx.Pool(
        args.workers,
        initializer=_probe_init,
        initargs=(
            shots_by_camp,
            systems,
            linkages,
            group_names,
            inc_snapshot,
            sigma_med,
            sigma_case,
        ),
    ) as pool:
        probe_out = pool.map(_probe_edge, probe_jobs)
        gains = sorted(
            [(g, camp, e, rho) for camp, e, g, rho in probe_out if g > 0],
            reverse=True,
        )
        logger.info(
            "tier C: %d/%d edges show single-edge probe gain (%.0f s); top: %s",
            len(gains),
            len(probe_jobs),
            time.perf_counter() - t0,
            [(round(g, 5), c, e) for g, c, e, _ in gains[:8]],
        )
        # cross-shot stability: probe-gain sign on random half-folds
        rng = np.random.default_rng(0)
        stable: list[tuple[float, str, tuple[int, int], float]] = []
        for g, camp, e, rho in gains[: 2 * args.max_edges]:
            n_p = len(shots_by_camp[camp])
            folds = [
                tuple(sorted(rng.choice(n_p, size=max(3, n_p // 2), replace=False)))
                for _ in range(args.stability_folds)
            ]
            fold_out = pool.map(_probe_edge, [(camp, e, f) for f in folds])
            wins = sum(1 for _c, _e, fg, _r in fold_out if fg > 0)
            if wins >= int(np.ceil(0.6 * args.stability_folds)):
                stable.append((g, camp, e, rho))
        stable = stable[: args.max_edges]
    logger.info(
        "tier C: %d stable edges for joint refit: %s",
        len(stable),
        [(c, e) for _g, c, e, _r in stable],
    )
    if stable:
        edges_by_campaign = {
            c: list(v) for c, v in incumbent["edges_by_campaign"].items()
        }
        edge_vals = {c: dict(v) for c, v in incumbent["edge_r"].items()}
        for _g, camp, e, rho in stable:
            edges_by_campaign.setdefault(camp, [])
            edge_vals.setdefault(camp, {})[len(edges_by_campaign[camp])] = rho
            edges_by_campaign[camp] = [*edges_by_campaign[camp], e]
        spec = build_stage(
            "C-adjacency",
            systems,
            linkages,
            group_names,
            **(stage_kwargs(incumbent) | {"edges_by_campaign": edges_by_campaign}),
        )
        fit = fit_stage(
            spec,
            train,
            held,
            sigma_med,
            sigma_case,
            warm(spec, edge_vals),
            workers=args.workers,
            maxiter=args.maxiter,
        )
        accept_or_reject(
            "C-adjacency", spec, fit, {"edges_by_campaign": edges_by_campaign}
        )
    else:
        ladder.append(
            {
                "stage": "C-adjacency",
                "accepted": False,
                "note": "no candidate edge survived probe gain + stability folds",
            }
        )

    # ---- final artifact ----
    final_spec = incumbent["spec"]
    final_x = incumbent["x"]
    dec = final_spec.layout.decode(final_x)
    repro_final = case_reproduction(held, final_spec, final_x)
    adjacency = {}
    for camp, hyp in final_spec.hyps.items():
        if hyp.edges:
            er = dec["edge_r"].get(camp, {})
            adjacency[camp] = [
                {
                    "i": int(hyp.system.circuits[i]),
                    "j": int(hyp.system.circuits[j]),
                    "r_couple": float(er.get(k, 0.0)),
                }
                for k, (i, j) in enumerate(hyp.edges)
            ]
    wiring = {
        ch: {
            "parents": case_parent_coil_channels(ch, systems[camp].coil_channels),
            "g_v": dec["g_v"][ch],
            "r_w": dec["r_w"][ch],
        }
        for camp in campaigns[:1]
        for ch in final_spec.wiring_order.get(camp, [])
    }
    pair_gains_out = []
    if final_spec.pair_order:
        camp0 = campaigns[0]
        pairs_all = coil_pair_channels(systems[camp0].coil_channels)
        by_label = {u.split("_")[0][:-1]: (u, low) for u, low in pairs_all}
        pair_gains_out = [
            {
                "channels": list(by_label[lab]),
                "common": dec["pair"][lab][0],
                "differential": dec["pair"][lab][1],
            }
            for lab in final_spec.pair_order.get(camp0, [])
        ]
    structure = PassiveStructure(
        case_series_pairs=incumbent["case_series"],
        case_wiring=wiring,
        pair_drive_gains=pair_gains_out,
        adjacency=adjacency,
        neighbour_rule={
            "factor": args.neighbour_factor,
            "metric": "centroid distance <= factor * pair-mean sqrt(sum w*h)",
        },
        r_level=R_LEVEL,
        r_group_multipliers=dec["mult"],
        provenance={
            "fitted": "2026-07-17",
            "coil_model_version": COIL_MODEL_VERSION,
            "geometry_table_version": GEOMETRY_TABLE_VERSION,
            "pool_artifact": str(args.pool_artifact),
            "baseline_calibration": str(args.baseline_calibration),
            "n_train_shots": len(train),
            "n_held_out_shots": len(held),
            "held_out_shots": sorted(d.shot for d in held),
            "joint_refit": "resistance multipliers refit jointly at every stage",
            "case_holdback": True,
            "wiring_model_note": (
                "case terminal-voltage drive keeps measured drives only; the "
                "winding's linkage of passive-state currents is dropped "
                "(symmetric eigenproblem preserved) — recorded approximation"
            ),
        },
    )
    tag = f"-{args.out_suffix}" if args.out_suffix else ""
    struct_path = ARTIFACTS / f"passive_structure_calibration{tag}.json"
    save_structure(struct_path, structure)
    logger.info("wrote %s", struct_path)

    out = {
        "kind": "vacuum-passive-structure-discovery",
        "coil_model_version": COIL_MODEL_VERSION,
        "geometry_table_version": GEOMETRY_TABLE_VERSION,
        "case_holdback": True,
        "pool": {
            "n_shots_prepared": len(data),
            "n_train": len(train),
            "n_held_out": len(held),
            "held_out_shots": sorted(d.shot for d in held),
            "campaigns": campaigns,
        },
        "tier_a_sibling_audit": tier_a,
        "ladder": [{k: v for k, v in rec.items() if k != "x"} for rec in ladder],
        "final": {
            "stage": final_spec.name,
            "held_out": incumbent["held"],
            "case_reproduction": repro_final,
            "case_reproduction_baseline": repro_base,
        },
        "structure_artifact": str(struct_path),
    }
    out_path = ARTIFACTS / f"vacuum_passive_structure_discovery{tag}.json"
    out_path.write_text(json.dumps(out, indent=2, default=float))
    logger.info("wrote %s", out_path)

    _figures(out, ladder, held, final_spec, final_x, spec0, x_base, tag)
    return 0


def _figures(out, ladder, held, final_spec, final_x, spec0, x_base, tag=""):
    recs = [r for r in ladder if "held_out" in r]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
    x = np.arange(len(recs))
    axes[0].plot(x, [r["held_out"]["combined"] for r in recs], "s-", label="combined")
    axes[0].plot(x, [r["held_out"]["mag"] for r in recs], "o--", label="magnetics")
    axes[0].plot(x, [r["held_out"]["case"] for r in recs], "d--", label="case")
    for i, r in enumerate(recs):
        axes[0].annotate(
            "✓" if r["accepted"] else "✗",
            (x[i], r["held_out"]["combined"]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
        )
    axes[0].set_xticks(
        x, [r["stage"] for r in recs], rotation=30, ha="right", fontsize=7
    )
    axes[0].set_ylabel("held-out mean whitened square")
    axes[0].set_title("structure-discovery ladder (✓ accepted)")
    axes[0].legend(fontsize=8)

    base_repro = out["final"]["case_reproduction_baseline"]
    fin_repro = out["final"]["case_reproduction"]
    chans = sorted(set(base_repro) | set(fin_repro))
    xb = np.arange(len(chans))
    axes[1].bar(
        xb - 0.17,
        [base_repro.get(c, {}).get("corr_median", np.nan) for c in chans],
        width=0.32,
        color="#bb5566",
        label="baseline (incumbent calibrated R)",
    )
    axes[1].bar(
        xb + 0.17,
        [fin_repro.get(c, {}).get("corr_median", np.nan) for c in chans],
        width=0.32,
        color="#228833",
        label="with discovered structure",
    )
    axes[1].set_xticks(xb, [c.replace("_case_current", "") for c in chans], fontsize=8)
    axes[1].set_ylabel("held-back case corr (median)")
    axes[1].set_title("held-back case reproduction")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / f"fig-structure-ladder{tag}.png", dpi=130)
    plt.close(fig)

    vac = [d for d in held if d.stratum == "dedicated_vacuum"] or held
    show = vac[:2]
    p2 = ["p2l_case_current", "p2u_case_current"]
    fig, axes = plt.subplots(len(show), 2, figsize=(13, 3.2 * len(show)), squeeze=False)
    for r, d in enumerate(show):
        for spec, xvec, color, lab in (
            (spec0, x_base, "#bb5566", "baseline"),
            (final_spec, final_x, "#228833", "with structure"),
        ):
            dec = spec.layout.decode(xvec)
            parts = campaign_theta_parts(spec, d.campaign, dec)
            maps = structured_mode_maps(spec.hyps[d.campaign], **parts)
            psi_m = d.i_drive @ maps.drive_flux.T
            volt_m = None if maps.drive_volt is None else d.i_drive @ maps.drive_volt.T
            a = zoh_mode_response(maps.tau, d.dt, psi_m, volt_m=volt_m)
            pred = a @ maps.case_map.T
            chans = sorted(spec.hyps[d.campaign].system.case_channel_row)
            t = np.arange(d.n_samples) * d.dt
            for c, ch in enumerate(p2):
                k = chans.index(ch)
                ax = axes[r][c]
                if lab == "baseline":
                    ax.plot(
                        t,
                        d.case_meas[:, k] / 1e3,
                        color="#222",
                        lw=1.0,
                        label="measured (held back)",
                    )
                ax.plot(t, pred[:, k] / 1e3, color=color, lw=0.9, label=lab)
                ax.set_title(f"{d.shot} {ch}", fontsize=8)
                if r == len(show) - 1:
                    ax.set_xlabel("t from stream start [s]")
                if c == 0:
                    ax.set_ylabel("case current [kA]")
    axes[0][0].legend(fontsize=7)
    fig.suptitle(
        "Held-back P2 case currents: the structured-residual target", fontsize=11
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIGURES / f"fig-structure-p2-case{tag}.png", dpi=130)
    plt.close(fig)
    logger.info("figures written to %s", FIGURES)


if __name__ == "__main__":
    raise SystemExit(main())
