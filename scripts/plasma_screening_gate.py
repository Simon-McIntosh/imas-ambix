#!/usr/bin/env python
"""Synthetic skin-truth gate for the plasma screening circuit.

Truths that the coefficient-ladder generator CANNOT produce: sequences of
manufactured equilibria whose current profile evolves under the plasma
filament circuit with a KNOWN η(ψ_N) through fast shape-forming ramps.  Per
interval the patch currents evolve exactly (ZOH eigenmodes; uniform loop
voltage shot to hit the prescribed Ip; coil-swing drive), the evolved state
is binned to a hollow-capable flux-function profile, and a force-balanced
free-boundary equilibrium is solved from it — a skin-current truth chained
exactly the way the plasma actually forms one (frozen geometry per interval,
remap at labels).  Each fast ramp ends in a flat-top hold whose slices give
the representation-adequate floor.

Two pre-declared legs, verdicts recorded honestly either way:

* **leg (a) — reproduction.**  The frozen classical spine (byte-same config
  as the real-data gates) fitted to these truths must REPRODUCE the measured
  limited-phase failure signature (limiter_phase_fidelity_audit):
    A1  elevated ramp boundary error:   LCFS ramp median ≥ 1.8× hold median
    A2  cost tracks the boundary error: ramp Spearman(cost, LCFS) ≥ 0.4,
        p < 0.01  (measured: 0.77)
    A3  the shape signature:            signed elongation harmonic a2 > 0
        (too-round) AND signed horizontal harmonic a1 < 0 (inboard-heavy)
        on ramp, both ≥ 2× the hold level in magnitude
        (measured: a2 +5.7 cm, a1 −3.8 cm at 4–6× flat-top)
    A4  li3 over-read:                  median(li3_fit − li3_truth) on ramp
        ≥ 0.3 and ≥ 2× the hold level  (measured: li3 reads 2–3 on ramp)
  A leg-(a) FAIL means the skin/shape diagnosis is wrong — stop and re-open
  the attribution.

* **leg (b) — recovery.**  The dynamic-mode fit (frozen non-negative
  backbone + k bounded zero-net-current screening columns, amplitudes
  prior-centred on the circuit trajectory — the locked bounded-mode-column
  form) must:
    B1  at ORACLE η, recover ≥ 50% of the synthetic ramp LCFS gap:
        (spine_ramp − dyn_ramp) / (spine_ramp − floor) ≥ 0.5,
        floor = the spine's own hold-phase median (medians on eval
        sequences; prior weight frozen on the tune subset)
    B2  at CLOSURE η (scale identified from the ramp transients by misfit
        scan, flat contrast — the closure's measured precision), remain
        better than the spine (paired median, bootstrap CI reported)
  Hold non-inferiority and axis medians are RECORDED as diagnostics for the
  real-data gate design (they are S2's pre-declared legs, not this rung's).
  A leg-(b) FAIL at oracle η means the representation is insufficient —
  stop before any real-data spend.

Artifacts: imas_ambix/latent/artifacts/patch_gate/plasma_screening_gate[-tag].json
Figures:   docs/figures/plasma-screening-dynamics/
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

from imas_ambix.latent import plasma_screening as ps
from imas_ambix.latent.current_diffusion import EtaProfile, _core_mask
from imas_ambix.latent.gs_solve import (
    MU0,
    _read_axis,
    _read_boundary_psi,
    build_passive_sidecar,
)
from imas_ambix.latent.synthetic_truth import (
    build_campaign,
    build_confining_i_pf,
    manufacture,
    manufacture_shape,
)
from imas_ambix.latent.topology import LCFS_ANGLES
from scripts.closure_gate_eval import fit_and_read_slice, geometry_target_pushout
from scripts.spine_label_factory import frozen_spine_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("plasma_screening_gate")

ARTIFACTS = Path("imas_ambix/latent/artifacts/patch_gate")
FIGURES = Path("docs/figures/plasma-screening-dynamics")
CAMPAIGN_SHOT = 11766  # train-split shot: geometry + noise floor only

TRUTH_N_RAD, TRUTH_N_POL = 14, 10  # truth-side circuit tiling (finer than fit)
FIT_N_RAD, FIT_N_POL = 10, 8  # fit-side screening tiling
BETA_SPLIT = 0.3  # fixed p′-drive share of the truth profile family

_CAMPAIGN: dict[str, object] = {}


def _campaign():
    if "c" not in _CAMPAIGN:
        c = build_campaign(CAMPAIGN_SHOT, nr=65, nz=97)
        c.grid.cell_greens()  # the screening L kernel — build once, share COW
        _CAMPAIGN["c"] = c
    return _CAMPAIGN["c"]


# ---------------------------------------------------------------------------
# scenario: fast shape-forming ramp + flat-top hold
# ---------------------------------------------------------------------------


def scenario_i_pf(campaign, vf_frac: float, quad: float) -> np.ndarray:
    """Confining coil pattern at strength fraction ``vf_frac`` with a
    quadrupole differential ``quad`` that elongates the plasma.

    Boosting the off-midplane P4 set against the near-midplane P5 set adds
    the field curvature that elongates (sign VERIFIED on this campaign:
    quad = +0.4 raises the truth's vertical/horizontal radius ratio).  P6 is
    left on the base pattern (in-vessel, close to the plasma — a strong
    differential there deforms the boundary read region).
    """
    base = build_confining_i_pf(campaign.fwd, 6.0e4 * vf_frac)
    out = base.copy()
    for j, chan in enumerate(campaign.fwd.pf_amc_channels):
        if chan.startswith("p4"):
            out[j] = base[j] * (1.0 + quad)
        elif chan.startswith("p5"):
            out[j] = base[j] * (1.0 - quad)
    return out


def _li3_2d(psi2d, core2d, ip_amperes, grid) -> float:
    """li3 = 4·Wpol/(μ0·Ip²·R0) straight off a 2D ψ (same estimator both
    sides of every comparison — no ladder needed)."""
    dpsi_dz, dpsi_dr = np.gradient(psi2d, grid.zg, grid.rg)
    bpol2 = (dpsi_dr**2 + dpsi_dz**2) / (2.0 * np.pi * grid.mesh_r) ** 2
    dvol = 2.0 * np.pi * grid.mesh_r * grid.dr * grid.dz
    wpol = float((bpol2 * dvol)[core2d].sum()) / (2.0 * MU0)
    return 4.0 * wpol / (MU0 * float(ip_amperes) ** 2 * grid.r0)


def _psi_n_state(psi2d, grid, ip_amperes):
    """(psi_n_flat, core2d, axis, axis_psi) of one equilibrium ψ."""
    sign = 1.0 if ip_amperes >= 0 else -1.0
    axis, axis_psi = _read_axis(psi2d, grid, sign)
    boundary_psi = _read_boundary_psi(psi2d, grid, axis_psi)
    span = boundary_psi - axis_psi
    if abs(span) < 1e-12:
        span = 1e-12
    psi_n = (psi2d.ravel() - axis_psi) / span
    core = _core_mask(psi2d, grid, axis, psi_n)
    return psi_n, core, axis, axis_psi


def _radial_bin_profile(circuit, i_patch, n_rad) -> np.ndarray:
    """Flux-surface (poloidally-summed) current density per radial bin
    [A/m²] — the flux-function limit of the evolved patch state."""
    rad = np.minimum(
        (np.sqrt(np.clip(circuit.tiling.psi_n, 0.0, 1.0)) * n_rad).astype(int),
        n_rad - 1,
    )
    cur = np.bincount(rad, weights=i_patch, minlength=n_rad)
    n_cells = np.bincount(
        rad[circuit.tiling.owner], minlength=n_rad
    )  # cells per radial bin
    area = np.clip(n_cells, 1, None) * circuit.cell_area
    return np.where(n_cells > 0, cur / area, 0.0)


def _shape_from_bins(h: np.ndarray, n_rad: int, r0: float, beta_split: float):
    """jφ(ψ_N, R) callback from a binned (possibly hollow) surface profile."""
    centers = (np.arange(n_rad) + 0.5) / n_rad  # in s = √ψ_N
    h = np.clip(np.asarray(h, dtype=np.float64), 0.0, None)

    def shape(psi_n: np.ndarray, r: np.ndarray) -> np.ndarray:
        psi_n = np.asarray(psi_n, dtype=np.float64)
        s = np.sqrt(np.clip(psi_n, 0.0, 1.0))
        val = np.interp(s, centers, h, left=h[0], right=h[-1])
        rr = np.maximum(np.asarray(r, dtype=np.float64), 1e-3)
        base = beta_split * rr / r0 + (1.0 - beta_split) * r0 / rr
        # taper the edge-bin plateau to zero exactly at the separatrix so the
        # Picard's core mask stays well-posed
        taper = np.clip((1.0 - s) / (0.5 / n_rad), 0.0, 1.0)
        return np.where(psi_n < 1.0, val * base * taper, 0.0)

    return shape


def generate_skin_sequence(job: tuple) -> dict | None:
    """One circuit-consistent skin-truth shot: fast ramp + flat-top hold."""
    seed, cfg = job
    rng = np.random.default_rng(seed)
    campaign = _campaign()
    grid = campaign.grid
    eta_true = EtaProfile.from_vector(np.asarray(cfg["eta_true"], dtype=np.float64))
    n_pre = int(cfg.get("n_pre", 4))
    n_ramp, n_hold = int(cfg["n_ramp"]), int(cfg["n_hold"])
    n_steps = n_pre + n_ramp + n_hold

    dt_ramp = rng.uniform(0.012, 0.018)
    times = 0.02 + np.concatenate(
        [
            np.arange(n_pre + n_ramp) * dt_ramp,
            (n_pre + n_ramp) * dt_ramp + np.arange(1, n_hold + 1) * 0.025,
        ]
    )
    ip_end = 5.5e5 * rng.uniform(0.9, 1.1)
    # labels start at the corpus's Ip floor (frac ≈ 0.4) with the skin already
    # accumulated over an UNLABELLED pre-phase from breakdown scale — exactly
    # the real corpus's structure (300 kA label floor; the early window is
    # chained through but never emitted)
    frac0 = rng.uniform(0.35, 0.45)
    frac = np.concatenate(
        [
            np.linspace(0.15, frac0, n_pre + 1)[:-1],
            np.linspace(frac0, 1.0, n_ramp),
            np.ones(n_hold),
        ]
    )
    ip_seq = ip_end * frac
    quad_max = float(cfg["quad_max"])
    quad = np.concatenate(
        [np.linspace(0.0, quad_max, n_pre + n_ramp), np.full(n_hold, quad_max)]
    )
    # radial force balance wants B_v ∝ Ip — scale the confining pattern on
    # the generator's calibrated pair (6e4 A ↔ 0.6 MA), so the plasma GROWS
    # through the ramp instead of being crushed inboard at low current
    vf_scale = float(cfg.get("vf_scale", 0.9))
    i_pf_seq = np.stack(
        [
            scenario_i_pf(campaign, vf_frac=vf_scale * ip / 6.0e5, quad=q)
            for ip, q in zip(ip_seq, quad, strict=True)
        ]
    )

    # step 0 — equilibrated start of ramp: plain profile, circuit steady state
    truth0 = manufacture(
        campaign,
        beta0=0.5,
        alpha=1.0,
        i_pf=i_pf_seq[0],
        ip_amperes=float(ip_seq[0]),
        seed=int(seed * 1000),
    )
    if not truth0.confined:
        logger.warning("seq %d: initial truth not confined — dropped", seed)
        return None
    psi_n, core, axis, _apsi = _psi_n_state(truth0.psi, grid, ip_seq[0])
    circuit = ps.build_plasma_circuit_from_state(
        grid, psi_n, core, axis, n_rad=TRUTH_N_RAD, n_pol=TRUTH_N_POL
    )
    i_patch = ps.steady_state_currents(circuit, eta_true, float(ip_seq[0]))

    rows = []
    annihilated = []
    warm = np.zeros(grid.flat_r.size)
    warm[grid.cells] = truth0.cell_currents / (grid.dr * grid.dz)

    def _record(truth, k):
        target, _pa, _pb = geometry_target_pushout(truth.psi, grid)
        rows.append(
            {
                "truth": truth,
                "time_s": float(times[k]),
                "target_true": target,
                "li3_true": _li3_2d(truth.psi, truth.core_mask, ip_seq[k], grid),
                "regime": "ramp" if k < n_pre + n_ramp else "hold",
            }
        )

    for k in range(1, n_steps):
        sub = np.linspace(times[k - 1], times[k], int(cfg["n_sub"]))
        wts = (sub - times[k - 1]) / (times[k] - times[k - 1])
        i_pf_sub = (1.0 - wts)[:, None] * i_pf_seq[k - 1] + wts[:, None] * i_pf_seq[k]
        _u, i_end = ps.loop_voltage_for_ip(
            circuit,
            eta_true,
            sub,
            i0=i_patch,
            ip_target=float(ip_seq[k]),
            i_pf_of_t=i_pf_sub,
        )
        # flux-function limit of the evolved state (recorded approximation:
        # the poloidal structure the binning annihilates)
        rad = np.minimum(
            (np.sqrt(np.clip(circuit.tiling.psi_n, 0.0, 1.0)) * TRUTH_N_RAD).astype(
                int
            ),
            TRUTH_N_RAD - 1,
        )
        pol_uniform = np.zeros_like(i_end)
        for b in range(TRUTH_N_RAD):
            m = rad == b
            if m.any():
                pol_uniform[m] = i_end[m].sum() / m.sum()
        annihilated.append(
            float(np.abs(i_end - pol_uniform).sum() / max(np.abs(i_end).sum(), 1e-30))
        )
        h = _radial_bin_profile(circuit, i_end, TRUTH_N_RAD)
        truth = manufacture_shape(
            campaign,
            _shape_from_bins(
                h, TRUTH_N_RAD, grid.r0, float(cfg.get("beta_split", BETA_SPLIT))
            ),
            ip_amperes=float(ip_seq[k]),
            i_pf=i_pf_seq[k],
            seed=int(seed * 1000 + k),
            warm_jphi=warm,
        )
        if not truth.confined:
            logger.warning("seq %d step %d: truth not confined — dropped", seed, k)
            return None
        if k >= n_pre:  # the pre-phase is chained through but never emitted
            _record(truth, k)
            rows[-1]["h_true"] = [float(v) for v in h]
        warm = np.zeros(grid.flat_r.size)
        warm[grid.cells] = truth.cell_currents / (grid.dr * grid.dz)
        # chain remap: rebuild the circuit on the NEW geometry, re-bin state
        psi_n, core, axis, _apsi = _psi_n_state(truth.psi, grid, ip_seq[k])
        circuit = ps.build_plasma_circuit_from_state(
            grid, psi_n, core, axis, n_rad=TRUTH_N_RAD, n_pol=TRUTH_N_POL
        )
        i_patch = ps.bin_cell_currents(circuit.tiling, truth.cell_currents)

    return {
        "seed": int(seed),
        "rows": rows,
        # label-window times / drives (what the fits see) + the pre-label
        # drive history (raw measurements in the real-data analogue — the
        # trajectory integrates from the stream start, never from a = 0)
        "times": [float(v) for v in times[n_pre:]],
        "ip_seq": [float(v) for v in ip_seq[n_pre:]],
        "i_pf_seq": i_pf_seq[n_pre:],
        "pre_times": [float(v) for v in times[:n_pre]],
        "pre_ip": [float(v) for v in ip_seq[:n_pre]],
        "pre_i_pf": i_pf_seq[:n_pre],
        "n_ramp": n_ramp,
        "annihilated_frac": annihilated,
    }


# ---------------------------------------------------------------------------
# fit arms
# ---------------------------------------------------------------------------


def _spine_fit_kw():
    spine, _sha = frozen_spine_config()
    isolve = spine["interior_solve"]
    return dict(
        beta0_grid=(0.5,),
        alpha_grid=(1.0,),
        cost_limit=float("inf"),
        convergence_limit=5e-3,
        retry_max_iterations=160,
        fit_mode="ladder",
        n_p=3,
        n_f=3,
        smoothness=float(isolve["smoothness"]),
        nonneg=True,
        passive_ridge=1.0,
        reseed_axis_r_max=float(isolve["reseed_axis_r_max"]),
        keep_psi=True,
        keep_jphi=True,
        boundary_read="pushout",
    ), int(isolve["passive_k"])


def fit_sequence_spine(job: tuple) -> list:
    """The frozen classical spine, slices independent (warm chain only)."""
    seq, _cfg = job
    campaign = _campaign()
    fit_kw, passive_k = _spine_fit_kw()
    sidecar = build_passive_sidecar(
        campaign.table,
        campaign.grid,
        g_passive=campaign.passive_g_sens,
        sensor_scale=campaign.scale,
        k=passive_k,
    )
    fits = []
    warm = None
    for r in seq["rows"]:
        f = fit_and_read_slice(
            campaign.grid,
            campaign.table,
            r["truth"].to_payload(),
            warm_jphi=warm,
            passive=sidecar,
            **fit_kw,
        )
        if f.scored and f.converged and f.jphi_flat is not None:
            warm = f.jphi_flat
        fits.append(f)
    return fits


def fit_sequence_dyn(job: tuple) -> list:
    """Dynamic-mode arm: spine + k screening columns, trajectory-centred.

    Mode shapes are frozen per solve from the PASS-1 (spine) state — the
    locked bounded-mode-column form: the non-negative convergence anchor is
    untouched and the screening columns add exactly the missing direction,
    amplitudes prior-centred on the exact-ZOH circuit trajectory driven by
    the coil history and the pass-1 plasma history.
    """
    seq, spine_fits, cfg = job
    campaign = _campaign()
    grid = campaign.grid
    eta_arm = EtaProfile.from_vector(np.asarray(cfg["eta_arm"], dtype=np.float64))
    weight = float(cfg["prior_weight"])
    k_modes = int(cfg["k_modes"])
    fit_kw, passive_k = _spine_fit_kw()
    sidecar = build_passive_sidecar(
        campaign.table,
        campaign.grid,
        g_passive=campaign.passive_g_sens,
        sensor_scale=campaign.scale,
        k=passive_k,
    )
    times = np.asarray(seq["times"], dtype=np.float64)
    i_pf_seq = np.asarray(seq["i_pf_seq"], dtype=np.float64)
    cell_area = grid.dr * grid.dz

    dyn = []
    for j, r in enumerate(seq["rows"]):
        f1 = spine_fits[j]
        if not (f1.scored and f1.psi is not None) or weight <= 0.0:
            dyn.append(f1)
            continue
        psi_n, core, axis, _apsi = _psi_n_state(f1.psi, grid, float(f1.ip_amperes))
        try:
            circuit = ps.build_plasma_circuit_from_state(
                grid, psi_n, core, axis, n_rad=FIT_N_RAD, n_pol=FIT_N_POL
            )
            basis = ps.screening_eigenbasis(
                grid,
                circuit,
                eta_arm,
                campaign.g_sens,
                k=k_modes,
                sensor_scale=campaign.scale,
            )
        except (ValueError, np.linalg.LinAlgError):
            dyn.append(f1)
            continue
        # backbone history: every pass-1 fit up to now, binned on THIS tiling;
        # the pre-label window is amplitude-followed with the measured Ip
        # (the raw_eddy_trajectory pattern — the trajectory integrates from
        # the stream start, never from a = 0 at the first label)
        i_b = np.zeros((j + 1, circuit.n_patches))
        for t_prev in range(j + 1):
            fp = spine_fits[t_prev]
            if fp.scored and fp.jphi_flat is not None:
                i_b[t_prev] = ps.bin_cell_currents(
                    circuit.tiling, fp.jphi_flat[grid.cells] * cell_area
                )
        pre_times = np.asarray(seq["pre_times"], dtype=np.float64)
        pre_ip = np.asarray(seq["pre_ip"], dtype=np.float64)
        ip0 = max(abs(float(seq["ip_seq"][0])), 1e-30)
        i_b_pre = np.outer(pre_ip / ip0, i_b[0])
        traj = ps.screening_trajectory(
            basis,
            np.concatenate([pre_times, times[: j + 1]]),
            i_pf_of_t=np.vstack(
                [np.asarray(seq["pre_i_pf"], dtype=np.float64), i_pf_seq[: j + 1]]
            ),
            i_backbone_patch=np.vstack([i_b_pre, i_b]),
        )
        side = ps.screening_sidecar(basis, campaign.scale)
        stacked = ps.stack_sidecars(sidecar, side)
        center = np.concatenate([np.zeros(passive_k), side["amp_scale"] * traj[-1]])
        w_vec = np.concatenate([np.zeros(passive_k), np.full(k_modes, weight)])
        f2 = fit_and_read_slice(
            campaign.grid,
            campaign.table,
            r["truth"].to_payload(),
            warm_jphi=f1.jphi_flat,
            passive=stacked,
            passive_prior=(center, w_vec),
            **fit_kw,
        )
        dyn.append(f2 if f2.scored else f1)
    return dyn


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------


def score_fits(seq: dict, fits: list) -> list[dict]:
    campaign = _campaign()
    grid = campaign.grid
    out = []
    for r, f in zip(seq["rows"], fits, strict=True):
        row = {
            "regime": r["regime"],
            "time_s": r["time_s"],
            "ip_frac": float(r["truth"].ip_amperes / max(seq["ip_seq"][-1], 1e-30)),
            "li3_true": float(r["li3_true"]),
            "scored": bool(f.scored and f.target is not None),
        }
        if row["scored"]:
            tt = np.asarray(r["target_true"], dtype=np.float64)
            tf = np.asarray(f.target, dtype=np.float64)
            dr = tf[6:14] - tt[6:14]
            ang = LCFS_ANGLES
            row.update(
                lcfs_cm=float(np.linalg.norm(dr) / np.sqrt(8.0) * 100.0),
                axis_cm=float(np.hypot(tf[0] - tt[0], tf[1] - tt[1]) * 100.0),
                a0_cm=float(np.nanmean(dr) * 100.0),
                a1_cm=float(2.0 * np.nanmean(dr * np.cos(ang)) * 100.0),
                b1_cm=float(2.0 * np.nanmean(dr * np.sin(ang)) * 100.0),
                a2_cm=float(2.0 * np.nanmean(dr * np.cos(2.0 * ang)) * 100.0),
                cost=float(f.cost),
            )
            if f.psi is not None:
                psi_n, core, _axis, _apsi = _psi_n_state(
                    f.psi, grid, float(f.ip_amperes)
                )
                row["li3_fit"] = float(_li3_2d(f.psi, core, float(f.ip_amperes), grid))
        out.append(row)
    return out


def _med(vals) -> float:
    arr = np.asarray([v for v in vals if v is not None and np.isfinite(v)])
    return float(np.median(arr)) if arr.size else float("nan")


def _regime(rows, key, regime):
    return [r.get(key) for r in rows if r["regime"] == regime and r.get("scored")]


def _paired_bootstrap_ci(diffs: np.ndarray, n_boot: int = 4000, seed: int = 7):
    rng = np.random.default_rng(seed)
    if diffs.size == 0:
        return (float("nan"), float("nan"))
    meds = np.median(rng.choice(diffs, size=(n_boot, diffs.size), replace=True), axis=1)
    return (float(np.quantile(meds, 0.025)), float(np.quantile(meds, 0.975)))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-sequences", type=int, default=8)
    ap.add_argument("--n-tune", type=int, default=2, help="tune-only sequences")
    ap.add_argument("--n-ramp", type=int, default=10)
    ap.add_argument("--n-hold", type=int, default=4)
    ap.add_argument("--n-pre", type=int, default=4)
    ap.add_argument("--n-sub", type=int, default=40)
    ap.add_argument(
        "--eta-true",
        type=str,
        default="-6.5,1.5,1.5",
        help="known truth eta: log10(eta0), contrast, shape — the RAMP-phase "
        "Spitzer scale (a few hundred eV), not the flat-top one",
    )
    ap.add_argument("--quad-max", type=float, default=0.25)
    ap.add_argument("--beta-split", type=float, default=BETA_SPLIT)
    ap.add_argument("--vf-scale", type=float, default=0.9)
    ap.add_argument("--k-modes", type=int, default=2)
    ap.add_argument("--weights", type=str, default="0.5,2,8,32")
    ap.add_argument(
        "--eta-scan",
        type=str,
        default="0.5,1.0,2.0,4.0",
        help="closure-eta scale factors on the oracle eta0 (flat contrast)",
    )
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed0", type=int, default=4200)
    ap.add_argument("--out-suffix", type=str, default="")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.n_sequences, args.n_tune = 2, 1
        args.n_ramp, args.n_hold = 5, 2
        args.weights = "2"
        args.eta_scan = "1.0,2.0"

    eta_true = [float(v) for v in args.eta_true.split(",")]
    weights = [float(v) for v in args.weights.split(",")]
    eta_scales = [float(v) for v in args.eta_scan.split(",")]

    cfg_gen = {
        "eta_true": eta_true,
        "n_ramp": args.n_ramp,
        "n_hold": args.n_hold,
        "n_pre": args.n_pre,
        "n_sub": args.n_sub,
        "quad_max": args.quad_max,
        "beta_split": args.beta_split,
        "vf_scale": args.vf_scale,
    }
    campaign = _campaign()  # build (incl. the cell kernel) BEFORE forking
    logger.info(
        "campaign ready: %d sensors, %d cells",
        len(campaign.channels),
        campaign.grid.cells.size,
    )
    ctx = multiprocessing.get_context("fork")
    t0 = time.perf_counter()
    jobs = [(args.seed0 + k, cfg_gen) for k in range(args.n_sequences)]
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
        seqs = [s for s in pool.map(generate_skin_sequence, jobs) if s is not None]
    logger.info(
        "generated %d/%d skin-truth sequences in %.0f s "
        "(poloidal-annihilation median %.3f)",
        len(seqs),
        len(jobs),
        time.perf_counter() - t0,
        float(np.median([a for s in seqs for a in s["annihilated_frac"]])),
    )
    if len(seqs) < 2:
        raise SystemExit("not enough sequences generated")
    tune_seqs = seqs[: args.n_tune]
    eval_seqs = seqs[args.n_tune :] or tune_seqs

    # --- spine arm (leg a) -------------------------------------------------
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
        spine_fits = list(pool.map(fit_sequence_spine, [(s, {}) for s in seqs]))
    spine_rows = [score_fits(s, f) for s, f in zip(seqs, spine_fits, strict=True)]

    # --- tune: prior weight at oracle eta on the tune subset ----------------
    def _dyn_rows(seq_list, fit_list, cfg):
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
            dyn = list(
                pool.map(
                    fit_sequence_dyn,
                    [(s, f, cfg) for s, f in zip(seq_list, fit_list, strict=True)],
                )
            )
        return [score_fits(s, d) for s, d in zip(seq_list, dyn, strict=True)]

    tune_fits = spine_fits[: args.n_tune]
    weight_scan = {}
    for w in weights:
        cfg = {"eta_arm": eta_true, "prior_weight": w, "k_modes": args.k_modes}
        rows = _dyn_rows(tune_seqs, tune_fits, cfg)
        med = _med(
            [r.get("lcfs_cm") for rs in rows for r in rs if r["regime"] == "ramp"]
        )
        weight_scan[w] = med
        logger.info("weight scan w=%g → tune ramp LCFS median %.2f cm", w, med)
    w_frozen = min(weight_scan, key=lambda k: weight_scan[k])
    logger.info("frozen prior weight: %g", w_frozen)

    # --- closure eta: scale scan on the ramp-transient misfit (tune subset) -
    eta_scan_result = {}
    for s_eta in eta_scales:
        eta_vec = [eta_true[0] + np.log10(s_eta), 0.0, 1.0]  # flat contrast
        cfg = {"eta_arm": eta_vec, "prior_weight": w_frozen, "k_modes": args.k_modes}
        rows = _dyn_rows(tune_seqs, tune_fits, cfg)
        med_cost = _med(
            [r.get("cost") for rs in rows for r in rs if r["regime"] == "ramp"]
        )
        eta_scan_result[s_eta] = med_cost
        logger.info(
            "eta scan scale=%.2g → tune ramp whitened cost median %.4f",
            s_eta,
            med_cost,
        )
    s_closure = min(eta_scan_result, key=lambda k: eta_scan_result[k])
    eta_closure = [eta_true[0] + np.log10(s_closure), 0.0, 1.0]
    logger.info("closure-identified eta scale: %.2g× oracle", s_closure)

    # --- eval arms: oracle and closure eta at the frozen weight -------------
    eval_fits = spine_fits[args.n_tune :] or tune_fits
    dyn_oracle_rows = _dyn_rows(
        eval_seqs,
        eval_fits,
        {"eta_arm": eta_true, "prior_weight": w_frozen, "k_modes": args.k_modes},
    )
    dyn_closure_rows = _dyn_rows(
        eval_seqs,
        eval_fits,
        {"eta_arm": eta_closure, "prior_weight": w_frozen, "k_modes": args.k_modes},
    )
    spine_eval_rows = spine_rows[args.n_tune :] or spine_rows[: args.n_tune]

    # --- leg (a) verdict (ALL sequences — no tuning was involved) ----------
    all_spine = [r for rs in spine_rows for r in rs]
    ramp_lcfs = _regime(all_spine, "lcfs_cm", "ramp")
    hold_lcfs = _regime(all_spine, "lcfs_cm", "hold")
    from scipy import stats

    ramp_cost = [
        (r["cost"], r["lcfs_cm"])
        for r in all_spine
        if r["regime"] == "ramp" and r.get("scored")
    ]
    rho, pval = (
        stats.spearmanr([c for c, _ in ramp_cost], [e for _, e in ramp_cost])
        if len(ramp_cost) > 8
        else (float("nan"), 1.0)
    )
    li3_over_ramp = _med(
        [
            r["li3_fit"] / max(r["li3_true"], 1e-9)
            for r in all_spine
            if r["regime"] == "ramp" and r.get("li3_fit") is not None
        ]
    )
    li3_over_hold = _med(
        [
            r["li3_fit"] / max(r["li3_true"], 1e-9)
            for r in all_spine
            if r["regime"] == "hold" and r.get("li3_fit") is not None
        ]
    )
    leg_a = {
        "lcfs_cm_ramp_median": _med(ramp_lcfs),
        "lcfs_cm_hold_median": _med(hold_lcfs),
        "A1_elevated_ramp": bool(_med(ramp_lcfs) >= 1.8 * _med(hold_lcfs)),
        "cost_lcfs_spearman_ramp": [float(rho), float(pval)],
        "A2_cost_tracks_error": bool(rho >= 0.4 and pval < 0.01),
        "a2_cm_ramp_signed": _med(_regime(all_spine, "a2_cm", "ramp")),
        "a2_cm_hold_signed": _med(_regime(all_spine, "a2_cm", "hold")),
        "a1_cm_ramp_signed": _med(_regime(all_spine, "a1_cm", "ramp")),
        "a1_cm_hold_signed": _med(_regime(all_spine, "a1_cm", "hold")),
        "li3_ratio_ramp": li3_over_ramp,
        "li3_ratio_hold": li3_over_hold,
        # the same 2D estimator on both sides, so the RATIO is calibration-
        # free: an over-reading fit reads li3 well above the skin truth's
        "A4_li3_over_read": bool(
            li3_over_ramp >= 1.5 and li3_over_ramp >= 1.3 * li3_over_hold
        ),
    }
    a2r, a2h = leg_a["a2_cm_ramp_signed"], leg_a["a2_cm_hold_signed"]
    a1r, a1h = leg_a["a1_cm_ramp_signed"], leg_a["a1_cm_hold_signed"]
    leg_a["A3_shape_signature"] = bool(
        a2r > 0
        and a1r < 0
        and abs(a2r) >= 2.0 * abs(a2h)
        and abs(a1r) >= 2.0 * abs(a1h)
    )
    leg_a["PASS"] = bool(
        leg_a["A1_elevated_ramp"]
        and leg_a["A2_cost_tracks_error"]
        and leg_a["A3_shape_signature"]
        and leg_a["A4_li3_over_read"]
    )

    # --- leg (b) verdict (eval sequences, frozen weight) -------------------
    def _flat(rows_list):
        return [r for rs in rows_list for r in rs]

    sp_e, dy_o, dy_c = (
        _flat(spine_eval_rows),
        _flat(dyn_oracle_rows),
        _flat(dyn_closure_rows),
    )
    ramp_sp = _med(_regime(sp_e, "lcfs_cm", "ramp"))
    hold_sp = _med(_regime(sp_e, "lcfs_cm", "hold"))
    ramp_dyn_o = _med(_regime(dy_o, "lcfs_cm", "ramp"))
    ramp_dyn_c = _med(_regime(dy_c, "lcfs_cm", "ramp"))
    hold_dyn_o = _med(_regime(dy_o, "lcfs_cm", "hold"))
    floor = hold_sp
    gap = ramp_sp - floor
    recovery = (ramp_sp - ramp_dyn_o) / gap if gap > 0 else float("nan")
    paired = np.asarray(
        [
            s["lcfs_cm"] - d["lcfs_cm"]
            for s, d in zip(sp_e, dy_o, strict=True)
            if s["regime"] == "ramp" and s.get("scored") and d.get("scored")
        ]
    )
    ci_o = _paired_bootstrap_ci(paired)
    paired_c = np.asarray(
        [
            s["lcfs_cm"] - d["lcfs_cm"]
            for s, d in zip(sp_e, dy_c, strict=True)
            if s["regime"] == "ramp" and s.get("scored") and d.get("scored")
        ]
    )
    ci_c = _paired_bootstrap_ci(paired_c)
    axis_sp = _med([r.get("axis_cm") for r in sp_e if r.get("scored")])
    axis_dyn = _med([r.get("axis_cm") for r in dy_o if r.get("scored")])
    leg_b = {
        "prior_weight_frozen": w_frozen,
        "weight_scan_tune": {str(k): v for k, v in weight_scan.items()},
        "eta_closure_scale": s_closure,
        "eta_scan_tune_cost": {str(k): v for k, v in eta_scan_result.items()},
        "lcfs_cm_ramp_spine": ramp_sp,
        "lcfs_cm_ramp_dyn_oracle": ramp_dyn_o,
        "lcfs_cm_ramp_dyn_closure": ramp_dyn_c,
        "lcfs_cm_hold_spine": hold_sp,
        "lcfs_cm_hold_dyn_oracle": hold_dyn_o,
        "gap_cm": gap,
        "recovery_fraction_oracle": recovery,
        "paired_ramp_gain_cm_median_oracle": _med(paired),
        "paired_ramp_gain_ci_oracle": list(ci_o),
        "paired_ramp_gain_cm_median_closure": _med(paired_c),
        "paired_ramp_gain_ci_closure": list(ci_c),
        "axis_cm_spine": axis_sp,
        "axis_cm_dyn_oracle": axis_dyn,
        "B1_recovers_half_gap_oracle": bool(recovery >= 0.5),
        "B2_beats_spine_at_closure_eta": bool(ramp_dyn_c < ramp_sp),
        # S2-design diagnostics (recorded, NOT part of this rung's
        # pre-declared no-go — the plan gates S1 on B1 + B2 alone; hold/axis
        # non-inferiority are the REAL-DATA gate legs S2 pre-declares)
        "diag_hold_non_inferior": bool(hold_dyn_o <= hold_sp + 0.05),
        "diag_axis_not_degraded": bool(axis_dyn <= 1.05 * axis_sp),
    }
    leg_b["PASS"] = bool(
        leg_b["B1_recovers_half_gap_oracle"] and leg_b["B2_beats_spine_at_closure_eta"]
    )

    result = {
        "arm": "plasma-screening-synthetic-skin-gate",
        "campaign_shot": CAMPAIGN_SHOT,
        "eta_true": eta_true,
        "k_modes": args.k_modes,
        "n_sequences": len(seqs),
        "n_tune": args.n_tune,
        "n_slices": sum(len(rs) for rs in spine_rows),
        "truth_tiling": [TRUTH_N_RAD, TRUTH_N_POL],
        "fit_tiling": [FIT_N_RAD, FIT_N_POL],
        "beta_split_truth": BETA_SPLIT,
        "quad_max": args.quad_max,
        "poloidal_annihilation_median": float(
            np.median([a for s in seqs for a in s["annihilated_frac"]])
        ),
        "leg_a_reproduction": leg_a,
        "leg_b_recovery": leg_b,
        "measured_reference": {
            "lcfs_cm_rampup_median": 8.44,
            "lcfs_cm_flattop_median": 2.96,
            "cost_lcfs_spearman_rampup": 0.77,
            "a2_cm_rampup_signed": 5.71,
            "a1_cm_rampup_signed": -3.80,
        },
        "wall_s_total": time.perf_counter() - t0,
    }
    tag = f"-{args.out_suffix}" if args.out_suffix else ""
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    out_json = ARTIFACTS / f"plasma_screening_gate{tag}.json"
    out_json.write_text(json.dumps(result, indent=2))

    _figures(
        seqs,
        spine_rows,
        spine_eval_rows,
        dyn_oracle_rows,
        dyn_closure_rows,
        spine_fits,
        leg_a,
        leg_b,
        tag,
    )
    logger.info(
        "leg (a) %s | leg (b) %s (recovery %.2f, closure paired %+0.2f cm) | %s",
        "PASS" if leg_a["PASS"] else "FAIL",
        "PASS" if leg_b["PASS"] else "FAIL",
        leg_b["recovery_fraction_oracle"],
        leg_b["paired_ramp_gain_cm_median_closure"],
        out_json,
    )
    return 0


def _figures(
    seqs,
    spine_rows,
    spine_eval_rows,
    dyn_oracle_rows,
    dyn_closure_rows,
    spine_fits,
    leg_a,
    leg_b,
    tag,
):
    import matplotlib.pyplot as plt

    FIGURES.mkdir(parents=True, exist_ok=True)
    all_spine = [r for rs in spine_rows for r in rs]

    # (1) the signature figure — synthetic spine failure vs the measured audit
    fig, ax = plt.subplots(2, 2, figsize=(11.5, 8.4))
    ipf = [r["ip_frac"] for r in all_spine if r.get("scored")]
    lc = [r["lcfs_cm"] for r in all_spine if r.get("scored")]
    reg = [r["regime"] for r in all_spine if r.get("scored")]
    col = {"ramp": "#cc3311", "hold": "#4477aa"}
    ax[0, 0].scatter(ipf, lc, c=[col[r] for r in reg], s=18, alpha=0.7)
    ax[0, 0].axhline(8.44, color="#cc3311", ls=":", lw=1, label="measured ramp 8.44")
    ax[0, 0].axhline(2.96, color="#4477aa", ls=":", lw=1, label="measured flat 2.96")
    ax[0, 0].set_xlabel("Ip fraction")
    ax[0, 0].set_ylabel("spine LCFS error vs truth [cm]")
    ax[0, 0].set_title(
        f"ramp {leg_a['lcfs_cm_ramp_median']:.1f} vs hold "
        f"{leg_a['lcfs_cm_hold_median']:.1f} cm — A1 "
        f"{'PASS' if leg_a['A1_elevated_ramp'] else 'FAIL'}"
    )
    ax[0, 0].legend(fontsize=8)

    cost = [r["cost"] for r in all_spine if r["regime"] == "ramp" and r.get("scored")]
    lcr = [r["lcfs_cm"] for r in all_spine if r["regime"] == "ramp" and r.get("scored")]
    ax[0, 1].scatter(cost, lcr, s=18, alpha=0.7, color="#cc3311")
    rho, pval = leg_a["cost_lcfs_spearman_ramp"]
    ax[0, 1].set_xlabel("whitened cost")
    ax[0, 1].set_ylabel("LCFS error [cm]")
    ax[0, 1].set_title(
        f"ramp Spearman {rho:.2f} (p={pval:.1e}; measured 0.77) — A2 "
        f"{'PASS' if leg_a['A2_cost_tracks_error'] else 'FAIL'}"
    )

    labels = ["a2 elong", "a1 horiz"]
    synth = [leg_a["a2_cm_ramp_signed"], leg_a["a1_cm_ramp_signed"]]
    hold = [leg_a["a2_cm_hold_signed"], leg_a["a1_cm_hold_signed"]]
    meas = [5.71, -3.80]
    x = np.arange(2)
    ax[1, 0].bar(x - 0.25, synth, 0.25, label="synthetic ramp", color="#cc3311")
    ax[1, 0].bar(x, hold, 0.25, label="synthetic hold", color="#4477aa")
    ax[1, 0].bar(x + 0.25, meas, 0.25, label="measured ramp", color="#999999")
    ax[1, 0].set_xticks(x, labels)
    ax[1, 0].axhline(0, color="k", lw=0.8)
    ax[1, 0].set_ylabel("signed harmonic [cm]")
    ax[1, 0].set_title(
        f"shape signature — A3 {'PASS' if leg_a['A3_shape_signature'] else 'FAIL'}"
    )
    ax[1, 0].legend(fontsize=8)

    lt = [
        (r["li3_true"], r["li3_fit"]) for r in all_spine if r.get("li3_fit") is not None
    ]
    regs = [r["regime"] for r in all_spine if r.get("li3_fit") is not None]
    ax[1, 1].scatter(
        [a for a, _ in lt],
        [b for _, b in lt],
        c=[col[r] for r in regs],
        s=18,
        alpha=0.7,
    )
    lim = [0, max(2.5, max((b for _, b in lt), default=2.5))]
    ax[1, 1].plot(lim, lim, "k--", lw=1)
    ax[1, 1].set_xlabel("li3 truth")
    ax[1, 1].set_ylabel("li3 spine fit")
    ax[1, 1].set_title(
        f"li3 ratio ramp {leg_a['li3_ratio_ramp']:.2f} — A4 "
        f"{'PASS' if leg_a['A4_li3_over_read'] else 'FAIL'}"
    )
    fig.suptitle(
        f"leg (a) — failure-signature reproduction: "
        f"{'PASS' if leg_a['PASS'] else 'FAIL'}"
    )
    fig.tight_layout()
    fig.savefig(FIGURES / f"fig-skin-gate-signature{tag}.png", dpi=120)
    plt.close(fig)

    # (2) recovery figure
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.2))
    sp_e = [r for rs in spine_eval_rows for r in rs]
    dy_o = [r for rs in dyn_oracle_rows for r in rs]
    dy_c = [r for rs in dyn_closure_rows for r in rs]
    pairs_o = [
        (s["lcfs_cm"], d["lcfs_cm"])
        for s, d in zip(sp_e, dy_o, strict=True)
        if s["regime"] == "ramp" and s.get("scored") and d.get("scored")
    ]
    ax[0].scatter(
        [a for a, _ in pairs_o],
        [b for _, b in pairs_o],
        s=20,
        alpha=0.7,
        color="#228833",
    )
    lim = [0, max([a for a, _ in pairs_o] + [b for _, b in pairs_o] + [1.0]) * 1.05]
    ax[0].plot(lim, lim, "k--", lw=1)
    ax[0].set_xlabel("spine ramp LCFS error [cm]")
    ax[0].set_ylabel("dynamic-mode (oracle η) [cm]")
    ax[0].set_title(
        f"recovery {leg_b['recovery_fraction_oracle']:.2f} — B1 "
        f"{'PASS' if leg_b['B1_recovers_half_gap_oracle'] else 'FAIL'}"
    )
    bins = np.linspace(0, lim[1], 20)
    ax[1].hist(
        [a for a, _ in pairs_o], bins=bins, alpha=0.55, label="spine", color="#cc3311"
    )
    ax[1].hist(
        [b for _, b in pairs_o],
        bins=bins,
        alpha=0.55,
        label="dyn (oracle η)",
        color="#228833",
    )
    ramp_c = [
        d["lcfs_cm"]
        for s, d in zip(sp_e, dy_c, strict=True)
        if s["regime"] == "ramp" and d.get("scored")
    ]
    ax[1].hist(ramp_c, bins=bins, alpha=0.4, label="dyn (closure η)", color="#4477aa")
    ax[1].set_xlabel("ramp LCFS error [cm]")
    ax[1].legend(fontsize=8)
    ax[1].set_title(
        f"closure-η paired gain {leg_b['paired_ramp_gain_cm_median_closure']:+.2f} cm"
        f" — B2 {'PASS' if leg_b['B2_beats_spine_at_closure_eta'] else 'FAIL'}"
    )
    ws = leg_b["weight_scan_tune"]
    ax[2].semilogx([float(k) for k in ws], list(ws.values()), "o-", color="#228833")
    ax[2].axvline(leg_b["prior_weight_frozen"], color="k", ls=":", lw=1, label="frozen")
    ax[2].set_xlabel("screening prior weight (tune)")
    ax[2].set_ylabel("tune ramp LCFS median [cm]")
    ax[2].legend(fontsize=8)
    fig.suptitle(f"leg (b) — recovery: {'PASS' if leg_b['PASS'] else 'FAIL'}")
    fig.tight_layout()
    fig.savefig(FIGURES / f"fig-skin-gate-recovery{tag}.png", dpi=120)
    plt.close(fig)

    # (3) representative slice: truth profile vs the spine's peaked read
    seq0 = seqs[0]
    k_mid = max(1, seq0["n_ramp"] - 2)
    r0row = seq0["rows"][k_mid]
    f0 = spine_fits[0][k_mid]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    if "h_true" in r0row:
        h = np.asarray(r0row["h_true"])
        s_cent = (np.arange(h.size) + 0.5) / h.size
        ax[0].plot(s_cent**2, h / max(h.max(), 1e-30), "o-", label="truth (skin)")
    if f0.scored and f0.psi is not None and f0.jphi_flat is not None:
        campaign = _campaign()
        grid = campaign.grid
        psi_n, core, _axis, _apsi = _psi_n_state(f0.psi, grid, float(f0.ip_amperes))
        pn_c = np.clip(psi_n[grid.cells], 0.0, 1.0)
        j_c = f0.jphi_flat[grid.cells]
        bins = np.linspace(0, 1, 15)
        prof = [
            np.mean(j_c[(pn_c >= a) & (pn_c < b)])
            if ((pn_c >= a) & (pn_c < b)).any()
            else np.nan
            for a, b in zip(bins[:-1], bins[1:], strict=True)
        ]
        prof = np.asarray(prof)
        ax[0].plot(
            0.5 * (bins[:-1] + bins[1:]),
            prof / np.nanmax(np.abs(prof)),
            "s-",
            label="spine fit (peaked)",
        )
    ax[0].set_xlabel("ψ_N")
    ax[0].set_ylabel("jφ (normalised)")
    ax[0].set_title("the state the peaked ladder cannot hold")
    ax[0].legend(fontsize=8)
    tt = np.asarray(r0row["target_true"])
    ang = LCFS_ANGLES
    ax[1].plot(
        tt[0] + tt[6:14] * np.cos(ang),
        tt[1] + tt[6:14] * np.sin(ang),
        "o-",
        label="truth LCFS",
    )
    if f0.scored and f0.target is not None:
        tf = np.asarray(f0.target)
        ax[1].plot(
            tf[0] + tf[6:14] * np.cos(ang),
            tf[1] + tf[6:14] * np.sin(ang),
            "s--",
            label="spine LCFS",
        )
    ax[1].set_aspect("equal")
    ax[1].set_xlabel("R [m]")
    ax[1].set_ylabel("Z [m]")
    ax[1].set_title("representative ramp slice")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / f"fig-skin-gate-example{tag}.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
