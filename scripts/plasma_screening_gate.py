#!/usr/bin/env python
"""Synthetic skin-truth gate for the plasma screening circuit.

Truths that the coefficient-ladder generator CANNOT produce: sequences of
manufactured equilibria whose current profile evolves under the COUPLED
plasma + vessel filament circuit with a KNOWN η(ψ_N) through fast
shape-forming ramps.  The chain starts from the machine-quiescent state —
a vacuum phase (coils ramp, vessel eddies build, no plasma), then the
plasma is caught at breakdown scale (Ip a few % of flat-top, its profile
the fully-diffused high-η limit — the unique early state fast diffusion
pins), and the current and profile GROW from there: every integral term
accumulates from zero, so the flux ledger carries no free constant.  Per
interval the coupled patch+vessel currents evolve exactly (ZOH eigenmodes;
uniform loop voltage on the plasma rows shot to the prescribed Ip; coil
swing and vessel eddies drive the same system), the evolved state is
binned to a hollow-capable flux-function profile, and a force-balanced
free-boundary equilibrium is solved from it with the evolved vessel state
injected — a skin-current truth chained exactly the way the plasma forms
one (frozen geometry per interval, remap at labels).  Each ramp ends in a
flat-top hold giving the representation-adequate floor.

MEASURED SCENARIO LIMITATION (recorded before the gate run, 8 design
probes): the limited→diverted transition band CANNOT be manufactured with
the current forward operator.  Within the confined fixed-point family the
nearest saddle stays ~1 m outboard of the boundary (out-limiter, between
the P4/P5/P6 coils) under every lever probed — P4/P5 quadrupole to
deconfinement (± radial-balance compensation), P2 divertor drive
both/lower-only at up to 3× the confining strength (± compensation; the
one marginal 'diverted' read was a knife-edge interior saddle that
vanishes under a 0.005 quad change), P6 weakened/off/flipped, vertical
asymmetry, and blends toward shot 11766's MEASURED flat-top coil pattern
(the real diverted program) — which has NO confined fixed point in this
operator even with the solenoid and case channels removed (the documented
outboard-attractor coil-model error).  The class-flip assertion C2 is
therefore expected to FAIL for scenario reasons, and A2 remains untestable
as measured; both are recorded honestly.  Unblocking the transition truth
family needs the coil-model attractor fix or a free-boundary-evolution
truth chain (S3 territory).

Pre-declared legs (RE-DECLARED for this rung BEFORE running, 2026-07-18 —
supersedes the first synthetic gate's oracle-primary rule; verdicts
recorded honestly either way):

* **leg (a) — reproduction** (unchanged).  The frozen classical spine
  (byte-same config as the real-data gates) fitted to these truths must
  REPRODUCE the measured limited-phase failure signature
  (limiter_phase_fidelity_audit):
    A1  elevated ramp boundary error:   LCFS ramp median ≥ 1.8× hold median
    A2  cost tracks the boundary error: ramp Spearman(cost, LCFS) ≥ 0.4,
        p < 0.01  (measured: 0.77) — now testable: the transition band the
        measured hotspot lives in exists in these truths
    A3  the shape signature:            a2 > 0 AND a1 < 0 on ramp, both
        ≥ 2× the hold level (measured: a2 +5.7, a1 −3.8 cm)
    A4  li3 over-read:                  ramp ratio ≥ 1.5 and ≥ 1.3× hold

* **leg (b) — recovery, PRIMARY at the closure-identified η.**  The
  dynamic-mode fit (frozen non-negative backbone + k bounded zero-net
  screening columns, amplitudes prior-centred on the coupled-circuit
  trajectory) at the η the misfit scan identifies from the ramp
  transients — the operating point the real-data rung would actually run,
  since S2 has no oracle — must:
    B1  recover ≥ 50% of the synthetic ramp LCFS gap:
        (spine_ramp − dyn_ramp) / (spine_ramp − floor) ≥ 0.5 with the
        paired-gain bootstrap CI clear of zero (floor = the spine's own
        hold-phase median; tuning is ORACLE-FREE: the closure η is
        identified first by misfit scan at a fixed scan weight, then the
        prior weight is scanned at the closure η — the protocol S2 runs)
  The ORACLE-η arm is retained as a reported DIAGNOSTIC (B2) — with the
  mode-build η sensitivity (flat vs truth-contrast at the closure scale)
  isolating what the S1 run attributed: the η contrast in the mode build,
  not the representation, breaks the oracle arm.

* **assertions (new, pre-declared)** — the dynamics must be complete:
    C1  flux ledger closes: ∫u dt = ΔΨ̄ + ∫mean(R·i) dt within every
        interval to < 1% of the drive volt-seconds, with the shape-remap
        (dL/dt) term carried explicitly across geometry updates and the
        breakdown-formation term recorded — no free integral constant
        (the chain starts at machine-quiescent zero)
    C2  the limited→diverted class flip occurs during the labelled ramp
        of every eval sequence (push-out class read), flip inside
        Ip-frac [0.55, 0.97]
    C3  all dΦ/dt terms folded: vessel eddies evolve in the SAME circuit
        as the plasma patches in the truth chain, and enter the fit-side
        trajectory drive (structural — recorded from the build)
    C4  the plasma inductance evolves along the chain (dL_p/dt captured):
        the patch-mean self-flux per ampere changes by ≥ 5% over the ramp

A leg-(a) FAIL re-opens the skin/shape diagnosis; a leg-(b) FAIL at the
closure operating point stops the plan before any real-data spend.

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
from imas_ambix.latent.temporal_operator import (
    build_passive_circuit_system,
    integrate_eddy_ode,
)
from imas_ambix.latent.topology import (
    LCFS_ANGLES,
    _inside_polygon,
    emergent_xpoints,
    find_critical_points,
    lcfs_contour,
)
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
TRANSITION_HALF_WIDTH = 0.06  # Ip-frac half width of the transition band

_CAMPAIGN: dict[str, object] = {}


def _campaign():
    if "c" not in _CAMPAIGN:
        c = build_campaign(CAMPAIGN_SHOT, nr=65, nz=97)
        c.grid.cell_greens()  # the screening L kernel — build once, share COW
        vsys = build_passive_circuit_system(c.table, c.grid)
        if vsys.n_circuits != c.n_passive:
            raise ValueError(
                f"passive-set mismatch: vessel system {vsys.n_circuits} vs "
                f"campaign columns {c.n_passive}"
            )
        # vessel↔coil linkage permuted into the i_pf channel order the
        # scenario drives (channels absent from the vessel build get zero)
        chan_idx = {ch: j for j, ch in enumerate(vsys.coil_channels)}
        m_vc_ipf = np.zeros((vsys.n_circuits, len(c.fwd.pf_amc_channels)))
        for j, chan in enumerate(c.fwd.pf_amc_channels):
            if chan in chan_idx:
                m_vc_ipf[:, j] = vsys.m_coil_circ[:, chan_idx[chan]]
        from scipy.linalg import eigh  # noqa: PLC0415

        w, vv = eigh(np.diag(vsys.r_diag), vsys.lmat)
        _CAMPAIGN["c"] = c
        _CAMPAIGN["vsys"] = vsys
        _CAMPAIGN["m_vc_ipf"] = m_vc_ipf
        _CAMPAIGN["vessel_eig"] = (1.0 / np.clip(w, 1e-12, None), vv)
    return _CAMPAIGN["c"]


def _vessel():
    _campaign()
    return _CAMPAIGN["vsys"], _CAMPAIGN["m_vc_ipf"], _CAMPAIGN["vessel_eig"]


# ---------------------------------------------------------------------------
# scenario: vacuum → breakdown catch → shape-forming ramp through the
# limited→diverted transition → flat-top hold
# ---------------------------------------------------------------------------


def scenario_i_pf(
    campaign,
    vf_frac: float,
    quad: float,
    div_frac: float = 0.0,
    boost: float = 0.0,
) -> np.ndarray:
    """Confining coil pattern at strength fraction ``vf_frac`` with a
    quadrupole differential ``quad``, a P2 divertor drive ``div_frac``, and
    a radial-balance compensation ``boost``.

    Boosting the off-midplane P4 set against the near-midplane P5 set adds
    the field curvature that elongates (sign VERIFIED on this campaign:
    quad = +0.4 raises the truth's vertical/horizontal radius ratio) — but
    the quad lever alone DECONFINES before an X-point enters the vessel
    (measured: quad ≥ 0.85 sends the axis outboard, no in-vessel saddle at
    any confined quad).  The transition lever is the P2 divertor set (the
    coils MAST forms its X-points with): current of the SAME sign as Ip at
    ``div_frac`` of the UNBOOSTED confining strength pulls a null onto the
    boundary — but it also weakens the net vertical field, so the confining
    set is co-scaled by ``(1 + boost)`` to hold radial force balance
    (measured operating point: div_frac ≈ 2.0, boost ≈ 0.6 diverts while
    confined).  P2 CASE channels are structural circuits, never driven.  P6
    stays on the base pattern (in-vessel, close to the plasma — a strong
    differential there deforms the boundary read region).
    """
    base = build_confining_i_pf(campaign.fwd, 6.0e4 * vf_frac * (1.0 + boost))
    out = base.copy()
    for j, chan in enumerate(campaign.fwd.pf_amc_channels):
        if chan.startswith("p4"):
            out[j] = base[j] * (1.0 + quad)
        elif chan.startswith("p5"):
            out[j] = base[j] * (1.0 - quad)
        elif chan.startswith("p2") and "coil" in chan:
            out[j] = +abs(div_frac) * 6.0e4 * vf_frac
    return out


def read_truth_class(psi2d, grid, axis) -> tuple[bool | None, np.ndarray]:
    """Limited/diverted class of one truth ψ via the push-out reader chain."""
    lc = lcfs_contour(
        np.asarray(psi2d),
        grid.rg,
        grid.zg,
        (float(axis[0]), float(axis[1])),
        limiter_r=grid.limiter_r,
        limiter_z=grid.limiter_z,
        clip_legs=True,
    )
    if not lc.found:
        return None, np.zeros((0, 2))
    cp = find_critical_points(np.asarray(psi2d), grid.rg, grid.zg)
    xpts = cp.x_points
    if xpts.shape[0]:
        ins = _inside_polygon(
            xpts[:, 0], xpts[:, 1], grid.limiter_r, grid.limiter_z
        ) & grid.clear_of_conductors(xpts[:, 0], xpts[:, 1])
        xpts = xpts[ins]
    xset, diverted = emergent_xpoints(xpts, lc.ring, tol=0.05)
    return bool(diverted), xset[~np.isnan(xset[:, 0])]


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


def _build_coupled(grid, psi_n, core, axis, n_rad, n_pol):
    """Plasma circuit on the current geometry, coupled to the vessel set."""
    campaign = _campaign()
    vsys, m_vc_ipf, _eig = _vessel()
    circuit = ps.build_plasma_circuit_from_state(
        grid, psi_n, core, axis, n_rad=n_rad, n_pol=n_pol
    )
    m_pv = ps.patch_external_linkage(grid, circuit.tiling, campaign.passive_psi_grid)
    coupled = ps.build_coupled_circuit(
        circuit,
        l_ext=vsys.lmat,
        r_ext=vsys.r_diag,
        m_ext_coil=m_vc_ipf,
        m_patch_ext=m_pv,
    )
    return circuit, coupled


def generate_skin_sequence(job: tuple) -> dict | None:
    """One coupled-circuit skin-truth shot: vacuum → breakdown → ramp → hold."""
    seed, cfg = job
    rng = np.random.default_rng(seed)
    campaign = _campaign()
    vsys, m_vc_ipf, (tau_v, vv) = _vessel()
    grid = campaign.grid
    eta_true = EtaProfile.from_vector(np.asarray(cfg["eta_true"], dtype=np.float64))
    n_vac = int(cfg.get("n_vac", 4))
    n_pre = int(cfg.get("n_pre", 6))
    n_ramp, n_hold = int(cfg["n_ramp"]), int(cfg["n_hold"])
    n_steps = n_pre + n_ramp + n_hold
    frac_bd = float(cfg.get("frac_bd", 0.05))

    dt_ramp = rng.uniform(0.012, 0.018)
    settle_s = float(cfg.get("settle_s", 0.05))
    # stream timeline: t = 0 is the machine-quiescent state (no coil current,
    # no vessel eddies, no plasma) — every integral term starts at zero.  The
    # coils ramp through the vacuum phase and then HOLD for a settle window
    # before breakdown: the ramp-induced vessel eddies (measured ~40% of the
    # catch-scale coil currents) must decay or they crush the breakdown
    # plasma — the real-operations pattern of waiting for error fields to
    # settle before initiating breakdown
    t_vac = np.arange(n_vac) * dt_ramp
    catch_t0 = float(t_vac[-1]) + settle_s if n_vac > 0 else settle_s
    times = catch_t0 + np.concatenate(
        [
            np.arange(n_pre + n_ramp) * dt_ramp,
            (n_pre + n_ramp) * dt_ramp + np.arange(1, n_hold + 1) * 0.025,
        ]
    )
    ip_end = 5.5e5 * rng.uniform(0.9, 1.1)
    # labels start at the corpus's Ip floor (frac ≈ 0.4); the pre-phase now
    # runs all the way down to the breakdown catch (frac_bd) and the vacuum
    # phase before it — chained through but never emitted, exactly the real
    # corpus's structure (300 kA label floor; early window unobserved)
    frac0 = rng.uniform(0.35, 0.45)
    frac = np.concatenate(
        [
            np.linspace(frac_bd, frac0, n_pre + 1)[:-1],
            np.linspace(frac0, 1.0, n_ramp),
            np.ones(n_hold),
        ]
    )
    ip_seq = ip_end * frac
    # shaping grows with the current: the P2 divertor drive and its
    # radial-balance compensation ramp ∝ Ip fraction and SATURATE at
    # frac_sat, crossing the X-point-forming threshold late in the ramp
    # (the transition band); the quad differential stays confinement-safe
    quad_max = float(cfg["quad_max"])
    div_max = float(cfg.get("div_max", 2.15))
    boost_max = float(cfg.get("boost_max", 0.65))
    frac_sat = float(cfg.get("frac_sat", 0.85))
    vf_scale = float(cfg.get("vf_scale", 0.9))

    def _i_pf_at(f_frac: float) -> np.ndarray:
        s = min(f_frac, frac_sat) / frac_sat
        return scenario_i_pf(
            campaign,
            vf_frac=vf_scale * f_frac * ip_end / 6.0e5,
            quad=quad_max * f_frac,
            div_frac=div_max * s,
            boost=boost_max * s,
        )

    # the vacuum ramp ENDS at the catch pattern (frac_bd) so the settle
    # interval [t_vac[-1], catch] holds the coils exactly constant — the
    # piecewise-linear ZOH then decays the eddies exactly
    frac_vac = np.linspace(0.0, frac_bd, n_vac)
    i_pf_vac = np.stack([_i_pf_at(f) for f in frac_vac])
    i_pf_seq = np.stack([_i_pf_at(f) for f in frac])

    # vacuum phase: vessel-only evolution from the quiescent zero state (the
    # coil ramp is piecewise-linear, so the ZOH integration is exact)
    t_vac_full = np.concatenate([t_vac, times[:1]])
    i_pf_vac_full = np.vstack([i_pf_vac, i_pf_seq[:1]])
    psi_v_m = (i_pf_vac_full @ m_vc_ipf.T) @ vv
    a_v, _u = integrate_eddy_ode(tau_v, t_vac_full, psi_v_m)
    i_vessel = a_v[-1] @ vv.T  # vessel eddies at the breakdown catch

    # breakdown catch: the plasma appears at a few % of flat-top current with
    # the fully-diffused (plain) profile — the unique early state the fast
    # high-η diffusion pins; its formation volt-seconds are ledgered below
    truth0 = manufacture(
        campaign,
        beta0=0.5,
        alpha=1.0,
        i_pf=i_pf_seq[0],
        ip_amperes=float(ip_seq[0]),
        passive_amplitudes=i_vessel,
        seed=int(seed * 1000),
        continuation=False,
    )
    if not truth0.confined:
        logger.warning("seq %d: breakdown-catch truth not confined — dropped", seed)
        return None
    psi_n, core, axis, _apsi = _psi_n_state(truth0.psi, grid, ip_seq[0])
    circuit, coupled = _build_coupled(grid, psi_n, core, axis, TRUTH_N_RAD, TRUTH_N_POL)
    i_patch = ps.bin_cell_currents(circuit.tiling, truth0.cell_currents)
    i_state = np.concatenate([i_patch, i_vessel])

    # flux ledger — every term accumulates from the machine-quiescent zero
    ledger = {
        "vs_drive": 0.0,  # ∫ u dt (loop volt-seconds applied)
        "delta_psi_bar": 0.0,  # Σ within-interval ΔΨ̄ (dΦ/dt carried exactly)
        "resistive_vs": 0.0,  # Σ ∫ mean(R·i) dt
        "formation_vs": ps.mean_plasma_linked_flux(coupled, i_state, i_pf_seq[0]),
        "remap_vs": 0.0,  # Σ shape-remap jumps (the frozen-geometry dL/dt carry)
        "closure_err_vs": 0.0,  # Σ |per-interval identity residual|
    }
    l_eff = [  # patch-mean plasma self-flux per ampere — dL_p/dt made visible
        float((circuit.lmat @ i_patch).mean() / max(abs(float(i_patch.sum())), 1e-30))
    ]

    rows = []
    annihilated = []
    warm = np.zeros(grid.flat_r.size)
    warm[grid.cells] = truth0.cell_currents / (grid.dr * grid.dz)

    def _record(truth, k):
        target, _pa, _pb = geometry_target_pushout(truth.psi, grid)
        diverted, xset = read_truth_class(truth.psi, grid, truth.axis)
        rows.append(
            {
                "truth": truth,
                "time_s": float(times[k]),
                "target_true": target,
                "li3_true": _li3_2d(truth.psi, truth.core_mask, ip_seq[k], grid),
                "regime": "ramp" if k < n_pre + n_ramp else "hold",
                "class_true": (
                    "diverted"
                    if diverted
                    else ("limited" if diverted is not None else "unread")
                ),
                "x_true": xset.tolist(),
            }
        )

    p = circuit.n_patches
    for k in range(1, n_steps):
        sub = np.linspace(times[k - 1], times[k], int(cfg["n_sub"]))
        wts = (sub - times[k - 1]) / (times[k] - times[k - 1])
        i_pf_sub = (1.0 - wts)[:, None] * i_pf_seq[k - 1] + wts[:, None] * i_pf_seq[k]
        u, _i_end = ps.coupled_loop_voltage_for_ip(
            coupled,
            eta_true,
            sub,
            i0=i_state,
            ip_target=float(ip_seq[k]),
            i_pf_of_t=i_pf_sub,
        )
        traj = ps.evolve_coupled(
            coupled,
            eta_true,
            sub,
            i0=i_state,
            loop_voltage=np.full(sub.size, u),
            i_pf_of_t=i_pf_sub,
        )
        i_end = traj[-1]
        # ledger: the per-patch circuit identity dΨ̄/dt + mean(R·i) = u,
        # integrated over the interval (trapezoid on the sub-cadence state)
        r_p = circuit.r_diag(eta_true)
        psi0 = ps.mean_plasma_linked_flux(coupled, traj[0], i_pf_sub[0])
        psi1 = ps.mean_plasma_linked_flux(coupled, traj[-1], i_pf_sub[-1])
        res = float(np.trapezoid((traj[:, :p] * r_p[np.newaxis, :]).mean(axis=1), sub))
        dt_k = float(sub[-1] - sub[0])
        ledger["vs_drive"] += u * dt_k
        ledger["delta_psi_bar"] += psi1 - psi0
        ledger["resistive_vs"] += res
        ledger["closure_err_vs"] += abs(u * dt_k - ((psi1 - psi0) + res))

        i_plasma_end, i_vessel = i_end[:p], i_end[p:]
        # flux-function limit of the evolved state (recorded approximation:
        # the poloidal structure the binning annihilates)
        rad = np.minimum(
            (np.sqrt(np.clip(circuit.tiling.psi_n, 0.0, 1.0)) * TRUTH_N_RAD).astype(
                int
            ),
            TRUTH_N_RAD - 1,
        )
        pol_uniform = np.zeros_like(i_plasma_end)
        for b in range(TRUTH_N_RAD):
            m = rad == b
            if m.any():
                pol_uniform[m] = i_plasma_end[m].sum() / m.sum()
        annihilated.append(
            float(
                np.abs(i_plasma_end - pol_uniform).sum()
                / max(np.abs(i_plasma_end).sum(), 1e-30)
            )
        )
        h = _radial_bin_profile(circuit, i_plasma_end, TRUTH_N_RAD)
        truth = manufacture_shape(
            campaign,
            _shape_from_bins(
                h, TRUTH_N_RAD, grid.r0, float(cfg.get("beta_split", BETA_SPLIT))
            ),
            ip_amperes=float(ip_seq[k]),
            i_pf=i_pf_seq[k],
            passive_amplitudes=i_vessel,
            seed=int(seed * 1000 + k),
            warm_jphi=warm,
        )
        if not truth.confined:
            logger.warning("seq %d step %d: truth not confined — dropped", seed, k)
            return None
        if k >= n_pre:  # the vacuum + pre phases are chained, never emitted
            _record(truth, k)
            rows[-1]["h_true"] = [float(v) for v in h]
        warm = np.zeros(grid.flat_r.size)
        warm[grid.cells] = truth.cell_currents / (grid.dr * grid.dz)
        # chain remap: rebuild the coupled system on the NEW geometry, re-bin
        # the plasma state (vessel rows carry over — their geometry is fixed);
        # the Ψ̄ jump across the remap is the shape/inductance (dL/dt) EMF the
        # frozen-geometry chain carries — ledgered explicitly
        psi_old = ps.mean_plasma_linked_flux(coupled, i_end, i_pf_seq[k])
        psi_n, core, axis, _apsi = _psi_n_state(truth.psi, grid, ip_seq[k])
        circuit, coupled = _build_coupled(
            grid, psi_n, core, axis, TRUTH_N_RAD, TRUTH_N_POL
        )
        p = circuit.n_patches
        i_patch = ps.bin_cell_currents(circuit.tiling, truth.cell_currents)
        i_state = np.concatenate([i_patch, i_vessel])
        ledger["remap_vs"] += (
            ps.mean_plasma_linked_flux(coupled, i_state, i_pf_seq[k]) - psi_old
        )
        l_eff.append(
            float(
                (circuit.lmat @ i_patch).mean() / max(abs(float(i_patch.sum())), 1e-30)
            )
        )

    # transition band from the truth class sequence (first limited→diverted
    # flip among the emitted labels)
    classes = [r["class_true"] for r in rows]
    fracs = [float(ip_seq[n_pre + j] / ip_end) for j in range(len(rows))]
    flip_frac = None
    for j in range(1, len(rows)):
        if classes[j] == "diverted" and classes[j - 1] == "limited":
            flip_frac = fracs[j]
            break
    for j, r in enumerate(rows):
        if flip_frac is None:
            r["band"] = r["class_true"]
        elif abs(fracs[j] - flip_frac) <= TRANSITION_HALF_WIDTH:
            r["band"] = "transition"
        else:
            r["band"] = "limited" if fracs[j] < flip_frac else "diverted"

    ledger["l_eff_wb_per_a"] = [float(v) for v in l_eff]
    ledger["closure_frac"] = float(
        ledger["closure_err_vs"] / max(abs(ledger["vs_drive"]), 1e-30)
    )

    return {
        "seed": int(seed),
        "rows": rows,
        # label-window times / drives (what the fits see) + the FULL pre-label
        # drive history from the machine-quiescent start (raw measurements in
        # the real-data analogue — the trajectory integrates from t = 0 with
        # a = 0 exactly, never from a fitted constant)
        "times": [float(v) for v in times[n_pre:]],
        "ip_seq": [float(v) for v in ip_seq[n_pre:]],
        "i_pf_seq": i_pf_seq[n_pre:],
        "pre_times": [float(v) for v in times[:n_pre]],
        "pre_ip": [float(v) for v in ip_seq[:n_pre]],
        "pre_i_pf": i_pf_seq[:n_pre],
        "vac_times": [float(v) for v in t_vac],
        "vac_i_pf": i_pf_vac,
        "n_ramp": n_ramp,
        "annihilated_frac": annihilated,
        "ledger": ledger,
        "flip_frac": flip_frac,
        "ip_end": float(ip_end),
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
    amplitudes prior-centred on the exact-ZOH circuit trajectory driven from
    the machine-quiescent stream start (vacuum → breakdown → labels, a = 0
    at t = 0 exactly) by the coil history, the PREDICTED vessel-eddy history
    (all dΦ/dt terms), and the pass-1 plasma history.
    """
    seq, spine_fits, cfg = job
    campaign = _campaign()
    _vsys, m_vc_ipf, (tau_v, vv) = _vessel()
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
    pre_times = np.asarray(seq["pre_times"], dtype=np.float64)
    pre_ip = np.asarray(seq["pre_ip"], dtype=np.float64)
    pre_i_pf = np.asarray(seq["pre_i_pf"], dtype=np.float64)
    vac_times = np.asarray(seq["vac_times"], dtype=np.float64)
    vac_i_pf = np.asarray(seq["vac_i_pf"], dtype=np.float64)
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
        # the pre-label window is amplitude-followed with the measured Ip down
        # to the breakdown catch and ZERO through the vacuum phase — the
        # trajectory integrates from the machine-quiescent stream start, so
        # the integral carries no fitted constant
        i_b = np.zeros((j + 1, circuit.n_patches))
        for t_prev in range(j + 1):
            fp = spine_fits[t_prev]
            if fp.scored and fp.jphi_flat is not None:
                i_b[t_prev] = ps.bin_cell_currents(
                    circuit.tiling, fp.jphi_flat[grid.cells] * cell_area
                )
        ip0 = max(abs(float(seq["ip_seq"][0])), 1e-30)
        i_b_pre = np.outer(pre_ip / ip0, i_b[0])
        i_b_vac = np.zeros((vac_times.size, circuit.n_patches))
        t_full = np.concatenate([vac_times, pre_times, times[: j + 1]])
        i_pf_full = np.vstack([vac_i_pf, pre_i_pf, i_pf_seq[: j + 1]])
        i_b_full = np.vstack([i_b_vac, i_b_pre, i_b])
        # predicted vessel-eddy history (nominal R, measured drives + backbone
        # plasma flux) — the external dΦ/dt term the coil channels don't carry
        m_pv_fit = ps.patch_external_linkage(
            grid, circuit.tiling, campaign.passive_psi_grid
        )
        psi_v_m = (i_pf_full @ m_vc_ipf.T + i_b_full @ m_pv_fit) @ vv
        a_v, _u = integrate_eddy_ode(tau_v, t_full, psi_v_m)
        i_v_pred = a_v @ vv.T
        psi_extra = i_v_pred @ (basis.v.T @ m_pv_fit).T
        traj = ps.screening_trajectory(
            basis,
            t_full,
            i_pf_of_t=i_pf_full,
            i_backbone_patch=i_b_full,
            psi_extra_m=psi_extra,
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
            "band": r.get("band", ""),
            "class_true": r.get("class_true", ""),
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


def _band(rows, key, band):
    return [r.get(key) for r in rows if r.get("band") == band and r.get("scored")]


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
    ap.add_argument("--n-pre", type=int, default=6)
    ap.add_argument("--n-vac", type=int, default=4)
    ap.add_argument(
        "--settle-s",
        type=float,
        default=0.05,
        help="coil-hold settle window between the vacuum ramp and breakdown "
        "(vessel eddies must decay before the plasma can be caught)",
    )
    ap.add_argument(
        "--frac-bd",
        type=float,
        default=0.05,
        help="breakdown-catch Ip fraction of flat-top (the plasma is caught "
        "at this current with the fully-diffused early profile)",
    )
    ap.add_argument("--n-sub", type=int, default=40)
    ap.add_argument(
        "--eta-true",
        type=str,
        default="-6.5,1.5,1.5",
        help="known truth eta: log10(eta0), contrast, shape — the RAMP-phase "
        "Spitzer scale (a few hundred eV), not the flat-top one",
    )
    ap.add_argument(
        "--quad-max",
        type=float,
        default=0.25,
        help="peak quadrupole differential (elongation; confinement-safe)",
    )
    ap.add_argument(
        "--div-max",
        type=float,
        default=0.0,
        help="peak P2 divertor drive as a fraction of the unboosted confining "
        "strength (measured NOT to produce a readable X-point at any confined "
        "strength — see the scenario-limitation note; kept for reproducing "
        "the design probes)",
    )
    ap.add_argument(
        "--boost-max",
        type=float,
        default=0.0,
        help="peak radial-balance compensation of the confining set (the P2 "
        "drive weakens the net vertical field)",
    )
    ap.add_argument(
        "--frac-sat",
        type=float,
        default=0.85,
        help="Ip fraction at which the divertor drive and boost saturate "
        "(the operating point then holds through flat-top)",
    )
    ap.add_argument("--beta-split", type=float, default=BETA_SPLIT)
    ap.add_argument("--vf-scale", type=float, default=0.9)
    ap.add_argument("--k-modes", type=int, default=2)
    ap.add_argument(
        "--weights",
        type=str,
        default="0.05,0.2,0.5,2,8",
        help="screening prior-weight scan (extends BELOW the S1 scan's frozen "
        "edge at 0.5)",
    )
    ap.add_argument(
        "--eta-scan",
        type=str,
        default="0.25,0.5,1.0,2.0,4.0",
        help="closure-eta scale factors on the oracle eta0 (flat contrast); "
        "extends below the S1 scan's frozen edge at 0.5",
    )
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed0", type=int, default=4200)
    ap.add_argument("--out-suffix", type=str, default="")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.n_sequences, args.n_tune = 3, 1
        args.n_ramp, args.n_hold = 5, 2
        args.n_pre, args.n_vac = 4, 2
        args.weights = "0.5"
        args.eta_scan = "0.5,1.0"

    eta_true = [float(v) for v in args.eta_true.split(",")]
    weights = [float(v) for v in args.weights.split(",")]
    eta_scales = [float(v) for v in args.eta_scan.split(",")]

    cfg_gen = {
        "eta_true": eta_true,
        "n_ramp": args.n_ramp,
        "n_hold": args.n_hold,
        "n_pre": args.n_pre,
        "n_vac": args.n_vac,
        "settle_s": args.settle_s,
        "frac_bd": args.frac_bd,
        "n_sub": args.n_sub,
        "quad_max": args.quad_max,
        "div_max": args.div_max,
        "boost_max": args.boost_max,
        "frac_sat": args.frac_sat,
        "beta_split": args.beta_split,
        "vf_scale": args.vf_scale,
    }
    campaign = _campaign()  # build (incl. cell kernel + vessel system) BEFORE forking
    logger.info(
        "campaign ready: %d sensors, %d cells, %d vessel circuits",
        len(campaign.channels),
        campaign.grid.cells.size,
        _CAMPAIGN["vsys"].n_circuits,
    )
    ctx = multiprocessing.get_context("fork")
    t0 = time.perf_counter()
    jobs = [(args.seed0 + k, cfg_gen) for k in range(args.n_sequences)]
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
        seqs = [s for s in pool.map(generate_skin_sequence, jobs) if s is not None]
    logger.info(
        "generated %d/%d skin-truth sequences in %.0f s "
        "(poloidal-annihilation median %.3f; ledger closure max %.2e; "
        "flips at Ip-frac %s)",
        len(seqs),
        len(jobs),
        time.perf_counter() - t0,
        float(np.median([a for s in seqs for a in s["annihilated_frac"]])),
        max(s["ledger"]["closure_frac"] for s in seqs),
        [None if s["flip_frac"] is None else round(s["flip_frac"], 2) for s in seqs],
    )
    if len(seqs) < 2:
        raise SystemExit("not enough sequences generated")
    tune_seqs = seqs[: args.n_tune]
    eval_seqs = seqs[args.n_tune :] or tune_seqs

    # --- spine arm (leg a) -------------------------------------------------
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
        spine_fits = list(pool.map(fit_sequence_spine, [(s, {}) for s in seqs]))
    spine_rows = [score_fits(s, f) for s, f in zip(seqs, spine_fits, strict=True)]

    # --- tune protocol (ORACLE-FREE — transferable verbatim to S2): first
    # identify the closure eta by misfit scan at a fixed scan weight, then
    # scan the prior weight AT the closure eta.  The oracle eta never enters
    # any tuning choice; it is an eval-side diagnostic only. ------------------
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
    w_scan = 0.5  # fixed scan weight for the eta identification
    eta_scan_result = {}
    for s_eta in eta_scales:
        eta_vec = [eta_true[0] + np.log10(s_eta), 0.0, 1.0]  # flat contrast
        cfg = {"eta_arm": eta_vec, "prior_weight": w_scan, "k_modes": args.k_modes}
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

    weight_scan = {}
    for w in weights:
        cfg = {"eta_arm": eta_closure, "prior_weight": w, "k_modes": args.k_modes}
        rows = _dyn_rows(tune_seqs, tune_fits, cfg)
        med = _med(
            [r.get("lcfs_cm") for rs in rows for r in rs if r["regime"] == "ramp"]
        )
        weight_scan[w] = med
        logger.info("weight scan w=%g → tune ramp LCFS median %.2f cm", w, med)
    w_frozen = min(weight_scan, key=lambda k: weight_scan[k])
    logger.info("frozen prior weight: %g", w_frozen)

    # --- eval arms at the frozen weight: closure (PRIMARY), oracle (diag),
    #     and the mode-build contrast sensitivity at the closure scale -------
    eval_fits = spine_fits[args.n_tune :] or tune_fits
    dyn_closure_rows = _dyn_rows(
        eval_seqs,
        eval_fits,
        {"eta_arm": eta_closure, "prior_weight": w_frozen, "k_modes": args.k_modes},
    )
    dyn_oracle_rows = _dyn_rows(
        eval_seqs,
        eval_fits,
        {"eta_arm": eta_true, "prior_weight": w_frozen, "k_modes": args.k_modes},
    )
    eta_contrast_closure = [
        eta_true[0] + np.log10(s_closure),
        eta_true[1],
        eta_true[2],
    ]
    dyn_contrast_rows = _dyn_rows(
        eval_seqs,
        eval_fits,
        {
            "eta_arm": eta_contrast_closure,
            "prior_weight": w_frozen,
            "k_modes": args.k_modes,
        },
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
        # per-band structure vs the measured hotspot (12.5 cm in transition)
        "lcfs_cm_band_limited": _med(_band(all_spine, "lcfs_cm", "limited")),
        "lcfs_cm_band_transition": _med(_band(all_spine, "lcfs_cm", "transition")),
        "lcfs_cm_band_diverted": _med(_band(all_spine, "lcfs_cm", "diverted")),
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

    # --- leg (b) verdict (eval sequences, frozen weight; PRIMARY = closure) -
    def _flat(rows_list):
        return [r for rs in rows_list for r in rs]

    sp_e, dy_o, dy_c, dy_x = (
        _flat(spine_eval_rows),
        _flat(dyn_oracle_rows),
        _flat(dyn_closure_rows),
        _flat(dyn_contrast_rows),
    )
    ramp_sp = _med(_regime(sp_e, "lcfs_cm", "ramp"))
    hold_sp = _med(_regime(sp_e, "lcfs_cm", "hold"))
    ramp_dyn_o = _med(_regime(dy_o, "lcfs_cm", "ramp"))
    ramp_dyn_c = _med(_regime(dy_c, "lcfs_cm", "ramp"))
    ramp_dyn_x = _med(_regime(dy_x, "lcfs_cm", "ramp"))
    hold_dyn_c = _med(_regime(dy_c, "lcfs_cm", "hold"))
    floor = hold_sp
    gap = ramp_sp - floor
    recovery_c = (ramp_sp - ramp_dyn_c) / gap if gap > 0 else float("nan")
    recovery_o = (ramp_sp - ramp_dyn_o) / gap if gap > 0 else float("nan")

    def _paired(a_rows, b_rows):
        return np.asarray(
            [
                s["lcfs_cm"] - d["lcfs_cm"]
                for s, d in zip(a_rows, b_rows, strict=True)
                if s["regime"] == "ramp" and s.get("scored") and d.get("scored")
            ]
        )

    paired_c = _paired(sp_e, dy_c)
    paired_o = _paired(sp_e, dy_o)
    ci_c = _paired_bootstrap_ci(paired_c)
    ci_o = _paired_bootstrap_ci(paired_o)
    axis_sp = _med([r.get("axis_cm") for r in sp_e if r.get("scored")])
    axis_dyn_c = _med([r.get("axis_cm") for r in dy_c if r.get("scored")])
    leg_b = {
        "prior_weight_frozen": w_frozen,
        "weight_scan_tune": {str(k): v for k, v in weight_scan.items()},
        "eta_closure_scale": s_closure,
        "eta_scan_tune_cost": {str(k): v for k, v in eta_scan_result.items()},
        "lcfs_cm_ramp_spine": ramp_sp,
        "lcfs_cm_ramp_dyn_closure": ramp_dyn_c,
        "lcfs_cm_ramp_dyn_oracle": ramp_dyn_o,
        "lcfs_cm_ramp_dyn_contrast_at_closure_scale": ramp_dyn_x,
        "lcfs_cm_hold_spine": hold_sp,
        "lcfs_cm_hold_dyn_closure": hold_dyn_c,
        "gap_cm": gap,
        "recovery_fraction_closure": recovery_c,
        "recovery_fraction_oracle_diag": recovery_o,
        "paired_ramp_gain_cm_median_closure": _med(paired_c),
        "paired_ramp_gain_ci_closure": list(ci_c),
        "paired_ramp_gain_cm_median_oracle": _med(paired_o),
        "paired_ramp_gain_ci_oracle": list(ci_o),
        "axis_cm_spine": axis_sp,
        "axis_cm_dyn_closure": axis_dyn_c,
        # PRIMARY (re-declared): the closure-identified eta IS the operating
        # point the real-data rung runs — recovery bar unchanged at 0.5,
        # paired CI must clear zero
        "B1_recovers_half_gap_closure": bool(recovery_c >= 0.5 and ci_c[0] > 0.0),
        # DIAGNOSTIC: the oracle-eta arm (the S1 attribution check)
        "B2_oracle_recovery_diag": recovery_o,
        "diag_hold_non_inferior": bool(hold_dyn_c <= hold_sp + 0.05),
        "diag_axis_not_degraded": bool(axis_dyn_c <= 1.05 * axis_sp),
    }
    leg_b["PASS"] = bool(leg_b["B1_recovers_half_gap_closure"])

    # --- pre-declared completeness assertions -------------------------------
    l_eff_spans = []
    for s in seqs:
        le = np.asarray(s["ledger"]["l_eff_wb_per_a"], dtype=np.float64)
        le = le[np.isfinite(le) & (le != 0.0)]
        if le.size >= 2:
            l_eff_spans.append(float(np.ptp(le) / np.abs(le).mean()))
    eval_flips = [s["flip_frac"] for s in eval_seqs]
    assertions = {
        "C1_ledger_closure_frac_max": max(s["ledger"]["closure_frac"] for s in seqs),
        "C1_flux_ledger_closes": bool(
            max(s["ledger"]["closure_frac"] for s in seqs) < 0.01
        ),
        "C2_flip_fracs_eval": [None if f is None else float(f) for f in eval_flips],
        "C2_transition_band_present": bool(
            all(f is not None and 0.55 <= f <= 0.97 for f in eval_flips)
        ),
        "C3_all_dynamics_folded": True,  # coupled truth chain + vessel-driven
        # trajectory are structural in this build (see module docstring)
        "C4_l_eff_rel_span_median": float(np.median(l_eff_spans))
        if l_eff_spans
        else float("nan"),
        "C4_inductance_evolves": bool(
            l_eff_spans and float(np.median(l_eff_spans)) >= 0.05
        ),
        "ledgers": [
            {k: (v if not isinstance(v, list) else v) for k, v in s["ledger"].items()}
            for s in seqs
        ],
    }
    assertions["PASS"] = bool(
        assertions["C1_flux_ledger_closes"]
        and assertions["C2_transition_band_present"]
        and assertions["C4_inductance_evolves"]
    )

    result = {
        "arm": "plasma-screening-synthetic-skin-gate-breakdown-start",
        "campaign_shot": CAMPAIGN_SHOT,
        "eta_true": eta_true,
        "k_modes": args.k_modes,
        "n_sequences": len(seqs),
        "n_tune": args.n_tune,
        "n_slices": sum(len(rs) for rs in spine_rows),
        "truth_tiling": [TRUTH_N_RAD, TRUTH_N_POL],
        "fit_tiling": [FIT_N_RAD, FIT_N_POL],
        "beta_split_truth": args.beta_split,
        "quad_max": args.quad_max,
        "div_max": args.div_max,
        "boost_max": args.boost_max,
        "frac_sat": args.frac_sat,
        "frac_bd": args.frac_bd,
        "n_vac": args.n_vac,
        "n_pre": args.n_pre,
        "poloidal_annihilation_median": float(
            np.median([a for s in seqs for a in s["annihilated_frac"]])
        ),
        "declared": {
            "primary": "B1 recovery >= 0.5 at closure-identified eta, paired CI "
            "clear of zero (the S2 operating point); oracle eta diagnostic only",
            "assertions": "C1 ledger closes <1% (no free constant, breakdown "
            "start), C2 limited->diverted flip in every eval sequence within "
            "Ip-frac [0.55, 0.97], C3 coupled plasma+vessel dynamics both "
            "sides, C4 plasma inductance evolves >=5% over the chain",
            "declared_before_run": "2026-07-18",
            "scenario_limitation_recorded_before_run": "8 design probes measured "
            "that no confined configuration of this forward operator reads "
            "diverted (saddles ~1 m out-limiter; real diverted coil pattern has "
            "no confined fixed point even minus sol/cases) — C2 is expected to "
            "FAIL for scenario reasons and A2 remains untestable as measured; "
            "unblock = coil-model attractor fix or free-boundary truth chain",
        },
        "leg_a_reproduction": leg_a,
        "leg_b_recovery": leg_b,
        "assertions": assertions,
        "PASS": bool(leg_a["PASS"] and leg_b["PASS"] and assertions["PASS"]),
        "measured_reference": {
            "lcfs_cm_rampup_median": 8.44,
            "lcfs_cm_flattop_median": 2.96,
            "lcfs_cm_transition_band": 12.5,
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
        "leg (a) %s | leg (b) %s (closure recovery %.2f CI [%.2f, %.2f]; oracle "
        "diag %.2f) | assertions %s | %s",
        "PASS" if leg_a["PASS"] else "FAIL",
        "PASS" if leg_b["PASS"] else "FAIL",
        leg_b["recovery_fraction_closure"],
        ci_c[0],
        ci_c[1],
        leg_b["recovery_fraction_oracle_diag"],
        "PASS" if assertions["PASS"] else "FAIL",
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
    fig, ax = plt.subplots(2, 3, figsize=(15.5, 8.4))
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

    bands = ["limited", "transition", "diverted"]
    band_meds = [leg_a.get(f"lcfs_cm_band_{b}", float("nan")) for b in bands]
    ax[0, 2].bar(bands, band_meds, color=["#4477aa", "#cc3311", "#228833"])
    ax[0, 2].axhline(12.5, color="k", ls=":", lw=1, label="measured transition 12.5")
    ax[0, 2].set_ylabel("spine LCFS error median [cm]")
    ax[0, 2].set_title("per-band structure (truth class read)")
    ax[0, 2].legend(fontsize=8)

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

    # class flip locations
    flips = [s["flip_frac"] for s in seqs]
    ax[1, 2].scatter(
        range(len(flips)),
        [f if f is not None else np.nan for f in flips],
        s=40,
        color="#228833",
    )
    ax[1, 2].axhspan(0.75, 0.85, color="#cccccc", alpha=0.5, label="measured hotspot")
    ax[1, 2].set_xlabel("sequence")
    ax[1, 2].set_ylabel("limited→diverted flip Ip-frac")
    ax[1, 2].set_ylim(0.4, 1.05)
    ax[1, 2].set_title("transition band location")
    ax[1, 2].legend(fontsize=8)
    fig.suptitle(
        f"leg (a) — failure-signature reproduction: "
        f"{'PASS' if leg_a['PASS'] else 'FAIL'}"
    )
    fig.tight_layout()
    fig.savefig(FIGURES / f"fig-skin-gate-signature{tag}.png", dpi=120)
    plt.close(fig)

    # (2) recovery figure — PRIMARY arm is the closure-eta fit
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.2))
    sp_e = [r for rs in spine_eval_rows for r in rs]
    dy_o = [r for rs in dyn_oracle_rows for r in rs]
    dy_c = [r for rs in dyn_closure_rows for r in rs]
    pairs_c = [
        (s["lcfs_cm"], d["lcfs_cm"])
        for s, d in zip(sp_e, dy_c, strict=True)
        if s["regime"] == "ramp" and s.get("scored") and d.get("scored")
    ]
    ax[0].scatter(
        [a for a, _ in pairs_c],
        [b for _, b in pairs_c],
        s=20,
        alpha=0.7,
        color="#228833",
    )
    lim = [0, max([a for a, _ in pairs_c] + [b for _, b in pairs_c] + [1.0]) * 1.05]
    ax[0].plot(lim, lim, "k--", lw=1)
    ax[0].set_xlabel("spine ramp LCFS error [cm]")
    ax[0].set_ylabel("dynamic-mode (closure η) [cm]")
    ax[0].set_title(
        f"closure recovery {leg_b['recovery_fraction_closure']:.2f} — B1 "
        f"{'PASS' if leg_b['B1_recovers_half_gap_closure'] else 'FAIL'}"
    )
    bins = np.linspace(0, lim[1], 20)
    ax[1].hist(
        [a for a, _ in pairs_c], bins=bins, alpha=0.55, label="spine", color="#cc3311"
    )
    ax[1].hist(
        [b for _, b in pairs_c],
        bins=bins,
        alpha=0.55,
        label="dyn (closure η)",
        color="#228833",
    )
    ramp_o = [
        d["lcfs_cm"]
        for s, d in zip(sp_e, dy_o, strict=True)
        if s["regime"] == "ramp" and d.get("scored")
    ]
    ax[1].hist(
        ramp_o, bins=bins, alpha=0.4, label="dyn (oracle η, diag)", color="#4477aa"
    )
    ax[1].set_xlabel("ramp LCFS error [cm]")
    ax[1].legend(fontsize=8)
    ax[1].set_title(
        f"closure-η paired gain {leg_b['paired_ramp_gain_cm_median_closure']:+.2f} cm"
        f" CI [{leg_b['paired_ramp_gain_ci_closure'][0]:+.2f},"
        f" {leg_b['paired_ramp_gain_ci_closure'][1]:+.2f}]"
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
    xt = np.asarray(r0row.get("x_true", []), dtype=np.float64).reshape(-1, 2)
    if xt.size:
        ax[1].plot(xt[:, 0], xt[:, 1], "kx", ms=10, mew=2, label="truth X-point")
    ax[1].set_aspect("equal")
    ax[1].set_xlabel("R [m]")
    ax[1].set_ylabel("Z [m]")
    ax[1].set_title(f"representative ramp slice ({r0row.get('class_true', '')})")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / f"fig-skin-gate-example{tag}.png", dpi=120)
    plt.close(fig)

    # (4) dynamics-completeness figure: the flux ledger and dL_p/dt made
    # visible (the breakdown-start / all-terms assertions)
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.2))
    terms = ["vs_drive", "delta_psi_bar", "resistive_vs", "formation_vs", "remap_vs"]
    term_labels = ["∫u dt", "ΔΨ̄ (dΦ/dt)", "∫R·i dt", "formation", "remap (dL/dt)"]
    vals = np.asarray([[s["ledger"][t] for t in terms] for s in seqs], dtype=np.float64)
    x = np.arange(len(terms))
    for i in range(vals.shape[0]):
        ax[0].plot(x, vals[i], "o-", alpha=0.6, lw=1)
    ax[0].set_xticks(x, term_labels, rotation=20, fontsize=8)
    ax[0].set_ylabel("volt-seconds [Wb]")
    closure_max = max(s["ledger"]["closure_frac"] for s in seqs)
    ax[0].set_title(f"flux ledger per sequence (closure max {closure_max:.1e})")

    for s in seqs:
        le = np.asarray(s["ledger"]["l_eff_wb_per_a"], dtype=np.float64) * 1e6
        ax[1].plot(np.arange(le.size), le, "-", alpha=0.6, lw=1)
    ax[1].set_xlabel("chain step (from breakdown catch)")
    ax[1].set_ylabel("patch-mean self-flux per ampere [μWb/A]")
    ax[1].set_title("plasma inductance evolves with the forming shape")

    for s in seqs:
        t_all = np.concatenate([s["vac_times"], s["pre_times"], s["times"]])
        ip_all = np.concatenate(
            [np.zeros(len(s["vac_times"])), s["pre_ip"], s["ip_seq"]]
        )
        ax[2].plot(t_all, np.asarray(ip_all) / 1e3, "-", alpha=0.6, lw=1)
    ax[2].axvspan(
        0.0,
        float(np.asarray(seqs[0]["pre_times"])[0]),
        color="#dddddd",
        alpha=0.6,
        label="vacuum",
    )
    ax[2].axvspan(
        float(np.asarray(seqs[0]["pre_times"])[0]),
        float(np.asarray(seqs[0]["times"])[0]),
        color="#ffeecc",
        alpha=0.6,
        label="breakdown → label floor",
    )
    ax[2].set_xlabel("t [s]")
    ax[2].set_ylabel("Ip [kA]")
    ax[2].set_title("the stream integrates from the quiescent state")
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / f"fig-skin-gate-dynamics{tag}.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
