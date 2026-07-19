"""Temporal equilibrium operator: causal sequence model over sensor history.

The temporal rung of the learned-equilibrium ladder.  A causal transformer
trunk attends over per-slice sensor-token codes (the same geometry-encoded
featurizer as :mod:`imas_ambix.latent.residual_operator`) and two heads emit
only physics-degenerate DOF:

* profile-DOF corrections ``dc`` about the classical solution, decoded through
  the exact Green's layer (:class:`ProfileGreensDecoder`) — identical contract
  to the static operator;
* vessel-eddy mode amplitudes ``da`` that enter the field as MORE EXTERNAL
  CURRENTS through the exact passive Green's columns — the boundary push-out
  readout is unchanged by construction.

The eddy latent is physics-structured, not free: the passive conductors'
L/R eigenmodes (inductance from the finite-area cylinder kernels, resistance
from the conductor cross-sections at a nominal steel resistivity) define a
diagonal state-space block whose per-mode decay constants are LEARNABLE but
initialised at the physical L/R times, and whose drive is the physically
computable flux swing the coil and plasma histories induce in each mode.  The
eddy state at time t is therefore an L/R convolution of the current history —
exactly the quantity the per-slice static fit provably cannot see.

Zero-initialised output heads: the untrained operator reproduces the classical
spine byte-exactly (``dc = 0``, ``da = 0``), so training starts at parity and
every gain is attributable to the learned temporal structure.

Conventions: total poloidal flux Φ = 2πR·A_φ [Wb]; thick-cylinder
finite-area Green's kernels throughout (never point-filament).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn

from imas_ambix.latent.residual_operator import TOKEN_FEATURES

if TYPE_CHECKING:
    from pathlib import Path

    from imas_ambix.gs.geometry import GeometryTable
    from imas_ambix.latent.gs_solve import EquilibriumGrid

logger = logging.getLogger(__name__)

#: nominal stainless-steel resistivity [Ω·m] for the vessel L/R times — the
#: decay constants are learnable, so this sets the INITIALISATION scale only
STEEL_RESISTIVITY = 7.2e-7


# ---------------------------------------------------------------------------
# Passive L/R eigenbasis — pure geometry (+ nominal resistivity), per campaign
# ---------------------------------------------------------------------------
@dataclass
class PassiveEigenbasis:
    """L/R eigenmodes of the passive conductors, reduced to the k modes kept.

    ``tau`` are the physical decay times [s]; ``v`` (n_passive, k) is the
    L-orthonormal eigenvector block; ``a_sens`` (S, k) / ``g_grid`` (nz·nr, k)
    map a mode amplitude to sensors / grid flux; ``m_coil`` (k, C) and
    ``m_cells`` (k, n_cells) are the flux linkages each mode picks up per
    ampere of coil channel / plasma cell current — the eddy DRIVE couplings.
    Mode amplitudes are in the L-orthonormal eigencoordinates throughout.
    """

    tau: np.ndarray
    v: np.ndarray
    a_sens: np.ndarray
    g_grid: np.ndarray
    m_coil: np.ndarray
    m_cells: np.ndarray
    resistivity: float
    #: (k, C) voltage-drive couplings [Ω] — the galvanic case-wiring EMF per
    #: ampere of drive channel current (``volt_m = i_pf @ volt_coil.T``);
    #: None when the basis carries no discovered galvanic terms
    volt_coil: np.ndarray | None = None

    @property
    def n_modes(self) -> int:
        return int(self.tau.size)


def _passive_circuit_filaments(table: GeometryTable) -> list[list]:
    """Filament groups of every ``inferred_passive`` circuit (sorted order)."""
    from imas_ambix.gs import operator as op  # noqa: PLC0415

    classes = op.classify_circuits(table.pf_filaments, table.amc_current_channels)
    passive = sorted(c.circuit for c in classes if c.role == "inferred_passive")
    by_circ: dict[int, list] = {}
    for f in table.pf_filaments:
        by_circ.setdefault(f.circuit, []).append(f)
    return [by_circ[c] for c in passive]


@dataclass
class PassiveCircuitSystem:
    """Circuit-space L/R system of the passive set, before eigen-reduction.

    The resistance DOF live HERE: ``r_diag`` is diagonal in circuit space
    (toroidal rings share no conductor path), so a candidate resistance model
    is a per-circuit multiplier vector applied to ``r_diag`` followed by one
    cheap generalised eigensolve — the geometry-exact ``lmat`` and all
    couplings never change.  ``case_channel_row`` is non-empty only when the
    measured-case circuits were moved INTO the passive set (their measured
    ``*_case_current`` channels then supply held-back per-circuit targets,
    never drives — ``coil_channels`` excludes them).
    """

    circuits: np.ndarray  # (P,) sorted circuit ids
    centroid_r: np.ndarray  # (P,) xmult-weighted circuit centroids [m]
    centroid_z: np.ndarray  # (P,)
    lmat: np.ndarray  # (P, P) SPD-guarded two-section flux linkage [Wb/A]
    r_diag: np.ndarray  # (P,) nominal toroidal-ring resistances [Ω]
    a_circ: np.ndarray  # (S, P) per-ampere sensor couplings
    g_circ: np.ndarray  # (nz·nr, P) per-ampere grid flux couplings
    m_coil_circ: np.ndarray  # (P, C) coil-channel flux linkage [Wb/A]
    coil_channels: list[str]  # C driven channel names (sorted)
    case_channel_row: dict[str, int]  # held-back case channel -> circuit row
    resistivity: float
    #: (P,) per-circuit conducting cross-section scale sqrt(Σ|w·h|) [m] — the
    #: geometric size that normalises the adjacency-neighbour rule; None on
    #: systems cached before the field existed (rebuild the cache to use it)
    section_scale: np.ndarray | None = None

    @property
    def n_circuits(self) -> int:
        return int(self.circuits.size)


def build_passive_circuit_system(
    table: GeometryTable,
    grid: EquilibriumGrid,
    *,
    resistivity: float = STEEL_RESISTIVITY,
    section_scale_frac: float = 1.0,
    section_n_max: int = 6,
    hold_back_cases: bool = False,
) -> PassiveCircuitSystem:
    """Circuit-space L, R, and couplings of the passive set — pure geometry.

    Inductance and resistance exactly as :func:`build_passive_eigenbasis`
    documents (two-section gridded linkage, true-area ring resistance).  With
    ``hold_back_cases`` the measured-case circuits (``known_case`` role) are
    re-classified into the passive set: their currents become STATE predicted
    from the remaining drives through the mutual couplings, their measured
    channels leave ``coil_channels``, and ``case_channel_row`` records which
    circuit row each held-back channel measures.
    """
    from imas_ambix.gs import operator as op  # noqa: PLC0415
    from imas_ambix.gs.cylinder import hybrid_greens  # noqa: PLC0415
    from imas_ambix.latent.boundary_disc import (  # noqa: PLC0415
        passive_coupling_matrices,
    )

    classes = op.classify_circuits(table.pf_filaments, table.amc_current_channels)
    passive_roles = (
        ("inferred_passive", "known_case") if hold_back_cases else ("inferred_passive",)
    )
    passive = sorted(c.circuit for c in classes if c.role in passive_roles)
    if not passive:
        raise ValueError("table has no passive circuits under the selected roles")
    by_circ: dict[int, list] = {}
    for f in table.pf_filaments:
        by_circ.setdefault(f.circuit, []).append(f)
    groups = [by_circ[c] for c in passive]
    n_pass = len(groups)

    case_channel_row = {
        c.amc_channel: passive.index(c.circuit)
        for c in classes
        if hold_back_cases and c.role == "known_case"
    }
    class_of = {c.circuit: c for c in classes}
    centroid_r = np.array([class_of[c].centroid_r for c in passive])
    centroid_z = np.array([class_of[c].centroid_z for c in passive])
    circuit_scale = np.array(
        [np.sqrt(sum(abs(f.width * f.height) for f in g)) for g in groups]
    )

    section_delta = section_scale_frac * _median_section_scale(groups)
    pr, pz, wt, owner = _section_grid(groups, section_delta, section_n_max)
    lmat = np.zeros((n_pass, n_pass))
    for j, gj in enumerate(groups):
        lmat[:, j] = _linked_flux_columns(gj, pr, pz, wt, owner, n_pass, hybrid_greens)
    # the analytic source + quadrature observer linkage is symmetric up to
    # observer-quadrature error — symmetrise, then guard SPD (physical L is)
    lmat = 0.5 * (lmat + lmat.T)
    w0, u0 = np.linalg.eigh(lmat)
    lmat = (u0 * np.clip(w0, 1e-4 * w0.max(), None)) @ u0.T

    r_diag = np.array(
        [
            sum(
                2.0
                * np.pi
                * f.r
                * resistivity
                / max(abs(f.width) * abs(f.height), 1e-8)
                * f.xmult**2
                for f in g
            )
            for g in groups
        ]
    )

    a_circ, g_circ = passive_coupling_matrices(grid, table, circuits=passive)

    # coil channel → circuit flux linkage, mirroring build_operator's channel
    # merge (average same-channel circuits; solenoid response scale applied);
    # held-back case channels never become drive columns
    pf_by_chan: dict[str, list[int]] = {}
    for cc in classes:
        if cc.role in op._KNOWN_ROLES:  # noqa: SLF001 — canonical role list
            if hold_back_cases and cc.role == "known_case":
                continue
            pf_by_chan.setdefault(cc.amc_channel, []).append(cc.circuit)
    coil_channels = sorted(pf_by_chan)
    m_coil_circ = np.zeros((n_pass, len(coil_channels)))
    for c_idx, chan in enumerate(coil_channels):
        cols = []
        for circ in sorted(pf_by_chan[chan]):
            cols.append(
                _linked_flux_columns(
                    by_circ[circ], pr, pz, wt, owner, n_pass, hybrid_greens
                )
            )
        col = np.mean(np.asarray(cols), axis=0)
        if chan == "sol_current":
            col = col * op.SOLENOID_RESPONSE_SCALE
        m_coil_circ[:, c_idx] = col

    return PassiveCircuitSystem(
        circuits=np.asarray(passive, dtype=np.int64),
        centroid_r=centroid_r,
        centroid_z=centroid_z,
        lmat=lmat,
        r_diag=r_diag,
        a_circ=a_circ,
        g_circ=g_circ,
        m_coil_circ=m_coil_circ,
        coil_channels=coil_channels,
        case_channel_row=case_channel_row,
        resistivity=float(resistivity),
        section_scale=circuit_scale,
    )


def build_drive_linkage(
    table: GeometryTable,
    *,
    section_scale_frac: float = 1.0,
    section_n_max: int = 6,
) -> tuple[list[str], np.ndarray]:
    """Flux linkage among the measured drive circuits, self terms included.

    Returns ``(channels, lam)`` with ``lam[i, j]`` the flux linked by channel
    ``i``'s circuit per ampere(-turn) of channel ``j``'s current [Wb/A] —
    the same two-section linkage as the passive system (finite-area source,
    section-averaged observer, machine-agnostic subdivision; the kernel is
    smooth inside conductors so the self term needs no special-casing).
    Same-channel redundant circuits are averaged exactly as
    :func:`build_passive_circuit_system` merges them; the solenoid response
    scale applies on the source side only.  The case-wiring discovery reads
    the winding rows: the flux a coil circuit links from every measured
    drive is the inductive part of its terminal voltage.
    """
    from imas_ambix.gs import operator as op  # noqa: PLC0415
    from imas_ambix.gs.cylinder import hybrid_greens  # noqa: PLC0415

    classes = op.classify_circuits(table.pf_filaments, table.amc_current_channels)
    by_circ: dict[int, list] = {}
    for f in table.pf_filaments:
        by_circ.setdefault(f.circuit, []).append(f)
    by_chan: dict[str, list[int]] = {}
    for cc in classes:
        if cc.role in op._KNOWN_ROLES:  # noqa: SLF001 — canonical role list
            by_chan.setdefault(cc.amc_channel, []).append(cc.circuit)
    channels = sorted(by_chan)
    groups = [
        [f for circ in sorted(by_chan[ch]) for f in by_circ[circ]] for ch in channels
    ]
    delta = section_scale_frac * _median_section_scale(groups)
    pr, pz, wt, owner = _section_grid(groups, delta, section_n_max)
    n = len(channels)
    lam = np.zeros((n, n))
    for j, chan in enumerate(channels):
        cols = []
        for circ in sorted(by_chan[chan]):
            cols.append(
                _linked_flux_columns(by_circ[circ], pr, pz, wt, owner, n, hybrid_greens)
            )
        col = np.mean(np.asarray(cols), axis=0)
        if chan == "sol_current":
            col = col * op.SOLENOID_RESPONSE_SCALE
        lam[:, j] = col
    # merged-observer normalisation: an averaged same-channel merge must also
    # average (not sum) on the observer side, mirroring the source merge
    for i, chan in enumerate(channels):
        n_merge = len(by_chan[chan])
        if n_merge > 1:
            lam[i, :] /= n_merge
    return channels, lam


def predict_vessel_currents(
    table: GeometryTable,
    system: PassiveCircuitSystem,
    i_pf_full: np.ndarray,
    channels: list[str],
    times: np.ndarray,
    *,
    ip_amperes: np.ndarray | None = None,
    axis_rz: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Vessel circuit currents from the measured drives, quiescent start.

    Exact-ZOH eigenmode integration of the passive circuit system driven by
    the full measured coil history (``i_pf_full`` (T, C) in ``channels``
    order; channels the system does not carry contribute zero).  When
    ``ip_amperes`` (T,) and ``axis_rz`` (T, 2) are given, the PLASMA
    current's own flux swing drives the vessel too — a toroidal filament at
    the given axis trace: the Lenz-anti-parallel image currents this induces
    during a fast Ip ramp contribute vertical field at the plasma in the
    CONFINING direction while they last, and decay on the vessel L/R times
    once the ramp holds — a drive term absent from every coil-only
    prediction.  Non-finite / zero-Ip samples contribute no plasma linkage
    (breakdown-start convention: the pre-plasma vessel state is coil-driven
    only).

    Returns ``(i_coil (T, P), i_full (T, P))`` — the coil-only and the
    coil+plasma-driven circuit states in ``system.circuits`` order
    (identical when the plasma drive is omitted).  The flux a downstream
    consumer injects is ``system.g_circ @ i_full[t]`` (grid) or
    ``system.a_circ @ i_full[t]`` (sensors).
    """
    from scipy.linalg import eigh  # noqa: PLC0415

    from imas_ambix.gs.operator import greens_psi  # noqa: PLC0415

    times = np.asarray(times, dtype=np.float64)
    i_pf_full = np.asarray(i_pf_full, dtype=np.float64)
    chan_idx = {ch: j for j, ch in enumerate(system.coil_channels)}
    m_vc = np.zeros((system.n_circuits, len(channels)))
    for j, chan in enumerate(channels):
        if chan in chan_idx:
            m_vc[:, j] = system.m_coil_circ[:, chan_idx[chan]]
    w, vv = eigh(np.diag(system.r_diag), system.lmat)
    tau = 1.0 / np.clip(w, 1e-12, None)
    psi_coil = i_pf_full @ m_vc.T
    a_coil, _u = integrate_eddy_ode(tau, times, psi_coil @ vv)
    i_coil = a_coil @ vv.T
    if ip_amperes is None or axis_rz is None:
        return i_coil, i_coil

    # plasma→circuit linkage per slice: xmult-weighted ψ per ampere of a
    # loop at the (time-varying) axis, vectorised over all passive filaments
    ip_amperes = np.asarray(ip_amperes, dtype=np.float64)
    axis_rz = np.asarray(axis_rz, dtype=np.float64)
    by_circ: dict[int, list] = {}
    for f in table.pf_filaments:
        by_circ.setdefault(f.circuit, []).append(f)
    fr, fz, fw, ci = [], [], [], []
    for i, c in enumerate(system.circuits):
        for f in by_circ[int(c)]:
            fr.append(f.r)
            fz.append(f.z)
            fw.append(f.xmult)
            ci.append(i)
    fr = np.array(fr)
    fz = np.array(fz)
    fw = np.array(fw)
    ci = np.array(ci)
    psi_plasma = np.zeros((times.size, system.n_circuits))
    for t in range(times.size):
        if (
            not np.isfinite(ip_amperes[t])
            or ip_amperes[t] == 0.0
            or not np.all(np.isfinite(axis_rz[t]))
        ):
            continue
        psi = np.atleast_1d(
            greens_psi(fr, fz, float(axis_rz[t, 0]), float(axis_rz[t, 1]))
        )
        psi_plasma[t] = np.bincount(
            ci, weights=fw * psi, minlength=system.n_circuits
        ) * float(ip_amperes[t])
    a_full, _u = integrate_eddy_ode(tau, times, (psi_coil + psi_plasma) @ vv)
    return i_coil, a_full @ vv.T


def reduce_passive_system(
    system: PassiveCircuitSystem,
    grid: EquilibriumGrid,
    *,
    sensor_scale: np.ndarray,
    k: int = 12,
    r_multipliers: np.ndarray | None = None,
) -> PassiveEigenbasis:
    """Eigen-reduce a circuit system to the k most history-relevant modes.

    ``r_multipliers`` (P,) scales the diagonal circuit resistances — the
    calibrated-resistance hook: the geometry-exact L and every coupling stay
    fixed while the data-led resistance model reshapes the eigenmodes.  With
    ``None`` (nominal R) this reproduces :func:`build_passive_eigenbasis`
    exactly.
    """
    from scipy.linalg import eigh  # noqa: PLC0415

    r_diag = system.r_diag
    if r_multipliers is not None:
        mult = np.asarray(r_multipliers, dtype=np.float64)
        if mult.shape != r_diag.shape:
            raise ValueError(
                f"r_multipliers shape {mult.shape} != circuits {r_diag.shape}"
            )
        if np.any(~np.isfinite(mult)) or np.any(mult <= 0):
            raise ValueError("r_multipliers must be finite and positive")
        r_diag = r_diag * mult
    w, v = eigh(np.diag(r_diag), system.lmat)  # R v = (1/τ) L v ; v L-orthonormal
    tau = 1.0 / np.clip(w, 1e-12, None)

    a_modes = system.a_circ @ v  # (S, n_pass)
    scale = np.clip(np.asarray(sensor_scale, dtype=np.float64), 1e-12, None)
    relevance = tau * np.linalg.norm(a_modes / scale[:, np.newaxis], axis=0)
    keep = np.argsort(relevance)[::-1][: int(k)]
    keep = keep[np.argsort(tau[keep])[::-1]]  # slowest-first for readability

    v_k = v[:, keep]
    g_modes = system.g_circ @ v_k
    m_cells = g_modes[grid.cells, :].T  # reciprocity: (k, n_cells)
    m_coil = v_k.T @ system.m_coil_circ  # (k, C)

    return PassiveEigenbasis(
        tau=tau[keep],
        v=v_k,
        a_sens=a_modes[:, keep],
        g_grid=g_modes,
        m_coil=m_coil,
        m_cells=m_cells,
        resistivity=float(system.resistivity),
    )


def _median_section_scale(groups: list[list]) -> float:
    """Median cross-section scale ``sqrt(w·h)`` of a conductor set [m].

    The machine-intrinsic length that normalises the observer sub-gridding:
    a fixed metre-level cell size would bake one machine's conductor sizes
    into the linkage accuracy, so the subdivision criterion is expressed as a
    FRACTION of this scale and transfers across machines unchanged.
    """
    dims = [
        np.sqrt(max(abs(f.width), 1e-4) * max(abs(f.height), 1e-4))
        for g in groups
        for f in g
    ]
    return float(np.median(dims))


def _section_points(
    r: float, z: float, width: float, height: float, delta: float, n_max: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Midpoint sub-grid of a rectangular cross-section, equal-area weighted.

    Sections smaller than ``delta`` in a dimension stay unsubdivided there
    (small elements see uniform flux — centroid linking is exact enough); a
    larger section is split so sub-cells are ≤ ``delta``, capped at ``n_max``
    per dimension.  Weights are the uniform current shares ``1/n``.  The
    TRUE section extents place the points (a thin 3 mm case wall's observer
    points must not smear over the 0.01 m kernel floor — that floor guards
    only the source-side flux integration).
    """
    w = max(abs(width), 1e-4)
    h = max(abs(height), 1e-4)
    nw = max(1, min(int(np.ceil(w / delta)), n_max))
    nh = max(1, min(int(np.ceil(h / delta)), n_max))
    rr = r + w * ((np.arange(nw) + 0.5) / nw - 0.5)
    zz = z + h * ((np.arange(nh) + 0.5) / nh - 0.5)
    rg, zg = np.meshgrid(rr, zz)
    return rg.ravel(), zg.ravel(), np.full(rg.size, 1.0 / rg.size)


def _section_grid(
    groups: list[list], delta: float, n_max: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Concatenated cross-section sub-points of every filament group.

    Returns ``(r, z, weight, owner)`` — the weight folds each filament's
    ``xmult`` current share with its section quadrature weight; ``owner``
    maps each point back to its group index (for :func:`numpy.bincount`
    reduction).
    """
    pr: list[np.ndarray] = []
    pz: list[np.ndarray] = []
    wt: list[np.ndarray] = []
    owner: list[np.ndarray] = []
    for gi, g in enumerate(groups):
        for f in g:
            r_pts, z_pts, w_pts = _section_points(
                f.r, f.z, f.width, f.height, delta, n_max
            )
            pr.append(r_pts)
            pz.append(z_pts)
            wt.append(f.xmult * w_pts)
            owner.append(np.full(r_pts.size, gi, dtype=np.int64))
    return (
        np.concatenate(pr),
        np.concatenate(pz),
        np.concatenate(wt),
        np.concatenate(owner),
    )


def _linked_flux_columns(
    src_filaments: list,
    pr: np.ndarray,
    pz: np.ndarray,
    wt: np.ndarray,
    owner: np.ndarray,
    n_groups: int,
    greens,
) -> np.ndarray:
    """Flux linkage [Wb/A] of one source group into every observer group.

    The finite-area cylinder kernel integrates the SOURCE cross-section
    exactly; the observer side is the section-averaged flux over each
    filament's sub-grid (the two-section flux linkage the nova inductance
    solver computes by gridding source and target and integrating the linkage
    out — necessary for the larger vessel elements where the flux varies
    materially across the section).
    """
    col = np.zeros(n_groups)
    for fs in src_filaments:
        psi, _br, _bz = greens(
            pr,
            pz,
            float(fs.r),
            float(fs.z),
            max(abs(fs.width), 0.01),
            max(abs(fs.height), 0.01),
        )
        col += float(fs.xmult) * np.bincount(
            owner, weights=wt * psi, minlength=n_groups
        )
    return col


def build_passive_eigenbasis(
    table: GeometryTable,
    grid: EquilibriumGrid,
    *,
    sensor_scale: np.ndarray,
    k: int = 12,
    resistivity: float = STEEL_RESISTIVITY,
    section_scale_frac: float = 1.0,
    section_n_max: int = 6,
    r_multipliers: np.ndarray | None = None,
    hold_back_cases: bool = False,
) -> PassiveEigenbasis:
    """L/R eigenmode reduction of the passive set — pure geometry, per campaign.

    Inductance: two-section mutual flux linkage between passive circuits —
    the finite-area cylinder kernel integrates the source section exactly and
    the observer section is averaged over a midpoint sub-grid (the nova
    gridded source+target linkage; small sections stay centroid-linked).  The
    subdivision criterion is MACHINE-AGNOSTIC: sub-cells are sized at
    ``section_scale_frac`` × the passive set's median section scale
    ``median(sqrt(w·h))`` (capped at ``section_n_max`` per dimension) — a
    dimensionless rule that transfers to other machines, never a fixed
    metre-level cell.  L is known EXACTLY from geometry — it is a prior no
    learner should re-fit.  Resistance: toroidal-ring resistance
    ``2πR·ρ/(w·h)`` per filament at the TRUE cross-section (the 0.01 m kernel
    floor is a flux-integration guard and must never inflate a thin shell's
    conducting area — 3 mm case walls carry 3.3× the clamped resistance),
    combined with the ``xmult²`` current-share weights (parallel paths at
    fixed shares), at the nominal steel resistivity (a bounded cross-shot
    scale is the calibratable unknown).  Generalised eigenproblem
    ``R v = (1/τ) L v`` with L-orthonormal ``v``.

    Mode selection keeps the ``k`` modes with the largest history-relevance
    ``τ_m · ||a_sens_m / scale||`` — a slow mode the sensors can see is
    exactly a mode whose history the static fit cannot absorb.  Drive
    couplings by reciprocity: ``m_cells = g_grid[cells].T`` (flux a mode links
    per ampere of plasma cell current == flux the cell sees per mode ampere).

    ``r_multipliers`` (per passive circuit, sorted-circuit order) applies a
    calibrated resistance model on top of the nominal ring resistances —
    see :func:`reduce_passive_system`.  ``hold_back_cases`` moves the
    measured-case circuits into the passive set (their channels leave the
    ``m_coil`` drive columns) — see :func:`build_passive_circuit_system`.
    """
    system = build_passive_circuit_system(
        table,
        grid,
        resistivity=resistivity,
        section_scale_frac=section_scale_frac,
        section_n_max=section_n_max,
        hold_back_cases=hold_back_cases,
    )
    return reduce_passive_system(
        system, grid, sensor_scale=sensor_scale, k=k, r_multipliers=r_multipliers
    )


def integrate_eddy_ode(
    tau: np.ndarray,
    times: np.ndarray,
    psi_m: np.ndarray,
    a0: np.ndarray | None = None,
    volt_m: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact-ZOH integration of ``da/dt + a/τ = −dΨ/dt + v`` along a series.

    ``psi_m`` (T, k) is the external flux each mode links, taken
    piecewise-linear between samples, for which the per-step update is exact::

        a_k = e^{−Δt/τ} a_{k−1} − (τ/Δt)(1 − e^{−Δt/τ}) ΔΨ

    ``volt_m`` (T, k), when given, is a voltage-type mode drive (a galvanic
    EMF that is NOT the derivative of a linked flux — e.g. the resistive term
    of a case wired across its winding).  It is taken piecewise-constant at
    the step midpoint value ``(v_t + v_{t−1})/2``, for which the update adds
    ``τ(1 − e^{−Δt/τ}) v̄`` exactly.

    Returns ``(a (T, k), u (T, k))`` — the mode state and the per-step flux
    swing ``−ΔΨ``.  ``a[0] = a0`` (zeros when None).
    """
    tau = np.asarray(tau, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    psi_m = np.asarray(psi_m, dtype=np.float64)
    n_t, k = psi_m.shape
    a = np.zeros((n_t, k))
    u = np.zeros((n_t, k))
    if a0 is not None:
        a[0] = np.asarray(a0, dtype=np.float64)
    if volt_m is not None:
        volt_m = np.asarray(volt_m, dtype=np.float64)
    for t in range(1, n_t):
        dt = max(float(times[t] - times[t - 1]), 1e-6)
        decay = np.exp(-dt / tau)
        coeff = tau / dt * (1.0 - decay)
        dpsi = psi_m[t] - psi_m[t - 1]
        u[t] = -dpsi
        a[t] = decay * a[t - 1] + coeff * u[t]
        if volt_m is not None:
            vbar = 0.5 * (volt_m[t] + volt_m[t - 1])
            a[t] = a[t] + tau * (1.0 - decay) * vbar
    return a, u


def physical_eddy_history(
    basis: PassiveEigenbasis,
    times: np.ndarray,
    i_pf: np.ndarray,
    i_cell: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact-ZOH integration of the mode eddy ODE along a slice sequence.

    Mode dynamics (L-orthonormal coordinates): ``da/dt + a/τ = −dΨ/dt`` with
    ``Ψ_m(t) = m_coil_m · i_pf(t) + m_cells_m · i_cell(t)`` the external flux
    the mode links.  Returns ``(a_phys (T, k), u_drive (T, k))`` — the
    physical eddy state and the per-step flux swing ``−ΔΨ`` (the drive
    feature).  ``a_phys[0] = 0``: the first labelled slice is taken as the
    eddy reference (label sequences start above the Ip threshold; earlier
    transients are unobserved here — :func:`raw_eddy_trajectory` removes this
    approximation by integrating the raw-cadence drives from the stream
    start).
    """
    psi_m = (
        np.asarray(i_pf, dtype=np.float64) @ basis.m_coil.T
        + np.asarray(i_cell, dtype=np.float64) @ basis.m_cells.T
    )  # (T, k)
    return integrate_eddy_ode(basis.tau, times, psi_m)


def raw_eddy_trajectory(
    basis: PassiveEigenbasis,
    raw_times: np.ndarray,
    i_pf_raw: np.ndarray,
    label_times: np.ndarray,
    i_cell_labels: np.ndarray,
    *,
    ip_raw: np.ndarray | None = None,
    tau_scale: float | np.ndarray = 1.0,
    volt_m_raw: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Mode eddy state at the labelled slices from RAW-cadence integration.

    The mode flux is assembled at the raw cadence and the ODE integrated from
    the raw stream start with ``a = 0`` (the pre-drive machine is quiescent),
    so the labelled sequence inherits the full drive history — solenoid
    precharge, breakdown flux swing — instead of the ``a[0] = 0``
    label-cadence approximation.

    Coil term: ``m_coil · i_pf_raw(t)`` per raw sample (measured drives).
    Plasma term: the exact per-label mode flux ``m_cells · i_cell(t_label)``
    — the FULL time-varying current distribution, so the changing plasma
    shape / position / internal inductance enters the flux swing as
    ``d(M(t)·I_p(t))/dt``, never the fixed-mutual approximation
    ``M·dI_p/dt`` — linearly interpolated to the raw cadence (interpolation
    commutes with the fixed linear map ``m_cells``).  Before the first label
    the first label's flux pattern is amplitude-followed with the measured
    plasma current (``ip_raw``, same raw grid; shape-frozen); with no
    ``ip_raw`` the plasma term is zero-ramped from the raw start.

    ``tau_scale`` is the bounded resistance-scale DOF: a UNIFORM scalar leaves
    the L/R eigenvectors invariant and maps every τ → τ/scale exactly; a
    per-mode array is the diagonal-in-eigenbasis approximation to a structured
    resistivity change (bounded, calibrated cross-shot, never per-slice).

    ``volt_m_raw`` (T_raw, k) adds a voltage-type mode drive at the raw
    cadence (see :func:`integrate_eddy_ode`) — the galvanic case-wiring term
    the structure discovery may accept; flux-type wiring terms fold into the
    drive columns and need no extra argument.

    Returns ``(a_labels (T_lab, k), a_raw (T_raw, k))`` in the L-orthonormal
    mode coordinates.
    """
    raw_times = np.asarray(raw_times, dtype=np.float64)
    label_times = np.asarray(label_times, dtype=np.float64)
    psi_coil = np.asarray(i_pf_raw, dtype=np.float64) @ basis.m_coil.T  # (T_raw, k)

    psi_cell_lab = (
        np.asarray(i_cell_labels, dtype=np.float64) @ basis.m_cells.T
    )  # (T_lab, k)
    psi_cell_raw = np.empty_like(psi_coil)
    for m in range(psi_coil.shape[1]):
        psi_cell_raw[:, m] = np.interp(
            raw_times, label_times, psi_cell_lab[:, m]
        )  # constant-extrapolates outside the label span
    before = raw_times < label_times[0]
    if np.any(before):
        if ip_raw is not None:
            ip_raw = np.asarray(ip_raw, dtype=np.float64)
            ip0 = float(np.interp(label_times[0], raw_times, ip_raw))
            frac = np.zeros(int(before.sum()))
            if abs(ip0) > 1e-12:
                frac = np.clip(ip_raw[before] / ip0, 0.0, 1.0)
            psi_cell_raw[before] = frac[:, np.newaxis] * psi_cell_lab[0]
        else:
            psi_cell_raw[before] = 0.0

    tau_eff = basis.tau / np.asarray(tau_scale, dtype=np.float64)
    a_raw, _u = integrate_eddy_ode(
        tau_eff, raw_times, psi_coil + psi_cell_raw, volt_m=volt_m_raw
    )
    a_labels = np.empty((label_times.size, basis.n_modes))
    for m in range(basis.n_modes):
        a_labels[:, m] = np.interp(label_times, raw_times, a_raw[:, m])
    return a_labels, a_raw


def save_circuit_system(path: Path | str, system: PassiveCircuitSystem) -> None:
    """Persist a campaign circuit system (the L build is minutes of kernels)."""
    import json  # noqa: PLC0415

    extra = (
        {"section_scale": system.section_scale}
        if system.section_scale is not None
        else {}
    )
    np.savez_compressed(
        path,
        circuits=system.circuits,
        centroid_r=system.centroid_r,
        centroid_z=system.centroid_z,
        lmat=system.lmat,
        r_diag=system.r_diag,
        a_circ=system.a_circ,
        g_circ=system.g_circ,
        m_coil_circ=system.m_coil_circ,
        coil_channels=np.array(system.coil_channels),
        case_channel_row=np.frombuffer(
            json.dumps(system.case_channel_row).encode(), dtype=np.uint8
        ),
        resistivity=np.float64(system.resistivity),
        **extra,
    )


def load_circuit_system(path: Path | str) -> PassiveCircuitSystem:
    import json  # noqa: PLC0415

    with np.load(path) as z:
        return PassiveCircuitSystem(
            circuits=z["circuits"],
            centroid_r=z["centroid_r"],
            centroid_z=z["centroid_z"],
            lmat=z["lmat"],
            r_diag=z["r_diag"],
            a_circ=z["a_circ"],
            g_circ=z["g_circ"],
            m_coil_circ=z["m_coil_circ"],
            coil_channels=[str(c) for c in z["coil_channels"]],
            case_channel_row={
                k: int(v)
                for k, v in json.loads(z["case_channel_row"].tobytes()).items()
            },
            resistivity=float(z["resistivity"]),
            section_scale=(z["section_scale"] if "section_scale" in z.files else None),
        )


def save_eigenbasis(path: Path | str, basis: PassiveEigenbasis) -> None:
    """Persist a campaign eigenbasis (the build is minutes of kernel sums)."""
    extra = {"volt_coil": basis.volt_coil} if basis.volt_coil is not None else {}
    np.savez_compressed(
        path,
        tau=basis.tau,
        v=basis.v,
        a_sens=basis.a_sens,
        g_grid=basis.g_grid,
        m_coil=basis.m_coil,
        m_cells=basis.m_cells,
        resistivity=np.float64(basis.resistivity),
        **extra,
    )


def load_eigenbasis(path: Path | str) -> PassiveEigenbasis:
    with np.load(path) as z:
        return PassiveEigenbasis(
            tau=z["tau"],
            v=z["v"],
            a_sens=z["a_sens"],
            g_grid=z["g_grid"],
            m_coil=z["m_coil"],
            m_cells=z["m_cells"],
            resistivity=float(z["resistivity"]),
            volt_coil=(z["volt_coil"] if "volt_coil" in z.files else None),
        )


# ---------------------------------------------------------------------------
# The causal temporal operator
# ---------------------------------------------------------------------------
class TemporalOperator(nn.Module):
    """Causal transformer over slice codes + diagonal L/R SSM eddy block.

    Inputs are sequences over one shot's labelled slices: per-slice sensor
    tokens (masked-pooled by the same encoder as the static operator),
    firewall-safe globals, the step Δt, and the physically-integrated eddy
    features (state + drive, standardised).  Outputs per step:

    * ``dc`` (B, T, n_dof) — bounded profile-DOF corrections (as R1);
    * ``da`` (B, T, k) — eddy mode amplitudes in PHYSICAL mode units
      (standardised internally, rescaled by ``eddy_std`` on output).

    The eddy block is a diagonal SSM: per-mode learnable decay rates
    initialised at the physical L/R times; drive = physical flux swing plus a
    zero-initialised trunk projection.  All output heads are zero-initialised,
    so the untrained operator is the identity on the classical spine.
    """

    def __init__(
        self,
        n_dof: int,
        tau_init: np.ndarray,
        eddy_std: np.ndarray,
        drive_std: np.ndarray,
        *,
        token_dim: int = len(TOKEN_FEATURES),
        n_global: int = 2,
        width: int = 96,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dc_scale: float = 0.3,
        eddy_scale: float = 3.0,
    ) -> None:
        super().__init__()
        self.n_dof = int(n_dof)
        self.n_modes = int(np.asarray(tau_init).size)
        self.dc_scale = float(dc_scale)
        self.eddy_scale = float(eddy_scale)
        self.width = int(width)
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.n_layers = int(n_layers)
        self.n_global = int(n_global)

        self.log_tau = nn.Parameter(
            torch.log(torch.as_tensor(tau_init, dtype=torch.float32))
        )
        self.register_buffer(
            "eddy_std",
            torch.clamp(torch.as_tensor(eddy_std, dtype=torch.float32), min=1e-30),
        )
        self.register_buffer(
            "drive_std",
            torch.clamp(torch.as_tensor(drive_std, dtype=torch.float32), min=1e-30),
        )

        self.token_mlp = nn.Sequential(
            nn.Linear(token_dim, width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.GELU(),
        )
        step_dim = 2 * width + n_global + 1 + 2 * self.n_modes  # +1 = log Δt
        self.step_proj = nn.Linear(step_dim, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            activation="gelu",
            batch_first=True,
            norm_first=True,
            dropout=0.0,
        )
        self.trunk = nn.TransformerEncoder(
            layer, num_layers=n_layers, enable_nested_tensor=False
        )

        self.dc_head = nn.Linear(d_model, n_dof)
        nn.init.zeros_(self.dc_head.weight)
        nn.init.zeros_(self.dc_head.bias)
        # eddy drive projection (adds to the physical drive) and output heads —
        # gate + trunk projection both zero-initialised: da == 0 at init while
        # the SSM state still carries the physically-driven trajectory
        self.drive_proj = nn.Linear(d_model, self.n_modes)
        nn.init.zeros_(self.drive_proj.weight)
        nn.init.zeros_(self.drive_proj.bias)
        self.eddy_gate = nn.Parameter(torch.zeros(self.n_modes))
        self.eddy_head = nn.Linear(d_model, self.n_modes)
        nn.init.zeros_(self.eddy_head.weight)
        nn.init.zeros_(self.eddy_head.bias)

    # -- featurization ------------------------------------------------------
    def encode_tokens(
        self, tokens: torch.Tensor, token_mask: torch.Tensor
    ) -> torch.Tensor:
        """(B, T, S, F) sensor tokens → (B, T, 2·width) masked mean+max codes."""
        h = self.token_mlp(tokens)
        w = token_mask.to(h.dtype).unsqueeze(-1)
        n_valid = w.sum(dim=2)
        mean_pool = (h * w).sum(dim=2) / n_valid.clamp(min=1.0)
        max_pool = torch.where(w > 0, h, h.new_full((), -1e30)).amax(dim=2)
        # a fully-masked timestep (padded tail of a mixed-length batch) must
        # emit zeros, not the -1e30 max sentinel — test on the TRUE valid
        # count, never on a clamped denominator that is always positive
        max_pool = torch.where(n_valid > 0, max_pool, torch.zeros_like(max_pool))
        return torch.cat([mean_pool, max_pool], dim=-1)

    def forward(
        self,
        tokens: torch.Tensor,  # (B, T, S, F)
        token_mask: torch.Tensor,  # (B, T, S) bool
        global_feats: torch.Tensor,  # (B, T, n_global)
        dt: torch.Tensor,  # (B, T) step Δt [s]; dt[:, 0] ignored
        a_phys: torch.Tensor,  # (B, T, k) physical eddy state (native units)
        u_drive: torch.Tensor,  # (B, T, k) physical flux swing (native units)
        pad_mask: torch.Tensor | None = None,  # (B, T) bool — True = PADDING
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, t = tokens.shape[:2]
        code = self.encode_tokens(tokens, token_mask)
        a_std = a_phys / self.eddy_std
        u_std = u_drive / self.drive_std
        log_dt = torch.log(torch.clamp(dt, min=1e-6)).unsqueeze(-1)
        feats = torch.cat([code, global_feats, log_dt, a_std, u_std], dim=-1)
        x = self.step_proj(feats)
        causal = torch.triu(
            torch.ones(t, t, dtype=torch.bool, device=x.device), diagonal=1
        )
        h = self.trunk(x, mask=causal, src_key_padding_mask=pad_mask)

        dc = self.dc_scale * torch.tanh(self.dc_head(h))

        # diagonal SSM over standardised mode amplitudes, exact-ZOH stepping
        tau = torch.exp(self.log_tau)  # (k,)
        drive = u_std + self.drive_proj(h)  # (B, T, k)
        dt_c = torch.clamp(dt, min=1e-6).unsqueeze(-1)
        decay = torch.exp(-dt_c / tau)
        coeff = tau / dt_c * (1.0 - decay)
        states = []
        s = torch.zeros(b, self.n_modes, device=x.device, dtype=x.dtype)
        for step in range(t):
            if step > 0:
                s = decay[:, step] * s + coeff[:, step] * drive[:, step]
            states.append(s)
        s_seq = torch.stack(states, dim=1)  # (B, T, k)

        da_std = self.eddy_scale * torch.tanh(
            self.eddy_gate * s_seq + self.eddy_head(h)
        )
        da = da_std * self.eddy_std
        if pad_mask is not None:
            # select, never multiply: any non-finite value a padded position
            # picks up would survive a multiplicative mask (NaN * 0 == NaN)
            # and poison the batch loss
            keep = (~pad_mask).unsqueeze(-1)
            dc = torch.where(keep, dc, torch.zeros_like(dc))
            da = torch.where(keep, da, torch.zeros_like(da))
        return dc, da


def save_checkpoint(path: Path | str, model: TemporalOperator, extra: dict) -> None:
    torch.save(
        {
            "state_dict": model.state_dict(),
            "n_dof": model.n_dof,
            "n_modes": model.n_modes,
            "dc_scale": model.dc_scale,
            "eddy_scale": model.eddy_scale,
            "width": model.width,
            "d_model": model.d_model,
            "n_heads": model.n_heads,
            "n_layers": model.n_layers,
            "n_global": model.n_global,
            "token_features": list(TOKEN_FEATURES),
            **extra,
        },
        path,
    )


def load_checkpoint(path: Path | str) -> tuple[TemporalOperator, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = TemporalOperator(
        int(ckpt["n_dof"]),
        np.ones(int(ckpt["n_modes"])),  # placeholders — state_dict overwrites
        np.ones(int(ckpt["n_modes"])),
        np.ones(int(ckpt["n_modes"])),
        width=int(ckpt["width"]),
        d_model=int(ckpt["d_model"]),
        n_heads=int(ckpt["n_heads"]),
        n_layers=int(ckpt["n_layers"]),
        n_global=int(ckpt["n_global"]),
        dc_scale=float(ckpt["dc_scale"]),
        eddy_scale=float(ckpt["eddy_scale"]),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


__all__ = [
    "STEEL_RESISTIVITY",
    "PassiveCircuitSystem",
    "PassiveEigenbasis",
    "TemporalOperator",
    "build_drive_linkage",
    "build_passive_circuit_system",
    "build_passive_eigenbasis",
    "integrate_eddy_ode",
    "reduce_passive_system",
    "load_checkpoint",
    "load_circuit_system",
    "load_eigenbasis",
    "save_circuit_system",
    "physical_eddy_history",
    "raw_eddy_trajectory",
    "save_checkpoint",
    "save_eigenbasis",
]
