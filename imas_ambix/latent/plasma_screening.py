"""Plasma screening circuit — the plasma as a dynamic filament system.

The vessel treatment applied to the plasma: the core region is tiled into
thick-cylinder patches (groups of in-limiter grid cells), the patch mutual
matrix comes exactly from the in-tree finite-area cylinder kernel (the grid's
cached cell Green's matrices — no new kernels), patch resistances come from
the bounded :class:`~imas_ambix.latent.current_diffusion.EtaProfile` family at
each patch's ψ_N, and the L/R eigenmodes — with their exact-ZOH-integrable
time constants — form the dynamic screening state.

Physics carried here and nowhere else in the classical spine:

* during a fast flux swing the inductance structure alone concentrates the
  incremental current in the outermost patches (the outer plasma shields the
  core — the skin effect IS circuit screening: applying a loop voltage to the
  nested system puts d i/dt ∝ M⁻¹·1 entirely on the outermost conductor);
* the current then penetrates inward on the local resistive time, so the
  filament resistances (the η(ψ_N) closure) set the decay of every screening
  mode — the ramp transients constrain η exactly the way coil-only transients
  constrained the vessel resistances.

In the flux-surface-averaged nested limit this circuit IS the landed 1D
ψ-diffusion operator (:func:`~imas_ambix.latent.current_diffusion.diffuse_psi`)
— pinned by test on a shared analytic nested-circle case — so the current
dynamics and the flux diffusion are one system with one unknown, η(ψ_N).

Screening modes are built in the ZERO-NET-CURRENT subspace: the measured
plasma current (Rogowski) already pins the total, so the free dynamics live in
the redistribution directions, and a mode column added to the per-slice fit
can never fight the Ip anchor.  Mode flux enters the free-boundary solve
through the same additive-column mechanism as the passive eigenmode sidecar
(:func:`~imas_ambix.latent.gs_solve.build_passive_sidecar`): sensor columns
from the cell Green's rows, grid-ψ columns from the grid's own Dirichlet
solve (superposition — exactly how plasma flux enters the Picard field).

All construction rules are machine-agnostic: patches are binned in the
dimensionless (√ψ_N, poloidal angle) coordinates, and every coupling is the
grid's own analytic kernel — a uniform geometric rescale leaves mode shapes
invariant and maps τ → τ·s² exactly (L ∝ s, R ∝ 1/s).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from imas_ambix.latent.temporal_operator import integrate_eddy_ode

if TYPE_CHECKING:
    from imas_ambix.latent.current_diffusion import EtaProfile
    from imas_ambix.latent.gs_solve import EquilibriumGrid

MU0 = 4.0e-7 * np.pi


# ---------------------------------------------------------------------------
# patch tiling — dimensionless (√ψ_N, θ) bins of the core cells
# ---------------------------------------------------------------------------


@dataclass
class PatchTiling:
    """Grouping of core grid cells into plasma patches.

    ``cell_index`` are positions into ``grid.cells`` (the in-limiter cell
    order every Green's matrix uses); ``owner`` maps each tiled cell to its
    patch; ``share`` is the cell's current share within its patch (uniform
    current density — shares sum to 1 per patch).  ``psi_n`` / ``r`` / ``z``
    are share-weighted patch centroids.
    """

    cell_index: np.ndarray  # (n_tiled,) positions into grid.cells
    owner: np.ndarray  # (n_tiled,) patch id per tiled cell
    share: np.ndarray  # (n_tiled,) current share within the patch
    psi_n: np.ndarray  # (P,) patch ψ_N (share-weighted)
    r: np.ndarray  # (P,) patch centroid R [m]
    z: np.ndarray  # (P,)

    @property
    def n_patches(self) -> int:
        return int(self.psi_n.size)

    def cell_matrix(self, n_cells: int) -> np.ndarray:
        """(n_cells, P) share matrix W: patch currents → cell currents [A]."""
        w = np.zeros((n_cells, self.n_patches))
        w[self.cell_index, self.owner] = self.share
        return w


def tile_core_patches(
    grid: EquilibriumGrid,
    psi_n_flat: np.ndarray,
    core_mask: np.ndarray,
    axis: tuple[float, float],
    *,
    n_rad: int = 10,
    n_pol: int = 8,
) -> PatchTiling:
    """Tile the axis-connected core into (√ψ_N × poloidal-angle) patches.

    Radial bins are uniform in √ψ_N (≈ minor radius — resolves the edge
    screening layer in physical depth); poloidal bins are uniform in the
    angle about the magnetic axis.  The innermost radial bin stays a single
    patch (no poloidal split — the axis region has no meaningful angle).
    Both coordinates are dimensionless, so the rule transfers across machines
    and grids unchanged.  Empty bins are dropped.
    """
    core_flat = np.asarray(core_mask, dtype=bool).ravel()
    psi_n_flat = np.asarray(psi_n_flat, dtype=np.float64)
    in_core = core_flat[grid.cells]
    cell_pos = np.where(in_core)[0]  # positions into grid.cells
    if cell_pos.size == 0:
        raise ValueError("core mask holds no in-limiter cells")
    cells_flat = grid.cells[cell_pos]
    pn = np.clip(psi_n_flat[cells_flat], 0.0, 1.0)
    rr = grid.flat_r[cells_flat]
    zz = grid.flat_z[cells_flat]

    rad = np.sqrt(pn)
    i_rad = np.minimum((rad * n_rad).astype(int), n_rad - 1)
    theta = np.arctan2(zz - axis[1], rr - axis[0])
    i_pol = np.minimum(((theta + np.pi) / (2.0 * np.pi) * n_pol).astype(int), n_pol - 1)
    bin_id = np.where(i_rad == 0, 0, 1 + (i_rad - 1) * n_pol + i_pol)

    uniq, owner = np.unique(bin_id, return_inverse=True)
    n_p = uniq.size
    counts = np.bincount(owner, minlength=n_p).astype(np.float64)
    share = 1.0 / counts[owner]

    psi_p = np.bincount(owner, weights=share * pn, minlength=n_p)
    r_p = np.bincount(owner, weights=share * rr, minlength=n_p)
    z_p = np.bincount(owner, weights=share * zz, minlength=n_p)
    return PatchTiling(
        cell_index=cell_pos,
        owner=owner,
        share=share,
        psi_n=psi_p,
        r=r_p,
        z=z_p,
    )


# ---------------------------------------------------------------------------
# the plasma circuit — exact L from the grid's analytic kernel, R from η(ψ_N)
# ---------------------------------------------------------------------------


@dataclass
class PlasmaCircuit:
    """Patch-space L/R system of the tiled plasma region.

    ``lmat`` is the exact two-section flux linkage [Wb/A] (finite-area
    cylinder source, cell-centre observer — the grid's own cached cell
    Green's matrix, SPD-guarded).  Resistance is per-cell exact:
    ``r_diag(η) = Σ_cells share²·2πR·η(ψ_N)/A_cell`` (parallel toroidal paths
    at fixed shares).  ``m_coil`` are the coil-channel flux linkages [Wb/A]
    (drive couplings, from the grid's coil ψ columns).
    """

    tiling: PatchTiling
    lmat: np.ndarray  # (P, P)
    m_coil: np.ndarray  # (P, C) coil channel → patch flux linkage
    cell_r: np.ndarray  # (n_tiled,) cell major radius [m]
    cell_psi_n: np.ndarray  # (n_tiled,) cell ψ_N
    cell_area: float  # grid cell area [m²]

    @property
    def n_patches(self) -> int:
        return self.tiling.n_patches

    def r_diag(self, eta: EtaProfile) -> np.ndarray:
        """Diagonal patch resistances [Ω] at the η(ψ_N) closure profile."""
        t = self.tiling
        eta_c = np.asarray(eta(self.cell_psi_n), dtype=np.float64)
        r_cell = 2.0 * np.pi * self.cell_r * eta_c / self.cell_area
        return np.bincount(
            t.owner, weights=t.share**2 * r_cell, minlength=self.n_patches
        )


def build_plasma_circuit(
    grid: EquilibriumGrid,
    tiling: PatchTiling,
    psi_n_flat: np.ndarray,
) -> PlasmaCircuit:
    """Assemble the patch L/R system from the grid's own analytic kernels.

    ``lmat = Wᵀ·Gψ·W`` with Gψ the cached in-limiter cell Green's ψ matrix —
    the finite-area cylinder kernel throughout, symmetrised and SPD-guarded
    exactly as the vessel circuit build.  Coil drive couplings are the grid's
    coil ψ columns share-averaged onto patches.  ``psi_n_flat`` is the
    equilibrium's ψ_N map (flat grid) — where the η closure is evaluated.
    """
    cg = grid.cell_greens()
    n_cells = grid.cells.size
    w = tiling.cell_matrix(n_cells)
    lmat = w.T @ cg["psi"] @ w
    lmat = 0.5 * (lmat + lmat.T)
    ev, u0 = np.linalg.eigh(lmat)
    lmat = (u0 * np.clip(ev, 1e-6 * ev.max(), None)) @ u0.T

    cells_flat = grid.cells[tiling.cell_index]
    if grid._coil_psi_columns.shape[1]:
        coil_cells = grid._coil_psi_columns[cells_flat, :]
        m_coil = np.zeros((tiling.n_patches, coil_cells.shape[1]))
        for c in range(coil_cells.shape[1]):
            m_coil[:, c] = np.bincount(
                tiling.owner,
                weights=tiling.share * coil_cells[:, c],
                minlength=tiling.n_patches,
            )
    else:
        m_coil = np.zeros((tiling.n_patches, 0))

    return PlasmaCircuit(
        tiling=tiling,
        lmat=lmat,
        m_coil=m_coil,
        cell_r=grid.flat_r[cells_flat],
        cell_psi_n=np.clip(
            np.asarray(psi_n_flat, dtype=np.float64)[cells_flat], 0.0, 1.0
        ),
        cell_area=grid.dr * grid.dz,
    )


def build_plasma_circuit_from_state(
    grid: EquilibriumGrid,
    psi_n_flat: np.ndarray,
    core_mask: np.ndarray,
    axis: tuple[float, float],
    *,
    n_rad: int = 10,
    n_pol: int = 8,
) -> PlasmaCircuit:
    """Tile + assemble in one step from an equilibrium's (ψ_N, core, axis)."""
    tiling = tile_core_patches(
        grid, psi_n_flat, core_mask, axis, n_rad=n_rad, n_pol=n_pol
    )
    return build_plasma_circuit(grid, tiling, psi_n_flat)


# ---------------------------------------------------------------------------
# full-system exact-ZOH evolution — the truth generator's engine
# ---------------------------------------------------------------------------


def circuit_eigensystem(
    circuit: PlasmaCircuit, eta: EtaProfile
) -> tuple[np.ndarray, np.ndarray]:
    """Full unconstrained eigensystem (τ, V) of ``R v = (1/τ) L v``.

    ``V`` is L-orthonormal (``Vᵀ L V = I``), so patch currents map to mode
    amplitudes by ``a = Vᵀ L i`` and back by ``i = V a``.
    """
    from scipy.linalg import eigh  # noqa: PLC0415

    w, v = eigh(np.diag(circuit.r_diag(eta)), circuit.lmat)
    tau = 1.0 / np.clip(w, 1e-12, None)
    return tau, v


def _evolve_lr_system(
    lmat: np.ndarray,
    r_vec: np.ndarray,
    m_coil: np.ndarray,
    volt_pattern: np.ndarray,
    times: np.ndarray,
    *,
    i0: np.ndarray,
    loop_voltage: np.ndarray,
    i_pf_of_t: np.ndarray | None,
) -> np.ndarray:
    """Exact-ZOH evolution of a generic diagonal-R L/R system.

    Dynamics: ``L·di/dt + R·i = u(t)·p − M_c·dI_c/dt`` with ``p`` the
    voltage-drive pattern (which conductors the loop voltage acts on).  In
    the L-orthonormal eigenbasis both terms are exactly integrable per step
    (piecewise-constant voltage / piecewise-linear flux — the vessel ZOH
    contract), so the integration is EXACT for those drive classes at any
    step size.  Returns the current state (T, N).
    """
    from scipy.linalg import eigh  # noqa: PLC0415

    times = np.asarray(times, dtype=np.float64)
    u = np.asarray(loop_voltage, dtype=np.float64)
    w, v = eigh(np.diag(np.asarray(r_vec, dtype=np.float64)), lmat)
    tau = 1.0 / np.clip(w, 1e-12, None)
    pat_m = v.T @ np.asarray(volt_pattern, dtype=np.float64)
    volt_m = u[:, np.newaxis] * pat_m[np.newaxis, :]
    if i_pf_of_t is not None and m_coil.shape[1]:
        psi_m = np.asarray(i_pf_of_t, dtype=np.float64) @ (v.T @ m_coil).T
    else:
        psi_m = np.zeros((times.size, tau.size))
    a0 = v.T @ (lmat @ np.asarray(i0, dtype=np.float64))
    a, _u = integrate_eddy_ode(tau, times, psi_m, a0=a0, volt_m=volt_m)
    return a @ v.T


def evolve_patch_currents(
    circuit: PlasmaCircuit,
    eta: EtaProfile,
    times: np.ndarray,
    *,
    i0: np.ndarray,
    loop_voltage: np.ndarray,
    i_pf_of_t: np.ndarray | None = None,
) -> np.ndarray:
    """Exact-ZOH evolution of the full patch-current state.

    Dynamics: ``L·di/dt + R·i = u(t)·1 − M_pc·dI_c/dt`` — ``u`` the loop
    voltage (identical for every toroidal loop enclosing the solenoid, so the
    central-solenoid drive is exactly the uniform vector), ``i_pf_of_t``
    (T, C) the coil currents whose swing drives the mode flux.  Returns patch
    currents (T, P).
    """
    return _evolve_lr_system(
        circuit.lmat,
        circuit.r_diag(eta),
        circuit.m_coil,
        np.ones(circuit.n_patches),
        times,
        i0=i0,
        loop_voltage=loop_voltage,
        i_pf_of_t=i_pf_of_t,
    )


def steady_state_currents(
    circuit: PlasmaCircuit, eta: EtaProfile, ip_amperes: float
) -> np.ndarray:
    """Fully-penetrated (resistive steady-state) patch currents at total Ip.

    Under a constant loop voltage the steady state is ``i ∝ R⁻¹·1`` — current
    proportional to conductance, the equilibrated profile the plasma reaches
    when the ramp is slow against the resistive time.
    """
    g = 1.0 / circuit.r_diag(eta)
    return g * (float(ip_amperes) / g.sum())


def loop_voltage_for_ip(
    circuit: PlasmaCircuit,
    eta: EtaProfile,
    times: np.ndarray,
    *,
    i0: np.ndarray,
    ip_target: float,
    i_pf_of_t: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    """Constant loop voltage over the interval that lands ΣI = ``ip_target``.

    The system is linear in the applied voltage, so two exact integrations
    (u = 0 and u = 1) determine it in closed form.  Returns ``(u, i_end)``.
    """
    times = np.asarray(times, dtype=np.float64)
    zero = np.zeros(times.size)
    i_free = evolve_patch_currents(
        circuit, eta, times, i0=i0, loop_voltage=zero, i_pf_of_t=i_pf_of_t
    )[-1]
    i_unit = evolve_patch_currents(
        circuit,
        eta,
        times,
        i0=np.zeros(circuit.n_patches),
        loop_voltage=np.ones(times.size),
        i_pf_of_t=None,
    )[-1]
    denom = float(i_unit.sum())
    if abs(denom) < 1e-30:
        raise ValueError("degenerate voltage response — check the circuit")
    u = (float(ip_target) - float(i_free.sum())) / denom
    return u, i_free + u * i_unit


# ---------------------------------------------------------------------------
# coupled system — plasma patches + fixed external conductors, ONE circuit
# ---------------------------------------------------------------------------


@dataclass
class CoupledCircuit:
    """Plasma patches + fixed external conductors (vessel) as ONE L/R system.

    The full dynamic content of the machine in a single block circuit: the
    plasma patch self/mutual block, the external-conductor block, and the
    exact cross-linkage between them — so the flux diffusion, the passive
    structures, and the plasma redistribution evolve together and every
    dΦ/dt term (coil swing, vessel eddies, plasma back-reaction) is carried
    by one integration.  The loop voltage acts on the plasma rows only (the
    solenoid EMF drives loops the plasma closes; the vessel is passive).
    """

    plasma: PlasmaCircuit
    lmat: np.ndarray  # (P+V, P+V) block flux linkage, SPD-guarded
    m_coil: np.ndarray  # (P+V, C) coil-channel flux linkage
    r_ext: np.ndarray  # (V,) external conductor resistances [Ω]

    @property
    def n_plasma(self) -> int:
        return self.plasma.n_patches

    @property
    def n_ext(self) -> int:
        return int(self.r_ext.size)

    @property
    def n_total(self) -> int:
        return int(self.lmat.shape[0])

    def r_vec(self, eta: EtaProfile) -> np.ndarray:
        return np.concatenate([self.plasma.r_diag(eta), self.r_ext])

    def volt_pattern(self) -> np.ndarray:
        return np.concatenate([np.ones(self.n_plasma), np.zeros(self.n_ext)])


def patch_external_linkage(
    grid: EquilibriumGrid,
    tiling: PatchTiling,
    psi_columns: np.ndarray,
) -> np.ndarray:
    """(P, V) flux linked by each patch per ampere of each external circuit.

    ``psi_columns`` (n_grid, V) are the external circuits' grid-flux columns
    (finite-area kernel — e.g. the campaign's passive columns); the patch
    linkage is the share-weighted average over the patch's cells, exactly the
    coil-linkage construction.
    """
    cells_flat = grid.cells[tiling.cell_index]
    cols = np.asarray(psi_columns, dtype=np.float64)[cells_flat, :]
    out = np.zeros((tiling.n_patches, cols.shape[1]))
    for c in range(cols.shape[1]):
        out[:, c] = np.bincount(
            tiling.owner, weights=tiling.share * cols[:, c], minlength=tiling.n_patches
        )
    return out


def build_coupled_circuit(
    circuit: PlasmaCircuit,
    *,
    l_ext: np.ndarray,
    r_ext: np.ndarray,
    m_ext_coil: np.ndarray,
    m_patch_ext: np.ndarray,
) -> CoupledCircuit:
    """Assemble the plasma+external block system.

    ``l_ext`` (V, V) external self/mutual linkage, ``r_ext`` (V,) external
    resistances, ``m_ext_coil`` (V, C) external↔coil linkage in the SAME coil
    channel order as ``circuit.m_coil``, ``m_patch_ext`` (P, V) plasma-patch↔
    external linkage (reciprocity supplies the transpose block).  The block L
    is symmetrised and SPD-guarded exactly as each sub-block build.
    """
    p = circuit.n_patches
    v = int(np.asarray(r_ext).size)
    lmat = np.zeros((p + v, p + v))
    lmat[:p, :p] = circuit.lmat
    lmat[:p, p:] = m_patch_ext
    lmat[p:, :p] = np.asarray(m_patch_ext, dtype=np.float64).T
    lmat[p:, p:] = l_ext
    lmat = 0.5 * (lmat + lmat.T)
    ev, u0 = np.linalg.eigh(lmat)
    lmat = (u0 * np.clip(ev, 1e-8 * ev.max(), None)) @ u0.T
    m_coil = np.vstack([circuit.m_coil, np.asarray(m_ext_coil, dtype=np.float64)])
    return CoupledCircuit(
        plasma=circuit,
        lmat=lmat,
        m_coil=m_coil,
        r_ext=np.asarray(r_ext, dtype=np.float64),
    )


def evolve_coupled(
    coupled: CoupledCircuit,
    eta: EtaProfile,
    times: np.ndarray,
    *,
    i0: np.ndarray,
    loop_voltage: np.ndarray,
    i_pf_of_t: np.ndarray | None = None,
) -> np.ndarray:
    """Exact-ZOH evolution of the coupled plasma+external state (T, P+V)."""
    return _evolve_lr_system(
        coupled.lmat,
        coupled.r_vec(eta),
        coupled.m_coil,
        coupled.volt_pattern(),
        times,
        i0=i0,
        loop_voltage=loop_voltage,
        i_pf_of_t=i_pf_of_t,
    )


def coupled_loop_voltage_for_ip(
    coupled: CoupledCircuit,
    eta: EtaProfile,
    times: np.ndarray,
    *,
    i0: np.ndarray,
    ip_target: float,
    i_pf_of_t: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    """Constant loop voltage landing Σ(plasma rows) = ``ip_target``.

    Same closed-form two-integration linearity as
    :func:`loop_voltage_for_ip`, with the total taken over the plasma rows
    only (the vessel rows carry eddies, not plasma current).  Returns
    ``(u, i_end)`` with ``i_end`` the full (P+V,) end state.
    """
    times = np.asarray(times, dtype=np.float64)
    p = coupled.n_plasma
    zero = np.zeros(times.size)
    i_free = evolve_coupled(
        coupled, eta, times, i0=i0, loop_voltage=zero, i_pf_of_t=i_pf_of_t
    )[-1]
    i_unit = evolve_coupled(
        coupled,
        eta,
        times,
        i0=np.zeros(coupled.n_total),
        loop_voltage=np.ones(times.size),
        i_pf_of_t=None,
    )[-1]
    denom = float(i_unit[:p].sum())
    if abs(denom) < 1e-30:
        raise ValueError("degenerate voltage response — check the coupled circuit")
    u = (float(ip_target) - float(i_free[:p].sum())) / denom
    return u, i_free + u * i_unit


def mean_plasma_linked_flux(
    coupled: CoupledCircuit,
    i_state: np.ndarray,
    i_pf: np.ndarray | None = None,
) -> float:
    """Patch-mean linked flux Ψ̄ of the plasma rows [Wb].

    ``Ψ̄ = mean_j (L·i + M_c·I_c)_j`` over the plasma rows — the scalar whose
    balance ``dΨ̄/dt + mean_j(R_j i_j) = u`` integrates the per-patch circuit
    equation.  With the state integrated from zero (vacuum start) the flux
    ledger ``∫u dt = ΔΨ̄ + Σ(shape-remap jumps) + ∫mean(R·i) dt`` closes with
    no free constant; the remap jumps are the frozen-geometry chain's carry
    of the shape/inductance change (dL/dt).
    """
    p = coupled.n_plasma
    psi = coupled.lmat[:p, :] @ np.asarray(i_state, dtype=np.float64)
    if i_pf is not None and coupled.m_coil.shape[1]:
        psi = psi + coupled.m_coil[:p, :] @ np.asarray(i_pf, dtype=np.float64)
    return float(psi.mean())


# ---------------------------------------------------------------------------
# one-matrix measured-trace solve — every circuit in one flux balance
# ---------------------------------------------------------------------------


def plasma_self_inductance(
    r_axis: np.ndarray,
    minor_radius: np.ndarray,
    elongation: np.ndarray | float = 1.0,
    internal_inductance: np.ndarray | float = 0.0,
) -> np.ndarray:
    """Large-aspect self-inductance of the plasma ring [H].

    ``L = μ0·R·(ln(8R/(a√κ)) − 2 + li/2)`` — the external term at the
    elongation-corrected effective minor radius plus the internal-inductance
    share.  All arguments broadcast, so a measured (R, a, κ, li) TRACE
    yields L(t) directly: the dL/dt of the growing, shifting, shaping column
    is as much a flux-balance term as the moving centroid, and both must be
    carried for the applied loop voltage to come out right.
    """
    r0 = np.asarray(r_axis, dtype=np.float64)
    a = np.asarray(minor_radius, dtype=np.float64)
    kappa = np.asarray(elongation, dtype=np.float64)
    li = np.asarray(internal_inductance, dtype=np.float64)
    a_eff = a * np.sqrt(kappa)
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = np.log(8.0 * r0 / a_eff) - 2.0 + 0.5 * li
        out = MU0 * r0 * lam
    return np.where((r0 > 0.0) & (a_eff > 0.0) & (a_eff < r0 * 8.0), out, np.nan)


@dataclass
class PinnedFluxSolve:
    """Result of the one-interaction-matrix solve at pinned plasma current.

    ``u_loop`` is the SOLVED applied loop voltage [V]; ``dpsi_terms`` maps
    each flux-balance component to its voltage share [V] (``plasma_self`` =
    d(L_p·Ip)/dt including the dL/dt of the evolving shape, ``coils``,
    ``vessel``, ``resistive`` = R_p·Ip) so nothing enters the balance
    silently.  Vessel states are in the passive system's circuit order.
    """

    u_loop: np.ndarray  # (T,) applied loop voltage [V]
    i_vessel: np.ndarray  # (T, P) coil+plasma-driven vessel state [A]
    i_vessel_coil: np.ndarray  # (T, P) coil-only vessel state [A]
    psi_plasma: np.ndarray  # (T,) plasma-row linked flux [Wb]
    plasma_l: np.ndarray  # (T,) plasma self-inductance [H]
    dpsi_terms: dict[str, np.ndarray]


def solve_pinned_plasma_circuit(
    table,
    passive,
    i_pf_full: np.ndarray,
    channels: list[str],
    times: np.ndarray,
    *,
    ip_amperes: np.ndarray,
    axis_rz: np.ndarray,
    minor_radius: np.ndarray,
    elongation: np.ndarray | float = 1.0,
    internal_inductance: np.ndarray | float = 0.0,
    plasma_resistance_ohm: float = 3.0e-6,
) -> PinnedFluxSolve:
    """One-interaction-matrix flux solve with the plasma row pinned.

    Design rule: when solving for flux, EVERY circuit — driven coil,
    measured case, passive structure, and the plasma itself — lives in the
    same interaction matrix, so every component of dψ/dt (coil swing,
    vessel eddies, plasma back-reaction, and the dL/dt of the evolving
    column) is accounted for and the applied loop voltage is SOLVED, never
    assumed.  The plasma row is resolved from measurement, not idealised:
    total current from the Ip trace, centroid from the first-moment trace
    (``axis_rz``), self-inductance from the evolving ``(R, a, κ, li)``
    through :func:`plasma_self_inductance` — the traces may come from the
    magnetics moment read (measurement path) or a firewalled referee
    (diagnostic path).

    With the plasma current pinned the block system decomposes exactly:
    the passive rows integrate the coil+plasma drive history
    (:func:`~imas_ambix.latent.temporal_operator.predict_vessel_currents`,
    moving-centroid mutuals included), and the plasma row's own balance is
    read out as the applied voltage

        u(t) = d/dt[L_p(t)·Ip(t) + M_pc(t)·I_c(t) + M_pv(t)·i_v(t)]
               + R_p·Ip(t)

    with every term reported separately in ``dpsi_terms``.  A passive
    system built with ``hold_back_cases=True`` moves the measured-case
    circuits from the drive columns into the state — the forward-chain
    configuration where no case measurements exist.
    """
    from imas_ambix.gs.force_balance import (  # noqa: PLC0415
        known_coil_psi,
        passive_circuit_psi,
    )
    from imas_ambix.latent.temporal_operator import (  # noqa: PLC0415
        predict_vessel_currents,
    )

    times = np.asarray(times, dtype=np.float64)
    ip = np.where(
        np.isfinite(np.asarray(ip_amperes, dtype=np.float64)),
        np.asarray(ip_amperes, dtype=np.float64),
        0.0,
    )
    axis_rz = np.asarray(axis_rz, dtype=np.float64)
    i_coil, i_full = predict_vessel_currents(
        table, passive, i_pf_full, channels, times, ip_amperes=ip, axis_rz=axis_rz
    )

    r_t = axis_rz[:, 0]
    z_t = axis_rz[:, 1]
    l_p = plasma_self_inductance(r_t, minor_radius, elongation, internal_inductance)
    psi_self = np.where(ip != 0.0, np.nan_to_num(l_p) * ip, 0.0)

    coil_chans, coil_cols = known_coil_psi(table, r_t, z_t)  # (T, C_known)
    idx = {ch: j for j, ch in enumerate(channels)}
    psi_coils = np.zeros(times.size)
    for j, chan in enumerate(coil_chans):
        if chan in idx:
            psi_coils += coil_cols[:, j] * np.asarray(i_pf_full)[:, idx[chan]]

    vessel_cols = passive_circuit_psi(table, passive.circuits, r_t, z_t)  # (T, P)
    psi_vessel = np.einsum("tp,tp->t", vessel_cols, i_full)

    psi_plasma = psi_self + psi_coils + psi_vessel
    dpsi_terms = {
        "plasma_self": np.gradient(psi_self, times),
        "coils": np.gradient(psi_coils, times),
        "vessel": np.gradient(psi_vessel, times),
        "resistive": float(plasma_resistance_ohm) * ip,
    }
    u_loop = (
        dpsi_terms["plasma_self"]
        + dpsi_terms["coils"]
        + dpsi_terms["vessel"]
        + dpsi_terms["resistive"]
    )
    return PinnedFluxSolve(
        u_loop=u_loop,
        i_vessel=i_full,
        i_vessel_coil=i_coil,
        psi_plasma=psi_plasma,
        plasma_l=l_p,
        dpsi_terms=dpsi_terms,
    )


# ---------------------------------------------------------------------------
# screening eigenbasis — zero-net-current modes for the per-slice fit
# ---------------------------------------------------------------------------


@dataclass
class ScreeningBasis:
    """Top-k zero-net-current L/R eigenmodes of the plasma circuit.

    ``tau`` are the resistive decay times [s] (the only place η enters);
    ``v`` (P, k) are the L-orthonormal patch-current patterns, each with
    exactly zero net current (the measured Ip pins the total, so the free
    screening dynamics live in the redistribution subspace); ``i_cells``
    (n_cells, k) expands a unit mode amplitude to in-limiter cell currents
    [A]; ``a_sens`` (S, k) and ``psi_grid`` (n_grid, k) are the sensor and
    full-grid flux columns per unit amplitude; ``m_coil`` (k, C) and the full
    projection pieces feed the drive trajectory.
    """

    tau: np.ndarray
    v: np.ndarray
    i_cells: np.ndarray
    a_sens: np.ndarray
    psi_grid: np.ndarray
    m_coil: np.ndarray
    lmat: np.ndarray  # (P, P) — the circuit L (for the backbone flux drive)
    r_diag: np.ndarray  # (P,) — at the basis η (for the backbone EMF drive)
    tiling: PatchTiling

    @property
    def n_modes(self) -> int:
        return int(self.tau.size)


def screening_eigenbasis(
    grid: EquilibriumGrid,
    circuit: PlasmaCircuit,
    eta: EtaProfile,
    g_sens: np.ndarray,
    *,
    k: int = 2,
    sensor_scale: np.ndarray | None = None,
) -> ScreeningBasis:
    """Reduce the plasma circuit to its k slowest zero-net-current modes.

    The net-current direction is deflated BEFORE the eigensolve (the
    generalised problem is restricted to the subspace {i : Σ i = 0}), so
    every kept mode is a pure radial/poloidal redistribution at fixed total
    current.  Modes are ranked by decay time (slowest first — a slow mode the
    sensors can see is exactly the history the static fit cannot absorb);
    when ``sensor_scale`` is given the ranking is the vessel-basis relevance
    ``τ·‖a_sens/σ‖`` instead.  Grid flux per mode comes from the grid's own
    Dirichlet solve (Δ*ψ = −2πμ0·R·jφ with the mode's edge Green's BC) — the
    identical path plasma current takes into the Picard field.
    """
    p = circuit.n_patches
    ones = np.ones(p)
    q, _r = np.linalg.qr(np.eye(p) - np.outer(ones, ones) / p)
    q = q[:, : p - 1]  # orthonormal complement of the net-current direction

    from scipy.linalg import eigh  # noqa: PLC0415

    r_diag = circuit.r_diag(eta)
    l_red = q.T @ circuit.lmat @ q
    r_red = q.T @ np.diag(r_diag) @ q
    w, v_red = eigh(r_red, l_red)
    tau = 1.0 / np.clip(w, 1e-12, None)
    v_full = q @ v_red  # L-orthonormal in the full space, zero net current

    n_cells = grid.cells.size
    w_mat = circuit.tiling.cell_matrix(n_cells)
    i_cells_all = w_mat @ v_full  # (n_cells, P-1)
    a_sens_all = np.asarray(g_sens, dtype=np.float64) @ i_cells_all

    if sensor_scale is not None:
        scale = np.clip(np.asarray(sensor_scale, dtype=np.float64), 1e-12, None)
        relevance = tau * np.linalg.norm(a_sens_all / scale[:, np.newaxis], axis=0)
        keep = np.argsort(relevance)[::-1][: int(k)]
        keep = keep[np.argsort(tau[keep])[::-1]]
    else:
        keep = np.argsort(tau)[::-1][: int(k)]

    v_k = v_full[:, keep]
    i_cells = i_cells_all[:, keep]
    a_sens = a_sens_all[:, keep]

    # exact mode flux on the full grid via the grid's own FD machinery
    psi_grid = np.zeros((grid.flat_r.size, len(keep)))
    cell_area = grid.dr * grid.dz
    for m in range(len(keep)):
        jphi = np.zeros(grid.flat_r.size)
        jphi[grid.cells] = i_cells[:, m] / cell_area
        rhs2d = (-(2.0 * np.pi * MU0) * grid.flat_r * jphi).reshape(grid.nz, grid.nr)
        psi_b2d = np.zeros((grid.nz, grid.nr))
        psi_b2d.ravel()[grid.edge_idx] = grid.g_edge @ i_cells[:, m]
        psi_grid[:, m] = grid.solve_dirichlet(rhs2d, psi_b2d).ravel()

    return ScreeningBasis(
        tau=tau[keep],
        v=v_k,
        i_cells=i_cells,
        a_sens=a_sens,
        psi_grid=psi_grid,
        m_coil=v_k.T @ circuit.m_coil,
        lmat=circuit.lmat,
        r_diag=r_diag,
        tiling=circuit.tiling,
    )


def screening_trajectory(
    basis: ScreeningBasis,
    times: np.ndarray,
    *,
    i_pf_of_t: np.ndarray | None = None,
    i_backbone_patch: np.ndarray | None = None,
    psi_extra_m: np.ndarray | None = None,
) -> np.ndarray:
    """Exact-ZOH screening-mode amplitudes along a drive history.

    Decomposing the patch state as backbone + screening deviation, the mode
    dynamics are ``da/dt + a/τ = −dΨ_m/dt − vᵀR·i_b`` with the linked flux
    ``Ψ_m = vᵀ(M_pc·I_c + L·i_b)`` — the coil swing plus the backbone's own
    flux history (the fixed-mutual trap avoided exactly as the vessel
    trajectory), and the resistive term the EMF the non-solution backbone
    leaves behind.  ``i_backbone_patch`` (T, P) is the backbone patch-current
    history (e.g. the pass-1 fits binned onto the tiling); None drops the
    backbone terms.  ``psi_extra_m`` (T, k) is additional linked flux per
    mode from external conductors the coil channels do not carry — e.g. the
    vessel-eddy history predicted from the measured drives — so all of the
    machine's dΦ/dt terms enter the mode dynamics.  Returns amplitudes (T, k)
    in L-orthonormal coordinates.
    """
    times = np.asarray(times, dtype=np.float64)
    k = basis.n_modes
    psi_m = np.zeros((times.size, k))
    volt_m = np.zeros((times.size, k))
    if i_pf_of_t is not None and basis.m_coil.shape[1]:
        psi_m += np.asarray(i_pf_of_t, dtype=np.float64) @ basis.m_coil.T
    if i_backbone_patch is not None:
        ib = np.asarray(i_backbone_patch, dtype=np.float64)
        psi_m += ib @ (basis.v.T @ basis.lmat).T
        volt_m -= ib @ (basis.v.T @ np.diag(basis.r_diag)).T
    if psi_extra_m is not None:
        psi_m += np.asarray(psi_extra_m, dtype=np.float64)
    a, _u = integrate_eddy_ode(basis.tau, times, psi_m, volt_m=volt_m)
    return a


def screening_sidecar(
    basis: ScreeningBasis,
    sensor_scale: np.ndarray,
) -> dict:
    """Fit-ready mode columns in the passive-sidecar contract.

    Columns are normalised so a unit fitted amplitude produces a unit-norm
    whitened sensor signal (the :func:`gs_solve.build_passive_sidecar`
    convention — without it the ampere-scale columns are invisible next to
    the Ip-normalised profile columns).  ``amp_scale`` converts a PHYSICAL
    mode amplitude (the trajectory's L-orthonormal coordinates) into the
    fit's whitened variable: ``x = amp_scale · a_physical`` — the prior
    centre must be passed through it.
    """
    scale = np.clip(np.asarray(sensor_scale, dtype=np.float64), 1e-12, None)
    white = basis.a_sens / scale[:, np.newaxis]
    norms = np.linalg.norm(white, axis=0)
    norms = np.clip(norms, 1e-12, None)
    return {
        "g_cols": basis.a_sens / norms[np.newaxis, :],
        "psi_cols": basis.psi_grid / norms[np.newaxis, :],
        "k": basis.n_modes,
        "modes": None,
        "amp_scale": norms,
    }


def stack_sidecars(base: dict | None, extra: dict) -> dict:
    """Concatenate two sidecar column sets (base first, extra after).

    The combined dict feeds :func:`gs_solve.solve_equilibrium_lsq` unchanged;
    prior-centre vectors must be assembled in the same concatenated order.
    ``modes`` is dropped (amplitude→circuit-current mapping is not defined
    across heterogeneous blocks).
    """
    if base is None:
        return dict(extra)
    return {
        "g_cols": np.hstack([base["g_cols"], extra["g_cols"]]),
        "psi_cols": np.hstack([base["psi_cols"], extra["psi_cols"]]),
        "br_cols": None,
        "bz_cols": None,
        "k": int(base["k"]) + int(extra["k"]),
        "modes": None,
    }


def bin_cell_currents(tiling: PatchTiling, cell_currents: np.ndarray) -> np.ndarray:
    """Patch currents [A] from in-limiter cell currents (chain remap)."""
    c = np.asarray(cell_currents, dtype=np.float64)[tiling.cell_index]
    return np.bincount(tiling.owner, weights=c, minlength=tiling.n_patches)


__all__ = [
    "MU0",
    "CoupledCircuit",
    "PatchTiling",
    "PinnedFluxSolve",
    "PlasmaCircuit",
    "ScreeningBasis",
    "bin_cell_currents",
    "build_coupled_circuit",
    "plasma_self_inductance",
    "solve_pinned_plasma_circuit",
    "build_plasma_circuit",
    "build_plasma_circuit_from_state",
    "circuit_eigensystem",
    "coupled_loop_voltage_for_ip",
    "evolve_coupled",
    "evolve_patch_currents",
    "loop_voltage_for_ip",
    "mean_plasma_linked_flux",
    "patch_external_linkage",
    "screening_eigenbasis",
    "screening_sidecar",
    "screening_trajectory",
    "stack_sidecars",
    "steady_state_currents",
    "tile_core_patches",
]
