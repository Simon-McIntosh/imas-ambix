"""Free-boundary Grad-Shafranov equilibrium solver — the force-balanced ψ decoder.

Why this exists (the representation finding): a LINEAR low-DOF current basis
spread over the vessel cannot produce an interior magnetic axis — the coil
vertical field dominates a broad, low-peakedness current, so the total ψ has no
closed flux region (demonstrated on real MAST data).  Localisation is
inherently nonlinear: the current lives inside the LCFS, the LCFS is a level
set of ψ, and ψ depends on the current — a fixed point.  That fixed point IS
the Grad-Shafranov equilibrium, so the ψ used for topology readouts must come
from an actual GS solve.  This is the pre-authorised fallback of the locked
``latent-to-psi-representation`` decision ("coarse ψ grid + 5-point Δ* stencil
+ free-boundary BC from known amc currents") for exactly the underfit condition
now demonstrated.

Scheme (standard free-boundary Picard, FreeGS-style), in TOTAL flux
Φ = 2π R A_φ [Wb] (the convention every Green's column here carries):

    Δ*Φ = −2π μ0 R jφ(ψ_N; θ)   with   jφ = λ·(β0·R/R0 + (1−β0)·R0/R)·(1−ψ_N)

inside the core (the axis-connected region with ψ_N < 1), zero elsewhere; λ is
rescaled every iteration so the total current equals the measured Ip (the
Rogowski constraint — a raw measurement, not a reconstruction).  Dirichlet
boundary values are re-evaluated each iteration from the current sources via
Green's functions (coil filaments subdivided over their physical winding pack;
plasma cells as filaments — the boundary is far from interior cells).  The
axis (sign-aware O-point inside the limiter polygon), the boundary flux
(innermost in-polygon X-point, else the limiter contact flux), and the
axis-connected core mask are read every iteration with the topology module.

All inputs are raw measurements or fixed machine geometry — no EFIT output
enters anywhere (the profile ansatz is a generic parametric family, pinned by
a static firewall test).  Non-converged slices carry ``converged=False`` and
must be masked by consumers, never silently used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy import ndimage  # type: ignore[import-untyped]

from imas_ambix.gs import operator as op
from imas_ambix.gs.cylinder import hybrid_greens
from imas_ambix.latent.topology import _inside_polygon, find_critical_points

if TYPE_CHECKING:
    from imas_ambix.gs.geometry import GeometryTable

MU0 = 4.0e-7 * np.pi


def profile_jphi_shape(
    psi_n: np.ndarray, r: np.ndarray, *, r0: float, beta0: float, alpha: float = 1.0
) -> np.ndarray:
    """The jφ(ψ_N) ansatz: ``(β0·R/R0 + (1−β0)·R0/R)·(1−ψ_N)^α`` inside ψ_N<1.

    A generic parametric family (the standard EFIT-class two-term form: the
    R/R0 term carries the pressure-gradient drive, the R0/R term the FF′
    drive); β0 ∈ [0,1] splits them, α sets peakedness.  Zero at and beyond the
    boundary.  Amplitude is NOT part of the ansatz — the solver rescales to
    the measured Ip every iteration.
    """
    psi_n = np.asarray(psi_n, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    inside = psi_n < 1.0
    shape = np.zeros_like(psi_n)
    rr = np.maximum(r, 1e-3)
    base = beta0 * rr / r0 + (1.0 - beta0) * r0 / rr
    shape[inside] = base[inside] * np.power(1.0 - psi_n[inside], alpha)
    return shape


@dataclass
class EquilibriumResult:
    """One converged (or honestly non-converged) free-boundary equilibrium."""

    psi: np.ndarray  # (nz, nr) total poloidal flux [Wb]
    axis: tuple[float, float]  # magnetic axis (R, Z) [m]
    axis_psi: float
    boundary_psi: float
    jphi: np.ndarray  # (nz, nr) toroidal current density [A/m^2]
    cell_currents: np.ndarray  # (n_cells,) per-cell current [A] (inside limiter)
    core_mask: np.ndarray  # (nz, nr) bool — axis-connected plasma region
    converged: bool
    residual: float  # last relative Picard update
    iterations: int


class EquilibriumGrid:
    """Per-campaign fixed geometry: grid, Δ* factorisation, Green's matrices.

    Everything here is pure device geometry, built once per campaign and
    reused across slices; a solve is then a handful of triangular solves and
    matmuls.
    """

    def __init__(
        self,
        *,
        rg: np.ndarray,
        zg: np.ndarray,
        limiter_r: np.ndarray,
        limiter_z: np.ndarray,
        coil_psi_columns: np.ndarray,  # (N, n_coil) ψ per unit coil current
        r0: float,
        conductor_rects: np.ndarray | None = None,  # (n, 4) r0, r1, z0, z1 packs
    ) -> None:
        self.rg = rg
        self.zg = zg
        self.nr = rg.size
        self.nz = zg.size
        self.dr = float(rg[1] - rg[0])
        self.dz = float(zg[1] - zg[0])
        self.limiter_r = np.asarray(limiter_r, dtype=np.float64)
        self.limiter_z = np.asarray(limiter_z, dtype=np.float64)
        self.r0 = float(r0)
        mesh_r, mesh_z = np.meshgrid(rg, zg)
        self.mesh_r = mesh_r
        self.mesh_z = mesh_z
        self.flat_r = mesh_r.ravel()
        self.flat_z = mesh_z.ravel()
        self.inside_limiter = _inside_polygon(
            self.flat_r, self.flat_z, self.limiter_r, self.limiter_z
        ).reshape(self.nz, self.nr)
        self.cells = np.where(self.inside_limiter.ravel())[0]
        self._coil_psi_columns = coil_psi_columns
        self.conductor_rects = (
            np.asarray(conductor_rects, dtype=np.float64)
            if conductor_rects is not None and len(conductor_rects)
            else np.zeros((0, 4))
        )

        interior = np.zeros((self.nz, self.nr), dtype=bool)
        interior[1:-1, 1:-1] = True
        self.interior = interior
        self.edge_idx = np.where(~interior.ravel())[0]
        self._lu = self._factorise()
        # plasma-cell → domain-edge ψ Green's matrix (edge is far from interior
        # cells, so point filaments per cell are accurate there)
        er, ez = self.flat_r[self.edge_idx], self.flat_z[self.edge_idx]
        cols = [
            hybrid_greens(
                er, ez, float(self.flat_r[c]), float(self.flat_z[c]), self.dr, self.dz
            )[0]
            for c in self.cells
        ]
        self.g_edge = np.column_stack(cols) if cols else np.zeros((er.size, 0))
        # nearest grid index of each limiter vertex (for the limiter-contact flux)
        self._limiter_grid_idx = np.array(
            [
                int(np.argmin(np.hypot(self.flat_r - lr, self.flat_z - lz)))
                for lr, lz in zip(self.limiter_r, self.limiter_z, strict=True)
            ]
        )
        # topology-candidate mask: inside the limiter AND clear of conductors.
        # The exact finite-area coil field has genuine extrema/saddles at every
        # winding pack — real field structure, but conductor interior, never a
        # plasma axis or plasma X-point.  Packs that straddle the limiter
        # contour (MAST in-vessel divertor coils) otherwise capture the axis
        # read and relocate the core mask onto the coil.
        self.topology_candidate = (
            self.inside_limiter.ravel()
            & self.clear_of_conductors(self.flat_r, self.flat_z)
        )

    def clear_of_conductors(self, r: np.ndarray, z: np.ndarray) -> np.ndarray:
        """True where (r, z) is outside every winding-pack rectangle.

        Rectangles are dilated by one grid cell — the reach of the discrete
        Hessian the critical-point finder evaluates.
        """
        r = np.asarray(r, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)
        clear = np.ones(r.shape, dtype=bool)
        for r0, r1, z0, z1 in self.conductor_rects:
            inside = (
                (r >= r0 - self.dr)
                & (r <= r1 + self.dr)
                & (z >= z0 - self.dz)
                & (z <= z1 + self.dz)
            )
            clear &= ~inside
        return clear

    # ---- construction ----

    @classmethod
    def from_table(
        cls, table: GeometryTable, *, nr: int = 65, nz: int = 97
    ) -> EquilibriumGrid:
        lr = np.asarray(table.limiter_r, dtype=np.float64)
        lz = np.asarray(table.limiter_z, dtype=np.float64)
        rg = np.linspace(max(float(lr.min()), 0.06), float(lr.max()), nr)
        zg = np.linspace(float(lz.min()), float(lz.max()), nz)
        mesh_r, mesh_z = np.meshgrid(rg, zg)
        flat_r, flat_z = mesh_r.ravel(), mesh_z.ravel()

        fwd = op.build_operator(table)
        by_circ: dict[int, list] = {}
        for f in table.pf_filaments:
            by_circ.setdefault(f.circuit, []).append(f)

        def circ_col(circ: int) -> np.ndarray:
            acc = np.zeros(flat_r.size)
            for f in by_circ[circ]:
                # finite-area winding pack: smooth and exact everywhere,
                # including AT in-vessel coils inside the solve domain
                # efm carries SIGNED pack extents (height < 0 occurs on real
                # tables) — the physical size is |extent|; clamping a negative
                # to the floor collapsed 29 cm solenoid packs to 1 cm
                psi_f, _br, _bz = hybrid_greens(
                    flat_r,
                    flat_z,
                    float(f.r),
                    float(f.z),
                    max(abs(f.width), 0.01),
                    max(abs(f.height), 0.01),
                )
                acc += f.xmult * psi_f
            return acc

        cols = []
        for circs in fwd.pf_merged_circuits:
            per = [circ_col(c) for c in circs]
            cols.append(np.mean(per, axis=0))
        coil_cols = np.column_stack(cols) if cols else np.zeros((flat_r.size, 0))
        rects = np.array(
            [
                [
                    f.r - abs(f.width) / 2.0,
                    f.r + abs(f.width) / 2.0,
                    f.z - abs(f.height) / 2.0,
                    f.z + abs(f.height) / 2.0,
                ]
                for f in table.pf_filaments
            ]
        )
        return cls(
            rg=rg,
            zg=zg,
            limiter_r=lr,
            limiter_z=lz,
            coil_psi_columns=coil_cols,
            r0=fwd.r0,
            conductor_rects=rects,
        )

    def _factorise(self):
        """LU-factorise the 5-point Δ* operator with Dirichlet edge rows."""
        nr, nz = self.nr, self.nz
        n = nr * nz
        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []
        for i in range(nz):
            for j in range(nr):
                k = i * nr + j
                if not self.interior[i, j]:
                    rows.append(k)
                    cols.append(k)
                    vals.append(1.0)
                    continue
                rj = self.rg[j]
                rp = 0.5 * (self.rg[j] + self.rg[j + 1])
                rm = 0.5 * (self.rg[j] + self.rg[j - 1])
                ce = rj / (rp * self.dr * self.dr)
                cw = rj / (rm * self.dr * self.dr)
                cn = 1.0 / (self.dz * self.dz)
                rows += [k, k, k, k, k]
                cols += [k, k + 1, k - 1, k + nr, k - nr]
                vals += [-(ce + cw + 2.0 * cn), ce, cw, cn, cn]
        a = sp.csc_matrix((vals, (rows, cols)), shape=(n, n))
        return spla.splu(a)

    # ---- primitive solves ----

    def solve_dirichlet(
        self, rhs_interior: np.ndarray, psi_boundary: np.ndarray
    ) -> np.ndarray:
        """Solve Δ*ψ = rhs with Dirichlet values from ``psi_boundary`` edge cells.

        ``rhs_interior`` and ``psi_boundary`` are (nz, nr) fields; only the
        interior of the former and the edge ring of the latter are read.
        Returns ψ as (nz, nr).
        """
        rhs = np.where(self.interior, rhs_interior, 0.0).ravel().astype(np.float64)
        rhs[self.edge_idx] = psi_boundary.ravel()[self.edge_idx]
        return self._lu.solve(rhs).reshape(self.nz, self.nr)

    def coil_psi(self, i_pf: np.ndarray) -> np.ndarray:
        """Vacuum coil ψ on the flattened grid [Wb] for coil currents [A]."""
        if self._coil_psi_columns.shape[1] == 0:
            return np.zeros(self.flat_r.size)
        return self._coil_psi_columns @ np.asarray(i_pf, dtype=np.float64)

    def sensor_greens(self, table: GeometryTable) -> tuple[np.ndarray, list[str]]:
        """Cell→sensor Green's matrix ``(n_sensor, n_cells)`` + channel names.

        Rows follow the mapped sensors in ``table.sensor_map`` order: flux
        loops get ψ [Wb per A], B-probes the orientation-projected field
        [T per A] of a unit filament at each in-limiter grid cell.  Cached on
        first call (pure geometry).
        """
        cached = getattr(self, "_sensor_greens_cache", None)
        if cached is not None:
            return cached

        rows: list[np.ndarray] = []
        channels: list[str] = []
        cr = self.flat_r[self.cells]
        cz = self.flat_z[self.cells]
        for m in table.sensor_map:
            row = np.empty(cr.size)
            ang = np.deg2rad(m.angle_deg if m.angle_deg is not None else 90.0)
            for k, (a, z0) in enumerate(zip(cr, cz, strict=True)):
                psi_k, br_k, bz_k = hybrid_greens(
                    np.array([m.r]),
                    np.array([m.z]),
                    float(a),
                    float(z0),
                    self.dr,
                    self.dz,
                )
                if m.kind == "flux_loop":
                    row[k] = psi_k[0]
                else:
                    row[k] = br_k[0] * np.cos(ang) + bz_k[0] * np.sin(ang)
            rows.append(row)
            channels.append(m.amb_channel)
        g = np.vstack(rows) if rows else np.zeros((0, cr.size))
        self._sensor_greens_cache = (g, channels)
        return g, channels


def _read_axis(
    psi2d: np.ndarray, grid: EquilibriumGrid, sign: float
) -> tuple[tuple[float, float], float]:
    """Sign-aware in-polygon, conductor-clear axis; grid-max fallback."""
    cp = find_critical_points(psi2d, grid.rg, grid.zg)
    if cp.o_points.shape[0]:
        ins = _inside_polygon(
            cp.o_points[:, 0], cp.o_points[:, 1], grid.limiter_r, grid.limiter_z
        ) & grid.clear_of_conductors(cp.o_points[:, 0], cp.o_points[:, 1])
        if ins.any():
            pts, vals = cp.o_points[ins], cp.o_psi[ins]
            k = int(np.argmax(sign * vals))
            return (float(pts[k, 0]), float(pts[k, 1])), float(vals[k])
    flat = np.where(grid.topology_candidate, sign * psi2d.ravel(), -np.inf)
    k = int(np.argmax(flat))
    return (float(grid.flat_r[k]), float(grid.flat_z[k])), float(psi2d.ravel()[k])


def _read_boundary_psi(
    psi2d: np.ndarray, grid: EquilibriumGrid, axis_psi: float
) -> float:
    """Innermost in-polygon X-point flux, else the limiter-contact flux."""
    cp = find_critical_points(psi2d, grid.rg, grid.zg)
    xb = None
    if cp.x_points.shape[0]:
        ins = _inside_polygon(
            cp.x_points[:, 0], cp.x_points[:, 1], grid.limiter_r, grid.limiter_z
        ) & grid.clear_of_conductors(cp.x_points[:, 0], cp.x_points[:, 1])
        if ins.any():
            xpsi = cp.x_psi[ins]
            xb = float(xpsi[int(np.argmin(np.abs(xpsi - axis_psi)))])
    lim_vals = psi2d.ravel()[grid._limiter_grid_idx]
    lim_b = float(lim_vals[int(np.argmin(np.abs(lim_vals - axis_psi)))])
    if xb is not None and abs(xb - axis_psi) < abs(lim_b - axis_psi):
        return xb
    return lim_b


def solve_equilibrium(
    grid: EquilibriumGrid,
    i_pf: np.ndarray,
    ip_amperes: float,
    *,
    beta0: float = 0.5,
    alpha: float = 1.0,
    max_iterations: int = 80,
    relax: float = 0.5,
    tolerance: float = 3e-4,
    seed_width: tuple[float, float] = (0.35, 0.5),
    coil_field_mode: str = "analytic-add",
    initial_jphi: np.ndarray | None = None,
    iteration_trace: list[dict] | None = None,
) -> EquilibriumResult:
    """Free-boundary Picard solve for one time slice.

    ``i_pf`` [A] are the KNOWN coil currents; ``ip_amperes`` the measured
    plasma current (its sign selects the axis extremum orientation); ``beta0``
    and ``alpha`` are the profile parameters θ the encoder (or a per-slice
    fit) supplies.

    ``coil_field_mode`` selects how the coil field enters the total ψ:
    ``"analytic-add"`` (default) solves the FD problem for the plasma part
    only and adds the exact finite-area coil field — correct at and inside
    the in-vessel coils; ``"boundary-continuation"`` reproduces the legacy
    structure (total-ψ Dirichlet BCs, the coil field entering as the
    Δ*-harmonic continuation of its boundary values — smooth but wrong near
    in-vessel coils), retained as a diagnostic arm.  ``iteration_trace``
    (a list) collects per-iteration axis / flux / residual dicts.
    """
    psi_coil = grid.coil_psi(np.asarray(i_pf, dtype=np.float64))
    sign = 1.0 if ip_amperes >= 0 else -1.0
    cell_area = grid.dr * grid.dz

    # compact plasma-like seed at the geometric centre — a uniform fill has no
    # interior O-point and the iteration locks onto corner fixed points.  A
    # caller-supplied ``initial_jphi`` (e.g. a converged distribution from an
    # easier configuration) replaces the seed for homotopy restarts.
    if initial_jphi is not None:
        jphi = np.where(
            grid.inside_limiter.ravel(),
            np.asarray(initial_jphi, dtype=np.float64).ravel(),
            0.0,
        )
        if not np.isfinite(jphi).all() or abs(jphi.sum()) < 1e-12:
            jphi = np.zeros(grid.flat_r.size)
    else:
        jphi = np.zeros(grid.flat_r.size)
    if abs(jphi.sum()) < 1e-12:
        jphi[grid.cells] = np.exp(
            -(
                ((grid.flat_r[grid.cells] - grid.r0) / seed_width[0]) ** 2
                + (grid.flat_z[grid.cells] / seed_width[1]) ** 2
            )
        )

    psi_flat: np.ndarray | None = None
    residual = np.inf
    axis = (grid.r0, 0.0)
    axis_psi = 0.0
    boundary_psi = 0.0
    core = grid.inside_limiter.copy()

    for iteration in range(1, max_iterations + 1):
        i_cell = jphi[grid.cells] * cell_area
        total = i_cell.sum()
        scale = ip_amperes / total if abs(total) > 1e-12 else 0.0
        i_cell = i_cell * scale

        # Solve the PLASMA part only (plasma RHS + plasma-only Green's BCs) and
        # add the coil field analytically: MAST's in-vessel coils sit INSIDE
        # the solve domain, where their field is not harmonic — Dirichlet
        # continuation of a total-psi BC would misrepresent it near the coils.
        # The finite-area coil columns are exact everywhere instead.
        # All Green's columns (coil, g_edge, sensors) carry TOTAL flux
        # Φ = 2π R A_φ [Wb], so the matching FD source is Δ*Φ = −2π μ0 R jφ
        # (per-radian −μ0 R jφ under-weights the plasma well by 2π against
        # the coil field — pinned by the flux-integral consistency test).
        rhs2d = (
            -(2.0 * np.pi * MU0) * grid.flat_r * jphi * scale
        ).reshape(grid.nz, grid.nr)
        psi_b2d = np.zeros((grid.nz, grid.nr))
        if coil_field_mode == "analytic-add":
            psi_b2d.ravel()[grid.edge_idx] = grid.g_edge @ i_cell
            psi_new = grid.solve_dirichlet(rhs2d, psi_b2d).ravel() + psi_coil
        else:  # boundary-continuation — the legacy diagnostic arm
            psi_b2d.ravel()[grid.edge_idx] = (
                psi_coil[grid.edge_idx] + grid.g_edge @ i_cell
            )
            psi_new = grid.solve_dirichlet(rhs2d, psi_b2d).ravel()

        if psi_flat is None:
            psi_flat = psi_new
        else:
            residual = float(
                np.abs(psi_new - psi_flat).max() / max(np.abs(psi_new).max(), 1e-12)
            )
            psi_flat = relax * psi_new + (1.0 - relax) * psi_flat

        psi2d = psi_flat.reshape(grid.nz, grid.nr)
        axis, axis_psi = _read_axis(psi2d, grid, sign)
        boundary_psi = _read_boundary_psi(psi2d, grid, axis_psi)
        span = boundary_psi - axis_psi
        if abs(span) < 1e-12:
            span = 1e-12
        psi_n = (psi_flat - axis_psi) / span

        closed = ((psi_n < 1.0) & grid.inside_limiter.ravel()).reshape(grid.nz, grid.nr)
        labels, _ = ndimage.label(closed)
        ia = int(np.argmin(np.abs(grid.zg - axis[1])))
        ja = int(np.argmin(np.abs(grid.rg - axis[0])))
        core_label = labels[ia, ja]
        core = (labels == core_label) if core_label != 0 else closed

        jphi = np.zeros_like(jphi)
        shape = profile_jphi_shape(
            psi_n, grid.flat_r, r0=grid.r0, beta0=beta0, alpha=alpha
        )
        jphi[core.ravel()] = shape[core.ravel()]

        if iteration_trace is not None:
            iteration_trace.append(
                {
                    "iteration": iteration,
                    "axis": axis,
                    "axis_psi": axis_psi,
                    "boundary_psi": boundary_psi,
                    "residual": residual if np.isfinite(residual) else None,
                    "core_cells": int(core.sum()),
                }
            )

        if iteration > 5 and residual < tolerance:
            break

    i_cell = jphi[grid.cells] * cell_area
    total = i_cell.sum()
    scale = ip_amperes / total if abs(total) > 1e-12 else 0.0
    i_cell = i_cell * scale
    jphi_final = (jphi * scale).reshape(grid.nz, grid.nr)

    return EquilibriumResult(
        psi=psi_flat.reshape(grid.nz, grid.nr)
        if psi_flat is not None
        else np.zeros((grid.nz, grid.nr)),
        axis=axis,
        axis_psi=axis_psi,
        boundary_psi=boundary_psi,
        jphi=jphi_final,
        cell_currents=i_cell,
        core_mask=core,
        converged=bool(residual < tolerance),
        residual=residual,
        iterations=iteration,
    )


def solve_equilibrium_bootstrapped(
    grid: EquilibriumGrid,
    i_pf: np.ndarray,
    ip_amperes: float,
    *,
    beta0: float = 0.5,
    alpha: float = 1.0,
    max_iterations: int = 80,
    bootstrap_iterations: int = 30,
    **kwargs,
) -> EquilibriumResult:
    """Two-stage solve: legacy-continuation bootstrap, then the exact field.

    With the exact in-vessel coil field the broad Gaussian seed has no
    interior O-point at iteration 1 (the correct coil well is deeper than the
    seed's ψ bump), so plain Picard locks onto a corner attractor.  The legacy
    boundary-continuation field is smoother and bootstraps reliably; its
    (possibly unconverged) current distribution then seeds the analytic-add
    Picard, which is the only result reported — the physically-correct field.
    """
    stage1 = solve_equilibrium(
        grid,
        i_pf,
        ip_amperes,
        beta0=beta0,
        alpha=alpha,
        max_iterations=bootstrap_iterations,
        coil_field_mode="boundary-continuation",
        **kwargs,
    )
    return solve_equilibrium(
        grid,
        i_pf,
        ip_amperes,
        beta0=beta0,
        alpha=alpha,
        max_iterations=max_iterations,
        initial_jphi=stage1.jphi.ravel(),
        **kwargs,
    )


@dataclass
class ProfileFit:
    """A per-slice profile fit: parameters, equilibrium, and whitened cost."""

    beta0: float
    alpha: float
    cost: float  # RMS whitened magnetics residual at the optimum
    result: EquilibriumResult


def fit_profile(
    grid: EquilibriumGrid,
    table: GeometryTable,
    *,
    i_pf: np.ndarray,
    ip_amperes: float,
    measured: np.ndarray,
    vacuum_prediction: np.ndarray,
    sensor_scale: np.ndarray,
    sensor_mask: np.ndarray,
    beta0_grid: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9),
    alpha_grid: tuple[float, ...] = (1.0,),
    convergence_limit: float = 5e-3,
) -> ProfileFit | None:
    """Per-slice profile fit: choose (β0, α) minimising the whitened magnetics
    residual over converged equilibria.

    This is the training-free inverse the learned encoder amortises: only that
    slice's measured magnetics, the KNOWN coil currents, and the measured Ip
    enter — no labels, no corpus, no EFIT.  ``measured``, ``vacuum_prediction``
    (the KNOWN-coil sensor field, e.g. ``ForwardOperator.vacuum_prediction``),
    ``sensor_scale`` and ``sensor_mask`` are all in the ``sensor_greens``
    channel order.  Returns None when no candidate converges (the caller must
    mask the slice, never fabricate a readout).
    """
    g_sens, _channels = grid.sensor_greens(table)

    meas = np.asarray(measured, dtype=np.float64)
    vac = np.asarray(vacuum_prediction, dtype=np.float64)
    scale = np.clip(np.asarray(sensor_scale, dtype=np.float64), 1e-12, None)
    mask = np.asarray(sensor_mask, dtype=bool) & np.isfinite(meas)

    best: ProfileFit | None = None
    for alpha in alpha_grid:
        for beta0 in beta0_grid:
            res = solve_equilibrium_bootstrapped(
                grid, i_pf, ip_amperes, beta0=float(beta0), alpha=float(alpha)
            )
            if not res.converged and res.residual > convergence_limit:
                continue
            pred = vac + g_sens @ res.cell_currents
            resid = (pred[mask] - meas[mask]) / scale[mask]
            cost = float(np.sqrt(np.mean(resid * resid))) if mask.any() else np.inf
            if best is None or cost < best.cost:
                best = ProfileFit(
                    beta0=float(beta0), alpha=float(alpha), cost=cost, result=res
                )
    return best


__all__ = [
    "MU0",
    "profile_jphi_shape",
    "EquilibriumGrid",
    "EquilibriumResult",
    "ProfileFit",
    "solve_equilibrium",
    "solve_equilibrium_bootstrapped",
    "fit_profile",
]
