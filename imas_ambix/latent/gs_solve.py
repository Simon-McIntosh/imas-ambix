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

Scheme (standard free-boundary Picard, FreeGS-style):

    Δ*ψ = −μ0 R jφ(ψ_N; θ)   with   jφ = λ·(β0·R/R0 + (1−β0)·R0/R)·(1−ψ_N)

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

        interior = np.zeros((self.nz, self.nr), dtype=bool)
        interior[1:-1, 1:-1] = True
        self.interior = interior
        self.edge_idx = np.where(~interior.ravel())[0]
        self._lu = self._factorise()
        # plasma-cell → domain-edge ψ Green's matrix (edge is far from interior
        # cells, so point filaments per cell are accurate there)
        er, ez = self.flat_r[self.edge_idx], self.flat_z[self.edge_idx]
        cols = [
            op.greens_psi(er, ez, float(self.flat_r[c]), float(self.flat_z[c]))
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
                wr = max(f.width, 0.01)
                wz = max(f.height, 0.01)
                # subdivide over the physical winding pack: the coil is a
                # distributed conductor, and grid points can sit within cm of
                # in-vessel coils where a point filament would be singular
                for orr in (-1 / 3, 0.0, 1 / 3):
                    for ozz in (-1 / 3, 0.0, 1 / 3):
                        a = max(f.r + orr * wr, 1e-3)
                        acc += (
                            f.xmult
                            * op.greens_psi(
                                flat_r, flat_z, float(a), float(f.z + ozz * wz)
                            )
                            / 9.0
                        )
            return acc

        cols = []
        for circs in fwd.pf_merged_circuits:
            per = [circ_col(c) for c in circs]
            cols.append(np.mean(per, axis=0))
        coil_cols = np.column_stack(cols) if cols else np.zeros((flat_r.size, 0))
        return cls(
            rg=rg,
            zg=zg,
            limiter_r=lr,
            limiter_z=lz,
            coil_psi_columns=coil_cols,
            r0=fwd.r0,
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


def _read_axis(
    psi2d: np.ndarray, grid: EquilibriumGrid, sign: float
) -> tuple[tuple[float, float], float]:
    """Sign-aware in-polygon axis; grid-max fallback for early iterations."""
    cp = find_critical_points(psi2d, grid.rg, grid.zg)
    if cp.o_points.shape[0]:
        ins = _inside_polygon(
            cp.o_points[:, 0], cp.o_points[:, 1], grid.limiter_r, grid.limiter_z
        )
        if ins.any():
            pts, vals = cp.o_points[ins], cp.o_psi[ins]
            k = int(np.argmax(sign * vals))
            return (float(pts[k, 0]), float(pts[k, 1])), float(vals[k])
    flat = np.where(grid.inside_limiter.ravel(), sign * psi2d.ravel(), -np.inf)
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
        )
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
) -> EquilibriumResult:
    """Free-boundary Picard solve for one time slice.

    ``i_pf`` [A] are the KNOWN coil currents; ``ip_amperes`` the measured
    plasma current (its sign selects the axis extremum orientation); ``beta0``
    and ``alpha`` are the profile parameters θ the encoder (or a per-slice
    fit) supplies.
    """
    psi_coil = grid.coil_psi(np.asarray(i_pf, dtype=np.float64))
    sign = 1.0 if ip_amperes >= 0 else -1.0
    cell_area = grid.dr * grid.dz

    # compact plasma-like seed at the geometric centre — a uniform fill has no
    # interior O-point and the iteration locks onto corner fixed points
    jphi = np.zeros(grid.flat_r.size)
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

        psi_edge = psi_coil[grid.edge_idx] + grid.g_edge @ i_cell
        rhs2d = (-(MU0) * grid.flat_r * jphi * scale).reshape(grid.nz, grid.nr)
        psi_b2d = np.zeros((grid.nz, grid.nr))
        psi_b2d.ravel()[grid.edge_idx] = psi_edge
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


__all__ = [
    "MU0",
    "profile_jphi_shape",
    "EquilibriumGrid",
    "EquilibriumResult",
    "solve_equilibrium",
]
