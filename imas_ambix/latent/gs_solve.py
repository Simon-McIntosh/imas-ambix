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
from imas_ambix.latent.topology import (
    CriticalPoints,
    _inside_polygon,
    boundary_flux_robust,
    find_critical_points,
)

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

    def plasma_greens_cells(self) -> dict:
        """Analytic thick-cylinder Green's matrices, in-limiter cell → in-limiter
        grid point, for the annulus soft-prior penalty.

        Returns ``{"cells", "psi", "br", "bz"}`` where ``psi``/``br``/``bz`` are
        ``(n_cells, n_cells)`` per-ampere total-flux ψ [Wb] and poloidal field
        [T] evaluated at every in-limiter grid point (rows, ``== self.cells``
        order) from a unit current in each in-limiter cell (columns, same order).
        ALL THREE come straight from the finite-area cylinder Biot–Savart kernel
        (the double cross-section integral) — ψ AND its field are analytic, so the
        grad-ψ annulus penalty needs NO finite differences and the matrices match
        the §2 annulus-consistency metric's own analytic carrier ψ.  The default
        near/far switch is kept deliberately: the annulus is the VACUUM region far
        from every current cell, exactly where the finite-area correction is the
        constant second-moment term (<0.2% in ψ, ~30× below the annulus
        consistency-RMS noise floor).  Forcing the full cylinder form at every
        distance was measured at ~3440 s vs ~6 s per grid (546×) for a <0.2%
        change in the far field — pointless here, so the near-band full form +
        far-field filament (identical to <0.2% wherever they differ) is used.
        The annulus point set is always a subset of ``self.cells``, so a slice
        indexes annulus rows directly.  Cached (pure geometry).
        """
        cached = getattr(self, "_plasma_greens_cells_cache", None)
        if cached is not None:
            return cached
        cr = self.flat_r[self.cells]
        cz = self.flat_z[self.cells]
        n = self.cells.size
        gpsi = np.empty((n, n), dtype=np.float64)
        gbr = np.empty((n, n), dtype=np.float64)
        gbz = np.empty((n, n), dtype=np.float64)
        for j, c in enumerate(self.cells):
            psi, br, bz = hybrid_greens(
                cr, cz, float(self.flat_r[c]), float(self.flat_z[c]), self.dr, self.dz
            )
            gpsi[:, j], gbr[:, j], gbz[:, j] = psi, br, bz
        out = {"cells": self.cells, "psi": gpsi, "br": gbr, "bz": gbz}
        self._plasma_greens_cells_cache = out
        return out


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


def _read_boundary_psi_robust(
    psi2d: np.ndarray,
    grid: EquilibriumGrid,
    axis: tuple[float, float],
    axis_psi: float,
    *,
    smooth_sigma: float = 0.0,
    min_axis_dist: float = 0.0,
) -> float:
    """Opt-in saddle-robust variant of :func:`_read_boundary_psi`.

    Two independent, composable robustifications of the LCFS boundary read
    (measured negative-then-positive: free-current ψ under-sizes the flat-top
    LCFS by tens of cm because the naive innermost-ψ pick locks onto a
    discretisation-scale saddle in the current's sensor-null-space
    concentration, not the genuine separatrix):

    ``smooth_sigma`` (grid cells): find candidate X-points on a
    Gaussian-smoothed copy of ψ instead of the raw field — small-scale ripple
    saddles vanish under smoothing while a genuine broad separatrix survives.

    ``min_axis_dist`` (metres): reject any candidate X-point closer than this
    to the magnetic axis before the innermost-ψ selection — a saddle that
    bounds the WHOLE confined region is never a hair's-breadth from the axis.

    ``smooth_sigma=0.0`` and ``min_axis_dist=0.0`` (defaults) reproduce
    :func:`_read_boundary_psi` exactly.
    """
    field = (
        psi2d if smooth_sigma <= 0.0 else ndimage.gaussian_filter(psi2d, smooth_sigma)
    )
    cp = find_critical_points(field, grid.rg, grid.zg)
    xb = None
    if cp.x_points.shape[0]:
        clear = grid.clear_of_conductors(cp.x_points[:, 0], cp.x_points[:, 1])
        cp_clear = CriticalPoints(
            cp.o_points, cp.o_psi, cp.x_points[clear], cp.x_psi[clear]
        )
        xb = boundary_flux_robust(
            cp_clear,
            axis,
            axis_psi,
            limiter_r=grid.limiter_r,
            limiter_z=grid.limiter_z,
            min_axis_dist=min_axis_dist,
        )
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
    seed_z0: float = 0.0,
    coil_field_mode: str = "analytic-add",
    initial_jphi: np.ndarray | None = None,
    iteration_trace: list[dict] | None = None,
) -> EquilibriumResult:
    """Free-boundary Picard solve for one time slice.

    ``i_pf`` [A] are the KNOWN coil currents; ``ip_amperes`` the measured
    plasma current (its sign selects the axis extremum orientation); ``beta0``
    and ``alpha`` are the profile parameters θ the encoder (or a per-slice
    fit) supplies.

    ``seed_z0`` [m] shifts the Gaussian seed's vertical centre — with a
    near-marginal vertical field the Picard iteration can admit distinct
    fixed-point branches, and the seed selects which one is approached; the
    default 0.0 reproduces the historical midplane seed exactly.

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
                + ((grid.flat_z[grid.cells] - seed_z0) / seed_width[1]) ** 2
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
        rhs2d = (-(2.0 * np.pi * MU0) * grid.flat_r * jphi * scale).reshape(
            grid.nz, grid.nr
        )
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

    A caller-supplied ``initial_jphi`` (e.g. the converged distribution of the
    previous time slice — temporal warm-starting) replaces the bootstrap
    entirely: the analytic-add Picard runs directly from it.
    """
    warm = kwargs.pop("initial_jphi", None)
    if warm is not None and np.isfinite(warm).all() and abs(np.sum(warm)) > 1e-12:
        return solve_equilibrium(
            grid,
            i_pf,
            ip_amperes,
            beta0=beta0,
            alpha=alpha,
            max_iterations=max_iterations,
            initial_jphi=np.asarray(warm, dtype=np.float64).ravel(),
            **kwargs,
        )
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
    z0: float = 0.0  # fitted seed vertical centre [m] (0.0 = midplane default)


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
    **solve_kwargs,
) -> ProfileFit | None:
    """Per-slice profile fit: choose (β0, α) minimising the whitened magnetics
    residual over converged equilibria.

    This is the training-free inverse the learned encoder amortises: only that
    slice's measured magnetics, the KNOWN coil currents, and the measured Ip
    enter — no labels, no corpus, no EFIT.  ``measured``, ``vacuum_prediction``
    (the KNOWN-coil sensor field, e.g. ``ForwardOperator.vacuum_prediction``),
    ``sensor_scale`` and ``sensor_mask`` are all in the ``sensor_greens``
    channel order.  ``solve_kwargs`` are forwarded to
    :func:`solve_equilibrium_bootstrapped` (e.g. ``max_iterations`` /
    ``tolerance`` for a relaxed-Picard retry pass that lifts convergence
    coverage).  Returns None when no candidate converges (the caller must mask
    the slice, never fabricate a readout).
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
                grid,
                i_pf,
                ip_amperes,
                beta0=float(beta0),
                alpha=float(alpha),
                **solve_kwargs,
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


def fit_profile_continuous(
    grid: EquilibriumGrid,
    table: GeometryTable,
    *,
    i_pf: np.ndarray,
    ip_amperes: float,
    measured: np.ndarray,
    vacuum_prediction: np.ndarray,
    sensor_scale: np.ndarray,
    sensor_mask: np.ndarray,
    x0: tuple[float, ...] = (0.5, 1.5),
    beta0_bounds: tuple[float, float] = (0.02, 0.98),
    alpha_bounds: tuple[float, float] = (0.4, 3.5),
    fit_z0: bool = False,
    z0_bounds: tuple[float, float] = (-0.25, 0.25),
    convergence_limit: float = 5e-3,
    maxfev: int = 60,
    **solve_kwargs,
) -> ProfileFit | None:
    """Continuous bounded (β0, α[, z0]) fit — retires the candidate grid.

    Same contract as :func:`fit_profile` (raw magnetics only, whitened cost,
    None when nothing converges) but the parameters are optimised continuously
    (Nelder–Mead within bounds) instead of enumerated, ``x0`` warm-starts the
    search (e.g. from the previous time slice's optimum), and ``fit_z0`` adds
    the vertical seed-centre degree of freedom (fixed-point branch selection).
    A non-converged candidate is penalised, never scored; the best CONVERGED
    equilibrium seen anywhere in the search is what is returned, so an
    optimiser step into a non-convergent pocket cannot corrupt the result.
    """
    from scipy import optimize  # noqa: PLC0415 — keep module import light

    g_sens, _channels = grid.sensor_greens(table)
    meas = np.asarray(measured, dtype=np.float64)
    vac = np.asarray(vacuum_prediction, dtype=np.float64)
    scale = np.clip(np.asarray(sensor_scale, dtype=np.float64), 1e-12, None)
    mask = np.asarray(sensor_mask, dtype=bool) & np.isfinite(meas)

    bounds = [beta0_bounds, alpha_bounds] + ([z0_bounds] if fit_z0 else [])
    x_start = np.asarray(x0[: 2 + int(fit_z0)], dtype=np.float64)
    if x_start.size < 2 + int(fit_z0):
        x_start = np.concatenate([x_start, [0.0]])
    x_start = np.clip(x_start, [b[0] for b in bounds], [b[1] for b in bounds])

    best: ProfileFit | None = None
    cache: dict[tuple[float, ...], float] = {}

    def objective(x: np.ndarray) -> float:
        nonlocal best
        key = tuple(np.round(x, 4))
        if key in cache:
            return cache[key]
        beta0 = float(np.clip(x[0], *beta0_bounds))
        alpha = float(np.clip(x[1], *alpha_bounds))
        z0 = float(np.clip(x[2], *z0_bounds)) if fit_z0 else 0.0
        res = solve_equilibrium_bootstrapped(
            grid,
            i_pf,
            ip_amperes,
            beta0=beta0,
            alpha=alpha,
            seed_z0=z0,
            **solve_kwargs,
        )
        if not res.converged and res.residual > convergence_limit:
            cost = 1e3 + min(float(res.residual), 1e3)
        else:
            pred = vac + g_sens @ res.cell_currents
            resid = (pred[mask] - meas[mask]) / scale[mask]
            cost = float(np.sqrt(np.mean(resid * resid))) if mask.any() else np.inf
            if best is None or cost < best.cost:
                best = ProfileFit(
                    beta0=beta0, alpha=alpha, cost=cost, result=res, z0=z0
                )
        cache[key] = cost
        return cost

    optimize.minimize(
        objective,
        x_start,
        method="Nelder-Mead",
        bounds=bounds,
        options={
            "maxfev": maxfev,
            "xatol": 5e-3,
            "fatol": 1e-4,
            "initial_simplex": None,
        },
    )
    return best


# ---------------------------------------------------------------------------
# K-coefficient profile-DOF ladder: p′/FF′ as linear basis expansions whose
# coefficients are solved by whitened linear least squares against the raw
# magnetics EVERY Picard sweep (the EFIT pattern), with the measured Ip as a
# hard linear anchor and an optional second-difference smoothness ridge.
# ---------------------------------------------------------------------------


def profile_basis(
    psi_n: np.ndarray,
    r: np.ndarray,
    *,
    r0: float,
    n_p: int,
    n_f: int,
    kind: str = "legendre",
    centrifugal_gamma=None,
) -> np.ndarray:
    """(n_points, n_p + n_f) jφ basis images, zero at and beyond the boundary.

    Columns 0..n_p−1 carry the pressure-gradient drive R/R0·φ_k(ψ_N); columns
    n_p..n_p+n_f−1 the FF′ drive R0/R·φ_k(ψ_N).

    ``kind="legendre"``: φ_k(ψ_N) = P_{k−1}(2ψ_N−1)·(1−ψ_N) (shifted Legendre
    × boundary factor — well-conditioned on [0, 1], exactly zero at ψ_N = 1,
    sign-indefinite).  ``kind="monomial-nonneg"``: φ_k(ψ_N) = (1−ψ_N)^e_k
    with the exponent ladder e = (0.5, 1, 1.5, 2, 3) — every basis function
    ≥ 0, so NON-NEGATIVE coefficients imply jφ ≥ 0 pointwise (the R1
    unidirectional-current fact imposed at the profile level).  The ladder
    starts at a SUB-LINEAR exponent because the continuous fit rails at its
    α = 0.4 lower bound (the calibrated magnetics demand a broader-than-
    linear edge-weighted profile); conditioning is poorer than Legendre but
    fine for ≤ 5 columns per family.
    n_p = n_f = 1 spans the two-term (β0, α=1) family with a free amplitude
    split in either kind.
    """
    from numpy.polynomial import legendre  # noqa: PLC0415

    psi_n = np.asarray(psi_n, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    inside = psi_n < 1.0
    rr = np.maximum(r, 1e-3)
    x = 2.0 * np.clip(psi_n, 0.0, 1.0) - 1.0
    edge = np.where(inside, 1.0 - np.clip(psi_n, 0.0, 1.0), 0.0)
    nonneg_exponents = (0.5, 1.0, 1.5, 2.0, 3.0)
    if centrifugal_gamma is not None:
        gam = np.asarray(centrifugal_gamma(np.clip(psi_n, 0.0, 1.0)))
        # clip the exponent: a wild measured profile must never overflow the
        # solve (|γ(R²−R₀²)| ~ M₀²·ΔR²/R₀² ≲ 1 on physical MAST inputs)
        cent = np.exp(np.clip(gam * (rr**2 - r0**2), -3.0, 3.0))
    else:
        cent = None
    cols = []
    for family, (drive, n_k) in enumerate(((rr / r0, n_p), (r0 / rr, n_f))):
        for k in range(n_k):
            if kind == "monomial-nonneg":
                phi = edge ** nonneg_exponents[k]
            elif kind == "legendre":
                phi = legendre.legval(x, [0.0] * k + [1.0]) * edge
            else:  # pragma: no cover — callers pass validated kinds
                raise ValueError(f"unknown basis kind {kind!r}")
            col = drive * phi
            if cent is not None and family == 0:  # pressure drive only
                col = col * cent
            cols.append(np.where(inside, col, 0.0))
    return (
        np.column_stack(cols) if cols else np.zeros((psi_n.size, 0), dtype=np.float64)
    )


def _bounded_profile_lsq(
    cols: np.ndarray,
    y: np.ndarray,
    anchor: np.ndarray,
    ip_amperes: float,
    *,
    k_dof: int,
    kp: int,
    s_gram: np.ndarray,
    smoothness_scale: float,
    passive_ridge_scale: float,
) -> np.ndarray | None:
    """Sign-constrained per-sweep coefficient solve: profile block ≥ 0,
    passive block free.

    Ridges enter as factor rows (matrix square root of the smoothness Gram —
    K is tiny, eigh is exact).  The Ip anchor enters as a MODERATELY weighted
    normalised row — weight ~20× the data-residual norm, so the anchor is
    held to ~1% (the caller's per-sweep rescale-to-Ip makes it exact) while
    the data gradients stay ABOVE the solver's optimality tolerance.  Both
    extremes are measured failure modes: weight 10⁴ drowned the misfit (the
    solver stopped at any Ip-consistent point); no anchor at all let the fit
    prefer ~0.2·Ip of net current and the loop-top rescale destroyed the
    optimum.  Returns None on solver failure so the sweep degrades to the
    previous coefficients, never fabricates.
    """
    from scipy import optimize  # noqa: PLC0415

    rows = [cols]
    rhs = [y]
    if smoothness_scale > 0.0 and k_dof:
        w, v = np.linalg.eigh(s_gram)
        factor = (v * np.sqrt(np.clip(w, 0.0, None) * smoothness_scale)).T
        rows.append(np.hstack([factor, np.zeros((k_dof, kp))]))
        rhs.append(np.zeros(k_dof))
    if kp and passive_ridge_scale > 0.0:
        rows.append(
            np.hstack(
                [np.zeros((kp, k_dof)), np.sqrt(passive_ridge_scale) * np.eye(kp)]
            )
        )
        rhs.append(np.zeros(kp))
    w_anchor = 20.0 * max(1.0, float(np.linalg.norm(y)))
    denom = max(abs(ip_amperes), 1e-30)
    rows.append((anchor / denom)[np.newaxis, :] * w_anchor)
    rhs.append(np.array([w_anchor * ip_amperes / denom]))
    a = np.vstack(rows)
    b = np.concatenate(rhs)
    lb = np.concatenate([np.zeros(k_dof), np.full(kp, -np.inf)])
    try:
        res = optimize.lsq_linear(a, b, bounds=(lb, np.full(k_dof + kp, np.inf)))
    except (ValueError, np.linalg.LinAlgError):
        return None
    return res.x if np.isfinite(res.x).all() else None


def _second_difference_gram(n_p: int, n_f: int, weight: float) -> np.ndarray:
    """Block-diagonal Gram matrix of per-family second differences (the
    :func:`coefficient_smoothness_penalty` form, assembled for the LSQ)."""
    k = n_p + n_f
    s = np.zeros((k, k))
    if weight <= 0.0:
        return s
    for lo, n in ((0, n_p), (n_p, n_f)):
        if n < 3:
            continue
        d2 = np.zeros((n - 2, n))
        for i in range(n - 2):
            d2[i, i], d2[i, i + 1], d2[i, i + 2] = 1.0, -2.0, 1.0
        s[lo : lo + n, lo : lo + n] = weight * (d2.T @ d2) / max(n - 2, 1)
    return s


@dataclass
class LadderFit:
    """A per-slice K-DOF ladder fit: coefficients, equilibrium, whitened cost.

    ``coeffs`` are the per-column amplitudes of :func:`profile_basis` under
    the L1 current normalisation used inside the solve (each normalised
    column carries |Ip| of gross current), so they are dimensionless O(1)
    numbers comparable across slices.  ``passive_amplitudes`` are the fitted
    passive currents mapped back to circuit space [A] (None when the sidecar
    is off).
    """

    coeffs: np.ndarray  # (n_p + n_f,)
    n_p: int
    n_f: int
    cost: float
    result: EquilibriumResult
    passive_amplitudes: np.ndarray | None = None

    @property
    def dof(self) -> int:
        return self.n_p + self.n_f


def build_passive_sidecar(
    table: GeometryTable,
    grid: EquilibriumGrid,
    *,
    g_passive: np.ndarray,
    sensor_scale: np.ndarray,
    k: int,
) -> dict:
    """Rank-k passive eigenmode sidecar: sensor + grid-ψ columns per mode.

    ``g_passive`` is the forward operator's passive block RE-MAPPED to the
    grid's sensor-channel order (rows absent on this campaign zeroed).  Modes
    are the top-k right singular vectors of the WHITENED sensor block — the
    passive current patterns the magnetics can actually see — so the sidecar
    spends its k DOF on observable structure, never on null-space fill.  Grid
    ψ columns are built from the same inferred-passive circuits' filaments
    (finite-area Green's functions), so the modes' flux enters the Picard
    field consistently, not just the sensor prediction.  Passive circuits are
    conductor packs already excluded from topology reads.

    Columns are normalised by their whitened singular values so a unit mode
    amplitude produces a unit-norm whitened sensor signal — without this the
    ampere-scale passive columns are ~10⁵ smaller than the Ip-normalised
    profile columns and any relative ridge silently zeroes the sidecar.
    ``modes`` maps fitted amplitudes back to per-circuit currents [A].
    Near-unobservable modes (singular value < 10⁻⁶ of the leading one) are
    dropped rather than amplified.
    """
    g_passive = np.asarray(g_passive, dtype=np.float64)
    scale = np.clip(np.asarray(sensor_scale, dtype=np.float64), 1e-12, None)
    k = int(min(k, g_passive.shape[1]))
    _u, sv, vt = np.linalg.svd(g_passive / scale[:, np.newaxis], full_matrices=False)
    keep = sv[:k] > 1e-6 * max(sv[0], 1e-300)
    k = int(np.count_nonzero(keep))
    v_over_s = vt[:k].T / sv[np.newaxis, :k]  # (n_passive, k), unit whitened norm

    classes = op.classify_circuits(table.pf_filaments, table.amc_current_channels)
    by_circ: dict[int, list] = {}
    for f in table.pf_filaments:
        by_circ.setdefault(f.circuit, []).append(f)
    psi_cols = []
    for cc in classes:
        if cc.role in op._KNOWN_ROLES:
            continue
        acc = np.zeros(grid.flat_r.size)
        for f in by_circ[cc.circuit]:
            psi_f, _br, _bz = hybrid_greens(
                grid.flat_r,
                grid.flat_z,
                float(f.r),
                float(f.z),
                max(abs(f.width), 0.01),
                max(abs(f.height), 0.01),
            )
            acc += f.xmult * psi_f
        psi_cols.append(acc)
    psi_full = (
        np.column_stack(psi_cols) if psi_cols else np.zeros((grid.flat_r.size, 0))
    )
    if psi_full.shape[1] != g_passive.shape[1]:
        raise ValueError(
            f"passive circuit count mismatch: table {psi_full.shape[1]} vs "
            f"g_passive {g_passive.shape[1]}"
        )
    return {
        "g_cols": g_passive @ v_over_s,
        "psi_cols": psi_full @ v_over_s,
        "k": k,
        "modes": v_over_s,
    }


def solve_equilibrium_lsq(
    grid: EquilibriumGrid,
    table: GeometryTable,
    i_pf: np.ndarray,
    ip_amperes: float,
    *,
    measured: np.ndarray,
    vacuum_prediction: np.ndarray,
    sensor_scale: np.ndarray,
    sensor_mask: np.ndarray,
    n_p: int = 1,
    n_f: int = 1,
    smoothness: float = 0.0,
    nonneg: bool = False,
    centrifugal_gamma=None,
    profile_relax: float = 1.0,
    passive: dict | None = None,
    passive_ridge: float = 1.0,
    max_iterations: int = 120,
    warmup_iterations: int = 8,
    relax: float = 0.5,
    tolerance: float = 3e-4,
    seed_z0: float = 0.0,
    seed_width: tuple[float, float] = (0.35, 0.5),
    initial_jphi: np.ndarray | None = None,
) -> LadderFit:
    """Free-boundary Picard solve with the profile coefficients re-fit by
    whitened linear least squares against the raw magnetics every sweep.

    Given ψ_N(R, Z) and the axis-connected core mask of the current iterate,
    the sensor prediction is LINEAR in the K = n_p + n_f basis coefficients,
    so each sweep solves the equality-constrained problem

        min_c ‖(G·U·c + vac − meas)/σ‖²  +  c·S·c     s.t.  1·U·c = Ip

    (S the second-difference smoothness Gram, the Ip anchor a hard KKT
    constraint — the Rogowski measurement, not a penalty) and the resulting
    jφ feeds the next Δ* solve.  The first ``warmup_iterations`` sweeps run
    the fixed default two-term shape rescaled to Ip so the core mask exists
    before the first LSQ; a degenerate or non-finite LSQ solution keeps the
    previous sweep's coefficients (the solve degrades to fixed-shape Picard,
    never fabricates).  Analytic-add coil field throughout; seed/warm-start
    semantics match :func:`solve_equilibrium`.

    ``passive`` (from :func:`build_passive_sidecar`) augments the LSQ with
    rank-k passive eigenmode amplitude columns — ridge-limited
    (``passive_ridge``, absolute on the unit-whitened-norm mode columns:
    1.0 halves a mode carrying a unit-norm signal), EXCLUDED from the plasma
    Ip anchor (vessel currents are not plasma current), and their flux added
    to the Picard field alongside the coil term.  ``passive=None`` (default)
    is byte-identical to the sidecar-free solve.
    """
    g_sens, _channels = grid.sensor_greens(table)
    meas = np.asarray(measured, dtype=np.float64)
    vac = np.asarray(vacuum_prediction, dtype=np.float64)
    scale = np.clip(np.asarray(sensor_scale, dtype=np.float64), 1e-12, None)
    mask = np.asarray(sensor_mask, dtype=bool) & np.isfinite(meas)
    y = (meas[mask] - vac[mask]) / scale[mask]
    w_inv = scale[mask]

    psi_coil = grid.coil_psi(np.asarray(i_pf, dtype=np.float64))
    sign = 1.0 if ip_amperes >= 0 else -1.0
    cell_area = grid.dr * grid.dz
    k_dof = n_p + n_f
    s_gram = _second_difference_gram(n_p, n_f, 1.0)

    kp = int(passive["k"]) if passive else 0
    a_pass = np.zeros(kp)
    if kp:
        g_pass_cols = np.asarray(passive["g_cols"], dtype=np.float64)
        psi_pass = np.asarray(passive["psi_cols"], dtype=np.float64)
        bp = g_pass_cols[mask, :] / w_inv[:, np.newaxis]
    else:
        g_pass_cols = np.zeros((meas.size, 0))
        psi_pass = np.zeros((grid.flat_r.size, 0))
        bp = np.zeros((int(mask.sum()), 0))

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
                + ((grid.flat_z[grid.cells] - seed_z0) / seed_width[1]) ** 2
            )
        )

    psi_flat: np.ndarray | None = None
    residual = np.inf
    axis = (grid.r0, 0.0)
    axis_psi = 0.0
    boundary_psi = 0.0
    core = grid.inside_limiter.copy()
    coeffs = np.zeros(k_dof)
    cost = np.inf

    for iteration in range(1, max_iterations + 1):
        i_cell = jphi[grid.cells] * cell_area
        total = i_cell.sum()
        scale_ip = ip_amperes / total if abs(total) > 1e-12 else 0.0
        i_cell = i_cell * scale_ip

        rhs2d = (-(2.0 * np.pi * MU0) * grid.flat_r * jphi * scale_ip).reshape(
            grid.nz, grid.nr
        )
        psi_b2d = np.zeros((grid.nz, grid.nr))
        psi_b2d.ravel()[grid.edge_idx] = grid.g_edge @ i_cell
        psi_new = grid.solve_dirichlet(rhs2d, psi_b2d).ravel() + psi_coil
        if kp:
            psi_new = psi_new + psi_pass @ a_pass

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

        if iteration <= warmup_iterations:
            jphi = np.zeros_like(jphi)
            shape = profile_jphi_shape(
                psi_n, grid.flat_r, r0=grid.r0, beta0=0.5, alpha=1.0
            )
            jphi[core.ravel()] = shape[core.ravel()]
        else:
            # basis images on the current core; L1-normalise each column to
            # carry |Ip| of gross current so the coefficients stay O(1).  In
            # nonneg mode normalisation is SIGNED so c ≥ 0 ⇒ jφ·sign(Ip) ≥ 0
            # (the R1 fact at the profile level) and the Ip anchor stays
            # reachable for either current direction.
            images = profile_basis(
                psi_n,
                grid.flat_r,
                r0=grid.r0,
                n_p=n_p,
                n_f=n_f,
                kind="monomial-nonneg" if nonneg else "legendre",
                centrifugal_gamma=centrifugal_gamma,
            )
            images[~core.ravel(), :] = 0.0
            u = images[grid.cells, :] * cell_area  # unit-coeff cell currents [A]
            norms = np.abs(u).sum(axis=0)
            ok_cols = norms > 1e-12 * max(abs(ip_amperes), 1.0)
            norm_scale = ip_amperes if nonneg else abs(ip_amperes)
            u_n = np.zeros_like(u)
            u_n[:, ok_cols] = u[:, ok_cols] * (norm_scale / norms[np.newaxis, ok_cols])
            a_anchor = u_n.sum(axis=0)  # net current per unit coefficient
            b_mat = (g_sens[mask, :] @ u_n) / w_inv[:, np.newaxis]
            n_data = int(mask.sum())
            n_var = k_dof + kp
            new_coeffs = None
            new_a_pass = None
            if n_data >= n_var and ok_cols.any():
                cols = np.hstack([b_mat, bp]) if kp else b_mat
                # ridges are RELATIVE: the unit-weight Grams are rescaled by
                # the data Gram's mean diagonal so --smoothness and
                # --passive-ridge are dimensionless O(0.01–1) knobs
                h_data = cols.T @ cols
                mean_diag = np.trace(h_data) / max(n_var, 1)
                anchor = np.concatenate([a_anchor, np.zeros(kp)])
                if nonneg:
                    sol = _bounded_profile_lsq(
                        cols,
                        y,
                        anchor,
                        ip_amperes,
                        k_dof=k_dof,
                        kp=kp,
                        s_gram=s_gram,
                        smoothness_scale=smoothness * mean_diag,
                        # passive mode columns are unit-whitened-norm by
                        # construction — the ridge is absolute on them
                        passive_ridge_scale=passive_ridge,
                    )
                else:
                    s_full = np.zeros((n_var, n_var))
                    s_full[:k_dof, :k_dof] = smoothness * mean_diag * s_gram
                    if kp:
                        # unit-whitened-norm mode columns: absolute ridge
                        s_full[k_dof:, k_dof:] = passive_ridge * np.eye(kp)
                    h = 2.0 * (h_data + s_full)
                    h += 1e-10 * np.trace(h) / max(n_var, 1) * np.eye(n_var)
                    kkt = np.zeros((n_var + 1, n_var + 1))
                    kkt[:n_var, :n_var] = h
                    kkt[:n_var, n_var] = anchor
                    kkt[n_var, :n_var] = anchor
                    rhs = np.concatenate([2.0 * (cols.T @ y), [ip_amperes]])
                    try:
                        sol = np.linalg.solve(kkt, rhs)[:n_var]
                    except np.linalg.LinAlgError:
                        sol = None
                if sol is not None and np.isfinite(sol[:n_var]).all():
                    new_coeffs = sol[:k_dof]
                    new_a_pass = sol[k_dof:n_var]
            if new_coeffs is not None:
                coeffs = new_coeffs
                if kp:
                    a_pass = new_a_pass
                jphi_cells = (u_n / cell_area) @ coeffs  # back to density [A/m²]
                jphi_lsq = np.zeros_like(jphi)
                jphi_lsq[grid.cells] = jphi_cells
                # under-relax the PROFILE update (independent of the ψ
                # relaxation): a full jump from the warmup shape to the LSQ
                # optimum moves the boundary read so far in one sweep that
                # the iteration escapes into the outboard corner attractor
                # (measured on held-out slices) — slow profile morphing keeps
                # the fixed point on the confined branch
                jphi = (
                    jphi_lsq
                    if profile_relax >= 1.0
                    else profile_relax * jphi_lsq + (1.0 - profile_relax) * jphi
                )
            # else: keep the previous sweep's jphi (fixed-shape degradation)

        if iteration > max(5, warmup_iterations + 2) and residual < tolerance:
            break

    i_cell = jphi[grid.cells] * cell_area
    total = i_cell.sum()
    scale_ip = ip_amperes / total if abs(total) > 1e-12 else 0.0
    i_cell = i_cell * scale_ip
    jphi_final = (jphi * scale_ip).reshape(grid.nz, grid.nr)
    # cost is always the FINAL equilibrium's whitened prediction residual (the
    # per-sweep LSQ cost is pre-relaxation and can be stale by one sweep)
    pred = vac + g_sens @ i_cell
    if kp:
        pred = pred + g_pass_cols @ a_pass
    resid_v = (pred[mask] - meas[mask]) / scale[mask]
    cost = float(np.sqrt(np.mean(resid_v * resid_v))) if mask.any() else np.inf

    result = EquilibriumResult(
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
    passive_currents = None
    if kp:
        modes = passive.get("modes")
        # report per-circuit currents [A] when the mode map is available
        passive_currents = np.asarray(modes) @ a_pass if modes is not None else a_pass
    return LadderFit(
        coeffs=coeffs,
        n_p=n_p,
        n_f=n_f,
        cost=cost,
        result=result,
        passive_amplitudes=passive_currents,
    )


def fit_profile_ladder(
    grid: EquilibriumGrid,
    table: GeometryTable,
    *,
    i_pf: np.ndarray,
    ip_amperes: float,
    measured: np.ndarray,
    vacuum_prediction: np.ndarray,
    sensor_scale: np.ndarray,
    sensor_mask: np.ndarray,
    n_p: int = 1,
    n_f: int = 1,
    smoothness: float = 0.0,
    bootstrap_iterations: int = 30,
    initial_jphi: np.ndarray | None = None,
    **solve_kwargs,
) -> LadderFit:  # centrifugal_gamma & friends flow through **solve_kwargs
    """Bootstrapped ladder solve: fixed-shape boundary-continuation stage 1
    (unless ``initial_jphi`` warm-starts it away), then the analytic-add
    LSQ-per-sweep Picard of :func:`solve_equilibrium_lsq`.
    """
    if initial_jphi is None:
        stage1 = solve_equilibrium(
            grid,
            i_pf,
            ip_amperes,
            beta0=0.5,
            alpha=1.0,
            max_iterations=bootstrap_iterations,
            coil_field_mode="boundary-continuation",
            seed_z0=solve_kwargs.get("seed_z0", 0.0),
        )
        initial_jphi = stage1.jphi.ravel()
        if solve_kwargs.get("nonneg", False):
            # the sign-constrained solve is basin-fragile: from a fixed-shape
            # seed the Picard escapes to the outboard corner attractor, while
            # an equally-good PHYSICAL fixed point exists on the confined
            # branch (measured: free-sign 0.428 vs seeded nonneg 0.434 on the
            # same slice).  The stable free-sign K=2 LSQ scouts the branch;
            # the sign-constrained solve then certifies a physical profile
            # there — Tier-3 instrument guides, Tier-2 carries.
            scout = solve_equilibrium_lsq(
                grid,
                table,
                i_pf,
                ip_amperes,
                measured=measured,
                vacuum_prediction=vacuum_prediction,
                sensor_scale=sensor_scale,
                sensor_mask=sensor_mask,
                n_p=1,
                n_f=1,
                initial_jphi=initial_jphi,
            )
            initial_jphi = scout.result.jphi.ravel()
    return solve_equilibrium_lsq(
        grid,
        table,
        i_pf,
        ip_amperes,
        measured=measured,
        vacuum_prediction=vacuum_prediction,
        sensor_scale=sensor_scale,
        sensor_mask=sensor_mask,
        n_p=n_p,
        n_f=n_f,
        smoothness=smoothness,
        initial_jphi=initial_jphi,
        **solve_kwargs,
    )


__all__ = [
    "MU0",
    "profile_jphi_shape",
    "profile_basis",
    "EquilibriumGrid",
    "EquilibriumResult",
    "ProfileFit",
    "LadderFit",
    "build_passive_sidecar",
    "solve_equilibrium",
    "solve_equilibrium_bootstrapped",
    "solve_equilibrium_lsq",
    "fit_profile",
    "fit_profile_continuous",
    "fit_profile_ladder",
]

# not part of the stable public API but imported directly by the gate-eval
# scripts and tests: _read_axis, _read_boundary_psi, _read_boundary_psi_robust
