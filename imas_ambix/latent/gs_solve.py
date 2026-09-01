"""Free-boundary Grad-Shafranov equilibrium solver — the force-balanced ψ decoder.

Why this exists (the representation finding): a LINEAR low-DOF current basis
spread over the vessel cannot produce an interior magnetic axis — the coil
vertical field dominates a broad, low-peakedness current, so the total ψ has no
closed flux region (demonstrated on real MAST data).  Localisation is
inherently nonlinear: the current lives inside the LCFS, the LCFS is a level
set of ψ, and ψ depends on the current — a fixed point.  That fixed point IS
the Grad-Shafranov equilibrium, so the ψ used for topology readouts must come
from an actual GS solve.  The implementation uses a coarse psi grid, a
five-point Grad-Shafranov stencil, and free-boundary conditions from measured
coil currents.

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

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy import ndimage  # type: ignore[import-untyped]

from imas_ambix.cocos import project_poloidal_field
from imas_ambix.gs import operator as op
from imas_ambix.gs.cylinder import hybrid_greens
from imas_ambix.latent.topology import (
    CriticalPoints,
    _inside_polygon,
    boundary_flux_robust,
    find_critical_points,
)
from imas_ambix.latent.wall_mask import (
    WallUnit,
    build_wall_mask,
    densify_units,
    vessel_unit,
)

logger = logging.getLogger("imas_ambix.gs_solve")

if TYPE_CHECKING:
    from imas_ambix.gs.machine_geometry import OperatorGeometry


def _geometry_member(geometry, projection_name: str, compatibility_name: str):
    """Read a projection field while synthetic table fixtures are retired."""
    if hasattr(geometry, projection_name):
        return getattr(geometry, projection_name)
    return getattr(geometry, compatibility_name)


def _representation_key(geometry) -> str | None:
    identity = getattr(geometry, "identity", None)
    if identity is not None:
        return str(identity.representation_key)
    signature = getattr(geometry, "signature", None)
    return None if signature is None else str(signature.key)


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
        wall_units: list[WallUnit] | None = None,  # arbitrary multi-unit wall
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
        # The wall enters ONLY as a raster boolean mask + a string of surface
        # nodes.  A single closed vessel loop (the ``limiter_r/limiter_z``
        # default — MAST) is byte-identical to the plain point-in-polygon read;
        # an explicit ``wall_units`` list (AUG discrete tiles, a movable WEST
        # wall) is DATA through the same code path (tiles-as-holes supercover).
        self.wall_units = wall_units
        if wall_units is not None:
            self.inside_limiter, self.wall_diagnostics = build_wall_mask(
                self.rg, self.zg, wall_units
            )
            units_for_nodes = wall_units
        else:
            self.inside_limiter = _inside_polygon(
                self.flat_r, self.flat_z, self.limiter_r, self.limiter_z
            ).reshape(self.nz, self.nr)
            self.wall_diagnostics = []
            units_for_nodes = (
                [vessel_unit(self.limiter_r, self.limiter_z)]
                if self.limiter_r.size >= 2
                else []
            )
        # wall surface nodes at ~Δ/2 arc spacing (the sub-grid tangency string,
        # tagged by unit); the exact node flux is the campaign ``g_wall`` GEMM.
        self.wall_r, self.wall_z, self.wall_unit_id = densify_units(
            units_for_nodes, 0.5 * min(self.dr, self.dz)
        )
        #: per-merged-circuit filament packs (r, z, w, h, weight) for the coil
        #: block of ``g_wall``; populated by ``from_table`` (the point where the
        #: filaments are available), None on a bare grid.
        self._coil_packs: list[np.ndarray] | None = None
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
        # The gridded Δ* operator is LU-factorised lazily, on the first
        # ``solve_dirichlet`` call.  A solve that evaluates ψ purely by the
        # analytic Green's matvec (``plasma_grid_psi``) never touches it, so
        # ``self._lu is None`` after such a solve is a machine-checkable proof
        # that no gridded elliptic operator was assembled or inverted.
        self._lu = None
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

    #: Campaign-scope build cache: the grid geometry, Δ* factorisation, and
    #: every Green's / interaction matrix are pure functions of the campaign
    #: (limiter + coil geometry + sensor map) and the resolution, so they are
    #: built ONCE per (campaign signature, nr, nz) and reused across every shot
    #: and slice of that campaign.  Opt-in (``cache=True``) so the
    #: default keeps building an independent grid per call (the gate's
    #: independent-arm invariant that proves ``_lu is None`` on the grid-free
    #: arm relies on distinct instances).
    _build_cache: dict = {}

    @classmethod
    def clear_grid_cache(cls) -> None:
        """Drop every cached campaign grid (free the interaction matrices)."""
        cls._build_cache.clear()

    @classmethod
    def from_table(
        cls, table: OperatorGeometry, *, nr: int = 65, nz: int = 97, cache: bool = False
    ) -> EquilibriumGrid:
        """Build the campaign grid and its interaction matrices from a table.

        The build cache is keyed by the setup signature and the resolution, and
        deliberately NOT by the machine's physical identity: every matrix it
        holds is a function of the discretization (filament count and positions,
        limiter vertices, sensor map), so two representations of the same machine
        need two different cached grids. Physical identity answers which machine
        a table describes -- see :mod:`imas_ambix.gs.machine_identity` -- and is
        never a compute-cache address.
        """
        key = None
        if cache:
            sig = getattr(getattr(table, "signature", None), "key", None)
            if sig is not None:
                key = (sig, int(nr), int(nz))
                hit = cls._build_cache.get(key)
                if hit is not None:
                    return hit
        lr = np.asarray(table.limiter_r, dtype=np.float64)
        lz = np.asarray(table.limiter_z, dtype=np.float64)
        rg = np.linspace(max(float(lr.min()), 0.06), float(lr.max()), nr)
        zg = np.linspace(float(lz.min()), float(lz.max()), nz)
        mesh_r, mesh_z = np.meshgrid(rg, zg)
        flat_r, flat_z = mesh_r.ravel(), mesh_z.ravel()

        fwd = op.build_operator(table)
        by_circ: dict[int, list] = {}
        for f in _geometry_member(table, "conductors", "pf_filaments"):
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
                for f in _geometry_member(table, "conductors", "pf_filaments")
            ]
        )
        grid = cls(
            rg=rg,
            zg=zg,
            limiter_r=lr,
            limiter_z=lz,
            coil_psi_columns=coil_cols,
            r0=fwd.r0,
            conductor_rects=rects,
        )
        # per-merged-circuit filament packs (r, z, clamped-w, clamped-h, weight)
        # so ``g_wall``'s coil block reproduces ``coil_psi`` at arbitrary wall
        # nodes: a merged column is the mean over its circuits of Σ xmult·greens,
        # i.e. Σ over (circuit, filament) of (xmult/n_circ)·greens.
        packs: list[np.ndarray] = []
        for circs in fwd.pf_merged_circuits:
            n_circ = max(len(circs), 1)
            rows = [
                (
                    float(f.r),
                    float(f.z),
                    max(abs(f.width), 0.01),
                    max(abs(f.height), 0.01),
                    float(f.xmult) / n_circ,
                )
                for c in circs
                for f in by_circ.get(c, [])
            ]
            packs.append(np.array(rows, dtype=np.float64) if rows else np.zeros((0, 5)))
        grid._coil_packs = packs
        if key is not None:
            cls._build_cache[key] = grid
        return grid

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
        Returns ψ as (nz, nr).  The Δ* LU is built here on first use.
        """
        if self._lu is None:
            self._lu = self._factorise()
        rhs = np.where(self.interior, rhs_interior, 0.0).ravel().astype(np.float64)
        rhs[self.edge_idx] = psi_boundary.ravel()[self.edge_idx]
        return self._lu.solve(rhs).reshape(self.nz, self.nr)

    def plasma_grid_psi_columns(self) -> np.ndarray:
        """Analytic plasma→grid ψ Green's matrix ``(n_grid, n_cell)`` [Wb per A].

        Column ``j`` is the total poloidal flux ψ at every grid point produced
        by unit current in in-limiter cell ``j`` (``self.cells`` order), from
        the finite-area :func:`hybrid_greens` kernel — the SAME kernel the
        coils, vessel, and the ``g_edge`` Dirichlet block already use, so this
        is the analytic Δ* inversion (the free-space Green's function of Δ*),
        not a second discretisation.  Matmul against a cell-current vector then
        gives ψ everywhere WITHOUT solving the gridded elliptic operator — the
        grid-free substrate.  Cached (pure geometry); the ``g_edge`` edge block
        is reproduced exactly here on the edge rows, so the grid-free field
        matches the grid solve's Dirichlet data on the boundary by construction
        and differs only in the interior (analytic vs 5-point-FD Green's).
        """
        cached = getattr(self, "_plasma_grid_psi_cache", None)
        if cached is not None:
            return cached
        cols = [
            hybrid_greens(
                self.flat_r,
                self.flat_z,
                float(self.flat_r[c]),
                float(self.flat_z[c]),
                self.dr,
                self.dz,
            )[0]
            for c in self.cells
        ]
        g = (
            np.column_stack(cols)
            if cols
            else np.zeros((self.flat_r.size, 0), dtype=np.float64)
        )
        self._plasma_grid_psi_cache = g
        return g

    def plasma_grid_psi(self, cell_currents: np.ndarray) -> np.ndarray:
        """Plasma-generated ψ on the flattened grid [Wb] from per-cell currents.

        ``cell_currents`` [A] are one value per in-limiter cell (``self.cells``
        order).  Returns ``G_cell→grid @ cell_currents`` — the analytic Δ*
        inversion evaluated at every grid point by matvec (no ``_lu`` solve).
        """
        g = self.plasma_grid_psi_columns()
        if g.shape[1] == 0:
            return np.zeros(self.flat_r.size, dtype=np.float64)
        return np.asarray(g @ np.asarray(cell_currents, dtype=np.float64))

    def coil_psi(self, i_pf: np.ndarray) -> np.ndarray:
        """Vacuum coil ψ on the flattened grid [Wb] for coil currents [A]."""
        if self._coil_psi_columns.shape[1] == 0:
            return np.zeros(self.flat_r.size)
        return self._coil_psi_columns @ np.asarray(i_pf, dtype=np.float64)

    def sensor_greens(self, table: OperatorGeometry) -> tuple[np.ndarray, list[str]]:
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
            if m.kind != "flux_loop" and m.angle_deg is None:
                raise ValueError(
                    f"poloidal probe {m.amb_channel!r} has no directed angle"
                )
            angle_deg = 0.0 if m.angle_deg is None else m.angle_deg
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
                    row[k] = project_poloidal_field(br_k[0], bz_k[0], angle_deg)
            rows.append(row)
            channels.append(m.amb_channel)
        g = np.vstack(rows) if rows else np.zeros((0, cr.size))
        self._sensor_greens_cache = (g, channels)
        return g, channels

    def cell_greens(self) -> dict:
        """Analytic thick-cylinder Green's matrices, in-limiter cell → in-limiter
        grid point, for the annulus soft-prior penalty.

        Returns ``{"cells", "psi", "br", "bz"}`` where ``psi``/``br``/``bz`` are
        ``(n_cells, n_cells)`` per-ampere total-flux ψ [Wb] and poloidal field
        [T] evaluated at every in-limiter grid point (rows, ``== self.cells``
        order) from a unit current in each in-limiter cell (columns, same order).
        ALL THREE come straight from the finite-area cylinder Biot–Savart kernel
        (the double cross-section integral) — ψ AND its field are analytic, so the
        grad-ψ annulus penalty needs NO finite differences and the matrices match
        the annulus-consistency metric's own analytic carrier psi.  The default
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
        cached = getattr(self, "_cell_greens_cache", None)
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
        self._cell_greens_cache = out
        return out

    def wall_greens(self) -> dict:
        """Wall-node ψ influence matrices ``g_wall`` — the campaign wall-flux GEMM.

        Returns ``{nodes_r, nodes_z, unit_id, g_coils, g_cells}`` where
        ``g_coils`` is ``(n_node, n_coil)`` and ``g_cells`` is
        ``(n_node, n_cell)`` per-ampere total-flux ψ [Wb] at every wall surface
        node (``self.wall_r/wall_z`` order) from a unit coil or in-limiter cell
        current, from the SAME finite-area :func:`hybrid_greens` kernel the coils,
        ``g_edge`` and the plasma-grid block use — but at the wall nodes, which
        are NEAR-field to the edge plasma cells, so the point-filament far-field
        shortcut that suffices for ``g_edge`` does not.  ``wall_flux`` then reads
        the tangency EXACTLY at the node (no O(Δ²) bilerp floor, largest exactly
        at a tangency where the plasma leans), linearly hence differentiably, in
        one matmul per solve.  Cached (pure campaign geometry); slots beside
        ``g_edge`` in the build cache.
        """
        cached = getattr(self, "_wall_greens_cache", None)
        if cached is not None:
            return cached
        wr = np.asarray(self.wall_r, dtype=np.float64)
        wz = np.asarray(self.wall_z, dtype=np.float64)
        # in-limiter cell → wall node (mirror of plasma_grid_psi_columns at nodes)
        cell_cols = [
            hybrid_greens(
                wr, wz, float(self.flat_r[c]), float(self.flat_z[c]), self.dr, self.dz
            )[0]
            for c in self.cells
        ]
        g_cells = (
            np.column_stack(cell_cols)
            if cell_cols
            else np.zeros((wr.size, 0), dtype=np.float64)
        )
        # coil → wall node, from the stored merged-circuit filament packs
        packs = self._coil_packs
        if packs:
            coil_cols = []
            for pack in packs:
                acc = np.zeros(wr.size, dtype=np.float64)
                for fr, fz, fw, fh, wt in pack:
                    acc += wt * hybrid_greens(wr, wz, float(fr), float(fz), fw, fh)[0]
                coil_cols.append(acc)
            g_coils = np.column_stack(coil_cols)
        else:
            g_coils = np.zeros((wr.size, 0), dtype=np.float64)
        out = {
            "nodes_r": wr,
            "nodes_z": wz,
            "unit_id": np.asarray(self.wall_unit_id),
            "g_coils": g_coils,
            "g_cells": g_cells,
        }
        self._wall_greens_cache = out
        return out

    def wall_flux(self, i_pf: np.ndarray, i_cell: np.ndarray) -> np.ndarray:
        """Exact total ψ [Wb] at every wall node for coil + cell currents (one GEMM).

        ``g_coils @ i_pf + g_cells @ i_cell`` on the same absolute scale as the
        gridded ψ (both are the finite-area Green's superposition of the same
        currents), so the connectivity read normalises it consistently against
        ``psi_axis``.  This is the sub-grid tangency source that replaces the
        bilinear read off the grid.
        """
        g = self.wall_greens()
        psi = np.zeros(g["nodes_r"].size, dtype=np.float64)
        if g["g_cells"].shape[1]:
            psi = psi + g["g_cells"] @ np.asarray(i_cell, dtype=np.float64)
        if g["g_coils"].shape[1]:
            psi = psi + g["g_coils"] @ np.asarray(i_pf, dtype=np.float64)
        return psi


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


#: The two ψ-evaluation substrates.  ``grid-delstar`` inverts the gridded
#: 5-point Δ* operator (the historical solve, byte-unchanged); ``greens-matvec``
#: evaluates ψ by the analytic finite-area Green's matvec and never assembles or
#: inverts the elliptic operator.
SUBSTRATE_GRID = "grid-delstar"
SUBSTRATE_GREENS = "greens-matvec"
SUBSTRATES = (SUBSTRATE_GRID, SUBSTRATE_GREENS)

#: The two per-sweep topology reads.  ``hard`` is the historical
#: hard-threshold host read (critical points + labelled core mask,
#: byte-unchanged); ``connectivity`` is the temperature-smoothed kernel —
#: softmin boundary binding + retracted-gate sigmoid core weight + sub-grid
#: stencil axis — under which the free-boundary fixed-point map is
#: end-to-end differentiable.
TOPOLOGY_HARD = "hard"
TOPOLOGY_CONNECTIVITY = "connectivity"
TOPOLOGY_READS = (TOPOLOGY_HARD, TOPOLOGY_CONNECTIVITY)


def _read_topology_smooth(
    psi_flat: np.ndarray,
    grid: EquilibriumGrid,
    seed_axis: tuple[float, float],
    core_cap: float,
    temperature: float,
) -> tuple[tuple[float, float], float, float, np.ndarray, np.ndarray]:
    """Continuous per-sweep topology read for the smooth-map solve path.

    The temperature-smoothed connectivity kernel
    (:func:`~imas_ambix.latent.connectivity_boundary.boundary_read_smooth`):
    the sub-grid stencil O-point is read first, the smooth boundary read is
    seeded at it (softmin over the wall-tangency / X-saddle binding
    candidates, retracted-gate sigmoid core weight — everything τ-controlled
    and end-to-end differentiable in ψ), and the returned ``core_weight`` IS
    the core membership: half-weight at the binding, width ``temperature``,
    gated by the axis-connected flood so a disconnected private-flux pocket
    never carries weight.  The fixed-point map under this read has no
    discrete mask-flip signature — exact autograd tangents flow through it.

    Returns ``(axis, axis_psi, boundary_psi, weight_flat, core_bool_2d)``;
    ``core_bool_2d`` is the reporting-only hard threshold (it never feeds the
    map).  Degenerate transients (no O-point / no closed level yet) fall back
    to the flood seed and the limiter-contact flux so the solve degrades the
    same way the hard read does instead of dying.
    """
    from imas_ambix.latent.connectivity_boundary import (  # noqa: PLC0415 (jax)
        boundary_read_smooth,
    )

    if core_cap != 1.0:
        raise ValueError(
            "the smooth connectivity read binds the core membership at the "
            f"boundary; a soft SOL cap (core_cap={core_cap!r}) has no "
            "smooth-kernel equivalent — run the SOL prior on the hard read"
        )
    psi2d = psi_flat.reshape(grid.nz, grid.nr)
    rd = boundary_read_smooth(psi2d, grid, seed_axis, temperature=temperature)
    ax_r = float(rd["axis_r"])
    ax_z = float(rd["axis_z"])
    axis = (ax_r, ax_z) if np.isfinite(ax_r) and np.isfinite(ax_z) else seed_axis
    axis_psi = (
        float(rd["axis_psi_sub"])
        if np.isfinite(rd["axis_psi_sub"])
        else float(rd["psi_axis"])
    )
    boundary_psi = float(rd["psi_bnd"])
    if not np.isfinite(boundary_psi):
        boundary_psi = _read_boundary_psi(psi2d, grid, axis_psi)
    weight = np.asarray(rd["core_weight"], dtype=np.float64).ravel()
    core = np.asarray(rd["core_weight"]) > 0.5
    return axis, axis_psi, boundary_psi, weight, core


def _plasma_psi_field(
    grid: EquilibriumGrid,
    jphi_scaled: np.ndarray,
    i_cell: np.ndarray,
    psi_coil: np.ndarray,
    *,
    substrate: str,
) -> np.ndarray:
    """Plasma+coil ψ on the flattened grid for one analytic-add Picard iterate.

    ``jphi_scaled`` is the Ip-rescaled current density on the full grid
    (``jphi * scale``); ``i_cell`` the matching per-cell current [A]
    (``jphi[cells] * cell_area * scale``, ``self.cells`` order); ``psi_coil``
    the vacuum coil ψ.

    ``substrate='grid-delstar'`` (default) is byte-identical to the historical
    analytic-add solve — Δ* inverted on the 5-point grid with the plasma's own
    Green's field as the Dirichlet edge, plus the analytic coil field.
    ``substrate='greens-matvec'`` drops the gridded solve: ψ_plasma is the
    analytic Green's matvec ``grid.plasma_grid_psi(i_cell)``, so no elliptic
    operator is assembled or inverted.
    """
    if substrate == SUBSTRATE_GREENS:
        return grid.plasma_grid_psi(i_cell) + psi_coil
    rhs2d = (-(2.0 * np.pi * MU0) * grid.flat_r * jphi_scaled).reshape(grid.nz, grid.nr)
    psi_b2d = np.zeros((grid.nz, grid.nr))
    psi_b2d.ravel()[grid.edge_idx] = grid.g_edge @ i_cell
    return grid.solve_dirichlet(rhs2d, psi_b2d).ravel() + psi_coil


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
    substrate: str = SUBSTRATE_GRID,
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

    ``substrate`` selects the ψ evaluator (:data:`SUBSTRATES`): the default
    ``grid-delstar`` is byte-unchanged; ``greens-matvec`` drops the gridded
    Δ* solve for the analytic Green's matvec (opt-in; ``analytic-add`` only —
    the legacy boundary-continuation arm has no grid-free form).
    """
    if substrate not in SUBSTRATES:
        raise ValueError(f"substrate must be one of {SUBSTRATES}, got {substrate!r}")
    if substrate == SUBSTRATE_GREENS and coil_field_mode != "analytic-add":
        raise ValueError(
            "greens-matvec substrate requires coil_field_mode='analytic-add'"
        )
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
        if coil_field_mode == "analytic-add":
            psi_new = _plasma_psi_field(
                grid, jphi * scale, i_cell, psi_coil, substrate=substrate
            )
        else:  # boundary-continuation — the legacy diagnostic arm (grid only)
            rhs2d = (-(2.0 * np.pi * MU0) * grid.flat_r * jphi * scale).reshape(
                grid.nz, grid.nr
            )
            psi_b2d = np.zeros((grid.nz, grid.nr))
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


def solve_equilibrium_nk(
    grid: EquilibriumGrid,
    i_pf: np.ndarray,
    ip_amperes: float,
    *,
    beta0: float = 0.5,
    alpha: float = 1.0,
    substrate: str = SUBSTRATE_GREENS,
    picard_warmup: int = 12,
    relax: float = 0.5,
    f_tol: float = 1e-7,
    maxiter: int = 80,
    seed_width: tuple[float, float] = (0.35, 0.5),
    seed_z0: float = 0.0,
    initial_jphi: np.ndarray | None = None,
    topology_read: str = TOPOLOGY_HARD,
    smooth_temperature: float = 1e-3,
) -> EquilibriumResult:
    """Jacobian-free Newton–Krylov free-boundary solve (nova's scheme).

    The free-boundary equilibrium is the fixed point of

        T(ψ) = ψ_coil + G · I(ψ)

    where ``I(ψ)`` are the force-balanced filament currents — the topology read
    gives ψ_N and the axis-connected core, ``jφ = R·p′(ψ_N) + FF′(ψ_N)/(μ₀R)``
    is applied pointwise, and the currents are renormalised so the net equals
    the measured Ip (filament turns sum to 1).  NK drives the residual
    ``F(ψ) = ψ − T(ψ)`` to zero (``scipy.optimize.newton_krylov``), exactly the
    nova ``residual = psi - self(psi)`` restated on this engine's machinery.
    Defaults to the grid-free ``greens-matvec`` substrate so ``G·I`` is the
    analytic Green's matvec and no Δ* is assembled.  A short Picard warmup seeds
    NK into the confined basin (the broad seed has no interior O-point, so a
    cold NK start would sit at the currentless fixed point).
    """
    from scipy.optimize import NoConvergence, newton_krylov  # noqa: PLC0415

    if substrate not in SUBSTRATES:
        raise ValueError(f"substrate must be one of {SUBSTRATES}, got {substrate!r}")
    if topology_read not in TOPOLOGY_READS:
        raise ValueError(
            f"topology_read must be one of {TOPOLOGY_READS}, got {topology_read!r}"
        )
    psi_coil = grid.coil_psi(np.asarray(i_pf, dtype=np.float64))
    sign = 1.0 if ip_amperes >= 0 else -1.0
    cell_area = grid.dr * grid.dz
    state: dict = {"axis": (grid.r0, seed_z0)}

    def currents_from_psi(psi_flat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Force-balanced (jφ_full·scale, i_cell) from a ψ iterate; caches reads."""
        psi2d = psi_flat.reshape(grid.nz, grid.nr)
        if topology_read == TOPOLOGY_CONNECTIVITY:
            # continuous read — the residual F(ψ) = ψ − T(ψ) is differentiable,
            # so the finite-difference directional derivatives of NK's GMRES
            # see a smooth map instead of discrete mask flips.
            axis, axis_psi, boundary_psi, weight, core = _read_topology_smooth(
                psi_flat, grid, state["axis"], 1.0, smooth_temperature
            )
            span = boundary_psi - axis_psi
            if abs(span) < 1e-12:
                span = 1e-12
            psi_n = (psi_flat - axis_psi) / span
            shape = profile_jphi_shape(
                psi_n, grid.flat_r, r0=grid.r0, beta0=beta0, alpha=alpha
            )
            jphi = shape * weight
        else:
            axis, axis_psi = _read_axis(psi2d, grid, sign)
            boundary_psi = _read_boundary_psi(psi2d, grid, axis_psi)
            span = boundary_psi - axis_psi
            if abs(span) < 1e-12:
                span = 1e-12
            psi_n = (psi_flat - axis_psi) / span
            closed = ((psi_n < 1.0) & grid.inside_limiter.ravel()).reshape(
                grid.nz, grid.nr
            )
            labels, _ = ndimage.label(closed)
            ia = int(np.argmin(np.abs(grid.zg - axis[1])))
            ja = int(np.argmin(np.abs(grid.rg - axis[0])))
            core_label = labels[ia, ja]
            core = (labels == core_label) if core_label != 0 else closed
            jphi = np.zeros(grid.flat_r.size)
            shape = profile_jphi_shape(
                psi_n, grid.flat_r, r0=grid.r0, beta0=beta0, alpha=alpha
            )
            jphi[core.ravel()] = shape[core.ravel()]
        i_cell = jphi[grid.cells] * cell_area
        total = i_cell.sum()
        scale = ip_amperes / total if abs(total) > 1e-12 else 0.0
        i_cell = i_cell * scale
        state.update(
            axis=axis,
            axis_psi=axis_psi,
            boundary_psi=boundary_psi,
            core=core,
            jphi=(jphi * scale).reshape(grid.nz, grid.nr),
        )
        return jphi * scale, i_cell

    def apply_map(psi_flat: np.ndarray) -> np.ndarray:
        jphi_scaled, i_cell = currents_from_psi(psi_flat)
        return _plasma_psi_field(
            grid, jphi_scaled, i_cell, psi_coil, substrate=substrate
        )

    # seed jφ (warm-start or the compact Gaussian) → an initial ψ with an axis
    if initial_jphi is not None:
        seed = np.where(
            grid.inside_limiter.ravel(),
            np.asarray(initial_jphi, dtype=np.float64).ravel(),
            0.0,
        )
        if not np.isfinite(seed).all() or abs(seed.sum()) < 1e-12:
            seed = np.zeros(grid.flat_r.size)
    else:
        seed = np.zeros(grid.flat_r.size)
    if abs(seed.sum()) < 1e-12:
        seed[grid.cells] = np.exp(
            -(
                ((grid.flat_r[grid.cells] - grid.r0) / seed_width[0]) ** 2
                + ((grid.flat_z[grid.cells] - seed_z0) / seed_width[1]) ** 2
            )
        )
    seed_cells = seed[grid.cells] * cell_area
    seed_total = seed_cells.sum()
    seed_cells = seed_cells * (
        ip_amperes / seed_total if abs(seed_total) > 1e-12 else 0.0
    )
    psi = _plasma_psi_field(grid, seed, seed_cells, psi_coil, substrate=substrate)

    # Picard warmup into the confined basin, then Newton–Krylov to the root
    for _ in range(max(0, picard_warmup)):
        psi = relax * apply_map(psi) + (1.0 - relax) * psi

    def residual_fn(psi_flat: np.ndarray) -> np.ndarray:
        return psi_flat - apply_map(psi_flat)

    converged = True
    iterations = maxiter
    try:
        psi = newton_krylov(residual_fn, psi, f_tol=f_tol, maxiter=maxiter)
    except NoConvergence as exc:
        psi = np.asarray(exc.args[0], dtype=np.float64)
        converged = False

    jphi_scaled, i_cell = currents_from_psi(psi)
    res = residual_fn(psi)
    residual = float(np.abs(res).max() / max(np.abs(psi).max(), 1e-12))
    return EquilibriumResult(
        psi=psi.reshape(grid.nz, grid.nr),
        axis=state["axis"],
        axis_psi=state["axis_psi"],
        boundary_psi=state["boundary_psi"],
        jphi=state["jphi"],
        cell_currents=i_cell,
        core_mask=state["core"],
        converged=bool(converged and residual < 1e-3),
        residual=residual,
        iterations=iterations,
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
    table: OperatorGeometry,
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
    table: OperatorGeometry,
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
    ≥ 0, so NON-NEGATIVE coefficients imply jφ ≥ 0 pointwise (the
    unidirectional-current invariant imposed at the profile level).  The ladder
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
    extra_rows: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray | None:
    """Sign-constrained per-sweep coefficient solve: profile block ≥ 0,
    passive block free.

    ``extra_rows`` (A, b), when given, are additional whitened soft-prior rows
    on ``x = [coeffs, a_pass]`` (same width as ``cols``) stacked into the
    bounded least-squares — the nonneg-arm route for the annulus anchor / q
    bound / moment priors (the abs-ψ gauge-offset column is unsupported here).

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
    if extra_rows is not None:
        a_extra, b_extra = extra_rows
        if a_extra.shape[0]:
            rows.append(np.asarray(a_extra, dtype=np.float64))
            rhs.append(np.asarray(b_extra, dtype=np.float64))
    a = np.vstack(rows)
    b = np.concatenate(rhs)
    lb = np.concatenate([np.zeros(k_dof), np.full(kp, -np.inf)])
    try:
        res = optimize.lsq_linear(a, b, bounds=(lb, np.full(k_dof + kp, np.inf)))
    except ValueError, np.linalg.LinAlgError:
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


def s_full_block(
    n_var: int,
    k_dof: int,
    kp: int,
    smoothness_scale: float,
    s_gram: np.ndarray,
    passive_ridge: float,
) -> np.ndarray:
    """The (n_var, n_var) smoothness + passive-ridge regulariser block.

    Coefficient block gets the (relatively-scaled) second-difference Gram; the
    unit-whitened-norm passive modes get an absolute ridge.  Factored out so the
    soft-prior-augmented normal equations reuse the exact same regulariser."""
    s = np.zeros((n_var, n_var))
    s[:k_dof, :k_dof] = smoothness_scale * s_gram
    if kp:
        s[k_dof:, k_dof:] = passive_ridge * np.eye(kp)
    return s


@dataclass
class SoftPriors:
    """Optional soft physics priors folded into the per-sweep LSQ.

    Every prior contributes weighted rows on the sweep's variable vector
    ``x = [coeffs (k_dof), a_pass (kp)]`` (the annulus abs-ψ form may add ONE
    trailing free gauge-offset column).  All knobs default OFF, so a bare
    ``SoftPriors()`` (or ``None``) leaves the solve byte-identical.  Weights and
    caps are dimensionless / geometry-scaled — machine-agnostic.

    Annulus anchor (the boundary read as a soft prior):
    ``anchor_form`` selects the gauge-keeping form (``"abs-psi"`` with a rank-1
    offset, or ``"grad-psi"`` field-matching); ``anchor_ann_rows`` are the
    annulus points as positions into ``grid.cells``; the harmonic target at
    those points is ``anchor_psi_target`` (abs-ψ) or ``anchor_grad_target``
    (grad-ψ, R then Z stacked).  Robust clip + per-slice uncertainty match the
    read's heavy-tailed consistency.

    Soft SOL edge: ``sol_cap`` > 1 admits current through a C¹ decay foot
    to ψ_N ≈ ``sol_cap`` (the axis-connected common-flux region only); the
    profile basis swaps to :func:`profile_regularization.profile_basis_foot`.

    q ≥ 1 (sawtooth): ``q_axis_max`` (from
    :func:`profile_regularization.q_axis_linear_bound`) softly bounds the
    on-axis current density with weight ``q_weight``.

    Moment priors (user extension): ``ip_soft_sigma`` switches the Ip anchor
    from the hard KKT to a soft covariance row; ``beta_li_*`` pins βp+li/2 to a
    magnetics-derived target via a caller-supplied sensitivity; ``pprime_*``
    pulls the p′-family coefficients toward a pressure-derived target (the
    p′/FF′ separation lever).
    """

    anchor_form: str | None = None
    anchor_weight: float = 0.0
    anchor_ann_rows: np.ndarray | None = None
    anchor_psi_target: np.ndarray | None = None
    anchor_grad_target: np.ndarray | None = None
    anchor_gauge_offset: bool = True
    anchor_robust_clip: float | None = 3.0
    anchor_uncertainty: float = 1.0

    sol_cap: float = 1.0
    sol_foot_w: float = 0.05

    q_axis_max: float | None = None
    q_weight: float = 1.0

    ip_soft_sigma: float | None = None

    beta_li_target: float | None = None
    beta_li_sensitivity: np.ndarray | None = None
    beta_li_sigma: float = 0.1

    pprime_target: np.ndarray | None = None
    pprime_basis: np.ndarray | None = None  # (n_samp, n_p) p'-family at samples
    pprime_sigma: float = 1.0

    # current-centroid position constraint (the well-posed position lever):
    # pin the R and/or Z toroidal-current centroid to a MEASURED magnetic
    # moment (firewall-safe, like the Ip anchor) so the free-boundary solve
    # holds the current at the measured position instead of drifting outboard
    # through the radially-unstable in-vessel-coil band.  The moments are
    # linear-homogeneous in the coefficients (∫R jφ = R_c·Ip, ∫Z jφ = Z_c·Ip),
    # assembled from the sweep's u_n exactly like a_anchor.  A None target
    # leaves that coordinate free; both None is byte-identical OFF.
    centroid_r_target: float | None = None
    centroid_z_target: float | None = None
    centroid_sigma_r: float = 0.02  # 1σ [m] on the R centroid (strong tether)
    centroid_sigma_z: float = 0.02  # 1σ [m] on the Z centroid

    # passive trajectory prior (dynamic passive spine): centre the passive
    # eddy amplitudes on a circuit-integrated trajectory instead of zero.
    # ``passive_prior_center`` is in the SIDECAR's whitened mode coordinates
    # (same variables as ``a_pass``); weight 0 / None is byte-identical OFF.
    # The weight may be a PER-MODE vector (kp,) so heterogeneous sidecar
    # blocks (e.g. vessel modes at 0 + plasma screening modes at w) can carry
    # different prior strengths in one solve; a scalar applies uniformly.
    passive_prior_center: np.ndarray | None = None
    passive_prior_weight: float | np.ndarray = 0.0

    # profile-coefficient temporal-consistency prior (current diffusion):
    # centre the ladder coefficients on the diffusion-evolved prediction from
    # the previous slice.  ``coeff_prior_center`` is in the solve's normalised
    # coefficient convention (O(1), same variables as ``coeffs``); weight 0 /
    # None is byte-identical OFF.
    coeff_prior_center: np.ndarray | None = None
    coeff_prior_weight: float = 0.0

    @property
    def sol_active(self) -> bool:
        return self.sol_cap > 1.0

    @property
    def any_penalty(self) -> bool:
        return (
            (self.anchor_form is not None and self.anchor_weight > 0.0)
            or self.q_axis_max is not None
            or self.ip_soft_sigma is not None
            or self.beta_li_target is not None
            or self.pprime_target is not None
            or self.centroid_r_target is not None
            or self.centroid_z_target is not None
            or (
                self.passive_prior_center is not None
                and bool(np.any(np.asarray(self.passive_prior_weight) > 0.0))
            )
            or (self.coeff_prior_center is not None and self.coeff_prior_weight > 0.0)
        )


def _assemble_soft_prior_rows(
    sp: SoftPriors,
    *,
    grid: EquilibriumGrid,
    u_n: np.ndarray,
    a_anchor: np.ndarray,
    axis_images_unit: np.ndarray,
    coeffs_prev: np.ndarray,
    psi_coil: np.ndarray,
    psi_pass: np.ndarray,
    a_pass: np.ndarray,
    ip_amperes: float,
    k_dof: int,
    kp: int,
    passive_br: np.ndarray | None = None,
    passive_bz: np.ndarray | None = None,
    data_gram_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Build the stacked soft-prior design rows for one Picard sweep.

    Returns ``(a_extra, b_extra, n_gauge)`` where ``a_extra`` has
    ``k_dof + kp + n_gauge`` columns (``n_gauge`` = 1 iff the annulus abs-ψ
    offset DOF is active) and the whitened residual rows drive
    ``a_extra @ x ≈ b_extra``.  Each prior is delegated to its own module
    (:mod:`boundary_prior`, :mod:`profile_regularization`, :mod:`moment_priors`)
    so the physics lives beside its tests; this only maps the sweep's linear
    quantities onto their inputs.
    """
    rows: list[np.ndarray] = []
    rhs: list[np.ndarray] = []
    n_var = k_dof + kp
    # the annulus abs-ψ offset is a shared free column appended AFTER [coeffs,
    # a_pass]; every other prior row is padded with a 0 there.
    n_gauge = int(
        sp.anchor_form == "abs-psi"
        and sp.anchor_gauge_offset
        and sp.anchor_weight > 0.0
        and sp.anchor_ann_rows is not None
    )

    def _pad(row2d: np.ndarray) -> np.ndarray:
        if n_gauge and row2d.shape[1] == n_var:
            return np.hstack([row2d, np.zeros((row2d.shape[0], n_gauge))])
        return row2d

    # --- annulus boundary anchor -------------------------------------------
    anchor_on = (
        sp.anchor_form is not None
        and sp.anchor_weight > 0.0
        and sp.anchor_ann_rows is not None
    )
    if anchor_on:
        from imas_ambix.latent.boundary_prior import annulus_penalty_rows

        ann = np.asarray(sp.anchor_ann_rows, dtype=int)
        cells_flat = grid.cells[ann]  # flat-grid indices of the annulus points
        cg = grid.cell_greens()
        if sp.anchor_form == "abs-psi":
            psi_basis_ann = cg["psi"][ann, :] @ u_n
            psi_pass_ann = psi_pass[cells_flat, :] if kp else np.zeros((ann.size, 0))
            # coil ψ is IDENTICAL on both sides (same i_pf, same kernel) and
            # cancels in the residual, so the anchor compares the
            # solve's plasma+passive ψ against the harmonic read directly — the
            # known coil term is not carried as a fixed offset here.  The target
            # (harmonic ψ) likewise excludes the coil field.
            psi_fixed_ann = np.zeros(ann.size)
            a_ann, b_ann = annulus_penalty_rows(
                form="abs-psi",
                psi_basis_ann=psi_basis_ann,
                psi_pass_ann=psi_pass_ann,
                psi_fixed_ann=psi_fixed_ann,
                psi_target_ann=np.asarray(sp.anchor_psi_target, dtype=np.float64),
                k_dof=k_dof,
                kp=kp,
                weight=sp.anchor_weight,
                per_slice_uncertainty=sp.anchor_uncertainty,
                robust_clip=sp.anchor_robust_clip,
                gauge_offset=bool(n_gauge),
            )
            # abs-ψ already returns the offset column when gauge_offset — pad
            # only the OTHER priors, so append this one directly.
            if a_ann.shape[1] == n_var and n_gauge:
                a_ann = np.hstack([a_ann, np.zeros((a_ann.shape[0], 1))])
            rows.append(a_ann)
            rhs.append(b_ann)
        elif sp.anchor_form == "grad-psi":
            # Field-matched "virtual magnetics": match the flux GRADIENT ∇Φ in the
            # annulus (gauge-free — invariant to any ψ datum, so NO offset DOF),
            # the densified near-plasma field the source-free TH read resolves.
            # Work in ∇Φ so both sides match the harmonic target directly:
            #   dΦ/dR = +2πR·B_Z,  dΦ/dZ = −2πR·B_R   (total-flux convention).
            # Coil B cancels (identical both sides), so only plasma + passive enter.
            r_ann = grid.flat_r[cells_flat]
            two_pi_r = 2.0 * np.pi * r_ann
            # plasma basis: (n_ann, k_dof) B → ∇Φ, stacked [dΦ/dR ; dΦ/dZ]
            b_r_basis = cg["br"][ann, :] @ u_n
            b_z_basis = cg["bz"][ann, :] @ u_n
            grad_basis = np.vstack(
                [two_pi_r[:, None] * b_z_basis, -two_pi_r[:, None] * b_r_basis]
            )
            if kp and passive_br is not None:
                bpr = passive_br[cells_flat, :]
                bpz = passive_bz[cells_flat, :]
                grad_pass = np.vstack(
                    [two_pi_r[:, None] * bpz, -two_pi_r[:, None] * bpr]
                )
            else:
                grad_pass = np.zeros((2 * ann.size, kp))
            a_ann, b_ann = annulus_penalty_rows(
                form="grad-psi",
                psi_basis_ann=None,
                psi_pass_ann=None,
                psi_fixed_ann=None,
                psi_target_ann=None,
                grad_basis_ann=grad_basis,
                grad_pass_ann=grad_pass,
                grad_fixed_ann=np.zeros(2 * ann.size),
                grad_target_ann=np.asarray(sp.anchor_grad_target, dtype=np.float64),
                k_dof=k_dof,
                kp=kp,
                weight=sp.anchor_weight,
                per_slice_uncertainty=sp.anchor_uncertainty,
                robust_clip=sp.anchor_robust_clip,
            )
            rows.append(_pad(a_ann))
            rhs.append(b_ann)

    # --- q ≥ 1 (sawtooth): soft upper bound on on-axis current density ---
    # One-sided by active-set: a linear row pulls two-sidedly, so include it only
    # when the current iterate VIOLATES j_axis > j_axis_max (q_0 < 1); a healthy
    # q_0 ≥ 1 iterate adds no row and is left untouched.
    if sp.q_axis_max is not None:
        from imas_ambix.latent.profile_regularization import q_axis_penalty_row

        bound = q_axis_penalty_row(
            images_axis_unit=np.asarray(axis_images_unit, dtype=np.float64),
            weight=float(sp.q_weight),
            j_axis_max=float(sp.q_axis_max),
        )
        if bound.j_axis(np.asarray(coeffs_prev, dtype=np.float64)) > sp.q_axis_max:
            row = np.zeros((1, n_var))
            row[0, :k_dof] = bound.row
            rows.append(_pad(row))
            rhs.append(np.array([bound.rhs]))

    # --- Ip soft prior (opt-in; default is the hard KKT elsewhere) ---
    if sp.ip_soft_sigma is not None:
        from imas_ambix.latent.moment_priors import ip_soft_prior_row

        row, r = ip_soft_prior_row(
            a_anchor, ip_amperes, sigma_rel=sp.ip_soft_sigma, k_dof=k_dof, kp=kp
        )
        rows.append(_pad(row[np.newaxis, :]))
        rhs.append(np.array([r]))

    # --- βp + li/2 moment consistency (caller supplies the iterate sensitivity) ---
    if sp.beta_li_target is not None and sp.beta_li_sensitivity is not None:
        from imas_ambix.latent.moment_priors import moment_consistency_rows

        row, r = moment_consistency_rows(
            computed_moment_unit_sensitivity=np.asarray(sp.beta_li_sensitivity),
            target_moment=float(sp.beta_li_target),
            sigma=sp.beta_li_sigma,
            k_dof=k_dof,
            kp=kp,
        )
        rows.append(_pad(row[np.newaxis, :]))
        rhs.append(np.array([r]))

    # --- pressure prior: pull the p′-family toward a target (p′/FF′ split) ---
    if sp.pprime_target is not None and sp.pprime_basis is not None:
        from imas_ambix.latent.moment_priors import pressure_gradient_prior_rows

        prows, prhs = pressure_gradient_prior_rows(
            p_basis_slice=np.asarray(sp.pprime_basis, dtype=np.float64),
            pprime_target=np.asarray(sp.pprime_target, dtype=np.float64),
            sigma=sp.pprime_sigma,
            k_dof=k_dof,
            kp=kp,
        )
        rows.append(_pad(prows))
        rhs.append(prhs)

    # --- current-centroid position constraint (the well-posed position lever) ---
    # ∫R jφ = R_c·Ip and ∫Z jφ = Z_c·Ip are linear-homogeneous in the profile
    # coefficients (same form as the Ip anchor a_anchor = Σ_cell u_n); the
    # per-coefficient sensitivities are Σ_cell coord·u_n.  A strong soft tether
    # holds the measured centroid while the profile shape stays free.
    if sp.centroid_r_target is not None or sp.centroid_z_target is not None:
        from imas_ambix.latent.moment_priors import centroid_moment_rows

        crows, crhs = centroid_moment_rows(
            cell_r=grid.flat_r[grid.cells],
            cell_z=grid.flat_z[grid.cells],
            unit_cell_currents=u_n,
            r_target=sp.centroid_r_target,
            z_target=sp.centroid_z_target,
            ip_amperes=ip_amperes,
            sigma_r=sp.centroid_sigma_r,
            sigma_z=sp.centroid_sigma_z,
            k_dof=k_dof,
            kp=kp,
        )
        if crows.shape[0]:
            rows.append(_pad(crows))
            rhs.append(crhs)

    # --- passive trajectory prior: centre the eddy amplitudes on the
    # circuit-integrated trajectory (the L/R mode ODE driven by measured coil
    # and plasma current histories).  The sidecar coordinates are already
    # whitened (unit mode amplitude == unit-norm whitened sensor signal), so
    # the rows are the identity on the a_pass block; sqrt-weight scaling makes
    # the penalty weight·‖a_pass − center‖² in the stacked least squares.
    if (
        sp.passive_prior_center is not None
        and bool(np.any(np.asarray(sp.passive_prior_weight) > 0.0))
        and kp
    ):
        center = np.asarray(sp.passive_prior_center, dtype=np.float64)
        if center.size != kp:
            raise ValueError(
                f"passive_prior_center size {center.size} != sidecar rank {kp}"
            )
        weight = np.broadcast_to(
            np.asarray(sp.passive_prior_weight, dtype=np.float64), (kp,)
        )
        w_sq = np.sqrt(np.clip(weight, 0.0, None))
        row = np.zeros((kp, n_var))
        row[:, k_dof : k_dof + kp] = np.diag(w_sq)
        rows.append(_pad(row))
        rhs.append(w_sq * center)

    # --- profile-coefficient temporal-consistency prior (current diffusion):
    # centre the ladder coefficients on the diffusion-evolved prediction from
    # the previous slice.  Coefficients are the solve's normalised O(1)
    # convention on both sides, so the rows are the identity on the coeff
    # block.  The weight is RELATIVE to the data Gram's mean diagonal (the
    # smoothness-knob convention): the Ip-normalised profile columns carry
    # whitened curvature ~10⁴–10⁵ per unit coefficient, so an absolute O(1)
    # weight is invisible — the penalty is weight·gram_scale·‖c − center‖².
    if sp.coeff_prior_center is not None and sp.coeff_prior_weight > 0.0:
        center = np.asarray(sp.coeff_prior_center, dtype=np.float64)
        if center.size != k_dof:
            raise ValueError(
                f"coeff_prior_center size {center.size} != profile DOF {k_dof}"
            )
        w_sq = np.sqrt(float(sp.coeff_prior_weight) * max(data_gram_scale, 1e-30))
        row = np.zeros((k_dof, n_var))
        row[:, :k_dof] = w_sq * np.eye(k_dof)
        rows.append(_pad(row))
        rhs.append(w_sq * center)

    if not rows:
        return np.zeros((0, n_var + n_gauge)), np.zeros(0), n_gauge
    return np.vstack(rows), np.concatenate(rhs), n_gauge


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
    table: OperatorGeometry,
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

    classes = op.classify_circuits(
        _geometry_member(table, "conductors", "pf_filaments"),
        _geometry_member(table, "available_current_channels", "amc_current_channels"),
        _geometry_member(table, "active_circuits", "active_circuits"),
        _geometry_member(table, "drive_map", "circuit_drives"),
    )
    by_circ: dict[int, list] = {}
    for f in _geometry_member(table, "conductors", "pf_filaments"):
        by_circ.setdefault(f.circuit, []).append(f)
    psi_cols = []
    br_cols = []
    bz_cols = []
    for cc in classes:
        if cc.role in op._KNOWN_ROLES:
            continue
        acc = np.zeros(grid.flat_r.size)
        acc_br = np.zeros(grid.flat_r.size)
        acc_bz = np.zeros(grid.flat_r.size)
        for f in by_circ[cc.circuit]:
            psi_f, br_f, bz_f = hybrid_greens(
                grid.flat_r,
                grid.flat_z,
                float(f.r),
                float(f.z),
                max(abs(f.width), 0.01),
                max(abs(f.height), 0.01),
            )
            acc += f.xmult * psi_f
            acc_br += f.xmult * br_f
            acc_bz += f.xmult * bz_f
        psi_cols.append(acc)
        br_cols.append(acc_br)
        bz_cols.append(acc_bz)
    psi_full = (
        np.column_stack(psi_cols) if psi_cols else np.zeros((grid.flat_r.size, 0))
    )
    br_full = np.column_stack(br_cols) if br_cols else np.zeros((grid.flat_r.size, 0))
    bz_full = np.column_stack(bz_cols) if bz_cols else np.zeros((grid.flat_r.size, 0))
    if psi_full.shape[1] != g_passive.shape[1]:
        raise ValueError(
            f"passive circuit count mismatch: table {psi_full.shape[1]} vs "
            f"g_passive {g_passive.shape[1]}"
        )
    return {
        "g_cols": g_passive @ v_over_s,
        "psi_cols": psi_full @ v_over_s,
        # passive poloidal-field grid columns (for the grad-ψ annulus anchor —
        # the field-matched "virtual magnetics" arm; coil B cancels like coil ψ)
        "br_cols": br_full @ v_over_s,
        "bz_cols": bz_full @ v_over_s,
        "k": k,
        "modes": v_over_s,
    }


class _AndersonMixer:
    """Safeguarded Anderson acceleration of a relaxed fixed-point iteration.

    The relaxed-Picard update x ← x + β(g(x)−x) converges linearly and, at
    β = 0.5 on the free-boundary map, slowly (tens–hundreds of sweeps).
    Anderson mixing reuses the last ``depth`` residual/iterate differences to
    take the (regularised) least-squares step

        x⁺ = x + β f − (ΔX + β ΔF) γ,   γ = argmin_γ ‖f − ΔF γ‖²

    (Walker–Ni type-II; ΔF, ΔX the consecutive differences of the residual
    f = g(x)−x and the iterate x).  It reduces to plain relaxed Picard while
    the history is shorter than two.

    The free-boundary map is only piecewise-smooth — the topology read moves
    the core mask and re-fits the profile discretely — but the position of the
    equilibrium is held on the physical branch by the current-centroid pin (a
    magnetics-derived soft prior), so the failure mode is NOT basin escape but
    slow linear convergence: the axis creeps toward its Shafranov-shifted fixed
    point over hundreds of sweeps while the raw residual oscillates ~1e-3.  That
    oscillation is normal, so restart-on-growth would falsely reset the physical
    branch and sabotage the acceleration.  Two cheap guards (no extra map
    evaluation) suffice:

    * **step cap** — the accepted move ‖x⁺−x‖ is capped at ``kappa``× the
      Picard step ‖βf‖, bounding any single Anderson step against the residual
      spikes of the early fixed-shape transient;
    * **ridge + finite check** — a Tikhonov term (relative to the mean diagonal
      of ΔFᵀΔF) conditions the small normal-equation solve, and a non-finite
      candidate falls back to the plain relaxed step.

    Engagement is delayed until after the fixed-shape warmup transient (the
    caller only calls :meth:`step` once the LSQ profile is live), so Anderson
    accelerates the slow crawl rather than amplifying the chaotic early sweeps.
    An Anderson run therefore reaches the SAME fixed point Picard approaches,
    in far fewer sweeps.
    """

    def __init__(
        self,
        depth: int = 6,
        ridge: float = 1e-6,
        kappa: float = 2.0,
    ) -> None:
        self.depth = max(1, int(depth))
        self.ridge = float(ridge)
        self.kappa = float(kappa)
        self._x: list[np.ndarray] = []
        self._f: list[np.ndarray] = []

    def step(self, x: np.ndarray, g: np.ndarray, beta: float) -> np.ndarray:
        """Next iterate from the current iterate ``x`` and its map image ``g``."""
        f = g - x
        picard = x + beta * f
        if not np.isfinite(f).all():
            self._x = []
            self._f = []
            return picard
        self._x.append(x)
        self._f.append(f)
        if len(self._x) > self.depth + 1:
            del self._x[0]
            del self._f[0]
        if len(self._x) < 2:
            return picard
        d_f = np.column_stack(
            [self._f[i + 1] - self._f[i] for i in range(len(self._f) - 1)]
        )
        d_x = np.column_stack(
            [self._x[i + 1] - self._x[i] for i in range(len(self._x) - 1)]
        )
        ftf = d_f.T @ d_f
        m = ftf.shape[0]
        reg = self.ridge * (np.trace(ftf) / max(m, 1) or 1.0) * np.eye(m)
        try:
            gamma = np.linalg.solve(ftf + reg, d_f.T @ f)
        except np.linalg.LinAlgError:
            return picard
        cand = picard - (d_x + beta * d_f) @ gamma
        if not np.isfinite(cand).all():
            return picard
        # step cap: never move more than kappa× the plain Picard step (bounds a
        # single wild step during the early transient; the branch itself is held
        # by the current-centroid pin, so no basin-escape guard is needed).
        step = cand - x
        step_norm = float(np.linalg.norm(step))
        picard_norm = float(np.linalg.norm(beta * f))
        if picard_norm > 0.0 and step_norm > self.kappa * picard_norm:
            cand = x + step * (self.kappa * picard_norm / step_norm)
        return cand


def solve_equilibrium_lsq(
    grid: EquilibriumGrid,
    table: OperatorGeometry,
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
    soft_priors: SoftPriors | None = None,
    substrate: str = SUBSTRATE_GRID,
    accelerator: str = "picard",
    anderson_depth: int = 6,
    anderson_ridge: float = 1e-8,
    topology_read: str = TOPOLOGY_HARD,
    smooth_temperature: float = 1e-3,
    iteration_trace: list[dict] | None = None,
) -> LadderFit:
    """Free-boundary Picard solve with the profile coefficients re-fit by
    whitened linear least squares against the raw magnetics every sweep.

    ``soft_priors`` (a :class:`SoftPriors`) folds optional physics priors into
    each sweep's LSQ: the annulus boundary anchor (the soft prior tying the
    near-edge field to the source-free harmonic read), the soft SOL current
    edge (a C¹ decay foot admitting public-SOL current to ψ_N ≈ cap), a q ≥ 1
    sawtooth clamp on the on-axis current density, and the Ip-soft / βp+li/2 /
    pressure moment priors.  ``None`` (default) is byte-identical to the
    prior-free solve.

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

    ``substrate`` selects the ψ evaluator (:data:`SUBSTRATES`): the default
    ``grid-delstar`` is byte-unchanged; ``greens-matvec`` drops the gridded Δ*
    solve for the analytic Green's matvec (the grid-free substrate spike).

    ``topology_read`` selects the per-sweep topology read
    (:data:`TOPOLOGY_READS`): the default ``hard`` is byte-unchanged;
    ``connectivity`` swaps in the temperature-smoothed kernel (softmin
    binding + retracted-gate sigmoid core weight + sub-grid stencil axis, at
    smoothing scale ``smooth_temperature``), under which the fixed-point map
    is differentiable — no discrete mask flips enter the Picard/Anderson
    path.  Incompatible with a soft SOL cap (``sol_cap`` > 1 raises).
    """
    if substrate not in SUBSTRATES:
        raise ValueError(f"substrate must be one of {SUBSTRATES}, got {substrate!r}")
    if accelerator not in ("picard", "anderson"):
        raise ValueError(
            f"accelerator must be 'picard' or 'anderson', got {accelerator!r}"
        )
    if topology_read not in TOPOLOGY_READS:
        raise ValueError(
            f"topology_read must be one of {TOPOLOGY_READS}, got {topology_read!r}"
        )
    # ``picard`` (default) is byte-identical to the historical relaxed solve;
    # ``anderson`` wraps the SAME relaxed-Picard update in a safeguarded
    # Anderson mixer (fewer sweeps to the identical fixed point).
    mixer = (
        _AndersonMixer(depth=anderson_depth, ridge=anderson_ridge)
        if accelerator == "anderson"
        else None
    )
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

    sp = soft_priors if soft_priors is not None else SoftPriors()
    core_cap = sp.sol_cap if sp.sol_active else 1.0

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

        psi_new = _plasma_psi_field(
            grid, jphi * scale_ip, i_cell, psi_coil, substrate=substrate
        )
        if kp:
            psi_new = psi_new + psi_pass @ a_pass

        if psi_flat is None:
            psi_flat = psi_new
        else:
            # residual is the RAW map update ‖g(ψ)−ψ‖ (pre-mixing), so the
            # convergence criterion is identical whether the mixing is plain
            # relaxed Picard or Anderson — the accelerator changes the path to
            # the fixed point, never the fixed point or its residual measure.
            residual = float(
                np.abs(psi_new - psi_flat).max() / max(np.abs(psi_new).max(), 1e-12)
            )
            # engage Anderson only once the LSQ profile is live and past the
            # fixed-shape warmup transient (before that the residual spikes are
            # the transient, not the slow crawl Anderson is meant to break).
            if mixer is not None and iteration > warmup_iterations + 2:
                psi_flat = mixer.step(psi_flat, psi_new, relax)
            else:
                psi_flat = relax * psi_new + (1.0 - relax) * psi_flat

        psi2d = psi_flat.reshape(grid.nz, grid.nr)
        if topology_read == TOPOLOGY_CONNECTIVITY:
            # continuous read: connectivity binding + stencil axis + smooth
            # core weight — the fixed-point map is differentiable, so the
            # residual decreases monotonically instead of limit-cycling on
            # discrete mask flips.  The previous sweep's axis seeds the flood.
            axis, axis_psi, boundary_psi, core_weight, core = _read_topology_smooth(
                psi_flat, grid, axis, core_cap, smooth_temperature
            )
            span = boundary_psi - axis_psi
            if abs(span) < 1e-12:
                span = 1e-12
            psi_n = (psi_flat - axis_psi) / span
            ia = int(np.argmin(np.abs(grid.zg - axis[1])))
            ja = int(np.argmin(np.abs(grid.rg - axis[0])))
        else:
            core_weight = None
            axis, axis_psi = _read_axis(psi2d, grid, sign)
            boundary_psi = _read_boundary_psi(psi2d, grid, axis_psi)
            span = boundary_psi - axis_psi
            if abs(span) < 1e-12:
                span = 1e-12
            psi_n = (psi_flat - axis_psi) / span

            # core support: axis-connected component of ψ_N < cap inside the
            # limiter.  cap = 1 is the hard separatrix mask; cap > 1 (soft SOL
            # edge) admits the public-SOL band, and the axis-connected component
            # keeps ONLY the common-flux region (private flux below an X-point is a
            # separate component that does not contain the axis).
            closed = ((psi_n < core_cap) & grid.inside_limiter.ravel()).reshape(
                grid.nz, grid.nr
            )
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
            if core_weight is None:
                jphi[core.ravel()] = shape[core.ravel()]
            else:
                jphi = shape * core_weight
        else:
            # basis images on the current core; L1-normalise each column to
            # carry |Ip| of gross current so the coefficients stay O(1).  In
            # nonneg mode normalisation is SIGNED so c ≥ 0 ⇒ jφ·sign(Ip) ≥ 0
            # (the non-negative profile invariant) and the Ip anchor stays
            # reachable for either current direction.  With the soft SOL edge
            # the basis swaps to the footed form (current decays through a C¹
            # foot into the SOL band instead of vanishing at ψ_N = 1).
            if sp.sol_active:
                from imas_ambix.latent.profile_regularization import (  # noqa: PLC0415
                    profile_basis_foot,
                )

                images = profile_basis_foot(
                    psi_n,
                    grid.flat_r,
                    r0=grid.r0,
                    n_p=n_p,
                    n_f=n_f,
                    kind="monomial-nonneg" if nonneg else "legendre",
                    w=sp.sol_foot_w,
                    cap=sp.sol_cap,
                    centrifugal_gamma=centrifugal_gamma,
                )
            else:
                images = profile_basis(
                    psi_n,
                    grid.flat_r,
                    r0=grid.r0,
                    n_p=n_p,
                    n_f=n_f,
                    kind="monomial-nonneg" if nonneg else "legendre",
                    centrifugal_gamma=centrifugal_gamma,
                )
            if core_weight is None:
                images[~core.ravel(), :] = 0.0
            else:
                # smooth core membership: a cell's contribution fades
                # continuously through the edge instead of flipping in/out
                images = images * core_weight[:, np.newaxis]
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
            # on-axis current density per unit (normalised) coefficient — the
            # sensitivity the q ≥ 1 sawtooth bound acts on.  The core cell
            # nearest the axis carries j_axis = (u_n / cell_area) · coeffs.
            axis_flat = int(ia * grid.nr + ja)
            axis_local = int(np.searchsorted(grid.cells, axis_flat))
            if axis_local < grid.cells.size and grid.cells[axis_local] == axis_flat:
                axis_images_unit = u_n[axis_local, :] / cell_area
            else:  # axis cell outside the in-limiter set (transient) — no bound
                axis_images_unit = np.zeros(k_dof)
            new_coeffs = None
            new_a_pass = None
            # the coefficients are determined by the magnetics data when it is
            # present (the reconstruction), OR by the position constraint alone
            # when it is not (the well-posed position-controlled solve: Ip +
            # current-centroid moment, profile otherwise free).  The data-rich
            # branch is unchanged, so the reconstruction stays byte-identical.
            if ok_cols.any() and (n_data >= n_var or sp.any_penalty):
                cols = np.hstack([b_mat, bp]) if kp else b_mat
                # ridges are RELATIVE: the unit-weight Grams are rescaled by
                # the data Gram's mean diagonal so --smoothness and
                # --passive-ridge are dimensionless O(0.01–1) knobs
                h_data = cols.T @ cols
                mean_diag = np.trace(h_data) / max(n_var, 1)
                # no magnetics data (position-controlled solve): the Gram is
                # zero, so give the relative regularisers an absolute unit
                # scale — leaves the data-rich reconstruction untouched.
                if mean_diag <= 0.0:
                    mean_diag = 1.0
                anchor = np.concatenate([a_anchor, np.zeros(kp)])
                # assemble optional soft-prior rows (annulus anchor, q bound,
                # Ip-soft, moment, pressure) on x = [coeffs, a_pass] (+1 gauge
                # offset column for the annulus abs-ψ form).
                a_extra, b_extra, n_gauge = (
                    _assemble_soft_prior_rows(
                        sp,
                        grid=grid,
                        u_n=u_n,
                        a_anchor=a_anchor,
                        axis_images_unit=axis_images_unit,
                        coeffs_prev=coeffs,
                        psi_coil=psi_coil,
                        psi_pass=psi_pass,
                        a_pass=a_pass,
                        ip_amperes=ip_amperes,
                        k_dof=k_dof,
                        kp=kp,
                        data_gram_scale=mean_diag,
                        passive_br=(
                            np.asarray(passive["br_cols"], dtype=np.float64)
                            if kp and "br_cols" in passive
                            else None
                        ),
                        passive_bz=(
                            np.asarray(passive["bz_cols"], dtype=np.float64)
                            if kp and "bz_cols" in passive
                            else None
                        ),
                    )
                    if sp.any_penalty
                    else (np.zeros((0, n_var)), np.zeros(0), 0)
                )
                if nonneg:
                    if n_gauge:
                        raise NotImplementedError(
                            "annulus abs-ψ gauge offset is unsupported in the "
                            "nonneg arm; use the free-sign solve for the anchor"
                        )
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
                        extra_rows=(a_extra, b_extra) if a_extra.shape[0] else None,
                    )
                else:
                    n_all = n_var + n_gauge
                    h_core = np.zeros((n_all, n_all))
                    h_core[:n_var, :n_var] = h_data + s_full_block(
                        n_var, k_dof, kp, smoothness * mean_diag, s_gram, passive_ridge
                    )
                    lin = np.zeros(n_all)
                    lin[:n_var] = cols.T @ y
                    if a_extra.shape[0]:
                        h_core += a_extra.T @ a_extra
                        lin += a_extra.T @ b_extra
                    h = 2.0 * h_core
                    h += 1e-10 * np.trace(h) / max(n_all, 1) * np.eye(n_all)
                    anchor_full = np.concatenate([a_anchor, np.zeros(kp + n_gauge)])
                    kkt = np.zeros((n_all + 1, n_all + 1))
                    kkt[:n_all, :n_all] = h
                    kkt[:n_all, n_all] = anchor_full
                    kkt[n_all, :n_all] = anchor_full
                    rhs = np.concatenate([2.0 * lin, [ip_amperes]])
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
    table: OperatorGeometry,
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

    Under the grid-free ``greens-matvec`` substrate the stage-1 bootstrap uses
    the analytic-add matvec (the boundary-continuation arm has no grid-free
    form), so the whole ladder stays free of the gridded Δ* solve.
    """
    substrate = solve_kwargs.get("substrate", SUBSTRATE_GRID)
    if initial_jphi is None:
        stage1 = solve_equilibrium(
            grid,
            i_pf,
            ip_amperes,
            beta0=0.5,
            alpha=1.0,
            max_iterations=bootstrap_iterations,
            coil_field_mode=(
                "analytic-add"
                if substrate == SUBSTRATE_GREENS
                else "boundary-continuation"
            ),
            substrate=substrate,
            seed_z0=solve_kwargs.get("seed_z0", 0.0),
        )
        initial_jphi = stage1.jphi.ravel()
        if solve_kwargs.get("nonneg", False):
            # the sign-constrained solve is basin-fragile: from a fixed-shape
            # seed the Picard escapes to the outboard corner attractor, while
            # an equally-good PHYSICAL fixed point exists on the confined
            # branch (measured: free-sign 0.428 vs seeded nonneg 0.434 on the
            # same slice).  The stable free-sign basin solve scouts the branch;
            # the sign-constrained solve then certifies a physical profile
            # there: the free-sign solve locates the branch and the
            # sign-constrained solve certifies a physical profile.
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
                substrate=substrate,
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
    "SoftPriors",
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
