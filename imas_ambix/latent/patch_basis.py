"""Patch-current forward substrate: precomputed finite-area Green's matrices.

The plasma is represented as piecewise-constant toroidal currents on the
in-limiter grid cells ("patches").  Flux, field, and diagnostics everywhere are
then a single matmul against precomputed per-campaign Green's interaction
matrices — no grid solve, no boundary conditions, no Picard iteration.  In this
basis Ampère's law and div B = 0 hold identically for ANY current vector (the
finite-area kernel is the exact axisymmetric field of a uniform rectangular
current section), so the only remaining physics content of Grad-Shafranov is the
flux-function structure ``jφ = a(ψ)·R + b(ψ)/R`` — carried by the separate
structure-residual module, not here.

This module lifts the scoping assembly (``scripts/patch_basis_studies.py``) into
a torch :class:`~torch.nn.Module` so the gate-scale inverse and the amortised
encoder can run the forward batched on GPU.

Conventions (matching :mod:`imas_ambix.latent.gs_solve` /
:mod:`imas_ambix.gs.operator`):

* Every ψ column carries the TOTAL poloidal flux ``Φ = 2π R A_φ`` [Wb per A]
  (the flux a flux loop measures — NOT the stream function ``Φ/2π``).
* B-probe sensor rows are the orientation-projected field [T per A].
* The patch→grid / patch→cell columns use the finite-area kernel
  (:func:`imas_ambix.gs.cylinder.hybrid_greens`: cylinder near the section,
  point filament far).  The kernel is regular INSIDE the conductor, so the
  patch→patch-centroid flux ``g_cc`` (and hence the structure residual read off
  it) is well-defined with NO self-inductance regularisation.

Shapes (``B`` = batch, ``n`` = in-limiter cells, ``S`` = sensors,
``C`` = KNOWN coils, ``G`` = grid points = nr·nz):

* ``i_cell`` : ``(n,)`` or ``(B, n)`` — per-cell current [A];
* ``i_pf``   : ``(C,)`` or ``(B, C)`` — KNOWN PF-coil currents [A];
* ``sensors``  → ``(B, S)`` sensor prediction [Wb / T];
* ``psi_grid`` → ``(B, G)`` flux [Wb]; ``psi_grid_2d`` → ``(B, nz, nr)``;
* ``psi_cells`` → ``(B, n)`` flux at the cell centroids [Wb].

Buffers are registered at ``dtype`` (fp32 by default) for the batched GPU
forward; fp64 numpy copies are retained for the ``*_np`` topology-read path,
which needs the extra precision to place the axis on the grid.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn

from imas_ambix.gs import operator as op
from imas_ambix.gs.cylinder import hybrid_greens
from imas_ambix.latent.gs_solve import EquilibriumGrid

if TYPE_CHECKING:
    from imas_ambix.gs.geometry import GeometryTable

# Reuse the scoping cache so an already-assembled 65x97 matrix is not rebuilt.
_DEFAULT_CACHE_DIR = Path("imas_ambix/latent/artifacts/patch_scoping")


def _assemble_g_pg(grid: EquilibriumGrid) -> np.ndarray:
    """(n_grid, n_cells) total-flux interaction matrix [Wb per A], fp64.

    Column ``c`` is ψ on the whole grid per ampere of current spread uniformly
    over cell ``c`` (finite-area kernel near the cell, point filament far).
    """
    cols = np.empty((grid.flat_r.size, grid.cells.size), dtype=np.float64)
    for k, c in enumerate(grid.cells):
        cols[:, k] = hybrid_greens(
            grid.flat_r,
            grid.flat_z,
            float(grid.flat_r[c]),
            float(grid.flat_z[c]),
            grid.dr,
            grid.dz,
        )[0]
    return cols


def _load_or_assemble_g_pg(grid: EquilibriumGrid, cache: Path | None) -> np.ndarray:
    """Load the patch→grid matrix from ``cache`` if present, else assemble + save."""
    if cache is not None and cache.exists():
        stored = np.load(cache)["g_pg"]
        if stored.shape == (grid.flat_r.size, grid.cells.size):
            return np.asarray(stored, dtype=np.float64)
    g_pg = _assemble_g_pg(grid)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, g_pg=g_pg)
    return g_pg


def _coil_sensor_matrix(table: GeometryTable) -> np.ndarray:
    """KNOWN-coil → sensor Green's matrix ``(n_sensor, n_coil)``.

    Assembled the same way as :meth:`EquilibriumGrid.from_table`'s coil ψ
    columns (one column per merged circuit group, averaged over redundant
    circuits, each a finite-area winding pack) but projected to each sensor:
    ψ [Wb per A] for flux loops, orientation-projected field [T per A] for
    B-probes.  Rows follow ``table.sensor_map`` (the ``sensor_greens`` order),
    columns follow ``fwd.pf_merged_circuits`` (the ``coil_psi`` / ``i_pf``
    order), so the two agree with the patch operators everywhere.
    """
    fwd = op.build_operator(table)
    by_circ: dict[int, list] = {}
    for f in table.pf_filaments:
        by_circ.setdefault(f.circuit, []).append(f)
    n_coil = len(fwd.pf_merged_circuits)
    rows: list[np.ndarray] = []
    for m in table.sensor_map:
        ang = np.deg2rad(m.angle_deg if m.angle_deg is not None else 90.0)
        tr = np.array([m.r], dtype=np.float64)
        tz = np.array([m.z], dtype=np.float64)
        row = np.zeros(n_coil, dtype=np.float64)
        for j, circs in enumerate(fwd.pf_merged_circuits):
            per_circ = []
            for c in circs:
                acc = 0.0
                for f in by_circ[c]:
                    psi_f, br_f, bz_f = hybrid_greens(
                        tr,
                        tz,
                        float(f.r),
                        float(f.z),
                        max(abs(f.width), 0.01),
                        max(abs(f.height), 0.01),
                    )
                    if m.kind == "flux_loop":
                        acc += f.xmult * float(psi_f[0])
                    else:
                        acc += f.xmult * float(
                            br_f[0] * np.cos(ang) + bz_f[0] * np.sin(ang)
                        )
                per_circ.append(acc)
            row[j] = float(np.mean(per_circ)) if per_circ else 0.0
        rows.append(row)
    return np.vstack(rows) if rows else np.zeros((0, n_coil), dtype=np.float64)


class PatchBasis(nn.Module):
    """Per-campaign patch-current forward substrate (fixed geometry, matmul-only).

    Built once per campaign :class:`~imas_ambix.gs.geometry.GeometryTable`; a
    forward pass is a batched matmul against the precomputed interaction
    matrices.  All buffers are pure device geometry — no EFIT, no labels.
    """

    def __init__(
        self,
        *,
        g_pg: np.ndarray,  # (G, n) patch→grid total flux [Wb/A]
        g_cc: np.ndarray,  # (n, n) patch→cell-centroid flux [Wb/A]
        m_sens: np.ndarray,  # (S, n) patch→sensor
        m_coil: np.ndarray,  # (S, C) coil→sensor
        psi_coil_grid: np.ndarray,  # (G, C) coil→grid ψ [Wb/A]
        psi_coil_cells: np.ndarray,  # (n, C) coil→cell ψ [Wb/A]
        r_cells: np.ndarray,  # (n,)
        z_cells: np.ndarray,  # (n,)
        candidate_mask: np.ndarray,  # (n,) conductor-clear in-limiter cells
        grid_r: np.ndarray,  # (nr,)
        grid_z: np.ndarray,  # (nz,)
        nr: int,
        nz: int,
        cell_area: float,
        r0: float,
        sensor_channels: list[str],
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.nr = int(nr)
        self.nz = int(nz)
        self.cell_area = float(cell_area)
        self.r0 = float(r0)
        self.sensor_channels = list(sensor_channels)
        self.dtype = dtype

        # fp64 numpy copies for the topology-read path (axis placement fidelity)
        self._g_pg_np = np.asarray(g_pg, dtype=np.float64)
        self._g_cc_np = np.asarray(g_cc, dtype=np.float64)
        self._psi_coil_grid_np = np.asarray(psi_coil_grid, dtype=np.float64)
        self._psi_coil_cells_np = np.asarray(psi_coil_cells, dtype=np.float64)

        def buf(x: np.ndarray) -> torch.Tensor:
            return torch.tensor(np.asarray(x, dtype=np.float64), dtype=dtype)

        self.register_buffer("g_pg", buf(g_pg))
        self.register_buffer("g_cc", buf(g_cc))
        self.register_buffer("m_sens", buf(m_sens))
        self.register_buffer("m_coil", buf(m_coil))
        self.register_buffer("psi_coil_grid", buf(psi_coil_grid))
        self.register_buffer("psi_coil_cells", buf(psi_coil_cells))
        self.register_buffer("r_cells", buf(r_cells))
        self.register_buffer("z_cells", buf(z_cells))
        self.register_buffer("candidate_mask", buf(candidate_mask))
        self.register_buffer("grid_r", buf(grid_r))
        self.register_buffer("grid_z", buf(grid_z))

    # ---- construction ----

    @classmethod
    def from_table(
        cls,
        table: GeometryTable,
        *,
        nr: int = 65,
        nz: int = 97,
        cache_dir: str | Path | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> PatchBasis:
        """Build from a campaign :class:`GeometryTable`.

        Assembles in fp64 numpy and registers buffers at ``dtype``.  The
        expensive patch→grid matrix is cached to
        ``{cache_dir}/g_pg_{signature.key}_{nr}x{nz}.npz`` (default
        ``imas_ambix/latent/artifacts/patch_scoping`` so an existing 65x97
        cache is reused); the remaining matrices are cheap and rebuilt each
        call from the :class:`EquilibriumGrid`.
        """
        grid = EquilibriumGrid.from_table(table, nr=nr, nz=nz)
        cache_root = Path(cache_dir) if cache_dir is not None else _DEFAULT_CACHE_DIR
        cache = cache_root / f"g_pg_{table.signature.key}_{nr}x{nz}.npz"

        g_pg = _load_or_assemble_g_pg(grid, cache)  # (G, n)
        g_cc = g_pg[grid.cells, :]  # (n, n) patch→cell-centroid flux
        m_sens, channels = grid.sensor_greens(table)  # (S, n)
        m_coil = _coil_sensor_matrix(table)  # (S, C)
        # grid.from_table averages redundant circuits into these columns in
        # fwd.pf_merged_circuits order — identical ordering to m_coil / i_pf.
        psi_coil_grid = np.asarray(grid._coil_psi_columns, dtype=np.float64)  # (G, C)
        psi_coil_cells = psi_coil_grid[grid.cells, :]  # (n, C)
        r_cells = grid.flat_r[grid.cells]
        z_cells = grid.flat_z[grid.cells]
        candidate_mask = grid.topology_candidate[grid.cells].astype(np.float64)

        return cls(
            g_pg=g_pg,
            g_cc=g_cc,
            m_sens=m_sens,
            m_coil=m_coil,
            psi_coil_grid=psi_coil_grid,
            psi_coil_cells=psi_coil_cells,
            r_cells=r_cells,
            z_cells=z_cells,
            candidate_mask=candidate_mask,
            grid_r=grid.rg,
            grid_z=grid.zg,
            nr=nr,
            nz=nz,
            cell_area=grid.dr * grid.dz,
            r0=grid.r0,
            sensor_channels=channels,
            dtype=dtype,
        )

    # ---- batched torch forward maps ----

    @staticmethod
    def _batched(x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(0) if x.dim() == 1 else x

    def _apply(
        self, mat: torch.Tensor, i_cell, coil_mat: torch.Tensor, i_pf
    ) -> torch.Tensor:
        """``i_cell @ mat.T + i_pf @ coil_mat.T`` with 1-D → batched promotion."""
        ic = self._batched(torch.as_tensor(i_cell))
        out = ic @ mat.to(device=ic.device, dtype=ic.dtype).T
        if coil_mat.shape[1] and i_pf is not None:
            ipf = torch.as_tensor(i_pf).to(device=ic.device, dtype=ic.dtype)
            ipf = self._batched(ipf)
            out = out + ipf @ coil_mat.to(device=ic.device, dtype=ic.dtype).T
        return out

    def sensors(self, i_cell, i_pf=None) -> torch.Tensor:
        """Predicted sensor magnetics ``(B, S)`` [Wb for flux loops, T for probes]."""
        return self._apply(self.m_sens, i_cell, self.m_coil, i_pf)

    def psi_grid(self, i_cell, i_pf=None) -> torch.Tensor:
        """Total poloidal flux ``(B, G)`` [Wb] on the flattened grid."""
        return self._apply(self.g_pg, i_cell, self.psi_coil_grid, i_pf)

    def psi_grid_2d(self, i_cell, i_pf=None) -> torch.Tensor:
        """Total poloidal flux ``(B, nz, nr)`` [Wb] (row = Z, col = R)."""
        psi = self.psi_grid(i_cell, i_pf)
        return psi.reshape(psi.shape[0], self.nz, self.nr)

    def psi_cells(self, i_cell, i_pf=None) -> torch.Tensor:
        """Total poloidal flux ``(B, n)`` [Wb] at the cell centroids."""
        return self._apply(self.g_cc, i_cell, self.psi_coil_cells, i_pf)

    def psi_coil_cells_for(self, i_pf: np.ndarray) -> torch.Tensor:
        """KNOWN-coil ψ at the cell centroids ``(n,)`` [Wb] for currents ``i_pf``."""
        ipf = torch.as_tensor(
            np.asarray(i_pf, dtype=np.float64),
            dtype=self.psi_coil_cells.dtype,
            device=self.psi_coil_cells.device,
        )
        return self.psi_coil_cells @ ipf

    # ---- fp64 numpy topology-read path ----

    def psi_grid_2d_np(self, i_cell: np.ndarray, i_pf: np.ndarray) -> np.ndarray:
        """Total poloidal flux ``(nz, nr)`` [Wb] in fp64 for one slice."""
        ic = np.asarray(i_cell, dtype=np.float64)
        psi = self._g_pg_np @ ic
        if self._psi_coil_grid_np.shape[1]:
            psi = psi + self._psi_coil_grid_np @ np.asarray(i_pf, dtype=np.float64)
        return psi.reshape(self.nz, self.nr)

    def psi_cells_np(self, i_cell: np.ndarray, i_pf: np.ndarray) -> np.ndarray:
        """Total poloidal flux ``(n,)`` [Wb] at the cell centroids in fp64."""
        ic = np.asarray(i_cell, dtype=np.float64)
        psi = self._g_cc_np @ ic
        if self._psi_coil_cells_np.shape[1]:
            psi = psi + self._psi_coil_cells_np @ np.asarray(i_pf, dtype=np.float64)
        return psi

    # ---- throughput bench ----

    def throughput(self, batch: int, n_iter: int, device: str | torch.device) -> float:
        """Median slices/s of the batched forward (``psi_grid`` + ``sensors``).

        Random currents; CUDA timers are synchronised around each iteration.
        """
        dev = torch.device(device)
        n = int(self.r_cells.shape[0])
        n_coil = int(self.psi_coil_grid.shape[1])
        g_pg = self.g_pg.to(dev)
        m_sens = self.m_sens.to(dev)
        psi_coil_grid = self.psi_coil_grid.to(dev)
        m_coil = self.m_coil.to(dev)
        i_cell = torch.randn(batch, n, dtype=self.dtype, device=dev)
        i_pf = torch.randn(batch, n_coil, dtype=self.dtype, device=dev)
        is_cuda = dev.type == "cuda"

        def one() -> None:
            _psi = i_cell @ g_pg.T
            _sens = i_cell @ m_sens.T
            if n_coil:
                _psi = _psi + i_pf @ psi_coil_grid.T
                _sens = _sens + i_pf @ m_coil.T

        for _ in range(3):  # warmup
            one()
        if is_cuda:
            torch.cuda.synchronize()
        times = np.empty(n_iter)
        for k in range(n_iter):
            t0 = time.perf_counter()
            one()
            if is_cuda:
                torch.cuda.synchronize()
            times[k] = time.perf_counter() - t0
        return float(batch / np.median(times))


__all__ = ["PatchBasis"]
