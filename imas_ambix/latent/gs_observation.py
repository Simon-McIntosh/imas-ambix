"""Differentiable (torch) Grad-Shafranov observation operator + ψ-field readout.

The geometry-only Green's-function :class:`imas_ambix.gs.operator.ForwardOperator`
is a fixed *linear* map from current amplitudes to sensor signals — pure device
geometry, no EFIT.  This module lifts it into a torch :class:`~torch.nn.Module`
so the hybrid latent (v0 stage 2) can be grounded in the raw magnetics *and* so
field topology can be read from the reconstructed ψ field, both differentiably.

Two products, one shared Green's physics:

1. **Sensor prediction** ``B̂ = G_pf·i_pf + G_plasma·c_plasma`` — matched, at
   training time, to the RAW measured magnetics (the spatial GS anchor, §8).
2. **ψ-field reconstruction** ``ψ(R,Z) = Σ_node Φ(R,Z; node)·c_plasma
   + Σ_coil Φ(R,Z; coil)·i_pf`` on an arbitrary grid — the *one solved flux
   field* from which axis / X-points / LCFS / public-private are read
   deterministically (§3, topology-from-psi).

The inferred plasma current ``c_plasma = B·θ`` is restricted to the locked
dimensionless polynomial profile basis ``B``
(:func:`imas_ambix.gs.residual.plasma_poly_basis`), so the latent carries only
the low-dimensional amplitudes ``θ``.  Everything downstream of ``θ`` is a
constant matmul, hence trivially differentiable — the GS residual and the
topology readout both back-propagate to the latent.

Units are raw SI (Wb for flux loops, T for B-probes, Wb for ψ, A for currents),
inherited from the forward operator; μ₀ is carried explicitly there.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn

from imas_ambix.gs import operator as op
from imas_ambix.gs.residual import plasma_poly_basis

if TYPE_CHECKING:
    from imas_ambix.gs.geometry import GeometryTable


def greens_psi_matrix(
    grid_r: np.ndarray,
    grid_z: np.ndarray,
    src_r: np.ndarray,
    src_z: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Poloidal-flux Green's matrix ``M[g, s] = Φ(grid_g; src_s)`` [Wb per A].

    ``weights`` (optional, one per source) pre-scales each source column — used
    to fold a coil's per-filament ``xmult`` into a single column.  Returns an
    ``(n_grid, n_src)`` matrix so ``ψ_grid = M @ amplitudes``.
    """
    gr = np.asarray(grid_r, dtype=np.float64)
    gz = np.asarray(grid_z, dtype=np.float64)
    sr = np.asarray(src_r, dtype=np.float64)
    sz = np.asarray(src_z, dtype=np.float64)
    w = np.ones(sr.shape, dtype=np.float64) if weights is None else np.asarray(weights)
    cols = [
        w[j] * op.greens_psi(gr, gz, float(sr[j]), float(sz[j])) for j in range(sr.size)
    ]
    if not cols:
        return np.zeros((gr.size, 0), dtype=np.float64)
    return np.column_stack(cols)


def _avoid_source_singularities(
    grid_r: np.ndarray, grid_z: np.ndarray, sources_rz: np.ndarray, min_distance: float
) -> tuple[np.ndarray, np.ndarray]:
    """Nudge any grid point within ``min_distance`` of a source radially outward.

    A point-filament Green's function diverges at zero source-field distance;
    a naive grid can land arbitrarily close to a real coil or plasma-basis
    node (confirmed on real MAST geometry — central-solenoid coils sit within
    1-4 cm of a modest-resolution reconstruction grid), producing a spurious
    near-singular ψ spike.  Each offending point is pushed to exactly
    ``min_distance`` from its nearest source, along the line joining them (a
    point exactly on a source is nudged along +R to break the degeneracy).
    """
    if sources_rz.size == 0:
        return grid_r, grid_z
    gr = np.array(grid_r, dtype=np.float64, copy=True)
    gz = np.array(grid_z, dtype=np.float64, copy=True)
    for sr, sz in sources_rz:
        dr = gr - sr
        dz = gz - sz
        dist = np.hypot(dr, dz)
        close = dist < min_distance
        if not close.any():
            continue
        # degenerate (exactly on the source): nudge along +R
        deg = close & (dist < 1e-12)
        dr = np.where(deg, 1.0, dr)
        dz = np.where(deg, 0.0, dz)
        dist = np.where(deg, 1.0, dist)
        scale = np.where(close, min_distance / dist, 1.0)
        gr = np.where(close, sr + dr * scale, gr)
        gz = np.where(close, sz + dz * scale, gz)
    return gr, gz


def _pf_psi_columns(
    table: GeometryTable,
    pf_amc_channels: list[str],
    pf_merged_circuits: list[list[int]],
    grid_r: np.ndarray,
    grid_z: np.ndarray,
) -> np.ndarray:
    """ψ Green's columns for the KNOWN PF coils, assembled exactly as ``g_pf``.

    Mirrors :func:`imas_ambix.gs.operator.build_operator`: one column per
    physical coil (per amc channel); each column is the AVERAGE over the coil's
    redundant fcoil circuits, each of which is ``Σ_filament xmult·Φ``.  This
    keeps the ψ-field PF contribution consistent with the sensor operator.
    """
    by_circ: dict[int, list] = {}
    for f in table.pf_filaments:
        by_circ.setdefault(f.circuit, []).append(f)

    def _circ_col(circ: int) -> np.ndarray:
        fs = by_circ[circ]
        fr = np.array([f.r for f in fs], dtype=np.float64)
        fz = np.array([f.z for f in fs], dtype=np.float64)
        fw = np.array([f.xmult for f in fs], dtype=np.float64)
        return greens_psi_matrix(grid_r, grid_z, fr, fz, fw).sum(axis=1)

    cols: list[np.ndarray] = []
    for circs in pf_merged_circuits:
        per_circ = [_circ_col(c) for c in circs]
        cols.append(np.mean(per_circ, axis=0))
    if not cols:
        return np.zeros((np.asarray(grid_r).size, 0), dtype=np.float64)
    return np.column_stack(cols)


class GSObservation(nn.Module):
    """Torch GS observation operator: latent amplitudes θ → magnetics + ψ-field.

    Built once per campaign geometry (all matrices are fixed device geometry).
    ``forward(theta, i_pf)`` predicts the sensor magnetics; ``psi_field`` and
    ``psi_field_2d`` reconstruct the poloidal flux on the reconstruction grid.

    Shapes (``B`` = batch, ``K`` = profile DOF, ``S`` = sensors, ``C`` = KNOWN
    coils, ``G`` = grid points):

    * ``theta``  : ``(B, K)`` — plasma-current profile amplitudes;
    * ``i_pf``   : ``(B, C)`` — KNOWN PF-coil currents [A];
    * forward → ``(B, S)`` sensor prediction [Wb / T];
    * psi_field → ``(B, G)`` flux [Wb];
    * psi_field_2d → ``(B, nz, nr)`` flux [Wb] (row = Z, col = R).
    """

    def __init__(
        self,
        *,
        sensor_channels: list[str],
        sensor_kind: list[str],
        pf_amc_channels: list[str],
        a_plasma: np.ndarray,  # (S, K) = g_plasma @ basis
        g_pf: np.ndarray,  # (S, C)
        psi_plasma_basis: np.ndarray,  # (G, K) = psi_grid_plasma @ basis
        psi_pf: np.ndarray,  # (G, C)
        grid_r: np.ndarray,  # (G,)
        grid_z: np.ndarray,  # (G,)
        grid_nr: int,
        grid_nz: int,
        basis: np.ndarray,  # (N, K)
        profile_order: int,
        plasma_rz: np.ndarray | None = None,  # (N, 2) plasma-current basis nodes
        coil_rz: np.ndarray | None = None,  # (M, 2) PF/passive filament locations
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        self.sensor_channels = list(sensor_channels)
        self.sensor_kind = list(sensor_kind)
        self.pf_amc_channels = list(pf_amc_channels)
        self.profile_order = int(profile_order)
        self.grid_nr = int(grid_nr)
        self.grid_nz = int(grid_nz)
        self.n_dof = int(a_plasma.shape[1])
        self.plasma_rz = (
            None if plasma_rz is None else np.asarray(plasma_rz, dtype=np.float64)
        )
        self.coil_rz = (
            None if coil_rz is None else np.asarray(coil_rz, dtype=np.float64)
        )

        def buf(x: np.ndarray) -> torch.Tensor:
            return torch.tensor(np.asarray(x, dtype=np.float64), dtype=dtype)

        self.register_buffer("a_plasma", buf(a_plasma))
        self.register_buffer("g_pf", buf(g_pf))
        self.register_buffer("psi_plasma_basis", buf(psi_plasma_basis))
        self.register_buffer("psi_pf", buf(psi_pf))
        self.register_buffer("grid_r", buf(grid_r))
        self.register_buffer("grid_z", buf(grid_z))
        self.register_buffer("basis", buf(basis))

    # ---- builders ----

    @classmethod
    def from_table(
        cls,
        table: GeometryTable,
        *,
        grid_nr: int = 65,
        grid_nz: int = 129,
        profile_order: int = 1,
        grid_pad: float = 0.0,
        min_source_distance: float = 0.02,
        dtype: torch.dtype = torch.float64,
    ) -> GSObservation:
        """Build from a campaign :class:`GeometryTable`.

        The ψ reconstruction grid spans the limiter bounding box (optionally
        padded by ``grid_pad`` metres each side) at ``grid_nr × grid_nz``
        resolution.  ``profile_order`` selects the dimensionless polynomial
        plasma-current basis (1→3 DOF, 2→6, 4→15).  Grid points within
        ``min_source_distance`` metres of any plasma-basis node or PF filament
        are nudged away (:func:`_avoid_source_singularities`) — real MAST
        in-vessel coils can sit within a few cm of a modest-resolution grid,
        and the point-filament Green's function diverges at zero distance.
        """
        fwd = op.build_operator(table)
        basis = plasma_poly_basis(
            fwd.plasma_rz, profile_order, fwd.r0, fwd.minor_radius
        )
        a_plasma = fwd.g_plasma @ basis  # (S, K)
        # PF + passive filament locations (MAST has IN-VESSEL PF coils, so these
        # sit inside the limiter and produce ψ O-points that are not the plasma).
        coil_rz = np.array(
            [[f.r, f.z] for f in table.pf_filaments], dtype=np.float64
        ) if table.pf_filaments else np.zeros((0, 2))

        lr = np.asarray(table.limiter_r, dtype=np.float64)
        lz = np.asarray(table.limiter_z, dtype=np.float64)
        if lr.size >= 3:
            r_lo, r_hi = float(lr.min()) - grid_pad, float(lr.max()) + grid_pad
            z_lo, z_hi = float(lz.min()) - grid_pad, float(lz.max()) + grid_pad
        else:
            r_lo, r_hi, z_lo, z_hi = 0.1, 2.0, -1.5, 1.5
        r_lo = max(r_lo, 1e-3)  # R>0 for the axisymmetric Green's function
        rg = np.linspace(r_lo, r_hi, grid_nr)
        zg = np.linspace(z_lo, z_hi, grid_nz)
        mesh_r, mesh_z = np.meshgrid(rg, zg)  # (nz, nr)
        grid_r = mesh_r.ravel()
        grid_z = mesh_z.ravel()

        all_sources = np.concatenate([fwd.plasma_rz, coil_rz], axis=0)
        if all_sources.size:
            grid_r, grid_z = _avoid_source_singularities(
                grid_r, grid_z, all_sources, min_source_distance
            )

        psi_grid_plasma = greens_psi_matrix(
            grid_r, grid_z, fwd.plasma_rz[:, 0], fwd.plasma_rz[:, 1]
        )  # (G, N)
        psi_plasma_basis = psi_grid_plasma @ basis  # (G, K)
        psi_pf = _pf_psi_columns(
            table,
            fwd.pf_amc_channels,
            fwd.pf_merged_circuits,
            grid_r,
            grid_z,
        )  # (G, C)

        return cls(
            sensor_channels=fwd.sensor_channels,
            sensor_kind=fwd.sensor_kind,
            pf_amc_channels=fwd.pf_amc_channels,
            a_plasma=a_plasma,
            g_pf=fwd.g_pf,
            psi_plasma_basis=psi_plasma_basis,
            psi_pf=psi_pf,
            grid_r=grid_r,
            grid_z=grid_z,
            grid_nr=grid_nr,
            grid_nz=grid_nz,
            basis=basis,
            profile_order=profile_order,
            plasma_rz=fwd.plasma_rz,
            coil_rz=coil_rz,
            dtype=dtype,
        )

    # ---- forward maps ----

    def forward(self, theta: torch.Tensor, i_pf: torch.Tensor) -> torch.Tensor:
        """Predicted sensor magnetics ``(B, S)`` [Wb for flux loops, T for probes]."""
        a = self.a_plasma.to(theta.dtype)
        gpf = self.g_pf.to(theta.dtype)
        pred = theta @ a.T
        if gpf.shape[1]:
            pred = pred + i_pf.to(theta.dtype) @ gpf.T
        return pred

    def psi_field(self, theta: torch.Tensor, i_pf: torch.Tensor) -> torch.Tensor:
        """Reconstructed poloidal flux ``(B, G)`` [Wb] on the flattened grid."""
        pb = self.psi_plasma_basis.to(theta.dtype)
        ppf = self.psi_pf.to(theta.dtype)
        psi = theta @ pb.T
        if ppf.shape[1]:
            psi = psi + i_pf.to(theta.dtype) @ ppf.T
        return psi

    def psi_field_2d(self, theta: torch.Tensor, i_pf: torch.Tensor) -> torch.Tensor:
        """Reconstructed poloidal flux ``(B, nz, nr)`` [Wb] (row = Z, col = R)."""
        psi = self.psi_field(theta, i_pf)
        return psi.reshape(psi.shape[0], self.grid_nz, self.grid_nr)

    def c_plasma(self, theta: torch.Tensor) -> torch.Tensor:
        """Node-space plasma current amplitudes ``c_plasma = B·θ`` ``(B, N)`` [A]."""
        return theta @ self.basis.to(theta.dtype).T

    @property
    def grid_r_1d(self) -> torch.Tensor:
        """Unique R coordinates of the reconstruction grid ``(nr,)``."""
        return self.grid_r.reshape(self.grid_nz, self.grid_nr)[0]

    @property
    def grid_z_1d(self) -> torch.Tensor:
        """Unique Z coordinates of the reconstruction grid ``(nz,)``."""
        return self.grid_z.reshape(self.grid_nz, self.grid_nr)[:, 0]

    def plasma_bbox(
        self, pad: float = 0.0
    ) -> tuple[float, float, float, float] | None:
        """(r_lo, r_hi, z_lo, z_hi) bounding the plasma-current basis nodes.

        The magnetic axis / X-points must lie within the region where plasma
        current is parameterised; passed as ``search_bbox`` to the topology read
        so PF-coil O-points (outside the plasma) are not picked as the axis.
        """
        if self.plasma_rz is None or self.plasma_rz.size == 0:
            return None
        r, z = self.plasma_rz[:, 0], self.plasma_rz[:, 1]
        return (
            float(r.min()) - pad,
            float(r.max()) + pad,
            float(z.min()) - pad,
            float(z.max()) + pad,
        )
