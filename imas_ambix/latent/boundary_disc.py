"""Staged-disc plasma boundary read: uniform current disc + gated quadrupole.

The boundary read that replaced both the whole-vessel cell-current moment fit
(ill-conditioned: its current centroid swings ~0.5 m across fit order) and the
toroidal-harmonic interior contour (pole-fragile, and a vacuum multipole carries
no current-EXTENT information, so its interior LCFS over-sizes at any
regularisation).  Staged, each stage small and well-conditioned:

1. **Ip** — Rogowski (pinned; never fit).
2. **Current centroid** — a 2-DOF filament-position fit to the coil-subtracted
   sensor signature (B-probes AND flux loops).  Direct and robust where the
   whole-vessel LSQ centroid is not.
3. **Uniform disc** — Ip spread uniformly over a disc about the centroid whose
   radius is self-consistently sized so the disc's own push-out boundary minor
   radius is a fixed point.  The plasma elongation is supplied by the real coil
   field shaping the psi_N=1 contour around the disc — no shape DOF needed for
   a robust boundary.
4. **Quadrupole-on-residual (gated)** — the three degree-2 zero-sum moments on
   the disc, fitted to the RESIDUAL sensor signature (stage-3 signature
   subtracted).  Degree-1 is SKIPPED: the position DOFs are already fixed by
   stage 2, so dipole terms only absorb noise and re-shift the centroid
   (measured: hurts on every slice).  The stage is accepted only if it moves
   the push-out boundary by less than ``gate_shift_frac`` of the disc radius —
   a machine-agnostic, referee-free over-fit gate (sensor misfit does NOT
   separate the regimes; the boundary shift does).
5. **Boundary** — one monotone flux-offset push (``lcfs_contour``, leg-clipped)
   on the ABSOLUTELY GAUGED total flux (plasma + coil), so the boundary flux
   is recovered by the push-out itself.

Conventions preserved from the patch substrate: total poloidal flux
Phi = 2 pi R A_phi [Wb], raw SI, MAST sign psi_axis > psi_boundary.  EFIT is
never an input — the read is scored against the firewalled referee only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from imas_ambix.gs.cylinder import hybrid_greens
from imas_ambix.latent.boundary_moment import (
    MomentFitConfig,
    build_moment_basis,
    fit_moment_currents,
)
from imas_ambix.latent.topology import lcfs_contour

if TYPE_CHECKING:
    from imas_ambix.gs.geometry import GeometryTable
    from imas_ambix.latent.patch_basis import PatchBasis


@dataclass(frozen=True)
class DiscReadConfig:
    """Staged-disc read knobs (defaults = the cohort-validated configuration)."""

    filament_w: float = 0.05  # centroid-fit filament cross-section [m]
    filament_h: float = 0.05
    rad_init_frac: float = 0.9  # initial radius / limiter-bounded minor distance
    rad_tol: float = 5e-3  # self-consistent radius fixed-point tolerance [m]
    max_radius_iter: int = 8
    min_cells: int = 5  # smallest disc worth evaluating
    quad_ridge: float = 1e-3  # column-normalised ridge on the 3-DOF quad stage
    gate_shift_frac: float = 0.15  # accept quad iff boundary shift < frac * radius
    # optional passive-structure (vessel eddy) stage — OFF by default.  Fitting
    # per-slice static eddy amplitudes is under-determined (the eddy state is an
    # L/R convolution of the current history, and no per-slice gate separates
    # help from harm: measured cohort-neutral).  Opt in for studies; the
    # principled treatment is time-coupled eddy evolution across slices.
    passive_k: int = 0  # top-k whitened sensor-SVD passive modes (0 = OFF)
    passive_ridge: float = 1.0  # strong ridge -> physical (~5-10 kA) amplitudes


@dataclass
class DiscInversion:
    """Result of :func:`disc_read` (``ring`` is ``None`` if no boundary found)."""

    ring: np.ndarray | None  # (n, 2) push-out LCFS polygon [m]
    psi_tot: np.ndarray  # (nz, nr) gauged total flux [Wb] (plasma + coil)
    psi_plasma: np.ndarray  # (nz, nr) plasma-only flux [Wb]
    centroid_r: float
    centroid_z: float
    radius: float  # converged disc radius [m]
    i_cell: np.ndarray  # (n_cells,) fitted per-cell current [A]
    misfit: float  # whitened sensor misfit of the ACCEPTED stage
    quad_applied: bool  # True if the quadrupole stage passed the gate
    quad_shift_frac: float  # boundary shift of the quad stage / disc radius
    axis_psi: float  # flux maximum inside the boundary [Wb]
    boundary_psi: float  # push-out separatrix / wall flux [Wb]
    passive_applied: bool = False  # True if the opt-in passive stage was accepted
    i_passive: np.ndarray | None = None  # (P,) per-circuit eddy currents [A]


def sensor_signature_arrays(
    table: GeometryTable,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(sr, sz, sang_deg, is_flux)`` in ``table.sensor_map`` row order.

    Payload ``measured`` / ``vacuum`` rows follow this order, so any design
    matrix built from these arrays is row-aligned with the measurements.
    """
    sr = np.array([m.r for m in table.sensor_map], dtype=np.float64)
    sz = np.array([m.z for m in table.sensor_map], dtype=np.float64)
    sang = np.array(
        [0.0 if m.angle_deg is None else float(m.angle_deg) for m in table.sensor_map],
        dtype=np.float64,
    )
    is_flux = np.array([m.kind == "flux_loop" for m in table.sensor_map], dtype=bool)
    return sr, sz, sang, is_flux


def _filament_signature(
    sr: np.ndarray,
    sz: np.ndarray,
    sang_deg: np.ndarray,
    is_flux: np.ndarray,
    r0: float,
    z0: float,
    w: float,
    h: float,
) -> np.ndarray:
    """Per-ampere sensor signature of one finite-area filament at ``(r0, z0)``."""
    psi, br, bz = hybrid_greens(sr, sz, r0, z0, w, h)
    ang = np.deg2rad(sang_deg)
    return np.where(is_flux, psi, br * np.cos(ang) + bz * np.sin(ang))


def fit_current_centroid(
    payload, table: GeometryTable, basis: PatchBasis, cfg: DiscReadConfig
) -> tuple[float, float]:
    """Robust 2-DOF current-centroid fit (filament position, Ip pinned).

    Minimises the whitened residual of a single Ip-carrying filament against
    the coil-subtracted sensor signature.  Seeded from the order-1 moment fit
    (whose centroid is rough but a fine starting point).
    """
    from scipy.optimize import (
        minimize,  # noqa: PLC0415 — scipy is an optional-heavy dep
    )

    sr, sz, sang, is_flux = sensor_signature_arrays(table)
    keep = np.asarray(payload.mask, dtype=bool)
    w = np.zeros(keep.size)
    w[keep] = 1.0 / np.maximum(np.asarray(payload.scale, dtype=np.float64)[keep], 1e-12)
    b = np.nan_to_num(np.asarray(payload.measured, dtype=np.float64)) - np.nan_to_num(
        np.asarray(payload.vacuum, dtype=np.float64)
    )
    ip = float(payload.ip_amperes)

    def objective(x: np.ndarray) -> float:
        g = _filament_signature(
            sr,
            sz,
            sang,
            is_flux,
            float(x[0]),
            float(x[1]),
            cfg.filament_w,
            cfg.filament_h,
        )
        return float(np.sum((w * (ip * g - b)) ** 2))

    seed = fit_moment_currents(basis, payload, MomentFitConfig(order=1))
    res = minimize(
        objective,
        [float(seed.centroid_r), float(seed.centroid_z)],
        method="Nelder-Mead",
        options={"xatol": 1e-3, "fatol": 1e-6},
    )
    return float(res.x[0]), float(res.x[1])


def limiter_radial_extent_at_z(
    limiter_r: np.ndarray, limiter_z: np.ndarray, z0: float
) -> tuple[float, float]:
    """``(R_hfs, R_lfs)`` where the limiter polygon crosses height ``z0``."""
    lr = np.asarray(limiter_r, dtype=np.float64)
    lz = np.asarray(limiter_z, dtype=np.float64)
    crossings = []
    for i in range(len(lr)):
        za, zb = lz[i], lz[(i + 1) % len(lr)]
        ra, rb = lr[i], lr[(i + 1) % len(lr)]
        if (za - z0) * (zb - z0) <= 0.0 and za != zb:
            crossings.append(ra + (z0 - za) / (zb - za) * (rb - ra))
    if not crossings:
        return float(lr.min()), float(lr.max())
    return float(min(crossings)), float(max(crossings))


def ring_shift_rms(
    ring_a: np.ndarray | None, ring_b: np.ndarray | None, centre: tuple[float, float]
) -> float:
    """RMS radial distance [m] between two boundary rings about ``centre``.

    The over-fit gate metric: how far a candidate stage moved the boundary.
    """
    if ring_a is None or ring_b is None:
        return float("inf")
    tha = np.arctan2(ring_a[:, 1] - centre[1], ring_a[:, 0] - centre[0])
    ra = np.hypot(ring_a[:, 0] - centre[0], ring_a[:, 1] - centre[1])
    thb = np.arctan2(ring_b[:, 1] - centre[1], ring_b[:, 0] - centre[0])
    rb = np.hypot(ring_b[:, 0] - centre[0], ring_b[:, 1] - centre[1])
    order = np.argsort(thb)
    ri = np.interp(tha, thb[order], rb[order], period=2.0 * np.pi)
    return float(np.sqrt(np.mean((ri - ra) ** 2)))


def passive_coupling_matrices(
    grid, table: GeometryTable, *, circuits: list[int] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Sensor + grid couplings of every ``inferred_passive`` circuit.

    Returns ``(a_sens (S, P), g_grid (nz*nr, P))`` — per-ampere sensor
    signatures (``table.sensor_map`` row order) and grid flux columns, the
    finite-area cylinder Biot-Savart kernel throughout.  Pure geometry: build
    once per campaign table and reuse across slices.  ``circuits`` overrides
    the circuit selection (column order follows the given list) — used when
    the passive set is extended beyond the ``inferred_passive`` role, e.g.
    measured-case circuits held back as prediction targets.
    """
    from imas_ambix.gs import operator as op  # noqa: PLC0415

    sr, sz, sang, is_flux = sensor_signature_arrays(table)
    ang = np.deg2rad(sang)
    classes = op.classify_circuits(table.pf_filaments, table.amc_current_channels)
    passive_circuits = (
        list(circuits)
        if circuits is not None
        else sorted(c.circuit for c in classes if c.role == "inferred_passive")
    )
    by_circ: dict[int, list] = {}
    for f in table.pf_filaments:
        by_circ.setdefault(f.circuit, []).append(f)
    a_cols, g_cols = [], []
    for circ in passive_circuits:
        a = np.zeros(sr.size)
        g = np.zeros(grid.flat_r.size)
        for f in by_circ[circ]:
            w = max(abs(f.width), 0.01)
            h = max(abs(f.height), 0.01)
            psi_s, br_s, bz_s = hybrid_greens(sr, sz, f.r, f.z, w, h)
            a += f.xmult * np.where(
                is_flux, psi_s, br_s * np.cos(ang) + bz_s * np.sin(ang)
            )
            g += f.xmult * hybrid_greens(grid.flat_r, grid.flat_z, f.r, f.z, w, h)[0]
        a_cols.append(a)
        g_cols.append(g)
    return np.column_stack(a_cols), np.column_stack(g_cols)


def _push_boundary(psi: np.ndarray, grid, centre: tuple[float, float]):
    return lcfs_contour(
        psi,
        grid.rg,
        grid.zg,
        centre,
        clip_legs=True,
        limiter_r=grid.limiter_r,
        limiter_z=grid.limiter_z,
    )


def disc_read(
    payload,
    grid,
    table: GeometryTable,
    basis: PatchBasis,
    cfg: DiscReadConfig | None = None,
    passive: tuple[np.ndarray, np.ndarray] | None = None,
) -> DiscInversion | None:
    """The full staged-disc boundary read for one slice (see module docstring).

    ``passive`` optionally supplies precomputed
    :func:`passive_coupling_matrices` (reused across a shot's slices); when
    ``cfg.passive_k > 0`` and it is omitted, the matrices are built here.
    """
    cfg = cfg or DiscReadConfig()
    r0, z0 = fit_current_centroid(payload, table, basis, cfg)
    ip = float(payload.ip_amperes)

    cell_r = grid.flat_r[grid.cells]
    cell_z = grid.flat_z[grid.cells]
    r_hfs, r_lfs = limiter_radial_extent_at_z(
        np.asarray(grid.limiter_r), np.asarray(grid.limiter_z), z0
    )
    d_minor = min(r0 - r_hfs, r_lfs - r0)
    if d_minor <= 0.0:
        return None

    def uniform_disc(radius: float) -> np.ndarray | None:
        sel = np.hypot(cell_r - r0, cell_z - z0) < radius
        if int(sel.sum()) < cfg.min_cells:
            return None
        ic = np.zeros(grid.cells.size)
        ic[sel] = ip / float(sel.sum())
        return ic

    def boundary_of(ic: np.ndarray):
        psi = np.asarray(
            basis.psi_grid_2d_np(ic, payload.i_pf), dtype=np.float64
        ).reshape(grid.nz, grid.nr)
        return psi, _push_boundary(psi, grid, (r0, z0))

    # stage 3: self-consistent disc radius (fixed point of the boundary minor radius)
    radius = cfg.rad_init_frac * d_minor
    ic0 = ring = None
    for _ in range(cfg.max_radius_iter):
        ic0 = uniform_disc(radius)
        if ic0 is None:
            return None
        _psi, lc = boundary_of(ic0)
        if not lc.found:
            return None
        ring = lc.ring
        b_minor = float(np.hypot(ring[:, 0] - r0, ring[:, 1] - z0).mean())
        new_radius = 0.5 * radius + 0.5 * b_minor
        if abs(new_radius - radius) < cfg.rad_tol:
            radius = new_radius
            break
        radius = new_radius
    ic0 = uniform_disc(radius)
    if ic0 is None:
        return None
    psi0, lc0 = boundary_of(ic0)
    if not lc0.found:
        return None

    # whitened residual after the uniform stage
    m_sens = np.asarray(basis.m_sens.detach().cpu().numpy(), dtype=np.float64)
    keep = np.asarray(payload.mask, dtype=bool)
    w = np.zeros(keep.size)
    w[keep] = 1.0 / np.maximum(np.asarray(payload.scale, dtype=np.float64)[keep], 1e-12)
    b = np.nan_to_num(np.asarray(payload.measured, dtype=np.float64)) - np.nan_to_num(
        np.asarray(payload.vacuum, dtype=np.float64)
    )
    resid = b - m_sens @ ic0
    n_keep = max(int(keep.sum()), 1)
    misfit0 = float(np.sum((w * resid)[keep] ** 2) / n_keep)

    def ridge_solve(a_fit: np.ndarray, rhs: np.ndarray, lam: float) -> np.ndarray:
        aw = a_fit * w[:, None]
        col_norm = np.linalg.norm(aw, axis=0)
        col_norm = np.where(col_norm > 0.0, col_norm, 1.0)
        a_n = aw / col_norm
        return (
            np.linalg.solve(
                a_n.T @ a_n + lam * np.eye(a_fit.shape[1]), a_n.T @ (rhs * w)
            )
            / col_norm
        )

    # optional passive-structure (vessel eddy) stage — see DiscReadConfig note
    psi_pas = np.zeros_like(psi0)
    passive_applied = False
    i_passive = None
    base_lc, base_misfit = lc0, misfit0
    if cfg.passive_k > 0:
        a_pas, g_pas = (
            passive if passive is not None else passive_coupling_matrices(grid, table)
        )
        _u, sv, vt = np.linalg.svd(a_pas * w[:, None], full_matrices=False)
        k = min(cfg.passive_k, int(np.sum(sv > 1e-10 * max(sv[0], 1e-300))))
        if k > 0:
            modes = vt[:k].T  # (P, k) circuit-current patterns
            a_modes = a_pas @ modes
            c_pas = ridge_solve(a_modes, resid, cfg.passive_ridge)
            i_pas = modes @ c_pas
            psi_try = (g_pas @ i_pas).reshape(grid.nz, grid.nr)
            lc_p = _push_boundary(psi0 + psi_try, grid, (r0, z0))
            shift_p = (
                ring_shift_rms(lc0.ring, lc_p.ring if lc_p.found else None, (r0, z0))
                / radius
            )
            if lc_p.found and shift_p < cfg.gate_shift_frac:
                passive_applied = True
                i_passive = i_pas
                psi_pas = psi_try
                base_lc = lc_p
                resid = resid - a_modes @ c_pas
                base_misfit = float(np.sum((w * resid)[keep] ** 2) / n_keep)

    # quadrupole stage on the (passive-consistent) residual — dipole skipped
    r_cells = np.asarray(basis.r_cells.detach().cpu().numpy(), dtype=np.float64)
    z_cells = np.asarray(basis.z_cells.detach().cpu().numpy(), dtype=np.float64)
    disc_mask = np.where(np.hypot(cell_r - r0, cell_z - z0) < radius, 1.0, 0.0)
    m_basis, _labels, _scale = build_moment_basis(
        r_cells, z_cells, disc_mask, r0, order=2, z0=z0
    )
    quad_cols = m_basis[:, 3:6]  # the degree-2 zero-sum moments {u^2, uv, v^2}
    c_quad = ridge_solve(m_sens @ quad_cols, resid, cfg.quad_ridge)
    ic_q = ic0 + quad_cols @ c_quad
    psi_q_plasma, _lc_unused = boundary_of(ic_q)
    psi_q = psi_q_plasma + psi_pas
    lc_q = _push_boundary(psi_q, grid, (r0, z0))
    shift_frac = (
        ring_shift_rms(base_lc.ring, lc_q.ring if lc_q.found else None, (r0, z0))
        / radius
    )
    quad_applied = bool(lc_q.found and shift_frac < cfg.gate_shift_frac)

    if quad_applied:
        i_cell, psi_tot, lc = ic_q, psi_q, lc_q
        resid_q = resid - (m_sens @ quad_cols) @ c_quad
        misfit = float(np.sum((w * resid_q)[keep] ** 2) / n_keep)
    else:
        i_cell, psi_tot, lc, misfit = ic0, psi0 + psi_pas, base_lc, base_misfit

    psi_plasma = np.asarray(
        basis.psi_grid_2d_np(i_cell, np.zeros_like(np.asarray(payload.i_pf))),
        dtype=np.float64,
    ).reshape(grid.nz, grid.nr)

    # flux maximum inside the boundary = the read's confined-side reference
    from matplotlib.path import Path as MplPath  # noqa: PLC0415

    rr, zz = np.meshgrid(grid.rg, grid.zg)
    inside = (
        MplPath(lc.ring)
        .contains_points(np.column_stack([rr.ravel(), zz.ravel()]))
        .reshape(grid.nz, grid.nr)
    )
    axis_psi = float(np.max(np.where(inside, psi_tot, -np.inf)))

    return DiscInversion(
        ring=lc.ring,
        psi_tot=psi_tot,
        psi_plasma=psi_plasma,
        centroid_r=r0,
        centroid_z=z0,
        radius=float(radius),
        i_cell=i_cell,
        misfit=misfit,
        quad_applied=quad_applied,
        quad_shift_frac=float(shift_frac),
        axis_psi=axis_psi,
        boundary_psi=float(lc.psi_bnd),
        passive_applied=passive_applied,
        i_passive=i_passive,
    )


__all__ = [
    "DiscInversion",
    "DiscReadConfig",
    "disc_read",
    "passive_coupling_matrices",
    "fit_current_centroid",
    "limiter_radial_extent_at_z",
    "ring_shift_rms",
    "sensor_signature_arrays",
]
