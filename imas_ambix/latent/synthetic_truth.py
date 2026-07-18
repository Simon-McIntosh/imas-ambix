"""Manufactured-equilibrium generator for the identifiability harness.

Every coefficient the inverse arms try to recover — the profile split (β0, α),
the rigid-rotation centrifugal coefficient γ(ψ_N), the per-channel static
calibration (offset, gain), and the passive-conductor currents — is INJECTED
here with a known value, pushed through the SAME free-boundary forward chain
the arms invert, and emitted as a synthetic sensor payload with noise drawn at
the measured whitening floor.  Recovery is then well-posed: inject β0, ask the
arm for β0.  No EFIT enters anywhere; the only "truth" is the coefficient the
generator wrote down.

Why a self-consistent forward solve (not a prescribed field).  The identity we
are testing is *can these sensors constrain this coefficient*, so the truth
must be a genuine fixed point of the operator the arms use: same Δ* grid, same
finite-area coil/plasma Green's functions, same 2π μ0 plasma-source scaling
(:mod:`imas_ambix.latent.gs_solve`).  A prescribed analytic ψ would make
"recovery" meaningless — the arms would be inverting a field their own forward
map cannot produce.

The rotation truth (the headline coefficient).  With a rigid toroidal rotation
the pressure gains the Maschke–Perrin centrifugal factor
``exp[γ(ψ_N)·(R²−R₀²)]`` on each surface (``structure_residual`` form
``affine-r2-rotation`` derivation), so the injected toroidal current density is

    jφ ∝ [ β0·(R/R₀)·exp(γ(ψ_N)(R²−R₀²))  +  (1−β0)·(R₀/R) ]·(1−ψ_N)^α

inside the core, rescaled to the measured Ip.  The centrifugal exponent adds
exactly the R⁴ structure the ``affine-r2-rotation`` design column detects; with
γ = 0 the shape is byte-identical to :func:`gs_solve.profile_jphi_shape`.

The confinement caveat (a measured fact this generator works around).  On real
MAST coil currents our forward operator has no stable *confined* fixed point —
the Picard drifts to an outboard corner attractor even warm-started at the
EFIT axis (the coil-model error the plan attacks separately).  A manufactured
truth must be genuinely confined, so the generator drives a manufactured
symmetric outer-coil vertical field (:func:`build_confining_i_pf`) strong
enough to hold an interior O-point, warm-starts from a compact core blob, and
*verifies* confinement (interior axis, localised current) before emitting —
rejecting a non-confined solve rather than shipping a corner artifact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch
from scipy import ndimage  # type: ignore[import-untyped]

from imas_ambix.gs import operator as op
from imas_ambix.gs.cylinder import hybrid_greens
from imas_ambix.latent.gs_solve import (
    MU0,
    EquilibriumGrid,
    _read_axis,
    _read_boundary_psi,
    profile_basis,
)
from imas_ambix.latent.patch_basis import PatchBasis
from imas_ambix.latent.patch_inverse import SlicePayload

if TYPE_CHECKING:
    from collections.abc import Callable

    from imas_ambix.gs.geometry import GeometryTable

# The manufactured vertical field that reliably confines an interior O-point at
# MAST geometry and Ip ~ 0.6 MA across the whole (β0, α) grid (measured:
# symmetric P4/P5/P6 at 60 kA → axis R 0.4–1.1 m over β0 ∈ [0.25, 0.75],
# α ∈ [0.8, 2.2], all converged; 44 kA confines only the plain profile — the
# fragile ones need the deeper well).  Not a real shot's currents — a
# manufactured confining scenario, see the module note.
DEFAULT_VF_STRENGTH = 6.0e4
DEFAULT_IP_AMPERES = 6.0e5
_CONFINED_AXIS_R_MAX = 1.4  # axis R above this ⇒ the outboard corner attractor


def build_confining_i_pf(fwd: op.ForwardOperator, vf_strength: float) -> np.ndarray:
    """Manufactured symmetric outer-coil vertical field over the KNOWN coils.

    Drives every P4/P5/P6 coil column at ``-vf_strength`` (the sign that
    confines a positive-Ip plasma) and leaves the inner coils / solenoid at
    zero — the minimal manufactured field that produces a stable interior
    O-point at MAST geometry.  Returned in the ``fwd.pf_amc_channels`` /
    ``i_pf`` column order the arms consume.
    """
    channels = fwd.pf_amc_channels
    i_pf = np.zeros(len(channels), dtype=np.float64)
    for j, chan in enumerate(channels):
        if any(chan.startswith(g) for g in ("p4", "p5", "p6")):
            i_pf[j] = -float(vf_strength)
    return i_pf


def rotation_gamma(
    gamma0: float, kind: str = "peaked"
) -> Callable[[np.ndarray], np.ndarray]:
    """A sign-definite centrifugal coefficient profile γ(ψ_N) ≥ 0.

    ``"peaked"`` (default): γ(ψ_N) = γ0·(1−ψ_N) — peaks on axis, vanishes at the
    boundary (the physical shape: rotation is largest in the hot core).
    ``"flat"``: γ(ψ_N) = γ0.  ``gamma0`` carries units 1/m² so that
    ``γ·(R²−R₀²)`` is the dimensionless centrifugal exponent; γ0 ≈ 0.7 gives an
    in–out pressure ratio ~4 across a MAST minor radius.
    """
    g0 = float(gamma0)
    if kind == "flat":
        return lambda psi_n: np.full_like(np.asarray(psi_n, dtype=np.float64), g0)
    if kind == "peaked":
        return lambda psi_n: (
            g0 * (1.0 - np.clip(np.asarray(psi_n, dtype=np.float64), 0.0, 1.0))
        )
    raise ValueError(f"unknown rotation kind {kind!r} (use 'peaked' or 'flat')")


def _two_term_shape_fn(
    beta0: float,
    alpha: float,
    r0: float,
    gamma_fn: Callable[[np.ndarray], np.ndarray] | None,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """jφ(ψ_N, R) shape callback for the two-term (+ optional rotation) family."""

    def shape(psi_n: np.ndarray, r: np.ndarray) -> np.ndarray:
        psi_n = np.asarray(psi_n, dtype=np.float64)
        r = np.asarray(r, dtype=np.float64)
        inside = psi_n < 1.0
        rr = np.maximum(r, 1e-3)
        pressure = beta0 * rr / r0
        if gamma_fn is not None:
            expo = np.clip(gamma_fn(psi_n) * (rr * rr - r0 * r0), -10.0, 3.0)
            pressure = pressure * np.exp(expo)
        base = pressure + (1.0 - beta0) * r0 / rr
        out = np.zeros_like(psi_n)
        out[inside] = base[inside] * np.power(1.0 - psi_n[inside], alpha)
        return out

    return shape


def _basis_shape_fn(
    coeffs: np.ndarray, n_p: int, n_f: int, r0: float, nonneg: bool
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """jφ(ψ_N, R) shape callback for a K-coefficient :func:`profile_basis` family."""
    coeffs = np.asarray(coeffs, dtype=np.float64)
    kind = "monomial-nonneg" if nonneg else "legendre"

    def shape(psi_n: np.ndarray, r: np.ndarray) -> np.ndarray:
        images = profile_basis(psi_n, r, r0=r0, n_p=n_p, n_f=n_f, kind=kind)
        return images @ coeffs

    return shape


@dataclass
class Campaign:
    """Shared fixed geometry for one campaign — built once, reused per truth.

    Holds the machine table, forward operator, Δ* grid, patch basis (the arms'
    forward map), the passive-conductor grid-ψ / sensor columns re-mapped to the
    grid's sensor-channel order, and the measured whitening scale.  All pure
    geometry + a noise floor; no truth, no EFIT.
    """

    table: GeometryTable
    fwd: op.ForwardOperator
    grid: EquilibriumGrid
    basis: PatchBasis
    channels: list[str]
    g_sens: np.ndarray  # (S, n_cells) plasma→sensor
    m_coil: np.ndarray  # (S, C) coil→sensor (grid channel order)
    scale: np.ndarray  # (S,) measured whitening floor
    passive_psi_grid: np.ndarray  # (n_grid, n_pass) passive circuit → grid ψ
    passive_g_sens: np.ndarray  # (S, n_pass) passive circuit → sensor
    n_passive: int


def _build_passive_columns(
    table: GeometryTable, grid: EquilibriumGrid, fwd: op.ForwardOperator
) -> tuple[np.ndarray, np.ndarray]:
    """(grid-ψ, sensor) columns for every INFERRED passive circuit, grid order.

    Grid-ψ columns are the finite-area Green's flux of each passive circuit's
    filaments (identical construction to :func:`gs_solve.build_passive_sidecar`);
    sensor columns are ``fwd.g_passive`` re-mapped from the forward operator's
    channel order onto the grid's ``sensor_greens`` order by NAME (rows absent
    on this campaign zeroed) — the alignment the gate applies to real payloads.
    """
    _g, channels = grid.sensor_greens(table)
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
    psi_grid = (
        np.column_stack(psi_cols) if psi_cols else np.zeros((grid.flat_r.size, 0))
    )
    row_of = {ch: i for i, ch in enumerate(fwd.sensor_channels)}
    g_sens_pass = np.zeros((len(channels), fwd.g_passive.shape[1]))
    for i, ch in enumerate(channels):
        j = row_of.get(ch, -1)
        if j >= 0:
            g_sens_pass[i] = fwd.g_passive[j]
    if psi_grid.shape[1] != g_sens_pass.shape[1]:
        raise ValueError(
            f"passive count mismatch: grid {psi_grid.shape[1]} vs "
            f"g_passive {g_sens_pass.shape[1]}"
        )
    return psi_grid, g_sens_pass


def build_campaign(
    shot: int = 18502,
    *,
    nr: int = 65,
    nz: int = 97,
    scale: np.ndarray | None = None,
    table: GeometryTable | None = None,
) -> Campaign:
    """Assemble the shared campaign geometry + noise floor.

    ``table`` may be supplied directly (e.g. an analytic test fixture) instead
    of loading ``shot`` from disk.  ``scale`` overrides the measured whitening
    floor (the fast tests pass an explicit floor); when omitted it is read from
    the real shot's raw-magnetics std through the training whitening convention
    (:func:`imas_ambix.latent.data.robust_channel_scale`), falling back to a
    5% relative floor on the coil vacuum field if the shot cannot be loaded.
    """
    from imas_ambix.gs.geometry import build_table_for_shot

    if table is None:
        table = build_table_for_shot(int(shot))
    fwd = op.build_operator(table)
    grid = EquilibriumGrid.from_table(table, nr=nr, nz=nz)
    basis = PatchBasis.from_table(table, nr=nr, nz=nz, dtype=torch.float64)
    g_sens, channels = grid.sensor_greens(table)
    m_coil = basis.m_coil.detach().cpu().numpy().astype(np.float64)
    passive_psi_grid, passive_g_sens = _build_passive_columns(table, grid, fwd)

    if scale is None:
        scale = _measured_noise_scale(int(shot), fwd, channels)
    scale = np.asarray(scale, dtype=np.float64)

    return Campaign(
        table=table,
        fwd=fwd,
        grid=grid,
        basis=basis,
        channels=list(channels),
        g_sens=g_sens,
        m_coil=m_coil,
        scale=scale,
        passive_psi_grid=passive_psi_grid,
        passive_g_sens=passive_g_sens,
        n_passive=passive_g_sens.shape[1],
    )


def _measured_noise_scale(
    shot: int, fwd: op.ForwardOperator, channels: list[str]
) -> np.ndarray:
    """Per-channel whitening floor from the real shot's raw magnetics, grid order.

    Reproduces the gate's own scale (:func:`patch_gate_eval.shot_payloads`):
    ``robust_channel_scale(nanstd(raw_mag))`` in forward-operator channel order,
    then indexed onto the grid's ``sensor_greens`` channel order by name.  Falls
    back to a 1.0 flat floor if the shot data is unavailable (tests supply an
    explicit scale instead).
    """
    from imas_ambix.latent.data import (
        feature_schema,
        load_shot_windows,
        robust_channel_scale,
    )

    try:
        w = load_shot_windows(
            int(shot), fwd, "train", feature_schema(), with_referee=False
        )
    except Exception:  # noqa: BLE001 — any load failure → flat fallback floor
        w = None
    if w is None:
        return np.ones(len(channels), dtype=np.float64)
    scale_fwd = robust_channel_scale(np.nanstd(w.raw_mag, axis=0), fwd.sensor_channels)
    row_of = {ch: i for i, ch in enumerate(fwd.sensor_channels)}
    out = np.ones(len(channels), dtype=np.float64)
    for i, ch in enumerate(channels):
        j = row_of.get(ch, -1)
        if j >= 0 and np.isfinite(scale_fwd[j]) and scale_fwd[j] > 0:
            out[i] = float(scale_fwd[j])
    return out


@dataclass
class SyntheticTruth:
    """One manufactured equilibrium: injected coefficients + emitted payload.

    Every ``*_true`` field is a coefficient the generator wrote down; the
    payload arrays (``measured``/``vacuum``/``scale``/``mask``) are in the
    campaign's ``sensor_greens`` channel order, ready for :meth:`to_payload`.
    """

    # injected profile family
    beta0_true: float
    alpha_true: float
    coeffs_true: np.ndarray | None
    n_p: int
    n_f: int
    # injected rotation / calibration / passive perturbations
    gamma0_true: float
    gamma_kind: str
    offsets_true: np.ndarray  # (S,) injected calibration offset
    gains_true: np.ndarray  # (S,) injected calibration gain
    passive_true: np.ndarray  # (n_pass,) injected passive currents [A]
    # scenario
    i_pf: np.ndarray
    ip_amperes: float
    vf_strength: float
    seed: int
    # generated equilibrium
    cell_currents: np.ndarray
    psi: np.ndarray
    axis: tuple[float, float]
    axis_psi: float
    boundary_psi: float
    core_mask: np.ndarray
    converged: bool
    confined: bool
    residual: float
    # emitted sensor payload (campaign channel order)
    channels: list[str]
    measured: np.ndarray  # corrupted + noisy — what an arm sees
    measured_clean: np.ndarray  # pre-noise, pre-calibration
    vacuum: np.ndarray  # KNOWN-coil prediction
    scale: np.ndarray
    mask: np.ndarray
    noise: np.ndarray = field(repr=False, default=None)

    @property
    def axis_r(self) -> float:
        return float(self.axis[0])

    def to_payload(self, apply_calibration_truth: bool = False) -> SlicePayload:
        """A :class:`SlicePayload` for the inverse arms.

        ``apply_calibration_truth=True`` un-corrupts the payload with the KNOWN
        injected (offset, gain) — the "arm sees the true calibration" oracle
        control; the default hands the arm the raw corrupted measurement.
        """
        measured = self.measured.copy()
        scale = self.scale.copy()
        if apply_calibration_truth:
            measured = (measured - self.offsets_true) / self.gains_true
            scale = scale / np.abs(self.gains_true)
        return SlicePayload(
            measured=measured,
            vacuum=self.vacuum.copy(),
            mask=self.mask.copy(),
            scale=scale,
            i_pf=self.i_pf.copy(),
            ip_amperes=float(self.ip_amperes),
            shot=int(self.seed),
            t_index=0,
            time_s=0.0,
        )


def _forward_picard(
    grid: EquilibriumGrid,
    i_pf: np.ndarray,
    ip_amperes: float,
    shape_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    *,
    psi_passive_grid: np.ndarray | None,
    seed_z0: float,
    seed_width: tuple[float, float],
    relax: float,
    max_iterations: int,
    tolerance: float,
    initial_jphi: np.ndarray | None,
):
    """Analytic-add free-boundary Picard with a caller-supplied jφ shape.

    Mirrors :func:`gs_solve.solve_equilibrium` (same Δ* solve, same 2π μ0
    plasma source, same analytic-add coil field, same topology reads) but the
    profile shape is injected, so rotation / K-coefficient / any manufactured
    family can drive the truth.  Optional ``psi_passive_grid`` is the injected
    passive-conductor flux added to the field every sweep (grid-flat [Wb]).
    Returns ``(psi2d, cell_currents, jphi_full, axis, axis_psi, boundary_psi,
    core_mask, converged, residual)`` — ``jphi_full`` is the un-rescaled shape
    density on the full grid (the warm-start seed for a continuation stage).
    """
    psi_coil = grid.coil_psi(np.asarray(i_pf, dtype=np.float64))
    if psi_passive_grid is not None:
        psi_coil = psi_coil + psi_passive_grid
    sign = 1.0 if ip_amperes >= 0 else -1.0
    cell_area = grid.dr * grid.dz

    jphi = np.zeros(grid.flat_r.size)
    if initial_jphi is not None:
        jphi = np.where(
            grid.inside_limiter.ravel(),
            np.asarray(initial_jphi, dtype=np.float64).ravel(),
            0.0,
        )
        if not np.isfinite(jphi).all() or abs(jphi.sum()) < 1e-12:
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
        rhs2d = (-(2.0 * np.pi * MU0) * grid.flat_r * jphi * scale).reshape(
            grid.nz, grid.nr
        )
        psi_b2d = np.zeros((grid.nz, grid.nr))
        psi_b2d.ravel()[grid.edge_idx] = grid.g_edge @ i_cell
        psi_new = grid.solve_dirichlet(rhs2d, psi_b2d).ravel() + psi_coil

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
        shape = shape_fn(psi_n, grid.flat_r)
        jphi[core.ravel()] = shape[core.ravel()]

        if iteration > 5 and residual < tolerance:
            break

    jphi_full = jphi.copy()
    i_cell = jphi[grid.cells] * cell_area
    total = i_cell.sum()
    scale = ip_amperes / total if abs(total) > 1e-12 else 0.0
    i_cell = i_cell * scale
    return (
        psi_flat.reshape(grid.nz, grid.nr),
        i_cell,
        jphi_full,
        axis,
        axis_psi,
        boundary_psi,
        core,
        bool(residual < tolerance),
        float(residual),
    )


def confined_seed(
    campaign: Campaign,
    *,
    ip_amperes: float = DEFAULT_IP_AMPERES,
    vf_strength: float = DEFAULT_VF_STRENGTH,
    i_pf: np.ndarray | None = None,
    seed_r0: float | None = None,
    seed_width: tuple[float, float] = (0.2, 0.35),
    relax: float = 0.3,
    max_iterations: int = 200,
    tolerance: float = 3e-4,
) -> tuple[np.ndarray, float]:
    """A confined plain-two-term jφ density seed + its axis R.

    Solved once per (campaign, i_pf) and reused as ``warm_jphi`` across a
    coefficient sweep so each :func:`manufacture` call runs a single Picard
    stage instead of the continuation's two.  Returns ``(jphi_full, axis_r)``;
    ``axis_r > 1.4`` signals the confining field failed to hold a branch.
    """
    grid = campaign.grid
    if i_pf is None:
        i_pf = build_confining_i_pf(campaign.fwd, vf_strength)
    r0seed = grid.r0 if seed_r0 is None else seed_r0
    blob = np.zeros(grid.flat_r.size)
    blob[grid.cells] = np.exp(
        -(
            ((grid.flat_r[grid.cells] - r0seed) / seed_width[0]) ** 2
            + (grid.flat_z[grid.cells] / seed_width[1]) ** 2
        )
    )
    plain = _two_term_shape_fn(0.5, 1.0, grid.r0, None)
    _p, _c, jphi_full, axis, *_rest = _forward_picard(
        grid,
        np.asarray(i_pf, dtype=np.float64),
        ip_amperes,
        plain,
        psi_passive_grid=None,
        seed_z0=0.0,
        seed_width=seed_width,
        relax=relax,
        max_iterations=max_iterations,
        tolerance=tolerance,
        initial_jphi=blob,
    )
    return jphi_full, float(axis[0])


def manufacture(
    campaign: Campaign,
    *,
    beta0: float = 0.5,
    alpha: float = 1.0,
    coeffs: np.ndarray | None = None,
    n_p: int = 1,
    n_f: int = 1,
    nonneg_basis: bool = False,
    gamma0: float = 0.0,
    gamma_kind: str = "peaked",
    offsets: np.ndarray | None = None,
    gains: np.ndarray | None = None,
    passive_amplitudes: np.ndarray | None = None,
    ip_amperes: float = DEFAULT_IP_AMPERES,
    vf_strength: float = DEFAULT_VF_STRENGTH,
    i_pf: np.ndarray | None = None,
    noise: bool = True,
    seed: int = 0,
    seed_r0: float | None = None,
    seed_width: tuple[float, float] = (0.2, 0.35),
    relax: float = 0.2,
    max_iterations: int = 300,
    tolerance: float = 3e-4,
    continuation: bool = True,
    warm_jphi: np.ndarray | None = None,
) -> SyntheticTruth:
    """Manufacture one confined equilibrium and emit its synthetic payload.

    The profile family is either the two-term (``beta0``, ``alpha``) with an
    optional rotation ``gamma0`` (a K-coefficient family instead when ``coeffs``
    is given), pushed through :func:`_forward_picard` on the campaign geometry
    with a manufactured confining vertical field (``i_pf`` overrides it).  The
    injected passive currents add their flux to the field and their signal to
    the sensors; the injected per-channel (``offsets``, ``gains``) corrupt the
    emitted ``measured`` as ``gain·clean + offset`` (the inverse of the gate's
    ``measured' = (measured − offset)/gain`` convention, so a fitter recovers
    them); Gaussian noise at the campaign whitening floor is added when
    ``noise``.  ``confined`` records whether the axis is a genuine interior
    O-point (R ≤ 1.4 m) — a caller must reject non-confined truths.

    ``continuation`` (default) first solves a known-confining plain two-term
    (β0=0.5, α=1, no rotation) stage from the compact seed, then warm-starts the
    requested family from its converged current.  The target profile / rotation
    is basin-fragile from a cold seed (peaked or fast-rotating families escape
    to the outboard attractor even where the plain profile confines), so the
    continuation makes the generator robust across the coefficient grid; set it
    ``False`` to solve the target profile directly (the basin study does this to
    map which cold seeds reach the confined branch).
    """
    grid = campaign.grid
    rng = np.random.default_rng(seed)
    if i_pf is None:
        i_pf = build_confining_i_pf(campaign.fwd, vf_strength)
    i_pf = np.asarray(i_pf, dtype=np.float64)

    gamma_fn = rotation_gamma(gamma0, gamma_kind) if gamma0 != 0.0 else None
    if coeffs is not None:
        coeffs = np.asarray(coeffs, dtype=np.float64)
        shape_fn = _basis_shape_fn(coeffs, n_p, n_f, grid.r0, nonneg_basis)
    else:
        shape_fn = _two_term_shape_fn(beta0, alpha, grid.r0, gamma_fn)

    # injected passive currents → grid flux + sensor signal
    n_pass = campaign.n_passive
    passive = (
        np.zeros(n_pass)
        if passive_amplitudes is None
        else np.asarray(passive_amplitudes, dtype=np.float64)
    )
    if passive.size != n_pass:
        raise ValueError(
            f"passive_amplitudes must have length {n_pass}, got {passive.size}"
        )
    psi_passive = (
        campaign.passive_psi_grid @ passive if n_pass and passive.any() else None
    )

    r0seed = grid.r0 if seed_r0 is None else seed_r0
    blob = np.zeros(grid.flat_r.size)
    blob[grid.cells] = np.exp(
        -(
            ((grid.flat_r[grid.cells] - r0seed) / seed_width[0]) ** 2
            + ((grid.flat_z[grid.cells] - 0.0) / seed_width[1]) ** 2
        )
    )
    # profile-continuation: reach the confined branch with a plain two-term
    # profile from the cold seed, then warm-start the requested (basin-fragile)
    # family from its converged current.
    seed_jphi = blob
    if warm_jphi is not None:
        seed_jphi = np.asarray(warm_jphi, dtype=np.float64)
    elif continuation and (
        coeffs is not None or gamma_fn is not None or alpha != 1.0 or beta0 != 0.5
    ):
        plain = _two_term_shape_fn(0.5, 1.0, grid.r0, None)
        _p, _c, jphi_plain, _ax, _apsi, _bpsi, _core, _cv, _rr = _forward_picard(
            grid,
            i_pf,
            ip_amperes,
            plain,
            psi_passive_grid=psi_passive,
            seed_z0=0.0,
            seed_width=seed_width,
            relax=relax,
            max_iterations=max_iterations,
            tolerance=tolerance,
            initial_jphi=blob,
        )
        if _ax[0] <= _CONFINED_AXIS_R_MAX:
            seed_jphi = jphi_plain
    (
        psi2d,
        cell_currents,
        _jphi_full,
        axis,
        axis_psi,
        boundary_psi,
        core,
        converged,
        residual,
    ) = _forward_picard(
        grid,
        i_pf,
        ip_amperes,
        shape_fn,
        psi_passive_grid=psi_passive,
        seed_z0=0.0,
        seed_width=seed_width,
        relax=relax,
        max_iterations=max_iterations,
        tolerance=tolerance,
        initial_jphi=seed_jphi,
    )
    confined = bool(axis[0] <= _CONFINED_AXIS_R_MAX and core.sum() > 4)

    # sensors: coil vacuum + plasma + passive
    vacuum = campaign.m_coil @ i_pf
    plasma = campaign.g_sens @ cell_currents
    passive_sens = campaign.passive_g_sens @ passive if n_pass else 0.0
    measured_clean = vacuum + plasma + passive_sens

    scale = campaign.scale.copy()
    noise_vec = (
        rng.normal(0.0, 1.0, size=measured_clean.shape) * scale
        if noise
        else np.zeros_like(measured_clean)
    )

    n_ch = measured_clean.size
    off = np.zeros(n_ch) if offsets is None else np.asarray(offsets, dtype=np.float64)
    gn = np.ones(n_ch) if gains is None else np.asarray(gains, dtype=np.float64)
    measured = gn * (measured_clean + noise_vec) + off

    mask = np.isfinite(measured) & (scale > 0)

    return SyntheticTruth(
        beta0_true=float(beta0),
        alpha_true=float(alpha),
        coeffs_true=coeffs,
        n_p=int(n_p),
        n_f=int(n_f),
        gamma0_true=float(gamma0),
        gamma_kind=str(gamma_kind),
        offsets_true=off,
        gains_true=gn,
        passive_true=passive,
        i_pf=i_pf,
        ip_amperes=float(ip_amperes),
        vf_strength=float(vf_strength),
        seed=int(seed),
        cell_currents=cell_currents,
        psi=psi2d,
        axis=(float(axis[0]), float(axis[1])),
        axis_psi=float(axis_psi),
        boundary_psi=float(boundary_psi),
        core_mask=core,
        converged=converged,
        confined=confined,
        residual=residual,
        channels=list(campaign.channels),
        measured=measured,
        measured_clean=measured_clean,
        vacuum=vacuum,
        scale=scale,
        mask=mask,
        noise=noise_vec,
    )


def manufacture_shape(
    campaign: Campaign,
    shape_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    *,
    ip_amperes: float,
    i_pf: np.ndarray,
    passive_amplitudes: np.ndarray | None = None,
    noise: bool = True,
    seed: int = 0,
    warm_jphi: np.ndarray | None = None,
    seed_width: tuple[float, float] = (0.2, 0.35),
    relax: float = 0.2,
    max_iterations: int = 300,
    tolerance: float = 3e-4,
) -> SyntheticTruth:
    """Manufacture one equilibrium from an ARBITRARY jφ(ψ_N, R) shape callback.

    The dynamically-evolved profile entry point: a caller (e.g. a circuit /
    flux-diffusion chain) supplies the shape directly — including
    hollow-but-non-negative skin-current states no coefficient ladder can
    produce — and the truth is a genuine fixed point of the SAME forward
    chain the inverse arms use (:func:`_forward_picard`).
    ``passive_amplitudes`` (n_passive,) injects evolved passive-conductor
    currents [A] exactly as :func:`manufacture` does — their flux enters the
    Picard field and their signal the sensors, so a dynamically-chained truth
    carries the vessel state its own drive history induced.  Payload emission
    is otherwise identical to :func:`manufacture` at default calibration (no
    injected offsets / gains).
    """
    grid = campaign.grid
    rng = np.random.default_rng(seed)
    i_pf = np.asarray(i_pf, dtype=np.float64)
    n_pass = campaign.n_passive
    passive = (
        np.zeros(n_pass)
        if passive_amplitudes is None
        else np.asarray(passive_amplitudes, dtype=np.float64)
    )
    if passive.size != n_pass:
        raise ValueError(
            f"passive_amplitudes must have length {n_pass}, got {passive.size}"
        )
    psi_passive = (
        campaign.passive_psi_grid @ passive if n_pass and passive.any() else None
    )
    (
        psi2d,
        cell_currents,
        _jphi_full,
        axis,
        axis_psi,
        boundary_psi,
        core,
        converged,
        residual,
    ) = _forward_picard(
        grid,
        i_pf,
        float(ip_amperes),
        shape_fn,
        psi_passive_grid=psi_passive,
        seed_z0=0.0,
        seed_width=seed_width,
        relax=relax,
        max_iterations=max_iterations,
        tolerance=tolerance,
        initial_jphi=warm_jphi,
    )
    confined = bool(axis[0] <= _CONFINED_AXIS_R_MAX and core.sum() > 4)

    vacuum = campaign.m_coil @ i_pf
    plasma = campaign.g_sens @ cell_currents
    passive_sens = campaign.passive_g_sens @ passive if n_pass else 0.0
    measured_clean = vacuum + plasma + passive_sens
    scale = campaign.scale.copy()
    noise_vec = (
        rng.normal(0.0, 1.0, size=measured_clean.shape) * scale
        if noise
        else np.zeros_like(measured_clean)
    )
    measured = measured_clean + noise_vec
    n_ch = measured_clean.size
    mask = np.isfinite(measured) & (scale > 0)

    return SyntheticTruth(
        beta0_true=float("nan"),
        alpha_true=float("nan"),
        coeffs_true=None,
        n_p=0,
        n_f=0,
        gamma0_true=0.0,
        gamma_kind="peaked",
        offsets_true=np.zeros(n_ch),
        gains_true=np.ones(n_ch),
        passive_true=passive,
        i_pf=i_pf,
        ip_amperes=float(ip_amperes),
        vf_strength=0.0,
        seed=int(seed),
        cell_currents=cell_currents,
        psi=psi2d,
        axis=(float(axis[0]), float(axis[1])),
        axis_psi=float(axis_psi),
        boundary_psi=float(boundary_psi),
        core_mask=core,
        converged=converged,
        confined=confined,
        residual=residual,
        channels=list(campaign.channels),
        measured=measured,
        measured_clean=measured_clean,
        vacuum=vacuum,
        scale=scale,
        mask=mask,
        noise=noise_vec,
    )


__all__ = [
    "DEFAULT_VF_STRENGTH",
    "DEFAULT_IP_AMPERES",
    "Campaign",
    "SyntheticTruth",
    "build_campaign",
    "build_confining_i_pf",
    "confined_seed",
    "rotation_gamma",
    "manufacture",
    "manufacture_shape",
]
