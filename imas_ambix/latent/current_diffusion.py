"""1D flux-surface-averaged resistive current diffusion — the temporal prior.

The per-slice classical fit treats successive profile-coefficient fits as
independent; physically the poloidal flux obeys the resistive diffusion
equation, whose boundary conditions (total plasma current, surface loop
voltage) are measured and whose only unknown is the parallel resistivity
profile η(ψ_N).  This module implements that equation on the spine's own
equilibria and turns it into a soft temporal-consistency prior on the
profile-coefficient evolution:

* :func:`flux_surface_geometry` — contour-integrated 1D metrics (V′, g2, g3,
  F, ⟨B²⟩, q, Φ_tor) from one converged spine equilibrium + its fitted
  coefficients, resampled onto a uniform normalised-toroidal-flux grid
  ρ̂ = √(Φ_tor/Φ_tor,b);
* :func:`diffuse_psi` — the poloidal-flux diffusion step (torch/fp64,
  implicit θ-scheme, tridiagonal): the formulation follows the TORAX FVM ψ
  equation (reference implementation, verified against it in form; never a
  runtime dependency),

      toc·∂ψ/∂t = ∂/∂ρ̂[ (g2·g3/ρ̂)·∂ψ/∂ρ̂ ],
      toc = σ∥·μ0·16π²·Φ_b²·ρ̂/F²,

  with ψ the TOTAL poloidal flux [Wb] (the spine's Φ = 2πRA_φ convention),
  a regularity BC on axis and the measured plasma current as the edge
  gradient BC  ∂ψ/∂ρ̂|₁ = Ip·16π³·μ0·Φ_b/(D₁·F₁);
* :func:`predicted_current` — the evolved state's flux-surface-averaged
  toroidal current density  j_tot = dI/dS  and its OHMIC parallel current
  ⟨J·B⟩ = σ∥·⟨E·B⟩ with ⟨E·B⟩ = ψ̇·F·⟨1/R²⟩/(2π);
* :func:`project_coefficients` — the predicted (j_tot, ⟨J·B⟩) profiles
  projected onto the ladder basis as a non-negative least squares.  The two
  targets weight the p′/FF′ families differently (⟨J·B⟩ per unit coefficient
  is F·φ_k/R0 for the p′ family but ⟨B²⟩·R0·φ_k/F for the FF′ family), so
  the parallel Ohm's law — pinned by the measured flux evolution — carries
  split information no single-slice magnetics fit has;
* :func:`flux_budget` — the consumption ledger: the surface flux swing
  dψ_bdry/dt decomposes structurally into RESISTIVE consumption dψ_axis/dt
  (Ohm's law at the axis: no flux is stored inside a zero-volume surface)
  plus INDUCTIVE internal storage d(ψ_bdry − ψ_axis)/dt, and the Ejima
  coefficient C_E = ΔΨ_res/(μ0·R0·ΔIp) falls out as a per-shot byproduct.

η(ψ_N) is an explicit bounded low-DOF parametric profile
(:class:`EtaProfile` — a Sauter/Spitzer-informed monotone family standing in
for the T_e^{-3/2} shape while Thomson T_e is gated), fitted CROSS-SHOT,
never per-slice.  All inputs are raw measurements or the spine's own fits;
EFIT enters nowhere.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy import ndimage  # type: ignore[import-untyped]

from imas_ambix.latent.gs_solve import (
    MU0,
    EquilibriumGrid,
    _read_axis,
    _read_boundary_psi,
    profile_basis,
)
from imas_ambix.latent.profile_regularization import f_from_ffprime

__all__ = [
    "EtaProfile",
    "FluxSurfaceGeometry",
    "flux_surface_geometry",
    "basis_projection_images",
    "beta_p_coeff_sensitivity",
    "diffuse_psi",
    "predicted_current",
    "project_coefficients",
    "flux_budget",
    "ejima_coefficient",
    "reconstruct_profile_scales",
    "wpol_li3",
]


def wpol_li3(geo) -> float:
    """li3 = 4·Wpol/(μ0·Ip²·R0) from the 1D state (TORAX Wpol identity)."""
    dpsi = np.gradient(geo.psi_face, geo.rho_face, edge_order=2)
    vpr = np.clip(geo.vpr_face, 1e-30, None)
    bpol2 = (dpsi / (2.0 * np.pi)) ** 2 * geo.g2_face / vpr**2
    bpol2[0] = 0.0
    wpol = float(np.trapezoid(bpol2 * geo.vpr_face, geo.rho_face)) / (2.0 * MU0)
    return 4.0 * wpol / (MU0 * geo.ip_amperes**2 * geo.r0)


_TWO_PI = 2.0 * np.pi
_16PI3 = 16.0 * np.pi**3
_16PI2 = 16.0 * np.pi**2


# ---------------------------------------------------------------------------
# η(ψ_N) — the explicit low-DOF unknown
# ---------------------------------------------------------------------------


@dataclass
class EtaProfile:
    """Bounded monotone parallel-resistivity profile η(ψ_N) [Ω·m].

    ``η(ψ_N) = eta0 · exp(contrast · ψ_N^shape)`` — axis value ``eta0``,
    edge/axis log-contrast ``contrast`` ≥ 0, and a shape exponent.  The
    family brackets the Spitzer η ∝ T_e^{-3/2} form for peaked-to-broad
    temperature profiles while staying finite at the separatrix (where the
    T_e-derived form diverges).  Cross-shot parameters; never per-slice.
    """

    eta0: float = 5.0e-8
    contrast: float = 2.0
    shape: float = 2.0

    BOUNDS = ((1.0e-9, 3.0e-6), (0.0, 8.0), (0.3, 4.0))

    def __call__(self, psi_n: np.ndarray) -> np.ndarray:
        pn = np.clip(np.asarray(psi_n, dtype=np.float64), 0.0, 1.0)
        return self.eta0 * np.exp(self.contrast * pn**self.shape)

    def as_vector(self) -> np.ndarray:
        return np.array(
            [np.log10(self.eta0), self.contrast, self.shape], dtype=np.float64
        )

    @classmethod
    def from_vector(cls, x: np.ndarray) -> EtaProfile:
        lo, hi = zip(*cls.BOUNDS, strict=True)
        e0 = float(np.clip(10.0 ** float(x[0]), lo[0], hi[0]))
        return cls(
            eta0=e0,
            contrast=float(np.clip(x[1], lo[1], hi[1])),
            shape=float(np.clip(x[2], lo[2], hi[2])),
        )


# ---------------------------------------------------------------------------
# flux-surface geometry from one spine equilibrium
# ---------------------------------------------------------------------------


@dataclass
class FluxSurfaceGeometry:
    """1D metrics of one equilibrium on a uniform ρ̂ grid (n faces = n_rho+1).

    Face arrays sit on ρ̂ ∈ [0, 1] inclusive; cell arrays midway.  ``psi_face``
    is the TOTAL poloidal flux [Wb] on the faces (the initial condition of the
    diffusion step); ``phi_b`` the boundary toroidal flux [Wb].  ``psi_n_face``
    maps ρ̂ back to the equilibrium's normalised poloidal flux (for η and the
    basis shapes).  Sign conventions are the equilibrium's own (MAST:
    ψ_axis > ψ_bdry, jφ > 0 for Ip > 0 in the solve).
    """

    rho_face: np.ndarray
    rho_cell: np.ndarray
    psi_face: np.ndarray  # (n_rho+1,) total poloidal flux [Wb]
    psi_n_face: np.ndarray
    psi_n_cell: np.ndarray
    vpr_face: np.ndarray  # dV/dρ̂ [m³]
    vpr_cell: np.ndarray
    g2_face: np.ndarray  # ⟨(∇V)²/R²⟩ [m²]
    g3_face: np.ndarray  # ⟨1/R²⟩ [m⁻²]
    g3_cell: np.ndarray
    f_face: np.ndarray  # F = R·B_φ [T·m]
    f_cell: np.ndarray
    b2_cell: np.ndarray  # ⟨B²⟩ [T²]
    inv_r_cell: np.ndarray  # ⟨1/R⟩ [m⁻¹]
    phi_b: float  # boundary toroidal flux [Wb]
    r0: float
    ip_amperes: float
    axis_psi: float
    boundary_psi: float
    volume: float  # plasma volume [m³]
    q_face: np.ndarray
    #: +1 when ψ increases outward (TORAX/COCOS-1), −1 on the MAST-sign
    #: equilibria (ψ_axis > ψ_bdry with jφ > 0) — normalises the Ampère
    #: identity and the Ip BC so callers always pass/read a POSITIVE Ip.
    flux_sign: float = 1.0
    #: per-column amplitude scales ŝ_k of the source fit's normalised coeffs
    #: (:func:`reconstruct_profile_scales`) — what the projection images need.
    s_k: np.ndarray | None = None

    @property
    def d_face(self) -> np.ndarray:
        """The diffusion face coefficient g2·g3/ρ̂ (regular limit 0 on axis)."""
        d = np.zeros_like(self.rho_face)
        d[1:] = self.g2_face[1:] * self.g3_face[1:] / self.rho_face[1:]
        return d

    def ip_edge_gradient(self, ip_amperes: float) -> float:
        """∂ψ/∂ρ̂ at ρ̂=1 carrying total current ``ip_amperes`` (TORAX BC).

        ``ip_amperes`` is the measured (positive-normalised) current; the
        stored ``flux_sign`` orients the gradient to the equilibrium's own
        flux convention.
        """
        return (
            self.flux_sign
            * ip_amperes
            * (_16PI3 * MU0 * self.phi_b)
            / (self.d_face[-1] * self.f_face[-1])
        )

    def enclosed_current(self, psi_face: np.ndarray) -> np.ndarray:
        """I(ρ̂) on faces from the flux gradient (the Ampère identity).

        Built from the SCHEME'S OWN stable discrete fluxes: the midpoint
        gradients (ψ_{i+1} − ψ_i)/Δ with midpoint-averaged D and F give the
        enclosed current at cell centres, averaged back onto interior faces
        (axis exactly 0; edge linearly extrapolated).  A pointwise gradient
        inversion (telescoping) is exact for the constructed initial state
        but amplifies odd-even components of evolved states into current
        oscillations (measured 6.7× RMS corruption on real geometry) — the
        midpoint read is second-order and unconditionally stable.
        """
        psi = np.asarray(psi_face, dtype=np.float64)
        drho = float(self.rho_face[1] - self.rho_face[0])
        d_mid = 0.5 * (self.d_face[:-1] + self.d_face[1:])
        f_mid = 0.5 * (self.f_face[:-1] + self.f_face[1:])
        i_mid = (
            self.flux_sign
            * d_mid
            * (np.diff(psi) / drho)
            * f_mid
            / (self.phi_b * _16PI3 * MU0)
        )
        i_face = np.empty_like(psi)
        i_face[0] = 0.0
        i_face[1:-1] = 0.5 * (i_mid[:-1] + i_mid[1:])
        i_face[-1] = 1.5 * i_mid[-1] - 0.5 * i_mid[-2]
        return i_face


def _core_mask(
    psi2d: np.ndarray,
    grid: EquilibriumGrid,
    axis: tuple[float, float],
    psi_n: np.ndarray,
) -> np.ndarray:
    """Axis-connected ψ_N < 1 component inside the limiter (the solve's rule)."""
    closed = ((psi_n < 1.0) & grid.inside_limiter.ravel()).reshape(grid.nz, grid.nr)
    labels, _ = ndimage.label(closed)
    ia = int(np.argmin(np.abs(grid.zg - axis[1])))
    ja = int(np.argmin(np.abs(grid.rg - axis[0])))
    lab = labels[ia, ja]
    return (labels == lab) if lab != 0 else closed


def reconstruct_profile_scales(
    psi2d: np.ndarray,
    grid: EquilibriumGrid,
    ip_amperes: float,
    *,
    n_p: int,
    n_f: int,
    nonneg: bool,
) -> dict:
    """Per-column amplitude scales ŝ_k [A/m²] of the fit's normalised coeffs.

    Reproduces the solve's own normalisation (L1 column currents = |Ip|;
    signed in the nonneg arm) on the fit's converged ψ so a stored coefficient
    vector maps back to physical jφ:  jφ(cell) = Σ_k c_k·images_k(cell)·ŝ_k.
    Returns axis/boundary flux, the ψ_N map, the core mask and ``s_k``.
    """
    psi_flat = np.asarray(psi2d, dtype=np.float64).ravel()
    sign = 1.0 if ip_amperes >= 0 else -1.0
    axis, axis_psi = _read_axis(psi2d, grid, sign)
    boundary_psi = _read_boundary_psi(psi2d, grid, axis_psi)
    span = boundary_psi - axis_psi
    if abs(span) < 1e-12:
        span = 1e-12
    psi_n = (psi_flat - axis_psi) / span
    core = _core_mask(psi2d, grid, axis, psi_n)

    kind = "monomial-nonneg" if nonneg else "legendre"
    images = profile_basis(psi_n, grid.flat_r, r0=grid.r0, n_p=n_p, n_f=n_f, kind=kind)
    images[~core.ravel(), :] = 0.0
    cell_area = grid.dr * grid.dz
    u = images[grid.cells, :] * cell_area
    norms = np.abs(u).sum(axis=0)
    norm_scale = ip_amperes if nonneg else abs(ip_amperes)
    s_k = np.zeros(n_p + n_f)
    ok = norms > 1e-12 * max(abs(ip_amperes), 1.0)
    s_k[ok] = norm_scale / norms[ok]
    return {
        "axis": axis,
        "axis_psi": float(axis_psi),
        "boundary_psi": float(boundary_psi),
        "psi_n": psi_n,
        "core": core,
        "s_k": s_k,
    }


_NONNEG_EXPONENTS = (0.5, 1.0, 1.5, 2.0, 3.0)


def beta_p_coeff_sensitivity(
    psi2d: np.ndarray,
    grid: EquilibriumGrid,
    ip_amperes: float,
    *,
    n_p: int,
    n_f: int,
    nonneg: bool,
) -> np.ndarray | None:
    """∂βp/∂coeffs of the fit's normalised ladder coefficients (FF′ rows 0).

    βp is LINEAR-HOMOGENEOUS in the p′-family coefficients at fixed ψ
    geometry: per unit coefficient the physical pressure gradient is
    p′_Φ,k(ψ_N) = ŝ_k·φ_k(ψ_N)/(2π·R0) (the solve's drive column
    ŝ_k·(R/R0)·φ_k equals 2πR·p′_Φ in the total-flux convention), the
    pressure integrates to p_k(ψ_N) = −span·p′-cumulative, and

        βp_k = 4/(μ0·R0·Ip²) · ∫ p_k dV .

    Combined with an EXTERNAL li estimate through the Shafranov moment
    m = βp + li/2 this row is the direct p′-family amplitude constraint —
    the split lever the magnetics alone cannot provide.  Returns None on a
    degenerate span/core (the caller skips the prior for that slice).
    """
    rec = reconstruct_profile_scales(
        psi2d, grid, ip_amperes, n_p=n_p, n_f=n_f, nonneg=nonneg
    )
    span = rec["boundary_psi"] - rec["axis_psi"]
    if abs(span) < 1e-9:
        return None
    core = rec["core"].ravel()
    if core.sum() < 8:
        return None
    psi_n = np.clip(rec["psi_n"], 0.0, 1.0)
    s_k = rec["s_k"]
    dvol = 2.0 * np.pi * grid.flat_r * grid.dr * grid.dz
    sens = np.zeros(n_p + n_f, dtype=np.float64)
    for k in range(n_p):
        if nonneg:
            e = _NONNEG_EXPONENTS[k]
            cum = (1.0 - psi_n) ** (e + 1.0) / (e + 1.0)  # ∫_{ψN}^1 (1−u)^e du
        else:
            from numpy.polynomial import legendre  # noqa: PLC0415

            u_fine = np.linspace(0.0, 1.0, 401)
            phi = legendre.legval(2.0 * u_fine - 1.0, [0.0] * k + [1.0]) * (
                1.0 - u_fine
            )
            tail = np.concatenate(
                [
                    np.cumsum((phi[::-1][:-1] + phi[::-1][1:]) * 0.5)[::-1]
                    * (u_fine[1] - u_fine[0]),
                    [0.0],
                ]
            )
            cum = np.interp(psi_n, u_fine, tail)
        p_cells = -span * (s_k[k] / (_TWO_PI * grid.r0)) * cum
        p_cells = np.where(core, p_cells, 0.0)
        sens[k] = (
            4.0
            * float(np.sum(p_cells * dvol))
            / (MU0 * grid.r0 * max(ip_amperes**2, 1e-30))
        )
    return sens


def flux_surface_geometry(
    psi2d: np.ndarray,
    grid: EquilibriumGrid,
    *,
    coeffs: np.ndarray,
    ip_amperes: float,
    n_p: int,
    n_f: int,
    nonneg: bool = True,
    b_phi0: float = 0.55,
    n_rho: int = 24,
    n_psin: int = 28,
    psin_min: float = 0.04,
    psin_max: float = 0.985,
    fsa_mode: str = "coarea",
    h_factor: float = 1.25,
) -> FluxSurfaceGeometry | None:
    """Contour-integrated 1D flux-surface metrics of one spine equilibrium.

    ``coeffs`` are the slice's fitted ladder coefficients (the solve's
    normalised convention) — the FF′ family integrates to F(ψ_N) from the
    vacuum ``F_bdry = R0·b_phi0`` inward, giving the toroidal-flux coordinate
    ρ̂ = √(Φ_tor/Φ_tor,b) and the F metric the diffusion needs.  Returns None
    when the core is too small to bin (tiny ramp-up plasmas) — the caller
    must skip the prior for that interval, never fabricate one.

    ``fsa_mode`` selects how the geometric flux-surface averages (dV/dψ_N,
    ⟨1/R²⟩, ⟨1/R⟩, ⟨|∇ψ|²/R²⟩) are computed — the only step that differs
    between modes; the F / ρ̂ / ψ assembly is shared and identical.

    * ``"coarea"`` (default): host-side coarea binning (``argsort`` +
      cumulative-sum, differenced at ψ_N levels).  Byte-unchanged when off.
    * ``"connectivity"``: the fixed-shape, contour-free, ``jit``/``vmap``-safe
      JAX kernel-coarea average on the analytic ψ
      (:func:`imas_ambix.latent.flux_surface_connectivity.flux_surface_bins`) —
      accelerator-native, with a flood-fill core (no ``scipy.ndimage.label``)
      and Gaussian-kernel surface averages (no sort).  ``h_factor`` scales the
      kernel bandwidth in units of the ψ_N level spacing.
    """
    rec = reconstruct_profile_scales(
        psi2d, grid, ip_amperes, n_p=n_p, n_f=n_f, nonneg=nonneg
    )
    axis_psi, boundary_psi = rec["axis_psi"], rec["boundary_psi"]
    span = boundary_psi - axis_psi
    s_k = rec["s_k"]
    coeffs = np.asarray(coeffs, dtype=np.float64)
    if coeffs.size != n_p + n_f:
        raise ValueError(f"coeffs size {coeffs.size} != n_p+n_f = {n_p + n_f}")

    # FF′ per unit ψ_rad (= Φ/2π) from the fitted FF′-family amplitudes:
    # jφ_f = (R0/R)·Σ c_k ŝ_k φ_k(ψ_N)  ≡  F F′_rad/(μ0 R)  ⇒
    # (F F′_rad)(ψ_N) = μ0·R0·Σ c_k ŝ_k φ_k(ψ_N)
    psin_grid = np.linspace(0.0, 1.0, 101)
    edge = 1.0 - psin_grid
    exps = (0.5, 1.0, 1.5, 2.0, 3.0)

    def _phi_shapes(pn: np.ndarray, n_k: int) -> np.ndarray:
        if nonneg:
            return np.column_stack([(1.0 - pn) ** exps[k] for k in range(n_k)])
        from numpy.polynomial import legendre  # noqa: PLC0415

        x = 2.0 * pn - 1.0
        return np.column_stack(
            [legendre.legval(x, [0.0] * k + [1.0]) * edge for k in range(n_k)]
        )

    phi_f = _phi_shapes(psin_grid, n_f)
    ffprime_rad = MU0 * grid.r0 * (phi_f @ (coeffs[n_p:] * s_k[n_p:]))
    f_of_grid = f_from_ffprime(
        psin_grid,
        ffprime_rad,
        f_boundary=grid.r0 * b_phi0,
        dpsi_dpsin=span / _TWO_PI,
    )

    # --- flux-surface metrics ⟨X⟩(ψ_N) = d(∫_{<ψ} X dV)/dV, dV/dΦ = d(∫dV)/dΦ ---
    # the surface identity q = |F|·⟨1/R²⟩·(dV/dΦ)/(2π) then replaces the traced
    # loop integral (the repo's canonical tracer, topology._axis_enclosing_ring,
    # remains the boundary/LCFS instrument — nothing here re-implements it).  The
    # geometric averages are the ONLY step that differs between fsa_mode values;
    # the core cells / order below feed the shared ψ→I initial condition.
    core_flat = rec["core"].ravel()
    cells_in = grid.cells[core_flat[grid.cells]]
    if cells_in.size < 200:  # tiny ramp-up plasma — too few cells to bin
        return None
    pn_cells = rec["psi_n"][cells_in]
    r_cells = grid.flat_r[cells_in]
    dphi_dz, dphi_dr = np.gradient(psi2d, grid.zg, grid.rg)
    grad2 = (dphi_dr**2 + dphi_dz**2).ravel()[cells_in]
    dvol = _TWO_PI * r_cells * grid.dr * grid.dz  # cell volume 2πR dA
    order = np.argsort(pn_cells)  # core-cell ψ_N order (the ψ→I initial condition)
    pn_sorted = pn_cells[order]

    if fsa_mode == "coarea":
        # host-side coarea binning: cumulative volume integrals differenced at
        # ψ_N levels.  Integrate-then-differentiate is numerically robust where
        # ring line-integrals of 1/|∇Φ| are not (bilinear |∇Φ| degrades near
        # stagnation regions — measured 36% dV/dΦ error at mid-radius on the
        # test fixture).
        cum = {
            "v": np.cumsum(dvol[order]),
            "r2": np.cumsum((dvol / r_cells**2)[order]),
            "ir": np.cumsum((dvol / r_cells)[order]),
            "g2": np.cumsum((dvol * grad2 / r_cells**2)[order]),
        }
        levels = np.linspace(psin_min, min(psin_max, float(pn_sorted[-1])), n_psin + 1)
        at = {k: np.interp(levels, pn_sorted, v, left=0.0) for k, v in cum.items()}
        dv_lvl = np.diff(at["v"])
        if np.any(dv_lvl <= 0):
            return None
        pn_s = 0.5 * (levels[:-1] + levels[1:])  # metric samples at mid-levels
        dv_dpn_s = dv_lvl / np.diff(levels)
        inv_r2_s = np.diff(at["r2"]) / dv_lvl  # ⟨1/R²⟩
        inv_r_s = np.diff(at["ir"]) / dv_lvl  # ⟨1/R⟩
        grad2_r2_s = np.diff(at["g2"]) / dv_lvl  # ⟨|∇Φ|²/R²⟩
        v_s = 0.5 * (at["v"][:-1] + at["v"][1:])  # cumulative V at mid-levels
        volume = float(dvol.sum())
    elif fsa_mode == "connectivity":
        # accelerator-native, contour-free flux-surface averages on the analytic
        # ψ: a flood-fill core (no scipy.ndimage.label) + Gaussian kernel-coarea
        # surface averages (no sort).  Fixed-shape, jit/vmap-safe — the device
        # inner-loop kernel for the batched corpus labeller.  The KDE smoothing
        # is intrinsic (no differencing of noisy cumulative sums).
        from imas_ambix.latent.flux_surface_connectivity import (  # noqa: PLC0415
            flux_surface_bins,
        )

        bins = flux_surface_bins(
            psi2d,
            grid,
            axis_psi=axis_psi,
            boundary_psi=boundary_psi,
            psin_min=psin_min,
            psin_max=psin_max,
            n_psin=n_psin,
            h_factor=h_factor,
        )
        if bins is None:
            return None
        pn_s = np.asarray(bins["pn_s"], dtype=np.float64)
        dv_dpn_s = np.asarray(bins["dv_dpn"], dtype=np.float64)
        inv_r2_s = np.asarray(bins["inv_r2"], dtype=np.float64)
        inv_r_s = np.asarray(bins["inv_r"], dtype=np.float64)
        grad2_r2_s = np.asarray(bins["grad2_r2"], dtype=np.float64)
        v_s = np.asarray(bins["v_cum"], dtype=np.float64)
        volume = float(bins["v_total"])
        if np.any(dv_dpn_s <= 0) or not np.all(np.isfinite(inv_r2_s)):
            return None
    else:
        raise ValueError(f"unknown fsa_mode {fsa_mode!r}")

    dv_dphi = dv_dpn_s / abs(span)  # |dV/dΦ_pol|
    f_s = np.interp(pn_s, psin_grid, f_of_grid)
    q_s = np.abs(f_s) * inv_r2_s * dv_dphi / _TWO_PI

    # Φ_tor(ψ_N) = ∫ q dΦ_pol  (q = dΦ_tor/dΦ_pol — same-convention fluxes)
    phi_tor_s = _cumtrapz_from_zero(pn_s, q_s * abs(span))
    # ⟨B²⟩ = ⟨B_p²⟩ + F²⟨1/R²⟩,  B_p = |∇Φ|/(2πR)
    b2_s = grad2_r2_s / (4.0 * np.pi**2) + f_s**2 * inv_r2_s

    if phi_tor_s[-1] <= 0:
        return None
    # extrapolate Φ_tor to the separatrix (``volume`` is set per fsa_mode above)
    phi_b = float(phi_tor_s[-1] + (1.0 - pn_s[-1]) * q_s[-1] * abs(span))

    rho_of_pn = np.sqrt(np.clip(phi_tor_s / phi_b, 0.0, 1.0))
    # guard monotonicity for the inverse map
    rho_of_pn = np.maximum.accumulate(rho_of_pn)

    # --- resample onto the uniform ρ̂ face grid ---
    rho_face = np.linspace(0.0, 1.0, n_rho + 1)
    rho_cell = 0.5 * (rho_face[:-1] + rho_face[1:])

    def _onto(rho: np.ndarray, vals: np.ndarray, axis_val: float, edge_val: float):
        out = np.interp(rho, rho_of_pn, vals, left=np.nan, right=edge_val)
        # inside the innermost traced surface: blend to the axis limit
        inner = rho < rho_of_pn[0]
        if inner.any():
            out[inner] = axis_val + (vals[0] - axis_val) * (
                rho[inner] / max(rho_of_pn[0], 1e-9)
            )
        return out

    pn_face = _onto(rho_face, pn_s, 0.0, 1.0)
    pn_cell = _onto(rho_cell, pn_s, 0.0, 1.0)
    f_face = _onto(rho_face, f_s, float(f_of_grid[0]), float(grid.r0 * b_phi0))
    f_cell = 0.5 * (f_face[:-1] + f_face[1:])
    g3_face = _onto(rho_face, inv_r2_s, float(inv_r2_s[0]), float(inv_r2_s[-1]))
    g3_cell = 0.5 * (g3_face[:-1] + g3_face[1:])
    inv_r_cell = np.interp(pn_cell, pn_s, inv_r_s)
    b2_cell = np.interp(pn_cell, pn_s, b2_s)
    q_face = _onto(rho_face, q_s, float(q_s[0]), float(q_s[-1]))

    # vpr = dV/dρ̂ on faces: differentiate the resampled V(ρ̂)
    v_face = _onto(rho_face, v_s, 0.0, volume)
    v_face = np.maximum.accumulate(np.nan_to_num(v_face))
    vpr_face = np.gradient(v_face, rho_face)
    vpr_face[0] = 0.0
    vpr_cell = np.diff(v_face) / np.diff(rho_face)

    # g2 = ⟨(∇V)²/R²⟩ = (dV/dΦ)²·⟨|∇Φ|²/R²⟩ resampled
    dv_dphi_rho = np.interp(pn_face, pn_s, dv_dphi)
    grad2_face = np.interp(pn_face, pn_s, grad2_r2_s)
    g2_face = dv_dphi_rho**2 * grad2_face
    g2_face[0] = 0.0

    # initial ψ(ρ̂) CONSISTENT with the fit's own current profile through the
    # Ampère identity: dψ/dρ̂ = sign·I(ρ̂)·16π³μ0Φ_b/(D·F).  Integrating the
    # binned enclosed-current profile (rather than resampling the raw ψ) makes
    # the t = 0 round trip ψ → I → j_tor exact, so the diffusion prior carries
    # NO spurious step-zero innovation from metric resampling noise.
    images_cells = profile_basis(
        pn_cells,
        r_cells,
        r0=grid.r0,
        n_p=n_p,
        n_f=n_f,
        kind="monomial-nonneg" if nonneg else "legendre",
    )
    jphi_cells_fit = images_cells @ (coeffs * s_k)
    cur_cells = jphi_cells_fit * grid.dr * grid.dz
    cum_cur = np.cumsum(cur_cells[order])
    i_face_target = np.interp(pn_face, pn_sorted, cum_cur, left=0.0)
    i_face_target[0] = 0.0
    i_face_target = np.maximum.accumulate(i_face_target)
    if abs(i_face_target[-1]) < 1e-6 * abs(ip_amperes):
        return None
    i_face_target = i_face_target * (abs(ip_amperes) / i_face_target[-1])
    flux_sign = float(np.sign(span)) if span != 0 else 1.0
    d_face_loc = np.zeros_like(rho_face)
    d_face_loc[1:] = g2_face[1:] * g3_face[1:] / rho_face[1:]
    dpsi_drho = np.zeros_like(rho_face)
    dpsi_drho[1:] = (
        flux_sign
        * i_face_target[1:]
        * (_16PI3 * MU0 * phi_b)
        / (d_face_loc[1:] * f_face[1:])
    )
    psi_face = axis_psi + np.concatenate(
        [[0.0], np.cumsum(0.5 * (dpsi_drho[1:] + dpsi_drho[:-1]) * np.diff(rho_face))]
    )

    return FluxSurfaceGeometry(
        rho_face=rho_face,
        rho_cell=rho_cell,
        psi_face=psi_face,
        psi_n_face=pn_face,
        psi_n_cell=pn_cell,
        vpr_face=vpr_face,
        vpr_cell=vpr_cell,
        g2_face=g2_face,
        g3_face=g3_face,
        g3_cell=g3_cell,
        f_face=f_face,
        f_cell=f_cell,
        b2_cell=b2_cell,
        inv_r_cell=inv_r_cell,
        phi_b=phi_b,
        r0=float(grid.r0),
        ip_amperes=float(ip_amperes),
        axis_psi=float(axis_psi),
        boundary_psi=float(boundary_psi),
        volume=volume,
        q_face=q_face,
        flux_sign=flux_sign,
        s_k=s_k,
    )


def _cumtrapz_from_zero(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Cumulative trapezoid anchored at x = 0 (integrand finite at the axis)."""
    out = np.zeros_like(y)
    out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))
    # the first traced surface already encloses ∫_0^{x0} ≈ y(0+)·x0
    return out + y[0] * x[0]


# ---------------------------------------------------------------------------
# the diffusion step (torch/fp64)
# ---------------------------------------------------------------------------


def diffuse_psi(
    geo: FluxSurfaceGeometry,
    eta: EtaProfile,
    *,
    t_grid: np.ndarray,
    ip_of_t: np.ndarray,
    psi0_face: np.ndarray | None = None,
    theta: float = 1.0,
) -> dict:
    """Integrate the ψ diffusion over ``t_grid`` with the Ip edge BC.

    ``t_grid`` (n_t,) are the sub-step times [s]; ``ip_of_t`` the measured
    plasma current at those times [A] (sign = the solve's convention).
    Frozen geometry over the interval (metrics from the starting slice —
    recorded approximation; the soft prior absorbs the residual).  Implicit
    θ-scheme (θ=1 backward Euler), tridiagonal solve per sub-step, fp64.

    Returns ``psi_face`` (n_t, n_rho+1) [Wb] plus the flux-budget traces:
    ``v_axis`` and ``v_bdry`` (dψ/dt at axis / boundary, [V]) and ``psidot``
    on cells at the final step (for the ohmic parallel current).
    """
    dt64 = torch.float64
    n_f = geo.rho_face.size
    drho = float(geo.rho_face[1] - geo.rho_face[0])
    d_face = torch.as_tensor(geo.d_face, dtype=dt64)
    sigma_cell = torch.as_tensor(1.0 / eta(geo.psi_n_cell), dtype=dt64)
    toc_cell = (
        sigma_cell
        * MU0
        * _16PI2
        * geo.phi_b**2
        * torch.as_tensor(geo.rho_cell, dtype=dt64)
        / torch.as_tensor(geo.f_cell, dtype=dt64) ** 2
    )
    # face-centred state: interpolate toc onto faces (interior faces only used)
    toc_face = torch.zeros(n_f, dtype=dt64)
    toc_face[1:-1] = 0.5 * (toc_cell[:-1] + toc_cell[1:])
    toc_face[0] = toc_cell[0]
    toc_face[-1] = toc_cell[-1]

    psi = torch.as_tensor(
        geo.psi_face if psi0_face is None else psi0_face, dtype=dt64
    ).clone()
    t = np.asarray(t_grid, dtype=np.float64)
    ip = np.asarray(ip_of_t, dtype=np.float64)
    out = torch.empty((t.size, n_f), dtype=dt64)
    out[0] = psi
    v_axis = np.zeros(t.size)
    v_bdry = np.zeros(t.size)

    # vertex-centred finite volume on the face grid (n_f points): interior
    # point i owns a cell of width drho with fluxes D_{i±1/2}·ψ′ at the
    # midpoints; the END points own HALF cells so the physical boundary flux
    # sits exactly at ρ̂ = 0 (regularity: zero) and ρ̂ = 1 (the Ip BC:
    # D_edge·grad_edge — the enclosed current is prescribed AT the boundary,
    # not half a cell outside it).
    d_mid = 0.5 * (d_face[:-1] + d_face[1:])  # D at midpoints between faces

    def _step(psi_in: torch.Tensor, dt_s: float, grad_edge: float) -> torch.Tensor:
        n = psi_in.numel()
        a = torch.zeros(n, dtype=dt64)  # sub-diagonal
        b = torch.zeros(n, dtype=dt64)  # diagonal
        c = torch.zeros(n, dtype=dt64)  # super-diagonal
        r = psi_in.clone()
        lam = dt_s / (toc_face * drho * drho)
        # interior rows i = 1..n−2: neighbours through d_mid[i−1] and d_mid[i]
        a[1:-1] = -theta * lam[1:-1] * d_mid[:-1]
        c[1:-1] = -theta * lam[1:-1] * d_mid[1:]
        b[1:-1] = 1.0 + theta * lam[1:-1] * (d_mid[:-1] + d_mid[1:])
        # axis half-cell: boundary flux 0 (regularity)
        b[0] = 1.0 + theta * 2.0 * lam[0] * d_mid[0]
        c[0] = -theta * 2.0 * lam[0] * d_mid[0]
        # edge half-cell: boundary flux = D_edge·grad_edge (the Ip BC)
        b[-1] = 1.0 + theta * 2.0 * lam[-1] * d_mid[-1]
        a[-1] = -theta * 2.0 * lam[-1] * d_mid[-1]
        r[-1] = r[-1] + 2.0 * lam[-1] * float(d_face[-1]) * grad_edge * drho
        if theta < 1.0:  # explicit part (θ-scheme)
            expl = torch.zeros(n, dtype=dt64)
            expl[1:-1] = d_mid[1:] * (psi_in[2:] - psi_in[1:-1]) - d_mid[:-1] * (
                psi_in[1:-1] - psi_in[:-2]
            )
            expl[0] = 2.0 * d_mid[0] * (psi_in[1] - psi_in[0])
            expl[-1] = -2.0 * d_mid[-1] * (psi_in[-1] - psi_in[-2])
            r = r + (1.0 - theta) * lam * expl
        # dense solve of the tridiagonal system (n ~ 25 — negligible, fp64)
        mat = torch.diag(b) + torch.diag(a[1:], -1) + torch.diag(c[:-1], 1)
        return torch.linalg.solve(mat, r)

    for k in range(1, t.size):
        dt_s = float(t[k] - t[k - 1])
        if dt_s <= 0:
            out[k] = psi
            continue
        grad_edge = geo.ip_edge_gradient(float(ip[k]))
        psi_new = _step(psi, dt_s, grad_edge)
        v_axis[k] = float((psi_new[0] - psi[0]) / dt_s)
        v_bdry[k] = float((psi_new[-1] - psi[-1]) / dt_s)
        psi = psi_new
        out[k] = psi

    psi_np = out.numpy()
    # ψ̇ on cells at the final step (backward difference) for the ohmic current
    if t.size > 1 and (t[-1] - t[-2]) > 0:
        psidot_face = (psi_np[-1] - psi_np[-2]) / (t[-1] - t[-2])
    else:
        psidot_face = np.zeros(n_f)
    return {
        "t": t,
        "psi_face": psi_np,
        "v_axis": v_axis,
        "v_bdry": v_bdry,
        "psidot_face": psidot_face,
    }


# ---------------------------------------------------------------------------
# predicted profiles + basis projection
# ---------------------------------------------------------------------------


def predicted_current(
    geo: FluxSurfaceGeometry,
    psi_face: np.ndarray,
    psidot_face: np.ndarray,
    eta: EtaProfile,
) -> dict:
    """Evolved-state current profiles on the cell grid.

    ``j_tor`` = dI/dS with dS = vpr·⟨1/R⟩·dρ̂/2π (the flux-surface-averaged
    toroidal current density, Felici Eq. 6.20); ``j_par_b`` = the OHMIC
    ⟨J·B⟩ = σ∥·⟨E·B⟩ with ⟨E·B⟩ = flux_sign·ψ̇·F·⟨1/R²⟩/2π — the
    ``flux_sign`` factor makes the dissipative channel POSITIVE for a
    positive normalised current in either flux convention (on the MAST-sign
    equilibria ψ̇ < 0 during consumption; without the factor the ohmic
    target flips sign and a non-negative projection can only reach it by
    annihilating the profile — the measured collapse mode).
    """
    i_face = geo.enclosed_current(psi_face)
    spr_cell = geo.vpr_cell * geo.inv_r_cell / _TWO_PI
    drho = np.diff(geo.rho_face)
    j_tor = np.diff(i_face) / (drho * np.clip(spr_cell, 1e-30, None))
    psidot_cell = 0.5 * (psidot_face[:-1] + psidot_face[1:])
    sigma_cell = 1.0 / eta(geo.psi_n_cell)
    e_dot_b = geo.flux_sign * psidot_cell * geo.f_cell * geo.g3_cell / _TWO_PI
    j_par_b = sigma_cell * e_dot_b
    return {"i_face": i_face, "j_tor": j_tor, "j_par_b": j_par_b}


def basis_projection_images(
    geo: FluxSurfaceGeometry,
    s_k: np.ndarray,
    *,
    n_p: int,
    n_f: int,
    nonneg: bool = True,
) -> dict:
    """Per-unit-coefficient (j_tor, ⟨J·B⟩) images on the geometry's cell grid.

    p′ family (drive R/R0):  j_tor = ŝφ_k/(R0⟨1/R⟩) ;  ⟨J·B⟩ = F·ŝφ_k/R0.
    FF′ family (drive R0/R): j_tor = ŝφ_k·R0⟨1/R²⟩/⟨1/R⟩ ;
                             ⟨J·B⟩ = ⟨B²⟩·R0·ŝφ_k/F.
    The differing F-weights are the split leverage the parallel Ohm's law
    provides; both matrices are (n_rho, n_p+n_f).
    """
    pn = geo.psi_n_cell
    exps = (0.5, 1.0, 1.5, 2.0, 3.0)
    if nonneg:
        phi_p = np.column_stack([(1.0 - pn) ** exps[k] for k in range(n_p)])
        phi_f = np.column_stack([(1.0 - pn) ** exps[k] for k in range(n_f)])
    else:
        from numpy.polynomial import legendre  # noqa: PLC0415

        x = 2.0 * np.clip(pn, 0.0, 1.0) - 1.0
        edge = 1.0 - np.clip(pn, 0.0, 1.0)
        phi_p = np.column_stack(
            [legendre.legval(x, [0.0] * k + [1.0]) * edge for k in range(n_p)]
        )
        phi_f = np.column_stack(
            [legendre.legval(x, [0.0] * k + [1.0]) * edge for k in range(n_f)]
        )
    s_p = np.asarray(s_k[:n_p], dtype=np.float64)
    s_f = np.asarray(s_k[n_p:], dtype=np.float64)
    inv_r = np.clip(geo.inv_r_cell, 1e-9, None)
    f_c = geo.f_cell
    a_tor = np.hstack(
        [
            phi_p * s_p[np.newaxis, :] / (geo.r0 * inv_r[:, np.newaxis]),
            phi_f * s_f[np.newaxis, :] * geo.r0 * (geo.g3_cell / inv_r)[:, np.newaxis],
        ]
    )
    a_par = np.hstack(
        [
            phi_p * s_p[np.newaxis, :] * (f_c / geo.r0)[:, np.newaxis],
            phi_f * s_f[np.newaxis, :] * (geo.b2_cell * geo.r0 / f_c)[:, np.newaxis],
        ]
    )
    return {"a_tor": a_tor, "a_par": a_par}


def project_coefficients(
    geo: FluxSurfaceGeometry,
    images: dict,
    j_tor_pred: np.ndarray,
    j_par_b_pred: np.ndarray,
    *,
    nonneg: bool = True,
    par_weight: float = 1.0,
    ridge: float = 1e-6,
) -> np.ndarray | None:
    """Predicted ladder coefficients from the evolved 1D profiles.

    Volume-weighted least squares over the cell grid, both current targets
    stacked (each block normalised to unit RMS target so ``par_weight`` is a
    dimensionless balance), non-negativity per the fit's arm.  Returns None
    on a degenerate solve — the caller skips the prior for that slice.
    """
    from scipy import optimize  # noqa: PLC0415

    w = np.sqrt(np.clip(geo.vpr_cell, 0.0, None))
    w = w / max(np.linalg.norm(w), 1e-30)

    def _block(a: np.ndarray, y: np.ndarray, weight: float):
        scale = max(float(np.sqrt(np.mean((w * y) ** 2))), 1e-30)
        return (np.sqrt(weight) / scale) * (a * w[:, np.newaxis]), (
            np.sqrt(weight) / scale
        ) * (y * w)

    a1, y1 = _block(images["a_tor"], np.asarray(j_tor_pred, dtype=np.float64), 1.0)
    a2, y2 = _block(
        images["a_par"], np.asarray(j_par_b_pred, dtype=np.float64), par_weight
    )
    a = np.vstack([a1, a2])
    y = np.concatenate([y1, y2])
    k = a.shape[1]
    a = np.vstack([a, np.sqrt(ridge) * np.eye(k)])
    y = np.concatenate([y, np.zeros(k)])
    try:
        if nonneg:
            res = optimize.lsq_linear(a, y, bounds=(np.zeros(k), np.full(k, np.inf)))
            x = res.x
        else:
            x, *_ = np.linalg.lstsq(a, y, rcond=None)
    except (ValueError, np.linalg.LinAlgError):
        return None
    return x if np.isfinite(x).all() else None


# ---------------------------------------------------------------------------
# flux-consumption ledger + Ejima
# ---------------------------------------------------------------------------


def flux_budget(step: dict, geo: FluxSurfaceGeometry) -> dict:
    """Inductive/resistive decomposition of the interval's flux consumption.

    Working in the equilibrium's own sign (MAST: ψ_axis > ψ_bdry; consumption
    drives both down), over the integration window:

    * ``d_psi_bdry``  — total surface flux swing  ∫ V_surf dt      [Wb]
    * ``d_psi_axis``  — RESISTIVE consumption      ∫ V_axis dt     [Wb]
      (Ohm's law at the axis: the axis loop voltage is purely resistive)
    * ``d_psi_internal`` — INDUCTIVE storage change Δ(ψ_bdry − ψ_axis) [Wb]

    The identity d_psi_bdry = d_psi_axis + d_psi_internal holds by
    construction; reporting all three keeps both consumption channels
    explicitly accounted.
    """
    psi = step["psi_face"]
    d_axis = float(psi[-1, 0] - psi[0, 0])
    d_bdry = float(psi[-1, -1] - psi[0, -1])
    return {
        "d_psi_bdry": d_bdry,
        "d_psi_axis": d_axis,
        "d_psi_internal": d_bdry - d_axis,
        "v_axis_mean": float(np.mean(step["v_axis"][1:])) if psi.shape[0] > 1 else 0.0,
        "v_bdry_mean": float(np.mean(step["v_bdry"][1:])) if psi.shape[0] > 1 else 0.0,
    }


def ejima_coefficient(d_psi_res: float, d_ip: float, r0: float) -> float:
    """Windowed Ejima coefficient C_E = |ΔΨ_res| / (μ0·R0·|ΔIp|).

    The resistive poloidal-flux consumption normalised by μ0·R0·ΔIp over the
    same window (Ejima 1982 uses breakdown→t; the chained window starts at
    the first labelled slice, so this is the INCREMENTAL coefficient over
    the covered ramp — comparable across shots and to literature values
    when the window covers most of the ramp).
    """
    denom = MU0 * abs(r0) * max(abs(d_ip), 1e-30)
    return float(abs(d_psi_res) / denom)
