"""SINDy distillation of the closed-form transition kernel + DMD/Koopman (S8-T8).

Stage-2 discovery track, geometry-INDEPENDENT.  Distils the Stage-1 RKN
*transition kernel* f_θ (``engine.trans_mean``) into a SPARSE symbolic
recurrence in the reduced (r≈3) coordinates that T7's SVD found, validated by
Dα-skill preservation through the FROZEN observation head.

Why this dissolves SINDy's hardest problem
-------------------------------------------
The RKN predict step is z_{t+1} = z_t + f_θ(z_t) with f_θ = ``trans_mean``: an
*exactly-evaluable autonomous map*.  So we never estimate a derivative from a
noisy finite difference.  We sample on-manifold z (the cached T7 trajectories)
and read the EXACT increment Δz = f_θ(z) by a forward pass.  STLSQ then fits a
sparse polynomial recurrence to (ξ, Δξ) pairs in reduced coordinates.

CRITICAL: Δz is ``trans_mean(z)`` evaluated FRESH on the cached z — NOT the
trajectory difference z_{t+1}−z_t (which folds in the Kalman update and is not
the pure transition kernel).  This module reads f_θ directly.

Two honesty cruxes (carried explicitly, never buried)
-----------------------------------------------------
(1) drift_reg confound.  The landed checkpoint trained with drift_reg=0.3,
    which penalises ‖f_θ(z)‖² on QUIESCENT steps → biases f_θ→IDENTITY there.
    This confound lives on the TRANSITION KERNEL — exactly this module's object
    (T7 separately showed the STATE SVD is full-effective-rank-3 even quiescent;
    see plan comment c-s8-t7-correction).  We therefore fit transient and
    quiescent strata SEPARATELY: a quiescent sparse map coming out ≈0 is the
    drift_reg artifact (flagged, NOT a conservation law); the transient stratum
    carries the real terms.  ‖Δz‖ per stratum is reported as a number so the
    bias is visible.

(2) Aliasing.  ama MHD modes are multi-kHz (median 3–6 kHz, p95 ~10 kHz) vs the
    1 kHz latent's 500 Hz Nyquist → eigenfrequency-to-MHD-mode matching is DEAD
    (aliased).  We report DECAY RATES (real part of the continuous eigenvalue —
    meaningful for genuinely slow modes) and explicitly DISCLAIM
    oscillation-frequency→MHD-mode matching.  Any spectral rate is reported
    sub-500 Hz only.

Skill-preservation: three-way attribution
------------------------------------------
A skill drop has two possible causes that must not be conflated:
  - true-fθ (full 16-d)         — the reference (recomputed in-process).
  - dense-reduced (r=3, NO threshold) — isolates the r=3 TRUNCATION loss
    (T7's r99=7 means dims 4–7 carry residual energy, so this is expected > 0).
  - sparse-reduced (STLSQ)      — adds the SPARSITY loss on top.
Only the dense→sparse gap is attributable to sparsity; the true→dense gap is
the dimension.  All three are simulated through the FROZEN obs head via a
runtime swap of ``model.trans_mean`` (no engine.py edit) and re-scored on the
same anchors/horizons as ``_score_horizons``.

Scope: import-only from engine.py / discovery_extract.py / filter.py; writes
to ``statespace/artifacts/``.  Zero new deps (numpy + sklearn STLSQ hand-roll +
sympy + scipy).  CPU only.

Usage
-----
    uv run python -m imas_ambix.statespace.discovery_sindy            # full run
    uv run python -m imas_ambix.statespace.discovery_sindy --quick    # smoke
"""

from __future__ import annotations

import itertools
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths — reuse the T7 scratch root + trajectory cache
# ---------------------------------------------------------------------------

_SCRATCH = Path("/work/projects/imas_gpu/mast/scratch/statespace_v0")
_TRAJ_CACHE = _SCRATCH / "discovery_trajectories_v0.npz"
_MANIFESTS = Path("/work/projects/imas_gpu/mast/manifests")
_SPLITS_MANIFEST = _MANIFESTS / "statespace_splits_dalpha_v0.json"

# Latent sampling-rate: the engine runs at 1 kHz → dt = 1 ms.  Used to convert
# discrete-time eigenvalues μ to continuous rates λ = log(μ)/dt.
_MODEL_HZ = 1000.0
_DT = 1.0 / _MODEL_HZ
_NYQUIST_HZ = _MODEL_HZ / 2.0  # 500 Hz — spectral validation ceiling (crux 2)

# Reduced dimension from T7's SVD (effective_rank_90pct = 3 on every stratum).
_R_DEFAULT = 3

# STLSQ defaults — sequentially-thresholded ridge (Brunton et al. 2016).
# The threshold is RELATIVE: a library column is dropped when its standardised
# coefficient (≈ that term's contribution to Δξ, after column-norm
# normalisation) falls below ``rel_threshold × RMS(Δξ_dim)``.  Raw-coefficient
# thresholding is meaningless here because the reduced coords are NOT unit-scale
# (ξ std ≈ 3.4 / 1.4 / 1.6) so quadratic columns have norms up to ~47 — a fixed
# absolute floor would compare apples to oranges and zero everything.
_STLSQ_REL_THRESHOLD = 0.10  # fraction of RMS(Δξ_dim) a term must contribute
# Sparsity/skill frontier — spans low (near-dense) to high (aggressive prune).
# f_θ in reduced coords turns out to be a DENSE low-order polynomial (no
# dominant-few-terms structure), so the frontier mostly shows that pruning only
# starts above ~0.75 — itself an honest finding reported in the artifact.
_STLSQ_FRONTIER = (0.10, 0.50, 1.0, 2.0, 3.0)
_STLSQ_ALPHA = 1e-3  # ridge regularisation
_STLSQ_MAX_ITER = 20
_POLY_DEGREE = 2  # library: 1, ξ_i, ξ_iξ_j


# ===========================================================================
# Polynomial feature library (zero-dep; mirrors PySINDy's PolynomialLibrary)
# ===========================================================================


def _poly_powers(n_vars: int, degree: int) -> list[tuple[int, ...]]:
    """Exponent tuples for a polynomial library up to ``degree`` (incl. constant).

    Returns the list of monomial exponent tuples, ordered constant → linear →
    quadratic …  e.g. for n_vars=2, degree=2:
        (0,0), (1,0), (0,1), (2,0), (1,1), (0,2)
    """
    powers: list[tuple[int, ...]] = []
    for total in range(degree + 1):
        # all exponent tuples summing to <= degree, grouped by total degree
        for combo in itertools.combinations_with_replacement(range(n_vars), total):
            exps = [0] * n_vars
            for c in combo:
                exps[c] += 1
            powers.append(tuple(exps))
    # de-dup while preserving order (combinations_with_replacement already unique)
    return powers


def _feature_names(powers: list[tuple[int, ...]], var: str = "xi") -> list[str]:
    """Human-readable monomial names for the exponent tuples."""
    names = []
    for exps in powers:
        if all(e == 0 for e in exps):
            names.append("1")
            continue
        terms = []
        for i, e in enumerate(exps):
            if e == 1:
                terms.append(f"{var}{i}")
            elif e > 1:
                terms.append(f"{var}{i}^{e}")
        names.append("*".join(terms))
    return names


def build_library(
    xi: np.ndarray, degree: int = _POLY_DEGREE
) -> tuple[np.ndarray, list]:
    """Polynomial feature matrix Θ(ξ) for an (n, r) reduced-coordinate array.

    Returns (Theta (n, n_features), powers) where ``powers`` is the exponent
    list so the symbolic recurrence can be rendered later.
    """
    n, r = xi.shape
    powers = _poly_powers(r, degree)
    cols = []
    for exps in powers:
        col = np.ones(n, dtype=np.float64)
        for i, e in enumerate(exps):
            if e:
                col = col * xi[:, i] ** e
        cols.append(col)
    return np.stack(cols, axis=1), powers


# ===========================================================================
# STLSQ — sequentially-thresholded least squares (the SINDy optimiser)
# ===========================================================================


def stlsq(
    theta: np.ndarray,
    dxi: np.ndarray,
    rel_threshold: float = _STLSQ_REL_THRESHOLD,
    alpha: float = _STLSQ_ALPHA,
    max_iter: int = _STLSQ_MAX_ITER,
) -> np.ndarray:
    """Sequentially-thresholded ridge regression (Brunton/Proctor/Kutz 2016).

    Solves Δξ ≈ Θ(ξ) Ξ with an L2 (ridge) penalty, then iteratively zeroes the
    least-contributing terms and refits on the survivors — the standard SINDy
    sparse-regression loop.  Zero new deps: ``np.linalg`` (ridge has a closed
    form).

    CRITICAL — column normalisation.  The library columns are NOT unit-scale
    (the reduced coords have std ≈ 3.4 / 1.4 / 1.6, so quadratic columns have
    norms up to ~47).  A raw-coefficient threshold compares apples to oranges:
    a column with norm 47 needs a tiny coefficient to contribute as much as a
    column with norm 1.5, so thresholding on |Ξ| preferentially keeps
    small-scale columns and drops the real terms regardless of contribution.
    We therefore fit on column-NORMALISED Θ, threshold on the STANDARDISED
    coefficient (≈ that term's contribution to Δξ), and rescale back to the
    physical Ξ.  The threshold is RELATIVE: a term survives only if its
    standardised |coeff| ≥ ``rel_threshold × RMS(Δξ_dim)``.

    Parameters
    ----------
    theta : (n, p) library feature matrix.
    dxi   : (n, r) exact reduced increments Δξ = f_θ projected.
    rel_threshold : fraction of RMS(Δξ_dim) a term must contribute to survive.
    alpha : ridge regularisation strength (on the normalised columns).
    max_iter : max refit iterations.

    Returns
    -------
    Xi : (p, r) sparse coefficient matrix in PHYSICAL (un-normalised) units.
    """
    p = theta.shape[1]
    r = dxi.shape[1]

    # Column-norm normalisation: Θ_n[:,k] = Θ[:,k] / ||Θ[:,k]||.  A unit-norm
    # column means its standardised coefficient is directly comparable to Δξ.
    col_norm = np.linalg.norm(theta, axis=0)
    col_norm = np.where(col_norm < 1e-30, 1.0, col_norm)
    theta_n = theta / col_norm  # (n, p) unit-norm columns

    def _ridge(th: np.ndarray, y: np.ndarray) -> np.ndarray:
        gram = th.T @ th + alpha * np.eye(th.shape[1])
        return np.linalg.solve(gram, th.T @ y)

    # Per-dimension absolute threshold = rel × RMS(Δξ_dim).  A near-zero
    # increment dim (drift_reg → identity) thus gets a near-zero threshold but
    # also a near-zero coefficient → it correctly resolves to the empty map.
    dxi_rms = np.sqrt(np.mean(dxi**2, axis=0))  # (r,)
    abs_thr = rel_threshold * np.maximum(dxi_rms, 1e-12)  # (r,)

    xi_n = _ridge(theta_n, dxi)  # (p, r) — standardised coefficients
    for _ in range(max_iter):
        small = np.abs(xi_n) < abs_thr[np.newaxis, :]
        xi_n[small] = 0.0
        changed = False
        for j in range(r):
            big = ~small[:, j]
            if not big.any():
                continue
            coeff = _ridge(theta_n[:, big], dxi[:, j : j + 1])[:, 0]
            new_col = np.zeros(p)
            new_col[big] = coeff
            if not np.allclose(new_col, xi_n[:, j]):
                changed = True
            xi_n[:, j] = new_col
        if not changed:
            break

    # Rescale standardised coefficients back to physical (un-normalised) units.
    return xi_n / col_norm[:, np.newaxis]


def _r2_score(theta: np.ndarray, dxi: np.ndarray, xi: np.ndarray) -> float:
    """In-sample R² of the closure Δξ ≈ Θ(ξ) Ξ (1.0 = perfect)."""
    pred = theta @ xi
    ss_res = float(np.sum((dxi - pred) ** 2))
    ss_tot = float(np.sum((dxi - dxi.mean(axis=0, keepdims=True)) ** 2))
    if ss_tot < 1e-30:
        return 1.0 if ss_res < 1e-30 else 0.0
    return 1.0 - ss_res / ss_tot


def render_recurrence(xi: np.ndarray, powers: list) -> list[str]:
    """Render the sparse recurrence Δξ_j = Σ_k Ξ_kj · monomial_k as strings.

    ``xi`` is the PHYSICAL (already-thresholded) coefficient matrix from
    :func:`stlsq`; dropped terms are exactly 0, so only the survivors are
    rendered.  Uses sympy to simplify each row to a clean symbolic expression.
    """
    import sympy as sp  # noqa: PLC0415

    r = xi.shape[1]
    syms = sp.symbols(f"xi0:{r}")
    exprs = []
    for j in range(r):
        expr = sp.Integer(0)
        for k, exps in enumerate(powers):
            c = xi[k, j]
            if c == 0.0:
                continue
            mono = sp.Integer(1)
            for i, e in enumerate(exps):
                if e:
                    mono = mono * syms[i] ** e
            expr = expr + sp.Float(round(float(c), 5)) * mono
        exprs.append(f"d(xi{j}) = {sp.sstr(sp.nsimplify(expr, rational=False))}")
    return exprs


# ===========================================================================
# Reduced coordinates (T7 SVD basis)
# ===========================================================================


@dataclass
class ReducedBasis:
    """SVD projection z ↦ ξ = V_rᵀ (z − z_mean) and its inverse lift.

    Built from the centered cached trajectories (the same data T7 SVD'd).  We
    use the RIGHT singular vectors V_r (the principal latent directions); the
    projection and lift are exact orthonormal-subspace operators.
    """

    z_mean: np.ndarray  # (L,)
    V_r: np.ndarray  # (L, r) — orthonormal columns (principal latent directions)
    singular_values: np.ndarray  # (L,) full spectrum (for reporting)
    r: int

    def project(self, z: np.ndarray) -> np.ndarray:
        """z (n, L) → ξ (n, r)."""
        return (z - self.z_mean) @ self.V_r

    def lift(self, xi: np.ndarray) -> np.ndarray:
        """ξ (n, r) → z (n, L) (mean-restored)."""
        return xi @ self.V_r.T + self.z_mean

    def project_increment(self, dz: np.ndarray) -> np.ndarray:
        """Δz (n, L) → Δξ (n, r) (mean-free, so no offset)."""
        return dz @ self.V_r

    def lift_increment(self, dxi: np.ndarray) -> np.ndarray:
        """Δξ (n, r) → Δz (n, L) (mean-free)."""
        return dxi @ self.V_r.T


def build_reduced_basis(z: np.ndarray, r: int = _R_DEFAULT) -> ReducedBasis:
    """Top-r right singular vectors of the centered (n, L) latent matrix.

    Matches T7's SVD convention (centre each dim; variance structure only).
    """
    z_mean = z.mean(axis=0)
    z_c = z - z_mean
    _u, s, vt = np.linalg.svd(z_c, full_matrices=False)
    v_r = vt[:r].T  # (L, r) orthonormal columns
    return ReducedBasis(z_mean=z_mean, V_r=v_r, singular_values=s, r=r)


# ===========================================================================
# Exact transition increments Δz = f_θ(z) (the autonomous map, read directly)
# ===========================================================================


@torch.no_grad()
def exact_increment(model, z: np.ndarray, device: str = "cpu") -> np.ndarray:
    """EXACT Δz = f_θ(z) = trans_mean(z), evaluated fresh on the given z.

    This is the heart of the distillation: f_θ is an exactly-evaluable
    autonomous map, so we read the increment directly — NOT a finite difference
    of the trajectory (which would fold in the Kalman update).
    """
    model.eval()
    zt = torch.from_numpy(np.ascontiguousarray(z)).float().to(device)
    incr = model.trans_mean(zt)  # (n, L) — the residual increment
    return incr.cpu().numpy().astype(np.float64)


# ===========================================================================
# The sparse / dense reduced transition (runtime swap for skill test)
# ===========================================================================


class ReducedTransition(torch.nn.Module):
    """A reduced-coordinate transition that drops in for ``model.trans_mean``.

    Computes Δz = V_r · g(V_rᵀ (z − z_mean)), where g maps reduced ξ → reduced
    Δξ.  Two modes:
      - dense:  g(ξ) = Θ(ξ) Ξ_dense   (least-squares, NO threshold) → r-truncation
      - sparse: g(ξ) = Θ(ξ) Ξ_sparse  (STLSQ) → adds sparsity on top

    Off-manifold z (far from the training cloud) still gets a well-defined Δz
    because the projection is linear; the increment simply lives in the r-dim
    subspace.  The variance path (trans_log_a / log_q) and obs head are
    UNTOUCHED — only the mean map is distilled (faithful "dynamics live in
    r dims" simulation).
    """

    def __init__(
        self,
        basis: ReducedBasis,
        xi_coeffs: np.ndarray,
        powers: list[tuple[int, ...]],
    ) -> None:
        super().__init__()
        self.r = basis.r
        self.register_buffer("z_mean", torch.from_numpy(basis.z_mean).float())
        self.register_buffer("V_r", torch.from_numpy(basis.V_r).float())  # (L, r)
        self.register_buffer("xi_coeffs", torch.from_numpy(xi_coeffs).float())  # (p,r)
        # Precompute power exponents as a (p, r) long tensor.
        self.register_buffer(
            "powers", torch.tensor(powers, dtype=torch.float32)
        )  # (p, r)

    def _library(self, xi: torch.Tensor) -> torch.Tensor:
        """Θ(ξ) for (n, r) ξ → (n, p)."""
        # xi: (n, r); powers: (p, r) → broadcast pow → product over r.
        # (n, 1, r) ** (1, p, r) → (n, p, r) → prod over last → (n, p)
        base = xi.unsqueeze(1).clamp(-1e6, 1e6)
        terms = base ** self.powers.unsqueeze(0)
        return terms.prod(dim=-1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        xi = (z - self.z_mean) @ self.V_r  # (n, r)
        theta = self._library(xi)  # (n, p)
        dxi = theta @ self.xi_coeffs  # (n, r)
        dz = dxi @ self.V_r.t()  # (n, L)
        return dz


# ===========================================================================
# DMD / Koopman + quiescent-fixed-point Jacobian (sub-500 Hz only)
# ===========================================================================


def _continuous_rates(eigvals: np.ndarray, dt: float = _DT) -> dict:
    """Discrete-time eigenvalues μ → continuous λ = log(μ)/dt.

    Returns growth/decay rate (Re λ, 1/s) and folded oscillation frequency
    (|Im λ|/2π, Hz).  CRUX 2: every true frequency aliases into [0, Nyquist],
    so the frequency is reported but oscillation→MHD-mode matching is DISCLAIMED
    in the artifact.  Decay rates (Re λ) are physically meaningful for slow modes.
    """
    rates = []
    for mu in eigvals:
        mu_c = complex(mu)
        # principal branch; a (near-)zero eigenvalue is an infinitely fast decay
        lam = complex(-np.inf, 0.0) if abs(mu_c) < 1e-12 else np.log(mu_c) / dt
        f_hz = abs(lam.imag) / (2.0 * math.pi)
        rates.append(
            {
                "mu_re": float(mu_c.real),
                "mu_im": float(mu_c.imag),
                "mu_abs": float(abs(mu_c)),
                "decay_rate_per_s": float(lam.real),
                "osc_freq_hz_folded": float(f_hz),
                "below_nyquist": bool(f_hz <= _NYQUIST_HZ),
            }
        )
    return {"rates": rates, "nyquist_hz": _NYQUIST_HZ}


def dmd_koopman(xi: np.ndarray, dxi: np.ndarray) -> dict:
    """Best-fit linear (Koopman/DMD) generator on reduced increments.

    The reduced one-step map is ξ_{t+1} = ξ_t + Δξ_t, and the best LINEAR fit is
    Δξ ≈ A ξ (least squares).  The discrete-time transition matrix is M = I + A;
    its eigenvalues give the discrete spectrum, converted to continuous rates.
    This is the linear (DMD) view of the same exact increments STLSQ fits
    nonlinearly — the discovery-method-first SECONDARY method.

    Returns the linear operator A, the eigenvalues of M = I + A, and the
    sub-500-Hz continuous-rate spectrum.
    """
    # least-squares Δξ = ξ Aᵀ → Aᵀ = (ξᵀξ)⁻¹ ξᵀ Δξ
    gram = xi.T @ xi + 1e-8 * np.eye(xi.shape[1])
    a_t = np.linalg.solve(gram, xi.T @ dxi)  # (r, r) = Aᵀ
    a_mat = a_t.T
    m_mat = np.eye(a_mat.shape[0]) + a_mat  # discrete transition I + A
    eigvals, eigvecs = np.linalg.eig(m_mat)
    spectrum = _continuous_rates(eigvals)
    return {
        "A": a_mat.tolist(),
        "M_eigenvalues_re": [float(e.real) for e in eigvals],
        "M_eigenvalues_im": [float(e.imag) for e in eigvals],
        "spectrum": spectrum,
        "eigvecs_abs": np.abs(eigvecs).tolist(),
    }


@torch.no_grad()
def _jacobian_full(model, z0: np.ndarray, device: str = "cpu") -> np.ndarray:
    """Analytic Jacobian ∂f_θ/∂z at a single latent point z0 (full L×L).

    Uses torch autograd on the exact transition map (no finite differences).
    Returns the Jacobian of the INCREMENT f_θ (so the one-step map Jacobian is
    I + J).
    """
    model.eval()
    zt = (
        torch.from_numpy(np.ascontiguousarray(z0))
        .float()
        .to(device)
        .requires_grad_(True)
    )
    with torch.enable_grad():
        jac = torch.autograd.functional.jacobian(
            lambda z: model.trans_mean(z.unsqueeze(0))[0], zt
        )
    return jac.cpu().numpy().astype(np.float64)


def fixed_point_jacobian(
    model, basis: ReducedBasis, z_mean_stratum: np.ndarray, device: str = "cpu"
) -> dict:
    """Reduced one-step-map Jacobian at the stratum mean → continuous spectrum.

    Linearises f_θ at the stratum-mean latent (the quiescent mean is the natural
    near-fixed-point), projects the increment-Jacobian J into the reduced
    subspace (V_rᵀ (I+J) V_r), and reports the continuous-rate spectrum.

    CRUX 1: for the QUIESCENT mean, drift_reg=0.3 drives f_θ→identity → J≈0 →
    reduced map ≈ I → eigenvalues μ≈1 → decay rate ≈0.  That is the drift_reg
    signature, NOT a near-conservation physics result — flagged in the artifact.
    """
    j_full = _jacobian_full(model, z_mean_stratum, device=device)  # (L, L) ∂f/∂z
    one_step = np.eye(j_full.shape[0]) + j_full  # I + J
    # project the one-step map into the reduced subspace
    m_red = basis.V_r.T @ one_step @ basis.V_r  # (r, r)
    eigvals = np.linalg.eigvals(m_red)
    spectrum = _continuous_rates(eigvals)
    return {
        "linearisation_point": "stratum_mean_latent",
        "reduced_map_eigenvalues_re": [float(e.real) for e in eigvals],
        "reduced_map_eigenvalues_im": [float(e.imag) for e in eigvals],
        "jacobian_increment_fro_norm": float(np.linalg.norm(j_full)),
        "spectrum": spectrum,
    }


# ===========================================================================
# Distillation per stratum
# ===========================================================================


@dataclass
class StratumFit:
    label: str
    n_samples: int
    dz_rms: float  # ‖Δz‖ RMS over the full latent (drift_reg visibility)
    dxi_rms: float  # ‖Δξ‖ RMS in reduced coords
    xi_coeffs: np.ndarray  # (p, r) sparse, PHYSICAL units (at rel_threshold)
    xi_coeffs_dense: np.ndarray  # (p, r) dense (no threshold)
    powers: list[tuple[int, ...]]
    rel_threshold: float  # the relative threshold used for xi_coeffs
    r2_sparse: float
    r2_dense: float
    n_active_terms: int
    recurrence: list[str] = field(default_factory=list)
    frontier: list[dict] = field(default_factory=list)  # sparsity/skill levels
    dmd: dict = field(default_factory=dict)
    jacobian: dict = field(default_factory=dict)


def distil_stratum(
    model,
    z_stratum: np.ndarray,
    basis: ReducedBasis,
    label: str,
    device: str = "cpu",
    rel_threshold: float = _STLSQ_REL_THRESHOLD,
    frontier: tuple[float, ...] = _STLSQ_FRONTIER,
    degree: int = _POLY_DEGREE,
    max_samples: int = 40000,
) -> StratumFit:
    """Distil f_θ on one stratum's latent cloud into a sparse reduced recurrence.

    Steps: subsample z (cap cost) → exact Δz = f_θ(z) → project both to reduced
    coords → build the polynomial library Θ(ξ) → STLSQ (sparse, column-normalised
    relative threshold) + dense ridge (no threshold) → sparsity/skill frontier
    over ``frontier`` thresholds → DMD/Koopman + fixed-point Jacobian.
    """
    rng = np.random.default_rng(0)
    if z_stratum.shape[0] > max_samples:
        idx = rng.choice(z_stratum.shape[0], size=max_samples, replace=False)
        z_s = z_stratum[idx]
    else:
        z_s = z_stratum

    dz = exact_increment(model, z_s, device=device)  # (n, L) EXACT
    dz_rms = float(np.sqrt(np.mean(dz**2)))

    xi = basis.project(z_s)  # (n, r)
    dxi = basis.project_increment(dz)  # (n, r)
    dxi_rms = float(np.sqrt(np.mean(dxi**2)))

    theta, powers = build_library(xi, degree=degree)
    # dense ridge with NO threshold → the r-truncation reference
    gram = theta.T @ theta + _STLSQ_ALPHA * np.eye(theta.shape[1])
    xi_dense = np.linalg.solve(gram, theta.T @ dxi)
    r2_dense = _r2_score(theta, dxi, xi_dense)

    # Sparsity/skill frontier: the same threshold-vs-fit trade reported elsewhere
    # in this plan (λ × profile-DOF).  The primary ``rel_threshold`` row is the
    # one the skill test uses.
    frontier_rows = []
    for f in sorted(set(frontier) | {rel_threshold}):
        xi_f = stlsq(theta, dxi, rel_threshold=f)
        frontier_rows.append(
            {
                "rel_threshold": float(f),
                "n_active_terms": int(np.count_nonzero(xi_f)),
                "r2": _r2_score(theta, dxi, xi_f),
            }
        )

    xi_sparse = stlsq(theta, dxi, rel_threshold=rel_threshold)
    r2_sparse = _r2_score(theta, dxi, xi_sparse)
    n_active = int(np.count_nonzero(xi_sparse))

    fit = StratumFit(
        label=label,
        n_samples=int(z_s.shape[0]),
        dz_rms=dz_rms,
        dxi_rms=dxi_rms,
        xi_coeffs=xi_sparse,
        xi_coeffs_dense=xi_dense,
        powers=powers,
        rel_threshold=rel_threshold,
        r2_sparse=r2_sparse,
        r2_dense=r2_dense,
        n_active_terms=n_active,
        frontier=frontier_rows,
    )
    fit.recurrence = render_recurrence(xi_sparse, powers)
    fit.dmd = dmd_koopman(xi, dxi)
    fit.jacobian = fixed_point_jacobian(
        model, basis, z_mean_stratum=z_s.mean(axis=0), device=device
    )
    logger.info(
        "[sindy/%s] n=%d ‖Δz‖=%.4g ‖Δξ‖=%.4g R²(sparse)=%.4f R²(dense)=%.4f active=%d",
        label,
        fit.n_samples,
        dz_rms,
        dxi_rms,
        r2_sparse,
        r2_dense,
        n_active,
    )
    return fit


# ===========================================================================
# Skill preservation through the frozen observation head (three-way)
# ===========================================================================


def _score_with_transition(
    model,
    runs: list,
    stats,
    horizons: tuple[int, ...],
    transition_module,
    label: str,
    device: str = "cpu",
    max_anchors_per_run: int = 64,
) -> dict:
    """Re-score autonomous forecasts with ``model.trans_mean`` swapped.

    Temporarily replaces the transition mean map (the variance path + frozen obs
    head are untouched), runs the SAME ``forecast_pairs`` + ``_score_horizons``
    pipeline the engine acceptance uses, restores the original, returns the
    per-horizon scores (CRPS/NLL/RMSE, overall + transient-target stratum).
    """
    from imas_ambix.statespace.baseline import compute_transient_mask  # noqa: PLC0415
    from imas_ambix.statespace.engine import _score_horizons  # noqa: PLC0415
    from imas_ambix.statespace.filter import forecast_pairs  # noqa: PLC0415

    h_sorted = sorted(horizons)
    h_max = max(horizons)
    nu_val = float(model.nu()[0].item()) if model.cfg.emission == "student_t" else None

    orig_trans = model.trans_mean
    if transition_module is not None:
        model.trans_mean = transition_module  # type: ignore[assignment]
    try:
        mus, vars_, ys, tflags = [], [], [], []
        rng = np.random.default_rng(0)
        for run in runs:
            T = run.X.shape[0]  # noqa: N806 — matches engine.py time-dim convention
            if h_max + 2 >= T:
                continue
            cand = np.arange(0, T - h_max - 1)
            if len(cand) > max_anchors_per_run:
                cand = np.sort(
                    rng.choice(cand, size=max_anchors_per_run, replace=False)
                )
            x_norm = stats.normalise_X(run.X.astype(np.float64))
            mu_n, var_n = forecast_pairs(
                model, x_norm, cand, list(horizons), device=device
            )
            if mu_n.shape[0] == 0:
                continue
            valid = [int(a) for a in cand if int(a) + h_max < T]
            mu_phys = stats.denormalise_y_mean(
                mu_n.reshape(-1, mu_n.shape[-1])
            ).reshape(mu_n.shape)
            var_phys = var_n * (stats.target_std**2)[np.newaxis, np.newaxis, :]
            D = run.y.shape[1]  # noqa: N806 — matches engine.py output-dim convention
            y_phys = np.full((len(valid), len(h_sorted), D), np.nan, dtype=np.float64)
            tflag = np.zeros((len(valid), len(h_sorted)), dtype=bool)
            tm = compute_transient_mask(run.y)
            for j, t in enumerate(valid):
                for i, h in enumerate(h_sorted):
                    if t + h < T:
                        y_phys[j, i] = run.y[t + h]
                        tflag[j, i] = bool(tm[t + h])
            mus.append(mu_phys)
            vars_.append(var_phys)
            ys.append(y_phys)
            tflags.append(tflag)
    finally:
        model.trans_mean = orig_trans  # ALWAYS restore

    if not mus:
        return {"label": label, "n": 0}
    mu = np.concatenate(mus)
    var = np.concatenate(vars_)
    y = np.concatenate(ys)
    tflag = np.concatenate(tflags)
    scores = _score_horizons(
        label, mu, var, y, horizons, transient_flag=tflag, nu=nu_val
    )
    return {"label": label, "n": int(mu.shape[0]), "per_horizon": scores}


def _mean_crps(s: dict, key: str = "crps_raw") -> float:
    """Mean over horizons of a per-horizon CRPS dict (NaN-safe)."""
    if not s or "per_horizon" not in s:
        return float("nan")
    vals = [
        s["per_horizon"][h].get(key)
        for h in s["per_horizon"]
        if isinstance(s["per_horizon"][h], dict)
    ]
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def skill_preservation(
    model,
    runs: list,
    stats,
    basis: ReducedBasis,
    transient_fit: StratumFit,
    horizons: tuple[int, ...],
    device: str = "cpu",
    max_runs: int = 40,
) -> dict:
    """Four-way skill attribution through the FROZEN observation head.

    identity (Δz≡0) → true-fθ (16-d) → dense-reduced (r, no threshold) →
    sparse-reduced (STLSQ).  Gaps:
      - true→dense   = r-TRUNCATION loss (the dimension).
      - dense→sparse = SPARSITY loss (the thresholding).
      - identity→true = how much the LEARNED dynamics buy over a frozen-belief
        rollout.  THIS IS THE DISCRIMINATION GUARD: the rule "skill-preserving
        sparsity ⇒ real dynamics" is only valid if identity scores MEANINGFULLY
        WORSE than true.  drift_reg=0.3 made f_θ tiny, so an identity rollout
        may already be nearly as good for Dα — in which case the skill metric
        CANNOT discriminate (a drift_reg confound propagating INTO validation,
        not a success).  All four re-scored on the SAME anchors/horizons.

    The transient-stratum fit's coefficients are used for the reduced maps (it
    carries the real dynamics; the quiescent fit is ≈identity by drift_reg).
    """
    runs_eval = runs[:max_runs]

    # Identity/null baseline: a ReducedTransition with ZERO coefficients → Δz≡0
    # → the autonomous rollout freezes the belief mean at the anchor (variance
    # still widens via Q, so the obs head still produces a calibrated forecast).
    null_coeffs = np.zeros_like(transient_fit.xi_coeffs)
    identity_mod = ReducedTransition(basis, null_coeffs, transient_fit.powers)
    dense_mod = ReducedTransition(
        basis, transient_fit.xi_coeffs_dense, transient_fit.powers
    )
    sparse_mod = ReducedTransition(basis, transient_fit.xi_coeffs, transient_fit.powers)

    true_scores = _score_with_transition(
        model, runs_eval, stats, horizons, None, "true_ftheta", device=device
    )
    identity_scores = _score_with_transition(
        model, runs_eval, stats, horizons, identity_mod, "identity_null", device=device
    )
    dense_scores = _score_with_transition(
        model, runs_eval, stats, horizons, dense_mod, "dense_reduced", device=device
    )
    sparse_scores = _score_with_transition(
        model, runs_eval, stats, horizons, sparse_mod, "sparse_reduced", device=device
    )

    def _crps_by_h(s: dict, key: str = "crps_raw") -> dict:
        if not s or "per_horizon" not in s:
            return {}
        return {
            h: s["per_horizon"][h].get(key)
            for h in s["per_horizon"]
            if isinstance(s["per_horizon"][h], dict)
        }

    # Discrimination guard: is identity meaningfully worse than true?
    crps_true = _mean_crps(true_scores)
    crps_identity = _mean_crps(identity_scores)
    rel_gap = (
        (crps_identity - crps_true) / crps_true
        if np.isfinite(crps_true) and crps_true > 1e-12
        else float("nan")
    )
    # 2% mean-CRPS gap is the (conservative) discrimination floor.
    discriminates = bool(np.isfinite(rel_gap) and rel_gap > 0.02)
    guard = {
        "mean_crps_true": crps_true,
        "mean_crps_identity": crps_identity,
        "identity_minus_true_rel": rel_gap,
        "metric_discriminates": discriminates,
        "interpretation": (
            "The skill metric DISCRIMINATES: identity (Δz≡0) is "
            f"{rel_gap:.1%} worse than the true kernel, so 'skill-preserving "
            "sparsity ⇒ captured real dynamics' is a valid inference here."
            if discriminates
            else (
                "WARNING — the skill metric does NOT discriminate at these "
                f"horizons: identity (Δz≡0) is only {rel_gap:.1%} worse than the "
                "true kernel. This is the drift_reg=0.3 confound propagating INTO "
                "validation (f_θ was made tiny, so a frozen-belief rollout is "
                "nearly as good for Dα). A 'skill-preserving' sparse map therefore "
                "does NOT by itself prove it captured real dynamics — the test is "
                "under-powered until the T6-grounded latent (larger f_θ) lands."
            )
        ),
    }

    delta = {
        "crps_overall": {
            "identity_null": _crps_by_h(identity_scores),
            "true": _crps_by_h(true_scores),
            "dense_reduced": _crps_by_h(dense_scores),
            "sparse_reduced": _crps_by_h(sparse_scores),
        },
        "crps_transient_target": {
            "identity_null": _crps_by_h(identity_scores, "crps_raw_transient"),
            "true": _crps_by_h(true_scores, "crps_raw_transient"),
            "dense_reduced": _crps_by_h(dense_scores, "crps_raw_transient"),
            "sparse_reduced": _crps_by_h(sparse_scores, "crps_raw_transient"),
        },
        "discrimination_guard": guard,
        "attribution": (
            "true→dense gap = r-truncation loss (the dimension); "
            "dense→sparse gap = sparsity loss (the thresholding); "
            "identity→true gap = value of the learned dynamics (the "
            "discrimination guard). Skill-preserving sparsity ⇒ the sparse map "
            "captured the real dynamics — BUT ONLY IF metric_discriminates is "
            "true; otherwise the test is under-powered (drift_reg confound)."
        ),
    }
    return {
        "n_runs_scored": len(runs_eval),
        "identity_null": identity_scores,
        "true_ftheta": true_scores,
        "dense_reduced": dense_scores,
        "sparse_reduced": sparse_scores,
        "delta": delta,
    }


# ===========================================================================
# Top-level pipeline
# ===========================================================================


def load_trajectories(path: Path = _TRAJ_CACHE) -> dict:
    """Load the T7 cached per-shot trajectories (filtered + smoothed + masks)."""
    if not path.exists():
        raise FileNotFoundError(
            f"T7 trajectory cache not found at {path}. Run discovery_extract first."
        )
    d = np.load(path)
    return {k: d[k] for k in d.files}


def run_distillation(
    r: int = _R_DEFAULT,
    rel_threshold: float = _STLSQ_REL_THRESHOLD,
    frontier: tuple[float, ...] = _STLSQ_FRONTIER,
    degree: int = _POLY_DEGREE,
    max_samples: int = 40000,
    max_skill_runs: int = 40,
    output: Path | None = None,
    device: str = "cpu",
) -> dict:
    """Full T8 distillation on the landed (Dα-grounded) latent.

    Returns the artifact dict (also written to artifacts/discovery_sindy_v0.json).
    """
    from imas_ambix.statespace.baseline import (  # noqa: PLC0415
        _FEATURE_SCHEMA_MAG_ANE,
        _LEVEL1_DIR,
        _XIM_CHANNELS_PRIMARY,
    )
    from imas_ambix.statespace.discovery_extract import (  # noqa: PLC0415
        load_or_train_engine,
    )
    from imas_ambix.statespace.engine import (  # noqa: PLC0415
        _load_split_runs,
    )

    model, stats = load_or_train_engine()
    traj = load_trajectories()

    # Use the FILTERED posterior latents — the rollout/skill test anchors on the
    # causal filter belief, and T7 showed filtered≈smoothed (r90=3 either way).
    z_all = traj["z_post"].astype(np.float64)  # (N, L)
    tmask = traj["transient_mask"].astype(bool)  # (N,)

    basis = build_reduced_basis(z_all, r=r)
    logger.info(
        "[sindy] reduced basis r=%d  σ[:6]=%s",
        r,
        np.round(basis.singular_values[:6], 3).tolist(),
    )

    z_trans = z_all[tmask]
    z_quies = z_all[~tmask]

    fit_trans = distil_stratum(
        model,
        z_trans,
        basis,
        "transient",
        device=device,
        rel_threshold=rel_threshold,
        frontier=frontier,
        degree=degree,
        max_samples=max_samples,
    )
    fit_quies = distil_stratum(
        model,
        z_quies,
        basis,
        "quiescent",
        device=device,
        rel_threshold=rel_threshold,
        frontier=frontier,
        degree=degree,
        max_samples=max_samples,
    )

    # Skill preservation needs ShotRun objects (forecast_pairs filters internally).
    # Reuse the SAME train split + horizons as the engine acceptance experiment.
    with open(_SPLITS_MANIFEST) as f:
        splits = json.load(f)
    train_shots = [int(x) for x in splits["train"]]
    runs = _load_split_runs(
        train_shots,
        _FEATURE_SCHEMA_MAG_ANE,
        _XIM_CHANNELS_PRIMARY,
        _LEVEL1_DIR,
        max_shots=500,
        seed=1,
        cache_tag="train",
    )
    horizons = (1, 2, 5, 10, 20)
    skill = skill_preservation(
        model,
        runs,
        stats,
        basis,
        fit_trans,
        horizons,
        device=device,
        max_runs=max_skill_runs,
    )

    artifact = _assemble_artifact(
        model, basis, fit_trans, fit_quies, skill, r, rel_threshold, degree
    )
    if output is None:
        output = Path(__file__).parent / "artifacts" / "discovery_sindy_v0.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    logger.info("[sindy] artifact written to %s", output)
    return artifact


def _stratum_to_dict(fit: StratumFit) -> dict:
    """Serialise a StratumFit (drop the dense coeff matrix; keep the sparse one)."""
    return {
        "label": fit.label,
        "n_samples": fit.n_samples,
        "dz_rms": fit.dz_rms,
        "dxi_rms": fit.dxi_rms,
        "rel_threshold": fit.rel_threshold,
        "n_active_terms": fit.n_active_terms,
        "r2_sparse": fit.r2_sparse,
        "r2_dense": fit.r2_dense,
        "sparsity_skill_frontier": fit.frontier,
        "recurrence": fit.recurrence,
        "sparse_coefficients": fit.xi_coeffs.tolist(),
        "feature_names": _feature_names(fit.powers),
        "dmd_koopman": fit.dmd,
        "fixed_point_jacobian": fit.jacobian,
    }


def _assemble_artifact(
    model, basis, fit_trans, fit_quies, skill, r, rel_threshold, degree
) -> dict:
    """Build the compact JSON artifact with the honesty cruxes baked in."""
    dz_ratio = (
        fit_quies.dz_rms / fit_trans.dz_rms
        if fit_trans.dz_rms > 1e-30
        else float("nan")
    )
    return {
        "description": (
            "T8 SINDy distillation of the closed-form RKN transition kernel f_θ "
            "(engine.trans_mean) into a sparse symbolic recurrence in the reduced "
            f"(r={r}) coordinates from T7's SVD. f_θ is an exactly-evaluable "
            "autonomous map → exact increments Δz=f_θ(z) read directly (no "
            "derivative estimation). STLSQ in reduced coords, stratified "
            "transient/quiescent. DMD/Koopman + quiescent-fixed-point Jacobian "
            "report SUB-500 Hz decay rates only. Validated by Dα-skill "
            "preservation through the FROZEN observation head (three-way "
            "attribution). FIRST distillation on the current Dα-grounded latent; "
            "the definitive discovery re-runs once T6 grounding lands."
        ),
        "crux_1_drift_reg_confound": (
            "The landed checkpoint trained with drift_reg=0.3, which penalises "
            "‖f_θ(z)‖² on QUIESCENT steps → biases f_θ→IDENTITY there. This "
            "confound lives on the TRANSITION KERNEL — exactly this object (T7 "
            "separately showed the STATE SVD is full-effective-rank-3 even "
            "quiescent; plan comment c-s8-t7-correction). EMPIRICAL signature: "
            f"‖Δz‖ quiescent={fit_quies.dz_rms:.4g} vs transient="
            f"{fit_trans.dz_rms:.4g} (ratio {dz_ratio:.3f}); the quiescent "
            "fixed-point Jacobian eigenvalues clustering at μ≈1 (decay≈0) are the "
            "drift_reg→identity signature, NOT a near-conservation physics result. "
            "The TRANSIENT stratum carries the real terms."
        ),
        "crux_2_aliasing": (
            "ama MHD modes are multi-kHz (median 3–6 kHz, p95 ~10 kHz) vs the "
            f"1 kHz latent's {_NYQUIST_HZ:.0f} Hz Nyquist → eigenfrequency-to-MHD-"
            "mode matching is DEAD (aliased). Spectral validation is SUB-500 Hz "
            "ONLY: decay RATES (Re λ) are reported (meaningful for genuinely slow "
            "modes); oscillation FREQUENCIES are reported folded into [0, 500] Hz "
            "but oscillation→MHD-mode matching is explicitly DISCLAIMED."
        ),
        "config": {
            "reduced_dim_r": r,
            "stlsq_rel_threshold": rel_threshold,
            "stlsq_threshold_units": (
                "RELATIVE: a term survives if its column-normalised |coeff| ≥ "
                "rel_threshold × RMS(Δξ_dim); coefficients reported in physical "
                "(un-normalised) reduced-coordinate units"
            ),
            "stlsq_alpha": _STLSQ_ALPHA,
            "poly_degree": degree,
            "model_hz": _MODEL_HZ,
            "nyquist_hz": _NYQUIST_HZ,
            "increment_source": (
                "EXACT trans_mean(z) on filtered z_post (not finite diff)"
            ),
            "latent_used": "filtered z_post (T7 cache; filtered≈smoothed)",
            "latent_dim": model.cfg.latent_dim,
            "drift_reg_weight": model.cfg.drift_reg_weight,
            "emission": model.cfg.emission,
            "seed": model.cfg.seed,
            "singular_values_top8": np.round(basis.singular_values[:8], 4).tolist(),
        },
        "open_decisions": {
            "uq-level-v0": "STILL-OPEN — not resolved by T8",
            "extrapolation-coordinates": (
                "STILL-OPEN — the reduced coords are V_r of the RAW latent SVD "
                "(not dimensionless); the dimensionless framing is unresolved"
            ),
        },
        "transient": _stratum_to_dict(fit_trans),
        "quiescent": _stratum_to_dict(fit_quies),
        "skill_preservation": skill,
        "gated_final_rerun": (
            "This is the FIRST distillation on the current Dα-grounded latent. "
            "The DEFINITIVE discovery (sparse recurrence + spectrum + skill "
            "delta) must be RE-RUN once the T6 GS-grounded latent lands — the "
            "transition kernel changes with the grounding, so these coefficients "
            "are provisional."
        ),
    }


# ===========================================================================
# CLI
# ===========================================================================


def main() -> None:
    import argparse  # noqa: PLC0415

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s"
    )
    p = argparse.ArgumentParser(description="T8: SINDy distillation of f_θ")
    p.add_argument("--r", type=int, default=_R_DEFAULT, help="reduced dimension")
    p.add_argument(
        "--rel-threshold",
        type=float,
        default=_STLSQ_REL_THRESHOLD,
        help="STLSQ relative threshold (fraction of RMS(Δξ_dim))",
    )
    p.add_argument("--degree", type=int, default=_POLY_DEGREE)
    p.add_argument("--max-samples", type=int, default=40000)
    p.add_argument("--max-skill-runs", type=int, default=40)
    p.add_argument("--quick", action="store_true", help="small smoke run")
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()

    if args.quick:
        args.max_samples = 4000
        args.max_skill_runs = 5

    artifact = run_distillation(
        r=args.r,
        rel_threshold=args.rel_threshold,
        degree=args.degree,
        max_samples=args.max_samples,
        max_skill_runs=args.max_skill_runs,
        output=args.output,
    )

    print("\n=== T8 SINDy Distillation Report ===")
    for stratum in ("transient", "quiescent"):
        s = artifact[stratum]
        print(f"\n--- {stratum.upper()} (n={s['n_samples']}) ---")
        print(f"  ‖Δz‖_rms={s['dz_rms']:.4g}  ‖Δξ‖_rms={s['dxi_rms']:.4g}")
        print(
            f"  R²(sparse)={s['r2_sparse']:.4f}  R²(dense)={s['r2_dense']:.4f}  "
            f"active terms={s['n_active_terms']}"
        )
        for line in s["recurrence"]:
            print(f"    {line}")
        print(f"  frontier: {s['sparsity_skill_frontier']}")
    print("\n--- SKILL PRESERVATION (4-way; CRPS overall) ---")
    sk = artifact["skill_preservation"]["delta"]
    print("  overall:", sk["crps_overall"])
    g = sk["discrimination_guard"]
    print(
        f"  GUARD: identity−true rel gap={g['identity_minus_true_rel']:.1%} "
        f"discriminates={g['metric_discriminates']}"
    )
    print(f"  {g['interpretation']}")


if __name__ == "__main__":
    main()
