"""Analytic / synthetic tests for the annulus soft-prior penalty rows.

The module under test assembles weighted least-squares rows that penalise
disagreement between the interior-solve flux and the frozen source-free
harmonic read over the shared vacuum annulus.  Every test here is pure numpy
on synthetic fields -- no data, no SLURM, no gate evals.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from imas_ambix.latent.boundary_prior import (
    _robust_weights,
    annulus_penalty_rows,
    annulus_point_set,
)

# --- synthetic grid ---------------------------------------------------------


def _circular_grid(n=41, half=1.0, limiter_radius=0.85, center=(1.0, 0.0)):
    """A tiny (nz, nr) grid duck-typed to what ``annulus_point_set`` reads.

    ``inside_limiter`` is a disk of ``limiter_radius`` about ``center`` in the
    (R, Z) plane -- enough to exercise the annulus region logic without
    building a full ``EquilibriumGrid`` (Green's assembly, LU, ...)."""
    r0, z0 = center
    rg = np.linspace(r0 - half, r0 + half, n)
    zg = np.linspace(z0 - half, z0 + half, n)
    mesh_r, mesh_z = np.meshgrid(rg, zg)  # (nz, nr)
    flat_r = mesh_r.ravel()
    flat_z = mesh_z.ravel()
    inside = ((mesh_r - r0) ** 2 + (mesh_z - z0) ** 2) <= limiter_radius**2
    return SimpleNamespace(
        rg=rg,
        zg=zg,
        nr=rg.size,
        nz=zg.size,
        flat_r=flat_r,
        flat_z=flat_z,
        inside_limiter=inside,
    )


def _confined_bump(grid, center=(1.0, 0.0), amp=1.0, sigma=0.35):
    """A Gaussian ψ bump peaked at ``center`` -- a stand-in confined region."""
    r0, z0 = center
    rr = grid.flat_r
    zz = grid.flat_z
    return amp * np.exp(-(((rr - r0) ** 2 + (zz - z0) ** 2) / (2.0 * sigma**2)))


def _gate_annulus_mask(psi_carrier, grid, axis_psi, boundary_psi):
    """Reproduce ``annulus_consistency_rms``'s region definition exactly."""
    sign = np.sign(axis_psi - boundary_psi)
    confined = (psi_carrier - boundary_psi) * sign > 0.0
    inside = np.asarray(grid.inside_limiter, dtype=bool).ravel()
    return inside & ~confined


# --- annulus point set ------------------------------------------------------


def test_annulus_point_set_matches_gate_region():
    grid = _circular_grid()
    psi = _confined_bump(grid)  # peak 1.0 at centre, decays outward
    axis_psi = 1.0
    boundary_psi = 0.4  # ring between this contour and the limiter is annulus
    expected = np.where(_gate_annulus_mask(psi, grid, axis_psi, boundary_psi))[0]
    got = annulus_point_set(
        grid, psi_carrier=psi, axis_psi=axis_psi, boundary_psi=boundary_psi
    )
    assert set(got.tolist()) == set(expected.tolist())
    # sanity: the annulus is non-empty and strictly inside the limiter
    assert got.size > 0
    assert np.all(np.asarray(grid.inside_limiter, dtype=bool).ravel()[got])


def test_annulus_point_set_sign_agnostic():
    """A flipped-sign carrier (axis a MINIMUM) yields the same geometric ring."""
    grid = _circular_grid()
    psi_up = _confined_bump(grid)
    idx_up = annulus_point_set(grid, psi_carrier=psi_up, axis_psi=1.0, boundary_psi=0.4)
    psi_dn = -psi_up
    idx_dn = annulus_point_set(
        grid, psi_carrier=psi_dn, axis_psi=-1.0, boundary_psi=-0.4
    )
    assert set(idx_up.tolist()) == set(idx_dn.tolist())


# --- helpers to build a tiny penalty problem --------------------------------


def _tiny_abs_case(seed=0, n_ann=12, k_dof=3, kp=2):
    rng = np.random.default_rng(seed)
    psi_basis = rng.normal(size=(n_ann, k_dof))
    psi_pass = rng.normal(size=(n_ann, kp))
    psi_fixed = rng.normal(size=n_ann)
    psi_target = rng.normal(size=n_ann)
    return psi_basis, psi_pass, psi_fixed, psi_target, k_dof, kp


def _tiny_grad_case(seed=1, n_grad=20, k_dof=3, kp=2):
    rng = np.random.default_rng(seed)
    grad_basis = rng.normal(size=(n_grad, k_dof))
    grad_pass = rng.normal(size=(n_grad, kp))
    grad_fixed = rng.normal(size=n_grad)
    grad_target = rng.normal(size=n_grad)
    return grad_basis, grad_pass, grad_fixed, grad_target, k_dof, kp


# --- gauge invariance -------------------------------------------------------


def test_grad_form_ignores_absolute_psi_shift():
    """grad-ψ matching is manifestly gauge-independent: a constant added to any
    ABSOLUTE-ψ input leaves the rows unchanged (and the ψ inputs are unused)."""
    gb, gp, gf, gt, k_dof, kp = _tiny_grad_case()
    a0, b0 = annulus_penalty_rows(
        form="grad-psi",
        psi_basis_ann=None,
        psi_pass_ann=None,
        psi_fixed_ann=None,
        psi_target_ann=None,
        grad_basis_ann=gb,
        grad_pass_ann=gp,
        grad_fixed_ann=gf,
        grad_target_ann=gt,
        k_dof=k_dof,
        kp=kp,
        weight=2.0,
    )
    # A constant added to the GRAD target/fixed would change things (grad of a
    # constant is zero physically, so the gauge shift lives in ψ, not ∇ψ) --
    # here we prove the rows do not depend on absolute ψ at all by passing ψ
    # arrays and confirming identical output.
    a1, b1 = annulus_penalty_rows(
        form="grad-psi",
        psi_basis_ann=np.ones((gb.shape[0], k_dof)) * 5.0,
        psi_pass_ann=np.ones((gb.shape[0], kp)) * 7.0,
        psi_fixed_ann=np.full(gb.shape[0], 9.0),
        psi_target_ann=np.full(gb.shape[0], 11.0),
        grad_basis_ann=gb,
        grad_pass_ann=gp,
        grad_fixed_ann=gf,
        grad_target_ann=gt,
        k_dof=k_dof,
        kp=kp,
        weight=2.0,
    )
    assert np.allclose(a0, a1)
    assert np.allclose(b0, b1)
    # width is exactly the DOF (no offset column for grad-ψ)
    assert a0.shape[1] == k_dof + kp


def test_abs_form_offset_absorbs_constant_shift():
    """abs-ψ with a free rank-1 offset: shifting the target by a constant leaves
    the achievable least-squares residual unchanged (g soaks up the DC)."""
    pb, pp, pf, pt, k_dof, kp = _tiny_abs_case(n_ann=15)

    def resid_norm(target):
        A, b = annulus_penalty_rows(
            form="abs-psi",
            psi_basis_ann=pb,
            psi_pass_ann=pp,
            psi_fixed_ann=pf,
            psi_target_ann=target,
            k_dof=k_dof,
            kp=kp,
            weight=1.0,
            gauge_offset=True,
        )
        assert A.shape[1] == k_dof + kp + 1  # trailing offset column
        x, *_ = np.linalg.lstsq(A, b, rcond=None)
        return np.linalg.norm(A @ x - b)

    r_base = resid_norm(pt)
    r_shift = resid_norm(pt + 3.14159)
    assert r_base > 1e-8  # random system is genuinely over-determined
    assert np.isclose(r_base, r_shift, rtol=1e-8, atol=1e-10)


def test_abs_form_without_offset_is_shift_sensitive():
    """gauge_offset=False (offset pre-removed) has NO free DC DOF, so a raw
    constant shift DOES move the residual -- the ablation control."""
    pb, pp, pf, pt, k_dof, kp = _tiny_abs_case(n_ann=15)

    def resid_norm(target):
        A, b = annulus_penalty_rows(
            form="abs-psi",
            psi_basis_ann=pb,
            psi_pass_ann=pp,
            psi_fixed_ann=pf,
            psi_target_ann=target,
            k_dof=k_dof,
            kp=kp,
            weight=1.0,
            gauge_offset=False,
        )
        assert A.shape[1] == k_dof + kp  # no offset column
        x, *_ = np.linalg.lstsq(A, b, rcond=None)
        return np.linalg.norm(A @ x - b)

    assert not np.isclose(resid_norm(pt), resid_norm(pt + 3.14159))


# --- exactness --------------------------------------------------------------


def test_grad_form_exact_recovery():
    """If the basis can represent the target exactly, the penalty rows put the
    residual at ~0 for the true coefficients."""
    rng = np.random.default_rng(3)
    n_grad, k_dof, kp = 30, 4, 3
    gb = rng.normal(size=(n_grad, k_dof))
    gp = rng.normal(size=(n_grad, kp))
    gf = rng.normal(size=n_grad)
    coeffs_true = rng.normal(size=k_dof)
    a_true = rng.normal(size=kp)
    gt = gb @ coeffs_true + gp @ a_true + gf  # target the basis reproduces
    A, b = annulus_penalty_rows(
        form="grad-psi",
        psi_basis_ann=None,
        psi_pass_ann=None,
        psi_fixed_ann=None,
        psi_target_ann=None,
        grad_basis_ann=gb,
        grad_pass_ann=gp,
        grad_fixed_ann=gf,
        grad_target_ann=gt,
        k_dof=k_dof,
        kp=kp,
        weight=3.0,
    )
    x_true = np.concatenate([coeffs_true, a_true])
    assert np.linalg.norm(A @ x_true - b) < 1e-9


def test_abs_form_exact_recovery_with_offset():
    rng = np.random.default_rng(4)
    n_ann, k_dof, kp = 25, 4, 2
    pb = rng.normal(size=(n_ann, k_dof))
    pp = rng.normal(size=(n_ann, kp))
    pf = rng.normal(size=n_ann)
    coeffs_true = rng.normal(size=k_dof)
    a_true = rng.normal(size=kp)
    g_true = 2.5
    # ψ_basis@c + ψ_pass@a + ψ_fixed - g == target  ⇒  target reproduced at x
    pt = pb @ coeffs_true + pp @ a_true + pf - g_true
    A, b = annulus_penalty_rows(
        form="abs-psi",
        psi_basis_ann=pb,
        psi_pass_ann=pp,
        psi_fixed_ann=pf,
        psi_target_ann=pt,
        k_dof=k_dof,
        kp=kp,
        weight=1.0,
        gauge_offset=True,
    )
    x_true = np.concatenate([coeffs_true, a_true, [g_true]])
    assert np.linalg.norm(A @ x_true - b) < 1e-9


# --- per-slice / robust weighting -------------------------------------------


def test_per_slice_uncertainty_scales_weight():
    """A higher per-slice uncertainty down-weights the whole slice's rows by
    1/σ (whitening a Gaussian of std σ)."""
    gb, gp, gf, gt, k_dof, kp = _tiny_grad_case()
    A1, b1 = annulus_penalty_rows(
        form="grad-psi",
        psi_basis_ann=None,
        psi_pass_ann=None,
        psi_fixed_ann=None,
        psi_target_ann=None,
        grad_basis_ann=gb,
        grad_pass_ann=gp,
        grad_fixed_ann=gf,
        grad_target_ann=gt,
        k_dof=k_dof,
        kp=kp,
        weight=1.0,
        per_slice_uncertainty=2.0,
    )
    A0, b0 = annulus_penalty_rows(
        form="grad-psi",
        psi_basis_ann=None,
        psi_pass_ann=None,
        psi_fixed_ann=None,
        psi_target_ann=None,
        grad_basis_ann=gb,
        grad_pass_ann=gp,
        grad_fixed_ann=gf,
        grad_target_ann=gt,
        k_dof=k_dof,
        kp=kp,
        weight=1.0,
    )
    assert np.allclose(A1, 0.5 * A0)
    assert np.allclose(b1, 0.5 * b0)


def test_robust_clip_downweights_outlier():
    """A single annulus point with an anomalous target-vs-fixed mismatch (e.g.
    a near-pole harmonic divergence) gets a reduced row weight."""
    rng = np.random.default_rng(5)
    n_ann, k_dof, kp = 40, 3, 2
    pb = rng.normal(size=(n_ann, k_dof))
    pp = rng.normal(size=(n_ann, kp))
    pf = rng.normal(size=n_ann)
    pt = pf + 0.05 * rng.normal(size=n_ann)  # tight bulk mismatch
    pt[7] += 1000.0  # inject an outlier point

    def rows(clip):
        return annulus_penalty_rows(
            form="abs-psi",
            psi_basis_ann=pb,
            psi_pass_ann=pp,
            psi_fixed_ann=pf,
            psi_target_ann=pt,
            k_dof=k_dof,
            kp=kp,
            weight=1.0,
            robust_clip=clip,
        )

    A_plain, _ = rows(None)
    A_rob, _ = rows(3.0)
    # bulk rows keep (nearly) full weight; the outlier row is strongly shrunk
    bulk = [i for i in range(n_ann) if i != 7]
    ratio_bulk = np.linalg.norm(A_rob[bulk], axis=1) / (
        np.linalg.norm(A_plain[bulk], axis=1) + 1e-30
    )
    ratio_out = np.linalg.norm(A_rob[7]) / (np.linalg.norm(A_plain[7]) + 1e-30)
    assert ratio_out < 0.1
    assert np.median(ratio_bulk) > 0.9


def test_robust_weights_uniform_shift_invariant():
    """The Huber weights are centred, so adding a constant to the residual proxy
    (an absolute-ψ gauge shift) does not change which points are outliers."""
    rng = np.random.default_rng(6)
    r = 0.02 * rng.normal(size=30)
    r[3] = 5.0
    w = _robust_weights(r, 3.0)
    w_shift = _robust_weights(r + 100.0, 3.0)
    assert np.allclose(w, w_shift)
    assert w[3] < 0.5
    assert np.all(w[np.arange(30) != 3] > 0.9)


def test_bad_form_raises():
    pb, pp, pf, pt, k_dof, kp = _tiny_abs_case()
    with pytest.raises(ValueError):
        annulus_penalty_rows(
            form="mean-projection",
            psi_basis_ann=pb,
            psi_pass_ann=pp,
            psi_fixed_ann=pf,
            psi_target_ann=pt,
            k_dof=k_dof,
            kp=kp,
            weight=1.0,
        )
