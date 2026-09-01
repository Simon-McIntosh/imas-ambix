"""Tests for the profile-free GS structure residual (fully synthetic).

Fixtures are built analytically — nested flux surfaces with a chosen ``a(ψ) =
p′``, ``b(ψ) = FF′/μ₀`` and ``jφ = a·R + b/R`` — so a GS-compliant current is
known in closed form.  No MAST data, no interaction matrices, no EFIT.  The
adversarial battery mirrors the scoping discrimination ratios (oracle floor
~0.0036, permuted ~20×, sensor-null-space ~5.7×; artifact
``imas_ambix/latent/artifacts/patch_scoping/discriminate-affine-r2.json``).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from imas_ambix.latent.structure_residual import (
    _bin_grid,
    _design,
    coefficient_smoothness_penalty,
    f2_integrability_penalty,
    fit_flux_functions,
    integrate_closures,
    structure_residual,
)

torch.manual_seed(0)


# --------------------------------------------------------------------------
# synthetic GS-compliant fixtures
# --------------------------------------------------------------------------


def _nested_fixture(nr=40, nz=50, r0=0.9, a0=1.0, a1=0.3, b0=0.5, b1=0.2):
    """Single-O-point nested-surface fixture with a linear a(ψ), b(ψ).

    ψ = −(((R−R0)/0.4)² + (Z/0.6)²): axis (R0, 0) at ψ=0, decreasing outward.
    Core mask ψ ≥ −1.  jφ = a(ψ)·R + b(ψ)/R inside, 0 outside — GS-compliant by
    construction (R·jφ = a·R² + b is exactly affine on each level set).
    """
    rg = np.linspace(0.4, 1.4, nr)
    zg = np.linspace(-0.8, 0.8, nz)
    rr, zz = np.meshgrid(rg, zg)
    psi = -(((rr - r0) / 0.4) ** 2 + (zz / 0.6) ** 2)
    core = psi >= -1.0
    a = a0 + a1 * psi
    b = b0 + b1 * psi
    jphi = np.where(core, a * rr + b / rr, 0.0)
    return (
        rr.ravel(),
        zz.ravel(),
        psi.ravel(),
        jphi.ravel(),
        core.ravel(),
        (a0, a1, b0, b1),
    )


def _disconnected_fixture():
    """Two disjoint current-carrying blobs sharing the SAME ψ values, DIFFERENT
    (a, b) per blob; each individually GS-compliant.

    Both blobs map their local squared-distance to ψ ∈ [−0.06, 0], so a naïve
    ψ-value grouping pools the two spatially separate components into one bin.
    The core blob carries the dominant current (so the current-weighted centroid
    sits on it, letting the single-centroid locality factor drop the minor one).
    """
    rg = np.linspace(0.4, 1.4, 46)
    zg = np.linspace(-0.8, 0.8, 56)
    rr, zz = np.meshgrid(rg, zg)
    r, z = rr.ravel(), zz.ravel()

    # core blob (dominant), private-flux blob (minor), well separated in (R, Z)
    core_c, priv_c = (0.95, 0.2), (0.6, -0.4)
    rad = 0.22
    d_core = np.hypot(r - core_c[0], z - core_c[1])
    d_priv = np.hypot(r - priv_c[0], z - priv_c[1])
    in_core = d_core < rad
    in_priv = d_priv < rad

    psi = np.full_like(r, -10.0)  # vacuum elsewhere (jφ=0, harmless)
    psi[in_core] = -(d_core[in_core] ** 2)  # ψ ∈ [−rad², 0]
    psi[in_priv] = -(d_priv[in_priv] ** 2)  # SAME ψ range, different location

    jphi = np.zeros_like(r)
    # different (a, b) per blob; core amplitude dominant
    jphi[in_core] = 1.0 * r[in_core] + 0.5 / r[in_core]
    jphi[in_priv] = 0.35 * (-0.4 * r[in_priv] + 0.9 / r[in_priv])

    labels = np.full(r.shape, 2, dtype=np.int64)  # 2 = vacuum
    labels[in_core] = 0
    labels[in_priv] = 1
    return r, z, psi, jphi, labels


def _t(x):
    return torch.as_tensor(x, dtype=torch.float64)


# --------------------------------------------------------------------------
# 1 — compliance floor (both forms)
# --------------------------------------------------------------------------


def test_compliance_floor_both_forms():
    r, z, psi, jphi, _core, _ = _nested_fixture()
    for form in ("affine-r2", "jphi"):
        res = float(structure_residual(_t(psi), _t(r), _t(jphi), form=form))
        assert res <= 1e-2, f"{form}: compliant residual {res:.4g} above floor"


# --------------------------------------------------------------------------
# 2 — permuted current
# --------------------------------------------------------------------------


def test_permuted_current_is_large():
    r, z, psi, jphi, core, _ = _nested_fixture()
    base = float(structure_residual(_t(psi), _t(r), _t(jphi)))
    rng = np.random.default_rng(1)
    jp = jphi.copy()
    idx = np.where(core)[0]
    jp[idx] = jp[idx][rng.permutation(idx.size)]
    perm = float(structure_residual(_t(psi), _t(r), _t(jp)))
    assert perm >= 10.0 * base, f"permuted {perm:.4g} not ≥10× floor {base:.4g}"


# --------------------------------------------------------------------------
# 3 — structureless random current
# --------------------------------------------------------------------------


def test_structureless_current_is_large():
    r, z, psi, jphi, core, _ = _nested_fixture()
    base = float(structure_residual(_t(psi), _t(r), _t(jphi)))
    rng = np.random.default_rng(2)
    jp = np.zeros_like(jphi)
    idx = np.where(core)[0]
    scale = np.sqrt(np.mean(jphi[idx] ** 2))
    jp[idx] = rng.normal(0.0, scale, idx.size)
    res = float(structure_residual(_t(psi), _t(r), _t(jp)))
    assert res >= 10.0 * base and res > 0.05, f"structureless residual {res:.4g}"


# --------------------------------------------------------------------------
# 4 — sensor-null-space perturbation
# --------------------------------------------------------------------------


def test_null_space_perturbation_is_detected():
    r, z, psi, jphi, core, _ = _nested_fixture()
    base = float(structure_residual(_t(psi), _t(r), _t(jphi)))
    idx = np.where(core)[0]
    rng = np.random.default_rng(3)
    a_sens = rng.normal(size=(60, idx.size))  # fat random "sensor" matrix
    # project a random perturbation onto the sensor null space (invisible to A)
    v = rng.normal(size=idx.size)
    gram = a_sens @ a_sens.T
    delta = v - a_sens.T @ np.linalg.solve(gram, a_sens @ v)
    delta *= 0.30 * np.linalg.norm(jphi[idx]) / np.linalg.norm(delta)
    jp = jphi.copy()
    jp[idx] += delta
    # confirm the perturbation really is (near-)invisible to the sensor matrix
    assert np.linalg.norm(a_sens @ delta) < 1e-6 * np.linalg.norm(delta)
    res = float(structure_residual(_t(psi), _t(r), _t(jp)))
    assert res >= 3.0 * base, f"null-space {res:.4g} not ≥3× floor {base:.4g}"


# --------------------------------------------------------------------------
# 5 — gradcheck (differentiability gate, fp64, small)
# --------------------------------------------------------------------------


def test_gradcheck_psi_and_jphi():
    r, z, psi, jphi, core, _ = _nested_fixture(nr=6, nz=7)
    idx = np.where(core)[0]
    r_c = _t(r[idx])
    psi_c = _t(psi[idx]).requires_grad_(True)
    jphi_c = _t(jphi[idx]).requires_grad_(True)

    # freeze the ψ-binning so the finite-difference check sees the SAME
    # (bins-detached) sensitivity autograd computes — gradients flow through the
    # kernel and jφ, not through where the (detached) bins sit
    w_amp = _t(jphi[idx]) ** 2
    grid = _bin_grid(_t(psi[idx]), w_amp / w_amp.sum(), 4, 1.0)

    def f(p, j):
        return structure_residual(p, r_c, j, n_bins=4, bin_grid=grid)

    assert torch.autograd.gradcheck(f, (psi_c, jphi_c), eps=1e-6, atol=1e-4, rtol=1e-3)


# --------------------------------------------------------------------------
# 6 — disconnected components (the connectivity caveat)
# --------------------------------------------------------------------------


def test_disconnected_components_need_connectivity():
    # a connected single-blob floor for reference
    r0, z0, psi0, jphi0, _c, _ = _nested_fixture()
    floor = float(structure_residual(_t(psi0), _t(r0), _t(jphi0)))

    r, z, psi, jphi, labels = _disconnected_fixture()
    args = dict(psi_c=_t(psi), r_c=_t(r), jphi_c=_t(jphi), z_c=_t(z))

    naive = float(structure_residual(**args, connectivity=None))
    labelled = float(
        structure_residual(
            **args, connectivity="labels", component_labels=torch.as_tensor(labels)
        )
    )
    local = float(
        structure_residual(**args, connectivity="locality", locality_scale=0.12)
    )

    assert naive >= 5.0 * floor, f"naive {naive:.4g} not ≥5× floor {floor:.4g}"
    assert labelled <= 0.4 * naive, f"labels {labelled:.4g} did not recover"
    assert local <= 0.4 * naive, f"locality {local:.4g} did not recover"

    # no free pass: a genuinely structureless single-region state stays large
    rng = np.random.default_rng(4)
    rN, zN, psiN, jphiN, coreN, _ = _nested_fixture()
    idx = np.where(coreN)[0]
    jbad = np.zeros_like(jphiN)
    jbad[idx] = rng.normal(0.0, np.sqrt(np.mean(jphiN[idx] ** 2)), idx.size)
    bad_local = float(
        structure_residual(
            _t(psiN),
            _t(rN),
            _t(jbad),
            z_c=_t(zN),
            connectivity="locality",
            locality_scale=0.12,
        )
    )
    assert bad_local > 0.05, f"connectivity gave a bad state a pass ({bad_local:.4g})"


# --------------------------------------------------------------------------
# 7 — closure recovery + integration
# --------------------------------------------------------------------------


def test_closure_recovery_and_integration():
    r, z, psi, jphi, core, (a0, a1, b0, b1) = _nested_fixture()
    fit = fit_flux_functions(_t(psi), _t(r), _t(jphi), n_bins=24)

    psi_c = np.asarray(fit.psi_centers)
    a_true = a0 + a1 * psi_c
    b_true = b0 + b1 * psi_c
    mass = np.asarray(fit.weight_mass)
    a_err = np.asarray(fit.a_err)
    b_err = np.asarray(fit.b_err)

    # well-populated interior bins only
    well = mass > 0.2 * mass.max()
    assert well.sum() >= 5
    a_rel = np.abs(np.asarray(fit.a_k)[well] - a_true[well]) / np.abs(a_true[well])
    b_rel = np.abs(np.asarray(fit.b_k)[well] - b_true[well]) / np.abs(b_true[well])
    assert a_rel.max() < 0.10, f"a(ψ) recovery {a_rel.max():.3f}"
    assert b_rel.max() < 0.10, f"b(ψ) recovery {b_rel.max():.3f}"
    assert np.all(np.isfinite(a_err[well])) and np.all(a_err[well] > 0)
    assert np.all(np.isfinite(b_err[well])) and np.all(b_err[well] > 0)

    integ = integrate_closures(fit, psi_axis=0.0, psi_boundary=-1.0, f_vac=0.85)
    p, f2 = integ["p"], integ["f_squared"]
    # positive-a fixture: p ≈ 0 at the boundary end, grows toward the axis end
    assert abs(p[0]) < 0.2 * np.abs(p).max()
    assert np.abs(p[-1]) > np.abs(p[0])
    assert np.all(f2 >= 0.0), "F² went negative for an adequate f_vac"


# --------------------------------------------------------------------------
# 8 — F² integrability penalty
# --------------------------------------------------------------------------


def test_f2_integrability_penalty():
    dpsi = 0.05
    b_ok = torch.full((20,), 0.4, dtype=torch.float64)
    assert float(f2_integrability_penalty(b_ok, dpsi, f_vac=0.85)) == 0.0

    b_bad = torch.full((20,), -5.0e6, dtype=torch.float64)
    assert float(f2_integrability_penalty(b_bad, dpsi, f_vac=0.1)) > 0.0

    # smoothness sanity: a straight line has zero second difference
    assert float(coefficient_smoothness_penalty(torch.linspace(0, 1, 10))) < 1e-12


# --------------------------------------------------------------------------
# 9 — cost at realistic size
# --------------------------------------------------------------------------


def test_cost_at_realistic_size():
    rng = np.random.default_rng(5)
    n = 5000
    r = torch.as_tensor(rng.uniform(0.4, 1.4, n), dtype=torch.float32)
    z = torch.as_tensor(rng.uniform(-0.8, 0.8, n), dtype=torch.float32)
    psi = torch.as_tensor(-((r - 0.9) ** 2 + z**2), dtype=torch.float32)
    jphi = torch.as_tensor(rng.normal(1.0, 0.3, n), dtype=torch.float32)

    # Measure single-threaded CPU time.  The guard targets algorithmic work, so
    # neither scheduler contention from concurrent test processes nor OpenMP
    # oversubscription should count against it.
    prev = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for _ in range(10):  # warmup (thread-pool spin-up, allocator)
            structure_residual(psi, r, jphi, n_bins=24)
        per_call = []
        for _ in range(21):
            t0 = time.process_time()
            structure_residual(psi, r, jphi, n_bins=24)
            per_call.append((time.process_time() - t0) * 1e3)
    finally:
        torch.set_num_threads(prev)
    ms = float(np.median(per_call))
    assert ms < 50.0, f"structure_residual too slow: {ms:.2f} ms/call (5000 cells)"


# --------------------------------------------------------------------------
# 10 — firewall static check
# --------------------------------------------------------------------------


def test_firewall_clean():
    src = Path("imas_ambix/latent/structure_residual.py").read_text()
    for banned in ("efit_referee", "equilibrium_labels", "worldmodel"):
        assert banned not in src, f"firewall leak: {banned!r} in structure_residual.py"


# --------------------------------------------------------------------------
# 11 — rigid-rotation column (the R⁴ extension of the structure-residual family)
# --------------------------------------------------------------------------


def _rotating_fixture(
    nr=44, nz=52, a0=1.0, a1=0.3, b0=0.5, b1=0.2, c0=1.5, c1=0.5, rotation=True
):
    """Full-R-coverage flux-level fixture for the rigid-rotation R⁴ column.

    ``jφ = a(ψ)·R + b(ψ)/R + c(ψ)·R³`` so that ``R·jφ = a·R² + b + c·R⁴`` is
    affine only once the R⁴ column is present; ``rotation=False`` gives the c=0
    baseline for matched-geometry floors.

    ψ is taken to depend on Z only, so every ψ level-set spans the FULL major-
    radius range.  That is the regime in which the ``[R², 1, R⁴]`` columns are
    individually well conditioned: on nested surfaces the near-axis level-sets
    span little R and R², R⁴ become collinear, so the *fit* (residual) stays good
    while the individual ``(a, b, c)`` are only weakly identified — a genuine
    feature of rotation inference near the axis, deliberately excluded here to
    isolate the column's recovery from that degeneracy.  ``form='affine-r2'``
    cannot represent the centrifugal R⁴ term and reports it as residual;
    ``form='affine-r2-rotation'`` absorbs it and returns to floor.
    """
    rg = np.linspace(0.3, 2.0, nr)
    zg = np.linspace(-0.9, 0.9, nz)
    rr, zz = np.meshgrid(rg, zg)
    psi = -((zz / 0.65) ** 2)  # depends on Z only ⇒ full-R flux levels
    core = psi >= -1.0
    a = a0 + a1 * psi
    b = b0 + b1 * psi
    jphi = a * rr + b / rr
    if rotation:
        jphi = jphi + (c0 + c1 * psi) * rr**3
    jphi = np.where(core, jphi, 0.0)
    return (
        rr.ravel(),
        zz.ravel(),
        psi.ravel(),
        jphi.ravel(),
        core.ravel(),
        (a0, a1, b0, b1, c0, c1),
    )


def test_rotation_form_discriminates_rotating_current():
    """The falsifiable signature: on a rotating current the affine form fails and
    the rotation form recovers to floor."""
    # binning floors: each form on the matched-geometry static (c=0) fixture
    rs, zs, psis, jphis, _cs, _p = _rotating_fixture(rotation=False)
    floor_affine = float(
        structure_residual(_t(psis), _t(rs), _t(jphis), form="affine-r2")
    )
    floor_rot = float(
        structure_residual(_t(psis), _t(rs), _t(jphis), form="affine-r2-rotation")
    )

    r, z, psi, jphi, _core, _pars = _rotating_fixture()
    res_affine = float(structure_residual(_t(psi), _t(r), _t(jphi), form="affine-r2"))
    res_rot = float(
        structure_residual(_t(psi), _t(r), _t(jphi), form="affine-r2-rotation")
    )

    assert res_affine >= 5.0 * floor_affine, (
        f"affine-r2 blind to rotation: {res_affine:.4g} vs floor {floor_affine:.4g}"
    )
    assert res_rot <= 2.0 * floor_rot, (
        f"rotation form did not recover: {res_rot:.4g} vs floor {floor_rot:.4g}"
    )


def test_rotation_form_no_free_lunch_on_static_and_permuted():
    """On a c=0 fixture the rotation form matches the affine form (no overfitting
    on a genuinely affine current), and it does NOT launder a permuted current to
    zero residual (the extra column is still only 3 DOF per bin)."""
    r, z, psi, jphi, core, _p = _rotating_fixture(rotation=False)
    res_affine = float(structure_residual(_t(psi), _t(r), _t(jphi), form="affine-r2"))
    res_rot = float(
        structure_residual(_t(psi), _t(r), _t(jphi), form="affine-r2-rotation")
    )
    # both at floor, and the extra column does not materially change a compliant fit
    assert res_rot <= 1e-2, f"rotation form off floor on static fixture: {res_rot:.4g}"
    assert res_rot <= 3.0 * res_affine + 1e-4, (
        f"rotation form overfit-collapsed on static: {res_rot:.4g} vs {res_affine:.4g}"
    )

    # permuted current: the R⁴ column must NOT drive the garbage residual to zero
    floor_rot = res_rot
    rng = np.random.default_rng(11)
    jp = jphi.copy()
    idx = np.where(core)[0]
    jp[idx] = jp[idx][rng.permutation(idx.size)]
    perm_rot = float(
        structure_residual(_t(psi), _t(r), _t(jp), form="affine-r2-rotation")
    )
    assert perm_rot >= 5.0 * floor_rot, (
        f"rotation form laundered permuted current: {perm_rot:.4g} vs {floor_rot:.4g}"
    )


def test_rotation_closure_recovery():
    """c_k recovered within ~15% on well-populated bins of the rotating fixture;
    a_k/b_k recovered on the static fixture by the rotation form (extra column
    does not corrupt the two affine coefficients when the true c is zero)."""
    # (i) c_k recovery on the rotating fixture
    r, z, psi, jphi, core, (a0, a1, b0, b1, c0, c1) = _rotating_fixture()
    fit = fit_flux_functions(
        _t(psi), _t(r), _t(jphi), n_bins=24, form="affine-r2-rotation"
    )
    assert fit.c_k is not None and fit.c_err is not None
    psi_c = np.asarray(fit.psi_centers)
    mass = np.asarray(fit.weight_mass)
    well = mass > 0.2 * mass.max()
    assert well.sum() >= 5
    c_true = c0 + c1 * psi_c
    c_rel = np.abs(np.asarray(fit.c_k)[well] - c_true[well]) / np.abs(c_true[well])
    assert c_rel.max() < 0.15, f"c(ψ) recovery {c_rel.max():.3f}"
    c_err = np.asarray(fit.c_err)
    assert np.all(np.isfinite(c_err[well])) and np.all(c_err[well] > 0)

    # (ii) a_k/b_k unchanged (vs truth) when the rotation form fits a c=0 fixture
    rs, zs, psis, jphis, cores, (sa0, sa1, sb0, sb1, _sc0, _sc1) = _rotating_fixture(
        rotation=False
    )
    sfit = fit_flux_functions(
        _t(psis), _t(rs), _t(jphis), n_bins=24, form="affine-r2-rotation"
    )
    spsi = np.asarray(sfit.psi_centers)
    smass = np.asarray(sfit.weight_mass)
    swell = smass > 0.2 * smass.max()
    a_true = sa0 + sa1 * spsi
    b_true = sb0 + sb1 * spsi
    a_rel = np.abs(np.asarray(sfit.a_k)[swell] - a_true[swell]) / np.abs(a_true[swell])
    b_rel = np.abs(np.asarray(sfit.b_k)[swell] - b_true[swell]) / np.abs(b_true[swell])
    assert a_rel.max() < 0.10, f"a(ψ) corrupted by rotation column: {a_rel.max():.3f}"
    assert b_rel.max() < 0.10, f"b(ψ) corrupted by rotation column: {b_rel.max():.3f}"


def test_rotation_form_gradcheck():
    r, z, psi, jphi, core, _p = _rotating_fixture(nr=7, nz=8)
    idx = np.where(core)[0]
    r_c = _t(r[idx])
    psi_c = _t(psi[idx]).requires_grad_(True)
    jphi_c = _t(jphi[idx]).requires_grad_(True)

    # freeze the ψ-binning so finite differences see the bins-detached sensitivity
    w_amp = _t(jphi[idx]) ** 2
    grid = _bin_grid(_t(psi[idx]), w_amp / w_amp.sum(), 4, 1.0)

    def f(p, j):
        return structure_residual(
            p, r_c, j, n_bins=4, form="affine-r2-rotation", bin_grid=grid
        )

    assert torch.autograd.gradcheck(f, (psi_c, jphi_c), eps=1e-6, atol=1e-4, rtol=1e-3)


def test_rotation_design_columns():
    """The rotation design is exactly [R², 1, R⁴] on y = R·jφ."""
    r = _t(np.array([0.5, 1.0, 1.3]))
    j = _t(np.array([2.0, 3.0, 1.5]))
    x, y = _design("affine-r2-rotation", r, j)
    assert x.shape == (3, 3)
    assert torch.allclose(x[:, 0], r * r)
    assert torch.allclose(x[:, 1], torch.ones_like(r))
    assert torch.allclose(x[:, 2], r**4)
    assert torch.allclose(y, r * j)
