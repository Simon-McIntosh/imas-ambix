"""Tests for the profile-free GS structure residual (fully synthetic).

Fixtures are built analytically — nested flux surfaces with a chosen ``a(ψ) =
p′``, ``b(ψ) = FF′/μ₀`` and ``jφ = a·R + b/R`` — so a GS-compliant current is
known in closed form.  No MAST data, no interaction matrices, no EFIT.  The
adversarial battery mirrors the scoping discrimination ratios (oracle floor
~0.0036, permuted ~20×, sensor-null-space ~5.7×; artifact
``imas_ambix/latent/artifacts/patch_scoping/discriminate-affine-r2.json``).
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import torch

from imas_ambix.latent.structure_residual import (
    _bin_grid,
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

    # Measure at a normal CPU-deployment thread count.  On very-high-core nodes
    # (e.g. 48-way login boxes) torch's OpenMP fan-out for elementwise
    # transcendentals (the Gaussian kernel's exp) has a pathological per-op
    # barrier cost — an environment artifact, not the algorithm's cost (the op is
    # ~5 ms single-threaded).  This test guards against ALGORITHMIC regressions,
    # so it pins a realistic thread count and restores it afterward.
    prev = torch.get_num_threads()
    torch.set_num_threads(min(8, os.cpu_count() or 8))
    try:
        for _ in range(10):  # warmup (thread-pool spin-up, allocator)
            structure_residual(psi, r, jphi, n_bins=24)
        per_call = []
        for _ in range(21):
            t0 = time.perf_counter()
            structure_residual(psi, r, jphi, n_bins=24)
            per_call.append((time.perf_counter() - t0) * 1e3)
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
