"""Analytic tests for the current-moment boundary read.

No corpus / zarr is needed: a tiny synthetic ``PatchBasis`` stub carries just
the attributes the fit reads (``r_cells``, ``z_cells``, ``candidate_mask``,
``m_sens``, ``r0``).  The physics claim under test is that a current confined to
the low-order moment span is recovered near-exactly from its own (noise-free)
external-field signature, and that the Rogowski-Ip anchor pins the total current.
"""

from __future__ import annotations

import numpy as np
import torch

from imas_ambix.latent.boundary_moment import (
    MomentFitConfig,
    build_moment_basis,
    fit_moment_currents,
    invert_slices_moment,
    moment_terms,
)
from imas_ambix.latent.patch_inverse import SlicePayload


class _FakeBasis:
    """Minimal stand-in exposing the attributes the moment fit reads."""

    def __init__(self, r_cells, z_cells, candidate_mask, m_sens, r0):
        self.r_cells = torch.as_tensor(r_cells, dtype=torch.float64)
        self.z_cells = torch.as_tensor(z_cells, dtype=torch.float64)
        self.candidate_mask = torch.as_tensor(candidate_mask, dtype=torch.float64)
        self.m_sens = torch.as_tensor(m_sens, dtype=torch.float64)
        self.r0 = float(r0)


def _make_basis(n_side=7, n_sensors=24, r0=0.9, seed=0):
    rng = np.random.default_rng(seed)
    rr = np.linspace(r0 - 0.4, r0 + 0.4, n_side)
    zz = np.linspace(-0.5, 0.5, n_side)
    R, Z = np.meshgrid(rr, zz, indexing="xy")
    r_cells = R.ravel()
    z_cells = Z.ravel()
    n = r_cells.size
    # every cell conductor-clear (candidate) for the recovery tests
    candidate = np.ones(n)
    # a well-conditioned random sensor Green's matrix (S, n), S > K
    m_sens = rng.standard_normal((n_sensors, n)) * 1e-6
    return _FakeBasis(r_cells, z_cells, candidate, m_sens, r0), r_cells, z_cells


def test_moment_fit_config_default_model_is_polynomial():
    """The library default must agree with the gate CLI's default (both
    'polynomial', the locked decision every shipped artifact scores against)
    — a silent disagreement here previously meant library callers that omit
    ``model=`` got a different current representation than the gate."""
    assert MomentFitConfig().model == "polynomial"


def test_moment_terms_counts():
    assert set(moment_terms(1)) == {(0, 0), (1, 0), (0, 1)}
    assert moment_terms(1)[0] == (0, 0)  # monopole first
    assert len(moment_terms(1)) == 3
    assert len(moment_terms(2)) == 6
    assert len(moment_terms(3)) == 10
    assert len(moment_terms(4)) == 15


def test_build_moment_basis_candidate_masked_and_monopole():
    _, r_cells, z_cells = _make_basis()
    cand = np.ones(r_cells.size)
    cand[: r_cells.size // 2] = 0.0  # mask out half
    M, labels, scale = build_moment_basis(r_cells, z_cells, cand, 0.9, order=2)
    assert M.shape == (r_cells.size, 6)
    assert labels[0] == "1"
    assert scale > 0
    # monopole column is exactly the candidate mask (1 inside, 0 outside)
    np.testing.assert_allclose(M[:, 0], cand)
    # every column is zero on masked cells
    assert np.all(M[cand == 0.0, :] == 0.0)


def test_recovers_current_in_moment_span():
    """A current that IS a low-order moment combination is recovered from its
    noise-free sensor signature to numerical precision (well-posed, K < S)."""
    basis, r_cells, z_cells = _make_basis()
    cand = np.ones(r_cells.size)
    M, _, _ = build_moment_basis(r_cells, z_cells, cand, basis.r0, order=3)
    rng = np.random.default_rng(3)
    c_true = rng.standard_normal(M.shape[1]) * 5.0e4
    i_true = M @ c_true
    ip_true = float(i_true.sum())

    m_sens = basis.m_sens.numpy()
    vacuum = rng.standard_normal(m_sens.shape[0]) * 1e-3  # arbitrary known coils
    measured = vacuum + m_sens @ i_true
    payload = SlicePayload(
        measured=measured,
        vacuum=vacuum,
        mask=np.ones(m_sens.shape[0], dtype=bool),
        scale=np.full(m_sens.shape[0], 1e-4),
        i_pf=np.zeros(3),
        ip_amperes=ip_true,
    )
    inv = fit_moment_currents(basis, payload, MomentFitConfig(order=3, ridge=1e-12))

    # sensor signature reproduced, currents recovered, Ip anchor honoured
    assert inv.misfit < 1e-6
    np.testing.assert_allclose(inv.i_cell, i_true, rtol=1e-4, atol=1e-2)
    assert inv.ip_rel_err < 1e-6  # hard anchor -> exact
    assert abs(inv.ip_fit - ip_true) / abs(ip_true) < 1e-6


def test_ip_anchor_pins_total_current():
    """With an underdetermined-in-monopole signature, the anchor still pins Ip."""
    basis, r_cells, z_cells = _make_basis(seed=1)
    m_sens = basis.m_sens.numpy()
    # a smooth true current (Gaussian blob) -- not exactly in the span
    blob = np.exp(-(((r_cells - basis.r0) / 0.25) ** 2 + (z_cells / 0.3) ** 2))
    i_true = blob / blob.sum() * 6.0e5  # 600 kA
    measured = m_sens @ i_true
    payload = SlicePayload(
        measured=measured,
        vacuum=np.zeros(m_sens.shape[0]),
        mask=np.ones(m_sens.shape[0], dtype=bool),
        scale=np.full(m_sens.shape[0], 1e-4),
        i_pf=np.zeros(3),
        ip_amperes=6.0e5,
    )
    # hard anchor pins Ip exactly, even for an out-of-span current
    anchored = fit_moment_currents(basis, payload, MomentFitConfig(order=3))
    assert anchored.ip_rel_err < 1e-9
    # centroid lands near the true blob centre (r0, 0)
    assert abs(anchored.centroid_r - basis.r0) < 0.15
    assert abs(anchored.centroid_z) < 0.15
    # free-fit ablation: no anchor -> Ip is not exactly pinned
    free = fit_moment_currents(
        basis, payload, MomentFitConfig(order=3, ip_anchor=False)
    )
    assert free.ip_rel_err >= anchored.ip_rel_err


def test_mask_ignores_untrusted_rows():
    """Corrupting masked-out sensor rows must not change the fit."""
    basis, r_cells, z_cells = _make_basis(seed=2)
    cand = np.ones(r_cells.size)
    M, _, _ = build_moment_basis(r_cells, z_cells, cand, basis.r0, order=2)
    rng = np.random.default_rng(11)
    c_true = rng.standard_normal(M.shape[1]) * 3.0e4
    i_true = M @ c_true
    m_sens = basis.m_sens.numpy()
    measured = m_sens @ i_true
    S = m_sens.shape[0]
    mask = np.ones(S, dtype=bool)
    mask[::4] = False  # drop a quarter of the rows

    def run(meas):
        p = SlicePayload(
            measured=meas,
            vacuum=np.zeros(S),
            mask=mask,
            scale=np.full(S, 1e-4),
            i_pf=np.zeros(3),
            ip_amperes=float(i_true.sum()),
        )
        return fit_moment_currents(basis, p, MomentFitConfig(order=2))

    clean = run(measured.copy())
    corrupted = measured.copy()
    corrupted[~mask] += 1e3  # garbage on untrusted rows only
    dirty = run(corrupted)
    np.testing.assert_allclose(clean.coeffs, dirty.coeffs, rtol=1e-8, atol=1e-6)


def test_invert_slices_moment_batch():
    basis, r_cells, z_cells = _make_basis(seed=4)
    m_sens = basis.m_sens.numpy()
    S = m_sens.shape[0]
    payloads = []
    for k in range(3):
        i_true = np.exp(-(((r_cells - basis.r0) / 0.3) ** 2)) * (1.0 + k)
        i_true = i_true / i_true.sum() * (4.0e5 + 1e5 * k)
        payloads.append(
            SlicePayload(
                measured=m_sens @ i_true,
                vacuum=np.zeros(S),
                mask=np.ones(S, dtype=bool),
                scale=np.full(S, 1e-4),
                i_pf=np.zeros(3),
                ip_amperes=float(i_true.sum()),
                t_index=k,
            )
        )
    out = invert_slices_moment(basis, payloads, MomentFitConfig(order=3))
    assert len(out) == 3
    assert [o.t_index for o in out] == [0, 1, 2]
    assert all(o.ip_rel_err < 0.1 for o in out)
