"""Tests for the training-free variational patch-current inverse.

Builds the Picard truth on the ``_confining_table`` synthetic fixture
(:mod:`tests.latent.test_gs_solve`) once per module, assembles a
:class:`~imas_ambix.latent.patch_basis.PatchBasis` on it, and synthesises a
self-consistent zero-coil sensor payload from the converged equilibrium.  The
optimiser runs against those fixed inputs to pin: end-to-end behaviour of all
three weight-policy arms, batching consistency, the ``_lambda_schedule``
per-policy update rule in isolation, unknown-policy error handling, candidate-
mask enforcement, misfit descent, the Ip anchor, and that the physics term
actually moves the structure residual.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from imas_ambix.latent.gs_solve import EquilibriumGrid, solve_equilibrium
from imas_ambix.latent.patch_basis import PatchBasis
from imas_ambix.latent.patch_inverse import (
    POLICIES,
    InverseConfig,
    SlicePayload,
    _lambda_schedule,
    invert_slices,
)
from tests.latent.test_gs_solve import _confining_table


@pytest.fixture(scope="module")
def synth_problem(tmp_path_factory):
    """Converged Picard truth + a self-consistent zero-coil sensor payload.

    The synthetic ``_confining_table`` coils cannot classify as KNOWN-PF (no
    centroid match), so the coil block is legitimately empty; the vacuum
    prediction is zero and all sensor signal comes from the plasma patch
    currents themselves.
    """
    table = _confining_table()
    grid = EquilibriumGrid.from_table(table, nr=49, nz=65)
    basis = PatchBasis.from_table(
        table, nr=49, nz=65, cache_dir=tmp_path_factory.mktemp("patch_cache")
    )
    ip = 4.0e5
    i_pf = np.zeros(0)
    res = solve_equilibrium(grid, i_pf, ip, beta0=0.5)
    assert res.converged
    g_sens, channels = grid.sensor_greens(table)
    vac = np.zeros(len(channels))
    meas = vac + g_sens @ res.cell_currents
    scale = np.abs(meas) + 1e-9
    payload = SlicePayload(
        measured=meas,
        vacuum=vac,
        mask=np.ones(meas.size, dtype=bool),
        scale=scale,
        i_pf=i_pf,
        ip_amperes=ip,
        shot=1,
        t_index=0,
    )
    return basis, payload, ip


@pytest.fixture(scope="module", params=POLICIES)
def policy_run(request, synth_problem):
    """One batched (2-slice, identical payload) inversion per policy."""
    basis, payload, ip = synth_problem
    cfg = InverseConfig(iters=200, policy=request.param, lambda_fb=3.0)
    out = invert_slices(basis, [payload, payload], cfg, device="cpu")
    return request.param, cfg, basis, payload, out


def test_policy_runs_end_to_end(policy_run):
    policy, _cfg, _basis, _payload, out = policy_run
    for inv in out:
        assert np.isfinite(inv.misfit)
        assert np.isfinite(inv.structure)
        assert np.isfinite(inv.lambda_final)
        assert inv.ip_rel_err < 0.02, f"{policy}: ip_rel_err {inv.ip_rel_err}"
        assert inv.misfit < 0.1, f"{policy}: misfit {inv.misfit}"


def test_batching_reproducible(policy_run):
    """Two identical payloads in one batch land on near-identical currents."""
    policy, _cfg, _basis, _payload, out = policy_run
    np.testing.assert_allclose(
        out[0].i_cell, out[1].i_cell, rtol=1e-6, atol=1e-6, err_msg=policy
    )


def test_ip_anchor_within_tolerance(policy_run):
    policy, _cfg, _basis, payload, out = policy_run
    for inv in out:
        rel = abs(inv.i_cell.sum() - payload.ip_amperes) / payload.ip_amperes
        assert rel < 0.02, f"{policy}: Ip anchor off by {rel}"


def test_misfit_decreases(policy_run):
    policy, _cfg, _basis, _payload, out = policy_run
    for inv in out:
        trace = inv.misfit_trace
        assert trace is not None
        start = trace[:10].mean()
        end = trace[-10:].mean()
        assert end < start, f"{policy}: misfit did not improve ({start} -> {end})"


def test_candidate_mask_zeros_excluded_cells(policy_run):
    policy, _cfg, basis, _payload, out = policy_run
    mask = basis.candidate_mask.cpu().numpy()
    excluded = mask == 0
    if not excluded.any():
        pytest.skip("fixture geometry excludes no cells")
    for inv in out:
        assert np.all(inv.i_cell[excluded] == 0.0), policy


# --------------------------------------------------------------------------
# _lambda_schedule in isolation (no optimisation)
# --------------------------------------------------------------------------


def _row(value: float, n: int = 3) -> torch.Tensor:
    return torch.full((n,), float(value), dtype=torch.float64)


def test_lambda_schedule_fixed_is_constant():
    cfg = InverseConfig(policy="fixed", lambda_fb=7.0, iters=100)
    lam = torch.zeros(3, dtype=torch.float64)
    for step in (0, 1, 50, 99):
        lam = _lambda_schedule(cfg, step, lam, _row(0.5), _row(float("inf")))
        torch.testing.assert_close(lam, _row(7.0))


def test_lambda_schedule_warm_start_switches_at_warmup_end():
    cfg = InverseConfig(
        policy="warm-start", lambda_fb=5.0, iters=100, warmup_fraction=0.25
    )
    warmup_end = int(cfg.warmup_fraction * cfg.iters)
    lam = torch.zeros(3, dtype=torch.float64)
    before = _lambda_schedule(cfg, warmup_end - 1, lam, _row(0.0), _row(float("inf")))
    torch.testing.assert_close(before, _row(0.0))
    at = _lambda_schedule(cfg, warmup_end, lam, _row(0.0), _row(float("inf")))
    torch.testing.assert_close(at, _row(5.0))
    after = _lambda_schedule(cfg, warmup_end + 1, lam, _row(0.0), _row(float("inf")))
    torch.testing.assert_close(after, _row(5.0))


def test_lambda_schedule_discrepancy_warmup_and_seed():
    cfg = InverseConfig(
        policy="discrepancy", lambda_fb=4.0, iters=100, warmup_fraction=0.25
    )
    warmup_end = int(cfg.warmup_fraction * cfg.iters)
    lam = torch.zeros(3, dtype=torch.float64)
    during = _lambda_schedule(cfg, warmup_end - 1, lam, _row(0.0), _row(float("inf")))
    torch.testing.assert_close(during, _row(0.0))
    seeded = _lambda_schedule(cfg, warmup_end, lam, _row(0.0), _row(float("inf")))
    torch.testing.assert_close(seeded, _row(4.0))


def test_lambda_schedule_discrepancy_adapts_up_and_down():
    cfg = InverseConfig(
        policy="discrepancy",
        lambda_fb=4.0,
        iters=200,
        warmup_fraction=0.25,
        adapt_every=25,
        adapt_factor=1.5,
        lambda_max=1e4,
    )
    warmup_end = int(cfg.warmup_fraction * cfg.iters)
    adapt_step = warmup_end + cfg.adapt_every
    target = _row(1.0)
    # misfit below target -> lambda scales UP by adapt_factor
    lam_up = _lambda_schedule(cfg, adapt_step, _row(4.0), _row(0.5), target)
    torch.testing.assert_close(lam_up, _row(4.0 * cfg.adapt_factor))
    # misfit above 1.2x target -> lambda scales DOWN by adapt_factor
    lam_down = _lambda_schedule(cfg, adapt_step, _row(4.0), _row(1.3), target)
    torch.testing.assert_close(lam_down, _row(4.0 / cfg.adapt_factor))
    # misfit within [target, 1.2x target] -> lambda unchanged even on cadence
    lam_same = _lambda_schedule(cfg, adapt_step, _row(4.0), _row(1.1), target)
    torch.testing.assert_close(lam_same, _row(4.0))
    # off-cadence step -> lambda unchanged regardless of misfit
    lam_off = _lambda_schedule(cfg, adapt_step + 1, _row(4.0), _row(0.1), target)
    torch.testing.assert_close(lam_off, _row(4.0))


def test_lambda_schedule_discrepancy_clamped_to_bounds():
    cfg = InverseConfig(
        policy="discrepancy",
        lambda_fb=4.0,
        iters=200,
        warmup_fraction=0.25,
        adapt_every=25,
        adapt_factor=1.5,
        lambda_max=10.0,
    )
    warmup_end = int(cfg.warmup_fraction * cfg.iters)
    adapt_step = warmup_end + cfg.adapt_every
    target = _row(1.0)
    lam_hi = _lambda_schedule(cfg, adapt_step, _row(9.0), _row(0.1), target)
    torch.testing.assert_close(lam_hi, _row(10.0))
    lam_lo = _lambda_schedule(
        cfg, adapt_step, _row(cfg.lambda_fb / cfg.lambda_max), _row(5.0), target
    )
    torch.testing.assert_close(lam_lo, _row(cfg.lambda_fb / cfg.lambda_max))


def test_lambda_schedule_unknown_policy_raises():
    cfg = InverseConfig(policy="not-a-policy", iters=10)
    with pytest.raises(ValueError, match="unknown weight policy"):
        _lambda_schedule(cfg, 0, _row(0.0), _row(0.0), _row(float("inf")))


def test_invert_slices_unknown_policy_raises():
    cfg = InverseConfig(policy="not-a-policy", iters=10)
    with pytest.raises(ValueError, match="unknown weight policy"):
        invert_slices(None, [], cfg, device="cpu")  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# physics regularisation actually moves the structure residual
# --------------------------------------------------------------------------


def test_structure_regularisation_reduces_residual(synth_problem):
    basis, payload, _ip = synth_problem
    cfg_reg = InverseConfig(iters=200, policy="fixed", lambda_fb=10.0)
    cfg_noreg = InverseConfig(iters=200, policy="fixed", lambda_fb=0.0)
    reg = invert_slices(basis, [payload], cfg_reg, device="cpu")[0]
    noreg = invert_slices(basis, [payload], cfg_noreg, device="cpu")[0]
    assert reg.structure < noreg.structure


# --------------------------------------------------------------------------
# physics priors: unidirectional current + free-boundary support consistency
# --------------------------------------------------------------------------


from imas_ambix.latent.patch_inverse import (  # noqa: E402
    negative_current_fraction,
    outside_current_fraction,
    support_outside_mask,
)


@pytest.fixture(scope="module")
def noisy_payload(synth_problem):
    """The synthetic payload with seeded multiplicative sensor noise.

    The clean payload is self-consistent, so an unregularised inverse has
    little incentive to fill the null space; 5 % relative noise gives the
    sensor term something to over-fit, which the free (λ=0) inverse absorbs
    with sign-indefinite null-space current.
    """
    _basis, payload, ip = synth_problem
    rng = np.random.default_rng(7)
    meas = payload.measured * (1.0 + 0.05 * rng.standard_normal(payload.measured.size))
    noisy = SlicePayload(
        measured=meas,
        vacuum=payload.vacuum,
        mask=payload.mask,
        scale=payload.scale,
        i_pf=payload.i_pf,
        ip_amperes=payload.ip_amperes,
        shot=2,
        t_index=0,
    )
    return noisy, ip


def test_priors_off_by_default():
    cfg = InverseConfig()
    assert cfg.sign_prior is None
    assert cfg.support_prior is False


def test_negative_fraction_helper():
    ip = 1.0e5
    i_cell = np.array([6.0e4, 5.0e4, -1.0e4])
    assert negative_current_fraction(i_cell, ip) == pytest.approx(0.1)
    # sign-aware: a negative-Ip plasma counts anti-parallel (positive) cells
    assert negative_current_fraction(-i_cell, -ip) == pytest.approx(0.1)
    assert negative_current_fraction(np.array([1.0, 2.0]), 3.0) == 0.0


def test_sign_prior_softplus_enforces_unidirectional(noisy_payload, synth_problem):
    basis, _payload, _ip = synth_problem
    payload, ip = noisy_payload
    cfg = InverseConfig(iters=300, policy="fixed", lambda_fb=0.0, sign_prior="softplus")
    inv = invert_slices(basis, [payload], cfg, device="cpu")[0]
    assert np.all(inv.i_cell * np.sign(ip) >= 0.0)
    assert inv.negative_fraction == 0.0
    assert inv.misfit < 1.0  # noise floor; must still explain the sensors
    assert inv.ip_rel_err < 0.02


def test_sign_prior_penalty_suppresses_negative_current(noisy_payload, synth_problem):
    basis, _payload, _ip = synth_problem
    payload, ip = noisy_payload
    cfg_off = InverseConfig(iters=300, policy="fixed", lambda_fb=0.0)
    cfg_pen = InverseConfig(
        iters=300, policy="fixed", lambda_fb=0.0, sign_prior="penalty", sign_weight=50.0
    )
    off = invert_slices(basis, [payload], cfg_off, device="cpu")[0]
    pen = invert_slices(basis, [payload], cfg_pen, device="cpu")[0]
    # precondition: the free inverse actually fills the null space sign-indefinitely
    assert off.negative_fraction > 1.0e-4
    assert pen.negative_fraction < 0.2 * off.negative_fraction


def test_negative_fraction_matches_currents(noisy_payload, synth_problem):
    basis, _payload, _ip = synth_problem
    payload, ip = noisy_payload
    cfg = InverseConfig(iters=200, policy="fixed", lambda_fb=0.0)
    inv = invert_slices(basis, [payload], cfg, device="cpu")[0]
    assert inv.negative_fraction == pytest.approx(
        negative_current_fraction(inv.i_cell, ip)
    )


def test_sign_prior_unknown_value_raises(synth_problem):
    basis, payload, _ip = synth_problem
    cfg = InverseConfig(iters=10, sign_prior="not-a-prior")
    with pytest.raises(ValueError, match="sign_prior"):
        invert_slices(basis, [payload], cfg, device="cpu")


def test_support_prior_bounds_outside_current(noisy_payload, synth_problem):
    basis, _payload, _ip = synth_problem
    payload, ip = noisy_payload
    table = _confining_table()
    lim_r = np.asarray(table.limiter_r, dtype=np.float64)
    lim_z = np.asarray(table.limiter_z, dtype=np.float64)
    budget = 0.05
    cfg_off = InverseConfig(
        iters=300, policy="fixed", lambda_fb=0.0, limiter_r=lim_r, limiter_z=lim_z
    )
    cfg_on = InverseConfig(
        iters=400,
        policy="fixed",
        lambda_fb=0.0,
        support_prior=True,
        support_weight=1000.0,
        halo_budget=budget,
        limiter_r=lim_r,
        limiter_z=lim_z,
    )
    off = invert_slices(basis, [payload], cfg_off, device="cpu")[0]
    on = invert_slices(basis, [payload], cfg_on, device="cpu")[0]
    # outside_fraction is a reported diagnostic whenever the limiter is given
    assert np.isfinite(off.outside_fraction)
    assert np.isfinite(on.outside_fraction)
    assert on.outside_fraction < off.outside_fraction
    assert on.outside_fraction <= budget + 0.03
    assert on.support_excess == pytest.approx(
        max(0.0, on.outside_fraction - budget), abs=1.0e-9
    )
    assert on.misfit < 1.5  # support must not destroy the sensor fit


def test_doublet_configuration_survives_both_priors(synth_problem):
    """Two same-sign lobes inside one flux envelope cost the priors ~nothing.

    The unidirectional prior sees no anti-parallel current, and the LCFS of
    the doublet's own ψ envelops both lobes, so the support penalty inside
    the halo budget is zero — doublets stay representable by construction.
    """
    basis, _payload, ip = synth_problem
    table = _confining_table()
    r_c = basis.r_cells.cpu().numpy()
    z_c = basis.z_cells.cpu().numpy()
    cand = basis.candidate_mask.cpu().numpy() > 0
    lobes = np.exp(-(((r_c - basis.r0) / 0.12) ** 2) - ((z_c - 0.22) / 0.10) ** 2)
    lobes += np.exp(-(((r_c - basis.r0) / 0.12) ** 2) - ((z_c + 0.22) / 0.10) ** 2)
    # compact support: a physical doublet carries no current outside its own
    # separatrix, so the fixture truncates the Gaussian tails
    lobes[lobes < 0.05 * lobes.max()] = 0.0
    i_doublet = lobes * cand
    i_doublet = i_doublet / i_doublet.sum() * ip
    assert negative_current_fraction(i_doublet, ip) == 0.0
    outside = support_outside_mask(
        basis,
        i_doublet,
        np.zeros(0),
        limiter_r=np.asarray(table.limiter_r, dtype=np.float64),
        limiter_z=np.asarray(table.limiter_z, dtype=np.float64),
    )
    assert outside_current_fraction(i_doublet, ip, outside) < 0.05
