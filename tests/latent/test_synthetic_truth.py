"""Standing gate for the synthetic-truth identifiability harness.

The fast subset (default) manufactures equilibria on the analytic confining
fixture (plasma self-field — no attractor, sub-second solves) and pins the
three properties the harness relies on: the generator round-trips a known β0 at
zero noise, the injected noise sits at the requested whitening floor, and an
injected calibration corruption is EXPOSED by the physics arm (its whitened
cost rises) while the free inverse absorbs it.  The basin contrast (scout /
warm-start reaches the confined branch where a fixed-shape cold seed drifts to
the outboard attractor) is inherently a MAST-geometry behaviour and is marked
``slow`` — it loads real shot geometry and runs several full-resolution solves.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from imas_ambix.latent import synthetic_truth as st
from tests.latent.test_gs_solve import _confining_table

_SCALE = np.full(5, 1.0e-3)  # explicit whitening floor for the 5-probe fixture


@pytest.fixture(scope="module", autouse=True)
def single_threaded_torch():
    """Keep synthetic solves independent of host OpenMP oversubscription."""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def fixture_campaign() -> st.Campaign:
    """A fast self-confining campaign (analytic fixture, coarse grid)."""
    return st.build_campaign(table=_confining_table(), nr=33, nz=45, scale=_SCALE)


def test_generator_roundtrip_recovers_beta0_at_zero_noise(fixture_campaign):
    """A KNOWN β0 manufactured at zero noise is recovered by the closure fit."""
    from imas_ambix.latent.gs_solve import fit_profile_continuous

    truth = st.manufacture(
        fixture_campaign, beta0=0.7, alpha=1.0, i_pf=None, noise=False
    )
    assert truth.confined and truth.converged
    p = truth.to_payload()
    fit = fit_profile_continuous(
        fixture_campaign.grid,
        fixture_campaign.table,
        i_pf=p.i_pf,
        ip_amperes=p.ip_amperes,
        measured=p.measured,
        vacuum_prediction=p.vacuum,
        sensor_scale=p.scale,
        sensor_mask=p.mask,
        x0=(0.6, 1.0),
        alpha_bounds=(0.5, 2.0),
        maxfev=30,
    )
    assert fit is not None and fit.result.converged
    assert abs(fit.beta0 - 0.7) < 0.12  # recovered generating β0
    assert abs(fit.alpha - 1.0) < 0.2


def test_noise_drawn_at_measured_floor(fixture_campaign):
    """The injected noise standard deviation equals the per-channel whitening
    floor: pooled ``noise/scale`` has unit standard deviation."""
    pooled = []
    for s in range(12):
        t = st.manufacture(
            fixture_campaign, beta0=0.6, alpha=1.0, i_pf=None, noise=True, seed=s
        )
        pooled.append(t.noise / t.scale)
    std = float(np.std(np.concatenate(pooled)))
    assert 0.75 < std < 1.35
    # noise=False is exactly the clean field (no calibration ⇒ measured==clean)
    clean = st.manufacture(
        fixture_campaign, beta0=0.6, alpha=1.0, i_pf=None, noise=False
    )
    np.testing.assert_allclose(clean.measured, clean.measured_clean, atol=1e-12)


def test_calibration_corruption_exposed_by_physics_arm(fixture_campaign):
    """An injected per-channel offset is EXPOSED by the physics (closure) arm —
    its whitened cost rises because the corrupted magnetics are no longer
    consistent with any force-balanced state — while the free inverse absorbs
    it into currents (a smoke of the aliasing distinction)."""
    from imas_ambix.latent.gs_solve import fit_profile_continuous
    from imas_ambix.latent.patch_inverse import InverseConfig, invert_slices

    n_ch = len(fixture_campaign.channels)
    offset = np.zeros(n_ch)
    offset[:3] = 4.0 * fixture_campaign.scale[:3]  # 4σ static corruption

    def closure_cost(off):
        t = st.manufacture(
            fixture_campaign, beta0=0.6, alpha=1.0, i_pf=None, noise=False, offsets=off
        )
        p = t.to_payload()
        fit = fit_profile_continuous(
            fixture_campaign.grid,
            fixture_campaign.table,
            i_pf=p.i_pf,
            ip_amperes=p.ip_amperes,
            measured=p.measured,
            vacuum_prediction=p.vacuum,
            sensor_scale=p.scale,
            sensor_mask=p.mask,
            x0=(0.6, 1.0),
            maxfev=25,
        )
        return fit.cost

    cost_clean = closure_cost(np.zeros(n_ch))
    cost_corrupt = closure_cost(offset)
    assert cost_corrupt > cost_clean  # the physics arm cannot hide the offset

    # the free inverse still fits (absorbs the corruption into currents)
    t = st.manufacture(
        fixture_campaign, beta0=0.6, alpha=1.0, i_pf=None, noise=False, offsets=offset
    )
    p = t.to_payload()
    inv = invert_slices(fixture_campaign.basis, [p], InverseConfig(iters=200))
    assert np.isfinite(inv[0].misfit)


def test_passive_injection_moves_sensors(fixture_campaign):
    """Injected passive-circuit currents add a finite signal to the sensors —
    the passive-recovery path has something to recover."""
    n_pass = fixture_campaign.n_passive
    assert n_pass >= 1
    base = st.manufacture(
        fixture_campaign, beta0=0.6, alpha=1.0, i_pf=None, noise=False
    )
    pa = np.zeros(n_pass)
    pa[0] = 3.0e4
    tp = st.manufacture(
        fixture_campaign,
        beta0=0.6,
        alpha=1.0,
        i_pf=None,
        noise=False,
        passive_amplitudes=pa,
    )
    delta = np.abs(tp.measured_clean - base.measured_clean) / fixture_campaign.scale
    assert float(np.max(delta)) > 0.1  # a detectable passive signature


def test_rotation_injection_adds_r4_structure(fixture_campaign):
    """A non-zero rotation γ0 changes the manufactured current (the centrifugal
    R⁴ term) — the field the rotation residual arm must detect."""
    base = st.manufacture(
        fixture_campaign, beta0=0.6, alpha=1.0, i_pf=None, noise=False
    )
    rot = st.manufacture(
        fixture_campaign, beta0=0.6, alpha=1.0, gamma0=0.7, i_pf=None, noise=False
    )
    # same Ip, but the current redistributes outward under rotation
    np.testing.assert_allclose(
        base.cell_currents.sum(), rot.cell_currents.sum(), rtol=1e-4
    )
    assert np.linalg.norm(rot.cell_currents - base.cell_currents) > 1e-6 * abs(
        base.cell_currents.sum()
    )


def test_z_symmetric_pins_the_midplane_branch(fixture_campaign):
    """``z_symmetric`` makes the manufactured equilibrium exactly up-down
    symmetric — the branch pin a chained truth family relies on where the
    plain Picard would amplify infinitesimal asymmetries near a vertical
    instability."""
    truth = st.manufacture(
        fixture_campaign,
        beta0=0.6,
        alpha=1.0,
        i_pf=None,
        noise=False,
        z_symmetric=True,
    )
    assert truth.confined
    assert abs(truth.axis[1]) < 2e-2  # axis on the midplane (grid-resolution)
    j2d = np.zeros(fixture_campaign.grid.flat_r.size)
    j2d[fixture_campaign.grid.cells] = truth.cell_currents
    j2d = j2d.reshape(fixture_campaign.grid.nz, fixture_campaign.grid.nr)
    asym = np.abs(j2d - j2d[::-1, :]).sum() / max(np.abs(j2d).sum(), 1e-30)
    assert asym < 1e-9  # current density exactly symmetric by construction


def test_firewall_no_evaluator_imports():
    """The generator must not import anything from the EFIT / evaluator side."""
    from pathlib import Path

    import imas_ambix.latent.synthetic_truth as m

    src = Path(m.__file__).read_text()
    for banned in ("efit_referee", "equilibrium_labels", "worldmodel", "ref_target"):
        assert banned not in src, f"synthetic_truth imports the firewalled {banned}"


@pytest.mark.slow
def test_basin_scout_reaches_confined_where_fixed_shape_drifts():
    """MAST geometry: a fixed-shape cold seed drifts to the outboard attractor
    for a fragile profile, while a free-sign scout / warm-start reaches the
    confined branch — the multi-branch fixed-point structure the harness maps.

    The contrast lives at a MARGINAL confining well.  The declared conductor
    elements carry winding turns in their Green's weights, so the corresponding
    current scale is about 0.085 of the acquisition-mesh scale.  A 4.4 kA well
    with a peaked broad profile sits between the cold and warm basins: the cold
    seed drifts outboard while the warm start holds the confined branch."""
    from imas_ambix.latent.gs_solve import solve_equilibrium

    vf_marginal = 4.4e3  # the well depth where both branches are reachable
    camp = st.build_campaign(18502, nr=49, nz=65)
    i_pf = st.build_confining_i_pf(camp.fwd, vf_marginal)
    ip = st.DEFAULT_IP_AMPERES
    warm, warm_axr = st.confined_seed(camp, vf_strength=vf_marginal)
    assert warm_axr <= st._CONFINED_AXIS_R_MAX  # the confining field holds a branch

    b, a = 0.75, 2.0  # a basin-fragile (peaked, broad-pressure) profile
    cold = solve_equilibrium(
        camp.grid, i_pf, ip, beta0=b, alpha=a, seed_width=(0.2, 0.35)
    )
    warm_solve = solve_equilibrium(
        camp.grid, i_pf, ip, beta0=b, alpha=a, initial_jphi=warm, relax=0.3
    )
    assert cold.axis[0] > st._CONFINED_AXIS_R_MAX  # fixed-shape cold seed → attractor
    assert warm_solve.axis[0] <= st._CONFINED_AXIS_R_MAX  # warm-start → confined branch
