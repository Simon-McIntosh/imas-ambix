"""Tests for the GS-readout-vs-referee scoring core (the gate-2 skill).

The gate scores the topology READ from the model's solved ψ against the
firewalled EFIT referee, per-quantity, exactly as the absolute-magnetics oracle
does: ``skill = 1 − RMSE_model / RMSE_baseline`` (baseline = train-mean
predictor).  The X-point is an order-invariant null set, so its error is a
PERMUTATION-INVARIANT match of the ≤2 predicted slots to the ≤2 reference slots.
"""

from __future__ import annotations

import numpy as np

from imas_ambix.latent.evaluate import (
    gs_inverse_theta,
    matched_xpoint_error,
    per_quantity_skill,
)


def test_matched_xpoint_error_is_permutation_invariant():
    pred = np.array([[1.0, 0.5], [1.2, -0.5]])
    ref_a = np.array([[1.0, 0.5], [1.2, -0.5]])
    ref_b = np.array([[1.2, -0.5], [1.0, 0.5]])  # swapped slots
    e_a = matched_xpoint_error(pred, ref_a)
    e_b = matched_xpoint_error(pred, ref_b)
    np.testing.assert_allclose(e_a, e_b)  # swap must not change the error
    np.testing.assert_allclose(e_a, 0.0, atol=1e-12)


def test_matched_xpoint_error_handles_absent_slots():
    pred = np.array([[1.0, 0.5], [np.nan, np.nan]])  # one predicted null
    ref = np.array([[1.05, 0.55], [np.nan, np.nan]])  # one reference null
    err = matched_xpoint_error(pred, ref)
    assert np.isfinite(err)
    np.testing.assert_allclose(err, np.hypot(0.05, 0.05), atol=1e-9)


def test_per_quantity_skill_matches_oracle_formula():
    """skill_i = 1 − RMSE_model_i / RMSE_baseline_i, per component."""
    # two samples, axis_R only for simplicity
    model = np.array([[1.01], [0.99]])
    ref = np.array([[1.00], [1.00]])
    baseline = np.array([[1.10], [0.90]])  # a worse (train-mean-like) predictor
    names = ["axis_R"]
    skill = per_quantity_skill(model, ref, baseline, names)
    rmse_m = np.sqrt(np.mean((model - ref) ** 2))
    rmse_b = np.sqrt(np.mean((baseline - ref) ** 2))
    np.testing.assert_allclose(skill["axis_R"], 1.0 - rmse_m / rmse_b, rtol=1e-9)
    assert skill["axis_R"] > 0  # model beats baseline


def test_per_quantity_skill_respects_mask():
    """A component with no finite reference yields NaN skill, not a crash."""
    model = np.array([[1.0, np.nan], [1.0, np.nan]])
    ref = np.array([[1.0, np.nan], [1.0, np.nan]])
    baseline = np.array([[1.5, np.nan], [0.5, np.nan]])
    names = ["axis_R", "axis_Z"]
    skill = per_quantity_skill(model, ref, baseline, names)
    assert np.isnan(skill["axis_Z"])  # no finite reference → undefined skill
    assert np.isfinite(skill["axis_R"])


def test_gs_inverse_theta_recovers_known_profile():
    """The ridge GS-inverse recovers the θ that generated the raw magnetics."""
    from imas_ambix.gs import geometry as gsg
    from imas_ambix.latent.gs_observation import GSObservation

    # a synthetic campaign with enough sensors to identify 3 profile DOF
    probes = [
        gsg.BProbe(
            index=i,
            r=1.4 + 0.02 * i,
            z=-0.4 + 0.16 * i,
            angle_deg=90.0 * (i % 2),
            length=0.02,
        )
        for i in range(8)
    ]
    sensor_map = [
        gsg.SensorMapping(f"obv{i:02d}", "b_probe", i, p.r, p.z, p.angle_deg, 0.001, "")
        for i, p in enumerate(probes)
    ]
    table = gsg.GeometryTable(
        signature=gsg.SetupSignature(
            n_bprobe=8, n_fluxloop=0, n_pf_filament=1, n_limiter=4, digest="eee0"
        ),
        shots=[1],
        b_probes=probes,
        flux_loops=[],
        pf_filaments=[
            gsg.PFFilament(
                r=1.5, z=1.1, turns=1.0, width=0.01, height=0.01, circuit=1, xmult=1.0
            )
        ],
        limiter_r=[0.3, 1.6, 1.6, 0.3],
        limiter_z=[-1.0, -1.0, 1.0, 1.0],
        sensor_map=sensor_map,
        passive_structures=[
            gsg.PassiveStructure(name="w", r=2.0, z=0.0, obsolete=False)
        ],
        amc_current_channels=["p4u_coil_current"],
        unmatched_amb=[],
    )
    gs = GSObservation.from_table(table, grid_nr=9, grid_nz=11, profile_order=1)
    a_plasma = gs.a_plasma.numpy()
    g_pf = gs.g_pf.numpy()
    s = a_plasma.shape[0]
    theta_true = np.array([[2.0, 0.7, -0.4], [1.0, -0.3, 0.2]])
    i_pf = np.zeros((2, g_pf.shape[1]))
    raw = theta_true @ a_plasma.T + i_pf @ g_pf.T
    mask = np.ones((2, s), dtype=bool)
    scale = np.abs(raw).mean(axis=0) + 1e-9
    theta = gs_inverse_theta(a_plasma, g_pf, raw, mask, i_pf, scale, ridge=1e-9)
    np.testing.assert_allclose(theta, theta_true, atol=1e-4)
