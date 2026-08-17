"""Loud failure checks for calculus, causality, and integral policy."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from imas_ambix.fluxstate import (
    AnalysisHandoff,
    DerivativeConsistencyError,
    FixedLagSmoothing,
    FluxFunctionState,
    ForecastHandoff,
    FreeAmplitude,
    FullSequenceSmoothing,
    HandoffEligibilityError,
    IntegralConstraintPolicy,
    IntegralMoment,
    IntegralPolicyKind,
    IntegralTarget,
    require_online_handoff,
)


def test_corrupted_pressure_derivative_fails_with_numeric_receipt(
    flux_state: FluxFunctionState,
):
    payload = flux_state.to_dict()
    payload["dpressure_dpsi"][4] += 500.0

    with pytest.raises(
        DerivativeConsistencyError,
        match=r"pressure derivative disagrees at index 4.*error=.*allowed=",
    ):
        FluxFunctionState.from_dict(payload)


def test_corrupted_field_drive_fails_loudly(flux_state: FluxFunctionState):
    payload = flux_state.to_dict()
    payload["f_df_dpsi"][3] *= -1.0

    with pytest.raises(
        DerivativeConsistencyError,
        match=r"F F' derivative disagrees at index 3",
    ):
        FluxFunctionState.from_dict(payload)


def test_forecast_and_analysis_are_type_distinct_from_smoothing():
    valid = datetime(2026, 8, 17, 12, 0, tzinfo=UTC)
    forecast = ForecastHandoff(
        valid_at=valid,
        information_cutoff=valid - timedelta(milliseconds=10),
        produced_at=valid - timedelta(milliseconds=5),
    )
    analysis = AnalysisHandoff(
        valid_at=valid,
        information_cutoff=valid,
        produced_at=valid + timedelta(milliseconds=1),
    )
    fixed = FixedLagSmoothing(
        valid_at=valid,
        information_cutoff=valid + timedelta(milliseconds=20),
        produced_at=valid + timedelta(milliseconds=25),
        lag_seconds=0.050,
    )
    full = FullSequenceSmoothing(
        valid_at=valid,
        information_cutoff=valid + timedelta(seconds=1),
        produced_at=valid + timedelta(seconds=2),
        sequence_start=valid - timedelta(seconds=1),
        sequence_end=valid + timedelta(seconds=1),
    )

    assert require_online_handoff(forecast) is forecast
    assert require_online_handoff(analysis) is analysis
    assert not isinstance(forecast, (FixedLagSmoothing, FullSequenceSmoothing))
    assert not isinstance(analysis, (FixedLagSmoothing, FullSequenceSmoothing))
    with pytest.raises(HandoffEligibilityError, match="cannot be an online handoff"):
        require_online_handoff(fixed)
    with pytest.raises(HandoffEligibilityError, match="cannot be an online handoff"):
        require_online_handoff(full)


def test_multimoment_policy_rejects_insufficient_physical_degrees_of_freedom():
    targets = (
        IntegralTarget(IntegralMoment.PLASMA_CURRENT, -800_000.0, 2_000.0, "A"),
        IntegralTarget(IntegralMoment.POLOIDAL_BETA, 0.8, 0.02, "1"),
    )

    with pytest.raises(
        ValueError,
        match="one physical amplitude per target",
    ):
        IntegralConstraintPolicy(
            kind=IntegralPolicyKind.MOMENT_CLOSURE,
            free_amplitudes=(FreeAmplitude("pressure_scale", "1", prior_mean=1.0),),
            targets=targets,
            prior_covariance=np.eye(1),
        )


def test_declared_multimoment_policy_preserves_prior_and_targets():
    policy = IntegralConstraintPolicy(
        kind=IntegralPolicyKind.MOMENT_CLOSURE,
        free_amplitudes=(
            FreeAmplitude("pressure_scale", "1", prior_mean=1.0),
            FreeAmplitude("field_drive_scale", "1", prior_mean=1.0),
        ),
        targets=(
            IntegralTarget(IntegralMoment.PLASMA_CURRENT, -800_000.0, 2_000.0, "A"),
            IntegralTarget(IntegralMoment.POLOIDAL_BETA, 0.8, 0.02, "1"),
        ),
        prior_covariance=np.asarray([[0.04, 0.01], [0.01, 0.09]]),
    )

    assert policy.kind is IntegralPolicyKind.MOMENT_CLOSURE
    assert [item.name for item in policy.free_amplitudes] == [
        "pressure_scale",
        "field_drive_scale",
    ]
    assert [item.moment for item in policy.targets] == [
        IntegralMoment.PLASMA_CURRENT,
        IntegralMoment.POLOIDAL_BETA,
    ]
    np.testing.assert_array_equal(
        policy.prior_covariance,
        np.asarray([[0.04, 0.01], [0.01, 0.09]]),
    )
