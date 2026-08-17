"""Producer-adapter evidence at the deterministic Nova call boundary."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from imas_ambix.fluxstate import (
    REVIEWED_CURRENT_DIFFUSION_REVISION,
    FixedLagSmoothing,
    FreeAmplitude,
    HandoffEligibilityError,
    IntegralConditioningError,
    IntegralConditioningRequest,
    IntegralConstraintConditioner,
    IntegralMoment,
    IntegralTarget,
    LearnedClosureCorrection,
    SequentialEstimatorAdapter,
    SequentialEstimatorBatch,
    SequentialProductKind,
    TransportForwardAdapter,
    TransportForwardResult,
)


class _TransportGeometry:
    def __init__(self, f_face: np.ndarray):
        self.f_face = f_face
        self.flux_sign = 1.0


class _CurrentDiffusion:
    def __init__(self, initial_flux: np.ndarray, f_face: np.ndarray):
        self.geometry = _TransportGeometry(f_face)
        self.initial_flux = initial_flux
        self.calls = 0

    def evolve(
        self,
        t_grid: np.ndarray,
        ip_of_t: np.ndarray,
        psi0_face: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        self.calls += 1
        initial = self.initial_flux if psi0_face is None else psi0_face
        radial_shift = np.linspace(0.0, 0.02, initial.size)
        fraction = (t_grid - t_grid[0]) / (t_grid[-1] - t_grid[0])
        history = initial[None, :] + fraction[:, None] * radial_shift[None, :]
        return {
            "psi_face": history,
            "psidot_face": radial_shift / (t_grid[-1] - t_grid[0]),
            "v_axis": np.zeros_like(t_grid),
            "v_bdry": np.full_like(t_grid, radial_shift[-1]),
        }


def _ensemble_members(flux_state, handoff):
    return tuple(
        replace(
            flux_state,
            ensemble=replace(flux_state.ensemble, active_member=index),
            handoff=handoff,
        )
        for index in range(len(flux_state.ensemble.member_ids))
    )


def test_transport_forward_adapter_round_trip_preserves_receipt_and_state(flux_state):
    time_grid = np.asarray([0.0, 0.01, 0.02])
    current = np.asarray([760_000.0, 770_000.0, 780_000.0])
    field_function = 1.9 + 0.08 * flux_state.coordinate.total_flux_wb
    transport = _CurrentDiffusion(flux_state.coordinate.total_flux_wb, field_function)
    correction = LearnedClosureCorrection(
        pressure_delta_pa=np.linspace(20.0, -20.0, flux_state.pressure.size),
        f_delta_tm=np.linspace(0.002, -0.002, flux_state.f.size),
        model_digest="sha256:learned-closure",
    )

    result = TransportForwardAdapter().evolve(
        transport,
        time_grid_s=time_grid,
        requested_current_a=current,
        source_state=flux_state,
        handoff=flux_state.handoff,
        correction=correction,
    )
    restored = TransportForwardResult.from_dict(result.to_dict())

    assert transport.calls == 1
    assert restored.receipt.nova_revision == REVIEWED_CURRENT_DIFFUSION_REVISION
    assert restored.receipt.correction_digest == "sha256:learned-closure"
    assert restored.state.handoff == flux_state.handoff
    assert restored.state.ensemble == flux_state.ensemble
    assert restored.state.integral_ledger.plasma_current.achieved == -780_000.0
    assert any(
        item == f"Nova CurrentDiffusion@{REVIEWED_CURRENT_DIFFUSION_REVISION}"
        for item in restored.state.provenance.source_diagnostics
    )
    np.testing.assert_allclose(
        restored.state.coordinate.total_flux_wb,
        restored.receipt.final_total_flux_wb,
    )
    np.testing.assert_array_equal(restored.state.pressure, result.state.pressure)
    np.testing.assert_array_equal(restored.state.f, result.state.f)


def test_sequential_adapter_round_trip_keeps_all_members_and_forecast_label(
    flux_state,
):
    batch = SequentialEstimatorAdapter().batch(
        _ensemble_members(flux_state, flux_state.handoff)
    )
    restored = SequentialEstimatorBatch.from_dict(batch.to_dict())

    assert restored.kind is SequentialProductKind.FORECAST
    assert restored.online_handoff() is restored
    assert len(restored.members) == 3
    assert [item.ensemble.active_member for item in restored.members] == [0, 1, 2]
    assert [item.ensemble.member_id for item in restored.members] == [
        "member-a",
        "member-b",
        "member-c",
    ]


def test_sequential_adapter_rejects_smoothed_batch_at_online_handoff(flux_state):
    valid = datetime(2026, 8, 17, 12, 0, tzinfo=UTC)
    smoothing = FixedLagSmoothing(
        valid_at=valid,
        information_cutoff=valid + timedelta(milliseconds=20),
        produced_at=valid + timedelta(milliseconds=25),
        lag_seconds=0.05,
    )
    batch = SequentialEstimatorAdapter().batch(_ensemble_members(flux_state, smoothing))

    assert batch.kind is SequentialProductKind.FIXED_LAG_SMOOTHING
    with pytest.raises(HandoffEligibilityError, match="cannot be an online handoff"):
        batch.online_handoff()


def test_conditioner_accepts_unique_one_amplitude_current_closure():
    target = IntegralTarget(
        moment=IntegralMoment.PLASMA_CURRENT,
        value=-800_000.0,
        tolerance=2_000.0,
        units="A",
    )
    request = IntegralConditioningRequest(
        free_amplitudes=(
            FreeAmplitude(name="current_profile_scale", units="1", prior_mean=1.0),
        ),
        targets=(target,),
        prior_covariance=np.asarray([[0.04]]),
    )

    result = IntegralConstraintConditioner().condition(
        request,
        achieved_moments={IntegralMoment.PLASMA_CURRENT: -780_000.0},
        jacobian=np.asarray([[-20_000.0]]),
    )

    assert result.policy.free_amplitudes == request.free_amplitudes
    assert result.policy.targets == request.targets
    assert result.jacobian_rank == 1
    assert result.jacobian_condition == pytest.approx(1.0)
    np.testing.assert_allclose(result.amplitude_delta, [1.0])
    np.testing.assert_allclose(result.posterior_amplitudes, [2.0])
    np.testing.assert_allclose(result.predicted_moments, [-800_000.0])
    np.testing.assert_allclose(result.residuals, [0.0])
    assert result.posterior_covariance.shape == (1, 1)


def test_conditioner_rejects_two_moments_without_two_amplitudes_and_prior():
    request = IntegralConditioningRequest(
        free_amplitudes=(
            FreeAmplitude(name="pressure_scale", units="1", prior_mean=1.0),
        ),
        targets=(
            IntegralTarget(IntegralMoment.PLASMA_CURRENT, -800_000.0, 2_000.0, "A"),
            IntegralTarget(IntegralMoment.POLOIDAL_BETA, 0.8, 0.02, "1"),
        ),
        prior_covariance=None,
    )

    with pytest.raises(IntegralConditioningError) as caught:
        IntegralConstraintConditioner().condition(
            request,
            achieved_moments={
                IntegralMoment.PLASMA_CURRENT: -780_000.0,
                IntegralMoment.POLOIDAL_BETA: 0.72,
            },
            jacobian=np.asarray([[-20_000.0], [0.05]]),
        )

    message = str(caught.value)
    assert "one named profile degree of freedom is required per moment" in message
    assert "an Ambix prior covariance is required" in message
