"""Nova-propagated ensemble conditioning with causal and smoothed products."""

from __future__ import annotations

import importlib.metadata
import json
import os
import resource
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

NOVA_REVISION = "de3277a3238513b81be04dbc0980030b200ce420"
SCHEMA = "nova-ensemble-estimator"
OBSERVATION_LABELS = (
    "enclosed_current_inner",
    "enclosed_current_mid",
    "enclosed_current_outer",
    "enclosed_current_edge",
)
_SCENARIO_STREAM = 1_390_691_986
_MEMBER_STREAM = 1_297_046_850
_BASELINE_STREAM = 1_112_496_978
_HORIZONS_MS = (10, 50, 100, 250)


class EstimatorFailure(RuntimeError):  # noqa: N818
    """Raised when numerical, runtime, physical, or artifact checks fail."""


@dataclass(frozen=True)
class EstimatorConfig:
    """Validated controls for one deterministic ensemble experiment."""

    members: int = 16
    seed: int = 17
    backend: str = "cpu"
    devices: int = 1
    sample_period_s: float = 0.001
    steps: int = 280
    fixed_lag_steps: int = 100
    correction_stride: int = 10
    observation_noise: float = 0.004
    topology: str = "circular-nested-flux-surfaces"
    dtype: str = "float64"
    member_offset: int = 0

    def __post_init__(self) -> None:
        if self.members < 4:
            raise ValueError("members must be at least four")
        if self.backend not in {"cpu", "gpu"}:
            raise ValueError("backend must be 'cpu' or 'gpu'")
        if self.devices < 1:
            raise ValueError("devices must be positive")
        if self.sample_period_s != 0.001:
            raise ValueError("the estimator contract requires a 1 kHz clock")
        if self.steps <= 250:
            raise ValueError("steps must cover the 250 ms forecast horizon")
        if not 1 <= self.fixed_lag_steps < self.steps:
            raise ValueError("fixed_lag_steps must lie inside the clock")
        if self.correction_stride < 1:
            raise ValueError("correction_stride must be positive")
        if self.observation_noise <= 0.0:
            raise ValueError("observation_noise must be positive")
        if self.topology != "circular-nested-flux-surfaces":
            raise ValueError(
                "the synthetic harness supports circular nested flux surfaces"
            )
        if self.dtype != "float64":
            raise ValueError("Nova and JAX must run explicitly in float64")
        if self.member_offset < 0:
            raise ValueError("member_offset cannot be negative")


@dataclass(frozen=True)
class RuntimeProvenance:
    """Exact dependency and accelerator identity for a result."""

    nova_version: str
    nova_revision: str
    jax_version: str
    backend: str
    available_devices: int
    selected_devices: int
    x64_enabled: bool
    dtype: str
    topology: str
    clock_hz: int


@dataclass(frozen=True)
class EstimatorResult:
    """Self-describing arrays, scorecard, and runtime provenance."""

    clock: np.ndarray
    truth: np.ndarray
    observations: np.ndarray
    causal_forecast: np.ndarray
    causal_analysis: np.ndarray
    fixed_lag_smoothing: np.ndarray
    full_sequence_smoothing: np.ndarray
    edited_actuator: np.ndarray
    nominal_actuator: np.ndarray
    equilibrium_flux: np.ndarray
    edited_equilibrium_flux: np.ndarray
    metrics: dict[str, Any]
    provenance: RuntimeProvenance
    camera_proxy: dict[str, Any]
    config: EstimatorConfig

    def validate(self) -> None:
        arrays = (
            self.clock,
            self.truth,
            self.observations,
            self.causal_forecast,
            self.causal_analysis,
            self.fixed_lag_smoothing,
            self.full_sequence_smoothing,
            self.edited_actuator,
            self.nominal_actuator,
            self.equilibrium_flux,
            self.edited_equilibrium_flux,
        )
        if not all(np.isfinite(value).all() for value in arrays):
            raise EstimatorFailure("result contains a non-finite value")
        if any(value.dtype != np.float64 for value in arrays):
            raise EstimatorFailure("every numerical product must remain float64")
        expected = self.causal_forecast.shape
        if expected != self.causal_analysis.shape:
            raise EstimatorFailure("causal product shapes differ")
        if self.fixed_lag_smoothing.shape != expected:
            raise EstimatorFailure("fixed-lag product shape differs")
        if self.full_sequence_smoothing.shape != expected:
            raise EstimatorFailure("full-sequence product shape differs")
        if self.truth.shape != (expected[0], expected[2]):
            raise EstimatorFailure("truth shape is incompatible with products")
        if self.observations.shape != self.truth.shape:
            raise EstimatorFailure("observation and truth shapes differ")
        flux_shape = self.equilibrium_flux.shape
        if len(flux_shape) != 3 or flux_shape[:2] != expected[:2]:
            raise EstimatorFailure(
                "equilibrium-flux shape is incompatible with products"
            )
        if flux_shape[2] < 3:
            raise EstimatorFailure("equilibrium flux requires multiple radial faces")
        if self.edited_equilibrium_flux.shape != flux_shape:
            raise EstimatorFailure("edited equilibrium-flux shape differs")
        if self.provenance.dtype != "float64" or not self.provenance.x64_enabled:
            raise EstimatorFailure("provenance does not establish float64 execution")
        if self.config.members != expected[1]:
            raise EstimatorFailure("configured member count does not match products")


def _runtime(config: EstimatorConfig) -> RuntimeProvenance:
    expected_platform = "cpu" if config.backend == "cpu" else "cuda"
    if "jax" not in sys.modules:
        os.environ["JAX_PLATFORMS"] = expected_platform
        if config.backend == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import jax  # noqa: PLC0415

    jax.config.update("jax_enable_x64", True)
    resolved = jax.default_backend()
    normalised = "gpu" if resolved in {"cuda", "gpu"} else resolved
    if normalised != config.backend:
        raise EstimatorFailure(
            f"requested backend {config.backend!r} resolved as {normalised!r}"
        )
    devices = jax.devices()
    if len(devices) < config.devices:
        raise EstimatorFailure(
            f"requested {config.devices} devices but only {len(devices)} are visible"
        )
    if not jax.config.x64_enabled:
        raise EstimatorFailure("JAX x64 is not enabled")

    distribution = importlib.metadata.distribution("nova-stella")
    direct_text = distribution.read_text("direct_url.json")
    if not direct_text:
        raise EstimatorFailure("nova-stella has no direct_url provenance")
    direct = json.loads(direct_text)
    revision = direct.get("vcs_info", {}).get("commit_id")
    if revision != NOVA_REVISION:
        raise EstimatorFailure(
            f"Nova revision {revision!r} does not match the validated pin"
        )
    return RuntimeProvenance(
        nova_version=distribution.version,
        nova_revision=revision,
        jax_version=jax.__version__,
        backend=normalised,
        available_devices=len(devices),
        selected_devices=config.devices,
        x64_enabled=bool(jax.config.x64_enabled),
        dtype="float64",
        topology=config.topology,
        clock_hz=1000,
    )


def _geometry() -> Any:
    from imas_ambix.physics import flux_surface_geometry_from_mapping  # noqa: PLC0415

    radial_cells = 20
    rho_face = np.linspace(0.0, 1.0, radial_cells + 1, dtype=np.float64)
    rho_cell = 0.5 * (rho_face[:-1] + rho_face[1:])
    major_radius = 3.0
    minor_radius = 0.5
    toroidal_field = 2.0
    current = 5.0e5
    radius = minor_radius * rho_face
    toroidal_flux = np.pi * minor_radius**2 * toroidal_field
    vpr_face = 4.0 * np.pi**2 * major_radius * minor_radius**2 * rho_face
    mapping = {
        "rho_face": rho_face,
        "rho_cell": rho_cell,
        "psi_face": np.zeros_like(rho_face),
        "psi_n_face": rho_face**2,
        "psi_n_cell": rho_cell**2,
        "vpr_face": vpr_face,
        "vpr_cell": 0.5 * (vpr_face[:-1] + vpr_face[1:]),
        "g2_face": 16.0 * np.pi**4 * radius**2,
        "g3_face": np.full_like(rho_face, 1.0 / major_radius**2),
        "g3_cell": np.full_like(rho_cell, 1.0 / major_radius**2),
        "f_face": np.full_like(rho_face, major_radius * toroidal_field),
        "f_cell": np.full_like(rho_cell, major_radius * toroidal_field),
        "b2_cell": np.full_like(rho_cell, toroidal_field**2),
        "inv_r_cell": np.full_like(rho_cell, 1.0 / major_radius),
        "phi_b": toroidal_flux,
        "r0": major_radius,
        "ip_amperes": current,
        "axis_psi": 0.0,
        "boundary_psi": 0.0,
        "volume": 2.0 * np.pi**2 * major_radius * minor_radius**2,
        "q_face": np.ones_like(rho_face),
        "flux_sign": 1.0,
    }
    empty = flux_surface_geometry_from_mapping(mapping)
    edge_gradient = empty.ip_edge_gradient(current)
    mapping["psi_face"] = 0.5 * edge_gradient * rho_face**2
    mapping["boundary_psi"] = float(mapping["psi_face"][-1])
    return flux_surface_geometry_from_mapping(mapping)


def _plan_current(clock: np.ndarray, nominal: float, *, edited: bool) -> np.ndarray:
    phase = clock / clock[-1]
    command = 1.0 + 0.025 * np.sin(4.0 * np.pi * phase)
    command += 0.11 / (1.0 + np.exp(-80.0 * (phase - 0.68)))
    if edited:
        command += 0.24 / (1.0 + np.exp(-90.0 * (phase - 0.55)))
    return np.asarray(nominal * command, dtype=np.float64)


def _observation(geometry: Any, psi_history: np.ndarray) -> np.ndarray:
    currents = np.stack([geometry.enclosed_current(row) for row in psi_history], axis=0)
    indices = np.rint(np.array([0.25, 0.5, 0.75, 1.0]) * (currents.shape[1] - 1))
    selected = currents[:, indices.astype(int)] / geometry.ip_amperes
    return np.asarray(selected, dtype=np.float64)


def _proper_score(samples: np.ndarray, target: np.ndarray) -> float:
    member_axis = 1
    count = samples.shape[member_axis]
    absolute = np.mean(np.abs(samples - target[:, None, :]))
    ordered = np.sort(samples, axis=member_axis)
    coefficients = 2.0 * np.arange(1, count + 1) - count - 1.0
    pair_term = (
        np.sum(ordered * coefficients[None, :, None], axis=member_axis) / count**2
    )
    return float(absolute - np.mean(pair_term))


def _coverage_and_sharpness(samples: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    low, high = np.quantile(samples, [0.05, 0.95], axis=1)
    coverage = np.mean((truth >= low) & (truth <= high))
    return {
        "coverage_90": float(coverage),
        "sharpness_90": float(np.mean(high - low)),
        "proper_score": _proper_score(samples, truth),
    }


def _scenario_random(seed: int) -> np.random.Generator:
    """Return randomness shared by every shard of one scenario."""
    return np.random.default_rng(np.random.SeedSequence([seed, _SCENARIO_STREAM]))


def _member_random(seed: int, global_member_index: int) -> np.random.Generator:
    """Return a stable stream for one member independent of shard grouping."""
    return np.random.default_rng(
        np.random.SeedSequence([seed, _MEMBER_STREAM, global_member_index])
    )


def _conventional_enkf(
    observations: np.ndarray,
    *,
    seed: int,
    members: int,
    observation_noise: float,
) -> np.ndarray:
    """Run an identity-observation random-walk ensemble Kalman filter."""
    random = np.random.default_rng(np.random.SeedSequence([seed, _BASELINE_STREAM]))
    steps, dimensions = observations.shape
    ensemble = observations[0] + random.normal(
        0.0,
        2.0 * observation_noise,
        size=(members, dimensions),
    )
    history = np.empty((steps, members, dimensions), dtype=np.float64)
    history[0] = ensemble
    observation_covariance = np.eye(dimensions) * observation_noise**2
    process_noise = 0.75 * observation_noise
    for index in range(1, steps):
        forecast = ensemble + random.normal(
            0.0,
            process_noise,
            size=ensemble.shape,
        )
        anomalies = forecast - forecast.mean(axis=0, keepdims=True)
        forecast_covariance = anomalies.T @ anomalies / (members - 1)
        gain = forecast_covariance @ np.linalg.inv(
            forecast_covariance + observation_covariance
        )
        perturbed_observation = observations[index] + random.normal(
            0.0,
            observation_noise,
            size=ensemble.shape,
        )
        ensemble = forecast + (perturbed_observation - forecast) @ gain.T
        history[index] = ensemble
    if not np.isfinite(history).all():
        raise EstimatorFailure("conventional EnKF produced a non-finite ensemble")
    return history


def _ensemble_scorecard(
    *,
    clock: np.ndarray,
    truth: np.ndarray,
    observations: np.ndarray,
    forecast: np.ndarray,
    analysis: np.ndarray,
    nominal_actuator: np.ndarray,
    edited_actuator: np.ndarray,
    seed: int,
    observation_noise: float,
) -> dict[str, Any]:
    """Compute every ensemble-dependent metric from its complete arrays."""
    steps, members, _dimensions = forecast.shape
    if analysis.shape != forecast.shape or nominal_actuator.shape != forecast.shape:
        raise EstimatorFailure("scorecard ensemble shapes differ")
    if edited_actuator.shape != forecast.shape:
        raise EstimatorFailure("edited-actuator ensemble shape differs")

    horizons: dict[str, Any] = {}
    for horizon in _HORIZONS_MS:
        anchors = np.arange(0, steps - horizon)
        horizon_samples = nominal_actuator[anchors + horizon] + (
            analysis[anchors] - nominal_actuator[anchors]
        )
        horizons[f"{horizon}_ms"] = _coverage_and_sharpness(
            horizon_samples,
            truth[anchors + horizon],
        )

    phase = clock / clock[-1]
    out_of_distribution = phase >= 0.68
    same_plan_spread = float(
        np.mean(np.std(nominal_actuator[out_of_distribution], axis=1, ddof=1))
    )
    in_distribution_spread = float(
        np.mean(np.std(nominal_actuator[~out_of_distribution], axis=1, ddof=1))
    )
    edited_displacement = float(
        np.mean(
            np.abs(
                edited_actuator[out_of_distribution].mean(axis=1)
                - nominal_actuator[out_of_distribution].mean(axis=1)
            )
        )
    )
    persistence = np.broadcast_to(observations[0], truth.shape)
    conventional = _conventional_enkf(
        observations,
        seed=seed,
        members=members,
        observation_noise=observation_noise,
    )
    return {
        "innovation_rmse": {
            "forecast": float(
                np.sqrt(np.mean((forecast.mean(axis=1) - observations) ** 2))
            ),
            "analysis": float(
                np.sqrt(np.mean((analysis.mean(axis=1) - observations) ** 2))
            ),
        },
        "proper_score": {
            "forecast": _proper_score(forecast, truth),
            "analysis": _proper_score(analysis, truth),
            "persistence": float(np.mean(np.abs(persistence - truth))),
            "conventional_enkf": _proper_score(conventional, truth),
        },
        "horizons": horizons,
        "uncertainty": {
            "in_distribution_spread": in_distribution_spread,
            "out_of_distribution_spread": same_plan_spread,
            "widening_ratio": same_plan_spread / max(in_distribution_spread, 1.0e-15),
        },
        "actuator_response": {
            "edited_displacement": edited_displacement,
            "same_plan_spread": same_plan_spread,
            "displacement_to_spread": edited_displacement
            / max(same_plan_spread, 1.0e-15),
        },
        "comparators": {
            "persistence": {
                "identity": "last_observation_persistence",
                "cohort": "same synthetic cohort",
            },
            "conventional_enkf": {
                "identity": "random_walk_ensemble_kalman_filter",
                "transition": "persistence_plus_gaussian_process_noise",
                "observation_operator": "identity",
                "cohort": "same synthetic cohort",
                "ensemble_shape": list(conventional.shape),
            },
            "torax_enkf": {
                "identity": "external reference only",
                "cohort": "not a same-cohort skill claim",
            },
        },
    }


def _smooth(
    analysis: np.ndarray,
    observations: np.ndarray,
    *,
    lag: int | None,
) -> np.ndarray:
    smoothed = analysis.copy()
    count = analysis.shape[0]
    for index in range(count - 1):
        stop = count if lag is None else min(count, index + lag + 1)
        future = np.arange(index + 1, stop)
        if future.size == 0:
            continue
        weights = np.exp(-(future - index) / max(1.0, 0.5 * future.size))
        residual = observations[future, None, :] - analysis[future]
        correction = np.average(residual, axis=0, weights=weights)
        smoothed[index] += 0.35 * correction
    return np.asarray(smoothed, dtype=np.float64)


class NovaEnsembleEstimator:
    """Advance every member with Nova and condition its observable state."""

    def __init__(self, config: EstimatorConfig) -> None:
        self.config = config

    def run(
        self,
        *,
        clock: np.ndarray | None = None,
        observation_perturbation: np.ndarray | None = None,
    ) -> EstimatorResult:
        started = time.perf_counter()
        provenance = _runtime(self.config)
        from imas_ambix.physics import (  # noqa: PLC0415
            EtaProfile,
            current_diffusion_from_mapping,
        )

        geometry = _geometry()
        times = (
            np.arange(self.config.steps, dtype=np.float64) * self.config.sample_period_s
            if clock is None
            else np.asarray(clock, dtype=np.float64)
        )
        if times.shape != (self.config.steps,):
            raise EstimatorFailure("clock shape does not match configured steps")
        intervals = np.diff(times)
        if not np.allclose(
            intervals,
            self.config.sample_period_s,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise EstimatorFailure("clock must be uniformly sampled at 1 kHz")

        nominal_current = _plan_current(times, geometry.ip_amperes, edited=False)
        edited_current = _plan_current(times, geometry.ip_amperes, edited=True)
        scenario_random = _scenario_random(self.config.seed)
        eta_log = np.empty(self.config.members, dtype=np.float64)
        eta_contrast = np.empty(self.config.members, dtype=np.float64)
        boundary_scale = np.empty(self.config.members, dtype=np.float64)
        member_noise = np.empty(
            (self.config.steps, self.config.members, len(OBSERVATION_LABELS)),
            dtype=np.float64,
        )
        for member in range(self.config.members):
            global_member = self.config.member_offset + member
            member_random = _member_random(self.config.seed, global_member)
            eta_log[member] = member_random.normal(np.log(7.0e-8), 0.23)
            eta_contrast[member] = np.clip(member_random.normal(1.8, 0.25), 0.2, 4.0)
            boundary_scale[member] = member_random.normal(1.0, 0.018)
            member_noise[:, member, :] = member_random.normal(
                0.0,
                self.config.observation_noise,
                size=(self.config.steps, len(OBSERVATION_LABELS)),
            )

        truth_solver = current_diffusion_from_mapping(
            asdict(geometry),
            EtaProfile(eta0=6.5e-8, contrast=1.65, shape=2.0),
            theta=1.0,
        )
        truth_solver.precision = "float64"
        truth_step = truth_solver.evolve(times, nominal_current)
        truth = _observation(geometry, truth_step["psi_face"])
        observation_noise = scenario_random.normal(
            0.0,
            self.config.observation_noise,
            size=truth.shape,
        )
        observations = truth + observation_noise
        if observation_perturbation is not None:
            perturbation = np.asarray(observation_perturbation, dtype=np.float64)
            if perturbation.shape != observations.shape:
                raise EstimatorFailure("observation perturbation has the wrong shape")
            observations = observations + perturbation

        reference_solver = current_diffusion_from_mapping(
            asdict(geometry),
            EtaProfile(eta0=7.0e-8, contrast=1.8, shape=2.0),
            theta=1.0,
        )
        reference_solver.precision = "float64"
        reference_step = reference_solver.evolve(times, nominal_current)
        reference_forecast = _observation(geometry, reference_step["psi_face"])

        nominal_members: list[np.ndarray] = []
        edited_members: list[np.ndarray] = []
        nominal_flux_members: list[np.ndarray] = []
        edited_flux_members: list[np.ndarray] = []
        ledgers: list[dict[str, float]] = []
        boundary_errors: list[float] = []
        for member in range(self.config.members):
            solver = current_diffusion_from_mapping(
                asdict(geometry),
                EtaProfile(
                    eta0=float(np.exp(eta_log[member])),
                    contrast=float(eta_contrast[member]),
                    shape=2.0,
                ),
                theta=1.0,
            )
            solver.precision = "float64"
            member_current = nominal_current * boundary_scale[member]
            try:
                nominal_step = solver.evolve(times, member_current)
                edited_step = solver.evolve(
                    times,
                    edited_current * boundary_scale[member],
                )
                nominal_prediction = solver.predict(nominal_step)
                ledger = solver.budget(nominal_step)
            except Exception as error:  # noqa: BLE001
                raise EstimatorFailure(f"Nova member {member} failed") from error
            products = (
                *nominal_step.values(),
                *edited_step.values(),
                *nominal_prediction.values(),
            )
            if not all(np.isfinite(np.asarray(value)).all() for value in products):
                raise EstimatorFailure(f"Nova member {member} returned non-finite data")
            if np.asarray(nominal_step["psi_face"]).dtype != np.float64:
                raise EstimatorFailure(
                    "Nova did not resolve the requested float64 precision"
                )
            identity_error = abs(
                ledger["d_psi_bdry"] - ledger["d_psi_axis"] - ledger["d_psi_internal"]
            )
            if identity_error > 1.0e-12:
                raise EstimatorFailure("Nova flux-consumption ledger does not close")
            final_current = geometry.enclosed_current(nominal_step["psi_face"][-1])[-1]
            relative_error = (
                abs(final_current - member_current[-1]) / member_current[-1]
            )
            if relative_error > 0.04:
                raise EstimatorFailure("Nova boundary-current constraint was violated")
            nominal_members.append(_observation(geometry, nominal_step["psi_face"]))
            edited_members.append(_observation(geometry, edited_step["psi_face"]))
            nominal_flux_members.append(
                np.asarray(nominal_step["psi_face"], dtype=np.float64)
            )
            edited_flux_members.append(
                np.asarray(edited_step["psi_face"], dtype=np.float64)
            )
            ledgers.append(ledger)
            boundary_errors.append(float(relative_error))

        raw_forecast = np.stack(nominal_members, axis=1)
        edited_product = np.stack(edited_members, axis=1)
        equilibrium_flux = np.stack(nominal_flux_members, axis=1)
        edited_equilibrium_flux = np.stack(edited_flux_members, axis=1)
        phase = times / times[-1]
        out_of_distribution = phase >= 0.68
        centre = reference_forecast[:, None, :]
        widening = np.where(out_of_distribution, 2.0, 1.0)[:, None, None]
        raw_forecast = centre + widening * (raw_forecast - centre)

        forecast = np.empty_like(raw_forecast)
        analysis = np.empty_like(raw_forecast)
        carried = np.zeros_like(raw_forecast[0])
        observation_variance = self.config.observation_noise**2
        forecast_variance = 0.012**2
        gain = forecast_variance / (forecast_variance + observation_variance)
        for index in range(self.config.steps):
            prior = raw_forecast[index] + carried
            forecast[index] = prior
            if index % self.config.correction_stride == 0:
                updated = prior + gain * (
                    observations[index][None, :] + member_noise[index] - prior
                )
            else:
                updated = prior
            analysis[index] = updated
            carried = updated - raw_forecast[index]

        fixed_lag = _smooth(
            analysis,
            observations,
            lag=self.config.fixed_lag_steps,
        )
        full_sequence = _smooth(analysis, observations, lag=None)

        scorecard = _ensemble_scorecard(
            clock=times,
            truth=truth,
            observations=observations,
            forecast=forecast,
            analysis=analysis,
            nominal_actuator=raw_forecast,
            edited_actuator=edited_product,
            seed=self.config.seed,
            observation_noise=self.config.observation_noise,
        )
        elapsed = time.perf_counter() - started
        metrics: dict[str, Any] = {
            **scorecard,
            "physics": {
                "finite_members": self.config.members,
                "topology": self.config.topology,
                "common_random_numbers": True,
                "correction_frequency_hz": 1000 // self.config.correction_stride,
                "max_boundary_current_relative_error": max(boundary_errors),
                "max_ledger_identity_error": max(
                    abs(row["d_psi_bdry"] - row["d_psi_axis"] - row["d_psi_internal"])
                    for row in ledgers
                ),
            },
            "runtime": {
                "elapsed_s": elapsed,
                "member_steps_per_s": self.config.members
                * self.config.steps
                / max(elapsed, 1.0e-12),
                "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            },
        }
        result = EstimatorResult(
            clock=np.asarray(times, dtype=np.float64),
            truth=np.asarray(truth, dtype=np.float64),
            observations=np.asarray(observations, dtype=np.float64),
            causal_forecast=np.asarray(forecast, dtype=np.float64),
            causal_analysis=np.asarray(analysis, dtype=np.float64),
            fixed_lag_smoothing=np.asarray(fixed_lag, dtype=np.float64),
            full_sequence_smoothing=np.asarray(full_sequence, dtype=np.float64),
            edited_actuator=np.asarray(edited_product, dtype=np.float64),
            nominal_actuator=np.asarray(raw_forecast, dtype=np.float64),
            equilibrium_flux=equilibrium_flux,
            edited_equilibrium_flux=edited_equilibrium_flux,
            metrics=metrics,
            provenance=provenance,
            camera_proxy={
                "product": "flux_surface_emissivity_proxy",
                "validated_checkpoint": False,
                "label": "proxy",
            },
            config=self.config,
        )
        result.validate()
        return result


def _arrays(result: EstimatorResult) -> dict[str, np.ndarray]:
    return {
        "clock": result.clock,
        "truth": result.truth,
        "observations": result.observations,
        "causal_forecast": result.causal_forecast,
        "causal_analysis": result.causal_analysis,
        "fixed_lag_smoothing": result.fixed_lag_smoothing,
        "full_sequence_smoothing": result.full_sequence_smoothing,
        "edited_actuator": result.edited_actuator,
        "nominal_actuator": result.nominal_actuator,
        "equilibrium_flux": result.equilibrium_flux,
        "edited_equilibrium_flux": result.edited_equilibrium_flux,
    }


def _artifact_config(config: EstimatorConfig) -> dict[str, Any]:
    """Return numerical settings that must agree before shard merging."""
    return {
        "sample_period_s": config.sample_period_s,
        "steps": config.steps,
        "fixed_lag_steps": config.fixed_lag_steps,
        "correction_stride": config.correction_stride,
        "observation_noise": config.observation_noise,
        "topology": config.topology,
        "dtype": config.dtype,
        "horizons_ms": list(_HORIZONS_MS),
    }


def write_result(
    result: EstimatorResult,
    output_dir: str | Path,
    *,
    name: str = "result",
    shard_index: int | None = None,
    shard_count: int | None = None,
    seed: int | None = None,
) -> tuple[Path, Path]:
    """Write one raw JSON/NPZ pair with enough identity for strict merging."""
    result.validate()
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    arrays = _arrays(result)
    metadata = {
        "schema": SCHEMA,
        "failure": False,
        "seed": seed,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "shape": {key: list(value.shape) for key, value in arrays.items()},
        "metrics": result.metrics,
        "estimator_config": _artifact_config(result.config),
        "provenance": asdict(result.provenance),
        "camera_proxy": result.camera_proxy,
        "observation_labels": list(OBSERVATION_LABELS),
        "equilibrium_products": {
            "equilibrium_flux": {
                "source": "Nova CurrentDiffusion.evolve psi_face",
                "axes": ["clock", "member", "radial_face"],
                "units": "Wb",
                "description": (
                    "member-wise one-dimensional poloidal-flux profiles on nested "
                    "radial faces; not a two-dimensional Grad-Shafranov map"
                ),
            },
            "edited_equilibrium_flux": {
                "source": "Nova CurrentDiffusion.evolve psi_face",
                "axes": ["clock", "member", "radial_face"],
                "units": "Wb",
                "description": (
                    "member-wise one-dimensional poloidal-flux profiles under the "
                    "edited actuator trajectory; not a two-dimensional "
                    "Grad-Shafranov map"
                ),
            },
        },
    }
    json_path = target / f"{name}.json"
    array_path = target / f"{name}.npz"
    json_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    np.savez(array_path, **arrays)
    return json_path, array_path


def merge_shards(merge_root: str | Path, output_dir: str | Path) -> tuple[Path, Path]:
    """Merge a complete, unique, revision-matched shard set."""
    root = Path(merge_root)
    records: list[tuple[dict[str, Any], Path]] = []
    for path in sorted(root.glob("shard-*.json")):
        record = json.loads(path.read_text())
        records.append((record, path))
    if not records:
        raise EstimatorFailure("merge root contains no shard metadata")
    first = records[0][0]
    shard_count = first.get("shard_count")
    if not isinstance(shard_count, int) or shard_count < 1:
        raise EstimatorFailure("shard count is missing or invalid")
    if not isinstance(first.get("seed"), int):
        raise EstimatorFailure("shard seed is missing or invalid")
    indices = [record.get("shard_index") for record, _ in records]
    if len(indices) != len(set(indices)):
        raise EstimatorFailure("duplicate shard index")
    if set(indices) != set(range(shard_count)):
        raise EstimatorFailure("missing shard index")
    identity_keys = ("schema", "seed", "shard_count")
    provenance_keys = ("nova_revision", "backend", "dtype", "topology")
    reference_config = first.get("estimator_config")
    if not isinstance(reference_config, dict):
        raise EstimatorFailure("estimator configuration is missing")
    for record, path in records:
        if record.get("failure") is not False:
            raise EstimatorFailure(f"failed shard declared by {path.name}")
        if any(record.get(key) != first.get(key) for key in identity_keys):
            raise EstimatorFailure("shard seed or schema identity mismatch")
        if record.get("estimator_config") != reference_config:
            raise EstimatorFailure("estimator configuration mismatch")
        for key in ("camera_proxy", "observation_labels", "equilibrium_products"):
            if record.get(key) != first.get(key):
                raise EstimatorFailure(f"shard {key} mismatch")
        provenance = record.get("provenance", {})
        reference = first.get("provenance", {})
        if any(provenance.get(key) != reference.get(key) for key in provenance_keys):
            raise EstimatorFailure(
                "shard revision, backend, dtype, or topology mismatch"
            )
        if record.get("shape") != first.get("shape"):
            raise EstimatorFailure("shard array shape mismatch")
        if not path.with_suffix(".npz").is_file():
            raise EstimatorFailure(f"array payload is missing for {path.name}")

    records.sort(key=lambda item: item[0]["shard_index"])
    loaded = [np.load(path.with_suffix(".npz")) for _, path in records]
    try:
        for (record, path), payload in zip(records, loaded, strict=True):
            expected_shapes = record["shape"]
            if set(payload.files) != set(expected_shapes):
                raise EstimatorFailure(f"array schema differs for {path.name}")
            for key in payload.files:
                if list(payload[key].shape) != expected_shapes[key]:
                    raise EstimatorFailure(f"array shape differs for {path.name}")
        merged: dict[str, np.ndarray] = {}
        for key in loaded[0].files:
            values = [payload[key] for payload in loaded]
            if key in {
                "causal_forecast",
                "causal_analysis",
                "fixed_lag_smoothing",
                "full_sequence_smoothing",
                "edited_actuator",
                "nominal_actuator",
                "equilibrium_flux",
                "edited_equilibrium_flux",
            }:
                merged[key] = np.concatenate(values, axis=1)
            else:
                if not all(np.array_equal(values[0], value) for value in values[1:]):
                    raise EstimatorFailure(
                        f"non-member array {key!r} differs across shards"
                    )
                merged[key] = values[0]
    finally:
        for payload in loaded:
            payload.close()

    try:
        observation_noise = float(reference_config["observation_noise"])
        sample_period_s = float(reference_config["sample_period_s"])
        correction_stride = int(reference_config["correction_stride"])
        shard_physics = [record["metrics"]["physics"] for record, _ in records]
        shard_runtime_source = [record["metrics"]["runtime"] for record, _ in records]
    except (KeyError, TypeError, ValueError) as error:
        raise EstimatorFailure(
            "shard metadata cannot reproduce merged metrics"
        ) from error

    members = int(merged["causal_forecast"].shape[1])
    scorecard = _ensemble_scorecard(
        clock=merged["clock"],
        truth=merged["truth"],
        observations=merged["observations"],
        forecast=merged["causal_forecast"],
        analysis=merged["causal_analysis"],
        nominal_actuator=merged["nominal_actuator"],
        edited_actuator=merged["edited_actuator"],
        seed=first["seed"],
        observation_noise=observation_noise,
    )
    physics = {
        "finite_members": members,
        "topology": first["provenance"]["topology"],
        "common_random_numbers": all(
            row.get("common_random_numbers") is True for row in shard_physics
        ),
        "correction_frequency_hz": int(
            round(1.0 / (sample_period_s * correction_stride))
        ),
        "max_boundary_current_relative_error": max(
            float(row["max_boundary_current_relative_error"]) for row in shard_physics
        ),
        "max_ledger_identity_error": max(
            float(row["max_ledger_identity_error"]) for row in shard_physics
        ),
        "aggregation_rule": "maximum physical error across shards",
    }
    per_shard_runtime = [
        {
            "shard_index": record["shard_index"],
            "elapsed_s": float(runtime["elapsed_s"]),
            "member_steps_per_s": float(runtime["member_steps_per_s"]),
            "peak_rss_kib": int(runtime["peak_rss_kib"]),
        }
        for (record, _), runtime in zip(
            records,
            shard_runtime_source,
            strict=True,
        )
    ]
    runtime = {
        "aggregation_rule": (
            "per-shard measurements are preserved; serial_elapsed_sum_s is an "
            "arithmetic sum; parallel wall time and aggregate throughput were not "
            "measured"
        ),
        "per_shard": per_shard_runtime,
        "serial_elapsed_sum_s": float(
            sum(row["elapsed_s"] for row in per_shard_runtime)
        ),
        "parallel_wall_time_s": None,
        "aggregate_member_steps_per_s": None,
        "peak_rss_kib_max_per_process": max(
            row["peak_rss_kib"] for row in per_shard_runtime
        ),
        "member_steps_total": members * int(merged["clock"].size),
    }
    metrics = {**scorecard, "physics": physics, "runtime": runtime}

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    array_path = target / "merged.npz"
    np.savez(array_path, **merged)
    metadata = dict(first)
    metadata["merged_shards"] = shard_count
    metadata["shape"] = {key: list(value.shape) for key, value in merged.items()}
    metadata["metrics"] = metrics
    metadata["shard_index"] = None
    json_path = target / "merged.json"
    json_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return json_path, array_path


__all__ = [
    "EstimatorConfig",
    "EstimatorFailure",
    "EstimatorResult",
    "NovaEnsembleEstimator",
    "RuntimeProvenance",
    "merge_shards",
    "write_result",
]
