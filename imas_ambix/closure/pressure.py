"""Electron-to-total pressure closure from derivative-level equilibrium receipts.

The calibration consumes extracted pressure gradients rather than equilibrium
maps.  A quadratic pressure-gradient shape, weighted toward the outer flux
surfaces, is the only label-derived profile freedom.  Thomson density and
temperature set electron pressure and a dimensionless, q-free collisionality
proxy; flux-surface-average geometry supplies the volume and aspect-ratio
scales.  The fitted closure is floored at unity because total pressure cannot
be smaller than its electron contribution.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from math import pi
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from scipy.constants import e, m_e

from imas_ambix.challenge.convention import DIIID_CONVENTION
from imas_ambix.challenge.loader import ChallengeShot, load_shot

_MINIMUM_MULTIPLIER = 1.0
_MAXIMUM_THOMSON_SEPARATION_MS = 15.0
_COLLISION_FREQUENCY_COEFFICIENT = 2.91e-6
_PRESSURE_SHAPE_DEGREE = 2
_BOOTSTRAP_DRAWS = 2000
_BOOTSTRAP_SEED = 214821
_HEATING_TOKENS = (
    "auxiliary_heating",
    "beam_power",
    "electron_cyclotron",
    "ion_cyclotron",
    "nbi",
    "ech",
    "ecrh",
    "ich",
    "icrh",
)


@dataclass(frozen=True)
class CollisionalityReceipt:
    """Inputs and result of the dimensionless collisionality proxy."""

    value: float
    density_m3: float
    temperature_ev: float
    major_radius_m: float
    effective_minor_radius_m: float
    inverse_aspect_ratio: float
    coulomb_logarithm: float


@dataclass(frozen=True)
class PressureClosureSample:
    """One aligned equilibrium/Thomson frame used by the calibration."""

    shot: str
    time_ms: float
    total_pressure_pa: float
    electron_pressure_pa: float
    raw_multiplier: float
    collisionality: CollisionalityReceipt
    actuators: dict[str, float]


@dataclass(frozen=True)
class PressureClosureCalibration:
    """Log-linear pressure closure with a hard physical lower bound."""

    reference_collisionality: float
    intercept_log_multiplier: float
    collisionality_slope: float
    collisionality_slope_confidence_interval: tuple[float, float]
    residual_scatter_log_multiplier: float
    residual_scatter_multiplier: float
    sample_count: int
    shot_count: int

    def multiplier(self, collisionality: Any) -> np.ndarray:
        """Predict total/electron pressure while enforcing the physical floor."""

        values = np.asarray(collisionality, dtype=float)
        if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("collisionality must be finite and positive")
        coordinate = np.log(values / self.reference_collisionality)
        prediction = np.exp(
            self.intercept_log_multiplier + self.collisionality_slope * coordinate
        )
        return np.maximum(prediction, _MINIMUM_MULTIPLIER)


def dimensionless_collisionality_proxy(
    density_m3: float,
    temperature_ev: float,
    major_radius_m: float,
    effective_minor_radius_m: float,
) -> CollisionalityReceipt:
    """Return a q-free electron collisionality proxy.

    The electron-ion collision frequency is normalized by the thermal transit
    frequency and the trapped-particle aspect-ratio factor.  Omitting q keeps
    the proxy wholly on the permitted Thomson plus FSA-geometry boundary.
    """

    values = np.asarray(
        [density_m3, temperature_ev, major_radius_m, effective_minor_radius_m],
        dtype=float,
    )
    if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("collisionality inputs must be finite and positive")
    inverse_aspect_ratio = effective_minor_radius_m / major_radius_m
    if not 0.0 < inverse_aspect_ratio < 1.0:
        raise ValueError("effective minor radius must be below major radius")
    density_cm3 = density_m3 * 1.0e-6
    coulomb_logarithm = float(
        np.clip(24.0 - np.log(np.sqrt(density_cm3) / temperature_ev), 10.0, 25.0)
    )
    collision_frequency_hz = (
        _COLLISION_FREQUENCY_COEFFICIENT
        * density_cm3
        * coulomb_logarithm
        / temperature_ev**1.5
    )
    thermal_speed_m_s = np.sqrt(2.0 * e * temperature_ev / m_e)
    value = (
        collision_frequency_hz
        * major_radius_m
        / (thermal_speed_m_s * inverse_aspect_ratio**1.5)
    )
    return CollisionalityReceipt(
        value=float(value),
        density_m3=float(density_m3),
        temperature_ev=float(temperature_ev),
        major_radius_m=float(major_radius_m),
        effective_minor_radius_m=float(effective_minor_radius_m),
        inverse_aspect_ratio=float(inverse_aspect_ratio),
        coulomb_logarithm=coulomb_logarithm,
    )


def collect_pressure_closure_samples(
    source_dir: str | Path,
    bank_dir: str | Path,
    *,
    maximum_thomson_separation_ms: float = _MAXIMUM_THOMSON_SEPARATION_MS,
) -> tuple[tuple[PressureClosureSample, ...], dict[str, Any]]:
    """Align the banked extraction corpus with Thomson and actuator channels."""

    source_root = Path(source_dir)
    bank_root = Path(bank_dir)
    pairs = _paired_paths(source_root, bank_root)
    if not pairs:
        raise FileNotFoundError("no paired challenge shots and extraction banks found")

    samples: list[PressureClosureSample] = []
    banked_frames = 0
    channel_sets: list[set[str]] = []
    schema_dynamic_sets: list[set[str]] = []
    for parquet_path, bank_path in pairs:
        shot = load_shot(parquet_path)
        with np.load(bank_path, allow_pickle=False) as loaded:
            bank = {name: loaded[name] for name in loaded.files}
        frame_count = min(len(shot.labels.time_ms), len(bank["times_ms"]))
        banked_frames += frame_count
        channel_sets.append(set(shot.actuators))
        schema_dynamic_sets.append(_dynamic_schema_channels(parquet_path))
        for frame in range(frame_count):
            sample = _frame_sample(
                shot,
                bank,
                frame,
                shot_id=parquet_path.stem,
                maximum_thomson_separation_ms=maximum_thomson_separation_ms,
            )
            if sample is not None:
                samples.append(sample)

    inventory_union = sorted(set().union(*channel_sets))
    inventory_intersection = sorted(set.intersection(*channel_sets))
    dynamic_union = sorted(set().union(*schema_dynamic_sets))
    heating_channels = [name for name in inventory_union if _is_heating_channel(name)]
    inventory = {
        "banked_shots": len(pairs),
        "banked_frames": banked_frames,
        "eligible_frames": len(samples),
        "actuator_channels_union": inventory_union,
        "actuator_channels_present_in_every_shot": inventory_intersection,
        "released_dynamic_channels": dynamic_union,
        "auxiliary_heating_channels": heating_channels,
        "excluded_dynamic_descriptors": sorted(
            set(dynamic_union) - set(inventory_union)
        ),
    }
    return tuple(samples), inventory


def fit_pressure_closure(
    samples: tuple[PressureClosureSample, ...] | list[PressureClosureSample],
    *,
    bootstrap_draws: int = _BOOTSTRAP_DRAWS,
    bootstrap_seed: int = _BOOTSTRAP_SEED,
) -> PressureClosureCalibration:
    """Fit a shot-balanced log-linear multiplier and cluster confidence interval."""

    if len(samples) < 3:
        raise ValueError("at least three pressure samples are required")
    collisionality = np.asarray([s.collisionality.value for s in samples], dtype=float)
    multiplier = np.maximum(
        np.asarray([s.raw_multiplier for s in samples], dtype=float),
        _MINIMUM_MULTIPLIER,
    )
    shots = np.asarray([s.shot for s in samples])
    reference = float(np.exp(np.median(np.log(collisionality))))
    x = np.log(collisionality / reference)
    y = np.log(multiplier)
    weights = _equal_shot_weights(shots)
    intercept, slope = _weighted_line(x, y, weights)
    fitted = np.maximum(intercept + slope * x, 0.0)
    prediction = np.exp(fitted)

    unique_shots = np.unique(shots)
    rng = np.random.default_rng(bootstrap_seed)
    bootstrap_slopes = []
    if bootstrap_draws > 0 and len(unique_shots) >= 2:
        indices_by_shot = {shot: np.flatnonzero(shots == shot) for shot in unique_shots}
        for _ in range(bootstrap_draws):
            chosen = rng.choice(unique_shots, size=len(unique_shots), replace=True)
            indices = np.concatenate([indices_by_shot[shot] for shot in chosen])
            draw_shots = np.concatenate(
                [
                    np.full(len(indices_by_shot[shot]), index)
                    for index, shot in enumerate(chosen)
                ]
            )
            draw_weights = _equal_shot_weights(draw_shots)
            _, draw_slope = _weighted_line(x[indices], y[indices], draw_weights)
            if np.isfinite(draw_slope):
                bootstrap_slopes.append(draw_slope)
    if bootstrap_slopes:
        interval = tuple(
            float(value) for value in np.quantile(bootstrap_slopes, (0.025, 0.975))
        )
    else:
        interval = (float("nan"), float("nan"))
    return PressureClosureCalibration(
        reference_collisionality=reference,
        intercept_log_multiplier=float(intercept),
        collisionality_slope=float(slope),
        collisionality_slope_confidence_interval=interval,
        residual_scatter_log_multiplier=float(
            np.sqrt(np.average((y - fitted) ** 2, weights=weights))
        ),
        residual_scatter_multiplier=float(
            np.sqrt(np.average((multiplier - prediction) ** 2, weights=weights))
        ),
        sample_count=len(samples),
        shot_count=len(unique_shots),
    )


def calibrate_pressure_closure(
    source_dir: str | Path,
    bank_dir: str | Path,
    output_path: str | Path,
    *,
    bootstrap_draws: int = _BOOTSTRAP_DRAWS,
) -> dict[str, Any]:
    """Fit the real banked corpus and write a reproducible JSON receipt."""

    samples, inventory = collect_pressure_closure_samples(source_dir, bank_dir)
    calibration = fit_pressure_closure(samples, bootstrap_draws=bootstrap_draws)
    collisionality = np.asarray([s.collisionality.value for s in samples], dtype=float)
    raw_multiplier = np.asarray([s.raw_multiplier for s in samples], dtype=float)
    fitted_multiplier = calibration.multiplier(collisionality)
    payload = {
        "schema_version": 1,
        "method": {
            "label_consumption": "derivative_level_pressure_gradient_only",
            "pressure_shape": "quadratic_outer_surface_weighted",
            "electron_pressure": "volume_average_of_thomson_ne_times_te",
            "collisionality_proxy": (
                "electron_ion_collision_frequency_over_thermal_transit_frequency_"
                "with_fsa_aspect_ratio_and_without_q"
            ),
            "fit": "equal_shot_weighted_log_multiplier_on_log_collisionality",
            "physical_floor": _MINIMUM_MULTIPLIER,
            "confidence_interval": (
                f"shot_cluster_bootstrap_95_percent_{bootstrap_draws}_draws"
            ),
        },
        "corpus": inventory,
        "fit": {
            **asdict(calibration),
            "collisionality_slope_confidence_interval": list(
                calibration.collisionality_slope_confidence_interval
            ),
            "collisionality_range": _summary(collisionality),
            "raw_multiplier_summary": _summary(raw_multiplier),
            "fitted_multiplier_summary": _summary(fitted_multiplier),
            "raw_multiplier_below_physical_floor_count": int(
                np.count_nonzero(raw_multiplier < _MINIMUM_MULTIPLIER)
            ),
        },
        "per_shot": _per_shot_receipts(samples, calibration),
        "auxiliary_heating_dependence": _heating_dependence(samples, inventory),
    }
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def _paired_paths(source_dir: Path, bank_dir: Path) -> list[tuple[Path, Path]]:
    pairs = []
    for bank_path in sorted(bank_dir.glob("*_flux_trajectory.npz")):
        stem = bank_path.name.removesuffix("_flux_trajectory.npz")
        parquet_path = source_dir / f"{stem}.parquet"
        if parquet_path.exists():
            pairs.append((parquet_path, bank_path))
    return pairs


def _dynamic_schema_channels(path: Path) -> set[str]:
    names = pq.ParquetFile(path).schema_arrow.names
    return {
        name
        for name in names
        if name.startswith("magnetics_") and not name.endswith(("_time", "_times"))
    }


def _is_heating_channel(name: str) -> bool:
    normalized = name.lower()
    return any(token in normalized for token in _HEATING_TOKENS)


def _frame_sample(
    shot: ChallengeShot,
    bank: dict[str, np.ndarray],
    frame: int,
    *,
    shot_id: str,
    maximum_thomson_separation_ms: float,
) -> PressureClosureSample | None:
    if not bool(bank["vacuum_pass"][frame]) or not bool(bank["fsa_well_posed"][frame]):
        return None
    profile = _thomson_surface_profile(
        shot,
        bank,
        frame,
        maximum_time_separation_ms=maximum_thomson_separation_ms,
    )
    if profile is None:
        return None
    measured_psi, measured_temperature, measured_density = profile

    psi = np.asarray(bank["fsa_psi_n"][frame], dtype=float)
    volume_derivative = np.abs(np.asarray(bank["fsa_dv_dpsi_n"][frame], dtype=float))
    pressure_gradient = np.abs(np.asarray(bank["p_prime"][frame], dtype=float))
    separation_quality = np.asarray(bank["separation_fit_fraction"][frame], dtype=float)
    finite = (
        np.isfinite(psi)
        & np.isfinite(volume_derivative)
        & np.isfinite(pressure_gradient)
        & np.isfinite(separation_quality)
        & (volume_derivative > 0.0)
        & (separation_quality > 0.2)
    )
    if np.count_nonzero(finite) < 12:
        return None
    psi = psi[finite]
    volume_derivative = volume_derivative[finite]
    pressure_gradient = pressure_gradient[finite]
    separation_quality = separation_quality[finite]
    order = np.argsort(psi)
    psi = psi[order]
    volume_derivative = volume_derivative[order]
    pressure_gradient = pressure_gradient[order]
    separation_quality = separation_quality[order]

    smooth_gradient = _edge_weighted_pressure_gradient(
        psi, pressure_gradient, separation_quality
    )
    flux_span = abs(
        float(bank["boundary_flux_wb"][frame] - bank["axis_flux_wb"][frame])
    )
    total_pressure_profile = _integrated_pressure(psi, smooth_gradient, flux_span)
    temperature_profile = np.interp(psi, measured_psi, measured_temperature)
    density_profile = np.interp(psi, measured_psi, measured_density)
    electron_pressure_profile = density_profile * temperature_profile * e
    normalization = float(np.trapezoid(volume_derivative, psi))
    if not np.isfinite(normalization) or normalization <= 0.0:
        return None
    total_pressure = float(
        np.trapezoid(total_pressure_profile * volume_derivative, psi) / normalization
    )
    electron_pressure = float(
        np.trapezoid(electron_pressure_profile * volume_derivative, psi) / normalization
    )
    average_temperature = float(
        np.trapezoid(temperature_profile * volume_derivative, psi) / normalization
    )
    average_density = float(
        np.trapezoid(density_profile * volume_derivative, psi) / normalization
    )
    inverse_radius = np.asarray(bank["fsa_inverse_r"][frame], dtype=float)[finite][
        order
    ]
    major_radius = float(
        1.0 / (np.trapezoid(inverse_radius * volume_derivative, psi) / normalization)
    )
    volume = float(bank["fsa_volume"][frame])
    minor_radius = float(np.sqrt(volume / (2.0 * pi**2 * major_radius)))
    if (
        not np.all(
            np.isfinite(
                [
                    total_pressure,
                    electron_pressure,
                    average_temperature,
                    average_density,
                ]
            )
        )
        or total_pressure <= 0.0
        or electron_pressure <= 0.0
    ):
        return None
    collisionality = dimensionless_collisionality_proxy(
        average_density,
        average_temperature,
        major_radius,
        minor_radius,
    )
    time_ms = float(shot.labels.time_ms[frame])
    actuators = {
        name: float(np.interp(time_ms, series.time_ms, series.values))
        for name, series in shot.actuators.items()
    }
    return PressureClosureSample(
        shot=shot_id,
        time_ms=time_ms,
        total_pressure_pa=total_pressure,
        electron_pressure_pa=electron_pressure,
        raw_multiplier=total_pressure / electron_pressure,
        collisionality=collisionality,
        actuators=actuators,
    )


def _thomson_surface_profile(
    shot: ChallengeShot,
    bank: dict[str, np.ndarray],
    frame: int,
    *,
    maximum_time_separation_ms: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    psi_parts: list[np.ndarray] = []
    temperature_parts: list[np.ndarray] = []
    density_parts: list[np.ndarray] = []
    for name in ("core", "edge"):
        profile = shot.thomson[name]
        profile_index = int(
            np.argmin(np.abs(profile.time_ms - shot.labels.time_ms[frame]))
        )
        if (
            abs(float(profile.time_ms[profile_index] - shot.labels.time_ms[frame]))
            > maximum_time_separation_ms
        ):
            continue
        radius, height = _profile_geometry(shot, name)
        psi_n = _sample_psi_n(shot, bank, frame, radius, height)
        temperature = np.asarray(profile.temperature_ev[profile_index], dtype=float)
        density = np.asarray(profile.density_m3[profile_index], dtype=float)
        keep = (
            np.isfinite(psi_n)
            & np.isfinite(temperature)
            & np.isfinite(density)
            & (psi_n >= 0.0)
            & (psi_n <= 1.05)
            & (temperature > 5.0)
            & (density > 1.0e17)
        )
        psi_parts.append(psi_n[keep])
        temperature_parts.append(temperature[keep])
        density_parts.append(density[keep])
    if not psi_parts or sum(len(values) for values in psi_parts) < 6:
        return None
    psi_n = np.concatenate(psi_parts)
    temperature = np.concatenate(temperature_parts)
    density = np.concatenate(density_parts)
    edges = np.linspace(0.0, 1.05, 22)
    bins = np.digitize(psi_n, edges) - 1
    binned = []
    for index in range(len(edges) - 1):
        selected = bins == index
        if np.any(selected):
            binned.append(
                (
                    float(np.median(psi_n[selected])),
                    float(np.median(temperature[selected])),
                    float(np.median(density[selected])),
                )
            )
    if len(binned) < 4:
        return None
    values = np.asarray(binned, dtype=float)
    if values[:, 0].min() > 0.45 or values[:, 0].max() < 0.72:
        return None
    return values[:, 0], values[:, 1], values[:, 2]


def _profile_geometry(
    shot: ChallengeShot, profile_name: str
) -> tuple[np.ndarray, np.ndarray]:
    names = shot.chord_geometry["thomson_chord_name"]
    prefix = "TS_core" if profile_name == "core" else "TS_tangential"
    indices = np.flatnonzero(np.char.startswith(names, prefix))
    expected = len(shot.thomson[profile_name].spatial_m)
    if len(indices) != expected:
        raise ValueError(
            f"{profile_name} geometry has {len(indices)} channels, expected {expected}"
        )
    return (
        np.asarray(shot.chord_geometry["thomson_chord_R"][indices], dtype=float),
        np.asarray(shot.chord_geometry["thomson_chord_Z"][indices], dtype=float),
    )


def _sample_psi_n(
    shot: ChallengeShot,
    bank: dict[str, np.ndarray],
    frame: int,
    radius: np.ndarray,
    height: np.ndarray,
) -> np.ndarray:
    values = _bilinear_sample(
        shot.labels.grid_r_m,
        shot.labels.grid_z_m,
        shot.labels.psirz[frame],
        radius,
        height,
    )
    axis = float(DIIID_CONVENTION.canonical_total_flux(bank["axis_flux_wb"][frame]))
    boundary = float(
        DIIID_CONVENTION.canonical_total_flux(bank["boundary_flux_wb"][frame])
    )
    return (values - axis) / (boundary - axis)


def _bilinear_sample(
    grid_r: np.ndarray,
    grid_z: np.ndarray,
    values_zr: np.ndarray,
    radius: np.ndarray,
    height: np.ndarray,
) -> np.ndarray:
    r_index = np.searchsorted(grid_r, radius, side="right") - 1
    z_index = np.searchsorted(grid_z, height, side="right") - 1
    inside = (
        (r_index >= 0)
        & (r_index < len(grid_r) - 1)
        & (z_index >= 0)
        & (z_index < len(grid_z) - 1)
    )
    r_index = np.clip(r_index, 0, len(grid_r) - 2)
    z_index = np.clip(z_index, 0, len(grid_z) - 2)
    r_fraction = (radius - grid_r[r_index]) / (grid_r[r_index + 1] - grid_r[r_index])
    z_fraction = (height - grid_z[z_index]) / (grid_z[z_index + 1] - grid_z[z_index])
    lower = (
        values_zr[z_index, r_index] * (1.0 - r_fraction)
        + values_zr[z_index, r_index + 1] * r_fraction
    )
    upper = (
        values_zr[z_index + 1, r_index] * (1.0 - r_fraction)
        + values_zr[z_index + 1, r_index + 1] * r_fraction
    )
    return np.where(inside, lower * (1.0 - z_fraction) + upper * z_fraction, np.nan)


def _edge_weighted_pressure_gradient(
    psi_n: np.ndarray,
    gradient: np.ndarray,
    separation_quality: np.ndarray,
) -> np.ndarray:
    design = np.vander(psi_n, N=_PRESSURE_SHAPE_DEGREE + 1, increasing=True)
    weights = np.clip(separation_quality, 0.0, 1.0) * (0.25 + 0.75 * psi_n**2)
    coefficients, _, _, _ = np.linalg.lstsq(
        design * np.sqrt(weights[:, None]),
        gradient * np.sqrt(weights),
        rcond=None,
    )
    return np.maximum(design @ coefficients, 0.0)


def _integrated_pressure(
    psi_n: np.ndarray, gradient: np.ndarray, flux_span: float
) -> np.ndarray:
    extended_psi = np.concatenate((psi_n, [1.0]))
    extended_gradient = np.concatenate((gradient, [gradient[-1]]))
    pressure = np.zeros_like(extended_psi)
    for index in range(len(extended_psi) - 2, -1, -1):
        pressure[index] = pressure[index + 1] + (
            0.5
            * (extended_gradient[index] + extended_gradient[index + 1])
            * (extended_psi[index + 1] - extended_psi[index])
            * flux_span
        )
    return pressure[:-1]


def _equal_shot_weights(shots: np.ndarray) -> np.ndarray:
    unique, counts = np.unique(shots, return_counts=True)
    count_by_shot = dict(zip(unique, counts, strict=True))
    return np.asarray([1.0 / count_by_shot[shot] for shot in shots], dtype=float)


def _weighted_line(
    coordinate: np.ndarray, response: np.ndarray, weights: np.ndarray
) -> tuple[float, float]:
    design = np.column_stack((np.ones_like(coordinate), coordinate))
    root_weight = np.sqrt(weights)
    coefficients, _, _, _ = np.linalg.lstsq(
        design * root_weight[:, None], response * root_weight, rcond=None
    )
    return float(coefficients[0]), float(coefficients[1])


def _summary(values: np.ndarray) -> dict[str, float]:
    quantiles = np.quantile(values, (0.0, 0.05, 0.5, 0.95, 1.0))
    return {
        "minimum": float(quantiles[0]),
        "p05": float(quantiles[1]),
        "median": float(quantiles[2]),
        "p95": float(quantiles[3]),
        "maximum": float(quantiles[4]),
    }


def _per_shot_receipts(
    samples: tuple[PressureClosureSample, ...],
    calibration: PressureClosureCalibration,
) -> list[dict[str, Any]]:
    receipts = []
    for shot in sorted({sample.shot for sample in samples}):
        selected = [sample for sample in samples if sample.shot == shot]
        collisionality = np.asarray(
            [sample.collisionality.value for sample in selected], dtype=float
        )
        raw = np.asarray([sample.raw_multiplier for sample in selected], dtype=float)
        fitted = calibration.multiplier(collisionality)
        receipts.append(
            {
                "shot": shot,
                "sample_count": len(selected),
                "collisionality": _summary(collisionality),
                "raw_multiplier": _summary(raw),
                "fitted_multiplier": _summary(fitted),
            }
        )
    return receipts


def _heating_dependence(
    samples: tuple[PressureClosureSample, ...], inventory: dict[str, Any]
) -> dict[str, Any]:
    channels = inventory["auxiliary_heating_channels"]
    if not channels:
        return {
            "channels": {},
            "finding": "not_estimable_no_auxiliary_heating_channel_in_banked_corpus",
        }
    collisionality = np.asarray([s.collisionality.value for s in samples], dtype=float)
    response = np.log(
        np.maximum([s.raw_multiplier for s in samples], _MINIMUM_MULTIPLIER)
    )
    x = np.log(collisionality / np.exp(np.median(np.log(collisionality))))
    weights = _equal_shot_weights(np.asarray([s.shot for s in samples]))
    results = {}
    for channel in channels:
        heating = np.asarray([s.actuators[channel] for s in samples], dtype=float)
        scale = float(np.std(heating))
        if not np.isfinite(scale) or scale == 0.0:
            results[channel] = {
                "finding": "not_estimable_constant_channel",
                "sample_count": len(samples),
            }
            continue
        standardized = (heating - np.mean(heating)) / scale
        design = np.column_stack((np.ones_like(x), x, standardized))
        root_weight = np.sqrt(weights)
        coefficients, _, _, _ = np.linalg.lstsq(
            design * root_weight[:, None], response * root_weight, rcond=None
        )
        results[channel] = {
            "log_multiplier_change_per_channel_standard_deviation": float(
                coefficients[2]
            ),
            "channel_standard_deviation": scale,
            "sample_count": len(samples),
        }
    return {"channels": results, "finding": "measured_for_every_identified_channel"}


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("bank_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--bootstrap-draws", type=int, default=_BOOTSTRAP_DRAWS)
    args = parser.parse_args()
    payload = calibrate_pressure_closure(
        args.source_dir,
        args.bank_dir,
        args.output,
        bootstrap_draws=args.bootstrap_draws,
    )
    fit = payload["fit"]
    print(
        json.dumps(
            {
                "shots": payload["corpus"]["banked_shots"],
                "banked_frames": payload["corpus"]["banked_frames"],
                "fit_samples": fit["sample_count"],
                "collisionality_slope": fit["collisionality_slope"],
                "slope_confidence_interval": fit[
                    "collisionality_slope_confidence_interval"
                ],
                "residual_scatter_multiplier": fit["residual_scatter_multiplier"],
                "heating_finding": payload["auxiliary_heating_dependence"]["finding"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    _main()
