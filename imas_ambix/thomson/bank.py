"""Read-only calibration adapters for challenge shots and extracted trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from imas_ambix.challenge.loader import ChallengeShot, load_shot

from .models import TopologyClass

_TWO_PI = 2.0 * np.pi
_MU_ZERO = 4.0e-7 * np.pi


@dataclass(frozen=True)
class BankedEquilibriumMoment:
    shot: str
    time_ms: float
    beta_p: float
    li: float
    beta_p_plus_li_half: float


def collect_pedestal_samples(
    parquet_path: str | Path,
    bank_path: str | Path,
    *,
    edge_psi_n_minimum: float = 0.55,
    maximum_time_separation_ms: float = 15.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect normalized temperature and label-derived chord ``psi_N``.

    The challenge loader is the sole corpus reader.  The extracted bank provides
    the axis and boundary flux receipts needed to normalize the label map.
    """

    shot = load_shot(parquet_path)
    with np.load(bank_path, allow_pickle=False) as loaded:
        bank = {name: loaded[name] for name in loaded.files}
    feature_parts: list[np.ndarray] = []
    psi_parts: list[np.ndarray] = []
    topology_parts: list[np.ndarray] = []
    for profile_name, topology in (
        ("core", TopologyClass.CORE_VERTICAL),
        ("edge", TopologyClass.TANGENTIAL_EDGE),
    ):
        profile = shot.thomson[profile_name]
        radius, height = _profile_geometry(shot, profile_name)
        for frame, label_time in enumerate(shot.labels.time_ms):
            profile_index = int(np.argmin(np.abs(profile.time_ms - label_time)))
            if (
                abs(float(profile.time_ms[profile_index] - label_time))
                > maximum_time_separation_ms
            ):
                continue
            temperature = np.asarray(profile.temperature_ev[profile_index], dtype=float)
            valid_temperature = np.isfinite(temperature) & (temperature > 1.0)
            if np.count_nonzero(valid_temperature) < 3:
                continue
            psi_n = _sample_psi_n(shot, bank, frame, radius, height)
            keep = (
                valid_temperature & np.isfinite(psi_n) & (psi_n >= edge_psi_n_minimum)
            )
            if np.count_nonzero(keep) < 2:
                continue
            reference = float(np.nanpercentile(temperature[valid_temperature], 90.0))
            feature_parts.append(np.log(temperature[keep] / reference))
            psi_parts.append(psi_n[keep])
            topology_parts.append(np.full(np.count_nonzero(keep), topology.value))
    if not feature_parts:
        return np.empty(0), np.empty(0), np.empty(0, dtype=str)
    return (
        np.concatenate(feature_parts),
        np.concatenate(psi_parts),
        np.concatenate(topology_parts),
    )


def banked_equilibrium_moments(
    parquet_path: str | Path,
    bank_path: str | Path,
    *,
    stride: int = 20,
) -> tuple[BankedEquilibriumMoment, ...]:
    """Compute ``beta_p + li/2`` receipts from banked equilibrium profiles."""

    shot = load_shot(parquet_path)
    with np.load(bank_path, allow_pickle=False) as loaded:
        bank = {name: loaded[name] for name in loaded.files}
    frame_count = min(len(shot.labels.time_ms), len(bank["times_ms"]))
    moments: list[BankedEquilibriumMoment] = []
    for frame in range(0, frame_count, stride):
        beta_p = _beta_p_from_banked_profiles(bank, frame)
        li = float(shot.labels.scalars["efit_li"][frame])
        if np.isfinite(beta_p) and np.isfinite(li) and beta_p >= 0.0:
            moments.append(
                BankedEquilibriumMoment(
                    shot=Path(parquet_path).stem,
                    time_ms=float(shot.labels.time_ms[frame]),
                    beta_p=beta_p,
                    li=li,
                    beta_p_plus_li_half=beta_p + 0.5 * li,
                )
            )
    return tuple(moments)


def _profile_geometry(
    shot: ChallengeShot, profile_name: str
) -> tuple[np.ndarray, np.ndarray]:
    names = shot.chord_geometry["thomson_chord_name"]
    if profile_name == "core":
        indices = np.flatnonzero(np.char.startswith(names, "TS_core"))
    elif profile_name == "edge":
        indices = np.flatnonzero(np.char.startswith(names, "TS_tangential"))
    else:
        raise ValueError(f"unsupported Thomson profile {profile_name}")
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
    raw = _bilinear_sample(
        shot.labels.grid_r_m,
        shot.labels.grid_z_m,
        shot.labels.psirz[frame],
        radius,
        height,
    )
    axis = float(bank["axis_flux_wb"][frame]) / _TWO_PI
    boundary = float(bank["boundary_flux_wb"][frame]) / _TWO_PI
    return (raw - axis) / (boundary - axis)


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
    result = lower * (1.0 - z_fraction) + upper * z_fraction
    return np.where(inside, result, np.nan)


def _beta_p_from_banked_profiles(bank: dict[str, np.ndarray], frame: int) -> float:
    psi_n = np.asarray(bank["surface_psi_n"][frame], dtype=float)
    p_prime = np.asarray(bank["p_prime"][frame], dtype=float)
    volume_derivative = np.asarray(bank["fsa_dv_dpsi_n"][frame], dtype=float)
    gradient = np.asarray(bank["fsa_gradient2_over_r2"][frame], dtype=float)
    flux_span = abs(
        float(bank["boundary_flux_wb"][frame] - bank["axis_flux_wb"][frame])
    )
    volume_derivative = np.abs(volume_derivative)
    pressure = np.zeros_like(p_prime)
    for index in range(len(psi_n) - 2, -1, -1):
        pressure[index] = pressure[index + 1] + (
            0.5
            * (abs(p_prime[index]) + abs(p_prime[index + 1]))
            * (psi_n[index + 1] - psi_n[index])
            * flux_span
        )
    pressure_energy = float(np.trapezoid(pressure * volume_derivative, psi_n))
    poloidal_energy = float(
        np.trapezoid(
            gradient * (flux_span / _TWO_PI) ** 2 * volume_derivative,
            psi_n,
        )
    )
    if poloidal_energy <= 0.0:
        return float("nan")
    return float(2.0 * _MU_ZERO * pressure_energy / poloidal_energy)
