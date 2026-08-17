"""Calibrated pedestal-foot detection from Thomson temperatures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .models import SeparatrixEstimate, TopologyClass


@dataclass(frozen=True)
class _TopologyCurve:
    feature_knots: np.ndarray
    psi_n_knots: np.ndarray
    psi_n_sigma: float
    sample_count: int


class PedestalCalibration:
    """Piecewise-linear temperature-ratio to ``psi_N`` calibration curve."""

    def __init__(self, curves: dict[TopologyClass, _TopologyCurve]) -> None:
        if not curves:
            raise ValueError("at least one topology curve is required")
        self._curves = dict(curves)

    @classmethod
    def fit(
        cls,
        log_temperature_ratio: np.ndarray,
        psi_n: np.ndarray,
        topology: np.ndarray,
        *,
        knot_count: int = 12,
        uncertainty_quantile: float = 0.68,
    ) -> PedestalCalibration:
        feature = np.asarray(log_temperature_ratio, dtype=float)
        target = np.asarray(psi_n, dtype=float)
        topology = np.asarray(topology, dtype=str)
        if feature.shape != target.shape or feature.shape != topology.shape:
            raise ValueError("calibration sample arrays must have identical shapes")
        curves: dict[TopologyClass, _TopologyCurve] = {}
        for topology_class in TopologyClass:
            keep = (
                (topology == topology_class.value)
                & np.isfinite(feature)
                & np.isfinite(target)
                & (target >= -0.1)
                & (target <= 1.35)
            )
            x = feature[keep]
            y = target[keep]
            if x.size < max(20, knot_count * 2):
                continue
            edges = np.unique(np.quantile(x, np.linspace(0.0, 1.0, knot_count + 1)))
            x_knots: list[float] = []
            y_knots: list[float] = []
            for lower, upper in zip(edges[:-1], edges[1:], strict=True):
                in_bin = (x >= lower) & (
                    x <= upper if upper == edges[-1] else x < upper
                )
                if np.count_nonzero(in_bin) < 3:
                    continue
                x_knots.append(float(np.median(x[in_bin])))
                y_knots.append(float(np.median(y[in_bin])))
            x_array = np.asarray(x_knots)
            y_array = _monotone_decreasing(np.asarray(y_knots))
            predicted = np.interp(x, x_array, y_array)
            sigma = float(np.quantile(np.abs(y - predicted), uncertainty_quantile))
            sigma = max(sigma, 0.015)
            curves[topology_class] = _TopologyCurve(
                feature_knots=x_array,
                psi_n_knots=y_array,
                psi_n_sigma=sigma,
                sample_count=int(x.size),
            )
        return cls(curves)

    def predict(
        self,
        log_temperature_ratio: np.ndarray,
        topology: TopologyClass,
    ) -> tuple[np.ndarray, np.ndarray]:
        feature = np.asarray(log_temperature_ratio, dtype=float)
        curve = self._curves.get(topology)
        if curve is None:
            raise ValueError(f"no calibration curve for {topology.value}")
        mean = np.interp(feature, curve.feature_knots, curve.psi_n_knots)
        sigma = np.full(mean.shape, curve.psi_n_sigma)
        return mean, sigma

    def coverage(
        self,
        log_temperature_ratio: np.ndarray,
        psi_n: np.ndarray,
        topology: np.ndarray,
    ) -> tuple[float, int]:
        feature = np.asarray(log_temperature_ratio, dtype=float)
        target = np.asarray(psi_n, dtype=float)
        topology = np.asarray(topology, dtype=str)
        covered: list[np.ndarray] = []
        for topology_class in self._curves:
            keep = (
                (topology == topology_class.value)
                & np.isfinite(feature)
                & np.isfinite(target)
            )
            if not np.any(keep):
                continue
            mean, sigma = self.predict(feature[keep], topology_class)
            covered.append(np.abs(target[keep] - mean) <= sigma)
        if not covered:
            return float("nan"), 0
        all_covered = np.concatenate(covered)
        return float(np.mean(all_covered)), int(all_covered.size)

    def to_dict(self) -> dict[str, Any]:
        return {
            topology.value: {
                "feature_knots": curve.feature_knots.tolist(),
                "psi_n_knots": curve.psi_n_knots.tolist(),
                "psi_n_sigma": curve.psi_n_sigma,
                "sample_count": curve.sample_count,
            }
            for topology, curve in self._curves.items()
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PedestalCalibration:
        curves = {
            TopologyClass(name): _TopologyCurve(
                feature_knots=np.asarray(values["feature_knots"], dtype=float),
                psi_n_knots=np.asarray(values["psi_n_knots"], dtype=float),
                psi_n_sigma=float(values["psi_n_sigma"]),
                sample_count=int(values["sample_count"]),
            )
            for name, values in payload.items()
        }
        return cls(curves)


class PedestalFootDetector:
    """Locate the separatrix by interpolating the calibrated ``psi_N=1`` foot."""

    def __init__(self, calibration: PedestalCalibration) -> None:
        self.calibration = calibration

    def locate(
        self,
        coordinate_m: np.ndarray,
        temperature_ev: np.ndarray,
        *,
        topology: TopologyClass,
        sigma_multiplier: np.ndarray | None = None,
    ) -> SeparatrixEstimate:
        coordinate = np.asarray(coordinate_m, dtype=float)
        temperature = np.asarray(temperature_ev, dtype=float)
        if coordinate.shape != temperature.shape or coordinate.ndim != 1:
            raise ValueError("coordinate and temperature must be matching vectors")
        positive = np.isfinite(temperature) & (temperature > 0.0)
        if np.count_nonzero(positive) < 2:
            raise ValueError("at least two positive finite temperatures are required")
        reference = float(np.nanpercentile(temperature[positive], 90.0))
        feature = np.log(np.maximum(temperature, 1.0e-12) / reference)
        mean, sigma = self.calibration.predict(feature, topology)
        if sigma_multiplier is not None:
            multiplier = np.asarray(sigma_multiplier, dtype=float)
            if multiplier.shape != sigma.shape or np.any(multiplier < 1.0):
                raise ValueError("sigma multiplier must match and be at least one")
            sigma = sigma * multiplier

        order = np.argsort(coordinate)
        coordinate = coordinate[order]
        mean = mean[order]
        sigma = sigma[order]
        crossings = np.flatnonzero((mean[:-1] - 1.0) * (mean[1:] - 1.0) <= 0.0)
        if crossings.size:
            index = int(crossings[np.argmin(np.abs(mean[crossings] - 1.0))])
            delta_psi = mean[index + 1] - mean[index]
            fraction = (1.0 - mean[index]) / delta_psi if delta_psi != 0.0 else 0.5
            fraction = float(np.clip(fraction, 0.0, 1.0))
            location = coordinate[index] + fraction * (
                coordinate[index + 1] - coordinate[index]
            )
            slope = abs(delta_psi / (coordinate[index + 1] - coordinate[index]))
            psi_sigma = float(np.hypot(sigma[index], sigma[index + 1]) / 2.0)
            coordinate_sigma = psi_sigma / max(slope, 1.0e-8)
            bracketed = True
        else:
            index = int(np.nanargmin(np.abs(mean - 1.0)))
            location = coordinate[index]
            psi_sigma = float(sigma[index])
            spacing = float(np.nanmedian(np.abs(np.diff(coordinate))))
            coordinate_sigma = spacing + psi_sigma
            bracketed = False
        return SeparatrixEstimate(
            coordinate_m=float(location),
            coordinate_sigma_m=float(coordinate_sigma),
            psi_n=1.0,
            psi_n_sigma=psi_sigma,
            bracketed=bracketed,
        )


def _monotone_decreasing(values: np.ndarray) -> np.ndarray:
    """Pool adjacent violations for a decreasing one-dimensional curve."""

    levels: list[float] = []
    weights: list[int] = []
    for value in values:
        levels.append(float(value))
        weights.append(1)
        while len(levels) >= 2 and levels[-2] < levels[-1]:
            weight = weights[-2] + weights[-1]
            level = (levels[-2] * weights[-2] + levels[-1] * weights[-1]) / weight
            levels[-2:] = [level]
            weights[-2:] = [weight]
    return np.concatenate(
        [np.full(weight, level) for level, weight in zip(levels, weights, strict=True)]
    )
