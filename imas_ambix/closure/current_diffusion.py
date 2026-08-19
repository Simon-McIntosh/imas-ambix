"""Cross-shot calibration of the resistive current-diffusion closure.

The extracted equilibrium bank is consumed only at derivative level.  Each
shot starts from its first extracted flux state and is then evolved with the
native flux-surface current-diffusion kernel.  The sole fitted quantity is a
bounded multiplicative correction to the axis resistivity of the existing
Sauter/Spitzer-informed profile; its contrast, shape, rank, and basis remain
fixed.  No per-shot or free-form profile correction is admitted.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np
from nova.transport.current_diffusion import (
    EtaProfile,
    FluxSurfaceGeometry,
    basis_projection_images,
    diffuse_psi,
    predicted_current,
    profile_shapes,
    project_coefficients,
    traced_assemble_flux_surface_geometry,
)
from scipy import optimize
from scipy.constants import mu_0

from imas_ambix.challenge.convention import DIIID_CONVENTION
from imas_ambix.challenge.loader import load_shot

if TYPE_CHECKING:
    from collections.abc import Callable

PROFILE_RANK = 3
CORRECTION_BOUNDS = (0.75, 1.25)
CORRECTION_PRIOR_LOG_SIGMA = 0.15
CORRECTION_PRIOR_WEIGHT = 0.01
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 917_431
MINIMUM_LI_CURRENT_A = 50_000.0

_PROFILE_COLUMNS = 2 * PROFILE_RANK
_TWO_PI = 2.0 * np.pi


@dataclass(frozen=True)
class CurrentDiffusionShot:
    """One derivative-level trajectory and its frozen initial geometry."""

    name: str
    times_s: np.ndarray
    current_a: np.ndarray
    rho_samples: np.ndarray
    truth_ff_prime: np.ndarray
    separation_quality: np.ndarray
    valid_frames: np.ndarray
    label_li: np.ndarray
    geometry: FluxSurfaceGeometry
    initial_psi_face: np.ndarray
    projection_images: dict[str, np.ndarray]


@dataclass(frozen=True)
class CorrectionFit:
    """Receipt for the tightly regularized scalar resistivity correction."""

    multiplier: float
    objective: float
    data_loss: float
    prior_penalty: float
    evaluations: int
    success: bool
    message: str


@dataclass(frozen=True)
class ArmPrediction:
    """Transport-evolved FF-prime and internal-inductance trajectories."""

    ff_prime: np.ndarray
    li: np.ndarray


def corrected_resistivity(baseline: EtaProfile, multiplier: float) -> EtaProfile:
    """Apply the allowed scalar correction without changing profile shape."""

    lower, upper = CORRECTION_BOUNDS
    if not np.isfinite(multiplier) or not lower <= multiplier <= upper:
        raise ValueError(f"resistivity multiplier must lie in [{lower}, {upper}]")
    return EtaProfile(
        eta0=baseline.eta0 * float(multiplier),
        contrast=baseline.contrast,
        shape=baseline.shape,
    )


def fit_resistivity_correction(
    score_multiplier: Callable[[float], float],
    *,
    prior_log_sigma: float = CORRECTION_PRIOR_LOG_SIGMA,
    prior_weight: float = CORRECTION_PRIOR_WEIGHT,
) -> CorrectionFit:
    """Fit one bounded multiplier with a log-normal prior centred on unity."""

    if prior_log_sigma <= 0.0 or prior_weight < 0.0:
        raise ValueError("correction prior parameters must be positive")
    evaluations = 0
    cache: dict[float, tuple[float, float, float]] = {}

    def objective(log_multiplier: float) -> float:
        nonlocal evaluations
        multiplier = float(np.exp(log_multiplier))
        key = float(np.round(multiplier, 10))
        if key not in cache:
            data_loss = float(score_multiplier(multiplier))
            penalty = prior_weight * (log_multiplier / prior_log_sigma) ** 2
            if not np.isfinite(data_loss):
                raise ValueError("correction score must be finite")
            cache[key] = (data_loss + penalty, data_loss, penalty)
            evaluations += 1
        return cache[key][0]

    lower, upper = np.log(CORRECTION_BOUNDS)
    result = optimize.minimize_scalar(
        objective,
        bounds=(lower, upper),
        method="bounded",
        options={"xatol": 2.0e-4, "maxiter": 24},
    )
    multiplier = float(np.exp(result.x))
    total = objective(float(result.x))
    _, data_loss, penalty = cache[float(np.round(multiplier, 10))]
    return CorrectionFit(
        multiplier=multiplier,
        objective=float(total),
        data_loss=float(data_loss),
        prior_penalty=float(penalty),
        evaluations=evaluations,
        success=bool(result.success),
        message=str(result.message),
    )


def paired_bootstrap_comparison(
    tuned_error: np.ndarray,
    untuned_error: np.ndarray,
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Compare paired shot errors; negative differences favour calibration."""

    tuned = np.asarray(tuned_error, dtype=float)
    untuned = np.asarray(untuned_error, dtype=float)
    if tuned.shape != untuned.shape or tuned.ndim != 1 or tuned.size < 2:
        raise ValueError("paired comparison needs equally sized one-dimensional arms")
    if np.any(~np.isfinite(tuned)) or np.any(~np.isfinite(untuned)):
        raise ValueError("paired comparison errors must be finite")
    if draws < 1:
        raise ValueError("at least one bootstrap draw is required")
    differences = tuned - untuned
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, tuned.size, size=(draws, tuned.size))
    draw_means = differences[indices].mean(axis=1)
    interval = np.quantile(draw_means, (0.025, 0.975))
    mean_difference = float(np.mean(differences))
    verdict = "BEAT" if float(interval[1]) < 0.0 else "MISS"
    return {
        "tuned_error_mean": float(np.mean(tuned)),
        "untuned_error_mean": float(np.mean(untuned)),
        "paired_difference_tuned_minus_untuned": mean_difference,
        "paired_bootstrap_confidence_interval_95": [
            float(interval[0]),
            float(interval[1]),
        ],
        "bootstrap_draws": int(draws),
        "bootstrap_seed": int(seed),
        "verdict": verdict,
        "verdict_rule": (
            "BEAT only when the complete confidence interval is below zero; "
            "otherwise MISS"
        ),
    }


def calibrate_current_diffusion_closure(
    source_dir: str | Path,
    bank_dir: str | Path,
    output_path: str | Path,
    *,
    bootstrap_draws: int = BOOTSTRAP_DRAWS,
) -> dict[str, Any]:
    """Fit on fifteen shots, evaluate five untouched shots, and bank evidence."""

    source_root = Path(source_dir)
    bank_root = Path(bank_dir)
    target = Path(output_path)
    pairs = _paired_paths(source_root, bank_root)
    if len(pairs) != 20:
        message = (
            f"the frozen calibration bank must contain 20 shots, found {len(pairs)}"
        )
        raise RuntimeError(message)
    split = 15
    fit_names = [
        bank.stem.removesuffix("_flux_trajectory") for _, bank in pairs[:split]
    ]
    heldout_names = [
        bank.stem.removesuffix("_flux_trajectory") for _, bank in pairs[split:]
    ]
    baseline = EtaProfile()
    payload: dict[str, Any] = {
        "status": "loading_corpus",
        "schema_version": 1,
        "method": {
            "label_consumption": "derivative_level_extracted_ff_prime_and_li",
            "transport_kernel": "native_flux_surface_resistive_current_diffusion",
            "initial_condition": "first_extracted_flux_state_per_shot",
            "score": "outer_weighted_nrmse_of_change_from_initial_ff_prime",
            "profile_family": "bounded_sauter_spitzer_informed_resistivity",
            "correction": "single_cross_shot_multiplicative_axis_resistivity",
            "free_form_profile_admitted": False,
            "basis_rank": PROFILE_RANK,
            "basis_columns": _PROFILE_COLUMNS,
            "same_rank_and_basis_in_both_arms": True,
            "geometry": "initial_extracted_flux_surface_geometry_frozen_per_shot",
            "convention": {
                "source_cocos": DIIID_CONVENTION.source_cocos,
                "target_cocos": DIIID_CONVENTION.target_cocos,
                "banked_total_flux_factor": DIIID_CONVENTION.total_flux_to_canonical,
            },
        },
        "corpus": {
            "banked_shots": len(pairs),
            "fit_shot_count": split,
            "heldout_shot_count": len(pairs) - split,
            "fit_shots": fit_names,
            "heldout_shots": heldout_names,
            "split_disjoint": not bool(set(fit_names) & set(heldout_names)),
        },
        "untuned_closure": _eta_receipt(baseline),
    }
    _write_increment(target, payload)

    shots = [_load_shot(source, bank) for source, bank in pairs]
    padded_length = max(len(shot.times_s) for shot in shots)
    fit_shots = shots[:split]
    heldout_shots = shots[split:]
    prediction_cache: dict[tuple[str, float], ArmPrediction] = {}

    def predict(shot: CurrentDiffusionShot, multiplier: float) -> ArmPrediction:
        key = (shot.name, float(np.round(multiplier, 10)))
        if key not in prediction_cache:
            prediction_cache[key] = _evolve(
                shot,
                corrected_resistivity(baseline, multiplier),
                padded_length,
            )
        return prediction_cache[key]

    def score_multiplier(multiplier: float) -> float:
        errors = [
            _ff_prime_error(shot, predict(shot, multiplier).ff_prime)
            for shot in fit_shots
        ]
        return float(np.mean(np.square(errors)))

    payload["status"] = "fitting_correction"
    payload["corpus"]["banked_frames"] = int(sum(len(shot.times_s) for shot in shots))
    _write_increment(target, payload)
    correction = fit_resistivity_correction(score_multiplier)
    tuned = corrected_resistivity(baseline, correction.multiplier)
    payload["fit"] = {
        "multiplicative_correction": correction.multiplier,
        "correction_bounds": list(CORRECTION_BOUNDS),
        "prior": {
            "distribution": "log_normal_centred_on_one",
            "log_sigma": CORRECTION_PRIOR_LOG_SIGMA,
            "objective_weight": CORRECTION_PRIOR_WEIGHT,
        },
        "objective": correction.objective,
        "data_loss": correction.data_loss,
        "prior_penalty": correction.prior_penalty,
        "evaluations": correction.evaluations,
        "success": correction.success,
        "message": correction.message,
        "tuned_closure": _eta_receipt(tuned),
        "shape_unchanged": bool(
            tuned.contrast == baseline.contrast and tuned.shape == baseline.shape
        ),
    }
    payload["status"] = "evaluating_heldout"
    _write_increment(target, payload)

    heldout_rows = []
    tuned_errors = []
    untuned_errors = []
    tuned_li_parts = []
    untuned_li_parts = []
    label_li_parts = []
    for shot in heldout_shots:
        tuned_prediction = predict(shot, correction.multiplier)
        untuned_prediction = predict(shot, 1.0)
        tuned_error = _ff_prime_error(shot, tuned_prediction.ff_prime)
        untuned_error = _ff_prime_error(shot, untuned_prediction.ff_prime)
        tuned_errors.append(tuned_error)
        untuned_errors.append(untuned_error)
        heldout_rows.append(
            {
                "shot": shot.name,
                "tuned_ff_prime_error": tuned_error,
                "untuned_ff_prime_error": untuned_error,
                "paired_difference_tuned_minus_untuned": tuned_error - untuned_error,
                "frames": len(shot.times_s),
            }
        )
        li_mask = (
            shot.valid_frames
            & np.isfinite(shot.label_li)
            & np.isfinite(tuned_prediction.li)
            & np.isfinite(untuned_prediction.li)
            & (shot.current_a >= MINIMUM_LI_CURRENT_A)
        )
        if np.any(li_mask):
            tuned_li_parts.append(tuned_prediction.li[li_mask])
            untuned_li_parts.append(untuned_prediction.li[li_mask])
            label_li_parts.append(shot.label_li[li_mask])
    comparison = paired_bootstrap_comparison(
        np.asarray(tuned_errors),
        np.asarray(untuned_errors),
        draws=bootstrap_draws,
    )
    payload["comparison"] = {
        **comparison,
        "heldout_shots": heldout_rows,
    }
    payload["li_diagnostic"] = _li_diagnostic(
        tuned_li_parts,
        untuned_li_parts,
        label_li_parts,
    )
    payload["status"] = "complete"
    _write_increment(target, payload)
    return payload


def _paired_paths(source_dir: Path, bank_dir: Path) -> list[tuple[Path, Path]]:
    pairs = []
    for bank_path in sorted(bank_dir.glob("*_flux_trajectory.npz")):
        name = bank_path.name.removesuffix("_flux_trajectory.npz")
        source_path = source_dir / f"{name}.parquet"
        if source_path.exists():
            pairs.append((source_path, bank_path))
    return pairs


def _load_shot(source_path: Path, bank_path: Path) -> CurrentDiffusionShot:
    shot = load_shot(source_path)
    with np.load(bank_path, allow_pickle=False) as loaded:
        bank = {name: loaded[name] for name in loaded.files}
    frame_count = min(len(shot.labels.time_ms), len(bank["times_ms"]))
    times_ms = np.asarray(shot.labels.time_ms[:frame_count], dtype=float)
    current_series = shot.actuators["magnetics_plasma_current"]
    current_a = np.abs(
        np.interp(times_ms, current_series.time_ms, current_series.values) * 1.0e3
    )
    major_radius = float(shot.labels.scalars["efit_r_axis"][0])
    total_flux_factor = DIIID_CONVENTION.total_flux_to_canonical
    p_prime = np.asarray(bank["p_prime"], dtype=float) / total_flux_factor
    ff_prime = np.asarray(bank["ff_prime"], dtype=float) / total_flux_factor
    coefficients = _profile_coefficients(
        np.asarray(bank["surface_psi_n"][0], dtype=float),
        p_prime[0],
        ff_prime[0],
        major_radius,
    )
    axis_flux = float(DIIID_CONVENTION.canonical_total_flux(bank["axis_flux_wb"][0]))
    boundary_flux = float(
        DIIID_CONVENTION.canonical_total_flux(bank["boundary_flux_wb"][0])
    )
    boundary_field = _boundary_field_function(
        bank,
        float(shot.labels.scalars["efit_q95"][0]),
        major_radius,
    )
    assembled = traced_assemble_flux_surface_geometry(
        _surface_bins(bank),
        jnp.asarray(shot.labels.psirz[0]),
        jnp.asarray(shot.labels.grid_r_m),
        jnp.asarray(shot.labels.grid_z_m),
        jnp.ones(shot.labels.psirz[0].shape, dtype=bool),
        axis_psi=axis_flux,
        boundary_psi=boundary_flux,
        profile_coefficients=jnp.asarray(coefficients),
        coefficient_scale=jnp.ones(_PROFILE_COLUMNS),
        ip_amperes=float(current_a[0]),
        major_radius=major_radius,
        boundary_toroidal_field=boundary_field,
        n_pressure=PROFILE_RANK,
        n_diamagnetic=PROFILE_RANK,
        n_radial_cells=len(bank["rho_hat_samples"]),
        nonnegative=False,
    )
    if not bool(assembled["valid"]):
        message = f"invalid extracted flux-surface geometry for {bank_path.stem}"
        raise ValueError(message)
    geometry = _as_geometry(assembled)
    psi_on_rho = DIIID_CONVENTION.canonical_total_flux(bank["psi_on_rho_wb"])
    initial_psi_face = np.interp(
        geometry.rho_face,
        bank["rho_hat_samples"],
        psi_on_rho[0],
        left=axis_flux,
        right=boundary_flux,
    )
    images = basis_projection_images(
        geometry,
        np.ones(_PROFILE_COLUMNS),
        n_pressure=PROFILE_RANK,
        n_diamagnetic=PROFILE_RANK,
        nonneg=False,
    )
    return CurrentDiffusionShot(
        name=bank_path.name.removesuffix("_flux_trajectory.npz"),
        times_s=(times_ms - times_ms[0]) * 1.0e-3,
        current_a=current_a,
        rho_samples=np.asarray(bank["rho_hat_samples"], dtype=float),
        truth_ff_prime=np.asarray(bank["ff_prime_on_rho"][:frame_count], dtype=float)
        / total_flux_factor,
        separation_quality=np.asarray(
            bank["separation_fit_fraction"][:frame_count], dtype=float
        ),
        valid_frames=(
            np.asarray(bank["vacuum_pass"][:frame_count], dtype=bool)
            & np.asarray(bank["fsa_well_posed"][:frame_count], dtype=bool)
        ),
        label_li=np.asarray(shot.labels.scalars["efit_li"][:frame_count], dtype=float),
        geometry=geometry,
        initial_psi_face=initial_psi_face,
        projection_images=images,
    )


def _surface_bins(bank: dict[str, np.ndarray]) -> dict[str, jnp.ndarray]:
    return {
        "pn_s": jnp.asarray(bank["fsa_psi_n"][0]),
        "dv_dpn": jnp.asarray(bank["fsa_dv_dpsi_n"][0]),
        "inv_r2": jnp.asarray(bank["fsa_inverse_r2"][0]),
        "inv_r": jnp.asarray(bank["fsa_inverse_r"][0]),
        "grad2_r2": jnp.asarray(bank["fsa_gradient2_over_r2"][0]),
        "v_cum": jnp.asarray(bank["fsa_cumulative_volume"][0]),
        "v_total": jnp.asarray(bank["fsa_volume"][0]),
        "well_posed": jnp.asarray(bank["fsa_well_posed"][0]),
    }


def _profile_coefficients(
    psi_n: np.ndarray,
    p_prime: np.ndarray,
    ff_prime: np.ndarray,
    major_radius: float,
) -> np.ndarray:
    basis = profile_shapes(psi_n, PROFILE_RANK, nonneg=False)
    pressure_drive = -_TWO_PI * major_radius * p_prime
    diamagnetic_drive = -_TWO_PI * ff_prime / (mu_0 * major_radius)
    pressure = np.linalg.lstsq(basis, pressure_drive, rcond=None)[0]
    diamagnetic = np.linalg.lstsq(basis, diamagnetic_drive, rcond=None)[0]
    return np.concatenate((pressure, diamagnetic))


def _boundary_field_function(
    bank: dict[str, np.ndarray], q95: float, major_radius: float
) -> float:
    span = abs(float(bank["boundary_flux_wb"][0] - bank["axis_flux_wb"][0]))
    metric = (
        bank["fsa_inverse_r2"][0, -1]
        * bank["fsa_dv_dpsi_n"][0, -1]
        / (_TWO_PI * max(span, 1.0e-12))
    )
    field_function = abs(float(q95)) / max(float(metric), 1.0e-12)
    return float(np.clip(field_function, 0.5, 12.0)) / major_radius


def _as_geometry(result: dict[str, jnp.ndarray]) -> FluxSurfaceGeometry:
    values = {}
    for key, value in result.items():
        if key == "valid":
            continue
        array = np.asarray(value)
        values[key] = float(array) if array.ndim == 0 else array
    return FluxSurfaceGeometry(**values)


def _evolve(
    shot: CurrentDiffusionShot, eta: EtaProfile, padded_length: int
) -> ArmPrediction:
    padding = padded_length - len(shot.times_s)
    times = np.pad(shot.times_s, (0, padding), mode="edge")
    current = np.pad(shot.current_a, (0, padding), mode="edge")
    result = diffuse_psi(
        shot.geometry,
        eta,
        t_grid=times,
        ip_of_t=current,
        psi0_face=shot.initial_psi_face,
    )
    face_history = np.asarray(result["psi_face"][: len(shot.times_s)])
    predicted = np.full_like(shot.truth_ff_prime, np.nan)
    li = np.full(len(shot.times_s), np.nan)
    basis = profile_shapes(shot.geometry.psi_n_cell, PROFILE_RANK, nonneg=False)
    for frame in range(len(shot.times_s)):
        if frame == 0:
            adjacent = 1
            delta_time = float(shot.times_s[adjacent] - shot.times_s[frame])
            flux_rate = (face_history[adjacent] - face_history[frame]) / delta_time
        else:
            adjacent = frame - 1
            delta_time = float(shot.times_s[frame] - shot.times_s[adjacent])
            flux_rate = (face_history[frame] - face_history[adjacent]) / delta_time
        if delta_time <= 0.0:
            continue
        currents = predicted_current(
            shot.geometry,
            face_history[frame],
            flux_rate,
            eta,
        )
        coefficients = project_coefficients(
            shot.geometry,
            shot.projection_images,
            currents["j_tor"],
            currents["j_par_b"],
            nonneg=False,
        )
        if coefficients is not None:
            diamagnetic_drive = basis @ coefficients[PROFILE_RANK:]
            cell_ff_prime = -diamagnetic_drive * mu_0 * shot.geometry.r0 / _TWO_PI
            predicted[frame] = np.interp(
                shot.rho_samples,
                shot.geometry.rho_cell,
                cell_ff_prime,
                left=cell_ff_prime[0],
                right=cell_ff_prime[-1],
            )
        li[frame] = _internal_inductance(
            shot.geometry,
            face_history[frame],
            shot.current_a[frame],
        )
    predicted = predicted - predicted[0] + shot.truth_ff_prime[0]
    return ArmPrediction(ff_prime=predicted, li=li)


def _internal_inductance(
    geometry: FluxSurfaceGeometry, psi_face: np.ndarray, current_a: float
) -> float:
    if not np.isfinite(current_a) or abs(current_a) < 1.0:
        return float("nan")
    enclosed_current = np.asarray(geometry.enclosed_current(psi_face), dtype=float)
    rho = np.asarray(geometry.rho_face, dtype=float)
    finite = np.isfinite(enclosed_current) & np.isfinite(rho) & (rho > 0.0)
    if np.count_nonzero(finite) < 3:
        return float("nan")
    current_fraction = enclosed_current[finite] / current_a
    return float(
        2.0
        * np.trapezoid(
            np.square(current_fraction) / rho[finite],
            rho[finite],
        )
    )


def _ff_prime_error(shot: CurrentDiffusionShot, prediction: np.ndarray) -> float:
    truth_change = shot.truth_ff_prime - shot.truth_ff_prime[0]
    prediction_change = prediction - prediction[0]
    radial_weight = 0.25 + 0.75 * shot.rho_samples**2
    weights = np.clip(shot.separation_quality, 0.0, 1.0) * radial_weight
    weights *= shot.valid_frames[:, np.newaxis]
    finite = (
        np.isfinite(truth_change)
        & np.isfinite(prediction_change)
        & np.isfinite(weights)
        & (weights > 0.0)
    )
    if np.count_nonzero(finite) < 2:
        raise ValueError(f"insufficient valid FF-prime values for {shot.name}")
    selected_weight = weights[finite]
    residual = truth_change[finite] - prediction_change[finite]
    scale = np.sqrt(
        np.average(np.square(truth_change[finite]), weights=selected_weight)
    )
    return float(
        np.sqrt(np.average(np.square(residual), weights=selected_weight))
        / max(float(scale), 1.0e-12)
    )


def _li_diagnostic(
    tuned_parts: list[np.ndarray],
    untuned_parts: list[np.ndarray],
    label_parts: list[np.ndarray],
) -> dict[str, Any]:
    if not label_parts:
        raise ValueError("no eligible held-out internal-inductance labels")
    tuned = np.concatenate(tuned_parts)
    untuned = np.concatenate(untuned_parts)
    label = np.concatenate(label_parts)
    reference = float(np.mean(label))
    denominator = float(np.sum(np.square(label - reference)))
    tuned_skill = 1.0 - float(np.sum(np.square(label - tuned))) / max(
        denominator, 1.0e-30
    )
    untuned_skill = 1.0 - float(np.sum(np.square(label - untuned))) / max(
        denominator, 1.0e-30
    )
    return {
        "definition": "one_minus_squared_error_over_heldout_label_mean_baseline",
        "label_channel": "efit_li",
        "interpretation": "derivative_level_closure_diagnostic_not_superiority_claim",
        "minimum_plasma_current_a": MINIMUM_LI_CURRENT_A,
        "eligible_frames": int(label.size),
        "tuned_skill": tuned_skill,
        "untuned_skill": untuned_skill,
    }


def _eta_receipt(eta: EtaProfile) -> dict[str, float]:
    return {
        "eta0_ohm_m": float(eta.eta0),
        "edge_axis_log_contrast": float(eta.contrast),
        "shape_exponent": float(eta.shape),
    }


def _write_increment(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("bank_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--bootstrap-draws", type=int, default=BOOTSTRAP_DRAWS)
    arguments = parser.parse_args()
    result = calibrate_current_diffusion_closure(
        arguments.source_dir,
        arguments.bank_dir,
        arguments.output,
        bootstrap_draws=arguments.bootstrap_draws,
    )
    print(json.dumps(result["comparison"], indent=2))
    print(json.dumps(result["li_diagnostic"], indent=2))


if __name__ == "__main__":
    main()
