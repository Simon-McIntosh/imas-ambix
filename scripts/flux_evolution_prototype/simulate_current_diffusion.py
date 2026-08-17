#!/usr/bin/env python3
"""Measure native current-diffusion forecasts against extracted trajectories.

The score is evaluated on change from each shot's extracted initial state. This
keeps the comparison about evolution rather than static cross-shot profile
offsets. One bounded resistivity profile is fitted across every shot; no slice
or shot receives its own closure parameters.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
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

PROFILE_TERMS = 3
PROFILE_COLUMNS = 2 * PROFILE_TERMS
TWO_PI = 2.0 * np.pi


@dataclass
class Shot:
    name: str
    times_s: np.ndarray
    current_a: np.ndarray
    normalised_current: np.ndarray
    rho_samples: np.ndarray
    truth_psi: np.ndarray
    truth_ff_prime: np.ndarray
    geometry: FluxSurfaceGeometry
    initial_psi_face: np.ndarray
    projection_images: dict[str, np.ndarray]
    source_path: str
    fsa_geometry_mode: str = "frozen_initial_frame"


def _profile_coefficients(
    psi_n: np.ndarray, p_prime: np.ndarray, ff_prime: np.ndarray, major_radius: float
) -> np.ndarray:
    basis = profile_shapes(psi_n, PROFILE_TERMS, nonneg=False)
    pressure_drive = -TWO_PI * major_radius * p_prime
    diamagnetic_drive = -TWO_PI * ff_prime / (mu_0 * major_radius)
    pressure = np.linalg.lstsq(basis, pressure_drive, rcond=None)[0]
    diamagnetic = np.linalg.lstsq(basis, diamagnetic_drive, rcond=None)[0]
    return np.concatenate((pressure, diamagnetic))


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


def _boundary_field_function(
    bank: dict[str, np.ndarray], q95: float, major_radius: float
) -> float:
    span = abs(float(bank["boundary_flux_wb"][0] - bank["axis_flux_wb"][0]))
    metric = (
        bank["fsa_inverse_r2"][0, -1]
        * bank["fsa_dv_dpsi_n"][0, -1]
        / (TWO_PI * max(span, 1e-12))
    )
    field_function = abs(float(q95)) / max(float(metric), 1e-12)
    field_function = float(np.clip(field_function, 0.5, 12.0))
    return field_function / major_radius


def _as_geometry(result: dict[str, jnp.ndarray]) -> FluxSurfaceGeometry:
    values = {}
    for key, value in result.items():
        if key == "valid":
            continue
        array = np.asarray(value)
        values[key] = float(array) if array.ndim == 0 else array
    return FluxSurfaceGeometry(**values)


def load_shot(bank_path: Path, source_path: Path) -> Shot:
    with np.load(bank_path) as loaded:
        bank = {key: loaded[key] for key in loaded.files}
    columns = [
        "efit_times",
        "efit_psirz",
        "efit_q95",
        "efit_grid_R",
        "efit_grid_Z",
        "efit_r_axis",
        "magnetics_plasma_current",
        "magnetics_plasma_current_times",
    ]
    row = pq.read_table(source_path, columns=columns).to_pylist()[0]
    times_ms = np.asarray(row["efit_times"], dtype=np.float64)
    current_ka = np.abs(
        np.interp(
            times_ms,
            np.asarray(row["magnetics_plasma_current_times"], dtype=np.float64),
            np.asarray(row["magnetics_plasma_current"], dtype=np.float64),
        )
    )
    peak_current = float(np.quantile(current_ka, 0.95))
    normalised_current = current_ka / max(peak_current, 1e-9)
    major_radius = float(np.asarray(row["efit_r_axis"], dtype=np.float64)[0])
    coefficients = _profile_coefficients(
        bank["surface_psi_n"][0],
        bank["p_prime"][0],
        bank["ff_prime"][0],
        major_radius,
    )
    boundary_field = _boundary_field_function(
        bank, np.asarray(row["efit_q95"], dtype=np.float64)[0], major_radius
    )
    initial_map = TWO_PI * np.asarray(row["efit_psirz"], dtype=np.float64)[0]
    radius = np.asarray(row["efit_grid_R"], dtype=np.float64)
    height = np.asarray(row["efit_grid_Z"], dtype=np.float64)
    assembled = traced_assemble_flux_surface_geometry(
        _surface_bins(bank),
        jnp.asarray(initial_map),
        jnp.asarray(radius),
        jnp.asarray(height),
        jnp.ones(initial_map.shape, dtype=bool),
        axis_psi=float(bank["axis_flux_wb"][0]),
        boundary_psi=float(bank["boundary_flux_wb"][0]),
        profile_coefficients=jnp.asarray(coefficients),
        coefficient_scale=jnp.ones(PROFILE_COLUMNS),
        ip_amperes=float(current_ka[0] * 1e3),
        major_radius=major_radius,
        boundary_toroidal_field=boundary_field,
        n_pressure=PROFILE_TERMS,
        n_diamagnetic=PROFILE_TERMS,
        n_radial_cells=len(bank["rho_hat_samples"]),
        nonnegative=False,
    )
    if not bool(assembled["valid"]):
        raise ValueError(f"native FSA geometry is invalid for {bank_path.stem}")
    geometry = _as_geometry(assembled)
    initial_psi_face = np.interp(
        geometry.rho_face,
        bank["rho_hat_samples"],
        bank["psi_on_rho_wb"][0],
        left=float(bank["axis_flux_wb"][0]),
        right=float(bank["boundary_flux_wb"][0]),
    )
    images = basis_projection_images(
        geometry,
        np.ones(PROFILE_COLUMNS),
        n_pressure=PROFILE_TERMS,
        n_diamagnetic=PROFILE_TERMS,
        nonneg=False,
    )
    return Shot(
        name=bank_path.name.removesuffix("_flux_trajectory.npz"),
        times_s=(times_ms - times_ms[0]) * 1e-3,
        current_a=current_ka * 1e3,
        normalised_current=normalised_current,
        rho_samples=bank["rho_hat_samples"],
        truth_psi=bank["psi_on_rho_wb"],
        truth_ff_prime=bank["ff_prime_on_rho"],
        geometry=geometry,
        initial_psi_face=initial_psi_face,
        projection_images=images,
        source_path=str(source_path),
    )


def _padded_drive(shot: Shot, length: int) -> tuple[np.ndarray, np.ndarray]:
    padding = length - len(shot.times_s)
    times = np.pad(shot.times_s, (0, padding), mode="edge")
    current = np.pad(shot.current_a, (0, padding), mode="edge")
    return times, current


def evolve_psi(shot: Shot, eta: EtaProfile, padded_length: int) -> np.ndarray:
    times, current = _padded_drive(shot, padded_length)
    result = diffuse_psi(
        shot.geometry,
        eta,
        t_grid=times,
        ip_of_t=current,
        psi0_face=shot.initial_psi_face,
    )
    face_history = result["psi_face"][: len(shot.times_s)]
    return np.vstack(
        [
            np.interp(shot.rho_samples, shot.geometry.rho_face, row)
            for row in face_history
        ]
    )


def recover_ff_prime(
    shot: Shot, eta: EtaProfile, padded_length: int
) -> np.ndarray:
    times, current = _padded_drive(shot, padded_length)
    result = diffuse_psi(
        shot.geometry,
        eta,
        t_grid=times,
        ip_of_t=current,
        psi0_face=shot.initial_psi_face,
    )
    face_history = result["psi_face"][: len(shot.times_s)]
    predicted = np.full_like(shot.truth_ff_prime, np.nan)
    basis = profile_shapes(shot.geometry.psi_n_cell, PROFILE_TERMS, nonneg=False)
    for frame in range(len(shot.times_s)):
        if frame == 0:
            delta_time = shot.times_s[1] - shot.times_s[0]
            flux_rate = (face_history[1] - face_history[0]) / delta_time
        else:
            delta_time = shot.times_s[frame] - shot.times_s[frame - 1]
            flux_rate = (face_history[frame] - face_history[frame - 1]) / delta_time
        currents = predicted_current(shot.geometry, face_history[frame], flux_rate, eta)
        coefficients = project_coefficients(
            shot.geometry,
            shot.projection_images,
            currents["j_tor"],
            currents["j_par_b"],
            nonneg=False,
        )
        if coefficients is None:
            continue
        diamagnetic_drive = basis @ coefficients[PROFILE_TERMS:]
        cell_ff_prime = -diamagnetic_drive * mu_0 * shot.geometry.r0 / TWO_PI
        predicted[frame] = np.interp(
            shot.rho_samples,
            shot.geometry.rho_cell,
            cell_ff_prime,
            left=cell_ff_prime[0],
            right=cell_ff_prime[-1],
        )
    return predicted


def _align_initial(prediction: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return prediction - prediction[0] + truth[0]


def _normalised_psi_loss(
    shots: list[Shot], eta: EtaProfile, padded_length: int
) -> float:
    losses = []
    for shot in shots:
        prediction = _align_initial(
            evolve_psi(shot, eta, padded_length), shot.truth_psi
        )
        truth_change = shot.truth_psi - shot.truth_psi[0]
        predicted_change = prediction - prediction[0]
        finite = np.isfinite(truth_change) & np.isfinite(predicted_change)
        scale = np.nanmean(truth_change[finite] ** 2)
        loss = np.nanmean((truth_change[finite] - predicted_change[finite]) ** 2)
        losses.append(loss / max(scale, 1e-12))
    return float(np.mean(losses))


def fit_resistivity(shots: list[Shot], padded_length: int) -> tuple[EtaProfile, dict]:
    bounds = [
        (np.log10(EtaProfile.BOUNDS[0][0]), np.log10(EtaProfile.BOUNDS[0][1])),
        EtaProfile.BOUNDS[1],
        EtaProfile.BOUNDS[2],
    ]
    cache: dict[tuple[float, ...], float] = {}

    def objective(vector):
        key = tuple(np.round(vector, 10))
        if key not in cache:
            cache[key] = _normalised_psi_loss(
                shots, EtaProfile.from_vector(vector), padded_length
            )
        return cache[key]

    default = EtaProfile()
    initial = default.as_vector()
    shaped_result = optimize.minimize(
        objective,
        initial,
        method="Powell",
        bounds=bounds,
        options={"maxiter": 35, "xtol": 2e-3, "ftol": 2e-4},
    )
    constant_result = optimize.minimize_scalar(
        lambda log_eta: objective(np.array([log_eta, 0.0, 2.0])),
        bounds=bounds[0],
        method="bounded",
        options={"xatol": 2e-3},
    )
    shaped = EtaProfile.from_vector(shaped_result.x)
    constant = EtaProfile.from_vector(
        np.array([constant_result.x, 0.0, 2.0], dtype=np.float64)
    )
    shaped_loss = objective(shaped.as_vector())
    constant_loss = objective(constant.as_vector())
    if constant_loss <= shaped_loss:
        fitted = constant
        selected_family = "constant_boundary_solution"
        selected_success = bool(constant_result.success)
        selected_message = str(constant_result.message)
    else:
        fitted = shaped
        selected_family = "shaped_profile"
        selected_success = bool(shaped_result.success)
        selected_message = str(shaped_result.message)
    receipt = {
        "success": selected_success,
        "message": selected_message,
        "selected_family": selected_family,
        "evaluations": int(shaped_result.nfev + constant_result.nfev),
        "shaped_iterations": int(shaped_result.nit),
        "default_loss": objective(initial),
        "fitted_loss": objective(fitted.as_vector()),
        "shaped_loss": shaped_loss,
        "constant_loss": constant_loss,
    }
    return fitted, receipt


def _regime_mask(shot: Shot, regime: str) -> np.ndarray:
    mask = np.ones(len(shot.times_s), dtype=bool)
    if regime == "flat_top":
        mask &= shot.normalised_current >= 0.8
    elif regime == "ramp":
        mask &= (shot.normalised_current >= 0.15) & (shot.normalised_current < 0.8)
    return mask


def _score_arrays(
    truth: np.ndarray, prediction: np.ndarray, frame_mask: np.ndarray
) -> tuple[float, float, int]:
    truth_change = truth - truth[0]
    prediction_change = prediction - prediction[0]
    mask = np.broadcast_to(frame_mask[:, np.newaxis], truth.shape)
    finite = mask & np.isfinite(truth_change) & np.isfinite(prediction_change)
    observed = truth_change[finite]
    modeled = prediction_change[finite]
    if observed.size < 2:
        return np.nan, np.nan, int(observed.size)
    residual = observed - modeled
    denominator = np.sum((observed - np.mean(observed)) ** 2)
    explained = 1.0 - np.sum(residual**2) / max(denominator, 1e-30)
    return float(explained), float(np.mean(residual**2)), int(observed.size)


def _all_predictions(shots: list[Shot], eta: EtaProfile, padded_length: int):
    predictions = {}
    for shot in shots:
        psi = evolve_psi(shot, eta, padded_length)
        ff_prime = recover_ff_prime(shot, eta, padded_length)
        predictions[shot.name] = {
            "psi": _align_initial(psi, shot.truth_psi),
            "ff_prime": _align_initial(ff_prime, shot.truth_ff_prime),
        }
    return predictions


def build_table(shots, tuned, default):
    rows = []
    pooled = {}
    for regime in ("all", "ramp", "flat_top"):
        for target in ("ff_prime", "psi"):
            for model in ("tuned_transport", "default_transport", "persistence"):
                pooled[(regime, target, model)] = {"truth": [], "prediction": []}
    for shot in shots:
        truths = {"ff_prime": shot.truth_ff_prime, "psi": shot.truth_psi}
        models = {
            "tuned_transport": tuned[shot.name],
            "default_transport": default[shot.name],
            "persistence": {
                key: np.broadcast_to(value[0], value.shape)
                for key, value in truths.items()
            },
        }
        for regime in ("all", "ramp", "flat_top"):
            frame_mask = _regime_mask(shot, regime)
            for target, truth in truths.items():
                for model, values in models.items():
                    prediction = values[target]
                    explained, mse, count = _score_arrays(truth, prediction, frame_mask)
                    rows.append(
                        {
                            "scope": shot.name,
                            "regime": regime,
                            "target": target,
                            "model": model,
                            "frames": int(np.count_nonzero(frame_mask)),
                            "values": count,
                            "explained_variance": explained,
                            "mse": mse,
                        }
                    )
                    truth_change = truth - truth[0]
                    prediction_change = prediction - prediction[0]
                    expanded = np.broadcast_to(frame_mask[:, np.newaxis], truth.shape)
                    finite = (
                        expanded
                        & np.isfinite(truth_change)
                        & np.isfinite(prediction_change)
                    )
                    pooled[(regime, target, model)]["truth"].append(
                        truth_change[finite]
                    )
                    pooled[(regime, target, model)]["prediction"].append(
                        prediction_change[finite]
                    )
    for (regime, target, model), values in pooled.items():
        truth = np.concatenate(values["truth"])
        prediction = np.concatenate(values["prediction"])
        residual = truth - prediction
        denominator = np.sum((truth - np.mean(truth)) ** 2)
        rows.append(
            {
                "scope": "pooled",
                "regime": regime,
                "target": target,
                "model": model,
                "frames": int(
                    sum(np.count_nonzero(_regime_mask(shot, regime)) for shot in shots)
                ),
                "values": int(truth.size),
                "explained_variance": float(
                    1.0 - np.sum(residual**2) / max(denominator, 1e-30)
                ),
                "mse": float(np.mean(residual**2)),
            }
        )
    return rows


def _write_table(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def _pooled_row(rows, regime, target, model):
    return next(
        row
        for row in rows
        if row["scope"] == "pooled"
        and row["regime"] == regime
        and row["target"] == target
        and row["model"] == model
    )


def _plot_representative(shot, tuned, default, path):
    channels = [4, 12, 20]
    colors = ("tab:blue", "tab:orange", "tab:green")
    fig, axes = plt.subplots(2, 1, figsize=(10.0, 7.0), sharex=True)
    for axis, target, label_text in (
        (axes[0], "ff_prime", "FF-prime change"),
        (axes[1], "psi", "total psi change [Wb]"),
    ):
        truth = shot.truth_ff_prime if target == "ff_prime" else shot.truth_psi
        for channel, color in zip(channels, colors, strict=True):
            rho = shot.rho_samples[channel]
            axis.plot(
                shot.times_s,
                truth[:, channel] - truth[0, channel],
                color=color,
                linewidth=2.0,
                label=f"extracted rho={rho:.2f}",
            )
            axis.plot(
                shot.times_s,
                tuned[target][:, channel] - tuned[target][0, channel],
                color=color,
                linestyle="--",
                linewidth=1.4,
                label=f"tuned rho={rho:.2f}",
            )
            axis.plot(
                shot.times_s,
                default[target][:, channel] - default[target][0, channel],
                color=color,
                linestyle=":",
                linewidth=1.1,
                label=f"default rho={rho:.2f}",
            )
        axis.axhline(0.0, color="0.5", linewidth=0.8, label="persistence")
        axis.set_ylabel(label_text)
    axes[0].legend(ncol=3, frameon=False, fontsize=8)
    axes[1].set_xlabel("time from first label [s]")
    fig.suptitle(f"Native diffusion forecast: {shot.name}")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bank_dir", type=Path)
    parser.add_argument("selection_json", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    selection = json.loads(args.selection_json.read_text())["selection"]
    shots = []
    for item in selection:
        bank = args.bank_dir / f"{item['shot']}_flux_trajectory.npz"
        shot = load_shot(bank, Path(item["path"]))
        shots.append(shot)
        print(f"geometry {shot.name}: frames={len(shot.times_s)}", flush=True)
    padded_length = max(len(shot.times_s) for shot in shots)
    fitted_eta, fit_receipt = fit_resistivity(shots, padded_length)
    print(
        "shared resistivity "
        f"eta0={fitted_eta.eta0:.6e} contrast={fitted_eta.contrast:.6f} "
        f"shape={fitted_eta.shape:.6f}",
        flush=True,
    )
    tuned = _all_predictions(shots, fitted_eta, padded_length)
    default = _all_predictions(shots, EtaProfile(), padded_length)
    rows = build_table(shots, tuned, default)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table_path = args.output_dir / "transport_explained_variance.csv"
    _write_table(rows, table_path)

    tuned_ff = _pooled_row(rows, "all", "ff_prime", "tuned_transport")
    persistence_ff = _pooled_row(rows, "all", "ff_prime", "persistence")
    margin = tuned_ff["explained_variance"] - persistence_ff["explained_variance"]
    outcome = "BEATS" if margin > 0.0 else "DOES NOT BEAT"
    verdict = (
        "PRE-REGISTERED VERDICT: tuned native transport "
        f"{outcome} persistence on pooled FF-prime evolution; "
        f"explained variance {tuned_ff['explained_variance']:.6f} versus "
        f"{persistence_ff['explained_variance']:.6f}, margin {margin:+.6f}."
    )
    print(verdict, flush=True)
    (args.output_dir / "transport_verdict.txt").write_text(verdict + "\n")
    pooled_rows = [row for row in rows if row["scope"] == "pooled"]
    summary = {
        "verdict": verdict,
        "fitted_eta": {
            "eta0_ohm_m": fitted_eta.eta0,
            "contrast": fitted_eta.contrast,
            "shape": fitted_eta.shape,
        },
        "default_eta": {
            "eta0_ohm_m": EtaProfile().eta0,
            "contrast": EtaProfile().contrast,
            "shape": EtaProfile().shape,
        },
        "fit_receipt": fit_receipt,
        "shot_count": len(shots),
        "frame_count": sum(len(shot.times_s) for shot in shots),
        "geometry_mode": "initial extracted FSA record frozen over each shot",
        "score_definition": "explained variance of change from extracted initial state",
        "rho_coordinate": "geometric proxy inherited from extraction bank",
        "pooled_scores": pooled_rows,
    }
    (args.output_dir / "transport_simulation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    representative = max(
        shots,
        key=lambda shot: np.count_nonzero(_regime_mask(shot, "flat_top")),
    )
    _plot_representative(
        representative,
        tuned[representative.name],
        default[representative.name],
        args.output_dir / "transport_trajectory_comparison.png",
    )


if __name__ == "__main__":
    main()
