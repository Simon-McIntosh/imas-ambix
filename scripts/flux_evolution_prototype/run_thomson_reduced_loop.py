#!/usr/bin/env python3
"""Close a frozen-geometry transport and Thomson correction loop on one shot.

The forward map deliberately stops short of a free-boundary equilibrium solve.
It samples each labeled map at the Thomson locations, converts those locations
to the banked flux-surface coordinate, and drives a temperature proxy with the
candidate p-prime profile and measured density.  FF-prime is corrected only
through cross-profile covariance in a joint basis learned on the other shots.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
import simulate_current_diffusion as diffusion
from nova.transport.current_diffusion import EtaProfile
from scipy import stats
from scipy.interpolate import RegularGridInterpolator

from imas_ambix.statespace.sequential_da import kalman_update, leading_observable_modes

TWO_PI = 2.0 * np.pi
CORE_CHANNELS = 44
TANGENTIAL_CHANNELS = 10
CANDIDATE_MODES = 10
CORRECTION_RANK = 6
ANALYSED_SLICES = 36


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as loaded:
        return {key: loaded[key] for key in loaded.files}


def _training_basis(
    bank_dir: Path, held_out: str
) -> tuple[np.ndarray, float, float, dict]:
    samples = []
    shot_count = 0
    for path in sorted(bank_dir.glob("*_flux_trajectory.npz")):
        if path.name.startswith(held_out):
            continue
        bank = _load_npz(path)
        p_change = bank["p_prime_on_rho"] - bank["p_prime_on_rho"][0]
        f_change = bank["ff_prime_on_rho"] - bank["ff_prime_on_rho"][0]
        samples.append((p_change, f_change))
        shot_count += 1
    p_scale = float(np.sqrt(np.nanmean(np.concatenate([x[0] for x in samples]) ** 2)))
    f_scale = float(np.sqrt(np.nanmean(np.concatenate([x[1] for x in samples]) ** 2)))
    matrix = np.concatenate(
        [np.concatenate((p / p_scale, f / f_scale), axis=1) for p, f in samples]
    )
    nonfinite = ~np.isfinite(matrix)
    column_median = np.nanmedian(np.where(nonfinite, np.nan, matrix), axis=0)
    matrix = np.where(nonfinite, column_median[np.newaxis, :], matrix)
    _u, singular, vt = np.linalg.svd(matrix, full_matrices=False)
    rank = min(CANDIDATE_MODES, vt.shape[0])
    normalised = vt[:rank].T
    n_rho = normalised.shape[0] // 2
    physical = normalised.copy()
    physical[:n_rho] *= p_scale
    physical[n_rho:] *= f_scale
    receipt = {
        "training_shots": shot_count,
        "candidate_modes": rank,
        "variance_fraction": float(np.sum(singular[:rank] ** 2) / np.sum(singular**2)),
        "p_prime_scale": p_scale,
        "ff_prime_scale": f_scale,
        "nonfinite_training_cells_filled": int(np.count_nonzero(nonfinite)),
    }
    return physical, p_scale, f_scale, receipt


def _nearest_row(times: np.ndarray, values: np.ndarray, time_ms: float) -> np.ndarray:
    index = int(np.argmin(np.abs(times - time_ms)))
    return np.asarray(values[index], dtype=np.float64).copy()


def _fill_measurement(
    row: np.ndarray, fallback: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    invalid = ~np.isfinite(row) | (row <= 1.0)
    filled = row.copy()
    filled[invalid] = fallback[invalid]
    still_invalid = ~np.isfinite(filled) | (filled <= 1.0)
    filled[still_invalid] = np.nanmedian(fallback[~still_invalid])
    return filled, invalid | still_invalid


def _sample_flux_coordinate(
    row: dict,
    bank: dict[str, np.ndarray],
    frame: int,
    radius: np.ndarray,
    height: np.ndarray,
) -> np.ndarray:
    raw_map = np.asarray(row["efit_psirz"], dtype=np.float64)[frame]
    grid_r = np.asarray(row["efit_grid_R"], dtype=np.float64)
    grid_z = np.asarray(row["efit_grid_Z"], dtype=np.float64)
    axis = float(bank["axis_flux_wb"][frame]) / TWO_PI
    boundary = float(bank["boundary_flux_wb"][frame]) / TWO_PI
    sampler = RegularGridInterpolator(
        (grid_z, grid_r), raw_map, bounds_error=False, fill_value=np.nan
    )
    raw = sampler(np.column_stack((height, radius)))
    psi_n = (raw - axis) / (boundary - axis)
    return np.clip(np.nan_to_num(psi_n, nan=1.2), 0.0, 1.2)


def _rho_from_psi(
    bank: dict[str, np.ndarray], frame: int, psi_n: np.ndarray
) -> np.ndarray:
    return np.interp(
        np.clip(psi_n, 0.0, 1.0),
        bank["surface_psi_n"][frame],
        bank["surface_rho_hat"][frame],
        left=float(bank["surface_rho_hat"][frame, 0]),
        right=float(bank["surface_rho_hat"][frame, -1]),
    )


def _pressure_proxy(p_prime: np.ndarray, psi_on_rho: np.ndarray) -> np.ndarray:
    drive = np.maximum(np.asarray(p_prime, dtype=np.float64), 0.0)
    spacing = np.abs(np.diff(np.asarray(psi_on_rho, dtype=np.float64)))
    shell = 0.5 * (drive[:-1] + drive[1:]) * spacing
    pressure = np.zeros_like(drive)
    pressure[:-1] = np.cumsum(shell[::-1])[::-1]
    floor = max(float(np.nanmax(pressure)) * 1.0e-5, 1.0e-9)
    return np.maximum(pressure, floor)


def _soft_pedestal_foot(values: np.ndarray, psi_n: np.ndarray) -> float:
    order = np.argsort(psi_n)
    x = np.asarray(psi_n)[order]
    y = np.log(np.maximum(np.asarray(values)[order], 1.0e-12))
    keep = (x >= 0.65) & (x <= 1.08)
    if np.count_nonzero(keep) >= 5:
        x, y = x[keep], y[keep]
    gradient = np.abs(np.diff(y) / np.maximum(np.diff(x), 1.0e-4))
    midpoint = 0.5 * (x[1:] + x[:-1])
    if gradient.size == 0:
        return float(np.nanmedian(x))
    scaled = gradient - np.nanmax(gradient)
    weights = np.exp(np.clip(scaled, -40.0, 0.0))
    return float(np.sum(weights * midpoint) / np.sum(weights))


def _measurement_context(
    row: dict,
    bank: dict[str, np.ndarray],
    frame: int,
    core_fallback: np.ndarray,
    tangential_fallback: np.ndarray,
) -> dict:
    time_ms = float(bank["times_ms"][frame])
    core_te, core_bad = _fill_measurement(
        _nearest_row(
            np.asarray(row["thomson_core_times"]),
            np.asarray(row["thomson_core_Te"]),
            time_ms,
        ),
        core_fallback,
    )
    tangential_te, tangential_bad = _fill_measurement(
        _nearest_row(
            np.asarray(row["thomson_edge_times"]),
            np.asarray(row["thomson_edge_Te"]),
            time_ms,
        ),
        tangential_fallback,
    )
    core_ne, core_ne_bad = _fill_measurement(
        _nearest_row(
            np.asarray(row["thomson_core_times"]),
            np.asarray(row["thomson_core_ne"]),
            time_ms,
        ),
        np.nanmedian(np.asarray(row["thomson_core_ne"]), axis=0),
    )
    tangential_ne, tangential_ne_bad = _fill_measurement(
        _nearest_row(
            np.asarray(row["thomson_edge_times"]),
            np.asarray(row["thomson_edge_ne"]),
            time_ms,
        ),
        np.nanmedian(np.asarray(row["thomson_edge_ne"]), axis=0),
    )
    chord_r = np.asarray(row["thomson_chord_R"], dtype=np.float64)[:54]
    chord_z = np.asarray(row["thomson_chord_Z"], dtype=np.float64)[:54]
    psi_n = _sample_flux_coordinate(row, bank, frame, chord_r, chord_z)
    rho = _rho_from_psi(bank, frame, psi_n)
    valid_core = np.flatnonzero(core_te > 1.0)
    matched_core = np.array(
        [
            valid_core[np.argmin(np.abs(np.log(core_te[valid_core]) - np.log(value)))]
            for value in tangential_te
        ],
        dtype=int,
    )
    mismatch = np.abs(
        np.log(core_te[matched_core]) - np.log(np.maximum(tangential_te, 1.0))
    )
    sigma = 0.18 + 0.5 * mismatch
    widened = (
        tangential_bad
        | tangential_ne_bad
        | core_bad[matched_core]
        | core_ne_bad[matched_core]
    )
    sigma[widened] *= 4.0
    observed_foot = _soft_pedestal_foot(core_te, psi_n[:CORE_CHANNELS])
    return {
        "core_te": core_te,
        "tangential_te": tangential_te,
        "core_ne": core_ne,
        "tangential_ne": tangential_ne,
        "psi_n": psi_n,
        "rho": rho,
        "matched_core": matched_core,
        "target": np.concatenate((np.zeros(TANGENTIAL_CHANNELS), [observed_foot])),
        "sigma": np.concatenate((sigma, [0.08 if not np.any(core_bad) else 0.20])),
        "widened": np.concatenate((widened, [np.any(core_bad)])),
    }


def _observe(
    state: np.ndarray,
    context: dict,
    bank: dict[str, np.ndarray],
    frame: int,
) -> np.ndarray:
    n_rho = len(bank["rho_hat_samples"])
    p_prime = state[:n_rho]
    pressure = _pressure_proxy(p_prime, bank["psi_on_rho_wb"][frame])
    sampled_pressure = np.interp(context["rho"], bank["rho_hat_samples"], pressure)
    core_proxy = sampled_pressure[:CORE_CHANNELS] / np.maximum(
        context["core_ne"], 1.0e16
    )
    tangential_proxy = sampled_pressure[CORE_CHANNELS:] / np.maximum(
        context["tangential_ne"], 1.0e16
    )
    pair_prediction = np.log(
        np.maximum(core_proxy[context["matched_core"]], 1.0e-30)
    ) - np.log(np.maximum(tangential_proxy, 1.0e-30))
    predicted_foot = _soft_pedestal_foot(core_proxy, context["psi_n"][:CORE_CHANNELS])
    return np.concatenate((pair_prediction, [predicted_foot]))


def _jacobian(
    state: np.ndarray,
    basis: np.ndarray,
    context: dict,
    bank: dict[str, np.ndarray],
    frame: int,
) -> np.ndarray:
    step = 0.08
    columns = []
    for mode in basis.T:
        plus = _observe(state + step * mode, context, bank, frame)
        minus = _observe(state - step * mode, context, bank, frame)
        columns.append((plus - minus) / (2.0 * step))
    return np.column_stack(columns)


def _normalised_error(
    estimate: np.ndarray, truth: np.ndarray, p_scale: float, f_scale: float
) -> tuple[float, float, float]:
    n_rho = truth.size // 2
    p_rmse = float(
        np.sqrt(np.mean(((estimate[:n_rho] - truth[:n_rho]) / p_scale) ** 2))
    )
    f_rmse = float(
        np.sqrt(np.mean(((estimate[n_rho:] - truth[n_rho:]) / f_scale) ** 2))
    )
    return p_rmse, f_rmse, float(np.sqrt(0.5 * (p_rmse**2 + f_rmse**2)))


def _whiteness(series: np.ndarray, max_lag: int = 5) -> tuple[float, float]:
    values = np.asarray(series, dtype=np.float64)
    values = values - np.mean(values)
    denom = float(np.dot(values, values))
    if denom <= 1.0e-12:
        return 0.0, 1.0
    correlations = []
    for lag in range(1, min(max_lag, len(values) - 1) + 1):
        correlations.append(float(np.dot(values[lag:], values[:-lag]) / denom))
    n = len(values)
    q_stat = (
        n
        * (n + 2.0)
        * sum(
            value * value / (n - lag) for lag, value in enumerate(correlations, start=1)
        )
    )
    return correlations[0], float(stats.chi2.sf(q_stat, len(correlations)))


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _plot(rows: list[dict], channel_rows: list[dict], path: Path) -> None:
    time = np.array([row["time_ms"] for row in rows]) * 1.0e-3
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.2))
    axes[0, 0].plot(
        time, [row["ff_prior_nrmse"] for row in rows], label="transport prior"
    )
    axes[0, 0].plot(time, [row["ff_posterior_nrmse"] for row in rows], label="updated")
    axes[0, 0].set_ylabel("FF-prime NRMSE")
    axes[0, 0].legend(frameon=False)
    axes[0, 1].plot(time, [row["p_prior_nrmse"] for row in rows], label="prior")
    axes[0, 1].plot(time, [row["p_posterior_nrmse"] for row in rows], label="updated")
    axes[0, 1].set_ylabel("p-prime NRMSE")
    axes[0, 1].legend(frameon=False)
    axes[1, 0].plot(time, [row["innovation_prior_rms"] for row in rows], label="prior")
    axes[1, 0].plot(
        time, [row["innovation_posterior_rms"] for row in rows], label="updated"
    )
    axes[1, 0].set_ylabel("normalised innovation RMS")
    axes[1, 0].set_xlabel("shot time [s]")
    axes[1, 0].legend(frameon=False)
    names = [row["channel"] for row in channel_rows]
    x = np.arange(len(names))
    axes[1, 1].bar(
        x - 0.18, [row["prior_rms"] for row in channel_rows], 0.36, label="prior"
    )
    axes[1, 1].bar(
        x + 0.18, [row["posterior_rms"] for row in channel_rows], 0.36, label="updated"
    )
    axes[1, 1].set_xticks(x, names, rotation=55, ha="right", fontsize=7)
    axes[1, 1].set_ylabel("per-channel normalised RMS")
    axes[1, 1].legend(frameon=False)
    fig.suptitle("Frozen-geometry Thomson reduced correction")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bank_dir", type=Path)
    parser.add_argument("selection_json", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--shot", default="d3d_shot_030fc156d6")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selection = json.loads(args.selection_json.read_text())["selection"]
    selected = next(item for item in selection if item["shot"] == args.shot)
    bank_path = args.bank_dir / f"{args.shot}_flux_trajectory.npz"
    bank = _load_npz(bank_path)
    source_path = Path(selected["path"])
    source = pq.read_table(source_path).to_pylist()[0]
    shot = diffusion.load_shot(bank_path, source_path)
    fit = json.loads(
        (args.output_dir / "transport_simulation_summary.json").read_text()
    )
    eta = EtaProfile(
        **{
            "eta0": fit["fitted_eta"]["eta0_ohm_m"],
            "contrast": fit["fitted_eta"]["contrast"],
            "shape": fit["fitted_eta"]["shape"],
        }
    )
    transport_ff = diffusion._align_initial(
        diffusion.recover_ff_prime(shot, eta, len(shot.times_s)), shot.truth_ff_prime
    )
    transport_psi = diffusion._align_initial(
        diffusion.evolve_psi(shot, eta, len(shot.times_s)), shot.truth_psi
    )

    candidate_basis, p_scale, f_scale, basis_receipt = _training_basis(
        args.bank_dir, args.shot
    )
    flat = np.flatnonzero(shot.normalised_current >= 0.8)
    indices = np.unique(
        np.rint(np.linspace(flat[0], flat[-1], ANALYSED_SLICES)).astype(int)
    )
    core_values = np.asarray(source["thomson_core_Te"], dtype=np.float64)
    tangential_values = np.asarray(source["thomson_edge_Te"], dtype=np.float64)
    core_fallback = np.nanmedian(
        np.where(core_values > 1.0, core_values, np.nan), axis=0
    )
    tangential_fallback = np.nanmedian(
        np.where(tangential_values > 1.0, tangential_values, np.nan), axis=0
    )

    initial_frame = int(indices[0])
    initial_state = np.concatenate(
        (bank["p_prime_on_rho"][initial_frame], transport_ff[initial_frame])
    )
    initial_context = _measurement_context(
        source, bank, initial_frame, core_fallback, tangential_fallback
    )
    candidate_h = _jacobian(
        initial_state, candidate_basis, initial_context, bank, initial_frame
    )
    observable_modes, singular_values = leading_observable_modes(
        candidate_h, initial_context["sigma"], CORRECTION_RANK
    )
    correction_basis = candidate_basis @ observable_modes
    all_singular = np.linalg.svd(
        candidate_h / initial_context["sigma"][:, np.newaxis],
        compute_uv=False,
    )
    observable_energy = float(
        np.sum(all_singular[:CORRECTION_RANK] ** 2) / np.sum(all_singular**2)
    )

    correction = np.zeros(correction_basis.shape[1])
    covariance = 0.20**2 * np.eye(correction.size)
    persistence = np.concatenate(
        (bank["p_prime_on_rho"][initial_frame], bank["ff_prime_on_rho"][initial_frame])
    )
    slice_rows = []
    prior_innovations = []
    posterior_innovations = []
    widened_count = 0
    psi_scale = max(
        float(
            np.sqrt(
                np.mean((shot.truth_psi[indices] - shot.truth_psi[initial_frame]) ** 2)
            )
        ),
        1.0e-9,
    )
    for frame in indices:
        context = _measurement_context(
            source, bank, int(frame), core_fallback, tangential_fallback
        )
        prior_state = np.concatenate(
            (bank["p_prime_on_rho"][initial_frame], transport_ff[frame])
        )
        truth_state = np.concatenate(
            (bank["p_prime_on_rho"][frame], bank["ff_prime_on_rho"][frame])
        )
        correction_prior = 0.94 * correction
        covariance_prior = 1.02 * (
            0.94**2 * covariance + 0.03**2 * np.eye(correction.size)
        )
        base_prediction = _observe(prior_state, context, bank, int(frame))
        h_mat = _jacobian(
            prior_state + correction_basis @ correction_prior,
            correction_basis,
            context,
            bank,
            int(frame),
        )
        correction, covariance, prior_norm, posterior_norm = kalman_update(
            correction_prior,
            covariance_prior,
            h_mat,
            context["target"] - base_prediction,
            context["sigma"],
            innovation_clip_sigma=8.0,
        )
        posterior_state = prior_state + correction_basis @ correction
        predicted_prior = base_prediction + h_mat @ correction_prior
        predicted_posterior = _observe(posterior_state, context, bank, int(frame))
        norm_prior = (context["target"] - predicted_prior) / context["sigma"]
        norm_posterior = (context["target"] - predicted_posterior) / context["sigma"]
        prior_innovations.append(norm_prior)
        posterior_innovations.append(norm_posterior)
        widened_count += int(np.count_nonzero(context["widened"]))
        p0, f0, joint0 = _normalised_error(persistence, truth_state, p_scale, f_scale)
        pp, fp, jointp = _normalised_error(prior_state, truth_state, p_scale, f_scale)
        pa, fa, jointa = _normalised_error(
            posterior_state, truth_state, p_scale, f_scale
        )
        psi_error = float(
            np.sqrt(np.mean((transport_psi[frame] - shot.truth_psi[frame]) ** 2))
            / psi_scale
        )
        slice_rows.append(
            {
                "frame": int(frame),
                "time_ms": float(bank["times_ms"][frame]),
                "plasma_current_a_boundary": float(shot.current_a[frame]),
                "p_persistence_nrmse": p0,
                "p_prior_nrmse": pp,
                "p_posterior_nrmse": pa,
                "ff_persistence_nrmse": f0,
                "ff_prior_nrmse": fp,
                "ff_posterior_nrmse": fa,
                "joint_persistence_nrmse": joint0,
                "joint_prior_nrmse": jointp,
                "joint_posterior_nrmse": jointa,
                "psi_transport_nrmse": psi_error,
                "innovation_prior_rms": prior_norm,
                "innovation_posterior_rms": posterior_norm,
                "widened_channels": int(np.count_nonzero(context["widened"])),
            }
        )

    prior_innovations = np.asarray(prior_innovations)
    posterior_innovations = np.asarray(posterior_innovations)
    labels = [f"isoflux_pair_{index:02d}" for index in range(TANGENTIAL_CHANNELS)] + [
        "pedestal_foot"
    ]
    channel_rows = []
    for channel, label in enumerate(labels):
        lag_prior, white_prior = _whiteness(prior_innovations[:, channel])
        lag_post, white_post = _whiteness(posterior_innovations[:, channel])
        channel_rows.append(
            {
                "channel": label,
                "prior_mean": float(np.mean(prior_innovations[:, channel])),
                "prior_rms": float(
                    np.sqrt(np.mean(prior_innovations[:, channel] ** 2))
                ),
                "posterior_mean": float(np.mean(posterior_innovations[:, channel])),
                "posterior_rms": float(
                    np.sqrt(np.mean(posterior_innovations[:, channel] ** 2))
                ),
                "prior_lag1": lag_prior,
                "prior_whiteness_p": white_prior,
                "posterior_lag1": lag_post,
                "posterior_whiteness_p": white_post,
            }
        )

    def mean_metric(key: str) -> float:
        return float(np.mean([row[key] for row in slice_rows]))

    joint_prior = mean_metric("joint_prior_nrmse")
    joint_post = mean_metric("joint_posterior_nrmse")
    joint_persistence = mean_metric("joint_persistence_nrmse")
    ff_prior = mean_metric("ff_prior_nrmse")
    ff_post = mean_metric("ff_posterior_nrmse")
    ff_persistence = mean_metric("ff_persistence_nrmse")
    joint_update_gain = joint_prior - joint_post
    joint_total_gain = joint_persistence - joint_post
    ff_update_gain = ff_prior - ff_post
    ff_total_gain = ff_persistence - ff_post
    update_share = (
        joint_update_gain / joint_total_gain if joint_total_gain > 0 else np.nan
    )
    ff_update_share = ff_update_gain / ff_total_gain if ff_total_gain > 0 else np.nan
    posterior_white = np.array([row["posterior_whiteness_p"] for row in channel_rows])
    adequate = bool(
        observable_energy >= 0.95
        and joint_update_gain > 0.0
        and ff_update_gain > 0.0
        and np.mean(posterior_white > 0.05) >= 0.5
    )
    summary = {
        "shot": args.shot,
        "analysed_slices": len(indices),
        "window_ms": [
            float(bank["times_ms"][indices[0]]),
            float(bank["times_ms"][indices[-1]]),
        ],
        "forward_substitution": (
            "No free-boundary solve was run. The observation forward samples "
            "the labeled psi map at Thomson coordinates, maps through the banked "
            "FSA/rho-hat geometry, "
            "and drives a pressure-over-measured-density temperature proxy."
        ),
        "ff_observation_path": (
            "Thomson has no direct FF-prime sensitivity in this surrogate; "
            "FF-prime is updated only through p-prime/FF-prime covariance in "
            "the joint cross-shot basis."
        ),
        "plasma_current_role": (
            "transport boundary condition only; absent from observation vector"
        ),
        "resistivity": fit["fitted_eta"],
        "basis": {
            **basis_receipt,
            "correction_rank": correction_basis.shape[1],
            "initial_whitened_singular_values": singular_values.tolist(),
            "retained_observable_energy": observable_energy,
            "adequate_for_thomson_only": adequate,
        },
        "channels": {
            "per_slice": len(labels),
            "isoflux_pairs": TANGENTIAL_CHANNELS,
            "pedestal_proxies": 1,
            "dropped": 0,
            "widened_instances": widened_count,
        },
        "tracking": {
            "joint_persistence_nrmse": joint_persistence,
            "joint_transport_prior_nrmse": joint_prior,
            "joint_posterior_nrmse": joint_post,
            "joint_update_error_reduction_fraction": joint_update_gain / joint_prior,
            "joint_update_share_of_total_gain": update_share,
            "ff_persistence_nrmse": ff_persistence,
            "ff_transport_prior_nrmse": ff_prior,
            "ff_posterior_nrmse": ff_post,
            "ff_update_error_reduction_fraction": ff_update_gain / ff_prior,
            "ff_update_share_of_total_gain": ff_update_share,
        },
        "innovation": {
            "prior_normalised_rms": float(np.sqrt(np.mean(prior_innovations**2))),
            "posterior_normalised_rms": float(
                np.sqrt(np.mean(posterior_innovations**2))
            ),
            "posterior_median_abs_lag1": float(
                np.median(np.abs([row["posterior_lag1"] for row in channel_rows]))
            ),
            "posterior_median_whiteness_p": float(np.median(posterior_white)),
            "posterior_white_channel_fraction_p_gt_0_05": float(
                np.mean(posterior_white > 0.05)
            ),
        },
    }
    joint_update_percent = 100.0 * joint_update_gain / joint_prior
    ff_update_percent = 100.0 * ff_update_gain / ff_prior
    summary["verdict"] = (
        "The reduced update changes joint tracking error by "
        f"{joint_update_percent:+.2f}% "
        f"relative to the transport prior and accounts for "
        f"{100.0 * update_share:.2f}% of the "
        "total gain over persistence. "
        f"For FF-prime alone it changes error by {ff_update_percent:+.2f}% "
        f"and accounts for {100.0 * ff_update_share:.2f}% of total FF-prime gain."
    )

    _write_csv(args.output_dir / "thomson_loop_slice_metrics.csv", slice_rows)
    _write_csv(args.output_dir / "thomson_loop_channel_innovations.csv", channel_rows)
    (args.output_dir / "thomson_loop_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    svd_assessment = (
        "adequate for this surrogate"
        if adequate
        else "not adequate as a Thomson-only production basis"
    )
    lessons = f"""# Frozen-geometry Thomson loop lessons

This demo substitutes a labeled-map/FSA forward for a free-boundary solve. It
therefore measures reduced flux-function tracking under fixed labeled geometry;
it does not demonstrate boundary, q95, or beta-N recovery.

- Keep: one cross-shot transport closure, plasma current solely as a transport
  boundary condition, a joint p-prime/FF-prime evolution basis trained away
  from the scored shot, row-whitened observable SVD, and explicit per-channel
  innovation receipts.
- Change: the production smoother needs a free-boundary solve, uncertainty in
  density and electron-to-total-pressure conversion, time-varying FSA geometry,
  and additional observations that directly constrain FF-prime. The present
  FF-prime correction is covariance-mediated because the Thomson surrogate has
  no direct FF-prime sensitivity.
- Drop: claims that Thomson-only frozen geometry identifies LCFS, q95, beta-N,
  or hidden FF-prime directions. No Thomson channel was dropped; uncertain
  instances were retained with widened sigma.
- SVD assessment: {svd_assessment}.
  Rank {correction_basis.shape[1]} retains {100.0 * observable_energy:.2f}% of
  the initial whitened observable energy;
  {100.0 * np.mean(posterior_white > 0.05):.1f}% of posterior
  channels pass the 5% whiteness test, and the update changes joint NRMSE by
  {100.0 * joint_update_gain / joint_prior:+.2f}% relative to its prior.

{summary["verdict"]}
"""
    (args.output_dir / "thomson_loop_lessons.md").write_text(lessons)
    _plot(
        slice_rows,
        channel_rows,
        args.output_dir / "thomson_loop_tracking_and_innovations.png",
    )
    print(summary["verdict"], flush=True)
    print(
        f"innovation rms {summary['innovation']['prior_normalised_rms']:.4f} -> "
        f"{summary['innovation']['posterior_normalised_rms']:.4f}; "
        "white channels="
        f"{summary['innovation']['posterior_white_channel_fraction_p_gt_0_05']:.3f}",
        flush=True,
    )
    print(f"summary={args.output_dir / 'thomson_loop_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
