#!/usr/bin/env python3
"""Add an analytic isotherm-centre-shift row to the banked Thomson loop."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
import run_thomson_reduced_loop as baseline
from nova.transport.current_diffusion import EtaProfile
from scipy.constants import mu_0

from imas_ambix.statespace.sequential_da import kalman_update, leading_observable_modes
from imas_ambix.thomson import IsothermAsymmetryOperator

PAIR_MISMATCH_LIMIT = 0.18
MINIMUM_SPAN_M = 0.30
CHANNEL_NAME = "isotherm_centre_shift"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fixed_reference_geometry(row: dict, frames: np.ndarray) -> tuple[float, float]:
    centres = []
    half_widths = []
    boundary_count = np.asarray(row["efit_lcfs_n"], dtype=int)
    boundary_radius = np.asarray(row["efit_lcfs_r"], dtype=float)
    for frame in frames:
        radius = boundary_radius[frame, : boundary_count[frame]]
        centres.append(0.5 * (np.nanmax(radius) + np.nanmin(radius)))
        half_widths.append(0.5 * (np.nanmax(radius) - np.nanmin(radius)))
    return float(np.median(centres)), float(np.median(half_widths))


def _asymmetry_context(
    row: dict,
    bank: dict[str, np.ndarray],
    frame: int,
    core_fallback: np.ndarray,
    tangential_fallback: np.ndarray,
    reference_major_radius_m: float,
    minor_radius_m: float,
) -> dict:
    context = baseline._measurement_context(
        row, bank, frame, core_fallback, tangential_fallback
    )
    radius = np.asarray(row["thomson_chord_R"], dtype=float)[
        baseline.CORE_CHANNELS : baseline.CORE_CHANNELS + baseline.TANGENTIAL_CHANNELS
    ]
    temperature = context["tangential_te"]
    candidates = []
    for inboard in np.flatnonzero(radius < reference_major_radius_m):
        for outboard in np.flatnonzero(radius > reference_major_radius_m):
            mismatch = abs(
                float(np.log(temperature[inboard]) - np.log(temperature[outboard]))
            )
            span = float(radius[outboard] - radius[inboard])
            candidates.append((mismatch, span, int(inboard), int(outboard)))
    matched = [item for item in candidates if item[0] <= PAIR_MISMATCH_LIMIT]
    if matched:
        mismatch, span, inboard, outboard = max(
            matched, key=lambda item: (item[1], -item[0])
        )
    else:
        mismatch, span, inboard, outboard = min(candidates)
    operator = IsothermAsymmetryOperator()
    measured = operator.measure(
        float(radius[inboard]),
        float(radius[outboard]),
        reference_major_radius_m=reference_major_radius_m,
        minor_radius_m=minor_radius_m,
    )
    radial_resolution = float(np.median(np.diff(radius)))
    propagated_radius_sigma = (
        np.sqrt(2.0)
        * 0.5
        * reference_major_radius_m
        * radial_resolution
        / minor_radius_m**2
    )
    coverage_multiplier = max(1.0, MINIMUM_SPAN_M / max(span, 1.0e-6))
    sigma = (propagated_radius_sigma + 0.35 * mismatch) * coverage_multiplier
    widened = mismatch > PAIR_MISMATCH_LIMIT or span < MINIMUM_SPAN_M
    context["target"] = np.concatenate((context["target"], [measured]))
    context["sigma"] = np.concatenate((context["sigma"], [sigma]))
    context["widened"] = np.concatenate((context["widened"], [widened]))
    context["asymmetry"] = {
        "inboard_radius_m": float(radius[inboard]),
        "outboard_radius_m": float(radius[outboard]),
        "log_temperature_mismatch": mismatch,
        "span_m": span,
        "measured_beta_p_plus_li_half": measured,
        "sigma": sigma,
        "widened": widened,
    }
    return context


def _state_moment(
    state: np.ndarray,
    bank: dict[str, np.ndarray],
    frame: int,
) -> tuple[float, float, float]:
    rho = np.asarray(bank["rho_hat_samples"], dtype=float)
    surface_rho = np.asarray(bank["surface_rho_hat"][frame], dtype=float)
    psi_n = np.asarray(bank["surface_psi_n"][frame], dtype=float)
    n_rho = len(rho)
    p_prime = np.interp(surface_rho, rho, state[:n_rho])
    ff_prime = np.interp(surface_rho, rho, state[n_rho:])
    volume_derivative = np.abs(np.asarray(bank["fsa_dv_dpsi_n"][frame], dtype=float))
    gradient = np.asarray(bank["fsa_gradient2_over_r2"][frame], dtype=float)
    inverse_radius_squared = np.asarray(bank["fsa_inverse_r2"][frame], dtype=float)
    flux_span = abs(
        float(bank["boundary_flux_wb"][frame] - bank["axis_flux_wb"][frame])
    )

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
            gradient * (flux_span / baseline.TWO_PI) ** 2 * volume_derivative,
            psi_n,
        )
    )
    beta_p = 2.0 * mu_0 * pressure_energy / max(poloidal_energy, 1.0e-30)

    current_drive = np.abs(p_prime + ff_prime * inverse_radius_squared / mu_0)
    shell_current = (
        0.5
        * (
            current_drive[1:] * volume_derivative[1:]
            + current_drive[:-1] * volume_derivative[:-1]
        )
        * np.diff(psi_n)
    )
    enclosed = np.concatenate(([0.0], np.cumsum(shell_current)))
    enclosed /= max(float(enclosed[-1]), 1.0e-30)
    radial_coordinate = np.concatenate(([0.0], surface_rho))
    enclosed_coordinate = np.concatenate(([0.0], enclosed))
    internal_inductance = float(
        2.0
        * np.trapezoid(
            enclosed_coordinate**2 / np.maximum(radial_coordinate, 1.0e-4),
            radial_coordinate,
        )
    )
    return beta_p, internal_inductance, beta_p + 0.5 * internal_inductance


def _observe(
    state: np.ndarray,
    context: dict,
    bank: dict[str, np.ndarray],
    frame: int,
) -> np.ndarray:
    thomson = baseline._observe(state, context, bank, frame)
    _beta_p, _internal_inductance, moment = _state_moment(state, bank, frame)
    return np.concatenate((thomson, [moment]))


def _moment_sensitivity(
    state: np.ndarray,
    bank: dict[str, np.ndarray],
    frame: int,
    p_scale: float,
    f_scale: float,
) -> dict[str, float]:
    n_rho = len(bank["rho_hat_samples"])
    gradient = np.zeros_like(state)
    for index in range(state.size):
        step = 1.0e-4 * (p_scale if index < n_rho else f_scale)
        perturbation = np.zeros_like(state)
        perturbation[index] = step
        plus = _state_moment(state + perturbation, bank, frame)[2]
        minus = _state_moment(state - perturbation, bank, frame)[2]
        gradient[index] = (plus - minus) / (2.0 * step)
    p_norm = float(np.linalg.norm(gradient[:n_rho]))
    f_norm = float(np.linalg.norm(gradient[n_rho:]))
    return {
        "p_prime_gradient_l2": p_norm,
        "ff_prime_gradient_l2": f_norm,
        "p_prime_scaled_sensitivity": p_norm * p_scale / np.sqrt(n_rho),
        "ff_prime_scaled_sensitivity": f_norm * f_scale / np.sqrt(n_rho),
    }


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


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _plot(
    slice_rows: list[dict],
    channel_rows: list[dict],
    baseline_summary: dict,
    path: Path,
) -> None:
    time = np.asarray([row["time_ms"] for row in slice_rows]) * 1.0e-3
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.2))
    axes[0, 0].plot(time, [row["ff_prior_nrmse"] for row in slice_rows], label="prior")
    axes[0, 0].plot(
        time, [row["ff_posterior_nrmse"] for row in slice_rows], label="with shift"
    )
    axes[0, 0].axhline(
        baseline_summary["tracking"]["ff_posterior_nrmse"],
        color="0.4",
        linestyle="--",
        label="banked posterior mean",
    )
    axes[0, 0].set_ylabel("FF-prime NRMSE")
    axes[0, 0].legend(frameon=False)
    axes[0, 1].plot(
        time,
        [row["asymmetry_target"] for row in slice_rows],
        label="measured shift",
    )
    axes[0, 1].plot(
        time,
        [row["asymmetry_prior"] for row in slice_rows],
        label="prior",
    )
    axes[0, 1].plot(
        time,
        [row["asymmetry_posterior"] for row in slice_rows],
        label="posterior",
    )
    axes[0, 1].set_ylabel("beta-p + li/2")
    axes[0, 1].legend(frameon=False)
    axes[1, 0].plot(
        time,
        [row["innovation_prior_rms"] for row in slice_rows],
        label="prior",
    )
    axes[1, 0].plot(
        time,
        [row["innovation_posterior_rms"] for row in slice_rows],
        label="posterior",
    )
    axes[1, 0].set_ylabel("normalised innovation RMS")
    axes[1, 0].set_xlabel("shot time [s]")
    axes[1, 0].legend(frameon=False)
    x = np.arange(len(channel_rows))
    axes[1, 1].bar(
        x - 0.18,
        [row["prior_rms"] for row in channel_rows],
        0.36,
        label="prior",
    )
    axes[1, 1].bar(
        x + 0.18,
        [row["posterior_rms"] for row in channel_rows],
        0.36,
        label="posterior",
    )
    axes[1, 1].set_xticks(
        x, [row["channel"] for row in channel_rows], rotation=55, ha="right", fontsize=7
    )
    axes[1, 1].set_ylabel("per-channel normalised RMS")
    axes[1, 1].legend(frameon=False)
    fig.suptitle("Isotherm-shift augmentation of the reduced Thomson loop")
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

    baseline_paths = [
        args.output_dir / "run_thomson_reduced_loop.py",
        args.output_dir / "thomson_loop_summary.json",
        args.output_dir / "thomson_loop_slice_metrics.csv",
        args.output_dir / "thomson_loop_channel_innovations.csv",
    ]
    baseline_hashes = {path.name: _sha256(path) for path in baseline_paths}
    baseline_summary = json.loads(
        (args.output_dir / "thomson_loop_summary.json").read_text()
    )
    selection = json.loads(args.selection_json.read_text())["selection"]
    selected = next(item for item in selection if item["shot"] == args.shot)
    bank_path = args.bank_dir / f"{args.shot}_flux_trajectory.npz"
    bank = baseline._load_npz(bank_path)
    source_path = Path(selected["path"])
    source = pq.read_table(source_path).to_pylist()[0]
    shot = baseline.diffusion.load_shot(bank_path, source_path)
    fit = json.loads(
        (args.output_dir / "transport_simulation_summary.json").read_text()
    )
    eta = EtaProfile(
        eta0=fit["fitted_eta"]["eta0_ohm_m"],
        contrast=fit["fitted_eta"]["contrast"],
        shape=fit["fitted_eta"]["shape"],
    )
    transport_ff = baseline.diffusion._align_initial(
        baseline.diffusion.recover_ff_prime(shot, eta, len(shot.times_s)),
        shot.truth_ff_prime,
    )

    candidate_basis, p_scale, f_scale, basis_receipt = baseline._training_basis(
        args.bank_dir, args.shot
    )
    flat = np.flatnonzero(shot.normalised_current >= 0.8)
    indices = np.unique(
        np.rint(np.linspace(flat[0], flat[-1], baseline.ANALYSED_SLICES)).astype(int)
    )
    core_values = np.asarray(source["thomson_core_Te"], dtype=float)
    tangential_values = np.asarray(source["thomson_edge_Te"], dtype=float)
    core_fallback = np.nanmedian(
        np.where(core_values > 1.0, core_values, np.nan), axis=0
    )
    tangential_fallback = np.nanmedian(
        np.where(tangential_values > 1.0, tangential_values, np.nan), axis=0
    )
    reference_major_radius, minor_radius = _fixed_reference_geometry(source, indices)

    initial_frame = int(indices[0])
    initial_state = np.concatenate(
        (bank["p_prime_on_rho"][initial_frame], transport_ff[initial_frame])
    )
    initial_context = _asymmetry_context(
        source,
        bank,
        initial_frame,
        core_fallback,
        tangential_fallback,
        reference_major_radius,
        minor_radius,
    )
    candidate_h = _jacobian(
        initial_state, candidate_basis, initial_context, bank, initial_frame
    )
    observable_modes, singular_values = leading_observable_modes(
        candidate_h, initial_context["sigma"], baseline.CORRECTION_RANK
    )
    correction_basis = candidate_basis @ observable_modes
    moment_sensitivity = _moment_sensitivity(
        initial_state,
        bank,
        initial_frame,
        p_scale,
        f_scale,
    )

    correction = np.zeros(correction_basis.shape[1])
    covariance = 0.20**2 * np.eye(correction.size)
    persistence = np.concatenate(
        (
            bank["p_prime_on_rho"][initial_frame],
            bank["ff_prime_on_rho"][initial_frame],
        )
    )
    slice_rows = []
    prior_innovations = []
    posterior_innovations = []
    widened_count = 0
    for frame in indices:
        context = _asymmetry_context(
            source,
            bank,
            int(frame),
            core_fallback,
            tangential_fallback,
            reference_major_radius,
            minor_radius,
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
        p0, f0, joint0 = baseline._normalised_error(
            persistence, truth_state, p_scale, f_scale
        )
        pp, fp, jointp = baseline._normalised_error(
            prior_state, truth_state, p_scale, f_scale
        )
        pa, fa, jointa = baseline._normalised_error(
            posterior_state, truth_state, p_scale, f_scale
        )
        prior_beta, prior_li, prior_moment = _state_moment(
            prior_state, bank, int(frame)
        )
        post_beta, post_li, post_moment = _state_moment(
            posterior_state, bank, int(frame)
        )
        asymmetry = context["asymmetry"]
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
                "innovation_prior_rms": prior_norm,
                "innovation_posterior_rms": posterior_norm,
                "asymmetry_target": asymmetry["measured_beta_p_plus_li_half"],
                "asymmetry_prior": prior_moment,
                "asymmetry_posterior": post_moment,
                "asymmetry_prior_beta_p": prior_beta,
                "asymmetry_prior_li": prior_li,
                "asymmetry_posterior_beta_p": post_beta,
                "asymmetry_posterior_li": post_li,
                "asymmetry_sigma": asymmetry["sigma"],
                "asymmetry_inboard_radius_m": asymmetry["inboard_radius_m"],
                "asymmetry_outboard_radius_m": asymmetry["outboard_radius_m"],
                "asymmetry_log_temperature_mismatch": asymmetry[
                    "log_temperature_mismatch"
                ],
                "widened_channels": int(np.count_nonzero(context["widened"])),
            }
        )

    prior_innovations = np.asarray(prior_innovations)
    posterior_innovations = np.asarray(posterior_innovations)
    labels = [
        f"isoflux_pair_{index:02d}" for index in range(baseline.TANGENTIAL_CHANNELS)
    ] + ["pedestal_foot", CHANNEL_NAME]
    channel_rows = []
    for channel, label in enumerate(labels):
        prior_lag, prior_white = baseline._whiteness(prior_innovations[:, channel])
        post_lag, post_white = baseline._whiteness(posterior_innovations[:, channel])
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
                "prior_lag1": prior_lag,
                "prior_whiteness_p": prior_white,
                "posterior_lag1": post_lag,
                "posterior_whiteness_p": post_white,
            }
        )

    def mean_metric(key: str) -> float:
        return float(np.mean([row[key] for row in slice_rows]))

    ff_persistence = mean_metric("ff_persistence_nrmse")
    ff_prior = mean_metric("ff_prior_nrmse")
    ff_posterior = mean_metric("ff_posterior_nrmse")
    joint_persistence = mean_metric("joint_persistence_nrmse")
    joint_prior = mean_metric("joint_prior_nrmse")
    joint_posterior = mean_metric("joint_posterior_nrmse")
    ff_update_share = (ff_prior - ff_posterior) / (ff_persistence - ff_posterior)
    baseline_tracking = baseline_summary["tracking"]
    baseline_share = baseline_tracking["ff_update_share_of_total_gain"]
    share_delta_points = 100.0 * (ff_update_share - baseline_share)
    posterior_delta = ff_posterior - baseline_tracking["ff_posterior_nrmse"]
    prior_delta = ff_posterior - ff_prior
    prior_worsening_percent = 100.0 * prior_delta / ff_prior
    traction = ff_posterior < ff_prior and posterior_delta < 0.0
    outcome = "GAINS" if traction else "DOES NOT GAIN"
    verdict = (
        f"VERDICT: the isotherm-shift observation {outcome} FF-prime traction; "
        f"posterior NRMSE changes by {posterior_delta:+.6f} against the banked "
        f"posterior, by {prior_delta:+.6f} ({prior_worsening_percent:+.3f}%) "
        f"against its prior, and update share changes by "
        f"{share_delta_points:+.3f} percentage points."
    )
    posterior_white = np.asarray([row["posterior_whiteness_p"] for row in channel_rows])
    summary = {
        "verdict": verdict,
        "shot": args.shot,
        "analysed_slices": len(indices),
        "window_ms": [
            float(bank["times_ms"][indices[0]]),
            float(bank["times_ms"][indices[-1]]),
        ],
        "invariants": {
            "training_shots": basis_receipt["training_shots"],
            "candidate_modes": basis_receipt["candidate_modes"],
            "correction_rank": correction_basis.shape[1],
            "resistivity": fit["fitted_eta"],
            "plasma_current_role": (
                "transport boundary condition only; absent from observation vector"
            ),
            "dropped_channels": 0,
            "widened_instances": widened_count,
        },
        "asymmetry_observation": {
            "operator": "imas_ambix.thomson.IsothermAsymmetryOperator",
            "reference_major_radius_m": reference_major_radius,
            "minor_radius_m": minor_radius,
            "mismatch_limit": PAIR_MISMATCH_LIMIT,
            "minimum_span_m": MINIMUM_SPAN_M,
            "mean_sigma": mean_metric("asymmetry_sigma"),
            "widened_slices": int(
                sum(
                    row["asymmetry_outboard_radius_m"]
                    - row["asymmetry_inboard_radius_m"]
                    < MINIMUM_SPAN_M
                    for row in slice_rows
                )
            ),
            "direct_state_path": (
                "p-prime sets beta-p; p-prime and FF-prime set the enclosed-current "
                "shape and large-aspect internal inductance"
            ),
            "initial_state_sensitivity": moment_sensitivity,
        },
        "tracking": {
            "ff_persistence_nrmse": ff_persistence,
            "ff_transport_prior_nrmse": ff_prior,
            "ff_posterior_nrmse": ff_posterior,
            "ff_update_share_of_total_gain": ff_update_share,
            "joint_persistence_nrmse": joint_persistence,
            "joint_transport_prior_nrmse": joint_prior,
            "joint_posterior_nrmse": joint_posterior,
        },
        "banked_baseline": {
            "ff_persistence_nrmse": baseline_tracking["ff_persistence_nrmse"],
            "ff_transport_prior_nrmse": baseline_tracking["ff_transport_prior_nrmse"],
            "ff_posterior_nrmse": baseline_tracking["ff_posterior_nrmse"],
            "ff_update_share_of_total_gain": baseline_share,
            "joint_posterior_nrmse": baseline_tracking["joint_posterior_nrmse"],
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
            "asymmetry_channel": next(
                row for row in channel_rows if row["channel"] == CHANNEL_NAME
            ),
        },
        "basis": {
            **basis_receipt,
            "correction_rank": correction_basis.shape[1],
            "initial_whitened_singular_values": singular_values.tolist(),
        },
        "interpretation": (
            "The added row has explicit nonzero FF-prime sensitivity, but its "
            "temporally correlated innovation and widened partial-span geometry "
            "degrade rather than improve FF-prime tracking on this shot."
        ),
        "baseline_hashes_before": baseline_hashes,
        "baseline_hashes_after": {path.name: _sha256(path) for path in baseline_paths},
    }

    _write_csv(args.output_dir / "shafranov_loop_slice_metrics.csv", slice_rows)
    _write_csv(args.output_dir / "shafranov_loop_channel_innovations.csv", channel_rows)
    (args.output_dir / "shafranov_loop_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    (args.output_dir / "shafranov_loop_verdict.txt").write_text(verdict + "\n")
    _plot(
        slice_rows,
        channel_rows,
        baseline_summary,
        args.output_dir / "shafranov_loop_comparison.png",
    )
    print(verdict, flush=True)
    print(
        f"FF-prime NRMSE persistence={ff_persistence:.6f} prior={ff_prior:.6f} "
        f"posterior={ff_posterior:.6f}; joint posterior={joint_posterior:.6f}",
        flush=True,
    )
    print(
        f"innovation rms {summary['innovation']['prior_normalised_rms']:.6f} -> "
        f"{summary['innovation']['posterior_normalised_rms']:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
