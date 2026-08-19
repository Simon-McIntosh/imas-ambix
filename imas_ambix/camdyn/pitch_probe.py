"""Frozen camera-representation probes for MSE sightline pitch.

Both trained camera arms are kept frozen.  A matched ridge readout is fit on
MSE calibration shots that also belong to the camera model's training split,
then converted back to the native per-shot sightline geometry and scored by
the canonical :mod:`imas_ambix.statespace.mse_eval` harness.  The locked MSE
held-out shots are never used to place radial nodes, standardise features, fit
the probe, or estimate its predictive spread.

Only deterministic camera windows selected from time-axis support are read.
The independent evaluation and bootstrap unit is a shot; adjacent camera or
MSE samples never count as independent observations.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from imas_ambix.camdyn.conditioning import CONDITIONING_CHANNELS
from imas_ambix.camdyn.dataset import _forward_dt
from imas_ambix.camdyn.loader import _hold_traces_to_frames, _read_shot_cond_traces
from imas_ambix.camdyn.masking import named_geometry_mask
from imas_ambix.camdyn.metrics import ProbeProtocol
from imas_ambix.camdyn.physics_probes import (
    _frame_representation,
    _load_arm,
    deterministic_window_starts,
    paired_bootstrap_difference,
)
from imas_ambix.camdyn.splits import DEFAULT_SPLIT_OUT, CamdynSplit
from imas_ambix.camdyn.train import _specs_for_shots
from imas_ambix.data.paths import LEVEL1_DIR, MANIFEST_DIR
from imas_ambix.statespace import mse_split
from imas_ambix.statespace.mse_eval import (
    MseTruth,
    ShotPrediction,
    load_manifest,
    score,
)

logger = logging.getLogger(__name__)

SUBSAMPLE_SEED = 20260819
WINDOW_FRAMES = 16
READOUT_FIRST_FRAME = WINDOW_FRAMES // 2
WINDOW_QUANTILES: tuple[float, ...] = (0.25, 0.50, 0.75)
MASK_GEOMETRY = "fixed_section2"
RADIAL_NODES = 7
RADIAL_QUANTILES = (0.10, 0.90)
RIDGE_LAMBDA = 1.0
DEFAULT_TRAIN_SHOTS = 48
CANDIDATE_POOL_MULTIPLIER = 8
BOOTSTRAP_REPLICATES = 10_000
DEFAULT_MSE_MANIFEST = MANIFEST_DIR / "mse_heldout_split_v0.json"


@dataclass(frozen=True)
class ShotLatents:
    """Matched arm representations at selected native MSE slice indices."""

    shot_id: int
    baseline: np.ndarray
    dynamics: np.ndarray
    slice_indices: np.ndarray


@dataclass
class RadialProbe:
    """Independent ridge readouts and train-residual spreads by radial node."""

    models: list[ProbeProtocol | None]
    sigma: np.ndarray

    def predict(self, features: np.ndarray) -> np.ndarray:
        output = np.full((features.shape[0], len(self.models)), np.nan)
        for node, model in enumerate(self.models):
            if model is not None:
                output[:, node] = model.predict(features)[:, 0]
        return output


def nearest_unique_indices(
    source_time: np.ndarray,
    query_time: np.ndarray,
    *,
    tolerance_factor: float = 0.75,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Map queries to unique nearest native samples within a cadence bound."""
    source = np.asarray(source_time, dtype=np.float64).reshape(-1)
    query = np.asarray(query_time, dtype=np.float64).reshape(-1)
    steps = np.diff(source)
    steps = steps[np.isfinite(steps) & (steps > 0)]
    if source.size < 2 or steps.size == 0:
        return np.empty(0, dtype=int), np.empty(0, dtype=int), 0.0
    tolerance = float(tolerance_factor * np.median(steps))
    right = np.clip(np.searchsorted(source, query, side="left"), 0, source.size - 1)
    left = np.clip(right - 1, 0, source.size - 1)
    choose_right = np.abs(source[right] - query) < np.abs(source[left] - query)
    nearest = np.where(choose_right, right, left)
    accepted = np.isfinite(query) & (np.abs(source[nearest] - query) <= tolerance)

    query_indices: list[int] = []
    source_indices: list[int] = []
    seen: set[int] = set()
    for query_index, source_index in enumerate(nearest):
        native_index = int(source_index)
        if accepted[query_index] and native_index not in seen:
            query_indices.append(query_index)
            source_indices.append(native_index)
            seen.add(native_index)
    return (
        np.asarray(query_indices, dtype=int),
        np.asarray(source_indices, dtype=int),
        tolerance,
    )


def fit_radial_nodes(manifest: dict, train_shot_ids: list[int]) -> np.ndarray:
    """Place the fixed radial grid using calibration/training geometry only."""
    radii: list[float] = []
    for shot_id in train_shot_ids:
        entry = manifest["shots"].get(str(shot_id))
        if entry is not None:
            radii.extend(float(value) for value in entry["active_channel_rpos"])
    finite = np.asarray(radii, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size < RADIAL_NODES:
        raise ValueError("insufficient calibration sightline geometry")
    lo, hi = np.quantile(finite, RADIAL_QUANTILES)
    return np.linspace(float(lo), float(hi), RADIAL_NODES)


def pitch_at_nodes(
    pitch: np.ndarray,
    pitch_error: np.ndarray,
    sightline_radii: np.ndarray,
    node_radii: np.ndarray,
) -> np.ndarray:
    """Interpolate physically gated sightline pitch onto fixed radial nodes."""
    values = np.asarray(pitch, dtype=np.float64)
    errors = np.asarray(pitch_error, dtype=np.float64)
    radii = np.asarray(sightline_radii, dtype=np.float64).reshape(-1)
    nodes = np.asarray(node_radii, dtype=np.float64).reshape(-1)
    if values.ndim == 1:
        values = values[None]
        errors = errors[None]
    gated = mse_split.pitch_point_gate(values, errors)
    output = np.full((values.shape[0], nodes.size), np.nan)
    for row in range(values.shape[0]):
        use = gated[row] & np.isfinite(radii)
        if use.sum() < 2:
            continue
        order = np.argsort(radii[use], kind="stable")
        rr = radii[use][order]
        pp = values[row, use][order]
        inside = (nodes >= rr[0]) & (nodes <= rr[-1])
        output[row, inside] = np.interp(nodes[inside], rr, pp)
    return output


def _stop_requested() -> bool:
    stop_file = os.environ.get("AMBIX_STOP_FILE")
    return bool(stop_file and Path(stop_file).exists())


def _extract_shot(
    spec,
    entry: dict,
    truth_shot,
    baseline,
    dynamics,
    baseline_stats,
    dynamics_stats,
    torch,
    device,
) -> ShotLatents | None:
    """Extract both arms at deterministic camera frames aligned to MSE slices."""
    import zarr  # noqa: PLC0415

    if spec.level1_path is None:
        return None
    level1 = zarr.open_group(str(spec.level1_path), mode="r")
    if "rbb" not in set(level1.group_keys()):
        return None
    camera_group = level1["rbb"]
    if "time" not in set(camera_group.array_keys()):
        return None
    camera_time = np.asarray(camera_group["time"], dtype=np.float64)
    mse_time = np.asarray(entry["beam_on_slice_times"], dtype=np.float64)
    if truth_shot is None or not np.array_equal(mse_time, truth_shot.time):
        return None
    starts = deterministic_window_starts(
        camera_time,
        [mse_time],
        n_frames=WINDOW_FRAMES,
        quantiles=WINDOW_QUANTILES,
    )
    if len(starts) != len(WINDOW_QUANTILES):
        return None

    token_store = zarr.open_group(str(spec.token_path), mode="r")
    token_array = token_store["tokens"]
    traces = _read_shot_cond_traces(spec.level1_path, CONDITIONING_CHANNELS)
    visible = named_geometry_mask(MASK_GEOMETRY, WINDOW_FRAMES)
    baseline_rows: list[np.ndarray] = []
    dynamics_rows: list[np.ndarray] = []
    slice_rows: list[np.ndarray] = []
    for start in starts:
        stop = start + WINDOW_FRAMES
        tokens = np.asarray(token_array[start:stop], dtype=np.int64)
        frame_time = camera_time[start:stop]
        if tokens.shape[0] != WINDOW_FRAMES or frame_time.size != WINDOW_FRAMES:
            return None
        cond_values, cond_missing = _hold_traces_to_frames(
            traces, frame_time, CONDITIONING_CHANNELS
        )
        arrays = {
            "tokens": tokens,
            "visible": visible,
            "cond_values": cond_values,
            "cond_missing": cond_missing,
            "dt": _forward_dt(frame_time).astype(np.float32),
        }
        baseline_rep = _frame_representation(
            baseline, arrays, baseline_stats, torch, device
        )[READOUT_FIRST_FRAME:]
        dynamics_rep = _frame_representation(
            dynamics, arrays, dynamics_stats, torch, device
        )[READOUT_FIRST_FRAME:]
        frame_subset = frame_time[READOUT_FIRST_FRAME:]
        query_indices, mse_indices, _ = nearest_unique_indices(mse_time, frame_subset)
        if mse_indices.size == 0:
            continue
        baseline_rows.append(baseline_rep[query_indices])
        dynamics_rows.append(dynamics_rep[query_indices])
        slice_rows.append(mse_indices)
    if not baseline_rows:
        return None

    baseline_all = np.concatenate(baseline_rows)
    dynamics_all = np.concatenate(dynamics_rows)
    slices_all = np.concatenate(slice_rows)
    _, unique_positions = np.unique(slices_all, return_index=True)
    keep = np.sort(unique_positions)
    return ShotLatents(
        shot_id=int(spec.shot_id),
        baseline=baseline_all[keep],
        dynamics=dynamics_all[keep],
        slice_indices=slices_all[keep],
    )


def _extract_cohort(
    specs,
    manifest,
    truth,
    baseline,
    dynamics,
    baseline_stats,
    dynamics_stats,
    torch,
    device,
    *,
    max_shots: int | None,
) -> tuple[list[ShotLatents], dict[int, str]]:
    rows: list[ShotLatents] = []
    failures: dict[int, str] = {}
    for spec in specs:
        if max_shots is not None and len(rows) >= max_shots:
            break
        if _stop_requested():
            raise InterruptedError("stop file requested during pitch extraction")
        entry = manifest["shots"].get(str(spec.shot_id))
        if entry is None:
            failures[int(spec.shot_id)] = "missing from MSE manifest"
            continue
        try:
            extracted = _extract_shot(
                spec,
                entry,
                truth.get(int(spec.shot_id)),
                baseline,
                dynamics,
                baseline_stats,
                dynamics_stats,
                torch,
                device,
            )
        except Exception as exc:  # corpus robustness
            failures[int(spec.shot_id)] = f"{type(exc).__name__}: {exc}"
            logger.warning("shot %s skipped: %s", spec.shot_id, exc)
            continue
        if extracted is None:
            failures[int(spec.shot_id)] = "no aligned deterministic camera windows"
            continue
        rows.append(extracted)
        logger.info(
            "pitch extraction %d%s: shot %s (%d slices)",
            len(rows),
            f"/{max_shots}" if max_shots is not None else "",
            spec.shot_id,
            extracted.slice_indices.size,
        )
    return rows, failures


def fit_radial_probe(
    features: np.ndarray,
    targets: np.ndarray,
) -> RadialProbe:
    """Fit one standardised ridge probe per node using finite labels only."""
    models: list[ProbeProtocol | None] = []
    sigma = np.full(targets.shape[1], np.nan)
    for node in range(targets.shape[1]):
        use = np.isfinite(targets[:, node])
        if use.sum() < 8:
            models.append(None)
            continue
        model = ProbeProtocol(
            probe_kind="linear",
            ridge_lambda=RIDGE_LAMBDA,
            standardize=True,
            targets=(f"radial_node_{node}",),
        ).fit(features[use], targets[use, node, None])
        residual = targets[use, node] - model.predict(features[use])[:, 0]
        spread = float(np.std(residual, ddof=1))
        floor = np.finfo(np.float64).eps * max(
            1.0, float(np.std(targets[use, node]))
        )
        sigma[node] = max(spread, floor)
        models.append(model)
    if sum(model is not None for model in models) < 2:
        raise RuntimeError("fewer than two radial nodes support a fitted probe")
    return RadialProbe(models=models, sigma=sigma)


def _interpolate_nodes(
    node_values: np.ndarray,
    node_radii: np.ndarray,
    sightline_radii: np.ndarray,
) -> np.ndarray:
    output = np.full((node_values.shape[0], sightline_radii.size), np.nan)
    for row in range(node_values.shape[0]):
        use = np.isfinite(node_values[row])
        if use.sum() < 2:
            continue
        lo = node_radii[use][0]
        hi = node_radii[use][-1]
        inside = (sightline_radii >= lo) & (sightline_radii <= hi)
        output[row, inside] = np.interp(
            sightline_radii[inside], node_radii[use], node_values[row, use]
        )
    return output


def build_shot_prediction(
    extracted: ShotLatents,
    arm_features: np.ndarray,
    probe: RadialProbe,
    node_radii: np.ndarray,
    entry: dict,
) -> ShotPrediction:
    """Map a radial-node readout into the harness's native sightline contract."""
    time = np.asarray(entry["beam_on_slice_times"], dtype=np.float64)
    sightline_radii = np.asarray(entry["active_channel_rpos"], dtype=np.float64)
    mean = np.full((time.size, sightline_radii.size), np.nan)
    spread = np.full_like(mean, np.nan)
    node_mean = probe.predict(arm_features)
    node_spread = np.broadcast_to(probe.sigma, node_mean.shape)
    mean[extracted.slice_indices] = _interpolate_nodes(
        node_mean, node_radii, sightline_radii
    )
    spread[extracted.slice_indices] = _interpolate_nodes(
        node_spread, node_radii, sightline_radii
    )
    return ShotPrediction(t=time, pitch_mean=mean, pitch_std=spread)


def _per_shot_pitch_metrics(
    predictions: dict[int, ShotPrediction], manifest: dict, truth: MseTruth
) -> dict[int, dict[str, float]]:
    output: dict[int, dict[str, float]] = {}
    for shot_id, prediction in predictions.items():
        entry = manifest["shots"][str(shot_id)]
        mini_manifest = {
            "version": manifest.get("version", "mse_heldout_split_v0"),
            "shots": {str(shot_id): entry},
        }
        result = score({shot_id: prediction}, mini_manifest, truth)
        pitch = result["primary"]["pitch"]
        if pitch["n_shots"]:
            output[shot_id] = {
                "rmse": float(pitch["rmse"]),
                "crps": float(pitch["crps"]),
                "n_points": int(pitch.get("n", 0)),
            }
    return output


def evaluate_pitch_probe(
    baseline_checkpoint,
    dynamics_checkpoint,
    *,
    mse_manifest_path=DEFAULT_MSE_MANIFEST,
    camdyn_split_path=DEFAULT_SPLIT_OUT,
    device="cuda",
    train_shots: int = DEFAULT_TRAIN_SHOTS,
) -> dict:
    """Fit matched frozen probes and score the locked held-out MSE cohort."""
    import torch  # noqa: PLC0415

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    baseline, baseline_stats = _load_arm(baseline_checkpoint, torch, dev)
    dynamics, dynamics_stats = _load_arm(dynamics_checkpoint, torch, dev)
    manifest = load_manifest(Path(mse_manifest_path))
    camdyn_split = CamdynSplit.load(Path(camdyn_split_path))
    camdyn_split.assert_invariants()
    truth = MseTruth(level1_dir=LEVEL1_DIR)

    calibration_ids = sorted(
        int(shot_id)
        for shot_id, entry in manifest["shots"].items()
        if entry["partition"] == "calibration"
        and int(shot_id) in set(camdyn_split.train)
    )
    rng = np.random.default_rng(SUBSAMPLE_SEED)
    pool_size = min(len(calibration_ids), CANDIDATE_POOL_MULTIPLIER * train_shots)
    train_candidates = np.asarray(calibration_ids)[
        rng.permutation(len(calibration_ids))[:pool_size]
    ].tolist()
    discovered_train_specs = _specs_for_shots(train_candidates, max_shots=None)
    train_specs = [
        discovered_train_specs[int(index)]
        for index in rng.permutation(len(discovered_train_specs))
    ]
    train_rows, train_failures = _extract_cohort(
        train_specs,
        manifest,
        truth,
        baseline,
        dynamics,
        baseline_stats,
        dynamics_stats,
        torch,
        dev,
        max_shots=train_shots,
    )
    if len(train_rows) < train_shots:
        raise RuntimeError(
            f"only {len(train_rows)}/{train_shots} calibration shots were usable"
        )
    train_ids = [row.shot_id for row in train_rows]
    node_radii = fit_radial_nodes(manifest, train_ids)
    train_baseline = np.concatenate([row.baseline for row in train_rows])
    train_dynamics = np.concatenate([row.dynamics for row in train_rows])
    train_targets = np.concatenate(
        [
            pitch_at_nodes(
                truth.get(row.shot_id).pitch[row.slice_indices],
                truth.get(row.shot_id).pitch_error[row.slice_indices],
                truth.get(row.shot_id).active_channel_rpos,
                node_radii,
            )
            for row in train_rows
        ]
    )
    baseline_probe = fit_radial_probe(train_baseline, train_targets)
    dynamics_probe = fit_radial_probe(train_dynamics, train_targets)

    locked_heldout = sorted(
        int(shot_id)
        for shot_id, entry in manifest["shots"].items()
        if entry["partition"] == "held_out"
    )
    tokenless = sorted(int(value) for value in camdyn_split.mse_heldout_without_tokens)
    expected_with_tokens = sorted(set(locked_heldout) - set(tokenless))
    if expected_with_tokens != sorted(camdyn_split.mse_heldout_forced):
        raise RuntimeError("camera and MSE held-out manifests disagree")
    held_specs = _specs_for_shots(expected_with_tokens, max_shots=None)
    held_rows, held_failures = _extract_cohort(
        held_specs,
        manifest,
        truth,
        baseline,
        dynamics,
        baseline_stats,
        dynamics_stats,
        torch,
        dev,
        max_shots=None,
    )

    predictions: dict[str, dict[int, ShotPrediction]] = {
        "baseline": {},
        "dynamics": {},
    }
    for row in held_rows:
        entry = manifest["shots"][str(row.shot_id)]
        predictions["baseline"][row.shot_id] = build_shot_prediction(
            row, row.baseline, baseline_probe, node_radii, entry
        )
        predictions["dynamics"][row.shot_id] = build_shot_prediction(
            row, row.dynamics, dynamics_probe, node_radii, entry
        )

    arm_scores = {
        name: score(arm_predictions, manifest, truth)["primary"]["pitch"]
        for name, arm_predictions in predictions.items()
    }
    per_shot = {
        name: _per_shot_pitch_metrics(arm_predictions, manifest, truth)
        for name, arm_predictions in predictions.items()
    }
    common_shots = sorted(set(per_shot["baseline"]) & set(per_shot["dynamics"]))
    baseline_rmse = np.asarray(
        [per_shot["baseline"][shot]["rmse"] for shot in common_shots]
    )
    dynamics_rmse = np.asarray(
        [per_shot["dynamics"][shot]["rmse"] for shot in common_shots]
    )
    baseline_crps = np.asarray(
        [per_shot["baseline"][shot]["crps"] for shot in common_shots]
    )
    dynamics_crps = np.asarray(
        [per_shot["dynamics"][shot]["crps"] for shot in common_shots]
    )
    paired = {
        "rmse": paired_bootstrap_difference(
            dynamics_rmse - baseline_rmse,
            seed=SUBSAMPLE_SEED,
            n_boot=BOOTSTRAP_REPLICATES,
        ),
        "crps": paired_bootstrap_difference(
            dynamics_crps - baseline_crps,
            seed=SUBSAMPLE_SEED + 1,
            n_boot=BOOTSTRAP_REPLICATES,
        ),
    }
    missing_after_tokens = sorted(set(expected_with_tokens) - set(common_shots))
    return {
        "schema_version": 1,
        "comparison": (
            "frozen ridge pitch readout: dynamics vs per-frame representation"
        ),
        "difference_orientation": (
            "dynamics minus baseline; negative favours dynamics for RMSE and CRPS"
        ),
        "checkpoints": {
            "baseline": str(baseline_checkpoint),
            "dynamics": str(dynamics_checkpoint),
        },
        "protocol": {
            "eval_harness": "imas_ambix.statespace.mse_eval.score",
            "mse_manifest": str(mse_manifest_path),
            "camera_split": str(camdyn_split_path),
            "probe": "independent linear ridge by radial node; lambda=1",
            "representation": (
                "final normalised trunk state, spatial mean per frame, frozen backbone"
            ),
            "train_partition": (
                "MSE calibration intersect camera-model train; "
                "shot-disjoint from held-out"
            ),
            "train_shot_ids": train_ids,
            "n_train_shots": len(train_ids),
            "n_train_samples": int(train_targets.shape[0]),
            "radial_node_radii_m": node_radii.tolist(),
            "mask_geometry": MASK_GEOMETRY,
            "window_frames": WINDOW_FRAMES,
            "readout_frames": [READOUT_FIRST_FRAME, WINDOW_FRAMES - 1],
            "window_selection": (
                "25/50/75% quantiles of common native camera/MSE time support; "
                "target values unseen"
            ),
            "bootstrap_unit": "held-out shot",
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "subsample_seed": SUBSAMPLE_SEED,
        },
        "heldout_cohort": {
            "locked_shots": len(locked_heldout),
            "locked_shot_ids": locked_heldout,
            "tokenless_exclusions": tokenless,
            "expected_with_tokens": len(expected_with_tokens),
            "usable_shots": len(common_shots),
            "usable_shot_ids": common_shots,
            "post_token_exclusions": missing_after_tokens,
            "post_token_exclusion_reasons": {
                str(shot): held_failures.get(shot, "no finite harness score")
                for shot in missing_after_tokens
            },
        },
        "arms": arm_scores,
        "paired_dynamics_minus_baseline": paired,
        "per_shot": {
            str(shot): {
                "baseline_rmse_rad": float(per_shot["baseline"][shot]["rmse"]),
                "dynamics_rmse_rad": float(per_shot["dynamics"][shot]["rmse"]),
                "baseline_crps_rad": float(per_shot["baseline"][shot]["crps"]),
                "dynamics_crps_rad": float(per_shot["dynamics"][shot]["crps"]),
            }
            for shot in common_shots
        },
        "verdict": {
            "dynamics_rmse_significantly_better": bool(
                paired["rmse"]["dynamics_better"]
            ),
            "dynamics_crps_significantly_better": bool(
                paired["crps"]["dynamics_better"]
            ),
            "clean_negative": bool(
                not paired["rmse"]["dynamics_better"]
                and not paired["crps"]["dynamics_better"]
            ),
        },
        "diagnostics": {
            "n_train_candidate_failures": len(train_failures),
            "n_heldout_extraction_failures": len(held_failures),
        },
    }


def write_comparison_figure(result: dict, path: str | Path) -> None:
    """Plot paired shot errors and the bootstrapped arm difference."""
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    rows = list(result["per_shot"].values())
    baseline = np.asarray([row["baseline_rmse_rad"] for row in rows])
    dynamics = np.asarray([row["dynamics_rmse_rad"] for row in rows])
    interval = result["paired_dynamics_minus_baseline"]["rmse"]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(9.2, 4.1),
        gridspec_kw={"width_ratios": [1.3, 1]},
    )
    fig.subplots_adjust(left=0.09, right=0.98, top=0.88, bottom=0.17, wspace=0.34)
    lo = min(float(baseline.min()), float(dynamics.min()))
    hi = max(float(baseline.max()), float(dynamics.max()))
    axes[0].scatter(baseline, dynamics, s=18, alpha=0.65, color="#2d6a9f", linewidths=0)
    axes[0].plot([lo, hi], [lo, hi], color="#555555", linewidth=1, linestyle="--")
    axes[0].set_xlabel("Per-frame latent RMSE (rad)")
    axes[0].set_ylabel("Dynamics latent RMSE (rad)")
    axes[0].set_title(f"Locked held-out shots (n={len(rows)})")

    differences = dynamics - baseline
    axes[1].axvline(0.0, color="#555555", linewidth=1)
    jitter = np.linspace(-0.12, 0.12, differences.size)
    axes[1].scatter(
        differences,
        jitter,
        s=16,
        alpha=0.45,
        color="#777777",
        linewidths=0,
    )
    axes[1].errorbar(
        interval["mean"],
        0.32,
        xerr=[[interval["mean"] - interval["lo"]], [interval["hi"] - interval["mean"]]],
        fmt="o",
        color="#b33c2e",
        capsize=4,
    )
    axes[1].set_yticks([0.32], ["mean + 95% CI"])
    axes[1].set_xlabel("Dynamics minus per-frame RMSE (rad)")
    axes[1].set_title("Paired shot bootstrap")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="both", color="#dddddd", linewidth=0.5, alpha=0.6)
    fig.suptitle("Frozen sightline-pitch readout")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Frozen camera-latent MSE pitch probe")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--dynamics", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--figure", required=True)
    parser.add_argument("--mse-manifest", default=str(DEFAULT_MSE_MANIFEST))
    parser.add_argument("--camdyn-split", default=str(DEFAULT_SPLIT_OUT))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--train-shots", type=int, default=DEFAULT_TRAIN_SHOTS)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    result = evaluate_pitch_probe(
        args.baseline,
        args.dynamics,
        mse_manifest_path=Path(args.mse_manifest),
        camdyn_split_path=Path(args.camdyn_split),
        device=args.device,
        train_shots=args.train_shots,
    )
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    write_comparison_figure(result, args.figure)
    logger.info("pitch-probe artifact written to %s", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
