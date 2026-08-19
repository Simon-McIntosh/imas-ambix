"""Frozen diagnostic readouts for the two trained camera representations.

The comparison attaches the same ridge probe to the final normalised trunk
representation of each frozen camera model.  Spatial cells are mean-pooled per
frame; only the latter half of each causal window is read out so that the
dynamics representation has an actual temporal context.  Targets are aligned
to native camera times without resampling the camera stream.

The independent evaluation unit is a shot.  Point errors are calculated per
shot before the paired bootstrap, preventing adjacent camera frames from
inflating the apparent sample size.  Negative dynamics-minus-baseline
differences favour the dynamics representation for both RMSE and CRPS.
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
from imas_ambix.camdyn.loader import (
    _hold_traces_to_frames,
    _read_shot_cond_traces,
)
from imas_ambix.camdyn.masking import named_geometry_mask
from imas_ambix.camdyn.metrics import ProbeProtocol, crps_gaussian, probe_rmse
from imas_ambix.camdyn.model import CamdynConfig, CamdynModel
from imas_ambix.camdyn.splits import DEFAULT_SPLIT_OUT, CamdynSplit
from imas_ambix.camdyn.train import _specs_for_shots

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DiagnosticTarget:
    """One scalar diagnostic readout and its level-1 binding."""

    key: str
    source: str
    array: str
    unit: str
    description: str
    was_conditioning_input: bool = False

    @property
    def source_path(self) -> str:
        return f"{self.source}/{self.array}"


DIAGNOSTIC_TARGETS: tuple[DiagnosticTarget, ...] = (
    DiagnosticTarget(
        "dalpha",
        "ada",
        "dalpha_integrated",
        "stored physical units",
        "integrated D-alpha emission",
    ),
    DiagnosticTarget(
        "ne_line_integrated",
        "ane",
        "density",
        "m^-2",
        "line-integrated electron density",
        was_conditioning_input=True,
    ),
    DiagnosticTarget(
        "te_core",
        "ayc",
        "te_core",
        "eV",
        "core Thomson electron temperature",
    ),
    DiagnosticTarget(
        "n2_mode_amp",
        "ama",
        "n=2_amplitude",
        "T",
        "toroidal n=2 mode amplitude",
    ),
)

SUBSAMPLE_SEED = 20260819
WINDOW_QUANTILES: tuple[float, ...] = (0.25, 0.50, 0.75)
WINDOW_FRAMES = 16
READOUT_FIRST_FRAME = WINDOW_FRAMES // 2
MASK_GEOMETRY = "fixed_section2"
DEFAULT_TRAIN_SHOTS = 48
DEFAULT_HELDOUT_SHOTS = 48
CANDIDATE_POOL_MULTIPLIER = 8
BOOTSTRAP_REPLICATES = 10_000
RIDGE_LAMBDA = 1.0


def deterministic_window_starts(
    camera_time: np.ndarray,
    target_times: list[np.ndarray],
    *,
    n_frames: int = WINDOW_FRAMES,
    quantiles: tuple[float, ...] = WINDOW_QUANTILES,
) -> list[int]:
    """Choose fixed windows inside the common diagnostic time support.

    Only time axes participate in selection; target values are never inspected.
    This prevents cherry-picking bright or otherwise easy plasma intervals.
    """
    ct = np.asarray(camera_time, dtype=np.float64).reshape(-1)
    if ct.size < n_frames or any(np.asarray(t).size < 2 for t in target_times):
        return []
    lo = max(float(ct[0]), *(float(np.asarray(t)[0]) for t in target_times))
    hi = min(float(ct[-1]), *(float(np.asarray(t)[-1]) for t in target_times))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return []

    starts: list[int] = []
    last = ct.size - n_frames
    for quantile in quantiles:
        centre_time = lo + float(quantile) * (hi - lo)
        centre_index = int(np.searchsorted(ct, centre_time, side="left"))
        start = int(np.clip(centre_index - n_frames // 2, 0, last))
        if start not in starts:
            starts.append(start)
    return starts


def align_nearest_native(
    signal_time: np.ndarray,
    signal_value: np.ndarray,
    frame_time: np.ndarray,
    *,
    tolerance_factor: float = 0.75,
) -> tuple[np.ndarray, float]:
    """Nearest-time alignment with a gap bound derived from native cadence."""
    st = np.asarray(signal_time, dtype=np.float64).reshape(-1)
    sv = np.asarray(signal_value, dtype=np.float64).reshape(-1)
    ft = np.asarray(frame_time, dtype=np.float64).reshape(-1)
    out = np.full(ft.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(st) & np.isfinite(sv)
    st, sv = st[valid], sv[valid]
    if st.size < 2:
        return out, 0.0
    order = np.argsort(st, kind="stable")
    st, sv = st[order], sv[order]
    native_steps = np.diff(st)
    native_steps = native_steps[np.isfinite(native_steps) & (native_steps > 0)]
    if native_steps.size == 0:
        return out, 0.0
    tolerance = float(tolerance_factor * np.median(native_steps))

    right = np.searchsorted(st, ft, side="left")
    left = np.clip(right - 1, 0, st.size - 1)
    right = np.clip(right, 0, st.size - 1)
    choose_right = np.abs(st[right] - ft) < np.abs(st[left] - ft)
    nearest = np.where(choose_right, right, left)
    gap = np.abs(st[nearest] - ft)
    accepted = np.isfinite(ft) & (gap <= tolerance)
    out[accepted] = sv[nearest[accepted]]
    return out, tolerance


def paired_bootstrap_difference(
    dynamics_minus_baseline: np.ndarray,
    *,
    seed: int = SUBSAMPLE_SEED,
    n_boot: int = BOOTSTRAP_REPLICATES,
) -> dict[str, float | int | bool]:
    """Paired percentile interval where a negative difference is better."""
    diff = np.asarray(dynamics_minus_baseline, dtype=np.float64).reshape(-1)
    diff = diff[np.isfinite(diff)]
    if diff.size == 0:
        return {
            "mean": float("nan"),
            "lo": float("nan"),
            "hi": float("nan"),
            "n_shots": 0,
            "dynamics_better": False,
            "clear_of_zero": False,
        }
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, diff.size, size=(n_boot, diff.size))
    means = diff[indices].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return {
        "mean": float(diff.mean()),
        "lo": float(lo),
        "hi": float(hi),
        "n_shots": int(diff.size),
        "dynamics_better": bool(hi < 0.0),
        "clear_of_zero": bool(hi < 0.0 or lo > 0.0),
    }


def _read_target_series(level1_store) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    series: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    groups = set(level1_store.group_keys())
    for target in DIAGNOSTIC_TARGETS:
        if target.source not in groups:
            continue
        group = level1_store[target.source]
        arrays = set(group.array_keys())
        if "time" not in arrays or target.array not in arrays:
            continue
        time = np.asarray(group["time"], dtype=np.float64).reshape(-1)
        value = np.asarray(group[target.array], dtype=np.float64)
        if value.ndim != 1 or value.size != time.size:
            continue
        finite_time = np.isfinite(time)
        if finite_time.sum() < 2:
            continue
        time, value = time[finite_time], value[finite_time]
        order = np.argsort(time, kind="stable")
        series[target.key] = (time[order], value[order])
    return series


def _load_arm(checkpoint, torch, device):
    payload = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    config = CamdynConfig.from_dict(payload["config"]["model"])
    model = CamdynModel.from_config(config)
    model.module.load_state_dict(payload["model_state"])
    model.module.to(device).eval()
    stats = (
        np.asarray(payload["cond_stats"][0], dtype=np.float32),
        np.asarray(payload["cond_stats"][1], dtype=np.float32),
    )
    return model, stats


def _frame_representation(
    model,
    arrays: dict[str, np.ndarray],
    cond_stats: tuple[np.ndarray, np.ndarray],
    torch,
    device,
) -> np.ndarray:
    """Return the spatially pooled final trunk state for every frame."""
    captured: list = []

    def capture(_module, _inputs, output):
        captured.append(output.detach())

    handle = model.module.out_norm.register_forward_hook(capture)
    mu, sd = cond_stats
    cond = (arrays["cond_values"] - mu) / sd
    try:
        with (
            torch.inference_mode(),
            torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ),
        ):
            model.module(
                torch.as_tensor(arrays["tokens"][None], device=device),
                torch.as_tensor(arrays["visible"][None], device=device),
                torch.as_tensor(cond[None], device=device),
                torch.as_tensor(arrays["cond_missing"][None], device=device),
                torch.as_tensor(arrays["dt"][None], device=device),
            )
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError(f"expected one trunk capture, received {len(captured)}")
    return captured[0].mean(dim=2).float().cpu().numpy()[0]


def _stop_requested() -> bool:
    stop_file = os.environ.get("AMBIX_STOP_FILE")
    return bool(stop_file and Path(stop_file).exists())


def _extract_split(
    specs,
    baseline,
    dynamics,
    baseline_stats,
    dynamics_stats,
    torch,
    device,
    *,
    max_shots: int,
    seed: int,
) -> dict[str, np.ndarray | list[int] | dict[str, list[float]]]:
    """Extract matched representations and aligned targets for one split."""
    import zarr  # noqa: PLC0415

    rng = np.random.default_rng(seed)
    ordered = [specs[int(i)] for i in rng.permutation(len(specs))]
    baseline_rows: list[np.ndarray] = []
    dynamics_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    shot_rows: list[np.ndarray] = []
    selected_shots: list[int] = []
    tolerances: dict[str, list[float]] = {
        target.key: [] for target in DIAGNOSTIC_TARGETS
    }

    visible = named_geometry_mask(MASK_GEOMETRY, WINDOW_FRAMES)
    for spec in ordered:
        if len(selected_shots) >= max_shots:
            break
        if _stop_requested():
            raise InterruptedError("stop file requested during probe extraction")
        if spec.level1_path is None:
            continue
        try:
            level1 = zarr.open_group(str(spec.level1_path), mode="r")
            if "rbb" not in set(level1.group_keys()):
                continue
            camera_group = level1["rbb"]
            if "time" not in set(camera_group.array_keys()):
                continue
            camera_time = np.asarray(camera_group["time"], dtype=np.float64)
            series = _read_target_series(level1)
            if set(series) != {target.key for target in DIAGNOSTIC_TARGETS}:
                continue
            starts = deterministic_window_starts(
                camera_time,
                [series[target.key][0] for target in DIAGNOSTIC_TARGETS],
            )
            if len(starts) != len(WINDOW_QUANTILES):
                continue
            token_store = zarr.open_group(str(spec.token_path), mode="r")
            token_array = token_store["tokens"]
            traces = _read_shot_cond_traces(spec.level1_path, CONDITIONING_CHANNELS)
        except Exception as exc:  # corpus robustness
            logger.warning("shot %s skipped during open: %s", spec.shot_id, exc)
            continue

        shot_baseline: list[np.ndarray] = []
        shot_dynamics: list[np.ndarray] = []
        shot_targets: list[np.ndarray] = []
        for start in starts:
            stop = start + WINDOW_FRAMES
            tokens = np.asarray(token_array[start:stop], dtype=np.int64)
            frame_time = camera_time[start:stop]
            if tokens.shape[0] != WINDOW_FRAMES or frame_time.size != WINDOW_FRAMES:
                continue
            cond_values, cond_missing = _hold_traces_to_frames(
                traces, frame_time, CONDITIONING_CHANNELS
            )
            aligned = []
            for target in DIAGNOSTIC_TARGETS:
                values, tolerance = align_nearest_native(
                    *series[target.key], frame_time
                )
                aligned.append(values)
                tolerances[target.key].append(tolerance)
            arrays = {
                "tokens": tokens,
                "visible": visible,
                "cond_values": cond_values,
                "cond_missing": cond_missing,
                "dt": _forward_dt(frame_time).astype(np.float32),
            }
            shot_baseline.append(
                _frame_representation(baseline, arrays, baseline_stats, torch, device)[
                    READOUT_FIRST_FRAME:
                ]
            )
            shot_dynamics.append(
                _frame_representation(dynamics, arrays, dynamics_stats, torch, device)[
                    READOUT_FIRST_FRAME:
                ]
            )
            shot_targets.append(np.stack(aligned, axis=1)[READOUT_FIRST_FRAME:])

        if len(shot_baseline) != len(WINDOW_QUANTILES):
            continue
        rows = len(shot_baseline) * (WINDOW_FRAMES - READOUT_FIRST_FRAME)
        baseline_rows.append(np.concatenate(shot_baseline))
        dynamics_rows.append(np.concatenate(shot_dynamics))
        target_rows.append(np.concatenate(shot_targets))
        shot_rows.append(np.full(rows, int(spec.shot_id), dtype=np.int64))
        selected_shots.append(int(spec.shot_id))
        logger.info(
            "probe extraction %d/%d: shot %s",
            len(selected_shots),
            max_shots,
            spec.shot_id,
        )

    if not baseline_rows:
        raise RuntimeError("no shots satisfied the frozen diagnostic subsample")
    return {
        "baseline": np.concatenate(baseline_rows),
        "dynamics": np.concatenate(dynamics_rows),
        "targets": np.concatenate(target_rows),
        "shot_ids": np.concatenate(shot_rows),
        "selected_shots": selected_shots,
        "alignment_tolerances_s": tolerances,
    }


def _per_shot_metrics(
    prediction: np.ndarray,
    truth: np.ndarray,
    shot_ids: np.ndarray,
    sigma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shots, rmse, crps = [], [], []
    for shot in np.unique(shot_ids):
        use = shot_ids == shot
        if not use.any():
            continue
        shots.append(int(shot))
        rmse.append(float(np.sqrt(np.mean((prediction[use] - truth[use]) ** 2))))
        spread = np.full((int(use.sum()), 1), sigma, dtype=np.float64)
        score = crps_gaussian(prediction[use, None], spread, truth[use, None])[0]
        crps.append(float(score))
    return np.asarray(shots), np.asarray(rmse), np.asarray(crps)


def score_frozen_target(
    train_baseline: np.ndarray,
    train_dynamics: np.ndarray,
    train_target: np.ndarray,
    held_baseline: np.ndarray,
    held_dynamics: np.ndarray,
    held_target: np.ndarray,
    held_shot_ids: np.ndarray,
    *,
    seed: int = SUBSAMPLE_SEED,
) -> dict:
    """Fit matched probes and return raw-unit held-out scores."""
    train_valid = np.isfinite(train_target)
    held_valid = np.isfinite(held_target)
    if train_valid.sum() < 8 or held_valid.sum() < 8:
        raise ValueError("insufficient finite target samples for frozen probe")

    arm_results: dict[str, dict[str, float]] = {}
    shot_results: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for name, train_features, held_features in (
        ("baseline", train_baseline, held_baseline),
        ("dynamics", train_dynamics, held_dynamics),
    ):
        probe = ProbeProtocol(
            probe_kind="linear", ridge_lambda=RIDGE_LAMBDA, standardize=True
        ).fit(train_features[train_valid], train_target[train_valid, None])
        train_prediction = probe.predict(train_features[train_valid])[:, 0]
        residual = train_target[train_valid] - train_prediction
        sigma = float(np.std(residual, ddof=1))
        scale_floor = np.finfo(np.float64).eps * max(
            1.0, float(np.nanstd(train_target[train_valid]))
        )
        sigma = max(sigma, scale_floor)
        prediction = probe.predict(held_features[held_valid])[:, 0]
        truth = held_target[held_valid]
        spread = np.full((truth.size, 1), sigma, dtype=np.float64)
        rmse = float(probe_rmse(prediction[:, None], truth[:, None])[0])
        crps = float(crps_gaussian(prediction[:, None], spread, truth[:, None])[0])
        arm_results[name] = {
            "rmse": rmse,
            "crps": crps,
            "train_residual_sigma": sigma,
        }
        shot_results[name] = _per_shot_metrics(
            prediction, truth, held_shot_ids[held_valid], sigma
        )

    baseline_shots, baseline_rmse, baseline_crps = shot_results["baseline"]
    dynamics_shots, dynamics_rmse, dynamics_crps = shot_results["dynamics"]
    if not np.array_equal(baseline_shots, dynamics_shots):
        raise RuntimeError("arm-specific held-out shot sets do not match")
    return {
        "n_train_samples": int(train_valid.sum()),
        "n_heldout_samples": int(held_valid.sum()),
        "n_heldout_shots": int(baseline_shots.size),
        "arms": arm_results,
        "paired_dynamics_minus_baseline": {
            "rmse": paired_bootstrap_difference(
                dynamics_rmse - baseline_rmse, seed=seed
            ),
            "crps": paired_bootstrap_difference(
                dynamics_crps - baseline_crps, seed=seed + 1
            ),
        },
    }


def evaluate_physics_probes(
    baseline_checkpoint,
    dynamics_checkpoint,
    *,
    split_path=None,
    device="cuda",
    train_shots: int = DEFAULT_TRAIN_SHOTS,
    heldout_shots: int = DEFAULT_HELDOUT_SHOTS,
) -> dict:
    """Run the complete frozen-backbone diagnostic comparison."""
    import torch  # noqa: PLC0415

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    baseline, baseline_stats = _load_arm(baseline_checkpoint, torch, dev)
    dynamics, dynamics_stats = _load_arm(dynamics_checkpoint, torch, dev)
    split = CamdynSplit.load(
        Path(split_path) if split_path is not None else DEFAULT_SPLIT_OUT
    )

    def candidate_specs(shot_ids, count, seed):
        rng = np.random.default_rng(seed)
        pool_size = min(len(shot_ids), CANDIDATE_POOL_MULTIPLIER * count)
        chosen = np.asarray(shot_ids)[rng.permutation(len(shot_ids))[:pool_size]]
        return _specs_for_shots(chosen.tolist(), max_shots=None)

    train_specs = candidate_specs(split.train, train_shots, SUBSAMPLE_SEED)
    held_specs = candidate_specs(split.held_out, heldout_shots, SUBSAMPLE_SEED + 1)

    train = _extract_split(
        train_specs,
        baseline,
        dynamics,
        baseline_stats,
        dynamics_stats,
        torch,
        dev,
        max_shots=train_shots,
        seed=SUBSAMPLE_SEED,
    )
    held = _extract_split(
        held_specs,
        baseline,
        dynamics,
        baseline_stats,
        dynamics_stats,
        torch,
        dev,
        max_shots=heldout_shots,
        seed=SUBSAMPLE_SEED + 1,
    )

    target_results = {}
    for index, target in enumerate(DIAGNOSTIC_TARGETS):
        result = score_frozen_target(
            train["baseline"],
            train["dynamics"],
            train["targets"][:, index],
            held["baseline"],
            held["dynamics"],
            held["targets"][:, index],
            held["shot_ids"],
            seed=SUBSAMPLE_SEED + 10 * index,
        )
        result.update(
            {
                "source_path": target.source_path,
                "description": target.description,
                "unit": target.unit,
                "was_conditioning_input": target.was_conditioning_input,
            }
        )
        target_results[target.key] = result

    better_rmse = sum(
        result["paired_dynamics_minus_baseline"]["rmse"]["dynamics_better"]
        for result in target_results.values()
    )
    better_crps = sum(
        result["paired_dynamics_minus_baseline"]["crps"]["dynamics_better"]
        for result in target_results.values()
    )
    return {
        "schema_version": 1,
        "comparison": (
            "frozen ridge probes: dynamics latent vs per-frame representation"
        ),
        "difference_orientation": (
            "dynamics minus baseline; negative favours dynamics for RMSE and CRPS"
        ),
        "checkpoints": {
            "baseline": str(baseline_checkpoint),
            "dynamics": str(dynamics_checkpoint),
        },
        "protocol": {
            "probe": "linear ridge; lambda=1; feature standardisation from train only",
            "representation": (
                "final normalised trunk state, spatial mean per frame, frozen backbone"
            ),
            "mask_geometry": MASK_GEOMETRY,
            "window_frames": WINDOW_FRAMES,
            "readout_frames": [READOUT_FIRST_FRAME, WINDOW_FRAMES - 1],
            "window_selection": (
                "25/50/75% quantiles of common native time-axis support; values unseen"
            ),
            "subsample_seed": SUBSAMPLE_SEED,
            "requested_train_shots": int(train_shots),
            "requested_heldout_shots": int(heldout_shots),
            "candidate_pool_multiplier": CANDIDATE_POOL_MULTIPLIER,
            "bootstrap_unit": "held-out shot",
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "alignment": (
                "nearest native diagnostic sample within 0.75 times its median cadence"
            ),
            "median_alignment_tolerance_s": {
                target.key: {
                    "train": float(
                        np.median(train["alignment_tolerances_s"][target.key])
                    ),
                    "heldout": float(
                        np.median(held["alignment_tolerances_s"][target.key])
                    ),
                }
                for target in DIAGNOSTIC_TARGETS
            },
            "train_shot_ids": train["selected_shots"],
            "heldout_shot_ids": held["selected_shots"],
        },
        "target_information_exposure": {
            "ne_line_integrated": (
                "ane/density conditioned both trained arms; this is an arm-to-arm "
                "representation readout, not an unseen-information result"
            ),
            "other_targets": "ada, ayc/te_core, and ama were not conditioning inputs",
        },
        "targets": target_results,
        "verdict": {
            "n_targets": len(target_results),
            "n_targets_rmse_significantly_better": int(better_rmse),
            "n_targets_crps_significantly_better": int(better_crps),
            "dynamics_better_all_targets_rmse": better_rmse == len(target_results),
            "dynamics_better_all_targets_crps": better_crps == len(target_results),
        },
    }


def write_comparison_figure(result: dict, path: str | Path) -> None:
    """Plot dimensionless improvement over the per-frame representation."""
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    keys = [target.key for target in DIAGNOSTIC_TARGETS]
    labels = ["D-alpha", "line ne", "Te core", "n=2 amp"]
    rmse_skill, crps_skill = [], []
    for key in keys:
        arms = result["targets"][key]["arms"]
        rmse_skill.append(
            100.0
            * (arms["baseline"]["rmse"] - arms["dynamics"]["rmse"])
            / arms["baseline"]["rmse"]
        )
        crps_skill.append(
            100.0
            * (arms["baseline"]["crps"] - arms["dynamics"]["crps"])
            / arms["baseline"]["crps"]
        )

    x = np.arange(len(keys))
    width = 0.34
    fig, ax = plt.subplots(figsize=(8.2, 4.5))
    fig.subplots_adjust(left=0.10, right=0.98, top=0.90, bottom=0.22)
    rmse_bars = ax.bar(x - width / 2, rmse_skill, width, label="RMSE", color="#2d6a9f")
    crps_bars = ax.bar(x + width / 2, crps_skill, width, label="CRPS", color="#d17a22")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Dynamics improvement over per-frame (%)")
    ax.set_title("Frozen diagnostic readout on held-out shots")
    ax.legend(frameon=False, ncols=2)
    ax.spines[["top", "right"]].set_visible(False)
    for bars in (rmse_bars, crps_bars):
        ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
    ax.text(
        0.01,
        -0.18,
        "Positive favours dynamics; line ne was supplied to both arms as conditioning.",
        transform=ax.transAxes,
        fontsize=8,
        color="#444444",
    )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Frozen camera-representation probes")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--dynamics", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--figure", required=True)
    parser.add_argument("--split-path", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--train-shots", type=int, default=DEFAULT_TRAIN_SHOTS)
    parser.add_argument("--heldout-shots", type=int, default=DEFAULT_HELDOUT_SHOTS)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    result = evaluate_physics_probes(
        args.baseline,
        args.dynamics,
        split_path=args.split_path,
        device=args.device,
        train_shots=args.train_shots,
        heldout_shots=args.heldout_shots,
    )
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_comparison_figure(result, args.figure)
    logger.info("physics-probe artifact written to %s", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
