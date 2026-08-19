"""Phase-sensitive filament-motion readout with adversarial controls.

The earlier temporal edge score compared inter-frame differences with SSIM.
That score is unsuitable for judging a temporal objective because independent
spectrum-matched noise can score as well as a model.  This module instead
combines local high-pass energy-centroid displacement skill with the signed,
amplitude-aware alignment of spatially high-passed inter-frame changes on
raw-camera-selected filament cells.

For each consecutive forecast-frame pair, the readout forms
``highpass(frame[j]) - highpass(frame[i])`` for the prediction and raw camera
truth.  It removes the spatial mean on the filament support, computes their
cosine alignment, and multiplies it by a symmetric motion-amplitude agreement.
The displacement term asks whether the predicted translation is closer to the
raw-camera translation than a frozen zero-motion reference.  The result is
phase and location sensitive: frozen persistence has no motion and scores
zero, exact truth scores one, and spectrum-matched coloured noise has the right
spatial power but random phase and scores near zero.

The evaluation is deliberately checkpoint-only.  It loads the existing
dynamics arm, scores MAP against raw camera frames at the exact decimated
timestamps, validates the controls, writes a JSON artifact, and renders the
dynamic-range figure.  It never trains or changes model state.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import tempfile
from pathlib import Path

import numpy as np

from imas_ambix.camdyn import recon_movie as mv
from imas_ambix.camdyn import reconstruction_demo as rd
from imas_ambix.camdyn import structure_fidelity as sf
from imas_ambix.camdyn.placement_trajectory import (
    _cell_mask_to_pixels,
    filament_cell_mask,
)

logger = logging.getLogger(__name__)

DEFAULT_JSON = Path("imas_ambix/camdyn/artifacts/motion_readout.json")
DEFAULT_FIGURE = Path("docs/figures/camera-dynamics-wm/fig-cdw-motion-readout.png")

# These thresholds are fixed before the checkpoint evaluation.  They define a
# useful measurement range, not a model-success target.
PERSISTENCE_ABS_MAX = 0.02
ORACLE_MIN = 0.95
COLOURED_NOISE_ABS_MAX = 0.10
MAP_SEPARATION_MIN = 0.03
MOTION_CAPTURED_MIN = 0.75
NOISE_SEEDS = 8
HIGH_PASS_WINDOW = 11
MAX_TRANSLATION_PIXELS = 8
MIN_TRUTH_TRANSLATION_PIXELS = 0.75


def _is_finite_number(value: object) -> bool:
    """Whether a scalar is numeric and finite without accepting ``None``."""
    return isinstance(value, (int, float, np.integer, np.floating)) and bool(
        np.isfinite(value)
    )


def _strict_json_data(value):
    """Encode arrays and non-finite measurements as strict-JSON values.

    A missing measurement is data, not a floating-point number.  JSON ``null``
    preserves that distinction while allowing ``allow_nan=False`` to remain a
    hard guard against accidental non-standard output.
    """
    if isinstance(value, dict):
        return {key: _strict_json_data(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict_json_data(item) for item in value]
    if isinstance(value, np.ndarray):
        return _strict_json_data(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _as_gray_stack(frames: np.ndarray) -> np.ndarray:
    """Return frames as a ``(time, 256, 256)`` float stack."""
    arr = np.asarray(frames)
    if arr.ndim not in (3, 4):
        raise ValueError("frames must have shape (time,height,width[,channels])")
    return np.stack([sf._to_gray256(frame) for frame in arr]).astype(np.float64)


def _high_pass_stack(frames: np.ndarray, *, window: int) -> np.ndarray:
    """Remove slowly varying brightness while retaining filament structure."""
    return np.stack([frame - sf._uniform_box(frame, window) for frame in frames])


def _transition_readout(
    pred_change: np.ndarray,
    truth_change: np.ndarray,
    pixel_mask: np.ndarray,
    *,
    truth_scale: float,
) -> dict[str, float]:
    """Score one predicted temporal change against the raw-camera change.

    A shared truth-derived scale preserves motion-amplitude information.  The
    cosine term alone would give a tiny correctly directed change a perfect
    score; the symmetric amplitude term prevents that degeneracy.
    """
    p = np.asarray(pred_change, dtype=np.float64)[pixel_mask] / truth_scale
    g = np.asarray(truth_change, dtype=np.float64)[pixel_mask] / truth_scale
    if not p.size:
        raise ValueError("filament pixel mask is empty")
    p = p - p.mean()
    g = g - g.mean()
    pred_norm = float(np.linalg.norm(p))
    truth_norm = float(np.linalg.norm(g))
    if truth_norm <= 1e-12:
        return {
            "phase_alignment": float("nan"),
            "cosine": float("nan"),
            "amplitude_agreement": float("nan"),
            "predicted_motion_norm": pred_norm,
            "truth_motion_norm": truth_norm,
        }
    if pred_norm <= 1e-12:
        return {
            "phase_alignment": 0.0,
            "cosine": 0.0,
            "amplitude_agreement": 0.0,
            "predicted_motion_norm": 0.0,
            "truth_motion_norm": truth_norm,
        }
    cosine = float(np.clip(np.dot(p, g) / (pred_norm * truth_norm), -1.0, 1.0))
    amplitude = float(2.0 * min(pred_norm, truth_norm) / (pred_norm + truth_norm))
    return {
        "phase_alignment": cosine * amplitude,
        "cosine": cosine,
        "amplitude_agreement": amplitude,
        "predicted_motion_norm": pred_norm,
        "truth_motion_norm": truth_norm,
    }


def _estimate_translation(
    previous: np.ndarray,
    current: np.ndarray,
    pixel_mask: np.ndarray,
    *,
    max_shift: int = MAX_TRANSLATION_PIXELS,
) -> dict[str, float]:
    """Estimate local displacement from the high-pass energy centroid."""
    mask = np.asarray(pixel_mask, dtype=bool)
    if mask.sum() < 64:
        return {"dy": 0.0, "dx": 0.0, "active_energy": 0.0}
    first = np.asarray(previous, dtype=np.float64)
    second = np.asarray(current, dtype=np.float64)

    def _centroid(frame: np.ndarray) -> tuple[float, float, float]:
        energy = np.abs(frame)
        threshold = float(np.percentile(energy[mask], 80.0))
        weights = np.where(mask, np.maximum(energy - threshold, 0.0), 0.0)
        total = float(weights.sum())
        if total <= 1e-12:
            return 0.0, 0.0, 0.0
        yy, xx = np.indices(frame.shape, dtype=np.float64)
        return (
            float((weights * yy).sum() / total),
            float((weights * xx).sum() / total),
            total,
        )

    first_y, first_x, first_energy = _centroid(first)
    second_y, second_x, second_energy = _centroid(second)
    shift = np.array([second_y - first_y, second_x - first_x], dtype=np.float64)
    norm = float(np.linalg.norm(shift))
    if norm > max_shift:
        shift *= max_shift / norm
    return {
        "dy": float(shift[0]),
        "dx": float(shift[1]),
        "active_energy": float(0.5 * (first_energy + second_energy)),
    }


def _displacement_skill(predicted: dict, truth: dict) -> float:
    """Improvement over frozen zero-motion displacement, bounded to ``[0,1]``."""
    truth_vector = np.array([truth["dy"], truth["dx"]], dtype=np.float64)
    predicted_vector = np.array([predicted["dy"], predicted["dx"]], dtype=np.float64)
    frozen_error = float(np.linalg.norm(truth_vector))
    if frozen_error < MIN_TRUTH_TRANSLATION_PIXELS:
        return float("nan")
    predicted_error = float(np.linalg.norm(predicted_vector - truth_vector))
    return float(np.clip(1.0 - predicted_error / frozen_error, 0.0, 1.0))


def motion_readout(
    predicted_frames: np.ndarray,
    truth_frames: np.ndarray,
    filament_cells: np.ndarray,
    post_frames: list[int] | tuple[int, ...],
    *,
    high_pass_window: int = HIGH_PASS_WINDOW,
) -> dict:
    """Measure phase-sensitive temporal evolution on filament cells.

    The returned headline score is an energy-weighted mean over consecutive
    post-frontier transitions.  Weighting by raw-truth motion energy prevents
    nearly static transitions from dominating the readout while retaining all
    finite transition-level values for inspection.
    """
    pred = _as_gray_stack(predicted_frames)
    truth = _as_gray_stack(truth_frames)
    if pred.shape != truth.shape:
        raise ValueError(
            f"prediction and truth shapes differ: {pred.shape} != {truth.shape}"
        )
    cells = np.asarray(filament_cells, dtype=bool)
    if cells.shape != (mv.GRID_H, mv.GRID_W):
        raise ValueError(
            f"filament_cells must be {(mv.GRID_H, mv.GRID_W)}, got {cells.shape}"
        )
    pixel_mask = _cell_mask_to_pixels(cells)
    indices = [int(index) for index in post_frames]
    if len(indices) < 2:
        raise ValueError("at least two post-frontier frames are required")
    if min(indices) < 0 or max(indices) >= pred.shape[0]:
        raise ValueError("post-frontier frame index is out of bounds")

    pred_hp = _high_pass_stack(pred, window=high_pass_window)
    truth_hp = _high_pass_stack(truth, window=high_pass_window)
    truth_values = np.abs(truth_hp[np.asarray(indices)][:, pixel_mask])
    truth_scale = float(np.percentile(truth_values, 95.0))
    if not np.isfinite(truth_scale) or truth_scale <= 1e-12:
        truth_scale = 1.0

    transitions: list[dict] = []
    for first, second in zip(indices[:-1], indices[1:], strict=True):
        read = _transition_readout(
            pred_hp[second] - pred_hp[first],
            truth_hp[second] - truth_hp[first],
            pixel_mask,
            truth_scale=truth_scale,
        )
        predicted_translation = _estimate_translation(
            pred_hp[first], pred_hp[second], pixel_mask
        )
        truth_translation = _estimate_translation(
            truth_hp[first], truth_hp[second], pixel_mask
        )
        displacement_skill = _displacement_skill(
            predicted_translation, truth_translation
        )
        truth_translation_norm = float(
            np.hypot(truth_translation["dy"], truth_translation["dx"])
        )
        phase_score = (
            max(0.0, float(read["phase_alignment"]))
            if _is_finite_number(read["phase_alignment"])
            else None
        )
        measurable = _is_finite_number(displacement_skill) and phase_score is not None
        read["score"] = (
            0.5 * (phase_score + float(displacement_skill)) if measurable else None
        )
        read["displacement_skill"] = (
            float(displacement_skill) if _is_finite_number(displacement_skill) else None
        )
        read["measurable"] = measurable
        read["nonmeasurable_reason"] = (
            None
            if measurable
            else "truth_translation_below_minimum_or_temporal_change_undefined"
        )
        read["truth_translation_norm_pixels"] = truth_translation_norm
        read["predicted_translation_pixels"] = predicted_translation
        read["truth_translation_pixels"] = truth_translation
        read.update({"from_frame": first, "to_frame": second})
        transitions.append(read)

    finite = [row for row in transitions if _is_finite_number(row["score"])]
    if not finite:
        score = float("nan")
    else:
        weights = np.square([row["truth_motion_norm"] for row in finite])
        score = float(np.average([row["score"] for row in finite], weights=weights))
    return {
        "score": score,
        "n_transitions": len(finite),
        "n_nonmeasurable_transitions": len(transitions) - len(finite),
        "truth_high_pass_scale": truth_scale,
        "transitions": transitions,
    }


def dynamic_range_verdict(role_scores: dict[str, float]) -> dict:
    """Validate the readout controls and state the model-headroom verdict."""
    required = {"persistence", "oracle", "coloured_noise", "map"}
    missing = sorted(required - set(role_scores))
    if missing:
        raise ValueError(f"missing role scores: {missing}")
    scores = {key: float(value) for key, value in role_scores.items()}
    checks = {
        "persistence_near_zero": bool(
            np.isfinite(scores["persistence"])
            and abs(scores["persistence"]) <= PERSISTENCE_ABS_MAX
        ),
        "oracle_high": bool(
            np.isfinite(scores["oracle"]) and scores["oracle"] >= ORACLE_MIN
        ),
        "coloured_noise_near_zero": bool(
            np.isfinite(scores["coloured_noise"])
            and abs(scores["coloured_noise"]) <= COLOURED_NOISE_ABS_MAX
        ),
        "map_separable_from_persistence": bool(
            np.isfinite(scores["map"])
            and np.isfinite(scores["persistence"])
            and scores["map"] - scores["persistence"] >= MAP_SEPARATION_MIN
        ),
    }
    validated = bool(all(checks.values()))
    map_fraction = float(scores["map"] / scores["oracle"])
    if not validated:
        headroom = "UNJUDGEABLE"
        detail = (
            "The control range is not fully validated; this readout cannot gate "
            "a temporal-evolution retrain."
        )
    elif map_fraction >= MOTION_CAPTURED_MIN:
        headroom = "LOW_HEADROOM"
        detail = (
            "MAP already captures at least 75% of the oracle motion readout; a "
            "temporal-evolution retrain has little measured headroom."
        )
    else:
        headroom = "HEADROOM"
        detail = (
            "MAP is separated from frozen persistence but remains below 75% of "
            "the oracle range; a temporal-evolution retrain now has a validated "
            "readout and measurable headroom."
        )
    return {
        "validated": validated,
        "checks": checks,
        "thresholds": {
            "persistence_abs_max": PERSISTENCE_ABS_MAX,
            "oracle_min": ORACLE_MIN,
            "coloured_noise_abs_max": COLOURED_NOISE_ABS_MAX,
            "map_minus_persistence_min": MAP_SEPARATION_MIN,
            "motion_captured_fraction_min": MOTION_CAPTURED_MIN,
        },
        "map_minus_persistence": scores["map"] - scores["persistence"],
        "map_fraction_of_oracle": map_fraction,
        "headroom": headroom,
        "detail": detail,
    }


def _raw_frames_at_times(win: rd.DemoWindow) -> tuple[np.ndarray, np.ndarray]:
    """Read raw rbb frames nearest the exact token-window timestamps.

    The forecast windows are decimated from a wider native-cadence window, so
    reading a contiguous slice from ``win.start`` would compare predictions to
    the wrong camera instants.  Timestamp lookup preserves the physical frame
    identity used by the model input.
    """
    import zarr

    from imas_ambix.camdyn.dataset import level1_shot_path

    path = level1_shot_path(win.shot_id)
    if path is None:
        raise FileNotFoundError(f"raw camera store unavailable for shot {win.shot_id}")
    group = zarr.open_group(str(path / "rbb"), mode="r")
    frame_time = np.asarray(group["time"], dtype=np.float64)
    target = np.asarray(win.frame_time, dtype=np.float64)
    right = np.searchsorted(frame_time, target, side="left")
    right = np.clip(right, 0, frame_time.size - 1)
    left = np.maximum(right - 1, 0)
    choose_left = np.abs(frame_time[left] - target) <= np.abs(
        frame_time[right] - target
    )
    indices = np.where(choose_left, left, right).astype(np.int64)
    errors = np.abs(frame_time[indices] - target)
    frames = np.asarray(group["data"].oindex[indices, :, :])
    return frames, errors


def _bootstrap_interval(values: list[float], *, seed: int = 20260819) -> list[float]:
    """Deterministic descriptive bootstrap interval over transition scores."""
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if not arr.size:
        return [None, None]
    if arr.size == 1:
        return [float(arr[0]), float(arr[0])]
    rng = np.random.default_rng(seed)
    draws = rng.choice(arr, size=(2000, arr.size), replace=True).mean(axis=1)
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def _aggregate_roles(per_window: list[dict]) -> dict:
    """Aggregate role readouts across windows and noise seeds."""
    roles: dict[str, dict] = {}
    for role in ("persistence", "oracle", "coloured_noise", "map"):
        rows = []
        for window in per_window:
            entries = window["roles"][role]
            if isinstance(entries, dict):
                entries = [entries]
            rows.extend(entries)
        transition_scores = [
            transition["score"]
            for entry in rows
            for transition in entry["transitions"]
            if _is_finite_number(transition["score"])
        ]
        entry_scores = [
            entry["score"] for entry in rows if _is_finite_number(entry["score"])
        ]
        total_transitions = sum(len(entry["transitions"]) for entry in rows)
        roles[role] = {
            "score": float(np.mean(entry_scores)) if entry_scores else None,
            "transition_ci95": _bootstrap_interval(transition_scores),
            "n_role_evaluations": len(entry_scores),
            "n_transitions": len(transition_scores),
            "n_nonmeasurable_transitions": total_transitions - len(transition_scores),
        }
    return roles


def _decode_roles(grids: dict, work_dir: Path, device: str) -> dict:
    """Decode from the frozen tokenizer checkout so cached weights resolve."""
    previous = Path.cwd()
    try:
        os.chdir(rd.MAGVIT2_ROOT)
        return sf._decode_grids(grids, work_dir, device)
    finally:
        os.chdir(previous)


def evaluate_checkpoint(*, device: str = "cuda", smoke: bool = False) -> dict:
    """Evaluate the existing dynamics checkpoint without modifying it."""
    import torch

    from imas_ambix.camdyn.arm_compare import _load_arm

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    windows = sf.select_structure_windows()
    if smoke:
        windows = windows[:1]
    if not windows:
        raise RuntimeError("no held-out filament windows could be selected")
    model, _cfg, cond_stats = _load_arm(rd.DYNAMICS_CKPT, torch, dev)
    work_root = Path(
        tempfile.mkdtemp(prefix="motion-readout-", dir=os.environ.get("TMPDIR", "/tmp"))
    )
    per_window: list[dict] = []
    try:
        for window_index, selected in enumerate(windows):
            win = selected.window
            logger.info("[motion] shot %d start %d", win.shot_id, win.start)
            _visible, bit_logits = sf.forward_bit_logits(
                model, torch, dev, win, cond_stats, sf.SCENARIO, sf.FRONTIER
            )
            map_tokens = sf.decode_map(bit_logits)
            persistence_tokens = mv.persistence_tokens(win.true_tokens, sf.FRONTIER)
            decode_dir = work_root / f"window-{window_index}"
            decode_dir.mkdir(parents=True, exist_ok=True)
            decoded = _decode_roles(
                {"map": map_tokens, "persistence": persistence_tokens},
                decode_dir,
                "cuda" if dev.type == "cuda" else "cpu",
            )
            raw, time_errors = _raw_frames_at_times(win)
            truth = _as_gray_stack(raw)
            cells = filament_cell_mask(raw, sf.FRONTIER)
            if not cells.get("calibrated"):
                raise RuntimeError(
                    f"shot {win.shot_id}: {cells.get('reason', 'uncalibrated cells')}"
                )
            post = list(range(sf.FRONTIER, win.true_tokens.shape[0]))
            role_results: dict[str, object] = {
                "map": motion_readout(decoded["map"], truth, cells["filament"], post),
                "persistence": motion_readout(
                    decoded["persistence"], truth, cells["filament"], post
                ),
                "oracle": motion_readout(truth, truth, cells["filament"], post),
            }
            noise_results = []
            for seed_offset in range(NOISE_SEEDS if not smoke else 2):
                noise = sf.coloured_noise_like(
                    truth, post, np.random.default_rng(sf.SEED0 + seed_offset)
                )
                noise_results.append(
                    motion_readout(noise, truth, cells["filament"], post)
                )
            role_results["coloured_noise"] = noise_results
            per_window.append(
                {
                    "shot_id": int(win.shot_id),
                    "start": int(win.start),
                    "frame_time_ms": (np.asarray(win.frame_time) * 1e3).tolist(),
                    "max_raw_time_alignment_error_us": float(time_errors.max() * 1e6),
                    "n_filament_cells": int(cells["n_filament"]),
                    "roles": role_results,
                }
            )
    finally:
        with contextlib.suppress(Exception):
            del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    roles = _aggregate_roles(per_window)
    scores = {role: result["score"] for role, result in roles.items()}
    verdict = dynamic_range_verdict(scores)
    return {
        "task": "phase-sensitive filament-motion readout dynamic-range validation",
        "metric": {
            "name": "filament_motion_alignment",
            "definition": (
                "equal-weight composite of high-pass energy-centroid displacement "
                "skill over frozen zero-motion and non-negative, amplitude-aware "
                "phase alignment of spatially high-passed inter-frame changes on "
                "raw-camera-selected filament cells"
            ),
            "range": [-1.0, 1.0],
            "support": "top-tercile high-frequency cells in the lower camera edge band",
            "truth": "raw rbb frames selected by exact model-window timestamps",
            "high_pass_window_pixels": HIGH_PASS_WINDOW,
            "max_translation_pixels": MAX_TRANSLATION_PIXELS,
            "min_truth_translation_pixels": MIN_TRUTH_TRANSLATION_PIXELS,
        },
        "checkpoint": str(rd.DYNAMICS_CKPT),
        "scenario": sf.SCENARIO,
        "frontier_frame": sf.FRONTIER,
        "n_windows": len(per_window),
        "shots": [window["shot_id"] for window in per_window],
        "noise_seeds_per_window": NOISE_SEEDS if not smoke else 2,
        "roles": roles,
        "dynamic_range": verdict,
        "per_window": per_window,
        "caveat": (
            "The held-out persistent-filament selector yielded two usable windows; "
            "transition-level intervals are descriptive because adjacent frames "
            "within a window are correlated."
        ),
    }


def render_figure(summary: dict, out_path: Path) -> None:
    """Render control range and transition-resolved MAP evidence."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    roles = summary["roles"]
    order = ["persistence", "coloured_noise", "map", "oracle"]
    colours = ["#777777", "#b07aa1", "#2a6fbb", "#2f8f46"]
    values = [roles[role]["score"] for role in order]
    fig, (ax_bar, ax_time) = plt.subplots(
        1, 2, figsize=(11.5, 4.6), constrained_layout=True
    )
    ax_bar.bar(order, values, color=colours, width=0.68)
    ax_bar.axhline(0.0, color="#333333", lw=0.8)
    ax_bar.axhline(MOTION_CAPTURED_MIN, color="#777777", lw=0.8, ls="--")
    ax_bar.set_ylim(-0.12, 1.05)
    ax_bar.set_ylabel("filament-motion alignment")
    ax_bar.set_title("Measured control range")
    ax_bar.tick_params(axis="x", rotation=18)
    for index, value in enumerate(values):
        ax_bar.text(
            index,
            value + (0.025 if value >= 0 else -0.055),
            f"{value:.3f}",
            ha="center",
        )

    for window in summary["per_window"]:
        transitions = window["roles"]["map"]["transitions"]
        times = np.asarray(window["frame_time_ms"])
        x = [times[row["to_frame"]] for row in transitions]
        y = [row["score"] for row in transitions]
        ax_time.plot(x, y, "-o", ms=4, label=f"shot {window['shot_id']}")
    ax_time.axhline(0.0, color="#333333", lw=0.8)
    ax_time.axhline(MOTION_CAPTURED_MIN, color="#777777", lw=0.8, ls="--")
    ax_time.set_ylim(-0.25, 1.05)
    ax_time.set_xlabel("camera time (ms)")
    ax_time.set_ylabel("MAP transition alignment")
    ax_time.set_title("Motion readout by forecast transition")
    ax_time.legend(frameon=False)

    verdict = summary["dynamic_range"]
    fig.suptitle(
        f"Filament motion: {verdict['headroom']}  "
        f"(MAP−persistence {verdict['map_minus_persistence']:+.3f})",
        fontsize=12,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def run(
    *,
    json_path: Path = DEFAULT_JSON,
    figure_path: Path = DEFAULT_FIGURE,
    device: str = "cuda",
    smoke: bool = False,
) -> dict:
    """Evaluate, render, and persist the motion-readout evidence."""
    summary = evaluate_checkpoint(device=device, smoke=smoke)
    render_figure(summary, figure_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    summary["figure"] = str(figure_path)
    strict_summary = _strict_json_data(summary)
    json_path.write_text(json.dumps(strict_summary, indent=2, allow_nan=False) + "\n")
    return strict_summary


def main(argv=None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    summary = run(
        json_path=args.json,
        figure_path=args.figure,
        device=args.device,
        smoke=args.smoke,
    )
    print(json.dumps(summary["roles"], indent=2))
    print(json.dumps(summary["dynamic_range"], indent=2))
    return 0 if summary["dynamic_range"]["validated"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
