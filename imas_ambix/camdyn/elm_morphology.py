"""Dalpha-selected ELM-frame morphology evaluation at a 10 ms horizon.

The selection diagnostic is never passed to either reconstruction arm.  A
fast native ``xim`` Dalpha burst supplies only a held-out target time.  Both
frozen arms then receive the same camera history ending 10 ms before that
target, and their target-frame predictions are scored on the fixed
edge/divertor support.

Uncertainty is estimated over independent selected windows.  Cell-level
values are reduced inside each window before the paired bootstrap so a shot
with more active pixels cannot masquerade as additional evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from imas_ambix.camdyn.metrics import (
    ELM_MORPHOLOGY_HORIZON_MS,
    bootstrap_ci,
    elm_edge_divertor_mask,
    elm_frame_morphology_fidelity,
)
from imas_ambix.camdyn.model import bitwise_nll

logger = logging.getLogger(__name__)

N_FRAMES = 16
FRONTIER_FRAME = 8
TARGET_FRAME = 12
DALPHA_SIGMA_GATE = 4.0
CAMERA_HEIGHT_REFERENCE = 112
CAMERA_WIDTH_REFERENCE = 156
CAMERA_HEIGHT_TOLERANCE = 24
CAMERA_WIDTH_TOLERANCE = 24
DEFAULT_MAX_CANDIDATES = 120
DEFAULT_MAX_WINDOWS = 12

DEFAULT_SPLIT = Path(__file__).parent / "artifacts" / "camdyn_split_v0.json"
DEFAULT_ARTIFACT = Path(__file__).parent / "artifacts" / "elm_morphology.json"
DEFAULT_FIGURE = Path("docs/figures/camera-dynamics-wm/fig-cdw-elm-morphology.png")


@dataclass(frozen=True)
class SelectedWindow:
    """One held-out camera window aligned to a native fast-Dalpha burst."""

    window: object
    target_frame: int
    dalpha_channel: str
    dalpha_burst_time_s: float
    dalpha_sigma: float
    dalpha_peak_to_baseline: float
    actual_horizon_ms: float
    target_alignment_ms: float
    camera_resolution: tuple[int, int]


def aligned_frame_indices(
    frame_time: np.ndarray,
    burst_time_s: float,
    *,
    n_frames: int = N_FRAMES,
    frontier_frame: int = FRONTIER_FRAME,
    target_frame: int = TARGET_FRAME,
    horizon_ms: float = ELM_MORPHOLOGY_HORIZON_MS,
) -> np.ndarray | None:
    """Map a physical 10 ms frontier/target grid to real camera frame indices.

    Desired sample times are uniformly spaced so ``target_frame`` is exactly
    ``horizon_ms`` after ``frontier_frame``.  Each is mapped to the nearest
    native camera frame.  Windows with duplicate or non-monotonic native
    indices are rejected rather than fabricating frames.
    """
    ft = np.asarray(frame_time, dtype=np.float64).reshape(-1)
    if ft.size < n_frames or not np.all(np.isfinite(ft)):
        return None
    if not 0 <= frontier_frame < target_frame < n_frames:
        raise ValueError("frontier and target frames must be ordered in the window")
    step_s = horizon_ms / 1000.0 / (target_frame - frontier_frame)
    desired = burst_time_s + (np.arange(n_frames) - target_frame) * step_s
    pos = np.searchsorted(ft, desired, side="left")
    pos = np.clip(pos, 0, ft.size - 1)
    left = np.clip(pos - 1, 0, ft.size - 1)
    choose_left = np.abs(ft[left] - desired) <= np.abs(ft[pos] - desired)
    idx = np.where(choose_left, left, pos).astype(np.int64)
    if np.any(np.diff(idx) <= 0):
        return None
    actual_ms = float((ft[idx[target_frame]] - ft[idx[frontier_frame]]) * 1e3)
    if not 0.75 * horizon_ms <= actual_ms <= 1.25 * horizon_ms:
        return None
    return idx


def _candidate_shots(split_path: Path, max_candidates: int) -> list[int]:
    payload = json.loads(Path(split_path).read_text())
    held_out = [int(s) for s in payload["held_out"]]
    if not held_out or max_candidates <= 0:
        return []
    step = max(1, len(held_out) // max_candidates)
    return held_out[::step][:max_candidates]


def _read_selected_tokens(shot_id: int, indices: np.ndarray) -> np.ndarray | None:
    import zarr

    from imas_ambix.camdyn.dataset import frames_token_path

    path = frames_token_path(shot_id)
    if not path.exists():
        return None
    group = zarr.open_group(str(path), mode="r")
    tokens = group["tokens"]
    if int(indices[-1]) >= int(tokens.shape[0]):
        return None
    return np.stack(
        [np.asarray(tokens[int(i)], dtype=np.int64) for i in indices], axis=0
    )


def select_dalpha_windows(
    *,
    split_path: Path = DEFAULT_SPLIT,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    max_windows: int = DEFAULT_MAX_WINDOWS,
) -> list[SelectedWindow]:
    """Select deterministic held-out windows from native fast-Dalpha bursts."""
    from imas_ambix.camdyn import reconstruction_demo as rd
    from imas_ambix.camdyn.elm_sampling_audit import (
        frame_resolution,
        read_camera_frame_times,
    )
    from imas_ambix.camdyn.tokenizer_fidelity import _fast_dalpha_burst

    selected: list[SelectedWindow] = []
    for shot_id in _candidate_shots(Path(split_path), max_candidates):
        burst = _fast_dalpha_burst(shot_id)
        if burst is None or burst.burst_sigma < DALPHA_SIGMA_GATE:
            continue
        resolution = frame_resolution(shot_id)
        if resolution is None:
            continue
        canonical_height = (
            abs(int(resolution[0]) - CAMERA_HEIGHT_REFERENCE) <= CAMERA_HEIGHT_TOLERANCE
        )
        canonical_width = (
            abs(int(resolution[1]) - CAMERA_WIDTH_REFERENCE) <= CAMERA_WIDTH_TOLERANCE
        )
        if not (canonical_height and canonical_width):
            continue
        frame_time = read_camera_frame_times(shot_id)
        if frame_time is None:
            continue
        indices = aligned_frame_indices(frame_time, burst.burst_time_s)
        if indices is None:
            continue
        tokens = _read_selected_tokens(shot_id, indices)
        if tokens is None:
            continue
        sampled_time = np.asarray(frame_time, dtype=np.float64)[indices]
        dt = np.concatenate([np.diff(sampled_time), np.diff(sampled_time)[-1:]])
        actual_horizon_ms = float(
            (sampled_time[TARGET_FRAME] - sampled_time[FRONTIER_FRAME]) * 1e3
        )
        selected.append(
            SelectedWindow(
                window=rd.DemoWindow(
                    shot_id=shot_id,
                    start=int(indices[0]),
                    frame_time=sampled_time,
                    dt=dt,
                    valid=np.ones(N_FRAMES, dtype=bool),
                    true_tokens=tokens,
                    motion_fraction=0.0,
                ),
                target_frame=TARGET_FRAME,
                dalpha_channel=burst.channel,
                dalpha_burst_time_s=float(burst.burst_time_s),
                dalpha_sigma=float(burst.burst_sigma),
                dalpha_peak_to_baseline=float(burst.burst_ratio),
                actual_horizon_ms=actual_horizon_ms,
                target_alignment_ms=float(
                    (sampled_time[TARGET_FRAME] - burst.burst_time_s) * 1e3
                ),
                camera_resolution=(int(resolution[0]), int(resolution[1])),
            )
        )
        logger.info(
            "selected shot %d %s burst %.3f sigma at %.3f ms alignment",
            shot_id,
            burst.channel,
            burst.burst_sigma,
            selected[-1].target_alignment_ms,
        )
        if len(selected) >= max_windows:
            break
    return selected


def _evaluate_arm(checkpoint: Path, windows: list[SelectedWindow], device: str):
    import torch

    from imas_ambix.camdyn import structure_fidelity as sf
    from imas_ambix.camdyn.arm_compare import _load_arm

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model, _cfg, cond_stats = _load_arm(checkpoint, torch, dev)
    out: list[dict] = []
    region = elm_edge_divertor_mask()
    try:
        for selected in windows:
            win = selected.window
            _visible, logits = sf.forward_bit_logits(
                model, torch, dev, win, cond_stats, "frontier", FRONTIER_FRAME
            )
            target = np.asarray(win.true_tokens[selected.target_frame], dtype=np.int64)
            target_logits = np.asarray(logits[selected.target_frame])
            shifts = np.arange(target_logits.shape[-1], dtype=np.int64)
            predicted = ((target_logits > 0.0).astype(np.int64) << shifts).sum(axis=-1)
            nll = bitwise_nll(target_logits, target)
            out.append(
                {
                    "edge_divertor_nll": float(nll[region].mean()),
                    "edge_divertor_top1": float(
                        (predicted[region] == target[region]).mean()
                    ),
                    "_predicted_tokens": predicted,
                }
            )
    finally:
        del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    return out


def _decode_morphology_scores(
    windows: list[SelectedWindow], baseline: list[dict], dynamics: list[dict]
) -> None:
    """Decode target/reference/predictions once and attach pixel-response scores."""
    from imas_ambix.camdyn import reconstruction_demo as rd
    from imas_ambix.camdyn.recon_movie_run import BundleBuilder

    with tempfile.TemporaryDirectory(prefix="elm-morphology-", dir="/tmp") as tmp:
        tmp_path = Path(tmp)
        token_bundle = tmp_path / "tokens.npz"
        image_bundle = tmp_path / "images.npz"
        builder = BundleBuilder()
        for index, selected in enumerate(windows):
            win = selected.window
            window_index = builder.add_window({"shot_id": int(win.shot_id)})
            grids = {
                "reference": win.true_tokens[FRONTIER_FRAME - 1],
                "target": win.true_tokens[selected.target_frame],
                "baseline": baseline[index]["_predicted_tokens"],
                "dynamics": dynamics[index]["_predicted_tokens"],
            }
            for role, grid in grids.items():
                builder.add_grid(grid[None], window_index, "elm", role)
        builder.save(token_bundle)
        # The frozen decoder resolves its already-staged LPIPS side weight
        # relative to the Open-MAGVIT2 root.  GPU nodes have no outbound
        # network, so launch from that root instead of triggering a download.
        original_cwd = Path.cwd()
        try:
            os.chdir(rd.MAGVIT2_ROOT)
            rd.run_decode_subprocess(token_bundle, image_bundle, "cuda")
        finally:
            os.chdir(original_cwd)
        decoded = np.load(image_bundle, allow_pickle=True)
        images = np.asarray(decoded["images"], dtype=np.uint8)
        index_rows = json.loads(str(decoded["index"]))
        slots = {
            (int(row["window"]), str(row["role"])): int(row["slot"])
            for row in index_rows
        }
        token_region = elm_edge_divertor_mask()
        pixel_region = np.repeat(np.repeat(token_region, 16, axis=0), 16, axis=1)
        for index in range(len(windows)):
            reference = images[slots[(index, "reference")], 0]
            target = images[slots[(index, "target")], 0]
            for role, records in (("baseline", baseline), ("dynamics", dynamics)):
                predicted = images[slots[(index, role)], 0]
                records[index].update(
                    elm_frame_morphology_fidelity(
                        predicted, target, reference, region_mask=pixel_region
                    )
                )
                del records[index]["_predicted_tokens"]


def summarise_paired_scores(
    baseline: list[dict], dynamics: list[dict], *, seed: int = 0
) -> dict:
    """Aggregate exact window-paired arm scores with window-level CIs."""
    if len(baseline) != len(dynamics) or not baseline:
        raise ValueError("baseline and dynamics scores must be non-empty and paired")
    base_morph = np.array([r["morphology_fidelity"] for r in baseline])
    dyn_morph = np.array([r["morphology_fidelity"] for r in dynamics])
    base_nll = np.array([r["edge_divertor_nll"] for r in baseline])
    dyn_nll = np.array([r["edge_divertor_nll"] for r in dynamics])
    morph_ci = bootstrap_ci(dyn_morph - base_morph, seed=seed)
    nll_ci = bootstrap_ci(base_nll - dyn_nll, seed=seed)
    morph_ci["n_pairs"] = len(baseline)
    nll_ci["n_pairs"] = len(baseline)
    return {
        "baseline_morphology_mean": float(base_morph.mean()),
        "dynamics_morphology_mean": float(dyn_morph.mean()),
        "dynamics_minus_baseline_morphology": morph_ci,
        "baseline_nll_mean": float(base_nll.mean()),
        "dynamics_nll_mean": float(dyn_nll.mean()),
        "baseline_minus_dynamics_nll": nll_ci,
        "reproduces_existing_arm_gap": bool(morph_ci["favours_dynamics"]),
        "edge_divertor_nll_favours_dynamics": bool(nll_ci["favours_dynamics"]),
    }


def _metric_spec_sha256() -> str:
    source = inspect.getsource(elm_frame_morphology_fidelity)
    source += inspect.getsource(elm_edge_divertor_mask)
    return hashlib.sha256(source.encode()).hexdigest()


def write_figure(payload: dict, path: Path = DEFAULT_FIGURE) -> Path:
    """Render paired window scores and the bootstrap gap without decoration."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = payload["per_window"]
    x = np.arange(len(rows))
    base = np.array([r["baseline"]["morphology_fidelity"] for r in rows])
    dyn = np.array([r["dynamics"]["morphology_fidelity"] for r in rows])
    ci = payload["aggregate"]["dynamics_minus_baseline_morphology"]

    fig, (ax, gap_ax) = plt.subplots(
        1,
        2,
        figsize=(9.2, 3.5),
        gridspec_kw={"width_ratios": [2.4, 1]},
    )
    for i in x:
        ax.plot([i, i], [base[i], dyn[i]], color="0.75", lw=1)
    ax.scatter(x, base, color="0.45", s=24, label="per-frame baseline")
    ax.scatter(x, dyn, color="#167a72", s=24, label="dynamics")
    ax.set_xlabel("Dalpha-selected held-out window")
    ax.set_ylabel("ELM response morphology fidelity")
    ax.legend(frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)

    gap_ax.axhline(0.0, color="0.45", lw=0.8)
    gap_ax.errorbar(
        [0],
        [ci["mean"]],
        yerr=[[ci["mean"] - ci["lo"]], [ci["hi"] - ci["mean"]]],
        fmt="o",
        color="#167a72",
        capsize=4,
    )
    gap_ax.set_xticks([0], ["dynamics − baseline"])
    gap_ax.set_ylabel("paired morphology gap")
    gap_ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("10 ms ELM-frame morphology on fast-Dalpha-selected windows")
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def run(
    *,
    baseline_checkpoint: Path,
    dynamics_checkpoint: Path,
    split_path: Path = DEFAULT_SPLIT,
    artifact_path: Path = DEFAULT_ARTIFACT,
    figure_path: Path = DEFAULT_FIGURE,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    max_windows: int = DEFAULT_MAX_WINDOWS,
    device: str = "cuda",
) -> dict:
    """Select windows, score both arms, and write the evidence artifacts."""
    windows = select_dalpha_windows(
        split_path=split_path,
        max_candidates=max_candidates,
        max_windows=max_windows,
    )
    if len(windows) < 2:
        raise RuntimeError("fewer than two eligible Dalpha-selected windows")
    baseline = _evaluate_arm(Path(baseline_checkpoint), windows, device)
    dynamics = _evaluate_arm(Path(dynamics_checkpoint), windows, device)
    _decode_morphology_scores(windows, baseline, dynamics)
    aggregate = summarise_paired_scores(baseline, dynamics)
    per_window = []
    for selected, base, dyn in zip(windows, baseline, dynamics, strict=True):
        per_window.append(
            {
                "shot_id": int(selected.window.shot_id),
                "dalpha_channel": selected.dalpha_channel,
                "dalpha_burst_time_s": selected.dalpha_burst_time_s,
                "dalpha_sigma": selected.dalpha_sigma,
                "dalpha_peak_to_baseline": selected.dalpha_peak_to_baseline,
                "actual_horizon_ms": selected.actual_horizon_ms,
                "target_alignment_ms": selected.target_alignment_ms,
                "camera_resolution": list(selected.camera_resolution),
                "baseline": base,
                "dynamics": dyn,
            }
        )
    payload = {
        "task": "Dalpha-selected ELM-frame response morphology",
        "pre_registered_metric": {
            "horizon_ms": ELM_MORPHOLOGY_HORIZON_MS,
            "frontier_frame": FRONTIER_FRAME,
            "target_frame": TARGET_FRAME,
            "region": "outer 3 token columns on each side plus lower 5 rows",
            "response": "decoded brightness change from the last observed frame",
            "score": (
                "positive spatial response correlation multiplied by signed "
                "mean-brightness fidelity in the fixed region"
            ),
            "decoder": "frozen imagenet_256_L Open-MAGVIT2 tokenizer decoder",
            "bootstrap_unit": "Dalpha-selected held-out window",
            "metric_spec_sha256": _metric_spec_sha256(),
        },
        "selection": {
            "diagnostic": "native fast xim Dalpha; selection only, never model input",
            "sigma_gate": DALPHA_SIGMA_GATE,
            "candidate_policy": "deterministic stride sample of held-out shots",
            "max_candidates": max_candidates,
            "eligible_windows": len(windows),
        },
        "baseline_checkpoint": str(baseline_checkpoint),
        "dynamics_checkpoint": str(dynamics_checkpoint),
        "per_window": per_window,
        "aggregate": aggregate,
    }
    artifact_path = Path(artifact_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(payload, indent=2) + "\n")
    write_figure(payload, Path(figure_path))
    return payload


def main() -> None:
    from imas_ambix.camdyn.reconstruction_demo import BASELINE_CKPT, DYNAMICS_CKPT

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=BASELINE_CKPT)
    parser.add_argument("--dynamics", type=Path, default=DYNAMICS_CKPT)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--max-candidates", type=int, default=DEFAULT_MAX_CANDIDATES)
    parser.add_argument("--max-windows", type=int, default=DEFAULT_MAX_WINDOWS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    payload = run(
        baseline_checkpoint=args.baseline,
        dynamics_checkpoint=args.dynamics,
        split_path=args.split,
        artifact_path=args.artifact,
        figure_path=args.figure,
        max_candidates=args.max_candidates,
        max_windows=args.max_windows,
        device=args.device,
    )
    print(json.dumps(payload["aggregate"], indent=2))


if __name__ == "__main__":
    main()
