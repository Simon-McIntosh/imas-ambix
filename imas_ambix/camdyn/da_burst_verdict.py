"""Conditioning verdict for matched native and control Dalpha burst runs.

The six trained runs are compared within architecture arm on exactly paired
held-out ELM frames at the registered 10 ms horizon.  Morphology uses the
registered decoded-frame metric, token NLL uses the same edge/divertor support,
and uncertainty is a paired bootstrap over shot means.  Positive morphology
deltas and negative NLL deltas favour native fast-Dalpha conditioning.

An evaluation artifact is used only when it contains frame-level ELM records.
Aggregate reconstruction summaries cannot support a paired comparison, so the
fallback scores every final checkpoint on one shared set of selected windows.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import logging
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from imas_ambix.camdyn import elm_morphology as morphology
from imas_ambix.camdyn.metrics import (
    ELM_MORPHOLOGY_HORIZON_MS,
    bootstrap_ci,
    elm_edge_divertor_mask,
    elm_frame_morphology_fidelity,
)

logger = logging.getLogger(__name__)

CHECKPOINT_ROOT = Path("/work/projects/imas_gpu/mast-checkpoints/camdyn")
DEFAULT_OUTPUT_DIR = Path("docs/figures/camera-dynamics-wm-v0/da-burst")
DEFAULT_ARTIFACT = DEFAULT_OUTPUT_DIR / "verdict.json"
DEFAULT_FIGURE = DEFAULT_OUTPUT_DIR / "verdict.png"
DEFAULT_REPORT = Path(
    "/home/ITER/mcintos/.config/reckon/crew/reports/"
    "camera-dynamics-wm-v0/da-burst-census.md"
)

RUN_DIRECTORIES = {
    "baseline": {
        "native": "da_burst_baseline",
        "shuffled": "da_burst_shuffled_baseline",
        "slow": "da_burst_slow_baseline",
    },
    "dynamics": {
        "native": "da_burst_dynamics",
        "shuffled": "da_burst_shuffled_dynamics",
        "slow": "da_burst_slow_dynamics",
    },
}
CONTROL_COMPARISONS = {
    "native_minus_shuffled": "shuffled",
    "native_minus_slow": "slow",
}
REGISTERED_MAX_CANDIDATES = 1_050
REGISTERED_FRAME_COUNT = 59


def _run_key(arm: str, variant: str) -> str:
    return f"{arm}_{variant}"


def _metric_value(record: dict[str, Any], *names: str) -> float | None:
    containers = [record]
    for name in ("metrics", "scores", "elm_frame"):
        value = record.get(name)
        if isinstance(value, dict):
            containers.append(value)
    for container in containers:
        for name in names:
            if name in container:
                try:
                    value = float(container[name])
                except TypeError, ValueError:
                    continue
                if math.isfinite(value):
                    return value
    return None


def _normalise_evaluation_record(record: dict[str, Any]) -> dict[str, Any] | None:
    """Return the common frame-record schema, or ``None`` when incomplete."""
    try:
        shot_id = int(record["shot_id"])
    except KeyError, TypeError, ValueError:
        return None
    morphology_score = _metric_value(record, "morphology_fidelity", "morphology")
    token_nll = _metric_value(
        record,
        "token_nll",
        "edge_divertor_nll",
        "masked_token_nll",
        "masked_nll",
    )
    horizon_ms = _metric_value(record, "actual_horizon_ms", "horizon_ms")
    if morphology_score is None or token_nll is None or horizon_ms is None:
        return None
    if (
        not 0.75 * ELM_MORPHOLOGY_HORIZON_MS
        <= horizon_ms
        <= 1.25 * ELM_MORPHOLOGY_HORIZON_MS
    ):
        return None
    if "frame_key" in record:
        frame_key = str(record["frame_key"])
    elif "dalpha_burst_time_s" in record:
        frame_key = f"{shot_id}:{float(record['dalpha_burst_time_s']):.9f}"
    elif "target_time_s" in record:
        frame_key = f"{shot_id}:{float(record['target_time_s']):.9f}"
    elif "window_index" in record:
        frame_key = f"{shot_id}:window-{int(record['window_index'])}"
    else:
        frame_key = f"{shot_id}:frame-{int(record.get('target_frame', 0))}"
    return {
        "shot_id": shot_id,
        "frame_key": frame_key,
        "actual_horizon_ms": horizon_ms,
        "morphology_fidelity": morphology_score,
        "token_nll": token_nll,
    }


def records_from_evaluation(payload: dict[str, Any]) -> list[dict[str, Any]] | None:
    """Extract per-frame ELM scores from supported evaluation layouts."""
    candidate_paths = (
        ("elm_frames",),
        ("per_window_elm_frames",),
        ("elm_morphology", "per_window"),
        ("held_out", "elm_frames"),
        ("held_out", "elm_morphology", "per_window"),
    )
    for path in candidate_paths:
        value: Any = payload
        for part in path:
            if not isinstance(value, dict) or part not in value:
                value = None
                break
            value = value[part]
        if not isinstance(value, list) or not value:
            continue
        records = [
            normalised
            for row in value
            if isinstance(row, dict)
            and (normalised := _normalise_evaluation_record(row)) is not None
        ]
        if len(records) == len(value):
            return records
    return None


def _frame_index(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for record in records:
        key = str(record["frame_key"])
        if key in index:
            raise ValueError(f"duplicate ELM frame key {key!r}")
        index[key] = record
    return index


def _within_horizon_tolerance(horizon_ms: float) -> bool:
    return bool(
        0.75 * ELM_MORPHOLOGY_HORIZON_MS
        <= horizon_ms
        <= 1.25 * ELM_MORPHOLOGY_HORIZON_MS
    )


def registered_frame_rows(
    windows: list[morphology.SelectedWindow],
) -> list[dict[str, Any]]:
    """Describe the fixed held-out ELM population without model scores."""
    return [
        {
            "shot_id": int(selected.window.shot_id),
            "frame_key": (
                f"{int(selected.window.shot_id)}:"
                f"{float(selected.dalpha_burst_time_s):.9f}"
            ),
            "dalpha_burst_time_s": float(selected.dalpha_burst_time_s),
            "actual_horizon_ms": float(selected.actual_horizon_ms),
        }
        for selected in windows
    ]


def build_census(
    registered_frames: list[dict[str, Any]],
    records: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Account for every registered frame and derive the six-way paired set."""
    expected_runs = [
        _run_key(arm, variant)
        for arm, variants in RUN_DIRECTORIES.items()
        for variant in variants
    ]
    indexed = {
        run_key: _frame_index(records.get(run_key, [])) for run_key in expected_runs
    }
    rows: list[dict[str, Any]] = []
    for registered in registered_frames:
        frame_key = str(registered["frame_key"])
        available_runs = [
            run_key for run_key in expected_runs if frame_key in indexed[run_key]
        ]
        missing_runs = [
            run_key for run_key in expected_runs if frame_key not in indexed[run_key]
        ]
        checkpoint_horizons_ms = {
            run_key: float(indexed[run_key][frame_key]["actual_horizon_ms"])
            for run_key in available_runs
        }
        registered_horizon_ms = float(registered["actual_horizon_ms"])
        registered_horizon_ok = _within_horizon_tolerance(registered_horizon_ms)
        checkpoint_horizon_ok = {
            run_key: _within_horizon_tolerance(value)
            for run_key, value in checkpoint_horizons_ms.items()
        }
        horizon_tolerance_failure = not registered_horizon_ok or any(
            not ok for ok in checkpoint_horizon_ok.values()
        )
        six_way_paired = not missing_runs
        support_disagreement = bool(available_runs and missing_runs)
        horizon_values = list(checkpoint_horizons_ms.values())
        horizon_disagreement = (
            len(set(checkpoint_horizon_ok.values())) > 1
            or bool(horizon_values)
            and (
                max(horizon_values + [registered_horizon_ms])
                - min(horizon_values + [registered_horizon_ms])
                > 1e-6
            )
        )
        checkpoint_disagreement = support_disagreement or horizon_disagreement
        if horizon_tolerance_failure:
            disposition = "horizon_tolerance_failure"
        elif not six_way_paired:
            disposition = "six_way_pairing_failure"
        elif checkpoint_disagreement:
            disposition = "checkpoint_disagreement"
        else:
            disposition = "scored"
        rows.append(
            {
                **registered,
                "registered_horizon_within_tolerance": registered_horizon_ok,
                "available_checkpoints": available_runs,
                "missing_checkpoints": missing_runs,
                "checkpoint_horizons_ms": checkpoint_horizons_ms,
                "checkpoint_horizon_within_tolerance": checkpoint_horizon_ok,
                "six_way_paired": six_way_paired,
                "checkpoint_disagreement": checkpoint_disagreement,
                "disposition": disposition,
            }
        )

    disposition_counts = {
        disposition: sum(row["disposition"] == disposition for row in rows)
        for disposition in (
            "scored",
            "six_way_pairing_failure",
            "horizon_tolerance_failure",
            "checkpoint_disagreement",
        )
    }
    scored_frame_keys = [
        str(row["frame_key"]) for row in rows if row["disposition"] == "scored"
    ]
    return {
        "population": (
            "deterministic fast-Dalpha selection over all held-out shots, capped "
            f"at the registered {REGISTERED_FRAME_COUNT} frames"
        ),
        "registered_frame_count": len(registered_frames),
        "scored_frame_count": len(scored_frame_keys),
        "disposition_counts": disposition_counts,
        "checkpoint_scored_frame_counts": {
            run_key: sum(
                str(frame["frame_key"]) in indexed[run_key]
                for frame in registered_frames
            )
            for run_key in expected_runs
        },
        "scored_frame_keys": scored_frame_keys,
        "frames": rows,
    }


def paired_census_records(
    records: dict[str, list[dict[str, Any]]], census: dict[str, Any]
) -> dict[str, list[dict[str, Any]]]:
    """Restrict every checkpoint to the census frames eligible for comparison."""
    scored = set(census["scored_frame_keys"])
    return {
        run_key: [row for row in rows if str(row["frame_key"]) in scored]
        for run_key, rows in records.items()
    }


def shot_clustered_delta(
    native: list[dict[str, Any]],
    control: list[dict[str, Any]],
    metric: str,
    *,
    lower_is_better: bool,
    seed: int = 0,
) -> dict[str, Any]:
    """Paired native-minus-control delta bootstrapped over shot means."""
    native_index = _frame_index(native)
    control_index = _frame_index(control)
    if not native_index or native_index.keys() != control_index.keys():
        raise ValueError("native and control ELM frames must be exactly paired")

    by_shot: dict[int, list[float]] = defaultdict(list)
    native_by_shot: dict[int, list[float]] = defaultdict(list)
    control_by_shot: dict[int, list[float]] = defaultdict(list)
    for key in sorted(native_index):
        native_row = native_index[key]
        control_row = control_index[key]
        if int(native_row["shot_id"]) != int(control_row["shot_id"]):
            raise ValueError(f"shot mismatch for paired ELM frame {key!r}")
        shot_id = int(native_row["shot_id"])
        native_value = float(native_row[metric])
        control_value = float(control_row[metric])
        by_shot[shot_id].append(native_value - control_value)
        native_by_shot[shot_id].append(native_value)
        control_by_shot[shot_id].append(control_value)

    shot_ids = sorted(by_shot)
    shot_deltas = np.array([np.mean(by_shot[s]) for s in shot_ids], dtype=np.float64)
    native_means = np.array(
        [np.mean(native_by_shot[s]) for s in shot_ids], dtype=np.float64
    )
    control_means = np.array(
        [np.mean(control_by_shot[s]) for s in shot_ids], dtype=np.float64
    )
    interval = bootstrap_ci(shot_deltas, seed=seed)
    favours_native = (
        bool(interval["hi"] < 0.0) if lower_is_better else bool(interval["lo"] > 0.0)
    )
    return {
        "orientation": "native_minus_control",
        "mean": interval["mean"],
        "lo": interval["lo"],
        "hi": interval["hi"],
        "alpha": interval["alpha"],
        "clear_of_zero": interval["clear_of_zero"],
        "favours_native": favours_native,
        "native_mean": float(native_means.mean()),
        "control_mean": float(control_means.mean()),
        "n_elm_frames": len(native_index),
        "n_shots": len(shot_ids),
        "bootstrap_unit": "shot mean",
    }


def build_verdict(
    records: dict[str, list[dict[str, Any]]], *, seed: int = 0
) -> dict[str, Any]:
    """Build strict within-arm conditioning comparisons from six run records."""
    arms: dict[str, Any] = {}
    run_summaries: dict[str, Any] = {}
    for arm, variants in RUN_DIRECTORIES.items():
        for variant in variants:
            key = _run_key(arm, variant)
            rows = records[key]
            run_summaries[key] = {
                "arm": arm,
                "conditioning": variant,
                "morphology_mean": float(
                    np.mean([row["morphology_fidelity"] for row in rows])
                ),
                "token_nll_mean": float(np.mean([row["token_nll"] for row in rows])),
                "n_elm_frames": len(rows),
                "n_shots": len({int(row["shot_id"]) for row in rows}),
            }

        native = records[_run_key(arm, "native")]
        comparisons: dict[str, Any] = {}
        for comparison, control in CONTROL_COMPARISONS.items():
            control_records = records[_run_key(arm, control)]
            morphology_delta = shot_clustered_delta(
                native,
                control_records,
                "morphology_fidelity",
                lower_is_better=False,
                seed=seed,
            )
            nll_delta = shot_clustered_delta(
                native,
                control_records,
                "token_nll",
                lower_is_better=True,
                seed=seed,
            )
            comparisons[comparison] = {
                "control": control,
                "morphology": morphology_delta,
                "token_nll": nll_delta,
                "native_beats_control": bool(
                    morphology_delta["favours_native"] and nll_delta["favours_native"]
                ),
            }
        arm_passes = all(row["native_beats_control"] for row in comparisons.values())
        arms[arm] = {
            "comparisons": comparisons,
            "fast_dalpha_beats_both_controls": arm_passes,
            "n_elm_frames": len(native),
            "n_shots": len({int(row["shot_id"]) for row in native}),
        }

    return {
        "arms": arms,
        "runs": run_summaries,
        "fast_dalpha_beats_both_controls": all(
            arm["fast_dalpha_beats_both_controls"] for arm in arms.values()
        ),
    }


def _load_all_evaluation_records(
    checkpoint_root: Path,
) -> tuple[dict[str, list[dict[str, Any]]] | None, dict[str, str]]:
    records: dict[str, list[dict[str, Any]]] = {}
    sources: dict[str, str] = {}
    for arm, variants in RUN_DIRECTORIES.items():
        for variant, directory in variants.items():
            key = _run_key(arm, variant)
            evaluation_path = checkpoint_root / directory / "evaluation.json"
            payload = json.loads(evaluation_path.read_text(encoding="utf-8"))
            extracted = records_from_evaluation(payload)
            if extracted is None:
                return None, {}
            records[key] = extracted
            sources[key] = str(evaluation_path)
    return records, sources


def _decode_rescored_morphology(
    windows: list[morphology.SelectedWindow],
    records: dict[str, list[dict[str, Any]]],
) -> None:
    from imas_ambix.camdyn import reconstruction_demo as reconstruction
    from imas_ambix.camdyn.recon_movie_run import BundleBuilder

    scratch_parent = os.environ.get("TMPDIR", "/tmp")
    with tempfile.TemporaryDirectory(
        prefix="da-burst-verdict-", dir=scratch_parent
    ) as scratch:
        scratch_path = Path(scratch)
        token_bundle = scratch_path / "tokens.npz"
        image_bundle = scratch_path / "images.npz"
        builder = BundleBuilder()
        for index, selected in enumerate(windows):
            window = selected.window
            window_index = builder.add_window({"shot_id": int(window.shot_id)})
            builder.add_grid(
                np.asarray(window.true_tokens[morphology.FRONTIER_FRAME - 1])[None],
                window_index,
                "elm",
                "reference",
            )
            builder.add_grid(
                np.asarray(window.true_tokens[selected.target_frame])[None],
                window_index,
                "elm",
                "target",
            )
            for run_key, run_records in records.items():
                builder.add_grid(
                    np.asarray(run_records[index]["_predicted_tokens"])[None],
                    window_index,
                    "elm",
                    run_key,
                )
        builder.save(token_bundle)
        original_cwd = Path.cwd()
        try:
            os.chdir(reconstruction.MAGVIT2_ROOT)
            reconstruction.run_decode_subprocess(token_bundle, image_bundle, "cpu")
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
            for run_key, run_records in records.items():
                predicted = images[slots[(index, run_key)], 0]
                run_records[index].update(
                    elm_frame_morphology_fidelity(
                        predicted,
                        target,
                        reference,
                        region_mask=pixel_region,
                    )
                )
                del run_records[index]["_predicted_tokens"]


def _rescore_all_checkpoints(
    checkpoint_root: Path,
    *,
    windows: list[morphology.SelectedWindow],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str]]:
    if len(windows) < 2:
        raise RuntimeError("fewer than two eligible held-out ELM frames")

    records: dict[str, list[dict[str, Any]]] = {}
    sources: dict[str, str] = {}
    for arm, variants in RUN_DIRECTORIES.items():
        for variant, directory in variants.items():
            key = _run_key(arm, variant)
            checkpoint = checkpoint_root / directory / "final.pt"
            logger.info("scoring %s on %d ELM frames using CPU", key, len(windows))
            scored = morphology._evaluate_arm(checkpoint, windows, "cpu")
            run_records: list[dict[str, Any]] = []
            for selected, score in zip(windows, scored, strict=True):
                shot_id = int(selected.window.shot_id)
                run_records.append(
                    {
                        "shot_id": shot_id,
                        "frame_key": (
                            f"{shot_id}:{float(selected.dalpha_burst_time_s):.9f}"
                        ),
                        "actual_horizon_ms": float(selected.actual_horizon_ms),
                        "dalpha_burst_time_s": float(selected.dalpha_burst_time_s),
                        "token_nll": float(score["edge_divertor_nll"]),
                        "_predicted_tokens": score["_predicted_tokens"],
                    }
                )
            records[key] = run_records
            sources[key] = str(checkpoint)
            gc.collect()
    logger.info("decoding all predicted ELM frames with the frozen decoder on CPU")
    _decode_rescored_morphology(windows, records)
    return records, sources


def _metric_spec_sha256() -> str:
    source = inspect.getsource(elm_frame_morphology_fidelity)
    source += inspect.getsource(elm_edge_divertor_mask)
    return hashlib.sha256(source.encode()).hexdigest()


def write_figure(payload: dict[str, Any], path: Path = DEFAULT_FIGURE) -> Path:
    """Plot all six run means and both paired control intervals."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = [
        _run_key(arm, variant)
        for arm in RUN_DIRECTORIES
        for variant in RUN_DIRECTORIES[arm]
    ]
    labels = [key.replace("_", "\n", 1) for key in order]
    colours = {"native": "#167a72", "shuffled": "#c18c2f", "slow": "#777777"}
    bar_colours = [colours[payload["runs"][key]["conditioning"]] for key in order]

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2))
    for ax, metric, ylabel in (
        (axes[0, 0], "morphology_mean", "ELM morphology fidelity"),
        (axes[0, 1], "token_nll_mean", "edge/divertor token NLL"),
    ):
        values = [payload["runs"][key][metric] for key in order]
        ax.bar(np.arange(len(order)), values, color=bar_colours, width=0.72)
        ax.set_xticks(np.arange(len(order)), labels, fontsize=8)
        ax.set_ylabel(ylabel)
        ax.spines[["top", "right"]].set_visible(False)

    comparison_order = [
        (arm, comparison)
        for arm in RUN_DIRECTORIES
        for comparison in CONTROL_COMPARISONS
    ]
    comparison_labels = [
        f"{arm}\nvs {payload['arms'][arm]['comparisons'][comparison]['control']}"
        for arm, comparison in comparison_order
    ]
    for ax, metric, ylabel, good_direction in (
        (
            axes[1, 0],
            "morphology",
            "native − control morphology",
            "positive favours native",
        ),
        (
            axes[1, 1],
            "token_nll",
            "native − control token NLL",
            "negative favours native",
        ),
    ):
        intervals = [
            payload["arms"][arm]["comparisons"][comparison][metric]
            for arm, comparison in comparison_order
        ]
        means = np.array([row["mean"] for row in intervals])
        lo = np.array([row["lo"] for row in intervals])
        hi = np.array([row["hi"] for row in intervals])
        x = np.arange(len(intervals))
        ax.axhline(0.0, color="0.35", lw=0.8)
        ax.errorbar(
            x,
            means,
            yerr=np.vstack((means - lo, hi - means)),
            fmt="o",
            color="#284b63",
            capsize=4,
        )
        ax.set_xticks(x, comparison_labels, fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(good_direction)
        ax.spines[["top", "right"]].set_visible(False)

    count = payload["elm_frame_count"]
    shots = payload["shot_count"]
    fig.suptitle(
        f"Fast-Dalpha conditioning at 10 ms — {count} paired ELM frames, {shots} shots"
    )
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def verdict_line(payload: dict[str, Any]) -> str:
    """Return the required one-line strict conditioning verdict."""
    result = bool(payload["fast_dalpha_beats_both_controls"])
    prefix = "YES" if result else "NO"
    qualifier = "beat" if result else "did not beat"
    return (
        f"{prefix} — native fast-Dalpha {qualifier} both shuffled-Dalpha and slow-only "
        "controls with shot-clustered 95% intervals clear of zero in both ELM-frame "
        "morphology and token NLL across both arms "
        f"({payload['elm_frame_count']} paired "
        f"ELM frames from {payload['shot_count']} shots at 10 ms)."
    )


def census_report_line(payload: dict[str, Any]) -> str:
    """Place the widened verdict beside the earlier eight-frame result."""
    return (
        f"Widened census: {verdict_line(payload)} "
        "Earlier result: NO under the same strict gate at 8 paired ELM frames."
    )


def run(
    *,
    checkpoint_root: Path = CHECKPOINT_ROOT,
    split_path: Path = morphology.DEFAULT_SPLIT,
    artifact_path: Path = DEFAULT_ARTIFACT,
    figure_path: Path = DEFAULT_FIGURE,
    report_path: Path = DEFAULT_REPORT,
    max_candidates: int = REGISTERED_MAX_CANDIDATES,
    max_windows: int = REGISTERED_FRAME_COUNT,
    seed: int = 0,
) -> dict[str, Any]:
    """Load frame evidence or rescore the six checkpoints and write the verdict."""
    checkpoint_root = Path(checkpoint_root)
    windows = morphology.select_dalpha_windows(
        split_path=Path(split_path),
        max_candidates=max_candidates,
        max_windows=max_windows,
    )
    if len(windows) != max_windows:
        raise RuntimeError(
            f"registered census requires {max_windows} frames; found {len(windows)}"
        )
    registered_frames = registered_frame_rows(windows)
    registered_keys = {str(frame["frame_key"]) for frame in registered_frames}
    records, sources = _load_all_evaluation_records(checkpoint_root)
    evaluation_is_complete = records is not None and all(
        registered_keys.issubset(_frame_index(run_records))
        for run_records in records.values()
    )
    if not evaluation_is_complete:
        evidence_mode = "checkpoint_cpu_rescore"
        records, sources = _rescore_all_checkpoints(
            checkpoint_root,
            windows=windows,
        )
    else:
        evidence_mode = "evaluation_frame_records"

    census = build_census(registered_frames, records)
    paired_records = paired_census_records(records, census)
    if census["scored_frame_count"] < 2:
        raise RuntimeError("fewer than two six-way paired ELM frames in the census")
    verdict = build_verdict(paired_records, seed=seed)
    first_records = next(iter(paired_records.values()))
    indexed_records = {
        run_key: _frame_index(run_records)
        for run_key, run_records in paired_records.items()
    }
    reference_run = next(iter(paired_records))
    frame_keys = sorted(indexed_records[reference_run])
    payload: dict[str, Any] = {
        "task": "fast-Dalpha conditioning verdict on held-out ELM frames",
        "horizon_ms": ELM_MORPHOLOGY_HORIZON_MS,
        "evidence_mode": evidence_mode,
        "metric_provenance": {
            "morphology": "camdyn.metrics.elm_frame_morphology_fidelity",
            "token_nll": "bitwise NLL on camdyn.metrics.elm_edge_divertor_mask",
            "bootstrap": "camdyn.metrics.bootstrap_ci over paired shot means",
            "metric_spec_sha256": _metric_spec_sha256(),
            "delta_orientation": "native_minus_control",
            "win_directions": {
                "morphology": "positive",
                "token_nll": "negative",
            },
        },
        "census": census,
        "sources": sources,
        "elm_frame_count": len(first_records),
        "shot_count": len({int(row["shot_id"]) for row in first_records}),
        "per_frame": [
            {
                "shot_id": int(indexed_records[reference_run][frame_key]["shot_id"]),
                "frame_key": frame_key,
                "actual_horizon_ms": float(
                    indexed_records[reference_run][frame_key]["actual_horizon_ms"]
                ),
                "runs": {
                    run_key: {
                        "morphology_fidelity": float(
                            indexed_records[run_key][frame_key]["morphology_fidelity"]
                        ),
                        "token_nll": float(
                            indexed_records[run_key][frame_key]["token_nll"]
                        ),
                    }
                    for run_key in paired_records
                },
            }
            for frame_key in frame_keys
        ],
        **verdict,
    }

    artifact_path = Path(artifact_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_figure(payload, Path(figure_path))
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(census_report_line(payload) + "\n", encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, default=CHECKPOINT_ROOT)
    parser.add_argument("--split", type=Path, default=morphology.DEFAULT_SPLIT)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--max-candidates", type=int, default=REGISTERED_MAX_CANDIDATES)
    parser.add_argument("--max-windows", type=int, default=REGISTERED_FRAME_COUNT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    payload = run(
        checkpoint_root=args.checkpoint_root,
        split_path=args.split,
        artifact_path=args.artifact,
        figure_path=args.figure,
        report_path=args.report,
        max_candidates=args.max_candidates,
        max_windows=args.max_windows,
        seed=args.seed,
    )
    print(verdict_line(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
