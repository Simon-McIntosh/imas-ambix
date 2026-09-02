"""Robustness scoring for the independent rbb camera topology referee.

The scorer keeps three quantities separate in every result:

* the source-frame census fixed by the cohort authorities;
* the windows actually evaluated by this invocation; and
* whether the invocation is eligible to count as qualification evidence.

Natural emission confounders are scored on their registered clean-test frame
memberships.  Brightness gains are deterministic transforms of the clean
frames.  The held-out campaign remains a separate joint campaign, camera-pose,
and plasma-appearance term because no camera calibration is available.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as nnf

from imas_ambix.worldmodel.camera_topology_labeller import (
    DEFAULT_COHORT_REPORT,
    DEFAULT_FULLSHOT_MANIFEST,
    DEFAULT_LEVEL1_ROOT,
    REGRESSION_NAMES,
    CameraTopologyLabeller,
    LabellerConfig,
    LabellerTargets,
    LoadedWindows,
    TargetStatistics,
    assemble_labeller_targets,
    brightness_centroid_features,
    read_cohort_frame_counts,
    read_fullshot_spans,
)
from imas_ambix.worldmodel.camera_topology_targets import (
    TOPOLOGY_CLASS_NAMES,
    TOPOLOGY_UNDEFINED,
    CameraTopologyTargets,
    load_camera_topology_targets,
)
from imas_ambix.worldmodel.equilibrium_labels import (
    DEFAULT_LEVEL2_ROOT,
    EquilibriumGeometry,
    load_equilibrium_geometry,
)

DEFAULT_DIVERTOR_REPORT = Path(
    "/home/ITER/mcintos/.config/reckon/crew/reports/"
    "physics-carried-playable-plasma/divertor-bright-screen.md"
)
BRIGHTNESS_GAINS: tuple[float, ...] = (0.50, 0.75, 1.25, 1.50)
REGISTERED_SOURCE_FRAMES: dict[str, int] = {
    "clean": 240_281,
    "brightness": 240_281,
    "gas_puff": 124_138,
    "divertor_bright": 2_042,
    "held_out_campaign": 107_880,
}
MAX_POSITION_ERROR_CM = 3.0
MIN_CLASS_ACCURACY = 0.95
MAX_ERROR_GROWTH = 1.50


@dataclass(frozen=True)
class RobustnessAuthorities:
    """Registered frame memberships and counts from the two evidence reports."""

    clean_counts: dict[int, int]
    campaign_counts: dict[int, int]
    gas_puff_counts: dict[int, int]
    divertor_indices: dict[int, np.ndarray]

    @property
    def source_frames(self) -> dict[str, int]:
        clean = sum(self.clean_counts.values())
        return {
            "clean": clean,
            "brightness": clean,
            "gas_puff": sum(self.gas_puff_counts.values()),
            "divertor_bright": sum(
                int(indices.size) for indices in self.divertor_indices.values()
            ),
            "held_out_campaign": sum(self.campaign_counts.values()),
        }


@dataclass(frozen=True)
class LoadedCheckpoint:
    """A labeller and the training-only coordinate statistics it requires."""

    model: CameraTopologyLabeller
    statistics: TargetStatistics
    training_shots: tuple[int, ...]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class BrightnessCentroidBaseline:
    """Affine coordinate and class predictions from brightness-centroid features."""

    coordinate_coefficients: np.ndarray
    class_coefficients: np.ndarray

    def predict(self, frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        features = brightness_centroid_features(frames)
        coordinates = features @ self.coordinate_coefficients
        classes = np.argmax(features @ self.class_coefficients, axis=1)
        return coordinates, classes.astype(np.int64, copy=False)


def _report_section(text: str, heading: str) -> str:
    match = re.search(
        rf"^### {re.escape(heading)}\b.*?(?=^### |^## |\Z)",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    if match is None:
        raise ValueError(f"report is missing the {heading!r} section")
    return match.group()


def _shot_counts(section: str) -> dict[int, int]:
    entries = {
        int(shot): int(count)
        for shot, count in re.findall(r"\b(\d{5}):(\d+)\b", section)
    }
    if not entries:
        raise ValueError("report section contains no shot frame counts")
    return entries


def read_divertor_frame_indices(
    path: Path = DEFAULT_DIVERTOR_REPORT,
) -> dict[int, np.ndarray]:
    """Read every native-frame member from the expanded camera-only screen."""
    text = Path(path).read_text(encoding="utf-8")
    match = re.search(
        r"^## Per-shot native-frame membership\b.*?"
        r"(?=^## Coverage and consistency checks)",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    if match is None:
        raise ValueError("divertor report is missing native-frame membership")
    result: dict[int, np.ndarray] = {}
    for shot, declared, values in re.findall(
        r"^(\d{5}) \((\d+)\): (.+)$", match.group(), flags=re.MULTILINE
    ):
        count = int(declared)
        indices = (
            np.empty(0, dtype=np.int64)
            if values.strip() == "none"
            else np.fromstring(values, dtype=np.int64, sep=" ")
        )
        if indices.size != count:
            raise ValueError(
                f"divertor report declares {count} frames for shot {shot} "
                f"but lists {indices.size}"
            )
        if np.unique(indices).size != indices.size:
            raise ValueError(f"divertor report repeats a frame for shot {shot}")
        result[int(shot)] = indices
    if not result:
        raise ValueError("divertor report contains no per-shot memberships")
    return result


def read_robustness_authorities(
    cohort_report: Path = DEFAULT_COHORT_REPORT,
    divertor_report: Path = DEFAULT_DIVERTOR_REPORT,
) -> RobustnessAuthorities:
    """Resolve the four registered test views from their evidence authorities."""
    cohort_text = Path(cohort_report).read_text(encoding="utf-8")
    partitions = read_cohort_frame_counts(cohort_report)
    return RobustnessAuthorities(
        clean_counts=partitions["clean_test"],
        campaign_counts=partitions["campaign_test"],
        gas_puff_counts=_shot_counts(_report_section(cohort_text, "Gas-puff")),
        divertor_indices=read_divertor_frame_indices(divertor_report),
    )


def validate_registered_authorities(authorities: RobustnessAuthorities) -> None:
    """Refuse evidence reports whose measured counts differ from registration."""
    observed = authorities.source_frames
    if observed != REGISTERED_SOURCE_FRAMES:
        raise ValueError(
            f"robustness source-frame census is {observed}; "
            f"expected {REGISTERED_SOURCE_FRAMES}"
        )
    unknown_gas = set(authorities.gas_puff_counts) - set(authorities.clean_counts)
    unknown_divertor = set(authorities.divertor_indices) - set(authorities.clean_counts)
    if unknown_gas or unknown_divertor:
        raise ValueError(
            "confounder membership includes frames outside the clean partition: "
            f"gas={sorted(unknown_gas)}, divertor={sorted(unknown_divertor)}"
        )


def _checkpoint_training_shots(payload: dict[str, Any], path: Path) -> tuple[int, ...]:
    if payload.get("train_shots"):
        return tuple(int(shot) for shot in payload["train_shots"])
    companion = path.with_suffix(".json")
    if companion.is_file():
        report = json.loads(companion.read_text(encoding="utf-8"))
        if report.get("train_shots"):
            return tuple(int(shot) for shot in report["train_shots"])
    raise ValueError(
        "checkpoint has no training-shot identity and no companion JSON supplies it"
    )


def load_labeller_checkpoint(path: Path, *, device: str = "cpu") -> LoadedCheckpoint:
    """Load and validate a camera-topology labeller checkpoint."""
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("labeller checkpoint must contain a mapping")
    required = {
        "state_dict",
        "config",
        "target_mean_m",
        "target_std_m",
        "regression_names",
        "class_names",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"labeller checkpoint is missing {sorted(missing)}")
    if tuple(payload["regression_names"]) != REGRESSION_NAMES:
        raise ValueError("checkpoint regression coordinates do not match the scorer")
    if tuple(payload["class_names"]) != TOPOLOGY_CLASS_NAMES:
        raise ValueError("checkpoint topology classes do not match the scorer")
    config = LabellerConfig(**payload["config"])
    model = CameraTopologyLabeller(config)
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    statistics = TargetStatistics(
        mean_m=np.asarray(payload["target_mean_m"], dtype=np.float32),
        std_m=np.asarray(payload["target_std_m"], dtype=np.float32),
    )
    if statistics.mean_m.shape != (len(REGRESSION_NAMES),):
        raise ValueError("checkpoint target statistics have the wrong shape")
    if (
        not np.isfinite(statistics.mean_m).all()
        or not (np.isfinite(statistics.std_m).all() & (statistics.std_m > 0.0)).all()
    ):
        raise ValueError("checkpoint target statistics must be finite and positive")
    return LoadedCheckpoint(
        model=model,
        statistics=statistics,
        training_shots=_checkpoint_training_shots(payload, Path(path)),
        metadata=payload,
    )


def fit_brightness_centroid_baseline(
    windows: LoadedWindows,
) -> BrightnessCentroidBaseline:
    """Fit the brightness-only comparator without using any evaluation frame."""
    features = brightness_centroid_features(windows.frames)
    coordinate_coefficients = np.zeros(
        (features.shape[1], len(REGRESSION_NAMES)), dtype=np.float64
    )
    for column in range(len(REGRESSION_NAMES)):
        mask = windows.targets.finite_mask[:, column]
        if np.count_nonzero(mask) >= features.shape[1]:
            coordinate_coefficients[:, column] = np.linalg.lstsq(
                features[mask], windows.targets.values_m[mask, column], rcond=None
            )[0]
        elif np.any(mask):
            coordinate_coefficients[0, column] = float(
                np.mean(windows.targets.values_m[mask, column])
            )

    defined = windows.targets.topology_class != TOPOLOGY_UNDEFINED
    class_coefficients = np.zeros(
        (features.shape[1], len(TOPOLOGY_CLASS_NAMES)), dtype=np.float64
    )
    if np.count_nonzero(defined) >= features.shape[1]:
        one_hot = np.eye(len(TOPOLOGY_CLASS_NAMES))[
            windows.targets.topology_class[defined]
        ]
        class_coefficients = np.linalg.lstsq(features[defined], one_hot, rcond=None)[0]
    return BrightnessCentroidBaseline(
        coordinate_coefficients=coordinate_coefficients,
        class_coefficients=class_coefficients,
    )


class _MetricAccumulator:
    def __init__(self) -> None:
        self.distance_sum = {
            name: 0.0 for name in ("o_point", "x_point", "strike_point")
        }
        self.distance_count = {name: 0 for name in self.distance_sum}
        self.shot_distances: dict[str, dict[int, list[float]]] = {
            name: {} for name in self.distance_sum
        }
        self.class_correct = 0
        self.class_count = 0
        self.class_support = np.zeros(len(TOPOLOGY_CLASS_NAMES), dtype=np.int64)
        self.class_correct_by_class = np.zeros(
            len(TOPOLOGY_CLASS_NAMES), dtype=np.int64
        )

    def update(
        self,
        prediction_m: np.ndarray,
        predicted_class: np.ndarray,
        targets: LabellerTargets,
        shot_id: int,
    ) -> None:
        point_slices = {
            "o_point": ((0, 2),),
            "x_point": ((2, 4),),
            "strike_point": ((4, 6), (6, 8)),
        }
        for name, slices in point_slices.items():
            shot_values: list[float] = []
            for start, stop in slices:
                mask = targets.finite_mask[:, start:stop].all(axis=1)
                if not np.any(mask):
                    continue
                distance = 100.0 * np.linalg.norm(
                    prediction_m[mask, start:stop] - targets.values_m[mask, start:stop],
                    axis=1,
                )
                self.distance_sum[name] += float(distance.sum())
                self.distance_count[name] += int(distance.size)
                shot_values.extend(distance.tolist())
            if shot_values:
                self.shot_distances[name].setdefault(shot_id, []).extend(shot_values)

        defined = targets.topology_class != TOPOLOGY_UNDEFINED
        truth = targets.topology_class[defined]
        predicted = predicted_class[defined]
        self.class_correct += int(np.count_nonzero(predicted == truth))
        self.class_count += int(truth.size)
        for index in range(len(TOPOLOGY_CLASS_NAMES)):
            mask = truth == index
            self.class_support[index] += int(np.count_nonzero(mask))
            self.class_correct_by_class[index] += int(
                np.count_nonzero(predicted[mask] == index)
            )

    def result(self) -> dict[str, Any]:
        position_error = {
            name: (
                self.distance_sum[name] / self.distance_count[name]
                if self.distance_count[name]
                else None
            )
            for name in self.distance_sum
        }
        shot_clustered = {
            name: (
                float(
                    np.mean(
                        [
                            np.mean(values)
                            for values in self.shot_distances[name].values()
                        ]
                    )
                )
                if self.shot_distances[name]
                else None
            )
            for name in self.distance_sum
        }
        per_class = {}
        for index, name in enumerate(TOPOLOGY_CLASS_NAMES):
            support = int(self.class_support[index])
            per_class[name] = {
                "support": support,
                "accuracy": (
                    float(self.class_correct_by_class[index] / support)
                    if support
                    else None
                ),
            }
        return {
            "position_error_cm": position_error,
            "position_support": dict(self.distance_count),
            "shot_clustered_position_error_cm": shot_clustered,
            "class_accuracy": (
                float(self.class_correct / self.class_count)
                if self.class_count
                else None
            ),
            "class_support": self.class_count,
            "per_class": per_class,
        }


def score_predictions(
    prediction_m: np.ndarray,
    predicted_class: np.ndarray,
    targets: LabellerTargets,
    *,
    shot_id: int = 0,
) -> dict[str, Any]:
    """Score position, class accuracy, and explicit per-class support."""
    accumulator = _MetricAccumulator()
    accumulator.update(prediction_m, predicted_class, targets, shot_id)
    return accumulator.result()


def _growth_ratio(value: float | None, reference: float | None) -> float | None:
    if value is None or reference is None:
        return None
    if reference > 0.0:
        return float(value / reference)
    return 1.0 if value == 0.0 else float("inf")


def error_growth_verdict(
    clean: dict[str, Any],
    confounded: dict[str, Any],
    *,
    maximum: float = MAX_ERROR_GROWTH,
) -> dict[str, Any]:
    """Compare shot-clustered errors to clean errors without hiding no-support."""
    ratios = {
        name: _growth_ratio(
            confounded["shot_clustered_position_error_cm"][name],
            clean["shot_clustered_position_error_cm"][name],
        )
        for name in ("o_point", "x_point", "strike_point")
    }
    clean_accuracy = clean["class_accuracy"]
    confounded_accuracy = confounded["class_accuracy"]
    class_error_ratio = _growth_ratio(
        None if confounded_accuracy is None else 1.0 - confounded_accuracy,
        None if clean_accuracy is None else 1.0 - clean_accuracy,
    )
    ratios["class_error"] = class_error_ratio
    missing = [name for name, value in ratios.items() if value is None]
    return {
        "maximum_allowed": maximum,
        "ratios": ratios,
        "missing_measures": missing,
        "passed": not missing
        and all(
            value <= maximum or np.isclose(value, maximum) for value in ratios.values()
        ),
    }


def _label_support_indices(
    shot_id: int,
    frame_span: tuple[int, int],
    *,
    level1_root: Path,
    level2_root: Path,
) -> tuple[np.ndarray, np.ndarray]:
    import zarr  # noqa: PLC0415

    camera = zarr.open_group(str(Path(level1_root) / f"{shot_id}.zarr"), mode="r")[
        "rbb"
    ]
    frame_times = np.asarray(camera["time"], dtype=np.float64)
    equilibrium = zarr.open_group(str(Path(level2_root) / f"{shot_id}.zarr"), mode="r")[
        "equilibrium"
    ]
    equilibrium_times = np.asarray(equilibrium["time"], dtype=np.float64)
    axis_r = np.asarray(equilibrium["magnetic_axis_r"], dtype=np.float64)
    axis_z = np.asarray(equilibrium["magnetic_axis_z"], dtype=np.float64)
    finite = np.isfinite(equilibrium_times) & np.isfinite(axis_r) & np.isfinite(axis_z)
    if np.count_nonzero(finite) < 2:
        raise ValueError(f"shot {shot_id} has fewer than two finite axis labels")
    start, stop = frame_span
    candidates = np.arange(max(0, start), min(stop, frame_times.size), dtype=np.int64)
    times = frame_times[candidates]
    label_start = float(np.min(equilibrium_times[finite]))
    label_stop = float(np.max(equilibrium_times[finite]))
    keep = (times >= label_start) & (times <= label_stop)
    return candidates[keep], frame_times


def _gas_puff_indices(
    shot_id: int,
    clean_indices: np.ndarray,
    frame_times: np.ndarray,
    *,
    level1_root: Path,
) -> np.ndarray:
    import zarr  # noqa: PLC0415

    store = zarr.open_group(str(Path(level1_root) / f"{shot_id}.zarr"), mode="r")
    if "aga" not in store:
        raise ValueError(f"shot {shot_id} has no readable gas-puff group")
    gas = store["aga"]
    gas_times = np.asarray(gas["time"], dtype=np.float64)
    gas_value = np.asarray(gas["inboard_total"], dtype=np.float64)
    finite = np.isfinite(gas_times) & np.isfinite(gas_value)
    gas_times = gas_times[finite]
    gas_value = gas_value[finite]
    order = np.argsort(gas_times)
    gas_times = gas_times[order]
    gas_value = gas_value[order]
    positive = gas_value[gas_value > 0.0]
    if gas_times.size == 0 or positive.size == 0:
        return np.empty(0, dtype=np.int64)
    threshold = max(1.0e20, 0.10 * float(np.percentile(positive, 99.0)))
    held = np.searchsorted(gas_times, frame_times[clean_indices], side="right") - 1
    valid = held >= 0
    active = np.zeros(clean_indices.size, dtype=bool)
    active[valid] = gas_value[held[valid]] >= threshold
    return clean_indices[active]


def _select_evenly(indices: np.ndarray, count: int) -> np.ndarray:
    if count <= 0 or indices.size <= count:
        return indices
    positions = np.linspace(0, indices.size - 1, count).round().astype(np.int64)
    return indices[positions]


def _load_windows(
    shot_id: int,
    indices: np.ndarray,
    config: LabellerConfig,
    *,
    level1_root: Path,
    level2_root: Path,
) -> LoadedWindows:
    import zarr  # noqa: PLC0415

    camera = zarr.open_group(str(Path(level1_root) / f"{shot_id}.zarr"), mode="r")[
        "rbb"
    ]
    frame_times = np.asarray(camera["time"], dtype=np.float64)
    offsets = np.arange(
        -(config.window_frames // 2), config.window_frames // 2 + 1, dtype=np.int64
    )
    window_indices = np.clip(
        indices[:, None] + offsets[None, :], 0, frame_times.size - 1
    )
    raw = np.asarray(
        camera["data"].oindex[window_indices.ravel(), :, :], dtype=np.float32
    )
    raw = raw.reshape(indices.size, config.window_frames, raw.shape[-2], raw.shape[-1])
    frames = nnf.interpolate(
        torch.from_numpy(raw / 255.0),
        size=(config.image_size, config.image_size),
        mode="bilinear",
        align_corners=False,
    ).numpy()
    selected_times = frame_times[indices]
    try:
        geometry_all = load_equilibrium_geometry(
            shot_id, selected_times, level2_root=level2_root
        )
    except KeyError:
        equilibrium = zarr.open_group(
            str(Path(level2_root) / f"{shot_id}.zarr"), mode="r"
        )["equilibrium"]
        equilibrium_times = np.asarray(equilibrium["time"], dtype=np.float64)
        axis_r = np.asarray(equilibrium["magnetic_axis_r"], dtype=np.float64)
        axis_z = np.asarray(equilibrium["magnetic_axis_z"], dtype=np.float64)
        target = np.full((selected_times.size, 14), np.nan, dtype=np.float32)
        for column, values in enumerate((axis_r, axis_z)):
            finite = np.isfinite(equilibrium_times) & np.isfinite(values)
            if np.count_nonzero(finite) >= 2:
                order = np.argsort(equilibrium_times[finite])
                times = equilibrium_times[finite][order]
                ordered_values = values[finite][order]
                in_range = (selected_times >= times[0]) & (selected_times <= times[-1])
                target[in_range, column] = np.interp(
                    selected_times[in_range], times, ordered_values
                )
        geometry_all = EquilibriumGeometry(
            shot_id=shot_id,
            frame_times=selected_times,
            target=target,
            finite_mask=np.isfinite(target),
        )
    geometry = EquilibriumGeometry(
        shot_id=geometry_all.shot_id,
        frame_times=selected_times,
        target=geometry_all.target,
        finite_mask=geometry_all.finite_mask,
        names=geometry_all.names,
        units=geometry_all.units,
    )
    try:
        topology = load_camera_topology_targets(
            shot_id, selected_times, level2_root=level2_root
        )
    except KeyError:
        topology = CameraTopologyTargets(
            shot_id=shot_id,
            frame_times=selected_times,
            primary_xpoint=np.full((selected_times.size, 2), np.nan, dtype=np.float32),
            primary_xpoint_mask=np.zeros(selected_times.size, dtype=bool),
            strike_points=np.full(
                (selected_times.size, 2, 2), np.nan, dtype=np.float32
            ),
            strike_point_mask=np.zeros((selected_times.size, 2), dtype=bool),
            topology_class=np.full(
                selected_times.size, TOPOLOGY_UNDEFINED, dtype=np.int8
            ),
            boundary_psi=np.full(selected_times.size, np.nan, dtype=np.float32),
            boundary_flux_mask=np.zeros(selected_times.size, dtype=bool),
        )
    return LoadedWindows(
        frames=frames.astype(np.float32, copy=False),
        targets=assemble_labeller_targets(geometry, topology),
        shot_ids=np.full(indices.size, shot_id, dtype=np.int64),
        frame_times=selected_times,
    )


def _predict_labeller(
    checkpoint: LoadedCheckpoint,
    frames: np.ndarray,
    *,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        coordinates, logits = checkpoint.model(
            torch.as_tensor(frames, dtype=torch.float32, device=device)
        )
    standard = coordinates.detach().cpu().numpy()
    prediction = standard * checkpoint.statistics.std_m + checkpoint.statistics.mean_m
    classes = logits.argmax(dim=1).detach().cpu().numpy()
    return prediction, classes


def _frame_digest(entries: list[tuple[int, int, float]]) -> str:
    digest = hashlib.sha256()
    for shot, index, gain in entries:
        digest.update(f"{shot}:{index}:{gain:.6f}\n".encode())
    return f"sha256:{digest.hexdigest()}"


def _score_membership(
    name: str,
    counts: dict[int, int],
    checkpoint: LoadedCheckpoint,
    baseline: BrightnessCentroidBaseline,
    frame_spans: dict[int, tuple[int, int]],
    *,
    config: LabellerConfig,
    level1_root: Path,
    level2_root: Path,
    device: str,
    max_shots: int,
    windows_per_shot: int,
    gains: tuple[float, ...] = (1.0,),
    gas_puff: bool = False,
    explicit_indices: dict[int, np.ndarray] | None = None,
) -> dict[str, Any]:
    model_metrics = _MetricAccumulator()
    baseline_metrics = _MetricAccumulator()
    condition_metrics = {gain: _MetricAccumulator() for gain in gains}
    selected_shots = list(counts)
    if max_shots > 0:
        selected_shots = selected_shots[:max_shots]
    scored_frames = 0
    resolved_selected_frames = 0
    identities: list[tuple[int, int, float]] = []
    for shot in selected_shots:
        if shot not in frame_spans:
            raise ValueError(f"full-shot manifest is missing shot {shot}")
        clean_indices, frame_times = _label_support_indices(
            shot,
            frame_spans[shot],
            level1_root=level1_root,
            level2_root=level2_root,
        )
        if explicit_indices is not None:
            indices = explicit_indices[shot]
            if not np.isin(indices, clean_indices).all():
                raise ValueError(
                    f"{name} contains out-of-support frames for shot {shot}"
                )
        elif gas_puff:
            indices = _gas_puff_indices(
                shot,
                clean_indices,
                frame_times,
                level1_root=level1_root,
            )
        else:
            indices = clean_indices
        if indices.size != counts[shot]:
            raise ValueError(
                f"{name} resolves {indices.size} frames for shot {shot}; "
                f"the authority declares {counts[shot]}"
            )
        resolved_selected_frames += int(indices.size)
        indices = _select_evenly(indices, windows_per_shot)
        if indices.size == 0:
            continue
        loaded = _load_windows(
            shot,
            indices,
            config,
            level1_root=level1_root,
            level2_root=level2_root,
        )
        scored_frames += int(indices.size)
        for gain in gains:
            frames = np.clip(loaded.frames * gain, 0.0, 1.0)
            model_prediction, model_class = _predict_labeller(
                checkpoint, frames, device=device
            )
            baseline_prediction, baseline_class = baseline.predict(frames)
            model_metrics.update(model_prediction, model_class, loaded.targets, shot)
            baseline_metrics.update(
                baseline_prediction, baseline_class, loaded.targets, shot
            )
            condition_metrics[gain].update(
                model_prediction, model_class, loaded.targets, shot
            )
            identities.extend((shot, int(index), gain) for index in indices)
    return {
        "name": name,
        "authority_source_frames": int(sum(counts.values())),
        "authority_shots": len(counts),
        "resolved_selected_source_frames": resolved_selected_frames,
        "scored_source_frames": scored_frames,
        "evaluation_instances": scored_frames * len(gains),
        "scored_shots": len(selected_shots),
        "frame_identity_digest": _frame_digest(identities),
        "identical_frames_for_labeller_and_baseline": True,
        "labeller": model_metrics.result(),
        "brightness_centroid_baseline": baseline_metrics.result(),
        "conditions": {
            f"gain_{gain:.2f}": metrics.result()
            for gain, metrics in condition_metrics.items()
        },
    }


def _fit_baseline_from_checkpoint_split(
    checkpoint: LoadedCheckpoint,
    frame_spans: dict[int, tuple[int, int]],
    *,
    config: LabellerConfig,
    level1_root: Path,
    level2_root: Path,
    max_shots: int,
    windows_per_shot: int,
) -> tuple[BrightnessCentroidBaseline, dict[str, int]]:
    shots = list(checkpoint.training_shots)
    if max_shots > 0:
        shots = shots[:max_shots]
    loaded: list[LoadedWindows] = []
    for shot in shots:
        if shot not in frame_spans:
            raise ValueError(f"full-shot manifest is missing baseline-fit shot {shot}")
        indices, _ = _label_support_indices(
            shot,
            frame_spans[shot],
            level1_root=level1_root,
            level2_root=level2_root,
        )
        indices = _select_evenly(indices, windows_per_shot)
        loaded.append(
            _load_windows(
                shot,
                indices,
                config,
                level1_root=level1_root,
                level2_root=level2_root,
            )
        )
    if not loaded:
        raise ValueError("checkpoint training split supplies no baseline-fit windows")
    joined = LoadedWindows(
        frames=np.concatenate([item.frames for item in loaded]),
        targets=LabellerTargets(
            values_m=np.concatenate([item.targets.values_m for item in loaded]),
            finite_mask=np.concatenate([item.targets.finite_mask for item in loaded]),
            topology_class=np.concatenate(
                [item.targets.topology_class for item in loaded]
            ),
        ),
        shot_ids=np.concatenate([item.shot_ids for item in loaded]),
        frame_times=np.concatenate([item.frame_times for item in loaded]),
    )
    return fit_brightness_centroid_baseline(joined), {
        "shots": len(shots),
        "windows": int(joined.frames.shape[0]),
    }


def _clean_verdict(metrics: dict[str, Any]) -> dict[str, Any]:
    position = metrics["position_error_cm"]
    checks = {
        "o_point": position["o_point"] is not None
        and position["o_point"] <= MAX_POSITION_ERROR_CM,
        "x_point": position["x_point"] is not None
        and position["x_point"] <= MAX_POSITION_ERROR_CM,
        "class_accuracy": metrics["class_accuracy"] is not None
        and metrics["class_accuracy"] >= MIN_CLASS_ACCURACY,
    }
    return {
        "thresholds": {
            "maximum_o_point_error_cm": MAX_POSITION_ERROR_CM,
            "maximum_x_point_error_cm": MAX_POSITION_ERROR_CM,
            "minimum_class_accuracy": MIN_CLASS_ACCURACY,
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def _checkpoint_is_full_corpus(checkpoint: LoadedCheckpoint) -> bool:
    validation_shots = checkpoint.metadata.get("validation_shots", ())
    return (
        len(checkpoint.training_shots) == 472
        and len(validation_shots) == 58
        and isinstance(checkpoint.metadata.get("epoch"), int)
    )


def run_robustness_scoring(args: argparse.Namespace) -> dict[str, Any]:
    """Run the registered robustness views and return one JSON-ready receipt."""
    authorities = read_robustness_authorities(args.cohort_report, args.divertor_report)
    validate_registered_authorities(authorities)
    checkpoint = load_labeller_checkpoint(args.checkpoint, device=args.device)
    config = checkpoint.model.config
    frame_spans = read_fullshot_spans(args.fullshot_manifest)
    baseline, baseline_fit = _fit_baseline_from_checkpoint_split(
        checkpoint,
        frame_spans,
        config=config,
        level1_root=args.level1_root,
        level2_root=args.level2_root,
        max_shots=args.baseline_max_shots,
        windows_per_shot=args.baseline_windows_per_shot,
    )

    clean = _score_membership(
        "clean",
        authorities.clean_counts,
        checkpoint,
        baseline,
        frame_spans,
        config=config,
        level1_root=args.level1_root,
        level2_root=args.level2_root,
        device=args.device,
        max_shots=args.max_shots_per_subset,
        windows_per_shot=args.windows_per_shot,
    )
    brightness = _score_membership(
        "brightness",
        authorities.clean_counts,
        checkpoint,
        baseline,
        frame_spans,
        config=config,
        level1_root=args.level1_root,
        level2_root=args.level2_root,
        device=args.device,
        max_shots=args.max_shots_per_subset,
        windows_per_shot=args.windows_per_shot,
        gains=BRIGHTNESS_GAINS,
    )
    gas_puff = _score_membership(
        "gas_puff",
        authorities.gas_puff_counts,
        checkpoint,
        baseline,
        frame_spans,
        config=config,
        level1_root=args.level1_root,
        level2_root=args.level2_root,
        device=args.device,
        max_shots=args.max_shots_per_subset,
        windows_per_shot=args.windows_per_shot,
        gas_puff=True,
    )
    divertor_counts = {
        shot: int(indices.size)
        for shot, indices in authorities.divertor_indices.items()
        if indices.size
    }
    divertor_bright = _score_membership(
        "divertor_bright",
        divertor_counts,
        checkpoint,
        baseline,
        frame_spans,
        config=config,
        level1_root=args.level1_root,
        level2_root=args.level2_root,
        device=args.device,
        max_shots=args.max_shots_per_subset,
        windows_per_shot=args.windows_per_shot,
        explicit_indices=authorities.divertor_indices,
    )
    campaign = _score_membership(
        "held_out_campaign",
        authorities.campaign_counts,
        checkpoint,
        baseline,
        frame_spans,
        config=config,
        level1_root=args.level1_root,
        level2_root=args.level2_root,
        device=args.device,
        max_shots=args.max_shots_per_subset,
        windows_per_shot=args.windows_per_shot,
    )

    emission = {
        "brightness": brightness,
        "gas_puff": gas_puff,
        "divertor_bright": divertor_bright,
    }
    for subset in emission.values():
        subset["labeller_error_growth"] = error_growth_verdict(
            clean["labeller"], subset["labeller"]
        )
        subset["brightness_centroid_error_growth"] = error_growth_verdict(
            clean["brightness_centroid_baseline"],
            subset["brightness_centroid_baseline"],
        )
    campaign["interpretation"] = (
        "joint campaign + camera-pose + plasma-appearance shift; reported "
        "separately from emission confounders"
    )
    campaign["labeller_error_growth"] = error_growth_verdict(
        clean["labeller"], campaign["labeller"]
    )
    campaign["brightness_centroid_error_growth"] = error_growth_verdict(
        clean["brightness_centroid_baseline"],
        campaign["brightness_centroid_baseline"],
    )

    complete = all(
        subset["scored_source_frames"] == subset["authority_source_frames"]
        for subset in [clean, *emission.values(), campaign]
    )
    numerical_pass = _clean_verdict(clean["labeller"])["passed"] and all(
        subset["labeller_error_growth"]["passed"]
        for subset in [*emission.values(), campaign]
    )
    checkpoint_is_full_corpus = _checkpoint_is_full_corpus(checkpoint)
    eligible = args.evidence_scale == "gate" and complete and checkpoint_is_full_corpus
    report: dict[str, Any] = {
        "evidence_scale": args.evidence_scale,
        "checkpoint": str(args.checkpoint),
        "checkpoint_model_parameters": checkpoint.model.n_parameters(),
        "checkpoint_is_full_corpus": checkpoint_is_full_corpus,
        "source_frame_authority": authorities.source_frames,
        "brightness_evaluation_instances": (
            authorities.source_frames["brightness"] * len(BRIGHTNESS_GAINS)
        ),
        "baseline_fit": baseline_fit,
        "clean": clean,
        "clean_threshold_verdict": _clean_verdict(clean["labeller"]),
        "emission_confounders": emission,
        "held_out_campaign": campaign,
        "qualification": {
            "complete_source_census_scored": complete,
            "eligible_as_gate_evidence": eligible,
            "numerical_thresholds_passed": numerical_pass,
            "verdict": (
                "pass"
                if eligible and numerical_pass
                else "fail"
                if eligible
                else "smoke-scale-not-a-gate-result"
                if args.evidence_scale == "smoke-scale"
                else "ineligible-gate-evidence"
            ),
        },
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cohort-report", type=Path, default=DEFAULT_COHORT_REPORT)
    parser.add_argument("--divertor-report", type=Path, default=DEFAULT_DIVERTOR_REPORT)
    parser.add_argument(
        "--fullshot-manifest", type=Path, default=DEFAULT_FULLSHOT_MANIFEST
    )
    parser.add_argument("--level1-root", type=Path, default=DEFAULT_LEVEL1_ROOT)
    parser.add_argument("--level2-root", type=Path, default=DEFAULT_LEVEL2_ROOT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--windows-per-shot", type=int, default=0)
    parser.add_argument("--max-shots-per-subset", type=int, default=0)
    parser.add_argument("--baseline-windows-per-shot", type=int, default=24)
    parser.add_argument("--baseline-max-shots", type=int, default=0)
    parser.add_argument(
        "--evidence-scale", choices=("smoke-scale", "gate"), default="smoke-scale"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = run_robustness_scoring(args)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "BRIGHTNESS_GAINS",
    "DEFAULT_DIVERTOR_REPORT",
    "MAX_ERROR_GROWTH",
    "REGISTERED_SOURCE_FRAMES",
    "BrightnessCentroidBaseline",
    "LoadedCheckpoint",
    "RobustnessAuthorities",
    "error_growth_verdict",
    "fit_brightness_centroid_baseline",
    "load_labeller_checkpoint",
    "main",
    "read_divertor_frame_indices",
    "read_robustness_authorities",
    "run_robustness_scoring",
    "score_predictions",
    "validate_registered_authorities",
]
