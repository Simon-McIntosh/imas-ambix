"""Independent rbb-camera referee for machine-coordinate plasma topology.

The labeller consumes a short window of raw ``rbb`` frames and predicts eight
metre-valued coordinates: magnetic axis, primary null, and radially ordered
inner/outer strike points.  A separate head predicts the topology class.  The
axis comes from :mod:`equilibrium_labels`; the remaining labels and all of
their absence masks come directly from :mod:`camera_topology_targets`.

The command-line entry point reads the shot-level cohort authority, fits only
cohort-train shots, evaluates disjoint held-out shots, and reports the learned
model beside an affine brightness-centroid baseline on exactly the same frames.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as nnf

from imas_ambix.worldmodel.camera_topology_targets import (
    TOPOLOGY_CLASS_NAMES,
    TOPOLOGY_UNDEFINED,
    CameraTopologyTargets,
    load_camera_topology_targets,
)
from imas_ambix.worldmodel.equilibrium_labels import (
    EquilibriumGeometry,
    load_equilibrium_geometry,
)

DEFAULT_LEVEL1_ROOT = Path("/work/projects/imas_gpu/mast/level1/shots")
DEFAULT_COHORT_REPORT = Path(
    "/home/ITER/mcintos/.config/reckon/crew/reports/"
    "physics-carried-playable-plasma/labeller-cohort-census.md"
)
DEFAULT_FULLSHOT_MANIFEST = Path(
    "/work/projects/imas_gpu/agents/excitation-corpus/curated_windows_fullshot.json"
)
FULL_CORPUS_COUNTS: dict[str, tuple[int, int]] = {
    "train": (472, 1_163_812),
    "validation": (58, 125_930),
}
REGRESSION_NAMES: tuple[str, ...] = (
    "axis_R",
    "axis_Z",
    "primary_null_R",
    "primary_null_Z",
    "inner_strike_R",
    "inner_strike_Z",
    "outer_strike_R",
    "outer_strike_Z",
)


@dataclass(frozen=True)
class LabellerConfig:
    """Architecture for the short-window camera topology labeller."""

    window_frames: int = 3
    image_size: int = 48
    width: int = 12
    hidden: int = 96
    n_classes: int = len(TOPOLOGY_CLASS_NAMES)


@dataclass(frozen=True)
class LabellerTargets:
    """Masked coordinate and class targets aligned to camera windows."""

    values_m: np.ndarray
    finite_mask: np.ndarray
    topology_class: np.ndarray


@dataclass(frozen=True)
class TargetStatistics:
    """Per-coordinate training statistics used to condition regression."""

    mean_m: np.ndarray
    std_m: np.ndarray


@dataclass(frozen=True)
class LoadedWindows:
    """In-memory windows and their source identity."""

    frames: np.ndarray
    targets: LabellerTargets
    shot_ids: np.ndarray
    frame_times: np.ndarray


class _ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        groups = max(1, min(4, out_channels // 4))
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class CameraTopologyLabeller(nn.Module):
    """Small CNN with masked coordinate regression and topology heads."""

    def __init__(self, config: LabellerConfig | None = None) -> None:
        super().__init__()
        self.config = config or LabellerConfig()
        cfg = self.config
        self.features = nn.Sequential(
            _ConvBlock(cfg.window_frames, cfg.width),
            _ConvBlock(cfg.width, 2 * cfg.width),
            _ConvBlock(2 * cfg.width, 4 * cfg.width),
            _ConvBlock(4 * cfg.width, 4 * cfg.width),
            nn.AdaptiveAvgPool2d(1),
        )
        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4 * cfg.width, cfg.hidden),
            nn.SiLU(),
        )
        self.coordinate_head = nn.Linear(cfg.hidden, len(REGRESSION_NAMES))
        self.class_head = nn.Linear(cfg.hidden, cfg.n_classes)

    def forward(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.shared(self.features(frames))
        return self.coordinate_head(hidden), self.class_head(hidden)

    def n_parameters(self) -> int:
        return int(sum(parameter.numel() for parameter in self.parameters()))


def assemble_labeller_targets(
    geometry: EquilibriumGeometry,
    topology: CameraTopologyTargets,
) -> LabellerTargets:
    """Join axis coordinates to topology-derived labels on one frame axis."""
    if geometry.shot_id != topology.shot_id:
        raise ValueError("geometry and topology must describe the same shot")
    if not np.array_equal(geometry.frame_times, topology.frame_times):
        raise ValueError("geometry and topology frame times must be identical")
    axis = np.asarray(geometry.target[:, :2], dtype=np.float32)
    values = np.concatenate(
        [
            axis,
            np.asarray(topology.primary_xpoint, dtype=np.float32),
            np.asarray(topology.strike_points, dtype=np.float32).reshape(-1, 4),
        ],
        axis=1,
    )
    mask = np.concatenate(
        [
            np.asarray(geometry.finite_mask[:, :2], dtype=bool),
            np.repeat(topology.primary_xpoint_mask[:, None], 2, axis=1),
            np.repeat(topology.strike_point_mask[:, :, None], 2, axis=2).reshape(-1, 4),
        ],
        axis=1,
    )
    values = np.where(mask, values, np.nan)
    return LabellerTargets(
        values_m=values,
        finite_mask=mask,
        topology_class=np.asarray(topology.topology_class, dtype=np.int64),
    )


def read_cohort_frame_counts(
    path: Path = DEFAULT_COHORT_REPORT,
) -> dict[str, dict[int, int]]:
    """Read ordered shot-to-frame counts for all cohort partitions."""
    text = Path(path).read_text(encoding="utf-8")
    headings = {
        "train": "Labeller train",
        "validation": "Labeller validation",
        "clean_test": "Clean same-campaign test",
        "campaign_test": "Held-out campaign test",
    }
    result: dict[str, dict[int, int]] = {}
    for key, heading in headings.items():
        match = re.search(
            rf"^### {re.escape(heading)}\b.*?(?=^### |\Z)",
            text,
            flags=re.MULTILINE | re.DOTALL,
        )
        if match is None:
            raise ValueError(f"cohort report is missing the {heading!r} section")
        entries = [
            (int(shot), int(count))
            for shot, count in re.findall(r"\b(\d{5}):(\d+)\b", match.group())
        ]
        if not entries:
            raise ValueError(f"cohort report contains no shots in {heading!r}")
        shots = [shot for shot, _ in entries]
        if len(shots) != len(set(shots)):
            raise ValueError(f"cohort report repeats a shot in {heading!r}")
        result[key] = dict(entries)
    owners: dict[int, str] = {}
    for partition, entries in result.items():
        for shot in entries:
            if shot in owners:
                raise ValueError(
                    f"shot {shot} appears in both {owners[shot]} and {partition}"
                )
            owners[shot] = partition
    return result


def read_cohort_split(path: Path = DEFAULT_COHORT_REPORT) -> dict[str, list[int]]:
    """Read the four whole-shot partitions from the cohort census report."""
    return {
        partition: list(entries)
        for partition, entries in read_cohort_frame_counts(path).items()
    }


def read_fullshot_spans(
    path: Path = DEFAULT_FULLSHOT_MANIFEST,
) -> dict[int, tuple[int, int]]:
    """Read the curated plasma-phase frame span for every available shot."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    spans: dict[int, tuple[int, int]] = {}
    for row in payload.get("windows", []):
        shot = int(row["shot_id"])
        start = int(row["start_frame"])
        end = int(row["end_frame"])
        if shot in spans:
            raise ValueError(f"full-shot manifest repeats shot {shot}")
        if end <= start:
            raise ValueError(f"full-shot manifest has an empty span for shot {shot}")
        spans[shot] = (start, end)
    if not spans:
        raise ValueError("full-shot manifest contains no windows")
    return spans


def validate_full_corpus_split(
    frame_counts: dict[str, dict[int, int]],
) -> None:
    """Refuse a cohort report that differs from the registered training census."""
    for partition, (expected_shots, expected_frames) in FULL_CORPUS_COUNTS.items():
        entries = frame_counts[partition]
        observed = (len(entries), sum(entries.values()))
        expected = (expected_shots, expected_frames)
        if observed != expected:
            raise ValueError(
                f"{partition} cohort is {observed[0]} shots/{observed[1]} frames; "
                f"expected {expected[0]} shots/{expected[1]} frames"
            )


def _select_evenly(indices: np.ndarray, count: int) -> np.ndarray:
    if count <= 0:
        return indices
    if indices.size <= count:
        return indices
    positions = np.linspace(0, indices.size - 1, count).round().astype(np.int64)
    return indices[positions]


def _select_one_shot_targets(
    shot_id: int,
    *,
    window_frames: int,
    windows_per_shot: int,
    level1_root: Path,
    frame_span: tuple[int, int] | None = None,
) -> tuple[Any, np.ndarray, np.ndarray, LabellerTargets]:
    """Select complete window centres and derive their compact targets."""
    import zarr  # noqa: PLC0415

    if window_frames < 1 or window_frames % 2 != 1:
        raise ValueError("window_frames must be a positive odd number")
    group = zarr.open_group(str(level1_root / f"{shot_id}.zarr"), mode="r")["rbb"]
    frame_times = np.asarray(group["time"], dtype=np.float64)
    geometry_all = load_equilibrium_geometry(shot_id, frame_times)
    half = window_frames // 2
    candidates = np.flatnonzero(geometry_all.finite_mask[:, :2].all(axis=1))
    if frame_span is not None:
        start, end = frame_span
        candidates = candidates[(candidates >= start) & (candidates < end)]
    candidates = candidates[
        (candidates >= half) & (candidates < frame_times.size - half)
    ]
    centers = _select_evenly(candidates, windows_per_shot)
    if centers.size == 0:
        raise ValueError(f"shot {shot_id} has no label-bearing complete frame window")

    selected_times = frame_times[centers]
    geometry = EquilibriumGeometry(
        shot_id=geometry_all.shot_id,
        frame_times=selected_times,
        target=geometry_all.target[centers],
        finite_mask=geometry_all.finite_mask[centers],
        names=geometry_all.names,
        units=geometry_all.units,
    )
    topology = load_camera_topology_targets(shot_id, selected_times)
    targets = assemble_labeller_targets(geometry, topology)
    return group, centers, selected_times, targets


def _load_one_shot(
    shot_id: int,
    *,
    window_frames: int,
    image_size: int,
    windows_per_shot: int,
    level1_root: Path,
    frame_span: tuple[int, int] | None = None,
) -> LoadedWindows:
    group, centers, selected_times, targets = _select_one_shot_targets(
        shot_id,
        window_frames=window_frames,
        windows_per_shot=windows_per_shot,
        level1_root=level1_root,
        frame_span=frame_span,
    )

    half = window_frames // 2
    offsets = np.arange(-half, half + 1, dtype=np.int64)
    window_indices = centers[:, None] + offsets[None, :]
    frames = np.empty(
        (centers.size, window_frames, image_size, image_size), dtype=np.float32
    )
    read_batch_size = 2048
    for start in range(0, centers.size, read_batch_size):
        stop = min(start + read_batch_size, centers.size)
        flat_indices = window_indices[start:stop].ravel()
        raw = np.asarray(group["data"].oindex[flat_indices, :, :], dtype=np.float32)
        raw = raw.reshape(stop - start, window_frames, raw.shape[-2], raw.shape[-1])
        resized = nnf.interpolate(
            torch.from_numpy(raw / 255.0),
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        )
        frames[start:stop] = resized.numpy()

    return LoadedWindows(
        frames=frames.astype(np.float32, copy=False),
        targets=targets,
        shot_ids=np.full(centers.size, shot_id, dtype=np.int64),
        frame_times=selected_times,
    )


def load_camera_windows(
    shot_ids: list[int],
    *,
    window_frames: int,
    image_size: int,
    windows_per_shot: int,
    level1_root: Path = DEFAULT_LEVEL1_ROOT,
    frame_spans: dict[int, tuple[int, int]] | None = None,
) -> LoadedWindows:
    """Load evenly sampled labelled windows from each named whole shot."""
    loaded = [
        _load_one_shot(
            shot,
            window_frames=window_frames,
            image_size=image_size,
            windows_per_shot=windows_per_shot,
            level1_root=level1_root,
            frame_span=None if frame_spans is None else frame_spans.get(shot),
        )
        for shot in shot_ids
    ]
    return LoadedWindows(
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


def target_statistics(targets: LabellerTargets) -> TargetStatistics:
    """Compute finite-only coordinate statistics from training targets."""
    values = np.asarray(targets.values_m, dtype=np.float64)
    masked = np.where(targets.finite_mask, values, np.nan)
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(masked, axis=0)
        std = np.nanstd(masked, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    std = np.where(np.isfinite(std) & (std >= 1.0e-3), std, 1.0)
    return TargetStatistics(
        mean_m=mean.astype(np.float32), std_m=std.astype(np.float32)
    )


def labeller_loss(
    coordinate_prediction: torch.Tensor,
    class_logits: torch.Tensor,
    target: torch.Tensor,
    finite_mask: torch.Tensor,
    topology_class: torch.Tensor,
    *,
    class_weight: float = 0.2,
) -> torch.Tensor:
    """Masked smooth-L1 coordinates plus defined-frame class cross entropy."""
    mask = finite_mask.to(coordinate_prediction.dtype)
    coordinate = nnf.smooth_l1_loss(coordinate_prediction, target, reduction="none")
    coordinate_loss = (coordinate * mask).sum() / mask.sum().clamp_min(1.0)
    class_mask = topology_class != TOPOLOGY_UNDEFINED
    if class_mask.any():
        class_loss = nnf.cross_entropy(
            class_logits[class_mask], topology_class[class_mask]
        )
    else:
        class_loss = class_logits.sum() * 0.0
    return coordinate_loss + class_weight * class_loss


def fit_labeller(
    model: CameraTopologyLabeller,
    frames: np.ndarray,
    targets: LabellerTargets,
    *,
    steps: int = 60,
    learning_rate: float = 3.0e-3,
    seed: int = 0,
    device: str = "cpu",
    cpu_threads: int = 1,
) -> tuple[TargetStatistics, list[float]]:
    """Fit a smoke-sized full batch and return statistics plus loss trace."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    previous_threads = torch.get_num_threads()
    if device == "cpu":
        torch.set_num_threads(cpu_threads)
    model.to(device)
    model.train()
    stats = target_statistics(targets)
    safe_values = np.where(targets.finite_mask, targets.values_m, stats.mean_m)
    normalised = (safe_values - stats.mean_m) / stats.std_m
    x = torch.as_tensor(frames, dtype=torch.float32, device=device)
    y = torch.as_tensor(normalised, dtype=torch.float32, device=device)
    mask = torch.as_tensor(targets.finite_mask, dtype=torch.bool, device=device)
    classes = torch.as_tensor(targets.topology_class, dtype=torch.long, device=device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    losses: list[float] = []
    try:
        for _ in range(steps + 1):
            coordinates, logits = model(x)
            loss = labeller_loss(coordinates, logits, y, mask, classes)
            losses.append(float(loss.detach().cpu()))
            if len(losses) <= steps:
                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                optimiser.step()
    finally:
        if device == "cpu":
            torch.set_num_threads(previous_threads)
    return stats, losses


def target_statistics_from_shots(
    shot_ids: list[int],
    *,
    window_frames: int,
    windows_per_shot: int,
    level1_root: Path,
    frame_spans: dict[int, tuple[int, int]],
) -> TargetStatistics:
    """Compute training-only target statistics without loading camera pixels."""
    total = np.zeros(len(REGRESSION_NAMES), dtype=np.float64)
    total_squared = np.zeros(len(REGRESSION_NAMES), dtype=np.float64)
    count = np.zeros(len(REGRESSION_NAMES), dtype=np.int64)
    for shot in shot_ids:
        _, _, _, targets = _select_one_shot_targets(
            shot,
            window_frames=window_frames,
            windows_per_shot=windows_per_shot,
            level1_root=level1_root,
            frame_span=frame_spans[shot],
        )
        values = np.asarray(targets.values_m, dtype=np.float64)
        mask = np.asarray(targets.finite_mask, dtype=bool)
        total += np.where(mask, values, 0.0).sum(axis=0)
        total_squared += np.where(mask, values * values, 0.0).sum(axis=0)
        count += mask.sum(axis=0)
    safe_count = np.maximum(count, 1)
    mean = total / safe_count
    variance = np.maximum(total_squared / safe_count - mean * mean, 0.0)
    std = np.sqrt(variance)
    mean = np.where(count > 0, mean, 0.0)
    std = np.where((count > 0) & (std >= 1.0e-3), std, 1.0)
    return TargetStatistics(mean.astype(np.float32), std.astype(np.float32))


def _batch_loss(
    model: CameraTopologyLabeller,
    frames: np.ndarray,
    targets: LabellerTargets,
    statistics: TargetStatistics,
    indices: np.ndarray,
    *,
    device: str,
) -> torch.Tensor:
    values = targets.values_m[indices]
    mask = targets.finite_mask[indices]
    safe_values = np.where(mask, values, statistics.mean_m)
    normalised = (safe_values - statistics.mean_m) / statistics.std_m
    x = torch.as_tensor(frames[indices], dtype=torch.float32, device=device)
    y = torch.as_tensor(normalised, dtype=torch.float32, device=device)
    finite = torch.as_tensor(mask, dtype=torch.bool, device=device)
    classes = torch.as_tensor(
        targets.topology_class[indices], dtype=torch.long, device=device
    )
    device_type = torch.device(device).type
    with torch.autocast(
        device_type=device_type,
        dtype=torch.bfloat16,
        enabled=device_type == "cuda",
    ):
        coordinates, logits = model(x)
        return labeller_loss(coordinates, logits, y, finite, classes)


def _loaded_validation_loss(
    model: CameraTopologyLabeller,
    loaded: LoadedWindows,
    statistics: TargetStatistics,
    *,
    batch_size: int,
    device: str,
) -> tuple[float, int]:
    model.eval()
    weighted_loss = 0.0
    sample_count = 0
    with torch.no_grad():
        for start in range(0, loaded.frames.shape[0], batch_size):
            indices = np.arange(start, min(start + batch_size, loaded.frames.shape[0]))
            loss = _batch_loss(
                model,
                loaded.frames,
                loaded.targets,
                statistics,
                indices,
                device=device,
            )
            weighted_loss += float(loss.detach().cpu()) * indices.size
            sample_count += indices.size
    return weighted_loss, sample_count


def validation_loss_from_shots(
    model: CameraTopologyLabeller,
    shot_ids: list[int],
    statistics: TargetStatistics,
    *,
    window_frames: int,
    image_size: int,
    windows_per_shot: int,
    level1_root: Path,
    frame_spans: dict[int, tuple[int, int]],
    batch_size: int,
    device: str,
    window_loader: Callable[..., LoadedWindows] = load_camera_windows,
) -> tuple[float, int]:
    """Evaluate held-out loss with one bounded-memory shot resident at a time."""
    weighted_loss = 0.0
    sample_count = 0
    for shot in shot_ids:
        loaded = window_loader(
            [shot],
            window_frames=window_frames,
            image_size=image_size,
            windows_per_shot=windows_per_shot,
            level1_root=level1_root,
            frame_spans=frame_spans,
        )
        shot_loss, shot_count = _loaded_validation_loss(
            model,
            loaded,
            statistics,
            batch_size=batch_size,
            device=device,
        )
        weighted_loss += shot_loss
        sample_count += shot_count
    if sample_count == 0:
        raise ValueError("held-out split contains no complete camera windows")
    return weighted_loss / sample_count, sample_count


def fit_full_corpus_labeller(
    model: CameraTopologyLabeller,
    train_shots: list[int],
    validation_shots: list[int],
    statistics: TargetStatistics,
    *,
    window_frames: int,
    image_size: int,
    windows_per_shot: int,
    level1_root: Path,
    frame_spans: dict[int, tuple[int, int]],
    batch_size: int,
    epochs: int,
    learning_rate: float,
    seed: int,
    device: str,
    output: Path,
    declared_frame_counts: dict[str, int],
    window_loader: Callable[..., LoadedWindows] = load_camera_windows,
) -> dict[str, object]:
    """Train all cohort shots with bounded memory and checkpoint on held-out gain."""
    if batch_size < 1 or epochs < 1:
        raise ValueError("batch_size and epochs must both be positive")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    model.to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    initial_validation_loss, initial_validation_windows = validation_loss_from_shots(
        model,
        validation_shots,
        statistics,
        window_frames=window_frames,
        image_size=image_size,
        windows_per_shot=windows_per_shot,
        level1_root=level1_root,
        frame_spans=frame_spans,
        batch_size=batch_size,
        device=device,
        window_loader=window_loader,
    )
    report: dict[str, object] = {
        "mode": "full-corpus",
        "train_shots": len(train_shots),
        "validation_shots": len(validation_shots),
        "declared_train_frames": declared_frame_counts["train"],
        "declared_validation_frames": declared_frame_counts["validation"],
        "initial_validation_loss": initial_validation_loss,
        "initial_validation_windows": initial_validation_windows,
        "model_parameters": model.n_parameters(),
        "epochs_requested": epochs,
        "history": [],
        "first_improving_checkpoint": None,
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    rng = np.random.default_rng(seed)
    best_validation_loss = initial_validation_loss
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_windows = 0
        shuffled_shots = np.asarray(train_shots, dtype=np.int64)
        rng.shuffle(shuffled_shots)
        for shot in shuffled_shots.tolist():
            loaded = window_loader(
                [shot],
                window_frames=window_frames,
                image_size=image_size,
                windows_per_shot=windows_per_shot,
                level1_root=level1_root,
                frame_spans=frame_spans,
            )
            order = rng.permutation(loaded.frames.shape[0])
            for start in range(0, order.size, batch_size):
                indices = order[start : start + batch_size]
                loss = _batch_loss(
                    model,
                    loaded.frames,
                    loaded.targets,
                    statistics,
                    indices,
                    device=device,
                )
                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                optimiser.step()
                epoch_loss += float(loss.detach().cpu()) * indices.size
                epoch_windows += indices.size

        validation_loss, validation_windows = validation_loss_from_shots(
            model,
            validation_shots,
            statistics,
            window_frames=window_frames,
            image_size=image_size,
            windows_per_shot=windows_per_shot,
            level1_root=level1_root,
            frame_spans=frame_spans,
            batch_size=batch_size,
            device=device,
            window_loader=window_loader,
        )
        history_row: dict[str, object] = {
            "epoch": epoch,
            "train_loss": epoch_loss / max(epoch_windows, 1),
            "train_windows": epoch_windows,
            "validation_loss": validation_loss,
            "validation_windows": validation_windows,
            "improved_from_initialisation": validation_loss < initial_validation_loss,
            "checkpoint": None,
        }
        if (
            validation_loss < initial_validation_loss
            and validation_loss < best_validation_loss
        ):
            checkpoint = output.parent / f"{output.stem}-epoch-{epoch:03d}.pt"
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "config": asdict(model.config),
                    "target_mean_m": statistics.mean_m,
                    "target_std_m": statistics.std_m,
                    "regression_names": REGRESSION_NAMES,
                    "class_names": TOPOLOGY_CLASS_NAMES,
                    "epoch": epoch,
                    "initial_validation_loss": initial_validation_loss,
                    "validation_loss": validation_loss,
                    "train_shots": train_shots,
                    "validation_shots": validation_shots,
                },
                checkpoint,
            )
            history_row["checkpoint"] = str(checkpoint)
            best_validation_loss = validation_loss
            if report["first_improving_checkpoint"] is None:
                report["first_improving_checkpoint"] = str(checkpoint)
        history = report["history"]
        assert isinstance(history, list)
        history.append(history_row)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def predict_metres(
    model: CameraTopologyLabeller,
    frames: np.ndarray,
    statistics: TargetStatistics,
    *,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        coordinates, logits = model(
            torch.as_tensor(frames, dtype=torch.float32, device=device)
        )
    standard = coordinates.detach().cpu().numpy()
    values = standard * statistics.std_m + statistics.mean_m
    return values, logits.argmax(dim=1).detach().cpu().numpy()


def brightness_centroid_features(frames: np.ndarray) -> np.ndarray:
    """Return affine features from the centre frame's brightness centroid."""
    centre = np.asarray(frames, dtype=np.float64)[:, frames.shape[1] // 2]
    height, width = centre.shape[-2:]
    yy, xx = np.mgrid[0:height, 0:width]
    mass = centre.sum(axis=(1, 2))
    safe_mass = np.maximum(mass, 1.0e-12)
    x = (centre * xx).sum(axis=(1, 2)) / safe_mass / max(width - 1, 1)
    y = (centre * yy).sum(axis=(1, 2)) / safe_mass / max(height - 1, 1)
    return np.column_stack([np.ones(centre.shape[0]), x, y, centre.mean((1, 2))])


def fit_centroid_baseline(frames: np.ndarray, targets: LabellerTargets) -> np.ndarray:
    """Fit one masked affine brightness-centroid regressor per coordinate."""
    features = brightness_centroid_features(frames)
    coefficients = np.zeros(
        (features.shape[1], len(REGRESSION_NAMES)), dtype=np.float64
    )
    for column in range(len(REGRESSION_NAMES)):
        mask = targets.finite_mask[:, column]
        if np.count_nonzero(mask) >= features.shape[1]:
            coefficients[:, column] = np.linalg.lstsq(
                features[mask], targets.values_m[mask, column], rcond=None
            )[0]
        elif np.any(mask):
            coefficients[0, column] = float(np.mean(targets.values_m[mask, column]))
    return coefficients


def _point_error_cm(
    prediction: np.ndarray, targets: LabellerTargets, start: int
) -> tuple[float | None, int]:
    mask = targets.finite_mask[:, start : start + 2].all(axis=1)
    if not np.any(mask):
        return None, 0
    distance = np.linalg.norm(
        prediction[mask, start : start + 2] - targets.values_m[mask, start : start + 2],
        axis=1,
    )
    return float(100.0 * np.mean(distance)), int(np.count_nonzero(mask))


def evaluate_predictions(
    prediction_m: np.ndarray,
    predicted_class: np.ndarray,
    targets: LabellerTargets,
) -> dict[str, object]:
    """Report physical errors and explicit support/accuracy for every class."""
    axis_error, axis_support = _point_error_cm(prediction_m, targets, 0)
    null_error, null_support = _point_error_cm(prediction_m, targets, 2)
    per_class: dict[str, dict[str, float | int | None]] = {}
    for index, name in enumerate(TOPOLOGY_CLASS_NAMES):
        mask = targets.topology_class == index
        support = int(np.count_nonzero(mask))
        per_class[name] = {
            "support": support,
            "accuracy": (
                float(np.mean(predicted_class[mask] == index)) if support else None
            ),
        }
    defined = targets.topology_class != TOPOLOGY_UNDEFINED
    return {
        "axis_error_cm": axis_error,
        "axis_support": axis_support,
        "primary_null_error_cm": null_error,
        "primary_null_support": null_support,
        "class_accuracy": (
            float(np.mean(predicted_class[defined] == targets.topology_class[defined]))
            if np.any(defined)
            else None
        ),
        "undefined_class_support": int(np.count_nonzero(~defined)),
        "per_class": per_class,
    }


def _parse_shots(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def run_training(args: argparse.Namespace) -> dict[str, object]:
    if getattr(args, "full_corpus", False):
        frame_counts = read_cohort_frame_counts(args.cohort_report)
        split = {
            partition: list(entries) for partition, entries in frame_counts.items()
        }
        validate_full_corpus_split(frame_counts)
        if args.output is None:
            raise ValueError("full-corpus training requires --output")
        frame_spans = read_fullshot_spans(args.fullshot_manifest)
        train_shots = split["train"]
        validation_shots = split["validation"]
        missing = [
            shot for shot in train_shots + validation_shots if shot not in frame_spans
        ]
        if missing:
            raise ValueError(
                f"full-shot manifest is missing {len(missing)} cohort shots: "
                f"{missing[:8]}"
            )
        config = LabellerConfig(
            window_frames=args.window_frames,
            image_size=args.image_size,
            width=args.width,
            hidden=args.hidden,
        )
        statistics = target_statistics_from_shots(
            train_shots,
            window_frames=config.window_frames,
            windows_per_shot=args.windows_per_shot,
            level1_root=args.level1_root,
            frame_spans=frame_spans,
        )
        model = CameraTopologyLabeller(config)
        return fit_full_corpus_labeller(
            model,
            train_shots,
            validation_shots,
            statistics,
            window_frames=config.window_frames,
            image_size=config.image_size,
            windows_per_shot=args.windows_per_shot,
            level1_root=args.level1_root,
            frame_spans=frame_spans,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            seed=args.seed,
            device=args.device,
            output=args.output,
            declared_frame_counts={
                partition: sum(frame_counts[partition].values())
                for partition in ("train", "validation")
            },
        )

    split = read_cohort_split(args.cohort_report)
    train_shots = _parse_shots(args.train_shots)
    heldout_shots = _parse_shots(args.heldout_shots)
    if not train_shots or any(shot not in split["train"] for shot in train_shots):
        raise ValueError("every training shot must belong to the cohort train split")
    heldout_allowed = set(
        split["validation"] + split["clean_test"] + split["campaign_test"]
    )
    if not heldout_shots or any(shot not in heldout_allowed for shot in heldout_shots):
        raise ValueError(
            "every held-out shot must belong to a cohort held-out partition"
        )

    config = LabellerConfig(
        window_frames=args.window_frames,
        image_size=args.image_size,
        width=args.width,
        hidden=args.hidden,
    )
    train = load_camera_windows(
        train_shots,
        window_frames=config.window_frames,
        image_size=config.image_size,
        windows_per_shot=args.windows_per_shot,
        level1_root=args.level1_root,
    )
    heldout = load_camera_windows(
        heldout_shots,
        window_frames=config.window_frames,
        image_size=config.image_size,
        windows_per_shot=args.windows_per_shot,
        level1_root=args.level1_root,
    )
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = CameraTopologyLabeller(config)
    statistics, losses = fit_labeller(
        model,
        train.frames,
        train.targets,
        steps=args.steps,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=args.device,
    )
    model_prediction, model_class = predict_metres(
        model, heldout.frames, statistics, device=args.device
    )
    baseline_coefficients = fit_centroid_baseline(train.frames, train.targets)
    baseline_prediction = (
        brightness_centroid_features(heldout.frames) @ baseline_coefficients
    )
    baseline_metrics = evaluate_predictions(
        baseline_prediction,
        np.full(heldout.targets.topology_class.shape, TOPOLOGY_UNDEFINED),
        heldout.targets,
    )
    report: dict[str, object] = {
        "train_shots": train_shots,
        "heldout_shots": heldout_shots,
        "train_windows": int(train.frames.shape[0]),
        "heldout_windows": int(heldout.frames.shape[0]),
        "model_parameters": model.n_parameters(),
        "training_loss": {
            "initial": losses[0],
            "final": losses[-1],
            "decreased": bool(losses[-1] < losses[0]),
        },
        "model": evaluate_predictions(model_prediction, model_class, heldout.targets),
        "brightness_centroid_baseline": {
            key: baseline_metrics[key]
            for key in (
                "axis_error_cm",
                "axis_support",
                "primary_null_error_cm",
                "primary_null_support",
            )
        },
        "cohort_partition_sizes": {key: len(value) for key, value in split.items()},
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = args.output.with_suffix(".pt")
        torch.save(
            {
                "state_dict": model.state_dict(),
                "config": asdict(config),
                "target_mean_m": statistics.mean_m,
                "target_std_m": statistics.std_m,
                "regression_names": REGRESSION_NAMES,
                "class_names": TOPOLOGY_CLASS_NAMES,
            },
            checkpoint,
        )
        report["checkpoint"] = str(checkpoint)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-report", type=Path, default=DEFAULT_COHORT_REPORT)
    parser.add_argument("--level1-root", type=Path, default=DEFAULT_LEVEL1_ROOT)
    parser.add_argument(
        "--fullshot-manifest", type=Path, default=DEFAULT_FULLSHOT_MANIFEST
    )
    parser.add_argument("--full-corpus", action="store_true")
    parser.add_argument("--train-shots", default="21983,21985")
    parser.add_argument("--heldout-shots", default="21989")
    parser.add_argument("--windows-per-shot", type=int, default=24)
    parser.add_argument("--window-frames", type=int, default=3)
    parser.add_argument("--image-size", type=int, default=48)
    parser.add_argument("--width", type=int, default=12)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3.0e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = run_training(args)
    print(json.dumps(report, indent=2))
    if args.full_corpus:
        return 0 if report["first_improving_checkpoint"] is not None else 1
    return 0 if report["training_loss"]["decreased"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DEFAULT_COHORT_REPORT",
    "DEFAULT_FULLSHOT_MANIFEST",
    "DEFAULT_LEVEL1_ROOT",
    "FULL_CORPUS_COUNTS",
    "REGRESSION_NAMES",
    "CameraTopologyLabeller",
    "LabellerConfig",
    "LabellerTargets",
    "LoadedWindows",
    "TargetStatistics",
    "assemble_labeller_targets",
    "brightness_centroid_features",
    "evaluate_predictions",
    "fit_centroid_baseline",
    "fit_full_corpus_labeller",
    "fit_labeller",
    "labeller_loss",
    "load_camera_windows",
    "main",
    "predict_metres",
    "read_cohort_frame_counts",
    "read_cohort_split",
    "read_fullshot_spans",
    "run_training",
    "target_statistics",
    "target_statistics_from_shots",
    "validate_full_corpus_split",
    "validation_loss_from_shots",
]
