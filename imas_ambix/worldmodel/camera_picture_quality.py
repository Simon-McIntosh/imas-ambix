"""Measure camera picture quality for the MAST response carriers.

The measurement deliberately uses the centre half of each image in both
spatial dimensions.  This matches the earlier ``rbb`` screen while avoiding
camera borders and makes the crop scale with each native frame shape.  Pixel
values are mapped to an 8-bit-equivalent scale from the camera's declared bit
depth before applying the shared thresholds:

* saturated pixels are at least 254;
* a frame is blank-like when its mean is below 10 or its standard deviation is
  below 5;
* robust range is the median, over frames, of the 1st-to-99th percentile span;
* motion is adjacent-frame mean absolute difference, reported at median and
  90th percentile.

The common ``rbb`` stream supplies the cross-shot ordering.  Other camera
groups remain fully reported but do not enter that ordering, avoiding a rank
that confounds camera identity with shot quality.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from imas_ambix.data.paths import LEVEL1_DIR

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from numpy.typing import NDArray

FROZEN_CARRIER_SHOTS: tuple[int, ...] = (21978, 21983, 21985, 21986, 21989, 22086)
DEFAULT_SESSION_ROOT = Path(
    "/work/projects/imas_gpu/sophelio/labeller_sessions/76906a29"
)
DEFAULT_OUTPUT_DIR = Path(
    "docs/figures/physics-carried-playable-plasma/label-cartoon/carrier-cameras"
)
REPORT_JSON = "camera-picture-quality.json"
DEFAULT_OUTPUT = DEFAULT_OUTPUT_DIR / REPORT_JSON
REFERENCE_CAMERA = "rbb"
SATURATED_LEVEL = 254.0
BLANK_MEAN_LEVEL = 10.0
BLANK_STD_LEVEL = 5.0


@dataclass(frozen=True)
class CameraPictureMetrics:
    """Picture metrics for one complete native camera stream."""

    camera_group: str
    frame_count: int
    height: int
    width: int
    channel_count: int
    bit_depth: int
    time_start_s: float
    time_end_s: float
    saturated_fraction: float
    blank_like_fraction: float
    robust_intensity_range: float
    frame_motion_median: float
    frame_motion_p90: float


def _spatial_crop(frames: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
    if frames.ndim not in (3, 4):
        raise ValueError(
            "camera frames must have shape (time, height, width[, channel]), "
            f"got {frames.shape}"
        )
    height, width = frames.shape[1:3]
    crop_height = max(1, height // 2)
    crop_width = max(1, width // 2)
    row = (height - crop_height) // 2
    column = (width - crop_width) // 2
    return frames[:, row : row + crop_height, column : column + crop_width, ...]


def _eight_bit_equivalent(frames: NDArray[Any], bit_depth: int) -> NDArray[np.float32]:
    if bit_depth < 1 or bit_depth > 32:
        raise ValueError(f"camera bit depth must be in [1, 32], got {bit_depth}")
    native_max = float((1 << bit_depth) - 1)
    values = np.asarray(frames, dtype=np.float32)
    return np.clip(values * (255.0 / native_max), 0.0, 255.0)


def measure_camera_frames(
    frames: NDArray[Any],
    *,
    camera_group: str,
    bit_depth: int,
    frame_times: NDArray[Any] | None = None,
) -> CameraPictureMetrics:
    """Measure one native camera stream on the shared 8-bit-equivalent scale."""
    values = np.asarray(frames)
    if values.shape[0] < 1:
        raise ValueError(f"camera group {camera_group!r} has no frames")
    scaled = _spatial_crop(_eight_bit_equivalent(values, bit_depth))
    reduce_axes = tuple(range(1, scaled.ndim))
    frame_mean = np.mean(scaled, axis=reduce_axes, dtype=np.float64)
    frame_std = np.std(scaled, axis=reduce_axes, dtype=np.float64)
    flattened = scaled.reshape(scaled.shape[0], -1)
    low, high = np.percentile(flattened, (1.0, 99.0), axis=1)
    if scaled.shape[0] > 1:
        motion = np.mean(
            np.abs(np.diff(scaled, axis=0)), axis=reduce_axes, dtype=np.float64
        )
        motion_median = float(np.median(motion))
        motion_p90 = float(np.percentile(motion, 90.0))
    else:
        motion_median = 0.0
        motion_p90 = 0.0

    times = (
        np.arange(values.shape[0], dtype=np.float64)
        if frame_times is None
        else np.asarray(frame_times, dtype=np.float64)
    )
    if times.shape != (values.shape[0],):
        raise ValueError(
            f"camera group {camera_group!r} has {values.shape[0]} frames but "
            f"time shape {times.shape}"
        )
    return CameraPictureMetrics(
        camera_group=camera_group,
        frame_count=int(values.shape[0]),
        height=int(values.shape[1]),
        width=int(values.shape[2]),
        channel_count=int(values.shape[3]) if values.ndim == 4 else 1,
        bit_depth=int(bit_depth),
        time_start_s=float(times[0]),
        time_end_s=float(times[-1]),
        saturated_fraction=float(np.mean(scaled >= SATURATED_LEVEL)),
        blank_like_fraction=float(
            np.mean((frame_mean < BLANK_MEAN_LEVEL) | (frame_std < BLANK_STD_LEVEL))
        ),
        robust_intensity_range=float(np.median(high - low)),
        frame_motion_median=motion_median,
        frame_motion_p90=motion_p90,
    )


def picture_quality_score(metrics: Mapping[str, Any]) -> float:
    """Balanced display score used only to make the cross-shot order explicit.

    The robust range is discounted by clipped pixels and blank-like frames.
    Motion is a bounded-strength multiplier: a 20-level adjacent-frame change
    increases the score by 20 percent, so motion breaks exposure ties without
    allowing flicker to overwhelm contrast and usable-frame coverage.
    """
    usable_pixels = 1.0 - float(metrics["saturated_fraction"])
    usable_frames = 1.0 - float(metrics["blank_like_fraction"])
    motion_factor = 1.0 + min(float(metrics["frame_motion_p90"]), 100.0) / 100.0
    return float(
        max(0.0, float(metrics["robust_intensity_range"]))
        * max(0.0, usable_pixels)
        * max(0.0, usable_frames)
        * motion_factor
    )


def _session_record(session_root: Path, shot: int) -> dict[str, Any]:
    manifest = session_root / f"{shot}.manifest.json"
    session = session_root / f"{shot}.nc"
    if not manifest.is_file():
        return {
            "status": "absent",
            "manifest": str(manifest),
            "session": str(session),
        }
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    status = str(payload.get("status", "present-without-status"))
    if status == "complete" and not session.is_file():
        status = "complete-manifest-missing-session"
    return {
        "status": status,
        "manifest": str(manifest),
        "session": str(session),
    }


def _thomson_record(store: Any) -> dict[str, Any]:
    groups: dict[str, int] = {}
    for name in ("atm", "ayc", "aye"):
        if name in store and "radius" in store[name]:
            groups[name] = int(store[name]["radius"].shape[0])
    preferred = "atm" if "atm" in groups else next(iter(groups), None)
    return {
        "group": preferred,
        "channel_count": groups.get(preferred, 0),
        "groups": groups,
    }


def measure_shot(
    shot: int,
    *,
    level1_root: Path = LEVEL1_DIR,
    session_root: Path = DEFAULT_SESSION_ROOT,
) -> dict[str, Any]:
    """Read every image group for one shot and return its complete measurement."""
    import zarr  # noqa: PLC0415

    path = level1_root / f"{shot}.zarr"
    store = zarr.open_group(str(path), mode="r")
    cameras: list[dict[str, Any]] = []
    for name in sorted(store.group_keys()):
        group = store[name]
        if not name.startswith("r") or "data" not in group or "time" not in group:
            continue
        attrs = group.attrs.asdict()
        depth = int(attrs.get("depth", np.dtype(group["data"].dtype).itemsize * 8))
        measured = measure_camera_frames(
            np.asarray(group["data"]),
            camera_group=name,
            bit_depth=depth,
            frame_times=np.asarray(group["time"]),
        )
        row = asdict(measured)
        row["source_quality"] = str(attrs.get("quality", "unknown"))
        row["source_uuid"] = attrs.get("uuid")
        cameras.append(row)

    reference = next(
        (camera for camera in cameras if camera["camera_group"] == REFERENCE_CAMERA),
        None,
    )
    if reference is None:
        raise ValueError(f"shot {shot} has no {REFERENCE_CAMERA!r} camera group")
    session = _session_record(session_root, shot)
    return {
        "shot": int(shot),
        "level1_store": str(path),
        "thomson": _thomson_record(store),
        "corpus_session": session,
        "label_cartoon_ready_today": session["status"] == "complete",
        "label_cartoon_blocker": (
            None
            if session["status"] == "complete"
            else "no complete nova labeller session"
        ),
        "reference_camera": REFERENCE_CAMERA,
        "picture_quality_score": picture_quality_score(reference),
        "cameras": cameras,
    }


def _source_revision() -> str:
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def measure_carrier_cameras(
    shots: Iterable[int] = FROZEN_CARRIER_SHOTS,
    *,
    level1_root: Path = LEVEL1_DIR,
    session_root: Path = DEFAULT_SESSION_ROOT,
    source_revision: str | None = None,
) -> dict[str, Any]:
    """Measure and rank the frozen response-carrier camera streams."""
    records = [
        measure_shot(int(shot), level1_root=level1_root, session_root=session_root)
        for shot in shots
    ]
    ranked = sorted(
        records,
        key=lambda record: (-float(record["picture_quality_score"]), record["shot"]),
    )
    for rank, record in enumerate(ranked, start=1):
        record["picture_quality_rank"] = rank
    records.sort(key=lambda record: record["picture_quality_rank"])
    ready_count = sum(bool(record["label_cartoon_ready_today"]) for record in records)
    return {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "source_revision": source_revision or _source_revision(),
        "level1_root": str(level1_root),
        "session_root": str(session_root),
        "reference_camera": REFERENCE_CAMERA,
        "metric_definition": {
            "crop": "centred half-height by half-width at native resolution",
            "intensity_scale": "8-bit equivalent from declared camera bit depth",
            "saturated_fraction": "fraction of cropped pixel-channel values >= 254",
            "blank_like_fraction": (
                "fraction of cropped frames with mean < 10 or standard deviation < 5"
            ),
            "robust_intensity_range": "median per-frame (p99 - p1)",
            "frame_motion": "adjacent-frame mean absolute difference, median and p90",
            "ranking": (
                "rbb robust range * (1-saturated) * (1-blank-like) * "
                "(1+min(p90 motion,100)/100)"
            ),
        },
        "ranking": [int(record["shot"]) for record in records],
        "label_cartoon_ready_count": ready_count,
        "label_cartoon_total_count": len(records),
        "conclusion": (
            f"{ready_count} of {len(records)} frozen carriers have a complete nova "
            "labeller session and can supply both panels today. Camera quality and "
            "session readiness are separate: a high picture-quality rank does not "
            "supply missing physics labels."
        ),
        "shots": records,
    }


def write_json_report(report: Mapping[str, Any], output: Path) -> Path:
    """Write the machine-readable camera-quality receipt."""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shots", nargs="+", type=int, default=list(FROZEN_CARRIER_SHOTS)
    )
    parser.add_argument("--level1-root", type=Path, default=LEVEL1_DIR)
    parser.add_argument("--session-root", type=Path, default=DEFAULT_SESSION_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = measure_carrier_cameras(
        args.shots, level1_root=args.level1_root, session_root=args.session_root
    )
    json_path = write_json_report(report, args.output)
    print(
        json.dumps(
            {
                "json_report": str(json_path),
                "ranking": report["ranking"],
                "label_cartoon_ready": (
                    f"{report['label_cartoon_ready_count']}/"
                    f"{report['label_cartoon_total_count']}"
                ),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
