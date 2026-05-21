"""JSON serialisation and deserialisation for calibration objects.

Save layout on disk::

    {CALIBRATION_ROOT}/signals/{group}.json   — dict[str, ChannelCalibration]
    {CALIBRATION_ROOT}/frames/{camera}.json   — FrameCalibration

Usage::

    from imas_ambix.calibration.persistence import (
        save_calibration,
        load_signal_calibration,
        load_frame_calibration,
        CALIBRATION_ROOT,
    )

    save_calibration(cal_dict, CALIBRATION_ROOT / "signals" / "summary.json")
    cal = load_signal_calibration(CALIBRATION_ROOT / "signals" / "summary.json")
"""

from __future__ import annotations

import json
from pathlib import Path

from imas_ambix.calibration.frames import FrameCalibration
from imas_ambix.calibration.signals import ChannelCalibration

# ---------------------------------------------------------------------------
# Root path
# ---------------------------------------------------------------------------

CALIBRATION_ROOT = Path("/work/projects/imas_gpu/mast-tokens/v1/calibration")
"""Default root for persisted calibration files.

Layout::

    {CALIBRATION_ROOT}/
        signals/
            {group}.json
        frames/
            {camera}.json
"""

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _channel_cal_to_dict(cal: ChannelCalibration) -> dict[str, object]:
    return {
        "name": cal.name,
        "mean": cal.mean,
        "std": cal.std,
        "min_value": cal.min_value,
        "max_value": cal.max_value,
        "q01": cal.q01,
        "q50": cal.q50,
        "q99": cal.q99,
        "n_samples": cal.n_samples,
        "n_shots": cal.n_shots,
    }


def _channel_cal_from_dict(d: dict[str, object]) -> ChannelCalibration:
    return ChannelCalibration(
        name=str(d["name"]),
        mean=float(d["mean"]),  # type: ignore[arg-type]
        std=float(d["std"]),  # type: ignore[arg-type]
        min_value=float(d["min_value"]),  # type: ignore[arg-type]
        max_value=float(d["max_value"]),  # type: ignore[arg-type]
        q01=float(d["q01"]),  # type: ignore[arg-type]
        q50=float(d["q50"]),  # type: ignore[arg-type]
        q99=float(d["q99"]),  # type: ignore[arg-type]
        n_samples=int(d["n_samples"]),  # type: ignore[arg-type]
        n_shots=int(d["n_shots"]),  # type: ignore[arg-type]
    )


def _frame_cal_to_dict(cal: FrameCalibration) -> dict[str, object]:
    return {
        "camera": cal.camera,
        "global_min": cal.global_min,
        "global_max": cal.global_max,
        "global_mean": cal.global_mean,
        "global_std": cal.global_std,
        # JSON keys must be strings; convert int keys to str on save
        "per_shot_min": {str(k): v for k, v in cal.per_shot_min.items()},
        "per_shot_max": {str(k): v for k, v in cal.per_shot_max.items()},
        "suggested": cal.suggested,
    }


def _frame_cal_from_dict(d: dict[str, object]) -> FrameCalibration:
    per_shot_min = {int(k): float(v) for k, v in d["per_shot_min"].items()}  # type: ignore[union-attr]
    per_shot_max = {int(k): float(v) for k, v in d["per_shot_max"].items()}  # type: ignore[union-attr]
    return FrameCalibration(
        camera=str(d["camera"]),
        global_min=float(d["global_min"]),  # type: ignore[arg-type]
        global_max=float(d["global_max"]),  # type: ignore[arg-type]
        global_mean=float(d["global_mean"]),  # type: ignore[arg-type]
        global_std=float(d["global_std"]),  # type: ignore[arg-type]
        per_shot_min=per_shot_min,
        per_shot_max=per_shot_max,
        suggested=str(d["suggested"]),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def save_calibration(
    calibration: dict[str, ChannelCalibration] | FrameCalibration,
    path: Path,
) -> None:
    """Serialise a calibration object to JSON.

    Auto-detects whether ``calibration`` is a
    ``dict[str, ChannelCalibration]`` (signal calibration) or a
    :class:`~imas_ambix.calibration.frames.FrameCalibration` (frame
    calibration) and picks the appropriate encoder.

    Parameters
    ----------
    calibration:
        The calibration to serialise.
    path:
        Target JSON file path.  Parent directories are created if absent.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(calibration, FrameCalibration):
        payload: dict[str, object] = {
            "__type__": "FrameCalibration",
            "data": _frame_cal_to_dict(calibration),
        }
    elif isinstance(calibration, dict):
        payload = {
            "__type__": "SignalCalibration",
            "data": {k: _channel_cal_to_dict(v) for k, v in calibration.items()},
        }
    else:
        raise TypeError(
            f"calibration must be dict[str, ChannelCalibration] or FrameCalibration, "
            f"got {type(calibration)}"
        )

    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, allow_nan=True)


def load_signal_calibration(path: Path) -> dict[str, ChannelCalibration]:
    """Load a signal calibration from a JSON file.

    Parameters
    ----------
    path:
        Path to a JSON file written by :func:`save_calibration` for a
        signal calibration (``__type__ == "SignalCalibration"``).

    Returns
    -------
    dict[str, ChannelCalibration]
    """
    with Path(path).open(encoding="utf-8") as fh:
        payload = json.load(fh)

    if payload.get("__type__") != "SignalCalibration":
        raise ValueError(
            f"Expected SignalCalibration JSON, got __type__={payload.get('__type__')!r}"
        )
    return {k: _channel_cal_from_dict(v) for k, v in payload["data"].items()}


def load_frame_calibration(path: Path) -> FrameCalibration:
    """Load a frame calibration from a JSON file.

    Parameters
    ----------
    path:
        Path to a JSON file written by :func:`save_calibration` for a
        frame calibration (``__type__ == "FrameCalibration"``).

    Returns
    -------
    FrameCalibration
    """
    with Path(path).open(encoding="utf-8") as fh:
        payload = json.load(fh)

    if payload.get("__type__") != "FrameCalibration":
        raise ValueError(
            f"Expected FrameCalibration JSON, got __type__={payload.get('__type__')!r}"
        )
    return _frame_cal_from_dict(payload["data"])
