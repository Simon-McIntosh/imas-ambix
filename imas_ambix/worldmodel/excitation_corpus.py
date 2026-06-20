"""Select the curated dynamic-excitation training corpus.

Ties the curation pieces together into the single decision the re-train needs:
given the rbb-bearing shots, produce a list of curated
``CuratedWindow(shot_id, start_frame, fps, n_frames, frame_stride, ...)`` that are

* DYNAMIC — the window has real coil-current excitation
  (:func:`imas_ambix.worldmodel.actuator_plan.find_excitation_window` scores the
  time-mean summed coil ``|dI/dt|`` and rejects flat drive), with breakdown/ramp
  preferred and disruption tails de-prioritised;
* PLASMA-PRESENT — ``max|Ip| >= 20 kA`` and a present-fraction gate
  (:mod:`imas_ambix.worldmodel.plasma_presence`); a window may START at
  ``Ip ~ 0`` (breakdown) provided plasma forms within it;
* LONG-HORIZON — each window's ``n_frames`` / ``frame_stride`` derive from a
  target PHYSICAL horizon (s) and the shot's measured fps
  (:mod:`imas_ambix.worldmodel.window_horizon`), so a window spans at least a full
  ramp-up regardless of the ~250x cadence spread.

Held-out shots (whole pulses kept out of training) are EXCLUDED here, so the
curated list carries the shot-level leakage guard exactly as the overlapping
window enumerator does — the curated list is built only from the train shots
handed in minus the held-out reserve.

This module reads only the camera frame-time axis + the ``amc`` actuator vector
through the existing loaders; it does NOT touch pixels (exposure balancing is a
re-encode step, applied separately by the build script).  It is a pure selection
helper — backward-compatible, no change to the existing dataset path.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from imas_ambix.worldmodel.actuator_plan import find_excitation_window
from imas_ambix.worldmodel.plasma_presence import (
    IP_PRESENT_THRESHOLD_A,
    MIN_PRESENT_FRACTION,
)
from imas_ambix.worldmodel.spacetime_dataset import REFERENCE_CAMERA, _frame_times
from imas_ambix.worldmodel.window_horizon import (
    DEFAULT_MAX_N_FRAMES,
    DEFAULT_TARGET_HORIZON_S,
    recommend_window,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

#: The held-out shot reserve — whole pulses kept OUT of any curated training set
#: (the controllability gate's default eval shots; the spacetime-v2 / corruption
#: runs hold the same set out).  Excluded by :func:`select_curated_windows`.
DEFAULT_HELD_OUT: tuple[int, ...] = (18502, 18503, 18504, 18505)


@dataclass(frozen=True)
class CuratedWindow:
    """One curated dynamic-excitation training window.

    Attributes
    ----------
    shot_id, start_frame:
        Which shot and the first camera frame of the long-horizon window.
    fps:
        The shot's measured camera frame rate (Hz) the window length derives
        from.
    n_frames, frame_stride:
        The long-horizon window shape for this shot (span
        ``(n_frames-1)*frame_stride+1`` native frames ≈ the target horizon).
    excitation_score:
        Time-mean summed coil ``|dI/dt|`` of the chosen window (the dynamic
        weight — higher = more excited).
    max_abs_ip:
        ``max|Ip|`` over the window (A) — the plasma-presence measure.
    present_fraction:
        Fraction of the window's frames that are plasma-present.
    """

    shot_id: int
    start_frame: int
    fps: float
    n_frames: int
    frame_stride: int
    excitation_score: float
    max_abs_ip: float
    present_fraction: float


def _shot_fps(shot_id: int, camera: str, *, token_root: Path | None) -> float | None:
    """Median camera frame rate (Hz) for a shot, or None if unreadable."""
    ftime = _frame_times(int(shot_id), camera, token_root=token_root)
    if ftime is None:
        return None
    ft = np.asarray(ftime, dtype=np.float64)
    if ft.size < 2:
        return None
    dt = np.diff(ft)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        return None
    return float(1.0 / np.median(dt))


def select_curated_window_for_shot(
    shot_id: int,
    *,
    camera: str = REFERENCE_CAMERA,
    token_root: Path | None = None,
    target_horizon_s: float = DEFAULT_TARGET_HORIZON_S,
    max_n_frames: int = DEFAULT_MAX_N_FRAMES,
    ip_present_threshold: float = IP_PRESENT_THRESHOLD_A,
    min_present_fraction: float = MIN_PRESENT_FRACTION,
    min_excitation: float = 1.0e3,
) -> CuratedWindow | None:
    """Pick the single best curated window for one shot, or None if rejected.

    Derives the long-horizon window shape from the shot's fps + the target
    horizon, then finds the most coil-excited plasma-present window of that span
    in the recording.  Returns ``None`` (shot rejected) when the recording is
    unreadable, too short for the horizon window, has no plasma, or has only flat
    coil drive.
    """
    fps = _shot_fps(shot_id, camera, token_root=token_root)
    if fps is None:
        return None
    rec = recommend_window(
        fps, target_horizon_s=target_horizon_s, max_n_frames=max_n_frames
    )
    span = (rec.n_frames - 1) * rec.frame_stride + 1
    ex = find_excitation_window(
        int(shot_id),
        span,
        camera=camera,
        token_root=token_root,
        ip_present_threshold=ip_present_threshold,
        min_present_fraction=min_present_fraction,
        min_ramp_rate=min_excitation,
    )
    if ex.start_frame is None:
        logger.debug("shot %s rejected: %s", shot_id, ex.reason)
        return None
    return CuratedWindow(
        shot_id=int(shot_id),
        start_frame=int(ex.start_frame),
        fps=float(fps),
        n_frames=int(rec.n_frames),
        frame_stride=int(rec.frame_stride),
        excitation_score=float(ex.score),
        max_abs_ip=float(ex.max_abs_ip),
        present_fraction=float(ex.present_fraction),
    )


def select_curated_windows(
    shot_ids: Sequence[int],
    *,
    camera: str = REFERENCE_CAMERA,
    token_root: Path | None = None,
    held_out: Sequence[int] = DEFAULT_HELD_OUT,
    target_horizon_s: float = DEFAULT_TARGET_HORIZON_S,
    max_n_frames: int = DEFAULT_MAX_N_FRAMES,
    ip_present_threshold: float = IP_PRESENT_THRESHOLD_A,
    min_present_fraction: float = MIN_PRESENT_FRACTION,
    min_excitation: float = 1.0e3,
    limit: int | None = None,
) -> list[CuratedWindow]:
    """Select curated dynamic windows over a shot list (held-out shots excluded).

    For each candidate shot (input list MINUS ``held_out``) picks the single most
    excited plasma-present long-horizon window via
    :func:`select_curated_window_for_shot`.  Returns a DETERMINISTIC list sorted
    by descending excitation score (most dynamic first), so a ``limit`` keeps the
    most excited shots — the dynamic-weighting the lead asked for.  Every emitted
    ``shot_id`` is an input shot not in ``held_out`` — the structural shot-level
    leakage guard.
    """
    held = {int(s) for s in held_out}
    candidates = [int(s) for s in shot_ids if int(s) not in held]
    out: list[CuratedWindow] = []
    for sid in candidates:
        cw = select_curated_window_for_shot(
            sid,
            camera=camera,
            token_root=token_root,
            target_horizon_s=target_horizon_s,
            max_n_frames=max_n_frames,
            ip_present_threshold=ip_present_threshold,
            min_present_fraction=min_present_fraction,
            min_excitation=min_excitation,
        )
        if cw is not None:
            out.append(cw)
    # most-excited first, then by shot id for stability
    out.sort(key=lambda c: (-c.excitation_score, c.shot_id))
    if limit is not None:
        out = out[: int(limit)]
    return out


__all__ = [
    "DEFAULT_HELD_OUT",
    "CuratedWindow",
    "select_curated_window_for_shot",
    "select_curated_windows",
]
