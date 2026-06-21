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

from imas_ambix.worldmodel.actuator_plan import (
    ACTUATOR_CHANNELS,
    coil_current_channel_indices,
    find_excitation_window,
    plasma_current_channel_index,
)
from imas_ambix.worldmodel.plasma_presence import (
    IP_PRESENT_THRESHOLD_A,
    MIN_PRESENT_FRACTION,
    evaluate_presence,
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
    phase:
        Plasma phase the window covers — ``"ramp"`` (Ip rising), ``"flat_top"``
        (Ip roughly constant and high), ``"termination"`` (Ip falling / quench),
        ``"full"`` (the whole plasma phase — full-shot mode), or ``""``
        (unclassified, single-window selector).
    end_frame:
        Last camera frame of the window (exclusive of trailing dark).  For the
        fixed-horizon modes this is ``start_frame + (n_frames-1)*frame_stride+1``;
        for FULL-SHOT mode it is the end of the plasma phase (the quench), so
        ``[start_frame, end_frame)`` spans the WHOLE breakdown->termination
        evolution and the trainer time-subsamples that span to its own n_frames.
        ``0`` for the legacy fixed-horizon rows (derive from n_frames/stride).
    plasma_duration_s:
        Wall-clock duration (s) of ``[start_frame, end_frame)`` — the per-shot
        plasma-phase length the trainer uses as ``target_horizon_s`` in full-shot
        mode.  ``0.0`` for the legacy fixed-horizon rows.
    """

    shot_id: int
    start_frame: int
    fps: float
    n_frames: int
    frame_stride: int
    excitation_score: float
    max_abs_ip: float
    present_fraction: float
    phase: str = ""
    end_frame: int = 0
    plasma_duration_s: float = 0.0


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


# ---------------------------------------------------------------------------
# Multi-window tiling (every pulse, every phase, disruptions INCLUDED)
# ---------------------------------------------------------------------------
#
# select_curated_windows above picks the SINGLE most-excited window per shot.
# For the re-train the lead wants MULTIPLE 0.25 s windows per pulse covering the
# WHOLE recording — ramp-up, flat-top, AND the termination/quench — so the model
# sees every phase, not just the one most-excited segment.  The disruption-tail
# de-prioritisation used by find_excitation_window is dropped here (a fast
# current quench is a strong dynamic training signal, not noise), so a
# terminating window is scored on its true dynamics and kept if it is
# plasma-present (it has plasma then quench; pure post-quench vacuum fails the
# presence gate, correctly).

#: Default time-stride (s) between consecutive tiled windows.  ~0.075 s gives
#: ~70 % overlap on a 0.25 s window — dense coverage of every pulse phase
#: without exploding the window count.
DEFAULT_WINDOW_TIME_STRIDE_S: float = 0.075


def _classify_phase(absip_window: np.ndarray, *, present_threshold: float) -> str:
    """Classify a window's plasma phase from its per-frame ``|Ip|`` (A).

    Compares the mean ``|Ip|`` of the first vs last third of the window against
    the window peak: a strong RISE (last third ≫ first third, relative to peak)
    is ``"ramp"``; a strong FALL (first third ≫ last third — a quench/rampdown)
    is ``"termination"``; otherwise ``"flat_top"``.  The thresholds are relative
    to the in-window peak so the classification is brightness/scale invariant.
    """
    a = np.asarray(absip_window, dtype=np.float64)
    a = np.where(np.isfinite(a), a, 0.0)
    n = a.shape[0]
    if n < 3 or float(a.max()) <= 0:
        return "flat_top"
    k = max(1, n // 3)
    first = float(a[:k].mean())
    last = float(a[-k:].mean())
    peak = float(a.max())
    rise = (last - first) / peak  # +ve: rising, -ve: falling
    if rise > 0.25:
        return "ramp"
    if rise < -0.25:
        return "termination"
    return "flat_top"


def enumerate_shot_windows(
    shot_id: int,
    *,
    camera: str = REFERENCE_CAMERA,
    token_root: Path | None = None,
    target_horizon_s: float = DEFAULT_TARGET_HORIZON_S,
    max_n_frames: int = DEFAULT_MAX_N_FRAMES,
    window_time_stride_s: float = DEFAULT_WINDOW_TIME_STRIDE_S,
    ip_present_threshold: float = IP_PRESENT_THRESHOLD_A,
    min_present_fraction: float = MIN_PRESENT_FRACTION,
    min_excitation: float = 1.0e3,
) -> list[CuratedWindow]:
    """Tile one pulse with overlapping plasma-present windows (every phase).

    Derives the per-shot 0.25 s window shape from the shot's fps, then slides a
    window of that span across the WHOLE recording at ``window_time_stride_s``
    (converted to a frame step via fps).  Each candidate window is scored on the
    time-mean summed coil ``|dI/dt|`` (NO disruption de-weight — quenches count),
    checked for plasma-presence (``max|Ip| >= ip_present_threshold`` AND
    present-fraction gate), classified ramp/flat_top/termination, and kept if it
    passes presence and clears ``min_excitation``.  Returns the kept windows in
    ascending start-frame order (empty if the shot is unreadable / too short /
    has no plasma anywhere).  The whole shot's conditioning is loaded ONCE.
    """
    from imas_ambix.camdyn.conditioning import load_conditioning  # noqa: PLC0415
    from imas_ambix.camdyn.dataset import level1_shot_path  # noqa: PLC0415

    fps = _shot_fps(shot_id, camera, token_root=token_root)
    if fps is None:
        return []
    rec = recommend_window(
        fps, target_horizon_s=target_horizon_s, max_n_frames=max_n_frames
    )
    span = (rec.n_frames - 1) * rec.frame_stride + 1

    ftime = _frame_times(int(shot_id), camera, token_root=token_root)
    if ftime is None:
        return []
    ftime = np.asarray(ftime, dtype=np.float64)
    n = ftime.shape[0]
    if n < span or span < 2:
        return []

    try:
        lpath = level1_shot_path(int(shot_id))
    except Exception:  # noqa: BLE001
        return []

    # Load the FULL actuator vector (coils + Ip) onto every camera frame once.
    cond = load_conditioning(
        lpath, ftime, int(shot_id), channels=ACTUATOR_CHANNELS, include_dalpha=False
    )
    raw = np.asarray(cond.values, dtype=np.float64)
    miss = np.asarray(cond.missing, dtype=np.float64)

    coil_cols = coil_current_channel_indices()
    ip_col = plasma_current_channel_index()
    if not coil_cols or ip_col is None:
        return []
    coil_present = miss[:, coil_cols].mean(axis=0) < 1.0
    cols = [c for c, ok in zip(coil_cols, coil_present, strict=True) if ok]
    if not cols:
        return []

    # Per-frame summed coil |dI/dt| (kA/s-ish), NO disruption de-weight.
    coil = raw[:, cols]
    dt = np.diff(ftime)
    dt = np.where(dt > 0, dt, np.nan)
    didt = np.full_like(coil, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = np.diff(coil, axis=0) / dt[:, None]
    didt[:-1] = rate
    didt[-1] = rate[-1] if rate.shape[0] else 0.0
    ramp = np.abs(np.where(np.isfinite(didt), didt, 0.0)).sum(axis=1)

    # Per-frame |Ip| (A), NaN where the Ip record is missing.
    absip = np.abs(raw[:, ip_col])
    ip_ok = miss[:, ip_col] < 1.0
    absip = np.where(ip_ok, absip, np.nan)

    # Frame step between consecutive tiled windows (>= 1 frame).
    frame_step = max(1, int(round(float(window_time_stride_s) * fps)))
    last_start = n - span
    out: list[CuratedWindow] = []
    for start in range(0, last_start + 1, frame_step):
        sl = slice(start, start + span)
        w_absip = absip[sl]
        pres = evaluate_presence(
            w_absip,
            threshold_a=ip_present_threshold,
            min_present_fraction=min_present_fraction,
        )
        if not pres.present:
            continue
        score = float(np.nanmean(ramp[sl]))
        if not np.isfinite(score) or score < float(min_excitation):
            continue
        phase = _classify_phase(w_absip, present_threshold=ip_present_threshold)
        out.append(
            CuratedWindow(
                shot_id=int(shot_id),
                start_frame=int(start),
                fps=float(fps),
                n_frames=int(rec.n_frames),
                frame_stride=int(rec.frame_stride),
                excitation_score=score,
                max_abs_ip=float(pres.max_abs_ip),
                present_fraction=float(pres.present_fraction),
                phase=phase,
            )
        )
    return out


def enumerate_curated_windows(
    shot_ids: Sequence[int],
    *,
    camera: str = REFERENCE_CAMERA,
    token_root: Path | None = None,
    held_out: Sequence[int] = DEFAULT_HELD_OUT,
    target_horizon_s: float = DEFAULT_TARGET_HORIZON_S,
    max_n_frames: int = DEFAULT_MAX_N_FRAMES,
    window_time_stride_s: float = DEFAULT_WINDOW_TIME_STRIDE_S,
    ip_present_threshold: float = IP_PRESENT_THRESHOLD_A,
    min_present_fraction: float = MIN_PRESENT_FRACTION,
    min_excitation: float = 1.0e3,
    max_windows_per_shot: int | None = None,
) -> list[CuratedWindow]:
    """Enumerate MULTIPLE tiled windows over every pulse (held-out fully excluded).

    For each candidate shot (input list MINUS ``held_out``) tiles the whole
    recording with overlapping plasma-present windows
    (:func:`enumerate_shot_windows`), covering ramp-up / flat-top / termination.
    ``max_windows_per_shot`` optionally caps a very long recording's share
    (the kept windows are then spread evenly across the recording so all phases
    survive the cap).  Returns a DETERMINISTIC list (ascending shot id, then
    ascending start frame).  EVERY emitted shot_id is an input shot not in
    ``held_out`` — and because ALL of a held-out shot's windows are skipped, the
    held-out reserve is fully disjoint (shot-level leakage guard).
    """
    held = {int(s) for s in held_out}
    candidates = sorted(int(s) for s in shot_ids if int(s) not in held)
    out: list[CuratedWindow] = []
    for sid in candidates:
        ws = enumerate_shot_windows(
            sid,
            camera=camera,
            token_root=token_root,
            target_horizon_s=target_horizon_s,
            max_n_frames=max_n_frames,
            window_time_stride_s=window_time_stride_s,
            ip_present_threshold=ip_present_threshold,
            min_present_fraction=min_present_fraction,
            min_excitation=min_excitation,
        )
        if max_windows_per_shot is not None and len(ws) > int(max_windows_per_shot):
            # spread the cap evenly across the recording (keep first..last) so a
            # capped shot still spans ramp -> flat-top -> termination.
            idx = np.linspace(0, len(ws) - 1, int(max_windows_per_shot))
            keep = sorted({int(round(i)) for i in idx})
            ws = [ws[i] for i in keep]
        out.extend(ws)
    return out


# ---------------------------------------------------------------------------
# FULL-SHOT windows (one window = the whole plasma phase)
# ---------------------------------------------------------------------------
#
# The fixed-0.25 s windows above (single or tiled) cut the pulse into slices.
# The tiled multi-window corpus REGRESSED controllability because the many
# flat-top slices, as SEPARATE training samples, diluted the action signal.  The
# fix: ONE window per shot spanning the WHOLE plasma phase (breakdown -> ramp ->
# flat-top -> termination), so flat-top is CONTEXT inside a dynamic sequence, not
# a standalone sample.  The trainer time-subsamples this full span to its own
# n_frames (target_horizon_s = the per-shot plasma duration), so here we report
# the SPAN (start_frame, end_frame, duration), not a fixed frame count.

#: Default tightened present-fraction for a full-shot plasma phase.  The window
#: may START dark (breakdown) but must not be MOSTLY dark — >= 70 % of the span's
#: frames carry plasma.  (The lead: "a section without plasma is OK, not the
#: whole lot.")
FULLSHOT_MIN_PRESENT_FRACTION: float = 0.7

#: Minimum plasma-phase length (frames) to keep a shot — below this the span is
#: too short to time-subsample to a useful sequence.
FULLSHOT_MIN_SPAN_FRAMES: int = 16


@dataclass(frozen=True)
class PlasmaPhaseSpan:
    """The plasma-phase span of one shot's camera recording.

    Attributes
    ----------
    start_frame, end_frame:
        ``[start_frame, end_frame)`` — first plasma-present frame (breakdown) to
        one past the last plasma-present frame (the quench).  ``end_frame`` is
        exclusive of trailing post-quench dark.
    present_fraction:
        Fraction of frames WITHIN the span that are plasma-present (a span with
        long dark gaps mid-pulse scores low — the mostly-dark reject).
    max_abs_ip:
        ``max|Ip|`` over the span (A).
    duration_s:
        Wall-clock duration of the span (s).
    valid:
        True when the span clears the min-span + present-fraction + max|Ip| bars.
    reason:
        Why rejected (``""`` when valid): ``unreadable`` / ``no_plasma`` /
        ``too_short`` / ``mostly_dark``.
    """

    start_frame: int
    end_frame: int
    present_fraction: float
    max_abs_ip: float
    duration_s: float
    valid: bool
    reason: str = ""


def _shot_abs_ip(shot_id, camera, token_root):
    """Per-frame (ftime, |Ip| in A) for a shot, or (None, None)."""
    from imas_ambix.camdyn.conditioning import load_conditioning  # noqa: PLC0415
    from imas_ambix.camdyn.dataset import level1_shot_path  # noqa: PLC0415

    ftime = _frame_times(int(shot_id), camera, token_root=token_root)
    if ftime is None:
        return None, None
    ftime = np.asarray(ftime, dtype=np.float64)
    if ftime.size < 2:
        return None, None
    ip_col = plasma_current_channel_index()
    if ip_col is None:
        return None, None
    try:
        lpath = level1_shot_path(int(shot_id))
    except Exception:  # noqa: BLE001
        return None, None
    cond = load_conditioning(
        lpath, ftime, int(shot_id), channels=ACTUATOR_CHANNELS, include_dalpha=False
    )
    raw = np.asarray(cond.values, dtype=np.float64)
    miss = np.asarray(cond.missing, dtype=np.float64)
    absip = np.abs(raw[:, ip_col])
    absip = np.where(miss[:, ip_col] < 1.0, absip, np.nan)
    return ftime, absip


def find_plasma_phase_span(
    shot_id: int,
    *,
    camera: str = REFERENCE_CAMERA,
    token_root: Path | None = None,
    ip_present_threshold: float = IP_PRESENT_THRESHOLD_A,
    min_present_fraction: float = FULLSHOT_MIN_PRESENT_FRACTION,
    min_span_frames: int = FULLSHOT_MIN_SPAN_FRAMES,
) -> PlasmaPhaseSpan:
    """Find one shot's plasma-phase span ``[breakdown, quench)``.

    The span runs from the FIRST plasma-present frame (breakdown — or the
    recording start if it already begins in plasma) to one past the LAST
    plasma-present frame (the quench), trimming trailing post-quench dark.  A
    span is ``valid`` when it is long enough (``>= min_span_frames``), its
    ``max|Ip|`` clears the threshold, and it is NOT mostly-dark (its internal
    present-fraction ``>= min_present_fraction`` — a pulse that only flickers into
    plasma fails).  The window may START dark (the breakdown frames are
    sub-threshold) — that is the point.
    """
    ftime, absip = _shot_abs_ip(shot_id, camera, token_root)
    if ftime is None or absip is None:
        return PlasmaPhaseSpan(0, 0, 0.0, 0.0, 0.0, False, "unreadable")
    present = np.isfinite(absip) & (absip >= float(ip_present_threshold))
    if not present.any():
        max_overall = float(np.nanmax(absip)) if np.isfinite(absip).any() else 0.0
        return PlasmaPhaseSpan(0, 0, 0.0, max_overall, 0.0, False, "no_plasma")
    idx = np.flatnonzero(present)
    start = int(idx[0])
    end = int(idx[-1]) + 1  # exclusive of trailing dark
    span = present[start:end]
    n_span = end - start
    present_fraction = float(span.mean()) if n_span > 0 else 0.0
    max_abs = float(np.nanmax(absip[start:end]))
    duration_s = float(ftime[end - 1] - ftime[start]) if n_span >= 2 else 0.0
    if n_span < int(min_span_frames):
        return PlasmaPhaseSpan(
            start, end, present_fraction, max_abs, duration_s, False, "too_short"
        )
    if present_fraction < float(min_present_fraction):
        return PlasmaPhaseSpan(
            start, end, present_fraction, max_abs, duration_s, False, "mostly_dark"
        )
    return PlasmaPhaseSpan(start, end, present_fraction, max_abs, duration_s, True, "")


def select_fullshot_window(
    shot_id: int,
    *,
    camera: str = REFERENCE_CAMERA,
    token_root: Path | None = None,
    ip_present_threshold: float = IP_PRESENT_THRESHOLD_A,
    min_present_fraction: float = FULLSHOT_MIN_PRESENT_FRACTION,
    min_span_frames: int = FULLSHOT_MIN_SPAN_FRAMES,
) -> CuratedWindow | None:
    """One full-shot window = the whole plasma phase, or None if rejected.

    Emits a :class:`CuratedWindow` spanning ``[start_frame, end_frame)`` (the
    whole breakdown->termination evolution) with ``phase="full"``,
    ``plasma_duration_s`` set, and ``n_frames``/``frame_stride`` carrying the
    NATIVE span (n_frames = span length, frame_stride = 1) as a record of the
    full extent — the trainer ignores these and time-subsamples
    ``[start_frame, end_frame)`` to its own n_frames using
    ``target_horizon_s = plasma_duration_s``.  ``excitation_score`` is the
    time-mean summed coil ``|dI/dt|`` over the span (for ordering / reporting).
    """
    span = find_plasma_phase_span(
        shot_id,
        camera=camera,
        token_root=token_root,
        ip_present_threshold=ip_present_threshold,
        min_present_fraction=min_present_fraction,
        min_span_frames=min_span_frames,
    )
    if not span.valid:
        logger.debug("shot %s full-shot rejected: %s", shot_id, span.reason)
        return None
    fps = _shot_fps(shot_id, camera, token_root=token_root)
    if fps is None:
        return None
    # excitation score over the span (coil |dI/dt|, for ordering/reporting only).
    score = _span_excitation_score(
        shot_id, camera, token_root, span.start_frame, span.end_frame
    )
    n_span = span.end_frame - span.start_frame
    return CuratedWindow(
        shot_id=int(shot_id),
        start_frame=int(span.start_frame),
        fps=float(fps),
        n_frames=int(n_span),  # native span length (record; trainer subsamples)
        frame_stride=1,
        excitation_score=float(score),
        max_abs_ip=float(span.max_abs_ip),
        present_fraction=float(span.present_fraction),
        phase="full",
        end_frame=int(span.end_frame),
        plasma_duration_s=float(span.duration_s),
    )


def _span_excitation_score(shot_id, camera, token_root, start, end) -> float:
    """Time-mean summed coil |dI/dt| over [start, end) (reporting/ordering)."""
    from imas_ambix.camdyn.conditioning import load_conditioning  # noqa: PLC0415
    from imas_ambix.camdyn.dataset import level1_shot_path  # noqa: PLC0415

    ftime = _frame_times(int(shot_id), camera, token_root=token_root)
    if ftime is None:
        return 0.0
    ftime = np.asarray(ftime, dtype=np.float64)
    try:
        lpath = level1_shot_path(int(shot_id))
    except Exception:  # noqa: BLE001
        return 0.0
    cond = load_conditioning(
        lpath, ftime, int(shot_id), channels=ACTUATOR_CHANNELS, include_dalpha=False
    )
    raw = np.asarray(cond.values, dtype=np.float64)
    miss = np.asarray(cond.missing, dtype=np.float64)
    coil_cols = coil_current_channel_indices()
    coil_present = miss[:, coil_cols].mean(axis=0) < 1.0
    cols = [c for c, ok in zip(coil_cols, coil_present, strict=True) if ok]
    if not cols:
        return 0.0
    coil = raw[:, cols]
    dt = np.diff(ftime)
    dt = np.where(dt > 0, dt, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = np.diff(coil, axis=0) / dt[:, None]
    ramp = np.abs(np.where(np.isfinite(rate), rate, 0.0)).sum(axis=1)
    sl = slice(int(start), max(int(start) + 1, int(end) - 1))
    seg = ramp[sl]
    return float(np.nanmean(seg)) if seg.size else 0.0


def select_fullshot_windows(
    shot_ids: Sequence[int],
    *,
    camera: str = REFERENCE_CAMERA,
    token_root: Path | None = None,
    held_out: Sequence[int] = DEFAULT_HELD_OUT,
    ip_present_threshold: float = IP_PRESENT_THRESHOLD_A,
    min_present_fraction: float = FULLSHOT_MIN_PRESENT_FRACTION,
    min_span_frames: int = FULLSHOT_MIN_SPAN_FRAMES,
) -> list[CuratedWindow]:
    """One full-shot plasma-phase window per shot (held-out fully excluded).

    For each candidate shot (input list MINUS ``held_out``) emits one window
    spanning its whole plasma phase via :func:`select_fullshot_window`, dropping
    shots whose plasma phase is unreadable / absent / too short / mostly-dark.
    Returns a DETERMINISTIC list (ascending shot id).  Every emitted shot_id is
    an input shot not in ``held_out`` — the shot-level leakage guard.
    """
    held = {int(s) for s in held_out}
    candidates = sorted(int(s) for s in shot_ids if int(s) not in held)
    out: list[CuratedWindow] = []
    for sid in candidates:
        cw = select_fullshot_window(
            sid,
            camera=camera,
            token_root=token_root,
            ip_present_threshold=ip_present_threshold,
            min_present_fraction=min_present_fraction,
            min_span_frames=min_span_frames,
        )
        if cw is not None:
            out.append(cw)
    return out


def probe_plasma_activity(
    shot_ids: Sequence[int],
    *,
    camera: str = REFERENCE_CAMERA,
    token_root: Path | None = None,
    ip_present_threshold: float = IP_PRESENT_THRESHOLD_A,
) -> dict[int, dict]:
    """Diagnostic: is each shot plasma-ACTIVE through its plasma phase?

    For each shot returns ``{max_abs_ip, present_fraction, duration_s, valid,
    reason}`` for its plasma-phase span — so a held-out shot that FAILED a
    controllability metric can be checked for being low-activity / mostly-dark
    (a dark-frame artifact) vs genuinely plasma-active.  Does NOT exclude
    held-out shots (this is the diagnostic that inspects them).
    """
    out: dict[int, dict] = {}
    for sid in shot_ids:
        span = find_plasma_phase_span(
            int(sid),
            camera=camera,
            token_root=token_root,
            ip_present_threshold=ip_present_threshold,
            min_present_fraction=0.0,  # report raw; don't pre-reject the probe
            min_span_frames=1,
        )
        out[int(sid)] = {
            "max_abs_ip": span.max_abs_ip,
            "present_fraction": span.present_fraction,
            "duration_s": span.duration_s,
            "start_frame": span.start_frame,
            "end_frame": span.end_frame,
            "reason": span.reason,
        }
    return out


__all__ = [
    "DEFAULT_HELD_OUT",
    "DEFAULT_WINDOW_TIME_STRIDE_S",
    "FULLSHOT_MIN_PRESENT_FRACTION",
    "FULLSHOT_MIN_SPAN_FRAMES",
    "CuratedWindow",
    "PlasmaPhaseSpan",
    "enumerate_curated_windows",
    "enumerate_shot_windows",
    "find_plasma_phase_span",
    "probe_plasma_activity",
    "select_curated_window_for_shot",
    "select_curated_windows",
    "select_fullshot_window",
    "select_fullshot_windows",
]
