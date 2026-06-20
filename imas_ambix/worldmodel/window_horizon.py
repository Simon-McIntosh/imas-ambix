"""Derive a LONG-HORIZON camera-window config from the coil/Ip timescales + fps.

The current spatiotemporal window (:class:`SpacetimeWindowConfig`, ``n_frames=24``,
``frame_stride=1``) spans only ``(24-1)*1+1 = 24`` camera frames.  At the
reference rbb cadence (~600 Hz) that is ~40 ms — far shorter than a MAST
ramp-up (the breakdown->flat-top current rise is ~50-160 ms; see the excitation
characterisation).  To "play the plasma" over a useful horizon the model must
predict over a window that spans **at least a full ramp-up**.

The complication this module solves: the rbb camera was run at WILDLY different
frame rates across shots (measured: ~600 Hz to ~77 kHz over a ~0.1-0.5 s
recording).  A FIXED frame count therefore spans a different physical horizon on
every shot.  The horizon that matters is in SECONDS of plasma evolution, so the
window must be specified as a **target physical horizon** and converted to a
frame span PER SHOT via that shot's measured fps:

    span_frames ≈ round(target_horizon_s * fps) + 1

and the model's temporal sequence length (``n_frames``) is bounded by a token
budget, so for a long physical span we sub-sample with a ``frame_stride`` that
keeps ``n_frames`` within budget while still covering the horizon:

    frame_stride = ceil(span_frames / n_frames_budget)

This module computes that recommendation (corpus-wide and per-shot) and builds a
:class:`SpacetimeWindowConfig` for it.  It is a pure recommendation helper — it
does not change the existing default config and is fully backward-compatible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

#: Default target physical horizon (s) for a long-horizon window — chosen to
#: span a full MAST ramp-up.  The breakdown->flat-top current rise is ~50-160 ms
#: (excitation characterisation); 0.25 s comfortably contains the ramp plus a
#: lead-in and a little early flat-top so the model sees the WHOLE driven phase.
DEFAULT_TARGET_HORIZON_S: float = 0.25

#: Default cap on the model's temporal sequence length (``n_frames``).  The
#: spatiotemporal model attends over ``n_frames * 256`` camera tokens; 48 frames
#: (vs the current 24) doubles the horizon at fixed stride and stays within a
#: tractable attention budget — a long horizon is then reached by ``frame_stride``
#: when the native span exceeds ``48`` frames.
DEFAULT_MAX_N_FRAMES: int = 48

#: Reference rbb cadence (Hz) used to fall back when a shot's fps is unknown.
REFERENCE_FPS: float = 600.0


@dataclass(frozen=True)
class WindowHorizonRecommendation:
    """A long-horizon window recommendation for a target physical horizon.

    Attributes
    ----------
    target_horizon_s:
        The physical horizon (s) the window should span.
    n_frames:
        Recommended temporal sequence length (frames the model sees).
    frame_stride:
        Recommended stride (take every ``frame_stride``-th native frame) so the
        modelled span covers the horizon while ``n_frames`` stays within budget.
    fps:
        The frame rate (Hz) the recommendation was derived against (per-shot
        measured, or the corpus median / reference fallback).
    native_span_frames:
        The native (stride-1) frame count that spans ``target_horizon_s`` at
        ``fps`` — i.e. ``round(target_horizon_s * fps) + 1``.
    covered_horizon_s:
        The horizon actually covered by ``(n_frames-1)*frame_stride+1`` frames at
        ``fps`` — may be slightly more/less than the target after rounding.
    context_frames:
        Recommended leading-context frame count (kept at ~1/3 of ``n_frames`` so
        the forecast window is the majority — the "play forward" horizon).
    """

    target_horizon_s: float
    n_frames: int
    frame_stride: int
    fps: float
    native_span_frames: int
    covered_horizon_s: float
    context_frames: int


def recommend_window(
    fps: float,
    *,
    target_horizon_s: float = DEFAULT_TARGET_HORIZON_S,
    max_n_frames: int = DEFAULT_MAX_N_FRAMES,
    context_fraction: float = 1.0 / 3.0,
) -> WindowHorizonRecommendation:
    """Recommend a window config that spans ``target_horizon_s`` at ``fps``.

    The native span (stride-1 frames) that covers the horizon is
    ``round(target_horizon_s * fps) + 1``.  If that fits within ``max_n_frames``
    we use stride 1 and ``n_frames = native_span`` (full native cadence over the
    horizon).  Otherwise we sub-sample: ``frame_stride`` is the smallest integer
    such that ``ceil(native_span / frame_stride) <= max_n_frames``, and
    ``n_frames`` is set so ``(n_frames-1)*frame_stride+1`` reaches the horizon.

    ``context_frames`` is ``max(1, round(context_fraction * n_frames))`` so the
    rollout has a short context and a long forecast horizon.
    """
    fps = float(fps) if (np.isfinite(fps) and fps > 0) else REFERENCE_FPS
    target_horizon_s = float(target_horizon_s)
    max_n_frames = max(2, int(max_n_frames))

    native_span = int(round(target_horizon_s * fps)) + 1
    native_span = max(2, native_span)

    if native_span <= max_n_frames:
        frame_stride = 1
        n_frames = native_span
    else:
        frame_stride = int(math.ceil(native_span / max_n_frames))
        # n_frames so the modelled span (n-1)*stride+1 first reaches the horizon
        n_frames = int(math.ceil((native_span - 1) / frame_stride)) + 1
        n_frames = min(n_frames, max_n_frames)

    covered = ((n_frames - 1) * frame_stride + 1) / fps
    context_frames = max(1, int(round(context_fraction * n_frames)))
    context_frames = min(context_frames, n_frames - 1)
    return WindowHorizonRecommendation(
        target_horizon_s=target_horizon_s,
        n_frames=int(n_frames),
        frame_stride=int(frame_stride),
        fps=float(fps),
        native_span_frames=int(native_span),
        covered_horizon_s=float(covered),
        context_frames=int(context_frames),
    )


def window_config_for(
    fps: float,
    *,
    target_horizon_s: float = DEFAULT_TARGET_HORIZON_S,
    max_n_frames: int = DEFAULT_MAX_N_FRAMES,
    n_plan: int = 8,
):
    """Build a :class:`SpacetimeWindowConfig` for a long-horizon window at ``fps``.

    Thin wrapper over :func:`recommend_window` that returns a ready
    ``SpacetimeWindowConfig`` (imported lazily so this module has no import-time
    dependency on the dataset module).  ``n_plan`` is passed through unchanged.
    """
    from imas_ambix.worldmodel.spacetime_dataset import (  # noqa: PLC0415
        SpacetimeWindowConfig,
    )

    rec = recommend_window(
        fps, target_horizon_s=target_horizon_s, max_n_frames=max_n_frames
    )
    return SpacetimeWindowConfig(
        n_frames=rec.n_frames,
        n_plan=int(n_plan),
        context_frames=rec.context_frames,
        frame_stride=rec.frame_stride,
    )


def corpus_fps_summary(fps_values: np.ndarray) -> dict:
    """Summarise a corpus fps distribution (for the characterisation doc).

    Returns the median / IQR / min / max and the fraction of shots at the
    reference ~600 Hz cadence (within 10%), so the doc can state how
    heterogeneous the cadence is and justify the per-shot (vs fixed-frame)
    window-length rule.
    """
    f = np.asarray(fps_values, dtype=np.float64)
    f = f[np.isfinite(f) & (f > 0)]
    if f.size == 0:
        return {"n": 0}
    near_ref = float(np.mean(np.abs(f - REFERENCE_FPS) <= 0.1 * REFERENCE_FPS))
    return {
        "n": int(f.size),
        "median": float(np.median(f)),
        "p25": float(np.percentile(f, 25)),
        "p75": float(np.percentile(f, 75)),
        "min": float(f.min()),
        "max": float(f.max()),
        "frac_near_reference": near_ref,
    }


__all__ = [
    "DEFAULT_MAX_N_FRAMES",
    "DEFAULT_TARGET_HORIZON_S",
    "REFERENCE_FPS",
    "WindowHorizonRecommendation",
    "corpus_fps_summary",
    "recommend_window",
    "window_config_for",
]
