"""Exposure-balancing transforms for the curated dynamic-excitation corpus.

The corpus is tokenised, and brightness lives in PIXELS — BEFORE tokenisation.
The current tokeniser normaliser
(:func:`imas_ambix.tokenizer.frames._normalise_frames_to_uint8`, mirrored in
:func:`imas_ambix.data.stream_encode.normalise_frames_to_uint8`) does a per-shot
GLOBAL min/max stretch:

    u8 = (f - f.min()) * 255 / (f.max() - f.min())

Its own docstring flags it as a v0 ("more careful normalisation can come later").
The excitation characterisation shows why "later" is now: the per-shot mean
brightness spans a ~870x range across shots, and the sensors are MIXED bit-depth
(most cap at 255 = 8-bit, a minority at 1023 = 10-bit).  Global min/max is
fragile to exactly this — a single saturated reflection or a dark-dropout frame
pins ``min``/``max`` and compresses the whole shot's usable range, so the SAME
plasma feature lands at a different token after stretch on different shots.  That
inconsistency is brightness leaking into the tokens — the opposite of bringing
shots to the same RELATIVE brightness.

This module provides robust per-shot exposure transforms to replace the v0
stretch for a SUBSET re-encode of the curated dynamic segments (NOT the full
4 B-token corpus — see the build script's compute gate):

* :func:`percentile_normalise` — clip to a per-shot ``[p_lo, p_hi]`` percentile
  window, then stretch.  Robust to saturated/dropout outliers; preserves the
  relative brightness of the bulk of the plasma signal.  The recommended
  default (cheap, deterministic, no per-frame flicker).
* :func:`clahe_normalise` — Contrast-Limited Adaptive Histogram Equalisation
  per frame after a percentile pre-clip.  For shots whose plasma feature is a
  low-contrast structure on a bright background; more invasive (per-frame, can
  amplify noise in dark frames), so applied only where the percentile transform
  leaves the feature under-exposed.

Both keep the ``normalise_frames_to_uint8`` contract: ``(T,H,W)`` (or
``(T,H,W,3)``) any-dtype in, ``(T,H,W[,3])`` uint8 out.  This module has NO heavy
import at module load (OpenCV is imported lazily inside :func:`clahe_normalise`)
so it is safe to import in the ambix tree and in the encode venv alike.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Default per-shot clip percentiles for :func:`percentile_normalise`.  [1, 99.5]
#: discards the brightest 0.5 % (saturated reflections / hot pixels) and darkest
#: 1 % (dropout) so the stretch is set by the bulk of the plasma signal, not by
#: outliers — the chief failure of the v0 global min/max.
DEFAULT_CLIP_PERCENTILES: tuple[float, float] = (1.0, 99.5)


def _to_float_gray(frames: np.ndarray) -> tuple[np.ndarray, bool]:
    """Return ``(T,H,W) float32`` plus whether the input was RGB ``(T,H,W,3)``.

    An RGB input is collapsed to a single channel (the camera signal is
    grey-replicated; the channels are identical) so the percentile window is
    computed once; the uint8 result is re-replicated to 3 channels on the way
    out to preserve the caller's channel layout.
    """
    f = np.asarray(frames)
    if f.ndim == 4 and f.shape[-1] == 3:
        return f[..., 0].astype(np.float32), True
    if f.ndim == 3:
        return f.astype(np.float32), False
    raise ValueError(f"frames must be (T,H,W) or (T,H,W,3), got {f.shape}")


def _restore_channels(u8: np.ndarray, was_rgb: bool) -> np.ndarray:
    """Re-replicate to ``(T,H,W,3)`` when the input was RGB; else pass through."""
    if was_rgb:
        return np.repeat(u8[..., None], 3, axis=-1)
    return u8


def cv2_available() -> bool:
    """True when OpenCV (``cv2``) can be imported (CLAHE requires it).

    The ambix env does NOT ship OpenCV by default — so the corpus-default
    transform is :func:`percentile_normalise` (pure-numpy) and ``"auto"`` falls
    back to it when ``cv2`` is absent rather than crashing the encode.  Add
    ``opencv-python-headless`` to the encode env to enable the CLAHE branch.
    """
    import importlib.util  # noqa: PLC0415

    return importlib.util.find_spec("cv2") is not None


def percentile_normalise(
    frames: np.ndarray,
    *,
    clip_percentiles: tuple[float, float] = DEFAULT_CLIP_PERCENTILES,
) -> np.ndarray:
    """Per-shot percentile-clip + stretch to uint8 [0,255].

    Computes the ``[p_lo, p_hi]`` percentiles over the WHOLE shot (all frames),
    clips to that window, and linearly stretches it to [0,255].  Robust drop-in
    for the v0 global min/max stretch: a bright outlier or dark dropout no longer
    pins the range, so the same plasma intensity maps to the same uint8 (hence
    the same token) across shots — bringing the corpus to a common RELATIVE
    brightness.  Returns the same shape/layout as ``frames``, dtype uint8.

    A degenerate shot (``p_hi <= p_lo``, e.g. a flat frame) returns all-zeros, as
    the v0 normaliser does.
    """
    f, was_rgb = _to_float_gray(frames)
    lo_p, hi_p = float(clip_percentiles[0]), float(clip_percentiles[1])
    lo = float(np.percentile(f, lo_p))
    hi = float(np.percentile(f, hi_p))
    if hi <= lo:
        u8 = np.zeros(f.shape, dtype=np.uint8)
        return _restore_channels(u8, was_rgb)
    stretched = (f - lo) * 255.0 / (hi - lo)
    u8 = np.clip(stretched, 0, 255).astype(np.uint8)
    return _restore_channels(u8, was_rgb)


def clahe_normalise(
    frames: np.ndarray,
    *,
    clip_percentiles: tuple[float, float] = DEFAULT_CLIP_PERCENTILES,
    clip_limit: float = 2.0,
    tile_grid: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Per-frame CLAHE after a per-shot percentile pre-clip → uint8 [0,255].

    Applies :func:`percentile_normalise` first (so the dynamic range is robustly
    set), then Contrast-Limited Adaptive Histogram Equalisation per frame to lift
    a low-contrast plasma structure off a bright background.  More invasive than
    the percentile transform (per-frame, can amplify noise in near-dark frames),
    so this is the targeted option for shots the percentile transform leaves
    under-exposed — NOT the corpus default.

    Requires OpenCV (``cv2``), imported lazily.  Returns the same shape/layout as
    ``frames``, dtype uint8.
    """
    import cv2  # noqa: PLC0415

    pre = percentile_normalise(frames, clip_percentiles=clip_percentiles)
    gray, was_rgb = _to_float_gray(pre)
    gray = gray.astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tuple(tile_grid))
    out = np.empty_like(gray)
    for i in range(gray.shape[0]):
        out[i] = clahe.apply(gray[i])
    return _restore_channels(out, was_rgb)


# ---------------------------------------------------------------------------
# Characterisation: recommend which transform fits a shot
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExposureProfile:
    """Per-shot brightness profile used to choose an exposure transform.

    Attributes
    ----------
    sensor_max:
        The brightest pixel seen in the shot — a bit-depth signature (255 = 8-bit,
        1023 = 10-bit).
    mean_brightness:
        Median over frames of the per-frame mean pixel value (raw units).
    p99_brightness:
        Median over frames of the per-frame 99th-percentile pixel value.
    saturated_fraction:
        Fraction of pixels at/above 99.5 % of ``sensor_max`` (saturation /
        bright-outlier load — high values are why global min/max fails).
    low_contrast:
        True when the bulk signal occupies a small fraction of the sensor range
        (``p99_brightness / sensor_max`` small) — the CLAHE-candidate signature.
    recommended:
        ``"percentile"`` (default) or ``"clahe"`` (low-contrast shots).
    """

    sensor_max: float
    mean_brightness: float
    p99_brightness: float
    saturated_fraction: float
    low_contrast: bool
    recommended: str


def profile_exposure(
    frames: np.ndarray,
    *,
    low_contrast_p99_frac: float = 0.25,
) -> ExposureProfile:
    """Profile a shot's brightness + recommend a transform.

    ``frames`` is the raw ``(T,H,W)`` (or ``(T,H,W,3)``) camera stack.  Recommends
    ``"clahe"`` when the plasma signal is low-contrast (the per-frame 99th
    percentile sits below ``low_contrast_p99_frac`` of the sensor max — the bulk
    signal uses only a sliver of the range, so per-frame adaptive equalisation
    helps), else ``"percentile"`` (the robust default).
    """
    f, _was_rgb = _to_float_gray(frames)
    sensor_max = float(f.max()) if f.size else 0.0
    per_frame_mean = (
        f.reshape(f.shape[0], -1).mean(axis=1) if f.size else np.array([0.0])
    )
    per_frame_p99 = (
        np.percentile(f.reshape(f.shape[0], -1), 99, axis=1)
        if f.size
        else np.array([0.0])
    )
    mean_b = float(np.median(per_frame_mean))
    p99_b = float(np.median(per_frame_p99))
    if sensor_max > 0:
        sat_thresh = 0.995 * sensor_max
        sat_frac = float(np.mean(f >= sat_thresh))
        low_contrast = (p99_b / sensor_max) < float(low_contrast_p99_frac)
    else:
        sat_frac = 0.0
        low_contrast = False
    recommended = "clahe" if low_contrast else "percentile"
    return ExposureProfile(
        sensor_max=sensor_max,
        mean_brightness=mean_b,
        p99_brightness=p99_b,
        saturated_fraction=sat_frac,
        low_contrast=low_contrast,
        recommended=recommended,
    )


def balance_exposure(
    frames: np.ndarray,
    *,
    strategy: str = "percentile",
    clip_percentiles: tuple[float, float] = DEFAULT_CLIP_PERCENTILES,
) -> np.ndarray:
    """Apply the named exposure-balancing strategy to a shot's frames.

    ``strategy`` is ``"percentile"`` (default, robust), ``"clahe"`` (per-frame
    adaptive, for low-contrast shots), or ``"auto"`` (profile the shot and pick).
    ``"global"`` reproduces the legacy v0 min/max stretch for an A/B comparison.
    Returns the same shape/layout as ``frames``, dtype uint8.
    """
    s = str(strategy).lower()
    if s == "auto":
        s = profile_exposure(frames).recommended
        # The encode env may lack OpenCV; degrade a CLAHE recommendation to the
        # pure-numpy percentile transform rather than crash the encode.
        if s == "clahe" and not cv2_available():
            s = "percentile"
    if s == "percentile":
        return percentile_normalise(frames, clip_percentiles=clip_percentiles)
    if s == "clahe":
        return clahe_normalise(frames, clip_percentiles=clip_percentiles)
    if s == "global":
        f, was_rgb = _to_float_gray(frames)
        lo, hi = float(f.min()), float(f.max())
        if hi <= lo:
            return _restore_channels(np.zeros(f.shape, dtype=np.uint8), was_rgb)
        u8 = ((f - lo) * 255.0 / (hi - lo)).clip(0, 255).astype(np.uint8)
        return _restore_channels(u8, was_rgb)
    raise ValueError(
        f"unknown exposure strategy {strategy!r}; expected percentile|clahe|auto|global"
    )


__all__ = [
    "DEFAULT_CLIP_PERCENTILES",
    "ExposureProfile",
    "balance_exposure",
    "clahe_normalise",
    "cv2_available",
    "percentile_normalise",
    "profile_exposure",
]
