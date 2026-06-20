"""Plasma-presence frame filter for the curated dynamic-excitation corpus.

A curated training window must show PLASMA, not vacuum.  The rbb camera records
over a short slice of the shot, and ~10 % of recordings are essentially
plasma-free (measured: the panel's 10th-percentile ``max|Ip|`` is ~5 kA, against
a ~700 kA median — a clean valley between vacuum and real plasma).  Feeding those
vacuum sequences to a "play the plasma" model teaches it nothing about driven
plasma dynamics.

The criterion (justified from the Ip characterisation)
------------------------------------------------------
A window is **plasma-present** when

* its ``max|Ip|`` clears :data:`IP_PRESENT_THRESHOLD_A` (20 kA), AND
* at least :data:`MIN_PRESENT_FRACTION` of its frames individually carry
  ``|Ip|`` above that threshold.

20 kA sits in the valley between vacuum shots (``max|Ip| ~ 5 kA``) and real
plasma (``~700 kA`` median), so it rules out vacuum without rejecting genuine
breakdown/ramp windows — a window is ALLOWED to START at ``Ip ~ 0`` (breakdown
is exactly the dynamic phase we want), provided the plasma forms within it and
the window is mostly present thereafter.  The fraction gate (default 0.5) admits
a window that begins in breakdown (the first frames are sub-threshold) while
rejecting one that only briefly flickers into plasma.

This is the SELECTION-side presence check used to accept/reject a window of a
shot; it reuses the camera-frame actuator loader (the ``amc`` ``plasma_current``
channel) and never reads a banned source.  Backward-compatible new module — it
adds no dependency to the existing dataset path.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from imas_ambix.worldmodel.actuator_plan import (
    ACTUATOR_CHANNELS,
    plasma_current_channel_index,
)

#: Plasma-presence threshold on ``|Ip|`` in AMPERES (physical units, after the
#: ``amc`` kA->A channel scale).  20 kA = 2.0e4 A — see the module docstring for
#: the valley-between-vacuum-and-plasma justification.
IP_PRESENT_THRESHOLD_A: float = 2.0e4

#: Minimum fraction of a window's frames that must individually be plasma-present
#: for the window to count as plasma-present.  0.5 admits a breakdown window
#: (early frames sub-threshold) while rejecting a brief plasma flicker.
MIN_PRESENT_FRACTION: float = 0.5


@dataclass(frozen=True)
class PresenceResult:
    """Outcome of the plasma-presence check on a window's Ip trace.

    Attributes
    ----------
    present:
        True when the window passes both the ``max|Ip|`` and the
        present-fraction gates.
    max_abs_ip:
        ``max|Ip|`` over the window (A).
    present_fraction:
        Fraction of frames with ``|Ip|`` above the threshold.
    n_frames:
        Number of frames inspected.
    """

    present: bool
    max_abs_ip: float
    present_fraction: float
    n_frames: int


def frame_presence_mask(
    abs_ip_a: np.ndarray,
    *,
    threshold_a: float = IP_PRESENT_THRESHOLD_A,
) -> np.ndarray:
    """Per-frame boolean plasma-present mask from a ``|Ip|`` (A) trace."""
    a = np.asarray(abs_ip_a, dtype=np.float64)
    return np.isfinite(a) & (a >= float(threshold_a))


def evaluate_presence(
    abs_ip_a: np.ndarray,
    *,
    threshold_a: float = IP_PRESENT_THRESHOLD_A,
    min_present_fraction: float = MIN_PRESENT_FRACTION,
) -> PresenceResult:
    """Apply the plasma-presence criterion to a window's ``|Ip|`` (A) trace.

    ``abs_ip_a`` is the per-frame ``|Ip|`` in amperes.  Returns a
    :class:`PresenceResult`; ``present`` is True when ``max|Ip|`` clears
    ``threshold_a`` AND at least ``min_present_fraction`` of frames are
    individually present.
    """
    a = np.asarray(abs_ip_a, dtype=np.float64)
    finite = a[np.isfinite(a)]
    n = int(a.shape[0])
    if finite.size == 0 or n == 0:
        return PresenceResult(False, 0.0, 0.0, n)
    mask = frame_presence_mask(a, threshold_a=threshold_a)
    max_abs = float(finite.max())
    frac = float(mask.mean())
    present = (max_abs >= float(threshold_a)) and (frac >= float(min_present_fraction))
    return PresenceResult(present, max_abs, frac, n)


def window_abs_ip(
    shot_id: int,
    frame_time: np.ndarray,
    *,
    level1_path=None,
) -> np.ndarray:
    """Per-frame ``|Ip|`` (A) for a camera window's frame times.

    Holds the ``amc`` ``plasma_current`` channel (physical A) onto ``frame_time``
    with the causal zero-order hold of the conditioning loader.  Frames before
    the Ip record (flagged missing) are returned as NaN so the presence check
    treats them as not-present rather than zero-current.  Returns ``(n_frames,)``.
    """
    from imas_ambix.camdyn.conditioning import load_conditioning  # noqa: PLC0415
    from imas_ambix.camdyn.dataset import level1_shot_path  # noqa: PLC0415

    ip_col = plasma_current_channel_index()
    ft = np.asarray(frame_time, dtype=np.float64)
    if ip_col is None or ft.size == 0:
        return np.full(ft.shape, np.nan)

    lpath = level1_path
    if lpath is None:
        try:
            lpath = level1_shot_path(int(shot_id))
        except Exception:  # noqa: BLE001
            return np.full(ft.shape, np.nan)

    cond = load_conditioning(
        lpath, ft, int(shot_id), channels=ACTUATOR_CHANNELS, include_dalpha=False
    )
    vals = np.asarray(cond.values, dtype=np.float64)[:, ip_col]
    miss = np.asarray(cond.missing, dtype=np.float64)[:, ip_col]
    return np.where(miss < 1.0, np.abs(vals), np.nan)


def is_window_plasma_present(
    shot_id: int,
    frame_time: np.ndarray,
    *,
    level1_path=None,
    threshold_a: float = IP_PRESENT_THRESHOLD_A,
    min_present_fraction: float = MIN_PRESENT_FRACTION,
) -> PresenceResult:
    """Plasma-presence check for a camera window given its frame times."""
    absip = window_abs_ip(shot_id, frame_time, level1_path=level1_path)
    return evaluate_presence(
        absip,
        threshold_a=threshold_a,
        min_present_fraction=min_present_fraction,
    )


__all__ = [
    "IP_PRESENT_THRESHOLD_A",
    "MIN_PRESENT_FRACTION",
    "PresenceResult",
    "evaluate_presence",
    "frame_presence_mask",
    "is_window_plasma_present",
    "window_abs_ip",
]
