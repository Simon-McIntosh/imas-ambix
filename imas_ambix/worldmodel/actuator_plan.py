"""Demanded actuator-plan conditioning for the camera world model (the drive surface).

Why this exists (the controllability bridge)
--------------------------------------------
The signal-conditioned camera transformer
(:mod:`imas_ambix.worldmodel.spacetime_model_v2`) conditions on the MEASURED
diagnostic streams — magnetics, interferometer, soft_x_rays, the gas-puff flow,
…  Those measured streams are mutually REDUNDANT (the realised plasma state is
written into all of them at once), so zeroing one barely moves the dream and
classifier-free guidance has nothing causal to amplify: the model is a good
FORECASTER but it is not DRIVEABLE.  To make it driveable the model must
condition on the DEMANDED actuator PLAN — what the operator asked the machine to
do — not on the realised observations.

This module reads that actuator plan for a camera window.  It reuses the
camera-dynamics actuator-vector loader
(:func:`imas_ambix.camdyn.conditioning.load_conditioning`) verbatim: the same
physical-unit channel set (``amc`` coil/solenoid/TF currents + plasma current,
``anb`` NBI beam powers, ``aga`` gas-puff flows, ``ane`` line-integrated
density), the same causal zero-order hold, and the same leakage ban
(EFIT/Solov'ev/pulse-schedule reconstruction sources are banned).  The actuator
vector is CONTINUOUS physical floats — unlike the tokenised measured streams it
is not embedded through a vocabulary table; the model projects it with a linear
encoder (:mod:`imas_ambix.worldmodel.controllable_model`).

A plan for a window
-------------------
The plan is sub-sampled to ``n_plan`` evenly-spaced steps spanning the camera
window's frame-time span (mirroring how the tokenised pulse-schedule plan is
prepended in v1).  Each step is the actuator vector held (causally) to that
step's time.  A per-channel ``missing`` flag rides alongside so the model can
learn to ignore an absent actuator.

Channel normalisation
---------------------
The raw channels span wildly different physical scales (coil currents ~1e5 A,
NBI ~1 MW, gas ~1e21 electrons/s, density ~1e19 m^-2).  A fixed per-channel
``log1p``-of-magnitude + sign normaliser maps every channel to an O(1) range so
the linear encoder is well-conditioned, WITHOUT needing a corpus-statistics
pass (the gate overfits a handful of shots).  The transform is monotone and
sign-preserving, so scaling a command up still scales the normalised drive up —
the property the controllability gate exercises when it amplifies the gas-puff
or NBI command.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from imas_ambix.camdyn.conditioning import (
    CONDITIONING_CHANNELS,
    ConditioningChannel,
    load_conditioning,
)
from imas_ambix.worldmodel.spacetime_dataset import REFERENCE_CAMERA

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from imas_ambix.worldmodel.spacetime_dataset import SpacetimeSample

logger = logging.getLogger(__name__)

#: The default actuator-plan channel set — the FULL physical actuator vector
#: (coil/solenoid/TF currents + plasma current + NBI powers + gas-puff flows +
#: line-integrated density).  Reused verbatim from the camera-dynamics loader so
#: the plan means the same thing here as in camera-dynamics-wm-v0.
ACTUATOR_CHANNELS: tuple[ConditioningChannel, ...] = CONDITIONING_CHANNELS

#: Number of actuator channels (the continuous drive-vector width).
N_ACTUATOR_CHANNELS: int = len(ACTUATOR_CHANNELS)

#: Channel keys, in order (matches the columns of the plan value array).
ACTUATOR_CHANNEL_KEYS: tuple[str, ...] = tuple(c.key for c in ACTUATOR_CHANNELS)


def actuator_channel_index(key: str) -> int:
    """Column index of an actuator channel by key (``ValueError`` if absent)."""
    try:
        return ACTUATOR_CHANNEL_KEYS.index(key)
    except ValueError as exc:  # pragma: no cover - defensive
        raise ValueError(
            f"unknown actuator channel {key!r}; have {ACTUATOR_CHANNEL_KEYS}"
        ) from exc


def gas_puff_channel_indices() -> list[int]:
    """Columns of the gas-puff actuator channels (``aga`` source).

    The gas-puff command is the controllability gate's primary lever (the
    inboard puff lights a bright spot — camdyn ``puff_attribution``); the gate
    scales these columns to test whether the decoded camera responds.
    """
    return [i for i, c in enumerate(ACTUATOR_CHANNELS) if c.source == "aga"]


def nbi_channel_indices() -> list[int]:
    """Columns of the NBI beam-power actuator channels (``anb`` source)."""
    return [i for i, c in enumerate(ACTUATOR_CHANNELS) if c.source == "anb"]


def coil_current_channel_indices() -> list[int]:
    """Columns of the PF/CS/TF COIL-CURRENT actuator channels.

    These are the ``amc`` source channels EXCEPT the plasma current — the
    poloidal-field coil currents, the central solenoid, and the toroidal-field
    current.  Their RAMP RATE ``|dI/dt|`` is the excitation signal that makes the
    control->camera map identifiable (the operator drives these; the plasma
    current is the RESPONSE, not a command), so the dynamic-segment selector
    scores variation on exactly these columns.
    """
    return [
        i
        for i, c in enumerate(ACTUATOR_CHANNELS)
        if c.source == "amc" and c.key != "plasma_current"
    ]


def plasma_current_channel_index() -> int | None:
    """Column of the plasma-current channel (``amc`` ``plasma_current``), or None.

    Ip is the plasma RESPONSE, not an actuator command — it is the
    plasma-presence signal (segment selection requires ``max|Ip|`` above a
    threshold), kept separate from the coil-current drive columns.
    """
    for i, c in enumerate(ACTUATOR_CHANNELS):
        if c.source == "amc" and c.key == "plasma_current":
            return i
    return None


def normalise_actuator_values(values: np.ndarray) -> np.ndarray:
    """Sign-preserving ``log1p``-of-magnitude normaliser to an O(1) range.

    ``values`` is ``(..., C)`` raw physical actuator values.  Returns the same
    shape, with each entry mapped by ``sign(x) * log1p(|x|)``.  This compresses
    the very different physical scales (currents ~1e5, gas ~1e21) into a small
    common range so the linear encoder is well-conditioned, is monotone and
    sign-preserving (so scaling a command up scales the drive up — the property
    the gate's amplification test relies on), and needs no corpus statistics.
    """
    v = np.asarray(values, dtype=np.float64)
    return (np.sign(v) * np.log1p(np.abs(v))).astype(np.float32)


@dataclass
class ActuatorPlan:
    """The demanded actuator plan for one camera window.

    Attributes
    ----------
    values:
        ``(n_plan, C)`` float32 NORMALISED actuator values (the always-on drive
        surface the model conditions on).
    missing:
        ``(n_plan, C)`` float32 in {0, 1}; 1.0 where the channel is a fill /
        absent for the shot, so the encoder can learn to ignore it.
    channel_keys:
        ``(C,)`` ordered channel names (matches the ``values`` columns).
    raw_values:
        ``(n_plan, C)`` float32 RAW physical values (pre-normalisation) — kept so
        the controllability gate can scale a specific channel's COMMAND in
        physical units and re-normalise, rather than perturbing the normalised
        space.
    """

    values: np.ndarray
    missing: np.ndarray
    channel_keys: list[str]
    raw_values: np.ndarray

    @property
    def n_plan(self) -> int:
        return int(self.values.shape[0])

    @property
    def n_channels(self) -> int:
        return int(self.values.shape[1])


def read_window_actuator_plan(
    shot_id: int,
    sample: SpacetimeSample,
    n_plan: int,
    *,
    channels: Sequence[ConditioningChannel] = ACTUATOR_CHANNELS,
    level1_path: Path | None = None,
) -> ActuatorPlan:
    """Read the demanded actuator plan for a camera window, sub-sampled to ``n_plan``.

    The window's time span comes from ``sample.frame_time`` (the camera frame
    timestamps).  The actuator vector is sub-sampled to ``n_plan`` evenly-spaced
    positions across that span and read with the causal zero-order hold of
    :func:`imas_ambix.camdyn.conditioning.load_conditioning` (no future leakage),
    then sign-preserving ``log1p`` normalised.  When the shot's level-1 store is
    unreadable every channel is flagged missing (an all-missing plan — the model
    then conditions on the optional measured observations only).

    Returns an :class:`ActuatorPlan`.
    """
    from imas_ambix.camdyn.dataset import level1_shot_path  # noqa: PLC0415

    n_plan = int(n_plan)
    chans = list(channels)
    keys = [c.key for c in chans]
    if n_plan <= 0:
        empty = np.zeros((0, len(chans)), dtype=np.float32)
        return ActuatorPlan(
            values=empty, missing=empty.copy(), channel_keys=keys, raw_values=empty
        )

    ftime = np.asarray(sample.frame_time, dtype=np.float64)
    if ftime.size < 2 or not (float(ftime.max()) > float(ftime.min())):
        # degenerate time axis — all-missing plan.
        miss = np.ones((n_plan, len(chans)), dtype=np.float32)
        zeros = np.zeros((n_plan, len(chans)), dtype=np.float32)
        return ActuatorPlan(
            values=zeros, missing=miss, channel_keys=keys, raw_values=zeros.copy()
        )

    t0, t1 = float(ftime.min()), float(ftime.max())
    grid = np.linspace(t0, t1, n_plan, dtype=np.float64)

    lpath = level1_path
    if lpath is None:
        try:
            lpath = level1_shot_path(int(shot_id))
        except Exception:  # noqa: BLE001
            lpath = None

    cond = load_conditioning(
        lpath, grid, int(shot_id), channels=tuple(chans), include_dalpha=False
    )
    raw = np.asarray(cond.values, dtype=np.float32)
    return ActuatorPlan(
        values=normalise_actuator_values(raw),
        missing=np.asarray(cond.missing, dtype=np.float32),
        channel_keys=list(cond.channel_keys),
        raw_values=raw,
    )


def find_transient_window(
    shot_id: int,
    span: int,
    *,
    camera: str = REFERENCE_CAMERA,
    token_root: Path | None = None,
    channels: Sequence[ConditioningChannel] = ACTUATOR_CHANNELS,
    level1_path: Path | None = None,
    min_variation: float = 1e-6,
) -> tuple[int | None, float]:
    """Find the camera-frame window where the actuator PLAN varies most.

    The controllability gate is only fair on a TRANSIENT window — one where the
    demanded actuator plan actually moves (a ramp-up / ramp-down / a gas-puff or
    NBI toggle).  On a flat-top window the plan is ~constant, so there is no
    control variation to learn from or respond to and the gate would FALSELY fail.
    This slides a ``span``-frame window over the whole shot, holds the
    (normalised) actuator vector to each window's camera frame times, and returns
    the start frame whose actuator drive has the largest summed per-channel
    standard deviation (the strongest combined ramp / toggle).

    Returns ``(start_frame, variation_score)``.  ``start_frame`` is ``None`` when
    the plan is flat everywhere (max windowed variation ``< min_variation``) or
    the shot / actuator record is unreadable — the caller then falls back to the
    centred window and the result is honestly a "no in-window actuator
    transition" case.  ``variation_score`` is the best window's summed std.
    """
    from imas_ambix.camdyn.dataset import level1_shot_path  # noqa: PLC0415
    from imas_ambix.worldmodel.spacetime_dataset import _frame_times  # noqa: PLC0415

    ftime = _frame_times(int(shot_id), camera, token_root=token_root)
    if ftime is None:
        return None, 0.0
    ftime = np.asarray(ftime, dtype=np.float64)
    n = ftime.shape[0]
    if n < int(span):
        return None, 0.0

    lpath = level1_path
    if lpath is None:
        try:
            lpath = level1_shot_path(int(shot_id))
        except Exception:  # noqa: BLE001
            return None, 0.0

    # Hold the actuator vector onto EVERY camera frame once, then score sliding
    # windows on that dense series — one load, cheap windowed std afterwards.
    cond = load_conditioning(
        lpath, ftime, int(shot_id), channels=tuple(channels), include_dalpha=False
    )
    vals = normalise_actuator_values(np.asarray(cond.values, dtype=np.float32))
    present = np.asarray(cond.missing, dtype=np.float32).mean(axis=0) < 1.0
    if not bool(present.any()):
        return None, 0.0
    vals = vals[:, present]  # only channels that exist for the shot

    span = int(span)
    best_start, best_score = 0, -1.0
    stride = max(1, span // 2)
    last_start = n - span
    for start in range(0, last_start + 1, stride):
        block = vals[start : start + span]
        score = float(np.std(block, axis=0).sum())
        if score > best_score:
            best_score, best_start = score, start
    if best_score < float(min_variation):
        return None, best_score
    return int(best_start), best_score


@dataclass
class ExcitationWindow:
    """The most dynamically-excited window of one shot's camera recording.

    Attributes
    ----------
    start_frame:
        First camera frame of the chosen window (``None`` when no window meets
        the plasma-presence / excitation bar — the shot is then rejected).
    score:
        Excitation score of the chosen window: the time-mean over the window of
        the summed per-coil ``|dI/dt|`` (physical units kA/s after the channel
        scale), de-weighted on any disruption tail.  Higher = more
        persistently-exciting coil drive.
    max_abs_ip:
        ``max|Ip|`` over the chosen window (physical units, A after the channel
        scale) — the plasma-presence measure.
    present_fraction:
        Fraction of the window's frames with ``|Ip|`` above the presence
        threshold.
    reason:
        Why the shot was rejected (``""`` when accepted) — one of
        ``"unreadable"``, ``"too_short"``, ``"no_plasma"``, ``"flat_drive"``.
    """

    start_frame: int | None
    score: float
    max_abs_ip: float
    present_fraction: float
    reason: str = ""


def find_excitation_window(
    shot_id: int,
    span: int,
    *,
    camera: str = REFERENCE_CAMERA,
    token_root: Path | None = None,
    level1_path: Path | None = None,
    ip_present_threshold: float = 2.0e4,
    min_present_fraction: float = 0.5,
    min_ramp_rate: float = 1.0e3,
    deprioritise_disruption: bool = True,
) -> ExcitationWindow:
    """Find the camera-frame window with the strongest COIL-CURRENT excitation.

    This is the dynamic-segment selector for the curated excitation corpus.
    Unlike :func:`find_transient_window` (which scores the summed per-channel
    *level* standard deviation of the FULL normalised actuator vector — a
    transient/flat-top discriminator for the controllability gate), this scores
    the **coil-current ramp rate** ``|dI/dt|`` of the ``amc`` PF/CS/TF channels
    (the operator's drive — :func:`coil_current_channel_indices`), which is the
    persistent-excitation signal that makes the control->camera map identifiable
    (per the control-conditioning survey §4).  It additionally enforces
    **plasma-presence** (``max|Ip|`` over the window must clear
    ``ip_present_threshold`` and a fraction of frames must be present) so a
    vacuum / no-plasma high-ramp window is rejected, and it **de-prioritises a
    disruption tail** (a window dominated by a sharp ``|Ip|`` collapse is
    down-weighted, since disruption tails are abundant but "perhaps not that
    useful" for learning the driven dynamics).

    The whole shot's coil channels and Ip are held onto EVERY camera frame once;
    ``|dI/dt|`` is finite-differenced on the (possibly non-uniform) frame grid;
    sliding ``span``-frame windows are scored on the time-mean of the summed
    coil ramp rate (a level-invariant measure, so a flat-but-large current does
    NOT score), masked to plasma-present frames and de-weighted on the tail.

    Returns an :class:`ExcitationWindow`.  ``start_frame`` is ``None`` with a
    populated ``reason`` when the shot is rejected.  Backward-compatible
    addition — does not touch :func:`find_transient_window`.
    """
    from imas_ambix.camdyn.dataset import level1_shot_path  # noqa: PLC0415
    from imas_ambix.worldmodel.spacetime_dataset import _frame_times  # noqa: PLC0415

    span = int(span)
    ftime = _frame_times(int(shot_id), camera, token_root=token_root)
    if ftime is None:
        return ExcitationWindow(None, 0.0, 0.0, 0.0, "unreadable")
    ftime = np.asarray(ftime, dtype=np.float64)
    n = ftime.shape[0]
    if n < span or span < 2:
        return ExcitationWindow(None, 0.0, 0.0, 0.0, "too_short")

    lpath = level1_path
    if lpath is None:
        try:
            lpath = level1_shot_path(int(shot_id))
        except Exception:  # noqa: BLE001
            return ExcitationWindow(None, 0.0, 0.0, 0.0, "unreadable")

    # Hold the FULL actuator vector (coils + Ip) onto every camera frame once.
    cond = load_conditioning(
        lpath, ftime, int(shot_id), channels=ACTUATOR_CHANNELS, include_dalpha=False
    )
    raw = np.asarray(cond.values, dtype=np.float64)  # (n, C) physical units
    miss = np.asarray(cond.missing, dtype=np.float64)  # (n, C)

    coil_cols = coil_current_channel_indices()
    ip_col = plasma_current_channel_index()
    if not coil_cols or ip_col is None:
        return ExcitationWindow(None, 0.0, 0.0, 0.0, "unreadable")

    # --- coil ramp rate |dI/dt| per frame, summed over present coil channels --
    coil_present = miss[:, coil_cols].mean(axis=0) < 1.0  # which coils exist
    cols = [c for c, ok in zip(coil_cols, coil_present, strict=True) if ok]
    if not cols:
        return ExcitationWindow(None, 0.0, 0.0, 0.0, "flat_drive")
    coil = raw[:, cols]  # (n, k) physical current
    dt = np.diff(ftime)
    dt = np.where(dt > 0, dt, np.nan)
    # forward |dI/dt|, last frame repeats the previous rate (no successor)
    didt = np.full_like(coil, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = np.diff(coil, axis=0) / dt[:, None]
    didt[:-1] = rate
    didt[-1] = rate[-1] if rate.shape[0] else 0.0
    ramp = np.abs(np.where(np.isfinite(didt), didt, 0.0)).sum(axis=1)  # (n,) kA/s-ish

    # --- plasma presence on the frame grid ---
    absip = np.abs(raw[:, ip_col])
    ip_ok = miss[:, ip_col] < 1.0
    absip = np.where(ip_ok, absip, 0.0)
    present = absip >= float(ip_present_threshold)

    # --- disruption-tail de-weight ---
    # A frame on a sharp |Ip| COLLAPSE (large negative dIp/dt while still
    # carrying current) is a disruption-tail frame: keep it (it is still
    # plasma-present) but halve its excitation contribution so a window cannot
    # win on the disruption spike alone.  Detected as the |Ip| decreasing faster
    # than 20% of the in-window peak per ms while above presence.
    weight = np.ones(n, dtype=np.float64)
    if deprioritise_disruption and absip.size and float(absip.max()) > 0:
        dip = np.full(n, 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.diff(absip) / dt
        dip[:-1] = np.where(np.isfinite(r), r, 0.0)
        collapse_rate = 0.2 * float(absip.max()) / 1.0e-3  # 20% of peak per ms
        is_tail = (dip < -collapse_rate) & present
        weight[is_tail] = 0.5

    scored = ramp * weight

    # --- slide windows, score time-mean excitation on plasma-present frames ---
    stride = max(1, span // 2)
    last_start = n - span
    best_start, best_score, best_maxip, best_pf = None, -1.0, 0.0, 0.0
    for start in range(0, last_start + 1, stride):
        sl = slice(start, start + span)
        w_present = present[sl]
        pf = float(w_present.mean())
        w_absip = absip[sl]
        max_absip = float(w_absip.max()) if w_absip.size else 0.0
        if max_absip < float(ip_present_threshold) or pf < float(min_present_fraction):
            continue  # not enough plasma in this window
        # time-mean excitation over the window (level-invariant; flat -> ~0)
        score = float(scored[sl].mean())
        if score > best_score:
            best_start, best_score, best_maxip, best_pf = start, score, max_absip, pf

    if best_start is None:
        # No window cleared the plasma bar anywhere in the recording.
        max_overall = float(absip.max()) if absip.size else 0.0
        reason = (
            "no_plasma" if max_overall < float(ip_present_threshold) else "no_plasma"
        )
        return ExcitationWindow(None, 0.0, max_overall, 0.0, reason)
    if best_score < float(min_ramp_rate):
        return ExcitationWindow(None, best_score, best_maxip, best_pf, "flat_drive")
    return ExcitationWindow(int(best_start), best_score, best_maxip, best_pf, "")


def scale_plan_channels(
    plan: ActuatorPlan,
    channel_indices: Sequence[int],
    factor: float,
) -> ActuatorPlan:
    """Return a copy of the plan with the given channels' COMMAND scaled.

    Scales the RAW physical command of the selected channels by ``factor``, then
    re-normalises — so ``factor > 1`` amplifies the demand and ``factor == 0``
    silences it (a zeroed-command counterfactual).  ``missing`` is preserved; a
    missing channel stays missing.  This is the controllability gate's lever: it
    edits the DEMAND (e.g. "fire the gas puff harder") and re-runs the dream.
    """
    idx = list(int(i) for i in channel_indices)
    raw = plan.raw_values.copy()
    if idx:
        raw[:, idx] = raw[:, idx] * float(factor)
    return ActuatorPlan(
        values=normalise_actuator_values(raw),
        missing=plan.missing.copy(),
        channel_keys=list(plan.channel_keys),
        raw_values=raw,
    )


def zero_plan(plan: ActuatorPlan) -> ActuatorPlan:
    """Return a copy of the plan with EVERY channel's command zeroed.

    The no-drive counterfactual: a silent actuator plan.  ``missing`` is
    preserved (a present-but-zero channel reads as a real zero command, the
    drive surface the model would see with no actuation demanded).
    """
    zeros = np.zeros_like(plan.raw_values)
    return ActuatorPlan(
        values=normalise_actuator_values(zeros),
        missing=plan.missing.copy(),
        channel_keys=list(plan.channel_keys),
        raw_values=zeros,
    )


__all__ = [
    "ACTUATOR_CHANNELS",
    "ACTUATOR_CHANNEL_KEYS",
    "N_ACTUATOR_CHANNELS",
    "ActuatorPlan",
    "ExcitationWindow",
    "actuator_channel_index",
    "coil_current_channel_indices",
    "find_excitation_window",
    "find_transient_window",
    "gas_puff_channel_indices",
    "nbi_channel_indices",
    "normalise_actuator_values",
    "plasma_current_channel_index",
    "read_window_actuator_plan",
    "scale_plan_channels",
    "zero_plan",
]
