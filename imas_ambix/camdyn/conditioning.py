"""Actuator / scalar conditioning loaders held to the camera frame times.

The dynamics arm is action-conditioned: alongside the clipped camera
token stream it sees the machine actuator vector and a few plasma
scalars, resampled/held to the rbb frame times in PHYSICAL units (A, MW,
electrons/s, m⁻²) — never raw DAC counts.  Physical units serve the
invariant-coordinates hook (the same channel means the same thing across
shots and machines).

Channel set (plan §4a / conditioning-set decision)
---------------------------------------------------
The FULL conditioning set is built here so any later
``conditioning-set`` decision (full actuator vector / Ip+ne / none /
+Dα) is a pure subset selection — D0 does not pre-empt that decision.

* ``amc`` coil / feed / case / sol / tf currents + plasma current (kA →
  A) — the poloidal-field + solenoid + toroidal-field actuators.
* ``anb`` beam powers (MW) — south + south-west sum + total.
* ``aga`` gas-puff flows (electrons/s) — inboard total/upper/lower
  (esp. inboard: the bright-spot cause) + outboard total.
* ``amc`` plasma current (kA → A) and ``ane`` line-integrated density
  (m⁻²) as the minimal scalar pair.
* ``ada`` Dα integrated — exposed as OPTIONAL conditioning, clearly
  separable, because it is a **W3 probe target**.  Off by default.

LEAKAGE BAN (hard)
------------------
EFIT (``efm``), Solov'ev (``esm``) and pulse-schedule (``xdc``) signals
are BANNED as inputs/conditioning everywhere: they embed the
reconstruction the model is supposed to produce.
:func:`assert_no_leakage_sources` enforces this and is called inside
:func:`load_conditioning`.

Resampling
----------
Each actuator lives on its own (faster) time grid; we **hold** the most
recent sample at or before each camera frame time (zero-order hold,
causal — no future leakage) via ``searchsorted``.  Frames before the
actuator record starts get the first sample and a missingness flag.
A per-channel float ``missing`` array (1.0 where the held value is a
fill / the channel is absent) accompanies the values so the model can
learn to ignore absent actuators.

The camera token stream is held NATIVE (no resampling) — only these
conditioning traces are placed on the frame grid; see the package
docstring's time-grid recommendation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Leakage ban
# ---------------------------------------------------------------------------

BANNED_CONDITIONING_SOURCES: frozenset[str] = frozenset({"efm", "esm", "xdc"})
"""Sources that embed the reconstruction → BANNED as conditioning/inputs.

``efm`` = EFIT equilibrium, ``esm`` = Solov'ev equilibrium, ``xdc`` =
pulse-schedule shape targets.  Same ban class throughout the project
(S9/S12 leakage findings)."""


def assert_no_leakage_sources(sources) -> None:
    """Raise ``ValueError`` if any banned (leaking) source is present."""
    bad = sorted(set(sources) & BANNED_CONDITIONING_SOURCES)
    if bad:
        raise ValueError(
            f"leakage: conditioning requested banned source(s) {bad}; "
            f"EFIT/Solov'ev/pulse-schedule embed the reconstruction and are "
            f"banned everywhere ({sorted(BANNED_CONDITIONING_SOURCES)})."
        )


# ---------------------------------------------------------------------------
# Channel spec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConditioningChannel:
    """One physical conditioning channel.

    Attributes
    ----------
    key:
        Stable channel name used in the conditioning vector / metadata.
    source:
        Level-1 source group (``amc`` / ``anb`` / ``aga`` / ``ane`` /
        ``ada``).
    array:
        Array name within the source group.
    unit:
        PHYSICAL unit after :attr:`scale` is applied.
    scale:
        Multiplicative factor from the stored value to physical SI-ish
        units (e.g. kA→A is 1e3).
    is_probe_target:
        True for channels (Dα) that are W3 probe targets; off by default
        and kept clearly separable.
    """

    key: str
    source: str
    array: str
    unit: str
    scale: float = 1.0
    is_probe_target: bool = False


# amc: stored in kA (verified) → A is ×1e3.  We keep the physically
# meaningful poloidal-field coil currents, the solenoid and TF currents,
# and the plasma current.  Feed/case currents are included as they are
# the actual actuator drive (coil + feed); per-shot missing ones get a flag.
_AMC_COIL_KEYS = [
    "p2iu_coil_current",
    "p2il_coil_current",
    "p2ou_coil_current",
    "p2ol_coil_current",
    "p3u_coil_current",
    "p3l_coil_current",
    "p4u_coil_current",
    "p4l_coil_current",
    "p5u_coil_current",
    "p5l_coil_current",
    "p6u_current",
    "p6l_current",
]

CONDITIONING_CHANNELS: tuple[ConditioningChannel, ...] = (
    # --- poloidal-field coil currents (kA → A) ---
    *[ConditioningChannel(k, "amc", k, "A", scale=1e3) for k in _AMC_COIL_KEYS],
    # --- solenoid + toroidal field (kA → A) ---
    ConditioningChannel("sol_current", "amc", "sol_current", "A", scale=1e3),
    ConditioningChannel("tf_current", "amc", "tf_current", "A", scale=1e3),
    # --- plasma current (kA → A) ---
    ConditioningChannel("plasma_current", "amc", "plasma_current", "A", scale=1e3),
    # --- NBI beam powers (MW) ---
    ConditioningChannel("nbi_ss_sum_power", "anb", "ss_sum_power", "MW"),
    ConditioningChannel("nbi_sw_sum_power", "anb", "sw_sum_power", "MW"),
    ConditioningChannel("nbi_tot_sum_power", "anb", "tot_sum_power", "MW"),
    # --- gas-puff flows (electrons/s) — inboard is the bright-spot cause ---
    ConditioningChannel("gas_inboard_total", "aga", "inboard_total", "electrons/s"),
    ConditioningChannel("gas_inboard_upper", "aga", "inboard_upper", "electrons/s"),
    ConditioningChannel("gas_inboard_lower", "aga", "inboard_lower", "electrons/s"),
    ConditioningChannel("gas_outboard_total", "aga", "outboard_total", "electrons/s"),
    # --- line-integrated density (m^-2) ---
    ConditioningChannel("ne_line_integrated", "ane", "density", "m^-2"),
)
"""The FULL physical conditioning channel set (probe targets excluded)."""

DALPHA_PROBE_CHANNEL = ConditioningChannel(
    "dalpha_integrated", "ada", "dalpha_integrated", "", is_probe_target=True
)
"""Dα integrated — optional conditioning, default OFF (it is a W3 probe target)."""


# ---------------------------------------------------------------------------
# Conditioning sample
# ---------------------------------------------------------------------------


@dataclass
class ConditioningSample:
    """Per-frame conditioning held to the camera frame times.

    Attributes
    ----------
    shot_id:
        Source shot.
    frame_time:
        ``(n_frames,)`` camera frame timestamps (s) the channels were
        held to.
    channel_keys:
        ``(C,)`` ordered channel names (matches ``values`` columns).
    units:
        ``(C,)`` physical units per channel.
    values:
        ``(n_frames, C)`` float32 physical values (zero-order hold).
    missing:
        ``(n_frames, C)`` float32 in {0,1}; 1.0 where the value is a fill
        (channel absent for this shot, or frame precedes the record).
    """

    shot_id: int
    frame_time: np.ndarray
    channel_keys: list[str]
    units: list[str]
    values: np.ndarray
    missing: np.ndarray

    def as_dict(self) -> dict:
        return {
            "shot_id": int(self.shot_id),
            "frame_time": self.frame_time,
            "channel_keys": list(self.channel_keys),
            "units": list(self.units),
            "values": self.values,
            "missing": self.missing,
        }


# ---------------------------------------------------------------------------
# Zero-order-hold resample (causal)
# ---------------------------------------------------------------------------


def resample_to_frames(
    sig_time: np.ndarray,
    sig_value: np.ndarray,
    frame_time: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Causal zero-order hold of ``sig_value`` onto ``frame_time``.

    For each frame time, take the most recent signal sample at or before
    it (``searchsorted`` 'right' − 1).  Frames before the signal record
    starts take the first sample and are flagged missing.

    Returns ``(held, missing)`` both shape ``(n_frames,)``; ``missing``
    is a float {0,1} flag.
    """
    sig_time = np.asarray(sig_time, dtype=np.float64).reshape(-1)
    sig_value = np.asarray(sig_value, dtype=np.float64).reshape(-1)
    ft = np.asarray(frame_time, dtype=np.float64).reshape(-1)
    n = ft.shape[0]
    if sig_time.size == 0 or sig_value.size == 0:
        return np.zeros(n, dtype=np.float32), np.ones(n, dtype=np.float32)

    # Sort the signal by time (defensive — most records are monotone).
    order = np.argsort(sig_time)
    st = sig_time[order]
    sv = sig_value[order]

    idx = np.searchsorted(st, ft, side="right") - 1
    before = idx < 0  # frame precedes the record start
    idx = np.clip(idx, 0, sv.size - 1)
    held = sv[idx]
    missing = before.astype(np.float32)
    # NaN samples → missing
    nan_held = ~np.isfinite(held)
    held = np.where(nan_held, 0.0, held)
    missing = np.maximum(missing, nan_held.astype(np.float32))
    return held.astype(np.float32), missing


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def _open_level1(level1_path):
    import zarr  # noqa: PLC0415

    try:
        return zarr.open_group(str(level1_path), mode="r")
    except Exception as e:  # pragma: no cover - corpus robustness
        logger.debug("Cannot open %s: %s", level1_path, e)
        return None


def _read_source_signal(store, source: str, array: str):
    """Return ``(time, value)`` for ``source/array`` or ``(None, None)``."""
    try:
        if source not in set(store.group_keys()):
            return None, None
        grp = store[source]
        keys = set(grp.array_keys())
        if array not in keys or "time" not in keys:
            return None, None
        return np.asarray(grp["time"]), np.asarray(grp[array])
    except Exception:  # pragma: no cover - corpus robustness
        return None, None


def load_conditioning(
    level1_path,
    frame_time: np.ndarray,
    shot_id: int,
    *,
    channels: tuple[ConditioningChannel, ...] = CONDITIONING_CHANNELS,
    include_dalpha: bool = False,
) -> ConditioningSample:
    """Build the per-frame physical conditioning sample for one window.

    Parameters
    ----------
    level1_path:
        V2 level-1 Zarr root for the shot (or None → all-missing sample).
    frame_time:
        ``(n_frames,)`` camera frame timestamps (s) to hold channels to.
    shot_id:
        For provenance.
    channels:
        Channel spec; defaults to the FULL set (probe targets excluded).
    include_dalpha:
        Append the Dα probe-target channel (clearly separable, default
        off).  Dα is a W3 target — keep it out of the default
        conditioning to avoid leaking a probe target into the inputs.

    Raises
    ------
    ValueError
        If any requested channel's source is in
        :data:`BANNED_CONDITIONING_SOURCES`.
    """
    chans = list(channels)
    if include_dalpha:
        chans = chans + [DALPHA_PROBE_CHANNEL]

    assert_no_leakage_sources(c.source for c in chans)

    ft = np.asarray(frame_time, dtype=np.float64).reshape(-1)
    n = ft.shape[0]
    c = len(chans)
    values = np.zeros((n, c), dtype=np.float32)
    missing = np.ones((n, c), dtype=np.float32)

    store = _open_level1(level1_path) if level1_path is not None else None
    for j, chan in enumerate(chans):
        if store is None:
            continue
        st, sv = _read_source_signal(store, chan.source, chan.array)
        if st is None or sv is None or sv.ndim != 1:
            continue
        held, miss = resample_to_frames(st, sv * chan.scale, ft)
        values[:, j] = held
        missing[:, j] = miss

    return ConditioningSample(
        shot_id=int(shot_id),
        frame_time=ft,
        channel_keys=[c.key for c in chans],
        units=[c.unit for c in chans],
        values=values,
        missing=missing,
    )
