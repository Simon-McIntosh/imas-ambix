"""Actuator / scalar conditioning loaders held to the camera frame times.

The dynamics arm is action-conditioned: alongside the clipped camera
token stream it sees the machine actuator vector and a few plasma
scalars, resampled/held to the rbb frame times in PHYSICAL units (A, MW,
electrons/s, m⁻²) — never raw DAC counts.  Physical units serve the
invariant-coordinates hook (the same channel means the same thing across
shots and machines).

Channel set
-----------
The full actuator set is built here so conditioning experiments remain
pure channel selections.  The default set additionally carries one fast
Dα burst-energy scalar; matched controls replace that scalar with either
a causal shuffled history or the slow radial-profile summary.

* ``amc`` coil / feed / case / sol / tf currents + plasma current (kA →
  A) — the poloidal-field + solenoid + toroidal-field actuators.
* ``anb`` beam powers (MW) — south + south-west sum + total.
* ``aga`` gas-puff flows (electrons/s) — inboard total/upper/lower
  (esp. inboard: the bright-spot cause) + outboard total.
* ``amc`` plasma current (kA → A) and ``ane`` line-integrated density
  (m⁻²) as the minimal scalar pair.
* ``xim/da_hm10_t`` fast Dα — reduced to one causal interval-RMS scalar.
* ``ada`` Dα integrated — exposed as optional conditioning, clearly
  separable, because it is a diagnostic probe target.  Off by default.

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
A fast Dα value is instead the RMS of native samples in the closed
inter-frame interval ending at the current frame.  Its shuffled control
can select only summaries from the causal prefix available by that frame.
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
from types import MappingProxyType

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Leakage ban
# ---------------------------------------------------------------------------

BANNED_CONDITIONING_SOURCES: frozenset[str] = frozenset({"efm", "esm", "xdc"})
"""Sources that embed the reconstruction → BANNED as conditioning/inputs.

``efm`` = EFIT equilibrium, ``esm`` = Solov'ev equilibrium, ``xdc`` =
pulse-schedule shape targets.  Same ban class throughout the project
(these fields embed the reconstruction being predicted)."""


def assert_no_leakage_sources(sources) -> None:
    """Raise ``ValueError`` if any banned (leaking) source is present.

    This is the *source-group* guard used by the camera-frame conditioning
    loader, whose channels are pulled from raw L1 source groups (``amc``,
    ``anb``, ``aga``, ``ane``, ``ada``).  At that layer the whole ``efm`` /
    ``esm`` / ``xdc`` source groups are banned: the loader never planned to
    pull a planned XDC waveform, so banning the group is correct *here*.

    The corrected reconstruction-vs-plan principle (planned XDC demands are
    authorised, only reconstructed state and reconstruction residuals are
    banned) is enforced at the **field level** by
    :func:`assert_no_leakage_fields`, which the L2 loader uses.  The two
    guards are complementary: this one keeps existing camera-frame callers
    unchanged; the field-level one admits planned actuators by inspecting
    the per-field ``uda_name``.
    """
    bad = sorted(set(sources) & BANNED_CONDITIONING_SOURCES)
    if bad:
        raise ValueError(
            f"leakage: conditioning requested banned source(s) {bad}; "
            f"EFIT/Solov'ev/pulse-schedule embed the reconstruction and are "
            f"banned everywhere ({sorted(BANNED_CONDITIONING_SOURCES)})."
        )


def assert_no_leakage_fields(fields) -> None:
    """Field-level leakage guard on the reconstruction-vs-plan principle.

    Unlike :func:`assert_no_leakage_sources` (which bans whole source
    groups), this inspects each L2 field's ``uda_name`` and rejects only
    those that carry **code-reconstructed state** or a **reconstruction
    residual** — while *authorising* planned/demanded pulse-schedule
    waveforms (demanded Ip/density, the feed-forward coil voltage, the gas
    valve demands).  This is the corrected layer the L2 loader must use so
    a forward world model can see the control plan without leaking the
    reconstruction it is meant to produce.

    Parameters
    ----------
    fields:
        Iterable of ``(group, var, uda_name)`` triples.

    Raises
    ------
    ValueError
        If any field classifies as ``banned`` (reconstructed equilibrium,
        a reconstruction-derived scalar, a reconstruction-residual XDC
        field, or an ambiguous field defaulted to banned for review).

    Notes
    -----
    Probe-target (Dα) and infra (geometry) fields are *not* leakage and do
    not raise here — but they are also not admissible default inputs; use
    :func:`imas_ambix.data.provenance.is_admissible_input` to select the
    input set.
    """
    from imas_ambix.data.provenance import (  # noqa: PLC0415
        BANNED,
        classify_l2_field,
    )

    bad = []
    for group, var, uda_name in fields:
        fc = classify_l2_field(group, var, uda_name)
        if fc.classification == BANNED:
            bad.append((group, var, uda_name, fc.reason))
    if bad:
        listing = "; ".join(f"{g}.{v} (uda={u!r}): {r}" for g, v, u, r in sorted(bad))
        raise ValueError(
            f"leakage: {len(bad)} field(s) carry reconstructed state / "
            f"reconstruction residuals and are banned as inputs — {listing}"
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
        True for channels used as diagnostic probe targets; off by default
        and kept clearly separable.
    reduction:
        ``"hold"`` for causal zero-order hold or ``"interval_rms"`` for a
        native-sample reduction over the inter-frame interval ending at the
        current camera frame.
    causal_shuffle:
        Replace each reduced value with a deterministic draw from summaries
        already available at that frame.  This is a timing-destruction
        control that never imports a future interval.
    is_fast_input:
        True when the channel carries native fast diagnostic information.
        Such a channel cannot also be a probe target.
    """

    key: str
    source: str
    array: str
    unit: str
    scale: float = 1.0
    is_probe_target: bool = False
    reduction: str = "hold"
    causal_shuffle: bool = False
    is_fast_input: bool = False


class ConditioningChannelCatalog(tuple):
    """Default channels plus named, width-matched control selections.

    Iteration preserves the ordinary default channel tuple used throughout
    the training pipeline.  :meth:`select` replaces its burst scalar with a
    named control while keeping channel count and order stable.
    """

    def __new__(cls, default_channels, burst_variants):
        obj = super().__new__(cls, tuple(default_channels))
        obj._burst_variants = MappingProxyType(dict(burst_variants))
        return obj

    def __reduce__(self):
        """Preserve control selections across spawned data-loader workers."""
        return type(self), (tuple(self), dict(self._burst_variants))

    @property
    def burst_variants(self) -> tuple[str, ...]:
        """Available burst-conditioning selections."""
        return tuple(self._burst_variants)

    def select(self, burst_variant: str = "native") -> tuple[ConditioningChannel, ...]:
        """Return a width-matched channel tuple for ``burst_variant``."""
        try:
            replacement = self._burst_variants[burst_variant]
        except KeyError as exc:
            choices = ", ".join(self._burst_variants)
            raise ValueError(
                f"unknown burst conditioning variant {burst_variant!r}; "
                f"choose {choices}"
            ) from exc
        selected = tuple(self[:-1]) + (replacement,)
        assert_fast_inputs_are_not_probe_targets(selected)
        return selected


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

_ACTUATOR_CONDITIONING_CHANNELS: tuple[ConditioningChannel, ...] = (
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

DALPHA_BURST_CHANNEL = ConditioningChannel(
    "dalpha_burst_rms",
    "xim",
    "da_hm10_t",
    "a.u.",
    reduction="interval_rms",
    is_fast_input=True,
)
"""Native fast Dα interval-RMS burst energy."""

DALPHA_SHUFFLED_CONTROL_CHANNEL = ConditioningChannel(
    "dalpha_burst_rms_shuffled",
    "xim",
    "da_hm10_t",
    "a.u.",
    reduction="interval_rms",
    causal_shuffle=True,
    is_fast_input=True,
)
"""Causally shuffled fast Dα burst-energy control."""

DALPHA_SLOW_CONTROL_CHANNEL = ConditioningChannel(
    "dalpha_slow_rms",
    "ada",
    "dalpha_raw_full",
    "a.u.",
    reduction="interval_rms",
)
"""Slow radial-profile interval-RMS control."""

CONDITIONING_CHANNELS = ConditioningChannelCatalog(
    (*_ACTUATOR_CONDITIONING_CHANNELS, DALPHA_BURST_CHANNEL),
    {
        "native": DALPHA_BURST_CHANNEL,
        "shuffled": DALPHA_SHUFFLED_CONTROL_CHANNEL,
        "slow_only": DALPHA_SLOW_CONTROL_CHANNEL,
    },
)
"""Physical conditioning channels with selectable Dα control variants."""

DALPHA_PROBE_CHANNEL = ConditioningChannel(
    "dalpha_integrated", "ada", "dalpha_integrated", "", is_probe_target=True
)
"""Dα integrated — optional conditioning, default off (a probe target)."""


def assert_fast_inputs_are_not_probe_targets(channels) -> None:
    """Reject a channel declared as both a fast input and a probe target."""
    bad = sorted(c.key for c in channels if c.is_fast_input and c.is_probe_target)
    if bad:
        raise ValueError(
            "fast conditioning channels cannot also be diagnostic probe targets: "
            f"{bad}"
        )


assert_fast_inputs_are_not_probe_targets(CONDITIONING_CHANNELS)


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


def _signal_on_time_axis(
    sig_time: np.ndarray, sig_value: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Sort a signal and collapse non-time axes to one finite sample trace."""
    st = np.asarray(sig_time, dtype=np.float64).reshape(-1)
    sv = np.asarray(sig_value, dtype=np.float64)
    if st.size == 0 or sv.size == 0:
        return st, np.empty(0, dtype=np.float64)
    if sv.ndim == 1:
        if sv.size != st.size:
            return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
        collapsed = sv
    else:
        candidates = [axis for axis, size in enumerate(sv.shape) if size == st.size]
        if not candidates:
            return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
        time_axis = candidates[-1]
        by_time = np.moveaxis(sv, time_axis, 0).reshape(st.size, -1)
        finite_count = np.isfinite(by_time).sum(axis=1)
        finite_sum = np.nansum(by_time, axis=1)
        collapsed = np.divide(
            finite_sum,
            finite_count,
            out=np.full(st.size, np.nan, dtype=np.float64),
            where=finite_count > 0,
        )
    order = np.argsort(st, kind="stable")
    return st[order], np.asarray(collapsed, dtype=np.float64)[order]


def reduce_interframe_rms(
    sig_time: np.ndarray,
    sig_value: np.ndarray,
    frame_time: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce native samples to causal inter-frame RMS burst energy.

    Frame ``i`` uses only signal samples in ``[frame[i-1], frame[i]]``.
    The first interval uses the following frame spacing as its causal width;
    a single-frame request can include only a sample on that frame boundary.
    Empty or wholly non-finite intervals are zero-filled and marked missing.
    """
    st, sv = _signal_on_time_axis(sig_time, sig_value)
    ft = np.asarray(frame_time, dtype=np.float64).reshape(-1)
    values = np.zeros(ft.size, dtype=np.float32)
    missing = np.ones(ft.size, dtype=np.float32)
    if st.size == 0 or sv.size == 0:
        return values, missing
    if np.any(np.diff(ft) < 0):
        raise ValueError("frame_time must be monotone non-decreasing")

    right = np.searchsorted(st, ft, side="right")
    left = np.searchsorted(st, ft, side="left")
    if ft.size > 1:
        first_interval_start = ft[0] - (ft[1] - ft[0])
        left[0] = np.searchsorted(st, first_interval_start, side="left")
        left[1:] = np.searchsorted(st, ft[:-1], side="left")
    for i, (lo, hi) in enumerate(zip(left, right, strict=True)):
        interval = sv[lo:hi]
        finite = interval[np.isfinite(interval)]
        if finite.size:
            values[i] = np.float32(np.sqrt(np.mean(np.square(finite))))
            missing[i] = 0.0
    return values, missing


def causal_shuffle_summaries(
    values: np.ndarray,
    missing: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shuffle summaries within each causal prefix.

    The source index for output frame ``i`` is always at most ``i``.  This
    destroys same-interval alignment while making future leakage impossible.
    """
    val = np.asarray(values, dtype=np.float32).reshape(-1)
    miss = np.asarray(missing, dtype=np.float32).reshape(-1)
    if val.shape != miss.shape:
        raise ValueError("values and missing must have identical shapes")
    rng = np.random.default_rng(int(seed))
    source = np.zeros(val.size, dtype=np.int64)
    for i in range(1, val.size):
        source[i] = int(rng.integers(0, i))
    return val[source], miss[source], source


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
        off).  Dα is a diagnostic probe target — keep it out of the default
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
    assert_fast_inputs_are_not_probe_targets(chans)

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
        if st is None or sv is None:
            continue
        scaled = np.asarray(sv) * chan.scale
        if chan.reduction == "hold":
            if scaled.ndim != 1:
                continue
            held, miss = resample_to_frames(st, scaled, ft)
        elif chan.reduction == "interval_rms":
            held, miss = reduce_interframe_rms(st, scaled, ft)
            if chan.causal_shuffle:
                held, miss, _ = causal_shuffle_summaries(
                    held, miss, seed=int(shot_id)
                )
        else:
            raise ValueError(
                f"unknown conditioning reduction {chan.reduction!r} for {chan.key}"
            )
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
