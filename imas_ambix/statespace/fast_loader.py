"""D0 — fast/exotic panel loaders: xma, xsx, aoe, act.

Interior-information-discovery-v0 §5 D0.

Loads the broad fast/exotic panel from MAST level-1 Zarr, returning compact
(finite-sample-only) arrays with availability flags and beam-gating.

Measured rates (shot 30460, GPFS):
  xma  5 kHz (5000 Hz, 200 µs spacing) — NOT 660 kHz as documented in the plan;
       plan text counted array size (660 000) as a rate.  Modern schema has a
       dedicated 5 kHz time axis (time1).  Legacy schema uses the full master
       time axis (sec, 17 kHz) with channel-by-channel NaN masking.
  xsx  100–500 kHz depending on MAST campaign (single-digit shot numbers use
       100 kHz; later shots use 500 kHz).  All fully finite inside their window.
  aoe  500 kHz, window ≈ 0.52 s; sparse (NaN outside the reflectometry gate).
  act  ~300 Hz, shape (n_chords, 96), beam-gated (NaN where beam was off).

I/O throughput (GPFS, login node, single-threaded):
  xma 51 key channels:  274 ms / shot  (28 MB compact data)
  xsx 36 channels:       93 ms / shot  (43 MB compact data)
  aoe 5 bands:           25 ms / shot  (4 MB compact data)
  act 12 quantities:     13 ms / shot  (<1 MB)
  Total panel:          ~405 ms / shot

Schema heterogeneity (MAST campaigns):
  xma_modern  (shot ≥ ~27000): ccbv_01..40, fl_cc01..09, dia_loop, time1
  xma_legacy  (shot < ~27000): ccbv01..40, flcc01..10, dialoop, sec
  xsx:         (18,T) or (18,T) hcam_l/hcam_u — T varies with campaign rate
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Channel-name constants (canonical for each schema)
# ---------------------------------------------------------------------------

XMA_MODERN_MIRNOV = [f"ccbv_{i:02d}" for i in range(1, 41)]
XMA_MODERN_FLUX = [f"fl_cc{i:02d}" for i in range(1, 10)]
XMA_MODERN_DIA = ["dia_loop", "dia_loopdot"]
# Rogowski and other channels omitted from core set; available via full_keys()
XMA_MODERN_CORE = XMA_MODERN_MIRNOV + XMA_MODERN_FLUX + XMA_MODERN_DIA

XMA_LEGACY_MIRNOV = [f"ccbv{i:02d}" for i in range(1, 41)]
XMA_LEGACY_FLUX = [f"flcc{i:02d}" for i in range(1, 11)]
XMA_LEGACY_DIA = ["dialoop", "dialoop_dot"]
XMA_LEGACY_CORE = XMA_LEGACY_MIRNOV + XMA_LEGACY_FLUX + XMA_LEGACY_DIA

XSX_CAMERA_ARRAYS = ["hcam_l", "hcam_u"]
XSX_POSITION_ARRAYS = ["hcam_l_r1", "hcam_u_r1"]  # major radius of each chord

AOE_BAND_ARRAYS = ["ka_band", "k_band", "u_band", "fast_k", "fast_ka"]

_ACT_QUANTITIES = [
    "temperature",
    "temperature_error",
    "velocity",
    "velocity_error",
    "cx_counts",
]
ACT_SYSTEMS = {"c_pla": _ACT_QUANTITIES, "c_ss": _ACT_QUANTITIES}


# ---------------------------------------------------------------------------
# Generic zarr helpers (follow mse_split.py pattern exactly)
# ---------------------------------------------------------------------------


def _open_group(shot_zarr_path: Path, group: str):
    """Open *group* inside *shot_zarr_path*; return None if missing/unreadable."""
    import zarr  # noqa: PLC0415

    grp_path = shot_zarr_path / group
    if not grp_path.exists():
        return None
    try:
        return zarr.open_group(str(grp_path), mode="r")
    except Exception as exc:  # pragma: no cover - corpus robustness
        logger.debug("Cannot open %s/%s: %s", shot_zarr_path.name, group, exc)
        return None


def _read_array(group, name: str) -> np.ndarray | None:
    """Read *name* from zarr *group*, tolerant of consolidated-metadata gaps."""
    try:
        if name not in set(group.array_keys()):
            return None
        return np.asarray(group[name])
    except Exception:  # pragma: no cover - corpus robustness
        return None


# ---------------------------------------------------------------------------
# xma — raw fast magnetics
# ---------------------------------------------------------------------------


@dataclass
class XmaShot:
    """Decoded fast-magnetics measurement for one shot (xma group).

    Arrays are compact: only the finite (plasma-on) samples are retained.
    Both ``time`` and ``data`` are aligned index-for-index.

    Attributes
    ----------
    shot_id:
        Integer shot identifier.
    schema:
        ``'modern'`` (ccbv_01, time1, ≥ shot ~27000) or
        ``'legacy'`` (ccbv01, sec, earlier shots).
    rate_hz:
        Measured sample rate in Hz (typically 5000).
    time:
        ``(T,)`` float64 — compact slice times (s).
    channel_names:
        ``(C,)`` list — channel names in data column order.
    data:
        ``(T, C)`` float32 — compact measurements, NaN where channel
        was not recorded at that slice (rare; most entries finite).
    avail_mask:
        ``(C,)`` bool — True if the channel has any finite data in this shot.
    """

    shot_id: int
    schema: str
    rate_hz: float
    time: np.ndarray
    channel_names: list[str]
    data: np.ndarray
    avail_mask: np.ndarray

    @property
    def n_slices(self) -> int:
        return int(self.time.shape[0])

    @property
    def n_channels(self) -> int:
        return len(self.channel_names)


def _xma_extract_modern(grp, shot_id: int) -> XmaShot | None:
    """Decode the modern xma schema (time1 + ccbv_01 naming).

    time1 and time2 are both 5 kHz masked axes from separate acq196 digitiser
    boards, interleaved in the 110 kHz storage array.  Channels on time2 have
    finite indices offset by ~11 samples (~100 µs) relative to time1.  Using
    ``time1[valid1]`` as the reference axis and extracting each channel via its
    OWN finite mask (rather than the shared time1 mask) avoids silently
    returning all-NaN for time2-clocked channels.  The ~100 µs timing error is
    negligible for MHD modes at 5 kHz Nyquist (mode periods ≥ 400 µs).
    """
    # Build the reference time axis from time1 (primary digitiser)
    t1 = _read_array(grp, "time1")
    if t1 is None:
        return None
    t1 = np.asarray(t1).reshape(-1)
    valid1 = np.isfinite(t1)
    n_ref = int(valid1.sum())
    if n_ref < 10:
        return None
    time = t1[valid1].astype(np.float64)

    dt_s = np.diff(time)
    rate_hz = float(1.0 / np.median(dt_s)) if dt_s.size > 0 else 5000.0

    # Build channel matrix — use each channel's OWN finite mask so that
    # time2-clocked channels (e.g. fl_cc01) are not silently all-NaN.
    cols: list[np.ndarray] = []
    names: list[str] = []
    for ch_name in XMA_MODERN_CORE:
        raw = _read_array(grp, ch_name)
        if raw is None:
            continue
        raw = np.asarray(raw).reshape(-1)
        if raw.size != t1.size:
            continue  # unexpected shape; skip
        ch_finite = np.isfinite(raw)
        n_ch = int(ch_finite.sum())
        if n_ch == n_ref:
            # Same count — use channel's own mask (handles time2-clocked channels)
            cols.append(raw[ch_finite].astype(np.float32))
        elif n_ch == 0:
            continue  # channel absent for this shot
        else:
            # Count mismatch: fall back to time1 mask (may introduce NaNs)
            logger.debug(
                "%s xma/%s: finite count %d ≠ ref %d, using time1 mask",
                shot_id,
                ch_name,
                n_ch,
                n_ref,
            )
            cols.append(raw[valid1].astype(np.float32))
        names.append(ch_name)

    if not cols:
        return None

    data = np.stack(cols, axis=1)  # (T, C)
    avail = np.isfinite(data).any(axis=0)  # (C,) bool
    return XmaShot(
        shot_id=shot_id,
        schema="modern",
        rate_hz=rate_hz,
        time=time,
        channel_names=names,
        data=data,
        avail_mask=avail,
    )


def _xma_extract_legacy(grp, shot_id: int) -> XmaShot | None:
    """Decode the legacy xma schema (sec master axis + ccbv01 naming)."""
    # Use ccbv01 finite mask to identify the plasma-on window
    cc01 = _read_array(grp, "ccbv01")
    if cc01 is None:
        # try without leading zero
        cc01 = _read_array(grp, "ccbv1")
    if cc01 is None:
        return None
    cc01 = np.asarray(cc01).reshape(-1)
    valid = np.isfinite(cc01)
    if valid.sum() < 10:
        return None

    # sec: the master time axis (may be ALL finite, not masked)
    sec = _read_array(grp, "sec")
    if sec is not None:
        sec = np.asarray(sec).reshape(-1)
        if sec.size == valid.size:
            time = sec[valid].astype(np.float64)
        else:
            time = np.where(valid)[0].astype(np.float64)  # index fallback
    else:
        time = np.where(valid)[0].astype(np.float64)

    dt_s = np.diff(time)
    rate_hz = float(1.0 / np.median(dt_s)) if dt_s.size > 0 else 5000.0

    cols: list[np.ndarray] = []
    names: list[str] = []
    for ch_name in XMA_LEGACY_CORE:
        raw = _read_array(grp, ch_name)
        if raw is None:
            continue
        raw = np.asarray(raw).reshape(-1)
        if raw.size == valid.size:
            cols.append(raw[valid].astype(np.float32))
        elif raw.size == time.size:
            cols.append(raw.astype(np.float32))
        else:
            continue
        names.append(ch_name)

    if not cols:
        return None

    data = np.stack(cols, axis=1)
    avail = np.isfinite(data).any(axis=0)
    return XmaShot(
        shot_id=shot_id,
        schema="legacy",
        rate_hz=rate_hz,
        time=time,
        channel_names=names,
        data=data,
        avail_mask=avail,
    )


def read_xma_shot(shot_zarr_path: Path) -> XmaShot | None:
    """Decode the fast-magnetics (xma) group for one shot.

    Tries the modern schema first (time1 + ccbv_01); falls back to the legacy
    schema (sec + ccbv01) when the modern time axis is absent.

    Returns None if the group is missing, unreadable, or carries no usable data.
    """
    grp = _open_group(shot_zarr_path, "xma")
    if grp is None:
        return None
    shot_id = int(shot_zarr_path.stem)
    keys = set(grp.array_keys())
    if "time1" in keys:
        return _xma_extract_modern(grp, shot_id)
    if any(k.startswith("ccbv") for k in keys):
        return _xma_extract_legacy(grp, shot_id)
    return None


# ---------------------------------------------------------------------------
# xim — fast visible spectroscopy (Dα / CII), ELM carriers
# ---------------------------------------------------------------------------
#
# xim ("spectrometer_visible") holds the fast Dα (da_*) and CII (cii_*) photo-
# multiplier channels at ~50 kHz on a dense, fully-finite time axis.  These
# carry the ELM / fluctuation carriers in the divertor and midplane sightlines.
#
# Schema heterogeneity (MAST campaigns):
#   modern (shot >= ~27000): da_hl11_l1, da_hu10_u1, da_hm10_r1, ... ; time
#   legacy (shot <  ~27000): da_hl11_l,  da_hu10_u,  da_hm10_r,  ... ; time
# The numeric-suffix variants (`_l1` vs `_l`) and the exact channel inventory
# vary per campaign, so the loader is schema-tolerant: it discovers da_*/cii_*
# channels present in the group rather than assuming a fixed inventory.

# Non-signal bookkeeping arrays that share the time axis but are NOT emission
# channels — excluded from the Dα/CII signal set.
_XIM_NON_SIGNAL = frozenset(
    {
        "time",
        "sec",
        "trigger",
        "target",
        "light_start",
        "light_end",
        "mass_start",
        "mass_end",
        "pellet_time",
        "pellet_halpha_2",
        "preion_trig",
        "ts_yag",
        "xsa_logic_out",
        "test_jd0",
    }
)


@dataclass
class XimShot:
    """Decoded fast visible-spectroscopy (xim) measurement for one shot.

    The time axis is dense and fully finite (no plasma-on masking needed at
    this digitiser); channels are kept on that native axis.

    Attributes
    ----------
    shot_id:
        Integer shot identifier.
    schema:
        ``'modern'`` (numeric-suffix channels, e.g. da_hl11_l1) or
        ``'legacy'`` (bare-suffix channels, e.g. da_hl11_l).
    rate_hz:
        Measured sample rate in Hz (typically ~50 000).
    time:
        ``(T,)`` float64 — native time axis (s).
    channel_names:
        ``(C,)`` list — da_*/cii_* channel names in data column order.
    data:
        ``(T, C)`` float32 — emission measurements.
    avail_mask:
        ``(C,)`` bool — True if the channel has any finite data in this shot.
    """

    shot_id: int
    schema: str
    rate_hz: float
    time: np.ndarray
    channel_names: list[str]
    data: np.ndarray
    avail_mask: np.ndarray

    @property
    def n_slices(self) -> int:
        return int(self.time.shape[0])

    @property
    def n_channels(self) -> int:
        return len(self.channel_names)


def read_xim_shot(shot_zarr_path: Path) -> XimShot | None:
    """Decode the fast visible-spectroscopy (xim) group for one shot.

    Schema-tolerant (copy of the read_xsx_shot pattern): discovers the
    da_*/cii_* emission channels actually present rather than assuming a fixed
    inventory, so it works across MAST campaigns whose channel suffixes differ
    (``da_hl11_l1`` modern vs ``da_hl11_l`` legacy).  Returns None if the group
    is missing, has no time axis, or carries no usable da_*/cii_* channel.
    """
    grp = _open_group(shot_zarr_path, "xim")
    if grp is None:
        return None
    shot_id = int(shot_zarr_path.stem)

    keys = set(grp.array_keys())
    t = _read_array(grp, "time")
    if t is None:
        return None
    t = np.asarray(t).reshape(-1).astype(np.float64)
    fin_t = np.isfinite(t)
    if fin_t.sum() < 10:
        return None
    time = t[fin_t]

    dt_s = np.diff(time[: min(time.size, 1000)])
    rate_hz = float(1.0 / np.median(dt_s)) if dt_s.size > 0 else 50_000.0

    # Discover da_*/cii_* emission channels present in this shot, in a stable
    # (sorted) order.  Schema is inferred from whether any modern numeric-suffix
    # channel name is present.
    signal_keys = sorted(
        k
        for k in keys
        if (k.startswith("da_") or k.startswith("cii_") or k.startswith("heii_"))
        and k not in _XIM_NON_SIGNAL
    )
    if not signal_keys:
        return None
    # Modern campaigns carry the numeric-suffix sightline variants
    # (da_hm10_r1, da_hl11_l1, …); legacy campaigns carry only the bare-suffix
    # names (da_hm10_r, da_hl11_l).  The bare names contain digits in the
    # detector label itself (da_bo10), so detect on the trailing "_<letter><n>"
    # variant marker rather than on a trailing digit.
    schema = (
        "modern"
        if any(len(k) > 2 and k[-1].isdigit() and k[-2].isalpha() for k in signal_keys)
        else "legacy"
    )

    cols: list[np.ndarray] = []
    names: list[str] = []
    for ch in signal_keys:
        raw = _read_array(grp, ch)
        if raw is None:
            continue
        raw = np.asarray(raw).reshape(-1)
        if raw.size != t.size:
            continue  # unexpected shape; skip
        cols.append(raw[fin_t].astype(np.float32))
        names.append(ch)

    if not cols:
        return None

    data = np.stack(cols, axis=1)  # (T, C)
    avail = np.isfinite(data).any(axis=0)  # (C,) bool
    return XimShot(
        shot_id=shot_id,
        schema=schema,
        rate_hz=rate_hz,
        time=time,
        channel_names=names,
        data=data,
        avail_mask=avail,
    )


# ---------------------------------------------------------------------------
# xsx — soft X-ray cameras
# ---------------------------------------------------------------------------


@dataclass
class XsxShot:
    """Decoded SXR camera measurement for one shot (xsx group).

    Both horizontal camera arrays are returned if present; ``hcam_u`` may be
    None for shots where only the lower camera was operational.

    Attributes
    ----------
    shot_id:
        Integer shot identifier.
    rate_hz:
        Sampling rate in Hz (100 000 or 500 000 depending on campaign era).
    time:
        ``(T,)`` float64 — dense time axis (s).
    hcam_l:
        ``(C, T)`` float32 — lower horizontal camera (C channels).
    hcam_l_r1:
        ``(C,)`` float64 — chord inner major radius (m); NaN if unavailable.
    hcam_u:
        ``(C, T)`` float32 or None — upper horizontal camera.
    hcam_u_r1:
        ``(C,)`` float64 or None — chord inner major radius for upper camera.
    avail_mask:
        ``(2,)`` bool — [hcam_l_present, hcam_u_present].
    """

    shot_id: int
    rate_hz: float
    time: np.ndarray
    hcam_l: np.ndarray
    hcam_l_r1: np.ndarray
    hcam_u: np.ndarray | None
    hcam_u_r1: np.ndarray | None
    avail_mask: np.ndarray


def read_xsx_shot(shot_zarr_path: Path) -> XsxShot | None:
    """Decode the SXR camera (xsx) group for one shot.

    Returns None if hcam_l is not present (the primary camera).
    """
    grp = _open_group(shot_zarr_path, "xsx")
    if grp is None:
        return None

    shot_id = int(shot_zarr_path.stem)

    t = _read_array(grp, "time")
    if t is None:
        return None
    t = np.asarray(t).reshape(-1).astype(np.float64)
    if t.size < 2:
        return None
    if t.size >= 100:
        rate_hz = float(1.0 / np.diff(t[:100]).mean())
    else:
        rate_hz = float(1.0 / (t[1] - t[0]))

    hl = _read_array(grp, "hcam_l")
    if hl is None:
        return None
    hl = np.asarray(hl).astype(np.float32)  # (C, T)
    if hl.ndim != 2:
        return None

    hl_r1_raw = _read_array(grp, "hcam_l_r1")
    hl_r1: np.ndarray
    if hl_r1_raw is not None:
        hl_r1 = np.asarray(hl_r1_raw).reshape(-1).astype(np.float64)
    else:
        hl_r1 = np.full(hl.shape[0], np.nan)

    hu = _read_array(grp, "hcam_u")
    hu_raw = np.asarray(hu) if hu is not None else None
    hu_arr: np.ndarray | None = (
        hu_raw.astype(np.float32) if hu_raw is not None and hu_raw.ndim == 2 else None
    )
    hu_r1_raw = _read_array(grp, "hcam_u_r1")
    hu_r1: np.ndarray | None = (
        np.asarray(hu_r1_raw).reshape(-1).astype(np.float64)
        if hu_r1_raw is not None and hu_arr is not None
        else None
    )

    avail = np.array([True, hu_arr is not None], dtype=bool)

    return XsxShot(
        shot_id=shot_id,
        rate_hz=rate_hz,
        time=t,
        hcam_l=hl,
        hcam_l_r1=hl_r1,
        hcam_u=hu_arr,
        hcam_u_r1=hu_r1,
        avail_mask=avail,
    )


# ---------------------------------------------------------------------------
# aoe — microwave reflectometry
# ---------------------------------------------------------------------------


@dataclass
class AoeShot:
    """Decoded reflectometry (aoe) measurement for one shot.

    Only the finite (reflectometry-active) window is retained.
    Bands absent from the shot are stored as NaN arrays of the same shape.

    Attributes
    ----------
    shot_id:
        Integer shot identifier.
    rate_hz:
        Effective sampling rate of the finite window (Hz).
    time:
        ``(T,)`` float64 — compact finite-window times (s).
    bands:
        ``{band_name: (T,) float32}`` — reflectometry band data.
    avail_mask:
        ``{band_name: bool}`` — True if the band has any finite data.
    """

    shot_id: int
    rate_hz: float
    time: np.ndarray
    bands: dict[str, np.ndarray]
    avail_mask: dict[str, bool]

    @property
    def n_slices(self) -> int:
        return int(self.time.shape[0])


def read_aoe_shot(shot_zarr_path: Path) -> AoeShot | None:
    """Decode the reflectometry (aoe) group for one shot.

    Returns None if the group is missing or carries no usable data.
    """
    grp = _open_group(shot_zarr_path, "aoe")
    if grp is None:
        return None

    shot_id = int(shot_zarr_path.stem)

    # Use ka_band NaN mask to define the active window; fall back to k_band
    for primary in ["ka_band", "k_band", "fast_ka"]:
        primary_arr = _read_array(grp, primary)
        if primary_arr is not None:
            break
    if primary_arr is None:
        return None

    primary_arr = np.asarray(primary_arr).reshape(-1)
    valid = np.isfinite(primary_arr)
    if valid.sum() < 10:
        return None

    t_full = _read_array(grp, "time")
    if t_full is None:
        return None
    t_full = np.asarray(t_full).reshape(-1).astype(np.float64)
    if t_full.size != primary_arr.size:
        return None
    time = t_full[valid]

    dt_s = np.diff(time)
    rate_hz = float(1.0 / np.median(dt_s)) if dt_s.size > 0 else 500_000.0

    bands: dict[str, np.ndarray] = {}
    avail: dict[str, bool] = {}
    for bname in AOE_BAND_ARRAYS:
        raw = _read_array(grp, bname)
        if raw is None:
            bands[bname] = np.full(valid.sum(), np.nan, dtype=np.float32)
            avail[bname] = False
        else:
            raw = np.asarray(raw).reshape(-1)
            if raw.size == valid.size:
                bands[bname] = raw[valid].astype(np.float32)
            elif raw.size == valid.sum():
                bands[bname] = raw.astype(np.float32)
            else:
                bands[bname] = np.full(valid.sum(), np.nan, dtype=np.float32)
                avail[bname] = False
                continue
            avail[bname] = bool(np.isfinite(bands[bname]).any())

    return AoeShot(
        shot_id=shot_id,
        rate_hz=rate_hz,
        time=time,
        bands=bands,
        avail_mask=avail,
    )


# ---------------------------------------------------------------------------
# act — CXRS (charge-exchange spectroscopy), beam-gated
# ---------------------------------------------------------------------------


@dataclass
class ActShot:
    """Decoded CXRS (act) measurement for one shot.

    Beam-gating: only slices where at least one chord has a finite temperature
    reading are included (NBI beam was on for that measurement frame).

    Attributes
    ----------
    shot_id:
        Integer shot identifier.
    system:
        CX system name (e.g. ``'c_pla'`` for the primary plasma system).
    n_chords:
        Number of spatial chords (rows in ``temperature``).
    n_slices:
        Total number of time slices in the raw data (NOT beam-filtered).
    time:
        ``(S,)`` float64 — time axis (s); may be index-based if no time
        array is present in the group.  Use ``beam_on_mask`` to select
        beam-on frames.
    major_radius:
        ``(S,)`` float64 — magnetic-axis R at each slice (m), or NaN.
    temperature:
        ``(C, S)`` float32 — ion temperature (eV) per chord per slice.
    velocity:
        ``(C, S)`` float32 — toroidal velocity (km/s) per chord per slice.
    cx_counts:
        ``(C, S)`` float32 — CX photon counts per chord per slice.
    beam_on_mask:
        ``(S,)`` bool — True for slices where ≥1 chord has a finite
        temperature (beam was on).
    avail_mask:
        ``{'temperature': bool, 'velocity': bool, 'cx_counts': bool}``.
    """

    shot_id: int
    system: str
    n_chords: int
    n_slices: int
    time: np.ndarray
    major_radius: np.ndarray
    temperature: np.ndarray
    velocity: np.ndarray
    cx_counts: np.ndarray
    beam_on_mask: np.ndarray
    avail_mask: dict[str, bool]


def read_act_shot(shot_zarr_path: Path, system: str = "c_pla") -> ActShot | None:
    """Decode the CXRS (act) group for one shot.

    Parameters
    ----------
    system:
        CX system prefix: ``'c_pla'`` (primary plasma) or ``'c_ss'`` (sawtooth).
    """
    grp = _open_group(shot_zarr_path, "act")
    if grp is None:
        return None

    shot_id = int(shot_zarr_path.stem)

    temp = _read_array(grp, f"{system}_temperature")
    if temp is None or np.asarray(temp).ndim != 2:
        return None
    temp = np.asarray(temp).astype(np.float32)  # (C, S)
    n_chords, n_slices = temp.shape

    vel = _read_array(grp, f"{system}_velocity")
    vel_raw = np.asarray(vel) if vel is not None else None
    vel_arr: np.ndarray = (
        vel_raw.astype(np.float32)
        if vel_raw is not None and vel_raw.shape == temp.shape
        else np.full_like(temp, np.nan)
    )

    cx = _read_array(grp, f"{system}_cx_counts")
    cx_raw = np.asarray(cx) if cx is not None else None
    cx_arr: np.ndarray = (
        cx_raw.astype(np.float32)
        if cx_raw is not None and cx_raw.shape == temp.shape
        else np.full_like(temp, np.nan)
    )

    # Time axis: `time` array has shape (T,) but may differ from n_slices.
    # Use majorradius (shape matches n_slices) as a fallback time reference.
    t_raw = _read_array(grp, "time")
    mr_raw = _read_array(grp, "majorradius")

    mr_raw_arr = np.asarray(mr_raw) if mr_raw is not None else None
    mr: np.ndarray = (
        mr_raw_arr.reshape(-1).astype(np.float64)
        if mr_raw_arr is not None and mr_raw_arr.size == n_slices
        else np.full(n_slices, np.nan)
    )

    if t_raw is not None and np.asarray(t_raw).size == n_slices:
        time = np.asarray(t_raw).reshape(-1).astype(np.float64)
    else:
        # Use index as proxy time (will be resolved in D1)
        time = np.arange(n_slices, dtype=np.float64)

    # Beam-on mask: ≥1 chord finite temperature at this slice
    beam_on = np.isfinite(temp).any(axis=0)  # (S,) bool

    avail = {
        "temperature": bool(np.isfinite(temp).any()),
        "velocity": bool(np.isfinite(vel_arr).any()),
        "cx_counts": bool(np.isfinite(cx_arr).any()),
    }

    return ActShot(
        shot_id=shot_id,
        system=system,
        n_chords=n_chords,
        n_slices=n_slices,
        time=time,
        major_radius=mr,
        temperature=temp,
        velocity=vel_arr,
        cx_counts=cx_arr,
        beam_on_mask=beam_on,
        avail_mask=avail,
    )


# ---------------------------------------------------------------------------
# Fast panel — availability probe (no data load; filesystem only)
# ---------------------------------------------------------------------------


@dataclass
class FastPanelAvail:
    """Lightweight per-shot availability record (no array reads)."""

    shot_id: int
    has_xma: bool
    xma_schema: str  # 'modern' | 'legacy' | 'empty' | 'missing'
    has_xsx: bool
    xsx_has_hcam: bool
    has_aoe: bool
    has_act: bool


def probe_fast_panel(shot_zarr_path: Path) -> FastPanelAvail:
    """Cheap filesystem probe — does NOT open Zarr arrays."""
    shot_id = int(shot_zarr_path.stem)

    # xma
    xma_p = shot_zarr_path / "xma"
    if not xma_p.exists():
        xma_schema = "missing"
        has_xma = False
    else:
        # Distinguish modern vs legacy by presence of time1 file
        if (xma_p / "time1").exists() or (xma_p / "time1" / ".zarray").exists():
            xma_schema = "modern"
            has_xma = True
        elif (xma_p / "ccbv01").exists() or (xma_p / "ccbv01" / ".zarray").exists():
            xma_schema = "legacy"
            has_xma = True
        elif any((xma_p / f"ccbv_{i:02d}").exists() for i in range(1, 5)):
            xma_schema = "modern"
            has_xma = True
        else:
            xma_schema = "empty"
            has_xma = False

    # xsx
    xsx_p = shot_zarr_path / "xsx"
    has_xsx = xsx_p.exists()
    xsx_has_hcam = has_xsx and (
        (xsx_p / "hcam_l").exists() or (xsx_p / "hcam_l" / ".zarray").exists()
    )

    # aoe / act
    has_aoe = (shot_zarr_path / "aoe").exists()
    has_act = (shot_zarr_path / "act").exists()

    return FastPanelAvail(
        shot_id=shot_id,
        has_xma=has_xma,
        xma_schema=xma_schema,
        has_xsx=has_xsx,
        xsx_has_hcam=xsx_has_hcam,
        has_aoe=has_aoe,
        has_act=has_act,
    )


# ---------------------------------------------------------------------------
# Window alignment — MSE eval slices
# ---------------------------------------------------------------------------


def align_to_mse_window(
    time: np.ndarray,
    data: np.ndarray,
    t_min: float,
    t_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Restrict *time*/*data* to the MSE eval window [t_min, t_max].

    Parameters
    ----------
    time:
        ``(T,)`` time axis.
    data:
        ``(T, ...)`` or ``(..., T)`` data array.  Only the leading dimension
        is sliced; for xsx/act (channels × time) the caller transposes first.
    t_min, t_max:
        Window bounds (s) derived from ``beam_on_slice_times`` in the MSE
        manifest.

    Returns
    -------
    (time_win, data_win)
        Arrays restricted to the window; empty if no overlap.
    """
    mask = (time >= t_min) & (time <= t_max)
    if data.ndim >= 1 and data.shape[0] == time.shape[0]:
        return time[mask], data[mask]
    return time[mask], data  # shape mismatch — return unchanged


def mse_eval_window(shot_manifest_entry: dict) -> tuple[float, float]:
    """Extract the MSE eval time window from a manifest shot entry.

    Parameters
    ----------
    shot_manifest_entry:
        One entry from ``mse_heldout_split_v0.json["shots"]``.

    Returns
    -------
    (t_min, t_max)
        Bounds of the beam-on MSE slice window (s).
    """
    times = shot_manifest_entry.get("beam_on_slice_times", [])
    if not times:
        return (float("-inf"), float("inf"))
    return float(min(times)), float(max(times))
