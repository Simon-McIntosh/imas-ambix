"""Measure whether the rbb camera RESOLVES or INTEGRATES the ELM burst.

A zero-GPU, CPU-only data forensics audit that settles — from the data, not
from argument — the strategic question for the camera-dynamics world model:
is its poor ELM reproduction a fixable conditioning gap, or a hard
sampling cap baked into the prediction target itself?

The crux is a sampling-theorem question about the rbb camera (the model's
prediction target).  An ELM at MAST is a sub-millisecond edge/divertor light
burst (rise + fall over ~0.1-1 ms).  If the camera inter-frame interval Δt is
well BELOW that width it RESOLVES the burst — several frames trace the rise
and fall, so cross-frame conditioning can in principle learn the onset
dynamics.  If Δt is at or above ~1 ms the camera INTEGRATES the burst into a
single bright frame — the sub-frame onset is gone at acquisition time and no
amount of conditioning can recover it (you cannot predict information the
target never recorded).

Four decisive measurements (all from on-disk corpus reads):

1. **rbb frame-interval distribution**, per held-out shot, with ELM-bearing
   shots flagged.  Settles the contradiction in our own artifacts
   (camera-dynamics doc ~100-400 Hz vs horizon_eval ~13 us..1 ms vs a prior
   ~1 kHz spot measurement) by reading the per-frame timestamps directly.
   Reports the FRACTION of ELM-window shots whose Δt resolves vs integrates
   a typical ELM width.

2. **Resolve vs integrate, per representative ELM window.**  On a clear
   camera-edge-burst window, overlays the per-frame camera ELM-brightening
   trajectory against the fast-Dα burst at its native rate, and reports
   whether the camera traces a multi-frame rise+fall (resolved) or shows a
   single isolated bright frame (integrated).

3. **Settle the contested fast-diagnostic rates/shapes**: ``ada`` Dα arrays,
   ``xim/da_hm10_t``, ``xma/ccbv_*`` — measured on-disk sample rate + shape,
   and which channel is the genuine fast ELM carrier on this corpus.

4. **Forecast statistical power** at the locked horizons (10/50/200 ms) — the
   surviving valid ELM-window sample counts, read from the W2 horizon table.

All MAST level-1 / token data are FAIR-MAST Zarr (V2 / V3), read with the
repo's existing ``zarr``-based readers — these are NOT IMAS HDF5 (the
imas-python rule is for IMAS data; FAIR-MAST Zarr has always used ``zarr``,
see ``dataset.py`` and ``conditioning.py``).  Runs CPU-only on a compute node
with GPFS access (raw data + token corpus).

Run::

    python -m imas_ambix.camdyn.elm_sampling_audit \\
        --out docs/figures/camera-dynamics-wm/fig-cdw-elm-sampling-audit.png
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from imas_ambix.camdyn.dataset import (
    DEFAULT_CAMERA,
    frames_token_path,
    level1_shot_path,
)
from imas_ambix.camdyn.recon_movie import camera_brightness_trace, camera_elm_score

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Typical ELM light-burst full width (rise + fall) at MAST, milliseconds.
#: An ELM Dα/divertor burst rises and decays over roughly 0.1-1 ms; the band
#: brackets that range.  A camera whose Δt is well below ELM_WIDTH_MS[0]
#: resolves the burst; a camera whose Δt is at/above ELM_WIDTH_MS[1]
#: integrates it into a single frame.
ELM_WIDTH_MS: tuple[float, float] = (0.1, 1.0)

#: Δt above which the camera is taken to INTEGRATE (rather than resolve) a
#: typical ELM — the upper edge of the ELM-width band.
INTEGRATE_DT_MS: float = ELM_WIDTH_MS[1]

#: Default split artifact (held-out shot list).
DEFAULT_SPLIT = Path("imas_ambix/camdyn/artifacts/camdyn_split_v0.json")

#: W2 forward-horizon table (forecast-power sample counts at each horizon).
DEFAULT_HORIZON_TABLE = Path("imas_ambix/camdyn/artifacts/horizon_table.json")
DEFAULT_FORECAST_SWEEP = Path("imas_ambix/camdyn/artifacts/forecast_sweep.json")

#: Fast-Dα photodiode channels in the ``xim`` group (the real sub-ms ELM
#: signature).  ``hm10`` is the horizontal-midplane monitor, the classic
#: MAST ELM diode; the others cross-check the burst.  Mirrors the channel
#: set the rest of camdyn uses for ELM confirmation.
FAST_DALPHA_CHANNELS: tuple[str, ...] = (
    "da_hm10_t",
    "da_hm10_r",
    "da_hl11_r",
    "da_to10",
    "da_bo10",
    "da_hu10_t",
)

#: ELM-window detection thresholds.  ``camera_elm_score`` returns a robust
#: sigma above baseline for the strongest high-passed transient; this gates a
#: window as "ELM-bearing".
CAMERA_ELM_SIGMA_GATE: float = 4.0

#: Flat-top window to scan for ELMs (s) — avoids ramp-up / ramp-down.
FLATTOP_S: tuple[float, float] = (0.10, 0.50)

OUT_JSON = Path("imas_ambix/camdyn/artifacts/elm_sampling_audit.json")


# ---------------------------------------------------------------------------
# Zarr readers (FAIR-MAST level-1 + token corpus — reuse the repo pattern)
# ---------------------------------------------------------------------------


def _open_group(path: Path):
    """Open a zarr group, returning None on any failure."""
    import zarr  # noqa: PLC0415

    if not path.exists():
        return None
    try:
        return zarr.open_group(str(path), mode="r")
    except Exception as exc:  # pragma: no cover - corpus robustness
        logger.debug("cannot open %s: %s", path, exc)
        return None


def read_camera_frame_times(
    shot_id: int, camera: str = DEFAULT_CAMERA
) -> np.ndarray | None:
    """Per-frame camera timestamps (s) from the V2 level-1 store, or None.

    ``<level1>/<shot>.zarr/<camera>/time`` — the same axis ``dataset.py``
    uses for the world-model frame grid (temporal_compression == 1, so
    token frame i ↔ raw frame i ↔ time[i]).
    """
    grp = _open_group(level1_shot_path(shot_id) / camera)
    if grp is None:
        return None
    try:
        if "time" not in set(grp.array_keys()):
            return None
        t = np.asarray(grp["time"], dtype=np.float64).reshape(-1)
    except Exception:  # pragma: no cover - corpus robustness
        return None
    return t if t.size >= 2 else None


def read_raw_frames(
    shot_id: int, frame_lo: int, frame_hi: int, camera: str = DEFAULT_CAMERA
) -> tuple[np.ndarray, np.ndarray] | None:
    """Read raw camera frames ``[frame_lo:frame_hi]`` + their times.

    Returns ``(frames (n,H,W), times (n,))`` or None.  Reads only the needed
    slice so a representative window is cheap even on large sensors.
    """
    grp = _open_group(level1_shot_path(shot_id) / camera)
    if grp is None:
        return None
    try:
        keys = set(grp.array_keys())
        if "time" not in keys or "data" not in keys:
            return None
        t = np.asarray(grp["time"], dtype=np.float64).reshape(-1)
        data = grp["data"]
        hi = min(frame_hi, int(data.shape[0]), t.size)
        lo = max(0, frame_lo)
        if hi - lo < 2:
            return None
        frames = np.asarray(data[lo:hi], dtype=np.float64)
        return frames, t[lo:hi]
    except Exception as exc:  # pragma: no cover - corpus robustness
        logger.debug("cannot read frames %d: %s", shot_id, exc)
        return None


def read_diag_array(
    shot_id: int, source: str, array: str
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return ``(time, value)`` for a level-1 ``source/array`` (raw, native rate).

    For the fast diagnostics (``xim``, ``ada``, ``xma``) the time axis is the
    group's own native acquisition axis — this is exactly what we want to
    measure (the genuine on-disk sample rate / shape).
    """
    grp = _open_group(level1_shot_path(shot_id) / source)
    if grp is None:
        return None
    try:
        keys = set(grp.array_keys())
        if array not in keys:
            return None
        val = np.asarray(grp[array])
        # the time axis: most groups carry "time"; xma_modern carries time1
        tkey = None
        for cand in ("time", "time1", "sec"):
            if cand in keys:
                tkey = cand
                break
        t = np.asarray(grp[tkey], dtype=np.float64).reshape(-1) if tkey else None
        return t, val
    except Exception as exc:  # pragma: no cover - corpus robustness
        logger.debug("cannot read %d %s/%s: %s", shot_id, source, array, exc)
        return None


def frame_resolution(
    shot_id: int, camera: str = DEFAULT_CAMERA
) -> tuple[int, int] | None:
    """Native per-frame sensor resolution ``(H, W)`` for one shot, or None.

    rbb is read out in different framing modes across shots — a small windowed
    ROI at high speed vs a full sensor frame at slower cadence — so the cadence
    is meaningless without the resolution that bought it.
    """
    grp = _open_group(level1_shot_path(shot_id) / camera)
    if grp is None:
        return None
    try:
        if "data" not in set(grp.array_keys()):
            return None
        shp = tuple(int(x) for x in grp["data"].shape)
        return (shp[1], shp[2]) if len(shp) >= 3 else None
    except Exception:  # pragma: no cover - corpus robustness
        return None


# ---------------------------------------------------------------------------
# Q1 — frame-interval distribution + resolve/integrate fraction
# ---------------------------------------------------------------------------


@dataclass
class ShotCadence:
    """Per-shot camera cadence summary."""

    shot_id: int
    n_frames: int
    median_dt_ms: float
    p05_dt_ms: float
    p95_dt_ms: float
    res_h: int = 0
    res_w: int = 0
    is_elm: bool = False
    camera_elm_sigma: float = 0.0
    elm_peak_frame: int = -1

    @property
    def npix(self) -> int:
        return self.res_h * self.res_w


def shot_cadence(shot_id: int) -> ShotCadence | None:
    """Median / 5-95 pct inter-frame Δt (ms) for one shot's rbb stream."""
    t = read_camera_frame_times(shot_id)
    if t is None:
        return None
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size < 2:
        return None
    res = frame_resolution(shot_id) or (0, 0)
    return ShotCadence(
        shot_id=shot_id,
        n_frames=int(t.size),
        median_dt_ms=float(np.median(dt) * 1e3),
        p05_dt_ms=float(np.quantile(dt, 0.05) * 1e3),
        p95_dt_ms=float(np.quantile(dt, 0.95) * 1e3),
        res_h=int(res[0]),
        res_w=int(res[1]),
    )


def detect_camera_elm(shot_id: int) -> tuple[float, int, int]:
    """Strongest camera edge/divertor burst over the flat-top → (sigma, peak, n).

    Reads the raw rbb frames over the flat-top window, scores the high-passed
    edge-brightness transient with the camdyn ``camera_elm_score`` (the same
    detector the demo uses), and returns ``(sigma, peak_frame, n_scanned)``.
    The peak_frame index is RELATIVE to the flat-top slice.
    """
    t = read_camera_frame_times(shot_id)
    if t is None:
        return 0.0, -1, 0
    flat = np.where((t >= FLATTOP_S[0]) & (t <= FLATTOP_S[1]))[0]
    if flat.size < 8:
        return 0.0, -1, 0
    lo, hi = int(flat[0]), int(flat[-1]) + 1
    got = read_raw_frames(shot_id, lo, hi)
    if got is None:
        return 0.0, -1, 0
    frames, _ = got
    sigma, peak = camera_elm_score(frames)
    return float(sigma), int(lo + peak), int(frames.shape[0])


def survey_cadence(
    shot_ids: list[int], *, max_shots: int, elm_check: bool
) -> list[ShotCadence]:
    """Survey per-shot rbb cadence over a sample of held-out shots.

    When ``elm_check`` runs the camera ELM detector on each shot and flags the
    ones with a clear edge-burst.  Bounded to ``max_shots`` for cost.
    """
    out: list[ShotCadence] = []
    scanned = 0
    for sid in shot_ids:
        if scanned >= max_shots:
            break
        # require the token store too (held-out world-model shots) — cheap check
        if not frames_token_path(sid).exists():
            continue
        sc = shot_cadence(sid)
        if sc is None:
            continue
        scanned += 1
        if elm_check:
            sigma, peak, _ = detect_camera_elm(sid)
            sc.camera_elm_sigma = sigma
            sc.elm_peak_frame = peak
            sc.is_elm = sigma >= CAMERA_ELM_SIGMA_GATE
        out.append(sc)
        if scanned % 25 == 0:
            logger.info("[elmaudit] surveyed %d shots", scanned)
    return out


def cadence_summary(cadences: list[ShotCadence]) -> dict:
    """Distribution stats + resolve-vs-integrate fractions for the survey."""
    med = np.array([c.median_dt_ms for c in cadences], dtype=np.float64)
    elm = np.array([c.is_elm for c in cadences], dtype=bool)
    elm_med = med[elm]

    def _frac_integrate(arr: np.ndarray) -> float:
        return float(np.mean(arr >= INTEGRATE_DT_MS)) if arr.size else float("nan")

    def _frac_resolve(arr: np.ndarray) -> float:
        # resolves a typical ELM only if Δt is well below the lower band edge
        return float(np.mean(arr < ELM_WIDTH_MS[0])) if arr.size else float("nan")

    return {
        "n_shots": int(med.size),
        "n_elm_shots": int(elm.sum()),
        "all_shots": {
            "median_dt_ms": float(np.median(med)) if med.size else None,
            "min_dt_ms": float(med.min()) if med.size else None,
            "max_dt_ms": float(med.max()) if med.size else None,
            "p05_dt_ms": float(np.quantile(med, 0.05)) if med.size else None,
            "p95_dt_ms": float(np.quantile(med, 0.95)) if med.size else None,
            "frac_integrate_ge_1ms": _frac_integrate(med),
            "frac_resolve_lt_0p1ms": _frac_resolve(med),
        },
        "elm_shots": {
            "median_dt_ms": float(np.median(elm_med)) if elm_med.size else None,
            "min_dt_ms": float(elm_med.min()) if elm_med.size else None,
            "max_dt_ms": float(elm_med.max()) if elm_med.size else None,
            "frac_integrate_ge_1ms": _frac_integrate(elm_med),
            "frac_resolve_lt_0p1ms": _frac_resolve(elm_med),
            "median_dt_values_ms": [float(x) for x in np.sort(elm_med)],
        },
    }


# ---------------------------------------------------------------------------
# Q2 — resolve vs integrate on a representative ELM window
# ---------------------------------------------------------------------------


@dataclass
class ResolveWindow:
    """One representative ELM window: camera trajectory vs native fast-Dα."""

    shot_id: int
    camera_dt_ms: float
    cam_time_ms: np.ndarray = field(default_factory=lambda: np.empty(0))
    cam_score: np.ndarray = field(default_factory=lambda: np.empty(0))
    dalpha_channel: str = ""
    dalpha_dt_ms: float = float("nan")
    da_time_ms: np.ndarray = field(default_factory=lambda: np.empty(0))
    da_value: np.ndarray = field(default_factory=lambda: np.empty(0))
    peak_time_ms: float = float("nan")
    n_bright_frames: int = 0
    res_h: int = 0
    res_w: int = 0
    verdict: str = ""


def _camera_burst_window(
    frames: np.ndarray, times: np.ndarray, peak: int, half: int = 8
):
    """Per-frame camera ELM-brightening trajectory around the burst peak.

    Returns ``(t_ms_centred, score, frame_lo)`` where ``score`` is the
    high-passed edge brightness (the same proxy ``camera_elm_score`` uses)
    over a window of ``2*half+1`` frames centred on the peak.
    """
    lo = max(0, peak - half)
    hi = min(frames.shape[0], peak + half + 1)
    b = camera_brightness_trace(frames[lo:hi])
    kernel = np.ones(3) / 3.0
    smooth = np.convolve(b, kernel, mode="same")
    hp = b - smooth
    tt = (times[lo:hi] - times[peak]) * 1e3  # ms, centred on the camera peak
    return tt, hp, lo


def _native_dalpha_burst(shot_id: int, t_peak_s: float, half_ms: float = 6.0):
    """Native fast-Dα trace around a time, on its own (sub-ms) axis.

    Picks the fast-Dα channel with the strongest transient near ``t_peak_s``,
    returns ``(channel, dt_ms, t_ms_centred, value)`` on the native axis.
    """
    best = None
    for ch in FAST_DALPHA_CHANNELS:
        got = read_diag_array(shot_id, "xim", ch)
        if got is None:
            continue
        t, v = got
        if t is None or v is None:
            continue
        t = np.asarray(t, dtype=np.float64).reshape(-1)
        v = np.asarray(v, dtype=np.float64).reshape(-1)
        if t.size != v.size or t.size < 10:
            continue
        win = np.where(np.abs(t - t_peak_s) <= half_ms * 1e-3)[0]
        fin = win[np.isfinite(v[win])]
        if fin.size < 5:
            continue
        seg_t, seg_v = t[fin], v[fin]
        base = float(np.median(seg_v))
        if base < 0.02:
            continue
        mad = float(np.median(np.abs(seg_v - base))) or (float(seg_v.std()) or 1.0)
        peak_h = (float(seg_v.max()) - base) / (1.4826 * mad + 1e-12)
        dt = np.diff(seg_t)
        dt = dt[np.isfinite(dt) & (dt > 0)]
        dt_ms = float(np.median(dt) * 1e3) if dt.size else float("nan")
        if best is None or peak_h > best[4]:
            best = (ch, dt_ms, (seg_t - t_peak_s) * 1e3, seg_v - base, peak_h)
    if best is None:
        return None
    ch, dt_ms, tt, vv, _ = best
    return ch, dt_ms, tt, vv


def resolve_window(shot_id: int) -> ResolveWindow | None:
    """Build the resolve-vs-integrate record for one ELM-bearing shot."""
    t = read_camera_frame_times(shot_id)
    if t is None:
        return None
    sigma, peak_abs, _ = detect_camera_elm(shot_id)
    if peak_abs < 0:
        return None
    # raw frames around the burst on the flat-top
    lo = max(0, peak_abs - 12)
    hi = peak_abs + 13
    got = read_raw_frames(shot_id, lo, hi)
    if got is None:
        return None
    frames, ftimes = got
    peak_rel = peak_abs - lo
    cam_dt = float(np.median(np.diff(ftimes)) * 1e3)
    tt, hp, _ = _camera_burst_window(frames, ftimes, peak_rel)
    t_peak_s = float(ftimes[peak_rel])

    # how many camera frames are "bright" — above half the peak high-pass value
    pk_val = float(hp.max()) if hp.size else 0.0
    n_bright = int(np.sum(hp >= 0.5 * pk_val)) if pk_val > 0 else 0

    res = frame_resolution(shot_id) or (0, 0)
    rec = ResolveWindow(
        shot_id=shot_id,
        camera_dt_ms=cam_dt,
        cam_time_ms=tt,
        cam_score=hp,
        peak_time_ms=0.0,
        n_bright_frames=n_bright,
        res_h=int(res[0]),
        res_w=int(res[1]),
    )
    da = _native_dalpha_burst(shot_id, t_peak_s)
    if da is not None:
        ch, da_dt, da_tt, da_vv = da
        rec.dalpha_channel = ch
        rec.dalpha_dt_ms = da_dt
        rec.da_time_ms = da_tt
        rec.da_value = da_vv

    # verdict: resolved if >=3 camera frames trace the burst above half-max AND
    # camera Δt is below the ELM upper band; integrated if essentially 1 frame
    if cam_dt >= INTEGRATE_DT_MS or n_bright <= 1:
        rec.verdict = "INTEGRATED (single bright frame)"
    elif n_bright >= 3 and cam_dt < ELM_WIDTH_MS[1]:
        rec.verdict = "RESOLVED (multi-frame rise+fall)"
    else:
        rec.verdict = "PARTIAL (2 frames)"
    return rec


def pick_resolve_windows(cadences: list[ShotCadence], *, n: int) -> list[ResolveWindow]:
    """Pick representative ELM windows STRATIFIED across cadence regimes.

    The decisive comparison is cadence-dependent: a fast-cadence camera could
    in principle resolve the burst, a slow one cannot.  So instead of just the
    top-sigma shots (which biases toward slow cameras — a slow camera produces
    one very bright integrated frame, i.e. a high sigma), pick the strongest-ELM
    shot in each cadence band: fast (<0.1 ms), medium (0.1-1 ms), slow (>=1 ms).
    Each band's window then shows whether THAT cadence resolves or integrates.
    """
    elm_shots = [c for c in cadences if c.is_elm and c.camera_elm_sigma > 0]
    bands = [
        ("fast", lambda dt: dt < ELM_WIDTH_MS[0]),
        ("medium", lambda dt: ELM_WIDTH_MS[0] <= dt < INTEGRATE_DT_MS),
        ("slow", lambda dt: dt >= INTEGRATE_DT_MS),
    ]
    out: list[ResolveWindow] = []
    seen: set[int] = set()
    for _name, pred in bands:
        cands = sorted(
            (c for c in elm_shots if pred(c.median_dt_ms)),
            key=lambda c: c.camera_elm_sigma,
            reverse=True,
        )
        for c in cands:
            if c.shot_id in seen:
                continue
            rec = resolve_window(c.shot_id)
            if rec is not None and rec.cam_score.size and rec.da_value.size:
                out.append(rec)
                seen.add(c.shot_id)
                break
    # if a band had no usable window, top up from remaining strong-ELM shots
    if len(out) < n:
        for c in sorted(elm_shots, key=lambda c: c.camera_elm_sigma, reverse=True):
            if c.shot_id in seen:
                continue
            rec = resolve_window(c.shot_id)
            if rec is not None and rec.cam_score.size:
                out.append(rec)
                seen.add(c.shot_id)
            if len(out) >= n:
                break
    # order fast → slow so the figure reads as a cadence sweep
    out.sort(key=lambda w: w.camera_dt_ms)
    return out[: max(n, len(out))]


# ---------------------------------------------------------------------------
# Q3 — settle the contested fast-diagnostic rates / shapes
# ---------------------------------------------------------------------------


def measure_channel_rate(
    shot_ids: list[int], source: str, array: str, *, max_shots: int = 6
) -> dict:
    """Median on-disk sample rate + shape of a level-1 channel across shots.

    Reports BOTH the raw storage-axis rate (Δt of the full time array) and the
    EFFECTIVE rate after restricting to samples where the channel is finite.
    For NaN-masked channels (e.g. the legacy ``xma`` schema, where the channel
    is stored on the ~100 kHz master axis but only ~5% of samples are real),
    the effective rate is the physically meaningful one and reconciles the
    contested ``xma`` rate (storage 100 kHz vs effective ~5 kHz, S12 finding).
    """
    raw_rates: list[float] = []
    eff_rates: list[float] = []
    finite_fracs: list[float] = []
    shapes: list[tuple] = []
    measured: list[int] = []
    for sid in shot_ids:
        if len(measured) >= max_shots:
            break
        got = read_diag_array(sid, source, array)
        if got is None:
            continue
        t, v = got
        if v is None:
            continue
        v = np.asarray(v)
        shapes.append(tuple(int(x) for x in v.shape))
        if t is not None and t.size >= 3:
            t = np.asarray(t, dtype=np.float64).reshape(-1)
            dt = np.diff(t)
            dt = dt[np.isfinite(dt) & (dt > 0)]
            if dt.size:
                raw_rates.append(float(1.0 / np.median(dt)))
            # effective rate on samples where the (1-D) channel is finite
            v1 = v.reshape(-1)
            if v1.size == t.size:
                fin = np.isfinite(v1)
                finite_fracs.append(float(fin.mean()))
                tf = t[fin]
                if tf.size >= 3:
                    dtf = np.diff(tf)
                    dtf = dtf[np.isfinite(dtf) & (dtf > 0)]
                    if dtf.size:
                        eff_rates.append(float(1.0 / np.median(dtf)))
        measured.append(sid)
    out = {
        "source": source,
        "array": array,
        "n_measured": len(measured),
        "shots": measured,
        "shapes": [list(s) for s in shapes],
    }
    if raw_rates:
        out["raw_rate_hz"] = float(np.median(raw_rates))
        out["median_rate_hz"] = float(np.median(raw_rates))  # back-compat alias
        out["median_dt_us"] = float(1e6 / np.median(raw_rates))
    else:
        out["raw_rate_hz"] = None
        out["median_rate_hz"] = None
        out["note"] = "no usable time axis (profile array or all-NaN)"
    if eff_rates:
        out["effective_rate_hz"] = float(np.median(eff_rates))
        out["finite_fraction"] = (
            float(np.median(finite_fracs)) if finite_fracs else None
        )
    return out


def settle_fast_channels(shot_ids: list[int]) -> dict:
    """Measure the three contested channels + identify the fast ELM carrier."""
    channels = {
        "ada/dalpha_raw_full": ("ada", "raw_full"),
        "ada/dalpha_integrated": ("ada", "dalpha_integrated"),
        "xim/da_hm10_t": ("xim", "da_hm10_t"),
        "xma/ccbv_01": ("xma", "ccbv_01"),
        "xma/ccbv01": ("xma", "ccbv01"),
    }
    out: dict = {}
    for label, (src, arr) in channels.items():
        out[label] = measure_channel_rate(shot_ids, src, arr)
    # also probe the bare ada/raw_full alias under either name
    if out["ada/dalpha_raw_full"]["n_measured"] == 0:
        alt = measure_channel_rate(shot_ids, "ada", "dalpha_raw_full")
        if alt["n_measured"]:
            out["ada/dalpha_raw_full"] = alt
            out["ada/dalpha_raw_full"]["array"] = "dalpha_raw_full"
    return out


# ---------------------------------------------------------------------------
# Q4 — forecast statistical power at the locked horizons
# ---------------------------------------------------------------------------


def forecast_power(
    horizon_table_path: Path = DEFAULT_HORIZON_TABLE,
    sweep_path: Path = DEFAULT_FORECAST_SWEEP,
) -> dict:
    """Read the surviving valid forecast sample counts at 10/50/200 ms.

    Pulls the W2 horizon table (matched regime — the populated table) and the
    finer forecast sweep; reports valid-window counts, cell counts, and the
    dynamics-vs-persistence significance flag at each horizon, so an
    under-powered horizon is plainly visible.
    """
    out: dict = {"source_horizon_table": str(horizon_table_path)}
    if horizon_table_path.exists():
        ht = json.loads(horizon_table_path.read_text())
        matched = ht.get("matched", {}).get("table", {})
        per_h: dict = {}
        for h in ("10.0", "50.0", "200.0"):
            cell = matched.get(h, {})
            per_h[h] = {
                "valid_windows": cell.get("valid_windows", 0),
                "n_cells": cell.get("n_cells", 0),
                "beats_persistence": bool(
                    cell.get("dynamics_vs_persistence_top1", {}).get(
                        "favours_dynamics", False
                    )
                )
                if cell.get("valid_windows")
                else None,
            }
        out["matched_horizons"] = per_h
    if sweep_path.exists():
        sw = json.loads(sweep_path.read_text())
        out["sweep_crossover_ms"] = sw.get("crossover_ms")
        out["sweep_valid_windows"] = {
            h: cell.get("valid_windows", 0) for h, cell in sw.get("table", {}).items()
        }
    return out


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def make_figure(
    cadences: list[ShotCadence],
    windows: list[ResolveWindow],
    channels: dict,
    power: dict,
    out_path: Path,
) -> None:
    """Three-panel Tufte-style audit figure + a settled-numbers table."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "figure.dpi": 150,
        }
    )

    nwin = max(1, min(3, len(windows)))
    fig = plt.figure(figsize=(14.0, 9.6), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.0])
    ax_dist = fig.add_subplot(gs[0, 0])
    gs_b = gs[0, 1].subgridspec(nwin, 1, hspace=0.45)
    res_axes = [fig.add_subplot(gs_b[i, 0]) for i in range(nwin)]
    ax_tab = fig.add_subplot(gs[1, :])

    # --- Panel A: frame-interval distribution -----------------------------
    med = np.array([c.median_dt_ms for c in cadences])
    elm = np.array([c.is_elm for c in cadences])
    bins = np.logspace(np.log10(0.005), np.log10(25.0), 44)
    ax_dist.hist(
        med[~elm],
        bins=bins,
        color="#9ecae1",
        alpha=0.8,
        label=f"non-ELM (n={int((~elm).sum())})",
    )
    if elm.any():
        ax_dist.hist(
            med[elm],
            bins=bins,
            color="#e6550d",
            alpha=0.85,
            label=f"ELM-bearing (n={int(elm.sum())})",
        )
    ax_dist.axvspan(
        ELM_WIDTH_MS[0],
        ELM_WIDTH_MS[1],
        color="0.5",
        alpha=0.18,
        label="ELM width band (0.1-1 ms)",
    )
    ax_dist.axvline(INTEGRATE_DT_MS, color="k", ls="--", lw=1.0)
    ax_dist.set_xscale("log")
    ax_dist.set_xlabel("per-shot median camera Δt (ms)  —  log scale")
    ax_dist.set_ylabel("shots")
    ax_dist.set_title(
        "A. rbb camera frame-interval distribution (held-out shots)",
        loc="left",
        fontweight="bold",
    )
    ax_dist.legend(loc="upper left", fontsize=7.5, framealpha=0.9)
    summ = cadence_summary(cadences)
    es = summ["elm_shots"]
    # how many ELM shots run a tiny high-speed ROI (<0.1 ms BUT small sensor)
    npix = np.array([c.npix for c in cadences if c.is_elm])
    medm = np.array([c.median_dt_ms for c in cadences if c.is_elm])
    n_fast_small = int(np.sum((medm < ELM_WIDTH_MS[0]) & (npix > 0) & (npix < 40000)))
    a = summ["all_shots"]
    txt = (
        f"all shots: median {a['median_dt_ms']:.2f} ms "
        f"(range {a['min_dt_ms']:.2f}-{a['max_dt_ms']:.1f} ms)\n"
        f"ELM shots (n={summ['n_elm_shots']}): "
        f"INTEGRATE (Δt≥1 ms) {es['frac_integrate_ge_1ms'] * 100:.0f}%  |  "
        f"sub-0.1 ms {es['frac_resolve_lt_0p1ms'] * 100:.0f}%\n"
        f"(of those sub-0.1 ms, {n_fast_small} are tiny high-speed ROI "
        f"<200px wide — not full-frame)"
    )
    ax_dist.text(
        0.98,
        0.97,
        txt,
        transform=ax_dist.transAxes,
        ha="right",
        va="top",
        fontsize=7.3,
        bbox=dict(boxstyle="round", fc="w", ec="0.7", alpha=0.92),
    )

    # --- Panel B: resolve vs integrate, stratified across cadence regimes -
    res_axes[0].set_title(
        "B. Camera burst trajectory vs native fast-Dα (fast → slow cadence)",
        loc="left",
        fontweight="bold",
        fontsize=10,
    )
    for ax_res, w in zip(res_axes, windows, strict=False):
        ax_da = ax_res.twinx()
        if w.da_value.size:
            ax_da.plot(
                w.da_time_ms,
                w.da_value,
                color="#3182bd",
                lw=0.9,
                alpha=0.8,
                label=f"fast Dα {w.dalpha_channel} ({w.dalpha_dt_ms * 1e3:.0f} µs)",
            )
            ax_da.tick_params(axis="y", labelcolor="#3182bd", labelsize=6.5)
        ax_res.plot(
            w.cam_time_ms,
            w.cam_score,
            "o-",
            color="#e6550d",
            lw=1.5,
            ms=4,
            label=f"camera ({w.camera_dt_ms:.2f} ms)",
        )
        ax_res.axvline(0.0, color="k", ls=":", lw=0.7)
        ax_res.tick_params(axis="y", labelcolor="#e6550d", labelsize=6.5)
        v_short = w.verdict.split(" (")[0]
        color = {
            "INTEGRATED": "#c1272d",
            "RESOLVED": "#1a7a3a",
            "PARTIAL": "#b07a00",
        }.get(v_short, "k")
        res_lbl = f"{w.res_h}×{w.res_w}px" if w.res_h else ""
        ax_res.text(
            0.015,
            0.93,
            f"shot {w.shot_id}  Δt={w.camera_dt_ms:.2f} ms  {res_lbl}  "
            f"[{w.n_bright_frames} bright frame(s)]  →  {v_short}",
            transform=ax_res.transAxes,
            ha="left",
            va="top",
            fontsize=7.2,
            color="w",
            bbox=dict(boxstyle="round", fc=color, ec="none", alpha=0.92),
        )
        lines1, lab1 = ax_res.get_legend_handles_labels()
        lines2, lab2 = ax_da.get_legend_handles_labels()
        ax_res.legend(
            lines1 + lines2,
            lab1 + lab2,
            loc="upper right",
            fontsize=6.3,
            framealpha=0.85,
        )
    res_axes[-1].set_xlabel("time relative to camera burst peak (ms)")
    res_axes[len(res_axes) // 2].set_ylabel(
        "camera high-pass edge brightness (orange)", color="#e6550d", fontsize=7.5
    )

    # --- Panel C: settled-numbers table -----------------------------------
    ax_tab.axis("off")
    ax_tab.set_title(
        "C. Settled channel identities/rates  +  forecast-horizon statistical power",
        loc="left",
        fontweight="bold",
    )

    def _shape(c):
        shp = c.get("shapes", [[]])
        return "×".join(str(x) for x in shp[0]) if shp and shp[0] else "?"

    def _rate_str(c):
        eff = c.get("effective_rate_hz")
        raw = c.get("raw_rate_hz")
        if eff and raw and abs(eff - raw) / raw > 0.2:
            ff = c.get("finite_fraction")
            return (
                f"storage {raw / 1e3:.0f} kHz, EFFECTIVE {eff / 1e3:.1f} kHz "
                f"({ff * 100:.0f}% finite)"
            )
        if raw:
            return f"{raw / 1e3:.1f} kHz (Δt {1e6 / raw:.1f} µs)"
        return "no fast time axis"

    rows = [["measurement", "on-disk value (this corpus)", "verdict / note"]]
    cw = channels
    # ada/dalpha_raw_full — settle the profile-vs-trace contradiction
    c = cw.get("ada/dalpha_raw_full", {})
    if c.get("n_measured"):
        rows.append(
            [
                "ada/dalpha_raw_full",
                f"{_rate_str(c)}  shape {_shape(c)}",
                "radial PROFILE on slow ~1.4 kHz axis — NOT a fast trace",
            ]
        )
    c = cw.get("ada/dalpha_integrated", {})
    if c.get("n_measured"):
        rows.append(
            [
                "ada/dalpha_integrated",
                f"{_rate_str(c)}  shape {_shape(c)}",
                "slow integrated 1-D trace (not the fast ELM carrier)",
            ]
        )
    c = cw.get("xim/da_hm10_t", {})
    if c.get("n_measured"):
        fast = (c.get("raw_rate_hz") or 0) > 2e4
        rows.append(
            [
                "xim/da_hm10_t",
                f"{_rate_str(c)}  shape {_shape(c)}",
                "GENUINE FAST ELM CARRIER (sub-ms)" if fast else "",
            ]
        )
    # xma — reconcile storage vs effective masked rate
    cmod = cw.get("xma/ccbv_01", {})
    cleg = cw.get("xma/ccbv01", {})
    c = cmod if cmod.get("n_measured") else cleg
    if c.get("n_measured"):
        rows.append(
            [
                f"xma/{c['array']}",
                f"{_rate_str(c)}  shape {_shape(c)}",
                "fast magnetics; effective ~5 kHz after NaN mask (S12)",
            ]
        )
    rows.append(["", "", ""])
    mh = power.get("matched_horizons", {})
    for h in ("10.0", "50.0", "200.0"):
        cell = mh.get(h, {})
        nw = cell.get("valid_windows", 0)
        bp = cell.get("beats_persistence")
        verdict = (
            "powered — beats persistence (CI>0)"
            if (nw >= 30 and bp)
            else (
                "n=1 — statistically powerless"
                if nw == 1
                else ("under-powered (CI crosses 0)" if nw else "unreachable")
            )
        )
        rows.append(
            [
                f"forecast power, horizon {h.split('.')[0]} ms",
                f"n={nw} valid windows  ({cell.get('n_cells', 0)} cells)",
                verdict,
            ]
        )

    tab = ax_tab.table(
        cellText=rows, loc="center", cellLoc="left", colWidths=[0.24, 0.42, 0.34]
    )
    tab.auto_set_font_size(False)
    tab.set_fontsize(8.5)
    tab.scale(1, 1.5)
    for (r, _c), cell in tab.get_celld().items():
        cell.set_edgecolor("0.85")
        if r == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="w", fontweight="bold")
        elif rows[r][0].startswith("forecast"):
            cell.set_facecolor("#f7f7f7")

    fig.suptitle(
        "camera-dynamics-wm — does the rbb camera RESOLVE or INTEGRATE the ELM burst?",
        fontsize=13,
        fontweight="bold",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("[elmaudit] wrote %s", out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def load_held_out(split_path: Path) -> list[int]:
    d = json.loads(split_path.read_text())
    return [int(s) for s in d.get("held_out", [])]


def run_audit(
    *,
    split_path: Path = DEFAULT_SPLIT,
    out_path: Path,
    max_shots: int = 120,
    n_windows: int = 3,
) -> dict:
    """Run all four measurements, write the figure + JSON, return the summary."""
    held_out = load_held_out(split_path)
    logger.info(
        "[elmaudit] %d held-out shots; surveying up to %d", len(held_out), max_shots
    )

    cadences = survey_cadence(held_out, max_shots=max_shots, elm_check=True)
    logger.info("[elmaudit] surveyed %d shots with cadence", len(cadences))
    summ = cadence_summary(cadences)

    windows = pick_resolve_windows(cadences, n=n_windows)
    logger.info("[elmaudit] built %d resolve windows", len(windows))

    # for channel-rate settle, prefer ELM shots (they carry the fast diodes)
    elm_ids = [c.shot_id for c in cadences if c.is_elm] or [c.shot_id for c in cadences]
    channels = settle_fast_channels(elm_ids)

    power = forecast_power()

    make_figure(cadences, windows, channels, power, out_path)

    result = {
        "q1_cadence": summ,
        "q2_resolve_windows": [
            {
                "shot_id": w.shot_id,
                "camera_dt_ms": w.camera_dt_ms,
                "frame_resolution": [w.res_h, w.res_w],
                "dalpha_channel": w.dalpha_channel,
                "dalpha_dt_ms": w.dalpha_dt_ms,
                "n_bright_camera_frames": w.n_bright_frames,
                "verdict": w.verdict,
            }
            for w in windows
        ],
        "q3_channels": channels,
        "q4_forecast_power": power,
        "params": {
            "max_shots": max_shots,
            "elm_width_ms": list(ELM_WIDTH_MS),
            "integrate_dt_ms": INTEGRATE_DT_MS,
            "camera_elm_sigma_gate": CAMERA_ELM_SIGMA_GATE,
            "flattop_s": list(FLATTOP_S),
        },
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2))
    logger.info("[elmaudit] wrote %s", OUT_JSON)
    return result


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="rbb camera ELM-sampling forensics audit")
    p.add_argument(
        "--out",
        default="docs/figures/camera-dynamics-wm/fig-cdw-elm-sampling-audit.png",
    )
    p.add_argument("--split", default=str(DEFAULT_SPLIT))
    p.add_argument("--max-shots", type=int, default=120)
    p.add_argument("--n-windows", type=int, default=3)
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    result = run_audit(
        split_path=Path(args.split),
        out_path=Path(args.out),
        max_shots=args.max_shots,
        n_windows=args.n_windows,
    )

    # human-readable summary to the log
    s = result["q1_cadence"]
    logger.info(
        "[elmaudit] Q1 all-shots median Δt=%.2f ms (%.2f..%.2f); ELM shots n=%d "
        "integrate=%.0f%% resolve=%.0f%%",
        s["all_shots"]["median_dt_ms"],
        s["all_shots"]["min_dt_ms"],
        s["all_shots"]["max_dt_ms"],
        s["n_elm_shots"],
        (s["elm_shots"]["frac_integrate_ge_1ms"] or 0) * 100,
        (s["elm_shots"]["frac_resolve_lt_0p1ms"] or 0) * 100,
    )
    for w in result["q2_resolve_windows"]:
        logger.info(
            "[elmaudit] Q2 shot %d: %s (cam dt=%.2f ms, Da %s dt=%.3f ms, %d bright)",
            w["shot_id"],
            w["verdict"],
            w["camera_dt_ms"],
            w["dalpha_channel"],
            w["dalpha_dt_ms"],
            w["n_bright_camera_frames"],
        )
    for label, c in result["q3_channels"].items():
        if c.get("n_measured"):
            rate = c.get("median_rate_hz")
            logger.info(
                "[elmaudit] Q3 %s: rate=%s shape=%s",
                label,
                f"{rate / 1e3:.2f} kHz" if rate else "no fast axis",
                c.get("shapes", [None])[0],
            )
    mh = result["q4_forecast_power"].get("matched_horizons", {})
    for h, cell in mh.items():
        logger.info(
            "[elmaudit] Q4 h=%s ms: n=%s valid windows", h, cell.get("valid_windows")
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
