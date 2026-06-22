"""Camera-frame windows + measured-signal conditioning for the v2 world model.

Extends :mod:`imas_ambix.worldmodel.spacetime_dataset` (camera frames + the
pulse-schedule plan) with the MEASURED diagnostic streams as additional
conditioning context: magnetics (``xma``), density (``interferometer``),
``soft_x_rays``, and the L2 measured groups (``summary`` / ``pf_active`` /
``gas_injection``).  The camera-frame target, the plan prefix, the windowing,
and the local-id rebasing are all reused from v1 — this module only adds the
signal read + grid-resample step and a v2 sample/collate that carry the signals
alongside the frames + plan.

A signal stream for a window
----------------------------
Each measured stream lives on its OWN native cadence (xma ~78 Hz, the L2 groups
4 kHz, …).  For a camera window spanning ``[t0, t1]`` we sub-sample each present
stream to ``n_signal_steps`` evenly-spaced positions across that span, taking the
nearest native token per position (no half-step tolerance gate — a conditioning
context is allowed to use the nearest reading, unlike a predicted target).  A
stream with no readable store for the shot is simply OMITTED (the model then
conditions on the plan + whatever streams ARE present, and touches the absent
streams' params with a zero contribution to keep DDP uniform).

Per-group-local ids + the boundary guard are inherited
-------------------------------------------------------
Signal reads go through :func:`imas_ambix.worldmodel.dataset._read_signal_hf`,
which rebases on-disk global ids to per-group-local ids and routes the store path
through :func:`imas_ambix.tokenizer.store_targets.assert_not_target_path` — so an
eval-only reconstruction target can never be ingested as a conditioning input.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from imas_ambix.worldmodel.dataset import PAD_LOCAL_ID, _read_signal_hf
from imas_ambix.worldmodel.spacetime_dataset import (
    REFERENCE_CAMERA,
    SpacetimeSample,
    SpacetimeWindowConfig,
    assemble_window,
    camera_frame_count,
)
from imas_ambix.worldmodel.spacetime_model_v2 import SignalStreamSpec

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default conditioning signal streams
# ---------------------------------------------------------------------------
#
# group is the on-disk signals_hf store-group name; vocab matches the encoder.
# The L2 measured groups share the 257-id uniform-quantiser vocab (256 bins +
# PAD); the HF magnetics ``xma`` is a small discrete codebook.  Density
# (interferometer) and soft_x_rays are L2 groups.  ``channels`` is 0 == "probe
# from disk" — the dataset fills it at discovery (coil/chord counts vary per
# shot), capped by ``max_channels`` so a wide stream cannot exceed the spatial
# lane budget.


@dataclass(frozen=True)
class SignalModalitySpec:
    """One measured conditioning stream (dataset side).

    Attributes
    ----------
    name:
        Stream key (matches the model's :class:`SignalStreamSpec` name).
    group:
        On-disk ``signals_hf`` store-group name.
    vocab:
        Local vocabulary size for the stream's value-embedding table.
    max_channels:
        Cap on the channels fed to the model (the first ``max_channels`` channels
        are kept).  Keeps a wide stream within the spatial-lane budget and bounds
        the conditioning cost.
    kind:
        Which on-disk store layout the stream is read from.  ``"signal_hf"``
        (default) is a PRE-tokenised ``signals_hf`` store read by
        :func:`imas_ambix.worldmodel.dataset._read_signal_hf` (the L2 light-path
        groups + the HF patch streams).  ``"ait"`` is the RAW-float divertor
        heat-flux store (``.../signals-ait/ait/<shot>/ait.zarr`` — a ``time`` axis
        + named 1-D traces + 2-D profiles) read + quantised on the fly by
        :func:`_read_ait_signal` to L2-style local ids (256 bins + PAD), so it
        joins the same per-stream embedding contract without a pre-tokenise step.
    """

    name: str
    group: str
    vocab: int
    max_channels: int = 64
    kind: str = "signal_hf"


def _l2_vocab() -> int:
    from imas_ambix.tokenizer.registry import L2_BLOCK_VOCAB  # noqa: PLC0415

    return int(L2_BLOCK_VOCAB) + 1


#: HF soft-x-ray + Dα/CII codebook sizes (the native-cadence ``signal_hf``
#: groups, matching the encoder — see :mod:`imas_ambix.worldmodel.dataset`).
_XSX_VOCAB = 1030
_XIM_VOCAB = 12806


def default_signal_modalities() -> list[SignalModalitySpec]:
    """The measured streams conditioned on by default.

    Magnetics (``xma``), density (``interferometer``), ``soft_x_rays``, and the
    L2 measured groups ``summary`` / ``pf_active`` / ``gas_injection``.  The plan
    (``pulse_schedule``) is NOT here — it rides v1's dedicated plan prefix.
    """
    l2 = _l2_vocab()
    return [
        SignalModalitySpec("summary", "summary_l2", l2, max_channels=48),
        SignalModalitySpec("pf_active", "pf_active_l2", l2, max_channels=48),
        SignalModalitySpec("interferometer", "interferometer_l2", l2, max_channels=24),
        SignalModalitySpec("gas_injection", "gas_injection_l2", l2, max_channels=24),
        SignalModalitySpec("soft_x_rays", "soft_x_rays_l2", l2, max_channels=48),
        SignalModalitySpec("xma", "xma", 8, max_channels=48),
    ]


def extended_signal_modalities() -> list[SignalModalitySpec]:
    """:func:`default_signal_modalities` + the tokenised-but-unfused HF streams.

    Adds the native-cadence HF soft-x-ray (``xsx``) and Dα/CII (``xim``)
    ``signal_hf`` groups — already on disk but not yet fused into the camera world
    model — plus a hook for an ``ait`` divertor-heat-flux stream (encoded through
    the L2 uniform quantiser, so it shares the L2 vocab).  A stream whose store is
    absent for a shot is silently OMITTED by the dataset read (the per-stream
    ``FileNotFoundError`` / ``KeyError`` guard in :func:`read_window_signals`), so
    listing ``ait`` here before its corpus-wide encode lands is safe — present
    shots condition on it, absent shots simply don't.

    Kept SEPARATE from :func:`default_signal_modalities` so the existing trained
    streams + their order are unchanged (a prior checkpoint's per-stream tables
    still load by name); the broad multi-camera run opts INTO the wider set.
    """
    l2 = _l2_vocab()
    mods = default_signal_modalities()
    mods += [
        SignalModalitySpec("xsx", "xsx", _XSX_VOCAB, max_channels=48),
        SignalModalitySpec("xim", "xim", _XIM_VOCAB, max_channels=24),
        # ait divertor heat flux — a RAW-float store (20 1-D traces + 4 (T,186)
        # heat-flux/temperature profiles) quantised ON READ to L2 local ids
        # (256 bins + PAD = 257 vocab); see :func:`_read_ait_signal`.  The
        # profiles dominate the channel count, so the R axis is sub-sampled
        # (profile_r_stride) — cap the kept channels to the lane budget.
        SignalModalitySpec("ait", "ait", l2, max_channels=48, kind="ait"),
    ]
    return mods


# ---------------------------------------------------------------------------
# v2 sample
# ---------------------------------------------------------------------------


@dataclass
class SignalSpacetimeSample:
    """A v1 :class:`SpacetimeSample` + the measured-signal conditioning.

    Attributes
    ----------
    base:
        The v1 camera-frame window + plan prefix (frames, plan, times, context).
    signals:
        ``{stream_name: (n_signal_steps, n_channels) int64 local ids}`` for every
        stream readable for the shot.  A stream absent for the shot is OMITTED
        (not zero-padded) so the model conditions only on present streams.
    """

    base: SpacetimeSample
    signals: dict[str, np.ndarray] = field(default_factory=dict)

    # convenient pass-throughs so a v2 sample is a drop-in for v1 consumers.
    @property
    def shot_id(self) -> int:
        return self.base.shot_id

    @property
    def camera(self) -> str:
        return self.base.camera

    @property
    def start_frame(self) -> int:
        return self.base.start_frame

    @property
    def frames(self) -> np.ndarray:
        return self.base.frames

    @property
    def plan(self) -> np.ndarray:
        return self.base.plan

    @property
    def frame_time(self) -> np.ndarray:
        return self.base.frame_time

    @property
    def context_frames(self) -> int:
        return self.base.context_frames

    @property
    def n_frames(self) -> int:
        return self.base.n_frames


# ---------------------------------------------------------------------------
# Signal read + grid resample
# ---------------------------------------------------------------------------


def _nearest_steps(
    native_time: np.ndarray,
    native_tokens: np.ndarray,
    grid_time: np.ndarray,
    *,
    max_channels: int,
) -> np.ndarray:
    """Sub-sample one stream to the grid steps (nearest native token per step).

    ``native_tokens`` is ``(n_native, n_ch)`` local ids; returns
    ``(n_grid, min(n_ch, max_channels)) int64``.  Unlike the predicted-target
    resampler in :mod:`imas_ambix.worldmodel.dataset`, a conditioning context may
    use the nearest reading regardless of distance (no half-step gate) — it is
    context, never scored — but a step with NO native data at all is PAD-filled.
    """
    n_grid = grid_time.shape[0]
    n_ch = native_tokens.shape[1] if native_tokens.ndim == 2 else 0
    n_ch = min(n_ch, int(max_channels))
    out = np.full((n_grid, n_ch), PAD_LOCAL_ID, dtype=np.int64)
    if native_time.size == 0 or n_ch == 0:
        return out
    order = np.argsort(native_time)
    nt = native_time[order]
    ntok = native_tokens[order][:, :n_ch]
    idx = np.searchsorted(nt, grid_time)
    idx = np.clip(idx, 0, nt.size - 1)
    left = np.clip(idx - 1, 0, nt.size - 1)
    pick_left = np.abs(nt[left] - grid_time) <= np.abs(nt[idx] - grid_time)
    chosen = np.where(pick_left, left, idx)
    out[:] = ntok[chosen]
    return out


# ---------------------------------------------------------------------------
# ait divertor-heat-flux RAW-float signal store -> L2-style tokens
# ---------------------------------------------------------------------------
#
# The ait store is NOT pre-tokenised (unlike the L2 / HF signal_hf groups): it
# holds a per-shot ``time`` axis plus named 1-D traces and 2-D (time, R) profiles
# of raw float divertor heat-flux quantities.  We quantise it ON READ with the
# SAME per-channel mean/std uniform-quantiser the L2 encoder uses
# (:class:`imas_ambix.tokenizer.signals.UniformQuantizer`) — z-score, clip to
# +-clip_sigma, map linearly into the bins — so an ait channel joins the
# per-stream embedding contract on the same 257-id vocab (256 bins + PAD) as the
# L2 groups.  There is no fitted corpus mean/std at read time, so we use the
# shot's OWN finite mean/std (the encoder's per-encode fallback), which is the
# honest scale for a single shot's window.

#: ait raw-float store quantiser params — match the L2 UniformQuantizer default
#: (256 magnitude bins, 4-sigma clip).  Bins are mapped to local ids 1..256 with
#: PAD reserved at 0 (so a real reading is never confused with PAD / an absent
#: step), giving the L2 257-id local vocab.
_AIT_N_BINS = 256
_AIT_CLIP_SIGMA = 4.0


def _ait_store_path(shot_id: int, *, token_root: Path | None = None):
    """Resolve the raw-float ait heat-flux store + route it through the guard.

    Layout: ``<token_root>/v1/signals-ait/ait/<shot>/ait.zarr``.  The path is run
    through :func:`imas_ambix.tokenizer.store_targets.assert_not_target_path` so
    a ``token_root`` resolving under the eval-only target root is hard-refused
    before any open (the same boundary guard the other signal reads honour).
    """
    from pathlib import Path as _Path  # noqa: PLC0415

    from imas_ambix.data.paths import TOKEN_ROOT  # noqa: PLC0415
    from imas_ambix.tokenizer.store_targets import (  # noqa: PLC0415
        assert_not_target_path,
    )

    root = _Path(token_root) if token_root is not None else _Path(TOKEN_ROOT)
    path = root / "v1" / "signals-ait" / "ait" / str(int(shot_id)) / "ait.zarr"
    return assert_not_target_path(path)


def _quantise_l2(values: np.ndarray) -> np.ndarray:
    """``(T, C) float -> (T, C) int64`` local ids 1..256 (L2 uniform-quantiser).

    Per-channel z-score against the channel's OWN finite mean/std (no fitted
    corpus stats at read time), clip to +-clip_sigma, map linearly into
    ``[0, n_bins-1]``, then shift by +1 so bin 0 becomes local id 1 and PAD (0)
    stays reserved.  NaNs map to the channel mean (z=0 -> the centre bin), never
    to PAD, so a finite-but-central reading and a missing step are distinct.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    out = np.zeros(arr.shape, dtype=np.int64)
    for c in range(arr.shape[1]):
        col = arr[:, c]
        finite = col[np.isfinite(col)]
        if finite.size == 0:
            out[:, c] = 1  # all-NaN channel -> centre-ish; valid mask handles it
            continue
        mean = float(finite.mean())
        std = float(finite.std())
        std = std if std > 1e-12 else 1.0
        z = np.nan_to_num((col - mean) / std, nan=0.0)
        z = np.clip(z, -_AIT_CLIP_SIGMA, _AIT_CLIP_SIGMA)
        local = np.round(
            ((z + _AIT_CLIP_SIGMA) / (2 * _AIT_CLIP_SIGMA)) * (_AIT_N_BINS - 1)
        ).astype(np.int64)
        local = np.clip(local, 0, _AIT_N_BINS - 1) + 1  # shift PAD off 0
        out[:, c] = local
    return out


def _read_ait_signal(
    shot_id: int,
    *,
    token_root: Path | None = None,
    profile_r_stride: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """Read + quantise the raw-float ait heat-flux store for a shot.

    Returns ``(tokens (T, C) int64 local ids, time (T,) float64)`` where the
    columns are the 1-D traces (in stored order) followed by the 2-D profiles
    sub-sampled along R by ``profile_r_stride`` (the rich heat-flux structure;
    sub-sampled like the xsx chords to bound the lane budget) and flattened.  A
    static R-axis array (``rcoord_*``) is NOT a channel (it is not time-resolved)
    and is skipped.  Raises ``FileNotFoundError`` / ``KeyError`` when the store is
    absent (a shot with no ait) so the caller omits the stream exactly like the
    other optional streams.
    """
    import zarr  # noqa: PLC0415

    path = _ait_store_path(shot_id, token_root=token_root)
    store = zarr.open_group(str(path), mode="r")  # raises if absent
    if "time" not in store:
        raise KeyError(f"ait store {path} has no 'time' axis")
    time = np.asarray(store["time"], dtype=np.float64).reshape(-1)
    n_t = time.shape[0]
    # Collect the 1-D TRACES first (the cheap, high-value channels) then the 2-D
    # PROFILES (the rich heat-flux structure, R-subsampled), each in sorted key
    # order — a DETERMINISTIC, traces-first column layout so a max_channels cap
    # keeps the traces preferentially and the order is stable across shots.
    traces: list[np.ndarray] = []
    profiles: list[np.ndarray] = []
    r_stride = max(1, int(profile_r_stride))
    for key in sorted(store.array_keys()):
        if key == "time":
            continue
        arr = np.asarray(store[key])
        if arr.ndim == 1 and arr.shape[0] == n_t:
            traces.append(_quantise_l2(arr.astype(np.float64)))  # (T, 1)
        elif arr.ndim == 2 and arr.shape[0] == n_t:
            profiles.append(_quantise_l2(arr[:, ::r_stride].astype(np.float64)))
        # any other shape (e.g. the static (R,) rcoord axis) is not a per-time
        # channel and is skipped.
    cols = traces + profiles
    if not cols:
        raise KeyError(f"ait store {path} has no time-resolved channels")
    tokens = np.concatenate(cols, axis=1)  # (T, C) — traces then profiles
    return tokens, time


def read_window_signals(
    shot_id: int,
    sample: SpacetimeSample,
    modalities: Sequence[SignalModalitySpec],
    n_signal_steps: int,
    *,
    token_root: Path | None = None,
) -> dict[str, np.ndarray]:
    """Read + resample every present measured stream onto the camera window.

    The window's time span comes from ``sample.frame_time`` (the camera frame
    timestamps).  Each present stream is sub-sampled to ``n_signal_steps``
    evenly-spaced positions across that span.  A stream with no readable store is
    omitted.  Returns ``{stream_name: (n_signal_steps, n_channels) int64}``.
    """
    if n_signal_steps <= 0 or not modalities:
        return {}
    ftime = np.asarray(sample.frame_time, dtype=np.float64)
    if ftime.size < 2:
        return {}
    t0, t1 = float(ftime.min()), float(ftime.max())
    if not (t1 > t0):
        return {}
    grid = np.linspace(t0, t1, int(n_signal_steps), dtype=np.float64)
    out: dict[str, np.ndarray] = {}
    for m in modalities:
        try:
            if m.kind == "ait":
                # RAW-float divertor heat-flux store, quantised on read to L2
                # local ids — resampled to the grid by the same nearest-step
                # rule as the pre-tokenised streams.
                tok, ttime = _read_ait_signal(shot_id, token_root=token_root)
            else:
                tok, ttime, _valid, _names, _base = _read_signal_hf(
                    shot_id, m.group, token_root=token_root
                )
        except (FileNotFoundError, KeyError) as exc:
            logger.debug("shot %s signal %s unreadable: %r", shot_id, m.name, exc)
            continue
        if tok.size == 0 or ttime.size == 0:
            continue
        steps = _nearest_steps(ttime, tok, grid, max_channels=m.max_channels)
        steps = np.clip(steps, 0, m.vocab - 1)
        if steps.shape[1] == 0:
            continue
        out[m.name] = steps.astype(np.int64)
    return out


def stream_specs_from_modalities(
    modalities: Sequence[SignalModalitySpec],
    channels: dict[str, int],
) -> list[SignalStreamSpec]:
    """Build the model's :class:`SignalStreamSpec` list from probed channels.

    ``channels`` maps stream name -> the channel count probed for the corpus
    (capped at ``max_channels``).  A stream with 0 probed channels is dropped (no
    shot carried it).  The order follows ``modalities`` so the model's stream
    order is deterministic and matches the dataset.
    """
    specs: list[SignalStreamSpec] = []
    for m in modalities:
        c = int(channels.get(m.name, 0))
        if c <= 0:
            continue
        specs.append(SignalStreamSpec(name=m.name, vocab=int(m.vocab), channels=c))
    return specs


def probe_signal_channels(
    shot_ids: Sequence[int],
    config: SpacetimeWindowConfig,
    modalities: Sequence[SignalModalitySpec],
    n_signal_steps: int,
    *,
    camera: str = REFERENCE_CAMERA,
    cameras: Sequence[str] | None = None,
    token_root: Path | None = None,
    max_probe: int = 8,
) -> dict[str, int]:
    """Probe a few shots for each stream's channel count (model sizing).

    Returns ``{stream_name: max channel count seen (capped at max_channels)}``.
    A stream never seen keeps 0 and is dropped from the model's stream list.

    ``cameras`` (optional, parallel to ``shot_ids``) gives a PER-SHOT camera — the
    multi-camera corpus mixes views, so a shot may have no recording in the single
    ``camera``; the per-shot camera is the one the dataset will actually read.  The
    measured-signal channel counts are camera-INDEPENDENT, but the camera frame
    window sets the time span the signals are resampled onto, so assembling with
    the right camera avoids skipping shots that lack the default view.
    """
    seen: dict[str, int] = {m.name: 0 for m in modalities}
    n = 0
    for i, sid in enumerate(shot_ids):
        if n >= max_probe:
            break
        cam = cameras[i] if cameras is not None and i < len(cameras) else camera
        try:
            sample = assemble_window(
                int(sid), config, camera=cam, token_root=token_root
            )
        except (ValueError, FileNotFoundError, KeyError):
            continue
        sigs = read_window_signals(
            int(sid), sample, modalities, n_signal_steps, token_root=token_root
        )
        for name, arr in sigs.items():
            seen[name] = max(seen[name], int(arr.shape[1]))
        n += 1
    return seen


# ---------------------------------------------------------------------------
# Assembly + dataset
# ---------------------------------------------------------------------------


def assemble_signal_window(
    shot_id: int,
    config: SpacetimeWindowConfig,
    modalities: Sequence[SignalModalitySpec],
    n_signal_steps: int,
    *,
    camera: str = REFERENCE_CAMERA,
    token_root: Path | None = None,
    start_frame: int | None = None,
) -> SignalSpacetimeSample:
    """Assemble a v1 camera window + the measured-signal conditioning."""
    base = assemble_window(
        shot_id, config, camera=camera, token_root=token_root, start_frame=start_frame
    )
    signals = read_window_signals(
        shot_id, base, modalities, n_signal_steps, token_root=token_root
    )
    return SignalSpacetimeSample(base=base, signals=signals)


class SignalSpacetimeDataset:
    """Map-style dataset of camera windows + measured-signal conditioning.

    Mirrors :class:`imas_ambix.worldmodel.spacetime_dataset.SpacetimeFrameDataset`
    (lazy Zarr-on-demand, worker-safe, optional random window jitter) and adds
    the per-window measured-signal read.
    """

    def __init__(
        self,
        shot_ids: Sequence[int],
        config: SpacetimeWindowConfig,
        modalities: Sequence[SignalModalitySpec],
        n_signal_steps: int,
        *,
        camera: str = REFERENCE_CAMERA,
        token_root: Path | None = None,
        random_window: bool = False,
        seed: int = 0,
    ) -> None:
        self._shot_ids = [int(s) for s in shot_ids]
        self._config = config
        self._modalities = list(modalities)
        self._n_signal_steps = int(n_signal_steps)
        self._camera = camera
        self._token_root = token_root
        self._random = bool(random_window)
        self._seed = int(seed)

    def __len__(self) -> int:
        return len(self._shot_ids)

    def __getitem__(self, index: int) -> SignalSpacetimeSample:
        sid = self._shot_ids[index]
        start = None
        if self._random:
            import random  # noqa: PLC0415

            span = (self._config.n_frames - 1) * self._config.frame_stride + 1
            try:
                n_total = camera_frame_count(
                    sid, self._camera, token_root=self._token_root
                )
            except (FileNotFoundError, KeyError, ValueError):
                n_total = span
            hi = max(0, n_total - span)
            rng = random.Random((self._seed * 1_000_003) ^ (sid * 31) ^ index)
            start = rng.randint(0, hi) if hi > 0 else 0
        return assemble_signal_window(
            sid,
            self._config,
            self._modalities,
            self._n_signal_steps,
            camera=self._camera,
            token_root=self._token_root,
            start_frame=start,
        )

    @property
    def config(self) -> SpacetimeWindowConfig:
        return self._config

    @property
    def modalities(self) -> list[SignalModalitySpec]:
        return self._modalities


# ---------------------------------------------------------------------------
# Overlapping-window enumeration (maximise signal from the fixed corpus)
# ---------------------------------------------------------------------------
#
# The map-style dataset above draws ONE window per shot per epoch (centred, or a
# random valid start when random_window).  A ~6000-frame recording therefore
# contributes a SINGLE 24-frame example per epoch — most of every shot is never
# seen.  Enumerating MULTIPLE OVERLAPPING windows (a sliding window, stride <
# n_frames-span) turns each long recording into many examples, a large increase
# in training signal from the SAME fixed corpus with no new data.
#
# LEAKAGE GUARD (binding): the window list is built ONLY from the shot ids handed
# in.  The train/val/held-out split stays at the SHOT level — whole pulses are
# held out — so as long as the caller passes ONLY train shots here, no window
# from a held-out shot can ever enter the list.  enumerate_windows takes a plain
# shot-id list and never reaches outside it; the trainer subtracts the held-out
# shots BEFORE calling it, and a test asserts the emitted set carries zero
# held-out ids.


def enumerate_windows(
    shot_ids: Sequence[int],
    config: SpacetimeWindowConfig,
    *,
    window_stride: int | None = None,
    camera: str = REFERENCE_CAMERA,
    token_root: Path | None = None,
    max_windows_per_shot: int | None = None,
) -> list[tuple[int, int]]:
    """Enumerate sliding ``(shot_id, start_frame)`` windows over each recording.

    For each shot the camera recording is tiled with windows of the configured
    frame span, advancing the start by ``window_stride`` frames (default
    ``max(1, span // 2)`` — 50 % overlap).  A shot too short for even one window
    is skipped; a shot long enough for one window contributes at least that one.

    The returned list is DETERMINISTIC (ascending shot id, then ascending start)
    so the train window set is reproducible across ranks and sessions.  Every
    emitted ``shot_id`` is one of the input ``shot_ids`` — the function never
    reaches outside the list, which is the structural shot-level leakage guard.

    Parameters
    ----------
    window_stride:
        Frames to advance the window start between consecutive windows.  Smaller
        => more overlap => more windows.  Defaults to half the window span.
    max_windows_per_shot:
        Optional cap on windows emitted per shot (the starts are then spread
        evenly across the recording), to bound a very long recording's share.
    """
    span = (config.n_frames - 1) * config.frame_stride + 1
    stride = int(window_stride) if window_stride else max(1, span // 2)
    if stride < 1:
        raise ValueError("window_stride must be >= 1")
    windows: list[tuple[int, int]] = []
    for sid in sorted(int(s) for s in shot_ids):
        try:
            n_total = camera_frame_count(sid, camera, token_root=token_root)
        except (FileNotFoundError, KeyError, ValueError):
            continue
        if n_total < span:
            continue
        last_start = n_total - span
        starts = list(range(0, last_start + 1, stride))
        if starts and starts[-1] != last_start:
            starts.append(last_start)  # always include the tail window
        if max_windows_per_shot is not None and len(starts) > int(max_windows_per_shot):
            # spread the cap evenly across the recording (keep first..last).
            sel = np.linspace(0, len(starts) - 1, int(max_windows_per_shot))
            starts = sorted({starts[int(round(i))] for i in sel})
        for st in starts:
            windows.append((sid, int(st)))
    return windows


class OverlappingSignalWindowDataset:
    """Map-style dataset over an explicit ``(shot_id, start_frame)`` window list.

    Unlike :class:`SignalSpacetimeDataset` (one window per shot per epoch), this
    is indexed by a PRECOMPUTED window list so a long recording contributes many
    overlapping windows — the signal-maximising pipeline.  The window list is the
    single source of truth for which (shot, frame-span) pairs train; because it
    is built only from the train-shot ids (see :func:`enumerate_windows`), the
    shot-level held-out guarantee is structural.

    Each ``__getitem__`` assembles the EXACT window at its list entry (no random
    jitter — the overlap already provides the augmentation).
    """

    def __init__(
        self,
        windows: Sequence[tuple[int, int]],
        config: SpacetimeWindowConfig,
        modalities: Sequence[SignalModalitySpec],
        n_signal_steps: int,
        *,
        camera: str = REFERENCE_CAMERA,
        token_root: Path | None = None,
    ) -> None:
        self._windows = [(int(s), int(f)) for s, f in windows]
        self._config = config
        self._modalities = list(modalities)
        self._n_signal_steps = int(n_signal_steps)
        self._camera = camera
        self._token_root = token_root

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, index: int) -> SignalSpacetimeSample:
        sid, start = self._windows[index]
        return assemble_signal_window(
            sid,
            self._config,
            self._modalities,
            self._n_signal_steps,
            camera=self._camera,
            token_root=self._token_root,
            start_frame=start,
        )

    @property
    def windows(self) -> list[tuple[int, int]]:
        return list(self._windows)

    @property
    def shot_ids(self) -> list[int]:
        """The distinct shot ids represented in the window list (sorted)."""
        return sorted({s for s, _ in self._windows})

    @property
    def config(self) -> SpacetimeWindowConfig:
        return self._config

    @property
    def modalities(self) -> list[SignalModalitySpec]:
        return self._modalities
