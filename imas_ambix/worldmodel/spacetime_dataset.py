"""Native-cadence camera-frame windows for the spatiotemporal video model.

This is the data substrate for the spatiotemporal camera transformer
(:mod:`imas_ambix.worldmodel.spacetime_model`).  It is DELIBERATELY separate
from :mod:`imas_ambix.worldmodel.dataset`: that module resamples every modality
onto a coarse common L2 grid (``n_steps`` ≈ 64), which throws away the camera's
native ~600 Hz temporal structure — exactly the structure a *video* model must
see.  Here we keep the camera on its OWN frame cadence.

What one sample is
------------------
A contiguous run of ``n_frames`` REAL recorded frames from one camera's
on-disk frame-token store, plus the shot's pulse-schedule plan as a short
conditioning prefix:

* Each frame is the full ``16 × 16 = 256`` LFQ token grid, raster-flattened
  (row-major ``for r for c``) — NO spatial subsample, every token kept, so
  spatial structure is fully present and the prediction decodes to a real
  256×256 image.
* Token ids are mapped from the on-disk STORE-ID space to LOCAL id space by
  subtracting :data:`REGISTRY_OFFSET` (= 4), so the model's vocabulary is
  exactly ``2**18`` and a local id indexes the embedding table directly.  The
  decode path adds the offset back (it expects store-ids), so
  :func:`local_to_store` is the single inverse used when handing a prediction
  to the MAGVIT2 decoder.
* Only frames that EXIST on disk are used.  The store holds only real recorded
  frames (the camera records over part of the shot); a window is taken from a
  contiguous slice of that array, so every frame in a sample is a genuine
  recording — token-0 "padding" is never fed as data.
* The plan prefix is the ``pulse_schedule_l2`` programmed-waveform tokens,
  rebased to their own local vocabulary, sub-sampled to ``n_plan`` steps that
  span the shot.  It conditions the dream ("this pulse schedule → this video").

Why native cadence + contiguous windows
----------------------------------------
A video model learns frame-to-frame dynamics; that signal lives at the camera's
own cadence.  Sampling a contiguous run (rather than nearest-neighbour onto a
coarse grid) preserves the true inter-frame evolution the temporal attention
must model.  Windows are drawn from the MIDDLE of each recording by default
(the established plasma, not the dark ramp-up / aborted tail).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from imas_ambix.camdyn.dataset import (
    DEFAULT_VOCAB_VERSION as FRAMES_VOCAB_VERSION,
)
from imas_ambix.camdyn.dataset import (
    FRAME_GRID,
    frames_token_path,
)
from imas_ambix.data.stream_encode import REGISTRY_OFFSET
from imas_ambix.worldmodel.dataset import (
    PAD_LOCAL_ID,
    _local_base_from_attrs,
    _signal_hf_store_path,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

#: The reference camera the demo predicts + decodes (its full-res frame is the
#: only one that round-trips to a real image through the frozen VQModel).
REFERENCE_CAMERA = "rbb"

#: Frame grid (rows, cols) and the flattened spatial token count per frame.
GRID_H, GRID_W = FRAME_GRID
N_SPATIAL = GRID_H * GRID_W  # 256

#: The LFQ codebook size = the model's local vocabulary (PAD reuses id 0, which
#: is also a real codebook id; the model NEVER masks camera tokens — every
#: frame in a window is a real recorded frame — so there is no PAD collision in
#: practice for the camera stream).
CAMERA_VOCAB = 1 << 18  # 262144

#: Plan (pulse_schedule) local vocab — the L2 block vocab + 1 for PAD, matching
#: ``default_modalities``' ``l2_vocab``.  Resolved lazily to avoid an import
#: cycle at module load.


def plan_vocab() -> int:
    """Local vocab for the pulse-schedule plan prefix (L2 block + PAD slot)."""
    from imas_ambix.tokenizer.registry import L2_BLOCK_VOCAB  # noqa: PLC0415

    return int(L2_BLOCK_VOCAB) + 1


def local_to_store(local_ids: np.ndarray) -> np.ndarray:
    """Map model LOCAL camera ids back to on-disk STORE-ids (decode expects).

    The inverse of the ``store_id - REGISTRY_OFFSET`` rebasing done at load.
    The MAGVIT2 decode path (``reconstruction_demo.decode_phase``) subtracts
    ``REGISTRY_OFFSET`` itself, so a prediction handed to it must be in STORE-id
    space — this is the single conversion used for both GT and prediction so a
    decoded prediction is never mislabelled.
    """
    return np.asarray(local_ids, dtype=np.int64) + REGISTRY_OFFSET


# ---------------------------------------------------------------------------
# Window configuration + one assembled sample
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpacetimeWindowConfig:
    """How a camera-frame window + plan prefix is assembled.

    Attributes
    ----------
    n_frames:
        Frames per sample (the temporal sequence length the model sees).
    n_plan:
        Plan-prefix steps prepended as conditioning (sub-sampled to span the
        shot's pulse schedule).
    context_frames:
        Leading frames given as context at eval/dream time (the rest are the
        forecast window).  Training is teacher-forced over the whole window;
        this only splits context vs forecast for the rollout + scoring.
    frame_stride:
        Take every ``frame_stride``-th frame from the contiguous run (1 = native
        cadence).  >1 widens the modelled time horizon at the cost of temporal
        resolution.  Used LITERALLY only when ``target_horizon_s <= 0``; otherwise
        a PER-SHOT stride is derived from the shot's cadence (see below).
    target_horizon_s:
        The PHYSICAL time span the ``n_frames`` should cover, in seconds.  When
        ``> 0`` (the default), :func:`assemble_window` IGNORES ``frame_stride`` and
        instead derives a per-shot stride
        ``max(1, round(target_horizon_s * fps / n_frames))`` from that shot's frame
        cadence, so every window spans ~``target_horizon_s`` regardless of the
        ~250x cadence spread across MAST shots.  This is essential for control
        learning: at MAST's 1-2 kHz native cadence, 24 CONSECUTIVE frames are only
        ~12-24 ms — far shorter than a ~125 ms ramp-up — so the model would see
        near-static clips and learn frame persistence, not actuator cause-effect.
        Set to 0 to use the literal ``frame_stride`` (legacy / debug).
    """

    n_frames: int = 24
    n_plan: int = 8
    context_frames: int = 8
    frame_stride: int = 1
    target_horizon_s: float = 0.25

    def __post_init__(self) -> None:
        if self.n_frames < 2:
            raise ValueError("n_frames must be >= 2 (need a next frame to predict)")
        if not (1 <= self.context_frames < self.n_frames):
            raise ValueError("context_frames must be in [1, n_frames)")
        if self.frame_stride < 1:
            raise ValueError("frame_stride must be >= 1")
        if self.n_plan < 0:
            raise ValueError("n_plan must be >= 0")


@dataclass
class SpacetimeSample:
    """One camera-frame window assembled for the spatiotemporal model.

    Attributes
    ----------
    shot_id, camera:
        Provenance.
    start_frame:
        Index of the first frame (in the on-disk store) of this window.
    frames:
        ``(n_frames, N_SPATIAL)`` int64 LOCAL camera token ids (store-id − 4),
        raster-flattened per frame.
    plan:
        ``(n_plan, n_plan_channels)`` int64 LOCAL plan token ids (the
        pulse-schedule conditioning prefix).  Empty ``(0, 0)`` when the shot has
        no readable plan.
    frame_time:
        ``(n_frames,)`` float64 per-frame timestamps (s) — for the demo strip.
    context_frames:
        Leading-context frame count (the rest are the forecast window).
    """

    shot_id: int
    camera: str
    start_frame: int
    frames: np.ndarray
    plan: np.ndarray
    frame_time: np.ndarray
    context_frames: int

    @property
    def n_frames(self) -> int:
        return int(self.frames.shape[0])

    def store_frames(self) -> np.ndarray:
        """The window's frames as ``(n_frames, 16, 16)`` STORE-ids (for decode)."""
        store = local_to_store(self.frames)
        return store.reshape(self.n_frames, GRID_H, GRID_W)


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------


def _camera_store(shot_id: int, camera: str, *, token_root: Path | None):
    """Open a camera frame-token store group (read-only), or raise."""
    import zarr  # noqa: PLC0415

    path = frames_token_path(
        shot_id, camera, FRAMES_VOCAB_VERSION, token_root=token_root
    )
    return zarr.open_group(str(path), mode="r")


def camera_frame_count(
    shot_id: int, camera: str, *, token_root: Path | None = None
) -> int:
    """Number of real recorded frames in a camera's on-disk token store."""
    grp = _camera_store(shot_id, camera, token_root=token_root)
    return int(grp["tokens"].shape[0])


def _read_plan(
    shot_id: int, n_plan: int, *, token_root: Path | None = None
) -> tuple[np.ndarray, int]:
    """Read the pulse-schedule plan, rebased to local ids, sub-sampled to n_plan.

    Returns ``(plan (n_plan, C) int64 local ids, n_channels)`` — the programmed
    waveform tokens at ``n_plan`` evenly-spaced positions spanning the shot's
    pulse schedule.  An empty ``(0, 0)`` array when the plan store is missing
    (the model then runs unconditioned for that shot — still trainable).
    """
    import zarr  # noqa: PLC0415

    if n_plan <= 0:
        return np.zeros((0, 0), dtype=np.int64), 0
    try:
        path = _signal_hf_store_path(
            shot_id, "pulse_schedule_l2", token_root=token_root
        )
        store = zarr.open_group(str(path), mode="r")
        tok = np.asarray(store["tokens"], dtype=np.int64)  # (T, C)
        base = _local_base_from_attrs(dict(store.attrs))
    except (FileNotFoundError, KeyError) as exc:
        logger.info(
            "shot %s: no pulse_schedule plan (%r) — running unconditioned", shot_id, exc
        )
        return np.zeros((0, 0), dtype=np.int64), 0
    if tok.ndim != 2 or tok.shape[0] == 0:
        return np.zeros((0, 0), dtype=np.int64), 0
    local = np.where(tok - base < 0, PAD_LOCAL_ID, tok - base)
    # evenly-spaced positions spanning the schedule (a coarse but complete view).
    t = local.shape[0]
    idx = np.linspace(0, t - 1, min(n_plan, t)).round().astype(int)
    sub = local[idx]
    if sub.shape[0] < n_plan:  # short schedule — repeat the last step to fill
        pad = np.repeat(sub[-1:], n_plan - sub.shape[0], axis=0)
        sub = np.concatenate([sub, pad], axis=0)
    return sub.astype(np.int64), int(sub.shape[1])


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def _fps_from_times(times: np.ndarray | None) -> float | None:
    """Native frame rate (Hz) from per-frame timestamps, or None.

    Uses the MEDIAN inter-frame dt (robust to dropped frames / non-uniform tails),
    so a 1-2 kHz MAST camera gives ~1000-2000.  Returns None when the timestamps
    are missing or degenerate (a constant / single-frame axis).
    """
    if times is None:
        return None
    t = np.asarray(times, dtype=np.float64)
    if t.size < 2:
        return None
    dt = np.diff(t)
    dt = dt[dt > 0]
    if dt.size == 0:
        return None
    med = float(np.median(dt))
    return (1.0 / med) if med > 0 else None


def effective_frame_stride(
    config: SpacetimeWindowConfig, fps: float | None
) -> int:
    """The per-shot frame stride so ``n_frames`` span ~``target_horizon_s``.

    ``max(1, round(target_horizon_s * fps / n_frames))`` when
    ``target_horizon_s > 0`` and ``fps`` is known; otherwise the literal
    ``config.frame_stride`` (legacy / when the cadence is unknown — a degenerate
    fallback, NOT the control-correct path).
    """
    if config.target_horizon_s and config.target_horizon_s > 0 and fps and fps > 0:
        stride = round(config.target_horizon_s * fps / config.n_frames)
        return max(1, int(stride))
    return int(config.frame_stride)


def effective_window_span(config: SpacetimeWindowConfig, fps: float | None) -> int:
    """Native-frame span the window occupies at the per-shot stride.

    ``(n_frames - 1) * effective_frame_stride(config, fps) + 1`` — the number of
    consecutive native frames a horizon-spanning window needs, so callers that
    pre-check recording length / pick a random start use the SAME geometry
    :func:`assemble_window` does.
    """
    return (config.n_frames - 1) * effective_frame_stride(config, fps) + 1


def window_span_for_shot(
    shot_id: int,
    config: SpacetimeWindowConfig,
    *,
    camera: str = REFERENCE_CAMERA,
    token_root: Path | None = None,
) -> int:
    """The per-shot native-frame window span (reads the shot's cadence).

    Convenience for callers (length filter, random-start, transient scan) that
    need the horizon-spanning span for a specific shot without assembling it.
    Falls back to the literal-stride span when the cadence is unreadable.
    """
    fps = _fps_from_times(_frame_times(shot_id, camera, token_root=token_root))
    return effective_window_span(config, fps)


def recording_time_span_s(
    shot_id: int,
    *,
    camera: str = REFERENCE_CAMERA,
    token_root: Path | None = None,
) -> float | None:
    """Total physical duration (s) of the camera recording, or None if unreadable.

    For the TIME-BASED window (the cadence-robust path), the only requirement on a
    shot is that its recording spans at least ``target_horizon_s`` — checked with
    this rather than a fixed native-frame count (which mis-estimates under a
    variable cadence).
    """
    t = _frame_times(shot_id, camera, token_root=token_root)
    if t is None or t.shape[0] < 2:
        return None
    return float(np.asarray(t, dtype=np.float64)[-1] - np.asarray(t)[0])


def _time_spanned_indices(
    times: np.ndarray,
    n_total: int,
    *,
    n_frames: int,
    horizon_s: float,
    start_frame: int | None,
) -> tuple[np.ndarray, float]:
    """``n_frames`` frame indices whose timestamps span ~``horizon_s`` from a start.

    Picks the frame nearest each target time ``t0 + k*(horizon_s/(n_frames-1))``
    for ``k=0..n_frames-1`` — robust to a NON-UNIFORM cadence (the index stride
    between picks shrinks where the camera is fast, grows where it is slow), so
    the window always covers ~``horizon_s`` of physical time.  ``start_frame``
    (the excited region's start) anchors ``t0``; if it would push the window's end
    past the last frame, the start is backed off so the full horizon still fits.
    Returns ``(indices (n_frames,) int, achieved_span_s)``.  Raises ``ValueError``
    when even the whole recording is shorter than ``horizon_s``.
    """
    t = np.asarray(times, dtype=np.float64)[:n_total]
    if t.size < n_frames:
        raise ValueError(f"only {t.size} frames, need {n_frames}")
    total_span = float(t[-1] - t[0])
    if total_span < horizon_s:
        raise ValueError(
            f"recording spans {total_span:.3f}s < horizon {horizon_s:.3f}s "
            f"({t.size} frames) — cannot form a horizon window"
        )
    s0 = 0 if start_frame is None else int(max(0, min(start_frame, n_total - 1)))
    t0 = float(t[s0])
    # back off the start so [t0, t0+horizon] fits within the recording.
    if t0 + horizon_s > float(t[-1]):
        t0 = float(t[-1]) - horizon_s
        s0 = int(np.searchsorted(t, t0, side="left"))
        s0 = max(0, min(s0, n_total - 1))
        t0 = float(t[s0])
    targets = t0 + np.linspace(0.0, horizon_s, n_frames)
    idx = np.searchsorted(t, targets, side="left")
    idx = np.clip(idx, 0, n_total - 1)
    # pick the closer neighbour (searchsorted gives the right side).
    left = np.clip(idx - 1, 0, n_total - 1)
    pick_left = np.abs(t[left] - targets) <= np.abs(t[idx] - targets)
    idx = np.where(pick_left, left, idx)
    # ensure strictly increasing (a tie / dense region can repeat an index).
    for i in range(1, idx.size):
        if idx[i] <= idx[i - 1]:
            idx[i] = min(idx[i - 1] + 1, n_total - 1)
    idx = np.clip(idx, 0, n_total - 1).astype(np.int64)
    achieved = float(t[idx[-1]] - t[idx[0]])
    return idx, achieved


def assemble_window(
    shot_id: int,
    config: SpacetimeWindowConfig,
    *,
    camera: str = REFERENCE_CAMERA,
    token_root: Path | None = None,
    start_frame: int | None = None,
) -> SpacetimeSample:
    """Assemble one frame window (spanning ~``target_horizon_s``) + plan prefix.

    The window is ``config.n_frames`` frames taken at a PER-SHOT stride derived
    from the shot's cadence so they span ~``config.target_horizon_s`` seconds
    (see :func:`effective_frame_stride`) — NOT ``config.n_frames`` consecutive
    native frames (which at MAST's 1-2 kHz is only ~12-24 ms, far shorter than a
    ramp-up, and trains frame persistence).  When ``target_horizon_s <= 0`` the
    literal ``config.frame_stride`` is used.  ``start_frame`` defaults to centring
    the run on the middle of the recording.  Token ids are rebased to LOCAL space
    (store-id − 4).  Raises ``ValueError`` when the recording is too short.
    """
    grp = _camera_store(shot_id, camera, token_root=token_root)
    grid = grp["tokens"]  # (T, 16, 16) store-ids
    n_total = int(grid.shape[0])
    times = _frame_times(shot_id, camera, token_root=token_root)

    use_time = (
        config.target_horizon_s
        and config.target_horizon_s > 0
        and times is not None
        and times.shape[0] >= n_total
    )
    if use_time:
        # TIME-BASED subsample: pick n_frames indices whose timestamps are nearest
        # to start_time + k*(horizon/(n_frames-1)).  This GUARANTEES the window
        # spans ~target_horizon_s even when the camera cadence CHANGES mid-shot
        # (MAST cameras accelerate/decelerate within a shot — a fixed per-shot
        # stride then under/over-spans; ~22% of curated windows undershot badly).
        idx, span = _time_spanned_indices(
            np.asarray(times, dtype=np.float64),
            n_total,
            n_frames=config.n_frames,
            horizon_s=float(config.target_horizon_s),
            start_frame=start_frame,
        )
        run = np.asarray(grid[: idx[-1] + 1], dtype=np.int64)[idx]  # (n_frames,16,16)
        start_frame = int(idx[0])
        ftime = np.asarray(times, dtype=np.float64)[idx]
    else:
        # fixed-stride fallback (target_horizon_s<=0, or no timestamps): the
        # literal/derived per-shot stride over a contiguous run.
        fps = _fps_from_times(times)
        stride = effective_frame_stride(config, fps)
        span = (config.n_frames - 1) * stride + 1
        if n_total < span:
            raise ValueError(
                f"shot {shot_id} camera {camera}: only {n_total} frames, need "
                f"{span} for n_frames={config.n_frames} stride={stride} "
                f"(fps={fps!r}, horizon={config.target_horizon_s}s)"
            )
        if start_frame is None:
            start_frame = max(0, (n_total - span) // 2)
        start_frame = int(min(start_frame, n_total - span))
        stop = start_frame + span
        run = np.asarray(grid[start_frame:stop], dtype=np.int64)[::stride]
        if times is not None and times.shape[0] >= stop:
            ftime = times[start_frame:stop][::stride].astype(np.float64)
        else:
            ftime = (np.arange(config.n_frames, dtype=np.float64) * stride) / 600.0

    # store-id -> local id; every frame here is a real recording (no padding).
    local = run - REGISTRY_OFFSET
    local = np.clip(local, 0, CAMERA_VOCAB - 1)
    frames = local.reshape(run.shape[0], N_SPATIAL).astype(np.int64)

    plan, _n_plan_ch = _read_plan(shot_id, config.n_plan, token_root=token_root)

    return SpacetimeSample(
        shot_id=int(shot_id),
        camera=camera,
        start_frame=int(start_frame),
        frames=frames,
        plan=plan,
        frame_time=ftime,
        context_frames=int(config.context_frames),
    )


def _frame_times(
    shot_id: int, camera: str, *, token_root: Path | None
) -> np.ndarray | None:
    """Per-frame timestamps from the camera level-1 store, or None."""
    from imas_ambix.camdyn.dataset import level1_shot_path  # noqa: PLC0415

    try:
        lpath = level1_shot_path(shot_id)
    except Exception:  # noqa: BLE001
        return None
    if lpath is None or not Path(lpath).exists():
        return None
    import zarr  # noqa: PLC0415

    try:
        store = zarr.open_group(str(lpath), mode="r")
        if camera not in set(store.group_keys()):
            return None
        grp = store[camera]
        if "time" not in set(grp.array_keys()):
            return None
        return np.asarray(grp["time"], dtype=np.float64)
    except Exception as exc:  # noqa: BLE001
        logger.debug("cannot read %s/%s/time: %r", lpath, camera, exc)
        return None


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def discover_camera_shots(
    *,
    camera: str = REFERENCE_CAMERA,
    token_root: Path | None = None,
    min_frames: int,
    limit: int | None = None,
    shot_ids: Sequence[int] | None = None,
) -> list[int]:
    """Shots whose ``camera`` recording has at least ``min_frames`` real frames.

    Scans the on-disk frame-token directory (or a given ``shot_ids`` list),
    keeping shots with a long-enough recording to yield a full window.  Returns
    a SORTED, DETERMINISTIC list (ascending shot id) so a train/eval split by
    index is reproducible across ranks and sessions.
    """
    from imas_ambix.data.paths import TOKEN_ROOT  # noqa: PLC0415

    root = Path(token_root) if token_root is not None else TOKEN_ROOT
    if shot_ids is not None:
        candidates = sorted(int(s) for s in shot_ids)
    else:
        frames_root = root / FRAMES_VOCAB_VERSION / "frames"
        if not frames_root.exists():
            return []
        candidates = sorted(
            int(p.name)
            for p in frames_root.iterdir()
            if p.is_dir() and p.name.isdigit()
        )
    kept: list[int] = []
    for sid in candidates:
        try:
            if camera_frame_count(sid, camera, token_root=token_root) >= min_frames:
                kept.append(sid)
        except (FileNotFoundError, KeyError, ValueError):
            continue
        if limit is not None and len(kept) >= limit:
            break
    return kept


# ---------------------------------------------------------------------------
# Torch Dataset
# ---------------------------------------------------------------------------


class SpacetimeFrameDataset:
    """Map-style torch Dataset of native-cadence camera-frame windows.

    Each item is a :class:`SpacetimeSample`.  Assembly is lazy in
    ``__getitem__`` (Zarr opened on demand) so the dataset pickles cheaply
    across DataLoader workers.  By default each access draws a window centred on
    the recording; pass ``random_window=True`` to draw a random valid start so a
    long recording yields varied windows across epochs (data augmentation for
    the corpus run).
    """

    def __init__(
        self,
        shot_ids: Sequence[int],
        config: SpacetimeWindowConfig,
        *,
        camera: str = REFERENCE_CAMERA,
        token_root: Path | None = None,
        random_window: bool = False,
        seed: int = 0,
    ) -> None:
        self._shot_ids = [int(s) for s in shot_ids]
        self._config = config
        self._camera = camera
        self._token_root = token_root
        self._random = bool(random_window)
        self._seed = int(seed)

    def __len__(self) -> int:
        return len(self._shot_ids)

    def __getitem__(self, index: int) -> SpacetimeSample:
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
            # per-(shot,index,epoch-less) deterministic-ish jitter; the sampler
            # reshuffles order across epochs, this jitters the window within a shot.
            rng = random.Random((self._seed * 1_000_003) ^ (sid * 31) ^ index)
            start = rng.randint(0, hi) if hi > 0 else 0
        return assemble_window(
            sid,
            self._config,
            camera=self._camera,
            token_root=self._token_root,
            start_frame=start,
        )

    @property
    def config(self) -> SpacetimeWindowConfig:
        return self._config

    @property
    def camera(self) -> str:
        return self._camera
