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
        resolution.
    """

    n_frames: int = 24
    n_plan: int = 8
    context_frames: int = 8
    frame_stride: int = 1

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


def assemble_window(
    shot_id: int,
    config: SpacetimeWindowConfig,
    *,
    camera: str = REFERENCE_CAMERA,
    token_root: Path | None = None,
    start_frame: int | None = None,
) -> SpacetimeSample:
    """Assemble one contiguous native-cadence frame window + plan prefix.

    The window is ``config.n_frames`` frames taken at ``config.frame_stride``
    from the camera's on-disk store; ``start_frame`` defaults to centring the
    run on the middle of the recording (the established plasma).  Token ids are
    rebased to LOCAL space (store-id − 4).  Raises ``ValueError`` when the
    recording is too short to yield a full window.
    """
    grp = _camera_store(shot_id, camera, token_root=token_root)
    grid = grp["tokens"]  # (T, 16, 16) store-ids
    n_total = int(grid.shape[0])
    span = (config.n_frames - 1) * config.frame_stride + 1
    if n_total < span:
        raise ValueError(
            f"shot {shot_id} camera {camera}: only {n_total} frames, need {span} "
            f"for n_frames={config.n_frames} stride={config.frame_stride}"
        )
    if start_frame is None:
        start_frame = max(0, (n_total - span) // 2)
    start_frame = int(min(start_frame, n_total - span))
    stop = start_frame + span
    run = np.asarray(grid[start_frame:stop], dtype=np.int64)  # (span,16,16)
    run = run[:: config.frame_stride]  # (n_frames,16,16)
    # store-id -> local id; every frame here is a real recording (no padding).
    local = run - REGISTRY_OFFSET
    local = np.clip(local, 0, CAMERA_VOCAB - 1)
    frames = local.reshape(run.shape[0], N_SPATIAL).astype(np.int64)

    # per-frame timestamps (best-effort; synthetic uniform fallback)
    times = _frame_times(shot_id, camera, token_root=token_root)
    if times is not None and times.shape[0] >= stop:
        ftime = times[start_frame:stop][:: config.frame_stride].astype(np.float64)
    else:
        ftime = (
            np.arange(frames.shape[0], dtype=np.float64) * config.frame_stride
        ) / 600.0

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
