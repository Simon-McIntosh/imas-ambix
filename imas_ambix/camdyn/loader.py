"""Bounded torch ``DataLoader`` over the locked D0 camdyn pieces.

Throughput layer for the camera-dynamics trainer.  The model is loaded
ONCE in the main process; the per-window CPU work — reading the V3 token
grid, holding the V2 level-1 conditioning to the frame times, and sampling
the clip mask — runs in PARALLEL worker processes and overlaps with GPU
compute.  No subprocess-per-item, no file-IPC daemon, no unbounded
prefetch threads (repo §2b): just the torch ``DataLoader`` worker pool
with a bounded ``prefetch_factor`` and ``persistent_workers``.

Why this exists
---------------
``train.py``'s original ``_window_iter`` read one window at a time
synchronously on the main thread.  For every window it re-opened the V3
token Zarr (``FrameTokenDataset.__getitem__``) AND re-opened the V2
level-1 Zarr for conditioning (``load_conditioning``) — ~48 Zarr opens
per training step on GPFS.  The configured ``num_workers`` was never
wired to a real ``DataLoader``, so the GPU sat idle (~6 s/step for a 19M
model on an H200).  This module fixes that without touching any locked D0
file: it COMPOSES :class:`~imas_ambix.camdyn.dataset.FrameTokenDataset`,
:func:`~imas_ambix.camdyn.masking.sample_clip_mask` and
:func:`~imas_ambix.camdyn.conditioning.load_conditioning`.

Per-worker caching
------------------
Many windows share a shot.  Each worker keeps a small per-shot LRU:

* the V3 token store's ``tokens`` array handle (one Zarr open per shot,
  not per window) — windows slice it directly;
* the per-shot RAW level-1 conditioning traces (time + scaled value per
  channel), read once and held-resampled to each window's frame times in
  pure numpy (no Zarr re-open per window).

The cache lives inside the worker process, so it is never shared across
processes (worker-safe) and is bounded (LRU eviction).

Output contract (matches what ``train.py`` consumes)
----------------------------------------------------
Each batch is a dict of stacked numpy arrays, identical in keys/shapes/
dtypes to the old ``_assemble_batch`` output so the trainer's forward,
loss, eval scoring, named-geometry override and motion-weighted subset
all work unchanged:

    tokens       (B, F, H, W) int64
    visible      (B, F, H, W) bool      (True = the model SEES the cell)
    loss_mask    (B, F, H, W) bool      (True = clipped-away = scored)
    cond_values  (B, F, C)    float32   (RAW physical units — NOT z-scored)
    cond_missing (B, F, C)    float32   (1.0 = fill / absent)
    dt           (B, F)       float32
    valid        (B, F)       bool
    frame_time   (B, F)       float64
    shot_id      (B,)         int64

Conditioning is emitted in RAW physical units; the trainer applies its
own ``_normalise_conditioning`` z-score with the precomputed per-channel
stats (kept in the main process) exactly as before.
"""

from __future__ import annotations

import contextlib
import logging
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from imas_ambix.camdyn.conditioning import (
    CONDITIONING_CHANNELS,
    assert_no_leakage_sources,
    resample_to_frames,
)
from imas_ambix.camdyn.dataset import (
    FRAME_GRID,
    FrameTokenDataset,
    FrameTokenShotSpec,
    FrameWindowConfig,
    _forward_dt,
)
from imas_ambix.camdyn.masking import ClipMaskConfig, MaskMode, sample_clip_mask

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = logging.getLogger(__name__)

BATCH_KEYS = (
    "tokens",
    "visible",
    "loss_mask",
    "cond_values",
    "cond_missing",
    "dt",
    "valid",
    "frame_time",
    "shot_id",
)


# ---------------------------------------------------------------------------
# Per-shot conditioning cache (worker-local) — read RAW traces once
# ---------------------------------------------------------------------------


def _read_shot_cond_traces(level1_path, channels=CONDITIONING_CHANNELS):
    """Read every conditioning channel's RAW (time, scaled-value) once.

    Returns a list aligned to ``channels``: each entry is ``(time, value)``
    float64 arrays (value already multiplied by ``chan.scale``) or
    ``None`` when the channel is absent for this shot.  Opening the V2
    level-1 store happens ONCE here (per shot, per worker) instead of once
    per window.
    """
    from imas_ambix.camdyn.conditioning import (  # noqa: PLC0415
        _open_level1,
        _read_source_signal,
    )

    n_chan = len(channels)
    out: list[tuple[np.ndarray, np.ndarray] | None] = [None] * n_chan
    if level1_path is None:
        return out
    store = _open_level1(level1_path)
    if store is None:
        return out
    for j, chan in enumerate(channels):
        st, sv = _read_source_signal(store, chan.source, chan.array)
        if st is None or sv is None or np.ndim(sv) != 1:
            continue
        out[j] = (
            np.asarray(st, dtype=np.float64),
            np.asarray(sv, dtype=np.float64) * chan.scale,
        )
    return out


def _hold_traces_to_frames(traces, frame_time, channels=CONDITIONING_CHANNELS):
    """Zero-order-hold cached RAW traces onto one window's frame times.

    Pure numpy — no Zarr re-open.  Produces the same ``(values, missing)``
    arrays as :func:`load_conditioning`, channel order preserved.
    """
    ft = np.asarray(frame_time, dtype=np.float64).reshape(-1)
    n = ft.shape[0]
    c = len(channels)
    values = np.zeros((n, c), dtype=np.float32)
    missing = np.ones((n, c), dtype=np.float32)
    for j in range(c):
        tr = traces[j]
        if tr is None:
            continue
        st, sv = tr
        held, miss = resample_to_frames(st, sv, ft)
        values[:, j] = held
        missing[:, j] = miss
    return values, missing


# ---------------------------------------------------------------------------
# Iterable dataset — composes the locked D0 pieces, shards across workers
# ---------------------------------------------------------------------------


@dataclass
class _Cached:
    tokens: object  # open zarr array handle (sliced per window)
    n_frames: int
    cond_traces: list
    frame_times: object  # (T,) float64 level-1 axis, or None (synthetic)


class CamdynWindowStream:
    """Iterable that yields fully-assembled per-window numpy dicts.

    One epoch = a seeded shuffle of all ``(spec, start)`` windows enumerated
    by :class:`FrameTokenDataset`.  When used inside a multi-worker torch
    ``DataLoader`` each worker takes a disjoint shard (round-robin on the
    flat window index) so the union over workers is the whole epoch with no
    duplication.  Iterating directly (workers == 0) yields the full epoch.

    The clip mask is sampled IN-WORKER so mask sampling overlaps with GPU
    compute too.  ``mode`` forces a single mask mode (eval); ``None`` draws
    the §4a mixture.  ``progress`` feeds the curriculum area anneal.

    This is a torch ``IterableDataset`` when torch is importable; the class
    body avoids importing torch at module load so the file stays usable in a
    torch-free environment (mirrors the rest of camdyn).
    """

    def __init__(
        self,
        specs: list[FrameTokenShotSpec],
        frame_cfg: FrameWindowConfig,
        mask_cfg: ClipMaskConfig,
        *,
        seed: int,
        mode: MaskMode | None = None,
        progress: float | None = None,
        max_windows: int | None = None,
        cache_size: int = 8,
        channels=CONDITIONING_CHANNELS,
    ) -> None:
        # Leakage ban is enforced once, up front (and again per-load inside
        # the locked loader path); fail fast before any worker spins up.
        assert_no_leakage_sources(c.source for c in channels)
        self._specs = specs
        self._frame_cfg = frame_cfg
        self._mask_cfg = mask_cfg
        self._seed = int(seed)
        self._mode = mode
        self._progress = progress
        self._max_windows = max_windows
        self._cache_size = int(cache_size)
        self._channels = tuple(channels)
        # Window enumeration is cheap (no Zarr opens) and identical to the
        # map-style dataset's, so we reuse it for a reproducible order.
        self._enum = FrameTokenDataset(specs, frame_cfg, as_dict=True)._windows

    # -- introspection -----------------------------------------------------

    def __len__(self) -> int:
        n = len(self._enum)
        return n if self._max_windows is None else min(n, self._max_windows)

    # -- per-shot cache (worker-local) + window assembly -------------------

    def _get_cached(self, cache: OrderedDict, spec_index: int) -> _Cached:
        """Open the token store + read RAW conditioning traces ONCE per shot.

        One Zarr open per shot (cached), not per window.  The per-shot frame
        time axis is read here too and stored on the cache entry, so the
        whole shot costs at most two Zarr opens regardless of how many
        windows it contributes.
        """
        spec = self._specs[spec_index]
        sid = int(spec.shot_id)
        c = cache.get(sid)
        if c is not None:
            cache.move_to_end(sid)
            return c
        import zarr  # noqa: PLC0415

        from imas_ambix.camdyn.dataset import _read_frame_times  # noqa: PLC0415

        store = zarr.open_group(str(spec.token_path), mode="r")
        tokens = store["tokens"]  # handle — sliced lazily per window
        traces = _read_shot_cond_traces(spec.level1_path, self._channels)
        times = _read_frame_times(spec.level1_path, spec.camera)
        c = _Cached(
            tokens=tokens,
            n_frames=int(tokens.shape[0]),
            cond_traces=traces,
            frame_times=times,
        )
        cache[sid] = c
        while len(cache) > self._cache_size:
            cache.popitem(last=False)  # LRU evict oldest
        return c

    def _assemble_one(self, cache, spec_index, start, rng) -> dict:
        sid = int(self._specs[spec_index].shot_id)
        c = self._get_cached(cache, spec_index)

        nf = self._frame_cfg.n_frames
        end = min(start + nf, c.n_frames)
        real = np.asarray(c.tokens[start:end], dtype=np.int64)
        n_real = int(real.shape[0])
        if n_real < nf:
            pad = np.zeros((nf - n_real, *FRAME_GRID), dtype=np.int64)
            real = np.concatenate([real, pad], axis=0)

        times = c.frame_times
        synthetic = False
        if times is not None and times.shape[0] >= start + n_real:
            ft = times[start : start + n_real].astype(np.float64)
        else:
            synthetic = True
            ft = (start + np.arange(n_real, dtype=np.float64)) * (
                self._frame_cfg.fallback_dt
            )
        if n_real < nf:
            base = ft[-1] if n_real > 0 else 0.0
            step = float(ft[-1] - ft[-2]) if n_real > 1 else self._frame_cfg.fallback_dt
            pad_t = base + step * np.arange(1, nf - n_real + 1, dtype=np.float64)
            frame_time = np.concatenate([ft, pad_t])
        else:
            frame_time = ft
        dt = _forward_dt(frame_time)
        valid = np.zeros(nf, dtype=bool)
        valid[:n_real] = True

        # conditioning held from cached RAW traces (no Zarr re-open)
        cv, cm = _hold_traces_to_frames(c.cond_traces, frame_time, self._channels)

        # clip mask sampled in-worker (overlaps GPU compute)
        m, _meta = sample_clip_mask(
            nf, self._mask_cfg, rng, mode=self._mode, progress=self._progress
        )
        visible = m
        loss_mask = ~m
        return {
            "tokens": real,
            "visible": visible,
            "loss_mask": loss_mask,
            "cond_values": cv,
            "cond_missing": cm,
            "dt": dt.astype(np.float32),
            "valid": valid,
            "frame_time": frame_time.astype(np.float64),
            "shot_id": np.int64(sid),
            "time_is_synthetic": synthetic,
        }

    # -- iteration ---------------------------------------------------------

    def __iter__(self) -> Iterator[dict]:
        # Worker sharding: split the flat window index round-robin so the
        # union over workers is the full epoch with no overlap.
        worker_id, num_workers = _worker_info()
        n = len(self._enum)
        rng = np.random.default_rng(self._seed)
        order = rng.permutation(n)
        if self._max_windows is not None:
            order = order[: self._max_windows]
        cache: OrderedDict = OrderedDict()
        # Per-window mask RNG seeded from the global seed + window index so
        # the mask stream is reproducible and independent of worker count.
        for pos, idx in enumerate(order):
            if num_workers > 1 and (pos % num_workers) != worker_id:
                continue
            spec_index, start = self._enum[int(idx)]
            wrng = np.random.default_rng(self._seed * 1_000_003 + int(idx))
            yield self._assemble_one(cache, spec_index, start, wrng)


def _worker_info() -> tuple[int, int]:
    """Return ``(worker_id, num_workers)`` (``(0, 1)`` outside a worker)."""
    try:
        import torch  # noqa: PLC0415

        info = torch.utils.data.get_worker_info()
    except Exception:  # pragma: no cover - torch-free env
        return 0, 1
    if info is None:
        return 0, 1
    return int(info.id), int(info.num_workers)


# ---------------------------------------------------------------------------
# Collate — stack worker dicts into the batch dict train.py expects
# ---------------------------------------------------------------------------


def collate_windows(items: list[dict]) -> dict:
    """Stack per-window dicts into the batch dict (numpy, not tensors).

    Tensor transfer + conditioning z-score stay in the main process
    (``train.py``) so the workers do pure CPU prep.
    """
    out = {
        "tokens": np.stack([it["tokens"] for it in items]).astype(np.int64),
        "visible": np.stack([it["visible"] for it in items]).astype(bool),
        "loss_mask": np.stack([it["loss_mask"] for it in items]).astype(bool),
        "cond_values": np.stack([it["cond_values"] for it in items]).astype(np.float32),
        "cond_missing": np.stack([it["cond_missing"] for it in items]).astype(
            np.float32
        ),
        "dt": np.stack([it["dt"] for it in items]).astype(np.float32),
        "valid": np.stack([it["valid"] for it in items]).astype(bool),
        "frame_time": np.stack([it["frame_time"] for it in items]).astype(np.float64),
        "shot_id": np.asarray([int(it["shot_id"]) for it in items], dtype=np.int64),
    }
    return out


# ---------------------------------------------------------------------------
# torch IterableDataset wrapper (module-level → picklable for forkserver)
# ---------------------------------------------------------------------------
#
# Python 3.14 defaults to the ``forkserver`` multiprocessing start method on
# POSIX, which pickles the worker's dataset.  A dataset class defined inside
# ``make_loader`` is NOT picklable (``Can't pickle local object``), so the
# IterableDataset wrapper MUST be a module-level class.  It is built only
# when torch is importable so the module stays usable in a torch-free env.

try:  # pragma: no cover - exercised whenever torch is installed
    from torch.utils.data import IterableDataset as _TorchIterableDataset

    class _StreamDataset(_TorchIterableDataset):
        """Picklable IterableDataset wrapping a :class:`CamdynWindowStream`.

        Holds the (picklable) stream config and rebuilds the stream in
        ``__iter__`` so torch's per-worker sharding (via
        :func:`_worker_info`) applies.  Module-level so ``forkserver`` can
        pickle it to the worker processes.
        """

        def __init__(self, stream: CamdynWindowStream) -> None:
            super().__init__()
            self._stream = stream

        def __iter__(self):
            return iter(self._stream)

        def __len__(self):
            return len(self._stream)

except Exception:  # pragma: no cover - torch-free env
    _StreamDataset = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------


def make_loader(
    specs: list[FrameTokenShotSpec],
    frame_cfg: FrameWindowConfig,
    mask_cfg: ClipMaskConfig,
    *,
    batch_size: int,
    num_workers: int,
    seed: int,
    mode: MaskMode | None = None,
    progress: float | None = None,
    max_windows: int | None = None,
    prefetch_factor: int = 4,
    cache_size: int = 8,
    channels=CONDITIONING_CHANNELS,
    persistent_workers: bool = True,
):
    """Build a bounded torch ``DataLoader`` over the window stream.

    Parameters
    ----------
    batch_size:
        Windows per batch.
    num_workers:
        Parallel worker processes doing CPU data prep.  ``0`` runs the
        stream in the main process (used by CPU smoke tests).
    prefetch_factor:
        Bounded look-ahead per worker (queue depth = num_workers *
        prefetch_factor).  Ignored when ``num_workers == 0``.
    max_windows:
        Cap the epoch length (eval reads a bounded number of windows).
    persistent_workers:
        Keep worker processes alive between epochs.  ``True`` (default) is
        right for the long training loop where the loader is iterated to
        exhaustion every epoch.  **Eval/val paths MUST pass ``False``** —
        they break out of the iterator early (after ``max_windows``) and
        build a NEW loader per call (val + once per named geometry).  An
        early-broken ``IterableDataset`` loader with ``persistent_workers=
        True`` leaves its worker processes alive; constructing many such
        loaders leaks/deadlocks the worker pool (the 2-hour eval hang at
        step=val_every, jobs 1216061/1216062).  With ``False`` the workers
        join when the iterator is exhausted or GC'd — see
        :func:`close_loader` for explicit teardown after an early break.
        Has no effect when ``num_workers == 0``.

    Returns the ``DataLoader``; iterate it for batch dicts (see
    :func:`collate_windows`).
    """
    import torch  # noqa: PLC0415
    from torch.utils.data import DataLoader

    stream = CamdynWindowStream(
        specs,
        frame_cfg,
        mask_cfg,
        seed=seed,
        mode=mode,
        progress=progress,
        max_windows=max_windows,
        cache_size=cache_size,
        channels=channels,
    )

    # Module-level IterableDataset wrapper (picklable for the forkserver
    # start method on Python 3.14) — torch's per-worker sharding then calls
    # our _worker_info()-based shard split in CamdynWindowStream.__iter__.
    ds = _StreamDataset(stream)
    kwargs: dict = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_windows,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
        kwargs["persistent_workers"] = bool(persistent_workers)
    return DataLoader(ds, **kwargs)


def close_loader(loader) -> None:
    """Tear down a ``DataLoader``'s worker processes / pin-memory thread.

    The eval/val path iterates a loader then breaks early (after
    ``eval_windows`` / ``val_windows``), so the underlying
    ``_MultiProcessingDataLoaderIter`` is not exhausted and its workers do
    not auto-join.  Calling this drains the held iterator (best-effort) so
    the worker processes and pin-memory thread are reaped before the next
    eval loader is built — preventing the worker-pool leak that hung
    ``evaluate_w1``.  Safe to call on a ``num_workers == 0`` loader (no-op)
    and idempotent.
    """
    it = getattr(loader, "_iterator", None)
    if it is None:
        return
    shutdown = getattr(it, "_shutdown_workers", None)
    if callable(shutdown):
        with contextlib.suppress(Exception):  # pragma: no cover - defensive
            shutdown()
    with contextlib.suppress(Exception):  # pragma: no cover - defensive
        loader._iterator = None
