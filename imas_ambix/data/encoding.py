"""Bulk-encode helpers for frame and signal tokenisation.

These pure-Python helpers (no Click) drive the per-shot encode/persist loop
and collect structured :class:`EncodeReport` objects.  The CLI in
:mod:`imas_ambix.data.cli` wires them up to ``ambix data bulk-encode-frames``
and ``ambix data bulk-encode-signals``.

Usage example
-------------
::

    from imas_ambix.data.encoding import bulk_encode_frames
    from imas_ambix.tokenizer.frames import PlaceholderFrameTokenizer

    reports = bulk_encode_frames(
        shot_ids=[15085, 15086],
        camera="rbb",
        tokenizer_factory=PlaceholderFrameTokenizer,
    )
    for r in reports:
        if r.error:
            print(f"shot {r.shot_id} FAILED: {r.error}")
        else:
            print(f"shot {r.shot_id}  {r.n_tokens} tokens  {r.elapsed_s:.2f}s")
"""

from __future__ import annotations

import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from imas_ambix.tokenizer.base import FrameTokenizer, SignalTokenizer

# Default precomputed-frame store root (rbb-256 fast path; R3b). When a shot's
# precomputed Zarr exists here it is fed straight to the daemon, skipping the
# L1 open + normalise + RGB + resize CPU work. Absent → legacy L1 path.
from imas_ambix.data.paths import MIRROR_ROOT  # noqa: E402

DEFAULT_PREPROCESSED_ROOT = MIRROR_ROOT / "preprocessed" / "rbb-256"


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EncodeReport:
    """Outcome record for one shot / modality encode call.

    Parameters
    ----------
    shot_id:
        Numeric shot identifier.
    modality:
        ``"frames"`` or ``"signals"``.
    group_or_camera:
        Camera name (frames) or signal group name (signals).
    tokenizer_name:
        Stable tokenizer identifier from :class:`~imas_ambix.tokenizer.base.Tokenizer`.
    n_tokens:
        Total number of tokens written (0 on error or skip).
    elapsed_s:
        Wall time in seconds (0.0 for skipped shots).
    output_path:
        Canonical Zarr path on disk.
    error:
        Exception message if the shot failed; ``None`` on success.
    """

    shot_id: int
    modality: str
    group_or_camera: str
    tokenizer_name: str
    n_tokens: int
    elapsed_s: float
    output_path: Path
    error: str | None = field(default=None)


# ---------------------------------------------------------------------------
# Single-shot helpers
# ---------------------------------------------------------------------------


def _precomputed_shot_path(preprocessed_root: Path | None, shot_id: int) -> Path | None:
    """Return the precomputed-frame Zarr for *shot_id*, or ``None`` if absent.

    Gates the R3b fast path: when ``preprocessed_root`` is set and the shot's
    ``<root>/<shot>.zarr`` exists, that store (``(T,256,256,3)`` uint8) is read
    directly. Returns ``None`` when fast-path is disabled or the shot has no
    precompute — callers then fall back to the legacy L1 path.
    """
    if preprocessed_root is None:
        return None
    candidate = preprocessed_root / f"{shot_id}.zarr"
    return candidate if candidate.exists() else None


def _load_frames_for_encode(
    shot_id: int,
    camera: str,
    *,
    preprocessed_root: Path | None,
    max_frames: int | None,
):
    """Load frames for one shot, preferring the precomputed fast path.

    Returns ``(frames, presized)`` where ``presized`` is ``True`` when the
    array came from a precomputed ``(T,H,W,3)`` uint8 store (already
    normalised + RGB + resized) and should be fed straight to the daemon.
    """
    import numpy as np

    pre_path = _precomputed_shot_path(preprocessed_root, shot_id)
    if pre_path is not None:
        import zarr

        arr = zarr.open_array(str(pre_path / "data"), mode="r")
        frames = np.asarray(arr[:])  # (T, S, S, 3) uint8
        if max_frames is not None:
            frames = frames[:max_frames]
        return frames, True

    import xarray as xr

    from imas_ambix.data.paths import LEVEL1_DIR

    shot_zarr = LEVEL1_DIR / f"{shot_id}.zarr"
    ds = xr.open_zarr(str(shot_zarr / camera))
    data_vars = list(ds.data_vars)
    if not data_vars:
        raise ValueError(f"no data variables in group '{camera}' of shot {shot_id}")
    frames = ds[data_vars[0]].values
    if max_frames is not None:
        frames = frames[:max_frames]
    return np.asarray(frames), False


def encode_one_shot_frames(
    shot_id: int,
    camera: str,
    tokenizer_factory: Callable[[], FrameTokenizer],
    *,
    vocab_version: str = "v1",
    max_frames: int | None = None,
    overwrite: bool = False,
    preprocessed_root: Path | None = None,
) -> EncodeReport:
    """Encode one shot's camera frames and persist tokens to Zarr.

    Parameters
    ----------
    shot_id:
        Numeric shot identifier.
    camera:
        Camera source name, e.g. ``"rbb"``.
    tokenizer_factory:
        Zero-argument callable that returns a fresh :class:`FrameTokenizer`.
        Called per-shot so instances are not shared across threads.
    vocab_version:
        Token vocabulary version directory (default ``"v1"``).
    max_frames:
        Truncate the frame array to this many frames before encoding.
    overwrite:
        If ``False`` (default) skip shots whose output path already exists.
    preprocessed_root:
        When set and ``<root>/<shot>.zarr`` exists, read that precomputed
        ``(T,256,256,3)`` uint8 store directly and skip the L1 open +
        normalise + resize (R3b fast path). ``None`` (default) always uses
        the legacy L1 path — keeping the running job's behaviour intact.

    Returns
    -------
    EncodeReport
        Contains ``error=None`` on success or the exception message on failure.
    """
    import numpy as np

    from imas_ambix.data.persist import frames_token_path, save_frame_tokens

    out_path = frames_token_path(shot_id, camera, vocab_version)

    if not overwrite and out_path.exists():
        tok = tokenizer_factory()
        return EncodeReport(
            shot_id=shot_id,
            modality="frames",
            group_or_camera=camera,
            tokenizer_name=tok.name,
            n_tokens=0,
            elapsed_s=0.0,
            output_path=out_path,
            error=None,
        )

    t0 = time.monotonic()
    try:
        frames, presized = _load_frames_for_encode(
            shot_id,
            camera,
            preprocessed_root=preprocessed_root,
            max_frames=max_frames,
        )

        tok = tokenizer_factory()
        encoded = _encode_frames(tok, frames, presized=presized)
        save_frame_tokens(
            shot_id=shot_id,
            camera=camera,
            encoded=encoded,
            vocab_version=vocab_version,
        )
        n_tokens = int(np.asarray(encoded.token_ids).size)
        return EncodeReport(
            shot_id=shot_id,
            modality="frames",
            group_or_camera=camera,
            tokenizer_name=tok.name,
            n_tokens=n_tokens,
            elapsed_s=time.monotonic() - t0,
            output_path=out_path,
            error=None,
        )
    except Exception as exc:  # noqa: BLE001
        return EncodeReport(
            shot_id=shot_id,
            modality="frames",
            group_or_camera=camera,
            tokenizer_name=_safe_tokenizer_name(tokenizer_factory),
            n_tokens=0,
            elapsed_s=time.monotonic() - t0,
            output_path=out_path,
            error=str(exc),
        )


def encode_one_shot_signals(
    shot_id: int,
    group: str,
    tokenizer_factory: Callable[[], SignalTokenizer],
    *,
    vocab_version: str = "v1",
    overwrite: bool = False,
) -> EncodeReport:
    """Encode one shot's signal group and persist tokens to Zarr.

    Parameters
    ----------
    shot_id:
        Numeric shot identifier.
    group:
        Signal group name, e.g. ``"magnetics"``.
    tokenizer_factory:
        Zero-argument callable that returns a fresh :class:`SignalTokenizer`.
    vocab_version:
        Token vocabulary version directory (default ``"v1"``).
    overwrite:
        If ``False`` (default) skip shots whose output path already exists.

    Returns
    -------
    EncodeReport
        Contains ``error=None`` on success or the exception message on failure.
    """
    import numpy as np
    import xarray as xr

    from imas_ambix.data.paths import LEVEL2_DIR
    from imas_ambix.data.persist import save_signal_tokens, signals_token_path

    out_path = signals_token_path(shot_id, group, vocab_version)

    if not overwrite and out_path.exists():
        tok = tokenizer_factory()
        return EncodeReport(
            shot_id=shot_id,
            modality="signals",
            group_or_camera=group,
            tokenizer_name=tok.name,
            n_tokens=0,
            elapsed_s=0.0,
            output_path=out_path,
            error=None,
        )

    t0 = time.monotonic()
    try:
        shot_zarr = LEVEL2_DIR / f"{shot_id}.zarr"
        ds = xr.open_zarr(str(shot_zarr / group))

        tok = tokenizer_factory()
        encoded = tok.encode(ds)
        save_signal_tokens(
            shot_id=shot_id,
            group=group,
            encoded=encoded,
            vocab_version=vocab_version,
        )
        n_tokens = int(np.asarray(encoded.token_ids).size)
        return EncodeReport(
            shot_id=shot_id,
            modality="signals",
            group_or_camera=group,
            tokenizer_name=tok.name,
            n_tokens=n_tokens,
            elapsed_s=time.monotonic() - t0,
            output_path=out_path,
            error=None,
        )
    except Exception as exc:  # noqa: BLE001
        return EncodeReport(
            shot_id=shot_id,
            modality="signals",
            group_or_camera=group,
            tokenizer_name=_safe_tokenizer_name(tokenizer_factory),
            n_tokens=0,
            elapsed_s=time.monotonic() - t0,
            output_path=out_path,
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# Bulk helpers
# ---------------------------------------------------------------------------


def bulk_encode_frames(
    shot_ids: list[int],
    camera: str,
    tokenizer_factory: Callable[[], FrameTokenizer],
    *,
    max_workers: int = 1,
    skip_existing: bool = True,
    max_frames_per_shot: int | None = None,
    vocab_version: str = "v1",
    prefetch: bool = True,
    prefetch_workers: int = 2,
    prefetch_queue_size: int = 4,
    preprocessed_root: Path | None = None,
) -> list[EncodeReport]:
    """Encode frames for multiple shots and return one :class:`EncodeReport` per shot.

    Parameters
    ----------
    shot_ids:
        List of shot IDs to process.
    camera:
        Camera source name.
    tokenizer_factory:
        Zero-argument callable returning a fresh :class:`FrameTokenizer`.
        A new instance is created per shot to avoid sharing state across
        threads (FrameTokenizer instances are not thread-safe).
    max_workers:
        Legacy thread-pool size for the non-prefetch path. Default 1
        (sequential). Ignored when ``prefetch`` is ``True``.
    skip_existing:
        Skip shots whose token file already exists (default ``True``).
    max_frames_per_shot:
        Truncate frames to this many before encoding (``None`` = all).
    vocab_version:
        Token vocabulary version directory (default ``"v1"``).
    prefetch:
        When ``True`` (default) a small producer pool stages the CPU/IO prep
        (open + normalise + RGB + write ``.npy``) for shot N+1 while the main
        thread issues the daemon GPU-encode of shot N — closing the per-shot
        GPU-idle gap. Set ``False`` to use the legacy serial / thread-pool
        path. The daemon is still called from a single thread; only prep is
        parallelised.
    prefetch_workers:
        Number of CPU-prep producer threads (default 2).
    prefetch_queue_size:
        Max prepped items buffered ahead of the consumer (default 4) — caps
        memory used by staged frame arrays.
    preprocessed_root:
        When set and ``<root>/<shot>.zarr`` exists, read that precomputed
        ``(T,256,256,3)`` uint8 store directly (R3b fast path). ``None``
        (default) always uses the legacy L1 path. The fast path is opt-in-safe
        per shot: any shot lacking a precompute falls back to L1.

    Returns
    -------
    list[EncodeReport]
        One report per shot ID, in input order.
    """

    # Build the tokenizer ONCE and share across shots. For tokenizers backed
    # by a persistent subprocess (Open-MAGVIT2 daemon mode), this is the
    # difference between loading the 250MB checkpoint once vs N_shots times.
    # Thread-safety: the persistent daemon serialises stdin writes via an
    # internal lock; placeholder tokenizers are stateless. Workers > 1 will
    # bottleneck on the lock — for true multi-GPU parallelism, run one driver
    # per GPU (e.g. sbatch --array).
    _shared_tok = tokenizer_factory()

    if prefetch:
        return _bulk_encode_frames_prefetch(
            shot_ids,
            camera,
            _shared_tok,
            skip_existing=skip_existing,
            max_frames_per_shot=max_frames_per_shot,
            vocab_version=vocab_version,
            prefetch_workers=prefetch_workers,
            prefetch_queue_size=prefetch_queue_size,
            preprocessed_root=preprocessed_root,
        )

    def _shared_factory() -> FrameTokenizer:
        return _shared_tok

    def _encode(sid: int) -> EncodeReport:
        return encode_one_shot_frames(
            sid,
            camera,
            _shared_factory,
            vocab_version=vocab_version,
            max_frames=max_frames_per_shot,
            overwrite=not skip_existing,
            preprocessed_root=preprocessed_root,
        )

    if max_workers <= 1:
        return [_encode(sid) for sid in shot_ids]

    reports: dict[int, EncodeReport] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_encode, sid): sid for sid in shot_ids}
        for fut in futures:
            sid = futures[fut]
            reports[sid] = fut.result()
    return [reports[sid] for sid in shot_ids]


# Sentinel pushed onto the prefetch queue when a producer finishes a shot.
@dataclass
class _PrepItem:
    """One unit of work flowing producer → consumer in the prefetch path."""

    shot_id: int
    # When skip/error short-circuits in the producer, `report` is set and the
    # consumer emits it directly (no daemon call). Otherwise `prepared`
    # carries either a tokenizer PreparedFrames (two-phase API available) or a
    # raw (frames, presized) tuple for tokenizers without the two-phase API.
    report: EncodeReport | None = None
    prepared: object = None
    frames: object = None
    presized: bool = False
    error: str | None = None


def _bulk_encode_frames_prefetch(
    shot_ids: list[int],
    camera: str,
    tok: FrameTokenizer,
    *,
    skip_existing: bool,
    max_frames_per_shot: int | None,
    vocab_version: str,
    prefetch_workers: int,
    prefetch_queue_size: int,
    preprocessed_root: Path | None,
) -> list[EncodeReport]:
    """Producer→consumer prefetch encode (R2).

    Producers (a bounded thread pool) run the CPU prep — load frames + stage a
    ``.npy`` (or fall back to raw frames for non-two-phase tokenizers) — and
    push results onto a bounded queue. The single consumer (this thread) pulls
    prepped items in submission order and issues the daemon encode + persist
    serially, so the GPU sees one stream while the next shot's CPU work
    overlaps. Skip-existing, per-shot error capture, EncodeReport shape and
    output paths all match the serial path.
    """
    import numpy as np

    from imas_ambix.data.persist import frames_token_path, save_frame_tokens

    has_two_phase = hasattr(tok, "prepare") and hasattr(tok, "encode_prepared")
    tok_name = getattr(tok, "name", "unknown")

    def _prepare_one(sid: int) -> _PrepItem:
        out_path = frames_token_path(sid, camera, vocab_version)
        if skip_existing and out_path.exists():
            return _PrepItem(
                shot_id=sid,
                report=EncodeReport(
                    shot_id=sid,
                    modality="frames",
                    group_or_camera=camera,
                    tokenizer_name=tok_name,
                    n_tokens=0,
                    elapsed_s=0.0,
                    output_path=out_path,
                    error=None,
                ),
            )
        try:
            frames, presized = _load_frames_for_encode(
                sid,
                camera,
                preprocessed_root=preprocessed_root,
                max_frames=max_frames_per_shot,
            )
            if has_two_phase:
                prepared = tok.prepare(frames, presized=presized)  # type: ignore[attr-defined]
                return _PrepItem(shot_id=sid, prepared=prepared)
            return _PrepItem(shot_id=sid, frames=frames, presized=presized)
        except Exception as exc:  # noqa: BLE001
            return _PrepItem(shot_id=sid, error=str(exc))

    def _consume_one(item: _PrepItem, t0: float) -> EncodeReport:
        out_path = frames_token_path(item.shot_id, camera, vocab_version)
        if item.report is not None:
            return item.report
        if item.error is not None:
            return EncodeReport(
                shot_id=item.shot_id,
                modality="frames",
                group_or_camera=camera,
                tokenizer_name=tok_name,
                n_tokens=0,
                elapsed_s=time.monotonic() - t0,
                output_path=out_path,
                error=item.error,
            )
        try:
            if item.prepared is not None:
                encoded = tok.encode_prepared(item.prepared)  # type: ignore[attr-defined]
            else:
                encoded = _encode_frames(tok, item.frames, presized=item.presized)
            save_frame_tokens(
                shot_id=item.shot_id,
                camera=camera,
                encoded=encoded,
                vocab_version=vocab_version,
            )
            n_tokens = int(np.asarray(encoded.token_ids).size)
            return EncodeReport(
                shot_id=item.shot_id,
                modality="frames",
                group_or_camera=camera,
                tokenizer_name=tok_name,
                n_tokens=n_tokens,
                elapsed_s=time.monotonic() - t0,
                output_path=out_path,
                error=None,
            )
        except Exception as exc:  # noqa: BLE001
            # Release any staged temp dir on failure.
            cleanup = getattr(item.prepared, "cleanup", None)
            if cleanup is not None:
                cleanup()
            return EncodeReport(
                shot_id=item.shot_id,
                modality="frames",
                group_or_camera=camera,
                tokenizer_name=tok_name,
                n_tokens=0,
                elapsed_s=time.monotonic() - t0,
                output_path=out_path,
                error=str(exc),
            )

    results: dict[int, EncodeReport] = {}
    work_q: queue.Queue[_PrepItem] = queue.Queue(maxsize=max(1, prefetch_queue_size))
    n_workers = max(1, prefetch_workers)
    # Bound how many shots are submitted ahead of the consumer so producers
    # block on the queue rather than racing the entire shotlist into memory.
    semaphore = threading.Semaphore(max(1, prefetch_queue_size) + n_workers)
    next_idx = 0
    next_idx_lock = threading.Lock()
    t_start = time.monotonic()

    def _producer() -> None:
        nonlocal next_idx
        while True:
            with next_idx_lock:
                i = next_idx
                if i >= len(shot_ids):
                    return
                next_idx += 1
            semaphore.acquire()
            work_q.put(_prepare_one(shot_ids[i]))

    pool = ThreadPoolExecutor(max_workers=n_workers)
    producer_futures = [pool.submit(_producer) for _ in range(n_workers)]
    try:
        for _ in range(len(shot_ids)):
            item = work_q.get()
            try:
                report = _consume_one(item, t_start)
            finally:
                semaphore.release()
            results[item.shot_id] = report
    finally:
        for fut in producer_futures:
            fut.result()  # surface any unexpected producer crash
        pool.shutdown(wait=True)

    return [results[sid] for sid in shot_ids]


def bulk_encode_signals(
    shot_ids: list[int],
    group: str,
    tokenizer_factory: Callable[[], SignalTokenizer],
    *,
    max_workers: int = 4,
    skip_existing: bool = True,
    vocab_version: str = "v1",
) -> list[EncodeReport]:
    """Encode signals for multiple shots and return one :class:`EncodeReport` per shot.

    Parameters
    ----------
    shot_ids:
        List of shot IDs to process.
    group:
        Signal group name, e.g. ``"magnetics"``.
    tokenizer_factory:
        Zero-argument callable returning a fresh :class:`SignalTokenizer`.
    max_workers:
        Thread-pool size (default 4).
    skip_existing:
        Skip shots whose token file already exists (default ``True``).
    vocab_version:
        Token vocabulary version directory (default ``"v1"``).

    Returns
    -------
    list[EncodeReport]
        One report per shot ID, in input order.
    """

    def _encode(sid: int) -> EncodeReport:
        return encode_one_shot_signals(
            sid,
            group,
            tokenizer_factory,
            vocab_version=vocab_version,
            overwrite=not skip_existing,
        )

    if max_workers <= 1:
        return [_encode(sid) for sid in shot_ids]

    reports: dict[int, EncodeReport] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_encode, sid): sid for sid in shot_ids}
        for fut in futures:
            sid = futures[fut]
            reports[sid] = fut.result()
    return [reports[sid] for sid in shot_ids]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _encode_frames(tok: FrameTokenizer, frames, *, presized: bool):
    """Encode *frames* with *tok*, honouring the precomputed fast path.

    Uses the two-phase ``prepare``/``encode_prepared`` API when the tokenizer
    exposes it (so the ``presized`` flag reaches the daemon-input staging).
    Tokenizers without the two-phase API (e.g. the placeholder) just call
    ``encode`` — for them ``presized`` input is already normalised RGB uint8,
    which the placeholder handles transparently.
    """
    if hasattr(tok, "prepare") and hasattr(tok, "encode_prepared"):
        prepared = tok.prepare(frames, presized=presized)  # type: ignore[attr-defined]
        return tok.encode_prepared(prepared)  # type: ignore[attr-defined]
    return tok.encode(frames)


def _safe_tokenizer_name(factory: Callable) -> str:
    """Best-effort tokenizer name without constructing the instance."""
    try:
        # Many tokenizers expose `name` as a class attribute (dataclass field)
        return str(factory.__self__.name)  # type: ignore[attr-defined]
    except AttributeError:
        pass
    try:
        return str(factory.__wrapped__.name)  # type: ignore[attr-defined]
    except AttributeError:
        pass
    try:
        # dataclass: inspect the default
        import dataclasses as _dc

        fields = _dc.fields(factory)  # type: ignore[arg-type]
        for f in fields:
            if f.name == "name":
                return str(f.default)
    except (TypeError, AttributeError):
        pass
    return "unknown"
