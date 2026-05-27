"""Encode helpers for frame and signal tokenisation.

These pure-Python helpers (no Click) drive the per-shot encode/persist loop
and collect structured :class:`EncodeReport` objects.  The CLI in
:mod:`imas_ambix.data.cli` wires :func:`bulk_encode_signals` up to
``ambix data bulk-encode-signals``; frame corpus encoding runs through the
in-process streaming encoder in :mod:`imas_ambix.data.stream_encode`.

Usage example
-------------
::

    from imas_ambix.data.encoding import bulk_encode_signals
    from imas_ambix.tokenizer.signals import UniformQuantizer

    reports = bulk_encode_signals(
        shot_ids=[15085, 15086],
        group="magnetics",
        tokenizer_factory=UniformQuantizer,
    )
    for r in reports:
        if r.error:
            print(f"shot {r.shot_id} FAILED: {r.error}")
        else:
            print(f"shot {r.shot_id}  {r.n_tokens} tokens  {r.elapsed_s:.2f}s")
"""

from __future__ import annotations

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
