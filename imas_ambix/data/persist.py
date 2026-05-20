"""Token persistence layer for MAST shot token files.

Stores and retrieves per-shot tokenised arrays using the Zarr v3 format under
``/work/projects/imas_gpu/mast-tokens/{vocab_version}/``.

Layout (matches plans/tokenizers.md §5)::

    mast-tokens/
      v1/
        frames/
          {shot_id}/
            {camera}.zarr   # int32 token_ids + optional block_kind + attrs
        signals/
          {shot_id}/
            {group}.zarr    # int32 token_ids + optional block_kind + attrs
        streams/
          {shot_id}.zarr    # 1-D int32 tokens + uint8 block_kind

Functions
---------
frames_token_path / signals_token_path / shot_stream_path
    Compute canonical paths without touching the filesystem.
save_frame_tokens / save_signal_tokens
    Write ``EncodedFrames`` / ``EncodedSignals`` to Zarr.
    Accept an optional *block_kind* array and persist it as a sibling
    ``block_kind`` array inside the same Zarr group.
load_frame_tokens / load_signal_tokens
    Read Zarr back into the dataclass types.
    If a ``block_kind`` array is present it is attached to
    ``EncodedFrames.metadata["block_kind"]`` /
    ``EncodedSignals.metadata["block_kind"]``.
save_shot_stream / load_shot_stream
    Convenience helpers for the unified per-shot token stream written by
    :meth:`~imas_ambix.tokenizer.multimodal.ShotTokenizer.encode_shot_with_block_kind`.
list_persisted_shots
    Return a sorted list of shot IDs that have at least one token file.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

import numpy as np

from imas_ambix.data.paths import TOKEN_ROOT
from imas_ambix.tokenizer.base import EncodedFrames, EncodedSignals

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def frames_token_path(shot_id: int, camera: str, vocab_version: str = "v1") -> Path:
    """Return the canonical Zarr path for a frame-token file.

    The returned path is
    ``TOKEN_ROOT/{vocab_version}/frames/{shot_id}/{camera}.zarr``.
    The path may not yet exist on disk.
    """
    return TOKEN_ROOT / vocab_version / "frames" / str(shot_id) / f"{camera}.zarr"


def signals_token_path(shot_id: int, group: str, vocab_version: str = "v1") -> Path:
    """Return the canonical Zarr path for a signal-token file.

    The returned path is
    ``TOKEN_ROOT/{vocab_version}/signals/{shot_id}/{group}.zarr``.
    The path may not yet exist on disk.
    """
    return TOKEN_ROOT / vocab_version / "signals" / str(shot_id) / f"{group}.zarr"


def shot_stream_path(shot_id: int, vocab_version: str = "v1") -> Path:
    """Return the canonical Zarr path for a unified per-shot token stream.

    The returned path is
    ``TOKEN_ROOT/{vocab_version}/streams/{shot_id}.zarr``.
    The path may not yet exist on disk.
    """
    return TOKEN_ROOT / vocab_version / "streams" / f"{shot_id}.zarr"


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------


def _to_json_safe(obj: object) -> object:
    """Recursively convert an object to a JSON-serialisable form."""
    if isinstance(obj, dict):
        return {str(k): _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(v) for v in obj]
    # numpy scalar / number
    try:
        if isinstance(obj, np.generic):
            return obj.item()
    except (AttributeError, TypeError):
        pass
    return obj


def save_frame_tokens(
    shot_id: int,
    camera: str,
    encoded: EncodedFrames,
    block_kind: np.ndarray | None = None,
    vocab_version: str = "v1",
) -> Path:
    """Write *encoded* frame tokens to Zarr and return the path.

    The ``token_ids`` array is stored as dataset variable ``tokens``
    (int32). Metadata is stored in Zarr ``.attrs``. If *block_kind* is
    provided it is stored as a sibling ``block_kind`` array (uint8).

    Parameters
    ----------
    shot_id:
        Numeric shot identifier.
    camera:
        Camera name, e.g. ``"rbb"``.
    encoded:
        ``EncodedFrames`` produced by a frame tokenizer.
    block_kind:
        Optional 1-D ``uint8`` array of :class:`~imas_ambix.tokenizer.base.BlockKind`
        codes, same length as ``encoded.token_ids`` when flattened.
    vocab_version:
        Token vocabulary version directory (default ``"v1"``).

    Returns
    -------
    Path
        The path of the written ``.zarr`` file.
    """
    import json

    import zarr

    path = frames_token_path(shot_id, camera, vocab_version)
    path.parent.mkdir(parents=True, exist_ok=True)

    store = zarr.open_group(str(path), mode="w")
    store.create_array(
        "tokens",
        data=np.asarray(encoded.token_ids, dtype=np.int32),
    )
    if block_kind is not None:
        store.create_array(
            "block_kind",
            data=np.asarray(block_kind, dtype=np.uint8),
        )
    store.attrs.update(
        {
            "shot_id": shot_id,
            "camera": camera,
            "vocab_version": vocab_version,
            "tokenizer_name": encoded.tokenizer_name,
            "shape": list(encoded.shape),
            "metadata": json.dumps(_to_json_safe(encoded.metadata)),
        }
    )
    return path


def save_signal_tokens(
    shot_id: int,
    group: str,
    encoded: EncodedSignals,
    block_kind: np.ndarray | None = None,
    vocab_version: str = "v1",
) -> Path:
    """Write *encoded* signal tokens to Zarr and return the path.

    Stores ``token_ids`` as ``tokens`` (int32) and ``channel_names`` as
    a separate JSON attribute. If *block_kind* is provided it is stored as
    a sibling ``block_kind`` array (uint8).

    Parameters
    ----------
    shot_id:
        Numeric shot identifier.
    group:
        Signal group name, e.g. ``"magnetics"``.
    encoded:
        ``EncodedSignals`` produced by a signal tokenizer.
    block_kind:
        Optional 1-D ``uint8`` array of :class:`~imas_ambix.tokenizer.base.BlockKind`
        codes, same length as ``encoded.token_ids`` when flattened.
    vocab_version:
        Token vocabulary version directory (default ``"v1"``).

    Returns
    -------
    Path
        The path of the written ``.zarr`` file.
    """
    import json

    import zarr

    path = signals_token_path(shot_id, group, vocab_version)
    path.parent.mkdir(parents=True, exist_ok=True)

    store = zarr.open_group(str(path), mode="w")
    store.create_array(
        "tokens",
        data=np.asarray(encoded.token_ids, dtype=np.int32),
    )
    if block_kind is not None:
        store.create_array(
            "block_kind",
            data=np.asarray(block_kind, dtype=np.uint8),
        )
    store.attrs.update(
        {
            "shot_id": shot_id,
            "group": group,
            "vocab_version": vocab_version,
            "tokenizer_name": encoded.tokenizer_name,
            "channel_names": list(encoded.channel_names),
            "metadata": json.dumps(_to_json_safe(encoded.metadata)),
        }
    )
    return path


def load_frame_tokens(
    shot_id: int,
    camera: str,
    vocab_version: str = "v1",
) -> EncodedFrames:
    """Read a persisted frame-token Zarr back into an :class:`EncodedFrames`.

    If the Zarr group contains a ``block_kind`` array it is attached to
    ``EncodedFrames.metadata["block_kind"]`` as a ``uint8`` numpy array.

    Parameters
    ----------
    shot_id:
        Numeric shot identifier.
    camera:
        Camera name, e.g. ``"rbb"``.
    vocab_version:
        Token vocabulary version directory (default ``"v1"``).

    Returns
    -------
    EncodedFrames
        Reconstructed dataclass with ``token_ids`` loaded from disk.

    Raises
    ------
    FileNotFoundError
        If the Zarr path does not exist.
    """
    import json

    import zarr

    path = frames_token_path(shot_id, camera, vocab_version)
    if not path.exists():
        raise FileNotFoundError(f"frame token file not found: {path}")

    store = zarr.open_group(str(path), mode="r")
    token_ids = np.asarray(store["tokens"], dtype=np.int32)
    attrs = dict(store.attrs)

    raw_shape = attrs.get("shape", list(token_ids.shape))
    shape = tuple(int(x) for x in raw_shape)
    tokenizer_name = str(attrs.get("tokenizer_name", ""))
    metadata_str = attrs.get("metadata", "{}")
    metadata: dict[str, object] = json.loads(metadata_str)

    if "block_kind" in store:
        metadata["block_kind"] = np.asarray(store["block_kind"], dtype=np.uint8)

    return EncodedFrames(
        token_ids=token_ids,
        shape=shape,
        tokenizer_name=tokenizer_name,
        metadata=metadata,
    )


def load_signal_tokens(
    shot_id: int,
    group: str,
    vocab_version: str = "v1",
) -> EncodedSignals:
    """Read a persisted signal-token Zarr back into an :class:`EncodedSignals`.

    If the Zarr group contains a ``block_kind`` array it is attached to
    ``EncodedSignals.metadata["block_kind"]`` as a ``uint8`` numpy array.

    Parameters
    ----------
    shot_id:
        Numeric shot identifier.
    group:
        Signal group name, e.g. ``"magnetics"``.
    vocab_version:
        Token vocabulary version directory (default ``"v1"``).

    Returns
    -------
    EncodedSignals
        Reconstructed dataclass with ``token_ids`` and ``channel_names`` from disk.

    Raises
    ------
    FileNotFoundError
        If the Zarr path does not exist.
    """
    import json

    import zarr

    path = signals_token_path(shot_id, group, vocab_version)
    if not path.exists():
        raise FileNotFoundError(f"signal token file not found: {path}")

    store = zarr.open_group(str(path), mode="r")
    token_ids = np.asarray(store["tokens"], dtype=np.int32)
    attrs = dict(store.attrs)

    channel_names = tuple(str(c) for c in attrs.get("channel_names", []))
    tokenizer_name = str(attrs.get("tokenizer_name", ""))
    metadata_str = attrs.get("metadata", "{}")
    metadata: dict[str, object] = json.loads(metadata_str)

    if "block_kind" in store:
        metadata["block_kind"] = np.asarray(store["block_kind"], dtype=np.uint8)

    return EncodedSignals(
        token_ids=token_ids,
        channel_names=channel_names,
        tokenizer_name=tokenizer_name,
        metadata=metadata,
    )


def save_shot_stream(
    shot_id: int,
    tokens: np.ndarray,
    block_kind: np.ndarray,
    vocab_version: str = "v1",
) -> Path:
    """Write the unified per-shot token stream to Zarr and return the path.

    This convenience function persists the output of
    :meth:`~imas_ambix.tokenizer.multimodal.ShotTokenizer.encode_shot_with_block_kind`
    to ``TOKEN_ROOT/{vocab_version}/streams/{shot_id}.zarr``.

    The Zarr group contains two arrays:

    ``tokens``
        1-D ``int32`` global token ids.
    ``block_kind``
        1-D ``uint8`` :class:`~imas_ambix.tokenizer.base.BlockKind` codes,
        same length as ``tokens``.

    Parameters
    ----------
    shot_id:
        Numeric shot identifier.
    tokens:
        1-D ``int32`` token array from ``encode_shot_with_block_kind``.
    block_kind:
        1-D ``uint8`` block-kind array from ``encode_shot_with_block_kind``.
    vocab_version:
        Token vocabulary version directory (default ``"v1"``).

    Returns
    -------
    Path
        The path of the written ``.zarr`` file.
    """
    import zarr

    path = shot_stream_path(shot_id, vocab_version)
    path.parent.mkdir(parents=True, exist_ok=True)

    store = zarr.open_group(str(path), mode="w")
    store.create_array("tokens", data=np.asarray(tokens, dtype=np.int32))
    store.create_array("block_kind", data=np.asarray(block_kind, dtype=np.uint8))
    store.attrs.update(
        {
            "shot_id": shot_id,
            "vocab_version": vocab_version,
            "n_tokens": int(np.asarray(tokens).shape[0]),
        }
    )
    return path


def load_shot_stream(
    shot_id: int,
    vocab_version: str = "v1",
) -> tuple[np.ndarray, np.ndarray]:
    """Read a unified per-shot token stream back from Zarr.

    Parameters
    ----------
    shot_id:
        Numeric shot identifier.
    vocab_version:
        Token vocabulary version directory (default ``"v1"``).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(tokens, block_kind)`` — both 1-D.  ``tokens`` is ``int32``;
        ``block_kind`` is ``uint8``.

    Raises
    ------
    FileNotFoundError
        If the Zarr path does not exist.
    """
    import zarr

    path = shot_stream_path(shot_id, vocab_version)
    if not path.exists():
        raise FileNotFoundError(f"shot stream file not found: {path}")

    store = zarr.open_group(str(path), mode="r")
    tokens = np.asarray(store["tokens"], dtype=np.int32)
    block_kind = np.asarray(store["block_kind"], dtype=np.uint8)
    return tokens, block_kind


def list_persisted_shots(
    modality: str = "frames",
    vocab_version: str = "v1",
) -> list[int]:
    """Return sorted shot IDs that have at least one persisted token file.

    Parameters
    ----------
    modality:
        ``"frames"``, ``"signals"``, or ``"streams"``.
    vocab_version:
        Token vocabulary version directory (default ``"v1"``).

    Returns
    -------
    list[int]
        Sorted list of integer shot IDs found on disk.
    """
    base = TOKEN_ROOT / vocab_version / modality
    if not base.exists():
        return []
    shot_ids: list[int] = []
    for child in base.iterdir():
        if modality == "streams":
            # streams layout: {shot_id}.zarr directly under base
            is_zarr = child.suffix == ".zarr" or (
                child.is_dir() and child.suffix == ".zarr"
            )
            if not is_zarr:
                continue
            stem = child.stem
            try:
                shot_ids.append(int(stem))
            except ValueError:
                continue
        else:
            if not child.is_dir():
                continue
            # Check at least one .zarr file exists in the shot directory
            has_tokens = any(
                p.suffix == ".zarr" or (p.is_dir() and p.suffix == ".zarr")
                for p in child.iterdir()
            )
            if not has_tokens:
                continue
            try:
                shot_ids.append(int(child.name))
            except ValueError:
                continue
    return sorted(shot_ids)
