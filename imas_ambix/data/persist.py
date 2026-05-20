"""Token persistence layer for MAST shot token files.

Stores and retrieves per-shot tokenised arrays using the Zarr v3 format under
``/work/projects/imas_gpu/mast-tokens/{vocab_version}/``.

Layout (matches plans/tokenizers.md §5)::

    mast-tokens/
      v1/
        frames/
          {shot_id}/
            {camera}.zarr   # 1-D or multi-D int32 token_ids + metadata attrs
        signals/
          {shot_id}/
            {group}.zarr    # 1-D or 2-D int32 token_ids + metadata attrs

Functions
---------
frames_token_path / signals_token_path
    Compute canonical paths without touching the filesystem.
save_frame_tokens / save_signal_tokens
    Write ``EncodedFrames`` / ``EncodedSignals`` to Zarr.
load_frame_tokens / load_signal_tokens
    Read Zarr back into the dataclass types.
list_persisted_shots
    Return a sorted list of shot IDs that have at least one token file.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

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
        import numpy as np  # noqa: PLC0415

        if isinstance(obj, np.generic):
            return obj.item()
    except ImportError:
        pass
    return obj


def save_frame_tokens(
    shot_id: int,
    camera: str,
    encoded: EncodedFrames,
    vocab_version: str = "v1",
) -> Path:
    """Write *encoded* frame tokens to Zarr and return the path.

    The ``token_ids`` array is stored as dataset variable ``tokens``
    (int32). Metadata is stored in Zarr ``.attrs``.

    Parameters
    ----------
    shot_id:
        Numeric shot identifier.
    camera:
        Camera name, e.g. ``"rbb"``.
    encoded:
        ``EncodedFrames`` produced by a frame tokenizer.
    vocab_version:
        Token vocabulary version directory (default ``"v1"``).

    Returns
    -------
    Path
        The path of the written ``.zarr`` file.
    """
    import json

    import numpy as np
    import zarr

    path = frames_token_path(shot_id, camera, vocab_version)
    path.parent.mkdir(parents=True, exist_ok=True)

    store = zarr.open_group(str(path), mode="w")
    store.create_array(
        "tokens",
        data=np.asarray(encoded.token_ids, dtype=np.int32),
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
    vocab_version: str = "v1",
) -> Path:
    """Write *encoded* signal tokens to Zarr and return the path.

    Stores ``token_ids`` as ``tokens`` (int32) and ``channel_names`` as
    a separate JSON attribute.

    Parameters
    ----------
    shot_id:
        Numeric shot identifier.
    group:
        Signal group name, e.g. ``"magnetics"``.
    encoded:
        ``EncodedSignals`` produced by a signal tokenizer.
    vocab_version:
        Token vocabulary version directory (default ``"v1"``).

    Returns
    -------
    Path
        The path of the written ``.zarr`` file.
    """
    import json

    import numpy as np
    import zarr

    path = signals_token_path(shot_id, group, vocab_version)
    path.parent.mkdir(parents=True, exist_ok=True)

    store = zarr.open_group(str(path), mode="w")
    store.create_array(
        "tokens",
        data=np.asarray(encoded.token_ids, dtype=np.int32),
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

    import numpy as np
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

    import numpy as np
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

    return EncodedSignals(
        token_ids=token_ids,
        channel_names=channel_names,
        tokenizer_name=tokenizer_name,
        metadata=metadata,
    )


def list_persisted_shots(
    modality: str = "frames",
    vocab_version: str = "v1",
) -> list[int]:
    """Return sorted shot IDs that have at least one persisted token file.

    Parameters
    ----------
    modality:
        ``"frames"`` or ``"signals"``.
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
