"""On-demand cache for JT-60SA EDDB channels.

Only the channels a map binds, for the shots a run asks for, are landed, as
``{shot}.zarr/{category}/{dname}`` under :data:`~imas_ambix.data.paths.JT60SA_ROOT`.
Each array is written on first fetch and never rewritten: a channel already on
disk is served from disk, so a warm read touches no network, and an attempt to
write over an existing channel is refused rather than allowed to silently
replace measured data.  The store is a plain Zarr group laid out so the
existing :class:`~imas_ambix.data.transform_engine.ZarrTransformEngine` and
:class:`~imas_ambix.data.virtual_zarr.VirtualZarrView` read it with no new
engine and no cache-specific reader.

This is a second transport beside the FAIR-MAST mirror, not an extension of
it: it shares the path owner and the Zarr layout the engine reads, and takes
none of the manifest or whole-shot download machinery.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import zarr

if TYPE_CHECKING:
    from collections.abc import Iterable

    from imas_ambix.data.eddb_remote import ChannelRecord, RemoteEddbExtractor

#: Attribute names carried on every cached channel array.  ``units`` is the
#: EDDB unit string verbatim, ``channel_count`` the number of channels in the
#: record and ``sequence_number`` its EDDB sequence number.
UNIT_ATTR = "units"
CHANNEL_COUNT_ATTR = "channel_count"
SEQUENCE_NUMBER_ATTR = "sequence_number"


class EddbCacheError(RuntimeError):
    """Raised when a channel cannot be cached without overwriting data."""


def channel_path(cache_root: Path | str, shot: str, category: str, dname: str) -> Path:
    """Return the array path a channel occupies inside the cache."""

    return Path(cache_root) / f"{shot}.zarr" / category / dname


def is_cached(cache_root: Path | str, shot: str, category: str, dname: str) -> bool:
    """Return whether a channel is already on disk and readable."""

    path = channel_path(cache_root, shot, category, dname)
    if not path.is_dir():
        return False
    try:
        zarr.open_array(path, mode="r")
    except (KeyError, ValueError, OSError):
        return False
    return True


def write_channel(cache_root: Path | str, record: ChannelRecord) -> Path:
    """Write one raw channel array and refuse to overwrite an existing one.

    The arrays are stored in raw units and raw sign, two-dimensional
    ``(channel, time)``, with the EDDB unit string, channel count and sequence
    number attached.  A second write of the same channel raises rather than
    replacing measured data.
    """

    root = Path(cache_root)
    root.mkdir(parents=True, exist_ok=True)
    store = zarr.open_group(root / f"{record.shot}.zarr", mode="a")
    values = np.ascontiguousarray(record.data, dtype="<f8")
    if values.ndim == 1:
        values = values.reshape(1, -1)

    category = store.require_group(record.category)
    if record.dname in category:
        raise EddbCacheError(
            f"channel {record.shot}/{record.category}/{record.dname} is already "
            "cached; the on-demand cache is written once and never rewritten"
        )
    array = category.create_array(record.dname, data=values, overwrite=False)
    array.attrs[UNIT_ATTR] = record.unit
    array.attrs[CHANNEL_COUNT_ATTR] = int(record.nch)
    array.attrs[SEQUENCE_NUMBER_ATTR] = int(record.seq)
    array.attrs["time_length"] = int(np.asarray(record.time).shape[-1])
    return channel_path(root, record.shot, record.category, record.dname)


def fetch_channels(
    extractor: RemoteEddbExtractor,
    cache_root: Path | str,
    requests: Iterable[object],
) -> list[ChannelRecord]:
    """Fetch the requested channels not already cached and write them.

    Requests already satisfied on disk make no transport call.  Every remaining
    request is read in a single remote process, so one batch is one ssh
    session, and each returned record is written with :func:`write_channel`.
    """

    ordered = list(requests)
    missing = [
        request
        for request in ordered
        if not is_cached(cache_root, request.shot, request.category, request.dname)
    ]
    if not missing:
        return []
    fetched = extractor.fetch_batch(missing)
    for record in fetched:
        write_channel(cache_root, record)
    return fetched


__all__ = [
    "CHANNEL_COUNT_ATTR",
    "SEQUENCE_NUMBER_ATTR",
    "UNIT_ATTR",
    "EddbCacheError",
    "channel_path",
    "fetch_channels",
    "is_cached",
    "write_channel",
]
