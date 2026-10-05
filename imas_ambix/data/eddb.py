"""On-demand cache for JT-60SA EDDB channels.

Only the channels a map binds, for the shots a run asks for, are landed, as
``{shot}.zarr/{category}/{dname}`` under :data:`~imas_ambix.data.paths.JT60SA_ROOT`,
with the channel's EDDB time base stored as a sibling array
(``{category}/{dname}_time``) so a cached channel carries its own time.  Each
array is written on first fetch and never rewritten: a channel already on disk
is served from disk, so a warm read touches no network, and an attempt to write
over an existing channel — readable as an array or not — is refused rather than
allowed to silently replace measured data.  The store is a plain Zarr group laid
out so the existing
:class:`~imas_ambix.data.transform_engine.ZarrTransformEngine` and
:class:`~imas_ambix.data.virtual_zarr.VirtualZarrView` read it with no new
engine and no cache-specific reader.

The shot token is normalised to its integer form for the directory name,
because the engine opens a pulse as ``f"{int(shot)}.zarr"``; a token such as
``051234`` therefore lands under ``51234.zarr`` and opens through the engine.

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

#: The channel's EDDB time vector is stored beside the data array under this
#: suffix, so the engine can read it as ``{category}/{dname}_time``.
TIME_SUFFIX = "_time"
TIME_UNIT = "s"


class EddbCacheError(RuntimeError):
    """Raised when a channel cannot be cached without overwriting data."""


def normalised_shot(shot: object) -> str:
    """Return the shot token in its integer form, the cache directory name.

    The engine opens a pulse as ``f"{int(shot)}.zarr"``, so the cache uses the
    same integer form; ``051234`` and ``51234`` name one shot, not two.
    """

    return str(int(shot))


def channel_path(
    cache_root: Path | str, shot: object, category: str, dname: str
) -> Path:
    """Return the data array path a channel occupies inside the cache."""

    return Path(cache_root) / f"{normalised_shot(shot)}.zarr" / category / dname


def time_array_name(dname: str) -> str:
    """Return the sibling array name holding a channel's EDDB time base."""

    return f"{dname}{TIME_SUFFIX}"


def channel_time_path(
    cache_root: Path | str, shot: object, category: str, dname: str
) -> Path:
    """Return the sibling array path holding a channel's EDDB time base."""

    return Path(cache_root) / f"{normalised_shot(shot)}.zarr" / category / (
        time_array_name(dname)
    )


def _channel_present(
    cache_root: Path | str, shot: object, category: str, dname: str
) -> bool:
    """The one predicate both the reader and the writer agree on.

    A node present at the channel path counts as cached even if it does not open
    as an array: the on-demand cache never rewrites, so a present-but-unreadable
    node is refused by the writer and treated as already-cached by the reader
    rather than refetched over.
    """

    return channel_path(cache_root, shot, category, dname).exists()


def is_cached(cache_root: Path | str, shot: object, category: str, dname: str) -> bool:
    """Return whether a channel already occupies its cache path."""

    return _channel_present(cache_root, shot, category, dname)


def write_channel(cache_root: Path | str, record: ChannelRecord) -> Path:
    """Write one raw channel array and its time base, refusing to overwrite.

    The data array is stored in raw units and raw sign, two-dimensional
    ``(channel, time)``, with the EDDB unit string, channel count and sequence
    number attached; the EDDB time vector is stored as the sibling array
    ``{dname}_time``.  A second write of the same channel raises rather than
    replacing measured data, whether or not the existing node opens as an array.
    """

    root = Path(cache_root)
    root.mkdir(parents=True, exist_ok=True)
    if _channel_present(root, record.shot, record.category, record.dname):
        raise EddbCacheError(
            f"channel {record.shot}/{record.category}/{record.dname} is already "
            "cached; the on-demand cache is written once and never rewritten"
        )

    values = np.ascontiguousarray(record.data, dtype="<f8")
    if values.ndim == 1:
        values = values.reshape(1, -1)
    times = np.ascontiguousarray(record.time, dtype="<f8").reshape(-1)

    store = zarr.open_group(root / f"{normalised_shot(record.shot)}.zarr", mode="a")
    category = store.require_group(record.category)
    array = category.create_array(record.dname, data=values, overwrite=False)
    array.attrs[UNIT_ATTR] = record.unit
    array.attrs[CHANNEL_COUNT_ATTR] = int(record.nch)
    array.attrs[SEQUENCE_NUMBER_ATTR] = int(record.seq)
    array.attrs["time_length"] = int(times.shape[-1])
    time_array = category.create_array(
        time_array_name(record.dname), data=times, overwrite=False
    )
    time_array.attrs[UNIT_ATTR] = TIME_UNIT
    time_array.attrs[SEQUENCE_NUMBER_ATTR] = int(record.seq)
    time_array.attrs["time_length"] = int(times.shape[-1])
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
    "TIME_SUFFIX",
    "TIME_UNIT",
    "UNIT_ATTR",
    "EddbCacheError",
    "channel_path",
    "channel_time_path",
    "fetch_channels",
    "is_cached",
    "normalised_shot",
    "time_array_name",
    "write_channel",
]
