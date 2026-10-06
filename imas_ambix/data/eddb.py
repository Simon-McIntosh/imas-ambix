"""On-demand cache for JT-60SA EDDB channels.

Only the channels a map binds, for the shots a run asks for, are landed, as
``{shot}.zarr/{category}/{dname}`` under :data:`~imas_ambix.data.paths.JT60SA_ROOT`,
with the channel's EDDB time base stored as a sibling array
(``{category}/{dname}_time``) so a cached channel carries its own time.

EDDB names a pulse by a series letter and six digits (``E101173``, ``C``-series
for commissioning), while the catalogue's shot ranges and both engines address a
shot as an integer.  The cache reconciles the two: it stores the pulse under the
integer of the token's digits, so ``ZarrTransformEngine`` opens it as
``f"{int(shot)}.zarr"``, and records the full EDDB token as an attribute on the
pulse group so a second token sharing those digits cannot be mixed in.  The
remote request keeps the full token, because EDDB is addressed by it.

A channel is cached only when both its data array and its time array are present
and the time length matches the data width.  A channel already cached is served
from disk, so a warm read touches no network, and a write over one is refused
rather than allowed to silently replace measured data.  A half-written channel —
data present, time missing or length-mismatched, as a failure between the two
writes would leave it — is completed by the next fetch rather than refused.  The
store is a plain Zarr group laid out so the existing
:class:`~imas_ambix.data.transform_engine.ZarrTransformEngine` and
:class:`~imas_ambix.data.virtual_zarr.VirtualZarrView` read it with no new
engine; :func:`read_channel` is the raw read beside that engine path, returning
one cached channel's data, time base and unit whole from the arrays
:func:`write_channel` laid down.

This is a second transport beside the FAIR-MAST mirror, not an extension of
it: it shares the path owner and the Zarr layout the engine reads, and takes
none of the manifest or whole-shot download machinery.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import zarr

from imas_ambix.data.eddb_remote import BatchResult, ChannelRecord

if TYPE_CHECKING:
    from collections.abc import Iterable

    from imas_ambix.data.eddb_remote import RemoteEddbExtractor

#: Attribute names carried on the cached arrays.  ``units`` is the EDDB unit
#: string verbatim, ``channel_count`` the number of channels in the record and
#: ``sequence_number`` its EDDB sequence number.
UNIT_ATTR = "units"
CHANNEL_COUNT_ATTR = "channel_count"
SEQUENCE_NUMBER_ATTR = "sequence_number"

#: The full EDDB token is recorded on the pulse group so two series tokens that
#: share the same digits cannot be written into one integer-named directory.
EDDB_TOKEN_ATTR = "eddb_token"

#: The channel's EDDB time vector is stored beside the data array under this
#: suffix, so the engine can read it as ``{category}/{dname}_time``.
TIME_SUFFIX = "_time"
TIME_UNIT = "s"

#: An EDDB shot token: an optional series letter followed by digits, or a bare
#: integer.  ``E101173`` and ``C510000`` are the two series; ``51234`` is bare.
_TOKEN_RE = re.compile(r"^(?P<series>[A-Za-z])?(?P<digits>\d+)$")


class EddbCacheError(RuntimeError):
    """Raised when a channel cannot be cached without overwriting data."""


def _split_token(shot: object) -> tuple[str | None, str]:
    """Return the optional series letter and the digits of an EDDB shot token."""

    text = str(shot).strip()
    match = _TOKEN_RE.match(text)
    if match is None:
        raise EddbCacheError(
            f"{shot!r} is not an EDDB shot token (a series letter and digits, or "
            "a bare integer)"
        )
    return match.group("series"), match.group("digits")


def normalised_shot(shot: object) -> str:
    """Return the integer of the token's digits, the cache directory name.

    The engine opens a pulse as ``f"{int(shot)}.zarr"``, so ``E101173`` and a
    bare ``101173`` name one directory, ``101173.zarr``.
    """

    _, digits = _split_token(shot)
    return str(int(digits))


def eddb_token(shot: object) -> str:
    """Return the canonical full EDDB token: series letter (upper) and digits."""

    series, digits = _split_token(shot)
    return (series.upper() if series else "") + digits


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


def read_eddb_token(cache_root: Path | str, shot: object) -> str | None:
    """Return the EDDB token recorded on a pulse group, or ``None`` if absent."""

    path = Path(cache_root) / f"{normalised_shot(shot)}.zarr"
    if not path.is_dir():
        return None
    try:
        group = zarr.open_group(path, mode="r")
    except (KeyError, ValueError, OSError):
        return None
    token = group.attrs.get(EDDB_TOKEN_ATTR)
    return None if token is None else str(token)


def _node_length(path: Path) -> int | None:
    """Return the last-axis length of an array at ``path``, else ``None``.

    ``None`` covers both a path that is absent and one that exists but does not
    open as an array; the caller that must tell those apart uses
    :func:`_node_state`.
    """

    try:
        array = zarr.open_array(path, mode="r")
    except (KeyError, ValueError, OSError):
        return None
    shape = array.shape
    return int(shape[-1]) if shape else 0


def _node_state(path: Path) -> tuple[bool, int | None]:
    """Return ``(exists, last-axis length)``; length is ``None`` when unreadable."""

    if not path.exists():
        return False, None
    return True, _node_length(path)


def _readable_length(path: Path, label: str) -> int | None:
    """Return the last-axis length of ``path``, or ``None`` when it is absent.

    A node that exists but does not open as an array is a corrupt cache entry,
    not an absent one, and is refused rather than reported as not-cached.
    """

    exists, length = _node_state(path)
    if exists and length is None:
        raise EddbCacheError(f"existing {label} node {path} is not a readable array")
    return length


def _channel_present(
    cache_root: Path | str, shot: object, category: str, dname: str
) -> bool:
    """The one predicate both the reader and the writer agree on.

    A channel is present only when its data array and its time array both exist
    and the time length matches the data width.  A channel with data but no time
    (or a mismatched time) is half-written, not cached.  A node that exists but
    does not open as an array is corrupt and is refused.
    """
    # fmt: off
    data_length = _readable_length(
        channel_path(cache_root, shot, category, dname), "channel"
    )
    time_length = _readable_length(
        channel_time_path(cache_root, shot, category, dname), "time"
    )
    # fmt: on
    if data_length is None or time_length is None:
        return False
    return time_length == data_length


def is_cached(cache_root: Path | str, shot: object, category: str, dname: str) -> bool:
    """Return whether a channel is completely cached."""

    return _channel_present(cache_root, shot, category, dname)


def write_channel(cache_root: Path | str, record: ChannelRecord) -> Path:
    """Write one raw channel and its time base, refusing to overwrite a cached one.

    The data array is stored in raw units and raw sign, two-dimensional
    ``(channel, time)``, with the EDDB unit string, channel count and sequence
    number attached; the EDDB time vector is written as the sibling array
    ``{dname}_time``.  A pulse group records the full EDDB token and refuses a
    second token that shares its digits.  A channel already cached is refused; a
    half-written channel — data without a matching time array — is completed.
    """

    root = Path(cache_root)
    root.mkdir(parents=True, exist_ok=True)
    token = eddb_token(record.shot)
    store = zarr.open_group(root / f"{normalised_shot(record.shot)}.zarr", mode="a")

    recorded = store.attrs.get(EDDB_TOKEN_ATTR)
    if recorded is not None and str(recorded) != token:
        raise EddbCacheError(
            f"pulse {normalised_shot(record.shot)} already holds EDDB token "
            f"{recorded!r}; refusing to write {token!r} into it"
        )
    if _channel_present(root, record.shot, record.category, record.dname):
        raise EddbCacheError(
            f"channel {token}/{record.category}/{record.dname} is already "
            "cached; the on-demand cache is written once and never rewritten"
        )

    values = np.ascontiguousarray(record.data, dtype="<f8")
    if values.ndim == 1:
        values = values.reshape(1, -1)
    times = np.ascontiguousarray(record.time, dtype="<f8").reshape(-1)
    if times.shape[-1] != values.shape[-1]:
        raise EddbCacheError(
            f"time base length {times.shape[-1]} does not match data width "
            f"{values.shape[-1]} for {token}/{record.category}/{record.dname}"
        )

    data_path = channel_path(root, record.shot, record.category, record.dname)
    time_path = channel_time_path(root, record.shot, record.category, record.dname)
    # Past _channel_present any existing node is a readable array, so these
    # lengths describe a half-written channel rather than a corrupt one.
    data_exists, data_length = _node_state(data_path)
    time_exists, time_length = _node_state(time_path)
    if data_exists and data_length != times.shape[-1]:
        raise EddbCacheError(
            f"stored data width {data_length} does not match the fetched time "
            f"base {times.shape[-1]} for {token}/{record.category}/{record.dname}"
        )

    store.attrs[EDDB_TOKEN_ATTR] = token
    category = store.require_group(record.category)
    if not data_exists:
        array = category.create_array(record.dname, data=values, overwrite=False)
        array.attrs[UNIT_ATTR] = record.unit
        array.attrs[CHANNEL_COUNT_ATTR] = int(record.nch)
        array.attrs[SEQUENCE_NUMBER_ATTR] = int(record.seq)
        array.attrs["time_length"] = int(times.shape[-1])

    if not (time_exists and time_length == times.shape[-1]):
        time_array = category.create_array(
            time_array_name(record.dname), data=times, overwrite=True
        )
        time_array.attrs[UNIT_ATTR] = TIME_UNIT
        time_array.attrs[SEQUENCE_NUMBER_ATTR] = int(record.seq)
        time_array.attrs["time_length"] = int(times.shape[-1])

    return data_path


def read_channel(
    cache_root: Path | str, shot: object, category: str, dname: str
) -> ChannelRecord:
    """Read one cached channel back as the record :func:`write_channel` stored.

    Returns the data array (two-dimensional ``(nch, time)``), its sibling time
    base and the :data:`UNIT_ATTR` unit carried on the data array, so a caller
    holds one channel whole without re-deriving the layout.  This is the raw
    read beside the engine path — :class:`ZarrTransformEngine` and
    :class:`VirtualZarrView` return canonical values with no time base, so the
    caller that needs the time base reads it here.  A channel that is absent or
    half-written is refused rather than reported as an empty record.
    """

    root = Path(cache_root)
    data_path = channel_path(root, shot, category, dname)
    time_path = channel_time_path(root, shot, category, dname)
    try:
        data_array = zarr.open_array(data_path, mode="r")
        time_array = zarr.open_array(time_path, mode="r")
    except (KeyError, ValueError, OSError) as error:
        raise EddbCacheError(
            f"channel {eddb_token(shot)}/{category}/{dname} is not cached at "
            f"{data_path}"
        ) from error

    data = np.asarray(data_array[...], dtype="<f8")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    time = np.asarray(time_array[...], dtype="<f8").reshape(-1)
    unit = str(data_array.attrs.get(UNIT_ATTR, ""))
    nch = int(data_array.attrs.get(CHANNEL_COUNT_ATTR, data.shape[0]))
    token = read_eddb_token(root, shot)
    return ChannelRecord(
        shot=token if token is not None else str(shot),
        category=category,
        dname=dname,
        data=data,
        time=time,
        unit=unit,
        nch=nch,
        seq=int(data_array.attrs.get(SEQUENCE_NUMBER_ATTR, 0)),
    )


def fetch_channels(
    extractor: RemoteEddbExtractor,
    cache_root: Path | str,
    requests: Iterable[object],
) -> BatchResult:
    """Fetch the requested channels not already cached and write them.

    Requests already satisfied on disk make no transport call.  Every remaining
    request — including a half-written one whose time array is missing — is read
    in a single remote process, so one batch is one ssh session.  Each served
    record is written with :func:`write_channel`; a refused channel is reported
    in the returned :class:`BatchResult` rather than raising, so a batch mixing
    served and refused channels caches the served ones and a batch that is
    entirely refused writes nothing.
    """

    ordered = list(requests)
    missing = [
        request
        for request in ordered
        if not is_cached(cache_root, request.shot, request.category, request.dname)
    ]
    if not missing:
        return BatchResult(records=[], refusals=[])
    result = extractor.fetch_batch(missing)
    for record in result.records:
        write_channel(cache_root, record)
    return result


__all__ = [
    "CHANNEL_COUNT_ATTR",
    "EDDB_TOKEN_ATTR",
    "SEQUENCE_NUMBER_ATTR",
    "TIME_SUFFIX",
    "TIME_UNIT",
    "UNIT_ATTR",
    "EddbCacheError",
    "channel_path",
    "channel_time_path",
    "eddb_token",
    "fetch_channels",
    "is_cached",
    "normalised_shot",
    "read_channel",
    "read_eddb_token",
    "time_array_name",
    "write_channel",
]
