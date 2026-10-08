"""Read a JT-60SA shot's SELENE reconstruction from the EDDB ``PSRC`` category.

SELENE writes its calibrated reconstruction into the EDDB ``PSRC`` category on
the JT-60SA analysis server.  The quantities this reader takes are the plasma
boundary above the X-point (``surfABVxp``), the X-point (``calRX``,
``calZX``), the magnetic axis (``calRp0``, ``calZp0``) and the plasma current
(``calIp``), each with its own time base.  ``surfABVxp`` is one packed record
per time slice, read as a (604,) slot: two header rows, a 300-slot R block in
millimetres, two more header rows and a 300-slot Z block in millimetres.  Each
block's second header row holds the count of valid points, and the rest of the
slot is zero fill; this reader keeps only the valid points, so a slice's
contour carries no fill entry, and divides the millimetre block by 1000 so the
record is returned in metres.

The arrays are read through the on-demand cache in :mod:`imas_alambic.eddb`, so
a cached channel touches no network.  A channel that is not cached is fetched
with :func:`~imas_alambic.eddb.fetch_channels` through the extractor the caller
supplies; this module adds no transport of its own and never reads the EDDB by
another path.  A required channel that is neither cached nor served is refused
with an error naming the shot and the channel.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from imas_alambic.eddb import (
    fetch_channels,
    is_cached,
    normalised_shot,
    read_channel,
)
from imas_alambic.eddb_remote import ChannelRequest
from imas_ambix.data.paths import JT60SA_ROOT

if TYPE_CHECKING:
    from imas_alambic.eddb_remote import RemoteEddbExtractor

__all__ = ["PsrcBoundary", "PsrcRecord", "SelenePsrcError", "read_psrc_record"]

#: The EDDB category every SELENE reconstruction array lives under.
PSRC_CATEGORY = "PSRC"

#: The packed ``surfABVxp`` slot: a type marker and a point count before each
#: block, then a 300-wide block of R (then Z) vertex coordinates in millimetres,
#: zero-padded after the count.  ``surfABVxp``'s total slot is 604 wide.
BOUNDARY_SLOT_LENGTH = 604
_R_COUNT_INDEX = 1
_R_START = 2
_Z_COUNT_INDEX = 303
_Z_START = 304
_MM_PER_M = 1000.0

#: The channels one PSRC record is built from, in the order they are read.
_RECORD_CHANNELS: tuple[str, ...] = (
    "calIp",
    "calRX",
    "calZX",
    "calRp0",
    "calZp0",
    "surfABVxp",
)


class SelenePsrcError(RuntimeError):
    """Raised when a shot's PSRC read cannot be completed."""


@dataclass(frozen=True)
class PsrcBoundary:
    """The PSRC boundary above the X-point, one closed contour per time slice.

    ``r`` and ``z`` each hold one 1-D array of coordinates in metres per slice,
    carrying only the slice's valid vertices so no zero-padded fill entry
    survives into the record.  ``time`` gives the shared time base.
    """

    r: tuple[np.ndarray, ...]
    z: tuple[np.ndarray, ...]
    time: np.ndarray


@dataclass(frozen=True)
class PsrcRecord:
    """One shot's SELENE reconstruction, in metres and amperes.

    Every quantity carries its own time base: the boundary above the X-point in
    metres, the X-point R and Z in metres on a shared base, the magnetic-axis R
    and Z in metres on a shared base, and the plasma current in amperes.
    """

    shot: str
    boundary: PsrcBoundary
    x_point_r: np.ndarray
    x_point_z: np.ndarray
    x_point_time: np.ndarray
    magnetic_axis_r: np.ndarray
    magnetic_axis_z: np.ndarray
    magnetic_axis_time: np.ndarray
    plasma_current: np.ndarray
    plasma_current_time: np.ndarray


def _flatten_series(record: object) -> np.ndarray:
    """Return a single-channel record's samples as a 1-D float64 array."""

    data = np.asarray(record.data, dtype=np.float64)
    return data.reshape(-1)


def _decode_boundary(data: np.ndarray, time: np.ndarray) -> PsrcBoundary:
    """Decode ``surfABVxp``'s packed slots into per-slice metre contours.

    ``data`` is (604, ntime) in millimetres; each column is decoded into its
    ``nA`` valid R and Z vertices, dropping the zero fill in each block.
    """

    if data.shape[0] != BOUNDARY_SLOT_LENGTH:
        raise SelenePsrcError(
            f"surfABVxp carries a {data.shape[0]}-wide slot; the packed boundary "
            f"layout is {BOUNDARY_SLOT_LENGTH} wide"
        )
    r_slices: list[np.ndarray] = []
    z_slices: list[np.ndarray] = []
    for column in range(data.shape[1]):
        slot = data[:, column]
        r_count = int(round(float(slot[_R_COUNT_INDEX])))
        r_slice = np.asarray(slot[_R_START : _R_START + r_count], dtype=np.float64)
        z_count = int(round(float(slot[_Z_COUNT_INDEX])))
        z_slice = np.asarray(slot[_Z_START : _Z_START + z_count], dtype=np.float64)
        r_slices.append(r_slice / _MM_PER_M)
        z_slices.append(z_slice / _MM_PER_M)
    return PsrcBoundary(r=tuple(r_slices), z=tuple(z_slices), time=np.asarray(time))


def read_psrc_record(
    shot: object,
    *,
    cache_root: Path | str = JT60SA_ROOT,
    extractor: RemoteEddbExtractor | None = None,
) -> PsrcRecord:
    """Return one shot's SELENE PSRC record, reading cached channels as needed.

    ``cache_root`` defaults to the JT-60SA EDDB cache root.  A channel not
    already cached is fetched through ``extractor``; with no extractor, or when
    the extractor does not serve it, the read is refused with an error naming
    the shot and the channel.
    """

    requests = [
        ChannelRequest(shot=str(shot), category=PSRC_CATEGORY, dname=dname)
        for dname in _RECORD_CHANNELS
    ]
    if extractor is not None:
        fetch_channels(extractor, cache_root, requests)

    for dname in _RECORD_CHANNELS:
        if not is_cached(cache_root, shot, PSRC_CATEGORY, dname):
            raise SelenePsrcError(
                f"PSRC channel {dname} for shot {normalised_shot(shot)} is neither "
                f"cached under {Path(cache_root)} nor served by the extractor"
            )

    channels = {
        dname: read_channel(cache_root, shot, PSRC_CATEGORY, dname)
        for dname in _RECORD_CHANNELS
    }
    boundary = _decode_boundary(
        np.asarray(channels["surfABVxp"].data, dtype=np.float64),
        np.asarray(channels["surfABVxp"].time, dtype=np.float64),
    )
    return PsrcRecord(
        shot=normalised_shot(shot),
        boundary=boundary,
        x_point_r=_flatten_series(channels["calRX"]),
        x_point_z=_flatten_series(channels["calZX"]),
        x_point_time=np.asarray(channels["calRX"].time, dtype=np.float64),
        magnetic_axis_r=_flatten_series(channels["calRp0"]),
        magnetic_axis_z=_flatten_series(channels["calZp0"]),
        magnetic_axis_time=np.asarray(channels["calRp0"].time, dtype=np.float64),
        plasma_current=_flatten_series(channels["calIp"]),
        plasma_current_time=np.asarray(channels["calIp"].time, dtype=np.float64),
    )
