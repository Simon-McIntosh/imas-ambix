"""Stage one shot's G-EQDSK poloidal flux into a cache group the engine reads.

An equilibrium poloidal-flux grid is a 2-D field, not a time series over named
coils, so the EDDB channel fetch a signal map normally drives cannot serve it.
The G-EQDSK file is the source; this module reads it through the single
:func:`imas_ambix.challenge.loader.load_geqdsk` reader and stages the flux into
a per-shot Zarr group laid out exactly as :mod:`imas_alambic.eddb`'s cache
writer lays a fetched channel — ``{shot}.zarr/{group}/{array}`` with a sibling
``{array}_time`` — so :class:`~imas_alambic.virtual_zarr.VirtualZarrView` reads
it like any cached channel, through a signal-map rule that binds it.

The stored flux is the G-EQDSK's own values, in the file's declared COCOS 1
convention (Wb/rad).  No convention factor is applied here: the COCOS-17
conversion lives in the map rule's ``source_cocos``, so the store keeps the
source measurement unchanged and the rule owns the COCOS 1 -> 17 correction.

The grid is staged once and never rewritten; a second call finds the channel
already cached and left as measured, exactly as a fetched channel is.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr

from imas_alambic.eddb import (
    UNIT_ATTR,
    normalised_shot,
    write_channel,
)
from imas_alambic.eddb_remote import ChannelRecord

from imas_ambix.challenge.loader import load_geqdsk
from imas_ambix.data.paths import JT60SA_ROOT

#: The cache group the staged flux occupies.  It sits beside the fetched EDDB
#: categories rather than inside one, so a rule binds it by this group name and
#: the engine reads it as it reads ``FAME`` or ``PSRC``.
GEQDSK_GROUP = "GEQDSK"

#: The flux array and the two grid-coordinate arrays the rule binds.
PSI_ARRAY = "PSI"
R_ARRAY = "GRID_R"
Z_ARRAY = "GRID_Z"

#: The flux unit as stored: the G-EQDSK's own per-radian convention.
PSI_UNIT = "Wb/rad"

#: The grid-coordinate unit.
GRID_UNIT = "m"

#: Milliseconds per second, for the store's IMAS-canonical time base.
_MS_PER_S = 1000.0


def default_geqdsk_path(shot: object, *, root: Path = JT60SA_ROOT) -> Path:
    """Return the G-EQDSK path a shot's file is staged from by default."""

    return Path(root) / f"{normalised_shot(shot)}.geqdsk"


def write_geqdsk_store(
    cache_root: Path | str,
    shot: object,
    *,
    geqdsk_path: Path | str | None = None,
    time_ms: float | None = None,
) -> Path:
    """Stage one shot's G-EQDSK flux and grid into its cache group.

    The flux is read through :func:`load_geqdsk` at ``time_ms`` (or the time the
    file's own header declares), flattened to the ``(channel, time)`` shape a
    cached EDDB channel carries — one time sample, one row per grid point — and
    written as ``{group}/{PSI_ARRAY}`` with its ``_time`` sibling.  The R and Z
    coordinate vectors are written as ``{group}/{R_ARRAY}`` and
    ``{group}/{Z_ARRAY}``; they are grid axes rather than time series, so they
    carry no time base.  A group already holding the flux is left as measured.

    ``cache_root`` is the store root the caller owns.  Nothing under the
    G-EQDSK's own tree is written.
    """

    path = Path(geqdsk_path) if geqdsk_path is not None else default_geqdsk_path(shot)
    labels = load_geqdsk(path, time_ms=time_ms)

    # psirz is (frame, Z, R); the store keeps the grid as (Z, R) with the single
    # time sample last, flattened so the cached channel is (channel, time).
    flux = np.asarray(labels.psirz[0], dtype=np.float64)
    radial = np.asarray(labels.grid_r_m, dtype=np.float64)
    vertical = np.asarray(labels.grid_z_m, dtype=np.float64)
    if flux.shape != (vertical.size, radial.size):
        raise ValueError(
            f"G-EQDSK flux grid {flux.shape} does not match its "
            f"{vertical.size}x{radial.size} coordinate vectors"
        )

    seconds = np.asarray(labels.time_ms, dtype=np.float64) / _MS_PER_S
    record = ChannelRecord(
        shot=str(shot),
        category=GEQDSK_GROUP,
        dname=PSI_ARRAY,
        data=flux.reshape(flux.size, 1),
        time=seconds,
        unit=PSI_UNIT,
        nch=int(flux.size),
        seq=0,
    )
    data_path = write_channel(cache_root, record)

    store = zarr.open_group(
        Path(cache_root) / f"{normalised_shot(shot)}.zarr", mode="a"
    )
    group = store.require_group(GEQDSK_GROUP)
    for name, values in ((R_ARRAY, radial), (Z_ARRAY, vertical)):
        if name in group:
            continue
        array = group.create_array(name, data=values)
        array.attrs[UNIT_ATTR] = GRID_UNIT

    return data_path


def read_geqdsk_grid(
    cache_root: Path | str, shot: object
) -> tuple[np.ndarray, np.ndarray]:
    """Return a staged shot's R and Z coordinate vectors as stored."""

    store = zarr.open_group(
        Path(cache_root) / f"{normalised_shot(shot)}.zarr", mode="r"
    )
    group = store[GEQDSK_GROUP]
    radial = np.asarray(group[R_ARRAY][...], dtype=np.float64).reshape(-1)
    vertical = np.asarray(group[Z_ARRAY][...], dtype=np.float64).reshape(-1)
    return radial, vertical


__all__ = [
    "GEQDSK_GROUP",
    "GRID_UNIT",
    "PSI_ARRAY",
    "PSI_UNIT",
    "R_ARRAY",
    "Z_ARRAY",
    "default_geqdsk_path",
    "read_geqdsk_grid",
    "write_geqdsk_store",
]