"""The one SELENE PSRC reader: boundary, X-point, axis and plasma current.

The tests seed a temporary on-demand cache by copying the real cached PSRC
arrays for shot 100599 read-only out of ``/work/projects/imas_gpu/jt60sa``, so
the axis, X-point and plasma-current reads are exercised against real data, and
add a synthetic ``surfABVxp`` packed in millimetres with zero-filled slots so
the boundary decode can be read at known values.  Nothing under ``/work`` is
written, and no test opens ssh.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import zarr

from imas_alambic.eddb import is_cached, write_channel
from imas_alambic.eddb_remote import (
    ChannelRecord,
    ChannelRefusal,
    RemoteEddbExtractor,
    encode_batch,
)
from imas_ambix.data.selene_psrc import SelenePsrcError, read_psrc_record

#: The real shot whose cached PSRC arrays seed the read.
SHOT = "E100599"
SHOT_INT = "100599"
CATEGORY = "PSRC"
SOURCE_PULSE = Path("/work/projects/imas_gpu/jt60sa/100599.zarr")

#: The real single-channel arrays copied from the source pulse.
REAL_CHANNELS = ("calRX", "calZX", "calRp0", "calZp0", "calIp")

#: The packed ``surfABVxp`` slot width: two header rows, a 300-slot R block in
#: millimetres, two more header rows and a 300-slot Z block in millimetres.
SLOT = 604
_R_COUNT = 1
_R_START = 2
_Z_COUNT = 303
_Z_START = 304
_FILL_MARKER = 3.0


def _seed_real_channels(cache_root: Path) -> None:
    """Copy the real single-channel arrays and their time bases into the cache."""

    pulse = cache_root / f"{SHOT_INT}.zarr"
    (pulse / CATEGORY).mkdir(parents=True, exist_ok=True)
    shutil.copy(SOURCE_PULSE / "zarr.json", pulse / "zarr.json")
    shutil.copy(SOURCE_PULSE / CATEGORY / "zarr.json", pulse / CATEGORY / "zarr.json")
    for name in REAL_CHANNELS:
        for node in (name, f"{name}_time"):
            shutil.copytree(SOURCE_PULSE / CATEGORY / node, pulse / CATEGORY / node)


def _pack_slice(r_mm: list[float], z_mm: list[float]) -> np.ndarray:
    """Pack one boundary slice into its 604-wide slot, zero-filling each block."""

    slot = np.zeros(SLOT)
    slot[0] = _FILL_MARKER
    slot[_R_COUNT] = len(r_mm)
    slot[_R_START : _R_START + len(r_mm)] = r_mm
    slot[302] = _FILL_MARKER
    slot[_Z_COUNT] = len(z_mm)
    slot[_Z_START : _Z_START + len(z_mm)] = z_mm
    return slot


def _synthetic_boundary() -> tuple[np.ndarray, np.ndarray]:
    """A packed (604, 2) boundary in millimetres and its two-sample time base."""

    slices = [
        _pack_slice([1000.0, 1100.0, 1200.0], [2000.0, 2100.0, 2200.0]),
        _pack_slice([1010.0, 1110.0], [2010.0, 2110.0]),
    ]
    return np.column_stack(slices), np.array([0.0, 0.5])


def _seed_boundary(cache_root: Path, data: np.ndarray, time: np.ndarray) -> None:
    write_channel(
        cache_root,
        ChannelRecord(
            shot=SHOT,
            category=CATEGORY,
            dname="surfABVxp",
            data=data,
            time=time,
            unit="mm",
            nch=SLOT,
            seq=0,
        ),
    )


def _source_array(name: str) -> np.ndarray:
    array = zarr.open_array(SOURCE_PULSE / CATEGORY / name, mode="r")
    return np.asarray(array[...], dtype=np.float64).reshape(-1)


class _StubTransport:
    """A transport answering a batch with the synthetic records it is given.

    ``records`` maps a data name to the :class:`ChannelRecord` to serve, and
    ``refuse`` maps a data name to the EDDB return code to answer instead.
    """

    def __init__(self, records, refuse=None) -> None:
        self.calls: list[list[str]] = []
        self._records = dict(records)
        self._refuse = dict(refuse or {})

    def run(self, argv, payload: bytes) -> bytes:
        self.calls.append(list(argv))
        request = json.loads(payload)
        records = []
        refusals = []
        for spec in request["requests"]:
            name = spec["dname"]
            if name in self._refuse:
                refusals.append(
                    ChannelRefusal(
                        spec["shot"], spec["category"], name, self._refuse[name]
                    )
                )
            else:
                records.append(self._records[name])
        return encode_batch(records, refusals)


def _stub_extractor(records, refuse=None):
    transport = _StubTransport(records, refuse=refuse)
    return RemoteEddbExtractor(transport=transport), transport


def _boundary_record(data: np.ndarray, time: np.ndarray) -> ChannelRecord:
    return ChannelRecord(
        shot=SHOT,
        category=CATEGORY,
        dname="surfABVxp",
        data=data,
        time=time,
        unit="mm",
        nch=SLOT,
        seq=0,
    )


def test_boundary_is_in_metres_with_fill_entries_dropped(tmp_path):
    cache = tmp_path / "cache"
    _seed_real_channels(cache)
    data, time = _synthetic_boundary()
    _seed_boundary(cache, data, time)

    record = read_psrc_record(SHOT, cache_root=cache)

    assert record.shot == SHOT_INT
    assert len(record.boundary.r) == 2
    assert np.allclose(record.boundary.r[0], [1.0, 1.1, 1.2])
    assert np.allclose(record.boundary.z[0], [2.0, 2.1, 2.2])
    assert np.allclose(record.boundary.r[1], [1.01, 1.11])
    assert np.allclose(record.boundary.time, [0.0, 0.5])
    # Fill entries are dropped: three real vertices, not the 300-wide R slot,
    # and no zero-padded slot value survives as a vertex.
    assert record.boundary.r[0].shape == (3,)
    assert 0.0 not in record.boundary.r[0]


def test_axis_xpoint_and_current_match_the_cached_arrays(tmp_path):
    cache = tmp_path / "cache"
    _seed_real_channels(cache)
    data, time = _synthetic_boundary()
    _seed_boundary(cache, data, time)

    record = read_psrc_record(SHOT, cache_root=cache)

    sample = 5
    assert record.magnetic_axis_r[sample] == _source_array("calRp0")[sample]
    assert record.magnetic_axis_z[sample] == _source_array("calZp0")[sample]
    assert record.x_point_r[sample] == _source_array("calRX")[sample]
    assert record.x_point_z[sample] == _source_array("calZX")[sample]
    assert record.plasma_current[sample] == _source_array("calIp")[sample]
    assert np.array_equal(record.magnetic_axis_time, _source_array("calRp0_time"))
    assert np.array_equal(record.x_point_time, _source_array("calRX_time"))
    assert np.array_equal(record.plasma_current_time, _source_array("calIp_time"))


def test_a_stub_extractor_serves_the_boundary_which_the_record_then_reads(tmp_path):
    cache = tmp_path / "cache"
    _seed_real_channels(cache)
    assert not is_cached(cache, SHOT, CATEGORY, "surfABVxp")

    data, time = _synthetic_boundary()
    extractor, transport = _stub_extractor({"surfABVxp": _boundary_record(data, time)})

    record = read_psrc_record(SHOT, cache_root=cache, extractor=extractor)

    assert len(transport.calls) == 1
    assert is_cached(cache, SHOT, CATEGORY, "surfABVxp")
    assert np.allclose(record.boundary.r[0], [1.0, 1.1, 1.2])


def test_a_refused_channel_raises_an_error_naming_the_shot_and_channel(tmp_path):
    cache = tmp_path / "cache"
    _seed_real_channels(cache)
    extractor, _ = _stub_extractor({}, refuse={"surfABVxp": 1015})

    with pytest.raises(SelenePsrcError) as excinfo:
        read_psrc_record(SHOT, cache_root=cache, extractor=extractor)

    message = str(excinfo.value)
    assert SHOT_INT in message
    assert "surfABVxp" in message


def test_an_uncached_channel_with_no_extractor_is_refused(tmp_path):
    cache = tmp_path / "cache"
    _seed_real_channels(cache)

    with pytest.raises(SelenePsrcError) as excinfo:
        read_psrc_record(SHOT, cache_root=cache)

    message = str(excinfo.value)
    assert SHOT_INT in message
    assert "surfABVxp" in message
