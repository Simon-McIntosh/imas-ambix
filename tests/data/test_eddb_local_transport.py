"""The local EDDB reader preserves cached facility arrays bit for bit."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pytest
import zarr

from imas_alambic.eddb_remote import (
    ChannelRequest,
    RemoteEddbExtractor,
    SubprocessTransport,
)
from imas_alambic.machine_map import MachineMapError, bundle_for_machine
from imas_ambix.data.paths import JT60SA_ROOT

LOGGER = logging.getLogger("test_eddb_local_transport")
PF_ACTIVE_ARRAYS = (
    "curCS1LKAT",
    "curCS2LKAT",
    "curCS3LKAT",
    "curCS4LKAT",
    "curEF1LKAT",
    "curEF2LKAT",
    "curEF3LKAT",
    "curEF4LKAT",
    "curEF5LKAT",
    "curEF6HiTe",
)
PF_ACTIVE_CATEGORY = "MMSYS"
PF_ACTIVE_SHOT = "E101154"

STANDIN_WRAPPER = """
import json, os
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_HERE, "channels.json")) as _fh:
    _CHANNELS = json.load(_fh)


class eddbWrapper:
    def __init__(self, lib_path):
        self.lib_path = lib_path

    def eddbOpen(self):
        return True

    def eddbClose(self):
        return True

    def eddbreadOne(self, *args, **kwargs):
        return False, None

    def eddbreadTime(self, shot, category, dname, t1, t2):
        spec = _CHANNELS[dname]
        data = np.load(spec["data"])
        time = np.load(spec["time"])
        return True, {
            "data": data,
            "time": time,
            "unit": ["A"],
            "seq": int(spec.get("seq", 0)),
        }
"""


def _write_standin(api_dir: Path, channels: dict) -> None:
    """Place the stand-in wrapper and the arrays it serves beside each other."""

    table = {}
    for dname, (data, time, seq) in channels.items():
        data_path = api_dir / f"{dname}_data.npy"
        time_path = api_dir / f"{dname}_time.npy"
        np.save(data_path, np.asarray(data, dtype="<f8"))
        np.save(time_path, np.asarray(time, dtype="<f8"))
        table[dname] = {"data": str(data_path), "time": str(time_path), "seq": int(seq)}
    (api_dir / "channels.json").write_text(json.dumps(table))
    (api_dir / "eddb_pwrapper.py").write_text(STANDIN_WRAPPER)


def _local_extractor(api_dir: Path, *, nice: bool = False) -> RemoteEddbExtractor:
    return RemoteEddbExtractor(
        ssh_command=(),
        remote_python=sys.executable,
        api_path=str(api_dir),
        nice=nice,
        transport=SubprocessTransport(),
    )


def _request(dname: str) -> ChannelRequest:
    return ChannelRequest(shot=PF_ACTIVE_SHOT, category=PF_ACTIVE_CATEGORY, dname=dname)


def test_local_route_matches_cached_pf_active_arrays_bit_for_bit(tmp_path):
    try:
        bundle_for_machine("jt-60sa")
    except MachineMapError:
        pytest.skip("JT-60SA map bundle is unavailable; set IMAS_ALAMBIC_MAP_PATH")
    store = JT60SA_ROOT / "101154.zarr"
    if not store.is_dir():
        pytest.skip(f"cached pulse {store} is not present")

    channels = {}
    for dname in PF_ACTIVE_ARRAYS:
        data = np.asarray(
            zarr.open_array(store / PF_ACTIVE_CATEGORY / dname, mode="r")[...]
        )
        time = np.asarray(
            zarr.open_array(store / PF_ACTIVE_CATEGORY / f"{dname}_time", mode="r")[...]
        )
        channels[dname] = (data, time, 0)
    _write_standin(tmp_path, channels)

    messages: list[str] = []

    class _Collect(logging.Handler):
        def emit(self, record):
            messages.append(record.getMessage())

    handler = _Collect()
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)
    try:
        result = _local_extractor(tmp_path).fetch_batch(
            [_request(d) for d in PF_ACTIVE_ARRAYS]
        )
        assert result.refusals == []
        assert len(result.records) == len(PF_ACTIVE_ARRAYS)
        for record in result.records:
            standin_data, standin_time, _ = channels[record.dname]
            assert record.data.shape == standin_data.shape
            assert record.time.shape == standin_time.shape
            assert np.array_equal(record.data, standin_data)
            assert np.array_equal(record.time, standin_time)
            LOGGER.info(
                "local pf_active %s data%s time%s equal stand-in arrays bit for bit",
                record.dname,
                record.data.shape,
                record.time.shape,
            )
    finally:
        LOGGER.removeHandler(handler)

    assert sum("equal stand-in arrays bit for bit" in m for m in messages) == len(
        PF_ACTIVE_ARRAYS
    )
