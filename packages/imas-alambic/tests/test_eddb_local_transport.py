"""The EDDB reader run locally, with no ssh hop.

On a host whose own python already has numpy, the extractor is given an empty
``ssh_command`` and ``RemoteEddbExtractor._argv`` composes
``nice -n 19 <python> -c REMOTE_SCRIPT`` as one argv list, with no ssh prefix
and no module-load shell.  These tests drive that local route end to end
through a stand-in ``eddb_pwrapper`` placed on the reader script's own path, so
the real :data:`REMOTE_SCRIPT` runs and its envelope is decoded by the same
:func:`decode_batch` the ssh route uses.  The arrays the stand-in returns are
compared bit for bit against what the local route delivers, first on synthetic
arrays and then on E101154's ``pf_active`` coil-current channels read from the
cached store under the description-store root.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pytest
import zarr

from imas_alambic.eddb_remote import (
    NICE_LEVEL,
    REMOTE_SCRIPT,
    ChannelRequest,
    RemoteEddbExtractor,
    SubprocessTransport,
)

LOGGER = logging.getLogger("test_eddb_local_transport")

#: The description-store root the cached pulse lives under.
JT60SA_ROOT = Path("/work/projects/imas_gpu/jt60sa")

#: The ``pf_active`` map's coil-current channels for the cached E101154 pulse.
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

#: A stand-in for the analysis server's ``eddb_pwrapper``.  It reads the channel
#: table written beside it (``channels.json``, one ``.npy`` data array and one
#: ``.npy`` time array per data name) and answers ``eddbreadTime`` with those
#: arrays in the shape the real wrapper returns: the data as a two-dimensional
#: ``(nch, ntime)`` array, the time as a one-dimensional array, the unit as a
#: one-entry list and an integer sequence number.
STANDIN_WRAPPER = '''
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
'''


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


def test_local_route_composes_one_argv_list_with_no_ssh_and_no_module_shell():
    extractor = RemoteEddbExtractor(ssh_command=(), remote_python="/venv/bin/python")

    argv = extractor._argv()

    assert argv == [
        "nice",
        "-n",
        str(NICE_LEVEL),
        "/venv/bin/python",
        "-c",
        REMOTE_SCRIPT,
    ]
    joined = " ".join(argv)
    assert "ssh" not in joined
    assert "module load" not in joined
    assert "module unload" not in joined


def test_local_route_reads_synthetic_arrays_through_the_reader_script(tmp_path):
    data = np.arange(8, dtype="<f8").reshape(2, 4)
    time = np.array([0.0, 0.5, 1.0, 1.5], dtype="<f8")
    _write_standin(tmp_path, {"SYN1": (data, time, 3)})

    result = _local_extractor(tmp_path).fetch_batch([_request("SYN1")])

    assert result.refusals == []
    (record,) = result.records
    assert np.array_equal(record.data, data)
    assert np.array_equal(record.time, time)
    assert record.unit == "A"
    assert record.nch == 2


def test_local_route_matches_cached_pf_active_arrays_bit_for_bit(tmp_path):
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
