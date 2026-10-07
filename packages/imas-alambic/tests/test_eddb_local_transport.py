"""The EDDB reader run locally, with no ssh hop.

On a host whose own python already has numpy, the extractor is given an empty
``ssh_command`` and ``RemoteEddbExtractor._argv`` composes
``nice -n 19 <python> -c REMOTE_SCRIPT`` as one argv list, with no ssh prefix
and no module-load shell.  These tests drive that local route end to end
through a stand-in ``eddb_pwrapper`` placed on the reader script's own path, so
the real :data:`REMOTE_SCRIPT` runs and its envelope is decoded by the same
:func:`decode_batch` the ssh route uses. The stand-in's synthetic arrays are
compared bit for bit against what the local route delivers.
"""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

from imas_alambic.eddb_remote import (
    NICE_LEVEL,
    REMOTE_SCRIPT,
    ChannelRequest,
    RemoteEddbExtractor,
    SubprocessTransport,
)

#: A stand-in for the analysis server's ``eddb_pwrapper``.  It reads the channel
#: table written beside it (``channels.json``, one ``.npy`` data array and one
#: ``.npy`` time array per data name) and answers ``eddbreadTime`` with those
#: arrays in the shape the real wrapper returns: the data as a two-dimensional
#: ``(nch, ntime)`` array, the time as a one-dimensional array, the unit as a
#: one-entry list and an integer sequence number.
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
    return ChannelRequest(shot="1", category="SYN", dname=dname)


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
