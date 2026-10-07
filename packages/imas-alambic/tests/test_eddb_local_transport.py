"""The EDDB reader run locally, with no ssh hop.

On a host whose own python already has numpy, the extractor is given an empty
``ssh_command`` and ``RemoteEddbExtractor._argv`` composes
``nice -n 19 <python> -c REMOTE_SCRIPT`` as one argv list, with no ssh prefix
and no module-load shell.  These tests drive that local route end to end
through a stand-in ``eddb_pwrapper`` placed on the reader script's own path, so
the real :data:`REMOTE_SCRIPT` runs and its envelope is decoded by the same
:func:`decode_batch` the ssh route uses. The stand-in's synthetic arrays are
compared bit for bit against what the local route delivers.

The stand-in wrapper and the helpers that place it are owned by
:mod:`eddb_standin` beside this file, so the stand-in has one definition; the
ambix suite reaches the same one rather than carrying a second copy.
"""

from __future__ import annotations

import numpy as np
from eddb_standin import local_extractor, write_standin

from imas_alambic.eddb_remote import (
    NICE_LEVEL,
    REMOTE_SCRIPT,
    ChannelRequest,
    RemoteEddbExtractor,
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
    write_standin(tmp_path, {"SYN1": (data, time, 3)})

    result = local_extractor(tmp_path).fetch_batch([_request("SYN1")])

    assert result.refusals == []
    (record,) = result.records
    assert np.array_equal(record.data, data)
    assert np.array_equal(record.time, time)
    assert record.unit == "A"
    assert record.nch == 2
