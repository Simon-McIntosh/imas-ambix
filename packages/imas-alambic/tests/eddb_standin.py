"""A stand-in for the analysis server's ``eddb_pwrapper``, owned once.

The engine's own test package and the ambix suite both drive the EDDB reader
locally, with no ssh hop, through a fake ``eddb_pwrapper`` placed on the reader
script's path.  Both used to carry a copy of the wrapper and the two helpers
that write it, so the two could drift apart.

The owner sits here, beside the engine test that first needed it, because the
engine's tests must keep running in a clean virtual environment holding only the
engine wheels and pytest -- so this helper cannot live in a shared package that
imports the wider project.  The engine test reaches it as a plain module in its
own directory; the ambix test loads it by the path recorded at the call site.

The wrapper reads the channel table written beside it (``channels.json``, one
``.npy`` data array and one ``.npy`` time array per data name) and answers
``eddbreadTime`` in the shape the real wrapper returns: the data as a
two-dimensional ``(nch, ntime)`` array, the time as a one-dimensional array, the
unit as a one-entry list and an integer sequence number.
"""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING

import numpy as np

from imas_alambic.eddb_remote import (
    ChannelRequest,
    RemoteEddbExtractor,
    SubprocessTransport,
)

if TYPE_CHECKING:
    from pathlib import Path

#: A stand-in for the analysis server's ``eddb_pwrapper``.
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


def write_standin(api_dir: Path, channels: dict) -> None:
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


def local_extractor(
    api_dir: Path, *, nice: bool = False, python: str | None = None
) -> RemoteEddbExtractor:
    """An extractor that runs the reader locally over the stand-in in ``api_dir``."""

    return RemoteEddbExtractor(
        ssh_command=(),
        remote_python=sys.executable if python is None else python,
        api_path=str(api_dir),
        nice=nice,
        transport=SubprocessTransport(),
    )


__all__ = [
    "STANDIN_WRAPPER",
    "ChannelRequest",
    "local_extractor",
    "write_standin",
]
