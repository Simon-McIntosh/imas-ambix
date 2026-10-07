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

#: A canned-response stand-in for the analysis server's ``eddb_pwrapper``.
#:
#: Unlike :data:`STANDIN_WRAPPER`, which reads a ``channels.json`` table the
#: caller writes, this one answers a fixed shape with no table: a default
#: two-by-two series, and four names that drive the extractor's edge paths --
#: ``NOTIME`` answers with no time base, ``REFUSED``/``ABSENT`` answer a false
#: return carrying an EDDB code, ``NOIRC`` a false return with no code, and
#: ``MULTIUNIT`` a two-entry unit list.  It exists so the reader envelope's
#: refusals and unit normalisation can be exercised without a data table.
CANNED_WRAPPER = '''
import numpy as np


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
        if dname == "NOTIME":
            return True, {"data": np.arange(6.0).reshape(2, 3)}
        if dname == "REFUSED":
            return False, {"irc": 1015, "ircgrp": 1}
        if dname == "ABSENT":
            return False, {"irc": 1013, "ircgrp": 1}
        if dname == "NOIRC":
            return False, {}
        if dname == "MULTIUNIT":
            return True, {
                "data": np.array([[1.0, 2.0, 3.0]]),
                "time": np.array([0.0, 0.5, 1.0]),
                "unit": ["A", "V"],
                "seq": 1,
            }
        return True, {
            "data": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            "time": np.array([0.0, 0.5, 1.0]),
            "unit": ["A"],
            "seq": 42,
        }
'''


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


def write_canned_wrapper(api_dir: Path) -> None:
    """Place :data:`CANNED_WRAPPER` as ``eddb_pwrapper.py`` in ``api_dir``."""

    (api_dir / "eddb_pwrapper.py").write_text(CANNED_WRAPPER)


__all__ = [
    "CANNED_WRAPPER",
    "STANDIN_WRAPPER",
    "ChannelRequest",
    "local_extractor",
    "write_canned_wrapper",
    "write_standin",
]
