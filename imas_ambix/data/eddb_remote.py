"""Batch reader for JT-60SA EDDB channels over ssh.

The authorised JT-60SA data path is the EDDB C library through its Python
wrapper on the Naka analysis server (``/analysis/src/eddb/eddb_pwrapper.py``
and ``/analysis/lib/libeddb.so``), reached as ``ssh jt-60sa``.  This module
owns the remote half of the on-demand cache: one batch of ``(shot, category,
data name)`` requests is sent down a single ssh session, the remote script
reads every channel in that session, and the arrays come back in one compact
binary envelope.  Ambix owns this extractor rather than importing imas-codex's
remote layer, so that reading a map does not drag the codex graph, LLM and
tool-installation dependencies into every ambix process.  Codex stays the
reference for the EDDB call shapes; it is not a dependency.

The remote script runs under the server's python with only the standard
library, numpy and ``eddb_pwrapper`` available.  It reads its request JSON on
stdin and writes the envelope to stdout, so nothing but the script itself has
to be staged on the far side.  The client encodes and decodes the same
envelope in :func:`encode_batch` / :func:`decode_batch`.
"""

from __future__ import annotations

import json
import os
import shlex
import struct
import subprocess
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

#: The ssh invocation the extractor uses by default.  It is configuration, not
#: a constant baked into the transport: a caller with a different control-master
#: alias or config passes its own ``ssh_command``.  ``ssh -F ~/.ssh/config``
#: does not expand ``~`` itself, so the config path is expanded at call time in
#: :meth:`RemoteEddbExtractor._argv`.
DEFAULT_SSH_COMMAND: tuple[str, ...] = ("ssh", "-F", "~/.ssh/config", "jt-60sa")

#: The python on the analysis server is selected through the environment
#: modules: the system ``python3`` is 3.9 without numpy, so the remote command
#: loads the 3.12 module and calls that module's ``python``.  The unload first
#: makes the command idempotent when an older module is already active.
PYTHON_MODULE_UNLOAD = "python/3.5.6"
PYTHON_MODULE_LOAD = "python/3.12"

#: The extractor runs niced so a batch never competes with the analysis
#: server's own work.
NICE_LEVEL = 19

_MAGIC = b"EDDB1\n"
_STRUCT = struct.Struct("<I")


class EddbRemoteError(RuntimeError):
    """Raised when a remote EDDB batch cannot be requested or decoded."""


@dataclass(frozen=True)
class ChannelRequest:
    """One ``(shot, category, data name)`` channel to read from the EDDB."""

    shot: str
    category: str
    dname: str

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.shot, self.category, self.dname)


@dataclass(frozen=True)
class ChannelRecord:
    """One channel as it comes back from the EDDB, in raw units and sign.

    ``data`` is two-dimensional ``(nch, ntime)``; a single-channel record is
    normalised to one row so a reader never has to branch on the EDDB class.
    ``unit`` is the EDDB unit string verbatim, ``nch`` the channel count and
    ``seq`` the EDDB sequence number, all carried onto the cached array as
    attributes.  ``time`` is the record's own time base.
    """

    shot: str
    category: str
    dname: str
    data: np.ndarray
    time: np.ndarray
    unit: str
    nch: int
    seq: int

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.shot, self.category, self.dname)


@dataclass(frozen=True)
class ChannelRefusal:
    """One requested channel EDDB did not serve, with its return code.

    ``code`` is the EDDB return code the remote read produced (1015 when the
    shot's store exists but the datum was never written, 1013 when the shot is
    absent from the data set).  A refusal names the whole request so a caller can
    report it without re-deriving which channel failed.
    """

    shot: str
    category: str
    dname: str
    code: int

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.shot, self.category, self.dname)


@dataclass(frozen=True)
class BatchResult:
    """What one remote session returned: served records and per-channel refusals.

    A batch mixing served and refused channels carries both here, so a caller
    caches the records and reports the refusals without the refusal aborting the
    batch.  ``records`` and ``refusals`` partition the requests the remote
    answered.
    """

    records: list[ChannelRecord]
    refusals: list[ChannelRefusal]


def normalise_unit(unit: object) -> str:
    """Return the EDDB unit string, refusing a multi-entry unit list.

    EDDB returns the unit as a list; a single-entry list is the unit itself
    (``["A"]`` is ``A``), while a multi-entry list names several units and is
    refused rather than joined into a string no reader can interpret.
    """

    if unit is None:
        return ""
    if isinstance(unit, (list, tuple)):
        if len(unit) == 1:
            return str(unit[0])
        raise EddbRemoteError(
            f"EDDB returned a multi-entry unit {unit!r}; refusing to join it"
        )
    return str(unit)


class Transport(Protocol):
    """The seam a batch runs through: one call is one remote process."""

    def run(self, argv: Sequence[str], payload: bytes) -> bytes:
        """Run the remote command with ``payload`` on stdin and return stdout."""


class SshTransport:
    """The default transport: one ``ssh`` process per batch."""

    def __init__(self, *, timeout: float = 600.0) -> None:
        self.timeout = timeout

    def run(self, argv: Sequence[str], payload: bytes) -> bytes:
        expanded = [os.path.expanduser(part) for part in argv]
        try:
            completed = subprocess.run(
                expanded,
                input=payload,
                capture_output=True,
                timeout=self.timeout,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise EddbRemoteError(f"remote EDDB command failed: {error}") from error
        if completed.returncode != 0:
            message = completed.stderr.decode("utf-8", "replace").strip()
            raise EddbRemoteError(
                f"remote EDDB command exited {completed.returncode}: {message[:400]}"
            )
        return completed.stdout


def encode_batch(
    records: Iterable[ChannelRecord],
    refusals: Iterable[ChannelRefusal] = (),
) -> bytes:
    """Encode records and refusals into the envelope the remote script emits.

    Layout: the magic, a little-endian uint32 header length, the UTF-8 JSON
    header, then every channel's ``data`` followed by its ``time``, both as
    little-endian float64 in C order.  The header lists each channel in the
    same order so a reader can walk the segment offsets without a second pass,
    and carries the refused channels' return codes beside them.
    """

    ordered = list(records)
    header_channels = [
        {
            "shot": record.shot,
            "category": record.category,
            "dname": record.dname,
            "unit": record.unit,
            "nch": int(record.nch),
            "seq": int(record.seq),
            "shape": [int(n) for n in record.data.shape],
            "time_shape": [int(n) for n in record.time.shape],
        }
        for record in ordered
    ]
    header_refusals = [
        {
            "shot": refusal.shot,
            "category": refusal.category,
            "dname": refusal.dname,
            "code": int(refusal.code),
        }
        for refusal in refusals
    ]
    header = json.dumps(
        {"channels": header_channels, "refusals": header_refusals}
    ).encode("utf-8")
    segments = bytearray()
    for record in ordered:
        segments += np.ascontiguousarray(record.data, dtype="<f8").tobytes()
        segments += np.ascontiguousarray(record.time, dtype="<f8").tobytes()
    return _MAGIC + _STRUCT.pack(len(header)) + header + bytes(segments)


def decode_batch(payload: bytes) -> BatchResult:
    """Decode an envelope produced by :func:`encode_batch` or the remote script."""

    if not payload.startswith(_MAGIC):
        raise EddbRemoteError("response is not an EDDB batch envelope")
    offset = len(_MAGIC)
    (header_length,) = _STRUCT.unpack_from(payload, offset)
    offset += _STRUCT.size
    try:
        header = json.loads(payload[offset : offset + header_length].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EddbRemoteError(f"batch header is unreadable: {error}") from error
    cursor = offset + header_length
    records: list[ChannelRecord] = []
    for entry in header.get("channels", []):
        shape = tuple(int(n) for n in entry["shape"])
        time_shape = tuple(int(n) for n in entry["time_shape"])
        data_count = int(np.prod(shape)) if shape else 1
        time_count = int(np.prod(time_shape)) if time_shape else 1
        data = np.frombuffer(
            payload, dtype="<f8", count=data_count, offset=cursor
        ).reshape(shape)
        cursor += data_count * 8
        time = np.frombuffer(
            payload, dtype="<f8", count=time_count, offset=cursor
        ).reshape(time_shape)
        cursor += time_count * 8
        records.append(
            ChannelRecord(
                shot=str(entry["shot"]),
                category=str(entry["category"]),
                dname=str(entry["dname"]),
                data=data,
                time=time,
                unit=normalise_unit(entry.get("unit", "")),
                nch=int(entry.get("nch", data.shape[0] if data.ndim else 1)),
                seq=int(entry.get("seq", 0)),
            )
        )
    refusals = [
        ChannelRefusal(
            shot=str(entry["shot"]),
            category=str(entry["category"]),
            dname=str(entry["dname"]),
            code=int(entry["code"]),
        )
        for entry in header.get("refusals", [])
    ]
    return BatchResult(records=records, refusals=refusals)


class RemoteEddbExtractor:
    """Read a batch of EDDB channels over one remote process per batch."""

    def __init__(
        self,
        *,
        ssh_command: Sequence[str] = DEFAULT_SSH_COMMAND,
        remote_python: str = "python",
        api_path: str = "/analysis/src/eddb",
        lib_path: str = "/analysis/lib/libeddb.so",
        nice: bool = True,
        transport: Transport | None = None,
    ) -> None:
        self.ssh_command = tuple(ssh_command)
        self.remote_python = remote_python
        self.api_path = api_path
        self.lib_path = lib_path
        self.nice = nice
        self.transport: Transport = (
            transport if transport is not None else SshTransport()
        )

    def _remote_shell_command(self) -> str:
        """Build the one remote shell command: module select, then the script.

        The system python on the analysis server is too old and lacks numpy, so
        the command loads the 3.12 module and runs that module's ``python``.
        The script is quoted as one argument to ``python -c`` and the request
        JSON still arrives on stdin.
        """
        python_call = f"{self.remote_python} -c {shlex.quote(REMOTE_SCRIPT)}"
        if self.nice:
            python_call = f"nice -n {NICE_LEVEL} {python_call}"
        return (
            f"module unload {PYTHON_MODULE_UNLOAD}; "
            f"module load {PYTHON_MODULE_LOAD}; "
            f"{python_call}"
        )

    def _argv(self) -> list[str]:
        # ssh -F does not expand ~ itself, so the config path is expanded here,
        # at call time, to an absolute path.
        prefix = [os.path.expanduser(part) for part in self.ssh_command]
        return [*prefix, self._remote_shell_command()]

    def fetch_batch(self, requests: Iterable[ChannelRequest]) -> BatchResult:
        """Read every request in one remote process and decode the result.

        The returned :class:`BatchResult` carries the served records and one
        refusal per refused channel, each naming the shot, category, dname and
        the EDDB return code, so a partial batch is reported rather than
        raising.  The transport raises only when the remote process itself
        fails; a channel the remote reported neither as served nor refused is a
        protocol failure of the session and raises here.
        """

        ordered = list(requests)
        if not ordered:
            return BatchResult(records=[], refusals=[])
        payload = json.dumps(
            {
                "api_path": self.api_path,
                "lib_path": self.lib_path,
                "requests": [
                    {
                        "shot": request.shot,
                        "category": request.category,
                        "dname": request.dname,
                    }
                    for request in ordered
                ],
            }
        ).encode("utf-8")
        response = self.transport.run(self._argv(), payload)
        result = decode_batch(response)
        answered = {r.key for r in result.records} | {r.key for r in result.refusals}
        missing = {request.key for request in ordered} - answered
        if missing:
            raise EddbRemoteError(
                f"remote batch reported neither a record nor a refusal for "
                f"{sorted(missing)!r}"
            )
        return result


# The script is executed by ``python3 -c`` on the analysis server, so it can
# only use the standard library, numpy and eddb_pwrapper.  It mirrors the
# envelope in :func:`encode_batch`: the same magic, the same uint32 header
# length, the same little-endian float64 segments, so the client decodes both
# identically.
REMOTE_SCRIPT = r'''
import json, struct, sys
import numpy as np

MAGIC = b"EDDB1\n"
HEADER = struct.Struct("<I")
TIME_BOUNDS = ("0", "99")


def _unit(value):
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return str(value[0])
        raise RuntimeError(
            "EDDB returned a multi-entry unit %r; refusing to join it" % (value,)
        )
    return str(value)


def _return_code(value):
    if isinstance(value, bool):
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    return 0


def _read_one(db, req):
    shot, cat, dname = req["shot"], req["category"], req["dname"]
    data_class = req.get("data_class", "")
    ok, rtn = True, None
    if data_class == "O":
        ok, rtn = db.eddbreadOne(shot, cat, dname, None, 0, 0)
    if rtn is None:
        ok, rtn = db.eddbreadTime(shot, cat, dname, TIME_BOUNDS[0], TIME_BOUNDS[1])
    if not ok or not rtn:
        return None, {
            "shot": shot,
            "category": cat,
            "dname": dname,
            "code": _return_code(rtn),
        }
    values = np.asarray(rtn.get("data"), dtype="<f8")
    if values.ndim == 1:
        values = values.reshape(1, -1)
    time = rtn.get("time")
    if time is None:
        raise RuntimeError(
            "EDDB returned no time base for shot=%s category=%s dname=%s"
            % (shot, cat, dname)
        )
    time = np.asarray(time, dtype="<f8").reshape(-1)
    unit = _unit(rtn.get("unit") or rtn.get("units"))
    return {
        "shot": shot,
        "category": cat,
        "dname": dname,
        "unit": unit,
        "nch": int(values.shape[0]),
        "seq": int(rtn.get("seq") or 0),
        "data": values,
        "time": time,
    }, None


def main():
    config = json.load(sys.stdin)
    sys.path.insert(0, config.get("api_path", ""))
    from eddb_pwrapper import eddbWrapper

    db = eddbWrapper(config["lib_path"])
    if not db.eddbOpen():
        raise SystemExit("eddbOpen() failed")

    header_channels = []
    header_refusals = []
    segments = bytearray()
    try:
        for req in config.get("requests", []):
            record, refusal = _read_one(db, req)
            if refusal is not None:
                header_refusals.append(refusal)
                continue
            header_channels.append({
                "shot": record["shot"],
                "category": record["category"],
                "dname": record["dname"],
                "unit": record["unit"],
                "nch": int(record["nch"]),
                "seq": int(record["seq"]),
                "shape": [int(n) for n in record["data"].shape],
                "time_shape": [int(n) for n in record["time"].shape],
            })
            segments += np.ascontiguousarray(record["data"]).tobytes()
            segments += np.ascontiguousarray(record["time"]).tobytes()
    finally:
        db.eddbClose()

    header = json.dumps(
        {"channels": header_channels, "refusals": header_refusals}
    ).encode("utf-8")
    out = sys.stdout.buffer
    out.write(MAGIC)
    out.write(HEADER.pack(len(header)))
    out.write(header)
    out.write(bytes(segments))
    out.flush()


if __name__ == "__main__":
    main()
'''


__all__ = [
    "DEFAULT_SSH_COMMAND",
    "REMOTE_SCRIPT",
    "BatchResult",
    "ChannelRecord",
    "ChannelRefusal",
    "ChannelRequest",
    "EddbRemoteError",
    "RemoteEddbExtractor",
    "SshTransport",
    "Transport",
    "decode_batch",
    "encode_batch",
    "normalise_unit",
]
