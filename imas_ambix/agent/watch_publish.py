"""``imas-ambix agent publish-watch`` — publish the watch document on a cadence.

``agent watch --json`` derives its figures from the recorded receipts every
time it runs, and on the live index that derivation is expensive: tens of
seconds and gigabytes of resident memory for the long periods. A consumer on
another host cannot pay that on every refresh, so this command runs the
derivation once per cadence and writes the result where a reader can pick it up
without re-deriving it.

Each iteration calls :func:`imas_ambix.agent.watch.watch_document` in-process,
adds two top-level fields and writes the document atomically:

* ``compute_seconds`` — the wall time of that call, a float.
* ``published_at`` — the UTC time, to the second, taken after the call returns
  and immediately before the temporary file is written.

The two fields exist only in the published file; ``agent watch --json`` itself
is unchanged.

**A failed iteration changes nothing on disk.** The document is written to a
temporary file in the target's directory and renamed over the target with
:func:`os.replace`, so a reader never sees a half-written file and a failure
before the rename leaves the previous publication byte-for-byte unchanged --
its ``published_at`` still names the last success. The loop prints exactly one
flushed line per iteration, success or failure, and continues to the next.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING

from imas_ambix.agent import watch as watch_mod

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import TextIO

#: Where the document is published by default. It sits beside the lane document
#: in ``~/public/imas-ambix`` so a consumer reads both from one directory.
DEFAULT_OUTPUT = Path.home() / "public" / "imas-ambix" / "watch.json"

#: The default seconds between iterations. The derivation is expensive, so a
#: refresh every five minutes keeps the file fresh without crowding the node.
DEFAULT_CADENCE = 300.0


def _utcnow() -> _dt.datetime:
    """The UTC wall clock, so an iteration's stamp is testable through one seam."""
    return _dt.datetime.now(_dt.UTC)


def _stamp(when: _dt.datetime) -> str:
    """A UTC instant as ISO-8601 to the second, the way the panels print it."""
    return when.astimezone(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _one_line(text: str) -> str:
    """One physical line: every newline in *text* becomes a space.

    An error message from a driver or another process routinely carries its own
    newlines, and a per-iteration line that spilled across several physical
    lines would break the one-line-per-iteration contract a log reader counts on.
    """
    return text.replace("\r", " ").replace("\n", " ")


def _write_atomic(target: Path, payload: str) -> None:
    """Write *payload* to *target* via a temporary file and a rename.

    The temporary file lives in the target's own directory so the rename stays
    within one filesystem and is atomic; the target is touched only by
    :func:`os.replace`, so a failure before that point leaves any existing file
    exactly as it was.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=str(target.parent), prefix=target.name + ".", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def publish_once(
    output: str | Path,
    *,
    record_dir: str | Path | None = None,
    index_path: str | Path | None = None,
) -> float:
    """Compute one document, add the two fields, and publish it atomically.

    Returns the compute wall time in seconds. Raises whatever the derivation or
    the write raises; :func:`run` is what turns that into a printed line and a
    kept-previous-file iteration.
    """
    target = Path(output).expanduser()
    started = time.monotonic()
    document = watch_mod.watch_document(record_dir=record_dir, index_path=index_path)
    compute_seconds = time.monotonic() - started
    document["compute_seconds"] = compute_seconds
    document["published_at"] = _stamp(_utcnow())
    _write_atomic(target, json.dumps(document, indent=2))
    return compute_seconds


def one_iteration(
    output: str | Path,
    *,
    record_dir: str | Path | None = None,
    index_path: str | Path | None = None,
) -> str:
    """Run one publish, returning the single line to print.

    Success and failure both return a line; the caller prints it and flushes.
    """
    target = Path(output).expanduser()
    try:
        compute_seconds = publish_once(
            target, record_dir=record_dir, index_path=index_path
        )
    except Exception as error:  # noqa: BLE001 - one line per iteration, never a crash
        return _one_line(
            f"{_stamp(_utcnow())} publish-watch failed: {error}"
        )
    size = target.stat().st_size
    return _one_line(
        f"{_stamp(_utcnow())} publish-watch wrote {target} "
        f"compute_seconds={compute_seconds:.2f} bytes={size}"
    )


def run(
    output: str | Path = DEFAULT_OUTPUT,
    *,
    record_dir: str | Path | None = None,
    index_path: str | Path | None = None,
    cadence: float = DEFAULT_CADENCE,
    iterations: int | None = None,
    sleep: Callable[[float], None] = time.sleep,
    stream: TextIO | None = None,
) -> None:
    """Publish the watch document every *cadence* seconds.

    Each iteration computes the document, publishes it atomically and writes one
    flushed line to *stream* (``sys.stdout`` when ``None``). A finite
    *iterations* count returns without sleeping past the last iteration, so
    ``--once`` pays nothing for a cadence it did not use; with ``None`` the loop
    runs until the process ends, which is the standing service the fleet
    supervisor keeps.

    *sleep* and *stream* are injected so a caller -- a test, or a service that
    logs each pass elsewhere -- can supply its own without the loop knowing.
    """
    import sys

    if cadence <= 0:
        raise ValueError(f"cadence must be positive seconds, not {cadence!r}")
    if iterations is not None and iterations < 0:
        raise ValueError(f"iterations must not be negative, not {iterations!r}")
    out = sys.stdout if stream is None else stream
    done = 0
    while True:
        line = one_iteration(output, record_dir=record_dir, index_path=index_path)
        out.write(line + "\n")
        out.flush()
        done += 1
        if iterations is not None and done >= iterations:
            return
        sleep(cadence)