"""Publish the watch document atomically for the ingest service or manual use.

``agent watch --json`` derives its figures from the recorded receipts every
time it runs, and on the live index that derivation is expensive. A consumer on
another host cannot pay that on every refresh, so ingest runs this derivation
on its publication cadence and writes the result for readers to pick up.

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
its ``published_at`` still names the last success. Each iteration returns one
log line, whether it succeeds or fails.
"""

from __future__ import annotations

import contextlib
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

#: Where the document is published by default. It sits beside the lane document
#: in ``~/public/imas-ambix`` so a consumer reads both from one directory.
DEFAULT_OUTPUT = watch_mod.PUBLISHED_WATCH_PATH


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

    ``mkstemp`` creates the temporary with mode 0600, which the rename would
    carry onto the target and leave the published document unreadable to anyone
    but its owner. The document sits beside ``lane.json`` for other users to
    read, so the temporary is given the conventional 0644 before it is renamed.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=str(target.parent), prefix=target.name + ".", suffix=".tmp"
    )
    try:
        os.fchmod(descriptor, 0o644)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
        os.replace(temporary, target)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(temporary)
        raise


def publish_once(
    output: str | Path,
    *,
    record_dir: str | Path | None = None,
    index_path: str | Path | None = None,
) -> tuple[float, int, str]:
    """Compute one document, add the two fields, and publish it atomically.

    Returns compute seconds, encoded bytes and publication time. These are
    prepared before the replace, so logging needs no fallible file read after
    the new document takes the old one's place. Derivation and write errors
    are turned into one failure line by :func:`one_iteration`.
    """
    target = Path(output).expanduser()
    started = time.monotonic()
    document = watch_mod.watch_document(record_dir=record_dir, index_path=index_path)
    compute_seconds = time.monotonic() - started
    document["compute_seconds"] = compute_seconds
    published_at = _stamp(_utcnow())
    document["published_at"] = published_at
    payload = json.dumps(document, indent=2)
    size = len(payload.encode("utf-8"))
    _write_atomic(target, payload)
    return compute_seconds, size, published_at


def one_iteration(
    output: str | Path,
    *,
    record_dir: str | Path | None = None,
    index_path: str | Path | None = None,
    on_success: Callable[[], None] | None = None,
) -> str:
    """Run one publish, returning the single line to print.

    Success and failure both return a line; the caller prints it and flushes.
    *on_success* lets the caller advance its cadence only after a write succeeds.
    """
    target = Path(output).expanduser()
    try:
        compute_seconds, size, published_at = publish_once(
            target, record_dir=record_dir, index_path=index_path
        )
    except Exception as error:  # noqa: BLE001 - one line per iteration, never a crash
        return _one_line(f"{_stamp(_utcnow())} publish-watch failed: {error}")
    if on_success is not None:
        on_success()
    return _one_line(
        f"{published_at} publish-watch wrote {target} "
        f"compute_seconds={compute_seconds:.2f} bytes={size}"
    )
