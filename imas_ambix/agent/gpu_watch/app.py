"""ambix-watch: the GPU servers and the lanes they serve, on one page.

The page is a single column: each server's cards, each lane, the period
ledger, then the queue. In a terminal the view is live and ``q`` quits.

Usage::

    ambix-watch                     # live: 1 s refresh, 30 s rolling mean
    ambix-watch 10                  # live, 10 s rolling mean
    ambix-watch --once              # one reading after a 30 s window
"""

from __future__ import annotations

import argparse
import contextlib
import select
import signal
import sys
import threading
import time

from rich.console import Console, Group
from rich.live import Live
from rich.text import Text

from imas_ambix.agent.gpu_watch import panels
from imas_ambix.agent.gpu_watch.fleet import Fleet, Reading
from imas_ambix.agent.gpu_watch.lane import lanes
from imas_ambix.agent.watch_style import page_width

#: Seconds between reads of the live serving document.
DOCUMENT_SECONDS = 5.0


class Document:
    """The live serving document, rebuilt on its own thread.

    A build reads the tail of the receipt record and of the request record,
    which on a loaded filesystem can take longer than a refresh; the page is
    drawn from the newest document already built.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._value: dict | None = None
        self._reason = ""
        self._stop = threading.Event()
        self.built = threading.Event()

    def start(self) -> None:
        threading.Thread(target=self._build_forever, daemon=True).start()

    def _build_forever(self) -> None:
        from imas_ambix.agent import llm_watch

        while not self._stop.is_set():
            value, reason = None, ""
            try:
                value = llm_watch.build_document()
            except Exception as error:
                reason = f"{type(error).__name__}: {error}"
            with self._lock:
                self._value, self._reason = value, reason
            self.built.set()
            self._stop.wait(DOCUMENT_SECONDS)

    def stop(self) -> None:
        self._stop.set()

    def get(self) -> dict | None:
        with self._lock:
            return self._value

    @property
    def reason(self) -> str:
        """Why the last build produced no document; empty when it did."""
        with self._lock:
            return self._reason


def render(
    readings: list[Reading],
    document: dict | None,
    window: int,
    size: tuple[int, int] | None = None,
    keys: bool = False,
    reason: str = "",
) -> Group:
    """The page from the readings and the serving document.

    ``reason`` is why there is no document, shown where the lanes would be.
    """
    return panels.view(
        readings,
        lanes(document),
        (document or {}).get("ledger"),
        window,
        published_at=(document or {}).get("ledger_published_at"),
        size=size,
        footer="[dim]q quit[/]" if keys else "",
        reason="" if document else reason,
        width=page_width(size[0] if size else None),
    )


class Keys:
    """Single key presses from the terminal, without waiting for a newline.

    Outside a terminal there are no keys and :meth:`read` only waits, so the
    view still runs under a pipe or a pager.
    """

    def __init__(self) -> None:
        self._fd: int | None = None
        self._saved: list | None = None

    def __enter__(self) -> Keys:
        with contextlib.suppress(Exception):
            import termios
            import tty

            if sys.stdin.isatty():
                self._fd = sys.stdin.fileno()
                self._saved = termios.tcgetattr(self._fd)
                tty.setcbreak(self._fd)
        return self

    def __exit__(self, *exc: object) -> None:
        if self._fd is not None and self._saved is not None:
            import termios

            with contextlib.suppress(Exception):
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved)

    def read(self, timeout: float) -> str | None:
        """The next key within ``timeout`` seconds, or None."""
        if self._fd is None:
            time.sleep(max(0.0, timeout))
            return None
        import os

        ready, _, _ = select.select([self._fd], [], [], max(0.0, timeout))
        if not ready:
            return None
        data = os.read(self._fd, 8).decode("utf-8", "replace")
        return None if data.startswith("\x1b") else data[:1] or None


def run_live(window: int) -> int:
    console = Console()
    fleet, document = Fleet(window), Document()
    fleet.start()
    document.start()
    try:
        with (
            Keys() as keys,
            Live(console=console, auto_refresh=False, screen=True) as live,
        ):
            live.update(Text("reading the scheduler…", style="dim"), refresh=True)
            fleet.polled.wait(30)
            while True:
                began = time.monotonic()
                live.update(
                    render(
                        fleet.tick(),
                        document.get(),
                        window,
                        console.size,
                        keys=True,
                        reason=document.reason,
                    ),
                    refresh=True,
                )
                while (left := 1.0 - (time.monotonic() - began)) > 0:
                    if keys.read(left) in ("q", "Q"):
                        return 0
    finally:
        document.stop()
        fleet.stop()


def run_once(window: int) -> int:
    """Sample for ``window`` seconds, then print one page."""
    fleet, document = Fleet(window), Document()
    fleet.start()
    document.start()
    try:
        fleet.polled.wait(30)
        readings: list[Reading] = []
        ends = time.monotonic() + window
        while True:
            readings = fleet.tick()
            if time.monotonic() >= ends:
                break
            time.sleep(1.0)
        document.built.wait(10)
        console = Console()
        console.print(
            render(
                readings,
                document.get(),
                window,
                (console.size.width, 0),
                reason=document.reason,
            )
        )
    finally:
        document.stop()
        fleet.stop()
    return 0


def live_by_default(live: bool, once: bool, terminal: bool) -> bool:
    """Whether the view keeps refreshing.

    A reader at a terminal is watching, so the view refreshes unless ``--once``
    asks for a single reading; output to a pipe or a file is one reading
    unless ``--live`` asks otherwise.
    """
    return not once and (live or terminal)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="GPU servers and the lanes they serve")
    parser.add_argument(
        "window",
        nargs="?",
        type=int,
        default=30,
        help="averaging window in seconds (default 30)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="refresh every second; the mean columns roll over the window. "
        "The default in a terminal",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="print one reading after the window and exit; the default when "
        "the output is not a terminal",
    )
    args = parser.parse_args(argv)
    # A terminal signal must end the samplers too, and Python does not unwind
    # for one by default.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    try:
        if live_by_default(args.live, args.once, sys.stdout.isatty()):
            return run_live(args.window)
        return run_once(args.window)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
