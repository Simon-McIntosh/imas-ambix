"""ambix-watch: the GPU servers and the lanes they serve, one page at a time.

Pages::

    overview     every server, every lane, the period ledger
    <server>     one server: cards, lanes served from it, queue
    lanes        the serving detail ``llm-watch`` draws

In ``--live`` the keys switch page without restarting: ``0`` is the overview,
``1``-``9`` a server in the order the overview lists them, ``l`` the lanes
detail, the arrow keys step through all of them, and ``q`` quits. With one
server the view opens on that server, and its page carries the ledger; with
more it opens on the overview.

Usage::

    ambix-watch --live              # 1 s refresh, 30 s rolling mean
    ambix-watch 10 --live           # 10 s rolling mean
    ambix-watch --live --page gpu-0003
    ambix-watch                     # one reading after a 30 s window
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
from imas_ambix.agent.gpu_watch.lane import Lane, lanes

OVERVIEW = "overview"
LANES = "lanes"

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


def page_names(readings: list[Reading]) -> list[str]:
    """Every page, in the order the keys step through them."""
    return [OVERVIEW, *(reading.server.name for reading in readings), LANES]


def default_page(readings: list[Reading]) -> str:
    return readings[0].server.name if len(readings) == 1 else OVERVIEW


def resolve_page(name: str | None, readings: list[Reading]) -> str:
    """The page a name selects: exact, then any page the name is part of."""
    names = page_names(readings)
    if not name:
        return default_page(readings)
    wanted = name.lower()
    for candidate in names:
        if candidate.lower() == wanted:
            return candidate
    for candidate in names:
        if wanted in candidate.lower():
            return candidate
    return default_page(readings)


def footer(page: str, readings: list[Reading]) -> str:
    """The keys, with the page on screen in bold."""

    def mark(key: str, name: str) -> str:
        text = f"{key} {name}"
        return f"[bold]{text}[/]" if name == page else f"[dim]{text}[/]"

    bits = [mark("0", OVERVIEW)]
    bits += [mark(str(i), r.server.name) for i, r in enumerate(readings[:9], 1)]
    bits += [mark("l", LANES), "[dim]q quit[/]"]
    return " [dim]·[/] ".join(bits)


def render(
    page: str,
    readings: list[Reading],
    document: dict | None,
    window: int,
    size: tuple[int, int] | None = None,
    keys: bool = False,
    reason: str = "",
) -> Group:
    """One page from the readings and the serving document.

    ``reason`` is why there is no document, shown where the lanes would be.
    """
    served: list[Lane] = lanes(document)
    ledger = (document or {}).get("ledger")
    published_at = (document or {}).get("ledger_published_at")
    foot = footer(page, readings) if keys else ""
    if page == LANES:
        from imas_ambix.agent import llm_watch

        body: list[object] = (
            [llm_watch.render(document)] if document else [panels.no_document(reason)]
        )
        if foot:
            body += [panels.rule(), Text.from_markup(foot)]
        return Group(*body)
    for reading in readings:
        if reading.server.name != page:
            continue
        here = [lane for lane in served if lane.server == reading.server.name]
        alone = len(readings) == 1
        return panels.server_page(
            reading,
            here,
            window,
            ledger=ledger if alone else None,
            published_at=published_at,
            size=size,
            footer=foot,
        )
    return panels.overview_page(
        readings,
        served,
        ledger,
        published_at,
        foot,
        reason="" if document else reason,
    )


def step(page: str, key: str, readings: list[Reading]) -> str:
    """The page a key press leads to; the same page for a key that means nothing."""
    names = page_names(readings)
    if key == "0":
        return OVERVIEW
    if key in ("l", "L"):
        return LANES
    if key.isdigit() and 1 <= int(key) <= len(readings):
        return readings[int(key) - 1].server.name
    if key in ("right", "left") and page in names:
        move = 1 if key == "right" else -1
        return names[(names.index(page) + move) % len(names)]
    return page


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
        if data.startswith("\x1b["):
            return {"C": "right", "D": "left"}.get(data[2:3])
        return data[:1] or None


def run_live(window: int, page: str | None) -> int:
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
            current: str | None = None
            while True:
                began = time.monotonic()
                readings = fleet.tick()
                if current is None or current not in page_names(readings):
                    current = resolve_page(page, readings)
                live.update(
                    render(
                        current,
                        readings,
                        document.get(),
                        window,
                        console.size,
                        keys=True,
                        reason=document.reason,
                    ),
                    refresh=True,
                )
                while (left := 1.0 - (time.monotonic() - began)) > 0:
                    key = keys.read(left)
                    if key in ("q", "Q"):
                        return 0
                    if key:
                        moved = step(current, key, readings)
                        if moved != current:
                            current = moved
                            live.update(
                                render(
                                    current,
                                    readings,
                                    document.get(),
                                    window,
                                    console.size,
                                    keys=True,
                                    reason=document.reason,
                                ),
                                refresh=True,
                            )
    finally:
        document.stop()
        fleet.stop()


def run_once(window: int, page: str | None) -> int:
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
        Console().print(
            render(
                resolve_page(page, readings),
                readings,
                document.get(),
                window,
                reason=document.reason,
            )
        )
    finally:
        document.stop()
        fleet.stop()
    return 0


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
        help="refresh every second; the mean columns roll over the window",
    )
    parser.add_argument(
        "--page",
        help="page to open: overview, lanes, or a server name such as gpu-0003",
    )
    args = parser.parse_args(argv)
    # A terminal signal must end the samplers too, and Python does not unwind
    # for one by default.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    try:
        if args.live:
            return run_live(args.window, args.page)
        return run_once(args.window, args.page)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
