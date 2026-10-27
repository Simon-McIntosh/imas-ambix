"""The gpu-watch view: each declared server's machine plus the engines on it.

gpu-watch renders one block per server declared in the published endpoint
document: the card table, the host row, a conditions line, the node's SLURM job
table, and one dim identity row per engine placed on that server. The engines'
own live figures belong to llm-watch, so the row carries identity -- model,
cards, port -- and says where the rest is drawn.

**The readings come from the node probe, not from a private sampler.** Cards,
host and the job table are :meth:`imas_ambix.agent.node_probe.NodeProbe.sample`'s
three sections, the same reader the recorder and the retention pass use, so one
place owns how a card's physical index is resolved and one place decides what an
unreadable job table means. ``NodeProbe.stream`` is the entry point a long-lived
on-node step runs; a one-shot run here takes a bounded window of the same
streamed samples, and ``--live`` holds a rolling window over it.

**A block with nothing to show says so.** A declared server with no running job
leaves a card query with no allocation to see through, and an empty card table
would read as a node with no cards rather than as a server nothing is serving
on -- so the block prints :data:`NO_JOB` instead. A server this host cannot
sample says that, rather than drawing the local node under another server's
name.

**One style module.** The column widths, the percentage formatter and the dash
come from :mod:`imas_ambix.agent.watch_style`, which the llm-watch view reads as
well, so one edit restyles both tools and the same figure takes the same colour
on either.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from rich import box
from rich.console import Console, Group
from rich.table import Table
from rich.text import Text

from imas_ambix.agent import node_probe
from imas_ambix.agent.registry import _short_host
from imas_ambix.agent.watch_style import DASH, WIDTHS, fmt_count, pct

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence

#: The window a one-shot run averages over, in seconds, and how often that
#: window is sampled. The window is the whole reason the mean columns exist: a
#: single ``nvidia-smi`` snapshot swings between idle and full each second.
DEFAULT_WINDOW_S = 30.0
SAMPLE_INTERVAL_S = 1.0

#: What a block says when its server holds no running job.
NO_JOB = "no job to sample"

#: What a block says when this host cannot read the server it declares. Drawing
#: the local node under another server's name would be a false reading.
NO_READING = "not sampled from this host"

#: What a block says when the job table reports work but the card query saw no
#: card: this process's device cgroup holds a different allocation.
CARDS_HIDDEN = "no cards visible from this process"

#: The page width a one-shot run renders at when the caller names none.
DEFAULT_WIDTH = WIDTHS[-1]


@dataclass(frozen=True)
class Reading:
    """One server's samples, oldest first, exactly as the probe reported them.

    A sample is one :meth:`NodeProbe.sample` mapping, so a section's absence
    means that read did not answer and is never drawn as a zero.

    *unavailable* names why a step that was opened produced nothing. It is set
    only when a sampler ran and answered no sample, so a server whose step
    failed is told apart from one that is simply not serving, which says
    :data:`NO_JOB`.
    """

    samples: tuple[dict[str, Any], ...] = ()
    unavailable: str | None = None

    @property
    def latest(self) -> dict[str, Any]:
        return self.samples[-1] if self.samples else {}

    def section(self, name: str) -> dict[str, Any] | None:
        """The newest sample that carried *name*, or ``None`` when none did."""
        for sample in reversed(self.samples):
            section = sample.get(name)
            if isinstance(section, dict):
                return section
        return None

    @property
    def cards(self) -> list[dict[str, Any]]:
        section = self.section("cards") or {}
        cards = section.get("cards")
        return list(cards or [])

    @property
    def jobs(self) -> dict[str, Any] | None:
        return self.section("jobs")

    @property
    def host(self) -> dict[str, Any] | None:
        return self.section("host")


def _mean(values: Iterable[float]) -> float | None:
    numbers = list(values)
    return round(sum(numbers) / len(numbers), 1) if numbers else None


def _mem_percent(card: Mapping[str, Any]) -> float | None:
    used = card.get("memory_used_mib")
    total = card.get("memory_total_mib")
    if isinstance(used, (int, float)) and isinstance(total, (int, float)) and total:
        return round(used / total * 100.0, 1)
    return None


def card_figures(reading: Reading) -> list[dict[str, Any]]:
    """One row per physical card, carrying its now and window-mean figure.

    Cards are paired by the index the probe resolved -- the physical silicon --
    so two samples of one allocation land on one row, and a card only one sample
    saw keeps the samples it has. A quantity no sample carried stays ``None``
    and renders as a dash.
    """
    seen: dict[Any, list[dict[str, Any]]] = {}
    for sample in reading.samples:
        section = sample.get("cards") or {}
        for card in section.get("cards") or []:
            seen.setdefault(card.get("index"), []).append(card)
    rows: list[dict[str, Any]] = []
    for index in sorted(seen, key=str):
        cards = seen[index]
        latest = cards[-1]
        rows.append(
            {
                "index": index,
                "name": latest.get("name"),
                "util_now": latest.get("utilisation_percent"),
                "util_mean": _mean(
                    card["utilisation_percent"]
                    for card in cards
                    if isinstance(card.get("utilisation_percent"), (int, float))
                ),
                "mem_now": _mem_percent(latest),
                "mem_now_mib": latest.get("memory_used_mib"),
                "mem_mean": _mean(
                    value
                    for value in (_mem_percent(card) for card in cards)
                    if value is not None
                ),
            }
        )
    return rows


def conditions(reading: Reading, declared_cap_watts: int | None) -> dict[str, Any]:
    """Temperature, draw and share of the cap, over the whole server.

    The cap is the server's declaration when it makes one, because that is the
    figure the operator sizes against and it is published where every other
    consumer reads it. The declaration is in watts while this section reports
    kilowatts, so it is scaled here -- a declared ``700`` is a 0.70 kW cap.
    Without one, the cards' own caps are summed, already divided to kilowatts;
    a node where neither is readable reports no percentage rather than a rate
    against a cap nobody stated.
    """
    cards = reading.cards
    temperatures = [
        card["temperature_c"]
        for card in cards
        if isinstance(card.get("temperature_c"), (int, float))
    ]
    draws = [
        card["power_draw_w"]
        for card in cards
        if isinstance(card.get("power_draw_w"), (int, float))
    ]
    # The declaration states watts (``SiteServerConfig.power_cap_watts``) and
    # the draw beside it is kilowatts, so the two are only comparable after the
    # declaration is scaled to the unit the line prints.
    cap = declared_cap_watts / 1000.0 if declared_cap_watts else None
    if cap is None:
        card_caps = [
            card["power_cap_w"]
            for card in cards
            if isinstance(card.get("power_cap_w"), (int, float))
        ]
        cap = sum(card_caps) / 1000.0 if card_caps else None
    kilowatts = round(sum(draws) / 1000.0, 2) if draws else None
    share = (
        round(kilowatts / cap * 100.0, 1) if (kilowatts is not None and cap) else None
    )
    return {
        "temp_mean": _mean(temperatures),
        "temp_max": max(temperatures) if temperatures else None,
        "kilowatts": kilowatts,
        "cap_kilowatts": cap,
        "cap_percent": share,
    }


def conditions_line(reading: Reading, declared_cap_watts: int | None) -> Text:
    """The conditions the serve grid used to carry, as one line."""
    figures = conditions(reading, declared_cap_watts)
    temp_mean = (
        DASH if figures["temp_mean"] is None else f"{figures['temp_mean']:.1f}°C"
    )
    temp_max = DASH if figures["temp_max"] is None else f"{figures['temp_max']:.0f}°C"
    kw = DASH if figures["kilowatts"] is None else f"{figures['kilowatts']:.2f} kW"
    cap = (
        DASH
        if figures["cap_kilowatts"] is None
        else f"{figures['cap_kilowatts']:.2f} kW cap"
    )
    share = (
        DASH
        if figures["cap_percent"] is None
        else f"{figures['cap_percent']:.0f}% of cap"
    )
    return Text.assemble(
        ("conditions  ", "bold"),
        ("temperature ", "dim"),
        (temp_mean, "bold"),
        (" mean / ", "dim"),
        (temp_max, "bold"),
        (" max    ", "dim"),
        ("power ", "dim"),
        (kw, "dim"),
        (" / ", "dim"),
        (cap, "dim"),
        ("    ", "dim"),
        (share, "bold"),
    )


def _host_line(reading: Reading) -> Text:
    host = reading.host or {}
    busy = host.get("cpu_busy_fraction")
    used = host.get("memory_used_mib")
    percent = host.get("memory_used_percent")
    cpu = DASH if not isinstance(busy, (int, float)) else f"{busy * 100:.0f}%"
    memory = DASH
    if isinstance(used, (int, float)):
        memory = f"{used / 1024.0:.0f} GiB"
        if isinstance(percent, (int, float)):
            memory = f"{memory} ({percent:.0f}%)"
    return Text.assemble(
        ("host        ", "bold"),
        ("cpu ", "dim"),
        (cpu, "bold"),
        ("    ", "dim"),
        ("memory ", "dim"),
        (memory, "bold"),
    )


def _card_table(rows: Sequence[Mapping[str, Any]]) -> Table:
    """The card table: utilisation and memory, now and over the window.

    A cell whose quantity no sample carried renders as a dash, so a dash on one
    row means that card exposed nothing rather than that the node was idle.
    """
    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False, expand=False)
    table.add_column("card", justify="right", style="bold")
    table.add_column("util now", justify="right", no_wrap=True)
    table.add_column("util mean", justify="right", no_wrap=True)
    table.add_column("mem now", justify="right", no_wrap=True)
    table.add_column("mem mean", justify="right", no_wrap=True)
    for row in rows:
        name = row.get("name")
        label = f"{row['index']}"
        if isinstance(name, str) and name:
            label = f"{label} {name}"
        table.add_row(
            label,
            pct(row.get("util_now"), width=4),
            pct(row.get("util_mean"), width=4),
            pct(row.get("mem_now"), width=4),
            pct(row.get("mem_mean"), width=4),
        )
    return table


def _job_table(jobs: Mapping[str, Any]) -> Group:
    """The node's SLURM job table, or a line saying why there is none.

    An unread table is a reading about the node, so it renders as the marker the
    probe section carries rather than as an empty table.
    """
    unread = jobs.get(node_probe.UNREAD_KEY)
    if unread:
        return Group(Text(f"jobs        {unread}", style="dim"))
    rows = jobs.get("jobs") or []
    if not rows:
        return Group(Text("jobs        none resident", style="dim"))
    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False, expand=False)
    for column in ("job", "user", "name", "state", "cpus", "gres", "elapsed"):
        table.add_column(
            column, justify="right" if column in ("job", "cpus") else "left"
        )
    for job in rows:
        table.add_row(
            str(job.get("job_id", DASH)),
            str(job.get("user", DASH)),
            str(job.get("name", DASH)),
            str(job.get("state", DASH)),
            fmt_count(job.get("cpus")),
            str(job.get("gres", DASH)),
            str(job.get("elapsed", DASH)),
        )
    return Group(Text("jobs", style="bold"), table)


def _entry_for(
    engine: str, endpoints: Sequence[Mapping[str, Any]]
) -> Mapping[str, Any]:
    """The endpoint entry describing *engine*, or an empty mapping.

    The engine id is ``<host short name>:<port>`` in the published document, so
    that is what the entry is matched on -- not the server's host spelling,
    which may carry the domain or omit it.
    """
    wanted = node_probe.scheduler_node_name(engine.rpartition(":")[0])
    for entry in endpoints:
        host = entry.get("host")
        port = entry.get("port")
        if not isinstance(host, str) or not isinstance(port, int):
            continue
        if node_probe.scheduler_node_name(_short_host(host)) == wanted:
            return entry
    return {}


def engine_rows(
    engines: Sequence[str], endpoints: Sequence[Mapping[str, Any]]
) -> list[Text]:
    """One identity row per engine: model, cards and port, and where the rest is.

    Identity only. An engine's live figures are llm-watch's, and drawing them in
    two places would give an operator two sources for one number.
    """
    rows: list[Text] = []
    for engine in engines:
        entry = _entry_for(engine, endpoints)
        model = entry.get("model_id") or DASH
        count = entry.get("accelerator_count")
        family = entry.get("accelerator_family") or DASH
        cards = f"{count}×{family}" if isinstance(count, int) else family
        port = entry.get("port", DASH)
        rows.append(
            Text.assemble(
                ("  engine ", "dim"),
                (engine, "dim bold"),
                ("  model ", "dim"),
                (str(model), "dim"),
                ("  cards ", "dim"),
                (cards, "dim"),
                ("  port ", "dim"),
                (fmt_count(port), "dim"),
                ("  → llm-watch", "dim italic"),
            )
        )
    return rows


def _server_header(server: Mapping[str, Any]) -> Text:
    """The block's first line: the server's identity, as it declares itself."""
    name = server.get("name") or server.get("host") or DASH
    family = server.get("accelerator_family")
    count = server.get("accelerator_count")
    detail = f"{count}×{family}" if isinstance(count, int) and family else ""
    groups = [str(group) for group in server.get("reservation_groups") or []]
    label = ", ".join(groups) if groups else DASH
    return Text.assemble(
        (str(name), "bold"),
        (f"  {detail}" if detail else "", "dim"),
        ("    groups ", "dim"),
        (label, "dim"),
    )


def _running_jobs(reading: Reading) -> bool:
    jobs = reading.jobs or {}
    for job in jobs.get("jobs") or []:
        if str(job.get("state", "")).upper().startswith("RUN"):
            return True
    return False


def _no_card_line(reading: Reading) -> Text:
    """What a block with no card reading says instead of an empty table."""
    jobs = reading.jobs
    if jobs is not None and jobs.get(node_probe.UNREAD_KEY):
        return Text(f"cards       {DASH}  (job table unread)", style="dim")
    if _running_jobs(reading):
        return Text(f"cards       {CARDS_HIDDEN}", style="dim")
    return Text(NO_JOB, style="bold yellow")


def server_block(
    server: Mapping[str, Any],
    reading: Reading | None,
    endpoints: Sequence[Mapping[str, Any]],
) -> Group:
    """One server's whole block: header, machine readings, then its engines."""
    parts: list[object] = [_server_header(server)]
    engines = [str(engine) for engine in server.get("engines") or []]
    if reading is None:
        parts.append(Text(NO_READING, style="dim"))
    elif reading.unavailable and not reading.samples:
        # A step was opened into a running job and answered nothing. Saying so
        # keeps a failed sample from reading as a server with no cards, and it
        # names the server it happened on.
        parts.append(Text(f"not sampled: {reading.unavailable}", style="dim"))
    else:
        declared_cap = server.get("power_cap_watts")
        cap = declared_cap if isinstance(declared_cap, int) else None
        if reading.cards:
            parts.append(conditions_line(reading, cap))
            parts.append(_card_table(card_figures(reading)))
            parts.append(_host_line(reading))
        else:
            parts.append(_no_card_line(reading))
        jobs = reading.jobs
        if jobs is not None:
            parts.append(_job_table(jobs))
    if engines:
        parts.append(Text("engines", style="bold"))
        parts.extend(engine_rows(engines, endpoints))
    else:
        parts.append(Text("engines     none declared", style="dim"))
    return Group(*parts)


def servers_of(document: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Every server the document declares, in declaration order."""
    return [
        server for server in document.get("servers") or [] if isinstance(server, dict)
    ]


def render(
    document: Mapping[str, Any],
    readings: Mapping[str, Reading] | None = None,
    *,
    server: str | None = None,
) -> Group:
    """Every selected server's block, stacked; ``server`` limits it to one."""
    declared = servers_of(document)
    chosen = [
        entry
        for entry in declared
        if server is None or server in (entry.get("name"), entry.get("host"))
    ]
    if not chosen:
        return Group(Text(f"gpu-watch: no server named {server!r}", style="bold red"))
    endpoints = [
        entry for entry in document.get("endpoints") or [] if isinstance(entry, dict)
    ]
    parts: list[object] = []
    for entry in chosen:
        name = str(entry.get("name") or entry.get("host") or "")
        reading = (readings or {}).get(name)
        if parts:
            parts.append(Text(""))
        parts.append(server_block(entry, reading, endpoints))
    return Group(*parts)


def render_text(
    document: Mapping[str, Any],
    readings: Mapping[str, Reading] | None = None,
    *,
    server: str | None = None,
    width: int = DEFAULT_WIDTH,
) -> str:
    """The view as plain text at *width*, so the output is comparable."""
    import io

    buffer = io.StringIO()
    Console(file=buffer, no_color=True, width=width, highlight=False).print(
        render(document, readings, server=server), soft_wrap=False
    )
    return buffer.getvalue()


def _is_local(server: Mapping[str, Any], hostname: str) -> bool:
    """Whether *server* names the node this process is running on."""
    host = server.get("host") or server.get("name")
    if not isinstance(host, str):
        return False
    local = node_probe.scheduler_node_name(_short_host(hostname))
    return node_probe.scheduler_node_name(_short_host(host)) == local


def _server_name(server: Mapping[str, Any]) -> str:
    return str(server.get("name") or server.get("host") or "")


def _machine(server: Mapping[str, Any]) -> str:
    """The scheduler name of the machine a declared server is on.

    Two declarations may name one machine with different spellings; they are
    the same cards, and sampling them twice would date the same silicon twice
    and could only disagree with itself. The name the scheduler accepts is kept
    whole, so it is also the key a sampler pool is held under.
    """
    host = server.get("host") or server.get("name") or ""
    return node_probe.scheduler_node_name(str(host))


def _no_job_reading(server: Mapping[str, Any]) -> Reading:
    """A reading carrying an empty job table, which renders :data:`NO_JOB`.

    A declared server with no running job holds nothing to attach to, and an
    empty card table would read as a node with no cards rather than as a
    server nothing is serving on.
    """
    return Reading(
        (
            {
                "jobs": {
                    "hostname": str(server.get("host") or server.get("name") or ""),
                    "count": 0,
                    "jobs": [],
                }
            },
        )
    )


def sample_readings(
    document: Mapping[str, Any],
    probe: node_probe.NodeProbe | None = None,
    *,
    window_s: float = DEFAULT_WINDOW_S,
    interval_s: float = SAMPLE_INTERVAL_S,
    hostname: str | None = None,
    sleep: Callable[[float], None] | None = time.sleep,
    remote: Any = None,
) -> dict[str, Reading]:
    """A bounded window for every declared server, in-process or over a step.

    A server on this host is sampled in-process, as it always was: the one
    window serves every local declaration because they name the same machine.
    A server elsewhere is sampled through its own running job, by one step per
    machine (:mod:`imas_ambix.agent.gpu_watch.launcher`), so a watcher on a
    login or fleet node reads the cards the allocation actually holds.

    *remote* is the sampler the elsewhere-servers are read with, and defaults
    to :class:`~imas_ambix.agent.gpu_watch.launcher.RemoteSampler`. It is a
    parameter so a test drives the view without a scheduler. *sleep* is ``None``
    where the caller's probe owns its own cadence.
    """
    from imas_ambix.agent.gpu_watch import launcher

    probe = node_probe.NodeProbe() if probe is None else probe
    host = probe.hostname if hostname is None else hostname
    sampler = launcher.RemoteSampler() if remote is None else remote
    servers = servers_of(document)

    readings: dict[str, Reading] = {}
    local = [entry for entry in servers if _is_local(entry, host)]
    if local:
        ticks = (
            max(1, int(math.ceil(window_s / interval_s)) + 1) if window_s > 0 else 1
        )
        if sleep is None:
            samples = tuple(probe.stream(interval_s, count=ticks))
        else:
            samples = tuple(probe.stream(interval_s, count=ticks, sleep=sleep))
        reading = Reading(samples)
        for entry in local:
            readings[_server_name(entry)] = reading

    remote_groups: dict[str, list[Mapping[str, Any]]] = {}
    for entry in servers:
        if _is_local(entry, host):
            continue
        remote_groups.setdefault(_machine(entry), []).append(entry)
    for machine, entries in remote_groups.items():
        step = sampler.read(machine, window_s=window_s, interval_s=interval_s)
        for entry in entries:
            if step.job_id is None:
                readings[_server_name(entry)] = _no_job_reading(entry)
            else:
                readings[_server_name(entry)] = Reading(
                    step.samples, unavailable=step.reason
                )
    return readings


def _live_groups(
    servers: list[Mapping[str, Any]], host: str
) -> tuple[list[Mapping[str, Any]], dict[str, list[Mapping[str, Any]]]]:
    """Split the declarations into those on this host and those grouped by machine."""
    local = [entry for entry in servers if _is_local(entry, host)]
    remote_groups: dict[str, list[Mapping[str, Any]]] = {}
    for entry in servers:
        if not _is_local(entry, host):
            remote_groups.setdefault(_machine(entry), []).append(entry)
    return local, remote_groups


def _bucket(history: dict[str, deque], name: str, capacity: int) -> deque:
    bucket = history.get(name)
    if bucket is None:
        bucket = history[name] = deque(maxlen=capacity)
    return bucket


def _live_tick(
    servers: list[Mapping[str, Any]],
    *,
    local: list[Mapping[str, Any]],
    remote_groups: dict[str, list[Mapping[str, Any]]],
    probe: Any,
    pool: Any,
    history: dict[str, deque],
    capacity: int,
) -> dict[str, Reading]:
    """One redraw: take what each machine offers now, and read it per server.

    Everything for a machine is filed under each of that machine's
    declarations, so two spellings of one machine render the same cards instead
    of sampling the silicon twice. The pool is keyed by the machine name for
    ``ensure``, ``drain`` and ``state`` alike -- a key that differs between the
    three asks one target and drains another, so a block whose step was
    launched never sees its samples.
    """
    if local:
        tick = probe.sample(time.monotonic())
        for entry in local:
            _bucket(history, _server_name(entry), capacity).append(tick)
    for machine, entries in remote_groups.items():
        pool.ensure(machine)
        for sample in pool.drain(machine):
            for entry in entries:
                _bucket(history, _server_name(entry), capacity).append(sample)

    readings: dict[str, Reading] = {}
    for entry in servers:
        name = _server_name(entry)
        samples = tuple(_bucket(history, name, capacity))
        if samples:
            readings[name] = Reading(samples)
            continue
        machine = _machine(entry)
        if machine not in remote_groups:
            readings[name] = Reading(())
            continue
        job_id, reason = pool.state(machine)
        if job_id is None:
            readings[name] = _no_job_reading(entry)
        else:
            readings[name] = Reading(
                (), unavailable=reason or "waiting for the step's first sample"
            )
    return readings


def _run_live(
    document: Mapping[str, Any],
    *,
    window_s: float,
    interval_s: float,
    server: str | None,
    width: int,
    probe: node_probe.NodeProbe | None = None,
    pool: Any = None,
) -> int:
    """Redraw every interval from the servers' own step streams.

    Each server is sampled where it lives: a server on this host in-process on
    each tick, and one elsewhere from the long-lived step, so the whole view
    costs one step per server rather than one per sample. The mean columns are
    a rolling window over the samples each block has received over *window_s*.
    The steps are reaped on the way out, so none outlives the watcher.
    """
    import signal

    from rich.live import Live

    from imas_ambix.agent.gpu_watch import launcher

    probe = node_probe.NodeProbe() if probe is None else probe
    host = probe.hostname
    sampler_pool = launcher.SamplerPool(interval_s=interval_s) if pool is None else pool
    servers = servers_of(document)
    local, remote_groups = _live_groups(servers, host)

    capacity = max(1, int(math.ceil(window_s / interval_s)) + 1) if window_s > 0 else 1
    history: dict[str, deque] = {}

    def stop(*_args: object) -> None:
        raise SystemExit(143)

    previous = signal.signal(signal.SIGTERM, stop)
    console = Console(width=width, highlight=False)
    try:
        with Live(console=console, auto_refresh=False, screen=True) as live:
            while True:
                started = time.monotonic()
                readings = _live_tick(
                    servers,
                    local=local,
                    remote_groups=remote_groups,
                    probe=probe,
                    pool=sampler_pool,
                    history=history,
                    capacity=capacity,
                )
                live.update(render(document, readings, server=server), refresh=True)
                elapsed = time.monotonic() - started
                if elapsed < interval_s:
                    time.sleep(max(0.0, interval_s - elapsed))
    finally:
        sampler_pool.close()
        signal.signal(signal.SIGTERM, previous)
    return 0


def _endpoint_document() -> str | None:
    """The published endpoint document, at the path the site declares."""
    from imas_ambix.agent.profile import SiteConfig

    return SiteConfig.from_env().endpoint_document


def main(argv: list[str] | None = None) -> int:
    """The ``gpu-watch`` console script."""
    parser = argparse.ArgumentParser(
        prog="gpu-watch",
        description="Show every declared serving server: cards, host, jobs, engines.",
    )
    parser.add_argument(
        "window",
        nargs="?",
        type=float,
        default=None,
        help=f"seconds to average over (default {DEFAULT_WINDOW_S:.0f})",
    )
    parser.add_argument(
        "--window",
        dest="window_flag",
        type=float,
        default=None,
        help="seconds to average over",
    )
    parser.add_argument(
        "--server",
        dest="server",
        default=None,
        help="show only the named declared server",
    )
    parser.add_argument(
        "--endpoints",
        dest="endpoints_path",
        default=None,
        help="the published endpoint document",
    )
    parser.add_argument(
        "--interval",
        dest="interval",
        type=float,
        default=SAMPLE_INTERVAL_S,
        help="seconds between samples",
    )
    parser.add_argument("--width", dest="width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument(
        "--live",
        dest="live",
        action="store_true",
        help="redraw every interval from the steps' streams until interrupted",
    )
    parser.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        help="emit the document and readings as JSON",
    )
    args = parser.parse_args(argv)

    path = args.endpoints_path or _endpoint_document()
    try:
        with open(path, encoding="utf-8") as handle:
            document = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as failure:
        sys.stderr.write(f"gpu-watch: cannot read {path}: {failure}\n")
        return 1
    if not isinstance(document, dict):
        sys.stderr.write(f"gpu-watch: {path} does not hold a document\n")
        return 1

    window = args.window_flag if args.window_flag is not None else args.window
    window = DEFAULT_WINDOW_S if window is None else window
    if args.live:
        return _run_live(
            document,
            window_s=window,
            interval_s=args.interval,
            server=args.server,
            width=args.width,
        )
    readings = sample_readings(document, window_s=window, interval_s=args.interval)

    if args.as_json:
        payload = {
            "servers": document.get("servers") or [],
            "endpoints": document.get("endpoints") or [],
            "readings": {
                name: list(reading.samples) for name, reading in readings.items()
            },
        }
        sys.stdout.write(json.dumps(payload, indent=2, default=str) + "\n")
        return 0
    sys.stdout.write(
        render_text(document, readings, server=args.server, width=args.width)
    )
    sys.stdout.flush()
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    raise SystemExit(main())
