"""The watch pages: boxes as wide as the page, card readings, totals, paging."""

from __future__ import annotations

import pytest
from rich.console import Console, Group

from imas_ambix.agent.gpu_watch import app, fleet, panels
from imas_ambix.agent.gpu_watch.fleet import Card, Job, Reading, Server

WINDOW = 30


def _job(
    job_id: str,
    user: str,
    account: str,
    name: str,
    state: str,
    node: str,
    gres: str = "N/A",
) -> Job:
    return Job(
        job_id,
        user,
        account,
        name,
        "betelgeuse",
        state,
        "1:00:00",
        gres,
        "",
        node if state == "RUNNING" else "",
    )


def _h200(jobs: int = 2) -> Reading:
    node = "98dci4-gpu-0003"
    queue = [
        _job("100", fleet.USER, "grpa", "served-model", "RUNNING", node, "gres/gpu:4"),
        _job("101", "peer", "grpb", "trainer", "RUNNING", node, "gres/gpu:3"),
    ]
    queue += [
        _job(str(200 + i), "peer", "grpb", f"wait-{i}", "PENDING", node)
        for i in range(jobs - 2)
    ]
    return Reading(
        Server(node, "betelgeuse", "H200", 8),
        state="mix",
        jobs=queue,
        allocated={0, 1, 3, 4, 5, 6, 7},
        job_cards={"100": {4, 5, 6, 7}, "101": {0, 1, 3}},
        cards=[
            Card(i, 100.0, 90.0 + i, 78.0, 50.0 + i, 300.0 + i, 600.0)
            for i in (4, 5, 6, 7)
        ],
        host=(7.0, 10.0, 30.0),
        group="A",
    )


def _b300() -> Reading:
    node = "98dci4-gpu-0004"
    return Reading(
        Server(node, "betelgeuse", "B300", 8),
        state="idle",
        jobs=[
            _job(
                "300", fleet.USER, "grpa", "other-model", "RUNNING", node, "gres/gpu:2"
            )
        ],
        allocated={0, 1},
        job_cards={"300": {0, 1}},
        cards=[
            Card(0, 40.0, 50.0, 20.0, 61.0, 900.0, 1200.0),
            Card(1, 60.0, 70.0, 40.0, 63.0, None, None),
        ],
    )


def _lane_block(model: str, host: str, job: str, running: float) -> dict:
    return {
        "model": model,
        "host": host,
        "engine": f"{host}:18810",
        "live": {
            "state": "measured",
            "served_name": model,
            "hostname": host,
            "job_id": job,
            "mean_running": running,
            "engine_queue": 0,
            "generation_toks_per_s": 300.0,
            "per_stream_toks_per_s": 50.0,
            "prefill_toks_per_s": 900.0,
            "median_time_to_first_token_s": 0.6,
            "cache_hit_rate": 0.99,
            "pool_occupancy": 0.19,
        },
        "gate": {
            "router_generation_gate": {
                "effective_width": 16,
                "in_flight": 7,
                "waiting": 0,
            },
            "admission": {"verdict": "open"},
            "worker_slots": {"worker_slots": 74, "live_runs": 12},
            "headroom": 9,
            "availability_percent": 56.0,
        },
    }


def _ledger(partial: bool = False) -> list[dict]:
    return [
        {
            "label": label,
            "tokens_in": 9.5e7,
            "cached": 9.2e7,
            "tokens_out": 5e5,
            "utilisation": 78.0,
            "cost": 2.33,
            "period": 3600.0,
            "covered": 1800.0 if partial and label == "1 day" else 3600.0,
        }
        for label in ("1 hour", "1 day")
    ]


def _document(engines: int = 1, partial: bool = False) -> dict:
    blocks = [
        _lane_block("served-model", "98dci4-gpu-0003.iter.org", "100", 5.5),
        _lane_block("other-model", "98dci4-gpu-0004.iter.org", "300", 2.0),
    ][:engines]
    first = blocks[0]
    document = {
        "lane": dict(first["gate"], model_id=first["model"]),
        "live": first["live"],
        "engine": {},
        "checkpoints": {},
        "ledger": _ledger(partial),
        "ledger_published_at": None,
    }
    if engines > 1:
        document["engines"] = blocks
    return document


NARROW, WIDE = 100, 190


def _lines(renderable: object, width: int = NARROW) -> list[str]:
    console = Console(width=width, force_terminal=False)
    with console.capture() as capture:
        console.print(renderable)
    return capture.get().rstrip("\n").splitlines()


def _page(
    readings: list[Reading],
    engines: int = 1,
    height: int | None = None,
    width: int = NARROW,
    partial: bool = False,
) -> list[str]:
    size = (width, height or 0)
    return _lines(
        app.render(
            readings, _document(engines, partial), WINDOW, size, keys=bool(height)
        ),
        width,
    )


def _figures(line: str) -> list[str]:
    return line.replace("█", " ").replace("░", " ").split()


def _row(lines: list[str], name: str) -> str:
    return next(line for line in lines if line.startswith(name))


def _titles(lines: list[str]) -> list[str]:
    """The first word of every section's title line, in order."""
    names = ("gpu-", "LANE", "LEDGER", "QUEUE", "SERVERS")
    return [line.split()[0] for line in lines if line.startswith(names)]


@pytest.mark.parametrize("width", [80, NARROW, WIDE])
def test_the_page_is_one_unframed_column_that_fits_its_width(width):
    lines = _page([_h200(), _b300()], engines=2, width=width)
    assert all(len(line) <= width for line in lines)
    assert not any(mark in line for line in lines for mark in "│╭╰─")
    assert _titles(lines) == ["gpu-0003", "gpu-0004", "LANE", "LANE", "LEDGER", "QUEUE"]


def test_a_card_row_is_utilisation_now_its_mean_and_memory():
    lines = _page([_h200()])
    assert _figures(_row(lines, "card")) == [
        "card",
        "util",
        "now",
        "util",
        "30s",
        "memory",
    ]
    assert _figures(_row(lines, "gpu 4")) == ["gpu", "4", "100%", "94%", "78%"]
    assert _figures(_row(lines, "gpu 4-7")) == [
        "gpu",
        "4-7",
        "served-model",
        "100%",
        "96%",
        "78%",
    ]


def test_temperature_and_power_are_the_server_s_and_not_each_card_s():
    lines = _page([_h200()])
    title = _row(lines, "gpu-0003")
    assert "57 °C max" in title
    assert "1.22 kW" in title
    assert "7/8 cards allocated" in title
    assert "°C" not in _row(lines, "card") and "kW" not in _row(lines, "card")


def test_a_wider_page_draws_longer_utilisation_bars():
    narrow = _row(_page([_h200()], width=80), "gpu 4").count("█")
    wide = _row(_page([_h200()], width=WIDE), "gpu 4").count("█")
    assert wide > narrow


def test_unreadable_cards_share_one_row_and_stay_out_of_the_mean():
    lines = _page([_h200()])
    row = _row(lines, "gpu 0-1 3")
    assert "another account's" in row
    assert "%" not in row


def test_a_card_whose_sensor_is_silent_leaves_the_total_to_the_others():
    title = _row(_page([_b300()]), "gpu-0004")
    assert "0.90 kW" in title
    assert "63 °C max" in title


def test_each_lane_is_titled_with_its_server_and_cards():
    lines = _page([_h200(), _b300()], engines=2)
    lanes = [line for line in lines if line.startswith("LANE")]
    assert "served-model" in lanes[0] and "gpu-0003 gpu 4-7" in lanes[0]
    assert "other-model" in lanes[1] and "gpu-0004 gpu 0-1" in lanes[1]


def test_a_lane_is_one_row_of_use_queues_and_rates():
    lines = _page([_h200()])
    at = next(i for i, line in enumerate(lines) if line.startswith("LIVE"))
    assert lines[at].split()[1:] == [
        "in",
        "use",
        "queue",
        "held",
        "tok/s",
        "/stream",
        "prefill/s",
        "ttft",
        "s",
    ]
    assert lines[at + 1].split() == ["7/16", "0", "0", "300", "50.0", "900", "0.6"]
    assert not any(line.startswith(("SERVING", "LOAD")) for line in lines)


def test_a_terminal_wider_than_the_page_does_not_stretch_it():
    lines = _page([_h200()], width=WIDE)
    assert max(len(line.rstrip()) for line in lines) <= 110


def test_two_servers_share_one_queue_that_names_each_job_s_server():
    lines = _page([_h200(), _b300()], engines=2)
    assert "QUEUE  3 running · 0 pending" in lines
    assert "server" in _row(lines, "job")
    assert "gpu-0004" in _row(lines, "300")
    assert "server" not in _row(_page([_h200()]), "job")


@pytest.mark.parametrize("width", [NARROW, WIDE])
def test_the_queue_takes_the_lines_left_and_counts_what_does_not_fit(width):
    height = 34
    lines = _page([_h200(jobs=30)], height=height, width=width)
    assert len(lines) == height
    assert lines[-1] == "q quit"
    more = next(line for line in lines if line.startswith("+"))
    shown = sum(1 for line in lines if line[:3].isdigit())
    assert more.startswith(f"+{30 - shown} more")
    assert shown >= panels.MIN_QUEUE_ROWS - 1


def test_a_short_screen_folds_the_cards_then_drops_the_ledger():
    roomy = _page([_h200(jobs=30)], height=37)
    assert any(line.startswith("gpu 4 ") for line in roomy)
    folded = _page([_h200(jobs=30)], height=33)
    assert not any(line.startswith("gpu 4 ") for line in folded)
    assert _figures(_row(folded, "gpu 4-7")) == [
        "gpu",
        "4-7",
        "served-model",
        "100%",
        "96%",
        "78%",
    ]
    assert "LEDGER" in folded
    short = _page([_h200(jobs=30)], height=27)
    assert "LEDGER" not in short
    assert sum(1 for line in short if line[:3].isdigit()) >= panels.MIN_QUEUE_ROWS - 1


def test_the_ledger_states_coverage_only_for_a_short_record():
    assert "covered" not in _row(_page([_h200()]), "period")
    assert "covered" in _row(_page([_h200()], partial=True), "period")


def test_a_missing_lane_document_says_why():
    lines = _lines(
        app.render([_h200()], None, WINDOW, reason="OSError: record unreadable")
    )
    assert "no lane document read — OSError: record unreadable" in lines


def test_other_gpu_servers_are_watched_unless_the_environment_names_them(
    monkeypatch,
):
    monkeypatch.delenv(fleet.PARTITIONS_ENV, raising=False)
    assert fleet.watched_partitions()[0] == ("betelgeuse", "titan")
    monkeypatch.setenv(fleet.PARTITIONS_ENV, "")
    assert fleet.watched_partitions()[0] == ("betelgeuse",)


def test_the_scheduler_output_is_read_as_servers_jobs_and_cards():
    nodes = fleet.parse_nodes(
        "98dci4-gpu-0003|betelgeuse|gpu:h200:8(S:0-1)|mix|none\n"
        "98dci4-gpu-0004|betelgeuse*|gpu:b300:8|drain|bad card\n"
        "98dci4-clu-0001|betelgeuse|(null)|idle|none\n"
    )
    assert [(s.name, s.hardware, state, why) for s, state, why in nodes] == [
        ("gpu-0003", "8×H200", "mix", ""),
        ("gpu-0004", "8×B300", "drain", "bad card"),
    ]
    jobs = fleet.parse_jobs(
        "1294597|mcintos|grpa|deepseek-v4-1-flash|betelgeuse|RUNNING|"
        "20-19:44:22|gres/gpu:4|gpu_0003_grpA|98dci4-gpu-0003\n"
        "1309674|buskirj|grpb|fluent_gpu|betelgeuse|PENDING|0:00|N/A||\n"
    )
    assert [(j.running, j.pending, j.gpus, j.group) for j in jobs] == [
        (True, False, 4, "A"),
        (False, True, 0, "B"),
    ]
    held = fleet.parse_card_indices(
        "   JobState=RUNNING\n"
        "     Nodes=98dci4-gpu-0003 CPU_IDs=15-26 Mem=65536 "
        "GRES=gpu:h200:6(IDX:0-3,6-7)\n"
    )
    assert held == {"98dci4-gpu-0003": {0, 1, 2, 3, 6, 7}}


QUEUE_AS_CAPTURED = (
    "1308623|murawan|grpb|qwen38_flash_next|betelgeuse|PENDING|0:00|N/A|"
    "gpu_0003_grpB|\n"
    "1308599|murawan|grpb|qwen38_flash_next|betelgeuse|RUNNING|5-15:04:23|N/A|"
    "gpu_0003_grpB|98dci4-gpu-0003\n"
    "1313440|mcintos|grpa|ambix-router|betelgeuse|RUNNING|1-00:56:43|N/A|"
    "gpu_0003_grpA|98dci4-gpu-0003\n"
    "1294597|mcintos|grpa|deepseek-v4-1-flash|betelgeuse|RUNNING|21-13:41:22|"
    "gres/gpu:4|gpu_0003_grpA|98dci4-gpu-0003\n"
    "1314914|mcintos|iter|identity-clean.sh|betelgeuse|RUNNING|35:01|"
    "gres/gpu:1|gpu_0003_grpA|98dci4-gpu-0003\n"
    "1313403|mcintos|iter|codex-neo4j|titan|RUNNING|1-01:14:19|N/A|(null)|"
    "98dci4-gpu-0002\n"
)

PER_JOB_REQUEST_DETAIL = (
    "JobId=1308599 JobName=qwen38_flash_next\n"
    "   ReqTRES=cpu=10,mem=360G,node=1,billing=10,gres/gpu=2\n"
    "   AllocTRES=cpu=10,mem=360G,node=1,billing=10,gres/gpu=2\n"
    "   JOB_GRES=gpu:h200:2\n"
    "     Nodes=98dci4-gpu-0003 CPU_IDs=12-13,38-45 Mem=368640 "
    "GRES=gpu:h200:2(IDX:1,3)\n"
    "   TresPerJob=gres/gpu:2\n"
)


def test_cards_asked_for_per_job_are_counted_from_the_job_detail():
    """The queue line of a job that asked for cards per job names none."""
    jobs = fleet.parse_jobs(QUEUE_AS_CAPTURED)
    peer = next(job for job in jobs if job.id == "1308599")
    assert peer.gpus == 0 and peer.group == "B"
    node = "98dci4-gpu-0003"
    held = fleet.parse_card_indices(PER_JOB_REQUEST_DETAIL)
    assert held == {node: {1, 3}}
    assert fleet.parse_asked_cards(PER_JOB_REQUEST_DETAIL) == 2
    assert fleet.parse_asked_cards("   ReqTRES=cpu=1,mem=4G,node=1\n") == 0
    reading = Reading(
        Server(node, "betelgeuse", "H200", 8),
        jobs=[job for job in jobs if job.node == node or job.pending],
        job_cards={"1308599": held[node], "1294597": {4, 5, 6, 7}},
        asked={"1308623": 2},
    )
    # Two from the detail, four from the indices, one from the queue line.
    assert reading.allocated_count == 7
    waiting = next(job for job in jobs if job.pending)
    assert reading.gpus(waiting) == 2
    queue = _lines(Group(*panels.queue_section([reading], NARROW)))
    assert _row(queue, "1308599").endswith("2")
    assert "QUEUE  4 running · 1 pending" in queue


def test_a_sampler_joins_each_job_with_cards_and_one_job_on_a_bare_server(
    monkeypatch,
):
    monkeypatch.setattr(fleet, "USER", "mcintos")
    jobs = fleet.parse_jobs(QUEUE_AS_CAPTURED)
    held = {
        "1294597": {"98dci4-gpu-0003": {4, 5, 6, 7}},
        "1314914": {"98dci4-gpu-0003": {0}},
        "1313440": {},
        "1313403": {},
    }
    nodes = {"98dci4-gpu-0003", "98dci4-gpu-0002"}
    assert fleet.sampled_jobs(jobs, held, nodes) == {
        "1294597": "98dci4-gpu-0003",
        "1314914": "98dci4-gpu-0003",
        "1313403": "98dci4-gpu-0002",
    }


def test_sampler_lines_become_cards_with_a_rolling_mean():
    node = "98dci4-gpu-0003"
    watch = fleet.Fleet(window=2)
    watch._servers = [Server(node, "betelgeuse", "H200", 8)]
    watch._jobs = [_job("100", fleet.USER, "grpa", "served", "RUNNING", node)]
    watch._samplers["100"] = object()
    owned: set = set()
    means = []
    for util in (20.0, 40.0, 90.0):
        watch._take_line(node, f"GPU\t4\tuuid\t{util}\t78\t60\t\t600\n", owned)
        watch._take_line(node, "NODE\t10\t100\t400\n", owned)
        (reading,) = watch.tick()
        means.append(reading.cards[0].util_mean)
    card = reading.cards[0]
    # The window holds two samples, so the first reading has left the mean.
    assert means == [20.0, 30.0, 65.0]
    assert (card.index, card.temp, card.watts, card.cap_watts) == (4, 60.0, None, 600.0)
    assert reading.host == (10.0, 10.0, 25.0)
    assert owned == {(node, 4)}


def test_the_view_is_live_in_a_terminal_and_one_reading_elsewhere():
    from imas_ambix.agent.gpu_watch.app import live_by_default

    assert live_by_default(False, False, True)
    assert not live_by_default(False, True, True)
    assert not live_by_default(False, False, False)
    assert live_by_default(True, False, False)


def test_cards_are_averaged_within_a_job_and_never_across_jobs():
    reading = _h200()
    reading.job_cards = {"100": {4, 5, 6, 7}, "102": {2}, "101": {0, 1, 3}}
    reading.allocated = set(range(8))
    reading.cards.insert(0, Card(2, 10.0, 20.0, 30.0))
    folded = _page([reading], height=20)
    reading.jobs.append(
        _job("102", fleet.USER, "iter", "solver", "RUNNING", "gpu-0003", "gres/gpu:1")
    )
    folded = _page([reading], height=20)
    assert _figures(_row(folded, "gpu 2 ")) == [
        "gpu",
        "2",
        "solver",
        "10%",
        "20%",
        "30%",
    ]
    assert "solver" in _row(_page([reading]), "gpu 2 ")
    assert _figures(_row(folded, "gpu 4-7")) == [
        "gpu",
        "4-7",
        "served-model",
        "100%",
        "96%",
        "78%",
    ]
    assert not any(line.startswith("gpu 2 4-7") for line in folded)
    assert not any(line.startswith("mean") for line in _page([reading]))


def test_a_job_submitted_without_a_name_is_named_by_its_working_directory():
    jobs = fleet.parse_jobs(
        "1315842|mcintos|iter|wrap|titan|RUNNING|1:29:31|gres/gpu:1||98dci4-gpu-0002|"
        "/home/user/worktrees/nova/diiid-external-current-propagation\n"
        "1294597|mcintos|grpa|served-model|betelgeuse|RUNNING|1:00|gres/gpu:4||"
        "98dci4-gpu-0003|/home/user/Code/imas-ambix\n"
        "1309674|buskirj|grpb|wrap|betelgeuse|PENDING|0:00|N/A||\n"
    )
    assert [job.label for job in jobs] == [
        "diiid-external-current-propagation",
        "served-model",
        "wrap",
    ]
