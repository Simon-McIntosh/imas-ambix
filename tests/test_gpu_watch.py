"""The watch pages: boxes as wide as the page, card readings, totals, paging."""

from __future__ import annotations

import pytest
from rich.console import Console

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


NARROW, WIDE = 120, 150


def _lines(renderable: object, width: int = NARROW) -> list[str]:
    console = Console(width=width, force_terminal=False)
    with console.capture() as capture:
        console.print(renderable)
    return capture.get().rstrip("\n").splitlines()


def _page(
    page: str,
    readings: list[Reading],
    engines: int = 1,
    height: int | None = None,
    width: int = NARROW,
) -> list[str]:
    size = (width, height or 0)
    return _lines(
        app.render(page, readings, _document(engines), WINDOW, size, keys=bool(height)),
        width,
    )


def _rows(lines: list[str]) -> list[str]:
    """Each line of a stacked page without the box's own edges."""
    return [line.strip("│").strip() for line in lines]


def _figures(line: str) -> list[str]:
    return line.replace("█", " ").replace("░", " ").replace("│", " ").split()


def _titles(lines: list[str]) -> list[str]:
    """The title of every box on a stacked page, in order."""
    return [line.strip("╭─╮ ") for line in lines if line.startswith("╭")]


@pytest.mark.parametrize("width", [80, NARROW, WIDE])
@pytest.mark.parametrize("page", ["overview", "gpu-0003", "gpu-0004", "lanes"])
def test_every_box_is_as_wide_as_the_page(page, width):
    lines = _page(page, [_h200(), _b300()], engines=2, width=width)
    assert max(len(line) for line in lines) == width
    # A line holding one box is that box's edge; two boxes side by side
    # together span the page.
    edges = [line.rstrip() for line in lines if line.startswith(("╭", "╰"))]
    assert edges and {len(line) for line in edges if line.count("╮") < 2} <= {
        width,
        (width - 1) // 2,
    }
    assert not any("…" in line for line in lines)


def test_a_wide_server_page_stands_the_cards_and_queue_beside_the_lanes():
    wide = _page("gpu-0003", [_h200()], width=WIDE)
    assert any("MACHINE" in line and "LANE" in line for line in wide)
    assert any("QUEUE" in line and "LEDGER" in line for line in wide)
    # Beside the cards there is room for the lane's memory band as well.
    assert any("kv pool %" in line for line in wide)
    stacked = _page("gpu-0003", [_h200()])
    assert [title.split()[0] for title in _titles(stacked)] == [
        "MACHINE",
        "LANE",
        "LEDGER",
        "QUEUE",
    ]
    assert not any("kv pool %" in line for line in stacked)


def test_each_card_carries_its_own_temperature_and_power():
    rows = _rows(_page("gpu-0003", [_h200()]))
    card = next(row for row in rows if row.startswith("gpu 5"))
    assert _figures(card)[-3:] == ["55", "0.30", "51"]


def test_the_totals_row_is_mean_load_hottest_card_and_summed_power():
    rows = _rows(_page("gpu-0003", [_h200()]))
    total = next(row for row in rows if row.startswith("4 cards"))
    # Mean of 94..97, memory 78, hottest 57, 4 x ~0.3 kW at half the cap.
    assert _figures(total) == [
        "4",
        "cards",
        "100%",
        "96%",
        "78%",
        "57",
        "max",
        "1.22",
        "51",
    ]


def test_unreadable_cards_share_one_row_and_stay_out_of_the_totals():
    rows = _rows(_page("gpu-0003", [_h200()]))
    hidden = next(row for row in rows if "unsampled" in row)
    assert hidden.startswith("gpu 0-1 3") and "allocated" in hidden
    assert sum(1 for row in rows if "unsampled" in row) == 1


def test_a_card_whose_sensor_is_silent_shows_a_dash_not_a_zero():
    rows = _rows(_page("gpu-0004", [_h200(), _b300()], engines=2))
    card = next(row for row in rows if row.startswith("gpu 1"))
    assert _figures(card)[-2:] == ["—", "—"]
    total = next(row for row in rows if row.startswith("2 cards"))
    # Power is summed over the one card that reported it.
    assert _figures(total)[-2:] == ["0.90", "75"]


def test_the_only_server_opens_on_its_own_page_and_carries_the_ledger():
    readings = [_h200()]
    assert app.resolve_page(None, readings) == "gpu-0003"
    titles = _titles(_page("gpu-0003", readings))
    assert "LEDGER" in titles
    assert any(t.startswith("LANE  served-model · gpu 4-7") for t in titles)


def test_two_servers_open_on_the_overview_with_totals():
    readings = [_h200(), _b300()]
    assert app.resolve_page(None, readings) == "overview"
    rows = _rows(_page("overview", readings, engines=2))
    servers = [row for row in rows if row.startswith(("gpu-000", "total"))][:3]
    assert [row.split()[0] for row in servers] == ["gpu-0003", "gpu-0004", "total"]
    assert _figures(servers[2])[1:3] == ["16", "9/16"]
    assert _figures(servers[2])[-1] == "3+0"
    lanes = [row for row in rows if row.startswith(("served-", "other-"))]
    assert [row.split()[1] for row in lanes] == ["gpu-0003", "gpu-0004"]
    assert any(row.split()[:2] == ["total", "7.5"] for row in rows)


def test_a_server_page_lists_only_the_lanes_served_from_it():
    titles = _titles(_page("gpu-0004", [_h200(), _b300()], engines=2))
    lanes = [title for title in titles if title.startswith("LANE")]
    assert len(lanes) == 1 and "other-model" in lanes[0]
    # With more than one server the ledger lives on the overview.
    assert "LEDGER" not in titles


@pytest.mark.parametrize("width", [NARROW, WIDE])
def test_the_queue_takes_the_lines_left_and_counts_what_does_not_fit(width):
    readings = [_h200(jobs=40)]
    lines = _page("gpu-0003", readings, height=44, width=width)
    assert len(lines) == 44
    cut = (width - 1) // 2 if width == WIDE else width
    column = [line[:cut] for line in lines]
    top = next(i for i, line in enumerate(column) if "QUEUE" in line)
    assert "2 running · 38 pending" in column[top]
    rows = _rows(column[top + 2 : -2])
    assert len(rows) >= panels.MIN_QUEUE_ROWS
    assert rows[-1].split()[:2] == [f"+{40 - len(rows) + 1}", "more"]
    assert "running" in rows[0] and "pending" in rows[-2]


def test_the_ledger_states_coverage_only_for_a_short_record():
    full = _lines(panels.ledger_box(_ledger(), None, NARROW))
    short = _lines(panels.ledger_box(_ledger(partial=True), None, NARROW))
    assert "covered" not in full[1] and "covered" in short[1]
    assert {len(line) for line in full + short} == {NARROW}


def test_the_lanes_page_titles_each_lane_with_its_server_and_cards():
    titles = _titles(_page("lanes", [_h200(), _b300()], engines=2))
    lanes = [title for title in titles if title.startswith("LANE ")]
    assert len(lanes) == 2
    assert "gpu-0003 gpu 4-7" in lanes[0] and "gpu-0004 gpu 0-1" in lanes[1]
    assert titles[0] == "LANES" and titles[-1] == "LEDGER"


def test_a_long_verdict_and_a_long_checkpoint_are_neither_joined_nor_cut():
    document = _document(1)
    document["lane"]["admission"]["verdict"] = "congested"
    document["lane"]["router_generation_gate"]["waiting"] = 1
    lane = app.lanes(document)[0]
    lane.model, lane.context = "deepseek-v4.1-flash", 262144
    lane.checkpoint = "DeepSeek-V4.1-Flash-1017@1bae0e10"
    for width in (80, NARROW):
        lines = _lines(panels.lanes_page([_h200()], [lane], None, width=width), width)
        row = next(r for r in _rows(lines) if r.startswith("deepseek"))
        assert row.split()[-2:] == ["1", "congested"]
        assert not any("…" in line for line in lines)
        assert any("DeepSeek-V4.1-Flash-1017@1bae0e10" in line for line in lines)
        assert any("262k ctx" in line for line in lines)


def test_keys_move_between_pages():
    readings = [_h200(), _b300()]
    assert app.step("overview", "2", readings) == "gpu-0004"
    assert app.step("gpu-0004", "0", readings) == "overview"
    assert app.step("overview", "left", readings) == "lanes"
    assert app.step("gpu-0003", "right", readings) == "gpu-0004"
    assert app.step("gpu-0003", "9", readings) == "gpu-0003"


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
    queue = _rows(_lines(panels.server_page(reading, [], WINDOW)))
    row = next(line for line in queue if line.startswith("1308599"))
    assert row.endswith("2")
    title = next(line for line in queue if "QUEUE" in line)
    assert "4 running · 1 pending" in title


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


def test_a_missing_lane_document_says_why():
    page = _lines(
        app.render("overview", [_h200()], None, WINDOW, reason="OSError: no record")
    )
    assert any("no lane document read — OSError: no record" in line for line in page)


def test_a_short_screen_keeps_ten_queue_rows_by_dropping_the_ledger():
    readings = [_h200(jobs=40)]
    roomy = _titles(_page("gpu-0003", readings, height=60))
    assert "LEDGER" in roomy
    short = _page("gpu-0003", readings, height=34)
    assert len(short) == 34
    assert "LEDGER" not in _titles(short)
    top = next(i for i, line in enumerate(short) if "QUEUE" in line)
    assert len(short[top + 2 : -2]) >= panels.MIN_QUEUE_ROWS


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
