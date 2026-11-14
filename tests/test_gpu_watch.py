"""The watch pages: one right edge, per-card readings, totals, and paging."""

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


def _lines(renderable: object) -> list[str]:
    console = Console(width=120, force_terminal=False)
    with console.capture() as capture:
        console.print(renderable)
    return capture.get().rstrip("\n").splitlines()


def _page(
    page: str, readings: list[Reading], engines: int = 1, height: int | None = None
) -> list[str]:
    size = (120, height) if height else None
    return _lines(
        app.render(page, readings, _document(engines), WINDOW, size, keys=bool(height))
    )


def _figures(line: str) -> list[str]:
    return line.replace("█", " ").replace("░", " ").split()


@pytest.mark.parametrize("page", ["overview", "gpu-0003", "gpu-0004"])
def test_every_table_row_ends_on_the_same_column(page):
    lines = _page(page, [_h200(), _b300()], engines=2)
    rules = [line for line in lines if set(line) == {"─"}]
    assert rules and {len(line) for line in rules} == {panels.WIDTH}
    assert max(len(line) for line in lines) == panels.WIDTH
    # Every header row, and every row of figures under it, reaches the edge.
    for name in ("MACHINE", "SERVERS", "LANES", "LEDGER", "QUEUE"):
        for line in lines:
            if line.startswith(name):
                assert len(line) == panels.WIDTH, line


def test_each_card_carries_its_own_temperature_and_power():
    lines = _page("gpu-0003", [_h200()])
    card = next(line for line in lines if line.startswith("gpu 5"))
    assert _figures(card)[-3:] == ["55", "0.30", "51"]


def test_the_totals_row_is_mean_load_hottest_card_and_summed_power():
    lines = _page("gpu-0003", [_h200()])
    total = next(line for line in lines if line.startswith("4 cards"))
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
    lines = _page("gpu-0003", [_h200()])
    hidden = next(line for line in lines if "unsampled" in line)
    assert hidden.startswith("gpu 0-1 3") and "allocated" in hidden
    assert sum(1 for line in lines if "unsampled" in line) == 1


def test_a_card_whose_sensor_is_silent_shows_a_dash_not_a_zero():
    lines = _page("gpu-0004", [_h200(), _b300()], engines=2)
    card = next(line for line in lines if line.startswith("gpu 1"))
    assert _figures(card)[-2:] == ["—", "—"]
    total = next(line for line in lines if line.startswith("2 cards"))
    # Power is summed over the one card that reported it.
    assert _figures(total)[-2:] == ["0.90", "75"]


def test_the_only_server_opens_on_its_own_page_and_carries_the_ledger():
    readings = [_h200()]
    assert app.resolve_page(None, readings) == "gpu-0003"
    lines = _page("gpu-0003", readings)
    assert any(line.startswith("LEDGER") for line in lines)
    assert any(line.startswith("LANE  served-model · gpu 4-7") for line in lines)


def test_two_servers_open_on_the_overview_with_totals():
    readings = [_h200(), _b300()]
    assert app.resolve_page(None, readings) == "overview"
    lines = _page("overview", readings, engines=2)
    servers = [line for line in lines if line.startswith(("gpu-000", "total"))]
    servers = servers[:3]
    assert [line.split()[0] for line in servers] == ["gpu-0003", "gpu-0004", "total"]
    assert _figures(servers[2])[1:3] == ["16", "9/16"]
    assert _figures(servers[2])[-1] == "3+0"
    lanes = [line for line in lines if line.startswith(("served-", "other-"))]
    assert [line.split()[1] for line in lanes] == ["gpu-0003", "gpu-0004"]
    assert any(line.split()[:2] == ["total", "7.5"] for line in lines)


def test_a_server_page_lists_only_the_lanes_served_from_it():
    lines = _page("gpu-0004", [_h200(), _b300()], engines=2)
    heads = [line for line in lines if line.startswith("LANE")]
    assert len(heads) == 1 and "other-model" in heads[0]
    # With more than one server the ledger lives on the overview.
    assert not any(line.startswith("LEDGER") for line in lines)


def test_the_queue_takes_the_lines_left_and_counts_what_does_not_fit():
    readings = [_h200(jobs=40)]
    lines = _page("gpu-0003", readings, height=40)
    assert len(lines) == 40
    head = next(i for i, line in enumerate(lines) if line.startswith("QUEUE"))
    rows = lines[head + 1 : -2]
    assert len(rows) >= panels.MIN_QUEUE_ROWS
    assert rows[-1].split()[:2] == [f"+{40 - len(rows) + 1}", "more"]
    assert "RUNNING" in rows[0] and "PENDING" in rows[-2]


def test_the_ledger_states_coverage_only_for_a_short_record():
    full = _lines(panels.Group(*panels.ledger_panel(_ledger())))
    short = _lines(panels.Group(*panels.ledger_panel(_ledger(partial=True))))
    assert "covered" not in full[0] and "covered" in short[0]
    assert {len(line) for line in full + short} == {panels.WIDTH}


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
