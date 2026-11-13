"""The gpu-watch layout: what each panel draws and how the queue is given room."""

from __future__ import annotations

import pytest
from rich.console import Console

from imas_ambix.agent.gpu_watch import view

WIDTH = 104


def _job(jobid: int, state: str, acct: str = "grpa", gres: str = "gres/gpu:1"):
    return [
        str(jobid),
        "someone",
        acct,
        f"job-{jobid}",
        "betelgeuse",
        state,
        "1:00",
        gres,
        "",
    ]


def _data(pending: int = 0) -> dict:
    jobs = [_job(100, "RUNNING", gres="gres/gpu:4"), _job(101, "RUNNING", "grpb")]
    jobs += [_job(200 + i, "PENDING", "grpb") for i in range(pending)]
    return {
        "nstate": "mix",
        "jobs": jobs,
        "gpu_group": "A",
        "vllm": None,
        "gpus": {str(i): (100.0, 98.0, 78.0, 78.0) for i in (4, 5, 6, 7)},
        "alloc_idx": {0, 1, 3, 4, 5, 6, 7},
        "node": (9.0, 300.0, 1000.0),
        "node_mean": (9.0, 30.0),
        "env": {str(i): (60.0, 350.0, 600.0) for i in (4, 5, 6, 7)},
    }


def _ledger_row(label: str, covered: float = 3600.0) -> dict:
    return {
        "label": label,
        "tokens_in": 1e8,
        "cached": 9.7e7,
        "tokens_out": 1e6,
        "utilisation": 80.0,
        "period": 3600.0,
        "covered": covered,
        "cost": 3.5,
    }


@pytest.fixture
def published(monkeypatch):
    """Stand in for every published source, so a render reads nothing live."""
    live = {
        "lane": {
            "router_generation_gate": {
                "effective_width": 12,
                "in_flight": 6,
                "waiting": 0,
            },
            "admission": {"verdict": "open"},
            "headroom": 6,
            "availability_percent": 50.0,
        },
        "live": {
            "state": "measured",
            "served_name": "served-model",
            "mean_running": 7.0,
            "engine_queue": 0,
            "generation_toks_per_s": 452.0,
            "per_stream_toks_per_s": 64.5,
            "prefill_toks_per_s": 645.0,
            "median_time_to_first_token_s": 0.7,
            "cache_hit_rate": 0.99,
            "pool_occupancy": 0.13,
        },
        "callers": {
            "state": "measured",
            "window_seconds": 300.0,
            "covered": True,
            "callers": [],
        },
    }
    watch = {"ledger": [_ledger_row("1 hour"), _ledger_row("1 day")]}
    monkeypatch.setattr(view, "_watch_live", lambda: live)
    monkeypatch.setattr(view, "serve_state", lambda: None)
    monkeypatch.setattr(view, "_checkpoint_label", lambda: "checkpoint@abcdef12")
    monkeypatch.setattr(view, "_watch_document", lambda: (watch, None))
    return {"live": live, "watch": watch}


def _lines(data: dict, height: int | None) -> list[str]:
    console = Console(width=WIDTH, force_terminal=False)
    size = (WIDTH, height) if height else None
    with console.capture() as capture:
        console.print(view.render(data, 30, size))
    return capture.get().rstrip("\n").splitlines()


def _job_rows(lines: list[str]) -> list[str]:
    head = next(i for i, line in enumerate(lines) if line.startswith("jobid"))
    return lines[head + 2 :]


def test_the_view_fits_the_screen_it_is_given(published):
    for height in (36, 40, 47, 60):
        assert len(_lines(_data(pending=30), height)) <= height


def test_the_queue_shows_ten_rows_on_a_forty_line_screen(published):
    rows = _job_rows(_lines(_data(pending=30), 40))
    assert len(rows) >= view.MIN_JOB_ROWS


def test_the_queue_takes_every_line_a_taller_screen_leaves(published):
    short = _job_rows(_lines(_data(pending=30), 47))
    tall = _job_rows(_lines(_data(pending=30), 60))
    assert len(tall) == len(short) + 13


def test_jobs_that_do_not_fit_are_counted_not_cut(published):
    rows = _job_rows(_lines(_data(pending=30), 47))
    shown = len(rows) - 1
    assert rows[-1].startswith(f"+{32 - shown} more")
    assert f"{32 - shown} pending" in rows[-1]


def test_running_jobs_come_before_pending_ones(published):
    data = _data(pending=2)
    data["jobs"].reverse()
    states = [row.split()[4] for row in _job_rows(_lines(data, None))]
    assert states == ["RUNNING", "RUNNING", "PENDING", "PENDING"]


def test_a_one_shot_reading_lists_every_job(published):
    assert len(_job_rows(_lines(_data(pending=30), None))) == 32


def test_unreadable_cards_share_one_row(published):
    text = "\n".join(_lines(_data(), None))
    assert text.count("unsampled") == 1
    assert "gpu 0 1 3" in text


def _caller(session: str, runs: int, held: float, wait: float) -> dict:
    return {
        "session": session,
        "hosts": [],
        "runs": runs,
        "requests": 10.0,
        "held": held,
        "held_per_run": held / runs,
        "mean_gate_wait_s": wait,
    }


def _callers_lines(lines: list[str]) -> list[str]:
    return [line for line in lines if line.startswith("callers")]


def test_an_empty_callers_window_on_an_idle_lane_draws_nothing(published):
    published["live"]["lane"]["router_generation_gate"]["in_flight"] = 0
    lines = _lines(_data(), None)
    assert _callers_lines(lines) == []
    assert not any(line.strip() == "idle" for line in lines)


def test_an_empty_callers_window_under_load_is_stated(published):
    assert _callers_lines(_lines(_data(), None)) == [
        "callers 5m   no finished request recorded — 6 in flight"
    ]


def test_callers_are_one_row_per_user(published, monkeypatch):
    published["live"]["callers"]["callers"] = [
        _caller("s-one", 2, 3.0, 4.0),
        _caller("s-two", 1, 1.0, 8.0),
        _caller("s-other", 1, 0.5, 0.0),
    ]
    owners = {"s-one": view.USER, "s-two": view.USER, "s-other": "?"}
    monkeypatch.setattr(
        view, "_caller_owner", lambda caller: ("a-project", owners[caller["session"]])
    )
    lines = _lines(_data(), None)
    head = next(i for i, line in enumerate(lines) if "sessions" in line)
    assert lines[head].split() == [
        "callers",
        "5m",
        "sessions",
        "runs",
        "in",
        "use",
        "per",
        "run",
        "wait",
        "s",
    ]
    rows = [line.split() for line in lines[head + 2 : head + 4]]
    assert rows == [
        [view.USER, "2", "3", "4.0", "1.3", "6"],
        ["unknown", "1", "1", "0.5", "0.5", "0"],
    ]
    assert "a-project" not in "\n".join(lines)


def test_the_ledger_states_coverage_only_for_a_short_record(published):
    assert "covered" not in "\n".join(_lines(_data(), None))
    published["watch"]["ledger"].append(_ledger_row("1 week", covered=600.0))
    assert "covered" in "\n".join(_lines(_data(), None))


def test_the_hottest_card_is_shown_and_the_mean_is_not(published):
    text = "\n".join(_lines(_data(), None))
    assert "°C max" in text and "°C mean" not in text
