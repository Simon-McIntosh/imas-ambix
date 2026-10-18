"""Tests for the caller breakdown of the lane, and the tail read it rests on.

Every figure is computed from a fixture request record written the way the
router writes it -- one object per line, appended when the request ends -- so
each expected value can be recomputed by hand from the rows in the test.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from imas_ambix.agent import request_receipts
from imas_ambix.agent.lane_callers import callers_block
from imas_ambix.agent.live_panel import live_panel
from imas_ambix.agent.request_receipts import receipt_interval, recent_receipt_rows

NOW = datetime(2026, 10, 18, 18, 40, 0, tzinfo=UTC)


def _row(
    admitted: float,
    *,
    duration: float,
    wait: float = 0.0,
    session: str | None = None,
    run_id: str | None = None,
    host: str = "10.154.6.12",
    agent: str = "claude-cli/2.1.302 (external, sd",
    prompt: int | None = 1000,
    sample_fraction: float = 1.0,
) -> dict:
    """One receipt row admitted *admitted* seconds after the window opens."""
    stamp = NOW - timedelta(seconds=300) + timedelta(seconds=admitted)
    return {
        "timestamp": stamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": "deepseek-v4.1-flash",
        "upstream": "http://98dci4-gpu-0003.iter.org:18810",
        "dialect": "anthropic",
        "prompt_tokens": prompt,
        "cached_prompt_tokens": None,
        "completion_tokens": 100 if prompt is not None else None,
        "reasoning_tokens": None,
        "time_to_first_token_s": 0.5,
        "duration_s": duration,
        "gate_wait_s": wait,
        "caller_hint": f"{host}:40000|{agent}",
        "run_id": run_id,
        "coordinator_session": session,
        "status": "completed",
        "sample_fraction": sample_fraction,
        "sampling_rate_per_s": 1.0,
    }


def _write(directory: Path, rows: list[dict], *, name: str = "requests.jsonl") -> Path:
    """Append *rows* in the order their requests end, as the sink does."""
    ordered = sorted(rows, key=lambda row: receipt_interval(row)[2])
    path = directory / name
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in ordered), encoding="utf-8"
    )
    return path


def _by_session(block: dict) -> dict:
    return {
        caller["session"] or caller["hosts"][0]: caller for caller in block["callers"]
    }


def test_held_is_the_time_averaged_requests_each_caller_kept_at_the_gate(
    tmp_path: Path,
) -> None:
    """Two sessions and an unkeyed host, each figure recomputable by hand.

    Session ``fan`` has one run holding three requests of 100 s at once, each
    having waited 50 s first: 3 x 150 s / 300 s = 1.5 held, 3 x 50 / 300 = 0.5
    of it waiting, and all three overlap, so the peak is 3. Session ``calm``
    has two runs of one 60 s request each, at different times: 2 x 60 / 300 =
    0.4 held, 0.2 per run, peak 1. The unkeyed curl probe holds 3 s: 0.01.
    """
    rows = [
        _row(100, duration=100, wait=50, session="fan", run_id="r-sweep")
        for _ in range(3)
    ]
    rows += [
        _row(10, duration=60, session="calm", run_id="r-a"),
        _row(200, duration=60, session="calm", run_id="r-b"),
        _row(5, duration=3, host="10.154.100.15", agent="curl/7.76.1", prompt=None),
    ]
    _write(tmp_path, rows)
    block = callers_block(tmp_path, now=NOW)

    assert block["state"] == "measured"
    assert block["covered"] is True
    callers = _by_session(block)
    assert [caller["session"] for caller in block["callers"]] == ["fan", "calm", None]

    fan = callers["fan"]
    assert fan["held"] == pytest.approx(1.5)
    assert fan["waiting"] == pytest.approx(0.5)
    assert fan["runs"] == 1
    assert fan["held_per_run"] == pytest.approx(1.5)
    assert fan["heaviest_run_held"] == pytest.approx(1.5)
    assert fan["peak"] == 3
    assert fan["mean_gate_wait_s"] == pytest.approx(50.0)
    assert fan["requests"] == 3
    assert fan["prompt_tokens"] == 3000

    calm = callers["calm"]
    assert calm["held"] == pytest.approx(0.4)
    assert calm["held_per_run"] == pytest.approx(0.2)
    assert calm["peak"] == 1
    assert calm["runs"] == 2

    probe = callers["10.154.100.15"]
    assert probe["session"] is None
    assert probe["clients"] == ["curl"]
    assert probe["held"] == pytest.approx(0.01)
    assert probe["held_per_run"] is None
    assert probe["prompt_tokens"] == 0

    assert block["held"] == pytest.approx(1.91)
    assert sum(caller["share"] for caller in block["callers"]) == pytest.approx(
        1.0, abs=0.002
    )


def test_a_request_straddling_the_window_contributes_only_its_overlap(
    tmp_path: Path,
) -> None:
    """Admitted 30 s before the window opened and relaying 90 s: 60 s inside."""
    _write(tmp_path, [_row(-30, duration=90, session="s", run_id="r")])
    caller = _by_session(callers_block(tmp_path, now=NOW))["s"]
    assert caller["held"] == pytest.approx(60 / 300)
    # Its admission falls before the window, so it is not one of the window's
    # requests and its wait and tokens are not the window's.
    assert caller["requests"] == 0
    assert caller["mean_gate_wait_s"] is None


def test_rows_that_ended_before_the_window_are_not_callers(tmp_path: Path) -> None:
    _write(
        tmp_path,
        [
            _row(-200, duration=20, session="gone", run_id="r-old"),
            _row(10, duration=20, session="here", run_id="r-new"),
        ],
    )
    assert set(_by_session(callers_block(tmp_path, now=NOW))) == {"here"}


def test_a_sampled_row_stands_for_the_requests_the_sink_dropped(tmp_path: Path) -> None:
    """A row kept at fraction 0.5 counts twice in held and requests, not in peak."""
    _write(
        tmp_path,
        [_row(0, duration=30, session="s", run_id="r", sample_fraction=0.5)],
    )
    caller = _by_session(callers_block(tmp_path, now=NOW))["s"]
    assert caller["held"] == pytest.approx(2 * 30 / 300)
    assert caller["requests"] == 2
    assert caller["peak"] == 1


def test_a_directory_with_no_record_is_unavailable_not_idle(tmp_path: Path) -> None:
    block = callers_block(tmp_path, now=NOW)
    assert block["state"] == "unavailable"
    assert "no request record" in block["detail"]
    assert "callers" not in block


def test_an_unfinished_final_line_is_not_a_defect(tmp_path: Path) -> None:
    path = _write(tmp_path, [_row(10, duration=20, session="s", run_id="r")])
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"timestamp": "2026-10-18T18:39')
    assert set(_by_session(callers_block(tmp_path, now=NOW))) == {"s"}


def test_a_malformed_interior_line_is_reported_not_skipped(tmp_path: Path) -> None:
    path = tmp_path / "requests.jsonl"
    row = json.dumps(_row(10, duration=20, session="s", run_id="r"))
    path.write_text(f"{row}\nnot json\n{row}\n", encoding="utf-8")
    block = callers_block(tmp_path, now=NOW)
    assert block["state"] == "unavailable"
    assert "unreadable" in block["detail"]


def test_the_tail_read_grows_until_it_reaches_the_window_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A first read too short to reach the window's start is widened, not trusted."""
    rows = [_row(-100 + 2 * i, duration=1, session="s", run_id="r") for i in range(150)]
    path = _write(tmp_path, rows)
    monkeypatch.setattr(request_receipts, "RECENT_TAIL_FIRST_BYTES", 256)
    monkeypatch.setattr(
        request_receipts, "RECENT_TAIL_MAX_BYTES", path.stat().st_size * 8
    )
    found, covered = recent_receipt_rows(tmp_path, since=NOW - timedelta(seconds=300))
    assert covered is True
    # Rows admitted from 0 s onwards ended inside the window: 100 of the 150.
    assert len(found) == 100


def test_a_tail_cut_at_the_ceiling_says_it_did_not_cover_the_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = [_row(2 * i, duration=1, session="s", run_id="r") for i in range(140)]
    _write(tmp_path, rows)
    monkeypatch.setattr(request_receipts, "RECENT_TAIL_FIRST_BYTES", 512)
    monkeypatch.setattr(request_receipts, "RECENT_TAIL_MAX_BYTES", 2048)
    found, covered = recent_receipt_rows(tmp_path, since=NOW - timedelta(seconds=300))
    assert covered is False
    assert 0 < len(found) < 140
    assert callers_block(tmp_path, now=NOW)["covered"] is False


def test_a_file_not_written_since_the_window_opened_is_not_read(tmp_path: Path) -> None:
    """A legacy file not written since the window opened is skipped unread."""
    legacy = _write(tmp_path, [_row(10, duration=5, session="stale", run_id="r")])
    old = (NOW - timedelta(hours=2)).timestamp()
    os.utime(legacy, (old, old))
    day = _write(
        tmp_path,
        [_row(20, duration=5, session="fresh", run_id="r")],
        name="requests-2026-10-18.jsonl",
    )
    recent = NOW.timestamp()
    os.utime(day, (recent, recent))
    assert set(_by_session(callers_block(tmp_path, now=NOW))) == {"fresh"}


def test_live_panel_reads_the_record_beside_the_lane_document(tmp_path: Path) -> None:
    lane = tmp_path / "lane.json"
    lane.write_text(
        json.dumps({"observed_at": "2026-10-18T18:40:00Z"}), encoding="utf-8"
    )
    _write(tmp_path, [_row(10, duration=20, session="s", run_id="r")])
    receipts = tmp_path / "receipts"
    receipts.mkdir()
    payload = live_panel(receipts_dir=receipts, lane_path=lane, now=NOW)
    assert set(payload) == {"lane", "live", "callers"}
    assert [caller["session"] for caller in payload["callers"]["callers"]] == ["s"]
