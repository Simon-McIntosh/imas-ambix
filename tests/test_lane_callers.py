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

from imas_ambix.agent import lane_callers, request_receipts
from imas_ambix.agent.lane_callers import callers_block, record_blocks
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
    account: str | None = None,
    project: str | None = None,
) -> dict:
    """One receipt row admitted *admitted* seconds after the window opens."""
    stamp = NOW - timedelta(seconds=300) + timedelta(seconds=admitted)
    return {
        "account": account,
        "project": project,
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


def _levels_fixture(tmp_path: Path) -> Path:
    """Two accounts, three projects and one unattributed host.

    ``alpha`` holds 0.4 under ``p1``, 0.1 under ``p2`` and 0.2 with no project
    reported; ``beta`` holds 0.5 under ``p3``; and an unkeyed curl probe from a
    third host holds 0.1. The four figures sum to 1.3 exactly, at 2 s of held
    request per 1 s of the 300 s window.
    """
    rows = [
        _row(
            0, duration=60, session="s-a", run_id="r-a1", account="alpha", project="p1"
        ),
        _row(
            60, duration=60, session="s-a", run_id="r-a2", account="alpha", project="p1"
        ),
        _row(
            120, duration=30, session="s-b", run_id="r-b", account="alpha", project="p2"
        ),
        _row(150, duration=60, session="s-d", run_id="r-d", account="alpha"),
        _row(
            0, duration=150, session="s-c", run_id="r-c", account="beta", project="p3"
        ),
        _row(
            180,
            duration=30,
            host="10.154.100.9",
            agent="curl/7.76.1",
            prompt=None,
        ),
    ]
    return _write(tmp_path, rows)


def test_accounts_and_projects_reconcile_with_the_block_and_with_each_other(
    tmp_path: Path,
) -> None:
    """The levels partition the window: every row is in one account, one project."""
    block = callers_block(_levels_fixture(tmp_path), now=NOW)
    assert block["state"] == "measured"
    assert block["held"] == pytest.approx(1.3)
    assert _by_session(block)  # the per-session list is still there, unchanged

    accounts = {row["account"]: row for row in block["accounts"]}
    assert set(accounts) == {"alpha", "beta", "unattributed"}
    assert sum(row["held"] for row in block["accounts"]) == pytest.approx(
        block["held"], abs=0.01
    )
    assert accounts["alpha"]["held"] == pytest.approx(0.7)
    assert accounts["alpha"]["sessions"] == 3
    assert accounts["alpha"]["runs"] == 4
    assert accounts["alpha"]["held_per_run"] == pytest.approx(0.17)
    assert accounts["alpha"]["share"] == pytest.approx(0.538, abs=0.001)
    assert accounts["beta"]["held"] == pytest.approx(0.5)
    assert accounts["beta"]["sessions"] == 1
    assert accounts["beta"]["held_per_run"] == pytest.approx(0.5)

    projects = {(row["account"], row["project"]): row for row in block["projects"]}
    assert {(a, p) for a, p in projects if p != "unattributed"} == {
        ("alpha", "p1"),
        ("alpha", "p2"),
        ("beta", "p3"),
    }
    assert projects[("alpha", "p1")]["held"] == pytest.approx(0.4)
    assert projects[("alpha", "p1")]["sessions"] == 1
    assert projects[("alpha", "p1")]["held_per_run"] == pytest.approx(0.2)
    assert projects[("alpha", "p2")]["held"] == pytest.approx(0.1)
    assert projects[("beta", "p3")]["held"] == pytest.approx(0.5)
    for account, total in (("alpha", 0.7), ("beta", 0.5), ("unattributed", 0.1)):
        mine = [row["held"] for row in block["projects"] if row["account"] == account]
        assert sum(mine) == pytest.approx(total, abs=0.01)


def test_an_account_reported_without_a_project_lands_in_project_unattributed(
    tmp_path: Path,
) -> None:
    """An account with no project reported is unattributed within that account."""
    block = callers_block(_levels_fixture(tmp_path), now=NOW)
    within = [
        row
        for row in block["projects"]
        if row["account"] == "alpha" and row["project"] == "unattributed"
    ]
    assert len(within) == 1
    assert within[0]["held"] == pytest.approx(0.2)
    # The only other unattributed project is the one under the unattributed
    # account, so no named account but alpha claims a project-less row.
    assert [
        row["account"] for row in block["projects"] if row["project"] == "unattributed"
    ] == ["alpha", "unattributed"]


def test_rows_with_no_account_are_totalled_under_one_unattributed_entry(
    tmp_path: Path,
) -> None:
    """The two unkeyed hosts land in one entry, which lists both of them."""
    rows = [
        _row(
            0, duration=60, session="s-a", run_id="r-a", account="alpha", project="p1"
        ),
        _row(60, duration=30, host="10.154.100.9", agent="curl/7.76.1", prompt=None),
        _row(90, duration=30, host="10.154.100.10", agent="curl/7.76.1", prompt=None),
    ]
    _write(tmp_path, rows)
    block = callers_block(tmp_path, now=NOW)
    accounts = {row["account"]: row for row in block["accounts"]}
    assert set(accounts) == {"alpha", "unattributed"}
    unattributed = accounts["unattributed"]
    assert unattributed["held"] == pytest.approx(0.1 + 0.1)
    assert unattributed["hosts"] == ["10.154.100.10", "10.154.100.9"]
    assert accounts["alpha"]["held"] == pytest.approx(0.2)
    # The account that did report a name carries no host list.
    assert "hosts" not in accounts["alpha"]
    assert [row["session"] for row in block["callers"]] == ["s-a", None, None]


def test_a_row_with_no_account_stays_unattributed_when_its_host_reports_one(
    tmp_path: Path,
) -> None:
    """Identity is what the caller reports, never what its address suggests.

    Both requests here come from the same peer address, and only one of them
    reports an account. The other is a request whose account the record does not
    hold, so it is totalled under ``unattributed`` and never credited to the
    account that happens to share its host.
    """
    rows = [
        _row(
            0,
            duration=60,
            host="10.0.0.5",
            session="s-g",
            run_id="r-g",
            account="gamma",
            project="pg",
        ),
        _row(30, duration=30, host="10.0.0.5", session="s-h", run_id="r-h"),
    ]
    _write(tmp_path, rows)
    block = callers_block(tmp_path, now=NOW)
    accounts = {row["account"]: row for row in block["accounts"]}
    assert set(accounts) == {"gamma", "unattributed"}
    assert accounts["gamma"]["held"] == pytest.approx(0.2)
    assert accounts["unattributed"]["held"] == pytest.approx(0.1)
    assert accounts["unattributed"]["hosts"] == ["10.0.0.5"]
    assert [row["session"] for row in block["callers"]] == ["s-g", "s-h"]


def test_every_session_entry_gains_the_account_and_project_it_reported(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path,
        [
            _row(
                0,
                duration=60,
                session="s-a",
                run_id="r-a",
                account="alpha",
                project="p1",
            ),
            _row(60, duration=60, session="s-b", run_id="r-b"),
        ],
    )
    by_session = _by_session(callers_block(tmp_path, now=NOW))
    assert by_session["s-a"]["account"] == "alpha"
    assert by_session["s-a"]["project"] == "p1"
    assert by_session["s-b"]["account"] == "unattributed"
    assert by_session["s-b"]["project"] == "unattributed"


def _live_fixture(tmp_path: Path) -> Path:
    """Today's lane document and request-record tail, copied verbatim."""
    (tmp_path / "lane.json").write_text(LANE_JSON, encoding="utf-8")
    (tmp_path / "requests.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in RECORD_TAIL), encoding="utf-8"
    )
    return tmp_path


def test_every_base_key_survives_on_a_fixture_copied_from_today(
    tmp_path: Path,
) -> None:
    """The added levels are additions: no base key changes and none disappears."""
    block = callers_block(_live_fixture(tmp_path), now=LIVE_NOW)
    assert block["state"] == "measured"
    assert set(block) - set(BASE_BLOCK) - {"callers"} == {"accounts", "projects"}
    for key, value in BASE_BLOCK.items():
        assert block[key] == value, key
    got = {
        caller["session"] or caller["hosts"][0]: caller for caller in block["callers"]
    }
    for expected in BASE_CALLERS:
        caller = got[expected["session"] or expected["hosts"][0]]
        assert set(caller) - set(expected) == {"account", "project"}
        for key, value in expected.items():
            assert caller[key] == value, key


def test_rows_copied_from_today_are_all_unattributed_with_their_hosts(
    tmp_path: Path,
) -> None:
    """Every row the record held, written before it carried the two keys."""
    block = callers_block(_live_fixture(tmp_path), now=LIVE_NOW)
    assert [row["account"] for row in block["accounts"]] == ["unattributed"]
    (entry,) = block["accounts"]
    assert entry["sessions"] == 3
    assert entry["runs"] == 5
    assert entry["held"] == pytest.approx(0.22)
    assert entry["hosts"] == ["10.154.6.78"]
    assert block["projects"] == [
        {
            "account": "unattributed",
            "project": "unattributed",
            "sessions": 3,
            "runs": 5,
            "held": 0.22,
            "held_per_run": 0.04,
            "mean_gate_wait_s": 0.0,
        }
    ]
    assert {caller["account"] for caller in block["callers"]} == {"unattributed"}


def test_live_panel_carries_the_levels_beside_the_session_list(tmp_path: Path) -> None:
    _live_fixture(tmp_path)
    receipts = tmp_path / "receipts"
    receipts.mkdir()
    payload = live_panel(
        receipts_dir=receipts, lane_path=tmp_path / "lane.json", now=LIVE_NOW
    )
    assert set(payload) == {"lane", "live", "callers"}
    callers = payload["callers"]
    assert callers["state"] == "measured"
    assert {"accounts", "projects", "callers", "held"} <= set(callers)
    assert [row["account"] for row in callers["accounts"]] == ["unattributed"]
    assert len(callers["callers"]) == 3


#: Today's published lane document, copied verbatim from the site the
#: watch document reads. Held here rather than read from the live path so the
#: fixture does not change under the test.
LANE_JSON = "\n".join(
        [
            '{',
            '  "admission": {',
            '    "active_sessions": 5,',
            '    "borrow_reserve": 2,',
            '    "fair_share": 6,',
            '    "headroom": 6,',
            '    "live_runs": 16,',
            '    "new_session_worker_slots": 5,',
            '    "observed_seconds": 900.0,',
            '    "oldest_wait_seconds": null,',
            '    "requests_per_run": 0.391,',
            '    "samples": 469,',
            '    "sessions": {',
            '      "67139f13-acc5-4116-8e4d-ecb63acaac2c": {',
            '        "live_runs": 1,',
            '        "worker_slots": 6',
            '      },',
            '      "704ee605-6100-4771-baae-9ea338ff1992": {',
            '        "live_runs": 2,',
            '        "worker_slots": 6',
            '      },',
            '      "b52968d9-6db7-419c-88fb-9893fefc4514": {',
            '        "live_runs": 3,',
            '        "worker_slots": 6',
            '      },',
            '      "e1b445e1-9530-49a8-a45c-81b39922e373": {',
            '        "live_runs": 3,',
            '        "worker_slots": 6',
            '      },',
            '      "efd0ee8c-9d4e-4fa3-a440-ac05791d4758": {',
            '        "live_runs": 1,',
            '        "worker_slots": 6',
            '      }',
            '    },',
            '    "unkeyed_share": 0.0,',
            '    "verdict": "open",',
            '    "waiting": 0,',
            '    "window_seconds": 900.0,',
            '    "worker_slots": 14,',
            '    "worker_slots_basis": "measured"',
            '  },',
            '  "availability_percent": 50.0,',
            '  "binding_observed": null,',
            '  "concurrent_requests": 26,',
            '  "concurrent_requests_instant": 26,',
            '  "derived_from": {',
            (
                '    "formula": "concurrent_requests = (pool_tokens'
                ' * occupancy_target) // mean_context, capped at 96'
                '; mean_context = MEDIAN over the window of (pool_t'
                'okens * kv_occupancy / running); headroom = concur'
                'rent_requests - running_at_observation",'
            ),
            '    "mean_context": 136000,',
            '    "occupancy_target": 0.9,',
            '    "pool_tokens": 4000000,',
            '    "running_at_observation": 5',
            '  },',
            '  "engine_headroom": 21,',
            '  "headroom": 6,',
            '  "headroom_instant": 21,',
            '  "headroom_is_upper_bound": true,',
            '  "hicache_host": {',
            '    "total_tokens": 8000000,',
            '    "used_fraction": 1.0,',
            '    "used_tokens": 8000000',
            '  },',
            '  "kv_occupancy": 0.17,',
            '  "mean_context": 136000,',
            '  "mean_context_instant": 136000,',
            '  "mean_context_spread": [',
            '    120000,',
            '    136000',
            '  ],',
            '  "model_id": "deepseek-v4.1-flash",',
            '  "observed_at": "2026-10-23T00:02:23Z",',
            '  "pool_tokens": 4000000,',
            '  "preemptions": null,',
            '  "prefix_hit_rate": 0.9925,',
            '  "router_generation_gate": {',
            (
                '    "config_path": "/work/projects/imas_gpu/agents'
                '/control/router-gate.json",'
            ),
            '    "context_estimate": 133696.35277176512,',
            '    "effective_width": 12,',
            '    "enabled": true,',
            '    "in_flight": 6,',
            '    "paused": false,',
            '    "reason": null,',
            '    "wait_seconds": 300.0,',
            '    "waiting": 0,',
            '    "width": 12',
            '  },',
            '  "running": 5,',
            '  "settling": null,',
            (
                '  "settling_caveat": "a reading taken within ~60s '
                'of a dispatch reports the fleet at its lightest; p'
                'refer a pre-dispatch reading",'
            ),
            '  "sizing_reason": "the working context is steady across this window",',
            '  "sizing_verdict": "usable",',
            '  "state": "measured",',
            '  "suggested_shelf_life_seconds": 45,',
            '  "throughput": {',
            '    "aggregate_tokens_per_second": 369.0,',
            '    "mean_tokens_per_second": 80.217,',
            '    "observed_at": "2026-10-23T00:02:23Z",',
            '    "runs": 4.6',
            '  },',
            '  "volatile": false,',
            '  "waiting": 0,',
            '  "window_samples": 5,',
            '  "window_seconds": 150',
            '}',
        ]
) + "\n"

#: The tail of today's request record, copied verbatim. No row carries an
#: account or a project: this is what the record held before it carried them.
RECORD_JSONL = "\n".join(
        [
            (
                '{"timestamp": "2026-10-23T00:02:27Z", "model": "de'
                'epseek-v4.1-flash", "upstream": "http://98dci4-gpu'
                '-0003.iter.org:18810", "dialect": "anthropic", "pr'
                'ompt_tokens": 59886, "cached_prompt_tokens": null,'
                ' "completion_tokens": 162, "reasoning_tokens": nul'
                'l, "time_to_first_token_s": 5.259409, "duration_s"'
                ': 6.919119, "gate_wait_s": 1.4e-05, "caller_hint":'
                ' "10.154.6.78:50890|claude-cli/2.1.309 (external, '
                'sd", "run_id": "r-20261022T235918417436-review-of-'
                'every-surface-lists-open-decisions-365a5debf3cd", '
                '"coordinator_session": "d768318d-3e86-4094-bdc7-3e'
                'd5964852ba", "status": "completed", "sample_fracti'
                'on": 1.0, "sampling_rate_per_s": 1.0}'
            ),
            (
                '{"timestamp": "2026-10-23T00:02:26Z", "model": "de'
                'epseek-v4.1-flash", "upstream": "http://98dci4-gpu'
                '-0003.iter.org:18810", "dialect": "anthropic", "pr'
                'ompt_tokens": 144828, "cached_prompt_tokens": null'
                ', "completion_tokens": 270, "reasoning_tokens": nu'
                'll, "time_to_first_token_s": 2.826957, "duration_s'
                '": 7.875644, "gate_wait_s": 1.4e-05, "caller_hint"'
                ': "10.154.6.78:46528|claude-cli/2.1.309 (external,'
                ' sd", "run_id": "r-20261022T234915855394-callers-r'
                'eport-account", "coordinator_session": "b52968d9-6'
                'db7-419c-88fb-9893fefc4514", "status": "completed"'
                ', "sample_fraction": 1.0, "sampling_rate_per_s": 2'
                '.0}'
            ),
            (
                '{"timestamp": "2026-10-23T00:02:27Z", "model": "de'
                'epseek-v4.1-flash", "upstream": "http://98dci4-gpu'
                '-0003.iter.org:18810", "dialect": "anthropic", "pr'
                'ompt_tokens": 136805, "cached_prompt_tokens": null'
                ', "completion_tokens": 272, "reasoning_tokens": nu'
                'll, "time_to_first_token_s": null, "duration_s": 7'
                '.302625, "gate_wait_s": 1.3e-05, "caller_hint": "1'
                '0.154.6.78:52278|claude-cli/2.1.309 (external, sd"'
                ', "run_id": "r-20261022T235354122944-servers-decla'
                'red-once", "coordinator_session": "b52968d9-6db7-4'
                '19c-88fb-9893fefc4514", "status": "completed", "sa'
                'mple_fraction": 1.0, "sampling_rate_per_s": 3.0}'
            ),
            (
                '{"timestamp": "2026-10-23T00:02:27Z", "model": "de'
                'epseek-v4.1-flash", "upstream": "http://98dci4-gpu'
                '-0003.iter.org:18810", "dialect": "anthropic", "pr'
                'ompt_tokens": 134936, "cached_prompt_tokens": null'
                ', "completion_tokens": 432, "reasoning_tokens": nu'
                'll, "time_to_first_token_s": null, "duration_s": 1'
                '0.008925, "gate_wait_s": 1.5e-05, "caller_hint": "'
                '10.154.6.78:38618|claude-cli/2.1.309 (external, sd'
                '", "run_id": "r-20261022T174957097929-tf-tape-resi'
                'due-cause", "coordinator_session": "e1b445e1-9530-'
                '49a8-a45c-81b39922e373", "status": "completed", "s'
                'ample_fraction": 1.0, "sampling_rate_per_s": 1.0}'
            ),
            (
                '{"timestamp": "2026-10-23T00:02:08Z", "model": "de'
                'epseek-v4.1-flash", "upstream": "http://98dci4-gpu'
                '-0003.iter.org:18810", "dialect": "anthropic", "pr'
                'ompt_tokens": 136625, "cached_prompt_tokens": null'
                ', "completion_tokens": 1668, "reasoning_tokens": n'
                'ull, "time_to_first_token_s": 0.735971, "duration_'
                's": 30.324246, "gate_wait_s": 1.4e-05, "caller_hin'
                't": "10.154.6.78:39456|claude-cli/2.1.309 (externa'
                'l, sd", "run_id": "r-20261022T235023436789-callers'
                '-grouped-by-account", "coordinator_session": "b529'
                '68d9-6db7-419c-88fb-9893fefc4514", "status": "comp'
                'leted", "sample_fraction": 1.0, "sampling_rate_per'
                '_s": 1.0}'
            ),
            (
                '{"timestamp": "2026-10-23T00:02:35Z", "model": "de'
                'epseek-v4.1-flash", "upstream": "http://98dci4-gpu'
                '-0003.iter.org:18810", "dialect": "anthropic", "pr'
                'ompt_tokens": 137188, "cached_prompt_tokens": null'
                ', "completion_tokens": 194, "reasoning_tokens": nu'
                'll, "time_to_first_token_s": null, "duration_s": 2'
                '.98276, "gate_wait_s": 1.5e-05, "caller_hint": "10'
                '.154.6.78:52278|claude-cli/2.1.309 (external, sd",'
                ' "run_id": "r-20261022T235354122944-servers-declar'
                'ed-once", "coordinator_session": "b52968d9-6db7-41'
                '9c-88fb-9893fefc4514", "status": "completed", "sam'
                'ple_fraction": 1.0, "sampling_rate_per_s": 2.0}'
            ),
        ]
) + "\n"
RECORD_TAIL = [
    json.loads(line) for line in RECORD_JSONL.splitlines() if line.strip()
]

#: The moment the fixture was taken at: 2026-10-23T00:02:40+00:00.
LIVE_NOW = datetime(2026, 10, 23, 0, 2, 40, tzinfo=UTC)

#: What the revision before this change emitted for the fixture above, run
#: with the same two files at the same moment. Every key here must survive
#: the added levels unchanged, block-level and per session.
BASE_BLOCK = (
{'state': 'measured',
 'window_seconds': 300.0,
 'window_start': '2026-10-22T23:57:40Z',
 'window_end': '2026-10-23T00:02:40Z',
 'basis': 'finished requests; one still waiting or generating is absent '
          'until it ends',
 'covered': True,
 'held': 0.22}
)

BASE_CALLERS = (
[{'session': 'b52968d9-6db7-419c-88fb-9893fefc4514',
  'hosts': ['10.154.6.78'],
  'clients': ['claude-cli'],
  'runs': 3,
  'requests': 4.0,
  'held': 0.16,
  'waiting': 0.0,
  'held_per_run': 0.05,
  'heaviest_run_held': 0.1,
  'peak': 3,
  'mean_gate_wait_s': 0.0,
  'prompt_tokens': 555446,
  'completion_tokens': 2404,
  'share': 0.741},
 {'session': 'e1b445e1-9530-49a8-a45c-81b39922e373',
  'hosts': ['10.154.6.78'],
  'clients': ['claude-cli'],
  'runs': 1,
  'requests': 1.0,
  'held': 0.03,
  'waiting': 0.0,
  'held_per_run': 0.03,
  'heaviest_run_held': 0.03,
  'peak': 1,
  'mean_gate_wait_s': 0.0,
  'prompt_tokens': 134936,
  'completion_tokens': 432,
  'share': 0.153},
 {'session': 'd768318d-3e86-4094-bdc7-3ed5964852ba',
  'hosts': ['10.154.6.78'],
  'clients': ['claude-cli'],
  'runs': 1,
  'requests': 1.0,
  'held': 0.02,
  'waiting': 0.0,
  'held_per_run': 0.02,
  'heaviest_run_held': 0.02,
  'peak': 1,
  'mean_gate_wait_s': 0.0,
  'prompt_tokens': 59886,
  'completion_tokens': 162,
  'share': 0.106}]
)


def test_live_panel_builds_every_request_record_block_from_one_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One poll of the document walks the record once, not once per block.

    The history strip reaches back half an hour and the callers block and the
    outcomes line reach back five minutes, so a builder that read per block
    would walk the same growing file three times on every poll.
    """
    lane = tmp_path / "lane.json"
    lane.write_text(
        json.dumps({"observed_at": "2026-10-18T18:40:00Z"}), encoding="utf-8"
    )
    _write(tmp_path, [_at(-10, duration=5, session="s", run_id="r")])
    receipts = tmp_path / "receipts"
    receipts.mkdir()
    reads: list[object] = []
    real = lane_callers.recent_receipt_rows

    def counted(location, *, since):
        reads.append(location)
        return real(location, since=since)

    monkeypatch.setattr(lane_callers, "recent_receipt_rows", counted)
    payload = live_panel(receipts_dir=receipts, lane_path=lane, now=NOW)

    assert len(reads) == 1
    assert payload["callers"]["state"] == "measured"
    for name in ("history", "outcomes"):
        assert payload["callers"][name]["state"] == "measured", name
    assert payload["callers"]["history"]["bins"]
    assert payload["callers"]["outcomes"]["completed"] == 1


def test_the_record_read_is_bounded_by_its_byte_ceiling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tail too long to read whole is cut, and every block says so.

    The control arm is the same record under the default ceiling: it is covered
    and its bins are built, so the cut below is the ceiling's doing and not an
    empty record reading as an absence.
    """
    rows = [
        _at(-1000 + 2 * index, duration=1, session="s", run_id="r")
        for index in range(140)
    ]
    _write(tmp_path, rows)

    whole = record_blocks(tmp_path, now=NOW)
    assert whole["callers"]["covered"] is True
    assert whole["history"]["covered"] is True
    assert sum(bin_["occupancy"] for bin_ in whole["history"]["bins"]) > 0

    monkeypatch.setattr(request_receipts, "RECENT_TAIL_FIRST_BYTES", 512)
    monkeypatch.setattr(request_receipts, "RECENT_TAIL_MAX_BYTES", 2048)
    cut = record_blocks(tmp_path, now=NOW)
    assert cut["callers"]["covered"] is False
    assert cut["history"]["covered"] is False
    assert cut["outcomes"]["covered"] is False
    assert "ceiling" in cut["history"]["reason"]


def _at(
    offset: float,
    *,
    duration: float,
    wait: float = 0.0,
    status: str = "completed",
    upstream: str | None = None,
    **kwargs,
) -> dict:
    """One row admitted *offset* seconds before ``NOW`` and ending before it."""
    row = _row(offset + 300, duration=duration, wait=wait, **kwargs)
    row["status"] = status
    if upstream is not None:
        row["upstream"] = upstream
    return row


def _history_fixture(tmp_path: Path) -> Path:
    """A request schedule whose occupancy per minute is computable by hand.

    Bins are one minute wide and end at ``NOW``, so bin ``k`` covers
    ``[NOW - 1800 + 60k, NOW - 1740 + 60k]``. Five rows:

    * ``A`` arrived at -600 s and ended at -290 s, so it fills bins 20-24 and
      ten seconds of bin 25;
    * ``B`` spans -1500 to -1380 s and ``C`` spans -1440 to -1360 s, so bin 6
      holds two requests at once and bin 7 holds C for twenty seconds;
    * four one-second rows are admitted inside bin 28 with waits of 1, 3, 4 and
      8 seconds, which is the bin whose p50 and p95 interpolate between two
      order statistics.
    """
    rows = [
        _at(-590, duration=300, wait=10, session="a", run_id="r-a"),
        _at(-1480, duration=100, wait=20, session="b", run_id="r-b"),
        _at(-1440, duration=80, session="c", run_id="r-c"),
    ]
    for offset, wait in ((-110, 1), (-100, 3), (-90, 4), (-80, 8)):
        rows.append(
            _at(offset, duration=1, wait=wait, session="d", run_id="r-d")
        )
    return _write(tmp_path, rows)


def test_history_bins_hold_the_occupancy_computed_by_hand(tmp_path: Path) -> None:
    block = record_blocks(_history_fixture(tmp_path), now=NOW)["history"]

    assert block["state"] == "measured"
    assert block["window_seconds"] == 1800.0
    assert block["bin_seconds"] == 60.0
    assert len(block["bins"]) == 30
    # 310 s of row A over bins 20-25, 20 s of row C in bin 7, and 20 s of the
    # four one-second rows in bin 28, each divided by the 60 s bin.
    expected = (
        [0.0] * 5
        + [1.0, 2.0, 0.33]
        + [0.0] * 12
        + [1.0, 1.0, 1.0, 1.0, 1.0, 0.17, 0.0, 0.0, 0.33, 0.0]
    )
    assert [bin_["occupancy"] for bin_ in block["bins"]] == expected
    assert [bin_["requests"] for bin_ in block["bins"]][5] == 1.0
    assert [bin_["requests"] for bin_ in block["bins"]][6] == 1.0
    assert [bin_["requests"] for bin_ in block["bins"]][28] == 4.0


def test_history_bin_edges_are_the_minute_boundaries_ending_at_now(
    tmp_path: Path,
) -> None:
    """Bin k starts 1800 - 60k seconds before now, so the strip ends at now."""
    block = record_blocks(_history_fixture(tmp_path), now=NOW)["history"]
    stamp = lambda seconds: (NOW + timedelta(seconds=seconds)).strftime(  # noqa: E731
        "%Y-%m-%dT%H:%M:%SZ"
    )
    assert block["bins"][0]["start"] == stamp(-1800)
    assert [bin_["start"] for bin_ in block["bins"]] == [
        stamp(-1800 + 60 * index) for index in range(30)
    ]
    assert block["bins"][25]["start"] == stamp(-300)
    assert block["bins"][-1]["end"] == stamp(0)


def test_history_waits_are_per_bin_percentiles_of_admitted_requests(
    tmp_path: Path,
) -> None:
    """Bin 28 admitted waits of 1, 3, 4 and 8 s: p50 interpolates to 3.5.

    p95 sits 85% of the way from the fourth-from-last to the last order
    statistic: 4 + 0.85 x (8 - 4) = 7.4. Bin 20 admitted row A's 10 s wait, and
    a bin that admitted nobody reports None rather than zero.
    """
    bins = record_blocks(_history_fixture(tmp_path), now=NOW)["history"]["bins"]
    assert bins[28]["gate_wait_p50_s"] == 3.5
    assert bins[28]["gate_wait_p95_s"] == 7.4
    assert bins[20]["gate_wait_p50_s"] == 10.0
    assert bins[6]["gate_wait_p50_s"] == 0.0
    assert bins[21]["gate_wait_p50_s"] is None
    assert bins[21]["gate_wait_p95_s"] is None


def _outcomes_fixture(tmp_path: Path) -> Path:
    """Five requests ending inside the five minutes, one ending before it."""
    return _write(
        tmp_path,
        [
            _at(-20, duration=10, session="s", run_id="r1"),
            _at(-70, duration=10, session="s", run_id="r2", status="aborted"),
            _at(-140, duration=20, session="s", run_id="r3", status="failed"),
            _at(-400, duration=10, session="s", run_id="r4"),
            _at(
                -280,
                duration=5,
                wait=300,
                session="s",
                run_id="r5",
                status="failed",
                upstream=request_receipts.SELF_ANSWERED_UPSTREAM,
            ),
            _at(
                -200,
                duration=2,
                session="s",
                run_id="r6",
                status="failed",
                upstream=request_receipts.SELF_ANSWERED_UPSTREAM,
            ),
        ],
    )


def test_outcomes_count_the_five_minutes_by_how_each_request_ended(
    tmp_path: Path,
) -> None:
    """A gate expiry is counted once, as an expiry and not also as a failure.

    The refusal the router answered itself with no gate wait (r6) is a failure,
    and the request that ended 390 s ago (r4) is outside the window.
    """
    block = record_blocks(_outcomes_fixture(tmp_path), now=NOW)["outcomes"]

    assert block["state"] == "measured"
    assert block["window_seconds"] == 300.0
    assert block["completed"] == 1
    assert block["aborted"] == 1
    assert block["failed"] == 2
    assert block["gate_expired"] == 1
    assert (
        block["completed"]
        + block["aborted"]
        + block["failed"]
        + block["gate_expired"]
        == 5
    )


def test_a_missing_request_record_leaves_every_record_block_unavailable(
    tmp_path: Path,
) -> None:
    blocks = record_blocks(tmp_path, now=NOW)
    for name in ("callers", "history", "outcomes"):
        assert blocks[name]["state"] == "unavailable", name
        assert "no request record" in blocks[name]["detail"], name
    assert blocks["history"]["bins"] == []
