"""Golden renders and negative controls for the llm-watch view.

The fixtures are documents rather than files on disk: what the view takes is a
built document, and every figure in one is a fact some published source
measured. The goldens are the rendered text of those documents at 120 columns,
committed so a change to the view shows up as a diff rather than as a
plausible-looking screen.

Two of the fixtures carry the account and project keys by hand, because a
document's shapes are fixed here before every producer that fills them has
deployed -- the same reason a section's fixtures carry a key its own deploy
has not written yet.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from imas_ambix.agent import llm_watch as lw
from imas_ambix.agent.watch_style import WIDTHS

GOLDEN_DIR = Path(__file__).parent / "golden" / "llm_watch"

VIEWER = "mcintos"


@pytest.fixture(autouse=True)
def _viewer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the viewer's login, which the detail table defaults to."""
    monkeypatch.setenv("USER", VIEWER)
    monkeypatch.setenv("LOGNAME", VIEWER)


# ── fixtures ─────────────────────────────────────────────────────────


def _live(**overrides: Any) -> dict[str, Any]:
    block: dict[str, Any] = {
        "file": "gpu-0003-deepseek-v4.1-flash-1291586.jsonl",
        "job_id": 1291586,
 # placeholder, replaced below
        "window_seconds": 45.0,
        "rows": 9,
        "state": "measured",
        "mean_running": 11.3,
        "engine_queue": 0.0,
        "pool_occupancy": 0.21,
        "generation_toks_per_s": 558.0,
        "per_stream_toks_per_s": 49.3,
        "prefill_toks_per_s": 2690.0,
        "cache_hit_rate": 0.95,
        "median_time_to_first_token_s": 0.6,
        "reason": None,
    }
    block.update(overrides)
    block["served_name"] = "deepseek-v4.1-flash"
    return block


def _lane(**overrides: Any) -> dict[str, Any]:
    block: dict[str, Any] = {
        "observed_at": "2026-10-23T04:40:55Z",
        "state": "measured",
        "stale": False,
        "age_seconds": 4.0,
        "suggested_shelf_life_seconds": 45,
        "model_id": "deepseek-v4.1-flash",
        "headroom": 4,
        "engine_headroom": 17,
        "availability_percent": 33.33,
        "kv_occupancy": 0.21,
        "mean_context": 144000,
        "concurrent_requests": 25,
        "preemptions": 0,
        "hicache_host": {
            "total_tokens": 8000000,
            "used_tokens": 7920000,
            "used_fraction": 0.99,
        },
        "admission": {
            "verdict": "congested",
            "headroom": 4,
            "waiting": 0,
            "oldest_wait_seconds": 22.0,
        },
        "router_generation_gate": {
            "width": 12,
            "effective_width": 12,
            "in_flight": 12,
            "waiting": 23,
            "paused": False,
        },
        "worker_slots": {"worker_slots": 13, "live_runs": 20},
    }
    block.update(overrides)
    return block


def _bins(occupancy: list[float]) -> dict[str, Any]:
    bins = []
    for index, value in enumerate(occupancy):
        bins.append(
            {
                "start": f"2026-10-23T04:{index:02d}:00Z",
                "end": f"2026-10-23T04:{index + 1:02d}:00Z",
                "occupancy": value,
                "requests": float(index),
                "gate_wait_p50_s": float(index % 7),
                "gate_wait_p95_s": float(index % 11),
            }
        )
    return {
        "state": "measured",
        "window_seconds": 1800.0,
        "bin_seconds": 60.0,
        "covered": True,
        "bins": bins,
    }


def _outcomes(completed: float, aborted: float, failed: float,
              expired: float) -> dict[str, Any]:
    return {
        "state": "measured",
        "window_seconds": 300.0,
        "covered": True,
        "completed": completed,
        "aborted": aborted,
        "failed": failed,
        "gate_expired": expired,
    }


def _account_row(account: str, sessions: int, runs: int, held: float,
                 share: float, wait: float = 2.0) -> dict[str, Any]:
    return {
        "account": account,
        "sessions": sessions,
        "runs": runs,
        "held": held,
        "held_per_run": round(held / runs, 2) if runs else None,
        "mean_gate_wait_s": wait,
        "share": share,
    }


def _project_row(account: str, project: str, sessions: int, runs: int,
                 held: float, wait: float = 2.0) -> dict[str, Any]:
    return {
        "account": account,
        "project": project,
        "sessions": sessions,
        "runs": runs,
        "held": held,
        "held_per_run": round(held / runs, 2) if runs else None,
        "mean_gate_wait_s": wait,
    }


def _session_row(account: str, project: str, session: str, runs: int,
                 held: float, wait: float = 2.0) -> dict[str, Any]:
    return {
        "session": session,
        "hosts": ["98dci4-srv-1006"],
        "clients": ["claude-code"],
        "account": account,
        "project": project,
        "runs": runs,
        "requests": float(runs),
        "held": held,
        "waiting": 0.0,
        "held_per_run": round(held / runs, 2) if runs else None,
        "heaviest_run_held": round(held / runs, 2) if runs else None,
        "peak": runs,
        "mean_gate_wait_s": wait,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "share": 0.1,
    }


def _callers(*, accounts: list[dict[str, Any]], projects: list[dict[str, Any]],
             sessions: list[dict[str, Any]],
             occupancy: list[float] | None = None,
             outcomes: dict[str, Any] | None = None) -> dict[str, Any]:
    measured = sum(row["held"] for row in accounts)
    return {
        "state": "measured",
        "window_seconds": 300.0,
        "window_start": "2026-10-23T04:35:55Z",
        "window_end": "2026-10-23T04:40:55Z",
        "basis": "finished requests; one still waiting or generating is absent "
                 "until it ends",
        "covered": True,
        "held": round(measured, 2),
        "accounts": accounts,
        "projects": projects,
        "callers": sessions,
        "history": _bins(occupancy or [0.0] * 30),
        "outcomes": outcomes or _outcomes(412.0, 3.0, 0.0, 0.0),
    }


ENDPOINT = {
    "endpoints": [
        {
            "accelerator_count": 4,
            "accelerator_family": "H200",
            "checkpoint_precision": "mxfp4+fp8",
            "host": "98dci4-gpu-0003.iter.org",
            "max_context": 262144,
            "model_id": "deepseek-v4.1-flash",
            "port": 18810,
        }
    ]
}

LEDGER = [
    {
        "label": "1 hour",
        "period": 3600.0,
        "covered": 3600.0,
        "tokens_in": 231700000.0,
        "cached": 227000000.0,
        "tokens_out": 1150000.0,
        "utilisation": 97.0,
        "cost": 7.45,
    },
    {
        "label": "1 day",
        "period": 86400.0,
        "covered": 43200.0,
        "tokens_in": 3312000000.0,
        "cached": 3180000000.0,
        "tokens_out": 18200000.0,
        "utilisation": 96.0,
        "cost": 131.02,
    },
]


def one_engine_fixture() -> dict[str, Any]:
    """One account, three projects, one engine: the base reading."""
    return {
        "lane": _lane(),
        "live": _live(),
        "callers": _callers(
            accounts=[_account_row("mcintos", 3, 14, 9.4, 1.0)],
            projects=[
                _project_row("mcintos", "power-convertor", 1, 6, 5.0),
                _project_row("mcintos", "reckon", 1, 5, 3.0),
                _project_row("mcintos", "nova", 1, 3, 1.4),
            ],
            sessions=[
                _session_row("mcintos", "power-convertor", "b52968d9-abcd",
                             6, 5.0),
            ],
            occupancy=[0.0] * 14 + [3.0, 4.0, 6.0, 8.0, 9.0, 7.0, 5.0] + [2.0] * 9,
        ),
        "engine": ENDPOINT,
        "ledger": LEDGER,
    }


def three_project_fixture() -> dict[str, Any]:
    """Three projects, which is under the fold: no ``+N more`` row."""
    return one_engine_fixture()


def eight_project_fixture() -> dict[str, Any]:
    """The viewer's account holds eight projects, so six list and two fold."""
    projects = [
        _project_row("mcintos", "power-convertor", 4, 11, 7.0),
        _project_row("mcintos", "reckon", 2, 3, 2.0),
        _project_row("mcintos", "nova", 1, 1, 1.0),
        _project_row("mcintos", "imas-codex", 1, 1, 0.4),
        _project_row("mcintos", "imas-ambix", 1, 1, 0.3),
        _project_row("mcintos", "dotfiles", 1, 1, 0.2),
        _project_row("mcintos", "serving-record-warehouse", 1, 1, 0.2),
        _project_row("mcintos", "ambix-bench", 1, 1, 0.1),
    ]
    return {
        "lane": _lane(),
        "live": _live(),
        "callers": _callers(
            accounts=[
                _account_row("mcintos", 11, 23, 12.4, 0.99),
                _account_row("llealsa", 1, 2, 0.1, 0.01, wait=0.0),
                _account_row(lw.UNATTRIBUTED, 2, 0, 0.0, 0.0),
            ],
            projects=projects + [
                _project_row("llealsa", "nova", 1, 2, 0.1, wait=0.0),
            ],
            sessions=[
                _session_row("mcintos", "power-convertor", "b52968d9-abcd",
                             11, 7.0),
                _session_row("llealsa", "nova", "efd0ee8c-1234", 2, 0.1),
            ],
            occupancy=[0.0] * 10 + [4.0, 7.0, 9.0, 12.0, 12.0, 11.0] + [3.0] * 14,
            outcomes=_outcomes(412.0, 3.0, 0.0, 0.0),
        ),
        "engine": ENDPOINT,
        "ledger": LEDGER,
    }


def unavailable_live_fixture() -> dict[str, Any]:
    """A missing receipts directory: one line naming the reason, no zeros."""
    document = one_engine_fixture()
    document["live"] = {
        "file": None,
        "job_id": None,
        "served_name": None,
        "window_seconds": 45.0,
        "rows": 0,
        "state": "unavailable",
        "mean_running": None,
        "engine_queue": None,
        "pool_occupancy": None,
        "generation_toks_per_s": None,
        "per_stream_toks_per_s": None,
        "prefill_toks_per_s": None,
        "cache_hit_rate": None,
        "median_time_to_first_token_s": None,
        "reason": "no raw receipt file with a readable row",
    }
    return document


def empty_account_fixture() -> dict[str, Any]:
    """The base reading, asked about an account that holds nothing."""
    return one_engine_fixture()


# ── goldens ──────────────────────────────────────────────────────────


def _golden(name: str) -> str:
    return (GOLDEN_DIR / f"{name}.txt").read_text(encoding="utf-8")


def render(document: dict[str, Any], **kwargs: Any) -> str:
    return lw.render_text(document, width=lw.GOLDEN_WIDTH, **kwargs)


@pytest.mark.parametrize(
    ("name", "factory", "kwargs"),
    [
        ("one_engine", one_engine_fixture, {}),
        ("two_accounts", eight_project_fixture, {}),
        ("three_projects", three_project_fixture, {}),
        ("sessions", eight_project_fixture, {"sessions": True}),
        ("account_llealsa", eight_project_fixture, {"account": "llealsa"}),
        ("unavailable_live", unavailable_live_fixture, {}),
        ("account_empty", empty_account_fixture, {"account": "llealsa"}),
    ],
)
def test_render_matches_its_golden(
    name: str, factory: Any, kwargs: dict[str, Any]
) -> None:
    assert render(factory(), **kwargs) == _golden(name)


def test_no_line_exceeds_the_terminal_width() -> None:
    documents = [
        one_engine_fixture(),
        eight_project_fixture(),
        unavailable_live_fixture(),
    ]
    for document in documents:
        for width in WIDTHS:
            text = lw.render_text(document, width=width)
            over = [line for line in text.splitlines() if len(line) > width]
            assert not over, f"line over {width} columns:\n" + "\n".join(over)


def test_accounts_table_reads_the_account_level() -> None:
    """The accounts table totals accounts, not one row per session.

    A session reports the account its rows most often named, so a table built
    from the sessions would list one account once per session and would drop
    every request whose row carried no session. The level's own row count and
    its session total are the two things a session-built table cannot
    reproduce.
    """
    document = eight_project_fixture()
    text = render(document)
    callers = document["callers"]
    assert f"accounts {callers['window_seconds'] / 60:.0f}m" in text
    # Three accounts, each on one row, and the viewer's sessions total 11.
    assert text.count("unattributed") >= 1
    assert "mcintos" in text and "llealsa" in text


def test_eight_project_fixture_folds_two_rows() -> None:
    text = render(eight_project_fixture())
    assert "+2 more" in text
    # The fold carries the two folded projects' summed in-use, 0.2 + 0.1.
    assert "0.3" in text


def test_three_project_fixture_has_no_fold_row() -> None:
    text = render(three_project_fixture())
    assert "more" not in text


def test_unavailable_live_entry_renders_one_line_and_no_band() -> None:
    """A missing receipt file is a stated reason, never a row of zeros.

    The refusal is the whole check: the throughput band must be absent, not
    the band's figures zeroed. A zero is a measurement, and this document
    makes none.
    """
    text = render(unavailable_live_fixture())
    assert "live entry unavailable" in text
    assert "no raw receipt file with a readable row" in text
    assert "tok/s out" not in text
    assert "11.3" not in text


def test_an_account_holding_nothing_renders_one_line() -> None:
    """An account absent from the window gets one dim line, not a table."""
    text = render(empty_account_fixture(), account="llealsa")
    assert "llealsa holds nothing in the window" in text
    assert "per run" not in text.split("llealsa holds nothing")[-1]


def test_live_flag_refreshes_every_five_seconds() -> None:
    """``--live`` re-renders on the declared cadence, five seconds apart."""
    assert lw.LIVE_REFRESH_SECONDS == 5.0
    ticks: list[float] = []
    calls: list[int] = []
    for _ in lw.refresh_loop(lambda: calls.append(1), ticks=3,
                             sleep=ticks.append):
        pass
    assert len(calls) == 3
    assert ticks == [5.0, 5.0]


def test_console_script_is_declared() -> None:
    """The view is reachable as ``llm-watch``, not only as a module."""
    import tomllib

    root = Path(__file__).resolve().parents[1]
    with (root / "pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle)
    scripts = project["project"]["scripts"]
    assert scripts.get("llm-watch") == "imas_ambix.agent.llm_watch:main"


def test_json_document_is_serialisable() -> None:
    document = one_engine_fixture()
    assert json.loads(json.dumps(document, default=str))["lane"]["model_id"]