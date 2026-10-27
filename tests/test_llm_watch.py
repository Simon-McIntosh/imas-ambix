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
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from rich.text import Text

from imas_ambix.agent import live_panel
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
    # The map the publishing side writes, key for key: the key is the engine id,
    # which shortens the cluster component out of the host, and the value is the
    # repository and the commit the engine id was loaded from.
    "checkpoints": {
        "gpu-0003.iter.org:18810":
            "deepseek-ai/DeepSeek-V4.1-Flash-1017@"
            "1bae0e1088993b9b82de08f59de5be5a34e42170",
    },
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


#: The lane document the router published on 2026-10-23T07:44:34Z, copied
#: key for key and value for value. It is here rather than read at test time so
#: the check runs against a document the deployment is known to write, and so a
#: lane that publishes nothing this morning does not turn the check green.
TODAYS_LANE: dict[str, Any] = {
    "observed_at": "2026-10-23T07:44:34Z",
    "state": "measured",
    "model_id": "deepseek-v4.1-flash",
    "suggested_shelf_life_seconds": 45,
    "headroom": 8,
    "engine_headroom": 23,
    "availability_percent": 66.67,
    "kv_occupancy": 0.13,
    "mean_context": 130000,
    "concurrent_requests": 27,
    "preemptions": None,
    "hicache_host": {
        "total_tokens": 8000000,
        "used_tokens": 7999744,
        "used_fraction": 0.999968,
    },
    "admission": {
        "headroom": 8,
        "verdict": "admit",
        "live_runs": 19,
        "observed_seconds": 900.0,
        "requests_per_run": 0.297,
        "worker_slots": 9,
        "fair_share": 6,
        "borrow_reserve": 2,
        "active_sessions": 6,
        "new_session_worker_slots": 9,
    },
    "router_generation_gate": {
        "width": 12,
        "effective_width": 12,
        "in_flight": 4,
        "waiting": 0,
        "paused": False,
        "width_mode": "auto",
        "reason": None,
    },
}


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
        "checkpoints": ENDPOINT["checkpoints"],
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
        "checkpoints": ENDPOINT["checkpoints"],
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


def _written_sources(tmp_path: Path) -> list[str]:
    """The console script's source arguments, pointing at written fixtures.

    A command-line test must not read the live site: the run would then depend
    on which engine happens to be serving, and the fixtures it wrote are the
    only documents the assertions are about.
    """
    lane = tmp_path / "lane.json"
    lane.write_text(json.dumps(_lane()), encoding="utf-8")
    endpoints = tmp_path / "endpoints.json"
    endpoints.write_text(json.dumps(ENDPOINT), encoding="utf-8")
    watch = tmp_path / "watch.json"
    watch.write_text(json.dumps({"published_at": "2026-10-23T04:40:55Z",
                                 "ledger": LEDGER}), encoding="utf-8")
    record = tmp_path / "requests"
    record.mkdir()
    return [
        "--record", str(record),
        "--lane", str(lane),
        "--endpoints", str(endpoints),
        "--watch", str(watch),
    ]


def render(document: dict[str, Any], **kwargs: Any) -> str:
    return lw.render_text(document, width=lw.GOLDEN_WIDTH, **kwargs)


_ACCOUNT_NAMES = {VIEWER, "llealsa", lw.UNATTRIBUTED}


def _accounts_region(text: str) -> str:
    """The accounts table alone: its heading up to the detail table's."""
    return text.split("accounts ", 1)[1].split(f"{VIEWER} 5m", 1)[0]


def _coloured(renderable: Any) -> str:
    """*renderable* as terminal text with its colour codes, unlike a golden."""
    import io

    from rich.console import Console

    buffer = io.StringIO()
    Console(file=buffer, force_terminal=True, color_system="standard",
            width=lw.GOLDEN_WIDTH, highlight=False).print(
        renderable, soft_wrap=False
    )
    return buffer.getvalue()


def _cell_colour(text: str, cell: str) -> str:
    """The colour class the *cell* figure was drawn in."""
    import re

    match = re.search(r"\x1b\[([0-9;]*)m" + re.escape(cell), text)
    assert match, f"{cell} is not drawn in any colour:\n{text}"
    return match.group(1)


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
        # The unattributed account has a row in the accounts table and no
        # project rows at all, so its detail table is the one-line refusal
        # rather than an empty table -- a case the account level exists to make
        # visible and a session-built table would drop entirely.
        ("unattributed", eight_project_fixture, {"account": lw.UNATTRIBUTED}),
    ],
)
def test_render_matches_its_golden(
    name: str, factory: Any, kwargs: dict[str, Any]
) -> None:
    assert render(factory(), **kwargs) == _golden(name)


def test_per_worker_cell_takes_the_serve_panels_colour() -> None:
    """40 tok/s is the per-worker reference, which the serve panel calls green.

    The cell is ramped against a share of what one worker can have, not against
    an absolute rate, so the same figure takes the same colour on both tools.
    Ramping the raw rate instead would call this reading -- the reference rate,
    the best a single stream does on this deployment -- yellow.
    """
    assert lw.PER_WORKER_REF == 40.0
    at_reference = _coloured(lw.throughput_band(_live(per_stream_toks_per_s=40.0)))
    assert "32" in _cell_colour(at_reference, "40.0").split(";")
    # Half the reference is 50 percent of it, under the 60 the ramp calls
    # healthy and over the 30 it calls poor.
    at_half = _coloured(lw.throughput_band(_live(per_stream_toks_per_s=20.0)))
    assert "33" in _cell_colour(at_half, "20.0").split(";")


def test_a_published_lane_document_fills_the_capacity_band() -> None:
    """The four capacity figures reach the band through the lane block.

    The fixture is a copy of the lane document the router published today,
    keys and values unchanged, so the test exercises the copy the live render
    depends on rather than a document shaped to suit the view.
    """
    lane = live_panel.lane_block(
        TODAYS_LANE, now=datetime.fromisoformat(TODAYS_LANE["observed_at"])
    )
    document = one_engine_fixture()
    document["lane"] = lane
    row = render(document).split("kv pool %", 1)[1].splitlines()[1]
    # kv pool is the engine's own counter; the three beside it are the lane
    # document's, and the palette never showed values for any of them before.
    assert row.split() == ["█░░░░", "21", "█████", "100", "130k", "27", "23", "—"]
    # The same render with the copy removed is the state this test exists to
    # catch: every figure the block fails to carry reads as a dash, not a zero.
    for key in ("hicache_host", "mean_context", "concurrent_requests"):
        del lane[key]
    bare = render(document).split("kv pool %", 1)[1].splitlines()[1]
    assert bare.split() == ["█░░░░", "21", "—", "—", "—", "23", "—"]


def test_head_line_names_the_checkpoint_the_served_engine_loaded() -> None:
    """The head line states which weights answered, not just the served name.

    The served name is unchanged across this lane's weights-only rotation, so a
    head line carrying only the name reads identically before and after it. The
    label comes from the endpoint document's map, where the repository's last
    path part and a short commit say which release is loaded.
    """
    assert "DeepSeek-V4.1-Flash-1017@1bae0e10" in render(one_engine_fixture())


def test_checkpoint_lookup_uses_the_engines_registry_id() -> None:
    """The lookup keys on the engine id, never on the host the entry carries.

    The map is written by the side that registers the serve, so its keys are
    built by the registry's engine-id function, which shortens the cluster
    component out of the host. A lookup built from the raw host would still be
    a plausible-looking string and would miss every entry, so this asserts both
    halves: the served engine's entry resolves, and the same entry keyed by the
    raw host does not.
    """
    from imas_ambix.agent.registry import _engine_id

    assert _engine_id("98dci4-gpu-0003.iter.org", 18810) == "gpu-0003.iter.org:18810"
    assert lw.checkpoint_label(one_engine_fixture()) == (
        "DeepSeek-V4.1-Flash-1017@1bae0e10"
    )
    raw = one_engine_fixture()
    recorded = ENDPOINT["checkpoints"]["gpu-0003.iter.org:18810"]
    raw["checkpoints"] = {"98dci4-gpu-0003.iter.org:18810": recorded}
    assert lw.checkpoint_label(raw) is None


def test_no_checkpoints_map_renders_checkpoint_unrecorded() -> None:
    """A document recording no checkpoint says so in yellow, never blank.

    A blank would read as "no checkpoint applies", and the view would look the
    same as one that simply dropped the label.
    """
    document = one_engine_fixture()
    del document["checkpoints"]
    head = lw.head_line(document)
    assert "[yellow]checkpoint unrecorded[/]" in head
    assert "checkpoint unrecorded" in lw.render_text(document)
    assert "33" in _cell_colour(_coloured(Text.from_markup(head)),
                                "checkpoint").split(";")
    # An empty map is the same absence, not a recorded checkpoint under no key.
    document["checkpoints"] = {}
    assert lw.checkpoint_label(document) is None


def test_document_build_reads_the_endpoint_document_the_site_declares(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no path given, the build reads the site's own endpoint document.

    The command line takes an override for a test or a one-off, but the live
    view is launched with none: a build that only read the argument would carry
    no endpoint entry and no checkpoints map at all.
    """
    endpoints = tmp_path / "endpoints.json"
    endpoints.write_text(json.dumps(ENDPOINT), encoding="utf-8")
    monkeypatch.setenv("AMBIX_AGENT_ENDPOINT_DOCUMENT", str(endpoints))
    watch = tmp_path / "watch.json"
    watch.write_text(json.dumps({"ledger": LEDGER}), encoding="utf-8")
    record = tmp_path / "requests"
    record.mkdir()
    document = lw.build_document(
        record_dir=record, lane_path=tmp_path / "lane.json", watch_path=watch
    )
    assert document["checkpoints"] == ENDPOINT["checkpoints"]
    assert document["engine"]["endpoints"] == ENDPOINT["endpoints"]


def test_document_build_reads_the_default_published_watch_document(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no watch path given, the build reads the publisher's own default.

    The live view is launched with no argument, so its ledger block can only
    arrive through the path the publisher writes. The build must resolve that
    path through the resolver the publisher actually offers: naming the output
    of the published ledger was an attribute that does not exist, and reaching
    for it raised on every refresh of the live view.
    """
    endpoints = tmp_path / "endpoints.json"
    endpoints.write_text(json.dumps(ENDPOINT), encoding="utf-8")
    monkeypatch.setenv("AMBIX_AGENT_ENDPOINT_DOCUMENT", str(endpoints))
    published = tmp_path / "watch.json"
    published.write_text(
        json.dumps({"ledger": LEDGER, "published_at": "2026-10-27T09:00:00Z"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(lw.watch_publish, "default_output_path", lambda: published)
    record = tmp_path / "requests"
    record.mkdir()
    document = lw.build_document(
        record_dir=record, lane_path=tmp_path / "lane.json"
    )
    assert document["ledger"] == LEDGER
    assert document["ledger_published_at"] == "2026-10-27T09:00:00Z"


def test_an_engine_other_than_the_served_one_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--engine`` cannot be silently ignored: this view renders one engine.

    Rendering another engine's id over this document's figures would label the
    served engine's numbers with the name of one that did not produce them,
    which is worse than refusing the request.
    """
    sources = _written_sources(tmp_path)
    with pytest.raises(SystemExit) as exit_:
        lw.main(["--engine", "glm-5.3", *sources])
    assert exit_.value.code == lw.REFUSED_ENGINE_STATUS
    # The refusal names the engine that is served -- the id a client sends,
    # which is what the caller mistyped, not the profile slug or file name.
    assert "deepseek-v4.1-flash" in capsys.readouterr().err
    # The served engine's own id is accepted, and so is naming none at all.
    assert lw.main(["--engine", "deepseek-v4.1-flash", *sources]) == 0
    assert lw.main(list(sources)) == 0


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
    # One row per account, in the order the level lists them. A table built
    # from the sessions would carry the viewer's account once per session --
    # the fixture holds two -- and would lose the unattributed row, whose
    # requests named no session at all.
    rows = [line.split()[0] for line in _accounts_region(text).splitlines()
            if line.strip() and line.split()[0] in _ACCOUNT_NAMES]
    assert rows == ["mcintos", "llealsa", "unattributed"]
    # The heading is the document's own window rather than a label the view
    # restates, so a document with a different window heads itself differently.
    wider = eight_project_fixture()
    wider["callers"]["window_seconds"] = 1800.0
    assert "accounts 5m" in text
    assert "accounts 30m" in render(wider)


def test_eight_project_fixture_folds_two_rows() -> None:
    """The fold row carries the folded projects' summed in-use.

    The assertion is on the row itself rather than on the figure appearing
    somewhere in the render: every project row's in-use is a small number, so
    a bare ``"0.3" in text`` would pass with the fold row deleted and 0.3
    elsewhere, which is the state this test exists to catch.
    """
    text = render(eight_project_fixture())
    folds = [line for line in text.splitlines()
             if line.strip().startswith("+2 more")]
    assert len(folds) == 1
    # The row reads ``+2 more`` and the summed in-use, and nothing else: a
    # sum of per-run rates would not be a rate of anything, so the fold
    # carries no sessions, runs or per-run figure.
    assert folds[0].split() == ["+2", "more", "0.3"]


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
