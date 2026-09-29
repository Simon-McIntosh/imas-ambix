"""``agent watch --live`` reads the published sources and opens no index.

The live panel exists so a consumer can render the shared lane and the serve's
last 45 s on a short cadence without paying for the index build the full
document does. Two published sources carry every figure it shows: the lane
document the router rewrites, and the recorder's raw receipt tail. So the flag
has one structural promise to keep -- it never constructs a
:class:`~imas_ambix.agent.telemetry_index.TelemetryIndex` and never reaches for
the provider price table -- and these tests make that promise fail loudly
rather than silently if a later edit routes ``--live`` back through the full
document path, which is the path that builds both.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from click.testing import CliRunner

from imas_ambix.agent import cli, provider_prices, telemetry_index, watch

if TYPE_CHECKING:
    import pytest

BASE = datetime(2026, 9, 29, 12, 0, 0, tzinfo=UTC)


def _stamp(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def _row(moment: datetime, *, job_id: int, generation: float, running: float) -> dict:
    """One receipt row in the shape the recorder writes it."""
    return {
        "timestamp": _stamp(moment),
        "job_id": job_id,
        "served_name": "deepseek-v4-1-flash",
        "num_requests_running": running,
        "num_requests_waiting": 0.0,
        "kv_cache_usage_perc": 0.42,
        "engine": {
            "generation_tokens": generation,
            "prompt_tokens": generation * 2.0,
            "uncached_prompt_tokens": generation / 2.0,
        },
    }


def _write_receipts(directory: Path, *, now: datetime) -> Path:
    """A three-row single-run tail inside the trailing window."""
    rows = [
        _row(now - timedelta(seconds=10), job_id=4811, generation=1000, running=4),
        _row(now - timedelta(seconds=5), job_id=4811, generation=1500, running=5),
        _row(now, job_id=4811, generation=2000, running=6),
    ]
    path = directory / "deepseek-v4-1-flash-4811.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _invoke(argv: list[str]):
    return CliRunner().invoke(cli.agent, argv)


#: A lane document carrying the top-level and engine headroom separately, so a
#: copy that dropped one would be visible.
_LANE_DOCUMENT = {
    "observed_at": _stamp(BASE),
    "suggested_shelf_life_seconds": 120,
    "state": "measured",
    "headroom": 11,
    "engine_headroom": 8,
    "headroom_is_upper_bound": False,
    "admission": {
        "headroom": 11,
        "verdict": "open",
        "waiting": 0,
        "oldest_wait_seconds": 0.0,
    },
    "router_generation_gate": {
        "width": 22,
        "effective_width": 22,
        "in_flight": 6,
        "waiting": 0,
        "paused": False,
        "width_mode": "fixed",
        "reason": None,
    },
}


def test_live_json_emits_only_the_two_blocks(tmp_path: Path) -> None:
    """``--json --live`` is the lane and live blocks and nothing else."""
    _write_receipts(tmp_path, now=datetime.now(UTC))
    result = _invoke(["watch", "--json", "--live", "--record", str(tmp_path)])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert set(payload) == {"lane", "live"}


def test_live_path_opens_no_index_and_fetches_no_prices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The live path builds no index and never reaches for the price table.

    Both are made to explode if touched, so the promise is checked by the
    refusal rather than by a figure that happens to agree.
    """

    class _RefuseIndex:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            raise AssertionError("the --live path constructed a TelemetryIndex")

    fetches: list[bool] = []

    def _refuse_fetch(**_kwargs: object) -> None:
        fetches.append(True)
        raise AssertionError("the --live path refreshed the price table")

    monkeypatch.setattr(watch, "TelemetryIndex", _RefuseIndex)
    monkeypatch.setattr(telemetry_index, "TelemetryIndex", _RefuseIndex)
    monkeypatch.setattr(provider_prices, "refresh", _refuse_fetch)

    _write_receipts(tmp_path, now=datetime.now(UTC))
    result = _invoke(["watch", "--json", "--live", "--record", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert fetches == []
    assert set(json.loads(result.output)) == {"lane", "live"}


def test_live_without_json_is_refused(tmp_path: Path) -> None:
    """The flag is JSON-only, so asking for it alone is a usage error."""
    result = _invoke(["watch", "--live", "--record", str(tmp_path)])
    assert result.exit_code != 0
    assert "json" in result.output.lower()


def test_full_json_document_carries_both_blocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The index-derived document carries the lane and live blocks too."""
    monkeypatch.setattr(watch, "_owned_prices", lambda: (None, None))
    _write_receipts(tmp_path, now=datetime.now(UTC))
    result = _invoke(
        [
            "watch",
            "--json",
            "--record",
            str(tmp_path),
            "--index",
            str(tmp_path / "index.sqlite3"),
        ]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert {"record", "ledger", "periods"} <= set(payload)
    assert {"lane", "live"} <= set(payload)


def test_live_document_reads_a_supplied_lane_document(tmp_path: Path) -> None:
    """The live builder copies the lane document's own published fields."""
    lane = tmp_path / "lane.json"
    lane.write_text(json.dumps(_LANE_DOCUMENT), encoding="utf-8")
    payload = watch.live_document(record_dir=tmp_path, now=BASE, lane_path=lane)
    assert set(payload) == {"lane", "live"}
    assert payload["lane"]["headroom"] == 11
    assert payload["lane"]["engine_headroom"] == 8
    assert payload["lane"]["router_generation_gate"]["effective_width"] == 22
    assert payload["lane"]["admission"]["verdict"] == "open"
