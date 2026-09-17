"""Tests for the Decode batch log harvester."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from imas_ambix.agent.decode_bins import (
    ASCENDING,
    DESCENDING,
    FLAT,
    harvest_decode_log,
    parse_decode_lines,
)

SPOT_LOG = Path("/home/ITER/mcintos/Code/imas-ambix/deepseek-v4-1-flash-1272783.log")


def _decode_line(
    *,
    second: int = 0,
    width: int,
    accept_length: float = 3.5,
    accept_rate: float = 0.5,
    generation_rate: float = 110.0,
) -> str:
    return (
        f"[2026-09-17 11:37:{second:02d} TP0 EP0] Decode batch, "
        f"#running-req: {width}, #full token: 7936, full token usage: 0.00, "
        f"#swa token: 512, swa token usage: 0.00, "
        f"accept len: {accept_length}, accept rate: {accept_rate}, "
        f"cuda graph: True, gen throughput (token/s): {generation_rate}, "
        f"#queue-req: 0"
    )


def _write_log(tmp_path: Path, lines: list[str]) -> Path:
    log = tmp_path / "serve.log"
    log.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
    return log


def test_rising_trajectory_labels_intervals_ascending(tmp_path: Path) -> None:
    log = _write_log(
        tmp_path,
        [
            _decode_line(second=1, width=1),
            _decode_line(second=2, width=2),
            _decode_line(second=3, width=3),
            _decode_line(second=4, width=4),
        ],
    )

    harvest = harvest_decode_log(log)

    assert [interval.width for interval in harvest.intervals] == [1, 2, 3, 4]
    assert [interval.limb for interval in harvest.intervals] == [
        None,
        ASCENDING,
        ASCENDING,
        ASCENDING,
    ]


def test_falling_trajectory_labels_intervals_descending(tmp_path: Path) -> None:
    log = _write_log(
        tmp_path,
        [
            _decode_line(second=1, width=6),
            _decode_line(second=2, width=4),
            _decode_line(second=3, width=2),
        ],
    )

    harvest = harvest_decode_log(log)

    assert [interval.limb for interval in harvest.intervals] == [
        None,
        DESCENDING,
        DESCENDING,
    ]


def test_flat_trajectory_labels_repeats_flat_and_leaves_the_first_unknown(
    tmp_path: Path,
) -> None:
    log = _write_log(
        tmp_path,
        [
            _decode_line(second=1, width=3),
            _decode_line(second=2, width=3),
            _decode_line(second=3, width=3),
        ],
    )

    harvest = harvest_decode_log(log)

    assert [interval.limb for interval in harvest.intervals] == [None, FLAT, FLAT]


def test_trajectory_turns_within_one_log(tmp_path: Path) -> None:
    log = _write_log(
        tmp_path,
        [
            _decode_line(second=1, width=2),
            _decode_line(second=2, width=4),
            _decode_line(second=3, width=4),
            _decode_line(second=4, width=3),
        ],
    )

    harvest = harvest_decode_log(log)

    assert [interval.limb for interval in harvest.intervals] == [
        None,
        ASCENDING,
        FLAT,
        DESCENDING,
    ]


def test_malformed_decode_line_is_reported_with_its_field_and_line_number(
    tmp_path: Path,
) -> None:
    bad = (
        "[2026-09-17 11:37:02 TP0 EP0] Decode batch, #running-req: 5, "
        "#full token: 7936, full token usage: 0.00, "
        "accept len: 3.40, accept rate: 0.48, cuda graph: True, #queue-req: 0"
    )
    log = _write_log(
        tmp_path,
        [
            _decode_line(second=1, width=4),
            bad,
            _decode_line(second=3, width=6),
        ],
    )

    harvest = harvest_decode_log(log)

    assert [interval.width for interval in harvest.intervals] == [4, 6]
    assert len(harvest.malformed) == 1
    rejected = harvest.malformed[0]
    assert rejected.line_number == 2
    assert "generation rate" in rejected.reason
    assert rejected.text == bad


def test_malformed_line_is_reported_rather_than_silently_absent(tmp_path: Path) -> None:
    bad = "[2026-09-17 11:37:01 TP0 EP0] Decode batch, accept len: 3.40"
    log = _write_log(tmp_path, [bad, _decode_line(second=2, width=1)])

    harvest = harvest_decode_log(log)

    assert len(harvest.intervals) == 1
    assert harvest.to_dict()["malformed"] == [
        {
            "line_number": 1,
            "reason": "unreadable running width, accept rate, generation rate",
            "text": bad,
        }
    ]


def test_prefill_lines_carry_running_width_and_are_not_decode_intervals(
    tmp_path: Path,
) -> None:
    prefill = (
        "[2026-09-17 11:37:01 TP0 EP0] Prefill batch, #new-seq: 1, "
        "#new-token: 3072, #cached-token: 0, full token usage: 0.00, "
        "swa token usage: 0.00, #running-req: 7, #queue-req: 0, "
        "#pending-token: 0, cuda graph: False, input throughput (token/s): 457.89"
    )
    log = _write_log(tmp_path, [prefill, _decode_line(second=2, width=1)])

    harvest = harvest_decode_log(log)

    assert [interval.width for interval in harvest.intervals] == [1]
    assert harvest.malformed == ()
    assert harvest.lines_read == 2


def test_progress_rewrites_split_on_carriage_return_count_as_one_line(
    tmp_path: Path,
) -> None:
    progress = (
        "Multi-thread loading shards:   0% Completed | 0/48 [00:00<?, ?it/s]\r"
        "Multi-thread loading shards:   2% Completed | 1/48 [00:00<00:25]\r"
        "Multi-thread loading shards:   6% Completed | 3/48 [00:04<01:09]"
    )
    log = _write_log(tmp_path, [progress, _decode_line(second=1, width=2)])

    harvest = harvest_decode_log(log)

    assert [interval.width for interval in harvest.intervals] == [2]
    assert harvest.lines_read == 2
    assert harvest.malformed == ()


def test_carriage_return_terminated_decode_line_still_parses(tmp_path: Path) -> None:
    log = tmp_path / "serve.log"
    log.write_bytes((_decode_line(second=1, width=3) + "\r\n").encode("utf-8"))

    harvest = harvest_decode_log(log)

    assert [interval.width for interval in harvest.intervals] == [3]
    assert harvest.intervals[0].timestamp == "2026-09-17 11:37:01"
    assert harvest.lines_read == 1


def test_width_summary_reports_medians_overall_and_split_by_limb(
    tmp_path: Path,
) -> None:
    log = _write_log(
        tmp_path,
        [
            _decode_line(second=1, width=2, generation_rate=200.0, accept_length=4.0),
            _decode_line(second=2, width=4, generation_rate=240.0, accept_length=4.2),
            _decode_line(second=3, width=2, generation_rate=260.0, accept_length=3.0),
            _decode_line(second=4, width=2, generation_rate=300.0, accept_length=3.4),
        ],
    )

    report = harvest_decode_log(log).to_dict()

    assert set(report["widths"]) == {"2", "4"}
    width_two = report["widths"]["2"]
    assert width_two["intervals"] == 3
    assert width_two["generation_rate_median"] == pytest.approx(260.0)
    assert width_two["accept_length_median"] == pytest.approx(3.4)
    assert width_two["limbs"] == {
        "unknown": {
            "intervals": 1,
            "generation_rate_median": pytest.approx(200.0),
            "accept_length_median": pytest.approx(4.0),
        },
        "descending": {
            "intervals": 1,
            "generation_rate_median": pytest.approx(260.0),
            "accept_length_median": pytest.approx(3.0),
        },
        "flat": {
            "intervals": 1,
            "generation_rate_median": pytest.approx(300.0),
            "accept_length_median": pytest.approx(3.4),
        },
    }
    assert report["widths"]["4"]["limbs"]["ascending"]["intervals"] == 1


def test_report_is_json_serialisable(tmp_path: Path) -> None:
    log = _write_log(tmp_path, [_decode_line(second=1, width=1)])

    json.dumps(harvest_decode_log(log).to_dict())


def test_parser_keeps_interval_order_so_limbs_can_be_derived() -> None:
    parsed = parse_decode_lines(
        [_decode_line(second=index, width=index) for index in range(1, 4)]
    )

    assert [interval.timestamp for interval in parsed.intervals] == [
        "2026-09-17 11:37:01",
        "2026-09-17 11:37:02",
        "2026-09-17 11:37:03",
    ]
    assert [interval.limb for interval in parsed.intervals] == [None, None, None]


@pytest.mark.skipif(not SPOT_LOG.exists(), reason="serve log absent on this host")
def test_harvest_reproduces_confirmed_single_width_spot_values() -> None:
    harvest = harvest_decode_log(SPOT_LOG)
    width_one = next(summary for summary in harvest.widths if summary.width == 1)

    assert width_one.intervals >= 30
    assert 100.0 <= width_one.generation_rate_median <= 118.0
    assert 3.3 <= width_one.accept_length_median <= 3.8
    assert harvest.malformed == ()
