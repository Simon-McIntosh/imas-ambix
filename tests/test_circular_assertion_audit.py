"""Tests for the circular-assertion audit over the test suite."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from scripts.circular_assertion_audit import AuditError, load_allowlist, main
from scripts.circular_assertion_audit import scan as run_scan

WIRING = """
from imas_ambix.agent.router import LaneCapacity


def test_published_field_is_fed_by_the_constant():
    derived = {"occupancy_target": 0.9}
    assert derived["occupancy_target"] == LaneCapacity.OCCUPANCY_TARGET
"""

WRITTEN_OUT_LITERAL = """
from imas_ambix.agent.router import LaneCapacity

_CARD_QUERY_FIELDS_LITERAL = ("a", "b")


def test_card_query_fields():
    assert _CARD_QUERY_FIELDS_LITERAL == LaneCapacity.CARD_QUERY_FIELDS
"""

PINNED_ONCE = """
from imas_ambix.agent.router import SELF_ANSWERED_UPSTREAM


def test_the_constant_is_pinned_here():
    assert SELF_ANSWERED_UPSTREAM == "(router)"


def test_later_rows_read_as_prose():
    rows = [{"upstream": "(router)"}]
    assert rows[0]["upstream"] == SELF_ANSWERED_UPSTREAM
"""

LITERAL_IN_CHAIN = """
from imas_ambix.agent.router import LaneCapacity


def test_occupancy_target_is_pinned_by_the_literal():
    assert LaneCapacity.OCCUPANCY_TARGET == 0.9
"""


def _plant(tmp_path: Path, name: str, source: str) -> Path:
    """Write *source* as ``tests/<name>`` under a fresh tree and return the root."""
    root = tmp_path / "project"
    tests = root / "tests"
    tests.mkdir(parents=True, exist_ok=True)
    (tests / name).write_text(textwrap.dedent(source).lstrip(), encoding="utf-8")
    return root


def _allowlist(root: Path, body: str = "", name: str = "allowlist.toml") -> Path:
    path = root / name
    path.write_text(textwrap.dedent(body).lstrip(), encoding="utf-8")
    return path


def _findings(root: Path, allowlist: Path) -> list:
    return run_scan(root, load_allowlist(allowlist))


def test_empty_tree_exits_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "project"
    (root / "tests").mkdir(parents=True)
    allowlist = _allowlist(root)
    assert main(["--root", str(root), "--allowlist", str(allowlist)]) == 0
    assert "TOTAL 0" in capsys.readouterr().out


def test_planted_circular_assertion_is_reported(tmp_path: Path) -> None:
    root = _plant(tmp_path, "test_planted.py", WIRING)
    findings = _findings(root, _allowlist(root))
    assert [finding.symbol for finding in findings] == ["LaneCapacity.OCCUPANCY_TARGET"]


def test_written_out_literal_is_silent(tmp_path: Path) -> None:
    root = _plant(tmp_path, "test_written_out.py", WRITTEN_OUT_LITERAL)
    assert _findings(root, _allowlist(root)) == []


def test_pinned_once_is_silent(tmp_path: Path) -> None:
    root = _plant(tmp_path, "test_pinned.py", PINNED_ONCE)
    assert _findings(root, _allowlist(root)) == []


def test_literal_in_the_chain_is_silent(tmp_path: Path) -> None:
    root = _plant(tmp_path, "test_literal.py", LITERAL_IN_CHAIN)
    assert _findings(root, _allowlist(root)) == []


def test_wiring_is_reported_then_silent_with_an_entry(tmp_path: Path) -> None:
    root = _plant(tmp_path, "test_wiring.py", WIRING)
    assert len(_findings(root, _allowlist(root))) == 1

    entry = _allowlist(
        root,
        """
        [[entry]]
        file = "tests/test_wiring.py"
        symbol = "LaneCapacity.OCCUPANCY_TARGET"
        shape = "wiring"
        reason = "the claim is that the field is fed by the constant, not its value"
        """,
    )
    assert _findings(root, entry) == []


def test_findings_make_the_audit_exit_one(tmp_path: Path) -> None:
    root = _plant(tmp_path, "test_wiring.py", WIRING)
    allowlist = _allowlist(root)
    assert main(["--root", str(root), "--allowlist", str(allowlist)]) == 1


@pytest.mark.parametrize(
    "body",
    [
        """
        [[entry]]
        file = "tests/test_wiring.py"
        symbol = "LaneCapacity.OCCUPANCY_TARGET"
        shape = "wiring"
        reason = ""
        """,
        """
        [[entry]]
        file = "tests/test_wiring.py"
        symbol = "LaneCapacity.OCCUPANCY_TARGET"
        shape = "wiring"
        """,
    ],
    ids=["empty-reason", "missing-reason"],
)
def test_allowlist_entry_without_a_reason_is_refused(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    body: str,
) -> None:
    root = _plant(tmp_path, "test_wiring.py", WIRING)
    allowlist = _allowlist(root, body)
    assert main(["--root", str(root), "--allowlist", str(allowlist)]) != 0
    captured = capsys.readouterr()
    assert "reason" in captured.err
    assert "tests/test_wiring.py" in captured.err
    assert "TOTAL" not in captured.out


def test_allowlist_entry_with_an_unknown_shape_is_refused(tmp_path: Path) -> None:
    allowlist = _allowlist(
        tmp_path,
        """
        [[entry]]
        file = "tests/test_wiring.py"
        symbol = "LaneCapacity.OCCUPANCY_TARGET"
        shape = "looks-fine"
        reason = "a reason is present but the shape is not in the vocabulary"
        """,
    )
    with pytest.raises(AuditError, match="shape"):
        load_allowlist(allowlist)


def test_missing_allowlist_file_is_refused(tmp_path: Path) -> None:
    with pytest.raises(AuditError, match="not readable"):
        load_allowlist(tmp_path / "absent.toml")


def test_unparsable_test_module_is_refused(tmp_path: Path) -> None:
    root = _plant(tmp_path, "test_broken.py", "def test_x(:\n    pass\n")
    with pytest.raises(AuditError, match="cannot parse"):
        _findings(root, _allowlist(root))
