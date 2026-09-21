"""The bench reader resolves serving quantities through the shared module.

A ``/metrics`` scrape is read by more than one caller, and each caller that
spells the exposition grammar or an engine's series names for itself is one
more place for the readings to drift apart. The measured cost of that drift
was a recorder that resolved only ``vllm:`` names against an SGLang engine
and wrote a null into fourteen fields of every one of 13,791 rows -- a
non-observation that read as a reading.

So the parser and the family-to-series mapping each have exactly one owner,
and this module asserts it rather than assuming it: the definition scan below
fails the moment a second module declares either. The scan is exercised
against a tree that does contain a duplicate, because a uniqueness check that
only ever runs on a unique tree reports a property it has not tested.
"""

from __future__ import annotations

import re
from pathlib import Path

from imas_ambix.agent import bench, engine_metrics

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PACKAGE = _REPO_ROOT / "imas_ambix"

#: Where the one copy lives. A symbol found in any other module is the
#: duplicate this test exists to catch.
_OWNER = "imas_ambix/agent/engine_metrics.py"

#: Spellings retired when the bench reader was delegated. Their reappearance
#: is the same defect under an old name.
_RETIRED_SYMBOLS = (
    "_parse_prometheus_text",
    "_bare_metric_name",
    "_SPEC_DECODE_COUNTER_NAMES",
    "_SPEC_DECODE_PER_POSITION_NAMES",
)

_SPEC_DECODE_SCRAPE = "\n".join(
    (
        "# TYPE vllm:spec_decode_num_draft_tokens_total counter",
        'vllm:spec_decode_num_draft_tokens_total{engine="0"} 9560.0',
        "vllm:spec_decode_num_draft_tokens_created 1.788616077069351e+09",
        "# TYPE vllm:spec_decode_num_accepted_tokens_total counter",
        'vllm:spec_decode_num_accepted_tokens_total{engine="0"} 4418.0',
        "vllm:spec_decode_num_accepted_tokens_created 1.7886160770694067e+09",
        "vllm:spec_decode_num_accepted_tokens_per_pos_total"
        '{engine="0",position="0"} 1423.0',
        "vllm:spec_decode_num_accepted_tokens_per_pos_total"
        '{engine="0",position="1"} 1108.0',
        "vllm:spec_decode_num_accepted_tokens_per_pos_created"
        '{engine="0",position="0"} 1.788616077069438e+09',
        "",
    )
)


def _defining_modules(symbol: str, package: Path = _PACKAGE) -> set[str]:
    """Paths relative to *package*'s parent declaring *symbol* at top level."""
    definition = re.compile(
        rf"^(?:def\s+{re.escape(symbol)}\b|{re.escape(symbol)}\s*[:=])"
    )
    found: set[str] = set()
    for path in sorted(package.rglob("*.py")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if definition.match(line):
                found.add(path.relative_to(package.parent).as_posix())
                break
    return found


def test_exposition_grammar_has_one_owner() -> None:
    """The exposition regexes and their parser are declared in one module."""
    for symbol in ("_SAMPLE_RE", "_LABEL_RE", "parse_metrics"):
        assert _defining_modules(symbol) == {_OWNER}, symbol


def test_spec_decode_name_tables_have_one_owner() -> None:
    """The speculative-decode series tables are declared in one module."""
    for symbol in ("SPEC_DECODE_SERIES", "SPEC_DECODE_PER_POSITION"):
        assert _defining_modules(symbol) == {_OWNER}, symbol


def test_no_retired_reader_symbol_survives() -> None:
    """The bench reader's superseded privates are gone, not shadowed."""
    for symbol in _RETIRED_SYMBOLS:
        assert _defining_modules(symbol) == set(), symbol


def test_definition_scan_finds_a_duplicate_when_one_exists(tmp_path: Path) -> None:
    """The uniqueness check above can fail: point it at a duplicated tree.

    Without this, a scan that silently matched nothing would report every
    symbol as singly-defined while proving nothing about the package.
    """
    package = tmp_path / "imas_ambix"
    (package / "agent").mkdir(parents=True)
    (package / "agent" / "engine_metrics.py").write_text(
        'parse_metrics = "first"\n_SPEC_DECODE_SERIES = {}\n', encoding="utf-8"
    )
    (package / "agent" / "rival_reader.py").write_text(
        "def parse_metrics(text):\n    return text\n", encoding="utf-8"
    )

    assert _defining_modules("parse_metrics", package) == {
        "imas_ambix/agent/engine_metrics.py",
        "imas_ambix/agent/rival_reader.py",
    }
    assert _defining_modules("_SPEC_DECODE_SERIES", package) == {
        "imas_ambix/agent/engine_metrics.py"
    }


def test_bench_snapshot_is_the_shared_reading() -> None:
    """The bench reader returns the shared module's own snapshot.

    The values are the counters alone: a ``_created`` timestamp folded into
    its token total would read ~1.79e9 rather than 9560.
    """
    snapshot = bench._spec_decode_snapshot(_SPEC_DECODE_SCRAPE)

    assert snapshot == engine_metrics.read_metrics(_SPEC_DECODE_SCRAPE).spec_decode
    assert snapshot["draft_tokens_total"] == 9560.0
    assert snapshot["accepted_tokens_total"] == 4418.0
    assert snapshot["num_accepted_per_pos"] == [1423.0, 1108.0]
    assert snapshot["draft_tokens_total"] < 1.0e6
    assert snapshot["accepted_tokens_total"] < 1.0e6
