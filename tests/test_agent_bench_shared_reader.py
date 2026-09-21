"""The bench reader resolves serving quantities through the shared module.

A ``/metrics`` scrape is read by more than one caller, and each caller that
spells the exposition grammar or an engine's series names for itself is one
more place for the readings to drift apart. The measured cost of that drift
was a recorder that resolved only ``vllm:`` names against an SGLang engine
and wrote a null into fourteen fields of every one of 13,791 rows -- a
non-observation that read as a reading.

So the parser and the family-to-series mapping each have exactly one owner,
and this module asserts it rather than assuming it: the definition scan below
fails the moment a second module declares either -- wherever in that module
the declaration sits. The scan is exercised against a tree that does contain a
duplicate, because a uniqueness check that only ever runs on a unique tree
reports a property it has not tested.
"""

from __future__ import annotations

import ast
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


def _target_names(target: ast.expr) -> set[str]:
    """Names bound by one assignment target, unpacking tuple and list ones."""
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        names: set[str] = set()
        for element in target.elts:
            names |= _target_names(element)
        return names
    return set()


def _declares(tree: ast.AST, symbol: str) -> bool:
    """Whether *tree* declares *symbol*, at any nesting depth.

    A declaration is a ``def``, an ``async def``, a ``class``, or a binding
    assignment of the name, and the walk deliberately does not filter by
    nesting: a rival arrives as a helper-local or inside a ``try``/``except``
    block as readily as at module level, and each of those is as separate an
    owner as a declaration at column 0.
    """
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and node.name == symbol
        ):
            return True
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AnnAssign, ast.NamedExpr)):
            targets = [node.target]
        else:
            continue
        if any(symbol in _target_names(target) for target in targets):
            return True
    return False


def _defining_modules(symbol: str, package: Path = _PACKAGE) -> set[str]:
    """Paths relative to *package*'s parent declaring *symbol* anywhere.

    The module is parsed rather than scanned line by line, so a declaration
    inside a function, a class body or a ``try``/``except`` block counts
    exactly as one at column 0. A line-anchored scan reports a tree whose
    only second declaration is indented as having a single owner, which is
    the shape a rival actually takes: a pattern table compiled inside the
    function that uses it, or a replacement declared in an ``except
    ImportError`` fallback.
    """
    found: set[str] = set()
    for path in sorted(package.rglob("*.py")):
        # Every module in the package is parseable at the scanning
        # interpreter, so a parse error is a real defect and not a file to
        # step over silently.
        tree = ast.parse(path.read_text(encoding="utf-8"))
        if _declares(tree, symbol):
            found.add(path.relative_to(package.parent).as_posix())
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


def test_definition_scan_sees_an_indented_declaration(tmp_path: Path) -> None:
    """A rival nested in a function or an import fallback is still a rival.

    The duplicate this scan exists to catch does not arrive at column 0, and
    both shapes it does arrive in are here: the exposition grammar compiled
    inside the helper that reads with it, and a replacement declared in an
    ``except ImportError`` fallback. A line-anchored scan calls a tree like
    this singly-owned, which is how a second owner survives the check that
    exists to find it. An import of the owner's name is deliberately not a
    declaration, or every future caller would read as a duplicate.
    """
    package = tmp_path / "imas_ambix"
    (package / "agent").mkdir(parents=True)
    (package / "agent" / "engine_metrics.py").write_text(
        "_SAMPLE_RE = re.compile('owner')\n"
        "_LABEL_RE = re.compile('owner')\n"
        "SPEC_DECODE_SERIES = {}\n"
        "SPEC_DECODE_PER_POSITION = ()\n"
        "\n"
        "\n"
        "def parse_metrics(text):\n"
        "    return text\n",
        encoding="utf-8",
    )
    (package / "agent" / "rival_reader.py").write_text(
        "def parse_scrape(text):\n"
        "    import re\n"
        "    _SAMPLE_RE = re.compile('rival')\n"
        "    _LABEL_RE = re.compile('rival')\n"
        "    _bare_metric_name = str\n"
        "    return text\n"
        "\n"
        "\n"
        "try:\n"
        "    from imas_ambix.agent.engine_metrics import parse_metrics\n"
        "except ImportError:\n"
        "\n"
        "    def parse_metrics(text):\n"
        "        return []\n"
        "\n"
        "    SPEC_DECODE_SERIES = {}\n",
        encoding="utf-8",
    )
    (package / "agent" / "caller.py").write_text(
        "from imas_ambix.agent.engine_metrics import parse_metrics\n"
        "\n"
        "\n"
        "def read(text):\n"
        "    return parse_metrics(text)\n",
        encoding="utf-8",
    )

    owner_path = "imas_ambix/agent/engine_metrics.py"
    both = {owner_path, "imas_ambix/agent/rival_reader.py"}
    expected = {
        "_SAMPLE_RE": both,
        "_LABEL_RE": both,
        "parse_metrics": both,
        "SPEC_DECODE_SERIES": both,
        "SPEC_DECODE_PER_POSITION": {owner_path},
        "_bare_metric_name": {"imas_ambix/agent/rival_reader.py"},
    }
    for symbol, modules in expected.items():
        assert _defining_modules(symbol, package) == modules, symbol


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
