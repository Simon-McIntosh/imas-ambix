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
against a tree that does contain duplicates, because a uniqueness check that
only ever runs on a unique tree reports a property it has not tested, and
against a tree whose local names merely resemble the grammar, because a guard
that reddens on an innocent module is a guard that gets switched off.

**What the scan counts.** A declaration is a ``def``, an ``async def``, a
``class``, or a binding assignment at *exporting* scope -- module level, or a
class body. A module-level ``try``/``except`` body is indented but still
module level, which is the shape an importer-facing fallback takes, so its
bindings count. A name bound inside a function, a lambda or a comprehension is
local to it and is never exportable, so it is not a second owner: an ordinary
local that happens to be called ``_SAMPLE_RE`` holds no copy of the exposition
grammar and no caller can resolve through it. Importing the same name is
likewise not a declaration, or every future caller would read as a duplicate.

**What it does not count.** The scan looks for those four forms only. Shapes
that are still invisible, and deliberately not chased: augmented assignment
(``_SAMPLE_RE += ...``), ``except Exception as parse_metrics``, ``for`` and
``with`` targets, parameter names, and a ``globals()[...]`` subscript. Each is
a rewrite no reader of this package has a reason to write, and the boundary is
recorded here so the next reader knows which side of it a rival has to land on.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from imas_ambix.agent import bench, engine_metrics

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PACKAGE = _REPO_ROOT / "imas_ambix"

#: Where the one copy lives. A symbol found in any other module is the
#: duplicate this test exists to catch.
_OWNER = "imas_ambix/agent/engine_metrics.py"

#: Every live name the shared module owns. The scan iterates this tuple, and a
#: separate test asserts its membership, so dropping a symbol from it reddens
#: the suite instead of silently narrowing what is covered.
_GUARDED_SYMBOLS = (
    "_SAMPLE_RE",
    "_LABEL_RE",
    "parse_metrics",
    "SPEC_DECODE_SERIES",
    "SPEC_DECODE_PER_POSITION",
)

#: Spellings retired when the bench reader was delegated. Their reappearance
#: is the same defect under an old name, and their correct owner count is zero.
_RETIRED_SYMBOLS = (
    "_parse_prometheus_text",
    "_bare_metric_name",
    "_SPEC_DECODE_COUNTER_NAMES",
    "_SPEC_DECODE_PER_POSITION_NAMES",
)

#: Nodes that open a namespace of their own. A name bound inside one is local
#: to it: it cannot be imported, so it cannot second-source a symbol its module
#: exports.
_LOCAL_SCOPE_NODES = (
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.Lambda,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
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


def _is_function_local(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> bool:
    """Whether *node* is bound inside a function, lambda or comprehension.

    The walk outward stops at the first node that opens a namespace of its own
    rather than at the outermost: a method body inside a class body is local to
    the method, and only a ``Module`` or a ``ClassDef`` body exports what it
    binds. Stopping at the first is what makes an ordinary function-local that
    merely resembles a guarded symbol a non-declaration while a module-level
    ``except`` body's binding stays a declaration.
    """
    parent = parents.get(node)
    while parent is not None:
        if isinstance(parent, (ast.Module, ast.ClassDef)):
            return False
        if isinstance(parent, _LOCAL_SCOPE_NODES):
            return True
        parent = parents.get(parent)
    return False


def _declares(tree: ast.AST, symbol: str) -> bool:
    """Whether *tree* declares *symbol* at exporting scope.

    A declaration is a ``def``, an ``async def``, a binding assignment of the
    name at module level or in a class body, and it counts wherever it sits
    inside that scope -- including indented inside a module-level
    ``try``/``except``, which is the shape an importer-facing fallback takes and
    which a scan anchored to column 0 reports as no rival can exist. A binding
    inside a function, a lambda or a comprehension is local to it and does not
    count.
    """
    parents = {
        child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)
    }
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name == symbol and not _is_function_local(node, parents):
                return True
            continue
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AnnAssign, ast.NamedExpr)):
            targets = [node.target]
        else:
            continue
        if _is_function_local(node, parents):
            continue
        if any(symbol in _target_names(target) for target in targets):
            return True
    return False


def _defining_modules(symbol: str, package: Path = _PACKAGE) -> set[str]:
    """Paths relative to *package*'s parent declaring *symbol* for export.

    The module is parsed rather than scanned line by line, so a declaration
    indented inside a module-level ``try``/``except`` counts exactly as one at
    column 0. A line-anchored scan reports a tree whose only second declaration
    is indented as having a single owner, and that is the shape a rival
    actually takes: a replacement declared in an ``except ImportError``
    fallback. Every module in the package is parseable at the scanning
    interpreter, so a parse error is a real defect and is raised rather than
    stepped over -- a file skipped silently would be a hole for exactly the
    declaration this scan exists to find.
    """
    found: set[str] = set()
    for path in sorted(package.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        if _declares(tree, symbol):
            found.add(path.relative_to(package.parent).as_posix())
    return found


def test_guarded_symbol_set_is_the_whole_declared_one() -> None:
    """The scan's own symbol lists are asserted, not assumed.

    Dropping a symbol from either tuple left every other test in this module
    green, so the suite reported full coverage while covering less, and the
    assertion against a second owner of that symbol quietly stopped being made.
    Comparing each tuple to a literal set is what a dropped symbol now fails.
    """
    assert set(_GUARDED_SYMBOLS) == {
        "_SAMPLE_RE",
        "_LABEL_RE",
        "parse_metrics",
        "SPEC_DECODE_SERIES",
        "SPEC_DECODE_PER_POSITION",
    }
    assert set(_RETIRED_SYMBOLS) == {
        "_parse_prometheus_text",
        "_bare_metric_name",
        "_SPEC_DECODE_COUNTER_NAMES",
        "_SPEC_DECODE_PER_POSITION_NAMES",
    }


def test_every_guarded_symbol_has_one_owner() -> None:
    """Every live name is declared in one module, and it is the shared one."""
    for symbol in _GUARDED_SYMBOLS:
        assert _defining_modules(symbol) == {_OWNER}, symbol


def test_no_retired_reader_symbol_survives() -> None:
    """The bench reader's superseded privates are gone, not shadowed."""
    for symbol in _RETIRED_SYMBOLS:
        assert _defining_modules(symbol) == set(), symbol


def test_definition_scan_refuses_a_module_it_cannot_parse(tmp_path: Path) -> None:
    """A module that will not parse is reported, never stepped over.

    The scan walks every module in the package, so a file it skipped silently
    would be a hole a declaration could hide in. Raising is the behaviour the
    scan's own docstring claims, and it is asserted here rather than trusted.
    """
    package = tmp_path / "imas_ambix"
    package.mkdir()
    (package / "truncated.py").write_text("def half_written(:\n", encoding="utf-8")

    with pytest.raises(SyntaxError):
        _defining_modules("parse_metrics", package)


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


def test_definition_scan_sees_an_indented_rival(tmp_path: Path) -> None:
    """A fallback declared inside an indented block is still a rival.

    The duplicate this scan exists to catch arrives indented: a replacement
    declared in an ``except ImportError`` fallback, whose binding sits inside a
    module-level ``try`` and so is indented but still exported. A scan anchored
    to column 0 calls a tree like this singly-owned, which is how a second
    owner survives the check that exists to find it. An import of the owner's
    name is deliberately not a declaration, or every future caller would read
    as a duplicate.
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
        "try:\n"
        "    from imas_ambix.agent.engine_metrics import parse_metrics\n"
        "except ImportError:\n"
        "\n"
        "    def parse_metrics(text):\n"
        "        return []\n"
        "\n"
        "    _SAMPLE_RE = re.compile('rival')\n"
        "    _LABEL_RE = re.compile('rival')\n"
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
    }
    for symbol, modules in expected.items():
        assert _defining_modules(symbol, package) == modules, symbol

    assert "imas_ambix/agent/caller.py" not in _defining_modules(
        "parse_metrics", package
    )


def test_definition_scan_ignores_a_local_named_like_a_symbol(tmp_path: Path) -> None:
    """A name local to a function is not a second owner.

    A function that unpacks a row into a variable it happens to call
    ``_SAMPLE_RE`` holds no copy of the exposition grammar: the name is local
    to that function, is never exported, and no caller can resolve through it.
    Counting it would redden the guard on an innocent module, and a guard that
    reddens on innocent modules is a guard that gets switched off -- a module
    whose only resemblance to a rival is a local variable name has to stay
    clean.
    """
    package = tmp_path / "imas_ambix"
    package.mkdir()
    (package / "reader.py").write_text(
        "def split_labels(text):\n"
        "    _SAMPLE_RE = text.split(',')\n"
        "    _bare_metric_name = str\n"
        "    return _SAMPLE_RE, _bare_metric_name\n"
        "\n"
        "\n"
        "class Collector:\n"
        "    def collect(self, text):\n"
        "        _LABEL_RE = text.split('=')\n"
        "        return _LABEL_RE\n",
        encoding="utf-8",
    )

    for symbol in ("_SAMPLE_RE", "_LABEL_RE", "_bare_metric_name", "parse_metrics"):
        assert _defining_modules(symbol, package) == set(), symbol


def test_definition_scan_counts_a_class_body_binding(tmp_path: Path) -> None:
    """A class-body binding is exported, so it is a declaration.

    The scope rule keeps module bodies and class bodies and drops function
    bodies; this is the half of it that still counts, and without it a rival
    could hide its table in a class attribute and stay invisible.
    """
    package = tmp_path / "imas_ambix"
    package.mkdir()
    (package / "reader.py").write_text(
        "class Grammar:\n"
        "    SPEC_DECODE_SERIES = {}\n"
        "    _SAMPLE_RE = None\n",
        encoding="utf-8",
    )

    assert _defining_modules("SPEC_DECODE_SERIES", package) == {
        "imas_ambix/reader.py"
    }
    assert _defining_modules("_SAMPLE_RE", package) == {"imas_ambix/reader.py"}


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
