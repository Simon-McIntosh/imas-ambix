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
``class``, or a binding assignment at *exporting* scope: a module body, or a
class body that is itself at exporting scope. The test is whether anything
between the binding and the module root opens a function scope -- not whether
the nearest enclosing scope is a class or the module, because a class body
nested inside a function holds attributes on a class no caller can reach, and
so is exactly as local as a plain function variable. The walk therefore goes
all the way to the root and a ``def``, an ``async def`` or a ``lambda``
anywhere on that path disqualifies what it encloses. A module-level
``try``/``except`` body is indented but still module level, which is the shape
an importer-facing fallback takes, so its bindings count. Importing the same
name is not a declaration either, or every future caller would read as a
duplicate.

**Assignment expressions are the one case where a comprehension is not
opaque.** A comprehension opens a scope for its own loop targets, which are
never counted, but a walrus target inside one binds in the scope *containing*
the comprehension: at module level ``pairs = [(_LABEL_RE := index) for index
in range(3)]`` really does export ``_LABEL_RE``, and a rival spelled that way
is a rival. A walrus inside a comprehension inside a function binds in that
function and, like any other function binding, is local.

**What it does not count.** The scan looks for those four forms only, and it
walks statements: a comprehension's own loop target is a plain name binding
inside the comprehension's scope and is not counted. ``_SAMPLE_RE`` used as a
comprehension target is therefore invisible, as it should be, because that
binding is not exported. Shapes likewise invisible, and deliberately not
chased: augmented assignment (``_SAMPLE_RE += ...``), ``except Exception as
parse_metrics``, ``for`` and ``with`` targets, parameter names, and a
``globals()[...]`` subscript. Each is a rewrite no reader of this package has
a reason to write, and the same reasoning bounds them as the walrus: only the
two exact forms above leave an exported name nothing else in this file would
report.

**No single declaration form is trusted alone.** The scan is exercised against
a duplicate tree, an innocent tree, a class body, an indented fallback, a
walrus it must find and a class nested in a function it must not, so a rule
that is wrong in either direction reddens the file rather than the package.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from imas_ambix.agent import bench

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

#: Nodes that open a function scope. A name bound anywhere inside one is local
#: to it, however many class bodies sit between, because nothing outside the
#: function can reach those attributes.
_FUNCTION_SCOPE_NODES = (
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.Lambda,
)

#: Nodes that open a comprehension scope. Their own loop targets bind in it and
#: are never counted, but a walrus target inside one binds in the containing
#: scope, so the walk treats these as transparent for an assignment expression
#: and as opaque for every other form.
_COMPREHENSION_SCOPE_NODES = (
    ast.DictComp,
    ast.GeneratorExp,
    ast.ListComp,
    ast.SetComp,
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


def _is_local(
    node: ast.AST,
    parents: dict[ast.AST, ast.AST],
    *,
    assignment_expression: bool = False,
) -> bool:
    """Whether *node*'s binding is local rather than exported.

    The walk goes all the way to the module root, and a function, an async
    function or a lambda anywhere on the way disqualifies what it encloses.
    Answering at the first ``Module`` or ``ClassDef`` instead is wrong in the
    direction that matters most: it reports a class body nested inside a
    function as exporting what it binds, because the ``def`` sits above the
    ``class`` and is never reached.

    *assignment_expression* selects the walrus rule. A comprehension opens a
    scope for its own loop targets, but a ``:=`` target inside one binds in the
    scope *containing* the comprehension, so a module-level comprehension's
    walrus is exported and the walk steps over comprehensions. Every other form
    considered here is a comprehension-incompatible statement -- a
    comprehension body holds expressions only -- so for those a comprehension
    ancestor means local and the walk stops there.
    """
    parent = parents.get(node)
    while parent is not None:
        if isinstance(parent, _FUNCTION_SCOPE_NODES):
            return True
        if not assignment_expression and isinstance(
            parent, _COMPREHENSION_SCOPE_NODES
        ):
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
    count, with the one exception the walk's walrus flag covers: an assignment
    expression's target binds through a comprehension into the scope containing
    it, so it is exported whenever that scope is the module.
    """
    parents = {
        child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)
    }
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name == symbol and not _is_local(node, parents):
                return True
            continue
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AnnAssign, ast.NamedExpr)):
            targets = [node.target]
        else:
            continue
        # An assignment expression binds through a comprehension into the scope
        # that contains it, so the walrus flag is what keeps a module-level
        # ``[... for ... (_x := ...) ...]`` a declaration.
        if _is_local(
            node, parents, assignment_expression=isinstance(node, ast.NamedExpr)
        ):
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


def test_definition_scan_ignores_a_class_body_inside_a_function(
    tmp_path: Path,
) -> None:
    """A class body nested in a function exports nothing a caller can reach.

    An attribute bound in a class body is exported only if something can name
    the class. When the class is declared inside a function, the class object
    never leaves that class's own scope, so the attribute is as unreachable as
    a plain local: counting it would redden the guard on a module holding no
    second copy of the grammar. Seeing this requires reaching the module root,
    because the ``def`` sits above the ``class`` and a walk that answers at the
    first class body never arrives at it.
    """
    package = tmp_path / "imas_ambix"
    package.mkdir()
    (package / "reader.py").write_text(
        "def build_grammar(text):\n"
        "    class Holder:\n"
        "        _SAMPLE_RE = text.split(',')\n"
        "        SPEC_DECODE_SERIES = {}\n"
        "\n"
        "        def parse_metrics(self):\n"
        "            return self._SAMPLE_RE\n"
        "\n"
        "    return Holder\n",
        encoding="utf-8",
    )

    for symbol in ("_SAMPLE_RE", "SPEC_DECODE_SERIES", "parse_metrics"):
        assert _defining_modules(symbol, package) == set(), symbol


def test_definition_scan_finds_a_walrus_binding_at_module_level(
    tmp_path: Path,
) -> None:
    """A walrus target in a module-level comprehension really does export.

    ``:=`` binds in the scope *containing* the comprehension, and at module
    level that scope is the module: after the comprehension runs the name is an
    attribute of the module. A rival that spells its grammar this way therefore
    holds a second owner, and a rule that treats every comprehension as opaque
    reports the tree as singly-owned.
    """
    package = tmp_path / "imas_ambix"
    package.mkdir()
    (package / "reader.py").write_text(
        "_LABEL_RE = None\n"
        "pairs = [(_SAMPLE_RE := index) for index in range(3)]\n",
        encoding="utf-8",
    )

    assert _defining_modules("_LABEL_RE", package) == {"imas_ambix/reader.py"}
    assert _defining_modules("_SAMPLE_RE", package) == {"imas_ambix/reader.py"}


def test_definition_scan_ignores_comprehension_scoped_bindings(
    tmp_path: Path,
    ) -> None:
    """The other half of the walrus rule: neither shape leaks out.

    A comprehension's own loop target binds in the comprehension's own scope,
    so the same name used as a target is not a module attribute and is not a
    declaration. The same walrus rule also stops at a function: inside ``def``
    the containing scope is the function, so the name never leaves it. Both are
    the reason the rule is a walk rather than a check for the nearest scope
    node.
    """
    package = tmp_path / "imas_ambix"
    package.mkdir()
    (package / "comp_target.py").write_text(
        "pairs = [_SAMPLE_RE for _SAMPLE_RE in range(3)]\n",
        encoding="utf-8",
    )
    (package / "function_walrus.py").write_text(
        "def build(text):\n"
        "    pairs = [(_LABEL_RE := index) for index in range(3)]\n"
        "    return pairs\n",
        encoding="utf-8",
    )

    assert _defining_modules("_SAMPLE_RE", package) == set()
    assert _defining_modules("_LABEL_RE", package) == set()


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
    """The bench reader returns the common reader's snapshot for this scrape.

    The expected mapping is written out literally rather than taken from
    ``engine_metrics.read_metrics`` on the same input: a defect that moved both
    sides together would leave a self-derived expectation green while the value
    the recorder writes is wrong, and the numbers are the whole point of the
    delegation. The values are the counters alone -- a ``_created`` timestamp
    folded into its token total would read ~1.79e9 rather than 9560.
    """
    snapshot = bench._spec_decode_snapshot(_SPEC_DECODE_SCRAPE)

    assert snapshot == {
        "draft_tokens_total": 9560.0,
        "accepted_tokens_total": 4418.0,
        "num_accepted_per_pos": [1423.0, 1108.0],
    }
    assert snapshot["draft_tokens_total"] == 9560.0
    assert snapshot["accepted_tokens_total"] == 4418.0
    assert snapshot["num_accepted_per_pos"] == [1423.0, 1108.0]
    assert snapshot["draft_tokens_total"] < 1.0e6
    assert snapshot["accepted_tokens_total"] < 1.0e6
