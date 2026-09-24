"""Audit the test suite for circular assertions.

WHAT
----
A circular assertion is a test that asserts a produced value equals the very
constant the code under test reads. Change the constant and both sides of the
``==`` move together, so the test stays green and reports that the behaviour
was checked when it was not. The defect is invisible to a reader holding both
sides of the comparison at once, which is exactly what a focused review does;
only a check that resolves the two sides *mechanically* sees it.

This module walks every ``.py`` file under ``tests/``, parses each with
:mod:`ast`, resolves which names each module binds locally and which it imports
from the package under test, and reports an ``assert`` whose all-``==``
comparison chain is anchored to nothing:

  * no operand is a literal (or a literal container) in the chain;
  * no operand is bound to a literal inside the test module itself; and
  * at least one operand is a bare reference into the package under test.

Three shapes look identical to a scanner and are sound, so they are silent
without needing an allowlist entry:

  * *written-out literal* -- the operand is a single name bound to a literal
    in the test module (``_FIELDS = ("a", "b")``), so the comparison does pin
    the value;
  * *pinned once* -- the imported constant is asserted against a literal
    somewhere else in the same file, so every later reference to it reads as
    prose rather than as a check;
  * *literal in the chain* -- some operand is written out directly.

The remaining shape, *wiring*, is not a value claim -- it says the published
field is fed by that constant, and pinning a number there would assert the
wrong thing and churn on every retune. It cannot be told from a defect by this
or any scanner, so it is recorded per site in the allowlist with the reason.

The allowlist sits beside this module in
:file:`circular_assertion_audit.toml`, schema ``[[entry]]`` with ``file``,
``symbol``, ``shape`` and ``reason``; ``shape`` is one of ``wiring``,
``pinned-once`` or ``written-out-literal``. An entry with no reason is a
failure of the guard rather than an exemption from it: the reason is the thing
a later reader checks.

WHY
---
Only the classification is hard: the scan that reports every ``assert a == b``
touching an imported name finds more sites by an order of magnitude than the
set worth looking at, because it swallows both sound shapes. So the guard is an AST
pass from the start -- recovering the test module's own bindings is what tells a
written-out literal apart from a circular reference, and that cannot be
retrofitted onto a regular expression.

Usage::

    uv run python -m scripts.circular_assertion_audit [--root PATH] [--allowlist PATH]

Exit status is 0 when no unallowlisted finding remains, 1 when findings exist,
and 2 when the allowlist or a scanned file cannot be read.
"""

from __future__ import annotations

import argparse
import ast
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

PACKAGE = "imas_ambix"
TESTS_DIRNAME = "tests"
ALLOWLIST_DEFAULT = Path(__file__).resolve().with_suffix(".toml")
REPO_ROOT_DEFAULT = Path(__file__).resolve().parents[1]

SHAPES = ("wiring", "pinned-once", "written-out-literal")

_LITERAL_CONTAINERS = (ast.Tuple, ast.List, ast.Set)
_FOLDING_BINOPS = (
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
)
_FOLDING_UNARYOPS = (ast.USub, ast.UAdd, ast.Invert)


class AuditError(Exception):
    """The audit cannot run: an unreadable allowlist or an unparsable module."""


@dataclass(frozen=True)
class Finding:
    """One unallowlisted comparison chain, identified by file, line and symbol."""

    path: str
    line: int
    symbol: str
    comparison: str


@dataclass(frozen=True)
class AllowlistEntry:  # noqa: N801
    """A recorded decision to keep one comparison chain as sound."""

    file: str
    symbol: str
    shape: str
    reason: str


def _is_literal(node: ast.expr) -> bool:
    """Whether *node* is written-out data rather than a reference to a value."""
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, _LITERAL_CONTAINERS):
        return all(_is_literal(element) for element in node.elts)
    if isinstance(node, ast.Dict):
        return all(
            key is not None and _is_literal(key) and _is_literal(value)
            for key, value in zip(node.keys, node.values, strict=True)
        )
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, _FOLDING_UNARYOPS):
        return _is_literal(node.operand)
    if isinstance(node, ast.BinOp) and isinstance(node.op, _FOLDING_BINOPS):
        return _is_literal(node.left) and _is_literal(node.right)
    return False


def _dotted(node: ast.expr) -> str | None:
    """The ``a.b.c`` spelling of an attribute chain rooted at a bare name."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value)
        return f"{base}.{node.attr}" if base is not None else None
    return None


def _base_name(node: ast.expr) -> str | None:
    """The leftmost ``Name`` of an attribute, subscript or call chain."""
    current: ast.AST = node
    while True:
        if isinstance(current, ast.Name):
            return current.id
        if isinstance(current, (ast.Attribute, ast.Subscript)):
            current = current.value
        elif isinstance(current, ast.Call):
            current = current.func
        else:
            return None


def _imported_names(tree: ast.AST) -> set[str]:
    """Local names under which *tree* imports the package under test."""
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == PACKAGE or module.startswith(f"{PACKAGE}."):
                for alias in node.names:
                    imported.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == PACKAGE or alias.name.startswith(f"{PACKAGE}."):
                    imported.add(alias.asname or alias.name.split(".", 1)[0])
    return imported


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


def _bound_literals(tree: ast.AST) -> set[str]:
    """Names *tree* binds to a written-out literal, at any scope.

    A local binding to a literal anchors a chain exactly as a written-out
    operand does: ``expected = "(router)"`` followed by ``assert x == expected``
    pins the value, and the test can fail on it.
    """
    bound: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            value = node.value
            targets = node.targets
        elif isinstance(node, (ast.AnnAssign, ast.NamedExpr)):
            value = node.value
            targets = [node.target]
        else:
            continue
        if _is_literal(value):
            for target in targets:
                bound |= _target_names(target)
    return bound


def _pinned_references(tree: ast.AST, imported: set[str]) -> set[str]:
    """Package references *tree* asserts against a literal, anywhere in the file.

    Pinning a literal exactly once and referring to the constant by name
    afterwards is the shape that makes the later references read as prose; the
    assertion that holds the value is the written-out one.
    """
    pinned: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assert):
            continue
        for compare in ast.walk(node.test):
            if not isinstance(compare, ast.Compare):
                continue
            operands = [compare.left, *compare.comparators]
            if not any(_is_literal(operand) for operand in operands):
                continue
            for operand in operands:
                if _is_literal(operand):
                    continue
                name = _dotted(operand)
                if name is not None and _base_name(operand) in imported:
                    pinned.add(name)
    return pinned


def _is_anchored(node: ast.expr, anchored: set[str]) -> bool:
    """Whether *node* names something this test module pins with a literal.

    A bare name is looked up directly; an attribute chain is looked up by its
    dotted spelling, so a pin written against ``LaneCapacity.OCCUPANCY_TARGET``
    anchors a later reference spelled the same way.
    """
    if isinstance(node, ast.Name):
        return node.id in anchored
    dotted = _dotted(node)
    return dotted is not None and dotted in anchored


def _chain_finding(
    compare: ast.Compare,
    imported: set[str],
    anchored: set[str],
) -> tuple[str, str] | None:
    """The ``(symbol, comparison)`` a chain reports, or ``None`` if it is sound."""
    if not compare.ops or not all(isinstance(op, ast.Eq) for op in compare.ops):
        return None
    operands = [compare.left, *compare.comparators]
    if len(operands) < 2:
        return None
    if any(_is_literal(operand) for operand in operands):
        return None
    references = [
        operand
        for operand in operands
        if (base := _base_name(operand)) is not None
        and base in imported
        and _dotted(operand) is not None
    ]
    if not references:
        return None
    for operand in operands:
        if _is_anchored(operand, anchored):
            return None
    symbol = min(
        name
        for operand in references
        if (name := (_dotted(operand) or _base_name(operand))) is not None
    )
    return symbol, ast.unparse(compare)


def load_allowlist(path: Path) -> tuple[AllowlistEntry, ...]:
    """Read and validate the allowlist, refusing any entry that is not complete."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise AuditError(f"allowlist not readable: {path}: {exc}") from exc
    try:
        raw = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise AuditError(f"allowlist {path} is not valid TOML: {exc}") from exc

    tables = raw.get("entry", [])
    if not isinstance(tables, list):
        raise AuditError(f"allowlist {path}: 'entry' must be an array of tables")

    entries: list[AllowlistEntry] = []
    for index, table in enumerate(tables):
        label = f"allowlist {path}: entry {index}"
        if not isinstance(table, dict):
            raise AuditError(f"{label} is not a table")
        file = str(table.get("file", "")).strip()
        symbol = str(table.get("symbol", "")).strip()
        shape = str(table.get("shape", "")).strip()
        reason = str(table.get("reason", "")).strip()
        named = f"{label} (file={file or '?'}, symbol={symbol or '?'})"
        if not file:
            raise AuditError(f"{named} has no 'file'")
        if not symbol:
            raise AuditError(f"{named} has no 'symbol'")
        if not reason:
            raise AuditError(f"{named} has no 'reason'; an exemption must state why")
        if shape not in SHAPES:
            raise AuditError(f"{named} has shape {shape!r}, expected one of {SHAPES}")
        entries.append(
            AllowlistEntry(file=file, symbol=symbol, shape=shape, reason=reason)
        )
    return tuple(entries)


def scan(root: Path, allowlist: Sequence[AllowlistEntry] = ()) -> list[Finding]:
    """Every unallowlisted finding under ``root/tests``, ordered by file and line."""
    allowed = {(entry.file, entry.symbol) for entry in allowlist}
    tests_root = Path(root) / TESTS_DIRNAME
    if not tests_root.is_dir():
        return []

    findings: list[Finding] = []
    for path in sorted(tests_root.rglob("*.py")):
        relative = path.relative_to(root).as_posix()
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError) as exc:
            raise AuditError(f"{relative}: cannot parse: {exc}") from exc
        imported = _imported_names(tree)
        if not imported:
            continue
        anchored = _bound_literals(tree) | _pinned_references(tree, imported)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assert):
                continue
            for compare in ast.walk(node.test):
                if not isinstance(compare, ast.Compare):
                    continue
                found = _chain_finding(compare, imported, anchored)
                if found is None:
                    continue
                symbol, comparison = found
                if (relative, symbol) in allowed:
                    continue
                findings.append(
                    Finding(
                        path=relative,
                        line=compare.lineno,
                        symbol=symbol,
                        comparison=comparison,
                    )
                )
    findings.sort(key=lambda finding: (finding.path, finding.line, finding.symbol))
    return findings


def main(argv: Sequence[str] | None = None) -> int:
    """Run the audit; print one line per finding plus the total."""
    parser = argparse.ArgumentParser(
        prog="circular_assertion_audit",
        description="Report test assertions whose equality chain has no anchor.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=REPO_ROOT_DEFAULT,
        help="repository root holding tests/ (default: this script's repository)",
    )
    parser.add_argument(
        "--allowlist",
        type=Path,
        default=ALLOWLIST_DEFAULT,
        help="TOML allowlist (default: scripts/circular_assertion_audit.toml)",
    )
    args = parser.parse_args(argv)

    try:
        allowlist = load_allowlist(args.allowlist)
        findings = scan(args.root, allowlist=allowlist)
    except AuditError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    for finding in findings:
        print(f"{finding.path}:{finding.line}: {finding.comparison}")
    print(f"TOTAL {len(findings)}")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
