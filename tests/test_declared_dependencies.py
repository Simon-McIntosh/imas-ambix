"""Census of directly imported distributions against the project declaration.

WHAT
----
Every distribution that something under ``imas_ambix/``, ``tests/`` or
``scripts/`` imports directly gets exactly one class:

* **default** -- the repository cannot do its ordinary work without it.  An
  ``import`` at any level (module or function) reaches it from code under
  ``imas_ambix/`` outside an optional subsystem, or from anywhere under
  ``tests/``; or it is the test and lint tooling every session runs.
* **optional** -- only an optional subsystem imports it: the GPU model serving
  stack (``imas_ambix/agent``), the world-model training stack
  (``imas_ambix/train``) or the FAIR-MAST acquisition package
  (``imas_ambix/data``).  A distribution reached only from ``scripts/`` (the
  analysis and figure tooling) counts as optional too: no core ``imas_ambix/``
  code outside those subsystems and no test reaches it.  The video extra's
  members (``LAZY_VIDEO_MODULES``) are optional by name.
* **unused** -- nothing under those three trees imports it.
* **transitive** -- nothing imports it, but another directly imported
  distribution requires it, so it needs no declaration of its own.

The classification lives in one function, :func:`build_census`, so the
assertion in :func:`test_every_directly_imported_distribution_is_declared` and
the renderer that writes the evidence fragment cannot disagree: both read the
same result.  ``packages_distributions`` maps an imported top-level module to
its distribution; path-source extras (``packages_distributions`` cannot see an
editable install that ships no ``top_level.txt``) are recovered from the
checkout the ``[tool.uv.sources]`` path points at.

WHY
---
The declaration must match what the code and tests actually import, so a plain
``uv sync`` installs the working stack and nothing a session relies on lives
outside the project's declared dependencies.  Keeping the census as a test
makes it a guard rather than a one-off scan: the failure lines name each
undeclared ``default`` and ``optional`` distribution, and the guard turns red
the moment the declaration falls behind the tree.

Regenerate the fragment after changing the tree, giving the output path and,
optionally, a pytest ``--durations`` log to record beside the census::

    uv run --no-sync python tests/test_declared_dependencies.py <out.html> <log>
"""

from __future__ import annotations

import ast
import os
import re
import sys
import tomllib
from collections import defaultdict
from dataclasses import dataclass
from importlib.metadata import distributions, packages_distributions
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from distribution_sources import editable_checkout  # noqa: E402

ROOT = TESTS_DIR.parent
SCAN_DIRS = ("imas_ambix", "tests", "scripts")

# Importing code under these roots belongs to an optional subsystem, so an
# import there does not make a distribution default.  Everything else under
# imas_ambix/, and every test import, does.  The FAIR-MAST acquisition package
# is optional too: its remote and object-store paths carry the data extra.
OPTIONAL_SUBSYSTEM_ROOTS = (
    "imas_ambix/agent",
    "imas_ambix/train",
    "imas_ambix/data",
)

# Video rendering and perceptual-metric paths import these only inside
# functions, and none is installed in an ordinary session.  They are named here
# as the video extra's members so the census checks each is declared there
# rather than pulling the video stack into a plain sync.
LAZY_VIDEO_MODULES = {
    "cv2": "opencv-python-headless",
    "torchvision": "torchvision",
    "imageio": "imageio",
    "imageio_ffmpeg": "imageio-ffmpeg",
    "lpips": "lpips",
}

# Tooling every session runs; default even though no import names some of them.
TOOLING = (
    "pytest",
    "pytest-cov",
    "pytest-timeout",
    "pytest-xdist",
    "ruff",
    "mypy",
    "pre-commit",
    "tokamap",
)

@dataclass(frozen=True)
class Row:
    """One directly imported distribution and how it stands against the declaration."""

    name: str
    kind: str
    deciding_file: str
    declaration: str


@dataclass(frozen=True)
class Census:
    """The whole census: rows, the transitive closure, and the failure lines."""

    rows: tuple[Row, ...]
    transitive: tuple[tuple[str, str], ...]
    unused_declared: tuple[str, ...]
    unmapped: tuple[str, ...]
    failures: tuple[str, ...]


def canonical(name: str) -> str:
    """Return the PEP 503 normalised distribution name."""
    return re.sub(r"[-_.]+", "-", name).lower()


def requirement_name(entry: object) -> str | None:
    """Extract the distribution name from a requirement or declaration string."""
    if not isinstance(entry, str):
        return None
    match = re.match(r"[A-Za-z0-9][A-Za-z0-9._-]*", entry.strip())
    return canonical(match.group(0)) if match else None


def declaration_map(project: dict) -> dict[str, str]:
    """Map every declared distribution name to where it is declared."""
    declared: dict[str, str] = {}
    for entry in project["project"].get("dependencies", []):
        name = requirement_name(entry)
        if name:
            declared[name] = "dependency"
    for group, entries in project.get("dependency-groups", {}).items():
        for entry in entries:
            name = requirement_name(entry)
            if name:
                declared.setdefault(name, f"group:{group}")
    for extra, entries in project["project"].get("optional-dependencies", {}).items():
        for entry in entries:
            name = requirement_name(entry)
            if name:
                declared.setdefault(name, f"extra:{extra}")
    return declared


def checkout_top_level_modules(path: Path) -> set[str]:
    """Top-level import names a sibling checkout exposes (flat or ``src/`` layout)."""
    base = path / "src" if (path / "src").is_dir() else path
    if not base.is_dir():
        return set()
    modules: set[str] = set()
    for child in base.iterdir():
        if child.suffix == ".py":
            modules.add(child.stem)
        elif child.is_dir() and (child / "__init__.py").is_file():
            modules.add(child.name)
    return modules


def installation_top_level_modules(dist) -> set[str]:
    """Top-level import names a single installed distribution exposes."""
    try:
        text = dist.read_text("top_level.txt")
    except OSError:
        text = None
    if text:
        return {line.strip() for line in text.splitlines() if line.strip()}
    try:
        raw = dist.read_text("direct_url.json")
    except OSError:
        return set()
    if not raw:
        return set()
    source_root = editable_checkout(raw)
    return checkout_top_level_modules(source_root) if source_root else set()


def module_distribution_map(sources: dict) -> dict[str, set[str]]:
    """Map each directly imported top-level module name to its distribution(s)."""
    del sources  # kept in the signature so a caller can pass [tool.uv.sources]
    mapping: dict[str, set[str]] = defaultdict(set)
    for module, dist_names in packages_distributions().items():
        for dist_name in dist_names:
            mapping[module].add(canonical(dist_name))

    # An editable path install ships no top_level.txt, so packages_distributions
    # cannot see it; recover the import names from the checkout it points at
    # (readable from its direct_url.json, so this works inside a worktree too).
    covered = {dist for dists in mapping.values() for dist in dists}
    for dist in distributions():
        name = dist.metadata["Name"]
        if not name:
            continue
        canonical_name = canonical(name)
        if canonical_name in covered:
            continue
        for module in installation_top_level_modules(dist):
            mapping[module].add(canonical_name)
    return mapping


def local_top_level_names() -> set[str]:
    """Top-level module names provided by this repository itself."""
    names = set(SCAN_DIRS)
    for child in ROOT.iterdir():
        if child.suffix == ".py" and child.stem != "__init__":
            names.add(child.stem)
        elif (child / "__init__.py").is_file():
            names.add(child.name)
    # Modules under the scanned trees import one another by bare name (the
    # analysis scripts especially), so every module stem is a local name.
    for base in SCAN_DIRS:
        for path in (ROOT / base).rglob("*.py"):
            if path.stem != "__init__":
                names.add(path.stem)
    return names


def iter_imports() -> dict[str, set[str]]:
    """Every top-level module imported anywhere, and the files importing it."""
    imports: dict[str, set[str]] = defaultdict(set)
    for base in SCAN_DIRS:
        for path in (ROOT / base).rglob("*.py"):
            try:
                tree = ast.parse(path.read_text(), filename=str(path))
            except SyntaxError:
                continue
            relative = path.relative_to(ROOT).as_posix()
            for node in ast.walk(tree):
                names: list[str] = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif (
                    isinstance(node, ast.ImportFrom)
                    and node.level == 0
                    and node.module
                ):
                    names = [node.module]
                for name in names:
                    imports[name.split(".")[0]].add(relative)
    return imports


def makes_default(relative: str) -> bool:
    """Whether an importing file's location makes its distribution default."""
    if relative.startswith("tests/"):
        return True
    if relative.startswith("imas_ambix/"):
        optional = tuple(f"{root}/" for root in OPTIONAL_SUBSYSTEM_ROOTS)
        return not relative.startswith(optional)
    return False  # scripts/ alone: optional tooling


def transitive_closure(roots: set[str]) -> dict[str, str]:
    """Map a required distribution to the directly imported one that pulls it in."""
    installed: dict[str, list[str] | None] = {}
    for dist in distributions():
        name = dist.metadata["Name"]
        if name:
            installed[canonical(name)] = dist.requires
    reached: dict[str, str] = {}
    origin: dict[str, str] = {root: root for root in roots}
    seen = set(roots)
    frontier = sorted(roots)
    while frontier:
        nxt: set[str] = set()
        for current in frontier:
            for entry in installed.get(current) or []:
                required = requirement_name(entry)
                if required and required in installed and required not in seen:
                    seen.add(required)
                    reached[required] = origin[current]
                    origin[required] = origin[current]
                    nxt.add(required)
        frontier = sorted(nxt)
    return reached


def build_census(pyproject_path: Path | None = None) -> Census:
    """Classify every directly imported distribution against the declaration.

    ``pyproject_path`` names the declaration to read; when it is omitted the
    ``AMBX_PYPROJECT`` environment variable, then the tree's own
    ``pyproject.toml``, is used, so a mutated declaration can be censused
    without editing the real file.
    """
    if pyproject_path is None:
        override = os.environ.get("AMBX_PYPROJECT")
        pyproject_path = Path(override) if override else ROOT / "pyproject.toml"
    project = tomllib.loads(pyproject_path.read_text())
    declared = declaration_map(project)
    sources = project.get("tool", {}).get("uv", {}).get("sources", {})
    module_map = module_distribution_map(sources)
    imports = iter_imports()
    local = local_top_level_names()
    stdlib = set(sys.stdlib_module_names)

    importers: dict[str, set[str]] = defaultdict(set)
    video_importers: dict[str, set[str]] = defaultdict(set)
    unmapped: set[str] = set()
    for module, files in imports.items():
        if module in stdlib or module in local:
            continue
        if module in LAZY_VIDEO_MODULES:
            video_importers[LAZY_VIDEO_MODULES[module]].update(files)
            continue
        dists = module_map.get(module)
        if not dists:
            unmapped.add(module)
            continue
        for dist in dists:
            importers[dist].update(files)

    rows: list[Row] = []
    for name, files in importers.items():
        default_files = sorted(f for f in files if makes_default(f))
        if default_files:
            rows.append(
                Row(name, "default", default_files[0], declared.get(name, "none"))
            )
        else:
            rows.append(
                Row(name, "optional", sorted(files)[0], declared.get(name, "none"))
            )
    for name, files in video_importers.items():
        rows.append(
            Row(name, "optional", sorted(files)[0], declared.get(name, "none"))
        )
    for tool in TOOLING:
        name = canonical(tool)
        if name not in importers:
            rows.append(
                Row(name, "default", "(no import; tooling)", declared.get(name, "none"))
            )
    rows.sort(key=lambda row: (row.kind, row.name))

    roots = set(importers) | {canonical(tool) for tool in TOOLING}
    reached = transitive_closure(roots)
    transitive = tuple(sorted((d, r) for d, r in reached.items() if d not in importers))

    directly_imported = set(importers)
    unused_declared = tuple(
        sorted(name for name in declared if name not in directly_imported)
    )

    failures: list[str] = []
    for row in rows:
        if row.kind == "default":
            if row.declaration != "dependency" and not row.declaration.startswith(
                "group:"
            ):
                failures.append(
                    f"default {row.name} (imported by {row.deciding_file}) is "
                    f"declared as {row.declaration}, not in [project.dependencies] "
                    f"or the dev dependency group"
                )
        elif not row.declaration.startswith("extra:"):
            failures.append(
                f"optional {row.name} (imported only by {row.deciding_file}) is "
                f"declared as {row.declaration}, not in an extra"
            )

    return Census(
        rows=tuple(rows),
        transitive=transitive,
        unused_declared=unused_declared,
        unmapped=tuple(sorted(unmapped)),
        failures=tuple(failures),
    )


_DURATION_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)s\s+(call|setup|teardown)\s+(\S+)")


def parse_durations(text: str) -> list[str]:
    """Extract the slowest-test lines from pytest ``--durations`` output."""
    found: list[str] = []
    for line in text.splitlines():
        match = _DURATION_RE.match(line)
        if match:
            found.append(f"{match.group(1)}s {match.group(2)} {match.group(3)}")
    return found


def render_fragment(
    census: Census, node: str, durations: list[str] | None = None
) -> str:
    """Render the census as an evidence fragment, from the same result.

    ``node`` labels the fragment (its ``id`` and header) and is supplied by the
    caller; ``durations`` optionally records the slowest tests beside the census.
    """
    lines: list[str] = [
        f'<section class="evidence-fragment" data-reckon="evidence" id="{node}">',
        "  <header>",
        "    <h3>Import census: which libraries a plain sync must declare</h3>",
        '    <p class="meta">',
        f"      Node <code>{node}</code> &middot; import census against the "
        "project declaration",
        "    </p>",
        "  </header>",
        "  <p>",
        "    Each distribution this repository imports directly, its class"
        " (default, optional, unused or transitive), the importing file that"
        " decided the class, and how the declaration currently carries it. The"
        " classifier in <code>tests/test_declared_dependencies.py</code> renders"
        " this fragment from the same result its assertion reads.",
        "  </p>",
        "  <table>",
        "    <thead><tr><th>Distribution</th><th>Class</th>"
        "<th>Deciding file</th><th>Declaration</th></tr></thead>",
        "    <tbody>",
    ]
    for row in census.rows:
        lines.append(
            f"      <tr><td><code>{row.name}</code></td><td>{row.kind}</td>"
            f"<td><code>{row.deciding_file}</code></td>"
            f"<td>{row.declaration}</td></tr>"
        )
    lines += [
        "    </tbody>",
        "  </table>",
        "  <h4>Transitive dependencies (nothing imports them; a directly imported"
        " distribution requires them)</h4>",
        "  <ul>",
    ]
    for dist, requirer in census.transitive:
        lines.append(
            f"    <li><code>{dist}</code> &mdash; required by "
            f"<code>{requirer}</code></li>"
        )
    lines += [
        "  </ul>",
        "  <h4>Declared dependencies nothing imports</h4>",
        "  <ul>",
    ]
    for name in census.unused_declared:
        lines.append(f"    <li><code>{name}</code></li>")
    lines += [
        "  </ul>",
        "  <h4>Imported but not mapped to an installed distribution</h4>",
        "  <ul>",
    ]
    for name in census.unmapped:
        lines.append(f"    <li><code>{name}</code></li>")
    lines += [
        "  </ul>",
        "  <h4>Assertion lines: undeclared default and optional distributions</h4>",
        "  <ul>",
    ]
    for failure in census.failures:
        lines.append(f"    <li>{failure}</li>")
    lines += ["  </ul>"]
    if durations:
        lines += ["  <h4>Slowest tests (whole suite)</h4>", "  <ul>"]
        for item in durations:
            lines.append(f"    <li><code>{item}</code></li>")
        lines += ["  </ul>"]
    lines += ["</section>", ""]
    return "\n".join(lines)


def write_fragment(
    census: Census, out_path: Path, durations: list[str] | None = None
) -> None:
    """Write ``census`` as an evidence fragment to ``out_path``.

    The fragment is labelled by the output file's stem, so this module carries
    no node identity of its own.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_fragment(census, out_path.stem, durations))


def test_every_directly_imported_distribution_is_declared() -> None:
    census = build_census()
    print("\n".join(census.failures))
    assert not census.failures, "undeclared distributions:\n" + "\n".join(
        census.failures
    )


def main(argv: list[str] | None = None) -> int:
    """Write the census fragment to the path given and report the failures.

    Usage: test_declared_dependencies.py <out.html> [durations.log]
    """
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print("usage: test_declared_dependencies.py <out.html> [durations.log]")
        return 2
    durations = parse_durations(Path(args[1]).read_text()) if len(args) > 1 else None
    out_path = Path(args[0])
    census = build_census()
    write_fragment(census, out_path, durations)
    count = len(census.failures)
    print(f"{len(census.rows)} distributions classified; {count} undeclared")
    for failure in census.failures:
        print(f"  {failure}")
    return 0 if not census.failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
