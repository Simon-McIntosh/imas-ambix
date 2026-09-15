"""Grammar compatibility checks for modules executed by serving environments."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from imas_ambix.agent.profile import SiteConfig

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_PACKAGE_ROOT = _REPOSITORY_ROOT / "imas_ambix"
_PARSER = """\
import ast
from pathlib import Path
import sys

failed = False
for argument in sys.argv[1:]:
    path = Path(argument)
    try:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as error:
        print(f"{path}:{error.lineno}: {error.msg}", file=sys.stderr)
        failed = True
raise SystemExit(failed)
"""


def _serving_python() -> Path:
    """Resolve the vLLM interpreter through the serving site configuration."""
    return SiteConfig().python_path("vllm")


def _parse_with_serving_python(*paths: Path) -> subprocess.CompletedProcess[str]:
    interpreter = _serving_python()
    if not interpreter.is_file():
        pytest.skip(
            "Serving Python is unavailable at "
            f"{interpreter}; run this check where the vLLM environment is installed."
        )
    return subprocess.run(
        [str(interpreter), "-c", _PARSER, *(str(path) for path in paths)],
        capture_output=True,
        text=True,
        check=False,
    )


def test_serving_interpreter_parses_every_module() -> None:
    module_paths = tuple(sorted(_PACKAGE_ROOT.rglob("*.py")))

    result = _parse_with_serving_python(*module_paths)

    assert result.returncode == 0, result.stderr


def test_serving_interpreter_gate_names_invalid_exception_tuple(tmp_path: Path) -> None:
    reverted_module = tmp_path / "reverted_module.py"
    reverted_module.write_text(
        "try:\n    pass\nexcept OSError, TypeError:\n    pass\n", encoding="utf-8"
    )

    result = _parse_with_serving_python(reverted_module)

    assert result.returncode == 1
    assert (
        f"{reverted_module}:3: multiple exception types must be parenthesized"
        in result.stderr
    )
