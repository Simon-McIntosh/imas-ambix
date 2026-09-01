"""Serve registration lifecycle and qualification tests."""

from __future__ import annotations

import ast
import re
import threading
import tomllib
from pathlib import Path

import pytest

from imas_ambix.agent.profile import SiteConfig, load_profile
from imas_ambix.agent.registry import (
    ServeRegistration,
    read_registrations,
    remove_registration,
    write_registration,
)
from imas_ambix.agent.slurm import generate_serve_script

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_SERVING_ENVIRONMENT = _REPOSITORY_ROOT / "imas_ambix/agent/envs/vllm/pyproject.toml"


def _serving_python_feature_version() -> tuple[int, int]:
    environment = tomllib.loads(_SERVING_ENVIRONMENT.read_text(encoding="utf-8"))
    requires_python = environment["project"]["requires-python"]
    lower_bound = re.search(r"(?:^|,)\s*>=\s*(\d+)\.(\d+)", requires_python)
    assert lower_bound is not None, (
        f"{_SERVING_ENVIRONMENT} requires-python has no inclusive minor lower bound"
    )
    return int(lower_bound.group(1)), int(lower_bound.group(2))


def _repository_modules_referenced_by(script: str) -> tuple[Path, ...]:
    paths: set[Path] = set()
    references = set(re.findall(r"\bimas_ambix(?:\.[A-Za-z_]\w*)+", script))
    for reference in references:
        parts = reference.split(".")
        for length in range(1, len(parts) + 1):
            package = _REPOSITORY_ROOT.joinpath(*parts[:length], "__init__.py")
            if package.is_file():
                paths.add(package)
            module = _REPOSITORY_ROOT.joinpath(*parts[:length]).with_suffix(".py")
            if module.is_file():
                paths.add(module)
                break
    return tuple(sorted(paths))


def _registration(job_id: str = "42", model_id: str = "glm-5.3") -> ServeRegistration:
    return ServeRegistration(
        model_id=model_id,
        host="98dci4-gpu-0003",
        port=18801,
        job_id=job_id,
        accelerator_count=4,
        checkpoint_precision="int4",
    )


def test_record_round_trip_carries_resolved_launch_identity(tmp_path):
    record = _registration()
    path = write_registration(record, tmp_path)
    read = read_registrations(tmp_path, job_is_running=lambda job_id: job_id == "42")

    assert path.name == "42.json"
    assert read.current == (record,)
    assert read.stale == ()
    assert read.current[0].origin == "http://98dci4-gpu-0003:18801"


def test_atomic_replacement_never_exposes_a_partial_record(tmp_path):
    first = _registration(model_id="glm-5.3")
    second = _registration(model_id="deepseek-v4-flash")
    write_registration(first, tmp_path)
    observed: list[tuple[ServeRegistration, ...]] = []

    def replace_repeatedly() -> None:
        for index in range(100):
            write_registration(first if index % 2 else second, tmp_path)

    writer = threading.Thread(target=replace_repeatedly)
    writer.start()
    while writer.is_alive():
        observed.append(
            read_registrations(tmp_path, job_is_running=lambda _job_id: True).current
        )
    writer.join()

    assert observed
    assert all(snapshot in {(first,), (second,)} for snapshot in observed)


def test_reader_returns_current_records_and_separates_stale_jobs(tmp_path):
    current = _registration(job_id="42", model_id="glm-5.3")
    stale = _registration(job_id="41", model_id="deepseek-v4-flash")
    write_registration(current, tmp_path)
    write_registration(stale, tmp_path)

    read = read_registrations(tmp_path, job_is_running=lambda job_id: job_id == "42")

    assert read.current == (current,)
    assert read.stale == (stale,)


def test_remove_registration_is_idempotent(tmp_path):
    path = write_registration(_registration(), tmp_path)

    remove_registration("42", tmp_path)
    remove_registration("42", tmp_path)

    assert not path.exists()


def test_generated_serve_script_owns_registration_lifecycle(tmp_path):
    profile = load_profile("glm-5-3").for_gpus(4)
    script = generate_serve_script(
        profile, SiteConfig(base_dir=str(tmp_path)), port=18801
    )

    assert "-m imas_ambix.agent.registry write" in script
    assert "--model-id glm-5.3" in script
    assert '--host "$(hostname)"' in script
    assert '--port "$PORT"' in script
    assert '--job-id "$SLURM_JOB_ID"' in script
    assert '--accelerator-count "${SLURM_GPUS_ON_NODE:-4}"' in script
    assert "--checkpoint-precision int4" in script
    assert "-m imas_ambix.agent.registry remove" in script
    assert "trap cleanup_serve EXIT" in script
    assert "trap terminate_serve TERM INT" in script


def test_generated_serve_script_modules_parse_with_serving_python(tmp_path):
    profile = load_profile("glm-5-3").for_gpus(4)
    script = generate_serve_script(
        profile, SiteConfig(base_dir=str(tmp_path)), port=18801
    )
    module_paths = _repository_modules_referenced_by(script)

    assert {path.relative_to(_REPOSITORY_ROOT).as_posix() for path in module_paths} >= {
        "imas_ambix/agent/registry.py",
        "imas_ambix/agent/vllm_catalog.py",
    }
    for path in module_paths:
        source = path.read_text(encoding="utf-8")
        try:
            ast.parse(
                source,
                filename=str(path),
                feature_version=_serving_python_feature_version(),
            )
        except SyntaxError as error:
            pytest.fail(
                f"{path}:{error.lineno}: is not valid for the serving Python: "
                f"{error.msg}",
                pytrace=False,
            )


def test_serving_python_guard_rejects_unparenthesized_exception_tuple():
    invalid_source = """\
try:
    pass
except OSError, TypeError:
    pass
"""

    with pytest.raises(SyntaxError):
        ast.parse(
            invalid_source,
            feature_version=_serving_python_feature_version(),
        )
