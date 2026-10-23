"""Serve registration lifecycle and qualification tests."""

from __future__ import annotations

import ast
import json
import re
import threading
import tomllib
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from imas_ambix.agent.profile import SiteConfig, load_profile
from imas_ambix.agent.registry import (
    ServeRegistration,
    build_endpoint_document,
    read_registrations,
    remove_registration,
    write_registration,
)
from imas_ambix.agent.slurm import generate_serve_script

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_AGENT_PACKAGE = _REPOSITORY_ROOT / "imas_ambix/agent"
_SERVING_ENVIRONMENT = _REPOSITORY_ROOT / "imas_ambix/agent/envs/vllm/pyproject.toml"


def _serving_python_feature_version() -> tuple[int, int]:
    environment = tomllib.loads(_SERVING_ENVIRONMENT.read_text(encoding="utf-8"))
    requires_python = environment["project"]["requires-python"]
    lower_bound = re.search(r"(?:^|,)\s*>=\s*(\d+)\.(\d+)", requires_python)
    assert lower_bound is not None, (
        f"{_SERVING_ENVIRONMENT} requires-python has no inclusive minor lower bound"
    )
    return int(lower_bound.group(1)), int(lower_bound.group(2))


def _agent_package_modules() -> tuple[Path, ...]:
    return tuple(sorted(_AGENT_PACKAGE.rglob("*.py")))


def _registration(
    job_id: str = "42", model_id: str = "glm-5.3", port: int = 18801
) -> ServeRegistration:
    return ServeRegistration(
        model_id=model_id,
        host="98dci4-gpu-0003",
        port=port,
        job_id=job_id,
        accelerator_count=4,
        checkpoint_precision="int4",
        accelerator_family="H200",
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
    assert f"--checkpoint-precision {profile.model.checkpoint_precision}" in script
    assert "-m imas_ambix.agent.registry remove" in script
    assert "trap cleanup_serve EXIT" in script
    assert "trap terminate_serve TERM INT" in script
    # A serve republishes the endpoint document once it answers anonymously,
    # and cleanup republishes again after withdrawing the registration, so a
    # new model is launcher-discoverable without an operator republish.
    write_pos = script.index("-m imas_ambix.agent.registry write")
    cleanup_start = script.index("cleanup_serve()")
    assert script.count("-m imas_ambix.agent.registry publish") == 2
    # One republish lives in the cleanup path, defined before the server run…
    cleanup_publish = script.index("-m imas_ambix.agent.registry publish")
    assert cleanup_start < cleanup_publish < write_pos
    # …and the other fires only after the write, once the engine is ready.
    assert write_pos < script.index("-m imas_ambix.agent.registry publish", write_pos)
    assert "readiness poll expired" in script


def test_agent_package_modules_parse_with_serving_python():
    module_paths = _agent_package_modules()

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


def test_a_dead_registration_sharing_a_port_does_not_empty_the_document():
    """A superseded job must not take the whole routing lane down with it.

    When a serve is replaced, the new job takes the port the old one released,
    so both registrations resolve to the same origin and both probe the one
    live serve. Refusing the ambiguity published an empty catalog while a
    healthy serve answered, which reads to every consumer as no endpoint at
    all -- observed three times in one day. Resolve to the later job instead.
    """
    superseded = _registration(
        job_id="1270620", model_id="deepseek-v4-flash", port=18800
    )
    live = ServeRegistration(
        model_id="deepseek-v4-flash",
        host="98dci4-gpu-0003",
        port=18800,
        job_id="1270628",
        accelerator_count=4,
        checkpoint_precision="int4",
        accelerator_family="H200",
    )

    def catalog(_origin: str) -> object:
        return {
            "data": [
                {
                    "id": "deepseek-v4-flash",
                    "max_model_len": 1048576,
                    "ambix": {
                        "accelerator_family": "H200",
                        "accelerator_count": 4,
                        "checkpoint_precision": "int4",
                    },
                }
            ]
        }

    endpoints = build_endpoint_document((superseded, live), fetch_catalog=catalog)

    assert len(endpoints) == 1, "one model id must publish exactly one endpoint"
    assert endpoints[0].port == 18800, "the later job's endpoint must win"


def test_a_second_model_survives_a_duplicate_on_the_first():
    """The document must lose at most the ambiguous entry, never every entry."""
    first = _registration(job_id="100", model_id="model-a")
    duplicate = _registration(job_id="101", model_id="model-a")
    other = _registration(job_id="102", model_id="model-b", port=18802)

    def catalog(_origin: str) -> object:
        return {
            "data": [
                {
                    "id": model_id,
                    "max_model_len": 1048576,
                    "ambix": {
                        "accelerator_family": "H200",
                        "accelerator_count": 4,
                        "checkpoint_precision": "int4",
                    },
                }
                for model_id in ("model-a", "model-b")
            ]
        }

    endpoints = build_endpoint_document(
        (first, duplicate, other), fetch_catalog=catalog
    )

    assert {endpoint.model_id for endpoint in endpoints} == {"model-a", "model-b"}


def test_two_engines_serving_one_model_both_stay_addressable():
    """One model on two engines is two routes, and both must publish.

    The deduplication key is the engine, not the model. Two engines of one
    model are two things a consumer can be pointed at, so collapsing them on
    the model id hides half the capacity the site declared -- while a restarted
    job keeps its predecessor's engine and still has to replace it.
    """
    first = _registration(job_id="100", model_id="glm-5.3", port=18801)
    second = _registration(job_id="101", model_id="glm-5.3", port=18802)

    def catalog(_origin: str) -> object:
        return {
            "data": [
                {
                    "id": "glm-5.3",
                    "max_model_len": 1048576,
                    "ambix": {
                        "accelerator_family": "H200",
                        "accelerator_count": 4,
                        "checkpoint_precision": "int4",
                    },
                }
            ]
        }

    endpoints = build_endpoint_document((first, second), fetch_catalog=catalog)

    assert [endpoint.engine_id for endpoint in endpoints] == [
        "gpu-0003:18801",
        "gpu-0003:18802",
    ]


def test_hold_is_armed_by_launch_and_only_release_clears_it(tmp_path):
    from imas_ambix.agent.registry import main

    record = replace(_registration(), held=True)
    args = ["write", "--directory", str(tmp_path)]
    for name, value in asdict(record).items():
        if name == "held":
            args.append("--held")
        elif value is None:
            # ``checkpoint`` is optional: a registration written without one must
            # round-trip, so the flag is omitted rather than passed as "None".
            continue
        else:
            args.extend(["--" + name.replace("_", "-"), str(value)])
    assert main(args) == 0
    write_registration(replace(record, held=False, port=19444), tmp_path)
    assert read_registrations(tmp_path, job_is_running=lambda _: True).current[0].held
    assert main(["release", "--directory", str(tmp_path), "--job-id", "42"]) == 0
    write_registration(record, tmp_path)
    released = json.loads((tmp_path / "42.json").read_text())
    assert released["held"] is False
    assert released["port"] == record.port


def test_held_launch_script_and_publication(tmp_path):
    profile = load_profile("glm-5-3").for_gpus(4)
    script = generate_serve_script(
        profile, SiteConfig(base_dir=str(tmp_path)), held=True
    )
    assert "registry write --held" in script
    assert script.index("registry write --held") < script.index("READINESS_URL=")
    called = []
    assert (
        build_endpoint_document(
            [replace(_registration(), held=True)],
            fetch_catalog=lambda url: called.append(url),
        )
        == ()
    )
    assert called == []


def test_legacy_registration_defaults_to_released(tmp_path):
    payload = asdict(_registration())
    payload.pop("held")
    (tmp_path / "42.json").write_text(json.dumps(payload))
    assert read_registrations(tmp_path, job_is_running=lambda _: True).current == (
        _registration(),
    )


@pytest.mark.parametrize("slug", ["glm-5-3", "deepseek-v4-1-flash"])
def test_held_registration_precedes_engine_launch(tmp_path, slug):
    profile = load_profile(slug)
    script = generate_serve_script(
        profile, SiteConfig(base_dir=str(tmp_path)), held=True
    )
    registration = script.index("registry write --held")
    launch = script.index(f'echo "[$(date)] Starting {profile.model.name} server"')
    assert registration < launch, (
        "engine can answer before its held registration exists"
    )
    assert script.index("--accelerator-family", registration) < launch
    assert script.index("_REGISTRATION_ACTIVE=1") < registration
    assert "set -euo pipefail" in script[:registration]


def _catalog_for(model_id: str, precision: str) -> dict[str, object]:
    return {
        "data": [
            {
                "id": model_id,
                "max_model_len": 512000,
                "ambix": {
                    "accelerator_family": "H200",
                    "accelerator_count": 4,
                    "checkpoint_precision": precision,
                },
            }
        ]
    }


def test_generated_script_records_the_profile_checkpoint_and_publishes_it(tmp_path):
    """A serve names the release the profile pins, and the endpoint carries it.

    The served name is held stable across a rotation, so the name no longer says
    which weights answered. The serve job writes the profile's
    ``<hf_repo>@<revision>`` onto its registration, the record round-trips that
    value, and the endpoint builder publishes it verbatim so a launcher reads
    the release without reaching into the registry directory.
    """
    profile = load_profile("deepseek-v4-1-flash")
    expected = f"{profile.model.hf_repo}@{profile.model.revision}"
    script = generate_serve_script(profile, SiteConfig(base_dir=str(tmp_path)))
    assert f"--checkpoint {expected}" in script

    registration = ServeRegistration(
        model_id=profile.model.served_name,
        host="98dci4-gpu-0003",
        port=18810,
        job_id="1273253",
        accelerator_count=4,
        checkpoint_precision=profile.model.checkpoint_precision,
        accelerator_family="H200",
        checkpoint=expected,
    )
    path = write_registration(registration, tmp_path)
    assert json.loads(path.read_text())["checkpoint"] == expected
    stored = read_registrations(tmp_path, job_is_running=lambda _job_id: True).current
    assert stored[0].checkpoint == expected

    endpoints = build_endpoint_document(
        stored,
        fetch_catalog=lambda _origin: _catalog_for(
            profile.model.served_name, profile.model.checkpoint_precision
        ),
    )
    assert endpoints[0].checkpoint == expected


def test_profile_without_a_revision_writes_an_unpinned_checkpoint(tmp_path):
    """An absent pin is visible, never a commit read from the downloaded cache.

    A profile that names a repository but no revision is not pinned to a commit,
    and the downloaded directory does not say which one, so the registration
    states ``unpinned`` in place of a commit rather than an identity that was
    never established.
    """
    profile = load_profile("glm-5-3").for_gpus(4)
    assert profile.model.revision is None
    script = generate_serve_script(profile, SiteConfig(base_dir=str(tmp_path)))
    assert f"--checkpoint {profile.model.hf_repo}@unpinned" in script


def test_registration_without_a_checkpoint_reads_null_and_still_builds(tmp_path):
    """A record written before the field existed is unknown, not backfilled."""
    payload = asdict(_registration())
    payload.pop("checkpoint", None)
    (tmp_path / "42.json").write_text(json.dumps(payload))

    read = read_registrations(tmp_path, job_is_running=lambda _job_id: True)
    assert read.current[0].checkpoint is None

    endpoints = build_endpoint_document(
        read.current,
        fetch_catalog=lambda _origin: _catalog_for("glm-5.3", "int4"),
    )
    assert len(endpoints) == 1
    assert endpoints[0].checkpoint is None
