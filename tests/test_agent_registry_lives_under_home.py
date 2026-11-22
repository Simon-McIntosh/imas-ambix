"""The serve registry and the publish that reads it.

A serve registers itself in a directory, and a publish turns those records into
the endpoint document every client reads. Both were originally anchored on the
site's base directory, beside the model weights, so reclaiming that tree
deleted the registry silently and the next publish wrote a document naming no
engine at all. These tests pin the directory to the operator's state home, pin
the generated scripts to that directory, and pin the publish that finds an
empty registry while a serve is running to the document it must not erase.
"""

from __future__ import annotations

import json
from pathlib import Path

from imas_ambix.agent import cli as cli_mod
from imas_ambix.agent import registry as registry_mod
from imas_ambix.agent import slurm as slurm_mod
from imas_ambix.agent.profile import SiteConfig, load_profile
from imas_ambix.agent.registry import (
    ServeRegistration,
    publish_registrations,
    read_endpoint_document,
    registration_directory,
    write_registration,
)


def _catalog(model_id: str) -> dict:
    return {
        "data": [
            {
                "id": model_id,
                "max_model_len": 524_288,
                "ambix": {
                    "accelerator_family": "H200",
                    "accelerator_count": 4,
                    "checkpoint_precision": "fp4+fp8",
                },
            }
        ]
    }


def _registration(job_id: str = "41") -> ServeRegistration:
    return ServeRegistration(
        model_id="deepseek-v4-flash",
        host="98dci4-gpu-0003",
        port=18800,
        job_id=job_id,
        accelerator_count=4,
        checkpoint_precision="fp4+fp8",
        accelerator_family="H200",
    )


def _existing_document(path: Path) -> str:
    """Write a document naming one engine and return its exact text."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "endpoints": [
                    {
                        "model_id": "deepseek-v4-flash",
                        "host": "98dci4-gpu-0003",
                        "port": 18800,
                        "accelerator_family": "H200",
                        "accelerator_count": 4,
                        "checkpoint_precision": "fp4+fp8",
                        "max_context": 524_288,
                    }
                ],
                "routing_origins": [],
                "servers": [],
                "checkpoints": {},
            }
        ),
        encoding="utf-8",
    )
    return path.read_text(encoding="utf-8")


# -- Where the registry lives -------------------------------------------------


def test_default_registry_directory_is_under_the_state_home(tmp_path) -> None:
    """The default registry is operator state under home, not site data."""
    base_dir = tmp_path / "site"
    site = SiteConfig(base_dir=str(base_dir))

    directory = registration_directory()

    assert directory == tmp_path / "state" / "ambix" / "serve-registry"
    assert base_dir not in directory.parents
    assert directory != Path(site.base_dir) / "agents" / "serve-registry"
    assert site.serve_registry_dir == directory


def test_registry_falls_back_to_local_state_under_home(tmp_path, monkeypatch) -> None:
    """With no XDG override the path is the documented home-relative one."""
    monkeypatch.delenv("XDG_STATE_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    assert registration_directory() == (
        tmp_path / "home" / ".local" / "state" / "ambix" / "serve-registry"
    )


def test_generated_serve_script_registers_in_the_home_directory(tmp_path) -> None:
    """A serve script writes and removes records in the home directory."""
    site = SiteConfig(base_dir=str(tmp_path / "site"))
    script = slurm_mod.generate_serve_script(load_profile("deepseek-v4-flash"), site)

    state_registry = tmp_path / "state" / "ambix" / "serve-registry"
    assert f"_REGISTRY_DIR={state_registry}" in script
    assert str(Path(site.base_dir) / "agents" / "serve-registry") not in script


def test_generated_router_script_publishes_from_the_home_directory(tmp_path) -> None:
    """The router's own republish names the same directory the serves write."""
    site = SiteConfig(base_dir=str(tmp_path / "site"))
    script = slurm_mod.generate_router_script(site, port=18802)

    state_registry = tmp_path / "state" / "ambix" / "serve-registry"
    assert f"imas_ambix.agent.registry publish --directory {state_registry}" in script
    assert str(Path(site.base_dir) / "agents" / "serve-registry") not in script


# -- The publish that must not erase a document -------------------------------


def test_publish_with_a_running_serve_and_an_empty_registry_keeps_the_document(
    tmp_path, monkeypatch, capsys
) -> None:
    """An empty registry beside a running serve preserves what it would erase."""
    directory = tmp_path / "registry"  # never created: the lost-registry shape
    document = tmp_path / "endpoints.json"
    before = _existing_document(document)
    monkeypatch.setattr(
        cli_mod,
        "_running_jobs",
        lambda _site, **_: [
            {"jobid": "41", "state": "RUNNING", "comment": "", "name": "serve"}
        ],
    )

    exit_code = registry_mod.main(
        ["publish", "--directory", str(directory), "--output", str(document)]
    )

    assert exit_code == 0
    assert document.read_text(encoding="utf-8") == before
    reported = capsys.readouterr()
    assert "holds no registration" in reported.err
    assert "41" in reported.err
    assert "keeping the endpoint document" in reported.err


def test_publish_with_a_registration_writes_the_document(tmp_path) -> None:
    """The guard does not refuse a publish that has something to say."""
    site = SiteConfig(
        base_dir=str(tmp_path / "site"),
        endpoint_document_path=str(tmp_path / "endpoints.json"),
    )
    write_registration(_registration(), site.serve_registry_dir)

    result = publish_registrations(
        site.serve_registry_dir,
        site.endpoint_document,
        running_jobs=["41"],
        fetch_catalog=lambda _origin: _catalog("deepseek-v4-flash"),
    )

    assert result.reason is None
    assert result.path == site.endpoint_document
    published = read_endpoint_document(site.endpoint_document)
    assert [endpoint.model_id for endpoint in published] == ["deepseek-v4-flash"]


def test_publish_after_a_serve_releases_its_own_record_writes_the_document(
    tmp_path,
) -> None:
    """A serve that has just deregistered is what makes an empty read legitimate."""
    site = SiteConfig(
        base_dir=str(tmp_path / "site"),
        endpoint_document_path=str(tmp_path / "endpoints.json"),
    )
    site.serve_registry_dir.mkdir(parents=True)
    _existing_document(site.endpoint_document)

    result = publish_registrations(
        site.serve_registry_dir,
        site.endpoint_document,
        running_jobs=["41"],
        fetch_catalog=lambda _origin: _catalog("deepseek-v4-flash"),
        released_jobs=["41"],
    )

    assert result.reason is None
    written = json.loads(site.endpoint_document.read_text(encoding="utf-8"))
    assert written["endpoints"] == []


def test_a_registration_that_cannot_be_probed_is_not_published(tmp_path) -> None:
    """A registration that does not answer drops out of the document."""
    site = SiteConfig(
        base_dir=str(tmp_path / "site"),
        endpoint_document_path=str(tmp_path / "endpoints.json"),
    )
    write_registration(_registration(), site.serve_registry_dir)
    _existing_document(site.endpoint_document)

    def _refuse(_origin):
        raise ValueError("catalog unreachable")

    result = publish_registrations(
        site.serve_registry_dir,
        site.endpoint_document,
        running_jobs=["41"],
        fetch_catalog=_refuse,
    )

    assert result.reason is None
    written = json.loads(site.endpoint_document.read_text(encoding="utf-8"))
    assert written["endpoints"] == []
