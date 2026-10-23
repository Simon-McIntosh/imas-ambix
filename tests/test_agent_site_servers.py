"""Declared servers: placement onto one, and the document's servers array."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from click.testing import CliRunner

from imas_ambix.agent.profile import SiteConfig, SiteServerConfig, load_profile
from imas_ambix.agent.registry import ServeRegistration, publish_endpoint_document
from imas_ambix.agent.slurm import generate_serve_script
from imas_ambix.cli import main

# Every value here is a literal, because the generated script embeds them: a
# digest taken under whatever paths the suite happens to be handed could not be
# compared against a recording made elsewhere.
_BASE_SITE = SiteConfig(
    base_dir="/tmp/ambix-site",
    engine_env_root="/tmp/ambix-engine-envs",
    endpoint_document_path="/tmp/ambix-endpoints.json",
)

# sha256 of the script `imas-ambix agent serve deepseek-v4-1-flash --dry-run`
# printed at the revision before a site could declare more than one server,
# under the site above. Declaring none must not move a byte of it.
#
# The digest is taken with the checkout's own root -- which the script embeds
# in PYTHONPATH and in the kernel-patch bind -- replaced by `_REPOSITORY_ROOT`,
# so it does not record where the tree happened to sit.
_BASE_SCRIPT_DIGEST = "d83d58cb79e1801bed704ed7b64c710bab0510afe9777b65e9b48cb860505773"

_REPOSITORY_ROOT = str(Path(__file__).resolve().parents[1])

_H200 = SiteServerConfig(
    name="h200",
    host="98dci4-gpu-0003",
    partition="betelgeuse",
    account="grpa",
    reservation="gpu_0003_grpA",
    accelerator_family="H200",
    accelerator_count=4,
    power_cap_watts=700,
    reservation_groups=("A", "B"),
)
_B300 = SiteServerConfig(
    name="b300",
    host="98dci4-gpu-0004",
    partition="blackwell",
    account="grpb",
    reservation="gpu_0004_grpB",
    accelerator_family="B300",
    accelerator_count=8,
    power_cap_watts=1000,
    reservation_groups=("C",),
)


def _two_server_site() -> SiteConfig:
    return _BASE_SITE.model_copy(update={"servers": [_H200, _B300]})


def test_no_declared_servers_prints_the_script_it_printed_before():
    """An unconfigured site resolves to one server and generates the same text."""
    profile = load_profile("deepseek-v4-1-flash")
    script = generate_serve_script(profile, _BASE_SITE, port=profile.slurm.port)
    normalised = script.replace(_REPOSITORY_ROOT, "<repo>")

    assert hashlib.sha256(normalised.encode()).hexdigest() == _BASE_SCRIPT_DIGEST


def test_the_flat_fields_are_the_single_server_when_none_is_declared():
    site = _BASE_SITE

    assert [server.name for server in site.declared_servers] == [site.gpu_host]
    server = site.resolve_server()
    assert (server.host, server.partition, server.account, server.reservation) == (
        site.gpu_host,
        site.partition,
        site.account,
        site.reservation,
    )


def test_each_declared_server_carries_its_own_partition_reservation_and_account():
    site = _two_server_site()
    profile = load_profile("deepseek-v4-1-flash")

    for server in (_H200, _B300):
        script = generate_serve_script(
            profile, site, port=profile.slurm.port, server=server
        )
        assert f"#SBATCH --partition={server.partition}" in script
        assert f"#SBATCH --reservation={server.reservation}" in script
        assert f"#SBATCH --account={server.account}" in script
        # The registration records which server it was placed on, which is what
        # the endpoint document groups engines by.
        assert f"--server {server.name}" in script

    b300_script = generate_serve_script(
        profile, site, port=profile.slurm.port, server=_B300
    )
    assert "#SBATCH --reservation=gpu_0003_grpA" not in b300_script
    assert "#SBATCH --account=grpa" not in b300_script
    # The accelerator family a serve declares follows its server, so a second
    # server of another family does not publish the first server's cards.
    assert "--accelerator-family B300" in b300_script


def _catalog(model_id: str, family: str, count: int, precision: str) -> dict:
    return {
        "data": [
            {
                "id": model_id,
                "max_model_len": 1048576,
                "ambix": {
                    "accelerator_family": family,
                    "accelerator_count": count,
                    "checkpoint_precision": precision,
                },
            }
        ]
    }


def test_the_document_lists_each_server_with_the_engines_placed_on_it(tmp_path):
    """The published servers array is what readers outside the package consume."""
    records = (
        ServeRegistration(
            "glm-5.3", "98dci4-gpu-0003", 18801, "41", 4, "int4", "H200", server="h200"
        ),
        ServeRegistration(
            "glm-5.3", "98dci4-gpu-0003", 18802, "42", 4, "int4", "H200", server="h200"
        ),
        ServeRegistration(
            "glm-5.3", "98dci4-gpu-0004", 18801, "43", 8, "int4", "B300", server="b300"
        ),
    )
    catalogs = {
        record.origin: _catalog(
            "glm-5.3", record.accelerator_family, record.accelerator_count, "int4"
        )
        for record in records
    }
    target = tmp_path / "endpoints.json"

    publish_endpoint_document(
        records, target, fetch_catalog=catalogs.__getitem__, servers=(_H200, _B300)
    )

    document = json.loads(target.read_text(encoding="utf-8"))
    assert [(item["host"], item["port"]) for item in document["endpoints"]] == [
        ("98dci4-gpu-0003", 18801),
        ("98dci4-gpu-0003", 18802),
        ("98dci4-gpu-0004", 18801),
    ]
    # Declaration order, so a reader renders the site's own ordering.
    assert [server["name"] for server in document["servers"]] == ["h200", "b300"]
    servers = {server["name"]: server for server in document["servers"]}
    assert servers["h200"]["host"] == "98dci4-gpu-0003"
    assert servers["h200"]["engines"] == ["gpu-0003:18801", "gpu-0003:18802"]
    assert servers["h200"]["accelerator_family"] == "H200"
    assert servers["h200"]["power_cap_watts"] == 700
    assert servers["h200"]["reservation_groups"] == ["A", "B"]
    assert servers["b300"]["host"] == "98dci4-gpu-0004"
    assert servers["b300"]["engines"] == ["gpu-0004:18801"]
    assert servers["b300"]["accelerator_family"] == "B300"
    assert servers["b300"]["reservation_groups"] == ["C"]


def test_a_declared_server_with_no_engine_is_still_listed(tmp_path):
    """A declared server is a row even when nothing is running on it.

    The job table renders the site's servers, so a server that is idle has to
    be visible as idle; a servers array built from the running engines would
    simply omit it, which reads the same as a server that is not declared.
    """
    records = (
        ServeRegistration(
            "glm-5.3", "98dci4-gpu-0003", 18801, "41", 4, "int4", "H200", server="h200"
        ),
    )
    catalogs = {records[0].origin: _catalog("glm-5.3", "H200", 4, "int4")}
    target = tmp_path / "endpoints.json"

    publish_endpoint_document(
        records, target, fetch_catalog=catalogs.__getitem__, servers=(_H200, _B300)
    )

    document = json.loads(target.read_text(encoding="utf-8"))
    servers = {server["name"]: server for server in document["servers"]}
    assert sorted(servers) == ["b300", "h200"]
    assert servers["h200"]["engines"] == ["gpu-0003:18801"]
    assert servers["b300"]["engines"] == []
    assert servers["b300"]["host"] == "98dci4-gpu-0004"
    assert servers["b300"]["accelerator_family"] == "B300"
    assert servers["b300"]["accelerator_count"] == 8


def test_a_servers_card_count_comes_from_its_declaration(tmp_path):
    """A serve may take fewer cards than its node has.

    The declaration is the node's card count and the engine is one placement on
    it, so deriving the count from the engine would publish a four-card serve
    as a four-card node and lose the cards the rest of the site can take.
    """
    node = _H200.model_copy(update={"accelerator_count": 8})
    record = ServeRegistration(
        "glm-5.3", "98dci4-gpu-0003", 18801, "41", 4, "int4", "H200", server="h200"
    )
    catalogs = {record.origin: _catalog("glm-5.3", "H200", 4, "int4")}
    target = tmp_path / "endpoints.json"

    publish_endpoint_document(
        (record,), target, fetch_catalog=catalogs.__getitem__, servers=(node,)
    )

    (server,) = json.loads(target.read_text(encoding="utf-8"))["servers"]
    assert server["name"] == "h200"
    assert server["engines"] == ["gpu-0003:18801"]
    # The engine registered four cards; the node declares eight.
    assert record.accelerator_count == 4
    assert server["accelerator_count"] == node.accelerator_count == 8
    assert server["power_cap_watts"] == 700
    assert server["reservation_groups"] == ["A", "B"]


def test_an_undeclared_host_still_carries_its_engines(tmp_path):
    """Nothing is dropped for want of a declaration."""
    record = ServeRegistration(
        "glm-5.3", "98dci4-gpu-0009", 18801, "41", 4, "int4", "H200", server=""
    )
    catalogs = {record.origin: _catalog("glm-5.3", "H200", 4, "int4")}
    target = tmp_path / "endpoints.json"

    publish_endpoint_document(
        (record,), target, fetch_catalog=catalogs.__getitem__, servers=(_H200,)
    )

    document = json.loads(target.read_text(encoding="utf-8"))
    assert [(item["host"], item["port"]) for item in document["endpoints"]] == [
        ("98dci4-gpu-0009", 18801)
    ]
    servers = {server["name"]: server for server in document["servers"]}
    assert sorted(servers) == ["98dci4-gpu-0009", "h200"]
    assert servers["98dci4-gpu-0009"]["engines"] == ["gpu-0009:18801"]
    assert servers["98dci4-gpu-0009"]["accelerator_count"] == 4
    assert servers["h200"]["engines"] == []


def test_a_configured_serve_is_placed_on_the_named_server(monkeypatch):
    monkeypatch.setenv(
        "AMBIX_AGENT_SERVERS", json.dumps([_H200.model_dump(), _B300.model_dump()])
    )

    result = CliRunner().invoke(
        main,
        ["agent", "serve", "deepseek-v4-1-flash", "--server", "b300", "--dry-run"],
    )

    assert result.exit_code == 0, result.output
    assert "#SBATCH --partition=blackwell" in result.output
    assert "#SBATCH --account=grpb" in result.output


def test_serve_refuses_a_server_that_no_declaration_names(monkeypatch):
    monkeypatch.setenv(
        "AMBIX_AGENT_SERVERS", json.dumps([_H200.model_dump(), _B300.model_dump()])
    )

    result = CliRunner().invoke(
        main,
        [
            "agent",
            "serve",
            "deepseek-v4-1-flash",
            "--server",
            "gpu-9999",
            "--dry-run",
        ],
    )
    message = result.output + (getattr(result, "stderr", "") or "")

    assert result.exit_code != 0
    assert "gpu-9999" in message
    assert "h200" in message and "b300" in message
    assert "#!/bin/bash" not in message
