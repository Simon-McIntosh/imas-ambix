"""Contracts for catalog-derived Clive picker presentation."""

from __future__ import annotations

import json
import os
import socket
import subprocess
from contextlib import contextmanager

import pytest

from imas_ambix.agent.clive import generate_clive_script
from imas_ambix.agent.litellm_service import LITELLM_PORT
from tests.agent.catalog_fixture import serve_catalog_items

CAPABILITY_SUFFIX = "_SUPPORTED_CAPABILITIES"


def _catalog_item(
    model_id: str,
    *,
    accelerator_count: int,
    max_model_len: int,
    precision: str = "fp8",
) -> dict[str, object]:
    return {
        "id": model_id,
        "max_model_len": max_model_len,
        "ambix": {
            "accelerator_family": "H200",
            "accelerator_count": accelerator_count,
            "checkpoint_precision": precision,
        },
    }


@contextmanager
def _openrouter_proxy_port():
    """Answer the readiness probe the hybrid branch makes before it starts the proxy."""
    listener = socket.socket()
    try:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            listener.bind(("127.0.0.1", LITELLM_PORT))
        except OSError:
            # The port already answers, so the launcher's probe succeeds
            # without this listener and the test does not need to hold the
            # port itself.
            yield
        else:
            listener.listen(1)
            yield
    finally:
        listener.close()


def _run_launcher(
    tmp_path,
    items,
    selected_model=None,
    *,
    preferred_release_id=None,
    mode="local",
    openrouter_native_release=None,
):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    arguments_file = tmp_path / "claude-arguments"
    environment_file = tmp_path / "claude-environment"
    launcher = tmp_path / "clive"

    (fake_bin / "claude").write_text(
        "#!/bin/sh\n"
        f"printf '%s\\n' \"$@\" > {arguments_file}\n"
        f"env > {environment_file}\n",
        encoding="utf-8",
    )
    (fake_bin / "claude").chmod(0o755)
    if mode != "local":
        # The hybrid branch asks the user manager for the proxy before it
        # probes the port; the probe itself is answered by the listener
        # opened below, not by this stub.
        (fake_bin / "systemctl").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        (fake_bin / "systemctl").chmod(0o755)

    with serve_catalog_items(items) as (site, requests):
        site = site.model_copy(update={"preferred_release_id": preferred_release_id})
        launcher.write_text(
            generate_clive_script(
                site,
                mode=mode,
                openrouter_native_release=openrouter_native_release,
            ),
            encoding="utf-8",
        )
        launcher.chmod(0o755)
        environment = os.environ.copy()
        environment["PATH"] = f"{fake_bin}:{environment['PATH']}"
        command = [str(launcher)]
        if mode != "local":
            command.extend(("--mode", mode))
        if selected_model is not None:
            command.extend(("--model", selected_model))
        with _openrouter_proxy_port():
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                env=environment,
            )

    arguments = arguments_file.read_text(encoding="utf-8").splitlines()
    harness_environment = dict(
        line.split("=", 1)
        for line in environment_file.read_text(encoding="utf-8").splitlines()
        if "=" in line
    )
    settings = json.loads(arguments[arguments.index("--settings") + 1])
    return result, settings, harness_environment, requests


def test_preferred_release_selects_without_narrowing_picker(tmp_path):
    items = [
        _catalog_item(
            "first-release",
            accelerator_count=2,
            max_model_len=524_288,
        ),
        _catalog_item(
            "preferred-release",
            accelerator_count=4,
            max_model_len=65_536,
        ),
    ]

    result, settings, environment, _requests = _run_launcher(
        tmp_path,
        items,
        preferred_release_id="preferred-release",
    )

    assert result.returncode == 0, result.stderr
    assert "Select model" not in result.stderr
    assert environment["ANTHROPIC_MODEL"] == "preferred-release"
    assert [row["model"] for row in settings["modelPicker"]["options"]] == [
        "first-release",
        "preferred-release",
    ]


def test_each_release_gets_its_own_topology_and_context(tmp_path):
    items = [
        _catalog_item(
            "narrow-release",
            accelerator_count=2,
            max_model_len=524_288,
            precision="int4",
        ),
        _catalog_item(
            "wide-release",
            accelerator_count=4,
            max_model_len=262_144,
        ),
    ]

    result, settings, environment, requests = _run_launcher(
        tmp_path, items, "narrow-release"
    )

    assert result.returncode == 0, result.stderr
    assert len(requests) == 1
    assert "Authorization" not in requests[0][1]
    assert settings["modelPicker"]["replaceBuiltInOptions"] is True
    assert settings["modelPicker"]["options"] == [
        {
            "model": "narrow-release",
            "label": "narrow-release",
            "description": "2×H200 · int4 · 512k context",
            "behavesAs": "claude-sonnet-5",
        },
        {
            "model": "wide-release",
            "label": "wide-release",
            "description": "4×H200 · fp8 · 256k context",
            "behavesAs": "claude-sonnet-5",
        },
    ]
    # The exported context is the input ceiling, not the served window, so the
    # harness plans against a budget the engine can actually accept.
    assert environment["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == "492288"
    assert environment["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == "32000"
    assert 492_288 + 32_000 <= 524_288
    assert "492288-token usable input budget" in result.stderr
    assert "32000-token output reservation" in result.stderr


def test_small_context_uses_its_own_safe_output_reservation(tmp_path):
    items = [
        _catalog_item(
            "large-release",
            accelerator_count=2,
            max_model_len=524_288,
        ),
        _catalog_item(
            "small-release",
            accelerator_count=4,
            max_model_len=65_536,
        ),
    ]

    result, _settings, environment, _requests = _run_launcher(
        tmp_path, items, "small-release"
    )

    assert result.returncode == 0, result.stderr
    usable_input = int(environment["CLAUDE_CODE_MAX_CONTEXT_TOKENS"])
    reservation = int(environment["CLAUDE_CODE_MAX_OUTPUT_TOKENS"])
    assert reservation == 16_384
    assert usable_input == 49_152
    assert usable_input > 33_537
    # A prompt filled to the declared ceiling still leaves the reservation
    # inside the served window, which is what keeps the engine from refusing.
    assert usable_input + reservation <= 65_536
    assert "49152-token usable input budget" in result.stderr
    assert "16384-token output reservation" in result.stderr


@pytest.mark.parametrize("max_model_len", [2, 3, 65_536, 524_288])
def test_output_reservation_always_leaves_a_minimal_prompt(tmp_path, max_model_len):
    items = [
        _catalog_item(
            "bounded-release",
            accelerator_count=2,
            max_model_len=max_model_len,
        )
    ]

    result, _settings, environment, _requests = _run_launcher(
        tmp_path, items, "bounded-release"
    )

    assert result.returncode == 0, result.stderr
    usable_input = int(environment["CLAUDE_CODE_MAX_CONTEXT_TOKENS"])
    reservation = int(environment["CLAUDE_CODE_MAX_OUTPUT_TOKENS"])
    assert reservation >= 1
    assert usable_input >= 1
    assert usable_input + reservation <= max_model_len


def test_picker_rows_do_not_create_or_redirect_aliases(tmp_path):
    items = [
        _catalog_item(
            f"catalog-release-{index}",
            accelerator_count=2 if index % 2 else 4,
            max_model_len=131_072 * index,
        )
        for index in range(1, 6)
    ]

    result, settings, environment, _requests = _run_launcher(
        tmp_path, items, "catalog-release-5"
    )

    assert result.returncode == 0, result.stderr
    assert len(settings["modelPicker"]["options"]) == 5
    assert settings["modelPicker"]["options"][-1]["model"] == "catalog-release-5"
    for row in settings["modelPicker"]["options"]:
        assert row["label"] == row["model"]
        assert "×H200" in row["description"]
        assert "fp8" in row["description"]
        assert "context" in row["description"]
    for alias in ("OPUS", "SONNET", "HAIKU", "FABLE"):
        assert environment[f"ANTHROPIC_DEFAULT_{alias}_MODEL"] == "catalog-release-5"


def test_every_declared_alias_has_supported_capabilities(tmp_path):
    items = [
        _catalog_item(
            "future-release",
            accelerator_count=6,
            max_model_len=393_216,
        )
    ]

    result, _settings, environment, _requests = _run_launcher(
        tmp_path, items, "future-release"
    )

    assert result.returncode == 0, result.stderr
    model_variables = {
        name
        for name in environment
        if name.startswith("ANTHROPIC_DEFAULT_")
        and name.endswith("_MODEL")
        and "_NAME" not in name
        and "_DESCRIPTION" not in name
    }
    assert model_variables == {
        "ANTHROPIC_DEFAULT_OPUS_MODEL",
        "ANTHROPIC_DEFAULT_SONNET_MODEL",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL",
        "ANTHROPIC_DEFAULT_FABLE_MODEL",
    }
    for model_variable in model_variables:
        assert environment[f"{model_variable}_SUPPORTED_CAPABILITIES"] == "thinking"


def test_every_hybrid_slot_declares_its_supported_capabilities(tmp_path):
    items = [
        _catalog_item(
            "frontier-native-release",
            accelerator_count=4,
            max_model_len=1_048_576,
        ),
        _catalog_item(
            "secondary-native-release",
            accelerator_count=2,
            max_model_len=524_288,
        ),
    ]

    result, _settings, environment, _requests = _run_launcher(
        tmp_path,
        items,
        "frontier-native-release",
        mode="hybrid",
        openrouter_native_release="frontier-native-release",
    )

    assert result.returncode == 0, result.stderr
    # A slot variable carries a NAME companion, which is the harness's own
    # model-registration convention. Deriving the roster from that rather than
    # listing it here is what makes a newly added slot visible: it arrives in
    # the roster and, absent its declaration, cannot match the mapping below.
    roster = {name for name in environment if f"{name}_NAME" in environment}
    declared = {
        name[: -len(CAPABILITY_SUFFIX)]: value
        for name, value in environment.items()
        if name.startswith("ANTHROPIC_") and name.endswith(CAPABILITY_SUFFIX)
    }

    assert roster == {
        "ANTHROPIC_DEFAULT_SONNET_MODEL",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL",
        "ANTHROPIC_DEFAULT_OPUS_MODEL",
        "ANTHROPIC_DEFAULT_FABLE_MODEL",
        "ANTHROPIC_CUSTOM_MODEL_OPTION",
    }
    # Exact set, not membership: a slot that arrives without its declaration
    # leaves this mapping short of the roster above, and one that arrives in a
    # spelling the consumer does not read fails the comparison with it.
    assert declared == {
        "ANTHROPIC_DEFAULT_SONNET_MODEL": "thinking",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": "thinking",
        "ANTHROPIC_DEFAULT_OPUS_MODEL": "thinking",
        "ANTHROPIC_DEFAULT_FABLE_MODEL": "thinking",
        "ANTHROPIC_CUSTOM_MODEL_OPTION": "thinking",
    }
    # The hybrid branch is the one that carries the hosted slots, so naming
    # their models is what shows this exercised it rather than the local one.
    assert environment["ANTHROPIC_DEFAULT_SONNET_MODEL"] == "frontier-native-release"
    assert environment["ANTHROPIC_DEFAULT_HAIKU_MODEL"] == "secondary-native-release"
    assert environment["ANTHROPIC_DEFAULT_OPUS_MODEL"] == "or-opus-4.8"
    assert environment["ANTHROPIC_DEFAULT_FABLE_MODEL"] == "or-glm-5.2"
    assert environment["ANTHROPIC_CUSTOM_MODEL_OPTION"] == "or-gpt-5.5"
