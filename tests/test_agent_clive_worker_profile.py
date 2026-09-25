"""Contracts for the opt-in Clive worker-profile expansion."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

from imas_ambix.agent.clive import generate_clive_script
from tests.agent.catalog_fixture import serve_catalog_items


def _catalog_item(model_id: str) -> dict[str, object]:
    return {
        "id": model_id,
        "max_model_len": 524_288,
        "ambix": {
            "accelerator_family": "H200",
            "accelerator_count": 2,
            "checkpoint_precision": "fp8",
        },
    }


def _launch(
    tmp_path: Path,
    *,
    profile: Path | None = None,
    user_settings: Path | None = None,
    agents: bool = True,
    receipt: Path | None = None,
):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(exist_ok=True)
    args_path = tmp_path / "claude-args"
    settings_path = tmp_path / "claude-settings"
    # The stub records the argv and the settings file it was handed. It must
    # not model whether a hook fires: that was decided by the live launch
    # named in the expansion record, so a stub that wrote a marker here would
    # let this suite claim a hook outcome it never measured.
    fake_claude = fake_bin / "claude"
    fake_claude.write_text(
        "#!/bin/sh\n"
        f"printf '%s\\0' \"$@\" > {args_path}\n"
        "settings=''\n"
        "previous=''\n"
        'for value in "$@"; do\n'
        "  if [ \"$previous\" = '--settings' ]; then settings=$value; fi\n"
        "  previous=$value\n"
        "done\n"
        f'python3 - "$settings" {settings_path} <<\'PY\'\n'
        "import json\n"
        "import pathlib\n"
        "import sys\n"
        "value = sys.argv[1]\n"
        "settings = (\n"
        "    json.loads(value)\n"
        "    if value.startswith('{')\n"
        "    else json.loads(pathlib.Path(value).read_text())\n"
        ")\n"
        "pathlib.Path(sys.argv[2]).write_text(json.dumps(settings))\n"
        "PY\n",
        encoding="utf-8",
    )
    fake_claude.chmod(0o755)
    if agents:
        (tmp_path / "AGENTS.md").write_text("repository guidance\n", encoding="utf-8")

    digest = tmp_path / "role-digest.txt"
    mcp = tmp_path / "mcp.json"
    profile_path = tmp_path / "worker-profile.json"
    if profile is None:
        digest.write_text("role digest\n", encoding="utf-8")
        mcp.write_text('{"mcpServers":{}}\n', encoding="utf-8")
        profile_path.write_text(
            json.dumps(
                {
                    "digest": str(digest),
                    "mcp_config": str(mcp),
                    "tools": ["Bash", "Read"],
                }
            ),
            encoding="utf-8",
        )
    else:
        profile_path = profile

    with serve_catalog_items([_catalog_item("release")]) as (site, _requests):
        launcher = tmp_path / "clive"
        launcher.write_text(generate_clive_script(site), encoding="utf-8")
        launcher.chmod(0o755)
        environment = os.environ.copy()
        environment["PATH"] = f"{fake_bin}:{environment['PATH']}"
        environment["CLIVE_USER_SETTINGS"] = str(
            user_settings or (tmp_path / "missing-settings.json")
        )
        if receipt is not None:
            environment["CLIVE_PROFILE_RECEIPT"] = str(receipt)
        if profile is None:
            command = [str(launcher), "--selector", "release"]
        else:
            command = [
                str(launcher),
                "--selector",
                "release",
                "--worker-profile",
                str(profile),
            ]
        result = subprocess.run(
            command,
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=False,
            env=environment,
        )
    args = (
        [value.decode() for value in args_path.read_bytes().split(b"\0")[:-1]]
        if args_path.exists()
        else []
    )
    settings = (
        json.loads(settings_path.read_text(encoding="utf-8"))
        if settings_path.exists()
        else {}
    )
    return result, args, settings, profile_path, digest, mcp


def test_no_profile_exec_arguments_match_the_baseline(tmp_path):
    result, args, _settings, _profile, _digest, _mcp = _launch(tmp_path)

    assert result.returncode == 0, result.stderr
    assert args[0] == "--settings"
    assert json.loads(args[1])["modelPicker"]["options"][0]["model"] == "release"
    expected_guidance = (
        "Clive dispatch guidance: the sonnet alias is the primary local worker "
        "and resolves to release (2×H200, 524,288-token engine-reported context). "
        "Send bulk, parallel, and mechanical work to sonnet. Keep adjudication "
        "and physics-critical judgement on a frontier slot when one is available; "
        "local-only mode does not provide a frontier slot."
    )
    assert args[2:] == [
        "--append-system-prompt",
        expected_guidance,
    ]
    # The whole exec statement is the base revision's, verbatim: a launch
    # without a profile takes the same line it always did.
    base_exec = (
        'exec claude --settings "$PICKER_SETTINGS" --append-system-prompt'
        ' "$DISPATCH_GUIDANCE" "${ARGS[@]}"'
    )
    script = (tmp_path / "clive").read_text(encoding="utf-8")
    assert script.count(base_exec) == 1


def test_profile_expands_verified_flags_and_records_file_digests(tmp_path):
    receipt = tmp_path / "profile-receipt.jsonl"
    result, args, _settings, profile, digest, mcp = _launch(
        tmp_path, profile=None
    )
    assert result.returncode == 0, result.stderr

    fake_bin = tmp_path / "profile-bin"
    fake_bin.mkdir()
    api_key = tmp_path / "api-key"
    (fake_bin / "claude").write_text(
        f"#!/bin/sh\nprintf '%s' \"$ANTHROPIC_API_KEY\" > {api_key}\nexit 0\n",
        encoding="utf-8",
    )
    (fake_bin / "claude").chmod(0o755)
    # Reuse the launch fixture's profile but make its receipt explicit.
    with serve_catalog_items([_catalog_item("release")]) as (site, _requests):
        launcher = tmp_path / "profile-clive"
        launcher.write_text(generate_clive_script(site), encoding="utf-8")
        launcher.chmod(0o755)
        environment = os.environ.copy()
        environment["PATH"] = f"{fake_bin}:{environment['PATH']}"
        environment["CLIVE_USER_SETTINGS"] = str(tmp_path / "missing-settings.json")
        environment["CLIVE_PROFILE_RECEIPT"] = str(receipt)
        result = subprocess.run(
            [str(launcher), "--selector", "release", "--worker-profile", str(profile)],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=False,
            env=environment,
        )

    assert result.returncode == 0, result.stderr
    record = json.loads(receipt.read_text(encoding="utf-8").splitlines()[0])
    assert record["route"] == "bare"
    assert (
        record["files"]["profile"]["sha256"]
        == hashlib.sha256(profile.read_bytes()).hexdigest()
    )
    assert (
        record["files"]["digest"]["sha256"]
        == hashlib.sha256(digest.read_bytes()).hexdigest()
    )
    assert (
        record["files"]["mcp_config"]["sha256"]
        == hashlib.sha256(mcp.read_bytes()).hexdigest()
    )
    assert "--bare" in record["added_arguments"]
    assert "--strict-mcp-config" in record["added_arguments"]
    assert "--mcp-config" in record["added_arguments"]
    assert "--tools" in record["added_arguments"]
    assert "Bash,Read" in record["added_arguments"]
    assert api_key.read_text(encoding="utf-8") == "clive-no-auth"
    help_result = subprocess.run(
        ["claude", "--help"], capture_output=True, text=True, check=True
    )
    help_text = help_result.stdout + help_result.stderr
    for flag in (
        "--bare",
        "--append-system-prompt",
        "--strict-mcp-config",
        "--mcp-config",
        "--tools",
        "--setting-sources",
        "--settings",
    ):
        assert flag in help_text


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda data: data.pop("digest"), "profile must contain"),
        (
            lambda data: data.update({"digest": str(Path("missing-digest.txt"))}),
            "digest does not exist",
        ),
        (
            lambda data: data.update({"mcp_config": str(Path("missing-mcp.json"))}),
            "mcp_config does not exist",
        ),
        (lambda data: data.update({"tools": []}), "tools must be a non-empty list"),
    ],
)
def test_malformed_profiles_are_refused(tmp_path, mutator, message):
    _result, _args, _settings, profile, _digest, _mcp = _launch(tmp_path)
    data = json.loads(profile.read_text(encoding="utf-8"))
    mutator(data)
    profile.write_text(json.dumps(data), encoding="utf-8")
    result, *_ = _launch(tmp_path, profile=profile)

    assert result.returncode == 2
    assert message in result.stderr


def test_missing_profile_is_refused(tmp_path):
    missing = tmp_path / "missing-profile.json"
    result, *_ = _launch(tmp_path, profile=missing)

    assert result.returncode == 2
    assert "profile does not exist" in result.stderr


def test_profile_without_agents_file_still_expands(tmp_path):
    _first, _args, _settings, profile, _digest, _mcp = _launch(
        tmp_path, agents=False
    )
    result, args, _settings, _profile, _digest, _mcp = _launch(
        tmp_path, profile=profile, agents=False
    )

    assert result.returncode == 0, result.stderr
    assert "--bare" in args
    prompt = args[args.index("--append-system-prompt") + 1]
    assert "role digest" in prompt
    assert "Repository guidance" not in prompt


def test_hooks_are_carried_and_force_the_settings_route(tmp_path):
    settings = tmp_path / "user-settings.json"
    hooks = {
        "Stop": [{"hooks": [{"type": "command", "command": "stop-hook"}]}],
        "PreToolUse": [{"hooks": [{"type": "command", "command": "guard-hook"}]}],
    }
    settings.write_text(json.dumps({"hooks": hooks}), encoding="utf-8")
    receipt = tmp_path / "hook-receipt.jsonl"
    _first, _args, _expanded, profile, _digest, _mcp = _launch(tmp_path)
    result, args, expanded, _profile, _digest, _mcp = _launch(
        tmp_path, profile=profile, user_settings=settings, receipt=receipt
    )

    assert result.returncode == 0, result.stderr
    # The hooks force the explicit-sources route: no --bare, an empty
    # --setting-sources so no config source is loaded from disk, and the hook
    # block read from the fixture user settings merged into the settings file
    # the harness is handed. Whether a hook in that file fires was decided by a
    # recorded live launch, not by this stub.
    assert "--bare" not in args
    assert args[args.index("--setting-sources") + 1] == ""
    assert expanded["hooks"] == hooks
    record = json.loads(receipt.read_text(encoding="utf-8").splitlines()[0])
    assert record["route"] == "settings"
    assert record["hooks"] == ["stop-hook", "guard-hook"]
    # The receipt and the emitted command share one definition of the added
    # arguments: what the record claims was added is exactly the prefix the
    # harness stub observed on its own command line.
    added = record["added_arguments"]
    assert added[:2] == ["--setting-sources", ""]
    assert args[: len(added)] == added
    prompt = args[args.index("--append-system-prompt") + 1]
    assert "role digest" in prompt
    assert "repository guidance" in prompt
    # The evidence pointer is the full report path, not a basename a reader
    # cannot open.
    assert record["evidence"]["hook_route_measurement"] == (
        "/home/ITER/mcintos/.config/reckon/crew/reports/cwp-settings-hook-check.md"
    )


def test_expansion_never_makes_the_global_agents_file_reachable(tmp_path):
    """A launch must not regain the global guidance the profile omits.

    --add-dir would grant the harness an extra directory whose CLAUDE.md it
    discovers, and a worktree's CLAUDE.md includes ~/.agents/AGENTS.md, so an
    --add-dir in the expansion silently reloads the global file the profile
    exists to drop.
    """
    _first, _args, _settings, profile, _digest, _mcp = _launch(tmp_path)
    result, args, _settings, _profile, _digest, _mcp = _launch(
        tmp_path, profile=profile
    )

    assert result.returncode == 0, result.stderr
    assert "--add-dir" not in args
    prompt = args[args.index("--append-system-prompt") + 1]
    global_agents = Path.home() / ".agents" / "AGENTS.md"
    if global_agents.is_file():
        assert global_agents.read_text(encoding="utf-8") not in prompt
