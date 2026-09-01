"""Focused contracts for Clive's explicit model-scope mode."""

from types import SimpleNamespace

from click.testing import CliRunner

from imas_ambix.agent.clive import generate_clive_script
from imas_ambix.agent.profile import SiteConfig
from imas_ambix.cli import main


def _invoke_print(*arguments: str, env: dict[str, str] | None = None):
    return CliRunner().invoke(
        main,
        ["agent", "clive", *arguments, "--print"],
        env=env,
    )


def test_implicit_and_explicit_local_generation_are_byte_identical():
    implicit = _invoke_print()
    explicit = _invoke_print("--mode", "local")

    assert implicit.exit_code == 0, implicit.output
    assert explicit.exit_code == 0, explicit.output
    assert implicit.output_bytes == explicit.output_bytes


def test_readable_hosted_credential_cannot_change_default_generation(tmp_path):
    home = tmp_path / "home"
    key_file = home / ".config" / "openrouter" / "key"
    key_file.parent.mkdir(parents=True)
    key_file.write_text("well-formed-personal-key\n", encoding="utf-8")
    key_file.chmod(0o600)

    implicit = _invoke_print(env={"HOME": str(home)})
    local = _invoke_print("--mode", "local", env={"HOME": str(home)})

    assert implicit.exit_code == 0, implicit.output
    assert implicit.output_bytes == local.output_bytes
    assert b"or-opus" not in implicit.output_bytes
    assert b"or-sonnet" not in implicit.output_bytes
    assert b"or-gpt" not in implicit.output_bytes


def test_hybrid_is_the_only_generation_mode_with_hosted_slots(monkeypatch):
    from imas_ambix.agent import cli as cli_mod

    profile = SimpleNamespace(model=SimpleNamespace(served_name="local-release"))
    monkeypatch.setattr(cli_mod, "_load_profile", lambda _slug: profile)

    local = _invoke_print("--mode", "local")
    hybrid = _invoke_print("local-profile", "--mode", "hybrid")

    assert local.exit_code == 0, local.output
    assert hybrid.exit_code == 0, hybrid.output
    for hosted_slot in (b"or-opus-4.8", b"or-glm-5.2", b"or-gpt-5.5"):
        assert hosted_slot not in local.output_bytes
        assert hosted_slot in hybrid.output_bytes
    assert b"or-sonnet-4.6" not in hybrid.output_bytes


def test_hybrid_installation_isolated_behind_explicit_mode(monkeypatch, tmp_path):
    from imas_ambix.agent import cli as cli_mod

    profile = SimpleNamespace(model=SimpleNamespace(served_name="local-release"))
    installed: list[tuple[SiteConfig, str]] = []
    monkeypatch.setattr(cli_mod, "_load_profile", lambda _slug: profile)
    monkeypatch.setattr(cli_mod, "_deploy_launcher", lambda *_args: None)
    monkeypatch.setattr(
        cli_mod,
        "_deploy_openrouter_proxy",
        lambda site, release: installed.append((site, release)),
    )

    local = CliRunner().invoke(
        main,
        [
            "agent",
            "clive",
            "--mode",
            "local",
            "--destination",
            str(tmp_path / "clive"),
        ],
    )
    assert local.exit_code == 0, local.output
    assert installed == []

    hybrid = CliRunner().invoke(
        main,
        [
            "agent",
            "clive",
            "local-profile",
            "--mode",
            "hybrid",
            "--destination",
            str(tmp_path / "clive"),
        ],
    )
    assert hybrid.exit_code == 0, hybrid.output
    assert len(installed) == 1
    assert installed[0][1] == "local-release"


def test_invalid_mode_names_valid_values_and_help_states_default():
    invalid = CliRunner().invoke(main, ["agent", "clive", "--mode", "remote"])
    help_result = CliRunner().invoke(main, ["agent", "clive", "--help"])

    assert invalid.exit_code != 0
    assert "local" in invalid.output
    assert "hybrid" in invalid.output
    assert help_result.exit_code == 0, help_result.output
    assert "--mode [local|hybrid]" in help_result.output
    assert "default: local" in help_result.output


def test_generated_launcher_keeps_mode_and_harness_axes_independent():
    local = generate_clive_script(SiteConfig(), mode="local")

    assert "--mode local|hybrid" in local
    assert '--claude) HARNESS="claude"' in local
    assert '--codex) HARNESS="codex"' in local
    assert 'local|hybrid) MODE="$2"' in local
    assert 'OPENAI_BASE_URL="${GLOBAL_ORIGIN}/v1"' in local
    assert 'ANTHROPIC_BASE_URL="$GLOBAL_ORIGIN"' in local
    assert 'ANTHROPIC_MODEL="$MODEL_ID"' in local
    assert "CLIVE_OPENROUTER" not in local


def test_local_generator_rejects_hosted_configuration():
    site = SiteConfig()

    try:
        generate_clive_script(
            site,
            mode="local",
            openrouter_native_release="local-release",
        )
    except ValueError as error:
        assert str(error) == "an OpenRouter native release requires hybrid mode"
    else:
        raise AssertionError("local generation accepted hosted configuration")
