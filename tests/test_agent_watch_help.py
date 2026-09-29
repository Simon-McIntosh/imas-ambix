"""The ``agent watch`` and ``agent receipts`` help name the resolved record source.

Both commands resolve the receipts directory at call time through
``imas_ambix.agent.watch.default_record_dir`` and the site configuration, so the
directory the recorder writes and the directory the reader discovers are one by
construction. A constant written into the help text cannot follow that: it was
still naming ``~/.local/share/ambix/receipts`` after the default moved to the
site's shared base, which is a reader -- or a recorder -- pointed where nothing
else looks. These pin the help to the mechanism so it cannot drift back.
"""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from imas_ambix.agent import cli, watch
from imas_ambix.agent.profile import SiteConfig

#: The pre-site-config default the help text used to name.
STALE_DEFAULT = "~/.local/share/ambix/receipts"


def _help(*argv: str) -> str:
    """The rendered ``--help`` output, whitespace-flattened.

    Flattened so a wrap that lands between the tokens these tests look for
    cannot make an absent string read as present, or a present one as absent.
    """
    result = CliRunner().invoke(cli.agent, [*argv, "--help"])
    assert result.exit_code == 0, result.output
    return " ".join(result.output.split())


def test_watch_help_does_not_name_the_superseded_record_default() -> None:
    """The help must not point the reader at a directory nothing writes."""
    assert STALE_DEFAULT not in _help("watch")


def test_watch_help_names_the_site_configuration_source() -> None:
    """The help must state where the default actually comes from."""
    text = _help("watch")
    assert "watch.default_record_dir" in text
    assert "site configuration" in text


def test_receipts_help_does_not_name_the_superseded_record_default() -> None:
    """The recorder's help must not point writing at a directory nothing reads."""
    assert STALE_DEFAULT not in _help("receipts")


def test_receipts_help_names_the_site_configuration_source() -> None:
    """The help must state where the output default actually comes from."""
    text = _help("receipts")
    assert "watch.default_record_dir" in text
    assert "site configuration" in text


def test_receipts_help_names_the_base_dir_variable_and_its_default() -> None:
    """The output help names the variable that relocates the default."""
    text = _help("receipts")
    assert "AMBIX_AGENT_BASE_DIR/agents/receipts" in text
    assert "/work/projects/imas_gpu" in text


def test_watch_help_names_the_base_dir_variable_and_its_default() -> None:
    """The reader's --record help names the variable that relocates it."""
    text = _help("watch")
    assert "AMBIX_AGENT_BASE_DIR/agents/receipts" in text
    assert "/work/projects/imas_gpu" in text


def test_ingest_help_names_the_base_dir_variable_and_its_default() -> None:
    """The ingest's --record help names the variable that relocates it."""
    text = _help("ingest")
    assert "AMBIX_AGENT_BASE_DIR/agents/receipts" in text
    assert "/work/projects/imas_gpu" in text


def test_receipts_default_output_follows_the_base_dir_variable(
    tmp_path, monkeypatch
) -> None:
    """Relocating the base relocates the recorder's default file.

    The help claims AMBIX_AGENT_BASE_DIR moves the directory, so this holds it
    to that: with the base set to a scratch tree, ``agent receipts`` resolves
    its default output file under ``<base>/agents/receipts`` through the same
    ``watch.default_record_dir`` the reader uses. The sampler is replaced with a
    stub, so no network poll runs and nothing is fetched.
    """
    from imas_ambix.agent import serving_receipts

    monkeypatch.setenv("AMBIX_AGENT_BASE_DIR", str(tmp_path))
    # No profile and no key file, so the command takes the bare --url path and
    # reads nothing from the environment beyond the base directory under test.
    monkeypatch.setattr(cli, "_default_profile", lambda: None)
    monkeypatch.setattr(cli, "_resolve_api_key", lambda _value: None)

    captured: list[Path] = []

    def fake_record(base_url, path, **kwargs):
        captured.append(Path(path))
        return 0

    monkeypatch.setattr(serving_receipts, "record_receipts", fake_record)

    result = CliRunner().invoke(
        cli.agent, ["receipts", "--url", "http://localhost:18800", "--duration", "0"]
    )

    assert result.exit_code == 0, result.output
    assert watch.default_record_dir() == tmp_path / "agents" / "receipts"
    assert captured == [tmp_path / "agents" / "receipts" / "endpoint.jsonl"]
    assert captured[0].parent == watch.default_record_dir()


def test_default_record_dir_is_callable_and_resolves_to_the_site_config() -> None:
    """The reader's resolver is a callable whose result is the site's directory.

    A function rather than a constant, so it follows the environment the process
    actually runs in; and equal to ``SiteConfig.from_env().receipts_dir``, so
    the writer and the reader resolve one directory rather than two that agree.
    """
    assert callable(watch.default_record_dir)
    assert watch.default_record_dir() == SiteConfig.from_env().receipts_dir
