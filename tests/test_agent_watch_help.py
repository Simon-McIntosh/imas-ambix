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


def test_default_record_dir_is_callable_and_resolves_to_the_site_config() -> None:
    """The reader's resolver is a callable whose result is the site's directory.

    A function rather than a constant, so it follows the environment the process
    actually runs in; and equal to ``SiteConfig.from_env().receipts_dir``, so
    the writer and the reader resolve one directory rather than two that agree.
    """
    assert callable(watch.default_record_dir)
    assert watch.default_record_dir() == SiteConfig.from_env().receipts_dir
