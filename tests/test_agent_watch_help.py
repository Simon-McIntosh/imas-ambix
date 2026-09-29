"""The ``agent watch`` help names the resolved record source, not a stale path.

The ``--record`` option resolves its default at call time through
``imas_ambix.agent.watch.default_record_dir`` and the site configuration, so the
directory the recorder writes and the directory the reader discovers are one by
construction. A constant written into the help text cannot follow that: it was
still naming ``~/.local/share/ambix/receipts`` after the default moved to the
site's shared base, which is a reader pointed where nothing writes. These pin
the help to the mechanism so it cannot drift back.
"""

from __future__ import annotations

from click.testing import CliRunner

from imas_ambix.agent import cli

#: The pre-site-config default the help text used to name.
STALE_DEFAULT = "~/.local/share/ambix/receipts"


def _watch_help() -> str:
    """The rendered ``--help`` output for the watch command, whitespace-flattened.

    Flattened so a wrap that lands between the tokens this test looks for cannot
    make an absent string read as present, or a present one as absent.
    """
    result = CliRunner().invoke(cli.agent, ["watch", "--help"])
    assert result.exit_code == 0, result.output
    return " ".join(result.output.split())


def test_watch_help_does_not_name_the_superseded_record_default() -> None:
    """The help must not point the reader at a directory nothing writes."""
    assert STALE_DEFAULT not in _watch_help()


def test_watch_help_names_the_site_configuration_source() -> None:
    """The help must state where the default actually comes from."""
    text = _watch_help()
    assert "watch.default_record_dir" in text
    assert "site configuration" in text
