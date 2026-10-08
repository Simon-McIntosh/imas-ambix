"""The engine's command line exposes ``write`` and ``config``.

``write`` takes positional pulses and the options that form each resolved
setting; ``config`` prints every resolved setting with its source.  This test
reads the group's own listing rather than spelling the retired verb.
"""

from __future__ import annotations

from click.testing import CliRunner

from imas_alambic.cli import main


def test_help_lists_the_write_command_and_its_options():
    group = CliRunner().invoke(main, ["--help"])
    assert group.exit_code == 0, group.output
    assert "write" in group.output
    assert sorted(main.commands) == ["config", "write"]

    command = CliRunner().invoke(main, ["write", "--help"])
    assert command.exit_code == 0, command.output
    for option in ("--run", "--machine", "--maps", "--out", "--cache", "--overwrite"):
        assert option in command.output
    assert "E101154" in command.output


def test_config_prints_each_setting_with_its_source(monkeypatch, tmp_path):
    import json

    from imas_alambic.settings import ENV_HOME

    monkeypatch.setattr("imas_alambic.machine_map.entry_points", lambda group: [])
    home = tmp_path / "IMASDB"
    bundle = home / "maps" / "current"
    bundle.mkdir(parents=True)
    (bundle / "bundle.json").write_text(
        json.dumps({"name": "facility", "version": "1.0.0", "machine": "jt-60sa"})
    )
    monkeypatch.setenv(ENV_HOME, str(home))

    result = CliRunner().invoke(main, ["config"])

    assert result.exit_code == 0, result.output
    for name in ("maps", "ids_root", "cache", "machine"):
        assert f"{name}:" in result.output
    assert str(bundle) in result.output
    assert str(home / "ids") in result.output
    assert "inferred" in result.output


def test_write_refuses_a_bare_number_through_the_cli_before_any_transport(
    tmp_path, monkeypatch
):
    from test_write_fetch import _prepare, _signal

    machine = "synth-machine"
    bundle, transport = _prepare(
        tmp_path,
        monkeypatch,
        machine,
        (_signal("ip", "plasma_current", "magnetics/ip/data"),),
    )
    cache = tmp_path / "cache"
    out = tmp_path / "ids"

    result = CliRunner().invoke(
        main,
        [
            "write",
            "900001",
            "--machine",
            machine,
            "--maps",
            str(bundle),
            "--cache",
            str(cache),
            "--out",
            str(out),
        ],
    )

    assert result.exit_code != 0, result.output
    assert "E900001" in result.output
    assert "series letter" in result.output
    assert transport.calls == 0
    assert not (cache / "900001.zarr").is_dir()
