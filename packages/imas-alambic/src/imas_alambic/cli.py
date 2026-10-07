"""The imas-alambic command line: ``write`` and ``config``, plus ``--version``.

``write`` resolves its settings from the flags on the command line and the
environment, so at the facility the only argument it needs is the pulse.
``config`` prints each resolved setting with the source it came from, which is
the only view of the precedence without reading the code.
"""

from __future__ import annotations

import click

from imas_alambic import __version__


@click.group(name="imas-alambic")
@click.version_option(__version__, prog_name="imas-alambic")
def main() -> None:
    """Write IMAS description IDSs from a facility map bundle."""


@main.command(name="write")
@click.argument("pulses", nargs=-1, required=True)
@click.option(
    "--run",
    "run",
    type=int,
    default=0,
    show_default=True,
    help="Run number, unpadded; defaults to 0.",
)
@click.option(
    "--machine",
    default=None,
    help="Machine name; inferred when exactly one machine is reachable.",
)
@click.option(
    "--maps",
    type=click.Path(),
    default=None,
    help="Map search path; the flag form of IMAS_ALAMBIC_MAP_PATH.",
)
@click.option(
    "--out",
    "out_dir",
    type=click.Path(),
    default=None,
    help="IDS root; the flag form of IMAS_ALAMBIC_IDS_ROOT.",
)
@click.option(
    "--cache",
    type=click.Path(),
    default=None,
    help="EDDB cache root; the flag form of IMAS_ALAMBIC_CACHE.",
)
@click.option(
    "--eddb-host",
    "eddb_host",
    default=None,
    help=(
        "EDDB transport: unset keeps the ssh route to jt-60sa, a host name "
        "composes the ssh route to that host, and local reads in place; the "
        "flag form of IMAS_ALAMBIC_EDDB_HOST."
    ),
)
@click.option(
    "--overwrite",
    is_flag=True,
    help="Replace an existing run file rather than refusing it.",
)
def write_cmd(
    pulses: tuple[str, ...],
    run: int,
    machine: str | None,
    maps: str | None,
    out_dir: str | None,
    cache: str | None,
    eddb_host: str | None,
    overwrite: bool,
) -> None:
    """Write each PULSE's description IDSs with the signals the maps serve.

    PULSE is the full EDDB token: its series letter and digits, for example
    E101154.  A bare number is refused before any fetch, because EDDB is
    addressed by the full token.
    """

    from imas_alambic.eddb import EddbCacheError
    from imas_alambic.machine_map import MachineMapError
    from imas_alambic.pulse_writer import PulseWriteError, write_pulse
    from imas_alambic.settings import (
        ENV_HOME,
        ENV_IDS_ROOT,
        SettingsFlags,
        require_setting,
        resolve_settings,
    )

    settings = resolve_settings(
        SettingsFlags(
            maps=maps,
            ids_root=out_dir,
            cache=cache,
            machine=machine,
            eddb_host=eddb_host,
        )
    )
    try:
        machine_name = require_setting("machine", settings.machine, "--machine")
        ids_root = require_setting(
            "ids_root", settings.ids_root, f"{ENV_IDS_ROOT} or {ENV_HOME}"
        )
    except MachineMapError as error:
        raise click.ClickException(str(error)) from error

    for pulse in pulses:
        try:
            receipt = write_pulse(
                machine_name,
                pulse,
                ids_root,
                run=run,
                overwrite=overwrite,
                maps=maps,
                cache=cache,
                eddb_host=eddb_host,
            )
        except (PulseWriteError, EddbCacheError) as error:
            raise click.ClickException(str(error)) from error

        click.echo(f"Wrote {receipt.shot} run {receipt.run} into {receipt.path}")
        for ids_name in receipt.ids_written:
            leaves = sum(1 for leaf in receipt.leaves if leaf.ids == ids_name)
            click.echo(f"  {ids_name}: {leaves} time-dependent leaves")
        if receipt.excluded:
            click.echo(
                f"  {len(receipt.excluded)} declared signals left out (see receipt)"
            )


@main.command(name="config")
def config_cmd() -> None:
    """Print each resolved setting with the source it came from."""

    from imas_alambic.settings import resolve_settings

    for name, setting in resolve_settings().as_pairs():
        value = "(unset)" if setting.value is None else str(setting.value)
        click.echo(f"{name}: {value}  [{setting.source}]")


if __name__ == "__main__":  # pragma: no cover - module entry point
    main()
