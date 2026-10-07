"""The imas-alambic command line: one command, ``write``, plus ``--version``."""

from __future__ import annotations

import click

from imas_alambic import __version__


@click.group(name="imas-alambic")
@click.version_option(__version__, prog_name="imas-alambic")
def main() -> None:
    """Write IMAS description IDSs from a facility map bundle."""


@main.command(name="write")
@click.option("--machine", required=True, help="Machine catalogue name, e.g. jt-60sa.")
@click.option(
    "--shot",
    required=True,
    help="EDDB shot token, e.g. E101154 or 101154.",
)
@click.option(
    "--out",
    "out_dir",
    required=True,
    type=click.Path(),
    help="Output root; one directory per shot is written under it.",
)
def write_cmd(machine: str, shot: str, out_dir: str) -> None:
    """Write a pulse's description IDSs with the signals the maps serve."""

    from imas_alambic.pulse_writer import PulseWriteError, write_pulse

    try:
        receipt = write_pulse(machine, shot, out_dir)
    except PulseWriteError as error:
        raise click.ClickException(str(error)) from error

    click.echo(f"Wrote {receipt.shot} phase {receipt.phase} into {receipt.out_dir}")
    for ids_name in receipt.ids_written:
        leaves = sum(1 for leaf in receipt.leaves if leaf.ids == ids_name)
        click.echo(f"  {ids_name}: {leaves} time-dependent leaves")
    if receipt.excluded:
        click.echo(f"  {len(receipt.excluded)} declared signals left out (see receipt)")


if __name__ == "__main__":  # pragma: no cover - module entry point
    main()
