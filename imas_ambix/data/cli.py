"""CLI commands for managing the FAIR-MAST mirror.

Subcommands:

- ``ambix data probe``               run the sizing probe (sirius node)
- ``ambix data manifest``            build / emit shot-id manifests
- ``ambix data download``            (plan-only for now) print the SLURM
                                      bulk-download script

The bulk-download submission itself is intentionally not wired up to
``sbatch`` from Python in v0 — see ``plans/data-acquisition.md`` §4. The
operator submits the script by hand the first time so we know the job
runs as expected; automation lands in a follow-up once the protocol is
proven.
"""

from __future__ import annotations

import json
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from imas_ambix.data.paths import (
    LEVEL2_DIR,
    MANIFEST_DIR,
    MIRROR_ROOT,
    PROBE_DIR,
    S3_BUCKET,
    S3_ENDPOINT,
)

console = Console()


@click.group(name="data")
def data() -> None:
    """FAIR-MAST data acquisition and access.

    See ``plans/data-acquisition.md`` for the end-to-end protocol.
    """


@data.command(name="probe")
@click.option(
    "--sample-size",
    default=50,
    show_default=True,
    help="Number of shots to copy for sizing + throughput measurement.",
)
@click.option(
    "--numworkers",
    default=32,
    show_default=True,
    help="s5cmd --numworkers value.",
)
@click.option(
    "--timeout-s",
    default=600.0,
    show_default=True,
    help="Per-shot s5cmd timeout in seconds.",
)
@click.option(
    "--camera-only",
    is_flag=True,
    default=False,
    help="Restrict the sample to camera-bearing shots.",
)
@click.option(
    "--output-dir",
    default=str(PROBE_DIR),
    show_default=True,
    help="Directory to write the JSON report into (created if missing).",
)
@click.option(
    "--seed",
    default=0,
    show_default=True,
    help="Random seed for shot sampling.",
)
def probe_cmd(
    sample_size: int,
    numworkers: int,
    timeout_s: float,
    camera_only: bool,
    output_dir: str,
    seed: int,
) -> None:
    """Run the FAIR-MAST sizing probe (intended for a sirius compute node)."""
    from imas_ambix.data.probe import run_probe

    report = run_probe(
        sample_size=sample_size,
        numworkers=numworkers,
        timeout_s=timeout_s,
        output_dir=Path(output_dir),
        keep_samples=False,
        camera_only=camera_only,
        seed=seed,
    )

    _render_report(report)


def _render_report(report: object) -> None:
    """Render a :class:`ProbeReport` as a rich table."""
    from imas_ambix.data.probe import ProbeReport

    if not isinstance(report, ProbeReport):  # narrow for the type checker
        raise TypeError("expected ProbeReport")

    table = Table(title="FAIR-MAST sizing probe", show_lines=False)
    table.add_column("metric")
    table.add_column("value")
    table.add_column("gate")

    acc = report.acceptance_summary()
    table.add_row(
        "throughput",
        f"{report.sustained_throughput_mbps:.0f} MB/s",
        acc["throughput"],
    )
    table.add_row(
        "median shot size",
        f"{report.median_shot_size_mb:.1f} MB",
        "-",
    )
    table.add_row(
        "p95 shot size",
        f"{report.p95_shot_size_mb:.1f} MB",
        acc["per_shot_p95"],
    )
    table.add_row(
        "extrapolated total",
        f"{report.extrapolated_total_size_tb:.2f} TB",
        acc["total_size"],
    )
    table.add_row(
        "camera-bearing shots",
        f"{report.n_camera_shots} / {report.n_shots_in_index}",
        acc["camera_shots"],
    )
    console.print(table)
    for note in report.notes:
        console.print(f"[yellow]note:[/yellow] {note}")


@data.command(name="manifest")
@click.option(
    "--camera-only",
    is_flag=True,
    default=False,
    help="Restrict the manifest to camera-bearing shots.",
)
@click.option(
    "--emit-ids",
    is_flag=True,
    default=False,
    help="Print just the shot IDs (newline-separated) for piping to s5cmd.",
)
@click.option(
    "--output",
    default=None,
    type=click.Path(),
    help="Path to write the JSON manifest to (default: stdout).",
)
def manifest_cmd(camera_only: bool, emit_ids: bool, output: str | None) -> None:
    """Build a shot manifest from the level-2 parquet index."""
    from imas_ambix.data.manifest import build_manifest, emit_shot_ids, load_index

    df = load_index()
    manifest = build_manifest(df, camera_only=camera_only)

    if emit_ids:
        click.echo(emit_shot_ids(manifest), nl=False)
        return

    payload = manifest.to_json()
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(payload, encoding="utf-8")
        console.print(
            f"[green]manifest written:[/green] {output} "
            f"({len(manifest.shot_ids)} shots)"
        )
    else:
        click.echo(payload)


@data.command(name="download")
@click.option(
    "--plan-only",
    is_flag=True,
    default=True,
    show_default=True,
    help="Print the SLURM bulk-download script instead of submitting.",
)
@click.option(
    "--partition",
    default="sun",
    show_default=True,
    help="SLURM partition (must have outbound network + GPFS access).",
)
@click.option(
    "--time-limit",
    default="24:00:00",
    show_default=True,
    help="SLURM --time value.",
)
@click.option(
    "--camera-only",
    is_flag=True,
    default=False,
    help=(
        "Only download camera-bearing shots (Pass A in plans/data-acquisition.md §4.5)."
    ),
)
def download_cmd(
    plan_only: bool,
    partition: str,
    time_limit: str,
    camera_only: bool,
) -> None:
    """Render the SLURM bulk-download script (submission is operator-driven)."""
    if not plan_only:
        raise click.UsageError(
            "Submission via sbatch is intentionally not wired up in v0. "
            "Pass --plan-only and submit the rendered script by hand. "
            "See plans/data-acquisition.md §4."
        )

    manifest_filter = "--camera-only" if camera_only else ""
    pass_label = "camera" if camera_only else "all"

    script = _render_slurm_script(
        partition=partition,
        time_limit=time_limit,
        dest=LEVEL2_DIR,
        manifest_filter=manifest_filter,
        pass_label=pass_label,
    )
    click.echo(script)


def _render_slurm_script(
    *,
    partition: str,
    time_limit: str,
    dest: Path,
    manifest_filter: str,
    pass_label: str,
) -> str:
    return f"""#!/bin/bash
#SBATCH --job-name=mast-mirror-{pass_label}
#SBATCH --partition={partition}
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time={time_limit}
#SBATCH --output=%x-%j.log

set -euo pipefail
export TMPDIR=/scratch_local/$SLURM_JOB_ID && mkdir -p "$TMPDIR"
export PATH="$HOME/.local/bin:$PATH"

DEST={dest}
mkdir -p "$DEST"

python -m imas_ambix.data.cli manifest --emit-ids {manifest_filter} \\
| while read SHOT; do
    s5cmd --no-sign-request --endpoint-url {S3_ENDPOINT} \\
      --numworkers 32 \\
      cp "s3://{S3_BUCKET}/level2/shots/${{SHOT}}.zarr/*" "${{DEST}}/${{SHOT}}.zarr/"
  done
""".lstrip()


@data.command(name="status")
def status_cmd() -> None:
    """Show local mirror progress against the latest manifest."""
    if not MIRROR_ROOT.exists():
        console.print(f"[yellow]mirror root does not exist yet:[/yellow] {MIRROR_ROOT}")
        return

    n_local = (
        sum(1 for p in LEVEL2_DIR.glob("*.zarr") if p.is_dir())
        if LEVEL2_DIR.exists()
        else 0
    )

    # Use the most recent manifest if present.
    manifest_files = (
        sorted(MANIFEST_DIR.glob("*.json")) if MANIFEST_DIR.exists() else []
    )
    if manifest_files:
        latest = manifest_files[-1]
        payload = json.loads(latest.read_text(encoding="utf-8"))
        target = len(payload.get("shot_ids", []))
        console.print(
            f"mirror progress: [green]{n_local}[/green] / {target} shots "
            f"(manifest: {latest.name})"
        )
    else:
        console.print(
            f"mirror progress: [green]{n_local}[/green] / ? shots "
            f"(no manifest under {MANIFEST_DIR})"
        )
