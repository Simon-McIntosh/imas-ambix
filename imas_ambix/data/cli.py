"""CLI commands for managing the FAIR-MAST mirror.

Subcommands:

- ``ambix data probe``         sizing + group-inventory probe (any host)
- ``ambix data inventory``     bulk shot → groups listing
- ``ambix data manifest``      build / emit shot-id manifests
- ``ambix data download``      (plan-only) print the SLURM bulk-download script
- ``ambix data status``        local mirror progress

The actual ``sbatch`` submission is intentionally not wired up to a
``submit`` flag in v0 — operators submit the rendered script by hand
the first time so we can spot pathology before scaling up.
"""

from __future__ import annotations

import json
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from imas_ambix.data.paths import (
    CAMERA_SOURCES,
    LEVEL1_DIR,
    LEVEL2_DIR,
    MANIFEST_DIR,
    MIRROR_ROOT,
    PROBE_DIR,
    S3_ENDPOINT,
)

console = Console()


@click.group(name="data")
def data() -> None:
    """FAIR-MAST data acquisition and access.

    See ``plans/data-acquisition.md`` for the end-to-end protocol.
    """


def _tier_option(fn):
    return click.option(
        "--tier",
        type=click.Choice(["level1", "level2"]),
        default="level2",
        show_default=True,
        help="Which FAIR-MAST tier to probe / inventory / download.",
    )(fn)


# --- probe ------------------------------------------------------------


@data.command(name="probe")
@_tier_option
@click.option(
    "--sample-size",
    default=20,
    show_default=True,
    help="Number of shots to copy for sizing + throughput measurement.",
)
@click.option(
    "--groups",
    default="",
    help=(
        "Comma-separated group names to fetch per sampled shot. "
        "Empty = whole shot. Useful for tier=level1 to probe a camera subset."
    ),
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
@click.option(
    "--shots-in-tier",
    default=None,
    type=int,
    help=(
        "Authoritative shot count at this tier (e.g. from "
        "`s5cmd ls s3://mast/level1/shots/ | wc -l`). "
        "Default = index size."
    ),
)
def probe_cmd(
    tier: str,
    sample_size: int,
    groups: str,
    numworkers: int,
    timeout_s: float,
    output_dir: str,
    seed: int,
    shots_in_tier: int | None,
) -> None:
    """Run the FAIR-MAST sizing + group-inventory probe."""
    from imas_ambix.data.probe import run_probe

    group_tuple = tuple(g.strip() for g in groups.split(",") if g.strip())

    report = run_probe(
        sample_size=sample_size,
        tier=tier,  # type: ignore[arg-type]
        groups=group_tuple,
        numworkers=numworkers,
        timeout_s=timeout_s,
        output_dir=Path(output_dir),
        seed=seed,
        n_shots_in_tier=shots_in_tier,
    )

    _render_report(report)


def _render_report(report: object) -> None:
    """Render a :class:`ProbeReport` as a rich table."""
    from imas_ambix.data.probe import ProbeReport

    if not isinstance(report, ProbeReport):
        raise TypeError("expected ProbeReport")

    acc = report.acceptance_summary()

    table = Table(title=f"FAIR-MAST sizing probe ({report.tier})")
    table.add_column("metric")
    table.add_column("value")
    table.add_column("gate")
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
        f"{report.extrapolated_total_size_tb:.3f} TB",
        acc["total_size"],
    )
    table.add_row(
        "camera coverage",
        f"{report.camera_coverage_fraction * 100:.1f}% of sample",
        acc["camera_coverage"],
    )
    table.add_row(
        "shots in tier",
        f"{report.n_shots_in_tier}",
        "-",
    )
    console.print(table)

    if report.group_coverage:
        coverage = Table(title="Group coverage in sample")
        coverage.add_column("group")
        coverage.add_column("shots", justify="right")
        for name, n in report.group_coverage.items():
            coverage.add_row(name, str(n))
        console.print(coverage)

    for note in report.notes:
        console.print(f"[yellow]note:[/yellow] {note}")


# --- inventory --------------------------------------------------------


@data.command(name="inventory")
@_tier_option
@click.option(
    "--sample-size",
    default=100,
    show_default=True,
    help="How many shots to inventory.",
)
@click.option(
    "--seed",
    default=0,
    show_default=True,
)
@click.option(
    "--output",
    default=None,
    type=click.Path(),
    help="Write the JSON inventory to this path (default: stdout summary).",
)
@click.option(
    "--shot-ids",
    default=None,
    help=(
        "Comma-separated explicit shot IDs to inventory (overrides "
        "--sample-size and --seed)."
    ),
)
@click.option(
    "--from-bucket",
    is_flag=True,
    default=False,
    help=(
        "List shot IDs by `s5cmd ls` against the bucket prefix instead of "
        "the parquet index. Required at tier=level1 — the level-1 bucket "
        "has more shots than the level-2 parquet index."
    ),
)
@click.option(
    "--workers",
    default=16,
    show_default=True,
    help="Number of parallel `s5cmd ls` threads.",
)
def inventory_cmd(
    tier: str,
    sample_size: int,
    seed: int,
    output: str | None,
    shot_ids: str | None,
    from_bucket: bool,
    workers: int,
) -> None:
    """List which IDS groups are present per shot at the given tier.

    This drives the v0 download decision — pick the subset of shots
    that carry cameras / required diagnostics.
    """
    from imas_ambix.data.manifest import (
        group_coverage,
        inventory_groups,
        load_index,
        shot_ids_from_bucket,
    )
    from imas_ambix.data.probe import sample_shots

    if shot_ids:
        ids = [int(s.strip()) for s in shot_ids.split(",") if s.strip()]
    elif from_bucket:
        ids = list(shot_ids_from_bucket(tier))  # type: ignore[arg-type]
    else:
        df = load_index()
        ids = sample_shots(df, sample_size, seed=seed)

    console.print(
        f"inventorying {len(ids)} shots at tier=[bold]{tier}[/bold] "
        f"({workers} workers)…"
    )
    inv = inventory_groups(ids, tier=tier, max_workers=workers)  # type: ignore[arg-type]
    coverage = group_coverage(inv)

    summary = Table(title=f"Group coverage ({tier})")
    summary.add_column("group")
    summary.add_column("shots", justify="right")
    summary.add_column("fraction", justify="right")
    n = len(inv)
    for name, hits in coverage.items():
        summary.add_row(name, str(hits), f"{hits / n * 100:.1f}%")
    console.print(summary)

    cam_hits = sum(
        1
        for sid, groups in inv.items()
        if set(groups) & set(CAMERA_SOURCES + ("camera_visible", "camera_ir"))
    )
    console.print(
        f"camera-bearing shots in sample: [bold]{cam_hits} / {n}[/bold] "
        f"({cam_hits / max(n, 1) * 100:.1f}%)"
    )

    if output:
        payload = {
            "tier": tier,
            "shot_count": n,
            "coverage": coverage,
            "by_shot": {str(sid): list(g) for sid, g in inv.items()},
        }
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        console.print(f"[green]inventory written:[/green] {output}")


# --- manifest ---------------------------------------------------------


@data.command(name="manifest")
@_tier_option
@click.option(
    "--groups",
    default="",
    help=("Comma-separated group names to include per shot. Empty = whole shot."),
)
@click.option(
    "--inventory",
    default=None,
    type=click.Path(),
    help=(
        "Path to a `ambix data inventory --output` JSON. When provided, "
        "the manifest is restricted to shots that have at least one of the "
        "--groups present."
    ),
)
@click.option(
    "--emit-ids",
    is_flag=True,
    default=False,
    help="Print just the shot IDs (newline-separated) for piping to s5cmd.",
)
@click.option(
    "--emit-s5cmd-script",
    is_flag=True,
    default=False,
    help="Print the s5cmd `run` script (one `cp` per target).",
)
@click.option(
    "--output",
    default=None,
    type=click.Path(),
    help="Path to write the JSON manifest to (default: stdout).",
)
def manifest_cmd(
    tier: str,
    groups: str,
    inventory: str | None,
    emit_ids: bool,
    emit_s5cmd_script: bool,
    output: str | None,
) -> None:
    """Build a shot/group manifest for the bulk-download SLURM job."""
    from imas_ambix.data.manifest import (
        build_manifest,
        emit_shot_ids,
        emit_targets_as_s5cmd,
        load_index,
        shot_ids_from_index,
    )

    df = load_index()
    group_tuple = tuple(g.strip() for g in groups.split(",") if g.strip())

    if inventory:
        inv_data = json.loads(Path(inventory).read_text(encoding="utf-8"))
        by_shot = inv_data.get("by_shot", {})
        if group_tuple:
            need = set(group_tuple)
            shot_ids = sorted(
                int(sid) for sid, present in by_shot.items() if need & set(present)
            )
            desc = f"{len(shot_ids)} shots at {tier} carrying any of {group_tuple}"
        else:
            shot_ids = sorted(int(sid) for sid in by_shot)
            desc = f"{len(shot_ids)} shots at {tier} (from inventory file)"
    else:
        shot_ids = list(shot_ids_from_index(df))
        desc = f"all {len(shot_ids)} shots in the index at {tier}"

    manifest = build_manifest(
        tier=tier,  # type: ignore[arg-type]
        shot_ids=shot_ids,
        groups=group_tuple,
        total_in_index=len(df),
        filter_description=desc,
    )

    if emit_ids:
        click.echo(emit_shot_ids(manifest), nl=False)
        return
    if emit_s5cmd_script:
        click.echo(emit_targets_as_s5cmd(manifest), nl=False)
        return

    payload = manifest.to_json()
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(payload, encoding="utf-8")
        console.print(
            f"[green]manifest written:[/green] {output} "
            f"({len(manifest.shot_ids)} shots, "
            f"{len(manifest.groups) or 'all'} groups)"
        )
    else:
        click.echo(payload)


# --- targets ----------------------------------------------------------


@data.command(name="targets")
@click.argument("manifest_path", type=click.Path(exists=True))
def targets_cmd(manifest_path: str) -> None:
    """Emit s5cmd `cp` lines from a built manifest JSON.

    Piped into ``s5cmd run`` for the actual download. The manifest's
    ``tier`` field selects the bucket prefix; ``groups`` (if non-empty)
    selects which sub-prefixes to copy per shot.
    """
    from imas_ambix.data.manifest import (
        build_manifest,
        emit_targets_as_s5cmd,
    )

    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    manifest = build_manifest(
        tier=payload["tier"],
        shot_ids=[int(s) for s in payload["shot_ids"]],
        groups=tuple(payload.get("groups", ())),
        total_in_index=payload.get("total_in_index", 0),
        filter_description=payload.get("filter_description", ""),
    )
    click.echo(emit_targets_as_s5cmd(manifest), nl=False)


# --- download ---------------------------------------------------------


@data.command(name="download")
@_tier_option
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
    help="SLURM partition with outbound network + GPFS access.",
)
@click.option(
    "--time-limit",
    default="24:00:00",
    show_default=True,
    help="SLURM --time value.",
)
@click.option(
    "--manifest",
    default=None,
    type=click.Path(),
    help=(
        "JSON manifest produced by `ambix data manifest`. The script "
        "consumes this via s5cmd `run`."
    ),
)
@click.option(
    "--host",
    type=click.Choice(["slurm-sun", "login"]),
    default="slurm-sun",
    show_default=True,
    help=(
        "Where the download runs: SLURM job on `sun` (default), or "
        "directly on the login node (no sbatch, just an s5cmd run script)."
    ),
)
@click.option(
    "--numworkers",
    default=32,
    show_default=True,
)
def download_cmd(
    tier: str,
    plan_only: bool,
    partition: str,
    time_limit: str,
    manifest: str | None,
    host: str,
    numworkers: int,
) -> None:
    """Render the bulk-download script (SLURM or plain login-node version)."""
    if not plan_only:
        raise click.UsageError(
            "Submission via sbatch is intentionally not wired up. "
            "Pass --plan-only and submit the rendered script by hand. "
            "See plans/data-acquisition.md §4."
        )

    if not manifest:
        raise click.UsageError(
            "--manifest <path.json> is required so the script knows what "
            "to download. Build one with `ambix data manifest --output …` first."
        )

    dest = LEVEL1_DIR if tier == "level1" else LEVEL2_DIR

    if host == "slurm-sun":
        script = _render_slurm_script(
            partition=partition,
            time_limit=time_limit,
            dest=dest,
            manifest_path=manifest,
            numworkers=numworkers,
        )
    else:
        script = _render_login_script(
            dest=dest,
            manifest_path=manifest,
            numworkers=numworkers,
        )
    click.echo(script)


def _render_slurm_script(
    *,
    partition: str,
    time_limit: str,
    dest: Path,
    manifest_path: str,
    numworkers: int,
) -> str:
    return f"""#!/bin/bash
#SBATCH --job-name=mast-mirror
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
cd "$DEST"

ambix data targets {manifest_path} \\
| s5cmd --no-sign-request --endpoint-url {S3_ENDPOINT} \\
    --numworkers {numworkers} \\
    run
"""


def _render_login_script(
    *,
    dest: Path,
    manifest_path: str,
    numworkers: int,
) -> str:
    return f"""#!/bin/bash
# Login-node bulk download. Run inside a `screen` or `tmux` session so
# the transfer survives session disconnects.
#
# Throughput on the login node was measured at ~18 MB/s single-stream
# during the 2026-05-19 probe; multi-shot parallel pulls saturate higher.
# Expect 6-12 h wall time for the level-2 corpus, 12-24 h for level-1
# cameras.

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"

DEST={dest}
mkdir -p "$DEST"
cd "$DEST"

ambix data targets {manifest_path} \\
| s5cmd --no-sign-request --endpoint-url {S3_ENDPOINT} \\
    --numworkers {numworkers} \\
    run
"""


# --- status -----------------------------------------------------------


@data.command(name="status")
@_tier_option
def status_cmd(tier: str) -> None:
    """Show local mirror progress against the latest manifest."""
    if not MIRROR_ROOT.exists():
        console.print(f"[yellow]mirror root does not exist yet:[/yellow] {MIRROR_ROOT}")
        return

    dest = LEVEL1_DIR if tier == "level1" else LEVEL2_DIR
    n_local = sum(1 for p in dest.glob("*.zarr") if p.is_dir()) if dest.exists() else 0

    manifest_files = (
        sorted(MANIFEST_DIR.glob("*.json")) if MANIFEST_DIR.exists() else []
    )
    if manifest_files:
        latest = manifest_files[-1]
        payload = json.loads(latest.read_text(encoding="utf-8"))
        target = len(payload.get("shot_ids", []))
        console.print(
            f"{tier} mirror progress: [green]{n_local}[/green] / {target} "
            f"shots (manifest: {latest.name})"
        )
    else:
        console.print(
            f"{tier} mirror progress: [green]{n_local}[/green] / ? shots "
            f"(no manifest under {MANIFEST_DIR})"
        )
