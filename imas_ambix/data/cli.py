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


# --- du ---------------------------------------------------------------


@data.command(name="du")
@_tier_option
@click.option(
    "--sample-size",
    default=100,
    show_default=True,
    help="How many shots to size. Use --from-bucket-all to size the entire tier.",
)
@click.option(
    "--groups",
    default="",
    help="Comma-separated group names. Empty = size the whole shot.",
)
@click.option(
    "--seed",
    default=0,
    show_default=True,
)
@click.option(
    "--from-bucket-all",
    is_flag=True,
    default=False,
    help="Size every shot at this tier (uses `s5cmd ls` to enumerate first).",
)
@click.option(
    "--workers",
    default=16,
    show_default=True,
)
@click.option(
    "--output",
    default=None,
    type=click.Path(),
    help="Path to write the JSON result to.",
)
def du_cmd(
    tier: str,
    sample_size: int,
    groups: str,
    seed: int,
    from_bucket_all: bool,
    workers: int,
    output: str | None,
) -> None:
    """Sum the on-bucket size of a (possibly filtered) corpus via s5cmd du.

    This is the accurate way to size a tier — it reads S3 metadata
    instead of extrapolating from a small sample, and so handles
    heavy-tailed distributions correctly (the §10 / §11 probe sampling
    under-extrapolated by 11×).
    """
    from imas_ambix.data.manifest import (
        load_index,
        shot_ids_from_bucket,
        sum_sizes_from_bucket,
    )
    from imas_ambix.data.probe import sample_shots

    group_tuple = tuple(g.strip() for g in groups.split(",") if g.strip())

    if from_bucket_all:
        ids = list(shot_ids_from_bucket(tier))  # type: ignore[arg-type]
    else:
        df = load_index()
        ids = sample_shots(df, sample_size, seed=seed)

    console.print(
        f"sizing {len(ids)} shots at tier=[bold]{tier}[/bold] "
        f"(groups={list(group_tuple) or 'all'}, workers={workers})…"
    )
    sizes = sum_sizes_from_bucket(
        ids,
        tier=tier,
        groups=group_tuple,
        max_workers=workers,  # type: ignore[arg-type]
    )

    total_bytes = sum(b for b, _ in sizes.values())
    total_objects = sum(o for _, o in sizes.values())
    n_with_data = sum(1 for b, _ in sizes.values() if b > 0)
    n_total = len(sizes)
    mean_mb = (total_bytes / n_with_data / 1e6) if n_with_data else 0.0
    extrap_tb = (mean_mb * n_total) / 1e6  # for sample-mode reporting only

    summary = Table(title=f"s5cmd du ({tier})")
    summary.add_column("metric")
    summary.add_column("value", justify="right")
    summary.add_row("shots sized", str(n_total))
    summary.add_row("shots with data", str(n_with_data))
    summary.add_row("total objects", f"{total_objects:,}")
    summary.add_row("total bytes", f"{total_bytes:,}")
    summary.add_row("total GB", f"{total_bytes / 1e9:,.2f}")
    summary.add_row("total TB", f"{total_bytes / 1e12:,.3f}")
    summary.add_row("mean shot size (MB)", f"{mean_mb:,.1f}")
    if not from_bucket_all:
        summary.add_row("extrapolated to tier (TB)", f"{extrap_tb:,.3f}")
    console.print(summary)

    if output:
        payload = {
            "tier": tier,
            "groups": list(group_tuple),
            "from_bucket_all": from_bucket_all,
            "shots_sized": n_total,
            "shots_with_data": n_with_data,
            "total_bytes": total_bytes,
            "total_objects": total_objects,
            "mean_shot_mb": mean_mb,
            "by_shot": {
                str(sid): {"bytes": b, "objects": o} for sid, (b, o) in sizes.items()
            },
        }
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        console.print(f"[green]du report written:[/green] {output}")


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


# --- encode-shot -------------------------------------------------------


@data.command(name="encode-shot")
@click.option("--shot", type=int, required=True, help="Shot ID to encode.")
@click.option("--camera", default="rbb", show_default=True, help="Camera source name.")
@click.option(
    "--tokenizer",
    type=click.Choice(["placeholder", "open-magvit2"]),
    default="open-magvit2",
    show_default=True,
    help="Frame tokenizer to use.",
)
@click.option(
    "--max-frames",
    type=int,
    default=None,
    help="Truncate input to this many frames (default: all).",
)
@click.option(
    "--device",
    default="cpu",
    show_default=True,
    help="Torch device for Open-MAGVIT2 (ignored for placeholder).",
)
@click.option(
    "--vocab-version",
    default="v1",
    show_default=True,
    help="Token vocabulary version (output path prefix).",
)
def encode_shot_cmd(
    shot: int,
    camera: str,
    tokenizer: str,
    max_frames: int | None,
    device: str,
    vocab_version: str,
) -> None:
    """Encode a single shot's camera frames and save to the token store.

    Loads the level-1 Zarr via xarray, encodes with the chosen tokenizer,
    and writes to ``mast-tokens/{vocab-version}/frames/{shot}/{camera}.zarr``.
    """
    import xarray as xr

    from imas_ambix.data.paths import LEVEL1_DIR
    from imas_ambix.data.persist import save_frame_tokens

    shot_zarr = LEVEL1_DIR / f"{shot}.zarr"
    if not shot_zarr.exists():
        raise click.ClickException(
            f"Level-1 shot Zarr not found at {shot_zarr}. "
            "Run `ambix data download` to mirror the data first."
        )

    console.print(f"Loading camera [bold]{camera}[/bold] from {shot_zarr} …")
    try:
        ds = xr.open_zarr(str(shot_zarr / camera))
        # Camera data is typically under a variable matching the source name
        # or the first available variable.
        data_vars = list(ds.data_vars)
        if not data_vars:
            raise click.ClickException(
                f"No data variables found in group '{camera}' of shot {shot}."
            )
        frames_da = ds[data_vars[0]]
        frames = frames_da.values
    except Exception as exc:
        raise click.ClickException(f"Failed to load camera data: {exc}") from exc

    if max_frames is not None:
        frames = frames[:max_frames]

    console.print(f"Frames shape: {frames.shape}, dtype: {frames.dtype}. Encoding …")

    if tokenizer == "placeholder":
        from imas_ambix.tokenizer.frames import PlaceholderFrameTokenizer

        tok = PlaceholderFrameTokenizer()
    else:
        from imas_ambix.tokenizer.frames import OpenMagvit2Tokenizer

        tok = OpenMagvit2Tokenizer(device=device)

    encoded = tok.encode(frames)
    out_path = save_frame_tokens(
        shot_id=shot,
        camera=camera,
        encoded=encoded,
        vocab_version=vocab_version,
    )
    console.print(
        f"[green]Saved:[/green] {out_path}  (token shape: {encoded.token_ids.shape})"
    )


# --- tokens-status -----------------------------------------------------


@data.command(name="tokens-status")
@click.option(
    "--vocab-version",
    default="v1",
    show_default=True,
    help="Token vocabulary version directory to inspect.",
)
def tokens_status_cmd(vocab_version: str) -> None:
    """Show how many shots have been tokenised for each modality / group.

    Walks ``TOKEN_ROOT/{vocab_version}/`` and counts persisted ``.zarr``
    files per modality (frames / signals) and per sub-group (camera or
    diagnostic name).
    """
    from imas_ambix.data.paths import TOKEN_ROOT

    root = TOKEN_ROOT / vocab_version
    if not root.exists():
        console.print(
            f"[yellow]Token root does not exist:[/yellow] {root} "
            f"(no shots tokenised yet at {vocab_version})"
        )
        return

    table = Table(title=f"Persisted tokens — {vocab_version}")
    table.add_column("modality")
    table.add_column("group / camera")
    table.add_column("shots", justify="right")

    total_shots: set[int] = set()

    for modality in ("frames", "signals"):
        modality_dir = root / modality
        if not modality_dir.exists():
            table.add_row(modality, "(none)", "0")
            continue

        # Collect per-group counts: group → set of shot ids
        group_shots: dict[str, set[int]] = {}
        for shot_dir in modality_dir.iterdir():
            if not shot_dir.is_dir():
                continue
            try:
                sid = int(shot_dir.name)
            except ValueError:
                continue
            for zarr_path in shot_dir.iterdir():
                if zarr_path.suffix != ".zarr" and not (
                    zarr_path.is_dir() and zarr_path.suffix == ".zarr"
                ):
                    continue
                grp = zarr_path.stem
                group_shots.setdefault(grp, set()).add(sid)
                total_shots.add(sid)

        if not group_shots:
            table.add_row(modality, "(none)", "0")
            continue

        for grp, shot_set in sorted(group_shots.items()):
            table.add_row(modality, grp, str(len(shot_set)))

    console.print(table)
    console.print(
        f"Total unique shots with any token file: [bold]{len(total_shots)}[/bold]"
    )


# --- audit ------------------------------------------------------------


@data.command(name="audit")
@_tier_option
@click.option(
    "--sample-size",
    default=50,
    show_default=True,
    help=(
        "Number of randomly sampled shots to audit (ignored when --shot-ids is given)."
    ),
)
@click.option(
    "--from-bucket-all",
    is_flag=True,
    default=False,
    help=(
        "Enumerate ALL shot IDs from the S3 bucket and audit them "
        "(ignores --sample-size)."
    ),
)
@click.option(
    "--workers",
    default=8,
    show_default=True,
    help="Number of parallel audit workers.",
)
@click.option(
    "--output",
    default=None,
    type=click.Path(),
    help="Write per-shot reports + aggregate to this JSON path.",
)
@click.option(
    "--shot-ids",
    default=None,
    help="Comma-separated explicit shot IDs to audit (overrides --sample-size).",
)
def audit_cmd(
    tier: str,
    sample_size: int,
    from_bucket_all: bool,
    workers: int,
    output: str | None,
    shot_ids: str | None,
) -> None:
    """Audit per-shot Zarr quality and emit a corpus report.

    Opens each downloaded shot at --tier, runs data-quality checks
    (openability, NaN fraction, dynamic range, time axis monotonicity,
    dd_version), and renders a Rich summary table.  Writes a JSON report
    when --output is given.

    Examples
    --------
    Audit three specific shots::

        ambix data audit --tier level2 --shot-ids 11766,11767,11768

    Audit a 50-shot random sample and save results::

        ambix data audit --tier level2 --sample-size 50 --output /tmp/audit.json
    """
    from imas_ambix.data.paths import LEVEL1_DIR as _L1
    from imas_ambix.data.paths import LEVEL2_DIR as _L2
    from imas_ambix.quality.audit import aggregate_corpus, audit_corpus

    # --- resolve shot IDs ------------------------------------------------
    if shot_ids:
        ids = [int(s.strip()) for s in shot_ids.split(",") if s.strip()]
    elif from_bucket_all:
        from imas_ambix.data.manifest import shot_ids_from_bucket

        ids = list(shot_ids_from_bucket(tier))  # type: ignore[arg-type]
    else:
        # Sample from locally present shots (avoids network dependency).
        shots_dir = _L1 if tier == "level1" else _L2
        if shots_dir.exists():
            local_ids = sorted(
                int(p.name.removesuffix(".zarr"))
                for p in shots_dir.glob("*.zarr")
                if p.is_dir()
            )
        else:
            local_ids = []
        if local_ids:
            import random

            rng = random.Random(0)
            ids = rng.sample(local_ids, min(sample_size, len(local_ids)))
            ids.sort()
        else:
            console.print(
                f"[yellow]warning:[/yellow] no local shots found under {shots_dir}. "
                "Pass --shot-ids or --from-bucket-all to specify shots explicitly."
            )
            ids = []

    if not ids:
        console.print("[red]No shot IDs to audit.[/red]")
        return

    console.print(
        f"Auditing [bold]{len(ids)}[/bold] shots at tier=[bold]{tier}[/bold] "
        f"({workers} workers)…"
    )

    reports = audit_corpus(ids, tier=tier, max_workers=workers)  # type: ignore[arg-type]
    agg = aggregate_corpus(reports)

    # --- render summary table -------------------------------------------
    from rich.table import Table

    summary = Table(title=f"Data-quality audit — {tier} ({len(ids)} shots)")
    summary.add_column("metric")
    summary.add_column("value", justify="right")
    summary.add_row("shots audited", str(agg["n_total"]))
    summary.add_row(
        "passed (all-green)",
        f"{agg['n_passed']} ({agg['pass_rate'] * 100:.1f}%)",
    )
    summary.add_row("warned", str(agg["n_warned"]))
    summary.add_row("failed", str(agg["n_failed"]))
    summary.add_row("usable_for_training", str(agg["usable_for_training"]))
    console.print(summary)

    # Campaign distribution
    if agg["campaign_distribution"]:
        camp_table = Table(title="Campaign distribution")
        camp_table.add_column("campaign")
        camp_table.add_column("shots", justify="right")
        for camp, cnt in list(agg["campaign_distribution"].items())[:10]:
            camp_table.add_row(camp, str(cnt))
        console.print(camp_table)

    # Quality flag rates
    if agg["quality_flag_rates"]:
        flag_table = Table(title="Quality flag rates")
        flag_table.add_column("flag")
        flag_table.add_column("fraction", justify="right")
        for flag, rate in agg["quality_flag_rates"].items():
            flag_table.add_row(flag, f"{rate * 100:.1f}%")
        console.print(flag_table)

    # Top failure modes
    if agg["top_failure_modes"]:
        fail_table = Table(title="Top failure modes")
        fail_table.add_column("check")
        fail_table.add_column("count", justify="right")
        for entry in agg["top_failure_modes"]:
            fail_table.add_row(entry["check"], str(entry["count"]))
        console.print(fail_table)

    # Plasma current deciles
    if agg["plasma_current_deciles"]:
        deciles = agg["plasma_current_deciles"]
        console.print(
            "Plasma current deciles (kA): "
            + ", ".join(f"{v / 1000:.1f}" for v in deciles)
        )

    # --- JSON output -----------------------------------------------------
    if output:
        import json

        payload = {
            "tier": tier,
            "shot_ids": ids,
            "aggregate": agg,
            "per_shot": [r.to_dict() for r in reports],
        }
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        console.print(f"[green]audit report written:[/green] {output}")


# --- bulk-encode-frames -----------------------------------------------


@data.command(name="bulk-encode-frames")
@click.option(
    "--shot-ids",
    default=None,
    help="Comma-separated shot IDs to encode.",
)
@click.option(
    "--from-quality-index",
    default=None,
    type=click.Path(),
    help=(
        "Path to an `ambix data audit --output` JSON; uses shots with "
        "usable_for_training=True."
    ),
)
@click.option(
    "--from-bucket-all",
    is_flag=True,
    default=False,
    help="Encode all shots present in the L1 bucket directory.",
)
@click.option("--camera", default="rbb", show_default=True)
@click.option(
    "--tokenizer",
    type=click.Choice(["placeholder", "open-magvit2"]),
    default="open-magvit2",
    show_default=True,
)
@click.option("--max-frames-per-shot", type=int, default=None)
@click.option("--device", default="cpu", show_default=True)
@click.option("--vocab-version", default="v1", show_default=True)
@click.option("--workers", default=1, type=int, show_default=True)
@click.option(
    "--skip-existing/--no-skip-existing",
    default=True,
    show_default=True,
)
@click.option(
    "--output",
    default=None,
    type=click.Path(),
    help="Write JSON report to this path.",
)
def bulk_encode_frames_cmd(
    shot_ids: str | None,
    from_quality_index: str | None,
    from_bucket_all: bool,
    camera: str,
    tokenizer: str,
    max_frames_per_shot: int | None,
    device: str,
    vocab_version: str,
    workers: int,
    skip_existing: bool,
    output: str | None,
) -> None:
    """Bulk-encode camera frames for multiple shots.

    Shot IDs can be provided via ``--shot-ids``, inferred from an audit
    quality index (``--from-quality-index``), or enumerated from the local
    L1 directory (``--from-bucket-all``).

    Examples
    --------
    Encode two shots using the placeholder tokenizer::

        ambix data bulk-encode-frames --shot-ids 15085,15086 --tokenizer placeholder

    Re-encode all shots whose quality index marks as usable::

        ambix data bulk-encode-frames --from-quality-index /tmp/audit.json \\
            --tokenizer open-magvit2
    """
    from imas_ambix.data.encoding import bulk_encode_frames

    # --- resolve shot list -----------------------------------------------
    ids = _resolve_shot_ids_frames(shot_ids, from_quality_index, from_bucket_all)
    if not ids:
        console.print("[red]No shot IDs resolved — nothing to encode.[/red]")
        return

    # --- build tokenizer factory -----------------------------------------
    def _frame_factory():
        if tokenizer == "placeholder":
            from imas_ambix.tokenizer.frames import (
                PlaceholderFrameTokenizer,  # noqa: PLC0415
            )

            return PlaceholderFrameTokenizer()
        from imas_ambix.tokenizer.frames import OpenMagvit2Tokenizer  # noqa: PLC0415

        return OpenMagvit2Tokenizer(device=device)

    console.print(
        f"Encoding [bold]{len(ids)}[/bold] shots  "
        f"camera=[bold]{camera}[/bold]  "
        f"tokenizer=[bold]{tokenizer}[/bold]  "
        f"workers={workers}"
    )

    t_start = time.monotonic()
    reports = bulk_encode_frames(
        shot_ids=ids,
        camera=camera,
        tokenizer_factory=_frame_factory,
        max_workers=workers,
        skip_existing=skip_existing,
        max_frames_per_shot=max_frames_per_shot,
        vocab_version=vocab_version,
    )
    total_elapsed = time.monotonic() - t_start

    _render_encode_summary(reports, total_elapsed, console)

    if output:
        _write_encode_report(reports, total_elapsed, Path(output))
        console.print(f"[green]report written:[/green] {output}")


# --- bulk-encode-signals -----------------------------------------------


@data.command(name="bulk-encode-signals")
@click.option(
    "--shot-ids",
    default=None,
    help="Comma-separated shot IDs to encode.",
)
@click.option(
    "--from-quality-index",
    default=None,
    type=click.Path(),
    help=(
        "Path to an `ambix data audit --output` JSON; uses shots with "
        "usable_for_training=True."
    ),
)
@click.option(
    "--from-bucket-all",
    is_flag=True,
    default=False,
    help="Encode all shots present in the L2 shots directory.",
)
@click.option("--group", default="magnetics", show_default=True)
@click.option(
    "--tokenizer",
    type=click.Choice(["uniform", "chronos", "patchtst"]),
    default="uniform",
    show_default=True,
)
@click.option("--vocab-version", default="v1", show_default=True)
@click.option("--workers", default=4, type=int, show_default=True)
@click.option(
    "--skip-existing/--no-skip-existing",
    default=True,
    show_default=True,
)
@click.option(
    "--output",
    default=None,
    type=click.Path(),
    help="Write JSON report to this path.",
)
def bulk_encode_signals_cmd(
    shot_ids: str | None,
    from_quality_index: str | None,
    from_bucket_all: bool,
    group: str,
    tokenizer: str,
    vocab_version: str,
    workers: int,
    skip_existing: bool,
    output: str | None,
) -> None:
    """Bulk-encode signal groups for multiple shots.

    Examples
    --------
    Encode magnetics for two shots::

        ambix data bulk-encode-signals --shot-ids 15085,15086 --group magnetics

    Use all usable shots from an audit index::

        ambix data bulk-encode-signals --from-quality-index /tmp/audit.json \\
            --tokenizer uniform
    """
    from imas_ambix.data.encoding import bulk_encode_signals

    # --- resolve shot list -----------------------------------------------
    ids = _resolve_shot_ids_signals(shot_ids, from_quality_index, from_bucket_all)
    if not ids:
        console.print("[red]No shot IDs resolved — nothing to encode.[/red]")
        return

    # --- build tokenizer factory -----------------------------------------
    def _signal_factory():
        if tokenizer == "chronos":
            from imas_ambix.tokenizer.signals import (
                ChronosSignalTokenizer,  # noqa: PLC0415
            )

            return ChronosSignalTokenizer()
        if tokenizer == "patchtst":
            from imas_ambix.tokenizer.signals import PatchTSTTokenizer  # noqa: PLC0415

            return PatchTSTTokenizer()
        from imas_ambix.tokenizer.signals import UniformQuantizer  # noqa: PLC0415

        return UniformQuantizer()

    console.print(
        f"Encoding [bold]{len(ids)}[/bold] shots  "
        f"group=[bold]{group}[/bold]  "
        f"tokenizer=[bold]{tokenizer}[/bold]  "
        f"workers={workers}"
    )

    t_start = time.monotonic()
    reports = bulk_encode_signals(
        shot_ids=ids,
        group=group,
        tokenizer_factory=_signal_factory,
        max_workers=workers,
        skip_existing=skip_existing,
        vocab_version=vocab_version,
    )
    total_elapsed = time.monotonic() - t_start

    _render_encode_summary(reports, total_elapsed, console)

    if output:
        _write_encode_report(reports, total_elapsed, Path(output))
        console.print(f"[green]report written:[/green] {output}")


# ---------------------------------------------------------------------------
# Shared render / resolve helpers for bulk-encode commands
# ---------------------------------------------------------------------------


import time as time  # noqa: E402,PLC0415  # needed by bulk-encode commands above


def _resolve_shot_ids_frames(
    shot_ids: str | None,
    from_quality_index: str | None,
    from_bucket_all: bool,
) -> list[int]:
    """Resolve a list of frame shot IDs from the various source options."""
    if shot_ids:
        return [int(s.strip()) for s in shot_ids.split(",") if s.strip()]
    if from_quality_index:
        return _usable_shots_from_quality_index(from_quality_index)
    if from_bucket_all:
        from imas_ambix.data.paths import LEVEL1_DIR  # noqa: PLC0415

        return sorted(int(p.stem) for p in LEVEL1_DIR.glob("*.zarr") if p.is_dir())
    return []


def _resolve_shot_ids_signals(
    shot_ids: str | None,
    from_quality_index: str | None,
    from_bucket_all: bool,
) -> list[int]:
    """Resolve a list of signal shot IDs from the various source options."""
    if shot_ids:
        return [int(s.strip()) for s in shot_ids.split(",") if s.strip()]
    if from_quality_index:
        return _usable_shots_from_quality_index(from_quality_index)
    if from_bucket_all:
        from imas_ambix.data.paths import LEVEL2_DIR  # noqa: PLC0415

        return sorted(int(p.stem) for p in LEVEL2_DIR.glob("*.zarr") if p.is_dir())
    return []


def _usable_shots_from_quality_index(index_path: str) -> list[int]:
    """Extract shot IDs with ``usable_for_training=True`` from an audit JSON."""
    payload = json.loads(Path(index_path).read_text(encoding="utf-8"))
    usable: list[int] = []
    for report in payload.get("per_shot", []):
        flags = report.get("quality_flags", {})
        if flags.get("usable_for_training", False):
            usable.append(int(report["shot_id"]))
    return sorted(usable)


def _render_encode_summary(
    reports: list,
    total_elapsed: float,
    con: Console,
) -> None:
    """Print a Rich progress table for a completed bulk-encode run."""
    from rich.table import Table  # noqa: PLC0415

    n_ok = sum(1 for r in reports if r.error is None and r.n_tokens > 0)
    n_skip = sum(1 for r in reports if r.error is None and r.n_tokens == 0)
    n_err = sum(1 for r in reports if r.error is not None)
    total_tokens = sum(r.n_tokens for r in reports)

    table = Table(title="Bulk encode summary")
    table.add_column("metric")
    table.add_column("value", justify="right")
    table.add_row("shots encoded", str(n_ok))
    table.add_row("shots skipped (existing)", str(n_skip))
    table.add_row("shots errored", str(n_err))
    table.add_row("total tokens", f"{total_tokens:,}")
    table.add_row("total elapsed (s)", f"{total_elapsed:.1f}")
    con.print(table)

    if n_err:
        err_table = Table(title="Errors")
        err_table.add_column("shot_id")
        err_table.add_column("error")
        for r in reports:
            if r.error is not None:
                err_table.add_row(str(r.shot_id), r.error[:120])
        con.print(err_table)


def _write_encode_report(
    reports: list,
    total_elapsed: float,
    path: Path,
) -> None:
    """Write a JSON encode report to *path*."""
    payload = {
        "total_elapsed_s": total_elapsed,
        "n_ok": sum(1 for r in reports if r.error is None and r.n_tokens > 0),
        "n_skipped": sum(1 for r in reports if r.error is None and r.n_tokens == 0),
        "n_errored": sum(1 for r in reports if r.error is not None),
        "total_tokens": sum(r.n_tokens for r in reports),
        "per_shot": [
            {
                "shot_id": r.shot_id,
                "modality": r.modality,
                "group_or_camera": r.group_or_camera,
                "tokenizer_name": r.tokenizer_name,
                "n_tokens": r.n_tokens,
                "elapsed_s": r.elapsed_s,
                "output_path": str(r.output_path),
                "error": r.error,
            }
            for r in reports
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
