"""CLI commands for the multi-modal tokenizer pipeline.

Subcommands:

- ``ambix tokenize inspect``     open a shot, print tokenizer-relevant shapes
- ``ambix tokenize frames``      encode a shot's camera frames + round-trip
- ``ambix tokenize signals``     encode a shot's signal groups + round-trip
- ``ambix tokenize registry``    print the global token vocabulary layout

The signal and frame paths use the placeholder tokenizers by default
so the plumbing can be exercised end-to-end before Open-MAGVIT2 and
Chronos checkpoints are staged.
"""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from imas_ambix.data.paths import LEVEL1_DIR, LEVEL2_DIR
from imas_ambix.tokenizer.frames import PlaceholderFrameTokenizer
from imas_ambix.tokenizer.registry import registry
from imas_ambix.tokenizer.signals import UniformQuantizer

console = Console()


@click.group(name="tokenize")
def tokenize() -> None:
    """Multi-modal tokenizers for FAIR-MAST shots.

    See ``plans/tokenizers.md`` for the design rationale.
    """


@tokenize.command(name="registry")
def registry_cmd() -> None:
    """Print the global token vocabulary layout."""
    # Trigger placeholder allocations so the registry has something to show
    PlaceholderFrameTokenizer()
    UniformQuantizer()

    table = Table(title=f"Global token vocabulary ({registry.version})")
    table.add_column("name")
    table.add_column("start", justify="right")
    table.add_column("end", justify="right")
    table.add_column("size", justify="right")
    for block in registry._blocks.values():  # noqa: SLF001 — internal access for display
        table.add_row(block.name, str(block.start), str(block.end), str(block.size))
    console.print(table)
    console.print(f"total_vocab_size: [bold]{registry.total_vocab_size()}[/bold]")


@tokenize.command(name="inspect")
@click.option("--shot", type=int, required=True)
@click.option(
    "--tier",
    type=click.Choice(["level1", "level2"]),
    default="level2",
    show_default=True,
)
@click.option(
    "--group",
    default=None,
    help="Specific group to inspect (e.g. 'rbb', 'magnetics'). Default: all.",
)
def inspect_cmd(shot: int, tier: str, group: str | None) -> None:
    """Open a shot via xarray and report shapes for the tokenizer."""
    import xarray as xr

    root = LEVEL1_DIR if tier == "level1" else LEVEL2_DIR
    shot_path = root / f"{shot}.zarr"
    if not shot_path.is_dir():
        raise click.UsageError(f"shot {shot} not on disk at {shot_path}")

    if group is None:
        groups = sorted(p.name for p in shot_path.iterdir() if p.is_dir())
        console.print(f"shot {shot} groups: {groups}")
        return

    ds = xr.open_zarr(str(shot_path), group=group, consolidated=False)
    table = Table(title=f"{tier}/{shot}/{group}")
    table.add_column("variable")
    table.add_column("shape")
    table.add_column("dtype")
    for name in ds.data_vars:
        var = ds[name]
        table.add_row(name, str(tuple(var.shape)), str(var.dtype))
    console.print(table)
    console.print(f"dims: {dict(ds.sizes)}")


@tokenize.command(name="frames")
@click.option("--shot", type=int, required=True)
@click.option(
    "--camera",
    default="rbb",
    show_default=True,
    help="Camera source name at level-1 (rba/rbb/rbc/rco/...).",
)
@click.option(
    "--tokenizer",
    type=click.Choice(["placeholder", "open-magvit2"]),
    default="placeholder",
    show_default=True,
    help="Which frame tokenizer to use.",
)
@click.option(
    "--temporal-compression",
    default=4,
    show_default=True,
    help="Placeholder tokenizer only.",
)
@click.option(
    "--spatial-compression",
    default=8,
    show_default=True,
    help="Placeholder tokenizer only.",
)
@click.option(
    "--max-frames",
    default=None,
    type=int,
    help=(
        "Only encode the first N frames (Open-MAGVIT2 on CPU is ~30 s/frame). "
        "Useful for smoke-testing without burning hours."
    ),
)
@click.option(
    "--device",
    default="cpu",
    show_default=True,
    help="open-magvit2: 'cpu' or 'cuda' (GPU node only).",
)
@click.option(
    "--output",
    default=None,
    type=click.Path(),
    help="Write the encoded tokens (numpy .npy) to this path.",
)
def frames_cmd(
    shot: int,
    camera: str,
    tokenizer: str,
    temporal_compression: int,
    spatial_compression: int,
    max_frames: int | None,
    device: str,
    output: str | None,
) -> None:
    """Encode + round-trip a shot's camera frames."""
    import numpy as np
    import xarray as xr

    shot_path = LEVEL1_DIR / f"{shot}.zarr"
    if not (shot_path / camera).is_dir():
        raise click.UsageError(
            f"shot {shot} has no level-1 {camera!r} group at {shot_path}"
        )
    ds = xr.open_zarr(str(shot_path), group=camera, consolidated=False)
    frames = np.asarray(ds["data"].values)
    if max_frames is not None:
        frames = frames[:max_frames]
    console.print(
        f"loaded {camera} for shot {shot}: shape={frames.shape}, dtype={frames.dtype}"
    )

    if tokenizer == "placeholder":
        tok = PlaceholderFrameTokenizer(
            temporal_compression=temporal_compression,
            spatial_compression=spatial_compression,
        )
    else:
        from imas_ambix.tokenizer.frames import OpenMagvit2Tokenizer

        tok = OpenMagvit2Tokenizer(device=device)

    enc = tok.encode(frames)
    decoded = tok.decode(enc)

    n_compare = min(decoded.shape[0], frames.shape[0])
    # Replicate single-channel input to 3-channel for comparison if needed.
    src = frames[:n_compare]
    if src.ndim == 3 and decoded.ndim == 4:
        src = np.repeat(src[..., None], 3, axis=-1)
    mae = float(
        abs(src.astype(np.float32) - decoded[:n_compare].astype(np.float32)).mean()
    )
    console.print(
        f"encoded shape: {enc.token_ids.shape}  "
        f"vocab range used: [{enc.token_ids.min()}, {enc.token_ids.max()}]"
    )
    console.print(f"decode shape:  {decoded.shape}  input vs decoded MAE: {mae:.2f}")

    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        np.save(output, enc.token_ids)
        console.print(f"[green]tokens saved:[/green] {output}")


@tokenize.command(name="signals")
@click.option("--shot", type=int, required=True)
@click.option(
    "--group",
    default="magnetics",
    show_default=True,
    help="Level-2 group to encode (e.g. magnetics, summary, pf_active).",
)
@click.option(
    "--n-bins",
    default=256,
    show_default=True,
)
@click.option(
    "--output",
    default=None,
    type=click.Path(),
)
def signals_cmd(shot: int, group: str, n_bins: int, output: str | None) -> None:
    """Encode + round-trip a level-2 signal group (placeholder tokenizer)."""
    import numpy as np
    import xarray as xr

    shot_path = LEVEL2_DIR / f"{shot}.zarr"
    if not (shot_path / group).is_dir():
        raise click.UsageError(
            f"shot {shot} has no level-2 {group!r} group at {shot_path}"
        )
    ds = xr.open_zarr(str(shot_path), group=group, consolidated=False)

    tok = UniformQuantizer(n_bins=n_bins)
    tok.fit([ds])
    enc = tok.encode(ds)

    table = Table(title=f"{group} signal tokenization (shot {shot})")
    table.add_column("metric")
    table.add_column("value")
    table.add_row("input vars", str(len(ds.data_vars)))
    table.add_row("tokenized channels", str(len(enc.channel_names)))
    table.add_row("token shape", str(tuple(enc.token_ids.shape)))
    table.add_row(
        "global id range",
        f"[{int(enc.token_ids.min())}, {int(enc.token_ids.max())}]",
    )
    table.add_row("vocab_size (per ch.)", str(tok.vocab_size))
    console.print(table)

    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        np.save(output, enc.token_ids)
        console.print(f"[green]tokens saved:[/green] {output}")
