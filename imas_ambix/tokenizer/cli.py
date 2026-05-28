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


# ---------------------------------------------------------------------------
# bench subcommand (appended — do not modify commands above this line)
# ---------------------------------------------------------------------------

_FRAME_TOKENIZER_CHOICES = ["placeholder", "open-magvit2"]
_SIGNAL_TOKENIZER_CHOICES = ["uniform", "chronos", "patchtst"]


def _make_frame_factory(name: str, device: str):
    """Return a zero-arg factory for the named frame tokenizer."""
    if name == "placeholder":
        return PlaceholderFrameTokenizer
    if name == "open-magvit2":
        from imas_ambix.tokenizer.frames import OpenMagvit2Tokenizer

        return lambda: OpenMagvit2Tokenizer(device=device)
    raise click.UsageError(f"Unknown frame tokenizer: {name!r}")


def _make_signal_factory(name: str):
    """Return a zero-arg factory for the named signal tokenizer."""
    if name == "uniform":
        return UniformQuantizer
    if name == "chronos":
        from imas_ambix.tokenizer.signals import ChronosSignalTokenizer

        return ChronosSignalTokenizer
    if name == "patchtst":
        from imas_ambix.tokenizer.signals import PatchTSTTokenizer

        return PatchTSTTokenizer
    raise click.UsageError(f"Unknown signal tokenizer: {name!r}")


@tokenize.command(name="bench")
@click.option(
    "--config",
    "config_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help=(
        "Path to a YAML bench config file. When provided, all tokenizer, "
        "shot, metrics, and kwargs are read from the file. Ad-hoc flags "
        "(--tokenizer, --shot-ids, etc.) are overridden by --config."
    ),
)
@click.option(
    "--tokenizer",
    "tokenizers",
    required=False,
    multiple=True,
    help=(
        "Repeat for each tokenizer to benchmark. "
        "Frame choices: placeholder, open-magvit2. "
        "Signal choices: uniform, chronos, patchtst. "
        "Required when --config is not provided."
    ),
)
@click.option(
    "--kind",
    type=click.Choice(["frame", "signal"]),
    default="frame",
    show_default=True,
    help="Whether to benchmark frame or signal tokenizers.",
)
@click.option(
    "--shot-ids",
    required=False,
    default=None,
    help=(
        "Comma-separated shot IDs, e.g. '15085,15086'. "
        "Required when --config is not provided."
    ),
)
@click.option(
    "--max-items-per-shot",
    default=None,
    type=int,
    help="Cap on frames/timesteps per shot (useful for CPU benchmarks).",
)
@click.option(
    "--camera",
    default="rbb",
    show_default=True,
    help="Camera source name (frame mode only).",
)
@click.option(
    "--group",
    default="magnetics",
    show_default=True,
    help="Level-2 group name (signal mode only).",
)
@click.option(
    "--tier",
    default=None,
    help="Data tier override. Defaults to level1 for frame, level2 for signal.",
)
@click.option(
    "--device",
    default="cpu",
    show_default=True,
    help="Device for tokenizers that support GPU (e.g. open-magvit2).",
)
@click.option(
    "--output",
    default=None,
    type=click.Path(),
    help="Write JSON results to this path.",
)
@click.option(
    "--in-process",
    "in_process",
    is_flag=True,
    default=False,
    help=(
        "Frame mode only. Launch a single worker subprocess that holds the "
        "VQModel in memory for the entire bench, eliminating the ~10 s/shot "
        "venv-init overhead. Requires the Open-MAGVIT2 venv at "
        "/work/projects/imas_gpu/mast-tokens/v1/open-magvit2/.venv. "
        "Expected speedup: ~10× (65 min → ~5-8 min for 100 shots)."
    ),
)
def bench_cmd(
    config_path: str | None,
    tokenizers: tuple[str, ...],
    kind: str,
    shot_ids: str | None,
    max_items_per_shot: int | None,
    camera: str,
    group: str,
    tier: str | None,
    device: str,
    output: str | None,
    in_process: bool,
) -> None:
    """Closed-loop tokenizer benchmark: encode → decode → measure → compare.

    Runs each requested tokenizer in turn, prints a Rich comparison table,
    and optionally writes JSON results.

    When --config is provided, the YAML file drives the benchmark (tokenizer
    factory, kwargs, shot_ids, metrics, max_items_per_shot, camera, device).
    Ad-hoc flags are ignored in that mode.

    Examples
    --------
    ::

        # YAML-driven (recommended — carries metrics + tokenizer kwargs):
        ambix tokenize bench --config bench-v0.yaml

        # Ad-hoc (legacy):
        ambix tokenize bench --tokenizer placeholder --kind frame \\
            --shot-ids 15085 --max-items-per-shot 4

        ambix tokenize bench --tokenizer uniform --kind signal \\
            --shot-ids 15085,15086
    """
    from imas_ambix.bench.report import render_comparison_table, save_results_json
    from imas_ambix.bench.tokenizer import (
        BenchConfig,
        benchmark_frame_tokenizer,
        benchmark_frame_tokenizer_in_process,
        benchmark_signal_tokenizer,
    )

    results = []

    if config_path is not None:
        # --config mode: YAML drives everything.
        if tokenizers or shot_ids is not None:
            console.print(
                "[yellow]WARNING:[/yellow] --config provided alongside ad-hoc flags "
                "(--tokenizer / --shot-ids). --config wins; ad-hoc flags are ignored."
            )

        from imas_ambix.bench.loader import load_bench_config

        cfg, run_kwargs = load_bench_config(config_path)

        console.print(f"[bold]Running benchmark:[/bold] {cfg.name} ...")

        if cfg.tokenizer_kind == "frame" and in_process:
            console.print("[dim]--in-process: dispatching to stream_worker (hold VQModel in memory)[/dim]")
            result = benchmark_frame_tokenizer_in_process(cfg, **run_kwargs)
        elif cfg.tokenizer_kind == "frame":
            result = benchmark_frame_tokenizer(cfg, **run_kwargs)
        else:
            result = benchmark_signal_tokenizer(cfg, **run_kwargs)
        results.append(result)

        for ps in result.per_shot:
            if ps.error:
                console.print(f"  [red]shot {ps.shot_id} FAILED:[/red]\n{ps.error}")

    else:
        # Ad-hoc mode: build BenchConfig from CLI flags.
        if not tokenizers:
            raise click.UsageError(
                "Either --config FILE or at least one --tokenizer NAME must be provided."
            )
        if shot_ids is None:
            raise click.UsageError(
                "Either --config FILE or --shot-ids must be provided."
            )

        parsed_shot_ids = [int(s.strip()) for s in shot_ids.split(",") if s.strip()]
        if not parsed_shot_ids:
            raise click.UsageError("--shot-ids must contain at least one integer shot id.")

        # Resolve default tier
        effective_tier = tier or ("level1" if kind == "frame" else "level2")

        # Default metrics per mode
        default_metrics: tuple[str, ...] = (
            ("psnr",) if kind == "frame" else ("mae", "nrmse", "correlation")
        )

        for tok_name in tokenizers:
            if kind == "frame":
                factory = _make_frame_factory(tok_name, device)
            else:
                factory = _make_signal_factory(tok_name)

            cfg = BenchConfig(
                name=f"{tok_name}-{device}" if kind == "frame" else tok_name,
                tokenizer_kind=kind,
                tokenizer_factory=factory,
                max_items_per_shot=max_items_per_shot,
                metrics=default_metrics,
                device=device,
            )

            console.print(f"[bold]Running benchmark:[/bold] {cfg.name} ...")

            if kind == "frame" and in_process:
                console.print("[dim]--in-process: dispatching to stream_worker (hold VQModel in memory)[/dim]")
                result = benchmark_frame_tokenizer_in_process(
                    cfg,
                    parsed_shot_ids,
                    camera=camera,
                    tier=effective_tier,  # type: ignore[arg-type]
                )
            elif kind == "frame":
                result = benchmark_frame_tokenizer(
                    cfg,
                    parsed_shot_ids,
                    camera=camera,
                    tier=effective_tier,  # type: ignore[arg-type]
                )
            else:
                result = benchmark_signal_tokenizer(
                    cfg,
                    parsed_shot_ids,
                    group=group,
                    tier=effective_tier,  # type: ignore[arg-type]
                )
            results.append(result)

            # Report per-shot errors inline
            for ps in result.per_shot:
                if ps.error:
                    console.print(f"  [red]shot {ps.shot_id} FAILED:[/red]\n{ps.error}")

    table = render_comparison_table(results)
    console.print(table)

    if output:
        out_path = Path(output)
        save_results_json(results, out_path)
        console.print(f"[green]results saved:[/green] {out_path}")


# ---------------------------------------------------------------------------
# finetune-decoder subcommand
# ---------------------------------------------------------------------------


@tokenize.command(name="finetune-decoder")
@click.option(
    "--train-shots",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Text file with newline-separated training shot IDs.",
)
@click.option(
    "--val-shots",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Text file with newline-separated validation shot IDs.",
)
@click.option(
    "--max-steps",
    default=10_000,
    show_default=True,
    type=int,
    help="Maximum number of training steps.",
)
@click.option(
    "--batch-size",
    default=16,
    show_default=True,
    type=int,
    help="Frames per step per GPU.",
)
@click.option(
    "--learning-rate",
    default=1e-4,
    show_default=True,
    type=float,
    help="AdamW initial learning rate.",
)
@click.option(
    "--output-path",
    default=None,
    type=click.Path(),
    help=(
        "Destination for the fine-tuned decoder weights (.safetensors). "
        "Defaults to {magvit2_root}/weights/plasma-decoder-v1.safetensors."
    ),
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Build and print the config without running training.",
)
def finetune_decoder_cmd(
    train_shots: str,
    val_shots: str,
    max_steps: int,
    batch_size: int,
    learning_rate: float,
    output_path: str | None,
    dry_run: bool,
) -> None:
    """Fine-tune the Open-MAGVIT2 decoder on plasma-domain camera frames.

    Freezes the encoder + codebook; trains only the decoder using pixel-level
    L1 + perceptual loss. Requires a 4×H200 exclusive GPU reservation.

    See ``plans/tokenizers.md`` §12.1 for the design rationale and trigger
    conditions.

    Examples
    --------
    ::

        # Dry-run: print config only
        ambix tokenize finetune-decoder \\
            --train-shots train_ids.txt --val-shots val_ids.txt --dry-run

        # Real run (GPU node required):
        ambix tokenize finetune-decoder \\
            --train-shots train_ids.txt --val-shots val_ids.txt \\
            --max-steps 10000 --batch-size 16
    """
    from imas_ambix.tokenizer.finetune_decoder import (
        DecoderFinetuneConfig,
        finetune_decoder,
    )

    # Parse shot ID files
    def _read_shot_ids(path: str) -> list[int]:
        ids = []
        for line in Path(path).read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                ids.append(int(line))
        return ids

    train_ids = _read_shot_ids(train_shots)
    val_ids = _read_shot_ids(val_shots)

    config = DecoderFinetuneConfig(
        train_shot_ids=train_ids,
        val_shot_ids=val_ids,
        max_steps=max_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )
    if output_path is not None:
        config.output_path = Path(output_path)

    if dry_run:
        console.print("[bold]DecoderFinetuneConfig (dry-run — no training)[/bold]")
        import dataclasses

        for f in dataclasses.fields(config):
            console.print(f"  [cyan]{f.name}[/cyan]: {getattr(config, f.name)}")
        return

    out = finetune_decoder(config)
    console.print(f"[green]Fine-tune complete. Weights saved to:[/green] {out}")
