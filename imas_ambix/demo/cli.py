"""Click CLI group for the WHAM demo pipeline (plans/demo.md §5).

Exposed as ``ambix demo`` via the lazy-group mechanism in
``imas_ambix/cli.py``.

Usage::

    ambix demo wham-mast \\
        --shot 30420 \\
        --checkpoint /path/to/checkpoint  \\
        --prefix-ms 150 \\
        --rollout-ms 1000 \\
        --output /tmp/demo-30420/

Pass ``--checkpoint mock`` to run the pipeline with a mock WhamModel
(no real trained weights required — useful for smoke-testing the plumbing).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import click

log = logging.getLogger(__name__)


@click.group(name="demo")
def demo() -> None:
    """Forward-prediction demo for held-out MAST shots (plans/demo.md)."""


@demo.command(name="wham-mast")
@click.option("--shot", required=True, type=int, help="MAST shot id, e.g. 30420.")
@click.option(
    "--checkpoint",
    required=True,
    type=str,
    help='Path to WhamModel checkpoint directory, or "mock" for a synthetic run.',
)
@click.option(
    "--prefix-ms",
    default=150,
    show_default=True,
    type=int,
    help="Initial context window in milliseconds.",
)
@click.option(
    "--rollout-ms",
    default=1000,
    show_default=True,
    type=int,
    help="Rollout horizon in milliseconds.",
)
@click.option(
    "--output",
    required=True,
    type=click.Path(),
    help="Output directory for all demo artefacts.",
)
@click.option(
    "--tokenizer-version",
    default="v1",
    show_default=True,
    type=str,
    help="Token vocabulary version (used to locate persisted streams).",
)
@click.option(
    "--top-k",
    default=64,
    show_default=True,
    type=int,
    help="Top-k sampling for frame tokens.",
)
@click.option(
    "--temperature",
    default=0.8,
    show_default=True,
    type=float,
    help="Sampling temperature for frame tokens.",
)
@click.option(
    "--no-video",
    is_flag=True,
    default=False,
    help="Skip MP4 video generation even if imageio-ffmpeg is available.",
)
def wham_mast_cmd(
    shot: int,
    checkpoint: str,
    prefix_ms: int,
    rollout_ms: int,
    output: str,
    tokenizer_version: str,
    top_k: int,
    temperature: float,
    no_video: bool,
) -> None:
    """Run the WHAM forward-prediction demo for a held-out MAST shot.

    Generates predictions for SHOT_ID using the model at CHECKPOINT, writes
    all artefacts to OUTPUT, and prints a metrics summary table.

    Per plans/demo.md §5 the artefacts are:

    \b
      ground-truth.zarr           decoded GT frames
      prediction.zarr             decoded predicted frames
      tokens-ground-truth.zarr    raw GT token sequence
      tokens-prediction.zarr      raw predicted token sequence
      metrics.json                psnr / lpips / rfid / centroid_mse / chord_nrmse
      demo-<shot>.png             4-panel GT vs prediction figure
      demo-<shot>.mp4             side-by-side video (optional)
    """
    from imas_ambix.demo.runner import run_demo
    from imas_ambix.eval.rollout import RolloutConfig

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    output_dir = Path(output)

    click.echo(
        f"Running WHAM demo: shot={shot}  checkpoint={checkpoint!r}  "
        f"prefix_ms={prefix_ms}  rollout_ms={rollout_ms}"
    )
    click.echo(f"Output directory: {output_dir}")

    cfg = RolloutConfig(
        top_k=top_k,
        temperature=temperature,
    )

    artefacts = run_demo(
        shot_id=shot,
        checkpoint_path=checkpoint,
        prefix_ms=prefix_ms,
        rollout_ms=rollout_ms,
        output_dir=output_dir,
        tokenizer_version=tokenizer_version,
        rollout_config=cfg,
        no_video=no_video,
    )

    # --- Print metrics table ------------------------------------------------
    if artefacts.metrics_json.exists():
        metrics: dict[str, object] = json.loads(artefacts.metrics_json.read_text())
        click.echo("\nEvaluation metrics")
        click.echo("=" * 40)
        col_w = 26
        for key, val in metrics.items():
            val_str = f"{val:.4f}" if isinstance(val, float) else str(val)
            click.echo(f"  {key:<{col_w}} {val_str}")
        click.echo("=" * 40)

    # --- Print artefact paths -----------------------------------------------
    click.echo("\nArtefacts written:")
    for label, path in [
        ("ground-truth.zarr    ", artefacts.ground_truth_zarr),
        ("prediction.zarr      ", artefacts.prediction_zarr),
        ("tokens-gt.zarr       ", artefacts.tokens_ground_truth_zarr),
        ("tokens-pred.zarr     ", artefacts.tokens_prediction_zarr),
        ("metrics.json         ", artefacts.metrics_json),
        ("figure.png           ", artefacts.figure_png),
        ("video.mp4            ", artefacts.video_mp4 or "(skipped)"),
    ]:
        click.echo(f"  {label}  {path}")

    click.echo("\nDone.")
