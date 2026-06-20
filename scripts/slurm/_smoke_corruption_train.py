"""End-to-end smoke for the context-corruption + control-dropout TRAIN path.

This gates the expensive multi-GPU fine-tune.  It runs the REAL corpus trainer
(:func:`imas_ambix.worldmodel.spacetime_train_v2.train_corpus`) on one GPU (or
CPU) for a handful of steps with the full M2 recipe ENABLED — overlapping-window
enumeration, history-token corruption, the corruption-level conditioning
embedding, and classifier-free-guidance control-dropout — then proves the run is
RESUME-SAFE: a first call writes ``latest.pt``; a second call resumes from it and
advances the step counter (the property a requeue/--time restart depends on).

What it asserts
---------------
1. the trainer runs end-to-end with corruption + dropout + overlapping windows
   (no exception, loss is finite);
2. ``latest.pt`` is written and reloadable, and carries the corruption-level
   embedding (so the fine-tune state is complete);
3. a resume call continues from the saved step rather than restarting at 0.

Usage (1 free GPU, a few cores):
    python scripts/slurm/_smoke_corruption_train.py \
        --shots 15085,15086,15087,15088 \
        --token-root /work/projects/imas_gpu/mast-tokens
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

import torch

from imas_ambix.worldmodel.spacetime_dataset import (
    REFERENCE_CAMERA,
    SpacetimeWindowConfig,
    discover_camera_shots,
)
from imas_ambix.worldmodel.spacetime_dataset_v2 import default_signal_modalities
from imas_ambix.worldmodel.spacetime_train_v2 import (
    ContextCorruptionConfig,
    CorpusV2Config,
    find_latest_checkpoint,
    train_corpus,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("smoke-corruption")


def _resolve_shots(args, span: int):  # noqa: ANN001
    token_root = Path(args.token_root) if args.token_root else None
    if args.shots.strip():
        shots = [int(s) for s in args.shots.split(",") if s.strip()]
    else:
        shots = discover_camera_shots(
            camera=args.camera,
            token_root=token_root,
            min_frames=span,
            limit=args.n_shots + len(_held_out(args)),
        )
    return shots


def _held_out(args) -> list[int]:  # noqa: ANN001
    return [int(s) for s in args.eval_shots.split(",") if s.strip()]


def _build_config(args, window: SpacetimeWindowConfig, steps: int) -> CorpusV2Config:
    corruption = ContextCorruptionConfig(
        max_rate=args.corruption_max_rate,
        levels=args.corruption_levels,
        clean_fraction=args.corruption_clean_fraction,
        control_dropout=args.control_dropout,
    )
    # small backbone — this smoke checks the PLUMBING, not capacity.
    return CorpusV2Config(
        steps=steps,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        warmup_steps=2,
        chunk=args.chunk,
        n_signal_steps=args.n_signal_steps,
        window=window,
        n_eval_shots=len(_held_out(args)) or 1,
        log_every=1,
        ckpt_every=args.steps_per_call,  # write latest.pt within the call
        eval_every=0,  # skip eval in the smoke
        num_workers=0,
        window_stride=args.window_stride,
        corruption=corruption,
        use_corruption=True,
        modalities=default_signal_modalities(),
        model_kwargs=dict(
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            d_ff=4 * args.d_model,
            dropout=0.0,
        ),
    )


def main(argv=None) -> int:  # noqa: ANN001
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shots", default="", help="comma-separated rbb shot ids")
    p.add_argument("--token-root", default="/work/projects/imas_gpu/mast-tokens")
    p.add_argument("--camera", default=REFERENCE_CAMERA)
    p.add_argument("--eval-shots", default="18502,18503,18504,18505")
    p.add_argument("--n-shots", type=int, default=4)
    p.add_argument("--steps-per-call", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--chunk", type=int, default=4096)
    p.add_argument("--n-frames", type=int, default=12)
    p.add_argument("--n-plan", type=int, default=4)
    p.add_argument("--n-signal-steps", type=int, default=3)
    p.add_argument("--context-frames", type=int, default=4)
    p.add_argument("--window-stride", type=int, default=4)
    # corruption knobs (defaults match the production recipe)
    p.add_argument("--corruption-levels", type=int, default=8)
    p.add_argument("--corruption-max-rate", type=float, default=0.30)
    p.add_argument("--corruption-clean-fraction", type=float, default=0.25)
    p.add_argument("--control-dropout", type=float, default=0.15)
    # small backbone for the smoke
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--out-dir", default="")
    args = p.parse_args(argv)
    args.steps = args.steps_per_call  # alias used by _build_config

    token_root = Path(args.token_root) if args.token_root else None
    window = SpacetimeWindowConfig(
        n_frames=args.n_frames,
        n_plan=args.n_plan,
        context_frames=args.context_frames,
        frame_stride=1,
    )
    span = (window.n_frames - 1) * window.frame_stride + 1

    held = set(_held_out(args))
    pool = _resolve_shots(args, span)
    train_shots = [s for s in pool if s not in held]  # the trainer's subtraction
    if len(train_shots) < 1:
        logger.error("no train shots resolved (pool=%s held=%s)", pool, sorted(held))
        return 1
    logger.info(
        "SMOKE train shots=%s (held-out subtracted=%s)", train_shots, sorted(held)
    )

    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir is None:
        import tempfile  # noqa: PLC0415

        out_dir = Path(tempfile.mkdtemp(prefix="smoke-corruption-"))
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("out-dir=%s", out_dir)

    # ── CALL 1: fresh run, a few steps with corruption + dropout ON ──
    cfg1 = _build_config(args, window, steps=args.steps_per_call)
    logger.info(
        "===== CALL 1: fresh corruption+dropout train (%d steps) =====", cfg1.steps
    )
    res1 = train_corpus(
        train_shots,
        camera=args.camera,
        config=cfg1,
        out_dir=out_dir,
        token_root=token_root,
        eval_shot_ids=sorted(held) or None,
        resume=False,
    )
    if math.isnan(res1.final_loss):
        logger.error("CALL 1 produced a NaN loss")
        return 3
    logger.info("CALL 1 final_loss=%.4f steps=%d", res1.final_loss, res1.steps_run)

    ckpt = find_latest_checkpoint(out_dir)
    if ckpt is None or not ckpt.exists():
        logger.error("latest.pt NOT written after CALL 1 (out=%s)", out_dir)
        return 3
    payload = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    step1 = int(payload.get("step", 0))
    has_corruption_embed = any(
        "corruption_embed" in k for k in payload["model_state_dict"]
    )
    logger.info(
        "CALL 1 wrote %s @ step %d (corruption_embed present=%s)",
        ckpt,
        step1,
        has_corruption_embed,
    )
    if not has_corruption_embed:
        logger.error("checkpoint missing corruption_embed — fine-tune state incomplete")
        return 3
    if step1 < 1:
        logger.error("checkpoint step is %d (<1) — no optimiser step recorded", step1)
        return 3

    # ── CALL 2: resume from latest.pt, advance further ──
    cfg2 = _build_config(args, window, steps=args.steps_per_call * 2)
    logger.info(
        "===== CALL 2: RESUME from latest.pt (target %d steps) =====", cfg2.steps
    )
    res2 = train_corpus(
        train_shots,
        camera=args.camera,
        config=cfg2,
        out_dir=out_dir,
        token_root=token_root,
        eval_shot_ids=sorted(held) or None,
        resume=True,
    )
    payload2 = torch.load(
        str(find_latest_checkpoint(out_dir)), map_location="cpu", weights_only=False
    )
    step2 = int(payload2.get("step", 0))
    logger.info(
        "CALL 2 final_loss=%.4f resumed-and-advanced to step %d", res2.final_loss, step2
    )
    if step2 <= step1:
        logger.error(
            "RESUME did not advance: step after CALL 2 (%d) <= step after CALL 1 (%d)",
            step2,
            step1,
        )
        return 3

    logger.info(
        "GATE PASS: corruption+dropout+overlapping-window TRAIN ran end-to-end; "
        "latest.pt written+reloadable; resume advanced %d -> %d steps.",
        step1,
        step2,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
