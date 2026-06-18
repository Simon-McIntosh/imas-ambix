"""Overfit gate for the spatiotemporal camera transformer.

Runs the WHOLE gate in one process on one GPU:

1. Overfit a small model on a handful of real rbb shots until the loss is very
   low (proves the architecture CAN fit the frame-token sequences).
2. Save the overfit checkpoint.
3. Decode the model's TEACHER-FORCED reconstruction AND a short autoregressive
   rollout back to 256x256 images through the frozen Open-MAGVIT2 VQModel
   (the validated two-venv decode path), writing GT-vs-prediction GIFs + PNGs.

PASS criterion (judged by a human looking at the PNG): the overfit
teacher-forced reconstruction is CLEARLY COHERENT — recognizable plasma
structure matching GT, not mush.  That proves spatial structure survives the
factorized space-time stack.  If it is NOT coherent, the corpus run does NOT
launch.

Run (single GPU; keep the neighbour's LLM server up)::

    .venv/bin/python scripts/slurm/spacetime_overfit_gate.py \\
        --shots 24065,23735 --steps 600 --out-dir /work/.../spacetime_smoke
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

logger = logging.getLogger("spacetime_gate")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--shots", default="24065,23735", help="comma-separated rbb shot ids"
    )
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--n-frames", type=int, default=24)
    p.add_argument("--n-plan", type=int, default=8)
    p.add_argument("--context-frames", type=int, default=8)
    p.add_argument("--frame-stride", type=int, default=1)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--n-layers", type=int, default=8)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--d-ff", type=int, default=2048)
    p.add_argument("--chunk", type=int, default=16384)
    p.add_argument(
        "--out-dir", default="/work/projects/imas_gpu/worldmodel/spacetime_smoke"
    )
    p.add_argument("--ckpt-dir", default=None, help="default: <out-dir>/overfit_ckpt")
    p.add_argument("--decode-shot", type=int, default=None, help="default: first shot")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    import torch

    from imas_ambix.worldmodel.spacetime_dataset import SpacetimeWindowConfig
    from imas_ambix.worldmodel.spacetime_train import (
        OverfitConfig,
        overfit,
        save_checkpoint,
    )

    shots = [int(s) for s in args.shots.split(",") if s.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else out_dir / "overfit_ckpt"

    window = SpacetimeWindowConfig(
        n_frames=args.n_frames,
        n_plan=args.n_plan,
        context_frames=args.context_frames,
        frame_stride=args.frame_stride,
    )
    cfg = OverfitConfig(
        steps=args.steps,
        lr=args.lr,
        chunk=args.chunk,
        window=window,
        model_kwargs={
            "d_model": args.d_model,
            "n_layers": args.n_layers,
            "n_heads": args.n_heads,
            "d_ff": args.d_ff,
        },
    )

    logger.info("=== GATE: overfit %s ===", shots)
    t0 = time.time()
    result, model, _samples = overfit(shots, config=cfg)
    logger.info(
        "overfit done in %.1fs: params=%d (%.1fM) initial=%.4f final=%.4f drop=%.4f",
        time.time() - t0,
        result.n_parameters,
        result.n_parameters / 1e6,
        result.initial_loss,
        result.final_loss,
        result.loss_drop_ratio,
    )

    # save the overfit checkpoint (the dream module reloads it for decode).
    opt = None
    ckpt = save_checkpoint(
        ckpt_dir, model=model, optimizer=opt, step=args.steps, window=window
    )
    logger.info("overfit checkpoint -> %s", ckpt)

    # free the training model before the decode (loads its own copy).
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── decode the overfit model's reconstruction + dream ───────────────────
    decode_shot = args.decode_shot or shots[0]
    logger.info("=== GATE: decode shot %s (teacher-forced + dream) ===", decode_shot)
    from imas_ambix.worldmodel.spacetime_dream import build

    try:
        summary = build(
            checkpoint=ckpt,
            shot_id=decode_shot,
            out_dir=out_dir,
            window=window,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
    except Exception as exc:  # noqa: BLE001 — record + still report the loss drop
        logger.exception("decode failed: %r", exc)
        gate = {
            "shots": shots,
            "overfit_initial_loss": result.initial_loss,
            "overfit_final_loss": result.final_loss,
            "loss_drop_ratio": result.loss_drop_ratio,
            "n_parameters": result.n_parameters,
            "decode_error": repr(exc),
        }
        (out_dir / "gate_summary.json").write_text(json.dumps(gate, indent=2))
        logger.error("GATE: decode unavailable — loss-drop only; inspect logs")
        return 3

    gate = {
        "shots": shots,
        "decode_shot": decode_shot,
        "overfit_initial_loss": result.initial_loss,
        "overfit_final_loss": result.final_loss,
        "loss_drop_ratio": result.loss_drop_ratio,
        "n_parameters": result.n_parameters,
        "teacher_forced_token_mismatch": summary["teacher_forced_token_mismatch"],
        "dream_token_mismatch": summary["dream_token_mismatch"],
        "dream_change_fraction": summary["dream_change_fraction"],
        "figure_paths": summary["figure_paths"],
    }
    (out_dir / "gate_summary.json").write_text(json.dumps(gate, indent=2))

    print("\n=== SPACETIME OVERFIT GATE SUMMARY ===")
    print(f"shots: {shots}  decode_shot: {decode_shot}")
    print(
        f"overfit loss: {result.initial_loss:.4f} -> {result.final_loss:.4f} "
        f"(drop {result.loss_drop_ratio:.4f})  params={result.n_parameters:,}"
    )
    print(
        "teacher-forced token mismatch (forecast window): "
        f"{summary['teacher_forced_token_mismatch']:.4f}"
    )
    print(f"dream token mismatch: {summary['dream_token_mismatch']:.4f}")
    print(f"dream change-from-last-context: {summary['dream_change_fraction']:.4f}")
    print(f"figures: {summary['figure_paths']}")
    print(
        "\nPASS = the teacher-forced PNG shows coherent plasma structure (human read)."
    )
    # A low overfit loss is necessary but not sufficient; the PNG is the gate.
    if result.loss_drop_ratio > 0.5:
        print("WARNING: loss barely dropped — the model did not overfit; investigate.")
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
