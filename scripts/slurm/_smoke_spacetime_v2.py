"""Overfit + memory smoke for the signal-conditioned camera model (v2).

Run on ONE GPU (or CPU fallback).  Two phases:

1. OVERFIT a few real rbb shots with measured-signal conditioning and confirm
   the loss drops sharply (it LEARNS) AND that the conditioning is LOAD-BEARING
   (a signals-zeroed forward gives a measurably different loss than the
   full-signals forward — the signals genuinely feed the prediction).
2. MEMORY: build the SCALED 4-GPU-target model and run a forward+backward at the
   per-rank batch on one card, reporting peak allocated memory so the 4-GPU
   launch is sized right.

Usage (1 free GPU, 1 core):
    python scripts/slurm/_smoke_spacetime_v2.py --shots 15085,15086,15087 \
        --token-root /work/projects/imas_gpu/mast-tokens
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

from imas_ambix.worldmodel.spacetime_dataset import (
    REFERENCE_CAMERA,
    SpacetimeWindowConfig,
    discover_camera_shots,
)
from imas_ambix.worldmodel.spacetime_dataset_v2 import (
    default_signal_modalities,
)
from imas_ambix.worldmodel.spacetime_train_v2 import (
    OverfitV2Config,
    _batch_to,
    build_signal_model,
    collate_signal_windows,
    overfit_signal,
    signal_ablation_delta,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("smoke-v2")


def _overfit_phase(args) -> int:  # noqa: ANN001
    token_root = Path(args.token_root) if args.token_root else None
    window = SpacetimeWindowConfig(
        n_frames=args.n_frames,
        n_plan=args.n_plan,
        context_frames=args.context_frames,
        frame_stride=1,
    )
    span = (window.n_frames - 1) * window.frame_stride + 1

    if args.shots.strip():
        shots = [int(s) for s in args.shots.split(",") if s.strip()]
    else:
        shots = discover_camera_shots(
            camera=REFERENCE_CAMERA,
            token_root=token_root,
            min_frames=span,
            limit=args.n_overfit_shots,
        )
    shots = shots[: args.n_overfit_shots]
    if len(shots) < 1:
        logger.error("no rbb shots discovered with >= %d frames", span)
        return 1
    logger.info("OVERFIT shots=%s", shots)

    cfg = OverfitV2Config(
        steps=args.steps,
        lr=args.lr,
        chunk=args.chunk,
        n_signal_steps=args.n_signal_steps,
        window=window,
        model_kwargs=dict(
            d_model=args.smoke_d_model,
            n_layers=args.smoke_n_layers,
            n_heads=args.smoke_n_heads,
            d_ff=4 * args.smoke_d_model,
            dropout=0.0,
        ),
    )
    result, model, samples = overfit_signal(shots, config=cfg, token_root=token_root)
    drop = result.loss_drop_ratio
    logger.info(
        "OVERFIT: params=%.2fM initial=%.4f final=%.4f drop_ratio=%.4f",
        result.n_parameters / 1e6,
        result.initial_loss,
        result.final_loss,
        drop,
    )

    # ── signal-ablation: prove the conditioning is LOAD-BEARING ──
    modalities = default_signal_modalities()
    channels: dict[str, int] = {}
    for s in samples:
        for name, arr in s.signals.items():
            channels[name] = max(channels.get(name, 0), int(arr.shape[1]))
    from imas_ambix.worldmodel.spacetime_dataset_v2 import (
        stream_specs_from_modalities,
    )

    streams = stream_specs_from_modalities(modalities, channels)
    stream_names = [st.name for st in streams]
    dev = next(model.parameters()).device
    batch = _batch_to(collate_signal_windows(samples, stream_names=stream_names), dev)
    full, zero = signal_ablation_delta(model, batch, chunk=args.chunk)
    delta = abs(full - zero)
    logger.info(
        "SIGNAL-ABLATION: present_streams=%s loss_full=%.5f loss_signals_zeroed=%.5f "
        "abs_delta=%.5f",
        [(st.name, st.channels) for st in streams],
        full,
        zero,
        delta,
    )

    pass_overfit = drop < 0.5  # final loss < half the initial = it learned
    pass_ablation = delta > 1e-3
    if not streams:
        logger.warning(
            "NO signal streams present for these shots — ablation is vacuous; "
            "pick shots known to carry the measured streams"
        )
    logger.info(
        "GATE: overfit_learns=%s (drop=%.4f<0.5) signals_load_bearing=%s "
        "(delta=%.5f>1e-3)",
        pass_overfit,
        drop,
        pass_ablation,
        delta,
    )
    return 0 if (pass_overfit and (pass_ablation or not streams)) else 3


def _memory_phase(args) -> int:  # noqa: ANN001
    if not torch.cuda.is_available():
        logger.warning("MEMORY phase skipped — no CUDA (run on a GPU to measure peak)")
        return 0
    token_root = Path(args.token_root) if args.token_root else None
    window = SpacetimeWindowConfig(
        n_frames=args.n_frames,
        n_plan=args.n_plan,
        context_frames=args.context_frames,
        frame_stride=1,
    )
    span = (window.n_frames - 1) * window.frame_stride + 1
    if args.shots.strip():
        shots = [int(s) for s in args.shots.split(",") if s.strip()]
    else:
        shots = discover_camera_shots(
            camera=REFERENCE_CAMERA,
            token_root=token_root,
            min_frames=span,
            limit=args.batch_size,
        )
    shots = shots[: args.batch_size]
    if len(shots) < 1:
        logger.error("no shots for the memory phase")
        return 1

    from imas_ambix.worldmodel.spacetime_dataset_v2 import (
        assemble_signal_window,
        stream_specs_from_modalities,
    )

    modalities = default_signal_modalities()
    samples = []
    for sid in shots:
        try:
            samples.append(
                assemble_signal_window(
                    sid, window, modalities, args.n_signal_steps, token_root=token_root
                )
            )
        except (ValueError, FileNotFoundError, KeyError) as exc:
            logger.warning("memory-phase shot %s unassemblable: %r", sid, exc)
    if not samples:
        logger.error("no assemblable shots for the memory phase")
        return 1
    # repeat to fill the per-rank batch if fewer shots than batch_size
    while len(samples) < args.batch_size:
        samples.append(samples[len(samples) % len(samples)])
    samples = samples[: args.batch_size]

    channels: dict[str, int] = {}
    for s in samples:
        for name, arr in s.signals.items():
            channels[name] = max(channels.get(name, 0), int(arr.shape[1]))
    streams = stream_specs_from_modalities(modalities, channels)
    stream_names = [st.name for st in streams]
    plan_ch = max(
        (int(s.plan.shape[1]) for s in samples if s.plan.ndim == 2 and s.plan.size),
        default=0,
    )

    dev = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats(dev)
    model = build_signal_model(
        window,
        plan_channels=plan_ch,
        signal_streams=streams,
        n_signal_steps=args.n_signal_steps,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
    ).to(dev)
    n_params = model.num_parameters()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)
    model.train()
    batch = _batch_to(collate_signal_windows(samples, stream_names=stream_names), dev)
    # one full train step (fwd + chunked-CE backward + opt) at the target size.
    for _ in range(2):
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(
                batch,
                loss_spec={
                    "chunk": args.chunk,
                    "context_frames": window.context_frames,
                },
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    peak = torch.cuda.max_memory_allocated(dev) / 1e9
    reserved = torch.cuda.max_memory_reserved(dev) / 1e9
    total = torch.cuda.get_device_properties(dev).total_memory / 1e9
    logger.info(
        "MEMORY @ d_model=%d n_layers=%d n_heads=%d d_ff=%d batch=%d "
        "n_signal_steps=%d prefix_frames=%d:",
        args.d_model,
        args.n_layers,
        args.n_heads,
        args.d_ff,
        args.batch_size,
        args.n_signal_steps,
        window.n_plan + len(streams) * args.n_signal_steps,
    )
    logger.info(
        "  params=%.1fM peak_alloc=%.2f GB peak_reserved=%.2f GB / card_total=%.1f GB "
        "loss=%.4f streams=%s",
        n_params / 1e6,
        peak,
        reserved,
        total,
        float(loss.detach()),
        [(st.name, st.channels) for st in streams],
    )
    del model, opt
    torch.cuda.empty_cache()
    return 0


def main(argv=None) -> int:  # noqa: ANN001
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shots", default="", help="comma-separated rbb shot ids")
    p.add_argument("--token-root", default="/work/projects/imas_gpu/mast-tokens")
    p.add_argument("--n-overfit-shots", type=int, default=3)
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--chunk", type=int, default=8192)
    p.add_argument("--n-frames", type=int, default=24)
    p.add_argument("--n-plan", type=int, default=8)
    p.add_argument("--n-signal-steps", type=int, default=4)
    p.add_argument("--context-frames", type=int, default=8)
    # overfit (small) model
    p.add_argument("--smoke-d-model", type=int, default=384)
    p.add_argument("--smoke-n-layers", type=int, default=4)
    p.add_argument("--smoke-n-heads", type=int, default=6)
    # memory-phase (scaled 4-GPU-target) model
    p.add_argument("--d-model", type=int, default=1536)
    p.add_argument("--n-layers", type=int, default=16)
    p.add_argument("--n-heads", type=int, default=16)
    p.add_argument("--d-ff", type=int, default=6144)
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--phase", choices=["overfit", "memory", "both"], default="both")
    args = p.parse_args(argv)

    rc = 0
    if args.phase in ("overfit", "both"):
        logger.info("===== OVERFIT PHASE =====")
        rc = _overfit_phase(args)
        if rc != 0 and args.phase == "overfit":
            return rc
    if args.phase in ("memory", "both"):
        logger.info("===== MEMORY PHASE =====")
        mrc = _memory_phase(args)
        rc = rc or mrc
    return rc


if __name__ == "__main__":
    sys.exit(main())
