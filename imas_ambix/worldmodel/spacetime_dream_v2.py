"""Decode the SIGNAL-CONDITIONED spacetime camera model to GT-vs-prediction.

The v2 counterpart of :mod:`imas_ambix.worldmodel.spacetime_dream`.  The v1
decoder cannot score a v2 checkpoint: it rebuilds a plain
:class:`~imas_ambix.worldmodel.spacetime_model.SpacetimeTransformer`, which
rejects the ``signal_*`` state-dict keys, and its prediction helpers never feed
the measured signals.  This module rebuilds the
:class:`~imas_ambix.worldmodel.spacetime_model_v2.SignalSpacetimeTransformer`
via :func:`imas_ambix.worldmodel.spacetime_train_v2.load_signal_model_from_checkpoint`,
assembles a held-out window WITH its measured signals, and runs the
signal-aware teacher-forced + autoregressive rollout
(:func:`...spacetime_train_v2.teacher_forced_signal_frames` /
``autoregressive_signal_dream``) that conditions on the plan AND the measured
plasma state at every step.

Everything downstream of the token prediction is REUSED verbatim from v1 so the
verdict is the SAME honest metric:

* Phase B — :func:`imas_ambix.camdyn.reconstruction_demo.run_decode_subprocess`
  decodes the ``(N, F, 16, 16)`` STORE-id bundle to ``(N, F, 256, 256, 3)`` via
  the frozen Open-MAGVIT2 venv (the two-venv handoff).
* The pixel verdict is the v1 ``spacetime_dream.forecast_pixel_errors`` — decoded
  model-pixel error vs the persistence (frozen-last-context) baseline on the
  forecast window — and the GIF + contact-sheet assembly is the v1
  ``spacetime_dream.assemble_figures``.

The pixel-vs-persistence ratio is the verdict, NOT the GIF coherence: a model can
look coherent while losing to persistence (the v1 camera-only lesson).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
import time
from pathlib import Path

import numpy as np

from imas_ambix.worldmodel.spacetime_dataset import (
    GRID_H,
    GRID_W,
    REFERENCE_CAMERA,
    SpacetimeWindowConfig,
    local_to_store,
)
from imas_ambix.worldmodel.spacetime_dataset_v2 import (
    SignalModalitySpec,
    assemble_signal_window,
    default_signal_modalities,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Phase A — predict + dump the STORE-id token bundle (signal-conditioned)
# ---------------------------------------------------------------------------


def run_phase_a(
    *,
    checkpoint: Path,
    shot_id: int,
    token_bundle: Path,
    window: SpacetimeWindowConfig,
    modalities: list[SignalModalitySpec] | None = None,
    n_signal_steps: int = 4,
    camera: str = REFERENCE_CAMERA,
    device: str = "cuda",
    token_root: Path | None = None,
) -> dict:
    """Load the v2 model once; build GT + teacher-forced + dream STORE-id grids.

    The signals are assembled for the held-out window and FED on every rollout
    step (teacher-forced and autoregressive), so the prediction always sees the
    measured plasma state.  The stream set/order is pinned to the model's so any
    stream the shot lacks is presented as an all-PAD block (the model's embedding
    tables stay in the graph and the prefix length is deterministic).
    """
    import torch

    from imas_ambix.worldmodel.spacetime_train_v2 import (
        autoregressive_signal_dream,
        load_signal_model_from_checkpoint,
        teacher_forced_signal_frames,
    )

    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if device == "cuda" and torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    else:
        device = "cpu"

    modalities = modalities or default_signal_modalities()

    logger.info("loading v2 checkpoint on %s: %s", device, checkpoint)
    model, payload = load_signal_model_from_checkpoint(checkpoint, map_location=device)
    model.eval()
    step = int(payload.get("step", -1))
    # The model knows the exact stream set it was trained with; pin to it so a
    # decode never silently drops a stream (or invents one the model lacks).
    model_streams = [st.name for st in model.config.signal_streams]
    logger.info(
        "checkpoint step=%d params=%d streams=%s",
        step,
        int(model.num_parameters()),
        model_streams,
    )

    logger.info("assembling held-out shot %s camera %s WITH signals", shot_id, camera)
    sample = assemble_signal_window(
        int(shot_id),
        window,
        modalities,
        int(n_signal_steps),
        camera=camera,
        token_root=token_root,
    )
    present = sorted(sample.signals.keys())
    logger.info("shot %s present signal streams: %s", shot_id, present)

    dev = torch.device(device)
    ctx = int(sample.context_frames)
    n_frames = int(sample.frames.shape[0])

    gt_local = np.asarray(sample.frames, dtype=np.int64).reshape(
        n_frames, GRID_H, GRID_W
    )

    t0 = time.time()
    tf_local = teacher_forced_signal_frames(
        model, sample, stream_names=model_streams, device=dev
    ).reshape(n_frames, GRID_H, GRID_W)
    logger.info("teacher-forced (signal) prediction in %.1fs", time.time() - t0)

    t0 = time.time()
    dream_local = autoregressive_signal_dream(
        model, sample, stream_names=model_streams, device=dev
    ).reshape(n_frames, GRID_H, GRID_W)
    logger.info("autoregressive (signal) dream in %.1fs", time.time() - t0)

    # honesty readouts over the forecast window (token space).
    tf_mismatch = float((tf_local[ctx:] != gt_local[ctx:]).mean())
    dream_mismatch = float((dream_local[ctx:] != gt_local[ctx:]).mean())
    last_ctx = dream_local[ctx - 1]
    dream_change = float((dream_local[ctx:] != last_ctx[None]).mean())

    try:
        del model
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:  # noqa: BLE001
        logger.warning("model release note: %r", exc)

    grids_stack = np.stack(
        [
            local_to_store(gt_local),
            local_to_store(tf_local),
            local_to_store(dream_local),
        ]
    ).astype(np.int64)
    index = [
        {"role": "gt", "slot": 0},
        {"role": "teacher_forced", "slot": 1},
        {"role": "dream", "slot": 2},
    ]
    meta = {
        "format": "reconstruction_demo",
        "camera": camera,
        "id_space": "store-id (local + REGISTRY_OFFSET; decode subtracts it)",
        "grid_hw": [GRID_H, GRID_W],
        "shot_id": int(shot_id),
        "checkpoint": str(checkpoint),
        "checkpoint_step": step,
        "n_frames": int(n_frames),
        "context_frames": int(ctx),
        "frame_time": np.asarray(sample.frame_time, dtype=np.float64).tolist(),
        "signal_streams": model_streams,
        "present_streams": present,
        "n_signal_steps": int(n_signal_steps),
    }
    token_bundle.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        token_bundle, grids=grids_stack, index=json.dumps(index), meta=json.dumps(meta)
    )
    logger.info("wrote token bundle -> %s", token_bundle)

    return {
        "checkpoint": str(checkpoint),
        "checkpoint_step": step,
        "shot_id": int(shot_id),
        "camera": camera,
        "n_frames": int(n_frames),
        "context_frames": int(ctx),
        "teacher_forced_token_mismatch": tf_mismatch,
        "dream_token_mismatch": dream_mismatch,
        "dream_change_fraction": dream_change,
        "frame_time": meta["frame_time"],
        "signal_streams": model_streams,
        "present_streams": present,
    }


# ---------------------------------------------------------------------------
# Orchestration — Phase A (here) + Phase B/C reused from v1
# ---------------------------------------------------------------------------


def build(
    *,
    checkpoint: Path,
    shot_id: int,
    out_dir: Path,
    window: SpacetimeWindowConfig,
    modalities: list[SignalModalitySpec] | None = None,
    n_signal_steps: int = 4,
    camera: str = REFERENCE_CAMERA,
    device: str = "cuda",
    token_root: Path | None = None,
    work_dir: Path | None = None,
) -> dict:
    # Phase B/C are byte-identical to v1 — the bundle format + the honest pixel
    # scorer + the GIF assembly are SHARED so the v2 verdict is the same metric.
    from imas_ambix.camdyn.reconstruction_demo import run_decode_subprocess
    from imas_ambix.worldmodel.spacetime_dream import assemble_figures

    out_dir = Path(out_dir)
    work_dir = work_dir or Path(
        tempfile.mkdtemp(prefix="st-dream-v2-", dir=os.environ.get("TMPDIR", "/tmp"))
    )
    work_dir.mkdir(parents=True, exist_ok=True)
    token_bundle = work_dir / "tokens.npz"
    image_bundle = work_dir / "images.npz"

    summary = run_phase_a(
        checkpoint=checkpoint,
        shot_id=shot_id,
        token_bundle=token_bundle,
        window=window,
        modalities=modalities,
        n_signal_steps=n_signal_steps,
        camera=camera,
        device=device,
        token_root=token_root,
    )
    logger.info("decoding token grids via the MAGVIT2 venv")
    run_decode_subprocess(token_bundle, image_bundle, "cuda")
    if not image_bundle.exists():
        raise RuntimeError(f"decode produced no image bundle at {image_bundle}")
    summary.update(
        assemble_figures(image_bundle=image_bundle, summary=summary, out_dir=out_dir)
    )
    return summary


def _verdict_line(label: str, px: dict | None) -> str:
    if px is None:
        return f"{label}: (no pixel metric)"
    verdict = (
        "BEATS persistence" if px["model_beats_persistence"] else "LOSES to persistence"
    )
    return (
        f"{label}: model={px['model_pixel_error']:.3f}  "
        f"persistence={px['persistence_pixel_error']:.3f}  "
        f"ratio={px['ratio']:.3f}x  -> {verdict}"
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument(
        "--shots",
        default="",
        help="comma-separated held-out shot ids (overrides --shot)",
    )
    p.add_argument("--shot", type=int, default=None)
    p.add_argument(
        "--out-dir", default="/work/projects/imas_gpu/worldmodel/spacetime_v2_decode"
    )
    p.add_argument("--camera", default=REFERENCE_CAMERA)
    p.add_argument("--n-frames", type=int, default=24)
    p.add_argument("--n-plan", type=int, default=8)
    p.add_argument("--context-frames", type=int, default=8)
    p.add_argument("--frame-stride", type=int, default=1)
    p.add_argument("--n-signal-steps", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--token-root", default=None)
    p.add_argument("--work-dir", default=None)
    p.add_argument(
        "--summary-json",
        default=None,
        help="optional path to dump the per-shot verdict summary as JSON",
    )
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    window = SpacetimeWindowConfig(
        n_frames=args.n_frames,
        n_plan=args.n_plan,
        context_frames=args.context_frames,
        frame_stride=args.frame_stride,
    )

    if args.shots.strip():
        shot_ids = [int(s) for s in args.shots.split(",") if s.strip()]
    elif args.shot is not None:
        shot_ids = [int(args.shot)]
    else:
        p.error("provide --shots (comma-separated) or --shot")

    summaries: list[dict] = []
    for sid in shot_ids:
        shot_out = Path(args.out_dir) / f"shot-{sid}"
        logger.info("==== v2 decode shot %s -> %s ====", sid, shot_out)
        summary = build(
            checkpoint=Path(args.checkpoint),
            shot_id=int(sid),
            out_dir=shot_out,
            window=window,
            n_signal_steps=args.n_signal_steps,
            camera=args.camera,
            device=args.device,
            token_root=Path(args.token_root) if args.token_root else None,
            work_dir=Path(args.work_dir) / f"shot-{sid}" if args.work_dir else None,
        )
        summaries.append(summary)

    # ---- the VERDICT -----------------------------------------------------
    print("\n=== spacetime-v2 signal-conditioned decode VERDICT ===")
    if summaries:
        print(
            f"checkpoint: {summaries[0]['checkpoint']} "
            f"(step {summaries[0]['checkpoint_step']})"
        )
        print(f"signal streams fed: {summaries[0].get('signal_streams')}")
    dream_beats = 0
    tf_beats = 0
    for s in summaries:
        sid = s["shot_id"]
        dp = s.get("dream_pixel")
        tp = s.get("teacher_forced_pixel")
        print(f"\n-- shot {sid} (present streams: {s.get('present_streams')}) --")
        print(
            "  token mismatch (forecast): "
            f"teacher_forced={s['teacher_forced_token_mismatch']:.4f}  "
            f"dream={s['dream_token_mismatch']:.4f}  "
            f"dream_change_from_last_ctx={s['dream_change_fraction']:.4f}"
        )
        print("  " + _verdict_line("teacher_forced pixel", tp))
        print("  " + _verdict_line("dream pixel          ", dp))
        if dp is not None and dp["model_beats_persistence"]:
            dream_beats += 1
        if tp is not None and tp["model_beats_persistence"]:
            tf_beats += 1
        print(f"  figures: {s.get('figure_paths')}")

    n = len(summaries)
    print(
        f"\n=== AGGREGATE over {n} held-out shots: dream beats persistence on "
        f"{dream_beats}/{n}; teacher-forced beats on {tf_beats}/{n} ==="
    )
    if n and dream_beats == 0:
        print(
            "VERDICT: the signal-conditioned model LOSES to persistence on the "
            "autoregressive forecast for every held-out shot (a measured negative)."
        )
    elif n and dream_beats == n:
        print(
            "VERDICT: the signal-conditioned model BEATS persistence on the "
            "autoregressive forecast for every held-out shot."
        )
    elif n:
        print(
            f"VERDICT: mixed — dream beats persistence on {dream_beats}/{n} "
            "held-out shots."
        )

    if args.summary_json:
        sj = Path(args.summary_json)
        sj.parent.mkdir(parents=True, exist_ok=True)
        sj.write_text(json.dumps(summaries, indent=2, default=str))
        print(f"\nwrote verdict summary -> {sj}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
