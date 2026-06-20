"""GATED GPU re-encode of the curated dynamic-excitation windows (exposure-balanced).

THIS IS THE MULTI-HOUR GPU STEP — it must NOT run unannounced.  Launch only
after the lead's go (the select-phase manifest carries the compute estimate;
for a curated subset it is well under 1 GPU-hour).

What it does (in-process, model-loaded-once, SIGTERM-clean — AGENTS.md §2b):
reads the curated-window manifest (built by build_excitation_corpus.py select),
loads the Open-MAGVIT2 VQModel ONCE, and for each curated shot:

  1. loads the shot's RAW level-1 frames,
  2. applies the EXPOSURE-BALANCING transform
     (:func:`imas_ambix.worldmodel.exposure_balance.balance_exposure`) — the
     robust per-shot percentile clip, NOT the v0 global min/max — to bring shots
     to a common relative brightness BEFORE tokenisation,
  3. resizes to 256² ON device and runs ``model.encode`` in the byte-identical
     ``MODEL_FORWARD_BATCH`` sub-chunks (reusing stream_encode's helpers verbatim),
  4. writes the token Zarr to a SEPARATE curated root (never the live tokens dir).

It reuses stream_encode's building-block functions (load_model,
frames_to_input_device, encode_batch_indices, save_stream_frame_tokens,
load_shot_frames) so the encode path is identical to production EXCEPT the
normalisation step — the one thing the exposure characterisation says to change
for the curated subset.  It does NOT modify stream_encode.py.

This runs in the Open-MAGVIT2 venv (no ambix import); the two ambix helpers it
needs (the manifest read + the exposure transform) are pure-numpy and imported
defensively.  Whole held-out shots are excluded at SELECT time, so the manifest
already carries the shot-level leakage guard.

Launch (only after the lead's go):
    sbatch scripts/slurm/build_excitation_corpus_encode.sbatch
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

import numpy as np

# Curated token root — SEPARATE from the live tokens dir, never collides.
DEFAULT_CURATED_ROOT = Path(
    "/work/projects/imas_gpu/mast-tokens/v1/frames-excitation-curated"
)
DEFAULT_MANIFEST = Path(
    "/work/projects/imas_gpu/agents/excitation-corpus/curated_windows.json"
)

STOP = False


def _install_handlers() -> None:
    def _h(signum, _frame):  # noqa: ANN001
        global STOP
        STOP = True
        print(f"[encode] signal {signum} -> graceful stop", flush=True)

    try:
        signal.signal(signal.SIGTERM, _h)
        signal.signal(signal.SIGINT, _h)
    except ValueError:
        pass


def _percentile_balance(frames_thw: np.ndarray) -> np.ndarray:
    """Exposure-balance raw (T,H,W) frames -> (T,H,W) uint8, percentile clip.

    Imports the ambix transform lazily; falls back to an inline percentile clip
    if the ambix tree is not importable in the encode venv (keeps the encoder
    self-contained, matching stream_encode's no-ambix-import contract).
    """
    try:
        from imas_ambix.worldmodel.exposure_balance import percentile_normalise

        return percentile_normalise(frames_thw)
    except Exception:  # noqa: BLE001
        f = np.asarray(frames_thw, dtype=np.float32)
        lo = float(np.percentile(f, 1.0))
        hi = float(np.percentile(f, 99.5))
        if hi <= lo:
            return np.zeros(f.shape, dtype=np.uint8)
        return np.clip((f - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)


def _output_exists(shot_id: int, camera: str, curated_root: Path) -> bool:
    """True if a readable, non-empty token zarr already exists for this shot.

    Mirrors stream_encode._output_complete: validates the zarr opens and its
    ``tokens`` array is non-empty, so a torn write is re-encoded rather than
    silently skipped.  Used by --skip-existing to make the encode resumable and
    let shards co-exist idempotently (all shards write the same root).
    """
    import zarr

    from imas_ambix.data.stream_encode import stream_frames_token_path

    path = stream_frames_token_path(int(shot_id), camera, curated_root)
    if not path.exists():
        return False
    try:
        store = zarr.open_group(str(path), mode="r")
        return int(np.asarray(store["tokens"]).shape[0]) > 0
    except Exception:  # noqa: BLE001
        return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--curated-root", default=str(DEFAULT_CURATED_ROOT))
    ap.add_argument(
        "--magvit2-root", default="/work/projects/imas_gpu/mast-tokens/v1/open-magvit2"
    )
    ap.add_argument("--l1-root", default="/work/projects/imas_gpu/mast/level1/shots")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--camera", default="rbb")
    ap.add_argument(
        "--max-shots", type=int, default=None, help="cap shots (PoC dry-run)"
    )
    ap.add_argument("--exposure", default="percentile", help="percentile|global (A/B)")
    ap.add_argument(
        "--shard",
        type=int,
        default=0,
        help="this worker's shard index (0..n_shards-1); stride-shards the window list",
    )
    ap.add_argument(
        "--n-shards",
        type=int,
        default=1,
        help="number of parallel encode workers (GPUs) over the manifest",
    )
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help="skip shots whose token zarr already exists in the curated root "
        "(makes the encode RESUMABLE + lets shards co-exist idempotently)",
    )
    args = ap.parse_args(argv)

    # stream_encode lives in the magvit2 venv; import its byte-identical helpers.
    from imas_ambix.data.stream_encode import (
        IMAGE_SIZE,
        MODEL_FORWARD_BATCH,
        REGISTRY_OFFSET,
        encode_batch_indices,
        frames_to_input_device,
        frames_to_rgb_uint8,
        load_model,
        load_shot_frames,
        save_stream_frame_tokens,
    )

    manifest = json.loads(Path(args.manifest).read_text())
    windows = manifest["windows"]
    if args.max_shots:
        windows = windows[: args.max_shots]
    curated_root = Path(args.curated_root)

    # Stride-shard the window list across parallel workers (one GPU each): worker
    # `shard` takes windows[shard::n_shards].  Deterministic + disjoint, so N
    # shards partition the manifest with no overlap and no coordination.
    n_shards = max(1, int(args.n_shards))
    shard = int(args.shard) % n_shards
    if n_shards > 1:
        windows = windows[shard::n_shards]

    # Skip-existing: drop shots already encoded in the curated root, so the
    # encode is resumable and shards never redo the PoC/earlier output.
    n_before = len(windows)
    if args.skip_existing:
        windows = [
            w
            for w in windows
            if not _output_exists(int(w["shot_id"]), args.camera, curated_root)
        ]
    print(
        f"[encode shard {shard}/{n_shards}] {len(windows)} windows to encode "
        f"({n_before - len(windows)} skipped existing) -> {curated_root} "
        f"(exposure={args.exposure}, device={args.device})",
        flush=True,
    )

    _install_handlers()

    import torch

    model = None
    n_ok = 0
    n_fail = 0
    t0 = time.monotonic()
    try:
        model = load_model(Path(args.magvit2_root), args.device)
        print(f"[encode] model loaded in {time.monotonic() - t0:.1f}s", flush=True)
        dtype = torch.bfloat16 if str(args.device).startswith("cuda") else torch.float32
        for w in windows:
            if STOP:
                print("[encode] STOP set — flushing and exiting", flush=True)
                break
            sid = int(w["shot_id"])
            try:
                raw = load_shot_frames(sid, args.camera, Path(args.l1_root))
                # EXPOSURE BALANCE (the one change vs production): replace the v0
                # global min/max stretch with the robust per-shot transform.
                if args.exposure == "percentile":
                    u8 = _percentile_balance(raw)
                    u8_rgb = np.repeat(u8[..., None], 3, axis=-1)
                else:  # "global" — reproduce the v0 path for an A/B comparison
                    u8_rgb = frames_to_rgb_uint8(raw, presized=False)
                images = frames_to_input_device(u8_rgb, IMAGE_SIZE, args.device, dtype)
                toks = encode_batch_indices(
                    model, images, args.device, MODEL_FORWARD_BATCH
                )  # (T,16,16) int64 local ids
                global_ids = (toks.astype(np.int64) + REGISTRY_OFFSET).astype(np.int32)
                save_stream_frame_tokens(
                    sid,
                    args.camera,
                    global_ids,
                    input_shape=tuple(int(x) for x in raw.shape),
                    original_hw=(int(raw.shape[1]), int(raw.shape[2])),
                    stream_root=curated_root,
                )
                n_ok += 1
                if n_ok % 25 == 0:
                    print(
                        f"[encode] {n_ok} shots done, {time.monotonic() - t0:.0f}s",
                        flush=True,
                    )
            except Exception as exc:  # noqa: BLE001
                n_fail += 1
                print(f"[encode] shot {sid} FAILED: {exc!r}", flush=True)
    finally:
        if model is not None:
            try:
                del model
                if str(args.device).startswith("cuda"):
                    torch.cuda.empty_cache()
            except Exception as exc:  # noqa: BLE001
                print(f"[encode] model release warning: {exc}", flush=True)

    el = time.monotonic() - t0
    print(
        json.dumps(
            {
                "n_windows": len(windows),
                "shots_ok": n_ok,
                "shots_fail": n_fail,
                "elapsed_s": round(el, 1),
                "curated_root": str(curated_root),
                "exposure": args.exposure,
                "aborted": bool(STOP),
            },
            indent=2,
        ),
        flush=True,
    )
    return 130 if STOP else 0


if __name__ == "__main__":
    sys.exit(main())
