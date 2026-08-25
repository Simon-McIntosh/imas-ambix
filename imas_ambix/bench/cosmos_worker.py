"""Zero-shot bench worker for NVIDIA Cosmos-Tokenizer DI16x16.

Mirrors :mod:`imas_ambix.bench.stream_worker` but swaps Open-MAGVIT2's VQModel
for Cosmos's .jit-compiled encoder + decoder.  Same in-process pattern:
load tokenizer once, encode + decode every shot in a single subprocess.

Manifest (JSON) shape matches stream_worker exactly so the parent bench
harness needs only minor branching:

    {
      "shots": [int, ...],
      "camera": "rbb",
      "l1_root": "/work/projects/imas_gpu/mast/level1/shots",
      "cosmos_root": "/work/projects/imas_gpu/mast-tokens/cosmos/v1/DI16x16",
      "max_items_per_shot": 32 | null,
      "output_dir": "/tmp/cosmos-bench-XXXX"
    }

Per-shot outputs:
    ``<shot_id>-tokens.npy``   — (T, 16, 16) int32 indices in [0, 65535]
    ``<shot_id>-decoded.npy``  — (T, H, W, 3) uint8 decoded at native (H,W)
    ``<shot_id>-src.npy``      — (T, H, W, 3) uint8 normalised RGB source

Cosmos encoder output:
    indices: (B, 16, 16) int32 in [0, 65535]
    codes:   (B, 6, 16, 16) bf16 in [-1, 1] — FSQ continuous codes (unused here)
    scalar:  (B, 1, 1, 1) fp32 — internal, ignored

Cosmos decoder takes indices, returns (B, 3, 256, 256) bf16 in [-1, 1].

Shots are subsampled to ``max_items_per_shot`` via an ``np.linspace`` uniform
stride across the full shot duration, matching fine-tune training and the
stream worker.
"""

from __future__ import annotations

import json
import signal
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

_TorchTensor = torch.Tensor

# ---------------------------------------------------------------------------
# Graceful-shutdown flag
# ---------------------------------------------------------------------------

STOP = threading.Event()
IMAGE_SIZE = 256
TOKEN_HW = 16
TOKENIZER_NAME = "frames_cosmos_di16x16_v1"
VOCAB_SIZE = 1 << 16  # 64 K FSQ codes
MODEL_FORWARD_BATCH = 4  # Mirror stream_worker; Cosmos JIT may be batch-invariant
                          # but using the same value keeps the comparison fair.


class StreamAborted(RuntimeError):  # noqa: N818  # not an Error-class, a control-flow unwind
    pass


def _install_signal_handlers() -> None:
    def _handler(signum, _frame) -> None:  # noqa: ANN001
        STOP.set()
        print(f"[cosmos-worker] signal {signum} → graceful stop", flush=True)

    try:
        signal.signal(signal.SIGTERM, _handler)
        signal.signal(signal.SIGINT, _handler)
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# Frame I/O (byte-identical to the stream-worker uniform-stride contract)
# ---------------------------------------------------------------------------


def normalise_frames_to_uint8(frames: np.ndarray) -> np.ndarray:
    if frames.dtype == np.uint8:
        return frames
    f = frames.astype(np.float32)
    lo = float(f.min())
    hi = float(f.max())
    if hi <= lo:
        return np.zeros_like(f, dtype=np.uint8)
    return ((f - lo) * 255.0 / (hi - lo)).clip(0, 255).astype(np.uint8)


def frames_to_rgb_uint8(frames: np.ndarray) -> np.ndarray:
    u8 = normalise_frames_to_uint8(frames)
    if u8.ndim == 3:
        return np.repeat(u8[..., None], 3, axis=-1)
    if u8.ndim == 4 and u8.shape[-1] == 3:
        return u8
    raise ValueError(f"frames must be (T,H,W) or (T,H,W,3), got {u8.shape}")


def load_shot_frames(
    shot_id: int, camera: str, l1_root: Path, max_frames: int | None = None
) -> np.ndarray:
    import xarray as xr

    shot_zarr = Path(l1_root) / f"{shot_id}.zarr"
    ds = xr.open_zarr(str(shot_zarr / camera))
    data_vars = list(ds.data_vars)
    if not data_vars:
        raise ValueError(f"no data_vars in '{camera}' of shot {shot_id}")
    frames = ds[data_vars[0]].values
    if max_frames is not None and frames.shape[0] > max_frames:
        # Uniform stride samples the full shot rather than only its prefix.
        indices = np.linspace(0, frames.shape[0] - 1, max_frames, dtype=int)
        frames = frames[indices]
    return np.asarray(frames)


def frames_to_input_device(
    frames_u8_rgb: np.ndarray,
    device: str,
    dtype,
) -> _TorchTensor:
    """(T, H, W, 3) uint8 → (T, 3, 256, 256) bf16 in [-1, 1].

    Mirrors stream_encode.frames_to_input_device — F.interpolate bilinear,
    no antialias, align_corners=False.  Same operator contract as the
    corpus encoder and fine-tune training.
    """
    import torch
    import torch.nn.functional as F

    if frames_u8_rgb.ndim != 4 or frames_u8_rgb.shape[-1] != 3:
        raise ValueError(f"expected (T, H, W, 3), got {frames_u8_rgb.shape}")
    t = torch.from_numpy(frames_u8_rgb).to(device)
    t = t.to(dtype).div(255.0).mul(2.0).sub(1.0)  # [-1, 1]
    t = t.permute(0, 3, 1, 2).contiguous(memory_format=torch.channels_last)
    t = F.interpolate(t, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear",
                      align_corners=False)
    return t


# ---------------------------------------------------------------------------
# Cosmos load + encode + decode
# ---------------------------------------------------------------------------


def load_cosmos(cosmos_root: Path, device: str):
    """Load Cosmos DI16x16 encoder + decoder JIT archives on ``device``."""
    import torch

    enc_path = Path(cosmos_root) / "encoder.jit"
    dec_path = Path(cosmos_root) / "decoder.jit"
    if not enc_path.exists() or not dec_path.exists():
        raise RuntimeError(
            f"Cosmos JIT files not found under {cosmos_root}: "
            f"need encoder.jit + decoder.jit"
        )
    encoder = torch.jit.load(str(enc_path), map_location=device)
    encoder.eval()
    decoder = torch.jit.load(str(dec_path), map_location=device)
    decoder.eval()
    if device.startswith("cuda"):
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    return encoder, decoder


def cosmos_encode_batch(
    encoder, images, device: str, model_forward_batch: int = MODEL_FORWARD_BATCH
) -> np.ndarray:
    """Encode ``(T, 3, 256, 256)`` bf16 in [-1, 1] → ``(T, 16, 16)`` int32 indices."""
    import torch

    images = images.to(device)
    n = images.shape[0]
    step = max(1, int(model_forward_batch))
    out: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, n, step):
            chunk = images[i : i + step]
            indices, _codes, _scalar = encoder(chunk)
            out.append(
                indices.detach().cpu().numpy().astype(np.int32).reshape(
                    chunk.shape[0], TOKEN_HW, TOKEN_HW
                )
            )
    return np.concatenate(out, axis=0)


def cosmos_decode_batch(
    decoder,
    indices_int32: np.ndarray,
    device: str,
    model_forward_batch: int,
    target_hw: tuple[int, int],
) -> np.ndarray:
    """Decode (T, 16, 16) int indices → (T, H, W, 3) uint8 at ``target_hw``."""
    import torch
    import torch.nn.functional as F

    T, h, w = indices_int32.shape
    out: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, T, model_forward_batch):
            chunk = indices_int32[i : i + model_forward_batch]
            idx = torch.from_numpy(chunk).to(device).to(torch.int32)
            recon = decoder(idx)  # (B, 3, 256, 256) bf16 in [-1, 1]
            t = recon.float().clamp(-1, 1).add(1.0).mul(127.5)
            t = F.interpolate(t, size=target_hw, mode="bilinear",
                              align_corners=False)
            t = t.clamp(0, 255).round().to(torch.uint8)
            t = t.permute(0, 2, 3, 1).contiguous().cpu().numpy()
            out.append(t)
    return np.concatenate(out, axis=0)


# ---------------------------------------------------------------------------
# Main worker loop
# ---------------------------------------------------------------------------


def _running_median(samples: list[float]) -> float:
    if not samples:
        return 0.0
    s = sorted(samples)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def run_worker(manifest: dict, device: str, model_forward_batch: int,
               batch_timeout_s: float) -> dict:
    import torch

    shots: list[int] = manifest["shots"]
    camera: str = manifest["camera"]
    l1_root = Path(manifest["l1_root"])
    cosmos_root = Path(manifest["cosmos_root"])
    max_items: int | None = manifest.get("max_items_per_shot")
    output_dir = Path(manifest["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    input_dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    timing_samples: list[float] = []

    def _shot_timeout() -> float:
        if batch_timeout_s > 0:
            return float(batch_timeout_s)
        if len(timing_samples) >= 3:
            med = _running_median(timing_samples)
            return max(60.0, 8.0 * med)
        return 60.0

    STOP.clear()
    _install_signal_handlers()
    t0_total = time.monotonic()

    encoder = decoder = None
    shots_ok = 0
    shots_fail = 0
    errors: list[dict] = []

    try:
        encoder, decoder = load_cosmos(cosmos_root, device)
        print(
            f"[cosmos-worker] tokenizer loaded in {time.monotonic() - t0_total:.1f}s "
            f"device={device} shots={len(shots)} camera={camera}",
            flush=True,
        )

        for shot_id in shots:
            if STOP.is_set():
                raise StreamAborted("STOP set before shot")
            t_shot_start = time.monotonic()
            try:
                raw = load_shot_frames(shot_id, camera, l1_root, max_items)
                u8_native = frames_to_rgb_uint8(raw)  # (T, H, W, 3)
                H, W = int(u8_native.shape[1]), int(u8_native.shape[2])

                t_enc0 = time.monotonic()
                images = frames_to_input_device(u8_native, device, input_dtype)
                indices = cosmos_encode_batch(
                    encoder, images, device, model_forward_batch
                )
                encode_s = time.monotonic() - t_enc0

                t_dec0 = time.monotonic()
                decoded = cosmos_decode_batch(
                    decoder, indices, device, model_forward_batch, (H, W)
                )
                decode_s = time.monotonic() - t_dec0

                np.save(str(output_dir / f"{shot_id}-tokens.npy"), indices)
                np.save(str(output_dir / f"{shot_id}-decoded.npy"), decoded)
                np.save(str(output_dir / f"{shot_id}-src.npy"), u8_native)

                shot_time = time.monotonic() - t_shot_start
                timing_samples.append(shot_time)
                if len(timing_samples) > 64:
                    del timing_samples[0]

                shots_ok += 1
                print(json.dumps({
                    "shot_id": shot_id,
                    "n_items": int(indices.shape[0]),
                    "encode_seconds": round(encode_s, 3),
                    "decode_seconds": round(decode_s, 3),
                    "native_hw": [H, W],
                    "error": None,
                }), flush=True)

            except StreamAborted:
                raise
            except Exception as exc:  # noqa: BLE001
                shots_fail += 1
                errors.append({"shot_id": shot_id, "error": str(exc)})
                print(json.dumps({
                    "shot_id": shot_id, "n_items": 0,
                    "encode_seconds": 0.0, "decode_seconds": 0.0,
                    "native_hw": None, "error": str(exc),
                }), flush=True)

    except StreamAborted:
        aborted = True
    else:
        aborted = False
    finally:
        if encoder is not None:
            try:
                del encoder, decoder
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass

    return {
        "shots_ok": shots_ok,
        "shots_fail": shots_fail,
        "elapsed_s": round(time.monotonic() - t0_total, 2),
        "aborted": aborted,
        "errors": errors[:50],
        "tokenizer_name": TOKENIZER_NAME,
        "vocab_size": VOCAB_SIZE,
        "registry_offset": 0,
        "model_forward_batch": model_forward_batch,
        "device": device,
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Cosmos DI16x16 bench worker")
    p.add_argument("--manifest", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--model-forward-batch", type=int, default=MODEL_FORWARD_BATCH)
    p.add_argument("--batch-timeout-s", type=float, default=0.0)
    p.add_argument("--report", default=None)
    args = p.parse_args(argv)

    manifest = json.loads(Path(args.manifest).read_text())
    summary = run_worker(
        manifest,
        device=args.device,
        model_forward_batch=args.model_forward_batch,
        batch_timeout_s=args.batch_timeout_s,
    )
    print(json.dumps({"summary": summary}, indent=2), flush=True)
    if args.report:
        Path(args.report).write_text(json.dumps(summary, indent=2))
    return 130 if summary.get("aborted") else 0


if __name__ == "__main__":
    sys.exit(main())
