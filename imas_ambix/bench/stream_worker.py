"""In-process bench worker: load VQModel once, encode + decode all shots.

Standalone script that runs inside the Open-MAGVIT2 venv
(``/work/projects/imas_gpu/mast-tokens/v1/open-magvit2/.venv/bin/python``).
Loaded as a subprocess by :func:`imas_ambix.bench.tokenizer.benchmark_frame_tokenizer_in_process`.

CLI
---
Reads a manifest JSON written by the caller::

    {
      "shots": [int, ...],
      "camera": "rbb",
      "l1_root": "/work/projects/imas_gpu/mast/level1/shots",
      "magvit2_root": "/work/projects/imas_gpu/mast-tokens/v1/open-magvit2",
      "max_items_per_shot": 32 | null,
      "output_dir": "/tmp/ambix-bench-XXXX"
    }

Outputs per shot (saved to ``output_dir``):
    ``<shot_id>-tokens.npy``   — (T, 16, 16) int32 with REGISTRY_OFFSET applied
    ``<shot_id>-decoded.npy``  — (T, H, W, 3) uint8 decoded, resized to native (H,W)
    ``<shot_id>-src.npy``      — (T, H, W, 3) uint8 live-path normalised RGB src

Emits one JSON-line to stdout per shot (progress), then a final summary JSON
to ``--report <path>`` on exit.

Exit codes: 0 = clean, 130 = SIGTERM/SIGINT aborted.

Hardening mirrors ``imas_ambix.data.stream_encode``:
  - SIGTERM/SIGINT handler sets global STOP flag (< 5 s graceful exit).
  - Per-shot watchdog: fires at max(60, 8× running-median shot time).
  - ``finally`` releases model + empties CUDA cache so the H200 node is
    never drained by a wedged process (see ``docs/rca-node-drain-2026-05-27.md``).

Imports from ``imas_ambix.data.stream_encode`` when available (the ambix
package may not be installed in the magvit2 venv); falls back to inlined
copies of the helpers it needs.
"""

from __future__ import annotations

import json
import signal
import sys
import threading
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Graceful-shutdown flag
# ---------------------------------------------------------------------------

STOP = threading.Event()


class StreamAborted(RuntimeError):
    """Raised to unwind the per-shot loop when STOP is set."""


def _install_signal_handlers() -> None:
    def _handler(signum, _frame):  # noqa: ANN001
        STOP.set()
        print(
            f"[bench-worker] signal {signum} received -> graceful stop",
            flush=True,
        )

    try:
        signal.signal(signal.SIGTERM, _handler)
        signal.signal(signal.SIGINT, _handler)
    except ValueError:
        print(
            "[bench-worker] could not install signal handlers (not main thread)",
            flush=True,
        )


# ---------------------------------------------------------------------------
# Helper imports — use stream_encode if available, else inline
# ---------------------------------------------------------------------------

try:
    from imas_ambix.data.stream_encode import (
        IMAGE_SIZE,
        MODEL_FORWARD_BATCH,
        REGISTRY_OFFSET,
        TOKEN_HW,
        TOKENIZER_NAME,
        VOCAB_SIZE,
        encode_batch_indices,
        frames_to_input_device,
        frames_to_rgb_uint8,
        load_model,
        load_shot_frames,
    )

    _HELPERS_FROM_STREAM_ENCODE = True
except ImportError:
    _HELPERS_FROM_STREAM_ENCODE = False

    # --- Inline copies (byte-identical to stream_encode) ---------------------

    IMAGE_SIZE = 256
    TOKEN_HW = 16
    REGISTRY_OFFSET = 4
    TOKENIZER_NAME = "frames_open_magvit2_v1"
    VOCAB_SIZE = 1 << 18
    MODEL_FORWARD_BATCH = 4

    def normalise_frames_to_uint8(frames: np.ndarray) -> np.ndarray:
        if frames.dtype == np.uint8:
            return frames
        f = frames.astype(np.float32)
        lo = float(f.min())
        hi = float(f.max())
        if hi <= lo:
            return np.zeros_like(f, dtype=np.uint8)
        return ((f - lo) * 255.0 / (hi - lo)).clip(0, 255).astype(np.uint8)

    def frames_to_rgb_uint8(frames: np.ndarray, *, presized: bool) -> np.ndarray:
        if presized:
            u8_rgb = np.asarray(frames)
            if u8_rgb.dtype != np.uint8 or u8_rgb.ndim != 4 or u8_rgb.shape[-1] != 3:
                raise ValueError(
                    f"presized frames must be (T,H,W,3) uint8, got "
                    f"shape={u8_rgb.shape} dtype={u8_rgb.dtype}"
                )
            return u8_rgb
        u8 = normalise_frames_to_uint8(frames)
        if u8.ndim == 3:
            return np.repeat(u8[..., None], 3, axis=-1)
        if u8.ndim == 4 and u8.shape[-1] == 3:
            return u8
        raise ValueError(f"frames must be (T,H,W) or (T,H,W,3), got {u8.shape}")

    def frames_to_input_device(
        frames_u8_rgb: np.ndarray,
        image_size: int,
        device: str,
        dtype,
        resize_chunk: int = 256,
    ):
        import torch
        import torch.nn.functional as F

        frames = frames_u8_rgb
        if frames.ndim == 3:
            frames = np.repeat(frames[..., None], 3, axis=-1)
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError(f"expected (T,H,W,3) or (T,H,W), got {frames.shape}")
        n = frames.shape[0]
        step = max(1, int(resize_chunk))
        outs = []
        for i in range(0, n, step):
            c = torch.from_numpy(frames[i : i + step]).to(device)
            c = c.to(dtype).div(255.0).mul(2.0).sub(1.0)
            c = c.permute(0, 3, 1, 2).contiguous(memory_format=torch.channels_last)
            c = F.interpolate(
                c, size=(image_size, image_size), mode="bilinear", align_corners=False
            )
            outs.append(c)
        return torch.cat(outs, dim=0) if len(outs) > 1 else outs[0]

    def encode_batch_indices(
        model, images, device: str, model_forward_batch: int = MODEL_FORWARD_BATCH
    ) -> np.ndarray:
        import torch

        images = images.to(device)
        n = images.shape[0]
        step = max(1, int(model_forward_batch))
        out: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, n, step):
                chunk = images[i : i + step]
                if model.use_ema:
                    with model.ema_scope():
                        _, _, idx, _ = model.encode(chunk)
                else:
                    _, _, idx, _ = model.encode(chunk)
                out.append(
                    idx.detach()
                    .cpu()
                    .numpy()
                    .astype(np.int64)
                    .reshape(chunk.shape[0], TOKEN_HW, TOKEN_HW)
                )
        return np.concatenate(out, axis=0)

    def load_shot_frames(
        shot_id: int, camera: str, l1_root: Path, max_frames: int | None = None
    ) -> np.ndarray:
        import xarray as xr

        shot_zarr = Path(l1_root) / f"{shot_id}.zarr"
        ds = xr.open_zarr(str(shot_zarr / camera))
        data_vars = list(ds.data_vars)
        if not data_vars:
            raise ValueError(
                f"no data variables in group '{camera}' of shot {shot_id}"
            )
        frames = ds[data_vars[0]].values
        if max_frames is not None:
            frames = frames[:max_frames]
        return np.asarray(frames)

    def load_model(magvit2_root: Path, device: str):
        import torch
        from omegaconf import OmegaConf

        src = Path(magvit2_root) / "src"
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))
        from src.Open_MAGVIT2.models.lfqgan import VQModel  # noqa: PLC0415

        config_path = (
            src / "configs" / "Open-MAGVIT2" / "gpu" / "imagenet_lfqgan_256_L.yaml"
        )
        ckpt_path = Path(magvit2_root) / "weights" / "imagenet_256_L.ckpt"
        config = OmegaConf.load(str(config_path))
        model = VQModel(**config.model.init_args)
        sd = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)[
            "state_dict"
        ]
        model.load_state_dict(sd, strict=False)
        model = model.eval()
        if device.startswith("cuda"):
            torch.set_float32_matmul_precision("high")
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[bench-worker] use_deterministic_algorithms note: {exc}", flush=True)
            model = model.to(
                device=device, dtype=torch.bfloat16, memory_format=torch.channels_last
            )
        else:
            model = model.to(device)
        return model


# ---------------------------------------------------------------------------
# Decode helper (bench-specific — not in stream_encode, which is encode-only)
# ---------------------------------------------------------------------------


def decode_batch(
    model,
    tokens_local_int64: np.ndarray,
    device: str,
    model_forward_batch: int,
    target_hw: tuple[int, int],
) -> np.ndarray:
    """Decode local token ids → (T, H, W, 3) uint8 at target resolution.

    Parameters
    ----------
    tokens_local_int64:
        Shape ``(T, 16, 16)`` int64 local codebook ids (WITHOUT REGISTRY_OFFSET).
    device:
        ``"cuda"`` or ``"cpu"``.
    model_forward_batch:
        Frames per model.decode() forward; matches encode chunk for consistency.
    target_hw:
        ``(H, W)`` to resize decoded frames to (the shot's native resolution).

    Returns
    -------
    np.ndarray
        ``(T, H, W, 3)`` uint8.
    """
    import torch
    import torch.nn.functional as F

    target_dtype = next(model.decoder.parameters()).dtype  # bf16 on cuda, fp32 on cpu
    embed_dim = int(model.quantize.codebook_dim)
    T, h, w = tokens_local_int64.shape
    out: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, T, model_forward_batch):
            chunk = tokens_local_int64[i : i + model_forward_batch]  # (B, h, w)
            B = chunk.shape[0]
            idx = torch.from_numpy(chunk).to(device).reshape(B, h * w)
            bhwc = (B, h, w, embed_dim)
            if model.use_ema:
                with model.ema_scope():
                    quant = model.quantize.get_codebook_entry(idx, bhwc=bhwc, order="pre")
                    quant = quant.to(target_dtype)
                    recon = model.decode(quant)  # (B, 3, 256, 256) in roughly [-1, 1]
            else:
                quant = model.quantize.get_codebook_entry(idx, bhwc=bhwc, order="pre")
                quant = quant.to(target_dtype)
                recon = model.decode(quant)
            # clamp [-1,1] → [0,255] → resize to target_hw → uint8 (B, H, W, 3)
            t = recon.float().clamp(-1, 1).add(1.0).mul(127.5)
            t = F.interpolate(t, size=target_hw, mode="bilinear", align_corners=False)
            t = t.clamp(0, 255).round().to(torch.uint8)
            t = t.permute(0, 2, 3, 1).contiguous().cpu().numpy()  # (B, H, W, 3) uint8
            out.append(t)
    return np.concatenate(out, axis=0)


# ---------------------------------------------------------------------------
# Per-shot watchdog helpers
# ---------------------------------------------------------------------------


def _running_median(samples: list[float]) -> float:
    """Median of a list of floats; 0.0 if empty."""
    if not samples:
        return 0.0
    s = sorted(samples)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


# ---------------------------------------------------------------------------
# Main worker loop
# ---------------------------------------------------------------------------


def run_worker(manifest: dict, device: str, model_forward_batch: int, batch_timeout_s: float) -> dict:
    """Load model once, encode + decode every shot in *manifest*.

    Returns a summary dict suitable for writing to ``--report``.
    """
    import torch

    shots: list[int] = manifest["shots"]
    camera: str = manifest["camera"]
    l1_root = Path(manifest["l1_root"])
    magvit2_root = Path(manifest["magvit2_root"])
    max_items: int | None = manifest.get("max_items_per_shot")
    output_dir = Path(manifest["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    _input_dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32

    # Per-shot timing samples for watchdog auto-calibration.
    _timing_samples: list[float] = []

    def _shot_timeout() -> float:
        if batch_timeout_s > 0:
            return float(batch_timeout_s)
        if len(_timing_samples) >= 3:
            med = _running_median(_timing_samples)
            return max(60.0, 8.0 * med)
        return 60.0

    STOP.clear()
    _install_signal_handlers()

    t0_total = time.monotonic()

    # Watchdog: arms a per-shot deadline; fires STOP if deadline passes.
    _wd_deadline: dict[str, float | None] = {"t": None}
    _wd_budget: dict[str, float] = {"s": 0.0}
    _wd_lock = threading.Lock()
    _wd_done = threading.Event()

    def _watchdog() -> None:
        while not _wd_done.is_set():
            with _wd_lock:
                deadline = _wd_deadline["t"]
                budget = _wd_budget["s"]
            if deadline is not None and time.monotonic() >= deadline:
                STOP.set()
                print(
                    f"[bench-worker] per-shot watchdog FIRED (shot exceeded "
                    f"{budget:.0f}s budget) -> graceful stop",
                    flush=True,
                )
                return
            _wd_done.wait(0.05)

    _wd_thread = threading.Thread(target=_watchdog, daemon=True)
    _wd_thread.start()

    model = None
    shots_ok = 0
    shots_fail = 0
    errors: list[dict] = []

    try:
        model = load_model(magvit2_root, device)
        print(
            f"[bench-worker] model loaded in {time.monotonic() - t0_total:.1f}s "
            f"device={device} shots={len(shots)} camera={camera}",
            flush=True,
        )

        for shot_id in shots:
            if STOP.is_set():
                raise StreamAborted("STOP set before shot")

            t_shot_start = time.monotonic()
            budget = _shot_timeout()
            with _wd_lock:
                _wd_budget["s"] = budget
                _wd_deadline["t"] = t_shot_start + budget

            try:
                raw = load_shot_frames(shot_id, camera, l1_root, max_items)
                u8_rgb_native = frames_to_rgb_uint8(raw, presized=False)  # (T,H,W,3) uint8
                H, W = int(u8_rgb_native.shape[1]), int(u8_rgb_native.shape[2])

                # Encode
                t_enc0 = time.monotonic()
                images_256 = frames_to_input_device(
                    u8_rgb_native, IMAGE_SIZE, device, _input_dtype
                )  # (T, 3, 256, 256)
                tokens_local = encode_batch_indices(
                    model, images_256, device, model_forward_batch
                )  # (T, 16, 16) int64
                encode_seconds = time.monotonic() - t_enc0

                # Decode
                t_dec0 = time.monotonic()
                decoded = decode_batch(
                    model, tokens_local, device, model_forward_batch, (H, W)
                )  # (T, H, W, 3) uint8
                decode_seconds = time.monotonic() - t_dec0

                # Global tokens (with REGISTRY_OFFSET, cast to int32 for corpus compat)
                tokens_global = (tokens_local.astype(np.int64) + REGISTRY_OFFSET).astype(
                    np.int32
                )

                # Save outputs
                np.save(str(output_dir / f"{shot_id}-tokens.npy"), tokens_global)
                np.save(str(output_dir / f"{shot_id}-decoded.npy"), decoded)
                np.save(str(output_dir / f"{shot_id}-src.npy"), u8_rgb_native)

                n_items = int(tokens_global.shape[0])
                shot_time = time.monotonic() - t_shot_start
                _timing_samples.append(shot_time)
                if len(_timing_samples) > 64:
                    del _timing_samples[0]

                shots_ok += 1
                line = {
                    "shot_id": shot_id,
                    "n_items": n_items,
                    "encode_seconds": round(encode_seconds, 3),
                    "decode_seconds": round(decode_seconds, 3),
                    "native_hw": [H, W],
                    "error": None,
                }
                print(json.dumps(line), flush=True)

            except StreamAborted:
                raise
            except Exception as exc:  # noqa: BLE001
                shots_fail += 1
                err_msg = str(exc)
                errors.append({"shot_id": shot_id, "error": err_msg})
                line = {
                    "shot_id": shot_id,
                    "n_items": 0,
                    "encode_seconds": 0.0,
                    "decode_seconds": 0.0,
                    "native_hw": None,
                    "error": err_msg,
                }
                print(json.dumps(line), flush=True)
            finally:
                with _wd_lock:
                    _wd_deadline["t"] = None

            if STOP.is_set():
                raise StreamAborted("STOP set after shot")

    except StreamAborted:
        aborted = True
    else:
        aborted = False
    finally:
        _wd_done.set()
        with _wd_lock:
            _wd_deadline["t"] = None
        _wd_thread.join(timeout=1.0)
        if model is not None:
            try:
                import torch as _torch

                del model
                if device.startswith("cuda"):
                    _torch.cuda.empty_cache()
            except Exception as exc:  # noqa: BLE001
                print(f"[bench-worker] model release warning: {exc}", flush=True)

    elapsed_s = time.monotonic() - t0_total
    return {
        "shots_ok": shots_ok,
        "shots_fail": shots_fail,
        "elapsed_s": round(elapsed_s, 2),
        "aborted": aborted,
        "errors": errors[:50],
        "tokenizer_name": TOKENIZER_NAME,
        "registry_offset": REGISTRY_OFFSET,
        "model_forward_batch": model_forward_batch,
        "device": device,
    }


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="In-process bench worker: encode + decode shots, hold model in memory"
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to JSON manifest: {shots, camera, l1_root, magvit2_root, max_items_per_shot, output_dir}",
    )
    parser.add_argument("--device", default="cuda", help="'cuda' or 'cpu'")
    parser.add_argument(
        "--model-forward-batch",
        type=int,
        default=MODEL_FORWARD_BATCH,
        help=f"frames per model.encode/decode forward (default {MODEL_FORWARD_BATCH} — byte-identity with corpus)",
    )
    parser.add_argument(
        "--batch-timeout-s",
        type=float,
        default=0.0,
        help="per-shot watchdog timeout in seconds; 0 = auto (max(60, 8x running-median))",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Write final summary JSON to this path",
    )
    args = parser.parse_args(argv)

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
