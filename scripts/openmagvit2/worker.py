"""Open-MAGVIT2 encode/decode worker — runs inside the isolated venv.

Bridge between imas-ambix (which can't co-install torch 2.1.1 + lightning
2.2.0 with its own torch 2.6+ stack) and the Open-MAGVIT2 model. Called
as a subprocess from
:class:`imas_ambix.tokenizer.frames.OpenMagvit2Tokenizer`.

Wire protocol (file-based to keep the IPC trivial):

- encode mode:
    worker.py encode --input frames.npy --output tokens.npy
      input  : numpy uint8 array, shape (T, H, W, 3), values in [0, 255]
      output : numpy int64 array, shape (T, h, w) where h = H/16, w = W/16
- decode mode:
    worker.py decode --input tokens.npy --output frames.npy
      input  : numpy int64 array, shape (T, h, w)
      output : numpy uint8 array, shape (T, H, W, 3), values in [0, 255]

Both modes accept ``--image-size`` (default 256) and ``--device``
(default ``cpu``; pass ``cuda`` on the GPU node).

The model and config locations are baked in:

- config : ``src/configs/Open-MAGVIT2/gpu/imagenet_lfqgan_256_L.yaml``
- ckpt   : ``weights/imagenet_256_L.ckpt``

Override via ``--config`` and ``--ckpt`` if a different checkpoint is
ever desired.
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "src"

# So `from src.Open_MAGVIT2....` imports resolve.
sys.path.insert(0, str(SRC))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from src.Open_MAGVIT2.models.lfqgan import VQModel  # noqa: E402

DEFAULT_CONFIG = (
    SRC / "configs" / "Open-MAGVIT2" / "gpu" / "imagenet_lfqgan_256_L.yaml"
)
DEFAULT_CKPT = HERE / "weights" / "imagenet_256_L.ckpt"


def load_model(config_path: Path, ckpt_path: Path, device: str) -> VQModel:
    """Build the VQModel and load the trained weights."""
    config = OmegaConf.load(str(config_path))
    model = VQModel(**config.model.init_args)
    sd = torch.load(str(ckpt_path), map_location="cpu")["state_dict"]
    model.load_state_dict(sd, strict=False)
    return model.eval().to(device)


def _frames_to_input(frames: np.ndarray, image_size: int) -> torch.Tensor:
    """Map (T, H, W, 3) uint8 frames → (T, 3, S, S) float in [-1, 1].

    Replicates a single-channel input across RGB if needed; resizes via
    bilinear interp to ``image_size`` square. Resizing happens on CPU
    via torch.nn.functional.interpolate to avoid bringing in PIL.
    """
    import torch.nn.functional as F

    if frames.ndim == 3:  # (T, H, W) single-channel
        frames = np.repeat(frames[..., None], 3, axis=-1)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"expected (T,H,W,3) or (T,H,W), got {frames.shape}")

    t = torch.from_numpy(frames).float().div(255.0).mul(2.0).sub(1.0)
    t = t.permute(0, 3, 1, 2).contiguous()  # (T, 3, H, W)
    t = F.interpolate(
        t, size=(image_size, image_size), mode="bilinear", align_corners=False
    )
    return t


def _output_to_frames(t: torch.Tensor, target_hw: tuple[int, int]) -> np.ndarray:
    """Map (T, 3, S, S) decoded output in [-1, 1] → (T, H, W, 3) uint8."""
    import torch.nn.functional as F

    t = torch.clamp(t, -1.0, 1.0)
    t = F.interpolate(t, size=target_hw, mode="bilinear", align_corners=False)
    t = (t + 1.0) / 2.0
    t = (t * 255.0).clamp(0, 255).to(torch.uint8)
    return t.permute(0, 2, 3, 1).contiguous().cpu().numpy()


def encode(args: argparse.Namespace) -> None:
    frames = np.load(args.input)
    if frames.dtype != np.uint8:
        raise ValueError(f"expected uint8 frames, got {frames.dtype}")

    device = args.device
    model = load_model(Path(args.config), Path(args.ckpt), device)
    images = _frames_to_input(frames, args.image_size).to(device)

    indices_per_batch: list[np.ndarray] = []
    batch = max(1, args.batch_size)
    n = images.shape[0]
    h_tok = args.image_size // 16  # 8× × 2 stride at LFQGAN = 16× spatial
    w_tok = h_tok
    with torch.no_grad():
        for i in range(0, n, batch):
            chunk = images[i : i + batch]
            if model.use_ema:
                with model.ema_scope():
                    _, _, idx, _ = model.encode(chunk)
            else:
                _, _, idx, _ = model.encode(chunk)
            idx_np = idx.detach().cpu().numpy().astype(np.int64)
            # Restore (B, h, w) shape — model returns flat indices.
            idx_np = idx_np.reshape(chunk.shape[0], h_tok, w_tok)
            indices_per_batch.append(idx_np)
    all_idx = np.concatenate(indices_per_batch, axis=0)
    np.save(args.output, all_idx)
    print(
        f"encoded {n} frames -> {all_idx.shape} int64 saved to {args.output}",
        file=sys.stderr,
    )


def decode(args: argparse.Namespace) -> None:
    tokens = np.load(args.input)
    if tokens.ndim != 3:
        raise ValueError(f"expected (T, h, w) tokens, got shape {tokens.shape}")

    target_h, target_w = (
        tuple(int(x) for x in args.target_hw.split(",")) if args.target_hw else (
            args.image_size,
            args.image_size,
        )
    )

    device = args.device
    model = load_model(Path(args.config), Path(args.ckpt), device)
    t, h, w = tokens.shape
    embed_dim = int(model.quantize.codebook_dim)
    batch = max(1, args.batch_size)
    recovered_per_batch: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, t, batch):
            chunk = tokens[i : i + batch]  # (B, h, w)
            b = chunk.shape[0]
            idx = torch.from_numpy(chunk).to(device).reshape(b, h * w)
            bhwc = (b, h, w, embed_dim)
            if model.use_ema:
                with model.ema_scope():
                    quant = model.quantize.get_codebook_entry(
                        idx, bhwc=bhwc, order="pre"
                    )
                    recon = model.decode(quant)
            else:
                quant = model.quantize.get_codebook_entry(
                    idx, bhwc=bhwc, order="pre"
                )
                recon = model.decode(quant)
            recovered_per_batch.append(
                _output_to_frames(recon, (target_h, target_w))
            )
    out = np.concatenate(recovered_per_batch, axis=0)
    np.save(args.output, out)
    print(
        f"decoded {t} token frames -> {out.shape} uint8 saved to {args.output}",
        file=sys.stderr,
    )


def daemon(args: argparse.Namespace) -> None:
    """Persistent server mode — load model once, stream requests via stdin.

    Wire protocol (line-delimited JSON):

    parent → child  one JSON object per line:
        {"op": "encode"|"decode", "input": <path>, "output": <path>,
         "target_hw": "H,W"|null}
        {"op": "shutdown"}

    child  → parent one JSON object per line:
        first line on startup:  {"ready": true, "device": "...", "torch": "..."}
        per request reply:      {"ok": true,  "output": <path>, "shape": [...]}
                            or  {"ok": false, "error": <str>}

    On EOF or {"op":"shutdown"} the daemon exits 0.
    """
    import json

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print(json.dumps({
            "ready": False,
            "error": f"cuda requested but torch.cuda.is_available()=False on {socket.gethostname()!r}",
        }), flush=True)
        sys.exit(1)

    model = load_model(Path(args.config), Path(args.ckpt), device)
    print(json.dumps({
        "ready": True,
        "device": device,
        "torch": torch.__version__,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
    }), flush=True)

    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            req = json.loads(raw)
        except Exception as exc:  # noqa: BLE001
            print(json.dumps({"ok": False, "error": f"bad json: {exc}"}), flush=True)
            continue
        op = req.get("op")
        if op == "shutdown":
            print(json.dumps({"ok": True, "bye": True}), flush=True)
            return
        try:
            if op == "encode":
                frames = np.load(req["input"])
                if frames.dtype != np.uint8:
                    raise ValueError(f"expected uint8 frames, got {frames.dtype}")
                images = _frames_to_input(frames, args.image_size).to(device)
                indices_per_batch: list[np.ndarray] = []
                batch = max(1, args.batch_size)
                n = images.shape[0]
                h_tok = args.image_size // 16
                w_tok = h_tok
                with torch.no_grad():
                    for i in range(0, n, batch):
                        chunk = images[i : i + batch]
                        if model.use_ema:
                            with model.ema_scope():
                                _, _, idx, _ = model.encode(chunk)
                        else:
                            _, _, idx, _ = model.encode(chunk)
                        idx_np = idx.detach().cpu().numpy().astype(np.int64)
                        idx_np = idx_np.reshape(chunk.shape[0], h_tok, w_tok)
                        indices_per_batch.append(idx_np)
                all_idx = np.concatenate(indices_per_batch, axis=0)
                np.save(req["output"], all_idx)
                print(json.dumps({
                    "ok": True, "output": req["output"], "shape": list(all_idx.shape),
                }), flush=True)
            elif op == "decode":
                tokens = np.load(req["input"])
                if tokens.ndim != 3:
                    raise ValueError(f"expected (T,h,w) tokens, got {tokens.shape}")
                target_hw_str = req.get("target_hw") or f"{args.image_size},{args.image_size}"
                target_h, target_w = (int(x) for x in target_hw_str.split(","))
                t, h, w = tokens.shape
                embed_dim = int(model.quantize.codebook_dim)
                batch = max(1, args.batch_size)
                recovered: list[np.ndarray] = []
                with torch.no_grad():
                    for i in range(0, t, batch):
                        chunk = tokens[i : i + batch]
                        b = chunk.shape[0]
                        idx_t = torch.from_numpy(chunk).to(device).reshape(b, h * w)
                        bhwc = (b, h, w, embed_dim)
                        if model.use_ema:
                            with model.ema_scope():
                                quant = model.quantize.get_codebook_entry(idx_t, bhwc=bhwc, order="pre")
                                recon = model.decode(quant)
                        else:
                            quant = model.quantize.get_codebook_entry(idx_t, bhwc=bhwc, order="pre")
                            recon = model.decode(quant)
                        recovered.append(_output_to_frames(recon, (target_h, target_w)))
                out = np.concatenate(recovered, axis=0)
                np.save(req["output"], out)
                print(json.dumps({
                    "ok": True, "output": req["output"], "shape": list(out.shape),
                }), flush=True)
            else:
                print(json.dumps({"ok": False, "error": f"unknown op {op!r}"}), flush=True)
        except Exception as exc:  # noqa: BLE001
            print(json.dumps({"ok": False, "error": str(exc)}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Open-MAGVIT2 worker")
    parser.add_argument(
        "mode", choices=("encode", "decode", "daemon"),
    )
    parser.add_argument("--input", required=False, default=None)
    parser.add_argument("--output", required=False, default=None)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--target-hw",
        default=None,
        help=(
            "Comma-separated 'H,W' to resize decoded frames back to (decode mode)"
        ),
    )
    args = parser.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # Validate device and resolve batch size default.
    if args.device == "cuda" and not torch.cuda.is_available():
        host = socket.gethostname()
        print(
            f"open-magvit2 worker: ERROR: --device cuda requested but "
            f"torch.cuda.is_available() is False on {host!r}. "
            "Submit via SLURM to the betelgeuse GPU partition.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.batch_size is None:
        args.batch_size = 8 if args.device == "cuda" else 4

    print(
        f"open-magvit2 worker: device={args.device}, "
        f"torch={torch.__version__}, "
        f"cuda_available={torch.cuda.is_available()}",
        file=sys.stderr,
    )

    if args.mode == "encode":
        if not args.input or not args.output:
            parser.error("encode mode requires --input and --output")
        encode(args)
    elif args.mode == "decode":
        if not args.input or not args.output:
            parser.error("decode mode requires --input and --output")
        decode(args)
    else:  # daemon
        daemon(args)


if __name__ == "__main__":
    main()
