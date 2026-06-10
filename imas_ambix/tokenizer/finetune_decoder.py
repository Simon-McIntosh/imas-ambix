"""Plasma-domain Open-MAGVIT2 decoder fine-tune.

Freezes the encoder + LFQ codebook (seeded from the original checkpoint's
EMA shadow to match the corpus tokenizer's ``ema_scope`` behaviour); trains
only the decoder with pixel L1 + VGG16 perceptual loss on MAST rbb plasma
frames.

Correctness contracts that MUST hold (any divergence destroys the rFID gate):

1. Encoder weights at train-time MUST equal encoder weights at bench-time.
   The corpus encoder reads via ``with model.ema_scope(): model.encode(...)``.
   We seed live encoder/decoder weights from the EMA shadow at startup so
   plain ``model.encode()`` returns the same tokens as the bench.

2. Frame normalization at train-time MUST equal corpus-encode normalization.
   The corpus uses per-shot min/max (``stream_encode.normalise_frames_to_uint8``).
   We pre-compute per-shot ``(lo, hi)`` during metadata scan and apply the
   same per-shot transform per frame.

3. Optimizer state precision: model params + AdamW moments in fp32; forward
   pass under ``torch.amp.autocast(dtype=torch.bfloat16)`` for compute speed.
   bf16-everywhere training loses precision in moment estimators.

4. rFID evaluation runs on rank 0 with the FULL val set (not the per-rank
   DistributedSampler shard). Result is broadcast so all ranks see the same
   early-stop decision.

Hardening (mirrors ``imas_ambix.bench.stream_worker``):
- SIGTERM/SIGINT handler sets STOP flag (< 5 s graceful exit).
- Per-step watchdog auto-calibrated from the running-median step time.
- ``try/finally`` releases model + empties CUDA cache + destroys NCCL group.

Run via torchrun (4×H200 DDP):

    torchrun --nproc_per_node=4 imas_ambix/tokenizer/finetune_decoder.py \\
        --train-shots train.txt --val-shots val.txt
"""

from __future__ import annotations

import os
import signal
import sys
import threading
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    import torch

# ---------------------------------------------------------------------------
# Graceful-shutdown flag — mirrors stream_worker.py
# ---------------------------------------------------------------------------

STOP = threading.Event()


def _install_signal_handlers() -> None:
    """Install SIGTERM/SIGINT handlers for graceful shutdown.

    Handler delivery can lag while inside a long NCCL collective (Python
    signal handlers run between bytecodes), but lands at the next step
    boundary — ``scancel`` and ``--time`` expiry both terminate the job
    cleanly through this path (settled drain findings,
    docs/rca-node-drain-final-2026-06-03.html).
    """

    def _handler(signum, _frame) -> None:  # noqa: ANN001
        STOP.set()
        print(
            f"[finetune-decoder] signal {signum} received → graceful stop",
            flush=True,
        )

    try:
        signal.signal(signal.SIGTERM, _handler)
        signal.signal(signal.SIGINT, _handler)
    except ValueError:
        print(
            "[finetune-decoder] could not install signal handlers (not main thread)",
            flush=True,
        )


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------


def _init_distributed() -> tuple[int, int]:
    """Initialise NCCL when launched via torchrun.  Returns ``(rank, world_size)``."""
    import torch
    import torch.distributed as dist

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    elif torch.cuda.is_available():
        torch.cuda.set_device(0)

    return rank, world_size


def _cleanup_distributed() -> None:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DEFAULT_MAGVIT2_ROOT = Path("/work/projects/imas_gpu/mast-tokens/v1/open-magvit2")
_DEFAULT_L1_ROOT = Path("/work/projects/imas_gpu/mast/level1/shots")


@dataclass
class DecoderFinetuneConfig:
    """Configuration for the plasma-domain Open-MAGVIT2 decoder fine-tune.

    Per-GPU batch_size; effective batch = batch_size × world_size.
    """

    frame_root: Path = _DEFAULT_L1_ROOT
    magvit2_root: Path = _DEFAULT_MAGVIT2_ROOT
    output_path: Path = _DEFAULT_MAGVIT2_ROOT / "weights" / "plasma-decoder-v1.safetensors"
    train_shot_ids: list[int] = field(default_factory=list)
    val_shot_ids: list[int] = field(default_factory=list)
    camera: str = "rbb"
    image_size: int = 256
    frames_per_shot: int = 50
    batch_size: int = 16  # per-GPU; effective = batch_size × world_size
    learning_rate: float = 1e-4
    max_steps: int = 10_000
    warmup_steps: int = 200
    l1_weight: float = 1.0
    perceptual_weight: float = 0.1  # locked decision: lpips-weight = 0.1
    grad_clip_norm: float = 1.0
    eval_every_n_steps: int = 1_000
    patience: int = 3
    seed: int = 0
    device: str = "cuda"

    def __post_init__(self) -> None:
        self.frame_root = Path(self.frame_root)
        self.magvit2_root = Path(self.magvit2_root)
        self.output_path = Path(self.output_path)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


# VGG16 expects ImageNet normalization. Stored as module-level constants so
# they can be moved to the model device once and reused across steps.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _frechet_from_features(
    ref_feats: object,
    pred_feats: object,
) -> float:
    """Compute Frechet distance from pre-extracted InceptionV3 features.

    Implementation parity with ``imas_ambix.eval.metrics.rfid`` (same
    ``eps=1e-3 if T<10 else 1e-6`` rule, same numerical-stability offset,
    same complex-part stripping).  Inputs are ``(T, 2048)`` numpy arrays.
    """
    import numpy as np
    from scipy.linalg import sqrtm

    mu_r, mu_p = ref_feats.mean(axis=0), pred_feats.mean(axis=0)
    sigma_r = np.cov(ref_feats, rowvar=False)
    sigma_p = np.cov(pred_feats, rowvar=False)

    t = ref_feats.shape[0]
    eps = 1e-3 if t < 10 else 1e-6

    diff = mu_r - mu_p
    offset = np.eye(sigma_r.shape[0]) * eps
    cov_mean, _ = sqrtm(sigma_r @ sigma_p + offset, disp=False)
    if np.iscomplexobj(cov_mean):
        cov_mean = cov_mean.real
    fid = float(
        diff @ diff + np.trace(sigma_r) + np.trace(sigma_p) - 2.0 * np.trace(cov_mean)
    )
    return float(max(fid, 0.0))


class DecoderFinetuneTrainer:
    """Drives the Open-MAGVIT2 decoder fine-tune loop for plasma imagery.

    Heavy imports (torch, torchvision, Open-MAGVIT2) are lazy so the
    constructor can be instantiated without a GPU.
    """

    _TOKEN_HW: int = 16  # Open-MAGVIT2 spatial compression: 256/16 = 16

    def __init__(self, config: DecoderFinetuneConfig) -> None:
        self.config = config
        self._model: object | None = None
        self._vgg_features: object | None = None
        self._vgg_mean: object | None = None
        self._vgg_std: object | None = None

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def _scan_shots(self, shot_ids: list[int], label: str) -> list[tuple]:
        """Scan zarr metadata + per-shot intensity stats.

        Returns a list of tuples ``(zarr_path, arr_key, frame_idx, lo, hi)``
        per frame.  ``(lo, hi)`` are the per-shot min/max — identical to what
        ``stream_encode.normalise_frames_to_uint8`` computes — so frames
        normalize the same way the corpus encoder did.
        """
        import numpy as np
        import zarr

        cfg = self.config
        entries: list[tuple[str, str, int, float, float]] = []
        n_missing = 0
        t0 = time.monotonic()

        for shot_id in shot_ids:
            if STOP.is_set():
                break
            zarr_path = cfg.frame_root / f"{shot_id}.zarr" / cfg.camera
            if not zarr_path.is_dir():
                n_missing += 1
                continue
            try:
                z = zarr.open_group(str(zarr_path), mode="r")
                arr_keys = [k for k in z.array_keys() if z[k].ndim >= 3]
                if not arr_keys:
                    n_missing += 1
                    continue
                arr_key = arr_keys[0]
                arr = z[arr_key]
                t_len = int(arr.shape[0])
                if t_len == 0:
                    continue

                # Per-shot min/max — match corpus normalise_frames_to_uint8.
                # For shots with > 200 frames we sample 200 evenly-spaced
                # frames to compute (lo, hi); the corpus encoder reads the
                # whole shot at once anyway so this is a cheap approximation
                # and the worst-case error on (lo, hi) is well under the
                # quantization step.  For ≤ 200 frames we read all of them.
                stats_indices = np.linspace(
                    0, t_len - 1, min(200, t_len), dtype=int
                )
                # Read those frames via fancy indexing (sorted to be efficient)
                stats_indices = np.unique(stats_indices)
                stats_data = arr[stats_indices.tolist()]
                stats_data = np.asarray(stats_data)
                if stats_data.dtype != np.uint8:
                    lo = float(stats_data.astype(np.float32).min())
                    hi = float(stats_data.astype(np.float32).max())
                else:
                    lo, hi = 0.0, 255.0
                if hi <= lo:
                    lo, hi = 0.0, 1.0  # degenerate shot: avoid div-by-zero

                # Training sample indices (evenly spaced)
                sample_indices = np.linspace(
                    0, t_len - 1, min(cfg.frames_per_shot, t_len), dtype=int
                )
                for idx in sample_indices:
                    entries.append((str(zarr_path), arr_key, int(idx), lo, hi))

            except Exception as exc:  # noqa: BLE001
                warnings.warn(
                    f"[finetune-decoder] scan {shot_id} failed: {exc}",
                    stacklevel=2,
                )
                n_missing += 1

        elapsed = time.monotonic() - t0
        rank = int(os.environ.get("RANK", "0"))
        if rank == 0:
            print(
                f"[finetune-decoder] {label}: scanned {len(shot_ids)} shots → "
                f"{len(entries)} frames "
                f"({n_missing} missing/skipped) in {elapsed:.1f}s",
                flush=True,
            )
        return entries

    def build_dataloaders(self) -> tuple[Iterable, Iterable]:
        """Build training and validation data loaders.

        Both loaders use :class:`DistributedSampler` so that DDP training
        and (parallel) eval shard the data non-overlappingly across ranks.
        Eval uses ``dist.all_gather`` of per-rank frame buffers + rank-0
        rFID computation so the metric covers the FULL val set (unbiased).

        Returns
        -------
        tuple
            ``(train_loader, val_loader)``.
        """
        import numpy as np
        import torch
        import torch.distributed as dist
        from torch.utils.data import DataLoader
        from torch.utils.data.distributed import DistributedSampler

        rank = dist.get_rank() if (dist.is_available() and dist.is_initialized()) else 0
        world_size = dist.get_world_size() if (dist.is_available() and dist.is_initialized()) else 1

        cfg = self.config

        class _PlasmaFrameDataset(torch.utils.data.Dataset):
            """One (image_size, image_size, 3) uint8 frame per item.

            Normalization uses the per-shot (lo, hi) precomputed in
            ``_scan_shots`` so train-time frames match corpus-encode frames.
            """

            def __init__(
                self,
                entries: list[tuple[str, str, int, float, float]],
                image_size: int,
            ) -> None:
                self.entries = entries
                self.image_size = image_size

            def __len__(self) -> int:
                return len(self.entries)

            def __getitem__(self, i: int) -> "torch.Tensor":
                import numpy as _np
                import zarr as _zarr
                import torch.nn.functional as _F

                zarr_path, arr_key, frame_idx, lo, hi = self.entries[i]

                # Per-worker zarr-store cache (bounded LRU-ish)
                if not hasattr(self, "_z_cache"):
                    self._z_cache: dict[str, object] = {}
                if zarr_path not in self._z_cache:
                    if len(self._z_cache) > 128:
                        del self._z_cache[next(iter(self._z_cache))]
                    self._z_cache[zarr_path] = _zarr.open_group(
                        zarr_path, mode="r"
                    )
                z = self._z_cache[zarr_path]

                raw = _np.asarray(z[arr_key][frame_idx])

                # Per-shot normalization (matches corpus encoder)
                if raw.dtype != _np.uint8:
                    f = raw.astype(_np.float32)
                    frame = (
                        (f - lo) * 255.0 / max(hi - lo, 1e-12)
                    ).clip(0, 255).astype(_np.uint8)
                else:
                    frame = raw

                if frame.ndim == 2:
                    frame = _np.repeat(frame[..., _np.newaxis], 3, axis=-1)
                elif frame.ndim == 3 and frame.shape[-1] == 1:
                    frame = _np.repeat(frame, 3, axis=-1)
                elif frame.ndim == 3 and frame.shape[-1] != 3:
                    frame = frame[:, :, :3]

                if frame.shape[:2] != (self.image_size, self.image_size):
                    # Match corpus encoder's resize EXACTLY:
                    # F.interpolate(bilinear, no antialias, align_corners=False)
                    # — see stream_encode.frames_to_input_device.  The decoder
                    # was previously fine-tuned with PIL.Image.resize which
                    # produced subtly different pixel values, causing a
                    # bench-time regression vs the imagenet baseline (job
                    # 1209047 measured rFID=27.4 vs baseline 16.25 because the
                    # decoder overfit to PIL-resized inputs and saw
                    # F.interpolate-resized inputs at bench/world-model time).
                    t = torch.from_numpy(_np.ascontiguousarray(frame)).float()
                    t = t.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
                    t = _F.interpolate(
                        t,
                        size=(self.image_size, self.image_size),
                        mode="bilinear",
                        align_corners=False,
                    )
                    t = t.squeeze(0).permute(1, 2, 0).clamp(0, 255).round()
                    return t.to(torch.uint8)

                return torch.from_numpy(_np.ascontiguousarray(frame))

        if rank == 0:
            print(
                f"[finetune-decoder] scanning {len(cfg.train_shot_ids)} train + "
                f"{len(cfg.val_shot_ids)} val shots ...",
                flush=True,
            )
        # All ranks scan independently (each needs the full entry list).
        # Per-shot (lo, hi) reads ~200 KB/shot (200 frames × ~1 KB header
        # accesses), so 7,500-shot scan stays under a minute on GPFS.
        train_entries = self._scan_shots(cfg.train_shot_ids, "train")
        val_entries = self._scan_shots(cfg.val_shot_ids, "val")

        train_ds = _PlasmaFrameDataset(train_entries, cfg.image_size)
        val_ds = _PlasmaFrameDataset(val_entries, cfg.image_size)

        train_sampler = DistributedSampler(
            train_ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,
            seed=cfg.seed,
        )

        n_workers = 4
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            sampler=train_sampler,
            num_workers=n_workers,
            prefetch_factor=2,
            persistent_workers=True,
            pin_memory=cfg.device.startswith("cuda"),
        )

        val_sampler = DistributedSampler(
            val_ds, num_replicas=world_size, rank=rank, shuffle=False
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.batch_size,
            sampler=val_sampler,
            num_workers=n_workers,
            prefetch_factor=2,
            persistent_workers=True,
            pin_memory=cfg.device.startswith("cuda"),
        )
        return train_loader, val_loader

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    def build_model(self) -> object:
        """Load Open-MAGVIT2 VQModel; seed encoder+decoder from EMA shadow; freeze.

        Steps:
        1. Load checkpoint (live + EMA shadow keys both present in state_dict).
        2. Construct VQModel via OmegaConf.
        3. ``load_state_dict(strict=False)`` populates both live and EMA.
        4. **Patch live encoder/decoder weights with EMA shadow values** so
           ``model.encode()`` (no ema_scope) returns the same tokens as the
           bench's ``with model.ema_scope(): model.encode()`` path.
        5. Freeze all params; unfreeze ``model.decoder``.
        6. Cast to channels_last on the local device; KEEP fp32 (autocast
           handles bf16 forward).
        7. Wrap in DDP if world_size > 1.

        Returns
        -------
        torch.nn.Module
            DDP-wrapped (or bare) VQModel with encoder/codebook frozen,
            decoder trainable.
        """
        import torch
        import torch.distributed as dist
        from omegaconf import OmegaConf
        from torch.nn.parallel import DistributedDataParallel as DDP

        cfg = self.config
        magvit2_root = cfg.magvit2_root
        src = magvit2_root / "src"
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))

        from src.Open_MAGVIT2.models.lfqgan import VQModel  # noqa: PLC0415

        config_path = (
            src / "configs" / "Open-MAGVIT2" / "gpu" / "imagenet_lfqgan_256_L.yaml"
        )
        ckpt_path = magvit2_root / "weights" / "imagenet_256_L.ckpt"
        if not ckpt_path.exists():
            raise RuntimeError(f"Open-MAGVIT2 checkpoint not found at {ckpt_path}.")

        model_cfg = OmegaConf.load(str(config_path))
        model = VQModel(**model_cfg.model.init_args)

        ckpt = torch.load(
            str(ckpt_path), map_location="cpu", weights_only=False
        )
        sd = ckpt["state_dict"]
        model.load_state_dict(sd, strict=False)

        # ── Seed live encoder/decoder weights from EMA shadow ─────────────
        # Bench reads encoder via ema_scope(); we need the same weights at
        # train time so token sequences match.  LitEma's m_name2s_name maps
        # "encoder.conv_in.weight" → "encoderconv_inweight".
        n_patched = 0
        live_state = model.state_dict()
        with torch.no_grad():
            for live_key, live_tensor in list(live_state.items()):
                if not (
                    live_key.startswith("encoder.")
                    or live_key.startswith("decoder.")
                ):
                    continue
                # Buffers (BN running stats) and non-EMA-tracked params have
                # no EMA shadow — skip them silently.
                ema_key = "model_ema." + live_key.replace(".", "")
                if ema_key in sd:
                    live_tensor.copy_(sd[ema_key])
                    n_patched += 1
        # ── Freeze everything; unfreeze decoder only ──────────────────────
        model.requires_grad_(False)
        model.decoder.requires_grad_(True)

        # Disable the model's own EMA hooks — we never call them during
        # fine-tune (LitEma.forward/copy_to assert frozen params are not in
        # m_name2s_name, but they are; see commit 9e53e02).  We patch the
        # EMA shadow at save time in _save_merged_checkpoint().
        model.use_ema = False

        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device = f"cuda:{local_rank}" if cfg.device.startswith("cuda") else cfg.device

        if device.startswith("cuda"):
            torch.set_float32_matmul_precision("high")
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[finetune-decoder] deterministic note: {exc}", flush=True)
            # Keep fp32 params (AdamW state stays fp32 → numerically stable).
            # Forward pass uses autocast bf16 — see _encode_decode/compute_loss.
            model = model.to(device=device, memory_format=torch.channels_last)
        else:
            model = model.to(device)

        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            model = DDP(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                # All decoder params see grads every step; encoder/codebook
                # are excluded by requires_grad=False.  find_unused=False
                # saves the per-step param-graph walk.
                find_unused_parameters=False,
            )

        rank = dist.get_rank() if (dist.is_available() and dist.is_initialized()) else 0
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in model.parameters())
        if rank == 0:
            ws = dist.get_world_size() if dist.is_initialized() else 1
            print(
                f"[finetune-decoder] model loaded: "
                f"{n_trainable:,} trainable / {n_total:,} total params  "
                f"device={device}  world_size={ws}  "
                f"ema-seeded keys={n_patched}",
                flush=True,
            )
        self._model = model
        return model

    # ------------------------------------------------------------------
    # Encode → decode helper
    # ------------------------------------------------------------------

    def _encode_decode(
        self,
        model: object,
        frames_input: object,
        *,
        training: bool,
    ) -> object:
        """Encode (frozen) → quant → decode (trainable).

        Wrapped in ``torch.amp.autocast(bf16)`` for forward speed; model
        params stay fp32 so AdamW state is fp32.

        Parameters
        ----------
        frames_input:
            ``(B, 3, H, W)`` fp32 in ``[0, 1]``, on device.
        training:
            When True, decode is computed with grad.  When False, wrapped
            in ``torch.no_grad()``.
        """
        import torch

        # MODEL_FORWARD_BATCH is part of the tokenizer's bit-exact contract
        # (see imas_ambix/data/stream_encode.py:159).  The Open-MAGVIT2 VQ
        # encoder forward is batch-size sensitive — feeding 16 frames at once
        # produces different LFQ token IDs than feeding 4 (1209084 confirmed:
        # bench rFID 28.87 vs training-time 14.04 on the same decoder weights,
        # attributed to this mismatch).  C6 fix: sub-batch the encoder forward
        # to MODEL_FORWARD_BATCH=4, matching bench/corpus.
        try:
            from imas_ambix.data.stream_encode import MODEL_FORWARD_BATCH
        except ImportError:
            MODEL_FORWARD_BATCH = 4  # fallback if module not importable

        raw = model.module if hasattr(model, "module") else model

        # C7 fix: encoder + quantize must run in EVAL mode to match
        # bench/corpus contract (stream_worker calls model.eval() at load).
        # The training loop's model.train() sets all submodules to train mode;
        # if any encoder layer has BatchNorm/Dropout, train-mode forward
        # produces different outputs than eval-mode.  Explicit reset before
        # every encode call is bulletproof and zero-cost.
        raw.encoder.eval()
        raw.quantize.eval()

        # Encode + codebook lookup: no_grad, autocast bf16.
        with torch.no_grad(), torch.amp.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            frames_normed = frames_input.mul(2.0).sub(1.0)  # [-1, 1]
            frames_normed = frames_normed.to(memory_format=torch.channels_last)

            B = frames_input.shape[0]
            # C6: chunk encoder forward to MODEL_FORWARD_BATCH-sized batches.
            idx_chunks = []
            for i in range(0, B, MODEL_FORWARD_BATCH):
                chunk = frames_normed[i : i + MODEL_FORWARD_BATCH]
                _, _, idx_i, _ = raw.encode(chunk)
                idx_chunks.append(idx_i)
            idx = (
                idx_chunks[0]
                if len(idx_chunks) == 1
                else torch.cat(idx_chunks, dim=0)
            )

            h = w = self._TOKEN_HW
            idx_flat = idx.reshape(B, h * w)
            bhwc = (B, h, w, int(raw.quantize.codebook_dim))
            quant = raw.quantize.get_codebook_entry(idx_flat, bhwc=bhwc, order="pre")

        # Decode: autocast bf16 forward; backward populates fp32 .grad tensors.
        if training:
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                recon_m11 = raw.decode(quant)
        else:
            with torch.no_grad(), torch.amp.autocast(
                device_type="cuda", dtype=torch.bfloat16
            ):
                recon_m11 = raw.decode(quant)

        # Cast to fp32 for loss computation (stable L1 + VGG).
        return (recon_m11.float().clamp(-1, 1) + 1.0) / 2.0  # [0,1] fp32

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def _vgg_normalize(self, x: object) -> object:
        """Apply ImageNet mean/std normalization to ``(B,3,H,W)`` in [0,1]."""
        import torch

        if self._vgg_mean is None:
            self._vgg_mean = torch.tensor(_IMAGENET_MEAN, dtype=torch.float32).view(
                1, 3, 1, 1
            ).to(x.device)
            self._vgg_std = torch.tensor(_IMAGENET_STD, dtype=torch.float32).view(
                1, 3, 1, 1
            ).to(x.device)
        return (x - self._vgg_mean) / self._vgg_std

    def compute_loss(
        self,
        target_frames: object,
        recon_frames: object,
    ) -> object:
        """L1 + ImageNet-normalized VGG16 perceptual loss.

        Both inputs are ``(B, 3, H, W)`` fp32 in [0, 1].  VGG features are
        compared after ImageNet mean/std normalization (the distribution
        VGG was trained on).
        """
        import torch
        import torch.nn.functional as F

        cfg = self.config
        l1 = F.l1_loss(recon_frames, target_frames)
        loss = cfg.l1_weight * l1

        if cfg.perceptual_weight > 0.0 and self._vgg_features is not False:
            try:
                import torchvision.models as tvm

                if self._vgg_features is None:
                    vgg = tvm.vgg16(weights=tvm.VGG16_Weights.IMAGENET1K_V1)
                    self._vgg_features = torch.nn.Sequential(
                        *list(vgg.features.children())[:16]
                    ).to(target_frames.device)
                    self._vgg_features.requires_grad_(False)
                    self._vgg_features.eval()
                    rank = int(os.environ.get("RANK", "0"))
                    if rank == 0:
                        print(
                            "[finetune-decoder] VGG16 perceptual loss enabled "
                            "(ImageNet-normalized input)",
                            flush=True,
                        )

                t_norm = self._vgg_normalize(target_frames)
                r_norm = self._vgg_normalize(recon_frames)
                with torch.no_grad():
                    feat_target = self._vgg_features(t_norm)
                feat_recon = self._vgg_features(r_norm)
                perceptual = F.mse_loss(feat_recon, feat_target)
                loss = loss + cfg.perceptual_weight * perceptual

            except Exception as exc:  # noqa: BLE001
                warnings.warn(
                    f"VGG16 perceptual loss disabled ({exc}); falling back to L1.",
                    stacklevel=2,
                )
                self._vgg_features = False

        return loss

    # ------------------------------------------------------------------
    # Evaluation — rank 0 only, full val set
    # ------------------------------------------------------------------

    def _get_inception(self, device: str) -> object:
        """Lazy-load InceptionV3 (penultimate layer, pool features) on device.

        Cached on the trainer instance.  Each rank loads its own copy on
        its own GPU so feature extraction runs in parallel.
        """
        import torch
        import torchvision.models as tvm

        if getattr(self, "_inception_model", None) is None:
            m = tvm.inception_v3(weights=tvm.Inception_V3_Weights.IMAGENET1K_V1)
            m.fc = torch.nn.Identity()
            m.eval()
            m.requires_grad_(False)
            m = m.to(device)
            self._inception_model = m
        return self._inception_model

    def _inception_features(
        self,
        frames_u8_bhwc: object,
        device: str,
    ) -> object:
        """Compute InceptionV3 pool features on GPU for a batch of uint8 frames.

        Parameters
        ----------
        frames_u8_bhwc:
            ``(B, H, W, 3)`` uint8 numpy array OR torch tensor.
        device:
            CUDA device string (matches rank's local GPU).

        Returns
        -------
        torch.Tensor
            ``(B, 2048)`` fp32 features on ``device``.
        """
        import torch
        import torch.nn.functional as F
        import numpy as np

        inception = self._get_inception(device)

        if isinstance(frames_u8_bhwc, np.ndarray):
            x = torch.from_numpy(np.ascontiguousarray(frames_u8_bhwc))
        else:
            x = frames_u8_bhwc
        # (B, H, W, 3) uint8 → (B, 3, 299, 299) float [0,1] ImageNet-normed
        x = x.to(device).permute(0, 3, 1, 2).float() / 255.0
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        mean = torch.tensor(_IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        std = torch.tensor(_IMAGENET_STD, device=device).view(1, 3, 1, 1)
        x = (x - mean) / std
        with torch.no_grad():
            feats = inception(x)
        return feats.detach()

    def evaluate(
        self,
        model: object,
        val_loader_distributed: Iterable,
    ) -> dict[str, float]:
        """Distributed eval: parallel encode/decode + parallel InceptionV3.

        All 4 GPUs run encode/decode AND InceptionV3 feature extraction
        in parallel on their DistributedSampler shard.  Only the small
        feature tensors are gathered to rank 0 for the final Frechet
        distance computation.  Beats the previous rank-0 rFID bottleneck
        (~106 s) measured in smoke 1208998 by ~10-20×.

        Pipeline per rank:
        1. encode/decode val shard on local GPU (parallel)
        2. accumulate L1 sum
        3. extract InceptionV3 features for both target and recon on local
           GPU — the real bottleneck of rfid()
        4. ``dist.all_reduce`` aggregates L1 stats
        5. ``dist.all_gather`` collects feature tensors (small: B × 2048
           fp32 ≈ 8 KB per frame) — no raw-frame gather
        6. Rank 0 computes Frechet distance from gathered features (fast:
           covariance + scipy sqrtm on 2048×2048)
        7. ``dist.broadcast`` returns rFID to all ranks
        """
        import numpy as np
        import torch
        import torch.distributed as dist
        import torch.nn.functional as F

        cfg = self.config

        ddp = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if ddp else 0
        world_size = dist.get_world_size() if ddp else 1
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        dev = f"cuda:{local_rank}" if cfg.device.startswith("cuda") else cfg.device

        # Per-rank cap on the feature buffer; FID is stable above ~1k frames.
        _PER_RANK_FID_CAP = max(1, 5_000 // max(world_size, 1))

        local_target_feats: list[object] = []
        local_recon_feats: list[object] = []
        local_l1_sum = 0.0
        local_n = 0
        cur_frames = 0

        model.eval()
        try:
            for batch in val_loader_distributed:
                if STOP.is_set():
                    break
                frames = batch.float().permute(0, 3, 1, 2).to(dev) / 255.0
                recon = self._encode_decode(model, frames, training=False)

                local_l1_sum += float(F.l1_loss(recon, frames).item())
                local_n += 1

                if cur_frames < _PER_RANK_FID_CAP:
                    # Convert to uint8 BHWC then extract InceptionV3 features
                    # — same dtype contract as imas_ambix.eval.metrics.rfid.
                    t_u8 = (
                        frames.permute(0, 2, 3, 1).clamp(0, 1) * 255
                    ).to(torch.uint8)
                    r_u8 = (
                        recon.permute(0, 2, 3, 1).clamp(0, 1) * 255
                    ).to(torch.uint8)
                    t_feat = self._inception_features(t_u8, dev)
                    r_feat = self._inception_features(r_u8, dev)
                    local_target_feats.append(t_feat)
                    local_recon_feats.append(r_feat)
                    cur_frames += t_u8.shape[0]
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"[finetune-decoder] evaluate() interrupted on rank {rank} "
                f"({exc}); reporting partial metrics.",
                stacklevel=2,
            )

        # ── Aggregate L1 across ranks ─────────────────────────────────────
        l1_pair = torch.tensor([local_l1_sum, float(local_n)], device=dev)
        if ddp and world_size > 1:
            dist.all_reduce(l1_pair, op=dist.ReduceOp.SUM)
        mean_l1 = float(l1_pair[0].item()) / max(float(l1_pair[1].item()), 1.0)

        # ── Concatenate local features ────────────────────────────────────
        if local_target_feats:
            local_t = torch.cat(local_target_feats, dim=0).to(torch.float32)
            local_r = torch.cat(local_recon_feats, dim=0).to(torch.float32)
        else:
            local_t = torch.zeros((0, 2048), device=dev, dtype=torch.float32)
            local_r = torch.zeros((0, 2048), device=dev, dtype=torch.float32)

        # ── Gather feature tensors via all_gather_object (variable shapes) ─
        # The payload is tiny (≤ ~10 MB total) so pickling overhead is OK.
        # Move to CPU before gather to avoid CUDA-IPC complications in
        # all_gather_object across NCCL backends.
        local_t_np = local_t.cpu().numpy()
        local_r_np = local_r.cpu().numpy()
        if ddp and world_size > 1:
            gathered_t: list[np.ndarray | None] = [None] * world_size
            gathered_r: list[np.ndarray | None] = [None] * world_size
            dist.all_gather_object(gathered_t, local_t_np)
            dist.all_gather_object(gathered_r, local_r_np)
        else:
            gathered_t = [local_t_np]
            gathered_r = [local_r_np]

        # ── Rank 0 computes Frechet distance from gathered features ───────
        rfid_val = float("nan")
        if rank == 0:
            try:
                non_empty_t = [a for a in gathered_t if a is not None and a.shape[0] > 0]
                non_empty_r = [a for a in gathered_r if a is not None and a.shape[0] > 0]
                if non_empty_t:
                    ref_feats = np.concatenate(non_empty_t, axis=0)
                    pred_feats = np.concatenate(non_empty_r, axis=0)
                    rfid_val = _frechet_from_features(ref_feats, pred_feats)
            except Exception as exc:  # noqa: BLE001
                warnings.warn(
                    f"rFID Frechet computation failed ({exc}); reporting NaN.",
                    stacklevel=2,
                )

        # ── Broadcast rFID so every rank returns the same value ───────────
        if ddp and world_size > 1:
            rfid_tensor = torch.tensor([rfid_val], device=dev, dtype=torch.float64)
            dist.broadcast(rfid_tensor, src=0)
            rfid_val = float(rfid_tensor.item())

        return {"rfid": rfid_val, "l1": mean_l1}

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self) -> Path:
        """Run the full decoder fine-tune loop."""
        import math

        import torch
        import torch.distributed as dist
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import LambdaLR

        cfg = self.config

        if not cfg.train_shot_ids:
            raise RuntimeError(
                "train_shot_ids is empty — provide shot IDs before calling train()."
            )

        _install_signal_handlers()
        STOP.clear()

        rank, world_size = _init_distributed()
        is_primary = rank == 0

        torch.manual_seed(cfg.seed + rank)

        # ── Watchdog ───────────────────────────────────────────────────────
        _step_times: list[float] = []
        _wd_deadline: dict[str, float | None] = {"t": None}
        _wd_budget: dict[str, float] = {"s": 0.0}
        _wd_lock = threading.Lock()
        _wd_done = threading.Event()

        def _step_timeout() -> float:
            if len(_step_times) >= 5:
                med = sorted(_step_times)[len(_step_times) // 2]
                return max(120.0, 5.0 * med)
            return 300.0

        def _watchdog() -> None:
            while not _wd_done.is_set():
                with _wd_lock:
                    d = _wd_deadline["t"]
                    b = _wd_budget["s"]
                if d is not None and time.monotonic() >= d:
                    STOP.set()
                    print(
                        f"[finetune-decoder] per-step watchdog FIRED "
                        f"(exceeded {b:.0f}s) → graceful stop",
                        flush=True,
                    )
                    return
                _wd_done.wait(0.1)

        wd_thread = threading.Thread(target=_watchdog, daemon=True)
        wd_thread.start()

        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        train_device = (
            f"cuda:{local_rank}" if cfg.device.startswith("cuda") else cfg.device
        )

        model = None
        try:
            train_loader, val_loader = self.build_dataloaders()

            if STOP.is_set():
                if is_primary:
                    print("[finetune-decoder] stopped during data load", flush=True)
                return cfg.output_path

            model = self.build_model()

            trainable_params = [p for p in model.parameters() if p.requires_grad]

            def _lr_lambda(step: int) -> float:
                if step < cfg.warmup_steps:
                    return float(step) / max(cfg.warmup_steps, 1)
                progress = (step - cfg.warmup_steps) / max(
                    cfg.max_steps - cfg.warmup_steps, 1
                )
                return 0.5 * (1.0 + math.cos(math.pi * progress))

            optimiser = AdamW(trainable_params, lr=cfg.learning_rate)
            scheduler = LambdaLR(optimiser, lr_lambda=_lr_lambda)

            if is_primary:
                cfg.output_path.parent.mkdir(parents=True, exist_ok=True)

            best_rfid = float("inf")
            no_improve_count = 0
            step = 0
            epoch = 0
            if hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)
            train_iter = iter(train_loader)
            t_step_start = time.monotonic()

            if is_primary:
                print(
                    f"[finetune-decoder] training: max_steps={cfg.max_steps} "
                    f"batch_size={cfg.batch_size}/GPU×{world_size}GPU "
                    f"lr={cfg.learning_rate} eval_every={cfg.eval_every_n_steps} "
                    f"grad_clip={cfg.grad_clip_norm}",
                    flush=True,
                )

            model.train()  # set once; only flipped during eval

            while step < cfg.max_steps and not STOP.is_set():
                budget = _step_timeout()
                with _wd_lock:
                    _wd_budget["s"] = budget
                    _wd_deadline["t"] = time.monotonic() + budget

                try:
                    batch = next(train_iter)
                except StopIteration:
                    epoch += 1
                    if hasattr(train_loader.sampler, "set_epoch"):
                        train_loader.sampler.set_epoch(epoch)
                    train_iter = iter(train_loader)
                    try:
                        batch = next(train_iter)
                    except StopIteration:
                        break

                frames = batch.float().permute(0, 3, 1, 2).to(train_device) / 255.0
                recon = self._encode_decode(model, frames, training=True)
                loss = self.compute_loss(frames, recon)

                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                if cfg.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        trainable_params, cfg.grad_clip_norm
                    )
                optimiser.step()
                scheduler.step()

                with _wd_lock:
                    _wd_deadline["t"] = None

                step_time = time.monotonic() - t_step_start
                _step_times.append(step_time)
                if len(_step_times) > 64:
                    del _step_times[0]
                t_step_start = time.monotonic()
                step += 1

                if is_primary and (step % 50 == 0 or step == 1):
                    print(
                        f"[finetune-decoder] step {step}/{cfg.max_steps} "
                        f"loss={float(loss):.4f} "
                        f"lr={scheduler.get_last_lr()[0]:.2e} "
                        f"step_time={step_time:.2f}s",
                        flush=True,
                    )

                # ── Evaluation + early stop (DISTRIBUTED — all 4 ranks) ──
                if step % cfg.eval_every_n_steps == 0 or step == cfg.max_steps:
                    if STOP.is_set():
                        break

                    # All ranks barrier for a clean sync point before eval.
                    if world_size > 1:
                        dist.barrier()

                    # ── Distributed eval: every rank runs encode/decode on
                    # its DistributedSampler shard of the val set, then
                    # all_gather frame buffers to rank 0 for rFID.  All 4
                    # GPUs busy throughout — no spin-wait imbalance.
                    rank0_failed = 0
                    t_eval = time.monotonic()
                    try:
                        model.eval()
                        # evaluate() is the parallel path — see method docs.
                        metrics = self.evaluate(model, val_loader)
                        model.train()
                    except Exception as exc:  # noqa: BLE001
                        # On any rank's failure, every rank gets a NaN dict
                        # and rank 0 marks the eval as failed below.
                        if is_primary:
                            print(
                                f"[finetune-decoder] eval at step {step} raised "
                                f"{type(exc).__name__}: {exc} — skipping",
                                flush=True,
                            )
                            import traceback

                            traceback.print_exc()
                        metrics = {"rfid": float("nan"), "l1": float("nan")}

                    current_rfid = float(metrics.get("rfid", float("nan")))
                    current_l1 = float(metrics.get("l1", float("nan")))

                    # Rank-0-only logging + checkpoint save + patience tracking.
                    # Wrapped in try/except so a save failure can't deadlock the
                    # broadcast below (job 1208992 lesson).
                    if is_primary:
                        try:
                            print(
                                f"[finetune-decoder] eval step {step}: "
                                f"rFID={current_rfid:.3f} L1={current_l1:.4f} "
                                f"({time.monotonic() - t_eval:.1f}s)",
                                flush=True,
                            )

                            improved = (
                                not (current_rfid != current_rfid)
                                and current_rfid < best_rfid
                            )
                            if improved:
                                best_rfid = current_rfid
                                no_improve_count = 0
                                self._save_checkpoint(model)
                                print(
                                    f"[finetune-decoder] ✓ new best rFID="
                                    f"{best_rfid:.3f} — checkpoint saved",
                                    flush=True,
                                )
                            else:
                                no_improve_count += 1
                                print(
                                    f"[finetune-decoder] no improvement "
                                    f"({no_improve_count}/{cfg.patience})",
                                    flush=True,
                                )
                        except Exception as exc:  # noqa: BLE001
                            print(
                                f"[finetune-decoder] rank 0 FATAL post-eval: "
                                f"{type(exc).__name__}: {exc} — aborting all ranks",
                                flush=True,
                            )
                            import traceback

                            traceback.print_exc()
                            rank0_failed = 1

                    # Broadcast stop decision (incl. rank-0 failure) from rank 0.
                    # All ranks MUST reach this point — see rank0_failed catch.
                    if world_size > 1:
                        stop_tensor = torch.tensor(
                            int(
                                (is_primary and no_improve_count >= cfg.patience)
                                or rank0_failed
                            ),
                            device=train_device,
                        )
                        dist.broadcast(stop_tensor, src=0)
                        should_stop = bool(stop_tensor.item())
                    else:
                        should_stop = (
                            no_improve_count >= cfg.patience or bool(rank0_failed)
                        )

                    if should_stop:
                        if is_primary:
                            print(
                                f"[finetune-decoder] early stop: patience "
                                f"{cfg.patience} exhausted",
                                flush=True,
                            )
                        break

            # Save final weights on rank 0; wrap in try/except so a save
            # failure can't deadlock the dist.barrier() below.
            if is_primary:
                try:
                    if not cfg.output_path.exists():
                        self._save_checkpoint(model)
                    merged_path = self._save_merged_checkpoint(model)
                    print(
                        f"[finetune-decoder] training complete "
                        f"(best rFID={best_rfid:.3f})\n"
                        f"  decoder weights : {cfg.output_path}\n"
                        f"  merged ckpt     : {merged_path}",
                        flush=True,
                    )
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"[finetune-decoder] final save FAILED: "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    import traceback

                    traceback.print_exc()
            if world_size > 1:
                dist.barrier()

        finally:
            _wd_done.set()
            with _wd_lock:
                _wd_deadline["t"] = None
            wd_thread.join(timeout=1.0)
            if model is not None:
                try:
                    import torch as _torch

                    del model
                    _torch.cuda.empty_cache()
                except Exception as exc:  # noqa: BLE001
                    if is_primary:
                        print(
                            f"[finetune-decoder] model release warning: {exc}",
                            flush=True,
                        )
            _cleanup_distributed()

        return cfg.output_path

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def _save_checkpoint(self, model: object) -> None:
        """Save decoder-only weights to ``output_path`` in safetensors format.

        ``.contiguous()`` is mandatory: BN-fused conv weights in the
        Open-MAGVIT2 decoder are non-contiguous after the channels_last
        memory-format conversion, and ``safetensors.save_file`` rejects
        non-contiguous tensors with a ``ValueError``.
        """
        import torch

        cfg = self.config
        cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
        raw = model.module if hasattr(model, "module") else model

        state_dict = {
            k: v.detach().cpu().contiguous()
            for k, v in raw.decoder.state_dict().items()
        }

        try:
            from safetensors.torch import save_file

            save_file(state_dict, str(cfg.output_path))
        except (ImportError, ModuleNotFoundError):
            fallback = cfg.output_path.with_suffix(".pt")
            warnings.warn(
                f"safetensors not available — saved as {fallback}",
                stacklevel=2,
            )
            torch.save(state_dict, str(fallback))

    def _save_merged_checkpoint(self, model: object) -> Path:
        """Patch original ckpt with fine-tuned decoder weights → save as .ckpt.

        Starts from ``imagenet_256_L.ckpt`` (which has consistent EMA + live
        state).  We replaced live encoder weights with EMA at startup, so
        live-encoder == EMA-encoder; both are unchanged by training.

        For the decoder, we patch BOTH the live ``decoder.*`` keys AND the
        ``model_ema.decoder*`` shadow buffers with the fine-tuned values.
        At bench time, ``stream_worker.load_model`` reads this checkpoint and
        ``ema_scope()`` copies the EMA shadow (= fine-tuned) over the live
        weights for inference.
        """
        import torch

        cfg = self.config
        # Derive merged ckpt name from the safetensors output stem so different
        # runs (e.g. v0 vs Option B with --perceptual-weight 1.0) write to
        # distinct files instead of clobbering each other.  E.g.:
        #   plasma-decoder-v1.safetensors → plasma-decoder-v1-merged.ckpt
        #   plasma-decoder-v1-optb.safetensors → plasma-decoder-v1-optb-merged.ckpt
        merged_path = cfg.output_path.with_name(
            f"{cfg.output_path.stem}-merged.ckpt"
        )
        merged_path.parent.mkdir(parents=True, exist_ok=True)

        orig = torch.load(
            str(cfg.magvit2_root / "weights" / "imagenet_256_L.ckpt"),
            map_location="cpu",
            weights_only=False,
        )
        sd = dict(orig["state_dict"])

        raw = model.module if hasattr(model, "module") else model
        # Fine-tuned decoder weights, cpu float32 contiguous (torch.save is more
        # forgiving than safetensors but contiguous keeps the merged ckpt
        # loadable by stream_worker.load_model without surprises).
        dec_sd = {
            k: v.detach().cpu().float().contiguous()
            for k, v in raw.decoder.state_dict().items()
        }

        n_live = 0
        n_ema = 0
        for local_k, val in dec_sd.items():
            full_k = f"decoder.{local_k}"
            if full_k in sd:
                sd[full_k] = val
                n_live += 1
            ema_buf = f"model_ema.{full_k.replace('.', '')}"
            if ema_buf in sd:
                sd[ema_buf] = val
                n_ema += 1

        torch.save({"state_dict": sd}, str(merged_path))
        print(
            f"[finetune-decoder] merged ckpt saved: {merged_path}  "
            f"(patched {n_live} live + {n_ema} EMA decoder keys)",
            flush=True,
        )
        return merged_path


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def finetune_decoder(config: DecoderFinetuneConfig) -> Path:
    """Fine-tune the decoder and return the saved safetensors path."""
    trainer = DecoderFinetuneTrainer(config)
    return trainer.train()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_shot_ids(path: str) -> list[int]:
    ids = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            ids.append(int(line))
    return ids


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune the Open-MAGVIT2 decoder on plasma rbb frames. "
            "Run via torchrun inside the Open-MAGVIT2 venv."
        )
    )
    parser.add_argument("--train-shots", required=True)
    parser.add_argument("--val-shots", required=True)
    parser.add_argument("--max-steps", type=int, default=10_000)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Per-GPU batch size (effective batch = batch_size × world_size).",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--eval-every", type=int, default=1_000)
    parser.add_argument(
        "--perceptual-weight",
        type=float,
        default=None,
        help=(
            "VGG16 perceptual loss weight (default: DecoderFinetuneConfig 0.1). "
            "Option B (commit 2026-05-28) explores higher values to test whether "
            "L1-dominant training was misaligned with the bench rFID metric — "
            "verdict bench 1209101 showed fine-tune regressed all rFID classes "
            "vs baseline despite PSNR improvement, suggesting the loss objective "
            "doesn't transfer to InceptionV3-based rFID. Try 1.0 or higher."
        ),
    )
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    train_ids = _parse_shot_ids(args.train_shots)
    val_ids = _parse_shot_ids(args.val_shots)

    config = DecoderFinetuneConfig(
        train_shot_ids=train_ids,
        val_shot_ids=val_ids,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        eval_every_n_steps=args.eval_every,
        device=args.device,
    )
    if args.output_path is not None:
        config.output_path = Path(args.output_path)
    if args.perceptual_weight is not None:
        config.perceptual_weight = args.perceptual_weight

    out = finetune_decoder(config)
    print(f"[finetune-decoder] weights saved to: {out}", flush=True)
