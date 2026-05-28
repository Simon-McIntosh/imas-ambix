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
    """Install best-effort SIGTERM/SIGINT handlers.

    NOTE: Signals cannot reliably stop CUDA-DDP training (Python signal
    handlers only run between bytecodes; we may be inside multi-second NCCL
    collectives).  These handlers are a fallback for non-CUDA paths and for
    forwarding cluster-level termination signals.  The PRIMARY cancellation
    mechanism is the STOP-FILE (see ``_check_stop_file`` and AGENTS.md
    §2a-cancel) — agents must use that, NEVER ``scancel``.
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


def _stop_file_path() -> Path | None:
    """Return the STOP-FILE path from ``AMBIX_STOP_FILE`` env, or None.

    See AGENTS.md §2a-cancel for the full contract.  When the file exists,
    the training loop breaks cleanly at the next step boundary — safe under
    NCCL/CUDA collectives in a way SIGTERM is not.
    """
    p = os.environ.get("AMBIX_STOP_FILE")
    if not p:
        return None
    return Path(p)


def _check_stop_file() -> bool:
    """Return True if the STOP-FILE exists. Single filesystem stat (~µs)."""
    p = _stop_file_path()
    return p is not None and p.exists()


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

    def build_dataloaders(self) -> tuple[Iterable, Iterable, Iterable | None]:
        """Build training and validation data loaders.

        Returns
        -------
        tuple
            ``(train_loader, val_loader_distributed, val_loader_full)``.
            ``val_loader_distributed`` is per-rank (used only for tests that
            want sharded eval).  ``val_loader_full`` is the unsharded loader
            built only on rank 0 — used by :meth:`evaluate` so rFID is
            computed on the full 93 k-frame val set.
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
                from PIL import Image

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
                    img = Image.fromarray(frame)
                    img = img.resize(
                        (self.image_size, self.image_size), Image.BILINEAR
                    )
                    frame = _np.asarray(img)

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

        # Distributed val loader — kept for completeness but currently
        # unused (evaluate() uses val_loader_full on rank 0 only).
        val_sampler = DistributedSampler(
            val_ds, num_replicas=world_size, rank=rank, shuffle=False
        )
        val_loader_distributed = DataLoader(
            val_ds,
            batch_size=cfg.batch_size,
            sampler=val_sampler,
            num_workers=n_workers,
            prefetch_factor=2,
            persistent_workers=True,
            pin_memory=cfg.device.startswith("cuda"),
        )

        # Full unsharded val loader — only used by rank 0 in evaluate().
        # We always build it (cheap, just a sampler-less DataLoader) so the
        # API is symmetric across ranks; non-rank-0 ranks never iterate it.
        val_loader_full = DataLoader(
            val_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=n_workers,
            prefetch_factor=2,
            persistent_workers=True,
            pin_memory=cfg.device.startswith("cuda"),
        )
        return train_loader, val_loader_distributed, val_loader_full

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

        raw = model.module if hasattr(model, "module") else model

        # Encode + codebook lookup: no_grad, autocast bf16.
        with torch.no_grad(), torch.amp.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            frames_normed = frames_input.mul(2.0).sub(1.0)  # [-1, 1]
            frames_normed = frames_normed.to(memory_format=torch.channels_last)
            _, _, idx, _ = raw.encode(frames_normed)

            B = frames_input.shape[0]
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

    def evaluate(self, model: object, val_loader: Iterable) -> dict[str, float]:
        """Evaluate rFID + L1 on the full validation set (rank 0 only).

        Other ranks block on the surrounding ``dist.barrier()`` while rank 0
        iterates; the rFID result tensor is broadcast back so all ranks agree
        on the early-stop decision.
        """
        import numpy as np
        import torch
        import torch.nn.functional as F

        cfg = self.config
        all_targets: list[np.ndarray] = []
        all_recons: list[np.ndarray] = []
        l1_sum = 0.0
        n_batches = 0

        # Cap rFID buffer at 5 k frames (statistically sufficient for FID).
        _RFID_MAX_FRAMES = 5_000
        model.eval()
        try:
            for batch in val_loader:
                # Honour STOP-FILE during eval too (long val passes still
                # need to be cancellable).
                if STOP.is_set() or _check_stop_file():
                    break
                # cfg.device may be "cuda"; current device set in build_model.
                local_rank = int(os.environ.get("LOCAL_RANK", "0"))
                dev = f"cuda:{local_rank}" if cfg.device.startswith("cuda") else cfg.device
                frames = batch.float().permute(0, 3, 1, 2).to(dev) / 255.0
                recon = self._encode_decode(model, frames, training=False)

                l1_sum += float(F.l1_loss(recon, frames).item())
                n_batches += 1

                current_total = sum(a.shape[0] for a in all_targets)
                if current_total < _RFID_MAX_FRAMES:
                    t_np = (
                        frames.permute(0, 2, 3, 1).cpu().numpy() * 255
                    ).clip(0, 255).astype(np.uint8)
                    r_np = (
                        recon.permute(0, 2, 3, 1).cpu().numpy() * 255
                    ).clip(0, 255).astype(np.uint8)
                    all_targets.append(t_np)
                    all_recons.append(r_np)
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"[finetune-decoder] evaluate() interrupted ({exc}); "
                "reporting partial metrics.",
                stacklevel=2,
            )

        mean_l1 = l1_sum / max(n_batches, 1)

        if not all_targets:
            return {"rfid": float("nan"), "l1": float("nan")}

        target_arr = np.concatenate(all_targets, axis=0)
        recon_arr = np.concatenate(all_recons, axis=0)

        rfid_val = float("nan")
        try:
            from imas_ambix.eval.metrics import rfid as _rfid  # type: ignore[import]

            rfid_val = _rfid(target_arr, recon_arr)
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"rFID computation failed ({exc}); reporting NaN.",
                stacklevel=2,
            )

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

        # Surface the STOP-FILE path so cancellation is straightforward.
        # See AGENTS.md §2a-cancel — touch this path to stop the job cleanly.
        stop_file = _stop_file_path()
        if is_primary:
            if stop_file is not None:
                print(
                    f"[finetune-decoder] STOP-FILE: touch {stop_file} to "
                    "request graceful exit (no scancel needed; no drain risk)",
                    flush=True,
                )
            else:
                print(
                    "[finetune-decoder] WARNING: AMBIX_STOP_FILE env var not "
                    "set — graceful cancellation unavailable. The sbatch "
                    "wrapper should export it (AGENTS.md §2a-cancel).",
                    flush=True,
                )

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
            train_loader, _val_loader_dist, val_loader_full = self.build_dataloaders()

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
                # STOP-FILE check: safe-cancel boundary (AGENTS.md §2a-cancel).
                # Runs between collectives → no NCCL deadlock, no drain risk.
                if _check_stop_file():
                    if is_primary:
                        print(
                            f"[finetune-decoder] STOP-FILE detected at step "
                            f"{step} → clean exit",
                            flush=True,
                        )
                    STOP.set()
                    break

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

                # ── Evaluation + early stop (rank 0 only) ─────────────────
                if step % cfg.eval_every_n_steps == 0 or step == cfg.max_steps:
                    if STOP.is_set():
                        break

                    # All ranks barrier first so we have a clean sync point.
                    if world_size > 1:
                        dist.barrier()

                    if is_primary:
                        model.eval()
                        t_eval = time.monotonic()
                        try:
                            metrics = self.evaluate(model, val_loader_full)
                        except Exception as exc:  # noqa: BLE001
                            print(
                                f"[finetune-decoder] eval at step {step} raised "
                                f"{type(exc).__name__}: {exc} — skipping",
                                flush=True,
                            )
                            metrics = {"rfid": float("nan"), "l1": float("nan")}
                        current_rfid = float(metrics.get("rfid", float("nan")))
                        current_l1 = float(metrics.get("l1", float("nan")))
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
                        model.train()

                    # Broadcast early-stop decision from rank 0.
                    if world_size > 1:
                        stop_tensor = torch.tensor(
                            int(is_primary and no_improve_count >= cfg.patience),
                            device=train_device,
                        )
                        dist.broadcast(stop_tensor, src=0)
                        should_stop = bool(stop_tensor.item())
                    else:
                        should_stop = no_improve_count >= cfg.patience

                    if should_stop:
                        if is_primary:
                            print(
                                f"[finetune-decoder] early stop: patience "
                                f"{cfg.patience} exhausted",
                                flush=True,
                            )
                        break

            # Save final weights on rank 0.
            if is_primary:
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
        """Save decoder-only weights to ``output_path`` in safetensors format."""
        import torch

        cfg = self.config
        cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
        raw = model.module if hasattr(model, "module") else model

        state_dict = {k: v.cpu() for k, v in raw.decoder.state_dict().items()}

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
        merged_path = cfg.output_path.with_name("plasma-decoder-v1-merged.ckpt")
        merged_path.parent.mkdir(parents=True, exist_ok=True)

        orig = torch.load(
            str(cfg.magvit2_root / "weights" / "imagenet_256_L.ckpt"),
            map_location="cpu",
            weights_only=False,
        )
        sd = dict(orig["state_dict"])

        raw = model.module if hasattr(model, "module") else model
        dec_sd = {k: v.cpu().float() for k, v in raw.decoder.state_dict().items()}

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

    out = finetune_decoder(config)
    print(f"[finetune-decoder] weights saved to: {out}", flush=True)
