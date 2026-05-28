"""Plasma-domain Open-MAGVIT2 decoder fine-tune.

Freezes the encoder + LFQ codebook; trains only the decoder with pixel L1 +
perceptual (VGG16) loss on MAST rbb plasma frames.  The encoder and codebook
remain ImageNet-derived — token IDs are stable, so no re-encoding of the
9,528-shot corpus is needed when the decoder is swapped.

Hardening mirrors ``imas_ambix.bench.stream_worker``:
- SIGTERM/SIGINT handler sets STOP flag (< 5 s graceful exit).
- Per-step watchdog auto-calibrated from the running-median step time.
- ``try/finally`` releases model + empties CUDA cache.

Run inside the Open-MAGVIT2 venv (Python 3.11, torch 2.1.1):

    PYTHONPATH=/path/to/imas-ambix \\
      /path/to/open-magvit2/.venv/bin/python \\
      imas_ambix/tokenizer/finetune_decoder.py \\
      --train-shots train_ids.txt --val-shots val_ids.txt

CLI is also exposed via ``ambix tokenize finetune-decoder`` (uses same
DecoderFinetuneTrainer internally).
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
    """Initialise NCCL process group when launched via torchrun.

    Returns ``(rank, world_size)``.  When not running under torchrun
    (WORLD_SIZE absent or 1) returns ``(0, 1)`` and leaves the process group
    uninitialised — single-GPU path is unchanged.
    """
    import torch
    import torch.distributed as dist

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    else:
        if torch.cuda.is_available():
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

    Defaults correspond to the recipe in the tokenizers plan §12.1:
    ~375 k frames from 7,500 rbb shots, 4×H200 exclusive (used as single GPU),
    AdamW + cosine LR, 10 k steps, early-stop on rFID plateau.
    """

    frame_root: Path = _DEFAULT_L1_ROOT
    """Level-1 zarr root. Shot zarrs at ``{frame_root}/{shot_id}.zarr/{camera}``."""
    magvit2_root: Path = _DEFAULT_MAGVIT2_ROOT
    output_path: Path = _DEFAULT_MAGVIT2_ROOT / "weights" / "plasma-decoder-v1.safetensors"
    train_shot_ids: list[int] = field(default_factory=list)
    val_shot_ids: list[int] = field(default_factory=list)
    camera: str = "rbb"
    image_size: int = 256
    frames_per_shot: int = 50
    batch_size: int = 16  # per-GPU; effective = batch_size × world_size (plan spec: 16 × 4 = 64)
    learning_rate: float = 1e-4
    max_steps: int = 10_000
    warmup_steps: int = 200
    l1_weight: float = 1.0
    perceptual_weight: float = 0.1  # locked decision: lpips-weight = 0.1
    adv_weight: float = 0.0  # no adversarial in v0
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


class DecoderFinetuneTrainer:
    """Drives the Open-MAGVIT2 decoder fine-tune loop for plasma imagery.

    All heavy imports (torch, torchvision, Open-MAGVIT2) are lazy — the
    constructor does not load weights or open data so the object can be
    instantiated without a GPU.

    Usage::

        config = DecoderFinetuneConfig(
            train_shot_ids=[15085, 15086, ...],
            val_shot_ids=[15100, 15101, ...],
        )
        trainer = DecoderFinetuneTrainer(config)
        output_path = trainer.train()
    """

    # TOKEN_HW: 256 / 16 = 16 (Open-MAGVIT2 spatial compression factor)
    _TOKEN_HW: int = 16

    def __init__(self, config: DecoderFinetuneConfig) -> None:
        self.config = config
        self._model: object | None = None
        self._vgg_features: object | None = None

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def build_dataloaders(self) -> tuple[Iterable, Iterable]:
        """Build training and validation data loaders (lazy, streaming).

        Pre-scans zarr metadata to enumerate (path, variable, frame_index) tuples
        without reading any pixel data — identical in spirit to stream_encode.py.
        Each DataLoader worker opens its own zarr store handles and reads single
        frames on demand, overlapping I/O with GPU compute.

        Level-1 zarr layout::

            {frame_root}/{shot_id}.zarr/{camera}/

        Returns
        -------
        tuple[Iterable, Iterable]
            ``(train_loader, val_loader)`` as :class:`torch.utils.data.DataLoader`.
        """
        import numpy as np
        import torch
        import torch.distributed as dist
        import zarr
        from torch.utils.data import DataLoader
        from torch.utils.data.distributed import DistributedSampler

        rank = dist.get_rank() if (dist.is_available() and dist.is_initialized()) else 0
        world_size = dist.get_world_size() if (dist.is_available() and dist.is_initialized()) else 1

        cfg = self.config

        # ------------------------------------------------------------------
        # Lazy frame dataset — no pixel data loaded at __init__ time
        # ------------------------------------------------------------------
        class _PlasmaFrameDataset(torch.utils.data.Dataset):
            """Each item is one (H, W, 3) uint8 frame, loaded lazily from zarr."""

            def __init__(
                self,
                entries: list[tuple[str, str, int]],
                image_size: int,
            ) -> None:
                self.entries = entries    # (zarr_path, arr_key, frame_idx)
                self.image_size = image_size

            def __len__(self) -> int:
                return len(self.entries)

            def __getitem__(self, i: int) -> "torch.Tensor":
                import numpy as np
                from PIL import Image

                zarr_path, arr_key, frame_idx = self.entries[i]

                # Per-worker zarr store cache (not shared across processes)
                if not hasattr(self, "_z_cache"):
                    self._z_cache: dict[str, object] = {}

                if zarr_path not in self._z_cache:
                    if len(self._z_cache) > 128:
                        del self._z_cache[next(iter(self._z_cache))]
                    self._z_cache[zarr_path] = zarr.open_group(
                        zarr_path, mode="r"
                    )

                z = self._z_cache[zarr_path]
                frame = np.asarray(z[arr_key][frame_idx])  # single frame read

                if frame.dtype != np.uint8:
                    lo = float(frame.min())
                    hi = float(frame.max())
                    if hi > lo:
                        frame = (
                            (frame.astype(np.float32) - lo) * 255.0 / (hi - lo)
                        ).clip(0, 255).astype(np.uint8)
                    else:
                        frame = np.zeros(frame.shape, dtype=np.uint8)

                if frame.ndim == 2:
                    frame = np.repeat(frame[..., np.newaxis], 3, axis=-1)
                elif frame.ndim == 3 and frame.shape[-1] == 1:
                    frame = np.repeat(frame, 3, axis=-1)
                elif frame.ndim == 3 and frame.shape[-1] != 3:
                    frame = frame[:, :, :3]

                if frame.shape[:2] != (self.image_size, self.image_size):
                    img = Image.fromarray(frame)
                    img = img.resize(
                        (self.image_size, self.image_size), Image.BILINEAR
                    )
                    frame = np.asarray(img)

                return torch.from_numpy(np.ascontiguousarray(frame))

        # ------------------------------------------------------------------
        # Metadata scan: enumerate frame (path, key, index) pairs only
        # ------------------------------------------------------------------
        def _scan_shots(
            shot_ids: list[int], label: str
        ) -> list[tuple[str, str, int]]:
            entries: list[tuple[str, str, int]] = []
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
                    arr_keys = [
                        k for k in z.array_keys()
                        if z[k].ndim >= 3  # skip 1D coord arrays
                    ]
                    if not arr_keys:
                        n_missing += 1
                        continue
                    arr_key = arr_keys[0]
                    t_len = int(z[arr_key].shape[0])
                    if t_len == 0:
                        continue
                    indices = np.linspace(
                        0, t_len - 1, min(cfg.frames_per_shot, t_len), dtype=int
                    )
                    for idx in indices:
                        entries.append((str(zarr_path), arr_key, int(idx)))
                except Exception as exc:  # noqa: BLE001
                    warnings.warn(
                        f"[finetune-decoder] scan {shot_id} failed: {exc}",
                        stacklevel=2,
                    )
                    n_missing += 1
            elapsed = time.monotonic() - t0
            print(
                f"[finetune-decoder] {label}: scanned {len(shot_ids)} shots "
                f"→ {len(entries)} frames "
                f"({n_missing} missing/skipped) in {elapsed:.1f}s",
                flush=True,
            )
            return entries

        # Only rank-0 needs to print the scan header; all ranks scan independently
        # (each needs the full entry list for the sampler).
        if rank == 0:
            print(
                f"[finetune-decoder] scanning {len(cfg.train_shot_ids)} train + "
                f"{len(cfg.val_shot_ids)} val shots (metadata only) ...",
                flush=True,
            )
        train_entries = _scan_shots(cfg.train_shot_ids, "train")
        val_entries = _scan_shots(cfg.val_shot_ids, "val")

        train_ds = _PlasmaFrameDataset(train_entries, cfg.image_size)
        val_ds = _PlasmaFrameDataset(val_entries, cfg.image_size)

        # DDP: each rank sees a non-overlapping shard of the data.
        # Single-GPU: DistributedSampler with world_size=1 == plain shuffle.
        train_sampler = DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
        )
        val_sampler = DistributedSampler(
            val_ds, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
        )

        # num_workers per rank: 4 GPUs × 4 workers = 16 total (fits in 30 CPUs).
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
        """Load the real Open-MAGVIT2 VQModel; freeze encoder + codebook.

        Uses the same ``load_model`` logic as ``imas_ambix.bench.stream_worker``
        to ensure byte-consistency with the corpus encode.  After loading,
        ALL parameters are frozen, then ``model.decoder`` is unfrozen for
        fine-tuning.

        Returns
        -------
        torch.nn.Module
            Full VQModel with encoder/codebook frozen and decoder trainable.
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
        sd = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)[
            "state_dict"
        ]
        model.load_state_dict(sd, strict=False)

        # Freeze everything, then unfreeze the decoder only
        model.requires_grad_(False)
        model.decoder.requires_grad_(True)

        # Device: use LOCAL_RANK when torchrun sets it, else fall back to cfg.device
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
            model = model.to(
                device=device,
                dtype=torch.bfloat16,
                memory_format=torch.channels_last,
            )
        else:
            model = model.to(device)

        # Wrap in DDP when running multi-GPU; only sync decoder gradients
        # (frozen encoder/codebook have no grad so they're ignored automatically).
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            model = DDP(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=True,  # frozen encoder/codebook have no grad
            )

        rank = dist.get_rank() if (dist.is_available() and dist.is_initialized()) else 0
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in model.parameters())
        if rank == 0:
            print(
                f"[finetune-decoder] model loaded: "
                f"{n_trainable:,} trainable / {n_total:,} total params  "
                f"device={device}  world_size={dist.get_world_size() if dist.is_initialized() else 1}",
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
        """Encode with frozen encoder; decode with trainable decoder.

        Parameters
        ----------
        model:
            VQModel returned by :meth:`build_model`.
        frames_input:
            ``(B, 3, H, W)`` float32 in [0, 1], on device.
        training:
            When True, decode is computed with grad (for backward pass).
            When False, wrapped in ``torch.no_grad()``.

        Returns
        -------
        torch.Tensor
            ``(B, 3, H, W)`` float32 in [0, 1].
        """
        import torch

        # Access the underlying VQModel when DDP-wrapped
        raw = model.module if hasattr(model, "module") else model

        target_dtype = next(raw.decoder.parameters()).dtype
        frames_normed = frames_input.to(target_dtype).mul(2.0).sub(1.0)  # [-1,1]

        # Encode with frozen encoder — never use ema_scope() during fine-tune.
        # LitEma.copy_to/forward assert that frozen params are NOT in m_name2s_name,
        # but they are (the EMA was built when all params were trainable), so
        # ema_scope() raises AssertionError after our requires_grad freeze.
        # The encoder is frozen anyway, so live weights == checkpoint weights == EMA.
        with torch.no_grad():
            _, _, idx, _ = raw.encode(frames_normed)

        # Reshape token indices
        B = frames_input.shape[0]
        h = w = self._TOKEN_HW
        idx_flat = idx.reshape(B, h * w)
        bhwc = (B, h, w, int(raw.quantize.codebook_dim))

        # Codebook lookup (codebook frozen; no ema_scope for same reason)
        quant = raw.quantize.get_codebook_entry(idx_flat, bhwc=bhwc, order="pre")
        quant = quant.to(target_dtype)

        # Decode via raw module.  DDP gradient sync is driven by parameter-level
        # autograd hooks registered at DDP.__init__ time — independent of the
        # call path — so raw.decode() correctly syncs decoder grads across ranks.
        if training:
            recon_m11 = raw.decode(quant)
        else:
            with torch.no_grad():
                recon_m11 = raw.decode(quant)

        return (recon_m11.float().clamp(-1, 1) + 1.0) / 2.0  # [0,1] float32

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        target_frames: object,
        recon_frames: object,
    ) -> object:
        """L1 + optional VGG16 perceptual loss.

        Falls back to L1-only if torchvision is absent or VGG16 weights
        cannot be loaded (e.g. GPU node has no outbound network).

        Parameters
        ----------
        target_frames:
            ``(B, 3, H, W)`` float32 in [0, 1].
        recon_frames:
            ``(B, 3, H, W)`` float32 in [0, 1].
        """
        import torch
        import torch.nn.functional as F

        cfg = self.config
        l1 = F.l1_loss(recon_frames, target_frames)
        loss = cfg.l1_weight * l1

        if cfg.perceptual_weight > 0.0 and self._vgg_features is not False:
            # _vgg_features states: None = not yet loaded, False = failed/disabled,
            # nn.Module = ready.  This avoids the TypeError bug where object() was
            # called as a function on every step after a failed download.
            try:
                import torchvision.models as tvm

                if self._vgg_features is None:
                    vgg = tvm.vgg16(weights=tvm.VGG16_Weights.IMAGENET1K_V1)
                    self._vgg_features = torch.nn.Sequential(
                        *list(vgg.features.children())[:16]
                    ).to(cfg.device)
                    self._vgg_features.requires_grad_(False)
                    self._vgg_features.eval()
                    print("[finetune-decoder] VGG16 perceptual loss enabled", flush=True)

                with torch.no_grad():
                    feat_target = self._vgg_features(target_frames)
                feat_recon = self._vgg_features(recon_frames)
                perceptual = F.mse_loss(feat_recon, feat_target)
                loss = loss + cfg.perceptual_weight * perceptual

            except Exception as exc:  # noqa: BLE001
                warnings.warn(
                    f"VGG16 perceptual loss disabled ({exc}); falling back to L1.",
                    stacklevel=2,
                )
                self._vgg_features = False  # permanently disable; never retry

        return loss

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, model: object, val_loader: Iterable) -> dict[str, float]:
        """Evaluate rFID + L1 on the validation set.

        Returns
        -------
        dict[str, float]
            ``{"rfid": <float>, "l1": <float>}``
        """
        import numpy as np
        import torch
        import torch.nn.functional as F

        cfg = self.config
        all_targets: list[np.ndarray] = []
        all_recons: list[np.ndarray] = []
        l1_sum = 0.0
        n_batches = 0

        # Cap rFID buffer at 5 k frames (statistically sufficient; avoids 36 GB accumulation)
        _RFID_MAX_FRAMES = 5_000
        model.eval()
        try:
            for batch in val_loader:
                if STOP.is_set():
                    break
                frames = batch.float().permute(0, 3, 1, 2).to(cfg.device) / 255.0
                recon = self._encode_decode(model, frames, training=False)

                l1_sum += float(F.l1_loss(recon, frames).item())
                n_batches += 1

                # Accumulate for rFID (capped to avoid multi-GB RAM usage)
                current_total = sum(a.shape[0] for a in all_targets)
                if current_total < _RFID_MAX_FRAMES:
                    t_np = (frames.permute(0, 2, 3, 1).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                    r_np = (recon.permute(0, 2, 3, 1).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                    all_targets.append(t_np)
                    all_recons.append(r_np)
        except Exception as exc:  # noqa: BLE001 — worker crash, CUDA error, etc.
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
        """Run the full decoder fine-tune loop.

        Algorithm
        ---------
        1. Install SIGTERM/SIGINT handlers + per-step watchdog.
        2. Build dataloaders (:meth:`build_dataloaders`).
        3. Build VQModel, freeze encoder + codebook (:meth:`build_model`).
        4. AdamW, cosine LR + linear warmup.
        5. Loop: encode (frozen) → decode (trainable) → L1+perceptual loss → step.
        6. Every ``eval_every_n_steps``: rFID eval; save on improvement; early-stop.
        7. Save decoder-only weights (.safetensors) + full merged weights (.ckpt).

        Returns
        -------
        pathlib.Path
            Path to the saved ``.safetensors`` decoder weights.
        """
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

        # Initialise NCCL when launched via torchrun; no-op for single-GPU.
        rank, world_size = _init_distributed()
        is_primary = rank == 0  # gate logging, checkpointing, early-stop

        torch.manual_seed(cfg.seed + rank)  # different seed per rank for data variety

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
            return 300.0  # generous initial timeout for data loading

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

        # Device string for moving batches (matches what build_model used)
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        train_device = f"cuda:{local_rank}" if cfg.device.startswith("cuda") else cfg.device

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
            # Set initial epoch for DistributedSampler shuffling
            if hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)
            train_iter = iter(train_loader)
            t_step_start = time.monotonic()

            if is_primary:
                print(
                    f"[finetune-decoder] training: max_steps={cfg.max_steps} "
                    f"batch_size={cfg.batch_size}/GPU×{world_size}GPU "
                    f"lr={cfg.learning_rate} eval_every={cfg.eval_every_n_steps}",
                    flush=True,
                )

            while step < cfg.max_steps and not STOP.is_set():
                # Arm watchdog for this step
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

                model.train()
                frames = batch.float().permute(0, 3, 1, 2).to(train_device) / 255.0
                recon = self._encode_decode(model, frames, training=True)

                loss = self.compute_loss(frames, recon)
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                scheduler.step()

                # EMA update intentionally skipped — see _save_merged_checkpoint.

                # Disarm watchdog
                with _wd_lock:
                    _wd_deadline["t"] = None

                step_time = time.monotonic() - t_step_start
                _step_times.append(step_time)
                if len(_step_times) > 64:
                    del _step_times[0]
                t_step_start = time.monotonic()

                step += 1

                if is_primary and (step % 100 == 0 or step == 1):
                    print(
                        f"[finetune-decoder] step {step}/{cfg.max_steps} "
                        f"loss={float(loss):.4f} "
                        f"lr={scheduler.get_last_lr()[0]:.2e}",
                        flush=True,
                    )

                # Evaluation + early stopping
                if step % cfg.eval_every_n_steps == 0 or step == cfg.max_steps:
                    if STOP.is_set():
                        break
                    model.eval()
                    t_eval = time.monotonic()
                    try:
                        metrics = self.evaluate(model, val_loader)
                    except Exception as exc:  # noqa: BLE001
                        print(
                            f"[finetune-decoder] eval at step {step} raised "
                            f"{type(exc).__name__}: {exc} — skipping",
                            flush=True,
                        )
                        metrics = {"rfid": float("nan"), "l1": float("nan")}
                    current_rfid = metrics.get("rfid", float("nan"))
                    print(
                        f"[finetune-decoder] eval step {step}: "
                        f"rFID={current_rfid:.3f} L1={metrics.get('l1', float('nan')):.4f} "
                        f"({time.monotonic() - t_eval:.1f}s)",
                        flush=True,
                    )

                    improved = (
                        not (current_rfid != current_rfid)  # not NaN
                        and current_rfid < best_rfid
                    )
                    if is_primary:
                        if improved:
                            best_rfid = current_rfid
                            no_improve_count = 0
                            self._save_checkpoint(model)
                            print(
                                f"[finetune-decoder] ✓ new best rFID={best_rfid:.3f} "
                                f"— checkpoint saved",
                                flush=True,
                            )
                        else:
                            no_improve_count += 1
                            print(
                                f"[finetune-decoder] no improvement "
                                f"({no_improve_count}/{cfg.patience})",
                                flush=True,
                            )

                    # Broadcast early-stop decision from rank 0 to all ranks
                    if world_size > 1:
                        import torch as _t
                        stop_tensor = _t.tensor(
                            int(is_primary and no_improve_count >= cfg.patience),
                            device=train_device,
                        )
                        dist.broadcast(stop_tensor, src=0)
                        should_stop = bool(stop_tensor.item())
                    else:
                        should_stop = is_primary and no_improve_count >= cfg.patience

                    if should_stop:
                        if is_primary:
                            print(
                                f"[finetune-decoder] early stop: patience "
                                f"{cfg.patience} exhausted",
                                flush=True,
                            )
                        break

            # Rank 0 saves final weights; all ranks wait.
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

        if hasattr(raw, "decoder") and hasattr(raw.decoder, "state_dict"):
            state_dict = {k: v.cpu() for k, v in raw.decoder.state_dict().items()}
        else:
            state_dict = {
                k: v.cpu() for k, v in raw.state_dict().items() if v.requires_grad
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
        """Patch the original checkpoint with fine-tuned decoder and save as .ckpt.

        Starts from the original ``imagenet_256_L.ckpt`` (which has a consistent
        EMA state) and replaces both the live ``decoder.*`` weights and the
        corresponding ``model_ema.*`` shadow buffers with the fine-tuned decoder.

        This lets ``stream_worker.load_model`` load the file and use
        ``ema_scope()`` for inference exactly as it does today — the EMA shadow
        now carries the fine-tuned decoder instead of the ImageNet one.

        Returns
        -------
        pathlib.Path
            Path to the saved merged ``.ckpt`` file.
        """
        import torch

        cfg = self.config
        merged_path = cfg.output_path.with_name("plasma-decoder-v1-merged.ckpt")
        merged_path.parent.mkdir(parents=True, exist_ok=True)

        # Load original checkpoint (encoder + codebook + EMA all consistent)
        orig = torch.load(
            str(cfg.magvit2_root / "weights" / "imagenet_256_L.ckpt"),
            map_location="cpu",
            weights_only=False,
        )
        sd = dict(orig["state_dict"])

        raw = model.module if hasattr(model, "module") else model
        # Fine-tuned decoder weights (cpu float32 for maximum compat)
        dec_sd = {k: v.cpu().float() for k, v in raw.decoder.state_dict().items()}

        # Replace live decoder keys: "decoder.<local>" → patched value
        for local_k, val in dec_sd.items():
            full_k = f"decoder.{local_k}"
            if full_k in sd:
                sd[full_k] = val

        # Replace EMA shadow keys for decoder.
        # LitEma stores buffer under name = full_param_name.replace('.', ''),
        # prefixed by "model_ema." in the module's state_dict.
        # E.g. "decoder.conv_in.weight" → buffer "model_ema.decoderconv_inweight"
        for local_k, val in dec_sd.items():
            full_param = f"decoder.{local_k}"
            ema_buf = f"model_ema.{full_param.replace('.', '')}"
            if ema_buf in sd:
                sd[ema_buf] = val

        torch.save({"state_dict": sd}, str(merged_path))
        print(f"[finetune-decoder] merged ckpt saved: {merged_path}", flush=True)
        return merged_path


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def finetune_decoder(config: DecoderFinetuneConfig) -> Path:
    """Fine-tune the decoder and return the saved safetensors path."""
    trainer = DecoderFinetuneTrainer(config)
    return trainer.train()


# ---------------------------------------------------------------------------
# CLI entry point (``python finetune_decoder.py`` or ``python -m ...``)
# ---------------------------------------------------------------------------


def _parse_shot_ids(path: str) -> list[int]:
    """Read one shot ID per line from a text file, skip blank/comment lines."""
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
            "Run inside the Open-MAGVIT2 venv (torch 2.1.1)."
        )
    )
    parser.add_argument(
        "--train-shots",
        required=True,
        help="Text file with one train shot ID per line.",
    )
    parser.add_argument(
        "--val-shots",
        required=True,
        help="Text file with one val shot ID per line.",
    )
    parser.add_argument(
        "--max-steps", type=int, default=10_000, help="Training steps."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Frames per gradient step (single GPU; plan spec 16×4=64).",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-4, help="AdamW initial LR."
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help=(
            "Destination for fine-tuned decoder weights (.safetensors). "
            "Defaults to {magvit2_root}/weights/plasma-decoder-v1.safetensors."
        ),
    )
    parser.add_argument(
        "--device", default="cuda", help="'cuda' or 'cpu'."
    )
    args = parser.parse_args()

    train_ids = _parse_shot_ids(args.train_shots)
    val_ids = _parse_shot_ids(args.val_shots)

    config = DecoderFinetuneConfig(
        train_shot_ids=train_ids,
        val_shot_ids=val_ids,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
    )
    if args.output_path is not None:
        config.output_path = Path(args.output_path)

    out = finetune_decoder(config)
    print(f"[finetune-decoder] weights saved to: {out}", flush=True)
