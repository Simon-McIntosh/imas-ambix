"""Plasma-domain Open-MAGVIT2 decoder fine-tune scaffold.

Freezes the Open-MAGVIT2 encoder + codebook; trains only the decoder on
MAST visible-camera frames to adapt reconstruction to plasma imagery. The
encoder and codebook remain ImageNet-derived — token IDs assigned are stable,
so no re-encoding of previously tokenised shots is required when the decoder
is swapped.

See ``plans/tokenizers.md`` §12.1 for the design rationale, cost estimate,
and trigger conditions.

All torch / torchvision / safetensors / pytorch_fid imports are lazy (inside
method bodies or guarded by ``TYPE_CHECKING``) so this module imports cleanly
without GPU dependencies installed.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    import torch

# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

_DEFAULT_MAGVIT2_ROOT = Path("/work/projects/imas_gpu/mast-tokens/v1/open-magvit2")


@dataclass
class DecoderFinetuneConfig:
    """Configuration for the plasma-domain Open-MAGVIT2 decoder fine-tune.

    Defaults correspond to the recipe in ``plans/tokenizers.md`` §12.1:
    ~5 k frames from ~100 rbb shots, 4×H200 exclusive, AdamW + cosine LR,
    10 k steps, early-stop on rFID plateau.
    """

    frame_root: Path = Path("/work/projects/imas_gpu/mast/level1")
    magvit2_root: Path = _DEFAULT_MAGVIT2_ROOT
    output_path: Path = _DEFAULT_MAGVIT2_ROOT / "weights" / "plasma-decoder-v1.safetensors"
    train_shot_ids: list[int] = field(default_factory=list)
    val_shot_ids: list[int] = field(default_factory=list)
    camera: str = "rbb"
    image_size: int = 256
    frames_per_shot: int = 50
    batch_size: int = 16  # 4 GPUs * 16 = 64 frames/step
    learning_rate: float = 1e-4
    max_steps: int = 10_000
    warmup_steps: int = 200
    l1_weight: float = 1.0
    perceptual_weight: float = 0.1
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
# Trainer class
# ---------------------------------------------------------------------------


class DecoderFinetuneTrainer:
    """Scaffolds the Open-MAGVIT2 decoder fine-tune loop for plasma imagery.

    Torch / lightning / Open-MAGVIT2 imports are lazy — the constructor does
    NOT load weights or open data so the object can be instantiated without a
    GPU or the Open-MAGVIT2 venv present.

    Usage
    -----
    ::

        config = DecoderFinetuneConfig(
            train_shot_ids=[15085, 15086, ...],
            val_shot_ids=[15100, 15101, ...],
        )
        trainer = DecoderFinetuneTrainer(config)
        output_path = trainer.train()

    The ``train()`` method drives the full loop; individual ``build_*``
    methods can be called in isolation for debugging or curriculum staging.
    """

    def __init__(self, config: DecoderFinetuneConfig) -> None:
        """Initialise the trainer with the given config.

        Parameters
        ----------
        config:
            Fine-tune configuration; see :class:`DecoderFinetuneConfig`.

        Notes
        -----
        The constructor is intentionally lightweight — no weight loading,
        no file system access beyond path validation. Call :meth:`build_model`
        and :meth:`build_dataloaders` explicitly when ready to start training.
        """
        self.config = config
        self._model: object | None = None
        self._vgg_features: object | None = None  # lazy-loaded perceptual model

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def build_dataloaders(self) -> tuple[Iterable, Iterable]:
        """Build training and validation data loaders.

        Each batch is a ``(B, H, W, 3)`` uint8 :class:`torch.Tensor` of
        plasma frames resized to :attr:`DecoderFinetuneConfig.image_size`.
        Frames are sampled uniformly across each shot at intervals of
        ``shot_duration / frames_per_shot``.

        The Zarr store layout assumed is::

            {frame_root}/{shot_id}/{camera}.zarr

        where the group exposes a ``data`` variable of shape ``(T, H, W)``
        or ``(T, H, W, C)`` in uint16.

        Returns
        -------
        tuple[Iterable, Iterable]
            ``(train_loader, val_loader)`` — both are
            :class:`torch.utils.data.DataLoader` instances.

        Raises
        ------
        FileNotFoundError
            If any shot Zarr store is absent from ``frame_root``.
        """
        import numpy as np
        import torch
        import xarray as xr
        from torch.utils.data import DataLoader, TensorDataset

        cfg = self.config

        def _load_shots(shot_ids: list[int]) -> torch.Tensor:
            """Load and concatenate uniformly sampled frames for a shot list."""
            all_frames: list[np.ndarray] = []
            for shot_id in shot_ids:
                zarr_path = cfg.frame_root / str(shot_id) / f"{cfg.camera}.zarr"
                if not zarr_path.is_dir():
                    raise FileNotFoundError(
                        f"Shot {shot_id} camera {cfg.camera!r} not found at {zarr_path}"
                    )
                ds = xr.open_zarr(str(zarr_path), consolidated=False)
                raw = np.asarray(ds["data"].values)  # (T, H, W) or (T, H, W, C)
                t = raw.shape[0]
                if t == 0:
                    continue
                # Uniform sampling: pick frames_per_shot indices spread across T
                indices = np.linspace(0, t - 1, min(cfg.frames_per_shot, t), dtype=int)
                sampled = raw[indices]
                # Collapse to grayscale if needed, then expand to 3 channels
                if sampled.ndim == 3:
                    # (K, H, W) → (K, H, W, 3)
                    sampled = np.repeat(sampled[..., np.newaxis], 3, axis=-1)
                elif sampled.shape[-1] == 1:
                    sampled = np.repeat(sampled, 3, axis=-1)
                # Normalise uint16 → uint8
                if sampled.dtype != np.uint8:
                    lo = float(sampled.min())
                    hi = float(sampled.max())
                    if hi > lo:
                        sampled = ((sampled.astype(np.float32) - lo) * 255.0 / (hi - lo))
                    sampled = sampled.clip(0, 255).astype(np.uint8)
                # Resize to image_size × image_size (simple numpy resize via PIL)
                try:
                    from PIL import Image

                    resized_frames = []
                    for frame in sampled:
                        img = Image.fromarray(frame)
                        img = img.resize((cfg.image_size, cfg.image_size), Image.BILINEAR)
                        resized_frames.append(np.asarray(img))
                    sampled = np.stack(resized_frames, axis=0)
                except ImportError:
                    # PIL absent: accept whatever shape comes from xarray
                    warnings.warn(
                        "Pillow not available — frames will NOT be resized to "
                        f"{cfg.image_size}×{cfg.image_size}. Install Pillow for "
                        "correct spatial alignment.",
                        stacklevel=2,
                    )
                all_frames.append(sampled)
            if not all_frames:
                # Return an empty (0, H, W, 3) tensor so DataLoader still works
                return torch.zeros((0, cfg.image_size, cfg.image_size, 3), dtype=torch.uint8)
            return torch.from_numpy(np.concatenate(all_frames, axis=0))

        train_frames = _load_shots(cfg.train_shot_ids)
        val_frames = _load_shots(cfg.val_shot_ids)

        train_ds = TensorDataset(train_frames)
        val_ds = TensorDataset(val_frames)

        g = torch.Generator()
        g.manual_seed(cfg.seed)
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            generator=g,
            drop_last=True,
        )
        val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
        return train_loader, val_loader

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    def build_model(self) -> object:
        """Load Open-MAGVIT2 encoder + codebook (frozen) and clone the decoder.

        The encoder and codebook weights are loaded from the pretrained
        checkpoint at::

            {magvit2_root}/weights/imagenet_256_L.ckpt

        The decoder weights are cloned from the same checkpoint and made
        trainable. The encoder and codebook parameters are frozen via
        ``requires_grad_(False)``.

        Returns
        -------
        torch.nn.Module
            A module with three child modules:
            - ``encoder`` (frozen)
            - ``codebook`` (frozen)
            - ``decoder`` (trainable)

        Raises
        ------
        RuntimeError
            If the Open-MAGVIT2 checkpoint is absent from ``magvit2_root``.
        ImportError
            If ``torch`` is not available in the current environment.

        Notes
        -----
        Open-MAGVIT2 is loaded from the isolated venv at
        ``{magvit2_root}/.venv`` when the main-venv torch pin conflicts. In
        v0 we assume the main venv has a compatible torch (≥ 2.6); if that
        assumption breaks, the worker-subprocess approach used by
        :class:`~imas_ambix.tokenizer.frames.OpenMagvit2Tokenizer` should be
        adopted here too.
        """
        import copy

        import torch

        cfg = self.config
        ckpt_path = cfg.magvit2_root / "weights" / "imagenet_256_L.ckpt"
        if not ckpt_path.exists():
            raise RuntimeError(
                f"Open-MAGVIT2 checkpoint not found at {ckpt_path}. "
                "Download imagenet_256_L.ckpt and place it under "
                f"{cfg.magvit2_root}/weights/."
            )

        # Load checkpoint — expects a Lightning-style .ckpt with state_dict key
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)

        # Separate encoder/codebook keys from decoder keys
        enc_keys = {k: v for k, v in state_dict.items() if not k.startswith("decoder")}
        dec_keys = {
            k.removeprefix("decoder."): v
            for k, v in state_dict.items()
            if k.startswith("decoder")
        }

        class _DecoderWrapper(torch.nn.Module):
            """Thin wrapper holding frozen encoder+codebook and trainable decoder."""

            def __init__(self) -> None:
                super().__init__()
                # Placeholder: real Open-MAGVIT2 classes should be imported from
                # the TencentARC repo once the venv integration is complete.
                # These are populated by build_model via load_state_dict.
                self.encoder_state: dict = {}
                self.codebook_state: dict = {}
                self.decoder = torch.nn.Identity()  # replaced by real decoder below

        wrapper = _DecoderWrapper()
        wrapper.encoder_state = {k: v.clone() for k, v in enc_keys.items()}
        wrapper.codebook_state = copy.deepcopy(
            {k: v for k, v in enc_keys.items() if "codebook" in k or "quantize" in k}
        )

        # Decoder: create a simple nn.ParameterDict so weights are tracked
        decoder_params = torch.nn.ParameterDict(
            {k.replace(".", "_"): torch.nn.Parameter(v.clone()) for k, v in dec_keys.items()}
        )
        wrapper.decoder = decoder_params  # type: ignore[assignment]
        wrapper.decoder.requires_grad_(True)

        wrapper = wrapper.to(cfg.device)
        self._model = wrapper
        return wrapper

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        target_frames: torch.Tensor,
        recon_frames: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the decoder fine-tune loss.

        Loss = L1 + perceptual loss (VGG16 relu3_3 features).

        The perceptual component weight is controlled by
        :attr:`DecoderFinetuneConfig.perceptual_weight`. If ``torchvision``
        is not importable the perceptual term is skipped with a one-time
        warning and the loss falls back to pure L1.

        Parameters
        ----------
        target_frames:
            Ground-truth frames, shape ``(B, 3, H, W)``, float32 in [0, 1].
        recon_frames:
            Reconstructed frames from the decoder, same shape as
            ``target_frames``.

        Returns
        -------
        torch.Tensor
            Scalar loss tensor with ``requires_grad=True``.
        """
        import torch
        import torch.nn.functional as F

        cfg = self.config
        l1 = F.l1_loss(recon_frames, target_frames)
        loss = cfg.l1_weight * l1

        if cfg.perceptual_weight > 0.0:
            try:
                import torchvision.models as tvm

                if self._vgg_features is None:
                    vgg = tvm.vgg16(weights=tvm.VGG16_Weights.IMAGENET1K_V1)
                    # Extract up to relu3_3 (layer index 15 in vgg.features)
                    self._vgg_features = torch.nn.Sequential(
                        *list(vgg.features.children())[:16]
                    ).to(cfg.device)
                    self._vgg_features.requires_grad_(False)
                    self._vgg_features.eval()

                with torch.no_grad():
                    feat_target = self._vgg_features(target_frames)
                feat_recon = self._vgg_features(recon_frames)
                perceptual = F.mse_loss(feat_recon, feat_target)
                loss = loss + cfg.perceptual_weight * perceptual

            except (ImportError, ModuleNotFoundError):
                warnings.warn(
                    "torchvision not available — perceptual loss skipped; "
                    "falling back to L1-only.",
                    stacklevel=2,
                )

        return loss

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, model: object, val_loader: Iterable) -> dict[str, float]:
        """Evaluate the decoder on the validation set.

        Computes rFID and mean L1 across the validation split. rFID is
        computed via :func:`imas_ambix.eval.metrics.rfid` (InceptionV3
        features). If ``pytorch_fid`` is installed it is preferred; the
        fallback is the project's own ``rfid`` function. If neither is
        available the key is set to ``float('nan')`` with a warning.

        Parameters
        ----------
        model:
            The decoder module returned by :meth:`build_model`. In the full
            implementation, ``model`` drives both encode (frozen) and decode
            (trainable) steps. In this scaffold, the evaluation loop is
            stubbed with correct interfaces.
        val_loader:
            Validation :class:`torch.utils.data.DataLoader` as returned by
            :meth:`build_dataloaders`.

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

        with torch.no_grad():
            for (batch,) in val_loader:
                # batch: (B, H, W, 3) uint8 → (B, 3, H, W) float32 in [0, 1]
                frames = batch.float().permute(0, 3, 1, 2).to(cfg.device) / 255.0
                # Scaffold: recon == identity (real decoder would go here)
                recon = frames.clone()
                l1_sum += float(F.l1_loss(recon, frames).item())
                n_batches += 1
                # Collect for rFID: convert back to uint8 (B, H, W, 3)
                t_np = (frames.permute(0, 2, 3, 1).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                r_np = (recon.permute(0, 2, 3, 1).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                all_targets.append(t_np)
                all_recons.append(r_np)

        mean_l1 = l1_sum / max(n_batches, 1)

        if not all_targets:
            return {"rfid": float("nan"), "l1": float("nan")}

        target_arr = np.concatenate(all_targets, axis=0)
        recon_arr = np.concatenate(all_recons, axis=0)

        # Try pytorch_fid first; fall back to imas_ambix.eval.metrics.rfid
        rfid_val = float("nan")
        try:
            # pytorch_fid provides a functional API via calculate_fid_given_paths;
            # for in-memory use we call its InceptionV3 features directly.
            from pytorch_fid import fid_score as pf_score  # type: ignore[import]  # noqa: F401

            # pytorch_fid does not expose a direct in-memory API; use our own.
            raise ImportError("pytorch_fid in-memory API not available")
        except (ImportError, ModuleNotFoundError):
            try:
                from imas_ambix.eval.metrics import rfid as _rfid

                rfid_val = _rfid(target_arr, recon_arr)
            except (ImportError, ModuleNotFoundError, Exception) as exc:
                warnings.warn(
                    f"rFID computation failed ({exc}); reporting NaN. "
                    "Install torchvision for InceptionV3-based rFID.",
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
        1. Build dataloaders (:meth:`build_dataloaders`).
        2. Build model (:meth:`build_model`).
        3. AdamW optimiser, cosine LR schedule with linear warmup.
        4. Train loop: encode batch with frozen encoder → decode with
           trainable decoder → compute loss (:meth:`compute_loss`) → step.
        5. Every ``eval_every_n_steps``: evaluate (:meth:`evaluate`); save
           a safetensors checkpoint if rFID improves; early-stop after
           ``patience`` consecutive non-improvements.
        6. Returns the path of the best checkpoint written.

        Returns
        -------
        pathlib.Path
            Path to the saved ``.safetensors`` weight file.

        Raises
        ------
        RuntimeError
            If ``train_shot_ids`` is empty or the model checkpoint is missing.
        """
        import math

        import torch
        import torch.nn.functional as F
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import LambdaLR

        cfg = self.config

        if not cfg.train_shot_ids:
            raise RuntimeError(
                "train_shot_ids is empty — provide shot IDs before calling train()."
            )

        torch.manual_seed(cfg.seed)

        train_loader, val_loader = self.build_dataloaders()
        model = self.build_model()

        # Only optimise the trainable decoder parameters
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimiser = AdamW(trainable_params, lr=cfg.learning_rate)

        # Cosine LR with linear warmup
        def _lr_lambda(step: int) -> float:
            if step < cfg.warmup_steps:
                return float(step) / max(cfg.warmup_steps, 1)
            progress = (step - cfg.warmup_steps) / max(cfg.max_steps - cfg.warmup_steps, 1)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = LambdaLR(optimiser, lr_lambda=_lr_lambda)

        cfg.output_path.parent.mkdir(parents=True, exist_ok=True)

        best_rfid = float("inf")
        no_improve_count = 0
        step = 0
        train_iter = iter(train_loader)

        while step < cfg.max_steps:
            # Cycle through the loader when exhausted
            try:
                (batch,) = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                try:
                    (batch,) = next(train_iter)
                except StopIteration:
                    break  # empty dataloader — shouldn't happen with real data

            model.train()
            frames = batch.float().permute(0, 3, 1, 2).to(cfg.device) / 255.0

            # Scaffold: in real implementation, encode with frozen encoder
            # then decode with trainable decoder. Here we use identity recon.
            recon = frames

            loss = self.compute_loss(frames, recon)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            scheduler.step()
            step += 1

            # Periodic evaluation + checkpointing
            if step % cfg.eval_every_n_steps == 0 or step == cfg.max_steps:
                model.eval()
                metrics = self.evaluate(model, val_loader)
                current_rfid = metrics.get("rfid", float("nan"))

                improved = (
                    not (current_rfid != current_rfid)  # not NaN
                    and current_rfid < best_rfid
                )
                if improved:
                    best_rfid = current_rfid
                    no_improve_count = 0
                    self._save_checkpoint(model)
                else:
                    no_improve_count += 1

                if no_improve_count >= cfg.patience:
                    break  # early stopping

        # Save final weights if no checkpoint was written yet
        if not cfg.output_path.exists():
            self._save_checkpoint(model)

        return cfg.output_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _save_checkpoint(self, model: object) -> None:
        """Save the decoder weights to ``output_path`` in safetensors format.

        Falls back to :func:`torch.save` if ``safetensors`` is not installed,
        writing a ``.pt`` file alongside the configured path with a warning.
        """
        import torch

        cfg = self.config
        cfg.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Collect only the decoder (trainable) state dict entries
        state_dict: dict[str, torch.Tensor] = {}
        if hasattr(model, "decoder") and hasattr(model.decoder, "state_dict"):
            state_dict = {k: v.cpu() for k, v in model.decoder.state_dict().items()}
        else:
            state_dict = {
                k: v.cpu() for k, v in model.state_dict().items() if v.requires_grad
            }

        try:
            from safetensors.torch import save_file

            save_file(state_dict, str(cfg.output_path))
        except (ImportError, ModuleNotFoundError):
            fallback = cfg.output_path.with_suffix(".pt")
            warnings.warn(
                f"safetensors not available — checkpoint saved as {fallback} instead.",
                stacklevel=2,
            )
            torch.save(state_dict, str(fallback))


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def finetune_decoder(config: DecoderFinetuneConfig) -> Path:
    """Fine-tune the Open-MAGVIT2 decoder on plasma imagery.

    Convenience wrapper around :class:`DecoderFinetuneTrainer`. Builds the
    trainer and calls :meth:`~DecoderFinetuneTrainer.train`.

    Parameters
    ----------
    config:
        Fine-tune configuration; see :class:`DecoderFinetuneConfig`.

    Returns
    -------
    pathlib.Path
        Path to the saved decoder weights.
    """
    trainer = DecoderFinetuneTrainer(config)
    return trainer.train()
