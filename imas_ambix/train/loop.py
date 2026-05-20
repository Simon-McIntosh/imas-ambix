"""WHAM training loop — ``python -m imas_ambix.train.loop``.

Invocation
----------
Normal training (FSDP, GPU)::

    accelerate launch -m imas_ambix.train.loop --config-name v0-500m

CPU debug smoke-test (single step, no W&B, synthetic data)::

    python -m imas_ambix.train.loop --config-name v0-125m ++training.debug=true

Hydra overrides::

    python -m imas_ambix.train.loop --config-name v0-125m \
        ++training.max_steps=100 ++training.debug=true

All configuration is Hydra-managed from ``configs/v0-125m.yaml`` (and
``configs/v0-500m.yaml``).  The ``debug`` flag caps training to one step,
disables W&B (``WANDB_MODE=disabled``), and falls back to a
:class:`SyntheticTokenDataset` when the manifest path does not exist.
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# Python 3.14 / hydra-core 1.3.x compatibility shim                          #
# hydra 1.3.2 passes a LazyCompletionHelp object as the help= argument to    #
# argparse.add_argument, but Python 3.14 made argparse stricter and now       #
# raises ValueError for non-string help values.  Patch before hydra imports.  #
# --------------------------------------------------------------------------- #
import argparse as _argparse
import contextlib as _contextlib

_orig_check_help = _argparse.ArgumentParser._check_help  # type: ignore[attr-defined]


def _lenient_check_help(self, action):  # type: ignore[override]
    with _contextlib.suppress(ValueError, TypeError):
        _orig_check_help(self, action)


_argparse.ArgumentParser._check_help = _lenient_check_help  # type: ignore[attr-defined]
# --------------------------------------------------------------------------- #

import logging  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
import uuid  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import TYPE_CHECKING  # noqa: E402

import hydra  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

if TYPE_CHECKING:
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

log = logging.getLogger(__name__)

# How often to log loss + lr to W&B / console (in steps)
_LOG_EVERY = 10


# ---------------------------------------------------------------------------
# Synthetic dataset for CPU smoke tests
# ---------------------------------------------------------------------------


class SyntheticTokenDataset:
    """Tiny in-memory dataset for CPU smoke tests.

    Yields ``n_batches`` dicts, each with ``(B, L)`` random ``int32`` token
    tensors in the range ``[0, vocab_size)``.  All ``loss_mask`` values are
    1.0 so the loss is never trivially zero.

    Parameters
    ----------
    batch_size:
        Sequences per batch.
    seq_len:
        Tokens per sequence.
    vocab_size:
        Upper bound (exclusive) for random token ids.
    n_batches:
        Number of distinct batches the dataset cycles over (default 2).
    seed:
        Random seed for reproducibility.
    """

    def __init__(
        self,
        batch_size: int = 2,
        seq_len: int = 64,
        vocab_size: int = 1024,
        n_batches: int = 2,
        seed: int = 42,
    ) -> None:
        import numpy as np

        self._rng = np.random.default_rng(seed)
        self._batches: list[dict[str, np.ndarray]] = []
        for _ in range(n_batches):
            ids = self._rng.integers(
                0, vocab_size, size=(batch_size, seq_len), dtype=np.int32
            )
            self._batches.append(
                {
                    "input_ids": ids,
                    "labels": ids.copy(),
                    "attn_mask": np.ones((batch_size, seq_len), dtype=np.int32),
                    "loss_mask": np.ones((batch_size, seq_len), dtype=np.float32),
                }
            )

    def __iter__(self):  # noqa: ANN204
        yield from self._batches


# ---------------------------------------------------------------------------
# Collation: numpy dict → torch tensor dict
# ---------------------------------------------------------------------------


def _collate_numpy(batch: list[dict]) -> dict:
    """Collate a list of numpy-array dicts into a single torch tensor dict."""
    import numpy as np
    import torch

    keys = batch[0].keys()
    out: dict[str, torch.Tensor] = {}
    for k in keys:
        arr = np.stack([b[k] for b in batch], axis=0)
        if k in ("input_ids", "labels", "attn_mask"):
            out[k] = torch.from_numpy(arr).long()
        else:
            out[k] = torch.from_numpy(arr).float()
    return out


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------


def _build_dataset(
    cfg: DictConfig,
    *,
    debug: bool,
    micro_batch_size: int,
) -> DataLoader:
    """Build the data-loader.

    Falls back to :class:`SyntheticTokenDataset` when ``debug=True`` and
    the manifest path does not exist.
    """
    from torch.utils.data import DataLoader

    manifest_path = Path(str(cfg.data.manifest))
    use_synthetic = debug and not manifest_path.exists()

    if use_synthetic:
        log.info("Manifest not found — using SyntheticTokenDataset (debug mode)")
        dataset = SyntheticTokenDataset(
            batch_size=micro_batch_size,
            seq_len=64,
            vocab_size=getattr(cfg.model, "vocab_size", 1024),
            n_batches=4,
        )
        # The synthetic dataset yields pre-batched dicts; wrap in a DataLoader
        # with batch_size=None (each item is already a batch)
        return DataLoader(
            list(dataset),  # type: ignore[arg-type]
            batch_size=None,
            collate_fn=lambda x: {
                k: (
                    v.long() if k in ("input_ids", "labels", "attn_mask") else v.float()
                )
                for k, v in {
                    kk: __import__("torch").from_numpy(vv) for kk, vv in x.items()
                }.items()
            },
            shuffle=False,
            num_workers=0,
        )

    # Real dataset path
    from imas_ambix.data.loaders import (  # noqa: PLC0415
        ShotTokenDataset,
        WindowSamplerConfig,
    )

    sampler_cfg = WindowSamplerConfig(
        context_length=int(cfg.data.context_length),
        stride=int(cfg.data.window_stride),
    )
    dataset = ShotTokenDataset.from_manifest(manifest_path, sampler_cfg)

    return DataLoader(
        list(dataset),  # type: ignore[arg-type]
        batch_size=micro_batch_size,
        collate_fn=_collate_numpy,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------


def _build_model(cfg: DictConfig) -> tuple:
    """Build WhamModel + parameter list from Hydra model config."""
    from imas_ambix.model.config import WhamConfig  # noqa: PLC0415
    from imas_ambix.model.wham import WhamModel  # noqa: PLC0415

    variant: str = str(cfg.model.variant)
    build_fn = getattr(WhamConfig, variant, None)
    if build_fn is None:
        raise ValueError(f"Unknown WhamConfig variant: {variant!r}")
    wham_cfg: WhamConfig = build_fn()

    # Apply any per-key overrides from the Hydra config
    for key in (
        "vocab_size",
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "intermediate_size",
        "max_position_embeddings",
    ):
        val = OmegaConf.select(cfg.model, key)
        if val is not None:
            object.__setattr__(wham_cfg, key, val)

    model = WhamModel.from_config(wham_cfg)
    return model, wham_cfg


# ---------------------------------------------------------------------------
# W&B init
# ---------------------------------------------------------------------------


def _init_wandb(cfg: DictConfig, *, run_id: str, debug: bool) -> None:
    """Initialise W&B (no-op in debug mode)."""
    if debug:
        os.environ["WANDB_MODE"] = "disabled"

    try:
        import wandb  # noqa: PLC0415

        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity if cfg.wandb.entity else None,
            id=run_id,
            config=OmegaConf.to_container(cfg, resolve=True),
            resume="allow",
        )
    except ImportError:
        log.warning("wandb not installed — skipping W&B logging")


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


@hydra.main(version_base=None, config_path="configs", config_name="v0-125m")
def main(cfg: DictConfig) -> None:
    """Training entry-point driven by Hydra config.

    Parameters
    ----------
    cfg:
        Resolved Hydra ``DictConfig``.  The ``++training.debug=true``
        override caps training to 1 step, disables W&B, and falls back to
        :class:`SyntheticTokenDataset` when the manifest is absent.
    """
    import torch

    from imas_ambix.train.launcher import build_accelerator  # noqa: PLC0415
    from imas_ambix.train.optim import (  # noqa: PLC0415
        build_adamw,
        build_cosine_schedule,
    )

    debug: bool = bool(OmegaConf.select(cfg, "training.debug", default=False))
    if debug:
        log.info("Debug mode enabled — capping to 1 step, disabling W&B")
        os.environ["WANDB_MODE"] = "disabled"

    max_steps: int = 1 if debug else int(cfg.training.max_steps)
    micro_batch_size: int = int(cfg.training.micro_batch_size)
    peak_lr: float = float(cfg.training.peak_lr)
    warmup_frac: float = float(cfg.training.warmup_frac)
    warmup_steps: int = max(1, int(warmup_frac * max_steps))
    min_lr_frac: float = float(cfg.training.min_lr_frac)
    weight_decay: float = float(cfg.training.weight_decay)
    betas: tuple[float, float] = tuple(float(b) for b in cfg.training.betas)  # type: ignore[assignment]
    precision: str = str(cfg.training.precision)
    use_fsdp: bool = str(cfg.training.fsdp_sharding).lower() != "no_shard"
    activation_checkpoint: bool = bool(cfg.training.activation_checkpoint)
    checkpoint_root: str = str(cfg.checkpoint.root)
    checkpoint_every: int = int(cfg.checkpoint.every_n_steps)
    eval_every: int = int(cfg.eval.every_n_steps)

    run_id: str = uuid.uuid4().hex[:8]

    # ------------------------------------------------------------------ #
    # Model                                                               #
    # ------------------------------------------------------------------ #
    log.info("Building WhamModel …")
    model, wham_cfg = _build_model(cfg)
    n_params = model.num_parameters() / 1e6
    log.info("  variant=%s, params=%.1fM", cfg.model.variant, n_params)

    # ------------------------------------------------------------------ #
    # Optimizer + scheduler                                               #
    # ------------------------------------------------------------------ #
    optimizer = build_adamw(
        list(model._model.parameters()),
        lr=peak_lr,
        weight_decay=weight_decay,
        betas=betas,
    )
    scheduler = build_cosine_schedule(
        optimizer,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        min_lr_frac=min_lr_frac,
    )

    # ------------------------------------------------------------------ #
    # Data                                                                #
    # ------------------------------------------------------------------ #
    log.info("Building data-loader …")
    loader = _build_dataset(cfg, debug=debug, micro_batch_size=micro_batch_size)

    # ------------------------------------------------------------------ #
    # Accelerator                                                         #
    # ------------------------------------------------------------------ #
    log.info("Building accelerator (fsdp=%s, precision=%s) …", use_fsdp, precision)
    # On CPU / debug, force no FSDP and no mixed precision
    if debug or not torch.cuda.is_available():
        use_fsdp = False
        precision = "no"

    accelerator = build_accelerator(
        precision=precision,
        fsdp=use_fsdp,
        activation_checkpoint=activation_checkpoint,
    )

    # accelerator.prepare returns new objects; unpack carefully
    prepared_model, prepared_optim, prepared_loader = accelerator.prepare(
        model._model, optimizer, loader
    )
    # Re-wrap WhamModel shell around the prepared HF model
    model._model = prepared_model

    # ------------------------------------------------------------------ #
    # W&B                                                                 #
    # ------------------------------------------------------------------ #
    _init_wandb(cfg, run_id=run_id, debug=debug)

    # ------------------------------------------------------------------ #
    # Training loop                                                       #
    # ------------------------------------------------------------------ #
    log.info("Starting training loop (max_steps=%d) …", max_steps)
    loader_iter = iter(prepared_loader)
    t0 = time.monotonic()

    for step in range(max_steps):
        # ---- get batch ----
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(prepared_loader)
            batch = next(loader_iter)

        # Move to accelerator device if not already there
        batch = {
            k: v.to(accelerator.device) if hasattr(v, "to") else v
            for k, v in batch.items()
        }

        # ---- forward ----
        out = model.forward(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attn_mask"),
            labels=batch["labels"],
            loss_mask=batch.get("loss_mask"),
        )
        loss: torch.Tensor = out["loss"]

        # ---- backward ----
        accelerator.backward(loss)
        prepared_optim.step()
        scheduler.step()
        prepared_optim.zero_grad()

        loss_val: float = float(loss.detach().cpu())
        current_lr: float = scheduler.get_last_lr()[0]

        # ---- logging ----
        if step % _LOG_EVERY == 0 or step == max_steps - 1:
            elapsed = time.monotonic() - t0
            log.info(
                "step=%d  loss=%.4f  lr=%.2e  elapsed=%.1fs",
                step,
                loss_val,
                current_lr,
                elapsed,
            )
            try:
                import wandb  # noqa: PLC0415

                if wandb.run is not None:
                    wandb.log(
                        {"train/loss": loss_val, "train/lr": current_lr},
                        step=step,
                    )
            except ImportError:
                pass

        # ---- checkpoint ----
        if checkpoint_every > 0 and step > 0 and step % checkpoint_every == 0:
            ckpt_dir = f"{checkpoint_root}/{run_id}/step-{step}"
            log.info("Saving checkpoint to %s …", ckpt_dir)
            accelerator.save_state(ckpt_dir)

        # ---- eval (placeholder) ----
        if eval_every > 0 and step > 0 and step % eval_every == 0:
            # Placeholder: log a synthetic val_loss
            import torch as _torch  # noqa: PLC0415

            synthetic_val_loss = float(_torch.tensor(0.0))
            log.info("step=%d  val_loss=%.4f (placeholder)", step, synthetic_val_loss)
            try:
                import wandb  # noqa: PLC0415

                if wandb.run is not None:
                    wandb.log({"eval/val_loss": synthetic_val_loss}, step=step)
            except ImportError:
                pass

        # ---- debug early-exit ----
        if debug and step >= 0:
            log.info("Debug mode: stopping after step %d (loss=%.4f)", step, loss_val)
            break

    # ------------------------------------------------------------------ #
    # Final checkpoint                                                    #
    # ------------------------------------------------------------------ #
    final_dir = f"{checkpoint_root}/{run_id}/final"
    log.info("Saving final checkpoint to %s …", final_dir)
    try:
        accelerator.save_state(final_dir)
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not save final checkpoint: %s", exc)

    log.info("Training complete. run_id=%s", run_id)

    try:
        import wandb  # noqa: PLC0415

        if wandb.run is not None:
            wandb.finish()
    except ImportError:
        pass


if __name__ == "__main__":
    main()
