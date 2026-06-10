"""Tests for the Open-MAGVIT2 decoder fine-tune scaffold.

Covers:
- Config defaults match the plan recipe.
- The module imports cleanly even when torch is absent.
- The CLI ``finetune-decoder --dry-run`` subcommand exits 0 without training.
- The trainer can be constructed without GPU access (no eager weight loading).
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from unittest.mock import patch


# ---------------------------------------------------------------------------
# test_config_defaults
# ---------------------------------------------------------------------------


def test_config_defaults() -> None:
    """DecoderFinetuneConfig defaults match the plan §12.1 recipe."""
    from imas_ambix.tokenizer.finetune_decoder import DecoderFinetuneConfig

    cfg = DecoderFinetuneConfig()
    assert cfg.learning_rate == 1e-4, f"Expected lr=1e-4, got {cfg.learning_rate}"
    assert cfg.max_steps == 10_000, f"Expected max_steps=10000, got {cfg.max_steps}"
    assert cfg.batch_size == 16, f"Expected batch_size=16, got {cfg.batch_size}"
    assert cfg.l1_weight == 1.0, f"Expected l1_weight=1.0, got {cfg.l1_weight}"
    assert cfg.perceptual_weight == 0.1, (
        f"Expected perceptual_weight=0.1, got {cfg.perceptual_weight}"
    )
    assert cfg.warmup_steps == 200
    assert cfg.eval_every_n_steps == 1_000
    assert cfg.patience == 3
    assert cfg.camera == "rbb"
    assert cfg.image_size == 256
    assert cfg.frames_per_shot == 50
    assert cfg.train_shot_ids == []
    assert cfg.val_shot_ids == []


# ---------------------------------------------------------------------------
# test_module_imports_without_torch
# ---------------------------------------------------------------------------


def test_module_imports_without_torch() -> None:
    """Module imports cleanly even when torch is absent from sys.modules."""
    # Remove cached module and all sub-modules to force a fresh import
    mod_name = "imas_ambix.tokenizer.finetune_decoder"

    # Stash originals
    saved: dict[str, object] = {}
    to_mask = [k for k in sys.modules if k == "torch" or k.startswith("torch.")]
    for k in to_mask:
        saved[k] = sys.modules.pop(k)

    # Also remove the target module so it is re-imported fresh
    saved[mod_name] = sys.modules.pop(mod_name, None)  # type: ignore[assignment]

    try:
        # Mark torch as unavailable
        sys.modules["torch"] = None  # type: ignore[assignment]
        mod = importlib.import_module(mod_name)
        # The dataclass and class should still be accessible
        assert hasattr(mod, "DecoderFinetuneConfig")
        assert hasattr(mod, "DecoderFinetuneTrainer")
        assert hasattr(mod, "finetune_decoder")
    finally:
        # Restore sys.modules to its original state
        sys.modules.pop("torch", None)
        for k, v in saved.items():
            if v is not None:
                sys.modules[k] = v  # type: ignore[assignment]
            else:
                sys.modules.pop(k, None)
        # Force re-import of the real module on next use
        sys.modules.pop(mod_name, None)


# ---------------------------------------------------------------------------
# test_cli_dry_run
# ---------------------------------------------------------------------------


def test_cli_dry_run(tmp_path: Path) -> None:
    """``finetune-decoder --dry-run`` exits 0 and does not call finetune_decoder."""
    from click.testing import CliRunner

    from imas_ambix.tokenizer.cli import tokenize

    # Create minimal shot-ID files
    train_file = tmp_path / "train_ids.txt"
    val_file = tmp_path / "val_ids.txt"
    train_file.write_text("15085\n15086\n")
    val_file.write_text("15100\n")

    runner = CliRunner()

    with patch(
        "imas_ambix.tokenizer.finetune_decoder.finetune_decoder"
    ) as mock_train:
        result = runner.invoke(
            tokenize,
            [
                "finetune-decoder",
                "--train-shots",
                str(train_file),
                "--val-shots",
                str(val_file),
                "--dry-run",
            ],
        )

    assert result.exit_code == 0, (
        f"Expected exit 0 for --dry-run, got {result.exit_code}.\n"
        f"Output:\n{result.output}\n"
        f"Exception:\n{result.exception}"
    )
    mock_train.assert_not_called()
    # Config values should appear in dry-run output
    assert "learning_rate" in result.output
    assert "max_steps" in result.output


# ---------------------------------------------------------------------------
# test_trainer_constructs_without_gpu
# ---------------------------------------------------------------------------


def test_trainer_constructs_without_gpu() -> None:
    """DecoderFinetuneTrainer constructs without GPU, no eager weight loading."""
    from imas_ambix.tokenizer.finetune_decoder import (
        DecoderFinetuneConfig,
        DecoderFinetuneTrainer,
    )

    cfg = DecoderFinetuneConfig(
        train_shot_ids=[15085],
        val_shot_ids=[15100],
        device="cpu",
    )
    # Must not raise, not access filesystem, not load weights
    trainer = DecoderFinetuneTrainer(cfg)
    assert trainer.config is cfg
    assert trainer._model is None, "model should not be loaded in __init__"
    assert trainer._vgg_features is None, "VGG should not be loaded in __init__"
