"""Tests for imas_ambix.model (WhamConfig, WhamModel)."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# WhamConfig tests
# ---------------------------------------------------------------------------


def test_wham_config_125m_validates() -> None:
    from imas_ambix.model import WhamConfig

    cfg = WhamConfig.variant_125m()
    assert cfg.hidden_size == 768
    assert cfg.num_hidden_layers == 12
    assert cfg.num_attention_heads == 12
    assert cfg.vocab_size == 280_000
    assert cfg.max_position_embeddings == 16384
    assert cfg.hidden_size % cfg.num_attention_heads == 0
    # Loss weights
    assert cfg.w_frame == 1.0
    assert cfg.w_signal == 0.3
    assert cfg.w_action == 0.1
    assert cfg.w_control == 0.0
    # Quick sanity on estimate — 125M arch has ~328M total params because
    # the 280K vocab embedding table dominates (280K × 768 ≈ 215M).
    params = cfg.estimate_params()
    assert 200_000_000 < params < 500_000_000, f"125M estimate out of range: {params}"


def test_wham_config_500m_estimate() -> None:
    from imas_ambix.model import WhamConfig

    cfg = WhamConfig.variant_500m()
    params = cfg.estimate_params()
    # 500M arch has ~689M total params due to the 280K vocab embedding table.
    assert 500_000_000 <= params <= 800_000_000, (
        f"500M estimate out of expected range [500M, 800M]: {params}"
    )


def test_wham_config_invalid_head_count() -> None:
    from imas_ambix.model import WhamConfig

    with pytest.raises(ValueError, match="divisible by num_attention_heads"):
        WhamConfig(hidden_size=768, num_attention_heads=11)


def test_wham_config_to_llama_config() -> None:
    transformers = pytest.importorskip("transformers")
    from imas_ambix.model import WhamConfig

    cfg = WhamConfig.variant_125m()
    llama_cfg = cfg.to_llama_config()
    assert isinstance(llama_cfg, transformers.LlamaConfig)
    assert llama_cfg.hidden_size == cfg.hidden_size
    assert llama_cfg.num_hidden_layers == cfg.num_hidden_layers
    assert llama_cfg.vocab_size == cfg.vocab_size


# ---------------------------------------------------------------------------
# WhamModel tests — require both torch and transformers
# ---------------------------------------------------------------------------


def _tiny_config():
    """Return a tiny WhamConfig that instantiates quickly on CPU."""
    from imas_ambix.model import WhamConfig

    return WhamConfig(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        max_position_embeddings=128,
        vocab_size=1024,
    )


def test_wham_model_instantiates_125m() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from imas_ambix.model import WhamConfig, WhamModel

    cfg = WhamConfig.variant_125m()
    model = WhamModel.from_config(cfg)
    n = model.num_parameters()
    # 125M arch has ~328M total params due to the 280K vocab embedding table.
    assert n < 500_000_000, f"125M model has too many parameters: {n}"
    assert n > 0


def test_wham_model_forward_finite_loss() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from imas_ambix.model import WhamModel

    cfg = _tiny_config()
    model = WhamModel.from_config(cfg)
    model._model.eval()

    batch, seq_len = 2, 64
    input_ids = torch.randint(0, cfg.vocab_size, (batch, seq_len))
    labels = input_ids.clone()

    with torch.no_grad():
        result = model.forward(input_ids=input_ids, labels=labels)

    assert "loss" in result
    assert "logits" in result
    loss_val = result["loss"].item()
    assert not math.isnan(loss_val), "loss is NaN"
    assert loss_val != float("inf"), "loss is infinite"
    assert loss_val > 0.0


def test_wham_model_loss_mask() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from imas_ambix.model import WhamModel

    cfg = _tiny_config()
    model = WhamModel.from_config(cfg)
    model._model.eval()

    batch, seq_len = 2, 64
    input_ids = torch.randint(0, cfg.vocab_size, (batch, seq_len))
    labels = input_ids.clone()

    # Half-zero mask — only first half positions are counted
    half_mask = torch.ones(batch, seq_len)
    half_mask[:, seq_len // 2 :] = 0.0

    with torch.no_grad():
        result_unmasked = model.forward(input_ids=input_ids, labels=labels)
        result_masked = model.forward(
            input_ids=input_ids, labels=labels, loss_mask=half_mask
        )

    unmasked_loss = result_unmasked["loss"].item()
    masked_loss = result_masked["loss"].item()

    assert not math.isnan(unmasked_loss), "unmasked loss is NaN"
    assert not math.isnan(masked_loss), "masked loss is NaN"
    assert unmasked_loss != float("inf")
    assert masked_loss != float("inf")

    # The masked loss should differ from the unmasked loss because different
    # subsets of positions are averaged. Assert strict inequality to verify
    # the loss_mask code path has an effect.
    assert masked_loss != unmasked_loss, (
        "masked and unmasked losses are identical — loss_mask has no effect"
    )


# ---------------------------------------------------------------------------
# YAML config parsing
# ---------------------------------------------------------------------------


def test_yaml_configs_parse() -> None:
    omegaconf = pytest.importorskip("omegaconf")

    configs_dir = Path(__file__).parent.parent / "imas_ambix" / "train" / "configs"

    cfg_125m = omegaconf.OmegaConf.load(configs_dir / "v0-125m.yaml")
    assert "model" in cfg_125m
    assert "training" in cfg_125m
    assert "data" in cfg_125m
    assert "checkpoint" in cfg_125m
    assert "eval" in cfg_125m
    assert "wandb" in cfg_125m
    assert cfg_125m.training.peak_lr == pytest.approx(3.0e-4)
    assert cfg_125m.model.vocab_size == 280000

    cfg_500m = omegaconf.OmegaConf.load(configs_dir / "v0-500m.yaml")
    assert "model" in cfg_500m
    assert "training" in cfg_500m
    assert cfg_500m.training.peak_lr == pytest.approx(1.5e-4)
    assert cfg_500m.model.variant == "variant_500m"
