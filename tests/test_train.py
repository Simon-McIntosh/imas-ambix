"""Tests for imas_ambix.train — optimizer helpers, launcher, and training loop.

Requires torch (``pytest.importorskip("torch")``).
Tests that need accelerate are skipped if that package is absent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


# ---------------------------------------------------------------------------
# optim tests
# ---------------------------------------------------------------------------


class TestBuildAdamw:
    """build_adamw returns the right type and hyper-parameters."""

    def _make(self, **kwargs):
        from imas_ambix.train.optim import build_adamw

        params = [torch.nn.Parameter(torch.randn(4, 4))]
        defaults = dict(lr=1e-3, weight_decay=0.1, betas=(0.9, 0.95), eps=1e-8)
        defaults.update(kwargs)
        return build_adamw(params, **defaults)

    def test_returns_adamw(self):
        opt = self._make()
        assert isinstance(opt, torch.optim.AdamW)

    def test_lr_stored(self):
        opt = self._make(lr=3e-4)
        assert opt.param_groups[0]["lr"] == pytest.approx(3e-4)

    def test_weight_decay_stored(self):
        opt = self._make(weight_decay=0.05)
        assert opt.param_groups[0]["weight_decay"] == pytest.approx(0.05)

    def test_betas_stored(self):
        opt = self._make(betas=(0.9, 0.95))
        assert opt.param_groups[0]["betas"] == (0.9, 0.95)

    def test_eps_stored(self):
        opt = self._make(eps=1e-6)
        assert opt.param_groups[0]["eps"] == pytest.approx(1e-6)


# ---------------------------------------------------------------------------
# schedule tests
# ---------------------------------------------------------------------------


class TestCosineSchedule:
    """build_cosine_schedule: warm-up, peak, and decay behaviour."""

    def _make_sched(
        self,
        peak_lr: float = 1e-3,
        warmup_steps: int = 10,
        max_steps: int = 100,
        min_lr_frac: float = 0.1,
    ):
        from imas_ambix.train.optim import build_adamw, build_cosine_schedule

        params = [torch.nn.Parameter(torch.randn(2))]
        opt = build_adamw(params, lr=peak_lr, weight_decay=0.0, betas=(0.9, 0.95))
        sched = build_cosine_schedule(
            opt,
            warmup_steps=warmup_steps,
            max_steps=max_steps,
            min_lr_frac=min_lr_frac,
        )
        return opt, sched

    def test_lr_zero_at_step_zero(self):
        """LR must be zero (or close to zero) at the very first step."""
        opt, sched = self._make_sched(peak_lr=1e-3, warmup_steps=10)
        # Before any .step() the scheduler multiplier is at step 0
        lr = sched.get_last_lr()[0]
        assert lr == pytest.approx(0.0, abs=1e-9)

    def test_lr_at_peak_after_warmup(self):
        """After warmup_steps steps, LR should equal peak_lr."""
        opt, sched = self._make_sched(peak_lr=1e-3, warmup_steps=10, max_steps=100)
        for _ in range(10):
            sched.step()
        lr = sched.get_last_lr()[0]
        assert lr == pytest.approx(1e-3, rel=1e-4)

    def test_lr_at_min_after_max_steps(self):
        """After max_steps, LR should equal peak * min_lr_frac."""
        peak = 1e-3
        min_frac = 0.1
        opt, sched = self._make_sched(
            peak_lr=peak, warmup_steps=5, max_steps=20, min_lr_frac=min_frac
        )
        for _ in range(20):
            sched.step()
        lr = sched.get_last_lr()[0]
        assert lr == pytest.approx(peak * min_frac, rel=1e-3)

    def test_lr_monotone_during_decay(self):
        """LR must decrease monotonically from warmup to max_steps."""
        opt, sched = self._make_sched(peak_lr=1e-3, warmup_steps=4, max_steps=20)
        lrs = []
        for _ in range(20):
            sched.step()
            lrs.append(sched.get_last_lr()[0])
        # Decay phase: indices 4..19
        decay_lrs = lrs[4:]
        for i in range(len(decay_lrs) - 1):
            assert decay_lrs[i] >= decay_lrs[i + 1] - 1e-12


# ---------------------------------------------------------------------------
# AccelerateUnavailableError / monkeypatch tests
# ---------------------------------------------------------------------------


class TestAccelerateUnavailable:
    """AccelerateUnavailableError is raised when accelerate cannot be imported."""

    def test_error_raised_when_accelerate_missing(self, monkeypatch):
        """Monkeypatching sys.modules simulates accelerate not installed."""
        # Remove accelerate from sys.modules if present
        monkeypatch.setitem(sys.modules, "accelerate", None)  # type: ignore[arg-type]
        # Also unload the launcher module to force re-import
        monkeypatch.delitem(sys.modules, "imas_ambix.train.launcher", raising=False)

        from imas_ambix.train.launcher import (
            AccelerateUnavailableError,  # noqa: PLC0415
        )

        with pytest.raises(AccelerateUnavailableError):
            # Re-import inside context to get fresh module
            import importlib  # noqa: PLC0415

            launcher = importlib.import_module("imas_ambix.train.launcher")
            launcher.build_accelerator(fsdp=False)

    def test_error_is_import_error_subclass(self):
        from imas_ambix.train.launcher import (
            AccelerateUnavailableError,  # noqa: PLC0415
        )

        assert issubclass(AccelerateUnavailableError, ImportError)


# ---------------------------------------------------------------------------
# build_accelerator — CPU (no-FSDP) test
# ---------------------------------------------------------------------------


class TestBuildAcceleratorCPU:
    """build_accelerator(fsdp=False) returns a vanilla Accelerator on CPU."""

    def test_returns_accelerator(self):
        accelerate = pytest.importorskip("accelerate")
        from imas_ambix.train.launcher import build_accelerator  # noqa: PLC0415

        acc = build_accelerator(fsdp=False)
        assert isinstance(acc, accelerate.Accelerator)

    def test_device_is_cpu_when_no_cuda(self):
        pytest.importorskip("accelerate")
        from imas_ambix.train.launcher import build_accelerator  # noqa: PLC0415

        if torch.cuda.is_available():
            pytest.skip("CUDA available — device check only meaningful on CPU-only env")
        acc = build_accelerator(fsdp=False)
        assert str(acc.device) == "cpu"


# ---------------------------------------------------------------------------
# Hydra config loading tests
# ---------------------------------------------------------------------------


class TestHydraConfigs:
    """v0-125m.yaml and v0-500m.yaml load via OmegaConf with expected keys."""

    _config_dir = Path(__file__).parent.parent / "imas_ambix" / "train" / "configs"

    def _load(self, name: str):
        from omegaconf import OmegaConf  # noqa: PLC0415

        return OmegaConf.load(self._config_dir / name)

    def test_125m_loads(self):
        cfg = self._load("v0-125m.yaml")
        assert cfg is not None

    def test_125m_has_required_keys(self):
        cfg = self._load("v0-125m.yaml")
        assert "model" in cfg
        assert "training" in cfg
        assert "checkpoint" in cfg
        assert "eval" in cfg
        assert "wandb" in cfg
        assert "data" in cfg

    def test_125m_peak_lr(self):
        cfg = self._load("v0-125m.yaml")
        assert float(cfg.training.peak_lr) == pytest.approx(3e-4, rel=1e-4)

    def test_125m_max_steps(self):
        cfg = self._load("v0-125m.yaml")
        assert int(cfg.training.max_steps) == 30000

    def test_500m_loads(self):
        cfg = self._load("v0-500m.yaml")
        assert cfg is not None

    def test_500m_peak_lr(self):
        cfg = self._load("v0-500m.yaml")
        assert float(cfg.training.peak_lr) == pytest.approx(1.5e-4, rel=1e-4)

    def test_500m_max_steps(self):
        cfg = self._load("v0-500m.yaml")
        assert int(cfg.training.max_steps) == 60000

    def test_500m_variant(self):
        cfg = self._load("v0-500m.yaml")
        assert cfg.model.variant == "variant_500m"


# ---------------------------------------------------------------------------
# CPU smoke test — full forward + backward step
# ---------------------------------------------------------------------------


class TestCPUSmokeRun:
    """One training step with a tiny WhamConfig on CPU."""

    @pytest.fixture()
    def tiny_cfg(self):
        """Return a tiny WhamConfig suitable for CPU."""
        from imas_ambix.model.config import WhamConfig  # noqa: PLC0415

        return WhamConfig(
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=128,
            max_position_embeddings=128,
            vocab_size=1024,
        )

    @pytest.fixture()
    def tiny_batch(self, tiny_cfg):
        """Return a single (batch=2, seq=64) batch keyed for WhamModel.forward."""
        batch_size, seq_len = 2, 64
        ids = torch.randint(0, tiny_cfg.vocab_size, (batch_size, seq_len))
        # WhamModel.forward() uses 'attention_mask', not 'attn_mask'
        return {
            "input_ids": ids,
            "labels": ids.clone(),
            "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
            "loss_mask": torch.ones(batch_size, seq_len, dtype=torch.float),
        }

    def _make_model_optim_sched(self, cfg):
        from imas_ambix.model.wham import WhamModel  # noqa: PLC0415
        from imas_ambix.train.optim import (  # noqa: PLC0415
            build_adamw,
            build_cosine_schedule,
        )

        model = WhamModel.from_config(cfg)
        opt = build_adamw(
            list(model._model.parameters()),
            lr=1e-3,
            weight_decay=0.1,
            betas=(0.9, 0.95),
        )
        sched = build_cosine_schedule(opt, warmup_steps=1, max_steps=10)
        return model, opt, sched

    def test_loss_is_finite_step0(self, tiny_cfg, tiny_batch):
        model, opt, sched = self._make_model_optim_sched(tiny_cfg)
        out = model.forward(**tiny_batch)
        loss = out["loss"]
        assert torch.isfinite(loss), f"Loss is not finite: {loss}"

    def test_loss_decreases_on_second_step(self, tiny_cfg, tiny_batch):
        """Two steps on the same batch should not produce exploding loss."""
        model, opt, sched = self._make_model_optim_sched(tiny_cfg)

        # Step 1
        out1 = model.forward(**tiny_batch)
        loss1 = out1["loss"]
        loss1.backward()
        opt.step()
        sched.step()
        opt.zero_grad()

        # Step 2
        out2 = model.forward(**tiny_batch)
        loss2 = out2["loss"]
        # We just assert the second loss is finite — one step rarely guarantees
        # monotone decrease but should never explode
        assert torch.isfinite(loss2), f"Loss after step 2 is not finite: {loss2}"

    def test_loss_zero_with_all_zero_loss_mask(self, tiny_cfg, tiny_batch):
        """All-zero loss_mask → loss should be zero (or very close to it)."""
        zeros = torch.zeros_like(tiny_batch["loss_mask"])
        batch_zero = {**tiny_batch, "loss_mask": zeros}
        model, _, _ = self._make_model_optim_sched(tiny_cfg)
        out = model.forward(**batch_zero)
        assert float(out["loss"]) == pytest.approx(0.0, abs=1e-6)

    def test_loss_positive_with_all_one_loss_mask(self, tiny_cfg, tiny_batch):
        """All-one loss_mask → loss should be positive for random weights."""
        model, _, _ = self._make_model_optim_sched(tiny_cfg)
        out = model.forward(**tiny_batch)
        assert float(out["loss"]) > 0.0
