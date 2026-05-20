"""Tests for imas_ambix.bench.loader."""

from __future__ import annotations

import pytest

from imas_ambix.bench.loader import bundled_config, load_bench_config


class TestLoadV0Rir25ShotConfig:
    def test_load_v0_rir_25shot_config(self):
        cfg, run_kwargs = load_bench_config(bundled_config("v0-rir-25shot"))
        assert cfg.tokenizer_kind == "frame"
        assert cfg.device == "cuda"
        assert cfg.metrics == ("psnr", "mae")
        assert run_kwargs["camera"] == "rir"
        assert len(run_kwargs["shot_ids"]) == 25


class TestLoadV0Rbb25ShotConfig:
    def test_load_v0_rbb_25shot_config(self):
        cfg, run_kwargs = load_bench_config(bundled_config("v0-rbb-25shot"))
        assert cfg.tokenizer_kind == "frame"
        assert cfg.device == "cuda"
        assert cfg.metrics == ("psnr", "mae")
        assert run_kwargs["camera"] == "rbb"
        assert len(run_kwargs["shot_ids"]) == 25


class TestFactoryIsCallable:
    def test_factory_is_callable(self):
        cfg, _ = load_bench_config(bundled_config("v0-rir-25shot"))
        assert callable(cfg.tokenizer_factory)


class TestLoadMissingFileRaises:
    def test_load_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_bench_config("/nonexistent/path/no-such-config.yaml")
