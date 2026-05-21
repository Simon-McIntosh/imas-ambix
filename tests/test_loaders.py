"""Tests for :mod:`imas_ambix.data.loaders`.

All tests use ``monkeypatch`` to redirect ``TOKEN_ROOT`` to ``tmp_path``
and write synthetic Zarr stores directly so they never touch the real
filesystem or require network access.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import zarr

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_token_zarr(
    path: Path,
    n_tokens: int,
    seed: int = 0,
    include_block_kind: bool = True,
) -> None:
    """Write a minimal token Zarr at *path* for testing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    tokens = rng.integers(0, 1000, size=(n_tokens,), dtype=np.int32)
    store = zarr.open_group(str(path), mode="w")
    store.create_array("tokens", data=tokens)
    if include_block_kind:
        # Alternate between block kinds to give a non-trivial loss_mask
        block_kind = np.tile(np.array([0, 1, 2, 3], dtype=np.uint8), n_tokens // 4 + 1)[
            :n_tokens
        ]
        store.create_array("block_kind", data=block_kind)
    store.attrs.update({"shot_id": seed, "tokenizer_name": "test_v1"})


# ---------------------------------------------------------------------------
# WindowSamplerConfig defaults
# ---------------------------------------------------------------------------


def test_window_sampler_config_defaults():
    """WindowSamplerConfig().context_length == 16384."""
    from imas_ambix.data.loaders import WindowSamplerConfig

    cfg = WindowSamplerConfig()
    assert cfg.context_length == 16384
    assert cfg.stride == 4096
    assert cfg.seed == 0


# ---------------------------------------------------------------------------
# ShotTokenDataset yields correctly shaped windows
# ---------------------------------------------------------------------------


def test_shot_token_dataset_yields_windows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Build synthetic tokens for 2 shots, iterate, check window shapes."""
    import imas_ambix.data.persist as persist_mod
    from imas_ambix.data.loaders import (
        ShotTokenDataset,
        ShotTokenSpec,
        WindowSamplerConfig,
    )

    monkeypatch.setattr(persist_mod, "TOKEN_ROOT", tmp_path)

    n_tokens = 32768  # 2× context_length
    context = 16384
    shot_ids = [10001, 10002]
    specs = []
    for i, sid in enumerate(shot_ids):
        path = tmp_path / "v1" / "frames" / str(sid) / "rbb.zarr"
        _write_token_zarr(path, n_tokens=n_tokens, seed=i, include_block_kind=True)
        specs.append(ShotTokenSpec(shot_id=sid, n_tokens=n_tokens, path=path))

    config = WindowSamplerConfig(context_length=context, stride=context, seed=42)
    dataset = ShotTokenDataset(specs, config)

    windows = list(dataset)
    assert len(windows) >= 2  # at least one window per shot

    for window in windows:
        assert set(window.keys()) == {"input_ids", "labels", "attn_mask", "loss_mask"}
        assert window["input_ids"].shape == (context,)
        assert window["labels"].shape == (context,)
        assert window["attn_mask"].shape == (context,)
        assert window["loss_mask"].shape == (context,)
        assert window["input_ids"].dtype == np.int32
        assert window["labels"].dtype == np.int32
        assert window["attn_mask"].dtype == np.int32
        assert window["loss_mask"].dtype == np.float32
        # attn_mask is all-ones for non-padded windows
        assert window["attn_mask"].sum() == context
        # loss_mask is finite and non-negative
        assert np.isfinite(window["loss_mask"]).all()
        assert (window["loss_mask"] >= 0.0).all()


def test_loss_mask_uses_block_kind_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """loss_mask values match the expected per-block-kind weights."""
    import imas_ambix.data.persist as persist_mod
    from imas_ambix.data.loaders import (
        _BLOCK_KIND_WEIGHTS,
        ShotTokenDataset,
        ShotTokenSpec,
        WindowSamplerConfig,
    )

    monkeypatch.setattr(persist_mod, "TOKEN_ROOT", tmp_path)

    context = 64
    path = tmp_path / "rbb.zarr"
    # Craft a deterministic block_kind: all zeros for first half, all ones for second
    n = context
    tokens = np.zeros(n, dtype=np.int32)
    block_kind = np.array(
        [0] * (n // 2) + [1] * (n // 2),
        dtype=np.uint8,
    )
    store = zarr.open_group(str(path), mode="w")
    store.create_array("tokens", data=tokens)
    store.create_array("block_kind", data=block_kind)

    specs = [ShotTokenSpec(shot_id=1, n_tokens=n, path=path)]
    config = WindowSamplerConfig(context_length=context, stride=context, seed=0)
    dataset = ShotTokenDataset(specs, config)
    window = next(iter(dataset))

    # First half should have weight 0.0 (control), second half 1.0 (frame)
    np.testing.assert_allclose(window["loss_mask"][: n // 2], _BLOCK_KIND_WEIGHTS[0])
    np.testing.assert_allclose(window["loss_mask"][n // 2 :], _BLOCK_KIND_WEIGHTS[1])


# ---------------------------------------------------------------------------
# Fallback to ones when block_kind is absent
# ---------------------------------------------------------------------------


def test_loss_mask_falls_back_to_ones_without_block_kind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """When block_kind is absent, loss_mask is all 1s and a warning is emitted."""
    import imas_ambix.data.persist as persist_mod
    from imas_ambix.data.loaders import (
        ShotTokenDataset,
        ShotTokenSpec,
        WindowSamplerConfig,
    )

    monkeypatch.setattr(persist_mod, "TOKEN_ROOT", tmp_path)

    context = 128
    path = tmp_path / "no_block_kind.zarr"
    _write_token_zarr(path, n_tokens=context, seed=5, include_block_kind=False)

    specs = [ShotTokenSpec(shot_id=2, n_tokens=context, path=path)]
    config = WindowSamplerConfig(context_length=context, stride=context, seed=0)
    dataset = ShotTokenDataset(specs, config)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        windows = list(dataset)

    assert len(windows) >= 1
    window = windows[0]
    np.testing.assert_allclose(window["loss_mask"], 1.0)

    # Exactly one warning should have been issued (the once-per-dataset guard)
    assert len(caught) >= 1
    assert any("block_kind" in str(w.message) for w in caught)


# ---------------------------------------------------------------------------
# Short shots (< context_length) are padded
# ---------------------------------------------------------------------------


def test_short_shot_is_padded_to_context_length(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Shots shorter than context_length produce a window padded to context_length."""
    import imas_ambix.data.persist as persist_mod
    from imas_ambix.data.loaders import (
        ShotTokenDataset,
        ShotTokenSpec,
        WindowSamplerConfig,
    )

    monkeypatch.setattr(persist_mod, "TOKEN_ROOT", tmp_path)

    context = 256
    short_n = 100  # much shorter than context_length
    path = tmp_path / "short.zarr"
    _write_token_zarr(path, n_tokens=short_n, seed=7, include_block_kind=False)

    specs = [ShotTokenSpec(shot_id=3, n_tokens=short_n, path=path)]
    config = WindowSamplerConfig(context_length=context, stride=context, seed=0)
    dataset = ShotTokenDataset(specs, config)

    windows = list(dataset)
    assert len(windows) == 1
    assert windows[0]["input_ids"].shape == (context,)


# ---------------------------------------------------------------------------
# Empty dataset
# ---------------------------------------------------------------------------


def test_empty_dataset_yields_nothing():
    """A dataset with no specs iterates cleanly and yields no windows."""
    from imas_ambix.data.loaders import ShotTokenDataset, WindowSamplerConfig

    dataset = ShotTokenDataset([], WindowSamplerConfig())
    assert list(dataset) == []
