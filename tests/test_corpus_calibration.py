"""Tests for corpus-level (absolute / SI) signal calibration in the encoders.

The blocker being fixed: the signal tokenizers standardise PER-SHOT (z-score
within each shot's own mean/std), so a given physical value maps to a DIFFERENT
token in every shot — absolute magnitude is destroyed.  Corpus calibration
supplies a SHOT-CONSTANT mean/std, so the same physical value maps to the same
token everywhere.  These tests prove:

(a) absolute mode maps the SAME physical value to the SAME bin across two
    synthetic "shots" with different per-shot ranges (per-shot mode does NOT);
(b) round-trip decode recovers the value within one bin width;
(c) byte-identical fallback when no calibration is set.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from imas_ambix.calibration.signals import ChannelCalibration
from imas_ambix.tokenizer.patch_transformer import (
    PatchTokenizerConfig,
    PatchTransformerTokenizer,
)
from imas_ambix.tokenizer.signals import UniformQuantizer


@pytest.fixture
def fresh_registry():
    """Reset the shared registry singleton to an empty state for each test."""
    from imas_ambix.tokenizer import registry as singleton
    from imas_ambix.tokenizer.registry import CONTROL_RANGE

    saved_blocks = dict(singleton._blocks)
    saved_cursor = singleton._cursor
    singleton._blocks.clear()
    singleton._cursor = CONTROL_RANGE[1]
    yield singleton
    singleton._blocks.clear()
    singleton._blocks.update(saved_blocks)
    singleton._cursor = saved_cursor


def _cal(name: str, mean: float, std: float) -> ChannelCalibration:
    """A minimal ChannelCalibration carrying only the load-bearing mean/std."""
    return ChannelCalibration(
        name=name,
        mean=mean,
        std=std,
        min_value=mean - 5 * std,
        max_value=mean + 5 * std,
        q01=mean - 2 * std,
        q50=mean,
        q99=mean + 2 * std,
        n_samples=1000,
        n_shots=10,
    )


def _shot(name: str, values: np.ndarray) -> xr.Dataset:
    return xr.Dataset(
        {name: (("time",), values.astype(np.float64))},
        coords={"time": np.arange(values.size, dtype=float)},
    )


# ---------------------------------------------------------------------------
# (a) absolute mode: SAME physical value → SAME bin across shots
# ---------------------------------------------------------------------------


def test_absolute_mode_maps_same_value_to_same_bin(fresh_registry):
    """A fixed physical value lands on the SAME bin in two shots with
    different per-shot ranges — that is the whole point of corpus calibration.
    """
    name = "b_field"
    # The corpus distribution: mean 1.0, std 0.5 (e.g. tesla).
    cal = {name: _cal(name, mean=1.0, std=0.5)}

    # Two shots whose own ranges differ wildly, but both contain the probe
    # value 0.5 T at index 0.
    probe = 0.5
    shot_a = _shot(name, np.array([probe, 0.6, 0.7, 0.8]))  # tight range
    shot_b = _shot(name, np.array([probe, 3.0, -2.0, 5.0]))  # wide range

    tok = UniformQuantizer(n_bins=256)
    tok.set_calibration(cal)

    enc_a = tok.encode(shot_a)
    enc_b = tok.encode(shot_b)

    bin_a = int(enc_a.token_ids[0, 0])
    bin_b = int(enc_b.token_ids[0, 0])
    assert bin_a == bin_b, (
        f"absolute mode must map {probe} to the same bin in both shots, "
        f"got {bin_a} vs {bin_b}"
    )
    assert enc_a.metadata["calibration"] == "absolute"


def test_per_shot_mode_maps_same_value_to_different_bins(fresh_registry):
    """Per-shot mode (no calibration, no fit) maps the same physical value to
    DIFFERENT bins in two shots with different ranges — the bug being fixed.
    """
    name = "b_field"
    probe = 0.5
    shot_a = _shot(name, np.array([probe, 0.6, 0.7, 0.8]))
    shot_b = _shot(name, np.array([probe, 3.0, -2.0, 5.0]))

    tok = UniformQuantizer(n_bins=256)  # no set_calibration, no fit → per-shot

    enc_a = tok.encode(shot_a)
    enc_b = tok.encode(shot_b)

    bin_a = int(enc_a.token_ids[0, 0])
    bin_b = int(enc_b.token_ids[0, 0])
    assert bin_a != bin_b, (
        "per-shot mode is expected to map the same value to different bins "
        "in shots with different ranges (this is the magnitude-destroying bug)"
    )
    assert enc_a.metadata["calibration"] == "per_shot"


# ---------------------------------------------------------------------------
# (b) round-trip decode recovers the value within one bin width
# ---------------------------------------------------------------------------


def test_absolute_round_trip_within_one_bin(fresh_registry):
    name = "b_field"
    cal = {name: _cal(name, mean=1.0, std=0.5)}
    n_bins = 256
    clip_sigma = 4.0
    bin_width = (2 * clip_sigma * 0.5) / (n_bins - 1)  # in physical units

    values = np.array([0.5, 1.0, 1.5, 0.2, 1.8])
    ds = _shot(name, values)

    tok = UniformQuantizer(n_bins=n_bins, clip_sigma=clip_sigma)
    tok.set_calibration(cal)

    enc = tok.encode(ds)
    dec = tok.decode(enc)
    recovered = np.asarray(dec[name].values)

    assert np.all(np.abs(recovered - values) <= bin_width + 1e-9), (
        f"round-trip error exceeds one bin width ({bin_width:.4f}): "
        f"{np.abs(recovered - values)}"
    )


def test_absolute_decode_uses_corpus_stats_not_default(fresh_registry):
    """Decode must invert with the corpus stats (not the mean=0/std=1 default),
    so a non-zero corpus mean is recovered."""
    name = "n_e"
    cal = {name: _cal(name, mean=5.0e19, std=1.0e19)}
    values = np.array([4.0e19, 5.0e19, 6.0e19])
    ds = _shot(name, values)

    tok = UniformQuantizer(n_bins=512)
    tok.set_calibration(cal)
    dec = tok.decode(tok.encode(ds))
    recovered = np.asarray(dec[name].values)
    # Recovered values must be in the 1e19 regime, not collapsed to ~0.
    assert np.all(recovered > 1.0e19)
    assert np.allclose(recovered, values, rtol=0.05)


# ---------------------------------------------------------------------------
# (c) byte-identical fallback when no calibration is set
# ---------------------------------------------------------------------------


def test_uniform_quantizer_byte_identical_without_calibration(fresh_registry):
    """With no calibration set and no fit, encode is byte-identical to the
    pre-calibration behaviour (per-encode normalisation)."""
    name = "x"
    values = np.array([0.1, 0.5, -0.3, 0.9, 0.2, 0.7])
    ds = _shot(name, values)

    baseline = UniformQuantizer(n_bins=128)
    enc_base = baseline.encode(ds)

    # A second quantizer with set_calibration(None) explicitly called.
    with_none = UniformQuantizer(n_bins=128)
    with_none.set_calibration(None)
    enc_none = with_none.encode(ds)

    np.testing.assert_array_equal(enc_base.token_ids, enc_none.token_ids)
    assert enc_base.channel_names == enc_none.channel_names


def test_uniform_quantizer_warns_on_missing_channel(fresh_registry, caplog):
    """A channel absent from the calibration falls back to per-shot with a
    visible warning naming the channel (never silent)."""
    import logging

    cal = {"present": _cal("present", mean=0.0, std=1.0)}
    ds = xr.Dataset(
        {
            "present": (("time",), np.array([0.0, 1.0, 2.0])),
            "missing": (("time",), np.array([10.0, 20.0, 30.0])),
        },
        coords={"time": np.arange(3.0)},
    )
    tok = UniformQuantizer(n_bins=64)
    tok.set_calibration(cal)
    with caplog.at_level(logging.WARNING):
        tok.encode(ds)
    assert any("missing" in rec.getMessage() for rec in caplog.records)


# ---------------------------------------------------------------------------
# patch-transformer hook: byte-identity + absolute normalisation
# ---------------------------------------------------------------------------


def _patch_tok() -> PatchTransformerTokenizer:
    cfg = PatchTokenizerConfig(
        patch_size=4, seq_patches=4, d_model=16, n_heads=2, n_layers=1, use_stft=False
    )
    tok = PatchTransformerTokenizer(cfg=cfg, name="signal_test_patch", device="cpu")
    tok.fit([np.random.default_rng(0).standard_normal((3, 64))], epochs=2)
    return tok


def test_patch_normalise_byte_identical_without_calibration():
    """``_normalise`` with corpus_calibration=None is byte-identical to the
    original per-window z-score."""
    tok = _patch_tok()
    x = np.random.default_rng(1).standard_normal((3, 64)).astype(np.float32)

    z0, m0, s0 = tok._normalise(x, fit=False)
    z1, m1, s1 = tok._normalise(
        x, fit=False, channel_names=["a", "b", "c"], corpus_calibration=None
    )
    np.testing.assert_array_equal(z0, z1)
    np.testing.assert_array_equal(m0, m1)
    np.testing.assert_array_equal(s0, s1)


def test_patch_normalise_absolute_uses_corpus_stats():
    """In absolute mode each row is standardised against its CORPUS mean/std,
    so two windows with different per-window ranges produce the same z for the
    same physical value."""
    tok = _patch_tok()
    cal = {"a": _cal("a", mean=1.0, std=0.5)}

    probe = 0.5
    win_a = np.concatenate([[probe], np.linspace(0.6, 0.9, 63)])[None, :].astype(
        np.float32
    )
    win_b = np.concatenate([[probe], np.linspace(-5.0, 5.0, 63)])[None, :].astype(
        np.float32
    )

    z_a, _, _ = tok._normalise(
        win_a, fit=False, channel_names=["a"], corpus_calibration=cal
    )
    z_b, _, _ = tok._normalise(
        win_b, fit=False, channel_names=["a"], corpus_calibration=cal
    )
    # The probe sample (index 0) standardises to the same z in both windows.
    expected = (probe - 1.0) / 0.5
    assert z_a[0, 0] == pytest.approx(expected)
    assert z_b[0, 0] == pytest.approx(expected)
    assert z_a[0, 0] == pytest.approx(z_b[0, 0])
