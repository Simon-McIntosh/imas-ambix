"""Tests for the shared factorized ST-transformer + LFQ bit-head (D1/D2).

Two layers:

* Pure-numpy bit utilities + the bit→vocab adapter — verified against the
  vocab-agnostic D0 metrics so the documented mapping is exact (no torch).
* The torch model — forward shapes, the matched-arm temporal toggle
  (identical params; D1 carries zero cross-frame information), and the
  masked bitwise loss.  Skipped cleanly if torch is unavailable.
"""

from __future__ import annotations

import numpy as np
import pytest

from imas_ambix.camdyn import metrics
from imas_ambix.camdyn.model import (
    LFQ_BITS,
    N_COND_CHANNELS,
    CamdynConfig,
    bit_logits_to_token_logits,
    bitwise_nll,
    restricted_vocab_logits,
    score_window_bits,
    token_ids_to_bits,
)

torch = pytest.importorskip  # placeholder so the name exists for skips


# ---------------------------------------------------------------------------
# Pure-numpy bit utilities
# ---------------------------------------------------------------------------


def test_token_ids_to_bits_roundtrip():
    ids = np.array([0, 1, 2, 7, (1 << 18) - 1, 12345], dtype=np.int64)
    bits = token_ids_to_bits(ids, bits=LFQ_BITS)
    assert bits.shape == (ids.size, LFQ_BITS)
    # reconstruct id from bits
    powers = (2 ** np.arange(LFQ_BITS)).astype(np.int64)
    recon = (bits.astype(np.int64) * powers).sum(axis=-1)
    np.testing.assert_array_equal(recon, ids)


def test_bit_logits_to_token_logits_matches_independent_bits():
    """Adapter score for an id == sum of per-bit log-sigmoids (exact)."""
    rng = np.random.default_rng(0)
    bit_logits = rng.standard_normal(LFQ_BITS)
    candidates = np.array([0, 1, 5, 99, (1 << 18) - 1], dtype=np.int64)
    scores = bit_logits_to_token_logits(bit_logits, candidates)
    assert scores.shape == (candidates.size,)
    # reference: explicit per-bit log-sigmoid sum
    for k, v in enumerate(candidates):
        bb = token_ids_to_bits(np.array([v]), LFQ_BITS)[0]  # (bits,)
        s = 2.0 * bb - 1.0
        ref = np.sum(-np.logaddexp(0.0, -(s * bit_logits)))
        assert scores[k] == pytest.approx(ref, rel=1e-9, abs=1e-9)


def test_adapter_argmax_picks_highest_probability_id():
    """The id whose bits agree with the logit signs scores highest."""
    # strong logits: bit pattern of id 0b101010101010101010
    target_id = int("101010101010101010", 2)
    bb = token_ids_to_bits(np.array([target_id]), LFQ_BITS)[0]
    bit_logits = (2.0 * bb - 1.0) * 6.0  # large-magnitude, correct sign
    candidates = np.array([target_id, 0, (1 << 18) - 1, 7], dtype=np.int64)
    scores = bit_logits_to_token_logits(bit_logits, candidates)
    assert int(np.argmax(scores)) == 0  # target_id is candidates[0]


def test_bitwise_nll_is_full_vocab_and_nonnegative():
    rng = np.random.default_rng(1)
    bit_logits = rng.standard_normal((4, 4, LFQ_BITS))
    targets = rng.integers(0, 1 << 18, size=(4, 4), dtype=np.int64)
    nll = bitwise_nll(bit_logits, targets)
    assert nll.shape == (4, 4)
    assert np.all(nll >= 0.0)
    # perfect prediction → NLL → 0
    tgt = np.array([[12345]])
    bb = token_ids_to_bits(tgt, LFQ_BITS)
    conf = (2.0 * bb - 1.0) * 30.0
    assert float(bitwise_nll(conf, tgt)[0, 0]) == pytest.approx(0.0, abs=1e-10)


def test_restricted_vocab_logits_remaps_targets_and_feeds_metrics():
    """The adapter output drops straight into the D0 metric interface."""
    rng = np.random.default_rng(2)
    bit_logits = rng.standard_normal((10, LFQ_BITS))
    targets = rng.integers(0, 50, size=10, dtype=np.int64)  # small id range
    dense, remapped = restricted_vocab_logits(bit_logits, targets)
    K = np.unique(targets).size
    assert dense.shape == (10, K)
    assert remapped.shape == (10,)
    # metrics consume (logits[...,V], targets) — must run without error
    full = np.ones(10, dtype=bool)
    nll = metrics.masked_token_nll(dense, remapped, full)
    acc = metrics.masked_top1_accuracy(dense, remapped, full)
    assert np.isfinite(nll)
    assert 0.0 <= acc <= 1.0


def test_score_window_bits_perfect_prediction():
    """Confident-correct bit logits → ~0 NLL and 100% accuracy on masked set."""
    rng = np.random.default_rng(3)
    tokens = rng.integers(0, 1 << 18, size=(5, 16, 16), dtype=np.int64)
    bb = token_ids_to_bits(tokens, LFQ_BITS)
    bit_logits = (2.0 * bb - 1.0) * 30.0
    loss_mask = rng.random((5, 16, 16)) < 0.4
    sc = score_window_bits(bit_logits, tokens, loss_mask)
    assert sc.n == int(loss_mask.sum())
    assert sc.nll_per_token.max() == pytest.approx(0.0, abs=1e-6)
    assert sc.acc_per_token.mean() == pytest.approx(1.0)


def test_score_window_bits_empty_mask():
    tokens = np.zeros((3, 16, 16), dtype=np.int64)
    bit_logits = np.zeros((3, 16, 16, LFQ_BITS))
    sc = score_window_bits(bit_logits, tokens, np.zeros((3, 16, 16), bool))
    assert sc.n == 0


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_config_roundtrip_and_cond_default():
    cfg = CamdynConfig(temporal_attention=True, dim=64, n_layers=3)
    d = cfg.to_dict()
    cfg2 = CamdynConfig.from_dict(d)
    assert cfg2.temporal_attention is True
    assert cfg2.dim == 64
    assert cfg2.n_layers == 3
    assert cfg2.grid == (16, 16)
    # default conditioning channels == full actuator vector
    assert CamdynConfig().cond_channels == N_COND_CHANNELS


# ---------------------------------------------------------------------------
# torch model
# ---------------------------------------------------------------------------


def _torch_or_skip():
    return pytest.importorskip("torch")


def _tiny_cfg(temporal: bool) -> CamdynConfig:
    return CamdynConfig(
        temporal_attention=temporal,
        dim=32,
        n_layers=2,
        n_heads=4,
        mlp_ratio=2.0,
        n_frames=5,
        cond_channels=N_COND_CHANNELS,
    )


def _fake_batch(t, cfg, b=2, f=5):
    tokens = t.randint(0, 1 << 18, (b, f, 16, 16))
    visible = t.rand(b, f, 16, 16) < 0.5
    cond_values = t.randn(b, f, cfg.cond_channels)
    cond_missing = (t.rand(b, f, cfg.cond_channels) < 0.2).float()
    dt = t.full((b, f), 1.0 / 600.0)
    return tokens, visible, cond_values, cond_missing, dt


def test_model_forward_shape():
    t = _torch_or_skip()
    from imas_ambix.camdyn.model import CamdynModel

    cfg = _tiny_cfg(temporal=False)
    model = CamdynModel.from_config(cfg)
    batch = _fake_batch(t, cfg)
    out = model.module(*batch)
    assert tuple(out.shape) == (2, 5, 16, 16, LFQ_BITS)


def test_temporal_toggle_is_matched_arm_identical_params():
    """D1 (OFF) and D2 (ON) have the SAME parameter shapes + count."""
    t = _torch_or_skip()
    from imas_ambix.camdyn.model import CamdynModel

    m_off = CamdynModel.from_config(_tiny_cfg(False))
    m_on = CamdynModel.from_config(_tiny_cfg(True))
    assert m_off.num_parameters() == m_on.num_parameters()
    names_off = {n: tuple(p.shape) for n, p in m_off.module.named_parameters()}
    names_on = {n: tuple(p.shape) for n, p in m_on.module.named_parameters()}
    assert names_off == names_on


def test_temporal_off_has_no_cross_frame_information_flow():
    """D1 baseline: changing a future frame must NOT alter an earlier frame.

    With temporal attention OFF the per-frame prediction depends only on
    that frame's own (visible) tokens + conditioning — perturbing a later
    frame's tokens cannot change an earlier frame's logits.  This is the
    defining property of the spatial-inpainting baseline.
    """
    t = _torch_or_skip()
    from imas_ambix.camdyn.model import CamdynModel

    t.manual_seed(0)
    cfg = _tiny_cfg(temporal=False)
    model = CamdynModel.from_config(cfg)
    model.module.eval()
    tokens, visible, cv, cm, dt = _fake_batch(t, cfg, b=1, f=5)
    with t.no_grad():
        out_a = model.module(tokens, visible, cv, cm, dt)
        # perturb ONLY the last frame's tokens
        tokens2 = tokens.clone()
        tokens2[:, -1] = t.randint(0, 1 << 18, (1, 16, 16))
        out_b = model.module(tokens2, visible, cv, cm, dt)
    # frame 0..F-2 logits must be unchanged (no info leaked backward)
    assert t.allclose(out_a[:, :-1], out_b[:, :-1], atol=1e-5)
    # the perturbed frame DID change (sanity — the model is not degenerate)
    assert not t.allclose(out_a[:, -1], out_b[:, -1], atol=1e-5)


def test_temporal_on_does_flow_information_across_frames():
    """D2 dynamics: an earlier frame's prediction DOES see later/earlier frames.

    Causal temporal attention means frame i attends to frames <= i, so
    perturbing frame 0 must change frame F-1's logits (forward flow).
    """
    t = _torch_or_skip()
    from imas_ambix.camdyn.model import CamdynModel

    t.manual_seed(0)
    cfg = _tiny_cfg(temporal=True)
    model = CamdynModel.from_config(cfg)
    model.module.eval()
    tokens, visible, cv, cm, dt = _fake_batch(t, cfg, b=1, f=5)
    with t.no_grad():
        out_a = model.module(tokens, visible, cv, cm, dt)
        tokens2 = tokens.clone()
        tokens2[:, 0] = t.randint(0, 1 << 18, (1, 16, 16))
        out_b = model.module(tokens2, visible, cv, cm, dt)
    # the LAST frame must respond to a change in the FIRST frame (causal flow)
    assert not t.allclose(out_a[:, -1], out_b[:, -1], atol=1e-5)


def test_masked_bit_bce_runs_and_backprops():
    t = _torch_or_skip()
    from imas_ambix.camdyn.model import CamdynModel, masked_bit_bce

    cfg = _tiny_cfg(temporal=False)
    model = CamdynModel.from_config(cfg)
    tokens, visible, cv, cm, dt = _fake_batch(t, cfg)
    loss_mask = ~visible
    valid = t.ones(2, 5, dtype=t.bool)
    logits = model.module(tokens, visible, cv, cm, dt)
    loss = masked_bit_bce(logits, tokens, loss_mask, valid)
    assert loss.ndim == 0 and float(loss) > 0.0
    loss.backward()
    grads = [p.grad for p in model.module.parameters() if p.grad is not None]
    assert len(grads) > 0
