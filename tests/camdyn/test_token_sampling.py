"""Tests for the truth-free sampled token decoders.

The decoders operate on the head's own per-bit logits (pure numpy, no torch,
no truth access).  These tests verify:

* MAP decode matches the renderer's per-bit-mode identity exactly.
* Both samplers are TRUTH-FREE (they take only bit_logits) and return valid
  global-id grids of the right shape / dtype.
* Confident logits → both samplers collapse onto the MAP id (no spurious
  structure when the head is certain).
* Hedged logits → sampling produces a SPREAD of ids around the MAP (the
  filament-recovery mechanism), while MAP stays a single id.
* The bit-beam candidates are real codebook ids near the MAP, ranked by the
  exact bit-factorised likelihood (the most-probable id under the head is the
  most-likely draw at low temperature).
* Determinism under a fixed seed; temperature monotonicity (hotter = more spread).
"""

from __future__ import annotations

import numpy as np
import pytest

from imas_ambix.camdyn.model import LFQ_BITS, bitwise_nll, token_ids_to_bits
from imas_ambix.camdyn.token_sampling import (
    DECODE_MODES,
    bernoulli_sample,
    bit_beam_sample,
    decode_tokens,
    map_decode,
)


def _grid_logits(rng, f=3, h=16, w=16, scale=1.0):
    return rng.standard_normal((f, h, w, LFQ_BITS)) * scale


def _confident_logits(ids):
    """Large-magnitude logits whose signs encode the given token ids exactly."""
    bb = token_ids_to_bits(ids, LFQ_BITS)
    return (2.0 * bb - 1.0) * 30.0


# ---------------------------------------------------------------------------
# MAP decode
# ---------------------------------------------------------------------------


def test_map_decode_matches_per_bit_mode():
    rng = np.random.default_rng(0)
    z = _grid_logits(rng)
    out = map_decode(z)
    shifts = np.arange(LFQ_BITS, dtype=np.int64)
    ref = ((z > 0.0).astype(np.int64) << shifts).sum(axis=-1)
    np.testing.assert_array_equal(out, ref)
    assert out.dtype == np.int64
    assert out.shape == z.shape[:-1]


def test_map_decode_recovers_confident_ids():
    ids = np.array([[[0, 1, 12345], [262143, 7, 99]]], dtype=np.int64)
    z = _confident_logits(ids)
    np.testing.assert_array_equal(map_decode(z), ids)


# ---------------------------------------------------------------------------
# Bernoulli sampler
# ---------------------------------------------------------------------------


def test_bernoulli_shape_dtype_and_range():
    rng = np.random.default_rng(1)
    z = _grid_logits(rng)
    out = bernoulli_sample(z, temperature=0.7, rng=rng)
    assert out.shape == z.shape[:-1]
    assert out.dtype == np.int64
    assert out.min() >= 0 and out.max() < (1 << LFQ_BITS)


def test_bernoulli_confident_collapses_to_map():
    ids = np.array([[[0, 1, 12345, 262143]]], dtype=np.int64)
    z = _confident_logits(ids)
    rng = np.random.default_rng(2)
    out = bernoulli_sample(z, temperature=1.0, rng=rng)
    np.testing.assert_array_equal(out, ids)  # certain head → no spread


def test_bernoulli_hedged_spreads_around_map():
    # near-zero logits = maximally hedged head; sampling MUST produce many ids
    z = np.zeros((1, 8, 8, LFQ_BITS))
    rng = np.random.default_rng(3)
    out = bernoulli_sample(z, temperature=1.0, rng=rng)
    assert np.unique(out).size > 1  # the mode is a single id; sampling spreads


def test_bernoulli_temperature_monotone_spread():
    rng = np.random.default_rng(4)
    z = _grid_logits(rng, f=4, scale=1.5)
    cold = bernoulli_sample(z, temperature=0.3, rng=np.random.default_rng(10))
    hot = bernoulli_sample(z, temperature=2.0, rng=np.random.default_rng(10))
    m = map_decode(z)
    # hotter sampling departs from the MAP id more often than colder
    assert (hot != m).mean() >= (cold != m).mean()


def test_bernoulli_deterministic_under_seed():
    rng = np.random.default_rng(5)
    z = _grid_logits(rng)
    a = bernoulli_sample(z, temperature=0.8, rng=np.random.default_rng(7))
    b = bernoulli_sample(z, temperature=0.8, rng=np.random.default_rng(7))
    np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# Bit-beam sampler (truth-free coherent joint)
# ---------------------------------------------------------------------------


def test_beam_shape_dtype_and_range():
    rng = np.random.default_rng(6)
    z = _grid_logits(rng)
    out = bit_beam_sample(z, temperature=1.0, n_expand_bits=6, rng=rng)
    assert out.shape == z.shape[:-1]
    assert out.dtype == np.int64
    assert out.min() >= 0 and out.max() < (1 << LFQ_BITS)


def test_beam_confident_collapses_to_map():
    ids = np.array([[[0, 1, 12345, 262143]]], dtype=np.int64)
    z = _confident_logits(ids)
    out = bit_beam_sample(
        z, temperature=1.0, n_expand_bits=8, rng=np.random.default_rng(8)
    )
    np.testing.assert_array_equal(out, ids)


def test_beam_candidates_are_near_map_in_hamming():
    """Beam draws flip only the head's most-uncertain bits → within n_expand of MAP."""
    rng = np.random.default_rng(9)
    z = _grid_logits(rng, f=2, scale=1.0)
    n_expand = 5
    m = map_decode(z)
    out = bit_beam_sample(
        z, temperature=1.5, n_expand_bits=n_expand, top_p=None, rng=rng
    )
    # Hamming distance between drawn id and MAP id must be <= n_expand bits
    diff = m ^ out
    hamming = ((diff[..., None] >> np.arange(LFQ_BITS)) & 1).sum(axis=-1)
    assert hamming.max() <= n_expand


def test_beam_zero_temperature_is_map():
    """At T→0 the beam snaps to the argmax candidate = the per-bit MAP id.

    The MAP id is the joint mode under bit-independence and is always a beam
    candidate (the empty bit-flip subset), so cold beam ≡ map_decode.
    """
    rng = np.random.default_rng(11)
    z = _grid_logits(rng, f=2, scale=2.0)  # moderately confident
    m = map_decode(z)
    out = bit_beam_sample(z, temperature=0.0, n_expand_bits=8, top_p=None, rng=rng)
    np.testing.assert_array_equal(out, m)


def test_beam_no_candidate_more_likely_than_map():
    """No beam draw is ever STRICTLY more likely than the MAP id (it is the mode)."""
    rng = np.random.default_rng(110)
    z = _grid_logits(rng, f=3, scale=1.5)
    m = map_decode(z)
    out = bit_beam_sample(z, temperature=1.0, n_expand_bits=8, rng=rng)
    nll_map = bitwise_nll(z, m)
    nll_out = bitwise_nll(z, out)
    # bit-independent mode → MAP NLL is the minimum; any draw is >= it
    assert (nll_out >= nll_map - 1e-9).all()


def test_beam_drawn_id_is_more_likely_than_random_far_id():
    """A beam draw has lower bitwise NLL than a random distant id (coherence)."""
    rng = np.random.default_rng(12)
    z = _grid_logits(rng, f=4, scale=1.0)
    drawn = bit_beam_sample(z, temperature=1.0, n_expand_bits=8, rng=rng)
    far = rng.integers(0, 1 << LFQ_BITS, size=drawn.shape, dtype=np.int64)
    nll_drawn = bitwise_nll(z, drawn).mean()
    nll_far = bitwise_nll(z, far).mean()
    assert nll_drawn < nll_far


def test_beam_deterministic_under_seed():
    rng = np.random.default_rng(13)
    z = _grid_logits(rng)
    a = bit_beam_sample(z, temperature=1.0, rng=np.random.default_rng(21))
    b = bit_beam_sample(z, temperature=1.0, rng=np.random.default_rng(21))
    np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# Dispatch + truth-free contract
# ---------------------------------------------------------------------------


def test_decode_tokens_dispatch_all_modes():
    rng = np.random.default_rng(14)
    z = _grid_logits(rng)
    single_pass = [m for m in DECODE_MODES if m != "maskgit"]
    for mode in single_pass:
        out = decode_tokens(z, mode, temperature=0.8, rng=np.random.default_rng(1))
        assert out.shape == z.shape[:-1]
        assert out.dtype == np.int64
    # maskgit is iterative — it cannot be produced from a static logits array and
    # must raise when routed through the single-pass dispatcher.
    with pytest.raises(ValueError, match="iterative"):
        decode_tokens(z, "maskgit", temperature=0.8)


def test_decode_tokens_map_is_deterministic_ignores_temp():
    rng = np.random.default_rng(15)
    z = _grid_logits(rng)
    a = decode_tokens(z, "map", temperature=0.6)
    b = decode_tokens(z, "map", temperature=1.0)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(a, map_decode(z))


def test_decode_tokens_rejects_unknown_mode():
    z = np.zeros((1, 2, 2, LFQ_BITS))
    with pytest.raises(ValueError, match="unknown decode mode"):
        decode_tokens(z, "argmax")


def test_samplers_take_only_bit_logits_no_truth():
    """Both samplers' signatures accept ONLY bit_logits + knobs — no truth arg.

    This is the shippability contract: a sampled decoder that needs the true
    tokens (like the oracle joint probe) is unshippable.
    """
    import inspect

    for fn in (bernoulli_sample, bit_beam_sample):
        params = list(inspect.signature(fn).parameters)
        assert params[0] == "bit_logits"
        assert not any("true" in p or "target" in p for p in params)
