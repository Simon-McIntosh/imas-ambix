"""Unit tests for the temperature + nucleus (top-p) sampling decode.

The camera world model collapses to persistence under a greedy argmax rollout;
the sampling decode is the zero-training mode-escape.  These pin its load-bearing
properties:

* ``temperature <= 0`` reproduces the deterministic argmax (same entry point,
  greedy baseline still reachable);
* a tiny ``top_p`` degrades to argmax (the nucleus always keeps the top-1 token —
  the support is never emptied);
* sampling with a fixed generator is REPRODUCIBLE, and two different generator
  states give DIFFERENT frames (the draw is genuinely stochastic);
* the nucleus mask keeps exactly the smallest top-p prefix and zeroes the rest.
"""

from __future__ import annotations

import torch

from imas_ambix.worldmodel.spacetime_model import _nucleus_mask_logits
from imas_ambix.worldmodel.spacetime_model_v2 import (
    SignalSpacetimeConfig,
    SignalSpacetimeTransformer,
    SignalStreamSpec,
)


def _tiny_cfg(**kw) -> SignalSpacetimeConfig:
    base = dict(
        vocab_size=64,
        grid_h=4,
        grid_w=4,
        max_frames=24,
        plan_vocab=16,
        plan_channels=2,
        d_model=32,
        n_layers=2,
        n_heads=4,
        d_ff=64,
        dropout=0.0,
        n_signal_steps=3,
        signal_streams=(SignalStreamSpec("xma", vocab=8, channels=3),),
    )
    base.update(kw)
    return SignalSpacetimeConfig(**base)


def _hidden(model, cfg, *, b=2, t=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    frames = torch.randint(0, cfg.vocab_size, (b, t, cfg.n_spatial), generator=g)
    plan = torch.randint(0, cfg.plan_vocab, (b, 3, cfg.plan_channels), generator=g)
    signals = {
        "xma": torch.randint(0, 8, (b, cfg.n_signal_steps, 3), generator=g),
    }
    with torch.no_grad():
        return model._forward_tokens(frames, plan, signals)  # (B, T, S, d)


# ---------------------------------------------------------------------------
# nucleus mask
# ---------------------------------------------------------------------------


def test_nucleus_mask_keeps_top1_for_tiny_p():
    logits = torch.tensor([[3.0, 1.0, 0.5, -2.0]])
    masked = _nucleus_mask_logits(logits, top_p=1e-6)
    # only the argmax (index 0) survives
    assert torch.isfinite(masked[0, 0])
    assert torch.isinf(masked[0, 1:]).all()


def test_nucleus_mask_keeps_smallest_prefix_reaching_p():
    # softmax of [4,3,1,0] ~ [0.6439,0.2369,0.0321,0.0118] cumsum
    # ~[0.644,0.881,0.913,1.0]; top_p=0.85 -> keep first two only.
    logits = torch.tensor([[4.0, 3.0, 1.0, 0.0]])
    masked = _nucleus_mask_logits(logits, top_p=0.85)
    assert torch.isfinite(masked[0, :2]).all()
    assert torch.isinf(masked[0, 2:]).all()


def test_nucleus_mask_full_p_keeps_everything():
    logits = torch.tensor([[4.0, 3.0, 1.0, 0.0]])
    masked = _nucleus_mask_logits(logits, top_p=1.0)
    assert torch.isfinite(masked).all()


# ---------------------------------------------------------------------------
# chunked_sample_frame
# ---------------------------------------------------------------------------


def test_temperature_zero_is_argmax():
    cfg = _tiny_cfg()
    model = SignalSpacetimeTransformer(cfg).eval()
    h = _hidden(model, cfg, seed=1)
    greedy = model.chunked_argmax_frame(h[:, -1])
    via_sample = model.chunked_sample_frame(h[:, -1], temperature=0.0)
    assert torch.equal(greedy, via_sample)


def test_tiny_top_p_degrades_to_argmax():
    cfg = _tiny_cfg()
    model = SignalSpacetimeTransformer(cfg).eval()
    h = _hidden(model, cfg, seed=2)
    greedy = model.chunked_argmax_frame(h[:, -1])
    # top_p so small only the top-1 survives -> sample is deterministic argmax
    sampled = model.chunked_sample_frame(h[:, -1], temperature=1.0, top_p=1e-6)
    assert torch.equal(greedy, sampled)


def test_sampling_is_reproducible_with_generator():
    cfg = _tiny_cfg()
    model = SignalSpacetimeTransformer(cfg).eval()
    h = _hidden(model, cfg, seed=3)
    g1 = torch.Generator().manual_seed(123)
    g2 = torch.Generator().manual_seed(123)
    a = model.chunked_sample_frame(h[:, -1], temperature=1.5, top_p=0.95, generator=g1)
    b = model.chunked_sample_frame(h[:, -1], temperature=1.5, top_p=0.95, generator=g2)
    assert torch.equal(a, b)


def test_sampling_varies_across_seeds():
    cfg = _tiny_cfg(vocab_size=256)  # enough mass spread to differ
    model = SignalSpacetimeTransformer(cfg).eval()
    h = _hidden(model, cfg, seed=4)
    g1 = torch.Generator().manual_seed(1)
    g2 = torch.Generator().manual_seed(2)
    a = model.chunked_sample_frame(h[:, -1], temperature=2.0, top_p=1.0, generator=g1)
    b = model.chunked_sample_frame(h[:, -1], temperature=2.0, top_p=1.0, generator=g2)
    # at high temperature over many tokens the two draws should differ somewhere
    assert not torch.equal(a, b)


def test_sample_shape_and_chunking_consistency():
    cfg = _tiny_cfg()
    model = SignalSpacetimeTransformer(cfg).eval()
    h = _hidden(model, cfg, b=2, t=4, seed=5)
    out = model.chunked_sample_frame(h[:, -1], temperature=0.0, chunk=3)
    assert out.shape == (2, cfg.n_spatial)
    # chunk size must not change the (deterministic at T=0) result
    out_full = model.chunked_sample_frame(h[:, -1], temperature=0.0, chunk=10_000)
    assert torch.equal(out, out_full)
