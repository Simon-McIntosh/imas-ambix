"""Unit tests for the per-frame camera-history embedding bottleneck.

The bottleneck is the corrected controllability lever (the M4 gate's key miss):
it must corrupt the PAST CAMERA-FRAME EMBEDDINGS reaching the dynamics head,
independently per frame (Diffusion Forcing), leaving the forecast frames intact,
with a graded strength the model can condition on.  These assert exactly those
load-bearing properties.
"""

from __future__ import annotations

import torch

from imas_ambix.worldmodel.history_bottleneck import (
    HistoryBottleneckConfig,
    bottleneck_history_embeddings,
    sample_frame_strengths,
)


def _emb(b=3, t=6, s=8, d=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(b, t, s, d, generator=g)


def test_forecast_frames_are_never_touched():
    cfg = HistoryBottleneckConfig(noise_std=2.0, mask_prob=0.9, max_strength=1.0)
    cam = _emb()
    ctx = 3
    strengths, _ = sample_frame_strengths(
        cam.shape[0], ctx, cfg, generator=torch.Generator().manual_seed(1)
    )
    out = bottleneck_history_embeddings(
        cam, strengths, cfg, context_frames=ctx, generator=torch.Generator().manual_seed(2)
    )
    # forecast frames (>= ctx) are returned byte-identical.
    assert torch.equal(out[:, ctx:], cam[:, ctx:]), (
        "forecast-window frames were corrupted — they are the prediction target"
    )
    # context frames moved (with this aggressive config + non-zero strengths).
    assert not torch.equal(out[:, :ctx], cam[:, :ctx])


def test_disabled_is_identity():
    cfg = HistoryBottleneckConfig(noise_std=0.0, mask_prob=0.0, max_strength=1.0)
    assert not cfg.enabled
    cam = _emb()
    strengths = torch.ones(cam.shape[0], 3)
    out = bottleneck_history_embeddings(cam, strengths, cfg, context_frames=3)
    assert torch.equal(out, cam)


def test_zero_strength_is_clean_history():
    cfg = HistoryBottleneckConfig(noise_std=2.0, mask_prob=0.9, max_strength=1.0)
    cam = _emb()
    ctx = 3
    strengths = torch.zeros(cam.shape[0], ctx)  # all-clean
    out = bottleneck_history_embeddings(
        cam, strengths, cfg, context_frames=ctx, generator=torch.Generator().manual_seed(3)
    )
    # strength 0 => no noise, no mask => identity even though the config is hot.
    assert torch.allclose(out, cam, atol=1e-6), (
        "a strength-0 (clean) history must be unchanged — the regime inference uses"
    )


def test_independent_per_frame_strengths_differ_across_frames():
    cfg = HistoryBottleneckConfig(
        independent_per_frame=True, clean_fraction=0.0, min_strength=0.1, max_strength=1.0
    )
    strengths, bins = sample_frame_strengths(
        64, 5, cfg, generator=torch.Generator().manual_seed(4)
    )
    # across the 5 context frames the per-frame strength is not constant (Diffusion
    # Forcing) — std over the frame axis is non-trivial for most samples.
    per_sample_std = strengths.std(dim=1)
    assert float(per_sample_std.mean()) > 1e-3
    # bins are a monotone-ish code: bin 0 only where strength is 0.
    assert bins.min().item() >= 0 and bins.max().item() < cfg.levels


def test_shared_strength_is_constant_across_frames():
    cfg = HistoryBottleneckConfig(
        independent_per_frame=False, clean_fraction=0.0, min_strength=0.1, max_strength=1.0
    )
    strengths, _ = sample_frame_strengths(
        16, 5, cfg, generator=torch.Generator().manual_seed(5)
    )
    # all frames of a sample share one strength.
    assert torch.allclose(strengths.std(dim=1), torch.zeros(16), atol=1e-6)


def test_clean_fraction_forces_some_clean_samples():
    cfg = HistoryBottleneckConfig(clean_fraction=1.0, min_strength=0.5, max_strength=1.0)
    strengths, bins = sample_frame_strengths(
        32, 4, cfg, generator=torch.Generator().manual_seed(6)
    )
    # clean_fraction 1.0 => every sample is fully clean.
    assert torch.equal(strengths, torch.zeros_like(strengths))
    assert torch.equal(bins, torch.zeros_like(bins))


def test_higher_strength_moves_embedding_more_on_average():
    cfg = HistoryBottleneckConfig(noise_std=1.0, mask_prob=0.0, max_strength=1.0)
    cam = _emb(b=128, t=4, s=8, d=16, seed=7)
    ctx = 2
    gen = torch.Generator().manual_seed(8)
    low = torch.full((128, ctx), 0.1)
    high = torch.full((128, ctx), 1.0)
    out_low = bottleneck_history_embeddings(
        cam, low, cfg, context_frames=ctx, generator=torch.Generator().manual_seed(9)
    )
    out_high = bottleneck_history_embeddings(
        cam, high, cfg, context_frames=ctx, generator=torch.Generator().manual_seed(9)
    )
    dl = (out_low[:, :ctx] - cam[:, :ctx]).abs().mean()
    dh = (out_high[:, :ctx] - cam[:, :ctx]).abs().mean()
    assert float(dh) > float(dl) * 3.0, (
        "a higher corruption strength must move the history embedding more — the "
        "strength knob is not graded"
    )


def test_mask_leg_zeroes_value_content_at_full_strength():
    # mask_prob 1.0, noise off => full-strength frames are scaled by mask_scale (0).
    cfg = HistoryBottleneckConfig(
        noise_std=0.0, mask_prob=1.0, mask_scale=0.0, max_strength=1.0
    )
    cam = _emb(b=64, t=4, s=8, d=16, seed=10)
    ctx = 2
    strengths = torch.ones(64, ctx)  # full strength => mask prob 1.0
    out = bottleneck_history_embeddings(
        cam, strengths, cfg, context_frames=ctx, generator=torch.Generator().manual_seed(11)
    )
    # every context frame masked to zero.
    assert torch.allclose(out[:, :ctx], torch.zeros_like(out[:, :ctx]), atol=1e-6)
    # forecast frames intact.
    assert torch.equal(out[:, ctx:], cam[:, ctx:])
