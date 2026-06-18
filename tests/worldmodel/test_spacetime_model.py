"""Unit tests for the spatiotemporal camera transformer (CPU, synthetic).

These assert the load-bearing properties:

* **shapes** — the backbone preserves ``(B, T, S, d)`` and the head spans the
  full vocab;
* **temporal causality** — a frame's prediction does NOT depend on any future
  frame's tokens (the property that makes autoregressive generation valid and
  the bug-class the old bag-of-channels model could not even express);
* **spatial information survives** — perturbing ONE spatial token changes the
  per-token hidden states (no summing-away of spatial structure);
* **chunked NLL == full-logit cross-entropy** — the memory-safe head matches
  the reference exactly;
* **overfit drops the loss** — a tiny model memorises a tiny synthetic clip
  (end-to-end wiring proof).
"""

from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F  # noqa: N812

from imas_ambix.worldmodel.spacetime_model import (
    SpacetimeConfig,
    SpacetimeTransformer,
)


def _tiny_cfg(**kw) -> SpacetimeConfig:
    base = dict(
        vocab_size=64,
        grid_h=4,
        grid_w=4,
        max_frames=12,
        plan_vocab=16,
        plan_channels=2,
        d_model=32,
        n_layers=2,
        n_heads=4,
        d_ff=64,
        dropout=0.0,
    )
    base.update(kw)
    return SpacetimeConfig(**base)


def _rand_batch(cfg: SpacetimeConfig, *, b=2, t=5, p=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    frames = torch.randint(0, cfg.vocab_size, (b, t, cfg.n_spatial), generator=g)
    plan = torch.randint(0, cfg.plan_vocab, (b, p, cfg.plan_channels), generator=g)
    return {"frames": frames, "plan": plan}


def test_forward_shapes():
    cfg = _tiny_cfg()
    model = SpacetimeTransformer(cfg).eval()
    batch = _rand_batch(cfg)
    out = model(batch, return_logits=True)
    b, t, s = batch["frames"].shape
    assert out.hidden.shape == (b, t, s, cfg.d_model)
    assert out.logits.shape == (b, t, s, cfg.vocab_size)


def test_unconditioned_model_runs():
    """plan_channels=0 -> no plan prefix, still produces per-token hidden."""
    cfg = _tiny_cfg(plan_channels=0)
    model = SpacetimeTransformer(cfg).eval()
    frames = torch.randint(0, cfg.vocab_size, (2, 4, cfg.n_spatial))
    out = model(
        {"frames": frames, "plan": torch.zeros((2, 0, 0), dtype=torch.long)},
        return_logits=False,
    )
    assert out.hidden.shape == (2, 4, cfg.n_spatial, cfg.d_model)


def test_temporal_causality():
    """hidden[:, t] must NOT change when a FUTURE frame's tokens change.

    This is the property that makes next-frame autoregressive generation valid:
    the state used to predict frame t+1 (hidden[:, t]) is a function of frames
    <= t only.  Perturb frame t+1 and assert hidden[:, :t+1] is bit-identical.
    """
    cfg = _tiny_cfg()
    model = SpacetimeTransformer(cfg).eval()
    batch = _rand_batch(cfg, t=5, seed=1)
    with torch.no_grad():
        h0 = model._forward_tokens(batch["frames"], batch["plan"])
    # perturb the LAST frame's tokens only.
    f2 = batch["frames"].clone()
    t = f2.shape[1]
    f2[:, t - 1] = (f2[:, t - 1] + 7) % cfg.vocab_size
    with torch.no_grad():
        h1 = model._forward_tokens(f2, batch["plan"])
    # frames before the perturbed one must be identical (causal: no future leak)
    assert torch.allclose(h0[:, : t - 1], h1[:, : t - 1], atol=1e-5), (
        "future frame leaked into an earlier frame's hidden state — temporal "
        "attention is not causal"
    )
    # the perturbed frame's own hidden MAY differ (it sees its own tokens).
    assert not torch.allclose(h0[:, t - 1], h1[:, t - 1], atol=1e-5)


def test_spatial_information_survives():
    """Changing ONE spatial token changes the hidden states (no spatial sum).

    The old model summed all spatial tokens into one vector; a single-cell
    change there could be swamped.  Here per-token identity is kept, so a
    single-cell change must move that frame's hidden states.
    """
    cfg = _tiny_cfg()
    model = SpacetimeTransformer(cfg).eval()
    batch = _rand_batch(cfg, t=4, seed=2)
    with torch.no_grad():
        h0 = model._forward_tokens(batch["frames"], batch["plan"])
    f2 = batch["frames"].clone()
    f2[:, 1, 0] = (f2[:, 1, 0] + 1) % cfg.vocab_size  # one cell of frame 1
    with torch.no_grad():
        h1 = model._forward_tokens(f2, batch["plan"])
    # frame 1's hidden must change in many positions (spatial attention spreads
    # the single-cell change across the frame), not be summed away.
    diff = (h0[:, 1] - h1[:, 1]).abs().sum(dim=-1)  # (B, S)
    assert (diff > 1e-6).float().mean() > 0.5


def test_chunked_nll_matches_full():
    """chunked_nll == cross_entropy over full logits (next-frame, no context)."""
    cfg = _tiny_cfg()
    model = SpacetimeTransformer(cfg).eval()
    batch = _rand_batch(cfg, b=2, t=5, seed=3)
    with torch.no_grad():
        hidden = model._forward_tokens(batch["frames"], batch["plan"])
        # reference: full logits, next-frame CE, mean over all (frame, pos) pairs
        logits = model.head(hidden[:, :-1])  # (B, T-1, S, V) predicts frames 1..T-1
        target = batch["frames"][:, 1:]
        ref = F.cross_entropy(
            logits.reshape(-1, cfg.vocab_size), target.reshape(-1), reduction="mean"
        )
        chunked = model.chunked_nll(hidden, batch["frames"], chunk=7)
    assert torch.allclose(ref, chunked, atol=1e-5), (ref.item(), chunked.item())


def test_chunked_nll_context_window():
    """context_frames restricts scoring to forecast-window targets only."""
    cfg = _tiny_cfg()
    model = SpacetimeTransformer(cfg).eval()
    batch = _rand_batch(cfg, b=1, t=6, seed=4)
    ctx = 3
    with torch.no_grad():
        hidden = model._forward_tokens(batch["frames"], batch["plan"])
        # reference: only predictions of target-frame index >= ctx (frames 3..5)
        logits = model.head(hidden[:, :-1])  # predicts frames 1..5
        target = batch["frames"][:, 1:]
        tgt_idx = torch.arange(1, 6)
        keep = tgt_idx >= ctx
        ref = F.cross_entropy(
            logits[:, keep].reshape(-1, cfg.vocab_size),
            target[:, keep].reshape(-1),
            reduction="mean",
        )
        chunked = model.chunked_nll(
            hidden, batch["frames"], chunk=5, context_frames=ctx
        )
    assert torch.allclose(ref, chunked, atol=1e-5), (ref.item(), chunked.item())


def test_chunked_argmax_frame_matches_full():
    cfg = _tiny_cfg()
    model = SpacetimeTransformer(cfg).eval()
    batch = _rand_batch(cfg, b=2, t=4, seed=5)
    with torch.no_grad():
        hidden = model._forward_tokens(batch["frames"], batch["plan"])
        prev = hidden[:, 1]  # (B, S, d)
        ref = model.head(prev).argmax(dim=-1)  # (B, S)
        chunked = model.chunked_argmax_frame(prev, chunk=6)
    assert torch.equal(ref, chunked)


def test_overfit_drops_loss():
    """A tiny model memorises a tiny synthetic clip (end-to-end wiring proof)."""
    torch.manual_seed(0)
    cfg = _tiny_cfg(d_model=64, n_layers=2, d_ff=128)
    model = SpacetimeTransformer(cfg).train()
    # one fixed clip the model must memorise.
    frames = torch.randint(0, cfg.vocab_size, (1, 5, cfg.n_spatial))
    plan = torch.randint(0, cfg.plan_vocab, (1, 3, cfg.plan_channels))
    batch = {"frames": frames, "plan": plan}
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    first = None
    last = None
    for _step in range(120):
        opt.zero_grad(set_to_none=True)
        loss = model(batch, loss_spec={"chunk": 64})
        loss.backward()
        opt.step()
        val = float(loss.detach())
        if first is None:
            first = val
        last = val
    assert last < first * 0.2, f"loss did not drop enough: {first:.3f} -> {last:.3f}"


def test_max_frames_guard():
    cfg = _tiny_cfg(max_frames=4, plan_channels=0)
    model = SpacetimeTransformer(cfg).eval()
    frames = torch.randint(0, cfg.vocab_size, (1, 5, cfg.n_spatial))  # 5 > 4
    with pytest.raises(ValueError, match="max_frames"):
        model({"frames": frames, "plan": torch.zeros((1, 0, 0), dtype=torch.long)})


def test_num_parameters_positive():
    model = SpacetimeTransformer(_tiny_cfg())
    assert model.num_parameters() > 0
    # weight tying: head shares the token embedding weight (same storage).
    assert model.head.weight.data_ptr() == model.token_embed.weight.data_ptr()
