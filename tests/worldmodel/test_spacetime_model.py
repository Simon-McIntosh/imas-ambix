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


def test_plan_params_get_grad_on_planless_batch():
    """A plan-capable model must touch ALL plan params even with no plan.

    Under DDP the reducer requires every rank to use the same parameter set on
    each step.  If a rank's shard is all-plan-less and the plan params get no
    gradient, the ring desynchronises and hangs.  The model adds a zero-magnitude
    plan touch so the plan params are always in the graph: the prediction is
    unchanged (contribution is *0.0) but every plan param receives a (zero)
    gradient.  Assert the gradients EXIST (are not None) for a plan-less batch.
    """
    cfg = _tiny_cfg(plan_channels=2)  # plan-capable
    model = SpacetimeTransformer(cfg).train()
    frames = torch.randint(0, cfg.vocab_size, (2, 4, cfg.n_spatial))
    # empty plan -> _embed_plan returns None -> the zero-touch path must fire.
    batch = {"frames": frames, "plan": torch.zeros((2, 0, 0), dtype=torch.long)}
    loss = model(batch, loss_spec={"chunk": 4096, "context_frames": None})
    loss.backward()
    for name in ("plan_embed", "plan_channel_embed", "plan_marker", "cam_marker"):
        param = getattr(model, name)
        weight = param.weight if hasattr(param, "weight") else param
        assert weight.grad is not None, (
            f"{name} got no gradient on a plan-less batch — DDP would hang when a "
            "rank's shard happens to contain no plan"
        )


def test_planless_zero_touch_does_not_change_prediction():
    """The zero-magnitude plan touch must not alter the plan-less forward."""
    cfg = _tiny_cfg(plan_channels=2)
    model = SpacetimeTransformer(cfg).eval()
    frames = torch.randint(0, cfg.vocab_size, (2, 4, cfg.n_spatial), generator=None)
    empty = {"frames": frames, "plan": torch.zeros((2, 0, 0), dtype=torch.long)}
    with torch.no_grad():
        h = model._forward_tokens(empty["frames"], empty["plan"])
    # mutate the plan params; a *0.0 touch means the output is unaffected.
    with torch.no_grad():
        model.plan_marker.add_(3.14)
        model.plan_channel_embed.add_(2.71)
        h2 = model._forward_tokens(empty["frames"], empty["plan"])
    assert torch.allclose(h, h2, atol=1e-6), (
        "plan-less forward changed when plan params changed — the touch is not "
        "zero-magnitude"
    )


def test_multi_rank_init_fails_loud_when_cuda_unavailable(monkeypatch):
    """A torchrun (multi-rank) launch with no working CUDA must FAIL, not degrade.

    A ``--gres=gpu:N`` job that lands on a node whose GPUs cannot initialise
    (driver/runtime defect) used to take the ``else`` branch and silently train
    on CPU/gloo — unusably slow, and it masked the broken node for ~12 minutes
    before anyone noticed.  When ``env.enabled`` (GPUs were requested) and CUDA
    is unavailable, init must raise a clear error in seconds instead.
    """
    from imas_ambix.worldmodel.spacetime_train import DistEnv, _init_distributed

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    env = DistEnv(rank=0, local_rank=0, world_size=2)  # enabled (multi-rank)
    with pytest.raises(RuntimeError, match="(?i)cuda"):
        _init_distributed(env)


def test_single_proc_cpu_init_is_allowed_when_cuda_unavailable(monkeypatch):
    """A single-process (WORLD_SIZE==1) CPU run is still allowed with no CUDA.

    The guard must only fire under a multi-rank launch (env.enabled); a CPU
    smoke test or single-process run must remain a no-op, never raising.
    """
    import torch.distributed as dist

    from imas_ambix.worldmodel.spacetime_train import DistEnv, _init_distributed

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    env = DistEnv(rank=0, local_rank=0, world_size=1)  # NOT enabled
    _init_distributed(env)  # must NOT raise and must NOT create a process group
    assert not dist.is_initialized()
