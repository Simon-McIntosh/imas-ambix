"""Unit tests for the signal-conditioned spatiotemporal camera transformer (v2).

These assert the v2-specific load-bearing properties on top of the v1 contract:

* **shapes** — the backbone still preserves ``(B, T, S, d)`` with signal frames
  prepended + stripped;
* **temporal causality survives** — a camera frame's prediction still does NOT
  depend on any future camera frame (the conditioning prefix is causal context,
  never a future leak);
* **signal conditioning is LOAD-BEARING** — perturbing the signal tokens changes
  the camera-frame hidden states, and zeroing them gives a different loss (proves
  the signals genuinely feed the prediction, not silently dropped);
* **signal params get grad on a signal-less batch** — the zero-touch keeps every
  signal param in the autograd graph (DDP would hang otherwise);
* **the zero-touch does not change the signal-less prediction**;
* **collate** stacks per-stream signals + pads to the batch-max, presenting the
  full model stream set;
* **chunked NLL == full-logit CE** still holds with conditioning present;
* **overfit drops the loss** on a tiny synthetic clip with signals.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F  # noqa: N812

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
        signal_streams=(
            SignalStreamSpec("xma", vocab=8, channels=3),
            SignalStreamSpec("summary", vocab=20, channels=4),
        ),
    )
    base.update(kw)
    return SignalSpacetimeConfig(**base)


def _rand_batch(cfg: SignalSpacetimeConfig, *, b=2, t=5, p=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    frames = torch.randint(0, cfg.vocab_size, (b, t, cfg.n_spatial), generator=g)
    plan = torch.randint(0, cfg.plan_vocab, (b, p, cfg.plan_channels), generator=g)
    signals = {}
    for st in cfg.signal_streams:
        signals[st.name] = torch.randint(
            0, st.vocab, (b, cfg.n_signal_steps, st.channels), generator=g
        )
    return {"frames": frames, "plan": plan, "signals": signals}


def test_forward_shapes():
    cfg = _tiny_cfg()
    model = SignalSpacetimeTransformer(cfg).eval()
    batch = _rand_batch(cfg)
    out = model(batch, return_logits=True)
    b, t, s = batch["frames"].shape
    # only the CAMERA frames survive in the output (conditioning prefix stripped)
    assert out.hidden.shape == (b, t, s, cfg.d_model)
    assert out.logits.shape == (b, t, s, cfg.vocab_size)


def test_temporal_causality_with_signals():
    """Future CAMERA frame must not leak into an earlier frame's hidden."""
    cfg = _tiny_cfg()
    model = SignalSpacetimeTransformer(cfg).eval()
    batch = _rand_batch(cfg, t=5, seed=1)
    with torch.no_grad():
        h0 = model._forward_tokens(batch["frames"], batch["plan"], batch["signals"])
    f2 = batch["frames"].clone()
    t = f2.shape[1]
    f2[:, t - 1] = (f2[:, t - 1] + 7) % cfg.vocab_size
    with torch.no_grad():
        h1 = model._forward_tokens(f2, batch["plan"], batch["signals"])
    assert torch.allclose(h0[:, : t - 1], h1[:, : t - 1], atol=1e-5), (
        "future frame leaked into an earlier frame's hidden — temporal attention "
        "is not causal with the signal prefix in place"
    )
    assert not torch.allclose(h0[:, t - 1], h1[:, t - 1], atol=1e-5)


def test_signal_conditioning_changes_hidden():
    """Perturbing the signal tokens MUST change the camera hidden states.

    If the signal block were silently dropped (a wiring bug) the camera hidden
    would be invariant to the signal ids — assert it is NOT.
    """
    cfg = _tiny_cfg()
    model = SignalSpacetimeTransformer(cfg).eval()
    batch = _rand_batch(cfg, t=4, seed=2)
    with torch.no_grad():
        h0 = model._forward_tokens(batch["frames"], batch["plan"], batch["signals"])
    sig2 = {k: v.clone() for k, v in batch["signals"].items()}
    # perturb every signal token of one stream.
    sig2["xma"] = (sig2["xma"] + 1) % cfg.signal_streams[0].vocab
    with torch.no_grad():
        h1 = model._forward_tokens(batch["frames"], batch["plan"], sig2)
    diff = (h0 - h1).abs().sum().item()
    assert diff > 1e-4, (
        "camera hidden did not change when the signal tokens changed — the "
        "conditioning is not feeding the prediction (silently dropped)"
    )


def test_signal_ablation_loss_delta():
    """Zeroing the signals gives a measurably different loss (load-bearing)."""
    cfg = _tiny_cfg()
    torch.manual_seed(0)
    model = SignalSpacetimeTransformer(cfg).eval()
    # train a few steps so the signal embeddings carry non-trivial weight.
    batch = _rand_batch(cfg, b=2, t=5, seed=3)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    model.train()
    for _ in range(40):
        opt.zero_grad(set_to_none=True)
        loss = model(batch, loss_spec={"chunk": 64})
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        full = float(model(batch, loss_spec={"chunk": 64}))
        zeroed = dict(batch)
        zeroed["signals"] = {
            k: torch.zeros_like(v) for k, v in batch["signals"].items()
        }
        zero = float(model(zeroed, loss_spec={"chunk": 64}))
    assert abs(full - zero) > 1e-3, (
        f"loss unchanged when signals zeroed (full={full:.5f} zero={zero:.5f}) — "
        "the model is ignoring the conditioning"
    )


def test_signal_params_get_grad_on_signalless_batch():
    """Every signal param must receive a grad even with NO signals (DDP)."""
    cfg = _tiny_cfg()
    model = SignalSpacetimeTransformer(cfg).train()
    frames = torch.randint(0, cfg.vocab_size, (2, 4, cfg.n_spatial))
    plan = torch.randint(0, cfg.plan_vocab, (2, 3, cfg.plan_channels))
    # empty signals -> the zero-touch path must fire for every signal param.
    batch = {"frames": frames, "plan": plan, "signals": {}}
    loss = model(batch, loss_spec={"chunk": 4096, "context_frames": None})
    loss.backward()
    assert model.signal_marker.grad is not None
    for name in cfg.signal_streams:
        assert model.signal_embed[name.name].weight.grad is not None, name.name
        assert model.signal_channel_embed[name.name].grad is not None, name.name
        assert model.signal_type_embed[name.name].grad is not None, name.name


def test_signalless_zero_touch_does_not_change_prediction():
    """The zero-magnitude signal touch must not alter a signal-less forward."""
    cfg = _tiny_cfg()
    model = SignalSpacetimeTransformer(cfg).eval()
    frames = torch.randint(0, cfg.vocab_size, (2, 4, cfg.n_spatial))
    plan = torch.randint(0, cfg.plan_vocab, (2, 3, cfg.plan_channels))
    empty = {"frames": frames, "plan": plan, "signals": {}}
    with torch.no_grad():
        h = model._forward_tokens(empty["frames"], empty["plan"], empty["signals"])
        model.signal_marker.add_(3.14)
        model.signal_type_embed["xma"].add_(2.71)
        model.signal_embed["xma"].weight.add_(1.23)
        h2 = model._forward_tokens(empty["frames"], empty["plan"], empty["signals"])
    assert torch.allclose(h, h2, atol=1e-6), (
        "signal-less forward changed when signal params changed — the touch is "
        "not zero-magnitude"
    )


def test_chunked_nll_matches_full_with_signals():
    cfg = _tiny_cfg()
    model = SignalSpacetimeTransformer(cfg).eval()
    batch = _rand_batch(cfg, b=2, t=5, seed=4)
    with torch.no_grad():
        hidden = model._forward_tokens(batch["frames"], batch["plan"], batch["signals"])
        logits = model.head(hidden[:, :-1])
        target = batch["frames"][:, 1:]
        ref = F.cross_entropy(
            logits.reshape(-1, cfg.vocab_size), target.reshape(-1), reduction="mean"
        )
        chunked = model.chunked_nll(hidden, batch["frames"], chunk=7)
    assert torch.allclose(ref, chunked, atol=1e-5), (ref.item(), chunked.item())


def test_signalless_equivalent_to_v1_with_no_streams():
    """A config with NO signal streams behaves like the v1 plan-only model."""
    cfg = _tiny_cfg(signal_streams=())
    assert not cfg.has_signals
    model = SignalSpacetimeTransformer(cfg).eval()
    frames = torch.randint(0, cfg.vocab_size, (2, 4, cfg.n_spatial))
    plan = torch.randint(0, cfg.plan_vocab, (2, 3, cfg.plan_channels))
    out = model({"frames": frames, "plan": plan, "signals": {}}, return_logits=False)
    assert out.hidden.shape == (2, 4, cfg.n_spatial, cfg.d_model)


def test_max_frames_guard_counts_signal_prefix():
    """The prefix (signals + plan) counts against max_frames."""
    # 2 streams x 3 signal steps = 6 prefix + 3 plan + 4 cam = 13 > max_frames 10
    cfg = _tiny_cfg(max_frames=10)
    model = SignalSpacetimeTransformer(cfg).eval()
    batch = _rand_batch(cfg, b=1, t=4, p=3)
    import pytest

    with pytest.raises(ValueError, match="max_frames"):
        model(batch, return_logits=False)


def test_overfit_drops_loss_with_signals():
    torch.manual_seed(0)
    cfg = _tiny_cfg(d_model=64, n_layers=2, d_ff=128)
    model = SignalSpacetimeTransformer(cfg).train()
    batch = _rand_batch(cfg, b=1, t=5, p=3, seed=7)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    first = last = None
    for _ in range(120):
        opt.zero_grad(set_to_none=True)
        loss = model(batch, loss_spec={"chunk": 64})
        loss.backward()
        opt.step()
        val = float(loss.detach())
        first = val if first is None else first
        last = val
    assert last < first * 0.2, f"loss did not drop enough: {first:.3f} -> {last:.3f}"


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------


def test_collate_stacks_and_pads_signals():
    from imas_ambix.worldmodel.spacetime_dataset import SpacetimeSample
    from imas_ambix.worldmodel.spacetime_dataset_v2 import (
        SignalSpacetimeSample,
    )
    from imas_ambix.worldmodel.spacetime_train_v2 import collate_signal_windows

    def _mk(shot, xma_ch, with_summary):
        base = SpacetimeSample(
            shot_id=shot,
            camera="rbb",
            start_frame=0,
            frames=np.zeros((4, 16), dtype=np.int64),
            plan=np.zeros((3, 2), dtype=np.int64),
            frame_time=np.linspace(0, 1, 4),
            context_frames=2,
        )
        sigs = {"xma": np.ones((3, xma_ch), dtype=np.int64)}
        if with_summary:
            sigs["summary"] = np.full((3, 4), 2, dtype=np.int64)
        return SignalSpacetimeSample(base=base, signals=sigs)

    samples = [_mk(1, 3, True), _mk(2, 5, False)]
    batch = collate_signal_windows(samples, stream_names=["xma", "summary"])
    # xma padded to batch-max channels (5); shot 1's lanes 3..4 are PAD 0.
    assert batch["signals"]["xma"].shape == (2, 3, 5)
    assert int(batch["signals"]["xma"][0, 0, 4]) == 0  # padded lane
    assert int(batch["signals"]["xma"][0, 0, 0]) == 1  # real
    # summary present for shot 1 only -> shot 2's block is all-PAD.
    assert batch["signals"]["summary"].shape == (2, 3, 4)
    assert int(batch["signals"]["summary"][1].abs().sum()) == 0  # shot2 all PAD
    assert int(batch["signals"]["summary"][0, 0, 0]) == 2  # shot1 real


def test_config_roundtrip_records_streams():
    from imas_ambix.worldmodel.spacetime_train_v2 import _config_to_dict

    cfg = _tiny_cfg()
    d = _config_to_dict(cfg)
    assert d["n_signal_steps"] == cfg.n_signal_steps
    names = {s["name"] for s in d["signal_streams"]}
    assert names == {"xma", "summary"}
