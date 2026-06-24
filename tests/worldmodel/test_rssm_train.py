"""CPU smoke tests for the RSSM corpus trainer's core (no DDP / no real corpus).

These exercise the trainer's load-bearing, NON-distributed pieces on a tiny
synthetic batch (so they run in seconds on a CPU node):

* **train-step decreases the loss** — a few :func:`_rssm_train_step` calls on a
  fixed tiny batch drive the real optimiser path (autocast forward → ELBO loss →
  backprop → clip → step → scheduler) and the loss must be finite and DROP, proving
  the step wires the model + optimiser + scheduler together correctly;
* **checkpoint round-trips** — :func:`save_rssm_checkpoint` →
  :func:`load_rssm_model_from_checkpoint` rebuilds the IDENTICAL config + state, and
  ``warm_start_from_phase1`` loads the reusable weights from that checkpoint;
* **build_rssm_model** sizes the config from the corpus command + streams;
* **config serialise round-trips** the full :class:`RSSMConfig`.

The full DDP launcher (:func:`train_rssm_corpus`) is NOT run here — it needs the
real curated token corpus + GPUs.  These tests cover the per-step core it calls.
"""

from __future__ import annotations

import torch

from imas_ambix.worldmodel.rssm import RSSMConfig, RSSMWorldModel, SignalStreamSpec
from imas_ambix.worldmodel.rssm_train import (
    _rssm_config_from_dict,
    _rssm_config_to_dict,
    _rssm_train_step,
    build_rssm_model,
    load_rssm_model_from_checkpoint,
    save_rssm_checkpoint,
    warm_start_reusable_from_checkpoint,
)


def _tiny_cfg(**kw) -> RSSMConfig:
    base = dict(
        vocab_size=64,
        grid_h=4,
        grid_w=4,
        d_model=32,
        h_dim=48,
        s_dim=8,
        a_dim=16,
        cmd_hidden=32,
        latent_hidden=32,
        decoder_hidden=48,
        beta=1.0,
        free_bits=1.0,
        diagnostic_weight=0.5,
        signal_streams=(
            SignalStreamSpec("gas_injection", vocab=8, channels=2),
            SignalStreamSpec("xma", vocab=8, channels=3),
        ),
        actuator_channels=6,
        masked_command_indices=(0, 1),
    )
    base.update(kw)
    return RSSMConfig(**base)


def _rand_batch(cfg, *, b=2, t=6, pa=4, ps=4, seed=0):
    """A controllable-collate-shaped batch (frames + actuator + signals)."""
    g = torch.Generator().manual_seed(seed)
    frames = torch.randint(0, cfg.vocab_size, (b, t, cfg.n_spatial), generator=g)
    signals = {}
    for st in cfg.signal_streams:
        # local ids >= 1 (id 0 is PAD / ignore) so the diagnostic CE actually scores.
        signals[st.name] = torch.randint(1, st.vocab, (b, ps, st.channels), generator=g)
    actuator = {
        "values": torch.randn(b, pa, cfg.actuator_channels, generator=g),
        "missing": torch.zeros(b, pa, cfg.actuator_channels),
    }
    return {"frames": frames, "signals": signals, "actuator": actuator}


# ---------------------------------------------------------------------------
# train-step decreases the loss
# ---------------------------------------------------------------------------


def test_train_step_decreases_loss():
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    model = RSSMWorldModel(cfg)
    model.train()
    batch = _rand_batch(cfg, b=3, t=6, seed=1)
    dev = torch.device("cpu")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)

    first = None
    last = None
    for i in range(40):
        out = _rssm_train_step(
            model, batch, opt, chunk=64, grad_clip=1.0, device=dev, scheduler=None
        )
        loss = float(out.loss.detach())
        assert torch.isfinite(out.loss), f"non-finite loss at step {i}"
        assert torch.isfinite(out.kl)
        if first is None:
            first = loss
        last = loss
    assert last < first, f"loss did not decrease: first={first:.4f} last={last:.4f}"


def test_train_step_decreases_loss_with_action_contrastive():
    """The step still finite + decreasing with the action-contrastive term ON."""
    torch.manual_seed(0)
    cfg = _tiny_cfg(action_contrastive=True, action_contrastive_weight=1.0)
    model = RSSMWorldModel(cfg)
    assert model.has_action_contrastive
    model.train()
    batch = _rand_batch(cfg, b=3, t=6, seed=1)
    dev = torch.device("cpu")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)

    first = None
    last = None
    for i in range(40):
        out = _rssm_train_step(
            model, batch, opt, chunk=64, grad_clip=1.0, device=dev, scheduler=None
        )
        loss = float(out.loss.detach())
        assert torch.isfinite(out.loss), f"non-finite loss at step {i}"
        assert torch.isfinite(out.action_contrastive)
        if first is None:
            first = loss
        last = loss
    assert last < first, f"loss did not decrease: first={first:.4f} last={last:.4f}"
    # the contrastive component is finite + scored (> 0 on a multi-sample batch).
    assert float(out.action_contrastive) > 0.0


def test_corpus_log_line_has_ac_suffix():
    """The per-step corpus log line exposes the action-contrastive component as ac=."""
    import inspect

    from imas_ambix.worldmodel import rssm_train

    src = inspect.getsource(rssm_train.train_rssm_corpus)
    # the log line carries an ac=%.4f field (the new component) and reads it off the
    # RSSMOutput.action_contrastive component.
    assert "ac=%.4f" in src
    assert "out.action_contrastive" in src


def test_train_step_advances_scheduler():
    """The factored step advances a passed LR scheduler (the corpus path uses one)."""
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    model = RSSMWorldModel(cfg)
    model.train()
    batch = _rand_batch(cfg, b=2, t=5, seed=2)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    # a simple warmup factor schedule so the LR moves on .step().
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda=lambda s: min(1.0, (s + 1) / 5)
    )
    lr0 = opt.param_groups[0]["lr"]
    _rssm_train_step(
        model,
        batch,
        opt,
        chunk=64,
        grad_clip=1.0,
        device=torch.device("cpu"),
        scheduler=sched,
    )
    lr1 = opt.param_groups[0]["lr"]
    assert lr1 != lr0  # the scheduler advanced


# ---------------------------------------------------------------------------
# checkpoint round-trip
# ---------------------------------------------------------------------------


def test_checkpoint_round_trips(tmp_path):
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    model = RSSMWorldModel(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    # take a step so the state is non-trivial.
    batch = _rand_batch(cfg, seed=3)
    _rssm_train_step(
        model, batch, opt, chunk=64, grad_clip=1.0, device=torch.device("cpu")
    )

    path = save_rssm_checkpoint(
        tmp_path,
        model=model,
        optimizer=opt,
        step=7,
        extra={"stream_names": ["gas_injection", "xma"]},
        snapshot=False,
    )
    assert path.exists()

    loaded, payload = load_rssm_model_from_checkpoint(str(path))
    assert payload["step"] == 7
    assert payload["extra"]["stream_names"] == ["gas_injection", "xma"]
    # config matches.
    assert loaded.config.actuator_channels == cfg.actuator_channels
    assert loaded.config.h_dim == cfg.h_dim
    assert loaded.config.s_dim == cfg.s_dim
    assert tuple(loaded.config.masked_command_indices) == tuple(
        cfg.masked_command_indices
    )
    assert [s.name for s in loaded.config.signal_streams] == ["gas_injection", "xma"]
    # weights match exactly (state round-trip).
    src_sd = model.state_dict()
    dst_sd = loaded.state_dict()
    for k in src_sd:
        assert torch.equal(src_sd[k], dst_sd[k].to(src_sd[k].dtype)), f"mismatch {k}"


def test_warm_start_from_checkpoint(tmp_path):
    """A saved RSSM checkpoint warm-starts a fresh RSSM's reusable weights."""
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    trained = RSSMWorldModel(cfg)
    # perturb the token embed so the warm start is observable.
    with torch.no_grad():
        trained.token_embed.weight.add_(1.234)
        trained.head.weight = trained.token_embed.weight
    opt = torch.optim.AdamW(trained.parameters(), lr=1e-3)
    path = save_rssm_checkpoint(
        tmp_path, model=trained, optimizer=opt, step=1, snapshot=False
    )

    fresh = RSSMWorldModel(_tiny_cfg())
    # the trainer's robust warm start recognises the model_state_dict wrapper key
    # both controllable + RSSM checkpoints use (the model's own
    # warm_start_from_phase1 misses it — it would load 0 tensors).
    counts = warm_start_reusable_from_checkpoint(fresh, str(path))
    assert counts["loaded"] > 0
    # the reusable token embed loaded (the latent core stays fresh).
    assert torch.equal(fresh.token_embed.weight, trained.token_embed.weight)
    # head stays tied to token_embed after the load.
    assert fresh.head.weight.data_ptr() == fresh.token_embed.weight.data_ptr()


# ---------------------------------------------------------------------------
# build + config serialise
# ---------------------------------------------------------------------------


def test_build_rssm_model_sizes_config():
    streams = (
        SignalStreamSpec("interferometer", vocab=257, channels=4),
        SignalStreamSpec("xma", vocab=8, channels=6),
    )
    model = build_rssm_model(
        actuator_channels=23,
        signal_streams=streams,
        masked_command_indices=(13, 14, 22),
        h_dim=128,
        s_dim=16,
        beta=2.0,
        free_bits=0.5,
        diagnostic_weight=0.25,
        action_contrastive=True,
        action_contrastive_weight=0.75,
    )
    assert model.config.actuator_channels == 23
    assert model.config.h_dim == 128
    assert model.config.s_dim == 16
    assert model.config.beta == 2.0
    assert model.config.free_bits == 0.5
    assert model.config.diagnostic_weight == 0.25
    assert model.config.action_contrastive is True
    assert model.config.action_contrastive_weight == 0.75
    assert tuple(model.config.masked_command_indices) == (13, 14, 22)
    assert [s.name for s in model.config.signal_streams] == ["interferometer", "xma"]
    assert model.config.has_actuator
    assert model.config.has_diagnostics
    assert model.config.has_action_contrastive  # ON + has a command path


def test_config_dict_round_trips():
    cfg = _tiny_cfg(
        actuator_channels=23,
        masked_command_indices=(13, 14, 22),
        action_contrastive=True,
        action_contrastive_weight=0.75,
        contrastive_dim=64,
        action_contrastive_temperature=0.2,
    )
    d = _rssm_config_to_dict(cfg)
    back = _rssm_config_from_dict(d)
    assert back.actuator_channels == cfg.actuator_channels
    assert back.h_dim == cfg.h_dim
    assert back.s_dim == cfg.s_dim
    assert back.beta == cfg.beta
    assert back.free_bits == cfg.free_bits
    assert back.diagnostic_weight == cfg.diagnostic_weight
    assert back.action_contrastive == cfg.action_contrastive
    assert back.action_contrastive_weight == cfg.action_contrastive_weight
    assert back.contrastive_dim == cfg.contrastive_dim
    assert back.action_contrastive_temperature == cfg.action_contrastive_temperature
    assert tuple(back.masked_command_indices) == tuple(cfg.masked_command_indices)
    assert [s.name for s in back.signal_streams] == [s.name for s in cfg.signal_streams]
    assert [s.vocab for s in back.signal_streams] == [
        s.vocab for s in cfg.signal_streams
    ]
