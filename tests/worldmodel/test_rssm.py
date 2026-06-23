"""Unit tests for the command-conditioned recurrent latent-dynamics world model.

These assert the structural properties the RSSM is built to guarantee — the fix
for the token backbone's failed controllability gate:

* **shapes / finite ELBO** — a teacher-forced forward on a tiny synthetic batch
  returns a finite loss, a non-negative KL, and finite camera / diagnostic CE;
* **controllability BY CONSTRUCTION (the key test)** — because the command lives
  INSIDE the recurrent transition, perturbing the command (on the unmasked
  command columns) CHANGES the prior rollout's decoded frames; a held / zeroed
  command vs a different command yields DIFFERENT rollouts.  Contrast the token
  backbone, where the plan was an AdaLN side-input the model could ignore;
* **KL free-bits** — the per-dim KL floor keeps the reported KL from collapsing
  below the free-bits budget;
* **masked commands** — perturbing ONLY the masked (state) columns does NOT move
  the rollout (drive-from-commands);
* **warm start** — the token embed + camera head + diagnostic heads load from a
  tiny synthetic phase1-like state_dict (shape-matched); the latent core stays
  fresh;
* **overfit smoke** — a tiny model fits a tiny synthetic batch (camera CE drops
  substantially over a few hundred steps) and the KL stays finite / non-collapsed.
"""

from __future__ import annotations

import torch

from imas_ambix.worldmodel.rssm import (
    RSSMConfig,
    RSSMWorldModel,
    SignalStreamSpec,
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
    g = torch.Generator().manual_seed(seed)
    frames = torch.randint(0, cfg.vocab_size, (b, t, cfg.n_spatial), generator=g)
    signals = {}
    for st in cfg.signal_streams:
        signals[st.name] = torch.randint(0, st.vocab, (b, ps, st.channels), generator=g)
    actuator = {
        "values": torch.randn(b, pa, cfg.actuator_channels, generator=g),
        "missing": torch.zeros(b, pa, cfg.actuator_channels),
    }
    return {"frames": frames, "signals": signals, "actuator": actuator}


# ---------------------------------------------------------------------------
# shapes / finite ELBO
# ---------------------------------------------------------------------------


def test_forward_shapes_and_finite_elbo():
    cfg = _tiny_cfg()
    model = RSSMWorldModel(cfg).eval()
    batch = _rand_batch(cfg)
    out = model(batch)
    b, t, s = batch["frames"].shape
    assert out.h.shape == (b, t, cfg.h_dim)
    assert out.s.shape == (b, t, cfg.s_dim)
    assert torch.isfinite(out.loss)
    assert torch.isfinite(out.camera_ce)
    assert torch.isfinite(out.diagnostic_ce)
    assert torch.isfinite(out.kl)
    assert float(out.kl.detach()) >= 0.0
    # the loss must carry gradient to the latent core (the GRU) — proves the
    # command/recurrence is in the autograd graph, not a dead side branch.
    out.loss.backward()
    grad = model.gru.weight_ih.grad
    assert grad is not None
    assert float(grad.abs().sum()) > 0.0


def test_diagnostics_off_runs():
    cfg = _tiny_cfg(signal_streams=())
    model = RSSMWorldModel(cfg).eval()
    batch = _rand_batch(cfg)
    out = model(batch)
    assert torch.isfinite(out.loss)
    assert float(out.diagnostic_ce) == 0.0


def test_no_actuator_runs():
    cfg = _tiny_cfg(actuator_channels=0, masked_command_indices=())
    model = RSSMWorldModel(cfg).eval()
    batch = _rand_batch(cfg)
    out = model(batch)
    assert torch.isfinite(out.loss)


# ---------------------------------------------------------------------------
# controllability BY CONSTRUCTION (the key test)
# ---------------------------------------------------------------------------


def test_command_changes_prior_rollout():
    """A different command MUST move the prior rollout (by construction)."""
    cfg = _tiny_cfg()
    model = RSSMWorldModel(cfg).eval()
    b, tc, n_steps = 2, 3, 4
    g = torch.Generator().manual_seed(7)
    ctx = torch.randint(0, cfg.vocab_size, (b, tc, cfg.n_spatial), generator=g)
    total_t = tc + n_steps
    base = torch.randn(b, total_t, cfg.actuator_channels, generator=g)

    def roll(values):
        act = {"values": values, "missing": torch.zeros_like(values)}
        with torch.no_grad():
            return model.rollout_prior(ctx, act, n_steps, sample=False)

    # TRUE plan vs a HELD (zeroed) plan vs a DIFFERENT plan.
    r_true = roll(base)
    r_zero = roll(torch.zeros_like(base))
    r_diff = roll(base * -2.0 + 1.0)

    # Different commands => different latent rollouts.  This is the structural
    # guarantee: the command is INSIDE the GRU transition, so a different command
    # is a different recurrence and therefore a different (h, s) trajectory.  We
    # check BOTH the deterministic recurrent state h and the stochastic state s.
    assert not torch.allclose(r_true.h, r_zero.h)
    assert not torch.allclose(r_true.s, r_zero.s)
    assert not torch.allclose(r_true.h, r_diff.h)
    assert not torch.allclose(r_true.s, r_diff.s)

    # The decoded camera surface is a deterministic function of the latent, so a
    # moved latent moves the decoded per-token hiddens (the argmax-discretised
    # frames can tie at init on a tiny untrained model, so we score the continuous
    # decode output the frames are read off of — the genuine "decoded frames
    # change" surface).
    z_true = torch.cat([r_true.h, r_true.s], dim=-1)
    z_zero = torch.cat([r_zero.h, r_zero.s], dim=-1)
    with torch.no_grad():
        dec_true = model.decode_hidden(z_true)
        dec_zero = model.decode_hidden(z_zero)
    assert not torch.allclose(dec_true, dec_zero)


def test_masked_columns_do_not_drive():
    """Perturbing ONLY the masked (state) columns must NOT move the rollout."""
    cfg = _tiny_cfg(masked_command_indices=(0, 1))
    model = RSSMWorldModel(cfg).eval()
    b, tc, n_steps = 2, 3, 4
    g = torch.Generator().manual_seed(3)
    ctx = torch.randint(0, cfg.vocab_size, (b, tc, cfg.n_spatial), generator=g)
    total_t = tc + n_steps
    base = torch.randn(b, total_t, cfg.actuator_channels, generator=g)
    perturbed = base.clone()
    # move ONLY the masked columns — these are zeroed before conditioning, so the
    # rollout must be identical.
    perturbed[..., list(cfg.masked_command_indices)] += 5.0

    def roll(values):
        act = {"values": values, "missing": torch.zeros_like(values)}
        with torch.no_grad():
            return model.rollout_prior(ctx, act, n_steps, sample=False)

    assert torch.allclose(roll(base).s, roll(perturbed).s, atol=1e-5)
    # but a non-masked (command) column DOES move it.
    cmd_perturbed = base.clone()
    cmd_perturbed[..., 2] += 5.0
    assert not torch.allclose(roll(base).s, roll(cmd_perturbed).s)


# ---------------------------------------------------------------------------
# KL free-bits
# ---------------------------------------------------------------------------


def test_kl_free_bits_floor():
    """The per-dim free-bits floor keeps the reported KL from collapsing below it."""
    # an EXACT-match Gaussian pair has true KL 0 per dim; with free_bits the
    # reported KL is floored at free_bits * s_dim.
    s_dim = 8
    free_bits = 1.0
    qm = torch.zeros(2, 5, s_dim)
    qs = torch.ones(2, 5, s_dim)
    kl = RSSMWorldModel._kl_diag_gaussian(qm, qs, qm, qs, free_bits=free_bits)
    assert abs(float(kl) - free_bits * s_dim) < 1e-5
    # without free-bits the same pair gives KL 0.
    kl0 = RSSMWorldModel._kl_diag_gaussian(qm, qs, qm, qs, free_bits=0.0)
    assert abs(float(kl0)) < 1e-6
    # a genuinely divergent pair exceeds the floor.
    pm = qm + 3.0
    kl_big = RSSMWorldModel._kl_diag_gaussian(qm, qs, pm, qs, free_bits=free_bits)
    assert float(kl_big) > free_bits * s_dim


def test_forward_kl_at_or_above_floor():
    cfg = _tiny_cfg(free_bits=1.0)
    model = RSSMWorldModel(cfg).eval()
    out = model(_rand_batch(cfg))
    # KL is summed over s_dim with each dim floored at free_bits.
    assert float(out.kl.detach()) >= cfg.free_bits * cfg.s_dim - 1e-4


# ---------------------------------------------------------------------------
# warm start
# ---------------------------------------------------------------------------


def test_warm_start_loads_reusable_and_keeps_core_fresh(tmp_path):
    cfg = _tiny_cfg()
    model = RSSMWorldModel(cfg)

    # build a tiny synthetic "phase1-like" state_dict carrying the reusable
    # tensors with MATCHING shapes (token_embed, head, row/col, diagnostic heads).
    fake = {
        "token_embed.weight": torch.randn(cfg.vocab_size, cfg.d_model),
        "head.weight": torch.randn(cfg.vocab_size, cfg.d_model),
        "row_embed.weight": torch.randn(cfg.grid_h, cfg.d_model),
        "col_embed.weight": torch.randn(cfg.grid_w, cfg.d_model),
        # an unrelated phase1 tensor with NO RSSM counterpart — must be ignored.
        "blocks.0.attn_s.qkv.weight": torch.randn(96, 32),
    }
    for st in cfg.signal_streams:
        fake[f"diagnostic_heads.{st.name}.weight"] = torch.randn(st.vocab, cfg.d_model)
        fake[f"diagnostic_heads.{st.name}.bias"] = torch.randn(st.vocab)

    ckpt = tmp_path / "phase1.pt"
    torch.save({"model": fake}, ckpt)

    # snapshot a latent-core tensor BEFORE the load — it must stay fresh.
    gru_before = model.gru.weight_ih.detach().clone()
    counts = model.warm_start_from_phase1(str(ckpt))

    # token embed loaded (head is tied to it — the loaded embed IS the head).
    assert torch.allclose(model.token_embed.weight.detach(), fake["token_embed.weight"])
    assert torch.allclose(model.row_embed.weight.detach(), fake["row_embed.weight"])
    for st in cfg.signal_streams:
        assert torch.allclose(
            model.diagnostic_heads[st.name].weight.detach(),
            fake[f"diagnostic_heads.{st.name}.weight"],
        )
    # the latent core stayed at its fresh init.
    assert torch.allclose(model.gru.weight_ih.detach(), gru_before)
    # the head is re-tied to the (loaded) token embed.
    assert model.head.weight.data_ptr() == model.token_embed.weight.data_ptr()
    # the reusable tensors were counted as loaded; the core as fresh.
    assert counts["loaded"] > 0
    assert counts["fresh"] > 0


def test_warm_start_skips_shape_mismatch(tmp_path):
    cfg = _tiny_cfg()
    model = RSSMWorldModel(cfg)
    before = model.token_embed.weight.detach().clone()
    # a phase1 with a WRONG-shape token embed (different vocab) — must be skipped,
    # NOT crash, and leave the fresh init intact.
    fake = {"token_embed.weight": torch.randn(cfg.vocab_size + 7, cfg.d_model)}
    ckpt = tmp_path / "mismatch.pt"
    torch.save(fake, ckpt)  # bare state_dict (no wrapper)
    counts = model.warm_start_from_phase1(str(ckpt))
    assert counts["skipped_shape"] >= 1
    assert torch.allclose(model.token_embed.weight.detach(), before)


# ---------------------------------------------------------------------------
# overfit smoke (tiny model + synthetic data — the de-risk gate)
# ---------------------------------------------------------------------------


def test_overfit_smoke_camera_ce_drops():
    """A tiny RSSM fits a tiny synthetic batch: camera CE drops substantially."""
    torch.manual_seed(0)
    cfg = _tiny_cfg(beta=1.0, free_bits=0.5)
    model = RSSMWorldModel(cfg).train()
    batch = _rand_batch(cfg, b=2, t=6, seed=42)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)

    first_ce = None
    last_ce = None
    last_kl = None
    for step in range(200):
        opt.zero_grad()
        out = model(batch)
        out.loss.backward()
        opt.step()
        if step == 0:
            first_ce = float(out.camera_ce)
        last_ce = float(out.camera_ce)
        last_kl = float(out.kl)

    assert first_ce is not None and last_ce is not None
    # the camera CE must drop substantially (the RSSM can fit the reconstruction).
    assert last_ce < first_ce * 0.6, f"camera CE did not drop: {first_ce} -> {last_ce}"
    # KL stays finite and does not collapse below the free-bits floor.
    assert last_kl is not None
    assert torch.isfinite(torch.tensor(last_kl))
    assert last_kl >= cfg.free_bits * cfg.s_dim - 1e-3
