"""Trainer-side wiring of the joint-generation (diagnostic CE) objective.

The model-side contract (per-stream heads + next-step CE on the measured-signal
tokens) is tested in ``test_controllable_model.py``.  This file tests that the
TRAINER actually USES it:

* the diagnostic-weight WARMUP schedule (``_diagnostic_weight``) — 0 at step 0,
  ramps to the target over the warmup fraction, holds, and is 0 when joint
  generation is off / the weight is 0;
* ``build_controllable_model`` threads ``generate_diagnostics`` to the model's
  ``has_diagnostics``;
* a tiny synthetic training smoke that drives the SAME loss-spec path the trainer
  uses (``return_components`` + a warmed ``diagnostic_weight``) and asserts the
  diagnostic CE component is finite and positive and the step loss is finite.
"""

from __future__ import annotations

import torch

from imas_ambix.worldmodel.controllable_train import (
    ControllableCorpusConfig,
    OverfitControllableConfig,
    _aux_log_suffix,
    _diagnostic_weight,
    _drop_observations,
    _mask_observations_per_stream,
    build_controllable_model,
)
from imas_ambix.worldmodel.spacetime_dataset import SpacetimeWindowConfig

# Reuse the tiny-config + random-batch helpers from the model unit tests.
from tests.worldmodel.test_controllable_model import _rand_batch, _tiny_cfg

# ---------------------------------------------------------------------------
# _diagnostic_weight warmup schedule
# ---------------------------------------------------------------------------


def test_diagnostic_weight_zero_at_start_and_max_after_warmup():
    cfg = OverfitControllableConfig(
        steps=100, diagnostic_weight=0.5, diagnostic_weight_warmup_frac=0.1
    )
    # 0 at the very first step.
    assert _diagnostic_weight(0, cfg.steps, cfg) == 0.0
    # ramps to the target by the warmup end (frac 0.1 -> full at step 10 of 100).
    assert abs(_diagnostic_weight(10, cfg.steps, cfg) - 0.5) < 1e-9
    # held at the max past the warmup.
    assert abs(_diagnostic_weight(50, cfg.steps, cfg) - 0.5) < 1e-9
    assert abs(_diagnostic_weight(99, cfg.steps, cfg) - 0.5) < 1e-9


def test_diagnostic_weight_is_monotone_nondecreasing_through_warmup():
    cfg = OverfitControllableConfig(
        steps=100, diagnostic_weight=0.5, diagnostic_weight_warmup_frac=0.1
    )
    prev = -1.0
    for step in range(0, 100, 2):
        w = _diagnostic_weight(step, cfg.steps, cfg)
        assert w >= prev - 1e-12, "diagnostic weight decreased during warmup"
        assert 0.0 <= w <= 0.5 + 1e-12
        prev = w


def test_diagnostic_weight_zero_when_disabled_or_weight_zero():
    off = OverfitControllableConfig(
        steps=100, generate_diagnostics=False, diagnostic_weight=0.5
    )
    # OFF -> always 0, even past the warmup.
    assert _diagnostic_weight(0, off.steps, off) == 0.0
    assert _diagnostic_weight(50, off.steps, off) == 0.0
    # weight 0 -> always 0 even when generate_diagnostics is True.
    zero = OverfitControllableConfig(
        steps=100, generate_diagnostics=True, diagnostic_weight=0.0
    )
    assert _diagnostic_weight(0, zero.steps, zero) == 0.0
    assert _diagnostic_weight(50, zero.steps, zero) == 0.0


def test_diagnostic_weight_no_warmup_is_full_immediately():
    cfg = OverfitControllableConfig(
        steps=100, diagnostic_weight=0.3, diagnostic_weight_warmup_frac=0.0
    )
    assert abs(_diagnostic_weight(0, cfg.steps, cfg) - 0.3) < 1e-9
    assert abs(_diagnostic_weight(50, cfg.steps, cfg) - 0.3) < 1e-9


def test_diagnostic_weight_schedule_matches_for_corpus_config():
    """The corpus config exposes the same fields, so the schedule is identical."""
    cfg = ControllableCorpusConfig(
        steps=200, diagnostic_weight=0.4, diagnostic_weight_warmup_frac=0.25
    )
    assert _diagnostic_weight(0, cfg.steps, cfg) == 0.0
    # full at step 0.25 * (200) ~ step 50 (using the (steps-1) denominator).
    assert abs(_diagnostic_weight(60, cfg.steps, cfg) - 0.4) < 1e-9
    # roughly half-way through the warmup window.
    mid = _diagnostic_weight(25, cfg.steps, cfg)
    assert 0.1 < mid < 0.3


# ---------------------------------------------------------------------------
# build_controllable_model threads generate_diagnostics
# ---------------------------------------------------------------------------


def _tiny_streams():
    from imas_ambix.worldmodel.spacetime_model_v2 import SignalStreamSpec

    return (
        SignalStreamSpec("gas_injection", vocab=8, channels=2),
        SignalStreamSpec("xma", vocab=8, channels=3),
    )


def _build(generate_diagnostics: bool):
    window = SpacetimeWindowConfig(n_frames=5, n_plan=3, context_frames=2)
    return build_controllable_model(
        window,
        plan_channels=2,
        signal_streams=_tiny_streams(),
        n_signal_steps=3,
        actuator_channels=6,
        n_act_steps=4,
        generate_diagnostics=generate_diagnostics,
        # keep the backbone tiny + CPU-fast.
        vocab_size=64,
        grid_h=4,
        grid_w=4,
        d_model=32,
        n_layers=2,
        n_heads=4,
        d_ff=64,
        dropout=0.0,
    )


def test_build_controllable_model_generate_diagnostics_on():
    model = _build(generate_diagnostics=True)
    assert model.has_diagnostics is True
    assert hasattr(model, "diagnostic_heads")
    assert set(model.diagnostic_heads.keys()) == {"gas_injection", "xma"}


def test_build_controllable_model_generate_diagnostics_off():
    model = _build(generate_diagnostics=False)
    assert model.has_diagnostics is False
    assert not hasattr(model, "diagnostic_heads")


# ---------------------------------------------------------------------------
# tiny overfit smoke through the trainer's loss-spec path
# ---------------------------------------------------------------------------


def test_overfit_smoke_diagnostic_ce_is_finite_positive_and_loss_drops():
    """Drive the SAME loss-spec path the trainer uses; the diagnostic CE fires.

    Builds a small generative model, runs a few training steps with the warmed
    ``diagnostic_weight`` + ``return_components`` (exactly the overfit loop's
    loss_spec), and asserts the diagnostic CE component is finite and > 0 and the
    step loss is finite (and the joint loss learns).
    """
    cfg = _tiny_cfg(generate_diagnostics=True)
    torch.manual_seed(0)
    model = _build(generate_diagnostics=True)
    # a fresh tiny model + a tiny synthetic batch (the trainer's collated shape).
    batch = _rand_batch(cfg, b=2, t=6, seed=3)
    train_cfg = OverfitControllableConfig(
        steps=20, diagnostic_weight=0.5, diagnostic_weight_warmup_frac=0.1
    )
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    model.train()
    losses: list[float] = []
    diag_ces: list[float] = []
    for step in range(train_cfg.steps):
        opt.zero_grad(set_to_none=True)
        out = model(
            batch,
            loss_spec={
                "chunk": 64,
                "context_frames": 2,
                "inverse_dynamics_weight": train_cfg.inverse_dynamics_weight,
                "diagnostic_weight": _diagnostic_weight(
                    step, train_cfg.steps, train_cfg
                ),
                "return_components": True,
            },
        )
        loss = out["loss"]
        assert torch.isfinite(loss), f"non-finite step loss at step {step}"
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
        diag_ces.append(float(out["diagnostic_ce"]))

    # the diagnostic CE component is finite + positive throughout (the heads are
    # being scored on real next-step targets, not all-PAD).
    assert all(d == d for d in diag_ces)  # no NaN
    assert max(diag_ces) > 0.0, "diagnostic CE never fired — joint objective dead"
    # and the gradient reached the diagnostic heads (the objective is wired in).
    for name in model.diagnostic_heads:
        g = model.diagnostic_heads[name].weight.grad
        assert g is not None and float(g.abs().sum()) > 0.0, (
            f"no gradient reached the {name} diagnostic head through the trainer path"
        )
    # the joint loss learns over the smoke run.
    assert losses[-1] < losses[0], "joint loss did not drop over the smoke run"


# ---------------------------------------------------------------------------
# Per-modality independent observation masking + clean diagnostic targets
# ---------------------------------------------------------------------------


def _signal_dict(b=64, p=4, seed=0):
    """A small {name: (B, P, C)} signal dict with all-NON-PAD tokens (ids >= 1)."""
    g = torch.Generator().manual_seed(seed)
    return {
        "gas_injection": torch.randint(1, 8, (b, p, 2), generator=g),
        "xma": torch.randint(1, 8, (b, p, 3), generator=g),
    }


def test_mask_observations_per_stream_is_independent_per_stream_and_rate_correct():
    """Two streams get DIFFERENT per-sample masks and the realized rate is ~rate."""
    torch.manual_seed(0)
    sig = _signal_dict(b=4000, p=4, seed=1)
    gen = torch.Generator().manual_seed(123)
    rate = 0.6
    masked = _mask_observations_per_stream(
        sig, rate, generator=gen, device=torch.device("cpu")
    )
    # a stream is "masked" for a sample iff its whole block is all-zero (PAD).
    gas_masked = (masked["gas_injection"].reshape(4000, -1) == 0).all(dim=1)
    xma_masked = (masked["xma"].reshape(4000, -1) == 0).all(dim=1)
    # realized rate close to the requested rate for EACH stream (full coverage:
    # ~(1-rate) of samples keep each stream as a clean CE target).
    assert abs(float(gas_masked.float().mean()) - rate) < 0.05
    assert abs(float(xma_masked.float().mean()) - rate) < 0.05
    # INDEPENDENT draws: the two streams' masks are NOT identical (an all-or-nothing
    # mask would make them equal sample-for-sample).
    assert not torch.equal(gas_masked, xma_masked), (
        "the two streams share the same mask — masking is not per-stream independent"
    )
    # and their disagreement is near the independent-Bernoulli expectation
    # (2*rate*(1-rate) ~ 0.48), not 0 (identical) — a sanity bound.
    disagree = float((gas_masked != xma_masked).float().mean())
    assert disagree > 0.2, "stream masks barely differ — draws look correlated"


def test_mask_observations_per_stream_does_not_mutate_input():
    sig = _signal_dict(b=32, p=4, seed=2)
    snap = {k: v.clone() for k, v in sig.items()}
    gen = torch.Generator().manual_seed(7)
    _ = _mask_observations_per_stream(
        sig, 0.5, generator=gen, device=torch.device("cpu")
    )
    for k in sig:
        assert torch.equal(sig[k], snap[k]), f"input stream {k} was mutated in place"


def test_mask_observations_per_stream_rate_zero_and_one():
    sig = _signal_dict(b=64, p=4, seed=3)
    gen = torch.Generator().manual_seed(9)
    none_masked = _mask_observations_per_stream(
        sig, 0.0, generator=gen, device=torch.device("cpu")
    )
    for k in sig:
        assert torch.equal(none_masked[k], sig[k]), "rate 0 masked something"
    all_masked = _mask_observations_per_stream(
        sig, 1.0, generator=gen, device=torch.device("cpu")
    )
    for k in sig:
        assert int(all_masked[k].abs().sum()) == 0, "rate 1 left a stream unmasked"


def test_overfit_step_prep_sets_clean_targets_and_masked_inputs():
    """The overfit step-prep keeps signal_targets == pre-mask signals; masks input.

    Replicates the exact code the overfit loop runs (per-modality masking branch):
    set ``signal_targets`` to a shallow copy of the CLEAN signals, then replace
    ``signals`` with the masked dict.  Asserts targets are byte-equal to the
    pre-mask signals while the input dict has at least one stream zeroed, and that
    commands (plan / actuator) are never touched.
    """
    batch = {
        "frames": torch.randint(0, 16, (8, 5, 4)),
        "plan": torch.randint(0, 16, (8, 3, 2)),
        "actuator": {
            "values": torch.randn(8, 4, 6),
            "missing": torch.zeros(8, 4, 6),
        },
        "signals": _signal_dict(b=8, p=4, seed=5),
    }
    plan_snap = batch["plan"].clone()
    act_v_snap = batch["actuator"]["values"].clone()
    sig_snap = {k: v.clone() for k, v in batch["signals"].items()}

    cfg = OverfitControllableConfig(per_modality_masking=True, observation_dropout=0.9)
    gen = torch.Generator().manual_seed(0)
    step_batch = dict(batch)
    # exactly the overfit loop's per-modality branch:
    step_batch["signal_targets"] = dict(batch["signals"])
    step_batch["signals"] = _mask_observations_per_stream(
        batch["signals"],
        cfg.observation_dropout,
        generator=gen,
        device=torch.device("cpu"),
    )

    # (c) clean targets equal the pre-mask signals.
    for k in sig_snap:
        assert torch.equal(step_batch["signal_targets"][k], sig_snap[k]), (
            f"signal_targets[{k}] is not the clean pre-mask reference"
        )
    # the input signals dict is a NEW dict (masking returns new tensors) and at
    # high rate at least one stream is zeroed for at least one sample.
    assert step_batch["signals"] is not batch["signals"]
    any_zeroed = any(
        bool((step_batch["signals"][k] == 0).all(dim=(1, 2)).any()) for k in sig_snap
    )
    assert any_zeroed, "no stream was masked at rate 0.9 — masking did not fire"
    # (b) commands untouched: plan + actuator are byte-identical.
    assert torch.equal(step_batch["plan"], plan_snap)
    assert torch.equal(step_batch["actuator"]["values"], act_v_snap)


def test_per_modality_masking_config_defaults_on_for_both_configs():
    assert OverfitControllableConfig().per_modality_masking is True
    assert ControllableCorpusConfig().per_modality_masking is True


def test_all_or_nothing_path_zeroes_whole_dict_for_flagged_sample():
    """The ablation path (per_modality_masking=False) masks the WHOLE dict at once."""
    sig = _signal_dict(b=10, p=4, seed=6)
    drop = torch.zeros(10, dtype=torch.bool)
    drop[3] = True  # flag exactly sample 3
    masked = _drop_observations(sig, drop)
    for k in sig:
        # sample 3 fully zeroed across EVERY stream (all-or-nothing).
        assert int(masked[k][3].abs().sum()) == 0
        # the other samples untouched.
        assert torch.equal(masked[k][:3], sig[k][:3])
        assert torch.equal(masked[k][4:], sig[k][4:])


# ---------------------------------------------------------------------------
# Auxiliary contrastive objectives — trainer-side wiring
# ---------------------------------------------------------------------------


def test_build_controllable_model_threads_contrastive_flags():
    """build_controllable_model passes the three contrastive flags to the model."""
    window = SpacetimeWindowConfig(n_frames=5, n_plan=3, context_frames=2)
    model = build_controllable_model(
        window,
        plan_channels=2,
        signal_streams=_tiny_streams(),
        n_signal_steps=3,
        actuator_channels=6,
        n_act_steps=4,
        cross_modal=True,
        self_predictive=True,
        action_contrastive=True,
        vocab_size=64,
        grid_h=4,
        grid_w=4,
        d_model=32,
        n_layers=2,
        n_heads=4,
        d_ff=64,
        dropout=0.0,
    )
    assert model.has_cross_modal is True
    assert model.has_self_predictive is True
    assert model.has_action_contrastive is True
    assert hasattr(model, "cross_modal_cam_proj")
    assert hasattr(model, "self_pred_predictor")
    assert hasattr(model, "action_contrastive_proj")


def test_contrastive_config_defaults_off_for_both_configs():
    for cfg in (OverfitControllableConfig(), ControllableCorpusConfig()):
        assert cfg.cross_modal is False
        assert cfg.self_predictive is False
        assert cfg.action_contrastive is False


def test_action_contrastive_weight_default_is_one_for_both_configs():
    """The InfoNCE term default weight is 1.0 (O(1) softmax CE), not the old 0.1."""
    for cfg in (OverfitControllableConfig(), ControllableCorpusConfig()):
        assert cfg.action_contrastive_weight == 1.0


def test_aux_log_suffix_shows_only_enabled_terms():
    """The log suffix prints the action-contrastive term (and the others) when ON.

    The last run's log printed only ``cam=``/``diag=``, so the action-contrastive
    term's ~0 contribution was invisible.  This locks the format: ``ac=`` appears
    iff ``action_contrastive`` is on, ``inv=`` iff inverse-dynamics is weighted,
    and the value comes from the ``return_components`` dict.
    """
    out = {
        "inv_dyn": 0.12,
        "action_contrastive": 1.5,
        "cross_modal": 0.7,
        "self_predictive": 0.3,
    }
    # action-contrastive ON, the others OFF, inverse-dynamics weighted.
    cfg = OverfitControllableConfig(
        action_contrastive=True,
        cross_modal=False,
        self_predictive=False,
        inverse_dynamics_weight=1.0,
    )
    suffix = _aux_log_suffix(out, cfg)
    assert " ac=1.5000" in suffix, suffix
    assert " inv=0.1200" in suffix, suffix
    assert "cm=" not in suffix and "sp=" not in suffix, suffix

    # all OFF + inverse-dynamics disabled -> empty suffix (byte-identical log line).
    cfg_off = OverfitControllableConfig(
        action_contrastive=False,
        cross_modal=False,
        self_predictive=False,
        inverse_dynamics_weight=0.0,
    )
    assert _aux_log_suffix(out, cfg_off) == ""

    # all three contrastive terms on.
    cfg_all = OverfitControllableConfig(
        action_contrastive=True,
        cross_modal=True,
        self_predictive=True,
        inverse_dynamics_weight=0.0,
    )
    s_all = _aux_log_suffix(out, cfg_all)
    assert " ac=1.5000" in s_all and " cm=0.7000" in s_all and " sp=0.3000" in s_all


def test_random_actuator_batch_perturbs_only_given_columns_holds_others():
    from imas_ambix.worldmodel.controllable_train import _random_actuator_batch

    g = torch.Generator().manual_seed(0)
    vals = torch.randn(3, 4, 6, generator=g)
    miss = torch.zeros(3, 4, 6)
    perturbable = [1, 3, 5]
    rb = _random_actuator_batch(
        {"values": vals, "missing": miss},
        perturbable_cols=perturbable,
        generator=torch.Generator().manual_seed(1),
    )
    assert rb is not None
    # the perturbable columns changed; the held columns are byte-identical.
    held = [0, 2, 4]
    assert not torch.allclose(rb["values"][:, :, perturbable], vals[:, :, perturbable])
    assert torch.equal(rb["values"][:, :, held], vals[:, :, held])
    # missing preserved + input not mutated.
    assert torch.equal(rb["missing"], miss)


def test_random_actuator_batch_none_inputs():
    from imas_ambix.worldmodel.controllable_train import _random_actuator_batch

    assert (
        _random_actuator_batch(None, perturbable_cols=[0], generator=torch.Generator())
        is None
    )
    vals = torch.randn(2, 3, 4)
    # no perturbable columns -> None (caller skips the action-contrastive term).
    assert (
        _random_actuator_batch(
            {"values": vals, "missing": torch.zeros_like(vals)},
            perturbable_cols=[],
            generator=torch.Generator(),
        )
        is None
    )


def test_overfit_smoke_with_all_contrastive_terms_on():
    """Drive the trainer loss-spec path with all three contrastive terms on."""
    cfg = _tiny_cfg(cross_modal=True, self_predictive=True, action_contrastive=True)
    torch.manual_seed(0)
    model = build_controllable_model(
        SpacetimeWindowConfig(n_frames=6, n_plan=3, context_frames=2),
        plan_channels=2,
        signal_streams=_tiny_streams(),
        n_signal_steps=3,
        actuator_channels=6,
        n_act_steps=4,
        cross_modal=True,
        self_predictive=True,
        action_contrastive=True,
        vocab_size=64,
        grid_h=4,
        grid_w=4,
        d_model=32,
        n_layers=2,
        n_heads=4,
        d_ff=64,
        dropout=0.0,
    )
    batch = _rand_batch(cfg, b=2, t=6, seed=3)
    batch["actuator_random"] = {
        "values": batch["actuator"]["values"] + 2.0,
        "missing": batch["actuator"]["missing"],
    }
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    model.train()
    for _ in range(5):
        opt.zero_grad(set_to_none=True)
        out = model(
            batch,
            loss_spec={
                "chunk": 64,
                "context_frames": 2,
                "cross_modal_weight": 0.1,
                "self_predictive_weight": 0.1,
                "action_contrastive_weight": 0.1,
                "return_components": True,
            },
        )
        loss = out["loss"]
        assert torch.isfinite(loss)
        loss.backward()
        opt.step()
    # the heads received gradients through the trainer path.
    assert model.cross_modal_cam_proj[0].weight.grad is not None
    assert model.self_pred_predictor[0].weight.grad is not None
    assert model.action_contrastive_proj[0].weight.grad is not None


def test_overfit_smoke_diagnostics_off_runs_and_has_zero_diag_ce():
    """With joint generation OFF the loss-spec path runs and the diag CE stays 0."""
    cfg = _tiny_cfg(generate_diagnostics=False)
    torch.manual_seed(0)
    model = _build(generate_diagnostics=False)
    batch = _rand_batch(cfg, b=2, t=6, seed=4)
    train_cfg = OverfitControllableConfig(
        steps=5, generate_diagnostics=False, diagnostic_weight=0.5
    )
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    model.train()
    for step in range(train_cfg.steps):
        opt.zero_grad(set_to_none=True)
        out = model(
            batch,
            loss_spec={
                "chunk": 64,
                "context_frames": 2,
                "diagnostic_weight": _diagnostic_weight(
                    step, train_cfg.steps, train_cfg
                ),
                "return_components": True,
            },
        )
        loss = out["loss"]
        assert torch.isfinite(loss)
        # weight schedule is 0 (disabled) and no heads exist -> diag CE detached 0.
        assert float(out["diagnostic_ce"]) == 0.0
        loss.backward()
        opt.step()
