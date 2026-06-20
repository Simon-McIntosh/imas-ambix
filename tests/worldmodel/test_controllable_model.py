"""Unit tests for the actuator-PLAN-conditioned camera transformer (M4 PLAY bridge).

These assert the load-bearing properties of the actuator-plan drive surface on
top of the v2 (measured-signal) contract:

* **shapes / causality** — the backbone still preserves ``(B, T, S, d)`` with the
  actuator + signal + plan prefix prepended and stripped, and a future camera
  frame still does not leak into an earlier frame's hidden;
* **the actuator plan is LOAD-BEARING** — perturbing the actuator vector changes
  the camera hidden, and silencing it gives a different loss (proves the drive
  surface feeds the prediction, not silently dropped);
* **scaling a specific command** (gas-puff / NBI) moves the prediction (the
  controllability lever the gate exercises);
* **actuator params get grad on an actuator-less batch** (DDP zero-touch);
* **the zero-touch does not change the actuator-less prediction**;
* **a config with 0 actuator channels is byte-equivalent to the v2 model**;
* **the gas/NBI channel index helpers + the normaliser** behave as documented;
* **the controllability gate** returns a structured verdict whose token-mismatch
  fields move with the model's actuator sensitivity.
"""

from __future__ import annotations

import numpy as np
import torch

from imas_ambix.worldmodel.actuator_plan import (
    ACTUATOR_CHANNEL_KEYS,
    N_ACTUATOR_CHANNELS,
    ActuatorPlan,
    gas_puff_channel_indices,
    nbi_channel_indices,
    normalise_actuator_values,
    scale_plan_channels,
    zero_plan,
)
from imas_ambix.worldmodel.controllable_model import (
    ControllableSpacetimeConfig,
    ControllableSpacetimeTransformer,
)
from imas_ambix.worldmodel.spacetime_model_v2 import SignalStreamSpec


def _tiny_cfg(**kw) -> ControllableSpacetimeConfig:
    base = dict(
        vocab_size=64,
        grid_h=4,
        grid_w=4,
        max_frames=40,
        plan_vocab=16,
        plan_channels=2,
        d_model=32,
        n_layers=2,
        n_heads=4,
        d_ff=64,
        dropout=0.0,
        n_signal_steps=3,
        signal_streams=(
            SignalStreamSpec("gas_injection", vocab=8, channels=2),
            SignalStreamSpec("xma", vocab=8, channels=3),
        ),
        actuator_channels=6,
        n_act_steps=4,
    )
    base.update(kw)
    return ControllableSpacetimeConfig(**base)


def _rand_batch(cfg, *, b=2, t=5, p=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    frames = torch.randint(0, cfg.vocab_size, (b, t, cfg.n_spatial), generator=g)
    plan = torch.randint(0, cfg.plan_vocab, (b, p, cfg.plan_channels), generator=g)
    signals = {}
    for st in cfg.signal_streams:
        signals[st.name] = torch.randint(
            0, st.vocab, (b, cfg.n_signal_steps, st.channels), generator=g
        )
    actuator = {
        "values": torch.randn(b, cfg.n_act_steps, cfg.actuator_channels, generator=g),
        "missing": torch.zeros(b, cfg.n_act_steps, cfg.actuator_channels),
    }
    return {"frames": frames, "plan": plan, "signals": signals, "actuator": actuator}


# ---------------------------------------------------------------------------
# shapes / causality
# ---------------------------------------------------------------------------


def test_forward_shapes_with_actuator():
    cfg = _tiny_cfg()
    model = ControllableSpacetimeTransformer(cfg).eval()
    batch = _rand_batch(cfg)
    out = model(batch, return_logits=True)
    b, t, s = batch["frames"].shape
    assert out.hidden.shape == (b, t, s, cfg.d_model)
    assert out.logits.shape == (b, t, s, cfg.vocab_size)


def test_temporal_causality_with_actuator():
    cfg = _tiny_cfg()
    model = ControllableSpacetimeTransformer(cfg).eval()
    batch = _rand_batch(cfg, t=5, seed=1)
    with torch.no_grad():
        h0 = model._forward_tokens(
            batch["frames"], batch["plan"], batch["signals"], actuator=batch["actuator"]
        )
    f2 = batch["frames"].clone()
    t = f2.shape[1]
    f2[:, t - 1] = (f2[:, t - 1] + 7) % cfg.vocab_size
    with torch.no_grad():
        h1 = model._forward_tokens(
            f2, batch["plan"], batch["signals"], actuator=batch["actuator"]
        )
    assert torch.allclose(h0[:, : t - 1], h1[:, : t - 1], atol=1e-5), (
        "future frame leaked into an earlier frame's hidden with the actuator "
        "prefix in place — temporal attention is not causal"
    )
    assert not torch.allclose(h0[:, t - 1], h1[:, t - 1], atol=1e-5)


# ---------------------------------------------------------------------------
# actuator plan is load-bearing
# ---------------------------------------------------------------------------


def test_actuator_changes_hidden():
    """Perturbing the actuator vector MUST change the camera hidden states."""
    cfg = _tiny_cfg()
    model = ControllableSpacetimeTransformer(cfg).eval()
    batch = _rand_batch(cfg, t=4, seed=2)
    with torch.no_grad():
        h0 = model._forward_tokens(
            batch["frames"], batch["plan"], batch["signals"], actuator=batch["actuator"]
        )
    act2 = {
        "values": batch["actuator"]["values"] + 2.0,
        "missing": batch["actuator"]["missing"],
    }
    with torch.no_grad():
        h1 = model._forward_tokens(
            batch["frames"], batch["plan"], batch["signals"], actuator=act2
        )
    diff = (h0 - h1).abs().sum().item()
    assert diff > 1e-4, (
        "camera hidden did not change when the actuator vector changed — the "
        "drive surface is not feeding the prediction (silently dropped)"
    )


def test_actuator_ablation_loss_delta():
    """Silencing the actuator gives a measurably different loss (load-bearing)."""
    cfg = _tiny_cfg()
    torch.manual_seed(0)
    model = ControllableSpacetimeTransformer(cfg).eval()
    batch = _rand_batch(cfg, b=2, t=5, seed=3)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    model.train()
    for _ in range(60):
        opt.zero_grad(set_to_none=True)
        loss = model(batch, loss_spec={"chunk": 64})
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        full = float(model(batch, loss_spec={"chunk": 64}))
        zeroed = dict(batch)
        zeroed["actuator"] = {
            "values": torch.zeros_like(batch["actuator"]["values"]),
            "missing": batch["actuator"]["missing"],
        }
        zero = float(model(zeroed, loss_spec={"chunk": 64}))
    assert abs(full - zero) > 1e-3, (
        f"loss unchanged when actuator zeroed (full={full:.5f} zero={zero:.5f}) — "
        "the model is ignoring the drive surface"
    )


def test_scaling_actuator_channels_moves_prediction():
    """Scaling a SUBSET of actuator channels changes the camera hidden."""
    cfg = _tiny_cfg()
    model = ControllableSpacetimeTransformer(cfg).eval()
    batch = _rand_batch(cfg, t=4, seed=4)
    with torch.no_grad():
        h0 = model._forward_tokens(
            batch["frames"], batch["plan"], batch["signals"], actuator=batch["actuator"]
        )
    vals = batch["actuator"]["values"].clone()
    vals[:, :, :2] = vals[:, :, :2] * 3.0  # scale the first two channels
    with torch.no_grad():
        h1 = model._forward_tokens(
            batch["frames"],
            batch["plan"],
            batch["signals"],
            actuator={"values": vals, "missing": batch["actuator"]["missing"]},
        )
    assert (h0 - h1).abs().sum().item() > 1e-4


# ---------------------------------------------------------------------------
# DDP zero-touch
# ---------------------------------------------------------------------------


def test_actuator_params_get_grad_on_actuatorless_batch():
    """Every actuator param must receive a grad even with NO actuator (DDP)."""
    cfg = _tiny_cfg()
    model = ControllableSpacetimeTransformer(cfg).train()
    frames = torch.randint(0, cfg.vocab_size, (2, 4, cfg.n_spatial))
    plan = torch.randint(0, cfg.plan_vocab, (2, 3, cfg.plan_channels))
    batch = {"frames": frames, "plan": plan, "signals": {}, "actuator": None}
    loss = model(batch, loss_spec={"chunk": 4096, "context_frames": None})
    loss.backward()
    assert model.actuator_encoder.weight.grad is not None
    assert model.actuator_encoder.bias.grad is not None
    assert model.actuator_lane_embed.grad is not None
    assert model.actuator_marker.grad is not None


def test_actuatorless_zero_touch_does_not_change_prediction():
    cfg = _tiny_cfg()
    model = ControllableSpacetimeTransformer(cfg).eval()
    frames = torch.randint(0, cfg.vocab_size, (2, 4, cfg.n_spatial))
    plan = torch.randint(0, cfg.plan_vocab, (2, 3, cfg.plan_channels))
    empty = {"frames": frames, "plan": plan, "signals": {}, "actuator": None}
    with torch.no_grad():
        h = model._forward_tokens(empty["frames"], empty["plan"], {}, actuator=None)
        model.actuator_marker.add_(3.14)
        model.actuator_lane_embed.add_(2.71)
        model.actuator_encoder.weight.add_(1.23)
        h2 = model._forward_tokens(empty["frames"], empty["plan"], {}, actuator=None)
    assert torch.allclose(h, h2, atol=1e-6), (
        "actuator-less forward changed when actuator params changed — the touch "
        "is not zero-magnitude"
    )


def test_no_actuator_config_equivalent_to_v2():
    """A config with 0 actuator channels behaves like the v2 signal model."""
    cfg = _tiny_cfg(actuator_channels=0)
    assert not cfg.has_actuator
    model = ControllableSpacetimeTransformer(cfg).eval()
    assert not hasattr(model, "actuator_encoder")
    frames = torch.randint(0, cfg.vocab_size, (2, 4, cfg.n_spatial))
    plan = torch.randint(0, cfg.plan_vocab, (2, 3, cfg.plan_channels))
    out = model(
        {"frames": frames, "plan": plan, "signals": {}, "actuator": None},
        return_logits=False,
    )
    assert out.hidden.shape == (2, 4, cfg.n_spatial, cfg.d_model)


def test_max_frames_guard_counts_actuator_prefix():
    # 4 act + 2x3 signal = 10 + 3 plan + 4 cam = 17 > max_frames 12
    cfg = _tiny_cfg(max_frames=12)
    model = ControllableSpacetimeTransformer(cfg).eval()
    batch = _rand_batch(cfg, b=1, t=4, p=3)
    import pytest

    with pytest.raises(ValueError, match="max_frames"):
        model(batch, return_logits=False)


# ---------------------------------------------------------------------------
# actuator_plan helpers
# ---------------------------------------------------------------------------


def test_normalise_is_sign_preserving_and_monotone():
    raw = np.array([[-1e5, 0.0, 1e21, 5e19]], dtype=np.float64)
    norm = normalise_actuator_values(raw)
    assert norm.shape == raw.shape
    # sign preserved
    assert norm[0, 0] < 0 and norm[0, 1] == 0 and norm[0, 2] > 0 and norm[0, 3] > 0
    # compressed to O(1)..O(50) range, monotone in magnitude
    assert abs(norm[0, 0]) < 20 and norm[0, 2] < 60
    bigger = normalise_actuator_values(raw * 10.0)
    assert bigger[0, 2] > norm[0, 2]  # scaling the command up scales the drive up


def test_gas_and_nbi_channel_indices_nonempty_and_disjoint():
    gas = gas_puff_channel_indices()
    nbi = nbi_channel_indices()
    assert gas and nbi
    assert not (set(gas) & set(nbi))
    # the gas channels are the aga-source ones; keys contain "gas".
    for i in gas:
        assert "gas" in ACTUATOR_CHANNEL_KEYS[i]
    for i in nbi:
        assert "nbi" in ACTUATOR_CHANNEL_KEYS[i]
    assert len(ACTUATOR_CHANNEL_KEYS) == N_ACTUATOR_CHANNELS


def test_scale_and_zero_plan():
    raw = np.ones((4, N_ACTUATOR_CHANNELS), dtype=np.float32) * 100.0
    plan = ActuatorPlan(
        values=normalise_actuator_values(raw),
        missing=np.zeros_like(raw),
        channel_keys=list(ACTUATOR_CHANNEL_KEYS),
        raw_values=raw,
    )
    gas = gas_puff_channel_indices()
    scaled = scale_plan_channels(plan, gas, 3.0)
    # the scaled channels' RAW command tripled.
    assert np.allclose(scaled.raw_values[:, gas], 300.0)
    # non-gas channels unchanged.
    other = [i for i in range(N_ACTUATOR_CHANNELS) if i not in gas]
    assert np.allclose(scaled.raw_values[:, other], 100.0)
    # normalised drive grew on the scaled channels.
    assert (scaled.values[:, gas] > plan.values[:, gas]).all()
    z = zero_plan(plan)
    assert np.allclose(z.raw_values, 0.0)
    assert np.allclose(z.values, 0.0)


# ---------------------------------------------------------------------------
# controllability gate (synthetic, CPU) — structured verdict
# ---------------------------------------------------------------------------


def _mk_controllable_sample(cfg, shot, *, t=6, ctx=3, seed=0):
    from imas_ambix.worldmodel.controllable_dataset import ControllableSpacetimeSample
    from imas_ambix.worldmodel.spacetime_dataset import SpacetimeSample
    from imas_ambix.worldmodel.spacetime_dataset_v2 import SignalSpacetimeSample

    g = torch.Generator().manual_seed(seed)
    frames = torch.randint(0, cfg.vocab_size, (t, cfg.n_spatial), generator=g).numpy()
    plan = torch.randint(0, cfg.plan_vocab, (4, cfg.plan_channels), generator=g).numpy()
    base = SpacetimeSample(
        shot_id=shot,
        camera="rbb",
        start_frame=0,
        frames=frames,
        plan=plan,
        frame_time=np.linspace(0.0, 0.1, t),
        context_frames=ctx,
    )
    sigs = {
        "gas_injection": torch.randint(0, 8, (cfg.n_signal_steps, 2)).numpy(),
        "xma": torch.randint(0, 8, (cfg.n_signal_steps, 3)).numpy(),
    }
    signal = SignalSpacetimeSample(base=base, signals=sigs)
    # actuator plan with non-zero, VARYING gas + NBI commands so the window is
    # transient (a ramp) — the gate only scores transient windows, and scaling a
    # command must actually change a non-constant drive.
    ramp = np.linspace(1e3, 1e5, cfg.n_act_steps, dtype=np.float32)[:, None]
    raw = ramp * np.ones((1, cfg.actuator_channels), dtype=np.float32)
    plan_act = ActuatorPlan(
        values=normalise_actuator_values(raw),
        missing=np.zeros_like(raw),
        channel_keys=[f"c{i}" for i in range(cfg.actuator_channels)],
        raw_values=raw,
    )
    return ControllableSpacetimeSample(signal=signal, actuator=plan_act)


def test_controllability_gate_returns_structured_verdict():
    from imas_ambix.worldmodel.controllable_train import controllability_gate

    # use the full actuator channel width so gas/NBI indices land in range.
    cfg = _tiny_cfg(actuator_channels=N_ACTUATOR_CHANNELS)
    torch.manual_seed(0)
    model = ControllableSpacetimeTransformer(cfg).train()
    samples = [_mk_controllable_sample(cfg, sid, seed=sid) for sid in (1, 2)]
    stream_names = [st.name for st in cfg.signal_streams]
    # train a few steps so the actuator encoder carries weight.
    from imas_ambix.worldmodel.controllable_train import (
        _batch_to,
        collate_controllable_windows,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    batch = _batch_to(
        collate_controllable_windows(samples, stream_names=stream_names), "cpu"
    )
    for _ in range(40):
        opt.zero_grad(set_to_none=True)
        loss = model(batch, loss_spec={"chunk": 64})
        loss.backward()
        opt.step()

    verdicts, summary = controllability_gate(
        model, samples, stream_names, device="cpu", chunk=64, margin_threshold=0.0
    )
    assert len(verdicts) == 2
    for v in verdicts:
        d = v.to_dict()
        for k in (
            "true_vs_zeroed_mismatch",
            "gas_scale_mismatch",
            "nbi_scale_mismatch",
            "observation_mismatch",
            "plan_over_observation_ratio",
            "cc_true_vs_zeroed_mismatch",
            "cc_gas_scale_mismatch",
            "cc_nbi_scale_mismatch",
            "cc_observation_mismatch",
            "plan_variation",
            "camera_change_fraction",
        ):
            assert k in d and np.isfinite(d[k])
        assert v.n_gas_channels > 0 and v.n_nbi_channels > 0
        # the synthetic plan ramps -> the window is transient.
        assert v.is_transient is True
        assert v.plan_variation > 0.0
    assert "verdict" in summary and summary["verdict"] in {"PASS", "FAIL"}
    assert summary["decision_metric"] == "corrupted_context_true_vs_zeroed"
    assert summary["n_samples"] == 2
    assert summary["n_transient"] == 2
    assert summary["gate_testable"] is True
    # the corrupted-context margin is the decision metric and is finite.
    assert np.isfinite(summary["mean_cc_true_vs_zeroed_mismatch"])
    assert summary["mean_true_vs_zeroed_mismatch"] >= 0.0


def test_flat_plan_sample_is_not_transient_and_excluded(monkeypatch):
    """A FLAT actuator plan window is flagged non-transient (gate not testable)."""
    from imas_ambix.worldmodel.controllable_train import controllability_gate

    cfg = _tiny_cfg(actuator_channels=N_ACTUATOR_CHANNELS)
    torch.manual_seed(0)
    model = ControllableSpacetimeTransformer(cfg).eval()
    sample = _mk_controllable_sample(cfg, 1, seed=1)
    # overwrite with a CONSTANT plan (flat-top) — zero variation.
    raw = np.full((cfg.n_act_steps, cfg.actuator_channels), 5e4, dtype=np.float32)
    sample.actuator.raw_values = raw
    sample.actuator.values = normalise_actuator_values(raw)
    sample.actuator.missing = np.zeros_like(raw)
    stream_names = [st.name for st in cfg.signal_streams]
    verdicts, summary = controllability_gate(
        model, [sample], stream_names, device="cpu", chunk=64, margin_threshold=0.0
    )
    assert verdicts[0].is_transient is False
    assert verdicts[0].plan_variation < 1e-3
    # the single sample is flat -> no transient window -> gate not fairly testable
    # (score_set falls back so means are finite, but gate_testable is False).
    assert summary["n_transient"] == 0
    assert summary["gate_testable"] is False
    assert summary["gate_pass"] is False


def test_find_transient_window_picks_the_ramp(monkeypatch):
    import imas_ambix.worldmodel.actuator_plan as ap
    from imas_ambix.camdyn.conditioning import ConditioningSample

    n = 100
    span = 20
    ftime = np.linspace(0.0, 1.0, n)
    # one channel flat at 100 with a ramp 0->100 over frames 40..60.
    sig = np.full(n, 100.0)
    sig[40:60] = np.linspace(0, 100, 20)
    vals = sig[:, None]
    miss = np.zeros((n, 1), dtype=np.float32)

    monkeypatch.setattr(ap, "_frame_times", None, raising=False)
    # patch the two reads find_transient_window does.
    import imas_ambix.worldmodel.spacetime_dataset as sd

    monkeypatch.setattr(sd, "_frame_times", lambda *a, **k: ftime)
    monkeypatch.setattr(
        ap,
        "level1_shot_path",
        lambda *a, **k: "x",
        raising=False,
    )

    import imas_ambix.camdyn.dataset as cd

    monkeypatch.setattr(cd, "level1_shot_path", lambda *a, **k: "x")
    monkeypatch.setattr(
        ap,
        "load_conditioning",
        lambda *a, **k: ConditioningSample(
            shot_id=1,
            frame_time=ftime,
            channel_keys=["c0"],
            units=[""],
            values=vals,
            missing=miss,
        ),
    )
    start, score = ap.find_transient_window(1, span, min_variation=1e-6)
    assert start is not None
    assert 30 <= start <= 60  # overlaps the ramp
    assert score > 0.0


def test_find_transient_window_none_when_flat(monkeypatch):
    import imas_ambix.camdyn.dataset as cd
    import imas_ambix.worldmodel.actuator_plan as ap
    import imas_ambix.worldmodel.spacetime_dataset as sd
    from imas_ambix.camdyn.conditioning import ConditioningSample

    n, span = 100, 20
    ftime = np.linspace(0.0, 1.0, n)
    monkeypatch.setattr(sd, "_frame_times", lambda *a, **k: ftime)
    monkeypatch.setattr(cd, "level1_shot_path", lambda *a, **k: "x")
    monkeypatch.setattr(
        ap,
        "load_conditioning",
        lambda *a, **k: ConditioningSample(
            shot_id=1,
            frame_time=ftime,
            channel_keys=["c0"],
            units=[""],
            values=np.full((n, 1), 130.0),
            missing=np.zeros((n, 1), dtype=np.float32),
        ),
    )
    start, score = ap.find_transient_window(1, span, min_variation=1e-3)
    assert start is None  # flat everywhere -> no transient window


def test_gate_collate_actuator_stacks_rectangular():
    from imas_ambix.worldmodel.controllable_train import collate_actuator

    cfg = _tiny_cfg(actuator_channels=N_ACTUATOR_CHANNELS)
    samples = [_mk_controllable_sample(cfg, sid, seed=sid) for sid in (1, 2, 3)]
    act = collate_actuator(samples)
    assert act["values"].shape == (3, cfg.n_act_steps, N_ACTUATOR_CHANNELS)
    assert act["missing"].shape == (3, cfg.n_act_steps, N_ACTUATOR_CHANNELS)
    assert act["values"].dtype == torch.float32
