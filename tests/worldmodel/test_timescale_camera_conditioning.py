"""Unit tests for the multi-timescale (Δt) + multi-camera conditioning.

These assert the contract of the four conditioning pieces added so ONE model can
ingest ALL the MAST imaging data (all cameras + both timescales + more signals):

* **Δt / camera encoding helpers** — log-Δt centres on the reference cadence and
  is monotone in cadence; the camera index is stable and falls back to the
  reference for an unknown / missing view;
* **OFF (default) is byte-identical** — a model with the knobs off matches a
  model built without them, frame-for-frame (no behavioural change to the prior
  model, so a prior checkpoint's results are preserved);
* **ON at INIT is identity** — the zero-init Δt head + zero-init camera table make
  a fresh timescale / camera-conditioned model byte-identical to the cadence- and
  view-blind backbone (warm-start starts AS the forecaster);
* **ON after training is load-bearing** — once trained, changing the cadence or
  the camera moves the prediction (the signal earned influence) and gradients
  reach the new params;
* **DDP-uniform** — the timescale encoder receives a gradient even when no Δt is
  supplied (it always runs at the reference offset), and the camera table when
  camera-conditioned;
* **warm-start tolerance** — a state dict missing the Δt / camera / extra-signal
  params loads with ``strict=False`` (the new params stay at their fresh init),
  and the extra HF signal streams (xsx / xim / ait) are admitted by the model.
"""

from __future__ import annotations

import numpy as np
import torch

from imas_ambix.worldmodel.controllable_model import (
    ControllableSpacetimeConfig,
    ControllableSpacetimeTransformer,
)
from imas_ambix.worldmodel.spacetime_model_v2 import SignalStreamSpec
from imas_ambix.worldmodel.timescale_conditioning import (
    CAMERA_IDS,
    REFERENCE_CAMERA_INDEX,
    REFERENCE_DT_SECONDS,
    TimescaleEncoder,
    camera_index,
    camera_indices,
    frame_dt_seconds,
    log_dt_offset,
)


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
        signal_streams=(SignalStreamSpec("xma", vocab=8, channels=3),),
        actuator_channels=6,
        n_act_steps=4,
    )
    base.update(kw)
    return ControllableSpacetimeConfig(**base)


def _rand_batch(cfg, *, b=2, t=5, p=3, seed=0, with_cond=False):
    g = torch.Generator().manual_seed(seed)
    frames = torch.randint(0, cfg.vocab_size, (b, t, cfg.n_spatial), generator=g)
    plan = torch.randint(0, cfg.plan_vocab, (b, p, cfg.plan_channels), generator=g)
    signals = {
        st.name: torch.randint(
            0, st.vocab, (b, cfg.n_signal_steps, st.channels), generator=g
        )
        for st in cfg.signal_streams
    }
    actuator = {
        "values": torch.randn(b, cfg.n_act_steps, cfg.actuator_channels, generator=g),
        "missing": torch.zeros(b, cfg.n_act_steps, cfg.actuator_channels),
    }
    batch = {"frames": frames, "plan": plan, "signals": signals, "actuator": actuator}
    if with_cond:
        # fast-cadence Δt for half the batch, slow for the other half; distinct
        # cameras — so per-sample conditioning is clearly different.
        ft_fast = np.linspace(0.0, 50e-6 * (t - 1), t)
        ft_slow = np.linspace(0.0, REFERENCE_DT_SECONDS * (t - 1), t)
        ld = np.stack(
            [
                log_dt_offset(frame_dt_seconds(ft_fast if i % 2 else ft_slow))
                for i in range(b)
            ]
        ).astype(np.float32)
        batch["frame_log_dt"] = torch.as_tensor(ld, dtype=torch.float32)
        batch["camera_id"] = torch.arange(b, dtype=torch.long) % len(CAMERA_IDS)
    return batch


# ---------------------------------------------------------------------------
# encoding helpers
# ---------------------------------------------------------------------------


def test_log_dt_offset_centres_and_is_monotone():
    # reference cadence -> 0
    ref = float(log_dt_offset(np.array([REFERENCE_DT_SECONDS]))[0])
    assert abs(ref) < 1e-9
    # 10x faster -> -1, 10x slower -> +1
    faster = float(log_dt_offset(np.array([REFERENCE_DT_SECONDS / 10]))[0])
    slower = float(log_dt_offset(np.array([REFERENCE_DT_SECONDS * 10]))[0])
    assert abs(faster + 1.0) < 1e-9
    assert abs(slower - 1.0) < 1e-9
    # monotone decreasing in cadence (faster -> more negative)
    fast = float(log_dt_offset(np.array([50e-6]))[0])
    slow = float(log_dt_offset(np.array([6e-3]))[0])
    assert fast < slow
    # the ~250x corpus span maps to ~2.4 decades
    assert -2.6 < fast < -2.0


def test_log_dt_offset_handles_torch_and_bad_values():
    # torch in -> torch out, same values as numpy
    t = torch.tensor([REFERENCE_DT_SECONDS, 50e-6])
    out = log_dt_offset(t)
    assert isinstance(out, torch.Tensor)
    assert abs(float(out[0])) < 1e-6
    # non-positive / non-finite clamp to reference (offset 0), no NaN/inf
    bad = log_dt_offset(np.array([0.0, -1.0, np.nan, np.inf]))
    assert np.all(np.isfinite(bad)) and np.allclose(bad, 0.0)


def test_frame_dt_forward_fills_and_is_robust():
    ft = np.array([0.0, 0.002, 0.004, 0.006])  # uniform 2 ms
    dt = frame_dt_seconds(ft)
    assert dt.shape == (4,)
    assert np.allclose(dt, 0.002)  # frame 0 copies frame 1's dt
    # < 2 frames -> reference everywhere
    assert np.allclose(frame_dt_seconds(np.array([0.1])), REFERENCE_DT_SECONDS)
    # non-increasing / degenerate -> reference (no zero/negative dt leaks)
    assert np.all(frame_dt_seconds(np.array([1.0, 1.0, 1.0])) == REFERENCE_DT_SECONDS)


def test_camera_index_stable_and_falls_back():
    assert camera_index("rbb") == REFERENCE_CAMERA_INDEX == 0
    assert camera_index("rco") == 1
    # unknown / None -> reference
    assert camera_index("not_a_camera") == REFERENCE_CAMERA_INDEX
    assert camera_index(None) == REFERENCE_CAMERA_INDEX
    assert camera_indices(["rbb", "rco", "zzz", None]) == [0, 1, 0, 0]
    # every known camera maps to a distinct index
    idx = [camera_index(c) for c in CAMERA_IDS]
    assert idx == list(range(len(CAMERA_IDS)))


def test_timescale_encoder_is_zero_at_init():
    enc = TimescaleEncoder(16, hidden=8)
    out = enc(torch.randn(3, 5))
    assert out.shape == (3, 5, 16)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-9), (
        "TimescaleEncoder output is not zero at init — the zero-init output layer "
        "is broken, so an ON-but-untrained Δt head is not identity"
    )


# ---------------------------------------------------------------------------
# OFF is byte-identical
# ---------------------------------------------------------------------------


def test_off_is_byte_identical_to_model_without_knobs():
    torch.manual_seed(0)
    m_off = ControllableSpacetimeTransformer(_tiny_cfg()).eval()
    torch.manual_seed(0)
    m_off2 = ControllableSpacetimeTransformer(
        _tiny_cfg(timescale_conditioning=False, camera_conditioning=False)
    ).eval()
    batch = _rand_batch(_tiny_cfg())
    with torch.no_grad():
        h1 = m_off(batch, return_logits=False).hidden
        h2 = m_off2(batch, return_logits=False).hidden
    assert torch.allclose(h1, h2, atol=1e-6)
    # the off model has NO timescale / camera params at all.
    assert not hasattr(m_off, "timescale_encoder")
    assert not hasattr(m_off, "camera_embed")
    assert not m_off.config.has_timescale and not m_off.config.has_camera


def test_off_model_ignores_supplied_conditioning():
    """An OFF model must ignore frame_log_dt / camera_id if they appear in batch."""
    torch.manual_seed(0)
    m = ControllableSpacetimeTransformer(_tiny_cfg()).eval()
    base = _rand_batch(_tiny_cfg())
    cond = _rand_batch(_tiny_cfg(), with_cond=True)
    # same frames/plan/signals/actuator, only the cond keys differ.
    cond = dict(base)
    cond["frame_log_dt"] = torch.randn(2, 5)
    cond["camera_id"] = torch.tensor([3, 4])
    with torch.no_grad():
        h0 = m(base, return_logits=False).hidden
        h1 = m(cond, return_logits=False).hidden
    assert torch.allclose(h0, h1, atol=1e-6), (
        "an OFF model changed when frame_log_dt / camera_id were supplied — the "
        "conditioning is not gated by the config flag"
    )


# ---------------------------------------------------------------------------
# ON at init is identity
# ---------------------------------------------------------------------------


def test_on_at_init_is_identity_vs_blind_backbone():
    """Zero-init Δt head + zero-init camera table => fresh ON == cadence/view-blind."""
    torch.manual_seed(0)
    m_on = ControllableSpacetimeTransformer(
        _tiny_cfg(timescale_conditioning=True, camera_conditioning=True)
    ).eval()
    torch.manual_seed(0)
    m_blind = ControllableSpacetimeTransformer(_tiny_cfg()).eval()
    blind_batch = _rand_batch(_tiny_cfg())
    on_batch = _rand_batch(_tiny_cfg(), with_cond=True)
    # identical frames/plan/signals/actuator across the two (only on_batch carries
    # the cond keys, which are no-ops at init).
    for k in ("frames", "plan", "signals", "actuator"):
        on_batch[k] = blind_batch[k]
    with torch.no_grad():
        h_on = m_on(on_batch, return_logits=False).hidden
        h_blind = m_blind(blind_batch, return_logits=False).hidden
    assert torch.allclose(h_on, h_blind, atol=1e-6), (
        "an ON-at-init timescale/camera model is not identical to the blind "
        "backbone — a zero-init head is not a no-op, so warm-start does not start "
        "as the forecaster"
    )


def test_on_at_init_insensitive_to_conditioning():
    """At INIT, changing Δt / camera is a no-op (the heads are zero-init)."""
    torch.manual_seed(0)
    m = ControllableSpacetimeTransformer(
        _tiny_cfg(timescale_conditioning=True, camera_conditioning=True)
    ).eval()
    batch = _rand_batch(_tiny_cfg(), with_cond=True)
    alt = dict(batch)
    alt["frame_log_dt"] = batch["frame_log_dt"] - 3.0  # very different cadence
    alt["camera_id"] = (batch["camera_id"] + 2) % len(CAMERA_IDS)
    with torch.no_grad():
        h0 = m(batch, return_logits=False).hidden
        h1 = m(alt, return_logits=False).hidden
    assert torch.allclose(h0, h1, atol=1e-6)


# ---------------------------------------------------------------------------
# ON after training is load-bearing
# ---------------------------------------------------------------------------


def _train_a_few_steps(model, batch, steps=60, lr=5e-3):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = model(batch, loss_spec={"chunk": 64, "context_frames": 2})
        loss.backward()
        opt.step()
    model.eval()


def test_timescale_becomes_load_bearing_after_training():
    """After training, changing the cadence moves the prediction (Δt earned influence).

    The model is trained on a batch whose target frames depend on the cadence (we
    make the per-sample frames a deterministic function of camera/cadence by using
    distinct seeds per row), then perturbing frame_log_dt at eval must change the
    camera hidden — proving the Δt head is wired into the prediction.
    """
    cfg = _tiny_cfg(timescale_conditioning=True, camera_conditioning=False)
    torch.manual_seed(0)
    model = ControllableSpacetimeTransformer(cfg)
    batch = _rand_batch(cfg, b=2, t=6, seed=3, with_cond=True)
    _train_a_few_steps(model, batch)
    # the Δt encoder output layer moved off its zero init.
    assert float(model.timescale_encoder.fc2.weight.detach().abs().sum()) > 0.0
    with torch.no_grad():
        h0 = model(batch, return_logits=False).hidden
        alt = dict(batch)
        alt["frame_log_dt"] = batch["frame_log_dt"] - 2.0
        h1 = model(alt, return_logits=False).hidden
    assert (h0 - h1).abs().sum().item() > 1e-4, (
        "after training the cadence does not move the hidden — the Δt head never "
        "became load-bearing"
    )


def test_camera_becomes_load_bearing_after_training():
    """After training, changing the camera id moves the prediction."""
    cfg = _tiny_cfg(timescale_conditioning=False, camera_conditioning=True)
    torch.manual_seed(0)
    model = ControllableSpacetimeTransformer(cfg)
    batch = _rand_batch(cfg, b=2, t=6, seed=4, with_cond=True)
    _train_a_few_steps(model, batch)
    # the camera table moved off its zero init.
    assert float(model.camera_embed.weight.detach().abs().sum()) > 0.0
    with torch.no_grad():
        h0 = model(batch, return_logits=False).hidden
        alt = dict(batch)
        alt["camera_id"] = (batch["camera_id"] + 1) % len(CAMERA_IDS)
        h1 = model(alt, return_logits=False).hidden
    assert (h0 - h1).abs().sum().item() > 1e-4, (
        "after training the camera id does not move the hidden — the camera "
        "embedding never became load-bearing"
    )


# ---------------------------------------------------------------------------
# DDP uniformity: the new params always get a gradient
# ---------------------------------------------------------------------------


def test_timescale_encoder_gets_grad_without_supplied_dt():
    """The Δt encoder must receive a grad even with NO frame_log_dt (DDP-uniform).

    It always runs at the reference offset when none is supplied, so it stays in
    the autograd graph on every step regardless of the batch.
    """
    cfg = _tiny_cfg(timescale_conditioning=True, camera_conditioning=True)
    model = ControllableSpacetimeTransformer(cfg).train()
    batch = _rand_batch(cfg, b=2, t=4, seed=1)  # NO cond keys
    loss = model(batch, loss_spec={"chunk": 4096, "context_frames": None})
    loss.backward()
    assert model.timescale_encoder.fc1.weight.grad is not None
    assert model.timescale_encoder.fc2.weight.grad is not None
    assert model.camera_embed.weight.grad is not None


def test_camera_zero_init_table_keeps_reference_row_zero_at_init():
    cfg = _tiny_cfg(camera_conditioning=True)
    model = ControllableSpacetimeTransformer(cfg)
    w = model.camera_embed.weight
    assert torch.allclose(w, torch.zeros_like(w))


# ---------------------------------------------------------------------------
# warm-start tolerance
# ---------------------------------------------------------------------------


def test_warm_start_tolerates_missing_timescale_camera_params():
    """A state dict WITHOUT the Δt / camera params loads with strict=False.

    Mirrors the forecaster -> controllable warm start: build a model WITHOUT the
    new knobs, take its state dict, and load it into a model WITH the knobs — the
    backbone loads by name, the new params stay at their fresh (zero) init.
    """
    torch.manual_seed(0)
    src = ControllableSpacetimeTransformer(_tiny_cfg())  # no Δt / camera params
    src_state = src.state_dict()
    torch.manual_seed(1)  # different init for the target so a copy is observable
    dst = ControllableSpacetimeTransformer(
        _tiny_cfg(timescale_conditioning=True, camera_conditioning=True)
    )
    missing, unexpected = dst.load_state_dict(src_state, strict=False)
    # the only missing keys are the new Δt / camera params; nothing unexpected.
    assert not unexpected
    miss = set(missing)
    assert any("timescale_encoder" in k for k in miss)
    assert any("camera_embed" in k for k in miss)
    # the new params kept their fresh init (zero) — load_state_dict didn't touch them.
    assert torch.allclose(
        dst.camera_embed.weight, torch.zeros_like(dst.camera_embed.weight)
    )
    assert torch.allclose(
        dst.timescale_encoder.fc2.weight,
        torch.zeros_like(dst.timescale_encoder.fc2.weight),
    )
    # a backbone tensor actually loaded from src (token embedding matches).
    assert torch.allclose(dst.token_embed.weight, src.token_embed.weight)


def test_config_roundtrips_the_new_flags():
    """The controllable config dict carries the new flags so a checkpoint rebuilds."""
    from imas_ambix.worldmodel.controllable_train import _controllable_config_to_dict

    cfg = _tiny_cfg(
        timescale_conditioning=True, timescale_hidden=48, camera_conditioning=True
    )
    d = _controllable_config_to_dict(cfg)
    assert d["timescale_conditioning"] is True
    assert d["timescale_hidden"] == 48
    assert d["camera_conditioning"] is True
    # the dict reconstructs the IDENTICAL config (the loader path).
    scalar = {
        k: d[k]
        for k in d
        if k in ControllableSpacetimeConfig.__dataclass_fields__
        and k not in ("signal_streams", "masked_command_indices")
    }
    rebuilt = ControllableSpacetimeConfig(
        signal_streams=cfg.signal_streams,
        masked_command_indices=cfg.masked_command_indices,
        **scalar,
    )
    assert rebuilt.timescale_conditioning and rebuilt.camera_conditioning
    assert rebuilt.timescale_hidden == 48


# ---------------------------------------------------------------------------
# extra signal streams (xsx / xim / ait) are admitted
# ---------------------------------------------------------------------------


def test_extra_hf_signal_streams_are_admitted_by_the_model():
    """The model embeds the extra HF streams (xsx / xim) exactly like xma."""
    cfg = _tiny_cfg(
        signal_streams=(
            SignalStreamSpec("xma", vocab=8, channels=3),
            SignalStreamSpec("xsx", vocab=1030, channels=4),
            SignalStreamSpec("xim", vocab=12806, channels=2),
            SignalStreamSpec("ait", vocab=257, channels=3),
        ),
    )
    model = ControllableSpacetimeTransformer(cfg).eval()
    # one embedding table per stream, sized to its vocab.
    assert model.signal_embed["xsx"].num_embeddings == 1030
    assert model.signal_embed["xim"].num_embeddings == 12806
    assert model.signal_embed["ait"].num_embeddings == 257
    batch = _rand_batch(cfg, b=2, t=5)
    out = model(batch, return_logits=False)
    assert out.hidden.shape == (2, 5, cfg.n_spatial, cfg.d_model)


def test_extended_modalities_include_xsx_xim_ait():
    from imas_ambix.worldmodel.controllable_dataset import (
        default_signal_modalities,
        extended_signal_modalities,
    )

    base = {m.name for m in default_signal_modalities()}
    ext = extended_signal_modalities()
    names = {m.name for m in ext}
    # the default streams are preserved IN ORDER (a prior checkpoint loads by name).
    assert [m.name for m in extended_signal_modalities()][: len(base)] == [
        m.name for m in default_signal_modalities()
    ]
    # the new HF streams are added.
    assert {"xsx", "xim", "ait"}.issubset(names)
    # vocabs match the encoder.
    by = {m.name: m for m in ext}
    assert by["xsx"].vocab == 1030
    assert by["xim"].vocab == 12806


# ---------------------------------------------------------------------------
# shapes / causality preserved with conditioning on
# ---------------------------------------------------------------------------


def test_causality_preserved_with_conditioning_on():
    """Δt / camera conditioning must not break temporal causality."""
    cfg = _tiny_cfg(timescale_conditioning=True, camera_conditioning=True)
    torch.manual_seed(1)
    model = ControllableSpacetimeTransformer(cfg)
    batch = _rand_batch(cfg, b=2, t=5, seed=2, with_cond=True)
    _train_a_few_steps(model, batch, steps=30)
    with torch.no_grad():
        h0 = model(batch, return_logits=False).hidden
        f2 = batch["frames"].clone()
        t = f2.shape[1]
        f2[:, t - 1] = (f2[:, t - 1] + 7) % cfg.vocab_size
        alt = dict(batch)
        alt["frames"] = f2
        h1 = model(alt, return_logits=False).hidden
    assert torch.allclose(h0[:, : t - 1], h1[:, : t - 1], atol=1e-4), (
        "a future frame leaked into an earlier frame's hidden with Δt/camera "
        "conditioning on — temporal causality broke"
    )
    assert not torch.allclose(h0[:, t - 1], h1[:, t - 1], atol=1e-4)


def test_collate_builds_frame_log_dt_and_camera_id():
    """collate_timescale_camera derives per-window log-Δt + camera index."""
    import imas_ambix.worldmodel.controllable_train as ct
    from imas_ambix.worldmodel.actuator_plan import (
        ACTUATOR_CHANNEL_KEYS,
        N_ACTUATOR_CHANNELS,
        ActuatorPlan,
        normalise_actuator_values,
    )
    from imas_ambix.worldmodel.controllable_dataset import ControllableSpacetimeSample
    from imas_ambix.worldmodel.spacetime_dataset import SpacetimeSample
    from imas_ambix.worldmodel.spacetime_dataset_v2 import SignalSpacetimeSample

    def mk(shot, camera, ft, t=5):
        frames = np.zeros((t, 16), dtype=np.int64)
        plan = np.zeros((4, 2), dtype=np.int64)
        base = SpacetimeSample(
            shot_id=shot,
            camera=camera,
            start_frame=0,
            frames=frames,
            plan=plan,
            frame_time=ft,
            context_frames=2,
        )
        sig = SignalSpacetimeSample(base=base, signals={})
        raw = np.zeros((4, N_ACTUATOR_CHANNELS), dtype=np.float32)
        act = ActuatorPlan(
            values=normalise_actuator_values(raw),
            missing=np.zeros_like(raw),
            channel_keys=list(ACTUATOR_CHANNEL_KEYS),
            raw_values=raw,
        )
        return ControllableSpacetimeSample(signal=sig, actuator=act)

    t = 5
    fast = np.linspace(0.0, 50e-6 * (t - 1), t)  # 50 us
    slow = np.linspace(0.0, REFERENCE_DT_SECONDS * (t - 1), t)  # 6 ms
    samples = [mk(1, "rbb", slow, t), mk(2, "rco", fast, t)]
    cond = ct.collate_timescale_camera(samples)
    assert cond["frame_log_dt"].shape == (2, t)
    assert cond["camera_id"].tolist() == [0, 1]
    # the slow sample maps to ~0 offset; the fast sample to ~-2.1.
    assert abs(float(cond["frame_log_dt"][0].mean())) < 1e-5
    assert float(cond["frame_log_dt"][1].mean()) < -2.0
