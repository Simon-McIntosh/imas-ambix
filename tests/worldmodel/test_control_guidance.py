"""Unit tests for classifier-free guidance on the controls + the W1 metrics.

These pin the load-bearing properties of the CFG decode and the gas-puff
falsification metrics, all on a tiny CPU model / synthetic frames:

* CFG with weight ``w == 1`` reproduces the plain conditioned rollout exactly;
* CFG with ``w == 0`` reproduces the UNCONDITIONED rollout (controls fully
  dropped) — so the guidance axis spans uncond..cond as it must;
* the guided ARGMAX equals ``argmax(l_u + w*(l_c - l_u))`` computed directly from
  the two forward passes (the CFG formula is applied to LOGITS, not tokens);
* zeroing a single signal stream leaves the plan + other streams intact;
* the inboard token-column band maps to the documented left-of-centre pixel band;
* the timing correlation, counterfactual delta, and control-divergence read the
  expected signs on constructed inputs.
"""

from __future__ import annotations

import numpy as np
import torch

from imas_ambix.worldmodel.control_guidance import (
    INBOARD_COLS,
    _guided_sample_frame,
    _zeroed_conditioning,
    cfg_guided_dream,
    control_divergence,
    counterfactual_delta,
    frame_l1,
    gas_command_per_frame,
    inboard_emission_series,
    inboard_pixel_cols,
    puff_timing_correlation,
)
from imas_ambix.worldmodel.spacetime_dataset import SpacetimeSample
from imas_ambix.worldmodel.spacetime_dataset_v2 import SignalSpacetimeSample
from imas_ambix.worldmodel.spacetime_model_v2 import (
    SignalSpacetimeConfig,
    SignalSpacetimeTransformer,
    SignalStreamSpec,
)

# ---------------------------------------------------------------------------
# Tiny model + sample fixtures
# ---------------------------------------------------------------------------


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
            SignalStreamSpec("gas_injection", vocab=8, channels=2),
            SignalStreamSpec("xma", vocab=8, channels=3),
        ),
    )
    base.update(kw)
    return SignalSpacetimeConfig(**base)


def _tiny_sample(cfg, *, t=6, ctx=3, seed=0) -> SignalSpacetimeSample:
    g = torch.Generator().manual_seed(seed)
    frames = torch.randint(0, cfg.vocab_size, (t, cfg.n_spatial), generator=g).numpy()
    plan = torch.randint(0, cfg.plan_vocab, (4, cfg.plan_channels), generator=g).numpy()
    base = SpacetimeSample(
        shot_id=1,
        camera="rbb",
        start_frame=0,
        frames=frames,
        plan=plan,
        frame_time=np.linspace(0.0, 0.1, t),
        context_frames=ctx,
    )
    signals = {
        "gas_injection": torch.randint(
            0, 8, (cfg.n_signal_steps, 2), generator=g
        ).numpy(),
        "xma": torch.randint(0, 8, (cfg.n_signal_steps, 3), generator=g).numpy(),
    }
    return SignalSpacetimeSample(base=base, signals=signals)


def _stream_names(cfg):
    return [st.name for st in cfg.signal_streams]


# ---------------------------------------------------------------------------
# CFG decode — formula + endpoints
# ---------------------------------------------------------------------------


def test_guided_sample_argmax_matches_logit_formula():
    """The guided argmax equals argmax of l_u + w*(l_c - l_u) over the head."""
    cfg = _tiny_cfg()
    model = SignalSpacetimeTransformer(cfg).eval()
    sample = _tiny_sample(cfg, seed=1)
    names = _stream_names(cfg)
    from imas_ambix.worldmodel.spacetime_train_v2 import (
        _batch_to,
        collate_signal_windows,
    )

    batch = _batch_to(collate_signal_windows([sample], stream_names=names), "cpu")
    plan, signals = batch["plan"], batch["signals"]
    plan_u, signals_u = _zeroed_conditioning(plan, signals, streams=None)
    cur = torch.as_tensor(sample.frames[:4][None], dtype=torch.long)
    with torch.no_grad():
        hc = model._forward_tokens(cur, plan, signals)[:, -1]  # (1,S,d)
        hu = model._forward_tokens(cur, plan_u, signals_u)[:, -1]
        lc = model.head(hc.reshape(-1, cfg.d_model)).float()
        lu = model.head(hu.reshape(-1, cfg.d_model)).float()
        w = 1.7
        expect = (lu + w * (lc - lu)).argmax(dim=-1).reshape(1, cfg.n_spatial)
        got = _guided_sample_frame(
            model,
            hc,
            hu,
            guidance=w,
            temperature=0.0,
            top_p=1.0,
            chunk=7,
            generator=None,
        )
    assert torch.equal(expect, got)


def test_cfg_dream_w1_equals_conditioned_dream():
    """w == 1 reproduces the plain conditioned rollout (autoregressive_signal_dream)."""
    cfg = _tiny_cfg()
    model = SignalSpacetimeTransformer(cfg).eval()
    sample = _tiny_sample(cfg, seed=2)
    names = _stream_names(cfg)
    from imas_ambix.worldmodel.spacetime_train_v2 import autoregressive_signal_dream

    g1 = torch.Generator().manual_seed(99)
    g2 = torch.Generator().manual_seed(99)
    plain = autoregressive_signal_dream(
        model, sample, stream_names=names, temperature=0.8, top_p=0.95, generator=g1
    )
    cfg_w1 = cfg_guided_dream(
        model,
        sample,
        stream_names=names,
        guidance=1.0,
        temperature=0.8,
        top_p=0.95,
        generator=g2,
    )
    assert np.array_equal(plain, cfg_w1)


def test_cfg_dream_w0_equals_unconditioned_dream():
    """w == 0 reproduces the rollout with ALL conditioning zeroed."""
    cfg = _tiny_cfg()
    model = SignalSpacetimeTransformer(cfg).eval()
    sample = _tiny_sample(cfg, seed=3)
    names = _stream_names(cfg)

    # unconditioned reference: zero the plan + signals on a copy of the sample.
    zero_sample = SignalSpacetimeSample(
        base=SpacetimeSample(
            shot_id=sample.shot_id,
            camera="rbb",
            start_frame=0,
            frames=sample.frames,
            plan=np.zeros_like(sample.plan),
            frame_time=sample.frame_time,
            context_frames=sample.context_frames,
        ),
        signals={k: np.zeros_like(v) for k, v in sample.signals.items()},
    )
    from imas_ambix.worldmodel.spacetime_train_v2 import autoregressive_signal_dream

    g1 = torch.Generator().manual_seed(7)
    g2 = torch.Generator().manual_seed(7)
    uncond = autoregressive_signal_dream(
        model,
        zero_sample,
        stream_names=names,
        temperature=0.7,
        top_p=1.0,
        generator=g1,
    )
    cfg_w0 = cfg_guided_dream(
        model,
        sample,
        stream_names=names,
        guidance=0.0,
        temperature=0.7,
        top_p=1.0,
        generator=g2,
    )
    assert np.array_equal(uncond, cfg_w0)


def test_cfg_dream_context_is_truth_and_shape():
    cfg = _tiny_cfg()
    model = SignalSpacetimeTransformer(cfg).eval()
    sample = _tiny_sample(cfg, t=6, ctx=3, seed=4)
    names = _stream_names(cfg)
    out = cfg_guided_dream(
        model,
        sample,
        stream_names=names,
        guidance=2.0,
        temperature=0.9,
        top_p=0.95,
        generator=torch.Generator().manual_seed(1),
    )
    assert out.shape == (6, cfg.n_spatial)
    # context frames are the truth (only forecast frames are generated).
    assert np.array_equal(out[:3], np.asarray(sample.frames[:3], dtype=np.int64))


def test_cfg_amplifies_difference_from_unconditioned():
    """A larger w pushes the guided logits FURTHER from the unconditioned ones."""
    cfg = _tiny_cfg()
    model = SignalSpacetimeTransformer(cfg).eval()
    sample = _tiny_sample(cfg, seed=5)
    names = _stream_names(cfg)
    from imas_ambix.worldmodel.spacetime_train_v2 import (
        _batch_to,
        collate_signal_windows,
    )

    batch = _batch_to(collate_signal_windows([sample], stream_names=names), "cpu")
    plan, signals = batch["plan"], batch["signals"]
    plan_u, signals_u = _zeroed_conditioning(plan, signals, streams=None)
    cur = torch.as_tensor(sample.frames[:4][None], dtype=torch.long)
    with torch.no_grad():
        hc = model._forward_tokens(cur, plan, signals)[:, -1]
        hu = model._forward_tokens(cur, plan_u, signals_u)[:, -1]
        lc = model.head(hc.reshape(-1, cfg.d_model)).float()
        lu = model.head(hu.reshape(-1, cfg.d_model)).float()
    g15 = (lu + 1.5 * (lc - lu)) - lu
    g20 = (lu + 2.0 * (lc - lu)) - lu
    # ||l_w - l_u|| grows linearly with w (away from the unconditioned point).
    assert g20.norm() > g15.norm()


# ---------------------------------------------------------------------------
# Zeroing semantics
# ---------------------------------------------------------------------------


def test_zero_single_stream_leaves_others_intact():
    plan = torch.randint(1, 9, (1, 4, 2))
    signals = {
        "gas_injection": torch.randint(1, 9, (1, 3, 2)),
        "xma": torch.randint(1, 9, (1, 3, 3)),
    }
    new_plan, new_signals = _zeroed_conditioning(
        plan, signals, streams=["gas_injection"]
    )
    assert torch.equal(new_plan, plan)  # plan untouched
    assert torch.all(new_signals["gas_injection"] == 0)  # gas zeroed
    assert torch.equal(new_signals["xma"], signals["xma"])  # xma untouched


def test_zero_all_zeroes_plan_and_every_stream():
    plan = torch.randint(1, 9, (1, 4, 2))
    signals = {"gas_injection": torch.randint(1, 9, (1, 3, 2))}
    new_plan, new_signals = _zeroed_conditioning(plan, signals, streams=None)
    assert torch.all(new_plan == 0)
    assert torch.all(new_signals["gas_injection"] == 0)


# ---------------------------------------------------------------------------
# Inboard band geometry
# ---------------------------------------------------------------------------


def test_inboard_pixel_cols_left_of_centre():
    # 16 token cols -> 256 px; inboard token band [2,8) -> px [32,128).
    p0, p1 = inboard_pixel_cols(256, token_cols=INBOARD_COLS, grid_w=16)
    assert (p0, p1) == (32, 128)
    # the band is strictly LEFT of the image centre (128).
    assert p1 <= 128


def test_inboard_emission_picks_up_left_band_only():
    # a frame bright only in the left band has high inboard emission; bright only
    # on the right has ~zero inboard emission.
    f_left = np.zeros((2, 16, 16), dtype=np.float64)
    f_left[:, :, 2:8] = 200.0
    f_right = np.zeros((2, 16, 16), dtype=np.float64)
    f_right[:, :, 10:16] = 200.0
    assert inboard_emission_series(f_left).mean() > 100.0
    assert inboard_emission_series(f_right).mean() < 1.0


# ---------------------------------------------------------------------------
# Timing correlation / counterfactual / divergence
# ---------------------------------------------------------------------------


def _frames_with_inboard_level(levels: np.ndarray, h=16, w=16) -> np.ndarray:
    """A frame stack whose left-band luminance follows ``levels`` per frame."""
    f = np.zeros((len(levels), h, w), dtype=np.float64)
    for i, lv in enumerate(levels):
        f[i, :, 2:8] = float(lv)
    return f


def test_timing_correlation_positive_when_emission_tracks_command():
    cmd = np.array([0, 0, 0, 1, 2, 3, 4, 5], dtype=np.float64)
    frames = _frames_with_inboard_level(cmd * 30.0)  # emission tracks command
    res = puff_timing_correlation(frames, cmd, ctx=3)
    assert res["pearson_emission_vs_command"] > 0.9
    assert res["n_forecast_frames"] == 5


def test_counterfactual_delta_positive_when_true_brighter():
    true = _frames_with_inboard_level(np.full(8, 150.0))
    nopuff = _frames_with_inboard_level(np.full(8, 10.0))
    res = counterfactual_delta(true, nopuff, ctx=3)
    assert res["counterfactual_delta"] > 0.0
    assert res["counterfactual_delta_positive"] is True
    assert res["true_mean_inboard_emission"] > res["zeroed_mean_inboard_emission"]


def test_control_divergence_exceeds_small_spread():
    # two conditionings differ a lot; same-conditioning members differ a little.
    a = _frames_with_inboard_level(np.full(8, 200.0))
    b = _frames_with_inboard_level(np.full(8, 0.0))
    m0 = _frames_with_inboard_level(np.full(8, 100.0))
    m1 = _frames_with_inboard_level(np.full(8, 102.0))
    res = control_divergence(a, b, [m0, m1], ctx=3)
    assert res["control_divergence_l1"] > res["same_conditioning_spread_l1"]
    assert res["control_exceeds_spread"] is True
    assert res["divergence_over_spread_ratio"] > 1.0


def test_frame_l1_zero_for_identical():
    a = _frames_with_inboard_level(np.full(6, 50.0))
    assert frame_l1(a, a.copy(), ctx=2) == 0.0


# ---------------------------------------------------------------------------
# Gas command extraction
# ---------------------------------------------------------------------------


def test_gas_command_per_frame_interpolates_to_frames():
    cfg = _tiny_cfg()
    sample = _tiny_sample(cfg, t=6, ctx=3, seed=8)
    cmd = gas_command_per_frame(sample)
    assert cmd.shape == (6,)
    # endpoints match the first/last signal-step means.
    block = np.asarray(sample.signals["gas_injection"], dtype=np.float64)
    assert np.isclose(cmd[0], block[0].mean())
    assert np.isclose(cmd[-1], block[-1].mean())


def test_gas_command_zero_when_absent():
    cfg = _tiny_cfg()
    sample = _tiny_sample(cfg, t=5, ctx=2, seed=9)
    sample.signals.pop("gas_injection")
    cmd = gas_command_per_frame(sample)
    assert cmd.shape == (5,)
    assert np.all(cmd == 0.0)


# ---------------------------------------------------------------------------
# Puff-window selection (find the window where the puff transitions most)
# ---------------------------------------------------------------------------


def test_find_puff_window_picks_the_ramp(monkeypatch):
    import imas_ambix.worldmodel.control_guidance as cg

    # 100-frame command flat at 100, with a ramp 50->150 over frames 40..60.
    cmd = np.full(100, 100.0)
    cmd[40:60] = np.linspace(50, 150, 20)
    monkeypatch.setattr(cg, "_inboard_gas_on_camera_frames", lambda *a, **k: cmd)
    start, std = cg.find_puff_window(1, span=20, min_std=1.0)
    assert start is not None
    # the chosen window must overlap the ramp region (high variance there).
    assert 30 <= start <= 60
    assert std > 1.0


def test_find_puff_window_none_when_flat(monkeypatch):
    import imas_ambix.worldmodel.control_guidance as cg

    monkeypatch.setattr(
        cg, "_inboard_gas_on_camera_frames", lambda *a, **k: np.full(100, 130.0)
    )
    start, std = cg.find_puff_window(1, span=20, min_std=1.0)
    assert start is None  # flat everywhere -> no puff-transition window
    assert std < 1.0


def test_find_puff_window_none_when_unreadable(monkeypatch):
    import imas_ambix.worldmodel.control_guidance as cg

    monkeypatch.setattr(cg, "_inboard_gas_on_camera_frames", lambda *a, **k: None)
    start, std = cg.find_puff_window(1, span=20)
    assert start is None
    assert std == 0.0


def test_signal_sample_exposes_start_frame_and_camera():
    # the driver reads sample.start_frame for the verdict meta; a v2 sample must
    # pass it through (it wraps a v1 SpacetimeSample) — regression guard.
    cfg = _tiny_cfg()
    sample = _tiny_sample(cfg, t=6, ctx=3, seed=0)
    assert sample.start_frame == sample.base.start_frame
    assert sample.camera == sample.base.camera
