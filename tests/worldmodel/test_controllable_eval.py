"""Unit tests for the post-train held-out controllability eval.

These cover the decoder-free / pure-numpy parts (centroid, coil selection,
bounded coil edit, the divergence scoring + verdict), so the eval harness is
green BEFORE the real re-train checkpoint exists.  The decoded-pixel + GIF paths
need the GPU VQ stack and are smoke-tested separately on a compute node.
"""

from __future__ import annotations

import numpy as np

from imas_ambix.worldmodel.controllable_eval import (
    EvalConfig,
    HeldoutDeltaNMVerdict,
    _bounded_coil_edit,
    _forecast_pixel_l1,
    _position_coil_columns,
    _summarise,
    _token_divergences,
    decoded_centroid,
    diagnostic_match,
)

# ---------------------------------------------------------------------------
# decoded_centroid
# ---------------------------------------------------------------------------


def test_decoded_centroid_finds_the_blob():
    # a 32x32 frame with a bright blob at (row=8, col=24).
    f = np.zeros((2, 32, 32), dtype=np.float64)
    f[0, 6:11, 22:27] = 200.0  # blob centred ~ (8, 24)
    f[1, 20:25, 4:9] = 200.0  # second frame blob ~ (22, 6)
    cen = decoded_centroid(f)
    assert cen.shape == (2, 2)
    assert abs(cen[0, 0] - 8.0) < 1.0 and abs(cen[0, 1] - 24.0) < 1.0
    assert abs(cen[1, 0] - 22.0) < 1.0 and abs(cen[1, 1] - 6.0) < 1.0


def test_decoded_centroid_dark_frame_is_centre():
    f = np.zeros((1, 16, 16), dtype=np.float64)
    cen = decoded_centroid(f)
    assert abs(cen[0, 0] - 7.5) < 1e-6 and abs(cen[0, 1] - 7.5) < 1e-6


def test_decoded_centroid_handles_rgb():
    f = np.zeros((1, 16, 16, 3), dtype=np.float64)
    f[0, 2:5, 10:13, :] = 150.0
    cen = decoded_centroid(f)
    assert abs(cen[0, 0] - 3.0) < 1.0 and abs(cen[0, 1] - 11.0) < 1.0


def test_forecast_pixel_l1_only_scores_forecast():
    a = np.zeros((4, 8, 8), dtype=np.float64)
    b = np.zeros((4, 8, 8), dtype=np.float64)
    a[:2] = 100.0  # differ only in the CONTEXT (frames < ctx=2)
    b[:2] = 0.0
    assert _forecast_pixel_l1(a, b, 2) == 0.0
    b[2:] = 50.0  # now differ in the forecast window
    assert _forecast_pixel_l1(a, b, 2) > 0.0


# ---------------------------------------------------------------------------
# position-coil selection + bounded edit
# ---------------------------------------------------------------------------


def test_position_coil_columns_picks_p4_p5_p6():
    from imas_ambix.worldmodel.actuator_plan import ACTUATOR_CHANNEL_KEYS

    cols = _position_coil_columns(list(ACTUATOR_CHANNEL_KEYS))
    picked = {ACTUATOR_CHANNEL_KEYS[c] for c in cols}
    # every picked key is a p4/p5/p6 coil/current.
    assert picked, "no position coil picked"
    for k in picked:
        assert any(tag in k for tag in ("p4", "p5", "p6"))
    # and it does NOT pick gas/nbi/density/Ip/tf/p2/p3.
    for k in picked:
        assert "gas" not in k and "nbi" not in k and "ne_line" not in k
        assert k not in ("plasma_current", "tf_current")


def test_bounded_coil_edit_scales_one_coil_holds_rest():
    from imas_ambix.worldmodel.actuator_plan import (
        ACTUATOR_CHANNEL_KEYS,
        N_ACTUATOR_CHANNELS,
        ActuatorPlan,
        normalise_actuator_values,
    )

    raw = np.ones((4, N_ACTUATOR_CHANNELS), dtype=np.float32) * 1e4
    plan = ActuatorPlan(
        values=normalise_actuator_values(raw),
        missing=np.zeros_like(raw),
        channel_keys=list(ACTUATOR_CHANNEL_KEYS),
        raw_values=raw,
    )
    col = _position_coil_columns(list(ACTUATOR_CHANNEL_KEYS))[0]
    edited = _bounded_coil_edit(plan, col, frac=0.5)
    # the edited coil's RAW command scaled by 1.5; the rest unchanged.
    assert np.allclose(edited.raw_values[:, col], 1.5e4)
    other = [i for i in range(N_ACTUATOR_CHANNELS) if i != col]
    assert np.allclose(edited.raw_values[:, other], 1e4)


# ---------------------------------------------------------------------------
# divergence scoring + verdict
# ---------------------------------------------------------------------------


def test_token_divergences_true_and_floor():
    # true rollout differs from the randoms in the forecast; the randoms also
    # differ from each other -> both signals non-zero.
    ctx = 2
    true_t = np.zeros((6, 4), dtype=np.int64)
    r1 = true_t.copy()
    r1[ctx:, 0] = 1  # r1 differs from true on the forecast
    r2 = true_t.copy()
    r2[ctx:, 1] = 1  # r2 differs from true elsewhere
    tvr, rvr = _token_divergences(true_t, [r1, r2], ctx)
    assert tvr > 0.0 and rvr > 0.0


def test_summarise_pass_when_margin_clears_floor():
    cfg = EvalConfig(margin_threshold=1.0, floor_ratio=1.5)
    # two transient shots, true-vs-random comfortably above the floor.
    vs = [
        HeldoutDeltaNMVerdict(
            shot_id=s,
            is_transient=True,
            plan_variation=10.0,
            true_vs_random=5.0,
            random_vs_random=1.0,
            margin=4.0,
            ratio=5.0,
            n_random=3,
            passed=True,
        )
        for s in (1, 2)
    ]
    summ = _summarise(vs, cfg, decode=True)
    assert summ["verdict"] == "PASS"
    assert summ["n_transient"] == 2 and summ["n_pass"] == 2
    assert summ["metric"] == "delta_nm_decoded_pixel"


def test_summarise_fail_when_below_floor():
    cfg = EvalConfig(margin_threshold=1.0, floor_ratio=1.5)
    vs = [
        HeldoutDeltaNMVerdict(
            shot_id=1,
            is_transient=True,
            plan_variation=10.0,
            true_vs_random=0.5,
            random_vs_random=1.0,
            margin=-0.5,
            ratio=0.5,
            n_random=3,
            passed=False,
        )
    ]
    summ = _summarise(vs, cfg, decode=True)
    assert summ["verdict"] == "FAIL"
    assert summ["n_pass"] == 0


def test_summarise_not_testable_when_no_transient():
    cfg = EvalConfig()
    vs = [
        HeldoutDeltaNMVerdict(
            shot_id=1,
            is_transient=False,
            plan_variation=0.0,
            true_vs_random=0.0,
            random_vs_random=0.0,
            margin=0.0,
            ratio=float("inf"),
            n_random=3,
            passed=False,
        )
    ]
    summ = _summarise(vs, cfg, decode=False)
    assert summ["gate_testable"] is False
    assert summ["gate_pass"] is False
    assert summ["metric"] == "delta_nm_token_lowerbound"


# ---------------------------------------------------------------------------
# dreamt-vs-real diagnostic-match (joint-generation quantitative axis)
# ---------------------------------------------------------------------------


def _tiny_diag_cfg(*, generate_diagnostics=True, **kw):
    """A tiny ControllableSpacetimeConfig with per-stream diagnostic heads on."""
    from imas_ambix.worldmodel.controllable_model import ControllableSpacetimeConfig
    from imas_ambix.worldmodel.spacetime_model_v2 import SignalStreamSpec

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
        n_signal_steps=4,
        signal_streams=(
            SignalStreamSpec("gas_injection", vocab=8, channels=2),
            SignalStreamSpec("xma", vocab=8, channels=3),
        ),
        actuator_channels=6,
        n_act_steps=4,
        generate_diagnostics=generate_diagnostics,
    )
    base.update(kw)
    return ControllableSpacetimeConfig(**base)


def _mk_diag_sample(cfg, shot, *, t=6, ctx=3, seed=0):
    """A synthetic ControllableSpacetimeSample with NON-PAD measured signals.

    Signal ids are drawn in ``[1, vocab)`` so every next-step target is non-PAD
    (id 0) and the diagnostic-match has scored positions on every stream.
    """
    import torch

    from imas_ambix.worldmodel.actuator_plan import (
        ACTUATOR_CHANNEL_KEYS,
        ActuatorPlan,
        normalise_actuator_values,
    )
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
    sigs = {}
    for st in cfg.signal_streams:
        # ids in [1, vocab) -> no PAD targets, so every stream scores.
        sigs[st.name] = torch.randint(
            1, st.vocab, (cfg.n_signal_steps, st.channels), generator=g
        ).numpy()
    signal = SignalSpacetimeSample(base=base, signals=sigs)
    raw = np.linspace(1e3, 1e5, cfg.n_act_steps, dtype=np.float32)[:, None] * np.ones(
        (1, cfg.actuator_channels), dtype=np.float32
    )
    keys = list(ACTUATOR_CHANNEL_KEYS[: cfg.actuator_channels]) + [
        f"c{i}"
        for i in range(
            cfg.actuator_channels - len(ACTUATOR_CHANNEL_KEYS[: cfg.actuator_channels])
        )
    ]
    plan_act = ActuatorPlan(
        values=normalise_actuator_values(raw),
        missing=np.zeros_like(raw),
        channel_keys=keys,
        raw_values=raw,
    )
    return ControllableSpacetimeSample(signal=signal, actuator=plan_act)


def test_diagnostic_match_scores_per_stream():
    """diagnostic_match returns finite per-stream accuracy in [0,1] + CE >= 0."""
    import torch

    from imas_ambix.worldmodel.controllable_model import (
        ControllableSpacetimeTransformer,
    )

    cfg = _tiny_diag_cfg(generate_diagnostics=True)
    torch.manual_seed(0)
    model = ControllableSpacetimeTransformer(cfg).eval()
    assert model.has_diagnostics
    sample = _mk_diag_sample(cfg, 18502, seed=1)
    stream_names = [st.name for st in cfg.signal_streams]

    res = diagnostic_match(model, sample, stream_names, device="cpu", chunk=64)
    assert res["diagnostics_generated"] is True
    assert set(res["per_stream"].keys()) == set(stream_names)
    for d in res["per_stream"].values():
        assert 0.0 <= d["accuracy"] <= 1.0 and np.isfinite(d["accuracy"])
        assert np.isfinite(d["ce"]) and d["ce"] >= 0.0
        assert d["n"] > 0
    assert 0.0 <= res["mean_accuracy"] <= 1.0 and np.isfinite(res["mean_accuracy"])
    assert np.isfinite(res["mean_ce"]) and res["mean_ce"] >= 0.0


def test_diagnostic_match_camera_only_baseline_returns_flag():
    """A model built generate_diagnostics=False reports diagnostics_generated=False."""
    import torch

    from imas_ambix.worldmodel.controllable_model import (
        ControllableSpacetimeTransformer,
    )

    cfg = _tiny_diag_cfg(generate_diagnostics=False)
    torch.manual_seed(0)
    model = ControllableSpacetimeTransformer(cfg).eval()
    assert not getattr(model, "has_diagnostics", False)
    sample = _mk_diag_sample(cfg, 18503, seed=2)
    stream_names = [st.name for st in cfg.signal_streams]

    res = diagnostic_match(model, sample, stream_names, device="cpu", chunk=64)
    assert res["diagnostics_generated"] is False
    assert res["per_stream"] == {}
    assert res["mean_accuracy"] == 0.0
    assert res["mean_ce"] == 0.0


def test_diagnostic_match_skips_all_pad_stream():
    """An all-PAD stream contributes no scored position but never errors/NaNs."""
    import torch

    from imas_ambix.worldmodel.controllable_model import (
        ControllableSpacetimeTransformer,
    )

    cfg = _tiny_diag_cfg(generate_diagnostics=True)
    torch.manual_seed(0)
    model = ControllableSpacetimeTransformer(cfg).eval()
    sample = _mk_diag_sample(cfg, 18504, seed=3)
    # force one stream to all-PAD (id 0) — it must drop out, the other scores.
    pad_name = cfg.signal_streams[0].name
    sample.signal.signals[pad_name] = np.zeros_like(sample.signal.signals[pad_name])
    stream_names = [st.name for st in cfg.signal_streams]

    res = diagnostic_match(model, sample, stream_names, device="cpu", chunk=64)
    assert pad_name not in res["per_stream"]
    assert res["per_stream"], "the non-PAD stream should still score"
    assert np.isfinite(res["mean_accuracy"]) and np.isfinite(res["mean_ce"])
