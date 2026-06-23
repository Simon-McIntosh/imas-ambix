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
    _bootstrap_mean_ratio_ci,
    _bounded_coil_edit,
    _decoded_divergences,
    _forecast_pixel_l1,
    _is_collapsed_rollout,
    _position_coil_columns,
    _summarise,
    _token_divergences,
    _variance_decomposition,
    _within_shot_ratio_std,
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
    tvr, rvr, tvr_s, rvr_s = _token_divergences(true_t, [r1, r2], ctx)
    assert tvr > 0.0 and rvr > 0.0
    # the per-random / per-pair sample lists back the means.
    assert len(tvr_s) == 2 and len(rvr_s) == 1
    assert abs(float(np.mean(tvr_s)) - tvr) < 1e-12
    assert abs(float(np.mean(rvr_s)) - rvr) < 1e-12


def test_summarise_pass_when_margin_clears_floor():
    # legacy absolute-margin gate (robust_gate=False).
    cfg = EvalConfig(margin_threshold=1.0, floor_ratio=1.5, robust_gate=False)
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
    cfg = EvalConfig(margin_threshold=1.0, floor_ratio=1.5, robust_gate=False)
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
    cfg = EvalConfig(robust_gate=False)
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
# robust gate: collapse rejection + normalised ratio + bootstrap CI
# ---------------------------------------------------------------------------


def _blob_stack(n, h, w, *, brightness, row, col, size=4):
    """A frame stack with a bright blob — high spatial std, NOT collapsed."""
    s = np.zeros((n, h, w), dtype=np.float64)
    s[:, row : row + size, col : col + size] = brightness
    return s


def test_is_collapsed_detects_near_uniform():
    ctx = 2
    flat = np.full((6, 16, 16), 3.0, dtype=np.float64)  # near-uniform low std
    assert _is_collapsed_rollout(flat, ctx) is True
    structured = _blob_stack(6, 16, 16, brightness=200.0, row=4, col=4)
    assert _is_collapsed_rollout(structured, ctx) is False


def test_is_collapsed_detects_near_black_vs_gt_scale():
    ctx = 2
    # a STRUCTURED dream (high spatial std -> not std-collapsed) but DIM on average
    # (a small bright patch on black): mean brightness is low against a bright GT.
    dim = _blob_stack(6, 16, 16, brightness=60.0, row=4, col=4, size=3)
    # not collapsed by the spatial-std test (the patch gives real structure).
    assert _is_collapsed_rollout(dim, ctx, gt_brightness=None) is False
    # vs a BRIGHT GT scale the dim dream is near-black -> collapse fires.
    assert _is_collapsed_rollout(dim, ctx, gt_brightness=200.0) is True
    # the same dream vs a comparably dim GT scale is NOT collapsed.
    assert _is_collapsed_rollout(dim, ctx, gt_brightness=10.0) is False


def test_decoded_divergences_excludes_collapsed_random_from_floor():
    """A collapsed (near-black) random must be dropped from the noise floor."""
    ctx = 2
    gh, gw = 16, 16

    # the decode is mocked: each role's pixels are a chosen pattern.  true + two
    # SIMILAR structured randoms (low pairwise floor) + one COLLAPSED near-black
    # random that diverges far from both (the 18503 degeneracy: a washed-out dream
    # that inflates the random-vs-random floor if KEPT).
    pix = {
        "true": _blob_stack(6, 16, 16, brightness=200.0, row=2, col=2),
        "rand0": _blob_stack(6, 16, 16, brightness=180.0, row=10, col=10),
        "rand1": _blob_stack(6, 16, 16, brightness=180.0, row=10, col=11),
        "rand2": np.full((6, 16, 16, 3), 1.0, dtype=np.float64),  # collapsed
    }

    def fake_decode(grids, roles, *, work_dir, device):
        return {e["role"]: pix[e["role"]] for e in roles}

    true_tok = np.zeros(6 * gh * gw, dtype=np.int64)
    rand_toks = [np.zeros(6 * gh * gw, dtype=np.int64) for _ in range(3)]

    tvr, rvr, n_collapsed, n_kept, tvr_s, rvr_s = _decoded_divergences(
        true_tok,
        rand_toks,
        ctx,
        device="cpu",
        work_dir=None,
        shot_id=1,
        grid_hw=(gh, gw),
        local_to_store=lambda a: a,
        decode_roles=fake_decode,
        reject_collapsed=True,
    )
    assert n_collapsed == 1 and n_kept == 2
    # the floor is the pairwise L1 among the TWO structured randoms only.
    assert rvr > 0.0 and np.isfinite(rvr)
    # the kept-random samples back the means (2 kept -> 2 tvr, 1 pair).
    assert len(tvr_s) == 2 and len(rvr_s) == 1
    # with the collapsed near-black random KEPT the floor would be inflated.
    _, rvr_keep, n_c2, n_k2, _tvr_s2, _rvr_s2 = _decoded_divergences(
        true_tok,
        rand_toks,
        ctx,
        device="cpu",
        work_dir=None,
        shot_id=1,
        grid_hw=(gh, gw),
        local_to_store=lambda a: a,
        decode_roles=fake_decode,
        reject_collapsed=False,
    )
    assert n_c2 == 0 and n_k2 == 3
    assert rvr_keep > rvr  # the collapsed random inflates the kept-everything floor


def test_bootstrap_mean_ratio_ci_brackets_the_mean():
    ratios = [2.0, 2.5, 3.0, 2.2, 2.8, 3.1, 2.6]
    mean, lo, hi = _bootstrap_mean_ratio_ci(
        ratios, n_boot=2000, ci_pct=(2.5, 97.5), seed=0
    )
    assert lo < mean < hi
    assert lo > 1.0  # a clearly-controllable cohort: CI clear of the noise floor


def test_bootstrap_ci_single_finite_is_degenerate():
    mean, lo, hi = _bootstrap_mean_ratio_ci([4.0], n_boot=500, ci_pct=(2.5, 97.5))
    assert mean == lo == hi == 4.0


def test_robust_summarise_pass_needs_ci_clear_of_floor():
    cfg = EvalConfig(robust_gate=True, ratio_threshold=1.5)
    vs = [
        HeldoutDeltaNMVerdict(
            shot_id=s,
            is_transient=True,
            plan_variation=10.0,
            true_vs_random=6.0,
            random_vs_random=2.0,
            margin=4.0,
            ratio=3.0,
            n_random=3,
            n_random_collapsed=0,
            n_random_kept=3,
            passed=True,
        )
        for s in range(8)
    ]
    summ = _summarise(vs, cfg, decode=True)
    assert summ["robust_gate"] is True
    assert summ["metric"] == "delta_nm_decoded_pixel_robust"
    assert summ["verdict"] == "PASS"
    assert summ["pass_fraction"] == 1.0
    assert summ["ratio_ci_lo"] > 1.0
    assert summ["mean_normalised_ratio"] == 3.0


def test_robust_summarise_fail_when_one_shot_carries_cohort():
    """One strong shot among many weak ones must NOT pass the robust gate."""
    cfg = EvalConfig(robust_gate=True, ratio_threshold=1.5)
    strong = HeldoutDeltaNMVerdict(
        shot_id=0,
        is_transient=True,
        plan_variation=10.0,
        true_vs_random=10.0,
        random_vs_random=1.0,
        margin=9.0,
        ratio=10.0,
        n_random=3,
        n_random_kept=3,
        passed=True,
    )
    weak = [
        HeldoutDeltaNMVerdict(
            shot_id=s,
            is_transient=True,
            plan_variation=10.0,
            true_vs_random=0.6,
            random_vs_random=1.0,
            margin=-0.4,
            ratio=0.6,
            n_random=3,
            n_random_kept=3,
            passed=False,
        )
        for s in range(1, 6)
    ]
    summ = _summarise([strong, *weak], cfg, decode=True)
    # 1/6 pass-fraction -> below the 0.5 majority -> FAIL, not carried by one shot.
    assert summ["verdict"] == "FAIL"
    assert summ["pass_fraction"] < 0.5


# ---------------------------------------------------------------------------
# variance decomposition: within-shot sampling noise vs across-shot heterogeneity
# ---------------------------------------------------------------------------


def test_within_shot_ratio_std_zero_for_constant_samples():
    """Constant per-random samples -> the ratio never wobbles -> std 0."""
    tvr = [4.0, 4.0, 4.0, 4.0]
    rvr = [2.0, 2.0, 2.0]
    std = _within_shot_ratio_std(tvr, rvr, n_boot=300, seed=0)
    assert std == 0.0


def test_within_shot_ratio_std_positive_for_noisy_samples():
    """Spread-out per-random samples -> the resampled ratio has real spread."""
    tvr = [1.0, 5.0, 9.0, 2.0, 8.0]
    rvr = [0.5, 4.0, 3.0, 1.0]
    std = _within_shot_ratio_std(tvr, rvr, n_boot=2000, seed=0)
    assert np.isfinite(std) and std > 0.0


def test_within_shot_ratio_std_nan_when_too_few_samples():
    assert np.isnan(_within_shot_ratio_std([4.0], [2.0], n_boot=100))
    assert np.isnan(_within_shot_ratio_std([4.0, 5.0], [], n_boot=100))


def _ratio_verdict(shot, ratio, *, within_std):
    """A minimal verdict carrying just the fields the decomposition reads."""
    return HeldoutDeltaNMVerdict(
        shot_id=shot,
        is_transient=True,
        plan_variation=10.0,
        true_vs_random=ratio,
        random_vs_random=1.0,
        margin=ratio - 1.0,
        ratio=ratio,
        n_random=10,
        n_random_kept=10,
        ratio_within_std=within_std,
        passed=ratio > 1.5,
    )


def test_variance_decomposition_flags_across_shot_heterogeneity():
    """Ratios spread wide across shots but each shot's own estimate is tight ->
    ACROSS-shot heterogeneity dominates; raising n_random would not help."""
    # the 25-shot cohort shape: most shots ~0.5-1.2, two big outliers.
    ratios = [0.5, 0.6, 0.7, 0.9, 1.0, 1.1, 1.2, 6.8, 10.6]
    vs = [_ratio_verdict(i, r, within_std=0.05) for i, r in enumerate(ratios)]
    dec = _variance_decomposition(vs)
    assert dec["across_shot_variance"] > dec["mean_within_shot_variance"]
    assert dec["across_over_within"] >= 3.0
    assert "ACROSS-shot heterogeneity dominates" in dec["interpretation"]


def test_variance_decomposition_flags_within_shot_noise():
    """Per-shot estimates noisy (big within-std) but the underlying ratios cluster
    -> WITHIN-shot sampling noise dominates; raising n_random WILL tighten it."""
    ratios = [1.45, 1.5, 1.55, 1.48, 1.52, 1.5]
    vs = [_ratio_verdict(i, r, within_std=0.8) for i, r in enumerate(ratios)]
    dec = _variance_decomposition(vs)
    assert dec["mean_within_shot_variance"] > dec["across_shot_variance"]
    assert dec["across_over_within"] <= 0.33
    assert "WITHIN-shot sampling noise dominates" in dec["interpretation"]


def test_variance_decomposition_nan_when_insufficient():
    vs = [_ratio_verdict(0, 1.5, within_std=float("nan"))]
    dec = _variance_decomposition(vs)
    assert not np.isfinite(dec["across_over_within"])
    assert "insufficient samples" in dec["interpretation"]


def test_summarise_exposes_variance_decomposition_and_distribution():
    """The robust summary surfaces the decomposition + the sorted ratio list +
    the pass-fraction so the distribution (not just the mean) is visible."""
    cfg = EvalConfig(robust_gate=True, ratio_threshold=1.5)
    ratios = [0.6, 0.8, 1.0, 1.2, 2.0, 3.0, 5.0]
    vs = [_ratio_verdict(i, r, within_std=0.1) for i, r in enumerate(ratios)]
    summ = _summarise(vs, cfg, decode=True)
    assert "variance_decomposition" in summ
    vd = summ["variance_decomposition"]
    assert "across_over_within" in vd and "interpretation" in vd
    # the sorted ratio list is exposed and is actually sorted.
    sr = summ["per_shot_ratios_sorted"]
    assert sr == sorted(sr) and len(sr) == len(ratios)
    # pass-fraction is the stable signal (3/7 here clear 1.5).
    assert abs(summ["pass_fraction"] - 3 / 7) < 1e-9
    assert "median_normalised_ratio" in summ
    assert np.isfinite(summ["mean_within_shot_ratio_std"])


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
