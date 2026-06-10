"""Pre-registered metric numerics on synthetic / random inputs (W1/W2/W3)."""

from __future__ import annotations

import numpy as np
import pytest

from imas_ambix.camdyn.metrics import (
    PROBE_TARGETS,
    ProbeProtocol,
    bootstrap_ci,
    crps_gaussian,
    horizon_frame_offsets,
    horizon_reconstruction_accuracy,
    masked_token_nll,
    masked_top1_accuracy,
    motion_weighted_subset,
    persistence_baseline_accuracy,
    probe_rmse,
)

# --------------------------------------------------------------------------
# W1 — masked NLL + top-1 + bootstrap CI
# --------------------------------------------------------------------------


def test_nll_perfect_logits_near_zero():
    V = 8
    targets = np.array([1, 3, 5])
    logits = np.full((3, V), -50.0)
    logits[np.arange(3), targets] = 50.0  # near-deterministic
    mask = np.ones(3, dtype=bool)
    nll = masked_token_nll(logits, targets, mask)
    assert nll == pytest.approx(0.0, abs=1e-6)


def test_nll_uniform_logits_equals_log_vocab():
    V = 16
    logits = np.zeros((10, V))  # uniform
    targets = np.random.default_rng(0).integers(0, V, size=10)
    mask = np.ones(10, dtype=bool)
    nll = masked_token_nll(logits, targets, mask)
    assert nll == pytest.approx(np.log(V), rel=1e-6)


def test_nll_only_scores_masked_positions():
    V = 4
    logits = np.zeros((5, V))
    targets = np.zeros(5, dtype=int)
    mask = np.array([True, False, True, False, False])
    # reduce=none returns one entry per masked token
    per = masked_token_nll(logits, targets, mask, reduce="none")
    assert per.shape == (2,)


def test_top1_accuracy_counts_argmax_hits():
    logits = np.array(
        [
            [0, 1, 0, 0, 0],  # pred 1
            [3, 0, 0, 0, 0],  # pred 0
            [0, 0, 0, 0, 2],  # pred 4
        ],
        dtype=float,
    )
    targets = np.array([1, 2, 4])  # hits: 1, miss, 4 → 2/3
    mask = np.ones(3, dtype=bool)
    assert masked_top1_accuracy(logits, targets, mask) == pytest.approx(2 / 3)


def test_bootstrap_ci_clear_of_zero_for_positive_signal():
    rng = np.random.default_rng(0)
    diff = rng.normal(0.5, 0.1, size=2000)  # strongly positive
    ci = bootstrap_ci(diff, seed=0)
    assert ci["mean"] == pytest.approx(0.5, abs=0.02)
    assert ci["lo"] > 0
    assert ci["clear_of_zero"]
    assert ci["favours_dynamics"]  # directional W1 win


def test_bootstrap_ci_significant_regression_is_not_a_w1_win():
    # paired diff oriented so positive favours dynamics; a strongly
    # NEGATIVE diff means dynamics is significantly WORSE.  The CI is
    # clear of zero (two-sided) but this is NOT a W1 win — the directional
    # gate favours_dynamics must be False so a regression cannot be
    # miscounted as a win.
    rng = np.random.default_rng(7)
    diff = rng.normal(-0.5, 0.1, size=2000)  # strongly negative
    ci = bootstrap_ci(diff, seed=0)
    assert ci["hi"] < 0
    assert ci["clear_of_zero"]  # significant in some direction
    assert not ci["favours_dynamics"]  # but dynamics did NOT win


def test_bootstrap_ci_not_clear_for_zero_mean():
    rng = np.random.default_rng(1)
    diff = rng.normal(0.0, 1.0, size=2000)
    ci = bootstrap_ci(diff, seed=0)
    assert not ci["clear_of_zero"]
    assert not ci["favours_dynamics"]
    assert ci["lo"] < 0 < ci["hi"]


def test_bootstrap_ci_empty_input():
    ci = bootstrap_ci(np.array([]))
    assert not ci["clear_of_zero"]
    assert not ci["favours_dynamics"]


# --------------------------------------------------------------------------
# Motion-weighted subset (reported alongside W1)
# --------------------------------------------------------------------------


def test_motion_subset_static_tokens_excluded():
    # all frames identical → nothing moves
    tokens = np.ones((10, 4, 4), dtype=int)
    ft = np.arange(10) * 0.001  # 1 ms cadence
    moving = motion_weighted_subset(tokens, ft, window_ms=50.0)
    assert moving.shape == (10, 4, 4)
    assert not moving.any()


def test_motion_subset_flags_changing_cell():
    tokens = np.zeros((10, 2, 2), dtype=int)
    tokens[5, 0, 0] = 99  # one cell changes at frame 5
    ft = np.arange(10) * 0.001  # 1 ms → all within ±50 ms
    moving = motion_weighted_subset(tokens, ft, window_ms=50.0)
    # the (0,0) cell is flagged moving in frames near 5; (1,1) never
    assert moving[:, 0, 0].any()
    assert not moving[:, 1, 1].any()


def test_motion_subset_window_respects_timestamps():
    tokens = np.zeros((6, 1, 1), dtype=int)
    tokens[3, 0, 0] = 7
    ft = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])  # 100 ms cadence
    # ±50 ms window covers no neighbour → no token sees the change of others
    moving = motion_weighted_subset(tokens, ft, window_ms=50.0)
    assert not moving.any()


# --------------------------------------------------------------------------
# W2 — horizon offsets + reconstruction + persistence baseline
# --------------------------------------------------------------------------


def test_horizon_frame_offsets_physical():
    ft = np.arange(300) * 0.001  # 1 ms cadence
    offs = horizon_frame_offsets(ft, horizons_ms=(10.0, 50.0, 200.0))
    assert offs[10.0] == 10
    assert offs[50.0] == 50
    assert offs[200.0] == 200


def test_horizon_offsets_scale_with_cadence():
    ft = np.arange(50) * 0.002  # 2 ms cadence → 10 ms = 5 frames
    offs = horizon_frame_offsets(ft, horizons_ms=(10.0,))
    assert offs[10.0] == 5


def test_horizon_reconstruction_valid_and_invalid():
    nfr, H, W, V = 60, 2, 2, 4
    rng = np.random.default_rng(0)
    target = rng.integers(0, V, size=(nfr, H, W))
    # perfect logits at every frame
    logits = np.full((nfr, H, W, V), -20.0)
    idx = np.indices((nfr, H, W))
    logits[idx[0], idx[1], idx[2], target] = 20.0
    ft = np.arange(nfr) * 0.001
    res = horizon_reconstruction_accuracy(logits, target, ft, frontier_frame=10)
    # 10 ms (off 10) and 50 ms (off 50): 10+50=60 == nfr → invalid; 10+10 valid
    assert res[10.0]["valid"] == 1.0
    assert res[10.0]["top1_acc"] == pytest.approx(1.0)
    assert res[200.0]["valid"] == 0.0  # 10+200 out of range


def test_persistence_baseline_perfect_for_static_future():
    nfr, H, W = 40, 2, 2
    target = np.zeros((nfr, H, W), dtype=int)  # static → persistence perfect
    ft = np.arange(nfr) * 0.001
    res = persistence_baseline_accuracy(target, ft, frontier_frame=5)
    assert res[10.0]["valid"] == 1.0
    assert res[10.0]["top1_acc"] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# W3 — frozen probe protocol + RMSE + CRPS
# --------------------------------------------------------------------------


def test_probe_targets_are_the_four_w3_diagnostics():
    assert set(PROBE_TARGETS) == {
        "dalpha",
        "ne_line_integrated",
        "te_core",
        "n2_mode_amp",
    }
    assert PROBE_TARGETS["dalpha"] == "ada"
    assert PROBE_TARGETS["ne_line_integrated"] == "ane"
    assert PROBE_TARGETS["te_core"] == "ayc"
    assert PROBE_TARGETS["n2_mode_amp"] == "ama"


def test_linear_probe_recovers_linear_target():
    rng = np.random.default_rng(0)
    N, D, T = 400, 8, 2
    X = rng.standard_normal((N, D))
    Wtrue = rng.standard_normal((D, T))
    Y = X @ Wtrue + 0.01 * rng.standard_normal((N, T))
    probe = ProbeProtocol(probe_kind="linear", ridge_lambda=1e-3).fit(X, Y)
    pred = probe.predict(X)
    rmse = probe_rmse(pred, Y)
    assert np.all(rmse < 0.1)


def test_probe_frozen_generalises_to_heldout():
    rng = np.random.default_rng(1)
    D, T = 6, 1
    Wtrue = rng.standard_normal((D, T))
    Xtr = rng.standard_normal((300, D))
    Ytr = Xtr @ Wtrue
    Xho = rng.standard_normal((100, D))
    Yho = Xho @ Wtrue
    probe = ProbeProtocol(probe_kind="linear", ridge_lambda=1e-4).fit(Xtr, Ytr)
    pred = probe.predict(Xho)  # frozen — no refit
    assert probe_rmse(pred, Yho)[0] < 0.05


def test_probe_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        ProbeProtocol().predict(np.zeros((2, 3)))


def test_probe_mlp1_runs_and_scores():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((200, 5))
    Y = X[:, :1] ** 2  # nonlinear-ish; mlp1 random features should help a bit
    probe = ProbeProtocol(probe_kind="mlp1", hidden_dim=32).fit(X, Y)
    pred = probe.predict(X)
    assert pred.shape == (200, 1)
    assert np.isfinite(probe_rmse(pred, Y)).all()


def test_probe_rmse_ignores_nan_truth():
    pred = np.array([[1.0], [2.0], [3.0]])
    truth = np.array([[1.0], [np.nan], [3.0]])
    rmse = probe_rmse(pred, truth)
    assert rmse[0] == pytest.approx(0.0)


def test_crps_lower_for_accurate_forecast():
    truth = np.zeros((100, 1))
    accurate = crps_gaussian(np.zeros((100, 1)), np.full((100, 1), 0.5), truth)
    biased = crps_gaussian(np.full((100, 1), 3.0), np.full((100, 1), 0.5), truth)
    assert accurate[0] < biased[0]


def test_crps_ignores_nan_truth():
    truth = np.array([[0.0], [np.nan]])
    out = crps_gaussian(np.zeros((2, 1)), np.ones((2, 1)), truth)
    assert np.isfinite(out[0])
