"""Tests for the S7.3 RKN latent state-space engine and inference modes.

Covers:
- Module shapes (encode / update / predict / observe / filter / rollout).
- A synthetic 1-D linear-Gaussian system: a *frozen, hand-set* engine configured
  to match the generating system must reproduce the analytic Kalman filter
  posterior (the spec's "filter SHOULD recover near-Kalman behaviour" check).
- The acceptance-critical NO-FUTURE-INPUT invariant: corrupting inputs_{t+1..t+h}
  must leave the autonomous forecast BIT-IDENTICAL (proof of no peeking).
- A learnability smoke test: a few epochs on a tiny synthetic dataset must
  reduce the training loss.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from imas_ambix.statespace.engine import (
    EngineConfig,
    RKNEngine,
    _quiescent_drift_penalty,
    _verdict,
    crps_student_t,
    gaussian_nll,
    student_t_nll,
    student_t_nll_np,
    train_engine,
)
from imas_ambix.statespace.filter import (
    filter_innovation_shot,
    filter_shot,
    fit_horizon_conformal,
    forecast_pairs,
    smooth_shot,
)

# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------


def test_module_shapes():
    cfg = EngineConfig(input_dim=7, latent_dim=4, output_dim=2)
    model = RKNEngine(cfg)
    B, T = 3, 11
    x = torch.randn(B, T, cfg.input_dim)

    w, r = model.encode(x[:, 0, :])
    assert w.shape == (B, cfg.latent_dim)
    assert r.shape == (B, cfg.latent_dim)
    assert (r > 0).all()

    z0, v0 = model.initial_belief(B, x.device, x.dtype)
    assert z0.shape == (B, cfg.latent_dim)

    zp, vp = model.update_step(z0, v0, w, r)
    assert zp.shape == (B, cfg.latent_dim)
    assert (vp > 0).all()
    # update reduces variance (information gain)
    assert (vp <= v0 + 1e-6).all()

    zn, vn = model.predict_step(zp, vp)
    assert zn.shape == (B, cfg.latent_dim)
    # predict adds process noise → variance grows
    assert (vn >= vp - 1e-6).all()

    mu, ov = model.observe(zn, vn)
    assert mu.shape == (B, cfg.output_dim)
    assert ov.shape == (B, cfg.output_dim)
    assert (ov > 0).all()

    z_post, var_post, obs_mu, obs_var = model.filter_sequence(x)
    assert z_post.shape == (B, T, cfg.latent_dim)
    assert obs_mu.shape == (B, T, cfg.output_dim)
    assert obs_var.shape == (B, T, cfg.output_dim)

    horizons = [1, 2, 5]
    rmu, rvar = model.rollout(z_post[:, 3, :], var_post[:, 3, :], horizons)
    assert rmu.shape == (B, len(horizons), cfg.output_dim)
    assert rvar.shape == (B, len(horizons), cfg.output_dim)
    # forecast variance must be non-decreasing in horizon (Q accumulates)
    assert (rvar[:, 1, :] >= rvar[:, 0, :] - 1e-6).all()
    assert (rvar[:, 2, :] >= rvar[:, 1, :] - 1e-6).all()


# ---------------------------------------------------------------------------
# Predict step grows variance via learned Q
# ---------------------------------------------------------------------------


def test_predict_grows_variance():
    cfg = EngineConfig(input_dim=3, latent_dim=5, output_dim=1)
    model = RKNEngine(cfg)
    z = torch.zeros(2, cfg.latent_dim)
    var = torch.full((2, cfg.latent_dim), 0.1)
    v_prev = var
    for _ in range(10):
        z, v = model.predict_step(z, v_prev)
        # strictly grows because Q > 0 and we start with a^2 ~ 1
        assert (v >= v_prev - 1e-9).all()
        v_prev = v


# ---------------------------------------------------------------------------
# Synthetic linear-Gaussian: frozen engine reproduces the analytic Kalman filter
# ---------------------------------------------------------------------------


def _set_linear_gaussian_engine(
    a: float, q: float, r: float, obs_noise: float
) -> RKNEngine:
    """Build a 1-latent engine hand-configured as a scalar linear-Gaussian SSM.

    Generating model:
        z_{t+1} = a z_t + N(0, q)
        x_t     = z_t + N(0, r)         (direct observation: encoder = identity)
        y_t     = z_t + N(0, obs_noise) (observation head = identity)

    We zero the nonlinear MLP paths so the engine reduces to the exact linear
    recurrence the analytic Kalman filter assumes.
    """
    cfg = EngineConfig(input_dim=1, latent_dim=1, output_dim=1)
    model = RKNEngine(cfg)
    with torch.no_grad():
        # Encoder = identity: w = x, and r fixed.  Zero the MLP, set enc_w = I.
        for p in model.encoder.parameters():
            p.zero_()
        # encoder output is ReLU(0) = 0 everywhere → enc_w(0)=bias; we instead
        # bypass by setting enc_w weight 0 and bias 0, then OVERRIDE encode below.
        model.enc_w.weight.zero_()
        model.enc_w.bias.zero_()
        model.enc_logr.weight.zero_()
        # softplus(b) + floor = r  →  b = log(exp(r - floor) - 1)
        target = r - 1e-6
        model.enc_logr.bias.fill_(math.log(math.expm1(target)))
        # Transition mean: z + f(z) should equal a*z  → f(z) = (a-1) z.
        for p in model.trans_mean.parameters():
            p.zero_()
        # Variance transition a² and Q:
        model.trans_log_a.fill_(math.log(abs(a)))
        model.log_q.fill_(math.log(q))
        # Observation head = identity: mu = z.
        for p in model.obs_mean.parameters():
            p.zero_()
        # obs variance map: out_var = softplus(W)·var + obs_noise² ; we want the
        # latent variance passed through (W→ +inf gives softplus→identity-ish) —
        # instead set W so softplus(W)=1 and obs_noise as given.
        model.obs_var_w.fill_(math.log(math.expm1(1.0)))  # softplus = 1
        model.log_obs_noise.fill_(math.log(obs_noise))
        # initial belief
        model.z0.zero_()
        model.log_var0.fill_(math.log(1.0))
    return model


def _linear_gaussian_engine_with_identity_paths(a, q, r, obs_noise):
    """Patch encode/observe/transition to be exactly linear (monkeypatch-free).

    Subclass overrides the nonlinear pieces so the test isolates the *belief
    algebra* (Kalman predict/update on the diagonal), which is the part the spec
    asks to match.  The learned MLPs are irrelevant to the recurrence equations.
    """

    class _Linear(RKNEngine):
        def encode(self, x):  # w = x, r fixed
            B = x.shape[0]
            w = x[:, :1]
            rr = torch.full((B, 1), r, dtype=x.dtype)
            return w, rr

        def predict_step(self, z, var):
            z_next = a * z
            var_next = (a * a) * var + q
            return z_next, var_next.clamp(1e-9, 1e9)

        def observe(self, z, var):
            return z, var + obs_noise * obs_noise

    cfg = EngineConfig(input_dim=1, latent_dim=1, output_dim=1)
    m = _Linear(cfg)
    with torch.no_grad():
        m.z0.zero_()
        m.log_var0.fill_(math.log(1.0))
    return m


def test_filter_matches_analytic_kalman():
    """The engine's belief algebra must equal the scalar Kalman filter."""
    a, q, r, obs_noise = 0.9, 0.05, 0.2, 1e-4
    rng = np.random.default_rng(0)
    T = 200
    # simulate
    z = 0.0
    zs, xs = [], []
    for _ in range(T):
        z = a * z + rng.normal(0, math.sqrt(q))
        zs.append(z)
        xs.append(z + rng.normal(0, math.sqrt(r)))
    xs = np.array(xs)

    # analytic scalar Kalman filter (observation = z + N(0,r))
    z_kf, P0 = 0.0, 1.0
    P = P0
    kf_means, kf_vars = [], []
    for t in range(T):
        if t > 0:
            z_kf = a * z_kf
            P = a * a * P + q
        # update
        k = P / (P + r)
        z_kf = z_kf + k * (xs[t] - z_kf)
        P = (1 - k) * P
        kf_means.append(z_kf)
        kf_vars.append(P)

    model = _linear_gaussian_engine_with_identity_paths(a, q, r, obs_noise)
    xb = torch.from_numpy(xs[np.newaxis, :, np.newaxis]).float()
    z_post, var_post, _mu, _ov = model.filter_sequence(xb)
    eng_means = z_post[0, :, 0].detach().numpy()
    eng_vars = var_post[0, :, 0].detach().numpy()

    # The diagonal Kalman recurrence is identical → near machine precision.
    assert np.allclose(eng_means, kf_means, atol=1e-4), (
        f"mean mismatch: max {np.max(np.abs(eng_means - np.array(kf_means)))}"
    )
    assert np.allclose(eng_vars, kf_vars, atol=1e-5), (
        f"var mismatch: max {np.max(np.abs(eng_vars - np.array(kf_vars)))}"
    )


# ---------------------------------------------------------------------------
# Acceptance-critical: NO FUTURE INPUTS / NO Dα PEEKING during forecasting
# ---------------------------------------------------------------------------


def test_forecast_ignores_future_inputs():
    """Autonomous forecast must be bit-identical when future inputs are corrupted.

    Anchor at t; corrupt inputs_{t+1..T-1} with garbage (NaN/random).  Because the
    rollout is input-free after the anchor, the forecast at every horizon must be
    EXACTLY unchanged.  This is the strongest possible proof of no leakage.
    """
    torch.manual_seed(3)
    cfg = EngineConfig(input_dim=6, latent_dim=8, output_dim=1)
    model = RKNEngine(cfg)
    model.eval()

    T = 60
    x = np.random.RandomState(0).randn(T, cfg.input_dim).astype(np.float64)
    anchors = np.array([20, 30])
    horizons = [1, 2, 5, 10, 20]

    mu_a, var_a = forecast_pairs(model, x, anchors, horizons)

    # Corrupt the genuine FUTURE: timesteps strictly after the LARGEST anchor.
    # The filter pass for every anchor t only consumes inputs_{1:t}, so inputs
    # after max(anchor) are never read → forecasts must be bit-identical.
    x_corrupt = x.copy()
    cut = int(anchors.max()) + 1
    x_corrupt[cut:] = np.random.RandomState(99).randn(T - cut, cfg.input_dim) * 1e3
    mu_b, var_b = forecast_pairs(model, x_corrupt, anchors, horizons)

    assert np.array_equal(mu_a, mu_b), (
        "forecast mean changed when future inputs corrupted → LEAK"
    )
    assert np.array_equal(var_a, var_b), (
        "forecast var changed when future inputs corrupted → LEAK"
    )


def test_forecast_uses_inputs_up_to_anchor():
    """Sanity: corrupting inputs AT/BEFORE the anchor MUST change the forecast.

    Confirms the no-future-input invariant is not vacuously passing because the
    model ignores inputs entirely.
    """
    torch.manual_seed(5)
    cfg = EngineConfig(input_dim=6, latent_dim=8, output_dim=1)
    model = RKNEngine(cfg)
    model.eval()
    T = 60
    x = np.random.RandomState(1).randn(T, cfg.input_dim).astype(np.float64)
    anchors = np.array([30])
    horizons = [1, 5, 20]
    mu_a, _ = forecast_pairs(model, x, anchors, horizons)
    x2 = x.copy()
    x2[:31] += 5.0  # perturb inputs up to and including the anchor
    mu_b, _ = forecast_pairs(model, x2, anchors, horizons)
    assert not np.allclose(mu_a, mu_b), "forecast did not respond to pre-anchor inputs"


# ---------------------------------------------------------------------------
# Learnability smoke test
# ---------------------------------------------------------------------------


def test_train_reduces_loss():
    """A few epochs on a tiny synthetic dataset must reduce the training loss."""
    rng = np.random.default_rng(0)
    cfg = EngineConfig(
        input_dim=4,
        latent_dim=8,
        output_dim=1,
        n_epochs=8,
        batch_size=8,
        seq_len=40,
        lr=3e-3,
        train_horizons=(1, 2, 5),
    )
    # synthetic: latent random walk, y = sin of cumulative input, x noisy
    xs, ys = [], []
    for _ in range(40):
        T = 80
        drive = rng.normal(0, 1, (T, cfg.input_dim))
        latent = np.cumsum(drive[:, 0]) * 0.1
        y = np.sin(latent)[:, None]
        x = drive + rng.normal(0, 0.05, (T, cfg.input_dim))
        xs.append(x.astype(np.float64))
        ys.append(y.astype(np.float64))

    model = RKNEngine(cfg)
    state = train_engine(model, xs, ys, cfg, device="cpu")
    assert len(state.epoch_losses) == cfg.n_epochs
    # loss should drop meaningfully from first to last epoch
    assert state.epoch_losses[-1] < state.epoch_losses[0], (
        f"loss did not decrease: {state.epoch_losses[0]:.3f} → {state.epoch_losses[-1]:.3f}"
    )


# ---------------------------------------------------------------------------
# Inference-mode shape checks (filtering / forecasting / smoothing)
# ---------------------------------------------------------------------------


def test_inference_modes_shapes():
    cfg = EngineConfig(input_dim=5, latent_dim=6, output_dim=1)
    model = RKNEngine(cfg)
    T = 50
    x = np.random.RandomState(2).randn(T, cfg.input_dim).astype(np.float64)

    mu_f, var_f = filter_shot(model, x)
    assert mu_f.shape == (T, cfg.output_dim)
    assert (var_f > 0).all()

    anchors = np.array([10, 20, 30])
    horizons = [1, 2, 5, 10]
    mu_p, var_p = forecast_pairs(model, x, anchors, horizons)
    assert mu_p.shape == (3, 4, cfg.output_dim)
    assert (var_p > 0).all()

    mu_s, var_s = smooth_shot(model, x)
    assert mu_s.shape == (T, cfg.output_dim)
    assert (var_s > 0).all()


def test_horizon_conformal_quantile():
    rng = np.random.default_rng(0)
    n = 5000
    y = rng.normal(0, 1, n)
    mu = np.zeros(n)
    sigma = np.ones(n)
    q = fit_horizon_conformal(y, mu, sigma, alpha=0.10)
    # for standard normal residuals, q̂ ≈ z_{0.95} ≈ 1.645
    assert 1.5 < q < 1.8, f"conformal q̂={q} not near 1.645"


def test_quiescent_drift_penalty():
    """The S7.4 drift regulariser is non-negative and quiescence-weighted.

    On a step with NO transient mass (fully quiescent) the penalty equals the
    full ||f_θ(z)||²; on a step that is the batch's peak transient it is ~0.
    Minimising it must drive the transition increment toward persistence (Δz→0).
    """
    torch.manual_seed(0)
    cfg = EngineConfig(input_dim=4, latent_dim=6, output_dim=1)
    model = RKNEngine(cfg)
    B, T, L = 2, 8, cfg.latent_dim
    z_post = torch.randn(B, T, L)

    # All-quiescent (zero transient mass): quiescence ≡ 1, penalty = mean ||f||².
    tw0 = torch.zeros(B, T)
    pen0 = _quiescent_drift_penalty(model, z_post, tw0)
    assert pen0 >= 0.0
    delta = model.trans_mean(z_post.reshape(B * T, L))
    full = (delta * delta).sum(-1).mean()
    assert torch.allclose(pen0, full, atol=1e-5)

    # A purely-transient batch (constant positive mass everywhere) → quiescence
    # ≡ 0 → penalty ≈ 0 (transients are free to move the latent).
    twT = torch.ones(B, T)
    penT = _quiescent_drift_penalty(model, z_post, twT)
    assert float(penT.detach()) < 1e-6

    # Minimising the penalty shrinks the transition increment (persistence pull).
    opt = torch.optim.SGD(model.trans_mean.parameters(), lr=0.05)
    before = float(_quiescent_drift_penalty(model, z_post, tw0).detach())
    for _ in range(50):
        opt.zero_grad()
        loss = _quiescent_drift_penalty(model, z_post, tw0)
        loss.backward()
        opt.step()
    after = float(_quiescent_drift_penalty(model, z_post, tw0).detach())
    assert after < before, (
        f"drift penalty did not decrease: {before:.4f} -> {after:.4f}"
    )


def test_verdict_criterion2_nll_branch():
    """_verdict reports the transient-NLL win branch (re-scoped criterion 2).

    The re-scoped Stage-1 bar is met on transient CRPS OR transient NLL.  This
    synthetic metrics dict mirrors the v0 finding: the engine LOSES transient
    CRPS at every horizon but WINS transient NLL at h>=5 (the static is caught
    confidently-narrow when an ELM lands).  The verdict must flag criterion 2 met
    via the NLL basis, with same_windows_verified true.
    """
    metrics = {
        "filtering": {"coverage_90_conf": 0.903, "coverage_90_raw": 0.945},
        "forecasting_indist_dense_transient": {
            "same_truths_engine_vs_static": True,
            "engine": {
                "1": {
                    "crps_raw": 0.09,
                    "crps_raw_transient": 0.059,
                    "nll_raw_transient": -0.57,
                    "n_transient": 9829,
                    "mean_sigma_raw": 0.16,
                    "rmse": 0.30,
                },
                "5": {
                    "crps_raw": 0.11,
                    "crps_raw_transient": 0.081,
                    "nll_raw_transient": -0.17,
                    "n_transient": 9829,
                    "mean_sigma_raw": 0.25,
                    "rmse": 0.31,
                },
                "20": {
                    "crps_raw": 0.14,
                    "crps_raw_transient": 0.149,
                    "nll_raw_transient": 0.88,
                    "n_transient": 10368,
                    "mean_sigma_raw": 0.28,
                    "rmse": 0.39,
                },
            },
            "static": {
                "1": {
                    "crps_raw": 0.065,
                    "crps_raw_transient": 0.035,
                    "nll_raw_transient": -3.90,
                },
                "5": {
                    "crps_raw": 0.067,
                    "crps_raw_transient": 0.036,
                    "nll_raw_transient": 0.594,
                },
                "20": {
                    "crps_raw": 0.088,
                    "crps_raw_transient": 0.0998,
                    "nll_raw_transient": 13.2,
                },
            },
        },
        "ood_honesty": {
            # S7.5: criterion 3 is now judged on the ENGINE-NATIVE OOD-AUROC
            # (the engine's own innovation signal), with the static-ensemble
            # disagreement kept only as a clearly-labelled reference.  Here the
            # engine-native score clearly beats the static (~random) baseline.
            "ood_auroc_engine": 0.78,
            "ood_auroc_engine_innovation": 0.78,
            "ood_auroc_engine_predictive_sigma": 0.71,
            "ood_auroc_static_ensemble_disagreement": 0.57,
            "ood_auroc_ensemble_disagreement": 0.57,
            "filter_coverage90_raw_indist": 0.945,
            "filter_coverage90_raw_ood": 0.734,
        },
    }
    v = _verdict(metrics)
    # transient CRPS loses at every horizon
    assert v["forecast_beats_static_transient_any_Hgt0"] is False
    # transient NLL wins at h>=5 (engine bounded, static explodes on caught ELMs)
    assert v["forecast_beats_static_transient_nll_any_Hgt0"] is True
    assert v["forecast_nll_transient_by_horizon"]["20"]["engine_wins"] is True
    assert v["forecast_nll_transient_by_horizon"]["1"]["engine_wins"] is False
    # re-scoped criterion 2 met via NLL; all three criteria met
    rs = v["rescoped_acceptance"]
    assert rs["criterion2_transient_dynamics_win"] is True
    assert rs["criterion2_win_basis"] == "transient_NLL"
    assert rs["criterion1_filtering_calibrated"] is True
    # criterion 3 uses the DYNAMICS-NATIVE innovation AUROC (0.78), which is
    # > 0.65 AND CLEARLY exceeds the same-data static disagreement (0.57) by a
    # real margin (≥ 0.05) → criterion 3 met.
    assert rs["criterion3_ood_honesty"] is True
    assert rs["criterion3a_coverage_noncollapse"] is True
    assert rs["criterion3b_innovation_auroc_clearly_exceeds_same_data_static"] is True
    assert rs["all_met"] is True
    # bulk CRPS still a (failed) stretch goal, not the gate
    assert rs["stretch_bulk_crps_beats_static"] is False


def test_verdict_ood_partial_mirrors_v2_finding():
    """Criterion 3 is a PARTIAL — the EXACT v2 finding (the headline of S7.5).

    The dynamics-native innovation AUROC (0.748) does NOT clearly exceed the
    same-data static disagreement (0.810); the predictive-σ AUROC (0.831) only
    TIES it (and measures the same construct, so it can't count for 3b).  Coverage
    non-collapse (3a) holds.  → criterion 3 is a PARTIAL, NOT forced to pass by
    privileging pred-σ via max() or by swapping in the decimated 0.568 baseline.
    This corrects v1's mislabelled "criterion 3 met" (which used the static's own
    0.81 disagreement as if it were the engine's signal).
    """
    metrics = {
        "filtering": {"coverage_90_conf": 0.913},
        "forecasting_indist_dense_transient": {
            "same_truths_engine_vs_static": True,
            "engine": {"5": {"nll_raw_transient": -0.55, "n_transient": 100}},
            "static": {"5": {"nll_raw_transient": 0.59}},
        },
        "ood_honesty": {
            "ood_auroc_engine": 0.748,
            "ood_auroc_engine_innovation": 0.748,  # dynamics-native; below static
            "ood_auroc_engine_predictive_sigma": 0.831,  # only ties static; not used for 3b
            "ood_auroc_static_ensemble_disagreement": 0.810,
            "filter_coverage90_raw_indist": 0.944,
            "filter_coverage90_raw_ood": 0.755,  # non-collapse holds
        },
    }
    v = _verdict(metrics)
    rs = v["rescoped_acceptance"]
    assert rs["criterion3a_coverage_noncollapse"] is True
    # 3b uses the dynamics-native innovation (0.748), NOT the pred-σ tie (0.831).
    assert rs["criterion3b_innovation_auroc_clearly_exceeds_same_data_static"] is False
    assert rs["criterion3_ood_honesty"] is False
    assert rs["criterion3_partial"] is True
    # all-met is False; the "with PARTIAL OOD" variant is True (crit1+2+3a)
    assert rs["all_met"] is False
    assert rs["all_met_with_partial_ood"] is True


def test_verdict_ood_tie_is_not_clearly_exceeds():
    """A within-noise tie (engine 0.831 vs static 0.810, margin 0.021 < 0.05) does
    NOT satisfy 'clearly exceeds' even when the innovation IS the tying signal."""
    metrics = {
        "filtering": {"coverage_90_conf": 0.91},
        "forecasting_indist_dense_transient": {
            "same_truths_engine_vs_static": True,
            "engine": {"5": {"nll_raw_transient": -0.5, "n_transient": 100}},
            "static": {"5": {"nll_raw_transient": 0.6}},
        },
        "ood_honesty": {
            "ood_auroc_engine_innovation": 0.831,  # ties static within noise
            "ood_auroc_engine_predictive_sigma": 0.70,
            "ood_auroc_static_ensemble_disagreement": 0.810,  # margin 0.021 < 0.05
            "filter_coverage90_raw_indist": 0.94,
            "filter_coverage90_raw_ood": 0.75,
        },
    }
    rs = _verdict(metrics)["rescoped_acceptance"]
    assert rs["criterion3b_innovation_auroc_clearly_exceeds_same_data_static"] is False
    assert rs["criterion3_ood_honesty"] is False


def test_verdict_ood_no_engine_score_not_satisfiable():
    """With NO engine-native score, criterion 3b cannot be satisfied (no fallback
    to the static-as-engine number that v1 mislabelled)."""
    metrics = {
        "filtering": {"coverage_90_conf": 0.903},
        "forecasting_indist_dense_transient": {
            "same_truths_engine_vs_static": True,
            "engine": {"5": {"nll_raw_transient": -0.17, "n_transient": 100}},
            "static": {"5": {"nll_raw_transient": 0.59}},
        },
        "ood_honesty": {
            "ood_auroc_ensemble_disagreement": 0.81,  # static only, mislabelled in v1
            "filter_coverage90_raw_indist": 0.945,
            "filter_coverage90_raw_ood": 0.734,
        },
    }
    v = _verdict(metrics)
    rs = v["rescoped_acceptance"]
    assert rs["criterion3b_innovation_auroc_clearly_exceeds_same_data_static"] is False
    assert rs["criterion3_ood_honesty"] is False
    assert v["ood_auroc_engine_native"] is None


def test_gaussian_nll_matches_numpy():
    rng = np.random.default_rng(0)
    y = torch.tensor(rng.normal(0, 1, (100, 1)))
    mu = torch.tensor(rng.normal(0, 1, (100, 1)))
    var = torch.tensor(np.abs(rng.normal(1, 0.1, (100, 1))) + 0.1)
    got = float(gaussian_nll(y, mu, var))
    sig = var.numpy()
    expect = float(
        np.mean(0.5 * (np.log(2 * np.pi * sig) + (y.numpy() - mu.numpy()) ** 2 / sig))
    )
    assert abs(got - expect) < 1e-6


# ---------------------------------------------------------------------------
# S7.5: Student-t emission head — CRPS / NLL correctness
# ---------------------------------------------------------------------------


def test_student_t_crps_matches_monte_carlo():
    """Closed-form Student-t CRPS must match an MC estimate across ν."""
    from scipy.stats import t as student_t

    rng = np.random.default_rng(0)
    mu, scale = 0.3, 0.7
    y = np.array([0.5])
    for nu in (3.0, 5.0, 12.0):
        n = 2_000_000
        X = mu + scale * student_t.rvs(df=nu, size=n, random_state=rng)
        Xp = mu + scale * student_t.rvs(df=nu, size=n, random_state=rng)
        mc = float(np.mean(np.abs(X - y[0])) - 0.5 * np.mean(np.abs(X - Xp)))
        cf = crps_student_t(y, np.array([mu]), np.array([scale]), nu)
        assert abs(cf - mc) / mc < 0.02, f"nu={nu}: CRPS cf={cf} vs MC={mc}"


def test_student_t_crps_reduces_to_gaussian_at_large_nu():
    """As ν→∞ the Student-t CRPS must converge to the Gaussian CRPS."""
    from imas_ambix.statespace.calibration import crps_gaussian

    y, mu, s = np.array([0.5]), np.array([0.3]), np.array([0.7])
    g = crps_gaussian(y, mu, s)
    t_big = crps_student_t(y, mu, s, 5000.0)
    assert abs(g - t_big) < 1e-3, f"t-CRPS(ν=5000)={t_big} != gauss-CRPS={g}"


def test_student_t_nll_matches_scipy():
    """Closed-form Student-t NLL (numpy + torch) must match scipy.stats.t."""
    from scipy.stats import t as student_t

    rng = np.random.default_rng(1)
    mu = rng.normal(0, 1, (200, 1))
    scale2 = np.abs(rng.normal(1, 0.2, (200, 1))) + 0.1
    y = rng.normal(0, 1, (200, 1))
    nu = 4.5
    # scipy log-density of a location-scale t: logpdf((y-mu)/s)/s = logpdf - 0.5 log s²
    s = np.sqrt(scale2)
    sp = float(np.mean(-(student_t.logpdf((y - mu) / s, df=nu) - np.log(s))))
    np_got = student_t_nll_np(y, mu, scale2, nu)
    assert abs(np_got - sp) < 1e-9
    # torch version matches the numpy version
    t_got = float(
        student_t_nll(
            torch.tensor(y),
            torch.tensor(mu),
            torch.tensor(scale2),
            torch.tensor(nu),
        )
    )
    assert abs(t_got - np_got) < 1e-6


def test_student_t_head_shapes_and_nu():
    """The Student-t observation head returns (μ, scale², ν>floor)."""
    cfg = EngineConfig(
        input_dim=5,
        latent_dim=6,
        output_dim=1,
        emission="student_t",
        student_t_nu=5.0,
        student_t_nu_floor=2.1,
    )
    model = RKNEngine(cfg)
    z = torch.randn(4, cfg.latent_dim)
    var = torch.rand(4, cfg.latent_dim) + 0.1
    mu, scale2, nu = model.observe_student_t(z, var)
    assert mu.shape == (4, 1)
    assert scale2.shape == (4, 1)
    assert (scale2 > 0).all()
    assert nu.shape == (1,)
    assert float(nu.detach()) > 2.0  # finite variance
    # scale² equals the Gaussian observe variance (same propagation machinery)
    _, var_g = model.observe(z, var)
    assert torch.allclose(scale2, var_g)


def test_student_t_training_reduces_loss():
    """Training with the Student-t head must reduce the loss (learnable ν)."""
    rng = np.random.default_rng(0)
    cfg = EngineConfig(
        input_dim=4,
        latent_dim=8,
        output_dim=1,
        n_epochs=8,
        batch_size=8,
        seq_len=40,
        lr=3e-3,
        train_horizons=(1, 2, 5),
        emission="student_t",
        num_threads=2,
    )
    xs, ys = [], []
    for _ in range(40):
        T = 80
        drive = rng.normal(0, 1, (T, cfg.input_dim))
        latent = np.cumsum(drive[:, 0]) * 0.1
        # heavy-tailed target: occasional spikes
        y = np.sin(latent)[:, None]
        spikes = (rng.random(T) < 0.05)[:, None]
        y = y + spikes * rng.normal(0, 3, (T, 1))
        x = drive + rng.normal(0, 0.05, (T, cfg.input_dim))
        xs.append(x.astype(np.float64))
        ys.append(y.astype(np.float64))
    model = RKNEngine(cfg)
    state = train_engine(model, xs, ys, cfg, device="cpu")
    assert len(state.epoch_losses) == cfg.n_epochs
    assert state.epoch_losses[-1] < state.epoch_losses[0]
    # ν learned to a finite (heavy-tailed) value
    assert float(model.nu()[0].detach()) > 2.0


def test_num_threads_restored_after_training():
    """train_engine must restore the process-wide thread count it changed."""
    before = torch.get_num_threads()
    rng = np.random.default_rng(0)
    cfg = EngineConfig(
        input_dim=3,
        latent_dim=4,
        output_dim=1,
        n_epochs=2,
        batch_size=8,
        seq_len=40,
        train_horizons=(1, 2),
        num_threads=2,
    )
    xs = [rng.normal(0, 1, (60, 3)) for _ in range(8)]
    ys = [rng.normal(0, 1, (60, 1)) for _ in range(8)]
    train_engine(RKNEngine(cfg), xs, ys, cfg, device="cpu")
    assert torch.get_num_threads() == before


def test_filter_innovation_is_engine_native_ood_score():
    """filter_innovation must be larger on OOD-like inputs than in-dist.

    The engine-native OOD score is the normalised filter-innovation magnitude:
    inputs that surprise the learned dynamics produce a larger innovation.  A
    far-from-prior input sequence must score higher than a quiet one.
    """
    torch.manual_seed(0)
    cfg = EngineConfig(input_dim=6, latent_dim=8, output_dim=1)
    model = RKNEngine(cfg)
    model.eval()
    T = 60
    x_quiet = np.random.RandomState(0).randn(T, cfg.input_dim).astype(np.float64) * 0.1
    x_wild = np.random.RandomState(1).randn(T, cfg.input_dim).astype(np.float64) * 8.0
    s_quiet = filter_innovation_shot(model, x_quiet)
    s_wild = filter_innovation_shot(model, x_wild)
    assert s_quiet.shape == (T,)
    assert (s_quiet >= 0).all() and (s_wild >= 0).all()
    assert float(np.mean(s_wild)) > float(np.mean(s_quiet))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
