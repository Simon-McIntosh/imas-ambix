"""Tests for the model-independent latent controllability gate.

All synthetic, CPU-only: a SYNTHETIC latent dynamics ``rollout_fn`` lets the gate
be exercised before any world model exists.

Two regimes:

- NULL dynamics — the latent trajectory IGNORES the plan (forecast frames depend
  only on the conditioning context + a per-rollout noise seed, not on the plan).
  The true-vs-random ratio must sit at the noise floor (~1.0) and its bootstrap CI
  must bracket 1.0 — the gate must NOT fire.
- CONTROLLED dynamics — the plan ADDS a systematic, plan-dependent push to the
  forecast latent.  The true plan then diverges from the randoms MORE than randoms
  diverge from each other, so the ratio + CI clear 1.0 — the gate FIRES.

These two regimes ARE the n=10 ΔN-M noise-floor characterisation the powered gate
requires (see ``test_noise_floor_characterisation``).
"""

from __future__ import annotations

import numpy as np
import pytest

from imas_ambix.eval.controllability_gate import (
    GateConfig,
    LatentShotVerdict,
    _bootstrap_mean_ratio_ci,
    _is_collapsed_latent,
    _variance_decomposition,
    controllability_gate,
    evaluate_shot,
    latent_divergence,
)

# A plan is just an opaque vector here; only the rollout_fn interprets it.
PLAN_DIM = 4
LATENT_DIM = 6
N_FRAMES = 24
CTX = 8


def _sample_random_plan(rng: np.random.Generator) -> np.ndarray:
    """An in-distribution counterfactual plan: a bounded random command vector."""
    return rng.normal(0.0, 1.0, size=PLAN_DIM)


def _make_null_rollout(seed_base: int):
    """Latent dynamics that IGNORE the plan (the null model).

    Each call draws a fresh per-rollout noise field for the forecast window — the
    forecast varies from rollout to rollout, but that variation is plan-INDEPENDENT,
    so the true plan diverges from randoms no more than randoms diverge from each
    other.  The context frames are shared (identical) across rollouts.
    """
    ctx_block = np.linspace(0.0, 1.0, CTX)[:, None] * np.ones((1, LATENT_DIM))
    counter = {"n": 0}

    def rollout(plan) -> np.ndarray:
        # a fresh, plan-independent forecast each call (the dead/null dynamics).
        rng = np.random.default_rng(seed_base + counter["n"])
        counter["n"] += 1
        forecast = rng.normal(0.0, 1.0, size=(N_FRAMES - CTX, LATENT_DIM))
        return np.concatenate([ctx_block, forecast], axis=0)

    return rollout


def _make_controlled_rollout(seed_base: int, gain: float = 6.0):
    """Latent dynamics where the plan drives the latent to a distinct operating point.

    The forecast latent is a small plan-independent noise PLUS a deterministic
    plan-dependent push ``gain * (plan @ W)`` broadcast over the forecast frames.
    A given plan produces a reproducible push, so the true plan reaches a latent
    operating point that the RANDOM counterfactuals (a different, tightly-clustered
    region of plan space — see ``_sample_random_plan_offset``) do not: true-vs-random
    divergence (the distance to the random cluster) then exceeds random-vs-random
    (the within-cluster spread), pushing the ratio clear of 1.0.

    This is the honest structure of a controllable system — the true actuator
    trajectory is a *distinguished* operating point, not just one more draw from the
    same distribution as the counterfactuals.  When randoms are drawn from the SAME
    distribution as the true plan (the null counterfactual scheme), the ratio sits
    at 1.0 even with strong dynamics — exactly the noise floor the gate measures.
    """
    ctx_block = np.linspace(0.0, 1.0, CTX)[:, None] * np.ones((1, LATENT_DIM))
    rng_w = np.random.default_rng(seed_base)
    w = rng_w.normal(0.0, 1.0, size=(PLAN_DIM, LATENT_DIM))
    counter = {"n": 0}

    def rollout(plan) -> np.ndarray:
        rng = np.random.default_rng(seed_base + 10_000 + counter["n"])
        counter["n"] += 1
        noise = rng.normal(0.0, 0.2, size=(N_FRAMES - CTX, LATENT_DIM))
        push = gain * (np.asarray(plan, dtype=np.float64) @ w)  # (LATENT_DIM,)
        ramp = np.linspace(0.0, 1.0, N_FRAMES - CTX)[:, None]
        forecast = noise + ramp * push[None, :]
        return np.concatenate([ctx_block, forecast], axis=0)

    return rollout


#: random counterfactuals are a TIGHT cluster offset away from the true plan
#: (which lives near the origin).  This is the in-distribution-but-distinct
#: counterfactual scheme under which a controllable system's true plan is
#: distinguished from its random alternatives.
_RANDOM_OFFSET = 3.0
_RANDOM_SPREAD = 0.25


def _sample_random_plan_offset(rng: np.random.Generator) -> np.ndarray:
    """A counterfactual from a tight cluster offset from the true (origin) plan."""
    return _RANDOM_OFFSET + rng.normal(0.0, _RANDOM_SPREAD, size=PLAN_DIM)


def _cohort(n_shots: int = 6):
    """Null cohort: true plans drawn from the SAME distribution as the randoms."""
    rng = np.random.default_rng(0)
    return [
        (1000 + i, rng.normal(0.0, 1.0, size=PLAN_DIM), CTX) for i in range(n_shots)
    ]


def _controlled_cohort(n_shots: int = 6):
    """Controlled cohort: the true plan is the distinguished (near-origin) operating
    point, while ``_sample_random_plan_offset`` draws the offset counterfactuals."""
    rng = np.random.default_rng(1)
    return [
        (2000 + i, rng.normal(0.0, _RANDOM_SPREAD, size=PLAN_DIM), CTX)
        for i in range(n_shots)
    ]


# ---------------------------------------------------------------------------
# (i) NULL dynamics -> ratio CI brackets the floor; gate does NOT fire
# ---------------------------------------------------------------------------


def test_null_dynamics_ratio_brackets_floor_gate_does_not_fire():
    cfg = GateConfig(n_random=10, seed=0)
    shots = _cohort()
    # one rollout_fn per shot so each shot's null noise is independent.
    rollouts = {sid: _make_null_rollout(seed_base=sid) for sid, _, _ in shots}

    # per-shot dynamics differ, so evaluate each shot then aggregate the cohort.
    from imas_ambix.eval.controllability_gate import _summarise

    verdicts = [
        evaluate_shot(
            rollouts[sid],
            true_plan,
            _sample_random_plan,
            context_frames=ctx,
            config=cfg,
            shot_id=sid,
        )
        for sid, true_plan, ctx in shots
    ]
    result = _summarise(verdicts, cfg)

    # Under the null the cohort mean ratio sits at the floor and the CI brackets it.
    assert result.ratio_ci_lo <= 1.0 <= result.ratio_ci_hi, (
        f"null ratio CI should bracket 1.0, got "
        f"[{result.ratio_ci_lo:.3f}, {result.ratio_ci_hi:.3f}]"
    )
    assert not result.gate_pass, "gate must NOT fire under null dynamics"
    assert result.verdict == "FAIL"


# ---------------------------------------------------------------------------
# (ii) CONTROLLED dynamics -> ratio + CI clear the floor; gate FIRES
# ---------------------------------------------------------------------------


def test_controlled_dynamics_ratio_clears_floor_gate_fires():
    cfg = GateConfig(n_random=10, seed=0)
    shots = _controlled_cohort()
    rollouts = {sid: _make_controlled_rollout(seed_base=sid) for sid, _, _ in shots}

    verdicts = [
        evaluate_shot(
            rollouts[sid],
            true_plan,
            _sample_random_plan_offset,
            context_frames=ctx,
            config=cfg,
            shot_id=sid,
        )
        for sid, true_plan, ctx in shots
    ]
    from imas_ambix.eval.controllability_gate import _summarise

    result = _summarise(verdicts, cfg)

    assert result.ratio_ci_lo > 1.0, (
        f"controlled ratio CI lower bound should clear 1.0, got "
        f"{result.ratio_ci_lo:.3f}"
    )
    assert result.gate_pass, "gate must FIRE under controlled dynamics"
    assert result.verdict == "PASS"
    assert result.pass_fraction >= 0.5


def test_controllability_gate_end_to_end_controlled():
    """The public ``controllability_gate`` dispatch fires on controlled dynamics."""
    cfg = GateConfig(n_random=10, seed=0)
    # one global controlled dynamics so a single rollout_fn serves all shots;
    # the shared W means every shot is driveable.
    rollout = _make_controlled_rollout(seed_base=777)
    shots = _controlled_cohort()
    result = controllability_gate(
        rollout, shots, _sample_random_plan_offset, config=cfg
    )
    assert result.gate_pass
    assert result.ratio_ci_lo > 1.0


def test_controllability_gate_end_to_end_null():
    cfg = GateConfig(n_random=10, seed=0)
    rollout = _make_null_rollout(seed_base=555)
    shots = _cohort()
    result = controllability_gate(rollout, shots, _sample_random_plan, config=cfg)
    assert not result.gate_pass
    assert result.ratio_ci_lo <= 1.0 <= result.ratio_ci_hi


# ---------------------------------------------------------------------------
# (iii) Collapsed rollout rejection
# ---------------------------------------------------------------------------


def test_collapsed_latent_detected():
    ctx = CTX
    # a near-constant forecast trajectory is collapsed.
    flat = np.ones((N_FRAMES, LATENT_DIM))
    flat[:CTX] = np.linspace(0, 1, CTX)[:, None]  # context moves; forecast does not
    assert _is_collapsed_latent(flat, ctx)
    # a moving forecast is not collapsed.
    moving = np.cumsum(np.ones((N_FRAMES, LATENT_DIM)), axis=0)
    assert not _is_collapsed_latent(moving, ctx)


def test_collapsed_random_excluded_from_floor():
    """A collapsed random rollout is dropped from the noise floor."""
    cfg = GateConfig(n_random=4, reject_collapsed=True, seed=0)
    ctx = CTX

    moving_ctx = np.linspace(0.0, 1.0, CTX)[:, None] * np.ones((1, LATENT_DIM))

    def make_traj(forecast):
        return np.concatenate([moving_ctx, forecast], axis=0)

    rng_state = {"n": 0}

    def rollout(plan) -> np.ndarray:
        # first random plan collapses (constant forecast); the rest move.
        i = rng_state["n"]
        rng_state["n"] += 1
        if i == 1:  # the second rollout (first random after the true) collapses
            return make_traj(np.zeros((N_FRAMES - CTX, LATENT_DIM)))
        r = np.random.default_rng(900 + i)
        return make_traj(r.normal(0.0, 1.0, size=(N_FRAMES - CTX, LATENT_DIM)))

    true_plan = np.ones(PLAN_DIM)
    v = evaluate_shot(
        rollout,
        true_plan,
        _sample_random_plan,
        context_frames=ctx,
        config=cfg,
        shot_id=1,
        rng=np.random.default_rng(0),
    )
    assert v.n_random_collapsed == 1
    assert v.n_random_kept == cfg.n_random - 1


# ---------------------------------------------------------------------------
# (iv) Bootstrap CI + variance-decomposition shapes
# ---------------------------------------------------------------------------


def test_bootstrap_ci_ordering_and_shape():
    ratios = [1.0, 1.5, 2.0, 2.5, 3.0, 0.8]
    mean, lo, hi = _bootstrap_mean_ratio_ci(
        ratios, n_boot=2000, ci_pct=(2.5, 97.5), seed=0
    )
    assert lo <= mean <= hi
    assert np.isclose(mean, float(np.mean(ratios)))


def test_bootstrap_ci_excludes_infinite_ratios():
    ratios = [1.5, 2.0, float("inf"), 2.5]
    mean, lo, hi = _bootstrap_mean_ratio_ci(
        ratios, n_boot=1000, ci_pct=(2.5, 97.5), seed=0
    )
    assert np.isfinite(mean) and np.isfinite(lo) and np.isfinite(hi)
    assert np.isclose(mean, np.mean([1.5, 2.0, 2.5]))


def test_variance_decomposition_shape_and_keys():
    verdicts = [
        LatentShotVerdict(
            shot_id=i,
            true_vs_random=2.0,
            random_vs_random=1.0,
            margin=1.0,
            ratio=2.0 + 0.3 * i,
            n_random=10,
            passed=True,
            ratio_within_std=0.1,
        )
        for i in range(5)
    ]
    vd = _variance_decomposition(verdicts)
    for key in (
        "mean_within_shot_variance",
        "across_shot_variance",
        "across_over_within",
        "n_shots_with_within_std",
        "interpretation",
    ):
        assert key in vd
    assert vd["n_shots_with_within_std"] == 5
    assert np.isfinite(vd["across_shot_variance"])


def test_latent_divergence_metrics():
    a = np.zeros((N_FRAMES, LATENT_DIM))
    b = np.zeros((N_FRAMES, LATENT_DIM))
    b[CTX:] = 1.0
    # l2 over forecast: each forecast frame differs by ||ones(D)|| = sqrt(D).
    assert np.isclose(latent_divergence(a, b, CTX, metric="l2"), np.sqrt(LATENT_DIM))
    # identical trajectories diverge by 0 in both metrics.
    assert latent_divergence(a, a, CTX, metric="l2") == 0.0
    assert latent_divergence(b, b, CTX, metric="cosine") == 0.0
    with pytest.raises(ValueError):
        latent_divergence(a, b, CTX, metric="nonsense")


# ---------------------------------------------------------------------------
# Noise-floor characterisation (the n=10 BAR the gate requires)
# ---------------------------------------------------------------------------


def test_noise_floor_characterisation(capsys):
    """Characterise the n=10 ΔN-M noise floor + verify the gate fires only with control.

    Reports (printed; visible with -s):
      - null ratio mean + bootstrap CI (the floor) — must bracket 1.0;
      - controlled ratio mean + CI — must clear 1.0;
      - the within- vs across-shot variance split at n=10 (powered-size justification).
    """
    cfg = GateConfig(n_random=10, seed=0)
    from imas_ambix.eval.controllability_gate import _summarise

    # NULL: the latent ignores the plan; randoms from the true plan's own
    # distribution.  This is the noise floor the gate must clear.
    null_shots = _cohort(n_shots=8)
    null_v = [
        evaluate_shot(
            _make_null_rollout(seed_base=sid),
            tp,
            _sample_random_plan,
            context_frames=ctx,
            config=cfg,
            shot_id=sid,
        )
        for sid, tp, ctx in null_shots
    ]
    null = _summarise(null_v, cfg)

    # CONTROLLED: the true plan reaches a distinguished latent operating point that
    # the offset random counterfactuals do not — the ratio clears the floor.
    ctrl_shots = _controlled_cohort(n_shots=8)
    ctrl_v = [
        evaluate_shot(
            _make_controlled_rollout(seed_base=sid),
            tp,
            _sample_random_plan_offset,
            context_frames=ctx,
            config=cfg,
            shot_id=sid,
        )
        for sid, tp, ctx in ctrl_shots
    ]
    ctrl = _summarise(ctrl_v, cfg)

    print("\n=== n=10 delta-N-M noise-floor characterisation ===")
    print(
        f"NULL  : ratio={null.mean_ratio:.3f} "
        f"CI=[{null.ratio_ci_lo:.3f},{null.ratio_ci_hi:.3f}] "
        f"pass_frac={null.pass_fraction:.2f} verdict={null.verdict}"
    )
    print(
        f"CTRL  : ratio={ctrl.mean_ratio:.3f} "
        f"CI=[{ctrl.ratio_ci_lo:.3f},{ctrl.ratio_ci_hi:.3f}] "
        f"pass_frac={ctrl.pass_fraction:.2f} verdict={ctrl.verdict}"
    )
    vd = null.variance_decomposition
    print(
        f"NULL var split: within={vd['mean_within_shot_variance']:.4g} "
        f"across={vd['across_shot_variance']:.4g} "
        f"across/within={vd['across_over_within']:.3f}"
    )
    print(f"  {vd['interpretation']}")

    # The bar: null brackets the floor (gate dark), control clears it (gate fires).
    assert null.ratio_ci_lo <= 1.0 <= null.ratio_ci_hi
    assert not null.gate_pass
    assert ctrl.ratio_ci_lo > 1.0
    assert ctrl.gate_pass
