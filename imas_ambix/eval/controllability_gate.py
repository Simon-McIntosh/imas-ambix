"""Model-independent powered controllability gate on the LATENT state.

This is the ΔN-M controllability verdict lifted off the camera world model: it
asks whether a TRUE actuator plan moves the LATENT trajectory MORE than RANDOM
plans do, and decides PASS/FAIL against the random-vs-random NOISE FLOOR with a
bootstrap CI and collapse rejection.  Because it operates on an abstract LATENT
trajectory (an ``(T, D)`` array) rather than camera brightness or VQ tokens, it
is decoupled from any particular decoder/codebook and reusable by the latent
world-model engine the instant that engine can produce closed-loop rollouts.

The camera-specific original (``imas_ambix.worldmodel.controllable_eval``) scored
the divergence in decoded-pixel L1 / token-mismatch.  Here the divergence is a
generic LATENT trajectory metric (L2 or cosine over the forecast window), so the
gate has no opinion about how the latent is rendered.

Interface (so it is testable NOW, before the model exists):

- ``rollout_fn(plan) -> np.ndarray`` of shape ``(T, D)`` — a closed-loop latent
  rollout under an actuator plan.  The orchestrator/engine supplies the real one;
  the tests supply synthetic dynamics.
- ``sample_random_plan(rng) -> plan`` — draw an in-distribution counterfactual
  plan.  ``plan`` is opaque to the gate (only ``rollout_fn`` interprets it).

For each shot the gate rolls the TRUE plan and ``n_random`` random plans, scores
the forecast-window true-vs-random divergence against the random-vs-random floor
(collapsed rollouts excluded from the floor), forms a per-shot ratio, and
aggregates the cohort with :func:`_bootstrap_mean_ratio_ci` /
:func:`_variance_decomposition`.  The result is a :class:`LatentDeltaNMVerdict`
(the cohort verdict) carrying the per-shot :class:`LatentShotVerdict` list — the
same statistical shape as the camera ``HeldoutDeltaNMVerdict``.

No GPU, no model, no IMAS data: pure NumPy on latent arrays.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

# ---------------------------------------------------------------------------
# Latent trajectory divergence (the camera-free generalisation of the kernel)
# ---------------------------------------------------------------------------


def _forecast_slice(traj: np.ndarray, ctx: int) -> np.ndarray:
    """The forecast window ``traj[ctx:]`` as a 2-D ``(F, D)`` float64 block.

    A latent trajectory is ``(T, D)`` (or ``(T,)`` for a scalar latent, promoted
    to ``(T, 1)``).  Frames ``0..ctx-1`` are the conditioning context the rollout
    shares with every plan; only the forecast frames ``>= ctx`` carry plan-driven
    divergence, so the divergence kernels score over this slice.
    """
    a = np.asarray(traj, dtype=np.float64)
    if a.ndim == 1:
        a = a[:, None]
    if a.shape[0] <= ctx:
        return a[:0]
    return a[ctx:]


def latent_divergence(
    a: np.ndarray, b: np.ndarray, ctx: int, *, metric: str = "l2"
) -> float:
    """Forecast-window divergence between two latent trajectories.

    ``metric="l2"`` (default): mean over forecast frames of the per-frame Euclidean
    distance ``||a_t - b_t||_2`` — the natural latent analogue of the decoded-pixel
    L1 the camera gate used.  ``metric="cosine"``: mean per-frame cosine DISTANCE
    ``1 - cos(a_t, b_t)`` (scale-invariant; a zero vector contributes 0).  Returns
    0.0 when there is no forecast window (trajectories no longer than ``ctx``).
    """
    fa = _forecast_slice(a, ctx)
    fb = _forecast_slice(b, ctx)
    n = min(fa.shape[0], fb.shape[0])
    if n == 0:
        return 0.0
    fa, fb = fa[:n], fb[:n]
    if metric == "l2":
        return float(np.linalg.norm(fa - fb, axis=1).mean())
    if metric == "cosine":
        na = np.linalg.norm(fa, axis=1)
        nb = np.linalg.norm(fb, axis=1)
        denom = na * nb
        dot = (fa * fb).sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            cos = np.where(denom > 0.0, dot / denom, 1.0)
        # clamp into [-1, 1] so floating-point round-off can't make an identical
        # pair report a tiny negative distance.
        cos = np.clip(cos, -1.0, 1.0)
        return float((1.0 - cos).mean())
    raise ValueError(f"unknown latent divergence metric {metric!r}")


#: A latent rollout is COLLAPSED when its forecast window carries almost no
#: temporal/spatial variation (a degenerate, near-constant trajectory) — its
#: variation then reflects a dead rollout, not plan-driven motion, so it is
#: excluded from the random-vs-random NOISE FLOOR (the collapse test).
COLLAPSE_MIN_STD = 1e-6


def _is_collapsed_latent(
    traj: np.ndarray, ctx: int, *, min_std: float = COLLAPSE_MIN_STD
) -> bool:
    """Is this latent rollout COLLAPSED (a near-constant forecast trajectory)?

    A counterfactual whose forecast latent barely moves across frames is a
    degenerate rollout — it inflates the random-vs-random floor with dead-rollout
    noise rather than real plan-driven variation, so it is dropped from the floor.
    The test is: the mean per-dimension temporal std over the forecast window is
    below ``min_std`` (the trajectory does not move).  Returns ``True`` when
    collapsed.  A forecast window shorter than 2 frames cannot be judged collapsed.
    """
    fwin = _forecast_slice(traj, ctx)
    if fwin.shape[0] < 2:
        return False
    return float(fwin.std(axis=0).mean()) < float(min_std)


# ---------------------------------------------------------------------------
# Verdict containers (same statistical shape as the camera HeldoutDeltaNMVerdict)
# ---------------------------------------------------------------------------


@dataclass
class LatentShotVerdict:
    """Per-shot true-vs-random latent divergence verdict."""

    shot_id: int
    true_vs_random: float  # mean latent divergence, true plan vs random plans
    random_vs_random: float  # mean pairwise divergence among NON-collapsed randoms
    margin: float
    ratio: float
    n_random: int
    passed: bool
    n_random_collapsed: int = 0
    n_random_kept: int = 0
    #: per-(kept-random) true-vs-random divergences — the samples behind ``tvr``.
    true_vs_random_samples: list = field(default_factory=list)
    #: per-pair random-vs-random divergences — the samples behind ``rvr``.
    random_vs_random_samples: list = field(default_factory=list)
    #: within-shot std of the per-shot ratio, bootstrapped over the random rollouts.
    ratio_within_std: float = float("nan")

    def to_dict(self) -> dict:
        return {
            "shot_id": self.shot_id,
            "true_vs_random": self.true_vs_random,
            "random_vs_random": self.random_vs_random,
            "margin": self.margin,
            "ratio": self.ratio,
            "ratio_within_std": self.ratio_within_std,
            "n_random": self.n_random,
            "n_random_collapsed": self.n_random_collapsed,
            "n_random_kept": self.n_random_kept,
            "passed": self.passed,
        }


@dataclass
class LatentDeltaNMVerdict:
    """Cohort controllability verdict — the powered, collapse-rejecting gate.

    Mirrors the camera ``HeldoutDeltaNMVerdict`` aggregation: a cohort mean ratio
    (true-vs-random / noise floor) with a bootstrap CI, a pass fraction over the
    shots, and the variance decomposition saying whether more random rollouts
    would tighten the CI.  ``gate_pass`` is the verdict: a majority of shots clear
    the per-shot ratio threshold AND the cohort mean-ratio bootstrap CI lower bound
    is clear of the noise floor (1.0) — so a single good shot cannot carry it.
    """

    n_shots: int
    n_pass: int
    pass_fraction: float
    mean_true_vs_random: float
    mean_random_vs_random_noise_floor: float
    mean_margin: float
    mean_ratio: float
    ratio_ci_lo: float
    ratio_ci_hi: float
    n_random: int
    n_ratio_infinite: int
    gate_pass: bool
    verdict: str
    variance_decomposition: dict = field(default_factory=dict)
    per_shot_ratios_sorted: list = field(default_factory=list)
    per_shot: list = field(default_factory=list)
    metric: str = "latent_delta_nm"

    def to_dict(self) -> dict:
        return {
            "metric": self.metric,
            "n_shots": self.n_shots,
            "n_pass": self.n_pass,
            "pass_fraction": self.pass_fraction,
            "mean_true_vs_random": self.mean_true_vs_random,
            "mean_random_vs_random_noise_floor": self.mean_random_vs_random_noise_floor,
            "mean_margin": self.mean_margin,
            "mean_ratio": self.mean_ratio,
            "ratio_ci_lo": self.ratio_ci_lo,
            "ratio_ci_hi": self.ratio_ci_hi,
            "n_random": self.n_random,
            "n_ratio_infinite": self.n_ratio_infinite,
            "variance_decomposition": self.variance_decomposition,
            "per_shot_ratios_sorted": self.per_shot_ratios_sorted,
            "per_shot": [v.to_dict() for v in self.per_shot],
            "gate_pass": self.gate_pass,
            "verdict": self.verdict,
        }


@dataclass
class GateConfig:
    """Knobs for the powered latent controllability gate."""

    #: random counterfactual rollouts per shot.  The per-shot ratio is a mean over
    #: these; more shrinks the WITHIN-shot sampling noise.  10 is the powered
    #: cohort size the noise-floor characterisation justifies.
    n_random: int = 10
    #: latent divergence metric — "l2" (default) or "cosine".
    metric: str = "l2"
    #: per-shot ratio (true_vs_random / floor) a shot must clear to count as
    #: controlled.
    ratio_threshold: float = 1.5
    #: minimum true divergence for a degenerate 0.0-floor shot to pass.
    margin_threshold: float = 0.0
    #: reject collapsed random rollouts from the noise floor (the collapse test).
    reject_collapsed: bool = True
    #: bootstrap resamples for the cohort mean-ratio CI.
    n_bootstrap: int = 2000
    #: cohort-level CI percentiles (lower, upper).
    ci_pct: tuple[float, float] = (2.5, 97.5)
    #: bootstrap resamples for the WITHIN-shot ratio std (variance diagnostic).
    n_within_bootstrap: int = 500
    #: cohort pass fraction the gate requires (majority of shots driveable).
    pass_fraction_threshold: float = 0.5
    seed: int = 0


# ---------------------------------------------------------------------------
# Statistics (generalised from controllable_eval — latent-divergence agnostic)
# ---------------------------------------------------------------------------


def _within_shot_ratio_std(tvr_samples, rvr_samples, *, n_boot, seed=0):
    """Bootstrap the WITHIN-shot std of one shot's ratio (true_vs_random / floor).

    The per-shot ratio is ``mean(tvr_samples) / mean(rvr_samples)`` — both means
    over the SAME shot's random rollouts.  Resampling those with replacement
    ``n_boot`` times and recomputing the ratio gives how much the ratio wobbles
    purely from the finite ``n_random`` sampling.  Returns the std of the resampled
    ratios; ``nan`` when there are too few samples (need >=2 tvr and >=1 rvr).
    """
    tvr = np.asarray([s for s in tvr_samples if np.isfinite(s)], dtype=np.float64)
    rvr = np.asarray([s for s in rvr_samples if np.isfinite(s)], dtype=np.float64)
    if tvr.size < 2 or rvr.size < 1:
        return float("nan")
    rng = np.random.default_rng(int(seed))
    t_idx = rng.integers(0, tvr.size, size=(int(n_boot), tvr.size))
    r_idx = rng.integers(0, rvr.size, size=(int(n_boot), rvr.size))
    t_means = tvr[t_idx].mean(axis=1)
    r_means = rvr[r_idx].mean(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.where(r_means > 0.0, t_means / r_means, np.nan)
    ratios = ratios[np.isfinite(ratios)]
    if ratios.size == 0:
        return float("nan")
    return float(ratios.std())


def _bootstrap_mean_ratio_ci(ratios, *, n_boot, ci_pct, seed=0):
    """Bootstrap CI for the cohort MEAN ratio (true_vs_random / floor).

    Resamples the per-shot finite ratios with replacement ``n_boot`` times, returns
    ``(mean, lo, hi)`` of the resampled means at ``ci_pct``.  Infinite ratios (a
    0.0 floor with positive true signal) are excluded from the CI math but counted
    separately — they are unambiguous passes, not a finite statistic to bootstrap.
    """
    finite = np.asarray([r for r in ratios if np.isfinite(r)], dtype=np.float64)
    if finite.size == 0:
        return 0.0, 0.0, 0.0
    if finite.size == 1:
        m = float(finite[0])
        return m, m, m
    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, finite.size, size=(int(n_boot), finite.size))
    boot_means = finite[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, list(ci_pct))
    return float(finite.mean()), float(lo), float(hi)


def _variance_decomposition(verdicts):
    """Decompose the per-shot ratio variance into WITHIN-shot vs ACROSS-shot.

    - ``mean_within_shot_variance`` — mean over shots of each shot's within-shot
      ratio variance (the bootstrapped ``ratio_within_std`` squared); the noise the
      finite ``n_random`` sampling alone injects.  RAISING n_random shrinks it.
    - ``across_shot_variance`` — variance of the per-shot ratio point estimates
      across shots (genuine shot-to-shot heterogeneity).  RAISING n_random does NOT
      shrink this.
    - ``across_over_within`` — their ratio.  >> 1: heterogeneity dominates (more
      rollouts won't tighten the cohort CI; enrich for driveable shots); << 1:
      within-shot sampling noise dominates (raising n_random WILL tighten it).
    - ``interpretation`` — a one-line read.

    Infinite per-shot ratios are excluded from the across-shot variance; shots with
    a NaN within-shot std are dropped from the within-shot mean.
    """
    finite = [v for v in verdicts if np.isfinite(v.ratio)]
    within_vars = [
        float(v.ratio_within_std) ** 2
        for v in finite
        if np.isfinite(v.ratio_within_std)
    ]
    mean_within = float(np.mean(within_vars)) if within_vars else float("nan")
    ratios = np.asarray([v.ratio for v in finite], dtype=np.float64)
    across = float(ratios.var()) if ratios.size >= 2 else float("nan")

    aow = float("nan")
    if np.isfinite(mean_within) and np.isfinite(across) and mean_within > 0.0:
        aow = across / mean_within

    if not np.isfinite(aow):
        interp = (
            "insufficient samples for a within/across decomposition "
            "(need >=2 shots with >=2 random rollouts each)"
        )
    elif aow >= 3.0:
        interp = (
            f"ACROSS-shot heterogeneity dominates (across/within={aow:.1f}): shots "
            "differ genuinely (some driveable, most not) — raising n_random will "
            "NOT tighten the cohort CI much; the pass-FRACTION over a screened "
            "cohort is the stable signal, and resolving levers needs a "
            "driveable-shot-enriched cohort, not more rollouts."
        )
    elif aow <= 0.33:
        interp = (
            f"WITHIN-shot sampling noise dominates (across/within={aow:.2f}): the "
            "per-shot ratios are noisy from too few random rollouts — RAISING "
            "n_random will tighten the cohort CI."
        )
    else:
        interp = (
            "within- and across-shot variance are comparable "
            f"(across/within={aow:.2f}): both raising n_random AND enriching the "
            "cohort for driveable shots help."
        )
    return {
        "mean_within_shot_variance": mean_within,
        "across_shot_variance": across,
        "across_over_within": aow,
        "n_shots_with_within_std": len(within_vars),
        "interpretation": interp,
    }


# ---------------------------------------------------------------------------
# Per-shot + cohort gate
# ---------------------------------------------------------------------------


def _shot_divergences(true_traj, rand_trajs, ctx, *, metric, reject_collapsed):
    """True-vs-random + random-vs-random floor for one shot's latent rollouts.

    Collapsed random rollouts (degenerate near-constant forecast) are excluded from
    BOTH the true-vs-random samples and the floor when ``reject_collapsed`` so the
    floor reflects real plan-driven variation.  Returns
    ``(tvr, rvr, n_collapsed, n_kept, tvr_samples, rvr_samples)``.
    """
    kept_idx = list(range(len(rand_trajs)))
    if reject_collapsed and rand_trajs:
        kept_idx = [
            i for i, r in enumerate(rand_trajs) if not _is_collapsed_latent(r, ctx)
        ]
    n_collapsed = len(rand_trajs) - len(kept_idx)
    kept = [rand_trajs[i] for i in kept_idx]

    tvr_samples = [latent_divergence(true_traj, r, ctx, metric=metric) for r in kept]
    rvr_samples = [
        latent_divergence(kept[i], kept[j], ctx, metric=metric)
        for i in range(len(kept))
        for j in range(i + 1, len(kept))
    ]
    tvr = float(np.mean(tvr_samples)) if tvr_samples else 0.0
    rvr = float(np.mean(rvr_samples)) if rvr_samples else 0.0
    return tvr, rvr, n_collapsed, len(kept), tvr_samples, rvr_samples


def evaluate_shot(
    rollout_fn: Callable[[object], np.ndarray],
    true_plan: object,
    sample_random_plan: Callable[[np.random.Generator], object],
    *,
    context_frames: int,
    config: GateConfig | None = None,
    shot_id: int = 0,
    rng: np.random.Generator | None = None,
) -> LatentShotVerdict:
    """Per-shot powered ΔN-M: does the TRUE plan move the latent more than random?

    Rolls the true plan + ``config.n_random`` random plans through ``rollout_fn``,
    scores the forecast-window latent divergence true-vs-random against the
    (collapse-rejected) random-vs-random floor, and forms the per-shot ratio + its
    bootstrapped within-shot std.  A shot PASSES when the ratio clears
    ``ratio_threshold`` (or, on a 0.0 floor, the true divergence clears
    ``margin_threshold``).
    """
    cfg = config or GateConfig()
    if rng is None:
        rng = np.random.default_rng((int(shot_id) * 1_000_003) ^ (cfg.seed * 31))
    ctx = int(context_frames)

    true_traj = np.asarray(rollout_fn(true_plan), dtype=np.float64)
    rand_trajs = [
        np.asarray(rollout_fn(sample_random_plan(rng)), dtype=np.float64)
        for _ in range(int(cfg.n_random))
    ]

    tvr, rvr, n_collapsed, n_kept, tvr_samples, rvr_samples = _shot_divergences(
        true_traj,
        rand_trajs,
        ctx,
        metric=cfg.metric,
        reject_collapsed=cfg.reject_collapsed,
    )
    margin = tvr - rvr
    ratio = float("inf") if rvr == 0.0 else tvr / rvr
    ratio_within_std = _within_shot_ratio_std(
        tvr_samples,
        rvr_samples,
        n_boot=cfg.n_within_bootstrap,
        seed=(int(shot_id) * 7919) ^ (cfg.seed * 31),
    )
    # noise-floor-NORMALISED pass: a flat 0/0 shot is not a win — the 0.0-floor
    # branch only passes when the true plan actually moved the latent.
    passed = bool(
        (rvr == 0.0 and tvr > cfg.margin_threshold) or ratio > cfg.ratio_threshold
    )
    return LatentShotVerdict(
        shot_id=int(shot_id),
        true_vs_random=tvr,
        random_vs_random=rvr,
        margin=margin,
        ratio=ratio,
        n_random=int(cfg.n_random),
        n_random_collapsed=int(n_collapsed),
        n_random_kept=int(n_kept),
        true_vs_random_samples=[float(x) for x in tvr_samples],
        random_vs_random_samples=[float(x) for x in rvr_samples],
        ratio_within_std=float(ratio_within_std),
        passed=passed,
    )


def _summarise(verdicts: Sequence[LatentShotVerdict], cfg: GateConfig):
    n_pass = sum(1 for v in verdicts if v.passed)
    pass_fraction = float(n_pass / len(verdicts)) if verdicts else 0.0
    mean_margin = float(np.mean([v.margin for v in verdicts])) if verdicts else 0.0
    mean_tvr = float(np.mean([v.true_vs_random for v in verdicts])) if verdicts else 0.0
    mean_rvr = (
        float(np.mean([v.random_vs_random for v in verdicts])) if verdicts else 0.0
    )
    ratios = [v.ratio for v in verdicts]
    n_inf = sum(1 for r in ratios if not np.isfinite(r))
    mean_ratio, ci_lo, ci_hi = _bootstrap_mean_ratio_ci(
        ratios, n_boot=cfg.n_bootstrap, ci_pct=cfg.ci_pct, seed=cfg.seed
    )
    var_decomp = _variance_decomposition(verdicts)
    sorted_ratios = sorted(float(v.ratio) for v in verdicts if np.isfinite(v.ratio))

    # ROBUST gate: a majority of shots clear the per-shot ratio AND the cohort
    # mean-ratio bootstrap CI lower bound is clear of the noise floor (1.0), so
    # the controllability win is not a 1-shot artifact.
    gate_pass = bool(
        len(verdicts) > 0
        and pass_fraction >= cfg.pass_fraction_threshold
        and ci_lo > 1.0
    )
    return LatentDeltaNMVerdict(
        n_shots=len(verdicts),
        n_pass=n_pass,
        pass_fraction=pass_fraction,
        mean_true_vs_random=mean_tvr,
        mean_random_vs_random_noise_floor=mean_rvr,
        mean_margin=mean_margin,
        mean_ratio=mean_ratio,
        ratio_ci_lo=ci_lo,
        ratio_ci_hi=ci_hi,
        n_random=int(cfg.n_random),
        n_ratio_infinite=n_inf,
        gate_pass=gate_pass,
        verdict="PASS" if gate_pass else "FAIL",
        variance_decomposition=var_decomp,
        per_shot_ratios_sorted=sorted_ratios,
        per_shot=list(verdicts),
        metric=f"latent_delta_nm_{cfg.metric}",
    )


def controllability_gate(
    rollout_fn: Callable[[object], np.ndarray],
    shots: Sequence[tuple[int, object, int]],
    sample_random_plan: Callable[[np.random.Generator], object],
    *,
    config: GateConfig | None = None,
) -> LatentDeltaNMVerdict:
    """Powered, model-independent latent controllability gate over a cohort.

    ``shots`` is a sequence of ``(shot_id, true_plan, context_frames)`` — one entry
    per held-out shot.  ``rollout_fn(plan) -> (T, D)`` produces a closed-loop latent
    rollout under a plan; ``sample_random_plan(rng) -> plan`` draws an
    in-distribution counterfactual.  For each shot the gate runs
    :func:`evaluate_shot`, then aggregates the cohort with a bootstrap-CI'd mean
    ratio and the variance decomposition.

    Returns the :class:`LatentDeltaNMVerdict`.  The gate FIRES (``gate_pass``) only
    when a majority of shots clear the per-shot ratio threshold AND the cohort
    mean-ratio CI lower bound clears the noise floor (1.0).
    """
    cfg = config or GateConfig()
    verdicts: list[LatentShotVerdict] = []
    for sid, true_plan, ctx in shots:
        rng = np.random.default_rng((int(sid) * 1_000_003) ^ (cfg.seed * 31))
        verdicts.append(
            evaluate_shot(
                rollout_fn,
                true_plan,
                sample_random_plan,
                context_frames=int(ctx),
                config=cfg,
                shot_id=int(sid),
                rng=rng,
            )
        )
    return _summarise(verdicts, cfg)
