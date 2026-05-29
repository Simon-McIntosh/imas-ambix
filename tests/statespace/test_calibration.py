"""Unit tests for imas_ambix.statespace.calibration.

All tests use synthetic data — no network or GPFS access.

CRPS closed form is validated against Monte-Carlo estimates:
    CRPS_MC(N(μ,σ), y) ≈ CRPS_analytical(N(μ,σ), y)
for large M (converges as O(1/√M)).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from imas_ambix.statespace.calibration import (
    compute_calibration_report,
    coverage_by_level,
    coverage_vs_distance,
    crps_ensemble,
    crps_gaussian,
    ece,
    ensemble_disagreement,
    interval_coverage,
    nll_gaussian,
    ood_auroc,
    ood_novelty_score,
    prediction_interval_width,
    reliability_diagram_data,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)


def _make_gaussian_predictions(
    n: int = 500,
    sigma_true: float = 1.0,
    sigma_pred: float = 1.0,
    bias: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (y_true, y_mean, y_std) with known calibration properties.

    y_mean is drawn from N(0, sigma_true), y_true ~ N(y_mean, sigma_true),
    so the model is calibrated when sigma_pred == sigma_true and bias == 0.
    """
    rng = np.random.default_rng(0)
    y_mean = rng.normal(0.0, sigma_true, size=n)
    # y_true is drawn from the true distribution centred on y_mean
    y_true = rng.normal(y_mean + bias, sigma_true)
    y_std = np.full(n, sigma_pred)
    return y_true, y_mean, y_std


# ---------------------------------------------------------------------------
# 1. Coverage
# ---------------------------------------------------------------------------


class TestIntervalCoverage:
    def test_perfect_calibration(self) -> None:
        """A perfectly calibrated model should hit ~90% at alpha=0.10."""
        y_true, y_mean, y_std = _make_gaussian_predictions(n=10_000)
        cov = interval_coverage(y_true, y_mean, y_std, alpha=0.10)
        assert abs(cov - 0.90) < 0.03, f"Expected ~0.90, got {cov:.4f}"

    def test_overconfident_model_undercoverage(self) -> None:
        """An overconfident model (std too small) undercoverts."""
        y_true, y_mean, _ = _make_gaussian_predictions(n=5000)
        y_std_small = np.full_like(y_true, 0.1)  # much too small
        cov = interval_coverage(y_true, y_mean, y_std_small, alpha=0.10)
        assert cov < 0.50, f"Expected under-coverage, got {cov:.4f}"

    def test_underconfident_model_overcoverage(self) -> None:
        """An underconfident model (std too large) overcovers."""
        y_true, y_mean, _ = _make_gaussian_predictions(n=5000)
        y_std_large = np.full_like(y_true, 10.0)  # much too large
        cov = interval_coverage(y_true, y_mean, y_std_large, alpha=0.10)
        assert cov > 0.99, f"Expected over-coverage, got {cov:.4f}"

    def test_coverage_by_level_ordering(self) -> None:
        """Larger alpha (tighter interval) should give lower coverage."""
        y_true, y_mean, y_std = _make_gaussian_predictions(n=5000)
        levels = coverage_by_level(y_true, y_mean, y_std)
        # Check monotonicity: smaller alpha → larger 1-alpha → more coverage
        sorted_alphas = sorted(levels.keys())
        covs = [levels[a] for a in sorted_alphas]
        for i in range(len(covs) - 1):
            # larger alpha → smaller interval → smaller coverage
            assert covs[i] >= covs[i + 1] - 0.05, (
                f"Coverage not monotone at alpha={sorted_alphas[i]}: "
                f"{covs[i]:.3f} vs {covs[i + 1]:.3f}"
            )


# ---------------------------------------------------------------------------
# 2. CRPS — Gaussian closed form vs Monte-Carlo validation
# ---------------------------------------------------------------------------


class TestCRPSGaussian:
    """Validate analytic CRPS against Monte-Carlo energy estimator."""

    def _crps_mc(
        self,
        y_true: np.ndarray,
        y_mean: np.ndarray,
        y_std: np.ndarray,
        n_samples: int = 4000,
    ) -> float:
        """Monte-Carlo CRPS: draw M samples from N(μ,σ) and use energy form."""
        rng = np.random.default_rng(1)
        N = len(y_true)  # noqa: N806
        M = n_samples  # noqa: N806
        samples = rng.normal(
            y_mean[:, np.newaxis],
            y_std[:, np.newaxis],
            size=(N, M),
        )
        return crps_ensemble(y_true, samples)

    def test_crps_gaussian_vs_mc(self) -> None:
        """Analytic CRPS must match MC within 1% for N(0,1) predictive."""
        rng = np.random.default_rng(10)
        N = 500
        y_true = rng.normal(0, 1, N)
        y_mean = rng.normal(0, 0.5, N)
        y_std = np.abs(rng.normal(1, 0.3, N)) + 0.1

        analytical = crps_gaussian(y_true, y_mean, y_std)
        mc = self._crps_mc(y_true, y_mean, y_std, n_samples=5000)

        rel_err = abs(analytical - mc) / max(abs(mc), 1e-12)
        assert rel_err < 0.02, (
            f"CRPS analytical={analytical:.6f} vs MC={mc:.6f} "
            f"(rel err {100 * rel_err:.2f}%)"
        )

    def test_crps_gaussian_perfect_prediction(self) -> None:
        """CRPS is non-negative and lower when the model is perfect."""
        rng = np.random.default_rng(20)
        N = 1000
        y_true = rng.normal(0, 1, N)

        # Perfect mean, correct std
        c_good = crps_gaussian(y_true, y_true.copy(), np.ones(N))

        # Biased predictions
        c_biased = crps_gaussian(y_true, y_true + 2.0, np.ones(N))

        assert c_good >= 0.0, f"CRPS must be non-negative, got {c_good}"
        assert c_good < c_biased, (
            f"Biased ({c_biased:.4f}) must have worse CRPS than unbiased ({c_good:.4f})"
        )

    def test_crps_gaussian_reduces_to_mae_for_certain_std(self) -> None:
        """For deterministic prediction (σ→0), CRPS → MAE."""
        rng = np.random.default_rng(30)
        N = 500
        y_true = rng.normal(0, 1, N)
        y_pred = rng.normal(0, 0.5, N)

        mae = float(np.mean(np.abs(y_true - y_pred)))
        crps_det = crps_gaussian(y_true, y_pred, np.full(N, 1e-6))

        assert abs(crps_det - mae) < 1e-3, (
            f"CRPS with σ≈0 ({crps_det:.6f}) should equal MAE ({mae:.6f})"
        )


# ---------------------------------------------------------------------------
# 3. CRPS ensemble form
# ---------------------------------------------------------------------------


class TestCRPSEnsemble:
    def test_ensemble_crps_1d(self) -> None:
        """Ensemble CRPS should be non-negative and lower for better ensemble."""
        rng = np.random.default_rng(40)
        N, M = 200, 50
        y_true = rng.normal(0, 1, N)

        # Good ensemble centred on truth
        ens_good = rng.normal(y_true[:, np.newaxis], np.ones((N, M)))
        crps_good = crps_ensemble(y_true, ens_good)

        # Bad ensemble far from truth
        ens_bad = rng.normal((y_true + 5.0)[:, np.newaxis], np.ones((N, M)))
        crps_bad = crps_ensemble(y_true, ens_bad)

        assert crps_good >= 0.0
        assert crps_good < crps_bad, (
            f"Good ens CRPS ({crps_good:.4f}) should < bad ens ({crps_bad:.4f})"
        )

    def test_ensemble_crps_multi_dim(self) -> None:
        """Ensemble CRPS for (N, M, D) input should be a scalar."""
        rng = np.random.default_rng(50)
        N, M, D = 100, 20, 3
        y_true = rng.normal(0, 1, (N, D))
        ens = rng.normal(0, 1, (N, M, D))
        result = crps_ensemble(y_true, ens)
        assert isinstance(result, float)
        assert result >= 0.0

    def test_ensemble_gaussian_crps_matches_analytical(self) -> None:
        """Ensemble CRPS (large M) should match analytical Gaussian CRPS."""
        rng = np.random.default_rng(60)
        N = 300
        mu = rng.normal(0, 1, N)
        sigma = np.abs(rng.normal(1, 0.2, N)) + 0.1
        y_true = rng.normal(mu, sigma)

        crps_anal = crps_gaussian(y_true, mu, sigma)
        # Draw large M to get accurate estimate
        M = 10_000
        ens = rng.normal(mu[:, None], sigma[:, None], (N, M))
        crps_ens = crps_ensemble(y_true, ens)

        rel_err = abs(crps_anal - crps_ens) / max(abs(crps_anal), 1e-12)
        assert rel_err < 0.02, (
            f"Analytical CRPS ({crps_anal:.5f}) vs ensemble ({crps_ens:.5f}), "
            f"rel error {100 * rel_err:.2f}%"
        )


# ---------------------------------------------------------------------------
# 4. NLL
# ---------------------------------------------------------------------------


class TestNLL:
    def test_nll_lower_for_correct_sigma(self) -> None:
        """NLL is minimised when σ matches the true noise level."""
        rng = np.random.default_rng(70)
        N = 1000
        y_true = rng.normal(0, 1, N)
        y_mean = np.zeros(N)

        # NLL at correct σ=1
        nll_correct = nll_gaussian(y_true, y_mean, np.ones(N))
        # NLL at σ=0.5 (overconfident)
        nll_overconfident = nll_gaussian(y_true, y_mean, np.full(N, 0.5))
        # NLL at σ=5 (underconfident)
        nll_underconfident = nll_gaussian(y_true, y_mean, np.full(N, 5.0))

        assert nll_correct < nll_overconfident, "Correct σ should beat overconfident"
        assert nll_correct < nll_underconfident, "Correct σ should beat underconfident"

    def test_nll_positive(self) -> None:
        """NLL for reasonable predictions should be positive."""
        y_true = np.array([0.0, 1.0, -1.0])
        y_mean = np.array([0.1, 0.9, -1.1])
        y_std = np.ones(3)
        assert nll_gaussian(y_true, y_mean, y_std) > 0.0


# ---------------------------------------------------------------------------
# 5. ECE and reliability diagram
# ---------------------------------------------------------------------------


class TestECE:
    def test_perfect_calibration_low_ece(self) -> None:
        """Well-calibrated model should have ECE < 0.05."""
        y_true, y_mean, y_std = _make_gaussian_predictions(n=10_000)
        e = ece(y_true, y_mean, y_std, n_bins=10)
        assert e < 0.05, f"Expected low ECE for calibrated model, got {e:.4f}"

    def test_biased_model_high_ece(self) -> None:
        """Biased model (σ×10 underconfident) should have high ECE."""
        y_true, y_mean, y_std = _make_gaussian_predictions(n=5000)
        y_std_bad = y_std * 10.0
        e = ece(y_true, y_mean, y_std_bad, n_bins=10)
        # Coverage will be ~100% at most levels → ECE high at low levels
        assert e > 0.10, f"Expected high ECE for underconfident model, got {e:.4f}"

    def test_reliability_diagram_returns_correct_structure(self) -> None:
        """Reliability diagram must return nominal, empirical, width lists."""
        y_true, y_mean, y_std = _make_gaussian_predictions(n=500)
        diag = reliability_diagram_data(y_true, y_mean, y_std, n_bins=5)
        assert "nominal" in diag
        assert "empirical" in diag
        assert "width" in diag
        assert len(diag["nominal"]) == 5
        assert all(0 <= v <= 1 for v in diag["empirical"])

    def test_reliability_diagram_near_diagonal_for_calibrated(self) -> None:
        """Calibrated model should have empirical ≈ nominal at most levels."""
        y_true, y_mean, y_std = _make_gaussian_predictions(n=20_000)
        diag = reliability_diagram_data(y_true, y_mean, y_std, n_bins=8)
        gaps = [
            abs(e - n)
            for e, n in zip(diag["empirical"], diag["nominal"], strict=True)
        ]
        mean_gap = sum(gaps) / len(gaps)
        assert mean_gap < 0.05, f"Mean calibration gap {mean_gap:.4f} > 0.05"


# ---------------------------------------------------------------------------
# 6. Ensemble epistemic and OOD scores
# ---------------------------------------------------------------------------


class TestEnsembleScores:
    def test_disagreement_shape(self) -> None:
        """ensemble_disagreement should return shape (N,)."""
        rng = np.random.default_rng(80)
        N, M, D = 50, 10, 4
        ens = rng.normal(0, 1, (N, M, D))
        score = ensemble_disagreement(ens)
        assert score.shape == (N,), f"Expected ({N},), got {score.shape}"

    def test_disagreement_higher_for_diverse_ensemble(self) -> None:
        """A more diverse ensemble should have higher disagreement."""
        rng = np.random.default_rng(90)
        N, M = 100, 20
        ens_tight = rng.normal(0, 0.01, (N, M))
        ens_wide = rng.normal(0, 10.0, (N, M))
        score_tight = np.mean(ensemble_disagreement(ens_tight))
        score_wide = np.mean(ensemble_disagreement(ens_wide))
        assert score_wide > score_tight * 100, (
            f"Wide ens score ({score_wide:.4f}) should >> tight ({score_tight:.4f})"
        )

    def test_ood_novelty_score_same_as_disagreement(self) -> None:
        """ood_novelty_score is the same as ensemble_disagreement."""
        rng = np.random.default_rng(100)
        ens = rng.normal(0, 1, (50, 10, 3))
        np.testing.assert_array_equal(
            ood_novelty_score(ens),
            ensemble_disagreement(ens),
        )

    def test_ood_auroc_perfect_separator(self) -> None:
        """Perfect separator should achieve AUROC=1.0."""
        in_scores = np.zeros(100)
        out_scores = np.ones(100)
        assert ood_auroc(in_scores, out_scores) == pytest.approx(1.0)

    def test_ood_auroc_random(self) -> None:
        """Random (shuffled) labels should give AUROC ≈ 0.5."""
        rng = np.random.default_rng(110)
        scores = rng.uniform(0, 1, 200)
        auroc = ood_auroc(scores[:100], scores[100:])
        assert abs(auroc - 0.5) < 0.15, f"Expected ≈0.5 for random, got {auroc:.4f}"


# ---------------------------------------------------------------------------
# 7. Coverage-vs-distance
# ---------------------------------------------------------------------------


class TestCoverageVsDistance:
    def test_coverage_decreases_with_distance(self) -> None:
        """Model should have lower coverage for high-distance (OOD) samples."""
        rng = np.random.default_rng(120)
        N = 1000
        distances = np.linspace(0, 5, N)  # 0 = in-dist, 5 = OOD
        # Bias increases with distance (simulating distribution shift)
        bias = distances * 0.5
        y_true = rng.normal(0, 1, N)
        y_mean = y_true + bias  # biased predictions for high-distance samples
        y_std = np.ones(N)

        result = coverage_vs_distance(
            y_true, y_mean, y_std, distances, n_bins=5, alpha=0.10
        )
        assert "bin_centers" in result
        assert len(result["coverage"]) == 5
        # Coverage should generally decrease with distance
        covs = result["coverage"]
        assert covs[0] > covs[-1], (
            f"Coverage at dist~0 ({covs[0]:.3f}) should > dist~5 ({covs[-1]:.3f})"
        )


# ---------------------------------------------------------------------------
# 8. Unified report
# ---------------------------------------------------------------------------


class TestCalibrationReport:
    def test_report_all_fields_populated(self) -> None:
        """compute_calibration_report should fill all scalar fields."""
        y_true, y_mean, y_std = _make_gaussian_predictions(n=500)
        report = compute_calibration_report(y_true, y_mean, y_std)
        assert not math.isnan(report.crps)
        assert not math.isnan(report.nll)
        assert not math.isnan(report.ece)
        assert not math.isnan(report.coverage_90)
        assert not math.isnan(report.pi_width_90)
        assert len(report.coverage_by_level) > 0
        assert len(report.reliability_diagram) > 0

    def test_report_with_ensemble(self) -> None:
        """With ensemble, ensemble_disagreement_mean should be populated."""
        rng = np.random.default_rng(130)
        N, M = 100, 10
        y_true = rng.normal(0, 1, N)
        y_mean = rng.normal(0, 1, N)
        y_std = np.abs(rng.normal(1, 0.2, N)) + 0.1
        ens = rng.normal(y_mean[:, None], y_std[:, None], (N, M))

        report = compute_calibration_report(y_true, y_mean, y_std, ensemble=ens)
        assert report.ensemble_disagreement_mean is not None
        assert report.ensemble_disagreement_mean >= 0.0

    def test_report_with_ood_auroc(self) -> None:
        """With ood scores, ood_auroc should be populated."""
        rng = np.random.default_rng(140)
        y_true, y_mean, y_std = _make_gaussian_predictions(n=200)
        in_dist = rng.uniform(0, 0.3, 100)
        ood = rng.uniform(0.7, 1.0, 50)

        report = compute_calibration_report(
            y_true,
            y_mean,
            y_std,
            ood_in_dist_scores=in_dist,
            ood_ood_scores=ood,
        )
        assert report.ood_auroc is not None
        assert report.ood_auroc > 0.5, (
            f"With separable scores, AUROC should > 0.5, got {report.ood_auroc}"
        )

    def test_report_to_dict_json_serialisable(self) -> None:
        """Report should serialise to a JSON-compatible dict."""
        import json

        y_true, y_mean, y_std = _make_gaussian_predictions(n=100)
        report = compute_calibration_report(y_true, y_mean, y_std)
        d = report.to_dict()
        # Should not raise
        json.dumps(d)

    def test_report_summary_string(self) -> None:
        """Report summary string should not raise and contain key fields."""
        y_true, y_mean, y_std = _make_gaussian_predictions(n=200)
        report = compute_calibration_report(y_true, y_mean, y_std)
        s = report.summary()
        assert "CRPS" in s
        assert "ECE" in s
        assert "Coverage" in s


# ---------------------------------------------------------------------------
# 9. PI width
# ---------------------------------------------------------------------------


class TestPIWidth:
    def test_width_proportional_to_sigma(self) -> None:
        """PI width should be proportional to y_std."""
        y_std_1 = np.ones(100)
        y_std_2 = np.ones(100) * 2.0
        w1 = prediction_interval_width(y_std_1, alpha=0.10)
        w2 = prediction_interval_width(y_std_2, alpha=0.10)
        assert abs(w2 / w1 - 2.0) < 0.01, f"Expected ratio 2.0, got {w2 / w1:.4f}"

    def test_width_decreases_with_alpha(self) -> None:
        """Larger alpha (tighter interval) → smaller width."""
        y_std = np.ones(100)
        w_wide = prediction_interval_width(y_std, alpha=0.01)  # 99% interval
        w_tight = prediction_interval_width(y_std, alpha=0.50)  # 50% interval
        assert w_wide > w_tight, f"Wide ({w_wide:.4f}) should > tight ({w_tight:.4f})"
