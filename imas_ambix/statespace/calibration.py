"""Probabilistic calibration harness for plasma-state-space-v0.

Implements the metrics defined in docs/probabilistic-state-space-methods.html §4:

- Interval coverage @ nominal alpha levels (split-conformal)
- Sharpness (prediction-interval width)
- CRPS (Continuous Ranked Probability Score)
  - Gaussian predictive: analytic closed form
  - Ensemble predictive: energy-form estimator
- NLL (Negative Log-Likelihood)
- Reliability diagram + Expected Calibration Error (ECE)
- Ensemble-disagreement epistemic score
- OOD novelty score (ensemble-variance based)
- OOD-detection AUROC / coverage-vs-distance

IMPORTANT: CRPS is implemented directly using numpy/scipy.
Do NOT add properscoring to the lockfile (it's absent, and lockfile changes
could propagate to an in-flight SLURM job).

All functions accept numpy arrays and return plain floats / dicts so they
can be unit-tested without any ML infrastructure.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases (numpy arrays throughout)
# ---------------------------------------------------------------------------
# y_true:  (N,) or (N, D) float  -- ground truth
# y_mean:  (N,) or (N, D) float  -- predictive mean
# y_std:   (N,) or (N, D) float  -- predictive std (> 0)
# ensemble: (N, M, D) float      -- M ensemble members, D output dims


# ---------------------------------------------------------------------------
# 1. Interval coverage and sharpness
# ---------------------------------------------------------------------------


def interval_coverage(
    y_true: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    alpha: float = 0.10,
) -> float:
    """Empirical coverage of the (1-alpha) Gaussian predictive interval.

    The interval is [y_mean - z * y_std, y_mean + z * y_std] where
    z = scipy.stats.norm.ppf(1 - alpha/2).

    Parameters
    ----------
    y_true, y_mean, y_std:
        Arrays of equal shape (N,) or (N, D).
    alpha:
        Miscoverage level (default 0.10 → 90 % interval).

    Returns
    -------
    float in [0, 1] — fraction of samples falling inside the interval.
    """
    from scipy.stats import norm  # noqa: PLC0415

    z = float(norm.ppf(1.0 - alpha / 2.0))
    lower = y_mean - z * y_std
    upper = y_mean + z * y_std
    inside = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(inside))


def prediction_interval_width(
    y_std: np.ndarray,
    alpha: float = 0.10,
) -> float:
    """Mean predictive-interval width (sharpness proxy).

    Width = 2 * z * y_std, averaged over all samples.
    Smaller width = sharper (more useful) predictions.
    """
    from scipy.stats import norm  # noqa: PLC0415

    z = float(norm.ppf(1.0 - alpha / 2.0))
    return float(np.mean(2.0 * z * np.abs(y_std)))


def coverage_by_level(
    y_true: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    alphas: list[float] | None = None,
) -> dict[float, float]:
    """Empirical coverage at multiple nominal levels.

    Parameters
    ----------
    alphas:
        List of miscoverage levels.  Defaults to
        ``[0.50, 0.20, 0.10, 0.05, 0.02, 0.01]``.

    Returns
    -------
    dict mapping alpha → empirical coverage.
    """
    if alphas is None:
        alphas = [0.50, 0.20, 0.10, 0.05, 0.02, 0.01]
    return {a: interval_coverage(y_true, y_mean, y_std, alpha=a) for a in alphas}


# ---------------------------------------------------------------------------
# 2. CRPS — Continuous Ranked Probability Score
# ---------------------------------------------------------------------------


def crps_gaussian(
    y_true: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
) -> float:
    """CRPS for a Gaussian predictive distribution (analytic closed form).

    CRPS(N(μ, σ), y) = σ [ (y-μ)/σ (2Φ((y-μ)/σ) - 1)
                          + 2φ((y-μ)/σ) - 1/√π ]

    where Φ is the standard-normal CDF and φ is the PDF.

    This is the standard Gneiting & Raftery (2007) closed form.
    Units are the same as y_true.

    Returns mean CRPS over all samples.
    """
    from scipy.stats import norm  # noqa: PLC0415

    sigma = np.abs(y_std)
    z = (y_true - y_mean) / np.where(sigma > 0, sigma, 1e-12)
    phi = norm.pdf(z)
    Phi = norm.cdf(z)  # noqa: N806 — conventional CDF notation
    crps_per_sample = sigma * (z * (2.0 * Phi - 1.0) + 2.0 * phi - 1.0 / np.sqrt(np.pi))
    return float(np.mean(crps_per_sample))


def crps_ensemble(
    y_true: np.ndarray,
    ensemble: np.ndarray,
) -> float:
    """CRPS via the energy form for ensemble predictive distributions.

    CRPS(F, y) = E|X - y| - 0.5 * E|X - X'|

    where X, X' are independent draws from the ensemble F.

    Parameters
    ----------
    y_true:
        Shape (N,) or (N, D) — ground truth values.
    ensemble:
        Shape (N, M) or (N, M, D) — M ensemble members.

    Returns
    -------
    float — mean CRPS over all N samples (and D dimensions if present).

    Notes
    -----
    The naive double-loop E|X - X'| is O(M²) per sample.  We use the
    equivalent O(M log M) form for 1-D:
        E|X - X'| = (2/M²) sum_{i<j} |x_i - x_j|
    For M ≤ 100 (typical ensemble size) the direct form is also fast.
    """
    y = np.asarray(y_true)
    ens = np.asarray(ensemble)

    if y.ndim == 1:
        y = y[:, np.newaxis]
    if ens.ndim == 2:
        ens = ens[:, :, np.newaxis]

    N, M, D = ens.shape  # noqa: N806 — conventional matrix dimension names
    assert y.shape == (N, D), f"Shape mismatch: y {y.shape} vs ens ({N},{M},{D})"

    # E|X - y| averaged over ensemble members
    e_abs = np.mean(np.abs(ens - y[:, np.newaxis, :]), axis=1)  # (N, D)

    # E|X - X'| via pair differences
    # For M <= 200, direct O(M^2): sum over all unordered pairs
    if M <= 200:
        pair_sum = np.zeros((N, D), dtype=np.float64)
        for i in range(M):
            for j in range(i + 1, M):
                pair_sum += np.abs(ens[:, i, :] - ens[:, j, :])
        e_pair = pair_sum * (2.0 / (M * M))  # 2 * n_pairs / M^2
    else:
        # Sort along M axis and use the sorted-form identity
        ens_sorted = np.sort(ens, axis=1)
        ranks = np.arange(M, dtype=np.float64)
        # E|X - X'| = (2/M^2) * sum_i (2i - M + 1) * x_{(i)}
        weights = (2.0 * ranks - M + 1.0)[np.newaxis, :, np.newaxis]  # (1, M, 1)
        e_pair = np.sum(weights * ens_sorted, axis=1) * (2.0 / (M * M))

    crps_per_sample = e_abs - 0.5 * e_pair  # (N, D)
    return float(np.mean(crps_per_sample))


# ---------------------------------------------------------------------------
# 3. NLL — Negative Log-Likelihood
# ---------------------------------------------------------------------------


def nll_gaussian(
    y_true: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
) -> float:
    """Mean NLL under a Gaussian predictive distribution.

    NLL = 0.5 * [log(2π σ²) + (y - μ)² / σ²]
    """
    sigma = np.abs(y_std)
    sigma = np.where(sigma > 0, sigma, 1e-12)
    nll = 0.5 * (np.log(2.0 * np.pi * sigma**2) + ((y_true - y_mean) / sigma) ** 2)
    return float(np.mean(nll))


# ---------------------------------------------------------------------------
# 4. Reliability diagram + ECE
# ---------------------------------------------------------------------------


def reliability_diagram_data(
    y_true: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    n_bins: int = 10,
    alphas: list[float] | None = None,
) -> dict[str, list[float]]:
    """Compute data for a reliability (calibration) diagram.

    For each nominal coverage level (1-alpha), computes the empirical
    coverage and average interval width.

    Parameters
    ----------
    n_bins:
        Number of equally-spaced nominal coverage levels from 0 to 1.
    alphas:
        Override the auto-generated alpha grid.

    Returns
    -------
    dict with keys:
        ``nominal`` : list of nominal coverage levels (1 - alpha)
        ``empirical`` : list of empirical coverage values
        ``width`` : list of mean PI widths at each level
    """
    if alphas is None:
        alphas = list(np.linspace(0.02, 0.98, n_bins))

    nominal = []
    empirical = []
    widths = []
    for alpha in alphas:
        cov = interval_coverage(y_true, y_mean, y_std, alpha=alpha)
        w = prediction_interval_width(y_std, alpha=alpha)
        nominal.append(1.0 - alpha)
        empirical.append(cov)
        widths.append(w)

    return {"nominal": nominal, "empirical": empirical, "width": widths}


def ece(
    y_true: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error (mean |empirical - nominal| coverage gap).

    Lower ECE = better calibration.
    """
    diag = reliability_diagram_data(y_true, y_mean, y_std, n_bins=n_bins)
    gaps = [abs(e - n) for e, n in zip(diag["empirical"], diag["nominal"], strict=True)]
    return float(np.mean(gaps))


# ---------------------------------------------------------------------------
# 5. Ensemble epistemic and OOD scores
# ---------------------------------------------------------------------------


def ensemble_disagreement(
    ensemble: np.ndarray,
) -> np.ndarray:
    """Per-sample epistemic uncertainty from ensemble variance.

    Parameters
    ----------
    ensemble:
        Shape (N, M) or (N, M, D).

    Returns
    -------
    (N,) array — mean variance over dimensions, one value per sample.
    Larger value = more epistemic uncertainty.
    """
    ens = np.asarray(ensemble)
    if ens.ndim == 2:
        ens = ens[:, :, np.newaxis]
    var = np.var(ens, axis=1)  # (N, D)
    return np.mean(var, axis=-1)  # (N,)


def ood_novelty_score(
    ensemble: np.ndarray,
) -> np.ndarray:
    """OOD novelty score based on ensemble disagreement.

    Returns the same as :func:`ensemble_disagreement` — a per-sample
    scalar.  Shots with high novelty scores are OOD candidates.

    See docs/probabilistic-state-space-methods.html §4 for the formal
    definition; the ensemble-variance threshold is calibrated on the
    in-distribution calibration set.
    """
    return ensemble_disagreement(ensemble)


def ood_auroc(
    in_dist_scores: np.ndarray,
    ood_scores: np.ndarray,
) -> float:
    """AUROC for OOD detection using novelty scores.

    Treats in-distribution shots as negative class and OOD shots as
    positive class.  A perfect OOD detector has AUROC = 1.0.

    Uses ``sklearn.metrics.roc_auc_score`` (sklearn is present in the env).

    Parameters
    ----------
    in_dist_scores:
        Novelty scores for in-distribution samples (shape (N_in,)).
    ood_scores:
        Novelty scores for OOD samples (shape (N_ood,)).

    Returns
    -------
    float AUROC in [0, 1].
    """
    from sklearn.metrics import roc_auc_score  # noqa: PLC0415

    labels = np.concatenate(
        [
            np.zeros(len(in_dist_scores), dtype=int),
            np.ones(len(ood_scores), dtype=int),
        ]
    )
    scores = np.concatenate([in_dist_scores, ood_scores])
    return float(roc_auc_score(labels, scores))


# ---------------------------------------------------------------------------
# 6. Coverage-vs-distance (distributional shift)
# ---------------------------------------------------------------------------


def coverage_vs_distance(
    y_true: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    distances: np.ndarray,
    n_bins: int = 5,
    alpha: float = 0.10,
) -> dict[str, list[float]]:
    """Bin samples by distance-to-training-set and compute coverage per bin.

    Parameters
    ----------
    distances:
        Per-sample distance from the training distribution (shape (N,)).
        Larger = more OOD.
    n_bins:
        Number of distance quantile bins.
    alpha:
        Miscoverage level (default 0.10 → 90% interval).

    Returns
    -------
    dict with keys ``bin_centers``, ``coverage``, ``n_samples``.
    """
    d = np.asarray(distances)
    bins = np.quantile(d, np.linspace(0, 1, n_bins + 1))
    bin_centers = []
    coverages = []
    counts = []
    for lo, hi in zip(bins[:-1], bins[1:], strict=True):
        mask = (d >= lo) & (d <= hi)
        if mask.sum() == 0:
            continue
        cov = interval_coverage(y_true[mask], y_mean[mask], y_std[mask], alpha=alpha)
        bin_centers.append(float(0.5 * (lo + hi)))
        coverages.append(cov)
        counts.append(int(mask.sum()))
    return {
        "bin_centers": bin_centers,
        "coverage": coverages,
        "n_samples": counts,
    }


# ---------------------------------------------------------------------------
# 7. Unified calibration report
# ---------------------------------------------------------------------------


@dataclass
class CalibrationReport:
    """Aggregate calibration metrics for one model prediction.

    All float fields are scalars; all list/dict fields are JSON-serialisable.

    Attributes
    ----------
    crps:
        CRPS (lower = better).
    nll:
        Negative log-likelihood (lower = better).
    ece:
        Expected calibration error (lower = better).
    coverage_90:
        Empirical coverage at 90% nominal level.
    pi_width_90:
        Mean PI width at 90% nominal level.
    coverage_by_level:
        {alpha: empirical_coverage} for multiple levels.
    reliability_diagram:
        Reliability diagram data (from :func:`reliability_diagram_data`).
    ensemble_disagreement_mean:
        Mean per-sample epistemic score (None if ensemble unavailable).
    ood_auroc:
        OOD-detection AUROC (None if OOD scores not provided).
    notes:
        Human-readable notes.
    """

    crps: float = float("nan")
    nll: float = float("nan")
    ece: float = float("nan")
    coverage_90: float = float("nan")
    pi_width_90: float = float("nan")
    coverage_by_level: dict[float, float] = field(default_factory=dict)
    reliability_diagram: dict[str, list[float]] = field(default_factory=dict)
    ensemble_disagreement_mean: float | None = None
    ood_auroc: float | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "crps": self.crps,
            "nll": self.nll,
            "ece": self.ece,
            "coverage_90": self.coverage_90,
            "pi_width_90": self.pi_width_90,
            "coverage_by_level": {str(k): v for k, v in self.coverage_by_level.items()},
            "reliability_diagram": self.reliability_diagram,
            "ensemble_disagreement_mean": self.ensemble_disagreement_mean,
            "ood_auroc": self.ood_auroc,
            "notes": self.notes,
        }

    def summary(self) -> str:
        lines = [
            "Calibration Report",
            "=" * 35,
            f"  CRPS          : {self.crps:.4f}",
            f"  NLL           : {self.nll:.4f}",
            f"  ECE           : {self.ece:.4f}",
            f"  Coverage @90% : {self.coverage_90:.3f}  (nominal 0.900)",
            f"  PI width @90% : {self.pi_width_90:.4f}",
        ]
        if self.ensemble_disagreement_mean is not None:
            lines.append(f"  Ens disagreement: {self.ensemble_disagreement_mean:.4f}")
        if self.ood_auroc is not None:
            lines.append(f"  OOD AUROC       : {self.ood_auroc:.4f}")
        return "\n".join(lines)


def compute_calibration_report(
    y_true: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    ensemble: np.ndarray | None = None,
    ood_in_dist_scores: np.ndarray | None = None,
    ood_ood_scores: np.ndarray | None = None,
    n_bins: int = 10,
) -> CalibrationReport:
    """Compute a full :class:`CalibrationReport` from predictive statistics.

    Parameters
    ----------
    y_true, y_mean, y_std:
        Predictive statistics.  All shape (N,) or (N, D).
    ensemble:
        Optional (N, M) or (N, M, D) ensemble array for epistemic scoring.
    ood_in_dist_scores, ood_ood_scores:
        Optional novelty scores for AUROC computation.
    n_bins:
        Number of reliability diagram bins.

    Returns
    -------
    CalibrationReport
    """
    report = CalibrationReport()
    report.crps = crps_gaussian(y_true, y_mean, y_std)
    report.nll = nll_gaussian(y_true, y_mean, y_std)
    report.ece = ece(y_true, y_mean, y_std, n_bins=n_bins)
    report.coverage_90 = interval_coverage(y_true, y_mean, y_std, alpha=0.10)
    report.pi_width_90 = prediction_interval_width(y_std, alpha=0.10)
    report.coverage_by_level = coverage_by_level(y_true, y_mean, y_std)
    report.reliability_diagram = reliability_diagram_data(
        y_true, y_mean, y_std, n_bins=n_bins
    )

    if ensemble is not None:
        disag = ensemble_disagreement(np.asarray(ensemble))
        report.ensemble_disagreement_mean = float(np.mean(disag))

    if ood_in_dist_scores is not None and ood_ood_scores is not None:
        try:
            report.ood_auroc = ood_auroc(ood_in_dist_scores, ood_ood_scores)
        except Exception as e:
            report.notes.append(f"OOD AUROC computation failed: {e}")

    return report
