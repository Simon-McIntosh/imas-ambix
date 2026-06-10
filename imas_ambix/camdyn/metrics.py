"""PRE-REGISTERED metric module — locks W1 / W2 / W3 before any model exists.

These are scoring INTERFACES.  No model number exists yet; the point of
D0 is to fix the exact definitions of the win conditions so the later
arms cannot move the goalposts.  Every metric here is exercised on
synthetic / random inputs in the tests.

Win conditions (verbatim from the plan, NOT changed here)
---------------------------------------------------------
W1 — dynamics beat statics
    At matched params + matched training tokens, the temporal model
    beats a per-frame spatial-inpainting baseline on **masked-token
    NLL / top-1 accuracy** over the held-out shot split, with a
    **bootstrap CI clear of zero** on the paired difference.
    → :func:`masked_token_nll`, :func:`masked_top1_accuracy`,
      :func:`bootstrap_ci` (on the per-token paired diff).

W2 — forward horizon
    Conditioned on the clipped stream up to time *t*, reconstruct full
    frames at *t + h* (h = 10, 50, 200 ms) **better than persistence and
    the per-frame baseline**.
    → :func:`horizon_frame_offsets` (ms → frame offsets via the per-frame
      timestamps), :func:`horizon_reconstruction_accuracy`.

W3 — latent knows physics
    A **frozen** linear / shallow probe from the latent predicts held-out
    raw diagnostics (Dα ``ada``, line-integrated density ``ane``,
    Thomson Te_core ``ayc``, n=2 mode amp ``ama``) **better than the
    baseline representation**, scored by RMSE / CRPS.
    → :class:`ProbeProtocol`, :func:`probe_rmse`, :func:`crps_gaussian`.

Reported alongside W1
    A MOTION-WEIGHTED token subset — tokens whose identity changes within
    ±50 ms — so the headline numbers are not dominated by static
    background tokens.
    → :func:`motion_weighted_subset`.

rFID is BANNED as a primary metric (S5 lesson).  Pixel L1 / SSIM via the
frozen decoder is secondary-only and not implemented in D0 (no decoding
needed to lock the token-space wins).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# W3 probe targets (named here so the protocol is self-documenting)
# ---------------------------------------------------------------------------

PROBE_TARGETS: dict[str, str] = {
    "dalpha": "ada",  # Dα integrated emission
    "ne_line_integrated": "ane",  # line-integrated electron density
    "te_core": "ayc",  # Thomson core electron temperature
    "n2_mode_amp": "ama",  # n=2 magnetic mode amplitude
}
"""W3 held-out diagnostic targets → their level-1 source group."""

# Horizon win condition (W2): the locked physical horizons.
HORIZON_MS: tuple[float, ...] = (10.0, 50.0, 200.0)
"""W2 forward-reconstruction horizons in milliseconds."""

# Motion-weighted subset window (reported alongside W1).
MOTION_WINDOW_MS: float = 50.0
"""±window (ms) over which a token must change identity to count as 'moving'."""


# ---------------------------------------------------------------------------
# W1 — masked-token NLL + top-1 accuracy (vocab-agnostic)
# ---------------------------------------------------------------------------


def _logsumexp(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    m = np.max(logits, axis=axis, keepdims=True)
    return (m + np.log(np.sum(np.exp(logits - m), axis=axis, keepdims=True))).squeeze(
        axis
    )


def masked_token_nll(
    logits: np.ndarray,
    targets: np.ndarray,
    mask: np.ndarray,
    *,
    reduce: str = "mean",
) -> np.ndarray | float:
    """Negative log-likelihood of the target token over the MASKED set.

    The metric is **vocab-agnostic**: ``logits`` are unnormalised scores
    over whatever head the arm uses (full 262 144-way softmax, 2×9-bit
    factorised, or LFQ bit-head reduced to a per-token logit vector).
    The only contract is ``logits[..., v]`` is the score for token id
    ``v``; the model-provenance / vocab-head decisions are free.

    Parameters
    ----------
    logits:
        ``(..., V)`` float — per-token logits over the vocabulary.
    targets:
        ``(...)`` int — true token ids (must index the last axis of
        ``logits``).
    mask:
        ``(...)`` bool — True where the token was MASKED (i.e. scored).
        Only masked positions contribute (W1 scores reconstruction of
        the hidden tokens, not the visible ones).
    reduce:
        ``"mean"`` (scalar mean NLL over masked tokens), ``"sum"``, or
        ``"none"`` (per-token NLL array, masked positions only, flattened).

    Returns
    -------
    Scalar NLL (mean/sum) or the per-token NLL vector for the masked set.
    The per-token vector feeds :func:`bootstrap_ci` for the paired W1 CI.
    """
    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets)
    mask = np.asarray(mask, dtype=bool)

    logZ = _logsumexp(logits, axis=-1)  # (...,)
    tgt_logit = np.take_along_axis(
        logits, targets[..., None].astype(np.int64), axis=-1
    ).squeeze(-1)
    nll = logZ - tgt_logit  # (...,)

    sel = nll[mask]
    if reduce == "none":
        return sel
    if sel.size == 0:
        return 0.0
    if reduce == "sum":
        return float(sel.sum())
    return float(sel.mean())


def masked_top1_accuracy(
    logits: np.ndarray,
    targets: np.ndarray,
    mask: np.ndarray,
    *,
    reduce: str = "mean",
) -> np.ndarray | float:
    """Top-1 token accuracy over the MASKED set (vocab-agnostic).

    ``reduce="none"`` returns the per-token correctness (0/1) vector for
    the masked positions — the paired input to :func:`bootstrap_ci`.
    """
    logits = np.asarray(logits)
    targets = np.asarray(targets)
    mask = np.asarray(mask, dtype=bool)
    pred = np.argmax(logits, axis=-1)
    correct = (pred == targets).astype(np.float64)
    sel = correct[mask]
    if reduce == "none":
        return sel
    if sel.size == 0:
        return 0.0
    if reduce == "sum":
        return float(sel.sum())
    return float(sel.mean())


def bootstrap_ci(
    paired_diff: np.ndarray,
    *,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, float]:
    """Percentile bootstrap CI on the mean of a paired difference.

    W1 is a *paired* comparison: for each masked token (or each shot, if
    the caller aggregates per-shot first), ``paired_diff`` is
    ``metric_baseline - metric_dynamics`` (for NLL, positive = dynamics
    better) or ``acc_dynamics - acc_baseline`` (positive = dynamics
    better) — i.e. the diff is ALWAYS oriented so positive favours the
    dynamics arm.

    **The W1 win gate is** ``favours_dynamics`` (the ``(1-alpha)`` CI
    lower bound > 0): the dynamics arm is significantly better.
    ``clear_of_zero`` is the weaker two-sided flag (CI excludes 0 in
    *either* direction) — a significant REGRESSION (dynamics worse) is
    ``clear_of_zero=True`` but ``favours_dynamics=False``.  Do not use
    ``clear_of_zero`` as the win verdict.

    Returns ``{mean, lo, hi, alpha, favours_dynamics, clear_of_zero}``.
    """
    x = np.asarray(paired_diff, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return {
            "mean": 0.0,
            "lo": 0.0,
            "hi": 0.0,
            "alpha": alpha,
            "favours_dynamics": False,
            "clear_of_zero": False,
        }
    rng = np.random.default_rng(seed)
    n = x.size
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = x[idx].mean(axis=1)
    lo = float(np.quantile(boot_means, alpha / 2))
    hi = float(np.quantile(boot_means, 1 - alpha / 2))
    mean = float(x.mean())
    return {
        "mean": mean,
        "lo": lo,
        "hi": hi,
        "alpha": alpha,
        "favours_dynamics": bool(lo > 0.0),
        "clear_of_zero": bool(lo > 0.0 or hi < 0.0),
    }


# ---------------------------------------------------------------------------
# Motion-weighted subset (reported alongside W1)
# ---------------------------------------------------------------------------


def motion_weighted_subset(
    tokens: np.ndarray,
    frame_time: np.ndarray,
    *,
    window_ms: float = MOTION_WINDOW_MS,
) -> np.ndarray:
    """Boolean ``(n_frames, H, W)`` — tokens whose identity CHANGES within ±window.

    A token at ``(f, i, j)`` is "moving" if its id differs from the id at
    the same grid cell in any frame within ``±window_ms`` (using the
    per-frame timestamps to convert ms → a frame range).  This subset is
    reported alongside the headline W1 numbers so static background
    tokens (which dominate the count) do not flatter the scores.

    Parameters
    ----------
    tokens:
        ``(n_frames, H, W)`` int token ids.
    frame_time:
        ``(n_frames,)`` timestamps (s).
    window_ms:
        Temporal half-window (ms).
    """
    tokens = np.asarray(tokens)
    ft = np.asarray(frame_time, dtype=np.float64).reshape(-1)
    nfr, h, w = tokens.shape
    win_s = window_ms / 1000.0
    moving = np.zeros((nfr, h, w), dtype=bool)
    for f in range(nfr):
        lo = np.searchsorted(ft, ft[f] - win_s, side="left")
        hi = np.searchsorted(ft, ft[f] + win_s, side="right")
        if hi - lo <= 1:
            continue
        neigh = tokens[lo:hi]  # (k, H, W)
        changed = np.any(neigh != tokens[f][None, :, :], axis=0)
        moving[f] = changed
    return moving


# ---------------------------------------------------------------------------
# W2 — forward-horizon reconstruction
# ---------------------------------------------------------------------------


def horizon_frame_offsets(
    frame_time: np.ndarray,
    *,
    horizons_ms: tuple[float, ...] = HORIZON_MS,
) -> dict[float, int]:
    """Map each physical horizon (ms) to a frame offset for this window.

    The offset is the median number of frames spanning ``h`` ms (using
    the per-frame Δt), so the W2 horizons are physical, not index-based —
    a 200 ms horizon is the same physical lead-time regardless of the
    shot's cadence.  Returns ``{h_ms: offset_frames}`` (offset ≥ 1).
    """
    ft = np.asarray(frame_time, dtype=np.float64).reshape(-1)
    if ft.size < 2:
        return {h: 1 for h in horizons_ms}
    dt = float(np.median(np.diff(ft)))
    if not np.isfinite(dt) or dt <= 0:
        return {h: 1 for h in horizons_ms}
    return {h: max(1, int(round((h / 1000.0) / dt))) for h in horizons_ms}


def horizon_reconstruction_accuracy(
    pred_logits: np.ndarray,
    target_tokens: np.ndarray,
    frame_time: np.ndarray,
    frontier_frame: int,
    *,
    horizons_ms: tuple[float, ...] = HORIZON_MS,
) -> dict[float, dict[str, float]]:
    """Per-horizon top-1 accuracy + NLL of reconstructed FUTURE frames (W2).

    Conditioned on frames ``< frontier_frame`` (the visible clipped
    stream), the arm predicts the full token grid at
    ``frontier_frame + offset(h)``.  This scores those predicted frames
    against truth at each locked horizon.

    Parameters
    ----------
    pred_logits:
        ``(n_frames, H, W, V)`` predicted logits (the arm's full-frame
        reconstruction at every frame).
    target_tokens:
        ``(n_frames, H, W)`` true token ids.
    frame_time:
        ``(n_frames,)`` timestamps (s) — converts ms horizons to offsets.
    frontier_frame:
        The conditioning frontier ``t`` (frames ``< t`` are observed).
    horizons_ms:
        Physical horizons to score.

    Returns
    -------
    ``{h_ms: {"top1_acc": .., "nll": .., "target_frame": .., "valid": ..}}``.
    A horizon whose target frame falls outside the window has
    ``valid=0.0`` and zeroed scores (the caller aggregates only valid
    horizons across windows).
    """
    pred_logits = np.asarray(pred_logits, dtype=np.float64)
    target_tokens = np.asarray(target_tokens)
    n_frames = target_tokens.shape[0]
    offsets = horizon_frame_offsets(frame_time, horizons_ms=horizons_ms)

    out: dict[float, dict[str, float]] = {}
    for h, off in offsets.items():
        tgt_f = frontier_frame + off
        if tgt_f >= n_frames or frontier_frame < 0:
            out[h] = {
                "top1_acc": 0.0,
                "nll": 0.0,
                "target_frame": float(tgt_f),
                "valid": 0.0,
            }
            continue
        logit_f = pred_logits[tgt_f]  # (H, W, V)
        tok_f = target_tokens[tgt_f]  # (H, W)
        full = np.ones(tok_f.shape, dtype=bool)
        out[h] = {
            "top1_acc": float(masked_top1_accuracy(logit_f, tok_f, full)),
            "nll": float(masked_token_nll(logit_f, tok_f, full)),
            "target_frame": float(tgt_f),
            "valid": 1.0,
        }
    return out


def persistence_baseline_accuracy(
    target_tokens: np.ndarray,
    frame_time: np.ndarray,
    frontier_frame: int,
    *,
    horizons_ms: tuple[float, ...] = HORIZON_MS,
) -> dict[float, dict[str, float]]:
    """W2 persistence baseline: predict the LAST observed frame for all horizons.

    Persistence copies the token grid at ``frontier_frame - 1`` (the last
    visible frame) to every future horizon.  W2 requires the arm to beat
    this (and the per-frame baseline).  Returns the same structure as
    :func:`horizon_reconstruction_accuracy` (top-1 acc + valid flag; NLL
    is undefined for a hard copy and reported as 0.0).
    """
    target_tokens = np.asarray(target_tokens)
    n_frames = target_tokens.shape[0]
    offsets = horizon_frame_offsets(frame_time, horizons_ms=horizons_ms)
    last_obs = max(0, frontier_frame - 1)
    persist = target_tokens[last_obs]  # (H, W)

    out: dict[float, dict[str, float]] = {}
    for h, off in offsets.items():
        tgt_f = frontier_frame + off
        if tgt_f >= n_frames or frontier_frame < 0:
            out[h] = {"top1_acc": 0.0, "target_frame": float(tgt_f), "valid": 0.0}
            continue
        acc = float((persist == target_tokens[tgt_f]).mean())
        out[h] = {"top1_acc": acc, "target_frame": float(tgt_f), "valid": 1.0}
    return out


# ---------------------------------------------------------------------------
# W3 — frozen-probe protocol
# ---------------------------------------------------------------------------


@dataclass
class ProbeProtocol:
    """The FROZEN linear/shallow probe protocol that locks W3.

    The probe is fit on TRAIN-split latents → target pairs, then FROZEN
    and scored on HELD-OUT latents.  The latent is never fine-tuned to
    the target (that is the whole point — W3 asks whether the dynamics
    latent *already* encodes the diagnostic).  Both the dynamics arm's
    latent and the per-frame-baseline representation are scored with the
    SAME protocol; W3 wins if the dynamics latent's held-out RMSE (or
    CRPS) beats the baseline's.

    Attributes
    ----------
    probe_kind:
        ``"linear"`` (ridge regression; the recommended frozen probe) or
        ``"mlp1"`` (one hidden layer — still shallow, still frozen).
    ridge_lambda:
        L2 regulariser for the linear probe.
    hidden_dim:
        Hidden width for ``mlp1`` (ignored for linear).
    standardize:
        Z-score the latent features using TRAIN statistics before fitting
        (recommended; keeps the comparison fair across representations of
        different scales).
    targets:
        Ordered W3 target names (see :data:`PROBE_TARGETS`).
    """

    probe_kind: str = "linear"
    ridge_lambda: float = 1.0
    hidden_dim: int = 64
    standardize: bool = True
    targets: tuple[str, ...] = tuple(PROBE_TARGETS)

    # The fitted state (populated by .fit; kept here so a frozen probe is
    # a single serialisable object).
    _W: np.ndarray | None = field(default=None, repr=False)
    _b: np.ndarray | None = field(default=None, repr=False)
    _mu: np.ndarray | None = field(default=None, repr=False)
    _sd: np.ndarray | None = field(default=None, repr=False)

    def fit(self, latents: np.ndarray, targets: np.ndarray) -> ProbeProtocol:
        """Fit the probe on (latents, targets); returns self (then FREEZE).

        ``latents`` is ``(N, D)``, ``targets`` is ``(N, T)``.  For
        ``probe_kind == "linear"`` this is closed-form ridge regression.
        ``mlp1`` falls back to ridge on a fixed random feature map (a
        cheap, deterministic shallow probe with no gradient training —
        keeping D0 CPU-only and the protocol frozen-by-construction).
        """
        X = np.asarray(latents, dtype=np.float64)
        Y = np.asarray(targets, dtype=np.float64)
        if X.ndim != 2 or Y.ndim != 2 or X.shape[0] != Y.shape[0]:
            raise ValueError("latents (N,D) and targets (N,T) must align on N")

        if self.standardize:
            self._mu = X.mean(axis=0)
            self._sd = X.std(axis=0)
            self._sd[self._sd == 0] = 1.0
            X = (X - self._mu) / self._sd

        if self.probe_kind == "mlp1":
            rng = np.random.default_rng(0)
            R = rng.standard_normal((X.shape[1], self.hidden_dim)) / np.sqrt(X.shape[1])
            self._R = R  # type: ignore[attr-defined]
            X = np.tanh(X @ R)

        Xb = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
        d = Xb.shape[1]
        reg = self.ridge_lambda * np.eye(d)
        reg[-1, -1] = 0.0  # do not regularise the bias
        beta = np.linalg.solve(Xb.T @ Xb + reg, Xb.T @ Y)  # (d, T)
        self._W = beta[:-1]
        self._b = beta[-1]
        return self

    def predict(self, latents: np.ndarray) -> np.ndarray:
        """Predict targets from latents using the FROZEN probe — ``(N, T)``."""
        if self._W is None or self._b is None:
            raise RuntimeError("probe is not fitted; call .fit first")
        X = np.asarray(latents, dtype=np.float64)
        if self.standardize and self._mu is not None:
            X = (X - self._mu) / self._sd
        if self.probe_kind == "mlp1":
            X = np.tanh(X @ self._R)  # type: ignore[attr-defined]
        return X @ self._W + self._b


def probe_rmse(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Per-target RMSE — ``(T,)`` (lower is better).  NaN truth ignored."""
    pred = np.asarray(pred, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    if pred.ndim == 1:
        pred = pred[:, None]
        truth = truth[:, None]
    out = np.full(pred.shape[1], np.nan)
    for t in range(pred.shape[1]):
        m = np.isfinite(truth[:, t]) & np.isfinite(pred[:, t])
        if m.any():
            out[t] = float(np.sqrt(np.mean((pred[m, t] - truth[m, t]) ** 2)))
    return out


def crps_gaussian(
    mean: np.ndarray,
    sigma: np.ndarray,
    truth: np.ndarray,
) -> np.ndarray:
    """Per-target mean CRPS of a Gaussian predictive — ``(T,)`` (lower better).

    Closed-form CRPS for a Gaussian forecast ``N(mean, sigma^2)`` against
    a scalar truth:

        CRPS = sigma * [ z (2Φ(z) − 1) + 2φ(z) − 1/√π ],   z = (y − μ)/σ

    A frozen probe that also emits a predictive spread (e.g. ridge
    residual variance, or an ensemble) is scored here; a point probe sets
    ``sigma`` to the train-residual std.  NaN truth is ignored.
    """
    from math import pi, sqrt  # noqa: PLC0415

    mean = np.asarray(mean, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    if mean.ndim == 1:
        mean = mean[:, None]
        sigma = sigma[:, None]
        truth = truth[:, None]

    def _phi(x):  # standard normal pdf
        return np.exp(-0.5 * x * x) / sqrt(2 * pi)

    def _Phi(x):  # standard normal cdf via erf
        from math import erf  # noqa: PLC0415

        return 0.5 * (1.0 + np.vectorize(erf)(x / sqrt(2.0)))

    out = np.full(mean.shape[1], np.nan)
    for t in range(mean.shape[1]):
        m = np.isfinite(truth[:, t]) & np.isfinite(mean[:, t]) & (sigma[:, t] > 0)
        if not m.any():
            continue
        s = sigma[m, t]
        z = (truth[m, t] - mean[m, t]) / s
        crps = s * (z * (2 * _Phi(z) - 1) + 2 * _phi(z) - 1.0 / sqrt(pi))
        out[t] = float(np.mean(crps))
    return out
