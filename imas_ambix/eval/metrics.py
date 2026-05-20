"""Evaluation metrics for the Fusion World Model frame rollout.

The model produces predicted frame sequences; we need quantitative signals
to compare them against ground-truth footage. This module collects every
metric referenced in ``plans/demo.md`` §4 and ``plans/world-model-v0.md``
§5 into a single, dependency-minimal file. Heavy deps (LPIPS, Inception-V3)
are lazily imported so the module loads instantly even in environments
where torch/torchvision are absent.

Related plans:
- ``plans/demo.md`` §4.1 reconstruction metrics (rFID, PSNR, LPIPS)
- ``plans/demo.md`` §4.2 physics-derived metrics (centroid, chord, edge)
- ``plans/world-model-v0.md`` §5 training-time evaluation hooks
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


# ---------------------------------------------------------------------------
# Reconstruction-quality metrics
# ---------------------------------------------------------------------------


def psnr(reference: np.ndarray, prediction: np.ndarray) -> float:
    """Mean PSNR between reference and prediction, in decibels.

    Averages over the T-frame axis. For identical inputs the result is
    ``+inf``; for heavily corrupted predictions it falls below 10 dB.
    Inputs must be uint8 arrays of the same shape: ``(T, H, W, 3)`` or
    ``(T, H, W)``.
    """
    import numpy as np

    ref = np.asarray(reference, dtype=np.float64)
    pred = np.asarray(prediction, dtype=np.float64)
    if ref.shape != pred.shape:
        raise ValueError(
            f"reference and prediction must have the same shape, "
            f"got {ref.shape} vs {pred.shape}"
        )
    t = ref.shape[0]
    psnr_sum = 0.0
    for i in range(t):
        mse = float(np.mean((ref[i] - pred[i]) ** 2))
        if mse == 0.0:
            psnr_sum += math.inf
        else:
            psnr_sum += 10.0 * math.log10(255.0**2 / mse)
    return psnr_sum / t


def lpips(reference: np.ndarray, prediction: np.ndarray) -> float:
    """Mean perceptual similarity (LPIPS) across T frames.

    Uses the ``lpips`` package with the default AlexNet backbone. Lower
    is better (perfect reconstruction → 0.0, unrelated images → ~1.0).
    Inputs must be uint8 ``(T, H, W, 3)``; grayscale ``(T, H, W)`` is
    automatically broadcast to three channels.

    Requires ``lpips`` and ``torch`` to be installed; use
    ``pytest.importorskip("lpips")`` in tests to skip when unavailable.
    """
    import lpips as lpips_lib  # lazy import
    import numpy as np
    import torch

    ref = np.asarray(reference, dtype=np.float32)
    pred = np.asarray(prediction, dtype=np.float32)
    # Grayscale → RGB
    if ref.ndim == 3:
        ref = np.stack([ref] * 3, axis=-1)
        pred = np.stack([pred] * 3, axis=-1)
    # Normalise [0, 255] → [-1, 1]
    ref = ref / 127.5 - 1.0
    pred = pred / 127.5 - 1.0
    # (T, H, W, 3) → (T, 3, H, W) for torch
    ref_t = torch.from_numpy(ref.transpose(0, 3, 1, 2))
    pred_t = torch.from_numpy(pred.transpose(0, 3, 1, 2))

    loss_fn = lpips_lib.LPIPS(net="alex", verbose=False)
    loss_fn.eval()
    with torch.no_grad():
        scores = loss_fn(ref_t, pred_t)  # (T, 1, 1, 1)
    return float(scores.mean().item())


def rfid(reference: np.ndarray, prediction: np.ndarray) -> float:
    """Reconstruction FID using InceptionV3 features.

    Computes the Fréchet distance between the InceptionV3 feature
    distributions of ``reference`` and ``prediction``. For small T
    (<10 frames), ``eps=1e-3`` is applied to the covariance matrix square
    root for numerical stability.

    Requires ``torch`` and ``torchvision``; use
    ``pytest.importorskip("torchvision")`` in tests to skip when unavailable.
    Inputs must be uint8 ``(T, H, W, 3)`` or ``(T, H, W)``.
    """
    import numpy as np
    import torch
    import torchvision.models as tvm
    import torchvision.transforms.functional as tvf
    from scipy.linalg import sqrtm

    def _extract_features(frames: np.ndarray, model: torch.nn.Module) -> np.ndarray:
        arr = np.asarray(frames, dtype=np.uint8)
        if arr.ndim == 3:
            arr = np.stack([arr] * 3, axis=-1)
        imgs = []
        for i in range(arr.shape[0]):
            t = torch.from_numpy(arr[i]).permute(2, 0, 1)  # (3, H, W)
            # Resize to 299 for Inception
            t = tvf.resize(t, [299, 299], antialias=True).float() / 255.0
            # Normalise to ImageNet stats
            t = tvf.normalize(t, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            imgs.append(t)
        batch = torch.stack(imgs)  # (T, 3, 299, 299)
        with torch.no_grad():
            feats = model(batch)
        return feats.cpu().numpy()

    # Load Inception, strip final classifier, keep pool layer
    inception = tvm.inception_v3(weights=tvm.Inception_V3_Weights.IMAGENET1K_V1)
    inception.fc = torch.nn.Identity()
    inception.eval()

    ref_feats = _extract_features(reference, inception)  # (T, 2048)
    pred_feats = _extract_features(prediction, inception)  # (T, 2048)

    mu_r, mu_p = ref_feats.mean(axis=0), pred_feats.mean(axis=0)
    sigma_r = np.cov(ref_feats, rowvar=False)
    sigma_p = np.cov(pred_feats, rowvar=False)

    t = ref_feats.shape[0]
    eps = 1e-3 if t < 10 else 1e-6

    # FID = ||mu_r - mu_p||^2 + Tr(sigma_r + sigma_p - 2 * sqrt(sigma_r @ sigma_p))
    diff = mu_r - mu_p
    # Add eps to diagonal for numerical stability
    offset = np.eye(sigma_r.shape[0]) * eps
    cov_mean, _ = sqrtm(sigma_r @ sigma_p + offset, disp=False)
    if np.iscomplexobj(cov_mean):
        cov_mean = cov_mean.real
    fid = float(
        diff @ diff + np.trace(sigma_r) + np.trace(sigma_p) - 2.0 * np.trace(cov_mean)
    )
    return float(max(fid, 0.0))


# ---------------------------------------------------------------------------
# Physics-derived metrics
# ---------------------------------------------------------------------------


def frame_centroid(frames: np.ndarray) -> np.ndarray:
    """Brightness-weighted centroid per frame.

    Converts to grayscale if needed, then computes the brightness-weighted
    mean position in pixel coordinates.

    Parameters
    ----------
    frames:
        ``(T, H, W)`` or ``(T, H, W, 3)`` uint8 array.

    Returns
    -------
    np.ndarray
        Shape ``(T, 2)`` of ``(x, y)`` in pixel units (x=column, y=row).
    """
    import numpy as np

    arr = np.asarray(frames, dtype=np.float64)
    if arr.ndim == 4:
        # Convert to grayscale: simple average over channels
        arr = arr.mean(axis=-1)
    t, h, w = arr.shape
    ys = np.arange(h, dtype=np.float64)  # row indices
    xs = np.arange(w, dtype=np.float64)  # column indices

    centroids = np.zeros((t, 2), dtype=np.float64)
    for i in range(t):
        frame = arr[i]
        total = frame.sum()
        if total == 0.0:
            centroids[i] = [w / 2.0, h / 2.0]
        else:
            cx = float((frame * xs[np.newaxis, :]).sum() / total)
            cy = float((frame * ys[:, np.newaxis]).sum() / total)
            centroids[i] = [cx, cy]
    return centroids


def modality_coherence(
    decoded_frames: np.ndarray,
    magnetic_axis_r: np.ndarray,
    frame_image_extent_m: tuple[float, float] | None = None,
) -> float:
    """Pearson r between the frame brightness centroid R and the equilibrium magnetic axis R.

    Measures cross-modality time-alignment quality: a high Pearson r (~0.7+)
    indicates the camera centroid tracks the equilibrium magnetic axis, as
    expected physically when frame and signal tokens are correctly aligned.

    Parameters
    ----------
    decoded_frames:
        ``(T, H, W)`` or ``(T, H, W, 3)`` uint8 frame array.
    magnetic_axis_r:
        ``(T,)`` float array of equilibrium magnetic axis R values (metres).
    frame_image_extent_m:
        Optional ``(R_min, R_max)`` in metres for the horizontal extent of
        the image. When provided, the centroid column index is linearly
        mapped to physical R before computing the correlation. When ``None``,
        the raw column index is used (still gives a meaningful correlation).

    Returns
    -------
    float
        Pearson r in ``[-1, 1]``, or ``float("nan")`` if fewer than 3
        valid paired samples exist or if either series is constant.
    """
    import numpy as np

    centroids = frame_centroid(decoded_frames)  # (T, 2): [col, row]
    col_r = centroids[:, 0]  # column index ≡ horizontal R-direction

    # Convert column index → physical R if extent is provided
    if frame_image_extent_m is not None:
        r_min, r_max = frame_image_extent_m
        arr = np.asarray(decoded_frames)
        # width from the frame array
        if arr.ndim == 4:
            w = arr.shape[2]
        else:
            w = arr.shape[2]
        col_r = r_min + col_r * (r_max - r_min) / max(w - 1, 1)

    axis_r = np.asarray(magnetic_axis_r, dtype=np.float64)
    col_r = np.asarray(col_r, dtype=np.float64)

    # Align lengths
    n = min(len(col_r), len(axis_r))
    x = col_r[:n]
    y = axis_r[:n]

    # Need at least 3 finite paired points
    finite_mask = np.isfinite(x) & np.isfinite(y)
    if finite_mask.sum() < 3:
        return float("nan")

    x = x[finite_mask]
    y = y[finite_mask]

    x_std = float(np.std(x))
    y_std = float(np.std(y))
    if x_std < 1e-12 or y_std < 1e-12:
        return float("nan")

    # Inline Pearson r (no scipy)
    x_z = (x - x.mean()) / x_std
    y_z = (y - y.mean()) / y_std
    r = float(np.mean(x_z * y_z))
    # Clamp to [-1, 1] to guard against floating-point overrun
    return max(-1.0, min(1.0, r))


def centroid_mse(reference_frames: np.ndarray, prediction_frames: np.ndarray) -> float:
    """MSE between centroid trajectories of reference and prediction frames.

    Computes brightness-weighted centroids for both inputs and returns the
    mean squared error across the T-frame trajectory. Returns 0.0 when
    reference and prediction are identical.
    """
    import numpy as np

    ref_c = frame_centroid(reference_frames)
    pred_c = frame_centroid(prediction_frames)
    return float(np.mean((ref_c - pred_c) ** 2))


def chord_integrated_emission(
    frames: np.ndarray, chord_y: int | None = None
) -> np.ndarray:
    """Integrate brightness along the horizontal midplane chord per frame.

    Sums pixel brightness along a single row (the midplane chord) for each
    frame in the sequence. This is a simplified proxy for the chord-integrated
    emission measured by line-integrated visible diagnostics.

    Parameters
    ----------
    frames:
        ``(T, H, W)`` or ``(T, H, W, 3)`` uint8 array.
    chord_y:
        Row index of the midplane chord. Defaults to ``H // 2``.

    Returns
    -------
    np.ndarray
        Shape ``(T,)`` — the sum of brightness along the chord per frame.
    """
    import numpy as np

    arr = np.asarray(frames, dtype=np.float64)
    if arr.ndim == 4:
        arr = arr.mean(axis=-1)
    t, h, _w = arr.shape
    if chord_y is None:
        chord_y = h // 2
    return arr[:, chord_y, :].sum(axis=-1)


def chord_nrmse(
    reference: np.ndarray,
    prediction: np.ndarray,
    chord_y: int | None = None,
) -> float:
    """Normalised RMSE of chord-integrated emission time series.

    Normalises the RMSE by the standard deviation of the reference chord
    series. Returns ``np.inf`` if the reference series has zero variance
    (e.g. a constant bright or dark frame sequence).

    Parameters
    ----------
    reference:
        ``(T, H, W)`` or ``(T, H, W, 3)`` uint8 ground-truth frames.
    prediction:
        Same shape as ``reference``.
    chord_y:
        Midplane chord row index (defaults to ``H // 2``).

    Returns
    -------
    float
        NRMSE value; 0.0 for perfect agreement, higher for larger errors.
    """
    import math

    import numpy as np

    ref_chord = chord_integrated_emission(reference, chord_y=chord_y)
    pred_chord = chord_integrated_emission(prediction, chord_y=chord_y)
    rmse = float(np.sqrt(np.mean((ref_chord - pred_chord) ** 2)))
    ref_std = float(np.std(ref_chord))
    if ref_std == 0.0:
        return math.inf if rmse > 0.0 else 0.0
    return rmse / ref_std


def edge_displacement(
    reference: np.ndarray,
    prediction: np.ndarray,
    threshold: int = 128,
) -> float:
    """Median-absolute-deviation of edge displacement between reference and prediction.

    For each frame, thresholds both images at ``threshold``, extracts the
    outer-boundary edges via scipy's Canny-equivalent (binary erosion), and
    computes the median absolute deviation between the edge pixel positions
    of reference and prediction.

    Returns the mean MAD across T frames. If either frame has no edge
    pixels after thresholding, that frame contributes 0.0 to the average.

    Requires ``scipy``; available in the project's standard venv via torch.
    """
    import numpy as np
    from scipy.ndimage import binary_erosion

    def _edge_pixels(frame_gray: np.ndarray) -> np.ndarray:
        """Return (N, 2) array of (row, col) edge pixel positions."""
        binary = frame_gray > threshold
        interior = binary_erosion(binary)
        edge_mask = binary & ~interior
        return np.argwhere(edge_mask)

    ref = np.asarray(reference, dtype=np.uint8)
    pred = np.asarray(prediction, dtype=np.uint8)
    if ref.ndim == 4:
        ref = ref.mean(axis=-1).astype(np.uint8)
    if pred.ndim == 4:
        pred = pred.mean(axis=-1).astype(np.uint8)

    t = ref.shape[0]
    mad_sum = 0.0
    for i in range(t):
        ref_edges = _edge_pixels(ref[i])
        pred_edges = _edge_pixels(pred[i])
        if len(ref_edges) == 0 or len(pred_edges) == 0:
            continue
        # Compute pairwise distances between nearest edges via broadcasting
        # (expensive for dense edges — acceptable for small T in eval)
        # Use median of min distances from ref to pred
        ref_f = ref_edges.astype(np.float32)
        pred_f = pred_edges.astype(np.float32)
        # Nearest-neighbour distance from each ref edge pixel to any pred edge pixel
        diffs = ref_f[:, np.newaxis, :] - pred_f[np.newaxis, :, :]  # (Nr, Np, 2)
        dists = np.sqrt((diffs**2).sum(axis=-1))  # (Nr, Np)
        min_dists = dists.min(axis=-1)  # (Nr,)
        mad_sum += float(np.median(np.abs(min_dists - np.median(min_dists))))
    return mad_sum / t


# ---------------------------------------------------------------------------
# Convenience aggregator
# ---------------------------------------------------------------------------


def compute_all_metrics(
    reference: np.ndarray,
    prediction: np.ndarray,
    *,
    chord_y: int | None = None,
) -> dict[str, float]:
    """Compute all evaluation metrics and return as a flat dictionary.

    Runs every metric in this module on the provided frame sequences. Heavy
    metrics (``lpips``, ``rfid``) are attempted; if their optional deps are
    absent the corresponding key is set to ``float('nan')`` rather than
    raising.

    Parameters
    ----------
    reference:
        Ground-truth frames, ``(T, H, W, 3)`` or ``(T, H, W)`` uint8.
    prediction:
        Predicted frames, same shape as ``reference``.
    chord_y:
        Midplane chord row index for chord metrics. Defaults to ``H // 2``.

    Returns
    -------
    dict[str, float]
        Keys: ``psnr``, ``lpips``, ``rfid``, ``centroid_mse``,
        ``chord_nrmse``, ``edge_displacement_mad``.
    """
    results: dict[str, float] = {}

    results["psnr"] = psnr(reference, prediction)
    results["centroid_mse"] = centroid_mse(reference, prediction)
    results["chord_nrmse"] = chord_nrmse(reference, prediction, chord_y=chord_y)
    results["edge_displacement_mad"] = edge_displacement(reference, prediction)

    try:
        results["lpips"] = lpips(reference, prediction)
    except (ImportError, ModuleNotFoundError):
        results["lpips"] = float("nan")

    try:
        results["rfid"] = rfid(reference, prediction)
    except (ImportError, ModuleNotFoundError):
        results["rfid"] = float("nan")

    return results
