"""Camera-derived visible-emission-edge prototype (rbb).

TIME-BOXED PROTOTYPE — feasibility note only.
This module provides a rough proof-of-concept for detecting the visible
plasma emission boundary from rbb (centre-stack camera) frames.

It is the BACKUP TARGET for plasma-state-space-v0.  The lock-gating
deliverables are the Dα/leakage inventory, the corpus N's, and the
calibration harness.  This code is deliberately minimal — do NOT invest
R&D here until the orchestrator confirms camera-boundary is the chosen
target.

Method (rough):
    1. Load one frame from the rbb Zarr group.
    2. Apply a Sobel edge filter to detect intensity gradients.
    3. Find the maximum-gradient contour along radial lines.
    4. Return the R, Z coordinates of the emission edge at each poloidal angle.

Feasibility notes (from 2026-05-29 exploration)
------------------------------------------------
- rbb frames: (T, H, W) uint16, H≈512, W≈640, at 100–400 Hz.
- The MAST rbb camera views the outboard midplane through a window.
  Emission from the main plasma and divertor are both visible.
- A gradient-based edge finder (Sobel) can locate the outer emission
  boundary, but calibration to physical coordinates (R, Z) requires
  the camera calibration matrix (not present in the Zarr store).
- WITHOUT camera calibration, we get pixel-space boundary — still useful
  as a relative shape change metric but not a physical LCFS proxy.
- rca (2D, no time axis) appears to be a single-frame reference image;
  it is not useful as a target signal.

Recommendation (for orchestrator):
- Camera-boundary is a technically feasible target in PIXEL SPACE.
- Converting to physical (R, Z) space requires camera calibration data
  (extrinsic/intrinsic matrices, plasma vessel geometry) that is not
  in the current corpus.
- If camera-boundary is chosen as target, the recommendation is to
  use a pixel-space boundary trace as a proxy for plasma shape, framed
  as "predict camera emission pattern" rather than "predict LCFS position".
- Corpus size with rbb: ~7,000 shots (shots >= ~20,000) vs ~17,000 for
  Dα-from-magnetics.  This roughly halves the corpus.
- STRONG RECOMMENDATION: use Dα-from-magnetics as v0 target.  Camera
  boundary can be added as a measured lift increment in v0.1.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class BoundaryProtoConfig:
    """Configuration for the camera boundary prototype.

    Attributes
    ----------
    threshold_fraction:
        Fraction of max gradient to use as boundary threshold (0–1).
    n_radial_lines:
        Number of radial lines for boundary extraction.
    """

    threshold_fraction: float = 0.3
    n_radial_lines: int = 36


def load_rbb_frame(
    shot_zarr_path: Path,
    frame_index: int = 0,
) -> np.ndarray | None:
    """Load one frame from the rbb camera Zarr group.

    Parameters
    ----------
    shot_zarr_path:
        Path to the shot's Zarr root (e.g. ``/path/to/30001.zarr``).
    frame_index:
        Frame index within the rbb group (default 0 = first frame).

    Returns
    -------
    (H, W) uint16 array, or None if rbb is absent or invalid.
    """
    import zarr  # noqa: PLC0415

    rbb_path = shot_zarr_path / "rbb"
    if not rbb_path.exists():
        return None
    try:
        store = zarr.open_group(str(shot_zarr_path), mode="r")
        data = store["rbb"]["data"]
        arr = np.asarray(data)
        if arr.ndim == 3:
            # (T, H, W) — return one frame
            if frame_index >= arr.shape[0]:
                frame_index = 0
            return arr[frame_index].astype(np.float32)
        elif arr.ndim == 2:
            return arr.astype(np.float32)
        else:
            return None
    except Exception as e:
        logger.debug("Cannot load rbb frame from %s: %s", shot_zarr_path, e)
        return None


def detect_emission_edge(
    frame: np.ndarray,
    config: BoundaryProtoConfig | None = None,
) -> dict[str, np.ndarray]:
    """Detect the visible plasma emission edge in a camera frame.

    Uses a Sobel gradient magnitude to find the emission boundary.
    Returns pixel-space boundary coordinates.

    Parameters
    ----------
    frame:
        (H, W) float32 intensity array.
    config:
        Detection configuration.

    Returns
    -------
    dict with keys:
        ``"gradient_magnitude"`` : (H, W) gradient magnitude image
        ``"edge_pixels_row"``    : (N,) edge pixel row indices
        ``"edge_pixels_col"``    : (N,) edge pixel column indices
        ``"boundary_radius_px"`` : (n_lines,) radial distance to edge
                                   per angular bin (pixel units)
    """
    from scipy import ndimage  # noqa: PLC0415

    if config is None:
        config = BoundaryProtoConfig()

    # Sobel gradient magnitude
    gy = ndimage.sobel(frame, axis=0)
    gx = ndimage.sobel(frame, axis=1)
    gradient_mag = np.hypot(gx, gy)

    # Threshold at fraction of max gradient
    thresh = config.threshold_fraction * gradient_mag.max()
    edge_mask = gradient_mag > thresh

    rows, cols = np.where(edge_mask)

    # Compute boundary radius along radial lines from image centre
    H, W = frame.shape  # noqa: N806 — conventional image dimension names
    cy, cx = H / 2.0, W / 2.0
    angles = np.linspace(0, 2 * np.pi, config.n_radial_lines, endpoint=False)

    boundary_radius = np.full(config.n_radial_lines, np.nan)
    for i, angle in enumerate(angles):
        # Direction vector
        dy = np.sin(angle)
        dx = np.cos(angle)
        # Find edge pixels along this radial direction
        if len(rows) == 0:
            continue
        # Project each edge pixel onto this ray
        rel_r = rows - cy
        rel_c = cols - cx
        proj = rel_r * dy + rel_c * dx
        perp = np.abs(rel_r * dx - rel_c * dy)
        # Select pixels close to the ray and positive projection
        on_ray = (perp < 5.0) & (proj > 0)
        if on_ray.sum() == 0:
            continue
        # Take the nearest edge along this ray
        boundary_radius[i] = float(proj[on_ray].min())

    return {
        "gradient_magnitude": gradient_mag,
        "edge_pixels_row": rows,
        "edge_pixels_col": cols,
        "boundary_radius_px": boundary_radius,
    }


def camera_boundary_timeseries(
    shot_zarr_path: Path,
    config: BoundaryProtoConfig | None = None,
    max_frames: int = 50,
) -> dict[str, np.ndarray] | None:
    """Extract a boundary-radius time series from the rbb camera.

    PROTOTYPE QUALITY — for feasibility demonstration only.

    Processes up to *max_frames* evenly-spaced frames and returns:
        ``"time"``               : (T,) frame time stamps (if available)
        ``"boundary_radius_px"`` : (T, n_lines) boundary radius per frame
        ``"mean_radius_px"``     : (T,) mean boundary radius per frame

    Returns None if rbb is absent or has fewer than 3 frames.
    """
    import zarr  # noqa: PLC0415

    if config is None:
        config = BoundaryProtoConfig()

    rbb_path = shot_zarr_path / "rbb"
    if not rbb_path.exists():
        return None

    try:
        store = zarr.open_group(str(shot_zarr_path), mode="r")
        data = np.asarray(store["rbb"]["data"])
    except Exception as e:
        logger.debug("Cannot load rbb data: %s", e)
        return None

    if data.ndim != 3 or data.shape[0] < 3:
        return None

    T = data.shape[0]  # noqa: N806 — conventional temporal dimension name
    frame_indices = np.round(np.linspace(0, T - 1, min(max_frames, T))).astype(int)

    # Time axis
    time_arr: np.ndarray | None = None
    try:
        time_arr = np.asarray(store["rbb"]["time"])
    except Exception:
        time_arr = np.arange(T, dtype=np.float32) / 200.0  # assume 200 Hz

    radii = []
    for idx in frame_indices:
        frame = data[idx].astype(np.float32)
        result = detect_emission_edge(frame, config)
        radii.append(result["boundary_radius_px"])

    radii_arr = np.stack(radii, axis=0)  # (n_frames, n_lines)
    mean_radius = np.nanmean(radii_arr, axis=1)

    return {
        "time": time_arr[frame_indices]
        if time_arr is not None
        else frame_indices.astype(float),
        "boundary_radius_px": radii_arr,
        "mean_radius_px": mean_radius,
        "frame_indices": frame_indices,
        "n_total_frames": T,
    }
