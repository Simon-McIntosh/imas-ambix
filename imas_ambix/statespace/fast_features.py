# ruff: noqa: N806
"""D1 — fast-panel feature extraction aligned to MSE time slices.

Interior-information-discovery-v0 §5 D1.

Extracts per-slice feature vectors from the D0 fast-panel loaders
(XmaShot, XsxShot) aligned to the MSE manifest beam_on_slice_times.
Alignment uses nearest-sample lookup with a configurable tolerance.

Feature blocks
--------------
xsx_hcam:
    17 chord amplitudes from hcam_l (chord 10 excluded — stuck channel).
    Background-subtracted using the pre-plasma mean (t < 0).
    (K, 17) per shot.  NaN where the xsx window doesn't cover a slice.

xma_ccbv:
    40 Mirnov coil amplitudes from ccbv_01..40 (modern) / ccbv01..40 (legacy).
    Instantaneous sample nearest to each MSE slice time.
    (K, 40) per shot.  NaN where xma window doesn't cover a slice.

xma_flux:
    8 flux-loop amplitudes from fl_cc01..09 (excluding fl_cc06 which is missing
    in the modern schema).  Equilibrium-adjacent but retained separately so the
    oracle attribution is auditable.
    (K, 8) per shot.

The three blocks are concatenated into a single (K, 65) feature matrix per shot.
Feature indices are recorded in FEATURE_SCHEMA for downstream attribution.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from imas_ambix.statespace.fast_loader import XmaShot, XsxShot

logger = logging.getLogger(__name__)

# Stuck channel in hcam_l (ch10, constant 0.315858 across all shots)
XSX_STUCK_CHANNEL = 10
# Active chord indices for hcam_l (all except the stuck one)
XSX_ACTIVE_CHORDS = [i for i in range(18) if i != XSX_STUCK_CHANNEL]  # 17 chords

XMA_CCBV_MODERN = [f"ccbv_{i:02d}" for i in range(1, 41)]
XMA_CCBV_LEGACY = [f"ccbv{i:02d}" for i in range(1, 41)]
XMA_FLUX_MODERN = [f"fl_cc{i:02d}" for i in range(1, 10) if i != 6]  # 8 channels
XMA_FLUX_LEGACY = [f"flcc{i:02d}" for i in range(1, 10) if i != 6]  # 8 channels


@dataclass
class FeatureSchema:
    """Maps feature-matrix columns to their source (for attribution)."""

    block_names: list[str] = field(default_factory=list)
    block_starts: list[int] = field(default_factory=list)
    block_widths: list[int] = field(default_factory=list)
    total_features: int = 0

    def slice_for(self, block: str) -> slice:
        i = self.block_names.index(block)
        s = self.block_starts[i]
        return slice(s, s + self.block_widths[i])


FEATURE_SCHEMA = FeatureSchema(
    block_names=["xsx_hcam", "xma_ccbv", "xma_flux"],
    block_starts=[0, 17, 57],
    block_widths=[17, 40, 8],
    total_features=65,
)


def _nearest_indices(ref_times: np.ndarray, query_times: np.ndarray) -> np.ndarray:
    """For each query time return the index of the nearest ref_time.

    Returns -1 if |nearest - query| > ``_MAX_GAP_S`` (set to 10 ms).
    """
    _MAX_GAP_S = 0.010
    idx = np.searchsorted(ref_times, query_times, side="left")
    idx = np.clip(idx, 0, len(ref_times) - 1)
    # Check also the left neighbour
    idx_left = np.maximum(idx - 1, 0)
    d_right = np.abs(ref_times[idx] - query_times)
    d_left = np.abs(ref_times[idx_left] - query_times)
    closer_left = d_left < d_right
    best_idx = np.where(closer_left, idx_left, idx)
    best_d = np.where(closer_left, d_left, d_right)
    return np.where(best_d <= _MAX_GAP_S, best_idx, -1)


def _xsx_background(xsx_shot: XsxShot) -> np.ndarray:
    """Pre-plasma background per chord (t < 0), shape (17,)."""
    pre = xsx_shot.time < 0.0
    if pre.sum() > 50:
        return xsx_shot.hcam_l[XSX_ACTIVE_CHORDS][:, pre].mean(axis=1)
    return np.zeros(17, dtype=np.float32)


def extract_xsx_features(
    xsx_shot: XsxShot,
    slice_times: np.ndarray,
) -> np.ndarray:
    """(K, 17) background-subtracted hcam_l amplitudes at each MSE slice time.

    NaN for slices outside the xsx acquisition window.
    """
    K = len(slice_times)
    out = np.full((K, 17), np.nan, dtype=np.float32)
    bg = _xsx_background(xsx_shot)
    idxs = _nearest_indices(xsx_shot.time, slice_times)
    valid = idxs >= 0
    if valid.any():
        raw = xsx_shot.hcam_l[XSX_ACTIVE_CHORDS][:, idxs[valid]]  # (17, n_valid)
        out[valid] = (raw - bg[:, None]).T.astype(np.float32)
    return out


def _xma_channel_data(xma_shot: XmaShot, channel_names: list[str]) -> np.ndarray:
    """(T, C_out) — select columns by channel_names, return NaN for missing."""
    cn = xma_shot.channel_names
    col_map = {n: i for i, n in enumerate(cn)}
    cols = []
    for name in channel_names:
        if name in col_map:
            cols.append(xma_shot.data[:, col_map[name]])
        else:
            cols.append(np.full(xma_shot.n_slices, np.nan, dtype=np.float32))
    return np.stack(cols, axis=1)  # (T, C_out)


def extract_xma_features(
    xma_shot: XmaShot,
    slice_times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """(K, 40) ccbv and (K, 8) flux features at each MSE slice time.

    NaN for slices outside the xma acquisition window.
    """
    K = len(slice_times)
    ccbv_out = np.full((K, 40), np.nan, dtype=np.float32)
    flux_out = np.full((K, 8), np.nan, dtype=np.float32)

    if xma_shot.schema == "modern":
        ccbv_names = XMA_CCBV_MODERN
        flux_names = XMA_FLUX_MODERN
    else:
        ccbv_names = XMA_CCBV_LEGACY
        flux_names = XMA_FLUX_LEGACY

    ccbv_data = _xma_channel_data(xma_shot, ccbv_names)  # (T, 40)
    flux_data = _xma_channel_data(xma_shot, flux_names)  # (T, 8)

    idxs = _nearest_indices(xma_shot.time, slice_times)
    valid = idxs >= 0
    if valid.any():
        ccbv_out[valid] = ccbv_data[idxs[valid]]
        flux_out[valid] = flux_data[idxs[valid]]
    return ccbv_out, flux_out


def build_feature_matrix(
    xsx_shot: XsxShot | None,
    xma_shot: XmaShot | None,
    slice_times: np.ndarray,
) -> np.ndarray:
    """(K, 65) feature matrix for one shot aligned to MSE slice times.

    Layout: [xsx_hcam(17) | xma_ccbv(40) | xma_flux(8)].
    Blocks missing for this shot are all-NaN.
    """
    K = len(slice_times)
    F = FEATURE_SCHEMA.total_features
    X = np.full((K, F), np.nan, dtype=np.float32)

    if xsx_shot is not None:
        X[:, FEATURE_SCHEMA.slice_for("xsx_hcam")] = extract_xsx_features(
            xsx_shot, slice_times
        )
    if xma_shot is not None:
        ccbv, flux = extract_xma_features(xma_shot, slice_times)
        X[:, FEATURE_SCHEMA.slice_for("xma_ccbv")] = ccbv
        X[:, FEATURE_SCHEMA.slice_for("xma_flux")] = flux
    return X
