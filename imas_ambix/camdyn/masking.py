"""Clip-mask sampler + frozen named-geometry eval suite (plan §4a).

A *visibility mask* is a boolean array over the ``(n_frames, 16, 16)``
token grid: ``True`` where the model SEES the token, ``False`` where it
is masked (must be reconstructed).  The mask is the clip applied to the
camera input; the reconstruction target is always the full grid.

Five §4a modes (all returned as ``(n_frames, 16, 16)`` bool):

``RANDOM``
    Random window position (centre uniform over the grid) and random
    size (visible-area fraction log-uniform in ``[area_min, area_max]``,
    with per-axis aspect jitter).  A curriculum hook
    (:meth:`ClipMaskConfig.curriculum_area_range`) anneals the area range
    from "large window / nearly full visibility" toward the full
    sampling range as training progresses.

``PANNING``
    A moving/panning clip: the visible window drifts linearly across the
    grid over the frames of the sequence (random start centre + random
    velocity), size fixed within the window.

``FRONTIER``
    Temporal frontier — every token at frame ``< t`` is visible, every
    token at frame ``>= t`` is masked.  This is the W2 forward-horizon
    mode: condition on the clipped stream up to ``t``, reconstruct the
    future.

``FULL``
    Full mask — nothing visible.  Reconstruct from conditioning alone.

``NAMED``
    Draw from the frozen named-geometry suite (see
    :data:`NAMED_GEOMETRIES`): the §2 fixed window, a divertor-only
    window, a centre-column strip, a standard pan, and the frontier
    mode.  Deterministic and seeded so every arm/ablation is scored on
    identical reconstruction tasks.

All randomness flows through an explicit ``numpy.random.Generator`` so
results are reproducible; the named geometries are pure functions of the
grid shape and frame count (no RNG) except the standard pan, whose
trajectory is fixed by construction.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

import numpy as np

GRID_H, GRID_W = 16, 16


class MaskMode(enum.StrEnum):
    """The five §4a masking modes."""

    RANDOM = "random"
    PANNING = "panning"
    FRONTIER = "frontier"
    FULL = "full"
    NAMED = "named"


# ---------------------------------------------------------------------------
# Sampler configuration
# ---------------------------------------------------------------------------


@dataclass
class ClipMaskConfig:
    """Hyper-parameters for the random/panning clip-mask sampler.

    Attributes
    ----------
    area_min, area_max:
        Visible-area fraction bounds (log-uniform).  Plan §4a calls for
        ~5 %–50 % visible.
    aspect_jitter:
        Multiplicative aspect-ratio jitter applied to the window's
        height/width split (drawn log-uniform in
        ``[1/aspect_jitter, aspect_jitter]``).
    pan_max_speed:
        Maximum per-frame window-centre drift (token cells/frame) for
        :data:`MaskMode.PANNING`.
    mode_weights:
        Sampling probabilities over RANDOM / PANNING / FRONTIER / FULL
        when :meth:`sample` is called with ``mode=None`` (mixture mode).
        NAMED is never drawn in the mixture — it is the held-out eval
        suite, drawn explicitly.
    grid:
        ``(H, W)`` token-grid shape.
    """

    area_min: float = 0.05
    area_max: float = 0.50
    aspect_jitter: float = 2.0
    pan_max_speed: float = 1.5
    mode_weights: dict[MaskMode, float] = field(
        default_factory=lambda: {
            MaskMode.RANDOM: 0.55,
            MaskMode.PANNING: 0.20,
            MaskMode.FRONTIER: 0.20,
            MaskMode.FULL: 0.05,
        }
    )
    grid: tuple[int, int] = (GRID_H, GRID_W)

    def curriculum_area_range(self, progress: float) -> tuple[float, float]:
        """Anneal the area range large→full over training (curriculum hook).

        ``progress`` in ``[0, 1]`` (e.g. step / total_steps).  At
        ``progress == 0`` the window is large and visibility is easy
        (area in ``[max(0.5, area_max), 1.0]``); at ``progress == 1`` it
        is the full configured range ``[area_min, area_max]``.  Linear
        interpolation of both bounds in log-space.
        """
        progress = float(np.clip(progress, 0.0, 1.0))
        easy_lo, easy_hi = max(0.5, self.area_max), 1.0
        lo = float(
            np.exp(
                np.log(easy_lo) + progress * (np.log(self.area_min) - np.log(easy_lo))
            )
        )
        hi = float(
            np.exp(
                np.log(easy_hi) + progress * (np.log(self.area_max) - np.log(easy_hi))
            )
        )
        return lo, hi


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _window_mask(
    grid: tuple[int, int],
    centre: tuple[float, float],
    half_h: float,
    half_w: float,
) -> np.ndarray:
    """Boolean (H, W) window: True inside the axis-aligned box."""
    h, w = grid
    rows = np.arange(h)[:, None]
    cols = np.arange(w)[None, :]
    r0, c0 = centre
    inside = (
        (rows >= np.floor(r0 - half_h))
        & (rows <= np.ceil(r0 + half_h))
        & (cols >= np.floor(c0 - half_w))
        & (cols <= np.ceil(c0 + half_w))
    )
    return inside


def _area_to_halfsizes(
    grid: tuple[int, int],
    area_frac: float,
    aspect: float,
) -> tuple[float, float]:
    """Convert a visible-area fraction + aspect into (half_h, half_w)."""
    h, w = grid
    n_visible = max(1.0, area_frac * h * w)
    # window area ≈ (2*half_h+1)*(2*half_w+1); solve with aspect = hh/hw
    # (2*aspect*hw+1)(2*hw+1) = n  →  approximate via sqrt
    base = np.sqrt(n_visible / max(aspect, 1e-6)) / 2.0
    half_w = base
    half_h = base * aspect
    half_h = float(np.clip(half_h, 0.0, h / 2.0))
    half_w = float(np.clip(half_w, 0.0, w / 2.0))
    return half_h, half_w


# ---------------------------------------------------------------------------
# The sampler
# ---------------------------------------------------------------------------


def sample_clip_mask(
    n_frames: int,
    config: ClipMaskConfig,
    rng: np.random.Generator,
    *,
    mode: MaskMode | None = None,
    progress: float | None = None,
) -> tuple[np.ndarray, dict]:
    """Sample one visibility mask of shape ``(n_frames, H, W)``.

    Parameters
    ----------
    n_frames:
        Temporal length of the window.
    config:
        :class:`ClipMaskConfig`.
    rng:
        Explicit generator (reproducibility).
    mode:
        Force a specific mode, or ``None`` to draw from
        ``config.mode_weights`` (NAMED excluded from the mixture).
    progress:
        Optional curriculum progress in ``[0, 1]``; when given, the area
        range is annealed via
        :meth:`ClipMaskConfig.curriculum_area_range`.

    Returns
    -------
    (mask, meta):
        ``mask`` is bool ``(n_frames, H, W)`` (True = visible); ``meta``
        records the realised mode and geometry for logging/audit.
    """
    h, w = config.grid
    if mode is None:
        modes = list(config.mode_weights.keys())
        probs = np.array([config.mode_weights[m] for m in modes], dtype=float)
        probs = probs / probs.sum()
        mode = modes[int(rng.choice(len(modes), p=probs))]

    if progress is not None:
        area_lo, area_hi = config.curriculum_area_range(progress)
    else:
        area_lo, area_hi = config.area_min, config.area_max

    meta: dict = {"mode": mode.value}

    if mode is MaskMode.FULL:
        mask = np.zeros((n_frames, h, w), dtype=bool)
        return mask, meta

    if mode is MaskMode.FRONTIER:
        # Visible up to frame t (exclusive of t); rest masked.
        t = int(rng.integers(1, max(2, n_frames)))
        mask = np.zeros((n_frames, h, w), dtype=bool)
        mask[:t] = True
        meta["frontier_t"] = t
        return mask, meta

    # RANDOM / PANNING share window-size sampling (log-uniform area).
    area = float(np.exp(rng.uniform(np.log(area_lo), np.log(area_hi))))
    aspect = float(
        np.exp(rng.uniform(-np.log(config.aspect_jitter), np.log(config.aspect_jitter)))
    )
    half_h, half_w = _area_to_halfsizes(config.grid, area, aspect)
    meta.update(area_frac=area, aspect=aspect, half_h=half_h, half_w=half_w)

    if mode is MaskMode.RANDOM:
        # Fixed centre across frames (the clip does not move).
        r0 = float(rng.uniform(0, h - 1))
        c0 = float(rng.uniform(0, w - 1))
        win = _window_mask(config.grid, (r0, c0), half_h, half_w)
        mask = np.broadcast_to(win, (n_frames, h, w)).copy()
        meta["centre"] = (r0, c0)
        return mask, meta

    # PANNING — window centre drifts linearly across frames.
    r0 = float(rng.uniform(0, h - 1))
    c0 = float(rng.uniform(0, w - 1))
    vr = float(rng.uniform(-config.pan_max_speed, config.pan_max_speed))
    vc = float(rng.uniform(-config.pan_max_speed, config.pan_max_speed))
    mask = np.zeros((n_frames, h, w), dtype=bool)
    for f in range(n_frames):
        cr = float(np.clip(r0 + vr * f, 0, h - 1))
        cc = float(np.clip(c0 + vc * f, 0, w - 1))
        mask[f] = _window_mask(config.grid, (cr, cc), half_h, half_w)
    meta.update(centre0=(r0, c0), velocity=(vr, vc))
    return mask, meta


# ---------------------------------------------------------------------------
# Frozen named-geometry eval suite
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NamedGeometry:
    """A deterministic, reproducible eval geometry.

    Attributes
    ----------
    name:
        Stable identifier used in metric tables.
    kind:
        One of ``"window"``, ``"strip"``, ``"pan"``, ``"frontier"``.
    description:
        Human-readable summary.
    params:
        Geometry parameters (centre/half-sizes/velocity/frontier-fraction).
    """

    name: str
    kind: str
    description: str
    params: dict

    def mask(
        self, n_frames: int, grid: tuple[int, int] = (GRID_H, GRID_W)
    ) -> np.ndarray:
        """Materialise this geometry's visibility mask, ``(n_frames, H, W)``."""
        h, w = grid
        p = self.params
        if self.kind == "frontier":
            frac = float(p["frontier_fraction"])
            t = max(1, int(round(frac * n_frames)))
            m = np.zeros((n_frames, h, w), dtype=bool)
            m[:t] = True
            return m
        if self.kind == "window":
            win = _window_mask(
                grid, tuple(p["centre"]), float(p["half_h"]), float(p["half_w"])
            )
            return np.broadcast_to(win, (n_frames, h, w)).copy()
        if self.kind == "strip":
            m2 = np.zeros((h, w), dtype=bool)
            c0, c1 = p["col_range"]
            r0, r1 = p.get("row_range", (0, h))
            m2[r0:r1, c0:c1] = True
            return np.broadcast_to(m2, (n_frames, h, w)).copy()
        if self.kind == "pan":
            c = p["centre0"]
            vr, vc = p["velocity"]
            hh, hw = float(p["half_h"]), float(p["half_w"])
            m = np.zeros((n_frames, h, w), dtype=bool)
            for f in range(n_frames):
                cr = float(np.clip(c[0] + vr * f, 0, h - 1))
                cc = float(np.clip(c[1] + vc * f, 0, w - 1))
                m[f] = _window_mask(grid, (cr, cc), hh, hw)
            return m
        raise ValueError(f"unknown named-geometry kind {self.kind!r}")


# The §2 fixed window: a central window covering the lower-outboard
# quadrant — the canonical "clipped task" geometry referenced in the plan.
NAMED_GEOMETRIES: dict[str, NamedGeometry] = {
    "fixed_section2": NamedGeometry(
        name="fixed_section2",
        kind="window",
        description=(
            "The §2 fixed clipped window — a central window over the "
            "lower-outboard machine quadrant (the canonical clipped task)."
        ),
        params={"centre": (10.0, 10.0), "half_h": 3.0, "half_w": 3.0},
    ),
    "divertor_only": NamedGeometry(
        name="divertor_only",
        kind="window",
        description=(
            "A small divertor-only window (lower rows, narrow) — tests "
            "reconstruction of the bulk plasma from a divertor-region view."
        ),
        params={"centre": (13.5, 8.0), "half_h": 1.5, "half_w": 3.0},
    ),
    "centre_column_strip": NamedGeometry(
        name="centre_column_strip",
        kind="strip",
        description=(
            "A vertical centre-column strip (full height, central columns) "
            "— the inboard/centre-stack sightline."
        ),
        params={"col_range": (6, 10), "row_range": (0, 16)},
    ),
    "standard_pan": NamedGeometry(
        name="standard_pan",
        kind="pan",
        description=(
            "One standard left→right pan: a fixed-size window drifting "
            "across the grid at +1 col/frame from the left edge."
        ),
        params={
            "centre0": (8.0, 2.0),
            "velocity": (0.0, 1.0),
            "half_h": 2.0,
            "half_w": 2.0,
        },
    ),
    "frontier_half": NamedGeometry(
        name="frontier_half",
        kind="frontier",
        description=(
            "Temporal frontier at the sequence midpoint — visible for the "
            "first half, masked for the second (the W2 forward-horizon task)."
        ),
        params={"frontier_fraction": 0.5},
    ),
}
"""The frozen, deterministic eval suite.  Every arm/ablation is scored on
these identical reconstruction tasks (plan §4a)."""


def named_geometry_mask(
    name: str,
    n_frames: int,
    grid: tuple[int, int] = (GRID_H, GRID_W),
) -> np.ndarray:
    """Visibility mask for a frozen named geometry, ``(n_frames, H, W)``."""
    if name not in NAMED_GEOMETRIES:
        raise KeyError(
            f"unknown named geometry {name!r}; available: {sorted(NAMED_GEOMETRIES)}"
        )
    return NAMED_GEOMETRIES[name].mask(n_frames, grid)
