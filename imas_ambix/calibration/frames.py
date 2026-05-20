"""Frame calibration — corpus-wide camera dynamic-range statistics.

Walks a collection of Level-1 shot Zarr stores (which hold camera data as
``/rbb``, ``/rba`` etc. arrays), samples a small number of frames per shot,
and accumulates global + per-shot dynamic-range statistics.

The ``suggested`` field of :class:`FrameCalibration` drives
``_normalise_frames_to_uint8`` in ``imas_ambix.tokenizer.frames``:

- ``"per_shot"`` — normalise each shot independently (current v0 behaviour).
- ``"global"`` — use corpus-wide min/max for cross-shot comparability.

Typical usage::

    from pathlib import Path
    from imas_ambix.calibration.frames import compute_frame_calibration

    shots = [Path(f"/work/.../shots/{s}.zarr") for s in shot_ids]
    cal = compute_frame_calibration(shots, camera="rbb")
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FrameCalibration:
    """Corpus-wide dynamic-range statistics for one camera source.

    Attributes
    ----------
    camera:
        Level-1 source key (e.g. ``"rbb"``).
    global_min:
        Minimum pixel value over all sampled frames across all shots.
    global_max:
        Maximum pixel value over all sampled frames across all shots.
    global_mean:
        Mean pixel value over all sampled frames (Welford streaming).
    global_std:
        Std of pixel values over all sampled frames.
    per_shot_min:
        Dict mapping shot index (0-based position in ``shot_paths``) to
        the per-shot minimum.
    per_shot_max:
        Same for per-shot maximum.
    suggested:
        ``"global"`` if per-shot maxima vary by ≤ 5× relative to
        ``global_max``; otherwise ``"per_shot"``.
    """

    camera: str
    global_min: float
    global_max: float
    global_mean: float
    global_std: float
    per_shot_min: dict[int, float]
    per_shot_max: dict[int, float]
    suggested: str  # "global" | "per_shot"


# ---------------------------------------------------------------------------
# Welford accumulator (re-used from signals but kept local to avoid import)
# ---------------------------------------------------------------------------


class _WelfordAccumulator:
    __slots__ = ("_n", "_mean", "_m2")

    def __init__(self) -> None:
        self._n: int = 0
        self._mean: float = 0.0
        self._m2: float = 0.0

    def update(self, values: np.ndarray) -> None:
        import numpy as np

        arr = np.asarray(values, dtype=np.float64).ravel()
        arr = arr[np.isfinite(arr)]
        n_b = int(arr.size)
        if n_b == 0:
            return
        mean_b = float(arr.mean())
        m2_b = float(np.sum((arr - mean_b) ** 2))
        n_a = self._n
        n_ab = n_a + n_b
        delta = mean_b - self._mean
        self._mean = (n_a * self._mean + n_b * mean_b) / n_ab
        self._m2 += m2_b + delta**2 * n_a * n_b / n_ab
        self._n = n_ab

    @property
    def mean(self) -> float:
        return self._mean if self._n > 0 else float("nan")

    @property
    def std(self) -> float:
        if self._n < 1:
            return float("nan")
        return float(self._m2 / self._n) ** 0.5


# ---------------------------------------------------------------------------
# Per-shot loader
# ---------------------------------------------------------------------------


def _load_shot_frames(
    shot_path: Path,
    shot_idx: int,
    camera: str,
    sample_frames_per_shot: int,
) -> tuple[int, np.ndarray | None, float, float]:
    """Load a sample of frames from one shot's camera array.

    Returns ``(shot_idx, sampled_frames, shot_min, shot_max)``.
    ``sampled_frames`` is ``None`` when the shot cannot be opened.

    The camera data is expected to live at ``<shot_path>/<camera>`` as a
    Zarr array (the Level-1 layout used by FAIR-MAST).  If the group does
    not exist, falls back to checking for an ``"frames"`` variable inside
    the group.
    """
    import numpy as np

    try:
        import zarr  # type: ignore[import-untyped]

        store = zarr.open(str(shot_path), mode="r")

        # Try <camera> as a direct array first, then as a group with a
        # 'frames' or first-available variable.
        cam_node = store.get(camera)
        if cam_node is None:
            return (shot_idx, None, float("nan"), float("nan"))

        if isinstance(cam_node, zarr.Array):
            arr_ref = cam_node
        else:
            # It's a group — look for a frames variable or take the first array
            arr_ref = None
            for key in ("frames", "data", "image"):
                candidate = cam_node.get(key)
                if isinstance(candidate, zarr.Array):
                    arr_ref = candidate
                    break
            if arr_ref is None:
                # Take first array in the group
                for _key, v in cam_node.items():
                    if isinstance(v, zarr.Array):
                        arr_ref = v
                        break
            if arr_ref is None:
                return (shot_idx, None, float("nan"), float("nan"))

        total_frames = arr_ref.shape[0]
        if total_frames == 0:
            return (shot_idx, None, float("nan"), float("nan"))

        # Sample evenly-spaced frame indices
        n_sample = min(sample_frames_per_shot, total_frames)
        indices = [
            int(round(i * (total_frames - 1) / max(n_sample - 1, 1)))
            for i in range(n_sample)
        ]
        # Deduplicate while preserving order
        seen: set[int] = set()
        indices_unique = [i for i in indices if not (i in seen or seen.add(i))]  # type: ignore[func-returns-value]

        sampled_list = []
        for idx in indices_unique:
            frame = np.asarray(arr_ref[idx], dtype=np.float32)
            sampled_list.append(frame.ravel())

        if not sampled_list:
            return (shot_idx, None, float("nan"), float("nan"))

        sampled = np.concatenate(sampled_list)
        return (shot_idx, sampled, float(sampled.min()), float(sampled.max()))

    except Exception:
        return (shot_idx, None, float("nan"), float("nan"))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_frame_calibration(
    shot_paths: list[Path],
    camera: str = "rbb",
    *,
    sample_frames_per_shot: int = 8,
    max_workers: int = 4,
) -> FrameCalibration:
    """Accumulate dynamic-range statistics for one camera across the corpus.

    Parameters
    ----------
    shot_paths:
        Paths to Level-1 shot ``.zarr`` stores.
    camera:
        Level-1 source key (e.g. ``"rbb"``).
    sample_frames_per_shot:
        Number of frames to sample from each shot (evenly spaced in time).
        Keeping this small (default 8) avoids loading the full frame stack.
    max_workers:
        Thread-pool concurrency.

    Returns
    -------
    FrameCalibration
        The ``suggested`` field is ``"per_shot"`` if any per-shot max
        exceeds ``5 × global_max / n_shots_valid`` — i.e. the dynamic
        range varies substantially across shots.  Otherwise ``"global"``.

    Notes
    -----
    The 5× threshold is derived from the requirement that cross-shot
    comparability is meaningful only when shot-to-shot dynamic range
    variation is modest.  If one shot has 10 000 ADU while another has
    100 ADU the same normalisation constant would crush the weak shot to
    noise.
    """
    import math

    acc = _WelfordAccumulator()
    global_min = math.inf
    global_max = -math.inf
    per_shot_min: dict[int, float] = {}
    per_shot_max: dict[int, float] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_load_shot_frames, p, idx, camera, sample_frames_per_shot): idx
            for idx, p in enumerate(shot_paths)
        }
        for fut in as_completed(futures):
            try:
                shot_idx, sampled, s_min, s_max = fut.result()
            except Exception:
                continue

            if sampled is None:
                continue

            acc.update(sampled)
            global_min = min(global_min, s_min)
            global_max = max(global_max, s_max)
            per_shot_min[shot_idx] = s_min
            per_shot_max[shot_idx] = s_max

    if math.isinf(global_min):
        global_min = float("nan")
    if math.isinf(global_max) or global_max < 0:
        global_max = float("nan")

    # Decide normalisation strategy.
    # "per_shot" when any shot's max/min spread is > 5× the inter-shot spread.
    suggested = "global"
    maxima = list(per_shot_max.values())
    if len(maxima) >= 2:
        shot_max_range = max(maxima) - min(maxima)
        # If the range of per-shot maxima exceeds 5× the minimum max, the
        # cross-shot dynamic range is too variable for a shared normalisation.
        min_max = min(m for m in maxima if m > 0) if any(m > 0 for m in maxima) else 1.0
        if shot_max_range > 5.0 * min_max:
            suggested = "per_shot"

    return FrameCalibration(
        camera=camera,
        global_min=float(global_min),
        global_max=float(global_max),
        global_mean=acc.mean,
        global_std=acc.std,
        per_shot_min=per_shot_min,
        per_shot_max=per_shot_max,
        suggested=suggested,
    )
