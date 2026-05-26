"""Pre-decode rbb frames into (T, H, H, 3) uint8 Zarr stores.

This module amortises the per-encode CPU work (uint16→uint8 normalisation,
RGB-replicate, 536×560→256×256 bilinear resize) by running it once and
writing the result next to the raw L1 Zarr.  The encode loop can then
skip straight from ``(T, image_size, image_size, 3) uint8`` to bf16/GPU,
cutting per-shot CPU time from ~0.4 s to ~0.02 s.

Output layout
--------------
``/work/projects/imas_gpu/mast/preprocessed/rbb-256/<shot>.zarr``
with one 4-D array ``data: (T, image_size, image_size, 3) uint8``.

Normalisation contract
-----------------------
The uint8 conversion MUST be bit-exact with
:func:`imas_ambix.tokenizer.frames._normalise_frames_to_uint8`:

  1. Cast raw uint16 → float32.
  2. Compute per-shot ``lo = f.min()``, ``hi = f.max()``.
  3. ``u8 = ((f - lo) * 255.0 / (hi - lo)).clip(0, 255).astype(uint8)``.
  4. If ``hi <= lo``, return zeros.

Resize backend (open decision q1)
-----------------------------------
Default ``resize_backend="cv2"`` uses ``cv2.INTER_LINEAR``.  Pass
``resize_backend="torch"`` to use ``torch.nn.functional.interpolate``
(bilinear, align_corners=False).  A 100-shot A/B for token-id divergence
should settle which backend the full corpus uses before the first
full-corpus run.

Usage
-----
::

    from imas_ambix.data.preprocess import preprocess_rbb_shot, bulk_preprocess
    from imas_ambix.data.paths import LEVEL1_DIR, MIRROR_ROOT

    dst_root = MIRROR_ROOT / "preprocessed" / "rbb-256"
    path = preprocess_rbb_shot(15085, LEVEL1_DIR, dst_root)

    paths = bulk_preprocess([15085, 15086, 15087], LEVEL1_DIR, dst_root, workers=4)
"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PreprocessReport:
    """Outcome of preprocessing one shot.

    Parameters
    ----------
    shot_id:
        Numeric shot identifier.
    camera:
        Camera name, e.g. ``"rbb"``.
    image_size:
        Target spatial resolution (square), e.g. 256.
    n_frames:
        Number of frames written (0 when skipped or error).
    elapsed_s:
        Wall time in seconds.
    output_path:
        Destination Zarr path on disk.
    skipped:
        ``True`` when output already existed and skip_existing=True.
    error:
        Exception message on failure; ``None`` on success.
    """

    shot_id: int
    camera: str
    image_size: int
    n_frames: int
    elapsed_s: float
    output_path: Path
    skipped: bool = field(default=False)
    error: str | None = field(default=None)


# ---------------------------------------------------------------------------
# Core single-shot function
# ---------------------------------------------------------------------------


def preprocess_rbb_shot(
    shot_id: int,
    src_root: Path,
    dst_root: Path,
    *,
    image_size: int = 256,
    resize_backend: str = "cv2",
    skip_existing: bool = True,
    camera: str = "rbb",
) -> PreprocessReport:
    """Preprocess one shot's camera frames and write to Zarr.

    Reads ``src_root/<shot_id>.zarr/<camera>``, normalises uint16→uint8
    with per-shot min/max (bit-exact with
    :func:`imas_ambix.tokenizer.frames._normalise_frames_to_uint8`),
    RGB-replicates, bilinear-resizes to ``image_size × image_size``, and
    writes ``(T, image_size, image_size, 3) uint8`` to
    ``dst_root/<shot_id>.zarr``.

    Parameters
    ----------
    shot_id:
        Numeric shot identifier.
    src_root:
        Directory containing ``<shot_id>.zarr`` sub-directories (L1 shots root).
    dst_root:
        Destination root; output is ``dst_root/<shot_id>.zarr``.
    image_size:
        Target square spatial resolution (locked decision: 256).
    resize_backend:
        ``"cv2"`` (default) or ``"torch"``.  See open decision q1.
    skip_existing:
        If ``True`` (default) return immediately when the output exists.
    camera:
        Camera group name inside the shot Zarr.

    Returns
    -------
    PreprocessReport
    """
    import numpy as np
    import xarray as xr
    import zarr

    out_path = dst_root / f"{shot_id}.zarr"
    t0 = time.monotonic()

    if skip_existing and out_path.exists():
        return PreprocessReport(
            shot_id=shot_id,
            camera=camera,
            image_size=image_size,
            n_frames=0,
            elapsed_s=0.0,
            output_path=out_path,
            skipped=True,
        )

    try:
        # --- Load raw frames --------------------------------------------------
        shot_zarr = src_root / f"{shot_id}.zarr"
        ds = xr.open_zarr(str(shot_zarr / camera))
        data_vars = list(ds.data_vars)
        if not data_vars:
            raise ValueError(
                f"no data variables in group '{camera}' of shot {shot_id}"
            )
        raw = ds[data_vars[0]].values  # (T, H, W) uint16 (or uint8 already)
        raw = np.asarray(raw)

        # --- Normalise to uint8 (MUST match _normalise_frames_to_uint8) ------
        u8 = _normalise_to_uint8(raw)  # (T, H, W) uint8

        # --- RGB-replicate: (T, H, W) → (T, H, W, 3) -----------------------
        rgb = np.repeat(u8[..., None], 3, axis=-1)  # (T, H, W, 3)

        # --- Resize each frame to image_size² --------------------------------
        # resized shape: (T, image_size, image_size, 3)
        resized = _resize_frames(rgb, image_size, backend=resize_backend)

        # --- Write output Zarr ----------------------------------------------
        dst_root.mkdir(parents=True, exist_ok=True)
        zarr.save_array(str(out_path / "data"), resized)

        return PreprocessReport(
            shot_id=shot_id,
            camera=camera,
            image_size=image_size,
            n_frames=int(resized.shape[0]),
            elapsed_s=time.monotonic() - t0,
            output_path=out_path,
        )

    except Exception as exc:  # noqa: BLE001
        return PreprocessReport(
            shot_id=shot_id,
            camera=camera,
            image_size=image_size,
            n_frames=0,
            elapsed_s=time.monotonic() - t0,
            output_path=out_path,
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# Bulk helper
# ---------------------------------------------------------------------------


def bulk_preprocess(
    shot_ids: Sequence[int],
    src_root: Path,
    dst_root: Path,
    *,
    image_size: int = 256,
    resize_backend: str = "cv2",
    skip_existing: bool = True,
    camera: str = "rbb",
    workers: int = 1,
    use_processes: bool = False,
) -> list[PreprocessReport]:
    """Preprocess frames for multiple shots, returning one report per shot.

    Parameters
    ----------
    shot_ids:
        Ordered sequence of shot IDs to process.
    src_root:
        L1 shots root containing ``<shot>.zarr`` dirs.
    dst_root:
        Destination root for preprocessed Zarr stores.
    image_size:
        Target square resolution (locked decision: 256).
    resize_backend:
        ``"cv2"`` or ``"torch"`` (open decision q1).
    skip_existing:
        Skip shots whose output Zarr already exists.
    camera:
        Camera group to read (default ``"rbb"``).
    workers:
        Number of parallel workers.  For CPU-bound resize work with many
        workers, prefer ``use_processes=True`` to avoid the GIL.
    use_processes:
        Use :class:`~concurrent.futures.ProcessPoolExecutor` instead of
        :class:`~concurrent.futures.ThreadPoolExecutor`.  Heavier on
        startup but avoids the GIL for pure NumPy / cv2 work.

    Returns
    -------
    list[PreprocessReport]
        One report per shot, in input order.
    """

    def _process_one(sid: int) -> PreprocessReport:
        return preprocess_rbb_shot(
            sid,
            src_root,
            dst_root,
            image_size=image_size,
            resize_backend=resize_backend,
            skip_existing=skip_existing,
            camera=camera,
        )

    ids = list(shot_ids)

    if workers <= 1:
        return [_process_one(sid) for sid in ids]

    executor_cls = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
    reports: dict[int, PreprocessReport] = {}
    with executor_cls(max_workers=workers) as pool:
        futures = {pool.submit(_process_one, sid): sid for sid in ids}
        for fut in futures:
            sid = futures[fut]
            reports[sid] = fut.result()
    return [reports[sid] for sid in ids]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _normalise_to_uint8(frames: np.ndarray) -> np.ndarray:  # type: ignore[name-defined]  # noqa: F821
    """Normalise any-dtype frames to uint8 in [0, 255] using per-shot min/max.

    This is a standalone copy of
    :func:`imas_ambix.tokenizer.frames._normalise_frames_to_uint8` so that
    the preprocess module does not import the tokenizer (which triggers the
    Open-MAGVIT2 subprocess chain on import).

    MUST remain bit-exact with the tokenizer's implementation.
    """
    import numpy as np

    if frames.dtype == np.uint8:
        return frames
    f = frames.astype(np.float32)
    lo = float(f.min())
    hi = float(f.max())
    if hi <= lo:
        return np.zeros_like(f, dtype=np.uint8)
    return ((f - lo) * 255.0 / (hi - lo)).clip(0, 255).astype(np.uint8)


def _resize_frames(
    rgb: np.ndarray,  # type: ignore[name-defined]  # noqa: F821
    image_size: int,
    *,
    backend: str = "cv2",
) -> np.ndarray:  # type: ignore[name-defined]  # noqa: F821
    """Resize ``(T, H, W, 3) uint8`` frames to ``(T, image_size, image_size, 3)``.

    Parameters
    ----------
    rgb:
        Input frames ``(T, H, W, 3) uint8``.
    image_size:
        Target square spatial resolution.
    backend:
        ``"cv2"`` (default) — ``cv2.resize(INTER_LINEAR)``.
        ``"torch"`` — ``torch.nn.functional.interpolate(bilinear)``.
        Open decision q1 tracks which should become the corpus default.

    Returns
    -------
    np.ndarray
        ``(T, image_size, image_size, 3) uint8``.
    """
    import numpy as np

    t = rgb.shape[0]
    h, w = rgb.shape[1], rgb.shape[2]
    if h == image_size and w == image_size:
        return rgb

    if backend == "cv2":
        import cv2  # type: ignore[import]

        out = np.empty((t, image_size, image_size, 3), dtype=np.uint8)
        for i in range(t):
            out[i] = cv2.resize(
                rgb[i], (image_size, image_size), interpolation=cv2.INTER_LINEAR
            )
        return out

    if backend == "torch":
        import torch
        import torch.nn.functional as functional  # noqa: N812

        # (T, H, W, 3) uint8 → (T, 3, H, W) float32
        x = torch.from_numpy(rgb).permute(0, 3, 1, 2).float()
        # interpolate expects (N, C, H, W) or (N, C, D, H, W)
        x = functional.interpolate(
            x,
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        )
        # → (T, 3, S, S) float32 → (T, S, S, 3) uint8
        x = x.permute(0, 2, 3, 1).clamp(0, 255).byte()
        return x.numpy()

    raise ValueError(
        f"Unknown resize_backend={backend!r}. Choose 'cv2' or 'torch'."
    )
