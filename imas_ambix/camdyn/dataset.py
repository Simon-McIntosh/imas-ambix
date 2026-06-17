"""Frame-grid-preserving token-stream dataset for camera dynamics.

This is the camera-dynamics counterpart to
:class:`imas_ambix.data.loaders.ShotTokenDataset`.  That loader FLATTENS
each frame's 16×16 token grid into a 1-D world-model stream; spatial clip
masking needs the grid intact, so this module keeps the per-frame
``(16, 16)`` structure and yields windows of shape ``(n_frames, 16, 16)``.

Data layout (verified on shot 24065)
-------------------------------------
Tokens (Zarr **V3** — ``zarr.json``)::

    /work/projects/imas_gpu/mast-tokens/v1/frames/<shot>/rbb.zarr
        store["tokens"]  (T, 16, 16) int32   vocab 2^18 = 262144

The token store carries no timestamps and
``metadata.temporal_compression == 1`` ⇒ token-frame *i* corresponds 1:1
to raw-frame *i*.

Raw frames + timestamps (Zarr **V2** — ``.zmetadata``)::

    /work/projects/imas_gpu/mast/level1/shots/<shot>.zarr
        store["rbb"]["time"]  (T,) float64  seconds  — time[i] ↔ frame i

The two tiers must be opened with the matching reader: the V3 token store
holds the integer tokens, the V2 level-1 store the per-frame timestamps.
Both are FAIR-MAST Zarr — use ``zarr``/``numpy``, NOT imas-python.

Window sampling reuses the spirit of
:class:`imas_ambix.data.loaders.WindowSamplerConfig` (``context_length``
→ ``n_frames`` here, plus ``stride`` and ``seed``) but operates over
frames, not flat tokens, and joins the timestamp axis so every yielded
window carries ``frame_time`` (s) and ``dt`` (s, per-frame forward
difference) for native-Δt conditioning (see the package docstring).

The dataset is lazy and torch-DataLoader-worker-safe: Zarr arrays are
opened on demand inside ``__getitem__``/``__iter__`` so the dataset
object is cheap to pickle and share across worker processes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from imas_ambix.data.paths import LEVEL1_DIR, TOKEN_ROOT
from imas_ambix.tokenizer.store_targets import assert_not_target_path

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FRAME_GRID: tuple[int, int] = (16, 16)
"""Spatial token-grid shape (H_tok, W_tok) per frame (Open-MAGVit2 LFQ)."""

VOCAB_SIZE: int = 1 << 18
"""LFQ vocabulary size (2^18 = 262144)."""

DEFAULT_CAMERA = "rbb"
DEFAULT_VOCAB_VERSION = "v1"


def frames_token_path(
    shot_id: int,
    camera: str = DEFAULT_CAMERA,
    vocab_version: str = DEFAULT_VOCAB_VERSION,
    token_root: Path | None = None,
) -> Path:
    """Return the V3 token Zarr path for one shot's camera frames.

    Every world-model INPUT path the camera loader opens is built here, so
    this is the chokepoint where the eval-only boundary guard fires: the
    resolved path is run through
    :func:`imas_ambix.tokenizer.store_targets.assert_not_target_path` before
    it is returned.  A ``token_root`` that resolves under ``TARGET_ROOT``
    (the eval-only L2 reconstruction-target store) is hard-refused at load
    time — the camera input stream can never admit a prediction target
    (Wall 3, now live in the real loader).
    """
    root = token_root or TOKEN_ROOT
    path = root / vocab_version / "frames" / str(shot_id) / f"{camera}.zarr"
    return assert_not_target_path(path)


def level1_shot_path(shot_id: int, level1_dir: Path | None = None) -> Path:
    """Return the V2 level-1 Zarr path carrying raw frames + timestamps."""
    root = level1_dir or LEVEL1_DIR
    return root / f"{shot_id}.zarr"


# ---------------------------------------------------------------------------
# Per-shot spec
# ---------------------------------------------------------------------------


@dataclass
class FrameTokenShotSpec:
    """Lazy descriptor for one shot's frame-grid token stream.

    Attributes
    ----------
    shot_id:
        Integer shot identifier.
    n_frames:
        Number of token frames (``tokens.shape[0]``).
    token_path:
        V3 Zarr root holding ``tokens`` of shape ``(n_frames, 16, 16)``.
    level1_path:
        V2 Zarr root holding ``rbb/time`` (per-frame timestamps).  May be
        ``None`` if the level-1 shot is absent — windows then carry a
        synthetic uniform time base (Δt from ``fallback_dt``) and
        ``time_is_synthetic=True``.
    """

    shot_id: int
    n_frames: int
    token_path: Path
    level1_path: Path | None = None
    camera: str = DEFAULT_CAMERA


@dataclass
class FrameWindowConfig:
    """Window-sampling hyper-parameters (frame-grid analogue of
    :class:`imas_ambix.data.loaders.WindowSamplerConfig`).

    Attributes
    ----------
    n_frames:
        Frames per sampled window (temporal context length).
    stride:
        Minimum gap between window starts when enumerating a shot.
    seed:
        Base RNG seed; each ``__iter__`` reseeds for reproducible epochs.
    drop_short:
        If True, shots with fewer than ``n_frames`` frames are skipped.
        If False, short shots yield a single zero-padded window with a
        ``valid_frames`` mask marking the padding.
    fallback_dt:
        Δt (s) used to synthesise a uniform time base when the level-1
        timestamp axis is unavailable.  Default 1/600 s (the reference
        rbb cadence).
    """

    n_frames: int = 16
    stride: int = 8
    seed: int = 0
    drop_short: bool = True
    fallback_dt: float = 1.0 / 600.0


@dataclass
class FrameWindow:
    """One sampled training window.

    Attributes
    ----------
    shot_id:
        Source shot.
    start:
        Index of the first frame in the source token stream.
    tokens:
        ``(n_frames, 16, 16)`` int32 token grid.
    frame_time:
        ``(n_frames,)`` float64 per-frame timestamps (s).  Synthetic
        (uniform ``fallback_dt``) when ``time_is_synthetic``.
    dt:
        ``(n_frames,)`` float64 per-frame forward Δt (s); the last entry
        repeats the previous Δt (no successor frame).
    valid_frames:
        ``(n_frames,)`` bool — True for real frames, False for padding on
        short shots.
    time_is_synthetic:
        True when ``frame_time`` was synthesised (level-1 axis missing).
    """

    shot_id: int
    start: int
    tokens: np.ndarray
    frame_time: np.ndarray
    dt: np.ndarray
    valid_frames: np.ndarray
    time_is_synthetic: bool = False

    def as_dict(self) -> dict[str, np.ndarray | int | bool]:
        """Return a plain dict (convenient for torch ``default_collate``)."""
        return {
            "shot_id": int(self.shot_id),
            "start": int(self.start),
            "tokens": self.tokens,
            "frame_time": self.frame_time,
            "dt": self.dt,
            "valid_frames": self.valid_frames,
            "time_is_synthetic": bool(self.time_is_synthetic),
        }


# ---------------------------------------------------------------------------
# Lazy frame readers (worker-safe — open on demand)
# ---------------------------------------------------------------------------


def _read_token_window(
    token_path: Path, start: int, n_frames: int
) -> tuple[np.ndarray, int]:
    """Read ``tokens[start:start+n_frames]`` from a V3 token store.

    Returns ``(window, n_real)`` where ``window`` is zero-padded to
    ``n_frames`` and ``n_real`` is the number of real frames read.
    """
    import zarr  # noqa: PLC0415

    store = zarr.open_group(str(token_path), mode="r")
    tok = store["tokens"]
    end = min(start + n_frames, int(tok.shape[0]))
    real = np.asarray(tok[start:end], dtype=np.int32)
    n_real = int(real.shape[0])
    if n_real < n_frames:
        pad = np.zeros((n_frames - n_real, *FRAME_GRID), dtype=np.int32)
        real = np.concatenate([real, pad], axis=0)
    return real, n_real


def _read_frame_times(level1_path: Path | None, camera: str) -> np.ndarray | None:
    """Read the per-frame timestamp axis from the V2 level-1 store.

    Returns the ``camera/time`` array (s) or ``None`` if unavailable.
    """
    if level1_path is None or not Path(level1_path).exists():
        return None
    import zarr  # noqa: PLC0415

    try:
        store = zarr.open_group(str(level1_path), mode="r")
        if camera not in set(store.group_keys()):
            return None
        grp = store[camera]
        if "time" not in set(grp.array_keys()):
            return None
        return np.asarray(grp["time"], dtype=np.float64)
    except Exception as e:  # pragma: no cover - corpus robustness
        logger.debug("Cannot read %s/%s/time: %s", level1_path, camera, e)
        return None


def _forward_dt(frame_time: np.ndarray) -> np.ndarray:
    """Per-frame forward Δt; last entry repeats the previous Δt."""
    if frame_time.size < 2:
        return np.zeros_like(frame_time)
    d = np.diff(frame_time)
    return np.concatenate([d, d[-1:]])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class FrameTokenDataset:
    """Frame-grid-preserving windowed token dataset.

    Supports both map-style (``__len__`` / ``__getitem__``) and
    iterable (``__iter__``, shuffled per epoch) access so it slots into a
    torch ``DataLoader`` either way.  Each item is a :class:`FrameWindow`
    (call ``.as_dict()`` for a collatable mapping).

    Parameters
    ----------
    specs:
        Per-shot :class:`FrameTokenShotSpec` descriptors.
    config:
        :class:`FrameWindowConfig` sampling configuration.
    as_dict:
        When True, ``__getitem__``/``__iter__`` yield the ``.as_dict()``
        mapping instead of the :class:`FrameWindow` object (torch-friendly).
    """

    def __init__(
        self,
        specs: list[FrameTokenShotSpec],
        config: FrameWindowConfig,
        *,
        as_dict: bool = False,
    ) -> None:
        self._specs = specs
        self._config = config
        self._as_dict = as_dict
        self._windows = self._enumerate_windows()

    # -- window enumeration ------------------------------------------------

    def _enumerate_windows(self) -> list[tuple[int, int]]:
        """Flat list of ``(spec_index, start_frame)`` windows."""
        nf = self._config.n_frames
        stride = self._config.stride
        out: list[tuple[int, int]] = []
        for i, spec in enumerate(self._specs):
            if spec.n_frames < nf:
                if not self._config.drop_short:
                    out.append((i, 0))
                continue
            last_start = spec.n_frames - nf
            starts = list(range(0, last_start + 1, stride))
            if starts and starts[-1] != last_start:
                starts.append(last_start)  # always cover the tail
            for s in starts:
                out.append((i, s))
        return out

    # -- materialise one window -------------------------------------------

    def _materialise(self, spec_index: int, start: int) -> FrameWindow:
        spec = self._specs[spec_index]
        nf = self._config.n_frames
        tokens, n_real = _read_token_window(spec.token_path, start, nf)

        times = _read_frame_times(spec.level1_path, spec.camera)
        synthetic = False
        if times is not None and times.shape[0] >= start + n_real:
            ft = times[start : start + n_real].astype(np.float64)
        else:
            synthetic = True
            ft = (
                start + np.arange(n_real, dtype=np.float64)
            ) * self._config.fallback_dt

        # Pad time / dt to n_frames using fallback Δt beyond real frames.
        if n_real < nf:
            base = ft[-1] if n_real > 0 else 0.0
            step = float(ft[-1] - ft[-2]) if n_real > 1 else self._config.fallback_dt
            pad_t = base + step * np.arange(1, nf - n_real + 1, dtype=np.float64)
            frame_time = np.concatenate([ft, pad_t])
        else:
            frame_time = ft

        dt = _forward_dt(frame_time)
        valid = np.zeros(nf, dtype=bool)
        valid[:n_real] = True

        return FrameWindow(
            shot_id=spec.shot_id,
            start=start,
            tokens=tokens,
            frame_time=frame_time,
            dt=dt,
            valid_frames=valid,
            time_is_synthetic=synthetic,
        )

    def _emit(self, win: FrameWindow):
        return win.as_dict() if self._as_dict else win

    # -- map-style ---------------------------------------------------------

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, index: int):
        spec_index, start = self._windows[index]
        return self._emit(self._materialise(spec_index, start))

    # -- iterable ----------------------------------------------------------

    def __iter__(self) -> Iterator:
        rng = np.random.default_rng(self._config.seed)
        order = rng.permutation(len(self._windows))
        for idx in order:
            spec_index, start = self._windows[int(idx)]
            yield self._emit(self._materialise(spec_index, start))

    # -- introspection -----------------------------------------------------

    @property
    def specs(self) -> list[FrameTokenShotSpec]:
        return self._specs

    @property
    def config(self) -> FrameWindowConfig:
        return self._config


# ---------------------------------------------------------------------------
# Corpus discovery + spec construction
# ---------------------------------------------------------------------------


def _token_n_frames(token_path: Path) -> int | None:
    """Return ``tokens.shape[0]`` for a V3 token store, or None if unreadable."""
    import zarr  # noqa: PLC0415

    try:
        store = zarr.open_group(str(token_path), mode="r")
        if "tokens" not in set(store.array_keys()):
            return None
        return int(store["tokens"].shape[0])
    except Exception as e:  # pragma: no cover - corpus robustness
        logger.debug("Cannot read %s/tokens: %s", token_path, e)
        return None


def discover_token_shots(
    camera: str = DEFAULT_CAMERA,
    vocab_version: str = DEFAULT_VOCAB_VERSION,
    token_root: Path | None = None,
    level1_dir: Path | None = None,
    shot_ids: list[int] | None = None,
    read_n_frames: bool = False,
) -> list[FrameTokenShotSpec]:
    """Enumerate shots with persisted camera tokens on disk.

    Scans ``<token_root>/<vocab_version>/frames/<shot>/<camera>.zarr``.
    By default this is a CHEAP directory scan: ``n_frames`` is left at 0
    and only filled when ``read_n_frames`` opens each store (slower —
    one Zarr open per shot).  Window enumeration needs ``n_frames``, so a
    caller that builds a dataset should pass ``read_n_frames=True`` for
    the subset of shots actually used (not all 9,527 at once).

    Parameters
    ----------
    shot_ids:
        Restrict discovery to these shots (skips the full directory
        scan).  When None, every shot directory under the frames root is
        considered.
    read_n_frames:
        When True, open each token store to populate ``n_frames``.
    """
    root = token_root or TOKEN_ROOT
    l1 = level1_dir or LEVEL1_DIR
    # Wall 3 (live): refuse a token root that resolves under TARGET_ROOT
    # before any directory scan — the eval-only reconstruction-target store
    # must never be enumerated as a world-model input.
    frames_dir = assert_not_target_path(root / vocab_version / "frames")

    if shot_ids is None:
        if not frames_dir.exists():
            return []
        candidate = sorted(
            int(p.name) for p in frames_dir.iterdir() if p.is_dir() and p.name.isdigit()
        )
    else:
        candidate = sorted(int(s) for s in shot_ids)

    specs: list[FrameTokenShotSpec] = []
    for sid in candidate:
        tpath = frames_token_path(sid, camera, vocab_version, token_root=root)
        if not (tpath / "zarr.json").exists() and not tpath.exists():
            continue
        n_frames = 0
        if read_n_frames:
            n = _token_n_frames(tpath)
            if n is None:
                continue
            n_frames = n
        lpath = level1_shot_path(sid, level1_dir=l1)
        specs.append(
            FrameTokenShotSpec(
                shot_id=sid,
                n_frames=n_frames,
                token_path=tpath,
                level1_path=lpath if lpath.exists() else None,
                camera=camera,
            )
        )
    return specs


def list_token_shot_ids(
    camera: str = DEFAULT_CAMERA,
    vocab_version: str = DEFAULT_VOCAB_VERSION,
    token_root: Path | None = None,
) -> list[int]:
    """Cheap scan: shot IDs that have a ``<camera>.zarr`` token store.

    Pure directory listing — no Zarr opens — so this is safe to run over
    the full 9,527-shot corpus.
    """
    root = token_root or TOKEN_ROOT
    # Wall 3 (live): a token root under TARGET_ROOT is refused at load time.
    frames_dir = assert_not_target_path(root / vocab_version / "frames")
    if not frames_dir.exists():
        return []
    out: list[int] = []
    for p in frames_dir.iterdir():
        if not (p.is_dir() and p.name.isdigit()):
            continue
        store = p / f"{camera}.zarr"
        if (store / "zarr.json").exists() or store.exists():
            out.append(int(p.name))
    return sorted(out)
