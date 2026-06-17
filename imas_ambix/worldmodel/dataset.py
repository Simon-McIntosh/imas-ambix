"""Multi-modal token-sequence assembly for the plan-conditioned world model.

This is §6 piece 1: take a shot's tokenised diagnostic streams (which each
live at their OWN native cadence and on their OWN time window), resample them
onto a single COMMON model time grid, attach the pulse-schedule conditioning
(the "plan" / programmed actuator waveforms), and split the grid into a
context window (given to the model) + a target window (the future the model
must predict).

The boundary guard is load-bearing
----------------------------------
Every store the input loader is about to open is routed through
:func:`imas_ambix.tokenizer.store_targets.assert_not_target_path`.  The
eval-only L2 reconstruction targets (psi, q, boundary, ...) live under
``TARGET_ROOT``, which is NOT a child of ``TOKEN_ROOT``; the guard hard-refuses
any path that resolves under ``TARGET_ROOT`` so a target can never be ingested
as an input even if a caller hands the dataset an explicit target path.  See
:mod:`imas_ambix.tokenizer.store_targets` (Wall 1 / Wall 3).

Per-group-local vocabularies
----------------------------
The substrate's locked decision is PER-GROUP-LOCAL token ids: each modality is
its own channel with its own embedding, disambiguated by group name (the
overlapping global ids are meaningless without the group).  This dataset keeps
each modality as a separate channel-group and records the per-group local-id
base so the model can build one embedding table per group (see
:mod:`imas_ambix.worldmodel.model`).  We therefore convert each store's
on-disk global ids back to LOCAL ids by subtracting the store's block start.

Common time grid
----------------
The modalities arrive at 4 kHz (L2 inputs), ~78 Hz (xma), ~600 Hz (camera
frames) etc.  We define a uniform model grid of ``n_steps`` samples over the
overlap of the requested modalities' windows, and for each grid step pick each
modality's NEAREST native token (within a half-step tolerance), recording a
per-step validity mask.  A grid step a modality has no token near is marked
invalid and filled with the modality's PAD local id (0) — never silently
treated as a real reading.

The dataset is lazy and torch-DataLoader-worker-safe: Zarr arrays are opened
on demand so the dataset object is cheap to pickle and share across workers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from imas_ambix.camdyn.dataset import (
    DEFAULT_CAMERA,
    frames_token_path,
    level1_shot_path,
)
from imas_ambix.camdyn.dataset import (
    DEFAULT_VOCAB_VERSION as FRAMES_VOCAB_VERSION,
)
from imas_ambix.data.paths import TOKEN_ROOT
from imas_ambix.tokenizer.store_targets import (
    SIGNALS_HF_GENERATION,
    assert_not_target_path,
)
from imas_ambix.tokenizer.store_v2 import signal_hf_token_path

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

# A control-token pad id local to every group (mirrors registry CONTROL_TOKENS
# "pad": 0).  A grid step with no native token for a modality is filled with
# this id and masked invalid — never a silent real-reading zero.
PAD_LOCAL_ID = 0


# ---------------------------------------------------------------------------
# Modality specification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModalitySpec:
    """One token modality the world model consumes as an input channel-group.

    A modality is a named group of token channels that shares a single
    embedding table and a single local-id vocabulary in the model.

    Attributes
    ----------
    name:
        Stable modality key (the model's embedding-table key).  E.g.
        ``"summary"``, ``"xma"``, ``"camera"``.
    kind:
        ``"signal_hf"`` for a native-cadence ``signals_hf/{shot}/{group}.zarr``
        store (this covers BOTH the L2 light-path ``*_l2`` groups and the
        xma/xim/xsx high-frequency groups), or ``"camera"`` for the camera
        frame token store under ``frames/{shot}/{camera}.zarr``.
    group:
        For ``kind="signal_hf"`` the on-disk store-group name (e.g.
        ``"summary_l2"``, ``"xma"``).  For ``kind="camera"`` the camera name
        (e.g. ``"rbb"``).
    vocab_size:
        Local vocabulary size for this modality's embedding table.  For a
        ``signal_hf`` group this is the quantiser bin count (256 for the L2
        light path); for the camera it is the LFQ codebook size.  Always sized
        so every local id (after rebasing) and the PAD id (0) fit.
    is_conditioning:
        When True this modality is the PLAN — the pulse-schedule programmed
        waveforms supplied as conditioning tokens, not a stream the model
        predicts.  Exactly the ``pulse_schedule_l2`` group.
    anchors_grid:
        When True this modality's native time window is part of the
        intersection that DEFINES the common model grid.  The reliable
        common-grid L2 light-path groups anchor; auxiliary modalities (xma,
        camera) that may carry a non-physical or disjoint time axis on some
        shots do NOT — they contribute tokens only where they overlap the
        anchored grid and are masked invalid elsewhere, so a single flaky
        auxiliary store never sinks the whole shot.
    n_channels:
        Number of token channels this modality contributes (filled at
        discovery from the store; 0 means "read from disk").
    camera_grid_stride:
        For ``kind="camera"`` only: spatial subsample stride over the 16x16
        token grid (keep every ``stride``-th token in each axis).  A stride of
        4 turns the 256-token frame into a 16-token frame — keeps the camera
        modality tractable for the prototype without changing the wiring.
    """

    name: str
    kind: str
    group: str
    vocab_size: int
    is_conditioning: bool = False
    anchors_grid: bool = True
    n_channels: int = 0
    camera_grid_stride: int = 4

    def __post_init__(self) -> None:
        if self.kind not in ("signal_hf", "camera"):
            raise ValueError(f"unknown modality kind {self.kind!r}")
        if self.vocab_size < 2:
            raise ValueError(f"{self.name}: vocab_size must be >= 2 (pad + data)")


def default_modalities(camera: str = DEFAULT_CAMERA) -> list[ModalitySpec]:
    """Return the prototype modality subset (§7: tractable, spans the kinds).

    The conditioning plan (``pulse_schedule``) plus a tractable measured-input
    subset (``summary``, ``pf_active``, ``interferometer``), the ``xma`` fast
    magnetics, and one camera.  Structured so adding a modality is a one-line
    append — e.g. ``ModalitySpec("xim", "signal_hf", "xim", 12806)`` to add the
    Dalpha/CII high-frequency stream.

    The L2 light-path groups share the 256-bin uniform-quantiser vocabulary
    (``L2_BLOCK_VOCAB``); ``xma``'s discrete codebook is degenerate (size 1 on
    disk — its payload lives in a continuous embedding, not the tokens), so it
    is included for structural completeness but contributes no discrete signal.
    """
    from imas_ambix.tokenizer.registry import L2_BLOCK_VOCAB

    # +1 so the PAD local id (0) and every quantiser bin id fit in the table.
    l2_vocab = L2_BLOCK_VOCAB + 1
    return [
        # The PLAN — programmed pulse-schedule demands (conditioning).
        ModalitySpec(
            "pulse_schedule",
            "signal_hf",
            "pulse_schedule_l2",
            l2_vocab,
            is_conditioning=True,
        ),
        # Measured L2 inputs (the observed scalars the model predicts forward).
        ModalitySpec("summary", "signal_hf", "summary_l2", l2_vocab),
        ModalitySpec("pf_active", "signal_hf", "pf_active_l2", l2_vocab),
        ModalitySpec("interferometer", "signal_hf", "interferometer_l2", l2_vocab),
        # High-frequency fast magnetics (xma).  Degenerate discrete codebook on
        # disk; kept so the wiring carries an HF modality end-to-end.  Does NOT
        # anchor the grid — some shots' xma stores carry a non-physical time
        # axis, which must not be allowed to sink the whole shot.
        ModalitySpec("xma", "signal_hf", "xma", 8, anchors_grid=False),
        # One camera (spatially subsampled to stay prototype-small).  Rides its
        # own frame-time axis, so it does not anchor the L2 grid either.
        ModalitySpec(
            "camera",
            "camera",
            camera,
            1 << 18,
            anchors_grid=False,
            camera_grid_stride=4,
        ),
    ]


# ---------------------------------------------------------------------------
# Window / grid configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WorldModelWindowConfig:
    """Common-grid + context/target split configuration.

    Attributes
    ----------
    n_steps:
        Total uniform model-grid steps over the modality-overlap window.
    context_steps:
        First ``context_steps`` grid steps are the given initial-condition
        context; the remaining ``n_steps - context_steps`` are the target
        window the model must forward-predict.
    grid_window:
        Optional explicit ``(t_start, t_end)`` model-grid window (s).  When
        ``None`` the window is the intersection of every requested modality's
        ``original_window`` for the shot (so every step has coverage).
    """

    n_steps: int = 64
    context_steps: int = 16
    grid_window: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if self.context_steps >= self.n_steps:
            raise ValueError(
                f"context_steps {self.context_steps} must be < n_steps {self.n_steps}"
            )
        if self.context_steps < 1:
            raise ValueError("context_steps must be >= 1")


# ---------------------------------------------------------------------------
# Per-shot assembled sample
# ---------------------------------------------------------------------------


@dataclass
class WorldModelSample:
    """One shot assembled on the common model grid.

    Attributes
    ----------
    shot_id:
        Source shot.
    grid_time:
        ``(n_steps,)`` float64 uniform model-grid times (s).
    tokens:
        ``{modality_name: (n_steps, n_channels) int64}`` PER-GROUP-LOCAL token
        ids (on-disk global ids rebased to local).  Includes the conditioning
        modality.
    valid:
        ``{modality_name: (n_steps, n_channels) bool}`` per-step-per-channel
        coverage mask (False where no native token fell near the grid step).
    channel_names:
        ``{modality_name: (channel names)}`` for introspection / decode.
    context_steps:
        Number of leading grid steps that are context (the rest are target).
    """

    shot_id: int
    grid_time: np.ndarray
    tokens: dict[str, np.ndarray]
    valid: dict[str, np.ndarray]
    channel_names: dict[str, tuple[str, ...]]
    context_steps: int

    @property
    def n_steps(self) -> int:
        return int(self.grid_time.shape[0])

    def as_dict(self) -> dict:
        """Return a plain dict (convenient for torch ``default_collate``)."""
        return {
            "shot_id": int(self.shot_id),
            "grid_time": self.grid_time,
            "tokens": self.tokens,
            "valid": self.valid,
            "context_steps": int(self.context_steps),
        }


# ---------------------------------------------------------------------------
# Lazy readers (worker-safe — open on demand) + the boundary guard
# ---------------------------------------------------------------------------


def _signal_hf_store_path(
    shot_id: int, group: str, *, token_root: Path | None = None
) -> Path:
    """Resolve a ``signals_hf`` store path and route it through the guard.

    The path is run through
    :func:`imas_ambix.tokenizer.store_targets.assert_not_target_path` so a
    ``token_root`` (or group) that resolves under ``TARGET_ROOT`` is hard-
    refused before any open — the eval-only reconstruction targets can never
    be admitted to the input stream.
    """
    if token_root is None:
        path = signal_hf_token_path(
            shot_id, group, store_generation=SIGNALS_HF_GENERATION
        )
    else:
        path = (
            Path(token_root)
            / SIGNALS_HF_GENERATION
            / "signals_hf"
            / str(shot_id)
            / f"{group}.zarr"
        )
    return assert_not_target_path(path)


def _read_signal_hf(
    shot_id: int, group: str, *, token_root: Path | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...], int]:
    """Read one ``signals_hf`` group: local tokens, time, valid, names, base.

    Returns ``(local_tokens, token_time, valid, channel_names, local_base)``
    where ``local_tokens = on_disk_global_id - local_base`` (the per-group
    local id), and ``local_base`` is the store's block start (the L2 light
    path records it in ``metadata.global_id_range``; the HF patch stores start
    at the control range, base 4).
    """
    import zarr  # noqa: PLC0415

    path = _signal_hf_store_path(shot_id, group, token_root=token_root)
    store = zarr.open_group(str(path), mode="r")
    tokens = np.asarray(store["tokens"], dtype=np.int64)
    token_time = np.asarray(store["token_time"], dtype=np.float64)
    valid = np.asarray(store["valid"], dtype=bool)
    attrs = dict(store.attrs)
    channel_names = tuple(str(c) for c in attrs.get("channel_names", ()))

    # The per-group local-id base.  The L2 light path records its absolute
    # block range in metadata.global_id_range; the HF patch stores begin at the
    # control range (CONTROL_RANGE[1] == 4).  Rebasing to local ids is what
    # keeps the per-group-local vocabulary contract (overlapping global ids are
    # meaningless without the group).
    local_base = _local_base_from_attrs(attrs)
    local_tokens = tokens - local_base
    # Any token that lands outside the local vocab (control/pad on disk) is
    # clamped to PAD and will be masked by ``valid`` anyway.
    local_tokens = np.where(local_tokens < 0, PAD_LOCAL_ID, local_tokens)
    return local_tokens, token_time, valid, channel_names, local_base


def _local_base_from_attrs(attrs: dict) -> int:
    """Recover a store's local-id base (block start) from its attrs."""
    import json  # noqa: PLC0415

    meta_raw = attrs.get("metadata", "{}")
    try:
        meta = json.loads(meta_raw) if isinstance(meta_raw, str) else dict(meta_raw)
    except (TypeError, ValueError):
        meta = {}
    rng = meta.get("global_id_range")
    if rng:
        return int(rng[0])
    # HF patch stores: every group's encode process restarts at the control
    # range, so the block starts at CONTROL_RANGE[1] == 4.
    from imas_ambix.tokenizer.registry import CONTROL_RANGE  # noqa: PLC0415

    return int(CONTROL_RANGE[1])


def _read_camera(
    shot_id: int,
    camera: str,
    stride: int,
    *,
    token_root: Path | None = None,
    level1_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    """Read one camera frame token stream, spatially subsampled.

    Returns ``(tokens, frame_time, valid, channel_names)`` where ``tokens`` is
    ``(n_frames, n_kept)`` int64 (the 16x16 grid kept at ``stride`` then
    flattened), ``frame_time`` the per-frame timestamps from the level-1 store
    (synthetic uniform fallback when absent), ``valid`` all-True, and
    ``channel_names`` ``("rNNcMM", ...)`` grid labels.

    The token path is built by
    :func:`imas_ambix.camdyn.dataset.frames_token_path`, which already routes
    through the target-boundary guard.
    """
    import zarr  # noqa: PLC0415

    path = frames_token_path(
        shot_id, camera, FRAMES_VOCAB_VERSION, token_root=token_root
    )
    # Belt-and-braces: re-assert the guard at this loader's chokepoint too.
    assert_not_target_path(path)
    store = zarr.open_group(str(path), mode="r")
    grid = np.asarray(store["tokens"], dtype=np.int64)  # (T, 16, 16)
    n_frames, h, w = grid.shape
    rsel = np.arange(0, h, stride)
    csel = np.arange(0, w, stride)
    sub = grid[:, rsel[:, None], csel[None, :]].reshape(n_frames, -1)
    names = tuple(f"r{r:02d}c{c:02d}" for r in rsel for c in csel)

    times = _read_camera_times(shot_id, camera, level1_dir=level1_dir)
    if times is None or times.shape[0] < n_frames:
        # synthetic uniform fallback at the reference rbb cadence
        times = np.arange(n_frames, dtype=np.float64) / 600.0
    else:
        times = times[:n_frames].astype(np.float64)
    valid = np.ones_like(sub, dtype=bool)
    return sub, times, valid, names


def _read_camera_times(
    shot_id: int, camera: str, *, level1_dir: Path | None = None
) -> np.ndarray | None:
    """Per-frame timestamps from the level-1 store, or None if unavailable."""
    import zarr  # noqa: PLC0415

    lpath = level1_shot_path(shot_id, level1_dir=level1_dir)
    if not Path(lpath).exists():
        return None
    try:
        store = zarr.open_group(str(lpath), mode="r")
        if camera not in set(store.group_keys()):
            return None
        grp = store[camera]
        if "time" not in set(grp.array_keys()):
            return None
        return np.asarray(grp["time"], dtype=np.float64)
    except Exception as exc:  # noqa: BLE001 — corpus robustness
        logger.debug("cannot read %s/%s/time: %r", lpath, camera, exc)
        return None


# ---------------------------------------------------------------------------
# Common-grid resampling
# ---------------------------------------------------------------------------


def _nearest_on_grid(
    native_time: np.ndarray,
    native_tokens: np.ndarray,
    native_valid: np.ndarray,
    grid_time: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Assign each grid step the nearest native token within a half-step.

    Returns ``(grid_tokens, grid_valid)`` of shape
    ``(n_steps, n_channels)``.  A grid step with no native sample within a
    half-step of the grid spacing is filled with PAD and marked invalid — a
    modality is never silently zero-filled.
    """
    n_steps = grid_time.shape[0]
    n_ch = native_tokens.shape[1] if native_tokens.ndim == 2 else 0
    grid_tokens = np.full((n_steps, n_ch), PAD_LOCAL_ID, dtype=np.int64)
    grid_valid = np.zeros((n_steps, n_ch), dtype=bool)
    if native_time.size == 0 or n_ch == 0:
        return grid_tokens, grid_valid

    order = np.argsort(native_time)
    nt = native_time[order]
    ntok = native_tokens[order]
    nval = native_valid[order]

    step = float(np.median(np.diff(grid_time))) if grid_time.size > 1 else 1.0
    tol = 0.5 * step if step > 0 else np.inf

    idx = np.searchsorted(nt, grid_time)
    idx = np.clip(idx, 0, nt.size - 1)
    # compare the candidate at idx and idx-1, pick the closer.
    left = np.clip(idx - 1, 0, nt.size - 1)
    pick_left = np.abs(nt[left] - grid_time) <= np.abs(nt[idx] - grid_time)
    chosen = np.where(pick_left, left, idx)
    dist = np.abs(nt[chosen] - grid_time)
    within = dist <= tol

    grid_tokens[within] = ntok[chosen[within]]
    grid_valid[within] = nval[chosen[within]]
    return grid_tokens, grid_valid


def build_shot_sample(
    shot_id: int,
    modalities: Sequence[ModalitySpec],
    config: WorldModelWindowConfig,
    *,
    token_root: Path | None = None,
    level1_dir: Path | None = None,
) -> WorldModelSample:
    """Assemble one shot's multi-modal token sample on the common grid.

    Reads every requested modality (each routed through the target-boundary
    guard), defines the common model grid over the modality-overlap window,
    resamples each modality onto it (nearest within a half-step), and records
    the per-step coverage mask + the context/target split.

    Raises ``ValueError`` if no requested modality is readable for the shot, or
    if the modalities' windows do not overlap.
    """
    reads: dict[str, tuple] = {}
    for m in modalities:
        try:
            if m.kind == "signal_hf":
                tok, t, val, names, _base = _read_signal_hf(
                    shot_id, m.group, token_root=token_root
                )
            else:  # camera
                tok, t, val, names = _read_camera(
                    shot_id,
                    m.group,
                    m.camera_grid_stride,
                    token_root=token_root,
                    level1_dir=level1_dir,
                )
        except (FileNotFoundError, KeyError) as exc:
            logger.info("shot %s modality %s unreadable: %r", shot_id, m.name, exc)
            continue
        if t.size == 0:
            continue
        reads[m.name] = (tok, t, val, names)

    if not reads:
        raise ValueError(f"shot {shot_id}: no requested modality readable")

    # Common grid window = intersection of the ANCHORING modalities' spans
    # (the reliable common-grid L2 light-path groups); auxiliary modalities
    # (xma, camera) ride their own axes and only contribute where they overlap.
    anchor_by_name = {m.name: m.anchors_grid for m in modalities}
    if config.grid_window is not None:
        t0, t1 = config.grid_window
    else:
        anchor_spans = [
            (float(t.min()), float(t.max()))
            for name, (_tok, t, _v, _n) in reads.items()
            if anchor_by_name.get(name, True)
        ]
        if not anchor_spans:
            # no anchor present — fall back to the intersection of all reads
            anchor_spans = [
                (float(t.min()), float(t.max())) for _tok, t, _v, _n in reads.values()
            ]
        t0 = max(lo for lo, _hi in anchor_spans)
        t1 = min(hi for _lo, hi in anchor_spans)
    if not (t1 > t0):
        raise ValueError(
            f"shot {shot_id}: anchoring modality windows do not overlap "
            f"(t0={t0}, t1={t1})"
        )
    grid_time = np.linspace(t0, t1, config.n_steps, dtype=np.float64)

    tokens: dict[str, np.ndarray] = {}
    valid: dict[str, np.ndarray] = {}
    channel_names: dict[str, tuple[str, ...]] = {}
    for m in modalities:
        if m.name not in reads:
            continue
        tok, t, val, names = reads[m.name]
        gt, gv = _nearest_on_grid(t, tok, val, grid_time)
        # clamp into the modality's local vocab so an out-of-range id can never
        # index past the embedding table (masked-invalid positions are PAD).
        gt = np.clip(gt, 0, m.vocab_size - 1)
        tokens[m.name] = gt
        valid[m.name] = gv
        channel_names[m.name] = names

    return WorldModelSample(
        shot_id=shot_id,
        grid_time=grid_time,
        tokens=tokens,
        valid=valid,
        channel_names=channel_names,
        context_steps=config.context_steps,
    )


# ---------------------------------------------------------------------------
# Corpus discovery
# ---------------------------------------------------------------------------


def discover_worldmodel_shots(
    modalities: Sequence[ModalitySpec],
    *,
    token_root: Path | None = None,
    shot_ids: Sequence[int] | None = None,
    limit: int | None = None,
    require_all: bool = True,
) -> list[int]:
    """Enumerate shots that carry the requested modalities on disk.

    A cheap directory scan: a shot qualifies when every ``require_all``
    modality's store exists (or, when ``require_all`` is False, at least one
    does).  ``shot_ids`` restricts the scan; ``limit`` caps the result.

    Every candidate path is routed through the target-boundary guard via the
    per-modality path builders, so the enumeration set can never include a
    target store.
    """
    root = Path(token_root) if token_root is not None else TOKEN_ROOT

    if shot_ids is not None:
        candidates = sorted(int(s) for s in shot_ids)
    else:
        sig_root = assert_not_target_path(root / SIGNALS_HF_GENERATION / "signals_hf")
        if not sig_root.exists():
            return []
        candidates = sorted(
            int(p.name) for p in sig_root.iterdir() if p.is_dir() and p.name.isdigit()
        )

    out: list[int] = []
    for sid in candidates:
        present = []
        for m in modalities:
            if m.kind == "signal_hf":
                p = _signal_hf_store_path(sid, m.group, token_root=root)
                present.append(p.exists())
            else:
                p = frames_token_path(
                    sid, m.group, FRAMES_VOCAB_VERSION, token_root=root
                )
                present.append((p / "zarr.json").exists() or p.exists())
        ok = all(present) if require_all else any(present)
        if ok:
            out.append(sid)
        if limit is not None and len(out) >= limit:
            break
    return out


# ---------------------------------------------------------------------------
# Torch Dataset
# ---------------------------------------------------------------------------


class WorldModelDataset:
    """Map-style torch Dataset of assembled multi-modal world-model samples.

    Each item is a :class:`WorldModelSample` (call ``.as_dict()`` for a
    collatable mapping).  Assembly is lazy in ``__getitem__`` so the dataset
    object is cheap to pickle across DataLoader workers.

    Parameters
    ----------
    shot_ids:
        Shots to serve (one sample per shot — the whole common grid).
    modalities:
        The modality channel-groups to assemble.
    config:
        Common-grid + context/target split configuration.
    """

    def __init__(
        self,
        shot_ids: Sequence[int],
        modalities: Sequence[ModalitySpec],
        config: WorldModelWindowConfig,
        *,
        token_root: Path | None = None,
        level1_dir: Path | None = None,
        as_dict: bool = False,
    ) -> None:
        self._shot_ids = [int(s) for s in shot_ids]
        self._modalities = list(modalities)
        self._config = config
        self._token_root = token_root
        self._level1_dir = level1_dir
        self._as_dict = as_dict

    def __len__(self) -> int:
        return len(self._shot_ids)

    def __getitem__(self, index: int):
        sid = self._shot_ids[index]
        sample = build_shot_sample(
            sid,
            self._modalities,
            self._config,
            token_root=self._token_root,
            level1_dir=self._level1_dir,
        )
        return sample.as_dict() if self._as_dict else sample

    @property
    def modalities(self) -> list[ModalitySpec]:
        return self._modalities

    @property
    def config(self) -> WorldModelWindowConfig:
        return self._config
