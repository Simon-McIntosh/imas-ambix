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
    DEFAULT_VOCAB_VERSION as FRAMES_VOCAB_VERSION,
)
from imas_ambix.camdyn.dataset import (
    frames_token_path,
    level1_shot_path,
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

# The five MAST frame-token cameras (camera id -> on-disk frame-token store
# count).  They do NOT all co-occur on every shot — most shots carry only a
# subset — so every camera is an OPTIONAL modality: consumed where present,
# all-PAD + masked where absent.  Requiring all five would collapse the corpus
# to the rare all-camera intersection; requiring none of them (cameras optional
# vs the required core) keeps the corpus large (≈ the rbb store count, the
# largest single camera, when rbb is in the core — see ``default_modalities``).
CAMERA_IDS: tuple[str, ...] = ("rbb", "rba", "rco", "rgb", "rgc")


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
    required:
        When True the modality's on-disk store MUST be present for a shot to
        qualify in :func:`discover_worldmodel_shots`.  The REQUIRED core is
        DELIBERATELY MINIMAL — just the conditioning plan (``pulse_schedule``,
        essential because the model is plan-conditioned) and ``summary`` (the
        universal base observation) — so the corpus stays LARGE (every shot
        carrying the plan + base observation qualifies).  When False the
        modality is OPTIONAL: consumed when its store is present for a shot and
        emitted as an all-PAD, masked block when absent (via
        :func:`~imas_ambix.worldmodel.train.pad_collate_batch`) — so an OPTIONAL
        modality never excludes a shot from the corpus.  EVERYTHING beyond the
        core is optional: the other L2 groups, every HF stream (xma/xim/xsx),
        and every camera.  Stream presence varies per shot, so requiring more
        than the core would shrink the corpus to a rare all-streams
        intersection.
    """

    name: str
    kind: str
    group: str
    vocab_size: int
    is_conditioning: bool = False
    anchors_grid: bool = True
    n_channels: int = 0
    camera_grid_stride: int = 4
    required: bool = True

    def __post_init__(self) -> None:
        if self.kind not in ("signal_hf", "camera"):
            raise ValueError(f"unknown modality kind {self.kind!r}")
        if self.vocab_size < 2:
            raise ValueError(f"{self.name}: vocab_size must be >= 2 (pad + data)")

    def fixed_channel_width(self) -> int | None:
        """The modality's channel width, when it can be fixed WITHOUT a shot.

        Returns the per-step channel count this modality contributes to the
        fused step embedding, when that width is a structural constant rather
        than a per-shot property:

        * an explicitly declared ``n_channels`` (>0) on the spec wins — the
          caller pinned the width;
        * a ``camera`` modality's width is fully determined by the 16×16 LFQ
          frame grid sub-sampled at ``camera_grid_stride`` (e.g. stride 4 → a
          4×4 = 16-token frame), so it never needs a probe shot.

        Returns ``None`` for a ``signal_hf`` group with no declared width — its
        channel count (coil counts, chord counts, …) genuinely varies per shot
        and must be probed (see :func:`camera_channel_width`).
        """
        if self.n_channels > 0:
            return int(self.n_channels)
        if self.kind == "camera":
            return camera_channel_width(self.camera_grid_stride)
        return None


def camera_channel_width(stride: int) -> int:
    """Channel count a camera modality contributes at a given grid stride.

    The camera token grid is the fixed ``FRAME_GRID`` (16×16) sub-sampled by
    keeping every ``stride``-th token on each axis, then flattened — exactly
    what :func:`_read_camera` produces.  This is a structural constant (no shot
    needed), so the model's camera embedding/head channel width is always
    correct even when no probed shot carries the camera.
    """
    from imas_ambix.camdyn.dataset import FRAME_GRID  # noqa: PLC0415

    h, w = FRAME_GRID
    return int(len(range(0, h, stride)) * len(range(0, w, stride)))


def default_modalities(
    cameras: Sequence[str] = CAMERA_IDS,
) -> list[ModalitySpec]:
    """Return the FULL tokenised input substrate as world-model modalities.

    Every tokenised input stream confirmed on disk is emitted as a modality so
    the trainer uses the whole multi-modal substrate:

    * **conditioning plan** — ``pulse_schedule`` (the programmed pulse-schedule
      demand waveforms);
    * **measured L2 light path** — ``summary``, ``pf_active``,
      ``interferometer``, ``gas_injection``, ``soft_x_rays`` (the
      provenance-verified Level-2 *input* observables, 256-bin uniform-quantiser
      vocab);
    * **L1 high-frequency patch-transformer streams** — ``xma`` (fast magnetics
      Mirnov array, degenerate size-1 discrete codebook on disk — kept for
      structural completeness), ``xim`` (Dα/CII, codebook 12800), ``xsx``
      (soft-X-ray chord array, codebook 1024);
    * **cameras** — all five MAST frame-token cameras (``rbb``, ``rba``,
      ``rco``, ``rgb``, ``rgc``), each its own LFQ ``1 << 18`` codebook.

    REQUIRED core vs OPTIONAL substrate
    -----------------------------------
    Stream presence VARIES per shot, so only a tiny REQUIRED core gates the
    corpus: the conditioning ``pulse_schedule`` (the plan is essential — the
    model is plan-conditioned) and ``summary`` (a near-universal base
    observation).  EVERYTHING ELSE — the other L2 groups, every HF stream, and
    every camera — is ``required=False``: consumed where present, emitted as an
    all-PAD, masked block (no loss) where absent (via
    :func:`~imas_ambix.worldmodel.train.pad_collate_batch`).  This keeps the
    corpus LARGE — :func:`discover_worldmodel_shots` admits every shot carrying
    the plan + base observation, NOT the rare all-streams intersection.

    Grid anchoring
    --------------
    Only the reliable common-grid L2 light-path groups (the L2 ``signal_hf``
    modalities) anchor the model time grid.  The HF streams (xma/xim/xsx) and
    the cameras ride their own native time axes and are ``anchors_grid=False``,
    so a single flaky/disjoint auxiliary store can never sink a whole shot.

    The two degenerate cross-channel SIBLING tokens (``xma_mode`` /
    ``xsx_profile``) are size-1 placeholder blocks on disk — their payload lives
    in a continuous metadata embedding, not the discrete tokens — so they carry
    NO discrete signal and are intentionally NOT emitted as predictive
    modalities.

    Each camera gets its OWN :class:`ModalitySpec` (distinct ``name`` == camera
    id), so the model builds one embedding + one next-token head per camera and
    per-camera identity is preserved.  Five separate ``1 << 18`` camera tables +
    heads are ~1.0 B params at ``d_model=384`` (≈16 GB worst-case fp32 AdamW),
    well within ONE H200's 140 GB — so the cameras stay per-camera (no shared
    codebook needed); a shot missing a camera simply leaves that camera's block
    all-PAD + masked.
    """
    from imas_ambix.tokenizer.registry import L2_BLOCK_VOCAB

    # +1 so the PAD local id (0) and every quantiser bin id fit in the table.
    l2_vocab = L2_BLOCK_VOCAB + 1
    mods: list[ModalitySpec] = [
        # ── Conditioning plan (REQUIRED core) ───────────────────────────────
        # The PLAN — programmed pulse-schedule demands (conditioning).  The
        # model is plan-conditioned, so the plan is essential: it gates the
        # corpus.
        ModalitySpec(
            "pulse_schedule",
            "signal_hf",
            "pulse_schedule_l2",
            l2_vocab,
            is_conditioning=True,
            required=True,
        ),
        # ── Measured L2 inputs ──────────────────────────────────────────────
        # ``summary`` is the universal base observation and the only measured
        # modality in the REQUIRED core; the remaining L2 groups are OPTIONAL so
        # a shot lacking one of them is still admitted (used where present,
        # all-PAD + masked where absent).  All L2 groups anchor the grid.
        ModalitySpec("summary", "signal_hf", "summary_l2", l2_vocab, required=True),
        ModalitySpec(
            "pf_active", "signal_hf", "pf_active_l2", l2_vocab, required=False
        ),
        ModalitySpec(
            "interferometer",
            "signal_hf",
            "interferometer_l2",
            l2_vocab,
            required=False,
        ),
        ModalitySpec(
            "gas_injection",
            "signal_hf",
            "gas_injection_l2",
            l2_vocab,
            required=False,
        ),
        ModalitySpec(
            "soft_x_rays",
            "signal_hf",
            "soft_x_rays_l2",
            l2_vocab,
            required=False,
        ),
        # ── L1 high-frequency patch-transformer streams (OPTIONAL) ──────────
        # Native-cadence phase-aware patch codes.  Coverage varies per shot
        # (xim ~14183, xsx ~13002, xma ~12045 shots), so each is OPTIONAL and
        # off-grid (its own native time axis).  ``xma``'s discrete patch
        # codebook is degenerate (size 1 on disk — payload in a continuous
        # embedding); the small vocab=8 leaves room for PAD + control ids.
        # ``xim`` codebook 12800, ``xsx`` codebook 1024 (+ slack for PAD/control
        # so every rebased local id and PAD=0 fit the table).
        ModalitySpec("xma", "signal_hf", "xma", 8, anchors_grid=False, required=False),
        ModalitySpec(
            "xim", "signal_hf", "xim", 12806, anchors_grid=False, required=False
        ),
        ModalitySpec(
            "xsx", "signal_hf", "xsx", 1030, anchors_grid=False, required=False
        ),
    ]
    # ── Cameras (OPTIONAL) ──────────────────────────────────────────────────
    # All five cameras, each OPTIONAL (used when present, all-PAD + masked when
    # absent).  Each rides its own frame-time axis, so none anchors the L2 grid.
    # The modality name IS the camera id so the model keeps per-camera tables /
    # heads and the loader reads each camera's own frames (``m.group``).
    for cam in cameras:
        mods.append(
            ModalitySpec(
                cam,
                "camera",
                cam,
                1 << 18,
                anchors_grid=False,
                camera_grid_stride=4,
                required=False,
            )
        )
    return mods


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


def _modality_store_present(
    sid: int, m: ModalitySpec, root: Path | None
) -> bool:
    """True when modality ``m``'s on-disk store exists for shot ``sid``.

    Routed through the per-modality path builders so the boundary guard fires
    (a path under ``TARGET_ROOT`` is hard-refused before any existence probe).
    """
    if m.kind == "signal_hf":
        p = _signal_hf_store_path(sid, m.group, token_root=root)
        return p.exists()
    p = frames_token_path(sid, m.group, FRAMES_VOCAB_VERSION, token_root=root)
    return (p / "zarr.json").exists() or p.exists()


def discover_worldmodel_shots(
    modalities: Sequence[ModalitySpec],
    *,
    token_root: Path | None = None,
    shot_ids: Sequence[int] | None = None,
    limit: int | None = None,
    require_all: bool = True,
    sample: str = "camera_first",
    seed: int = 0,
) -> list[int]:
    """Enumerate shots that carry the REQUIRED core modalities on disk.

    A cheap directory scan keyed on the REQUIRED-vs-OPTIONAL split:

    * When ``require_all`` is True (the default), a shot qualifies when every
      modality with ``ModalitySpec.required`` True has its store present.
      Modalities with ``required=False`` (the non-core L2 groups, every HF
      stream, and the five cameras) are IGNORED by discovery — a shot is
      admitted whether or not its optional stores exist, and the optional
      modality is consumed where present / emitted all-PAD + masked where
      absent at collate time.  This is what keeps the corpus LARGE: requiring
      the whole substrate would shrink it to the rare all-streams intersection,
      but requiring only the minimal core (the conditioning plan + the universal
      ``summary`` base observation) admits every shot that carries them, with
      every other stream used opportunistically.
    * When ``require_all`` is False a shot qualifies when AT LEAST ONE of the
      passed modalities' stores exists (the permissive any-of scan — unchanged).

    Sampling (``sample`` + ``seed``) — why this matters
    ---------------------------------------------------
    Cameras live in HIGH shot-ids (rbb ≥ 15085, rco ≥ 19156, …).  A naive
    ascending scan with a small ``limit`` returns ONLY the lowest-id band,
    which carries ZERO cameras — the camera (and the partly-covered xma) heads
    then never see a single token.  To stop that, ALL qualifying shots are
    enumerated first and only THEN truncated to ``limit``, after a deterministic
    (seeded) resample so the kept corpus and the channel-sizing probe both SEE
    camera-bearing shots:

    * ``"camera_first"`` (default): qualifying shots that carry AT LEAST ONE of
      the passed cameras are moved to the FRONT (each band internally shuffled
      with ``seed``), so a small ``limit`` is camera-dense; the remaining
      core-only shots follow.  Camera presence is probed per shot (boundary-
      guarded) only over the camera specs, so the scan stays cheap.
    * ``"shuffle"``: a single seeded shuffle of all qualifying shots (no camera
      bias) — still removes the ascending-low-id pathology.
    * ``"ascending"``: the legacy strict ascending order (no resample) — kept
      for explicit callers that want determinism by id; NOT the default because
      it is the camera-free-band bug.

    ``shot_ids`` restricts the scan; ``limit`` caps the result (applied AFTER
    the resample so the truncation keeps camera-bearing shots).

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

    # In the default (require_all) mode the gate is the REQUIRED core only;
    # optional modalities (cameras) never gate discovery.  When no modality is
    # marked required (a caller passing only optional specs), fall back to
    # requiring every passed modality so the call still means "shots with these".
    required_mods = [m for m in modalities if m.required]
    gate_mods = required_mods if (require_all and required_mods) else list(modalities)

    # Enumerate EVERY qualifying shot first (do NOT break early at ``limit`` —
    # that is the ascending-low-id pathology).  ``limit`` is applied only after
    # the resample below.
    qualified: list[int] = []
    for sid in candidates:
        present = [_modality_store_present(sid, m, root) for m in gate_mods]
        ok = all(present) if require_all else any(present)
        if ok:
            qualified.append(sid)

    out = _sample_discovered_shots(
        qualified, modalities, root, sample=sample, seed=seed
    )
    if limit is not None:
        out = out[:limit]
    return out


def _sample_discovered_shots(
    qualified: Sequence[int],
    modalities: Sequence[ModalitySpec],
    root: Path | None,
    *,
    sample: str,
    seed: int,
) -> list[int]:
    """Resample qualifying shots so a small ``limit`` sees camera-bearing ones.

    See :func:`discover_worldmodel_shots` for the ``sample`` modes.  Returns the
    full reordered list; the caller applies ``limit`` afterwards.
    """
    import random  # noqa: PLC0415

    shots = list(qualified)
    if sample == "ascending":
        return shots
    rng = random.Random(seed)
    if sample == "shuffle":
        rng.shuffle(shots)
        return shots
    if sample != "camera_first":
        raise ValueError(
            f"unknown sample mode {sample!r} "
            "(want 'camera_first', 'shuffle', or 'ascending')"
        )
    # camera_first: split into camera-bearing vs core-only, shuffle each band
    # (seeded), and put the camera-bearing band first so a small limit is dense
    # in cameras.  If no camera modalities are declared, this degenerates to a
    # plain seeded shuffle.
    camera_mods = [m for m in modalities if m.kind == "camera"]
    if not camera_mods:
        rng.shuffle(shots)
        return shots
    with_cam: list[int] = []
    without_cam: list[int] = []
    for sid in shots:
        has_cam = any(_modality_store_present(sid, m, root) for m in camera_mods)
        (with_cam if has_cam else without_cam).append(sid)
    rng.shuffle(with_cam)
    rng.shuffle(without_cam)
    return with_cam + without_cam


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
