"""Multi-family state-space dataset wrapper — multi-modal edition (D3, S9).

Provides a minimal iterable dataset that opens FAIR-MAST level-1 Zarr
stores on demand and yields per-shot multi-family data dicts.

This is a TARGET-AGNOSTIC dataset: the caller specifies which groups are
inputs and which group is the held-out target.  The dataset returns both
so downstream code can compute predictions and evaluate calibration.

Multi-modal additions (D3, S9)
-------------------------------
The loader now handles four input kinds in addition to the original
``signal_1d`` path:

``signal_1d``   — 1-D time-series signals (the original path; kept intact).
                  Used for magnetics (ama, amb, amc, ane, …).

``chord_2d``    — 2-D chord arrays (T, C) where channels are spatial
                  chords.  Used for:
                    * ``xsx`` — SXR cameras (3 × 18 chords → 54 ch per
                      time point, after transposing from the on-disk
                      (C, T) layout and concatenating hcam_l/hcam_u/tcam).
                    * ``abm`` — bolometer (32 ch radiated-power ``i-bol``).
                  Groups are absent-filled with zeros (T_grid, C) plus a
                  boolean ``present`` flag.

``camera``      — frame arrays (T, H, W) from the visible camera ``rbb``.
                  Frames are mean-pool downsampled to ``frame_hw`` (default
                  64 × 64) via block-average pooling.  Native 544 × 640
                  pixels: H is cropped to the largest multiple of 8 (512),
                  W is exactly divisible by 10 — both are then pooled to
                  64 × 64.
                  Time-aligned via nearest-frame assignment (not linear
                  interpolation — blending frames has no physical meaning).
                  A ``raw_rate`` flag on the spec skips grid alignment and
                  returns frames at the native camera rate.
                  Absent-filled with zeros (T_grid, 64, 64) + present flag.

``thomson``     — Thomson scattering as a MASKED channel.  Reuses
                  :func:`~imas_ambix.statespace.integrated_inputs.load_thomson_stream`
                  which forward-fills the (T_grid, 14) feature vector at
                  the native Thomson cadence.  The per-time freshness scalar
                  (last column) carries the missing/stale signal — the
                  model never sees an interpolated-dense profile.
                  Absent-filled with zeros (T_grid, 14) + present flag.

One shared time grid per shot
-------------------------------
All modalities (including absent ones) are aligned to a SINGLE shot-level
model grid built from the intersection of all PRESENT modalities' time
windows via :func:`~imas_ambix.tokenizer.alignment.shot_time_window`.
This guarantees that absent-group zero arrays have the same T as the
present ones — the missingness flag is meaningful.

Usage
-----
    from imas_ambix.statespace.dataset import (
        DatasetConfig, ModalitySpec, StatespaceDataset
    )

    # Minimal (signal_1d only — backward-compatible):
    cfg = DatasetConfig(
        input_groups=["ama", "amb", "amc", "ane"],
        target_group="xim",
        model_hz=100.0,
    )

    # Multi-modal:
    cfg = DatasetConfig(
        input_groups=["amc", "rbb", "xsx", "abm", "ayc"],
        target_group="",
        model_hz=100.0,
        modality_spec={
            "amc":  ModalitySpec(kind="signal_1d"),
            "rbb":  ModalitySpec(kind="camera",    frame_hw=(64, 64)),
            "xsx":  ModalitySpec(kind="chord_2d",  n_channels=54),
            "abm":  ModalitySpec(kind="chord_2d",  n_channels=32),
            "ayc":  ModalitySpec(kind="thomson"),
        },
    )
    ds = StatespaceDataset(shot_ids=[23142, 23143, 23144], config=cfg)
    for sample in ds:
        inputs     = sample["inputs"]        # dict[group -> ndarray | Dataset]
        present    = sample["present_flags"] # dict[group -> bool]
        shot_grid  = sample["shot_grid"]     # np.ndarray (T,) model grid (s)
        sid        = sample["shot_id"]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    import numpy as np
    import xarray as xr

from imas_ambix.data.paths import LEVEL1_DIR
from imas_ambix.statespace.align import (
    MODEL_HZ_DEFAULT,
    align_camera_to_grid,
    align_chord2d_to_grid,
    align_family_dataset,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Groups that are NEVER loaded as inputs (leakage / scope)
# ---------------------------------------------------------------------------

#: Groups excluded unconditionally from the input stream (held-out targets,
#: equilibrium reconstruction outputs, and derived Dα channels).
EXCLUDED_INPUT_GROUPS: frozenset[str] = frozenset(
    {"efm", "esm", "ada", "aim", "xim", "ams"}
)

#: Channels within ``xdc`` that must be dropped (shape flux-error series).
XDC_EXCLUDED_CHANNEL_PREFIX: str = "shape_s_fluxerr"


# ---------------------------------------------------------------------------
# Modality specification
# ---------------------------------------------------------------------------


@dataclass
class ModalitySpec:
    """Specification for one input group's modality kind and missingness policy.

    Attributes
    ----------
    kind:
        One of ``"signal_1d"``, ``"chord_2d"``, ``"camera"``, ``"thomson"``.
    n_channels:
        For ``chord_2d`` only: expected number of channels C in (T, C).
        Used to synthesize the absent-group zero array.
        * ``xsx`` → 54  (hcam_l + hcam_u + tcam, each 18 chords)
        * ``abm`` → 32  (``i-bol`` channels)
    frame_hw:
        For ``camera`` only: (height, width) of the output spatial grid
        after pooling.  Default ``(64, 64)``.
    raw_rate:
        For ``camera`` only: if True, skip model-grid alignment and return
        frames at the native camera cadence.  The caller is responsible for
        handling the variable-length time axis.
    missingness:
        How to represent an absent group.
        ``"zero"``   — return a zero-filled array of the right shape (default).
        ``"nan"``    — return a NaN-filled array.
    """

    kind: str = "signal_1d"
    n_channels: int = 0  # chord_2d
    frame_hw: tuple[int, int] = (64, 64)  # camera
    raw_rate: bool = False  # camera
    missingness: str = "zero"  # "zero" | "nan"


# ---------------------------------------------------------------------------
# DatasetConfig
# ---------------------------------------------------------------------------


@dataclass
class DatasetConfig:
    """Configuration for :class:`StatespaceDataset`.

    Attributes
    ----------
    input_groups:
        Zarr group names to load as model inputs (never includes target
        groups or excluded groups).
    target_group:
        Primary Zarr group of the held-out family (e.g. ``"xim"`` for Dα).
    target_channels:
        Channel names within *target_group* to use as the target signal.
        If empty, all channels are used.
    additional_leaking_target_groups:
        Additional groups that carry the same physical quantity as the
        target (from the leakage audit).  Loaded but held-out.
    model_hz:
        Model grid frequency for time alignment.
    level1_dir:
        Override for the level-1 Zarr root directory.
    control_groups:
        Groups for the exogenous control / conditioning stream (e.g.
        ``["xdc", "anb", "aga"]``).  Available to later stages as
        conditioning inputs; NOT diagnostic inputs, NOT prediction target.
    modality_spec:
        Optional per-group :class:`ModalitySpec`.  Any group not listed
        here defaults to ``ModalitySpec(kind="signal_1d")``.  This enables
        backward-compatible usage where only 1-D groups appear.
    """

    input_groups: list[str] = field(default_factory=list)
    target_group: str = ""
    target_channels: list[str] = field(default_factory=list)
    additional_leaking_target_groups: list[str] = field(default_factory=list)
    model_hz: float = MODEL_HZ_DEFAULT
    level1_dir: Path | None = None
    control_groups: list[str] = field(default_factory=lambda: ["xdc", "anb", "aga"])
    modality_spec: dict[str, ModalitySpec] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal per-group loaders
# ---------------------------------------------------------------------------


def _open_group_as_dataset(shot_zarr_path: Path, group: str) -> xr.Dataset | None:
    """Open one Zarr group as an xarray Dataset (1-D signals only).

    Returns None on any error (missing group, etc.).
    """
    import numpy as np  # noqa: PLC0415
    import xarray as xr  # noqa: PLC0415
    import zarr  # noqa: PLC0415

    grp_path = shot_zarr_path / group
    if not grp_path.exists():
        return None

    try:
        store = zarr.open_group(str(shot_zarr_path), mode="r")
        grp = store[group]

        # Collect time axis
        time_arr: np.ndarray | None = None
        for time_name in ("time",):
            if time_name in grp:
                time_arr = np.asarray(grp[time_name])
                break

        data_vars: dict[str, tuple] = {}
        for key in grp:
            if key in ("time", "passnumber", "status", "svn_revision"):
                continue
            # Skip xdc shape_s_fluxerr* channels (leakage guard)
            if group == "xdc" and key.startswith(XDC_EXCLUDED_CHANNEL_PREFIX):
                continue
            try:
                arr = np.asarray(grp[key])
            except Exception:
                continue
            if (
                arr.ndim == 1
                and time_arr is not None
                and arr.shape[0] == time_arr.shape[0]
            ):
                data_vars[key] = (("time",), arr)
            elif arr.ndim == 0:
                data_vars[key] = ((), arr)
            # 2-D+ arrays handled by modality-specific loaders — skip here

        if not data_vars:
            return None

        coords: dict[str, np.ndarray] = {}
        if time_arr is not None:
            coords["time"] = time_arr

        return xr.Dataset(data_vars, coords=coords)

    except Exception as e:
        logger.debug("Cannot open %s/%s: %s", shot_zarr_path.name, group, e)
        return None


def _open_group_time(shot_zarr_path: Path, group: str) -> np.ndarray | None:
    """Return the time array for a Zarr group, or None if absent."""
    import numpy as np  # noqa: PLC0415
    import zarr  # noqa: PLC0415

    grp_path = shot_zarr_path / group
    if not grp_path.exists():
        return None
    try:
        store = zarr.open_group(str(shot_zarr_path), mode="r")
        grp = store[group]
        if "time" not in grp:
            return None
        return np.asarray(grp["time"], dtype=np.float64)
    except Exception:
        return None


def _load_chord2d(
    shot_zarr_path: Path,
    group: str,
    spec: ModalitySpec,
    grid_times: np.ndarray,
) -> tuple[np.ndarray, bool]:
    """Load a chord-2D group aligned to ``grid_times``.

    Returns ``(array (T_grid, C), present_flag)``.

    For ``xsx``:  concatenates hcam_l, hcam_u, tcam (each (18, T_xsx))
    transposing to (T_xsx, 18) → (T_xsx, 54).

    For ``abm``:  uses ``i-bol`` (T_abm, 32) directly.

    Any other group: tries to find a (T, C) 2-D array where T matches
    the time axis length.
    """
    import numpy as np  # noqa: PLC0415
    import zarr  # noqa: PLC0415

    T_grid = len(grid_times)  # noqa: N806
    n_ch = spec.n_channels
    fill = 0.0 if spec.missingness == "zero" else float("nan")

    grp_path = shot_zarr_path / group
    if not grp_path.exists():
        return np.full((T_grid, n_ch), fill, dtype=np.float32), False

    try:
        store = zarr.open_group(str(shot_zarr_path), mode="r")
        grp = store[group]
        t_native = np.asarray(grp["time"], dtype=np.float64)

        if group == "xsx":
            # On-disk layout: each camera is (18_chords, T_time).
            # Transpose to (T, 18), then concat across 3 cameras → (T, 54).
            cams = []
            for cam_key in ("hcam_l", "hcam_u", "tcam"):
                if cam_key not in grp:
                    cams.append(np.full((len(t_native), 18), fill, dtype=np.float32))
                    continue
                cam = np.asarray(grp[cam_key], dtype=np.float32)  # (18, T)
                if cam.shape[1] == len(t_native):
                    cam = cam.T  # → (T, 18)
                elif cam.shape[0] == len(t_native):
                    pass  # already (T, 18)
                else:
                    cams.append(np.full((len(t_native), 18), fill, dtype=np.float32))
                    continue
                cams.append(cam)
            arr_native = np.concatenate(cams, axis=1)  # (T_xsx, 54)

        elif group == "abm":
            # On-disk layout: i-bol is (T_abm, 32).
            ibol = np.asarray(grp["i-bol"], dtype=np.float32)  # (T, 32)
            if ibol.shape[0] != len(t_native) and ibol.shape[1] == len(t_native):
                # Transpose if needed (defensive)
                ibol = ibol.T
            arr_native = ibol  # (T_abm, 32)

        else:
            # Generic: find first 2-D array whose one axis == len(t_native)
            arr_native = None
            for key in grp:
                if key in ("time", "passnumber", "status"):
                    continue
                try:
                    a = np.asarray(grp[key], dtype=np.float32)
                except Exception:
                    continue
                if a.ndim == 2:
                    if a.shape[0] == len(t_native):
                        arr_native = a
                        break
                    if a.shape[1] == len(t_native):
                        arr_native = a.T
                        break
            if arr_native is None:
                return np.full((T_grid, n_ch), fill, dtype=np.float32), False

        aligned = align_chord2d_to_grid(arr_native, t_native, grid_times)

        # Pad / crop to declared n_channels
        if n_ch > 0 and aligned.shape[1] != n_ch:
            out = np.full((T_grid, n_ch), fill, dtype=np.float32)
            nc = min(aligned.shape[1], n_ch)
            out[:, :nc] = aligned[:, :nc]
            return out, True

        return aligned.astype(np.float32), True

    except Exception as e:
        logger.debug("chord2d load failed for %s/%s: %s", shot_zarr_path.name, group, e)
        return np.full((T_grid, max(n_ch, 1)), fill, dtype=np.float32), False


def _load_camera(
    shot_zarr_path: Path,
    group: str,
    spec: ModalitySpec,
    grid_times: np.ndarray,
) -> tuple[np.ndarray, bool]:
    """Load a visible-camera group and downsample to ``spec.frame_hw``.

    Returns ``(array (T, H_out, W_out), present_flag)`` where T is either
    ``len(grid_times)`` (aligned) or the native frame count (raw_rate).

    Downsampling: block-average pool from (H_src, W_src) to ``frame_hw``.
    H_src is cropped to the nearest multiple of block_h; W_src to block_w.
    ``uint8`` frames are converted to ``float32`` before pooling.
    """
    import numpy as np  # noqa: PLC0415
    import zarr  # noqa: PLC0415

    T_grid = len(grid_times)  # noqa: N806
    H_out, W_out = spec.frame_hw  # noqa: N806
    fill = 0.0 if spec.missingness == "zero" else float("nan")

    grp_path = shot_zarr_path / group
    if not grp_path.exists():
        if spec.raw_rate:
            return np.full((0, H_out, W_out), fill, dtype=np.float32), False
        return np.full((T_grid, H_out, W_out), fill, dtype=np.float32), False

    try:
        store = zarr.open_group(str(shot_zarr_path), mode="r")
        grp = store[group]

        # On-disk: data (T_cam, H, W), time (T_cam,)
        frames_raw = np.asarray(grp["data"], dtype=np.float32)  # uint8→float32
        t_cam = np.asarray(grp["time"], dtype=np.float64)

        if frames_raw.ndim != 3:
            raise ValueError(f"rbb data expected 3-D, got {frames_raw.ndim}")

        T_cam, H_src, W_src = frames_raw.shape  # noqa: N806

        # Block-average downsample to H_out × W_out
        bh = H_src // H_out  # block height (e.g. 544//64 = 8)
        bw = W_src // W_out  # block width  (e.g. 640//64 = 10)
        H_crop = bh * H_out  # noqa: N806  # largest H divisible by block (512)
        W_crop = bw * W_out  # noqa: N806  # largest W divisible by block (640)
        cropped = frames_raw[:, :H_crop, :W_crop]  # (T_cam, H_crop, W_crop)
        # Reshape to expose block dimensions, then mean-pool
        pooled = cropped.reshape(T_cam, H_out, bh, W_out, bw).mean(
            axis=(2, 4)
        )  # (T_cam, H_out, W_out)

        if spec.raw_rate:
            return pooled, True

        aligned = align_camera_to_grid(pooled, t_cam, grid_times)
        return aligned, True

    except Exception as e:
        logger.debug("camera load failed for %s/%s: %s", shot_zarr_path.name, group, e)
        if spec.raw_rate:
            return np.full((0, H_out, W_out), fill, dtype=np.float32), False
        return np.full((T_grid, H_out, W_out), fill, dtype=np.float32), False


def _load_thomson(
    shot_zarr_path: Path,
    spec: ModalitySpec,
    grid_times: np.ndarray,
) -> tuple[np.ndarray, bool]:
    """Load Thomson scattering as a masked (T_grid, 14) feature stream.

    Delegates to :func:`~imas_ambix.statespace.integrated_inputs.load_thomson_stream`.
    Returns ``(features (T_grid, N_THOMSON_FEATURES), present_flag)`` where
    ``present_flag`` is True iff at least one of ayc/atm/aye was found with
    usable pe profiles.  The freshness column (last feature) acts as a
    per-time missingness signal.
    """
    import numpy as np  # noqa: PLC0415
    import zarr  # noqa: PLC0415

    from imas_ambix.statespace.integrated_inputs import (  # noqa: PLC0415
        N_THOMSON_FEATURES,
        load_thomson_stream,
    )

    T_grid = len(grid_times)  # noqa: N806
    fill = 0.0 if spec.missingness == "zero" else float("nan")

    try:
        store = zarr.open_group(str(shot_zarr_path), mode="r")
        ts = load_thomson_stream(store, grid_times)
        present = ts.system != "none" and ts.n_measurements > 0
        return ts.features.astype(np.float32), present
    except Exception as e:
        logger.debug("thomson load failed for %s: %s", shot_zarr_path.name, e)
        return np.full((T_grid, N_THOMSON_FEATURES), fill, dtype=np.float32), False


# ---------------------------------------------------------------------------
# Shot-level grid builder
# ---------------------------------------------------------------------------


def _build_shot_grid(
    shot_zarr_path: Path,
    all_groups: list[str],
    model_hz: float,
) -> np.ndarray | None:
    """Build the shared model grid from the intersection of all present groups' time.

    Gathers time arrays from every present group, computes the intersection
    window via :func:`~imas_ambix.tokenizer.alignment.shot_time_window`, and
    returns a uniform grid at ``model_hz`` Hz.  Returns None when fewer than 2
    groups supply a time array (can't define an intersection).
    """

    from imas_ambix.tokenizer.alignment import (  # noqa: PLC0415
        TimeGrid,
        shot_time_window,
    )

    time_arrays: list[np.ndarray] = []
    for grp in all_groups:
        t = _open_group_time(shot_zarr_path, grp)
        if t is not None and t.size >= 2:
            time_arrays.append(t)

    if not time_arrays:
        return None

    try:
        t_start, t_end = shot_time_window(*time_arrays)
    except ValueError:
        return None

    if t_end <= t_start:
        return None

    grid = TimeGrid(t_start=t_start, t_end=t_end, hz=model_hz)
    arr = grid.as_array()
    if arr.size < 2:
        return None
    return arr


# ---------------------------------------------------------------------------
# StatespaceDataset
# ---------------------------------------------------------------------------


class StatespaceDataset:
    """Iterable dataset yielding per-shot multi-family data dicts.

    Each item is a dict with keys:

    ``"shot_id"``
        int — shot identifier.
    ``"shot_grid"``
        np.ndarray (T,) — the shared model-grid time axis (s).
    ``"inputs"``
        dict[group -> ndarray | xr.Dataset]

        * ``signal_1d`` groups → xr.Dataset (aligned to model grid).
        * ``chord_2d`` groups  → np.ndarray (T, C).
        * ``camera`` groups    → np.ndarray (T, H, W) or (T_cam, H, W) if raw_rate.
        * ``thomson`` groups   → np.ndarray (T, 14).

    ``"present_flags"``
        dict[group -> bool] — True iff the group was found and loaded
        successfully; False for absent / failed groups.
    ``"target"``
        xr.Dataset | None — held-out family, time-aligned.
    ``"control"``
        dict[str, xr.Dataset] — control/conditioning groups.
    ``"missing_inputs"``
        list[str] — groups absent for this shot (complement of present_flags).
    ``"missing_control"``
        list[str] — control groups absent for this shot.

    Parameters
    ----------
    shot_ids:
        List of shot IDs to yield.
    config:
        :class:`DatasetConfig` describing what to load.
    skip_missing_target:
        If True (default), skip shots where the target group is absent.
    """

    def __init__(
        self,
        shot_ids: list[int],
        config: DatasetConfig,
        skip_missing_target: bool = True,
    ) -> None:
        self._shot_ids = list(shot_ids)
        self._config = config
        self._skip_missing_target = skip_missing_target
        self._level1_dir = config.level1_dir or LEVEL1_DIR

        # Validate: reject excluded groups in input_groups
        bad = EXCLUDED_INPUT_GROUPS & set(config.input_groups)
        if bad:
            raise ValueError(
                f"input_groups contains excluded/held-out groups: {sorted(bad)}.  "
                "Remove them from input_groups."
            )

    def _spec(self, group: str) -> ModalitySpec:
        return self._config.modality_spec.get(group, ModalitySpec(kind="signal_1d"))

    def __len__(self) -> int:
        return len(self._shot_ids)

    def __iter__(self) -> Iterator[dict]:

        cfg = self._config

        for sid in self._shot_ids:
            shot_path = self._level1_dir / f"{sid}.zarr"
            if not shot_path.exists():
                logger.debug("Shot %d not found at %s — skipping", sid, shot_path)
                continue

            # Build the shared shot-level model grid from all relevant groups.
            # Include inputs, target, and control so the grid covers the full
            # plasma window.
            all_groups = list(cfg.input_groups)
            if cfg.target_group:
                all_groups.append(cfg.target_group)
            all_groups.extend(cfg.control_groups)
            # De-duplicate while preserving order
            seen: set[str] = set()
            unique_groups: list[str] = []
            for g in all_groups:
                if g not in seen:
                    seen.add(g)
                    unique_groups.append(g)

            grid_times = _build_shot_grid(shot_path, unique_groups, cfg.model_hz)
            if grid_times is None:
                logger.debug(
                    "Shot %d: cannot build model grid (no usable time arrays)"
                    " — skipping",
                    sid,
                )
                continue

            # ---- Load target group ----
            target_ds: xr.Dataset | None = None
            if cfg.target_group:
                raw_target = _open_group_as_dataset(shot_path, cfg.target_group)
                if raw_target is not None:
                    if cfg.target_channels:
                        keep = [
                            c for c in cfg.target_channels if c in raw_target.data_vars
                        ]
                        raw_target = raw_target[keep] if keep else None
                    if raw_target is not None:
                        target_ds = align_family_dataset(
                            raw_target, "dalpha", cfg.model_hz
                        )

            if self._skip_missing_target and cfg.target_group and target_ds is None:
                logger.debug(
                    "Shot %d: target group '%s' absent — skipping",
                    sid,
                    cfg.target_group,
                )
                continue

            # ---- Load input groups (multi-modal) ----
            inputs: dict[str, object] = {}
            present_flags: dict[str, bool] = {}
            missing_inputs: list[str] = []

            for grp in cfg.input_groups:
                spec = self._spec(grp)

                if spec.kind == "camera":
                    arr, present = _load_camera(shot_path, grp, spec, grid_times)
                    inputs[grp] = arr
                    present_flags[grp] = present

                elif spec.kind == "chord_2d":
                    arr, present = _load_chord2d(shot_path, grp, spec, grid_times)
                    inputs[grp] = arr
                    present_flags[grp] = present

                elif spec.kind == "thomson":
                    arr, present = _load_thomson(shot_path, spec, grid_times)
                    inputs[grp] = arr
                    present_flags[grp] = present

                else:
                    # signal_1d (default / backward-compat)
                    ds = _open_group_as_dataset(shot_path, grp)
                    if ds is None:
                        present_flags[grp] = False
                        missing_inputs.append(grp)
                        continue
                    inputs[grp] = align_family_dataset(ds, grp, cfg.model_hz)
                    present_flags[grp] = True

                if not present_flags[grp]:
                    missing_inputs.append(grp)

            # ---- Load control groups (signal_1d, with xdc channel filter) ----
            control: dict[str, xr.Dataset] = {}
            missing_control: list[str] = []
            for grp in cfg.control_groups:
                ds = _open_group_as_dataset(shot_path, grp)
                if ds is None:
                    missing_control.append(grp)
                    continue
                control[grp] = align_family_dataset(ds, grp, cfg.model_hz)

            yield {
                "shot_id": sid,
                "shot_grid": grid_times,
                "inputs": inputs,
                "present_flags": present_flags,
                "target": target_ds,
                "control": control,
                "missing_inputs": missing_inputs,
                "missing_control": missing_control,
            }
