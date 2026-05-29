"""Multi-family state-space dataset wrapper.

Provides a minimal iterable dataset that opens FAIR-MAST level-1 Zarr
stores on demand and yields per-shot multi-family data dicts.

This is a TARGET-AGNOSTIC dataset: the caller specifies which groups are
inputs and which group is the held-out target.  The dataset returns both
so downstream code can compute predictions and evaluate calibration.

Usage
-----
    from imas_ambix.statespace.dataset import StatespaceDataset, DatasetConfig
    cfg = DatasetConfig(
        input_groups=["ama", "amb", "amc", "ane"],
        target_group="xim",
        target_channels=["da_hm10_t", "da_to10"],
        model_hz=100.0,
    )
    ds = StatespaceDataset(shot_ids=[30001, 30002, ...], config=cfg)
    for sample in ds:
        x = sample["inputs"]     # dict[group -> xr.Dataset]
        y = sample["target"]     # xr.Dataset (held-out family)
        sid = sample["shot_id"]  # int
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    import xarray as xr

from imas_ambix.data.paths import LEVEL1_DIR
from imas_ambix.statespace.align import MODEL_HZ_DEFAULT, align_family_dataset

logger = logging.getLogger(__name__)


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
        target (from the leakage audit).  These are loaded but marked
        as target / held-out, NOT available as inputs.
    model_hz:
        Model grid frequency for time alignment.
    level1_dir:
        Override for the level-1 Zarr root directory.
    control_groups:
        Groups for the exogenous control / conditioning stream (e.g.
        ``["xdc", "anb", "aga"]``).  These are available to later
        stages as conditioning inputs; they are NOT diagnostic inputs
        and NOT the prediction target.
    """

    input_groups: list[str] = field(default_factory=list)
    target_group: str = ""
    target_channels: list[str] = field(default_factory=list)
    additional_leaking_target_groups: list[str] = field(default_factory=list)
    model_hz: float = MODEL_HZ_DEFAULT
    level1_dir: Path | None = None
    control_groups: list[str] = field(default_factory=lambda: ["xdc", "anb", "aga"])


def _open_group_as_dataset(shot_zarr_path: Path, group: str) -> xr.Dataset | None:
    """Open one Zarr group as an xarray Dataset.

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

        # Collect channels and time axis
        time_arr: np.ndarray | None = None
        for time_name in ("time",):
            if time_name in grp:
                time_arr = np.asarray(grp[time_name])
                break

        data_vars: dict[str, tuple] = {}
        for key in grp:
            if key in ("time", "passnumber", "status", "svn_revision"):
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
            # Skip 2D+ arrays (e.g. camera frames, matrix data)

        if not data_vars:
            return None

        coords: dict[str, np.ndarray] = {}
        if time_arr is not None:
            coords["time"] = time_arr

        return xr.Dataset(data_vars, coords=coords)

    except Exception as e:
        logger.debug("Cannot open %s/%s: %s", shot_zarr_path.name, group, e)
        return None


class StatespaceDataset:
    """Iterable dataset yielding per-shot multi-family data dicts.

    Each item is a dict with keys:
        ``"shot_id"``  : int
        ``"inputs"``   : dict[str, xr.Dataset]  -- group → time-aligned dataset
        ``"target"``   : xr.Dataset | None       -- held-out family, time-aligned
        ``"control"``  : dict[str, xr.Dataset]   -- control/conditioning groups
        ``"missing_inputs"``  : list[str]         -- groups absent for this shot
        ``"missing_control"`` : list[str]

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

    def __len__(self) -> int:
        return len(self._shot_ids)

    def __iter__(self) -> Iterator[dict]:
        cfg = self._config
        for sid in self._shot_ids:
            shot_path = self._level1_dir / f"{sid}.zarr"
            if not shot_path.exists():
                logger.debug("Shot %d not found at %s — skipping", sid, shot_path)
                continue

            # Load target group
            target_ds: xr.Dataset | None = None
            if cfg.target_group:
                raw_target = _open_group_as_dataset(shot_path, cfg.target_group)
                if raw_target is not None:
                    # Filter to requested channels if specified
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

            # Load input groups
            inputs: dict[str, xr.Dataset] = {}
            missing_inputs: list[str] = []
            for grp in cfg.input_groups:
                ds = _open_group_as_dataset(shot_path, grp)
                if ds is None:
                    missing_inputs.append(grp)
                    continue
                inputs[grp] = align_family_dataset(ds, grp, cfg.model_hz)

            # Load control groups
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
                "inputs": inputs,
                "target": target_ds,
                "control": control,
                "missing_inputs": missing_inputs,
                "missing_control": missing_control,
            }
