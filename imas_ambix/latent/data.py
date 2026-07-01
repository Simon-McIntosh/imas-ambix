"""Real-data assembly for the GS-grounded latent engine (train + eval).

Turns MAST level-1 shots into the aligned per-slice arrays the engine needs:

* **input features** — the absolute-calibrated (corpus-level SI, NOT per-shot)
  feature vector (ama ⊕ amb ⊕ amc ⊕ ane), the encoder's input;
* **raw magnetics** — the ``amb`` flux-loop [Wb] / B-probe [T] channels aligned
  BY NAME to a campaign :class:`~imas_ambix.gs.operator.ForwardOperator`'s
  ``sensor_channels`` (the GS observation targets), with a per-sensor mask for
  channels the operator predicts but the shot does not carry;
* **known PF currents ``i_pf``** — the ``amc`` coil channels assembled to
  amperes via :meth:`ForwardOperator.assemble_pf_currents`;
* **anchored raw scalars** — Ip (Rogowski) + line-averaged density n_e;
* **firewalled referee target** — the EFIT axis / X-point / LCFS geometry, read
  ONLY inside :func:`imas_ambix.eval.efit_referee.evaluator_context` and aligned
  by the shared 1 kHz ``times`` grid (evaluation only, never a training input).

The sensor-alignment helper :func:`align_sensor_columns` is pure and offline-
testable; the loaders touch the ``/work`` mirror and run on the compute node.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# X column layout of the MAG+ANE feature schema (baseline._FEATURE_SCHEMA_MAG_ANE):
# [ ama(6) | amb(73) | amc(42) | ane(1) ] = 122.
_AMA_OFFSET = 0
_AMB_OFFSET = 6
_AMC_OFFSET = 79
_ANE_OFFSET = 121


def align_sensor_columns(
    sensor_channels: list[str], amb_names: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Map operator sensor rows ↔ amb feature columns BY CHANNEL NAME.

    Returns ``(op_rows, x_cols)`` — parallel int arrays such that operator
    sensor row ``op_rows[k]`` is measured by feature column
    ``_AMB_OFFSET + x_cols[k]`` (i.e. absolute X column ``6 + x_cols[k]``).  Only
    operator sensors whose channel name appears in ``amb_names`` are matched;
    the rest are unmeasured (masked out of the GS residual for this campaign).
    """
    idx = {name: j for j, name in enumerate(amb_names)}
    op_rows: list[int] = []
    x_cols: list[int] = []
    for r, ch in enumerate(sensor_channels):
        if ch in idx:
            op_rows.append(r)
            x_cols.append(idx[ch])
    return np.array(op_rows, dtype=np.int64), np.array(x_cols, dtype=np.int64)


@dataclass
class CorpusStats:
    """Corpus-level (SI) per-feature mean/std — the absolute calibration."""

    mean: np.ndarray  # (n_feat,)
    std: np.ndarray  # (n_feat,)

    def normalise(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / np.clip(self.std, 1e-9, None)


def fit_corpus_stats(x_list: list[np.ndarray]) -> CorpusStats:
    """Fit ONE corpus-level mean/std across all (shot, slice) rows.

    Corpus-level (not per-shot) is the absolute-calibration requirement: the
    same physical value maps to the same normalised code in every shot.
    """
    stacked = np.concatenate([np.asarray(x) for x in x_list], axis=0)
    mean = np.nanmean(stacked, axis=0)
    std = np.nanstd(stacked, axis=0)
    return CorpusStats(mean=mean, std=std)


@dataclass
class ShotWindows:
    """Per-plasma-on-slice aligned arrays for one shot on one campaign operator."""

    shot_id: int
    campaign: str
    features_raw: np.ndarray  # (T, n_feat) raw SI input features
    raw_mag: np.ndarray  # (T, S) raw magnetics on operator rows (NaN if unmeasured)
    mag_mask: np.ndarray  # (T, S) bool — operator sensor is measured this shot
    i_pf: np.ndarray  # (T, C) KNOWN PF-coil currents [A]
    anchored: np.ndarray  # (T, n_anchored) raw scalars [Ip(kA), n_e(m^-2)]
    times: np.ndarray  # (T,) seconds (shared 1 kHz grid)
    ref_target: np.ndarray | None = None  # (T, 14) firewalled EFIT geometry (eval)
    ref_mask: np.ndarray | None = None  # (T, 14) bool


# --- anchored-scalar channel indices in X (MAG+ANE schema) ---
ANCHORED_IP_COL = 120  # amc/plasma_current (Rogowski) [kA]
ANCHORED_NE_COL = 121  # ane/density [m^-2]
ANCHORED_NAMES = ("ip", "n_e")


def load_shot_windows(
    shot_id: int,
    operator,  # ForwardOperator for the shot's campaign
    campaign: str,
    feature_schema: dict[str, list[str]],
    *,
    level1_dir=None,
    level2_root=None,
    model_hz: float = 1000.0,
    with_referee: bool = False,
    target_channels: list[str] | None = None,
) -> ShotWindows | None:
    """Assemble one shot's aligned per-slice arrays (plasma-on slices only).

    Reads level-1 via :func:`imas_ambix.statespace.baseline.load_shot_slices`,
    aligns the amb magnetics to ``operator.sensor_channels`` by name, assembles
    ``i_pf`` from the amc block, extracts the anchored scalars, and (eval only,
    inside the firewall) reads the EFIT referee geometry on the same ``times``.
    Returns None if the shot has no usable plasma-on slices.
    """
    from imas_ambix.statespace.baseline import load_shot_slices  # noqa: PLC0415

    tgt = target_channels or ["da_hm10_t"]
    loaded = load_shot_slices(
        shot_id,
        feature_schema,
        tgt,
        model_hz=model_hz,
        **({} if level1_dir is None else {"level1_dir": level1_dir}),
    )
    if loaded is None:
        return None
    x, _y, times, plasma_on = loaded
    if plasma_on is None or not np.any(plasma_on):
        return None
    x = np.asarray(x, dtype=np.float64)[plasma_on]
    times = np.asarray(times, dtype=np.float64)[plasma_on]
    n = x.shape[0]
    if n == 0:
        return None

    amb_names = feature_schema["amb"]
    amc_names = feature_schema["amc"]
    op_rows, x_cols = align_sensor_columns(operator.sensor_channels, amb_names)
    n_sensor = len(operator.sensor_channels)

    raw_mag = np.full((n, n_sensor), np.nan, dtype=np.float64)
    mag_mask = np.zeros((n, n_sensor), dtype=bool)
    if op_rows.size:
        raw_mag[:, op_rows] = x[:, _AMB_OFFSET + x_cols]
        mag_mask[:, op_rows] = True

    # i_pf per slice via the operator's amc assembly (kA·turn → A inside)
    n_coil = len(operator.pf_amc_channels)
    i_pf = np.zeros((n, n_coil), dtype=np.float64)
    amc_block = x[:, _AMC_OFFSET : _AMC_OFFSET + len(amc_names)]
    for t in range(n):
        amc_values = {ch: float(amc_block[t, j]) for j, ch in enumerate(amc_names)}
        i_pf[t] = operator.assemble_pf_currents(amc_values)

    anchored = np.column_stack([x[:, ANCHORED_IP_COL], x[:, ANCHORED_NE_COL]]).astype(
        np.float64
    )

    ref_target = ref_mask = None
    if with_referee:
        ref_target, ref_mask = _load_referee(shot_id, times, level2_root)

    return ShotWindows(
        shot_id=int(shot_id),
        campaign=str(campaign),
        features_raw=x,
        raw_mag=raw_mag,
        mag_mask=mag_mask,
        i_pf=i_pf,
        anchored=anchored,
        times=times,
        ref_target=ref_target,
        ref_mask=ref_mask,
    )


def _load_referee(shot_id: int, times: np.ndarray, level2_root):
    """Read the firewalled EFIT geometry on ``times`` (evaluator-only)."""
    from imas_ambix.eval.efit_referee import evaluator_context  # noqa: PLC0415
    from imas_ambix.worldmodel.equilibrium_labels import (  # noqa: PLC0415
        load_equilibrium_geometry,
    )

    try:
        with evaluator_context():
            geom = load_equilibrium_geometry(
                shot_id,
                times,
                **({} if level2_root is None else {"level2_root": level2_root}),
            )
        return np.asarray(geom.target, dtype=np.float64), np.asarray(geom.finite_mask)
    except Exception as exc:  # noqa: BLE001 — a shot w/o equilibrium → no referee
        logger.warning("shot %d: referee unavailable (%s)", shot_id, exc)
        return None, None


__all__ = [
    "align_sensor_columns",
    "CorpusStats",
    "fit_corpus_stats",
    "ShotWindows",
    "ANCHORED_NAMES",
    "load_shot_windows",
]
