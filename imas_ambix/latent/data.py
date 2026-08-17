"""Real-data assembly for the GS-grounded latent engine (train + eval).

Turns MAST level-1 shots into the aligned per-slice arrays the engine needs:

* **input features** — the absolute-calibrated (corpus-level SI, NOT per-shot)
  feature vector (ama ⊕ amb ⊕ amc ⊕ ane), the encoder's input;
* **raw magnetics** — the ``amb`` flux-loop [Wb] / B-probe [T] channels aligned
  BY NAME to a campaign :class:`~imas_ambix.gs.operator.ForwardOperator`'s
  ``sensor_channels`` (the GS observation targets), with a per-sensor mask for
  channels the operator predicts but the shot does not carry, and with each
  channel referred to one acquisition range setting
  (:func:`divide_out_acquisition_scale`) so an amplitude means the same thing on
  every shot;
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

import json
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

#: Standing held-out cohort (per the eval harness) — never trained on.
STANDING_HELD_OUT = (18502, 18503, 18504, 18505)

#: Default train/test-OOD shot-list manifest (per-machine campaign geometry
#: build reads shots individually; this only supplies the split membership).
DEFAULT_SPLITS_MANIFEST = Path(
    "/work/projects/imas_gpu/mast/manifests/statespace_splits_dalpha_v0.json"
)

# Column layout is derived from the schema at run time (schema_group_offsets)
# — never hard-coded: the per-group channel lists are alphabetically sorted, so
# literal indices silently drift when a channel is added or renamed.


def align_sensor_columns(
    sensor_channels: list[str], amb_names: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Map operator sensor rows ↔ amb feature columns BY CHANNEL NAME.

    Returns ``(op_rows, x_cols)`` — parallel int arrays such that operator
    sensor row ``op_rows[k]`` is measured by amb-group column ``x_cols[k]``
    (add the amb group's offset from :func:`schema_group_offsets` for the
    absolute feature column).  Only operator sensors whose channel name appears
    in ``amb_names`` are matched; the rest are unmeasured (masked out of the GS
    residual for this campaign).
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
    with warnings.catch_warnings():
        # all-NaN columns (channels dead across the whole corpus) are handled
        # below — silence the empty-slice warning they trigger
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = np.nanmean(stacked, axis=0)
        std = np.nanstd(stacked, axis=0)
    # a column with no finite samples carries no information: normalise it to
    # exactly zero (mean-code) rather than propagating NaN into the encoder
    mean = np.where(np.isfinite(mean), mean, 0.0)
    std = np.where(np.isfinite(std) & (std > 0), std, 1.0)
    return CorpusStats(mean=mean, std=std)


@dataclass
class ShotWindows:
    """Per-plasma-on-slice aligned arrays for one shot on one campaign operator."""

    shot_id: int
    campaign: str
    features_raw: np.ndarray  # (T, n_feat) raw SI input features
    raw_mag: np.ndarray  # (T, S) magnetics on operator rows (NaN if unmeasured),
    # with each channel's acquisition range setting divided out
    mag_mask: np.ndarray  # (T, S) bool — operator sensor is measured this shot
    i_pf: np.ndarray  # (T, C) KNOWN PF-coil currents [A]
    anchored: np.ndarray  # (T, n_anchored) raw scalars [Ip(kA), n_e(m^-2)]
    times: np.ndarray  # (T,) seconds (shared 1 kHz grid)
    ref_target: np.ndarray | None = None  # (T, 14) firewalled EFIT geometry (eval)
    ref_mask: np.ndarray | None = None  # (T, 14) bool
    #: One warrant per operator sensor channel for the setting ``raw_mag`` was
    #: divided by — what the read did and what justified it.
    scale_corrections: tuple = ()


# --- anchored raw scalars (resolved BY NAME from the schema, never by index:
# the amc channel list is alphabetically sorted, so a hard-coded index silently
# reads the wrong channel — a fixed defect read tf_current as Ip) ---
ANCHORED_NAMES = ("ip", "n_e")


def schema_group_offsets(feature_schema: dict[str, list[str]]) -> dict[str, int]:
    """Column offset of each group in the concatenated feature vector."""
    offsets: dict[str, int] = {}
    col = 0
    for group, channels in feature_schema.items():
        offsets[group] = col
        col += len(channels)
    return offsets


def anchored_columns(feature_schema: dict[str, list[str]]) -> tuple[int, int]:
    """(ip_col, ne_col) — plasma_current + density columns, resolved by name.

    Raises ``KeyError`` (fail-loud) if either channel is absent from the schema.
    """
    offsets = schema_group_offsets(feature_schema)
    try:
        ip_col = offsets["amc"] + feature_schema["amc"].index("plasma_current")
    except (KeyError, ValueError) as exc:
        raise KeyError("feature schema has no amc/plasma_current channel") from exc
    try:
        ne_col = offsets["ane"] + feature_schema["ane"].index("density")
    except (KeyError, ValueError) as exc:
        raise KeyError("feature schema has no ane/density channel") from exc
    return ip_col, ne_col


def load_shot_slices_raw(
    shot_id: int,
    feature_schema: dict[str, list[str]],
    *,
    level1_dir=None,
    model_hz: float = 1000.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Target-free, imputation-free feature loader: ``(X_raw, times, plasma_on)``.

    Unlike :func:`imas_ambix.statespace.baseline.load_shot_slices` (built for
    the Dα statespace work) this loader

    * requires NO target channel — the statespace loader drops every slice
      where its Dα target is NaN, which collapsed a ~0.5 s shot to a ~26 ms
      sliver (<5% of the plasma);
    * preserves NaN — the statespace loader imputes missing features with the
      column mean BEFORE returning, which made downstream finiteness masks
      treat imputed sensor values as real measurements in the GS residual.

    ``X_raw`` is ``(T, F)`` raw SI with NaN where a channel is absent/invalid;
    masks must be derived from THIS array, before any fill.
    """
    import zarr  # noqa: PLC0415

    from imas_ambix.statespace.baseline import (  # noqa: PLC0415
        _LEVEL1_DIR,
        _build_common_time_grid,
        _read_group_channels,
    )

    shot_path = (level1_dir or _LEVEL1_DIR) / f"{shot_id}.zarr"
    if not shot_path.exists():
        return None
    try:
        store = zarr.open_group(str(shot_path), mode="r")
    except Exception:  # noqa: BLE001
        return None
    times = _build_common_time_grid(store, model_hz=model_hz)
    if times is None:
        return None

    parts: list[np.ndarray] = []
    for group, channels in feature_schema.items():
        mat = _read_group_channels(store, group, channels, times)
        if mat is None:
            mat = np.full((times.size, len(channels)), np.nan)
        parts.append(mat)
    x = np.concatenate(parts, axis=1)

    plasma_on = np.zeros(times.size, dtype=bool)
    if "amc" in store and "plasma_current" in store["amc"]:
        grp = store["amc"]
        ip = np.asarray(grp["plasma_current"], dtype=np.float64)
        ip_t = np.asarray(grp["time"], dtype=np.float64) if "time" in grp else None
        if ip_t is not None and ip.shape == ip_t.shape:
            ip_on_grid = np.interp(times, ip_t, np.abs(ip))
            peak = float(np.nanmax(ip_on_grid))
            if peak > 50.0:  # kA
                plasma_on = ip_on_grid > max(50.0, 0.2 * peak)
    return x, times, plasma_on


def divide_out_acquisition_scale(
    values: np.ndarray,
    sensor_channels: list[str],
    shot: int,
    *,
    table=None,
) -> tuple[np.ndarray, tuple]:
    """Refer every channel's amplitude to one acquisition range setting.

    Nineteen MAST probe channels were not recorded at a single setting: each sits
    at one range for a run of shots, steps by a rung of a binary ladder, holds,
    and steps back.  A channel like that means a different number of tesla per
    stored unit depending on which shot is being read, so a forward model that
    predicts the field correctly still misses the recorded value by the rung —
    and a misfit metric charges that arithmetic to the description.  Dividing the
    rung out here refers every shot to the same setting, which is what makes a
    residual over a mixed set of shots a statement about the machine.

    ``table`` supplies the settings; the promoted table nova carries is the
    default, and an empty :class:`~nova.imas.mast_block_scale.BlockScaleTable`
    reads the archive exactly as published.  A channel is divided only where a
    measurement warrants it: a block whose step is not a ladder rung is refused
    rather than rounded onto one, and a shot falling in the gap between two
    blocks is left alone because the switch could be on either side of it.  So an
    unchanged column is not evidence of a unit setting — the returned warrants
    are, one per channel, and they are what a consumer records.

    Only the measurement is corrected.  The encoder's feature block carries the
    same channels at the amplitudes its corpus statistics and checkpoints were
    fitted on, so moving those is a retraining decision rather than a read one.
    """
    if table is None:
        from nova.imas.mast_block_scale import (  # noqa: PLC0415
            promoted_block_scales,
        )

        table = promoted_block_scales()
    corrected = np.array(values, dtype=np.float64)
    warrants = table.corrections(int(shot), sensor_channels)
    by_channel = {row.channel: row for row in warrants}
    for column, channel in enumerate(sensor_channels):
        warrant = by_channel.get(channel)
        if warrant is not None and warrant.applied:
            corrected[:, column] = warrant.normalise(corrected[:, column])
    return corrected, tuple(by_channel[channel] for channel in sensor_channels)


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

    Reads level-1 via :func:`load_shot_slices_raw` (target-free, NaN-preserving
    — see its docstring for why the statespace loader is unsuitable), aligns
    the amb magnetics to ``operator.sensor_channels`` by name, assembles
    ``i_pf`` from the amc block, extracts the anchored scalars by CHANNEL NAME,
    and (eval only, inside the firewall) reads the EFIT referee geometry on the
    same ``times``.  Returns None if the shot has no usable plasma-on slices.

    ``mag_mask`` is derived from the pre-fill raw array, so a sensor absent on
    this shot is honestly masked out of the GS residual (never imputed).

    ``raw_mag`` comes back with each channel's acquisition range setting divided
    out and one warrant per channel in ``scale_corrections``; a channel with no
    measured setting is unchanged and says so through its warrant.
    """
    del target_channels  # kept in the signature for call-site compatibility
    loaded = load_shot_slices_raw(
        shot_id, feature_schema, level1_dir=level1_dir, model_hz=model_hz
    )
    if loaded is None:
        return None
    x, times, plasma_on = loaded
    if plasma_on is None or not np.any(plasma_on):
        return None
    x = np.asarray(x, dtype=np.float64)[plasma_on]
    times = np.asarray(times, dtype=np.float64)[plasma_on]
    n = x.shape[0]
    if n == 0:
        return None

    amb_names = feature_schema["amb"]
    amc_names = feature_schema["amc"]
    offsets = schema_group_offsets(feature_schema)
    op_rows, x_cols = align_sensor_columns(operator.sensor_channels, amb_names)
    n_sensor = len(operator.sensor_channels)

    raw_mag = np.full((n, n_sensor), np.nan, dtype=np.float64)
    mag_mask = np.zeros((n, n_sensor), dtype=bool)
    if op_rows.size:
        raw_mag[:, op_rows] = x[:, offsets["amb"] + x_cols]
        mag_mask[:, op_rows] = np.isfinite(raw_mag[:, op_rows])
    raw_mag, scale_corrections = divide_out_acquisition_scale(
        raw_mag, list(operator.sensor_channels), int(shot_id)
    )

    # i_pf per slice via the operator's amc assembly (kA·turn → A inside);
    # a NaN coil current contributes zero (assemble_pf_currents skips missing).
    n_coil = len(operator.pf_amc_channels)
    i_pf = np.zeros((n, n_coil), dtype=np.float64)
    amc_block = x[:, offsets["amc"] : offsets["amc"] + len(amc_names)]
    for t in range(n):
        amc_values = {
            ch: float(amc_block[t, j])
            for j, ch in enumerate(amc_names)
            if np.isfinite(amc_block[t, j])
        }
        i_pf[t] = operator.assemble_pf_currents(amc_values)

    ip_col, ne_col = anchored_columns(feature_schema)
    anchored = np.column_stack([x[:, ip_col], x[:, ne_col]]).astype(np.float64)

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
        scale_corrections=scale_corrections,
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


# --- corpus-level assembly (shared by the training + gate-eval drivers) ----


def feature_schema() -> dict[str, list[str]]:
    """The MAG+ANE absolute-calibrated feature schema (ama⊕amb⊕amc⊕ane)."""
    from imas_ambix.statespace.baseline import _FEATURE_SCHEMA_MAG_ANE  # noqa: PLC0415

    return _FEATURE_SCHEMA_MAG_ANE


def read_split_shot_lists(
    n_train: int, n_heldout: int, *, manifest: Path | None = None
) -> tuple[list[int], list[int]]:
    """Train + held-out shot lists (the standing cohort is forced into held-out)."""
    with open(manifest or DEFAULT_SPLITS_MANIFEST) as f:
        splits = json.load(f)
    train = [int(x) for x in splits.get("train", [])]
    test = [int(x) for x in splits.get("test_ood_regime", [])]
    held = list(STANDING_HELD_OUT) + [s for s in test if s not in STANDING_HELD_OUT]
    train = [s for s in train if s not in set(held)]
    return train[:n_train], held[:n_heldout]


def build_campaign_operators(
    shots: list[int], *, grid_nr: int, grid_nz: int, profile_order: int
) -> tuple[dict, dict, dict[int, str]]:
    """Build one :class:`GSObservation` + limiter per campaign signature.

    Returns ``(gs_by_campaign, limiter_by_campaign, campaign_of)`` where
    ``campaign_of`` maps each shot whose geometry table built successfully to
    its signature key; shots without a buildable table are simply absent.
    """
    from imas_ambix.data.description_reader import (  # noqa: PLC0415
        read_geometry_table,
    )
    from imas_ambix.latent.gs_observation import GSObservation  # noqa: PLC0415

    gs_by_campaign: dict = {}
    limiter_by_campaign: dict = {}
    campaign_of: dict[int, str] = {}
    for s in shots:
        try:
            table = read_geometry_table(int(s))
        except Exception as exc:  # noqa: BLE001 — unavailable description → skip
            logger.warning("shot %d: no geometry table (%s)", s, exc)
            continue
        key = table.signature.key
        campaign_of[int(s)] = key
        if key not in gs_by_campaign:
            gs_by_campaign[key] = GSObservation.from_table(
                table, grid_nr=grid_nr, grid_nz=grid_nz, profile_order=profile_order
            )
            limiter_by_campaign[key] = (
                np.asarray(table.limiter_r, dtype=np.float64),
                np.asarray(table.limiter_z, dtype=np.float64),
            )
    return gs_by_campaign, limiter_by_campaign, campaign_of


def assemble_shot_windows(
    shots: list[int],
    campaign_of: dict[int, str],
    schema: dict[str, list[str]],
    *,
    with_referee: bool,
) -> list[ShotWindows]:
    """:func:`load_shot_windows` for every shot that has a campaign operator."""
    from imas_ambix.data.description_reader import (  # noqa: PLC0415
        read_geometry_table,
    )
    from imas_ambix.gs.operator import build_operator  # noqa: PLC0415

    out: list[ShotWindows] = []
    for s in shots:
        key = campaign_of.get(int(s))
        if key is None:
            continue
        try:
            operator = build_operator(read_geometry_table(int(s)))
        except Exception as exc:  # noqa: BLE001
            logger.warning("shot %d: operator build failed (%s)", s, exc)
            continue
        w = load_shot_windows(int(s), operator, key, schema, with_referee=with_referee)
        if w is not None:
            out.append(w)
    return out


def sensor_scale_for_campaign(
    windows: list[ShotWindows], campaign: str, n_sensor: int
) -> np.ndarray:
    """Per-sensor whitening scale = std of measured raw magnetics over slices."""
    cols = [w.raw_mag for w in windows if w.campaign == campaign]
    if not cols:
        return np.ones(n_sensor)
    stacked = np.concatenate(cols, axis=0)
    scale = np.nanstd(stacked, axis=0)
    return np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)


#: kind-relative scale floor (locked convention): a channel's whitening scale
#: may never be floored below this fraction of its own KIND's (b-probe vs flux
#: loop) per-shot median scale — see :func:`robust_channel_scale`.
CHANNEL_SCALE_KIND_FLOOR_REL = 0.05


def _channel_kind(name: str) -> str:
    """amb naming convention: ``fl_*`` is a flux loop, everything else here is
    a B-probe (the only two amb sensor kinds this whitening scale covers)."""
    return "flux_loop" if str(name).lower().startswith("fl") else "b_probe"


def robust_channel_scale(
    scale: np.ndarray,
    channels: list[str],
    *,
    rel_floor: float = CHANNEL_SCALE_KIND_FLOOR_REL,
) -> np.ndarray:
    """Floor a per-channel whitening ``scale`` at ``rel_floor`` x its KIND's
    per-shot median scale, computed locally from ``scale`` itself.

    A channel's own natural variability over one shot can be near-zero (e.g. a
    flux loop that barely moves for that particular shot) purely as a data
    characteristic, not a fault — but dividing a modest vacuum/measured
    mismatch by that near-zero scale in the amortised-encoder / variational-
    inverse whitened misfit amplifies it by orders of magnitude (observed: a
    single such channel drove the whitened misfit to ~3.7e6 on an otherwise
    unremarkable example).  Flooring at a fraction of the SAME shot's SAME-KIND
    channels bounds this (no channel can claim more than ``1/rel_floor`` times
    its kind's typical whitened weight) while requiring no corpus-wide
    statistics — computable identically wherever ``scale`` is consumed, so the
    assembly path (``scripts/train_patch_encoder.py``) and the gate/inverse
    path (``scripts/patch_gate_eval.py``) apply the IDENTICAL convention from
    one shared implementation.  Falls back to the pre-existing absolute ``1.0``
    floor when a whole kind is itself degenerate (median non-finite or ``<=0``
    — e.g. no b-probes at all in ``channels``).

    ``scale`` may be ``(S,)`` (one shot) or ``(N, S)`` (N examples, vectorised
    — each row floored against its OWN row's kind medians, e.g. for a cached
    corpus's per-example scale array where every row is a different shot).
    ``channels`` is constant across rows (the basis/operator's fixed sensor
    order) — kind grouping never depends on which row is being floored.
    """
    arr = np.asarray(scale, dtype=np.float64)
    one_d = arr.ndim == 1
    s2 = arr[None, :] if one_d else arr
    kinds = np.array([_channel_kind(c) for c in channels])
    out = np.array(s2, dtype=np.float64)
    for kind in np.unique(kinds):
        sel = kinds == kind
        block = s2[:, sel]
        finite_pos = np.isfinite(block) & (block > 0)
        masked = np.where(finite_pos, block, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN rows
            medians = np.nanmedian(masked, axis=1)
        medians = np.where(np.isfinite(medians), medians, 0.0)
        floor = rel_floor * medians
        safe = np.where(finite_pos, block, 0.0)
        floored = np.where(medians[:, None] > 0, np.maximum(safe, floor[:, None]), 1.0)
        out[:, sel] = floored
    return out[0] if one_d else out


__all__ = [
    "align_sensor_columns",
    "robust_channel_scale",
    "CHANNEL_SCALE_KIND_FLOOR_REL",
    "CorpusStats",
    "fit_corpus_stats",
    "ShotWindows",
    "ANCHORED_NAMES",
    "load_shot_windows",
    "load_shot_slices_raw",
    "anchored_columns",
    "schema_group_offsets",
    "STANDING_HELD_OUT",
    "DEFAULT_SPLITS_MANIFEST",
    "feature_schema",
    "read_split_shot_lists",
    "build_campaign_operators",
    "assemble_shot_windows",
    "sensor_scale_for_campaign",
]
