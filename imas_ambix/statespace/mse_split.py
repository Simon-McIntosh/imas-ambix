"""D1 — held-out-MSE split builder for the S9 MSE-free current-recovery gate.

This module builds the *locked* cross-shot + cross-family split that makes the
MSE eval scorable, and persists a per-shot manifest that the eval harness
(:mod:`imas_ambix.statespace.mse_eval`) and downstream predictors
(D2 EnKF baseline, D4 neural filter) read.

Partition taxonomy
------------------
TRAIN
    The multi-modal **Tier-1 union** corpus (magnetics + camera + bolometer +
    soft-X-ray), MSE-**independent**.  MSE (the ``ams`` group) is *never* an
    input or a label here.  Disjoint from CAL ∪ HELD-OUT by construction.

CAL  (conformal calibration)
    beam-on ``ams`` shots OUTSIDE the locked v0 OOD box — in-distribution
    relative to the held-out hole, used for split-conformal coverage
    calibration.

HELD-OUT  (OOD test)
    beam-on ``ams`` shots INSIDE the locked v0 OOD box (high-Iₚ × high-density
    corner).  This is the scored set; target N ≈ 128.

The MSE eval truth lives in the level-1 ``ams`` group and is *beam-state
dependent*:

* ``beam_ok == 0`` shots carry ONLY a static geometry/calibration table
  (no ``time``/``pitcha``/``gamma``) → **EXCLUDED** from CAL/HELD-OUT.
* ``beam_ok == 1`` shots carry the full MSE measurement on a ~2 kHz ``time``
  axis: ``pitcha`` (magnetic pitch, PRIMARY clean truth), ``gamma``
  (polarisation angle), and the derived ``q0_kappa1.85_4pt`` / ``rax_4pt``
  (SECONDARY, noisy).

Channel mapping (verified empirically)
--------------------------------------
``ch`` and ``rpos`` are *compacted*: their finite entries occupy the first K
positions and give the active channel ids and major radii in order.  The
finite *columns* of ``pitcha`` / ``gamma`` are *sparse* in the 1898-wide
padded array, but there are exactly K of them and they appear in the same
radial order.  The k-th finite ``pitcha`` column therefore corresponds to the
k-th compacted ``ch``/``rpos`` entry (positional pairing — NOT an arithmetic
column-offset formula, which is not constant).

Disjointness / zero-MSE guarantee
----------------------------------
TRAIN shot-ids are disjoint from CAL ∪ HELD-OUT (all beam-on ams shots are
removed from TRAIN), and the TRAIN input-channel spec contains no ``ams`` /
MSE reference.  Both are asserted in the smoke test.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — manifest / gate defaults
# ---------------------------------------------------------------------------

MSE_GROUP = "ams"
MODEL_GRID_HZ = 1000  # the model state grid; 2 kHz MSE decimated → 1 kHz
HELDOUT_N_TARGET = 128
HELDOUT_N_FLOOR = 80  # below this, widen the OOD hole and document

# Tier-1 union families (MSE-INDEPENDENT).  Sourced from families.FAMILY_GROUPS.
TIER1_MAGNETICS = ("ama", "amb", "amc", "amh", "amm", "asm")
TIER1_CAMERA = ("rba", "rbb", "rbc", "rco", "rgb", "rgc", "rca", "rir", "rit")
TIER1_BOLOMETER = ("abm",)
TIER1_SXR = ("xsx",)

# PRE-REGISTERED physical gates (from the orchestrator's measured scan).
Q0_MIN, Q0_MAX = 0.5, 3.0
RAX_MIN, RAX_MAX = 0.7, 1.1
# error-cut: |q0_err| < ERR_FRAC * |q0|  (and likewise for rax)
ERR_FRAC = 0.5
# pitch is physical in [-pi/2, pi/2]; require >= this many finite channels
PITCH_MIN_FINITE_CH = 4
# a slice counts as "pitch valid" if at least this many active channels finite
PITCH_VALID_MIN_CH = 6

# The LOCKED v0 OOD box (joint_p84, by-current-density) — IDENTICAL across the
# dalpha and integrated_core v0 split artifacts.  Reused verbatim so the
# held-out hole matches all prior S8/S9 work.  Units: Iₚ in kA, ne in 1e19 m⁻².
LOCKED_OOD_BOX_V0 = {
    "ip_min_kA": 666.9210327148437,
    "ip_max_kA": 986.0563916015625,
    "ne_min_1e19": 13.40141355440582,
    "ne_max_1e19": 27.403331717508483,
    "description": "High-Iₚ (>667 kA) × high-density (>13.40×10¹⁹ m⁻²) corner",
}


# ---------------------------------------------------------------------------
# Robust Zarr ams reader (handles the mixed V2/V3 consolidated-metadata corpus)
# ---------------------------------------------------------------------------


@dataclass
class AmsShot:
    """Decoded beam-on MSE measurement for one shot (1 kHz model grid).

    All arrays are plain numpy.  ``pitch`` and ``pitch_error`` are decimated to
    the model grid and reduced to the K active channels (radial order).
    """

    shot_id: int
    beam_ok: bool
    time: np.ndarray  # (K_t,) decimated slice times (s)
    active_channel_ids: np.ndarray  # (C,) int — channel ids in radial order
    active_channel_rpos: np.ndarray  # (C,) float — major radius (m), radial order
    pitch: np.ndarray  # (K_t, C) pitch angle (rad)
    pitch_error: np.ndarray  # (K_t, C) pitch error (rad)
    gamma: np.ndarray  # (K_t, C) polarisation angle (rad)
    gamma_error: np.ndarray
    q0: np.ndarray  # (K_t,) derived on-axis q (q0_kappa1.85_4pt)
    q0_error: np.ndarray
    rax: np.ndarray  # (K_t,) derived magnetic-axis R (rax_4pt)
    rax_error: np.ndarray


def _open_ams(shot_zarr_path: Path):
    """Open the ``ams`` group of a shot, robust to consolidated-metadata gaps.

    Returns the zarr group or None if the shot/group is unreadable.
    """
    import zarr  # noqa: PLC0415

    if not (shot_zarr_path / MSE_GROUP).exists():
        return None
    try:
        return zarr.open_group(str(shot_zarr_path / MSE_GROUP), mode="r")
    except Exception as e:  # pragma: no cover - corpus robustness
        logger.debug("Cannot open %s/ams: %s", shot_zarr_path.name, e)
        return None


def _read_array(group, name: str) -> np.ndarray | None:
    """Read an array from a zarr group, tolerant of consolidated-metadata gaps."""
    try:
        if name not in set(group.array_keys()):
            return None
        return np.asarray(group[name])
    except Exception:  # pragma: no cover - corpus robustness
        return None


def probe_beam_ok(shot_zarr_path: Path) -> bool | None:
    """Cheap pass: is this an ams shot with a beam-on MSE measurement?

    Returns True if ``beam_ok == 1`` AND a time-resolved ``pitcha`` is present,
    False if the shot is a beam-off (static geometry only), None if unreadable.
    """
    grp = _open_ams(shot_zarr_path)
    if grp is None:
        return None
    keys = set(grp.array_keys())
    if "beam_ok" not in keys:
        return None
    bo = _read_array(grp, "beam_ok")
    if bo is None:
        return None
    beam_on = float(np.asarray(bo).reshape(-1)[0]) == 1.0
    has_measurement = "time" in keys and "pitcha" in keys
    return bool(beam_on and has_measurement)


def _active_channels(grp) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Return (channel_ids, rpos, radial_order) for the K active channels.

    ``channel_ids`` / ``rpos`` are the *compacted* finite entries; ``radial_order``
    sorts them by increasing major radius.  All three are returned already in
    radial order.  Returns None if channel metadata is missing.
    """
    ch = _read_array(grp, "ch")
    rpos = _read_array(grp, "rpos")
    if ch is None or rpos is None:
        return None
    ch = np.asarray(ch).reshape(-1)
    rpos = np.asarray(rpos).reshape(-1)
    fin = np.isfinite(ch) & np.isfinite(rpos)
    ch_f = ch[fin]
    r_f = rpos[fin]
    if ch_f.size == 0:
        return None
    order = np.argsort(r_f)
    return ch_f[order].astype(int), r_f[order].astype(float), order


def _finite_columns(arr2d: np.ndarray) -> np.ndarray:
    """Indices of columns that are finite for at least one row (active channels)."""
    return np.where(np.isfinite(arr2d).any(axis=0))[0]


def _decimate_indices(time: np.ndarray, grid_hz: int) -> np.ndarray:
    """Indices that thin a ~2 kHz axis down to ~grid_hz (nearest-sample bin).

    We keep at most one sample per 1/grid_hz bin (the first sample whose time
    falls into a new bin).  This is a deterministic, monotone decimation that
    preserves the slice times exactly (no interpolation of physical truth).
    """
    if time.size == 0:
        return np.array([], dtype=int)
    bin_width = 1.0 / float(grid_hz)
    # bin index for each sample
    bins = np.floor((time - time[0]) / bin_width).astype(np.int64)
    # first occurrence of each bin
    _, first_idx = np.unique(bins, return_index=True)
    return np.sort(first_idx)


def read_ams_shot(shot_zarr_path: Path, grid_hz: int = MODEL_GRID_HZ) -> AmsShot | None:
    """Decode the beam-on MSE measurement for one shot onto the model grid.

    Returns None unless the shot is beam-on with a usable pitch measurement.
    Pitch/gamma are reduced to the K active channels (radial order) and the
    time axis is decimated from ~2 kHz to ``grid_hz``.
    """
    grp = _open_ams(shot_zarr_path)
    if grp is None:
        return None
    keys = set(grp.array_keys())
    bo = _read_array(grp, "beam_ok")
    if bo is None or float(np.asarray(bo).reshape(-1)[0]) != 1.0:
        return None
    if "time" not in keys or "pitcha" not in keys:
        return None

    time = _read_array(grp, "time")
    pitcha = _read_array(grp, "pitcha")
    if time is None or pitcha is None or pitcha.ndim != 2:
        return None
    time = np.asarray(time).reshape(-1)

    ac = _active_channels(grp)
    if ac is None:
        return None
    ch_ids, ch_rpos, order = ac

    cols = _finite_columns(pitcha)
    # The active channels and the finite pitch columns must agree in count;
    # positional pairing maps the k-th radial channel to the k-th finite column.
    if cols.size != ch_ids.size:
        # Fall back to the min count to stay robust; both are radially ordered.
        k = min(cols.size, ch_ids.size)
        if k == 0:
            return None
        cols = cols[:k]
        ch_ids = ch_ids[:k]
        ch_rpos = ch_rpos[:k]

    def _reduce(name: str) -> np.ndarray:
        a = _read_array(grp, name)
        if a is None or a.ndim != 2:
            return np.full((time.size, cols.size), np.nan)
        # select finite columns, then reorder to radial order via `order`
        sel = a[:, cols]
        # cols are already in column-index order; `order` sorted the compacted
        # ch/rpos by radius. The finite columns appear in the SAME (radial)
        # order as the compacted ch entries, so cols[k] ↔ compacted entry k,
        # which `order` then sorts. Apply the same permutation to the columns.
        return sel[:, order[: cols.size]] if order.size >= cols.size else sel

    pitch = _reduce("pitcha")
    pitch_err = _reduce("pitcha_error")
    gamma = _reduce("gamma")
    gamma_err = _reduce("gamma_error")

    def _scalar(name: str) -> np.ndarray:
        a = _read_array(grp, name)
        if a is None:
            return np.full((time.size,), np.nan)
        return np.asarray(a).reshape(-1)

    q0 = _scalar("q0_kappa1.85_4pt")
    q0_err = _scalar("q0_kappa1.85_4pt_error")
    rax = _scalar("rax_4pt")
    rax_err = _scalar("rax_4pt_error")

    # Decimate the time axis (and every per-slice array) onto the model grid.
    keep = _decimate_indices(time, grid_hz)

    def _take(a: np.ndarray) -> np.ndarray:
        return a[keep] if a.shape[0] == time.size else a

    return AmsShot(
        shot_id=int(shot_zarr_path.stem),
        beam_ok=True,
        time=time[keep],
        active_channel_ids=ch_ids,
        active_channel_rpos=ch_rpos,
        pitch=_take(pitch),
        pitch_error=_take(pitch_err),
        gamma=_take(gamma),
        gamma_error=_take(gamma_err),
        q0=_take(q0),
        q0_error=_take(q0_err),
        rax=_take(rax),
        rax_error=_take(rax_err),
    )


# ---------------------------------------------------------------------------
# Per-slice gating masks
# ---------------------------------------------------------------------------


def pitch_valid_mask(shot: AmsShot) -> np.ndarray:
    """(K_t,) bool — slices with enough finite, physical pitch channels.

    A slice is pitch-valid if it has >= ``PITCH_VALID_MIN_CH`` finite channels
    whose pitch lies in the physical range [-pi/2, pi/2].
    """
    phys = np.isfinite(shot.pitch) & (np.abs(shot.pitch) <= np.pi / 2.0)
    return phys.sum(axis=1) >= PITCH_VALID_MIN_CH


def q0_gated_mask(shot: AmsShot) -> np.ndarray:
    """(K_t,) bool — slices passing the SECONDARY q0 physical + error gate."""
    q0 = shot.q0
    q0e = shot.q0_error
    m = np.isfinite(q0) & (q0 >= Q0_MIN) & (q0 <= Q0_MAX)
    with np.errstate(invalid="ignore"):
        err_ok = np.isfinite(q0e) & (np.abs(q0e) < ERR_FRAC * np.abs(q0))
    # if error is absent treat error-cut as pass (physical gate already strong)
    err_ok = np.where(np.isfinite(q0e), err_ok, m)
    return m & err_ok


def rax_gated_mask(shot: AmsShot) -> np.ndarray:
    """(K_t,) bool — slices passing the SECONDARY rax physical + error gate."""
    rax = shot.rax
    raxe = shot.rax_error
    m = np.isfinite(rax) & (rax >= RAX_MIN) & (rax <= RAX_MAX)
    with np.errstate(invalid="ignore"):
        err_ok = np.isfinite(raxe) & (np.abs(raxe) < ERR_FRAC * np.abs(rax))
    err_ok = np.where(np.isfinite(raxe), err_ok, m)
    return m & err_ok


# ---------------------------------------------------------------------------
# Tier-1 union corpus + split build
# ---------------------------------------------------------------------------


def load_inventory_shot_groups(inventory_path: Path) -> dict[int, set[str]]:
    """Load {shot_id -> set(groups)} from the family inventory manifest."""
    inv = json.loads(inventory_path.read_text(encoding="utf-8"))
    return {int(row[0]): set(row[1]) for row in inv["shot_groups"]}


def tier1_union_shots(shot_groups: dict[int, set[str]]) -> list[int]:
    """Shots in the MSE-independent Tier-1 union (mag ∧ cam ∧ bolo ∧ sxr).

    The union requires presence of at least one channel from each of the four
    multi-modal pillars.  MSE (``ams``) plays no role here.
    """
    mag = set(TIER1_MAGNETICS)
    cam = set(TIER1_CAMERA)
    bolo = set(TIER1_BOLOMETER)
    sxr = set(TIER1_SXR)
    out = []
    for sid, g in shot_groups.items():
        if (g & mag) and (g & cam) and (g & bolo) and (g & sxr):
            out.append(sid)
    return sorted(out)


def ams_shots(shot_groups: dict[int, set[str]]) -> list[int]:
    """All shots whose inventory lists the ``ams`` group (beam state unknown)."""
    return sorted(sid for sid, g in shot_groups.items() if MSE_GROUP in g)


@dataclass
class MseSplit:
    """The locked MSE held-out split + summary counts."""

    train: list[int] = field(default_factory=list)  # Tier-1 union, MSE-independent
    calibration: list[int] = field(default_factory=list)  # beam-on ams, in-dist
    held_out: list[int] = field(default_factory=list)  # beam-on ams, OOD box
    ood_box: dict = field(default_factory=lambda: dict(LOCKED_OOD_BOX_V0))
    train_input_groups: list[str] = field(default_factory=list)
    n_ams_total: int = 0
    n_ams_beam_on: int = 0
    n_ams_beam_off: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def n_train(self) -> int:
        return len(self.train)

    @property
    def n_cal(self) -> int:
        return len(self.calibration)

    @property
    def n_heldout(self) -> int:
        return len(self.held_out)

    def assert_no_mse_in_train(self) -> None:
        """Hard gate: TRAIN is disjoint from eval AND has no MSE input ref."""
        eval_set = set(self.calibration) | set(self.held_out)
        overlap = set(self.train) & eval_set
        assert not overlap, f"TRAIN ∩ (CAL∪HELDOUT) non-empty: {sorted(overlap)[:10]}"
        assert MSE_GROUP not in self.train_input_groups, (
            f"TRAIN input groups contain MSE group {MSE_GROUP!r}: "
            f"{self.train_input_groups}"
        )

    def summary_dict(self) -> dict:
        return {
            "version": "mse_split_v0",
            "model_grid_hz": MODEL_GRID_HZ,
            "n_train": self.n_train,
            "n_calibration": self.n_cal,
            "n_held_out": self.n_heldout,
            "n_ams_total": self.n_ams_total,
            "n_ams_beam_on": self.n_ams_beam_on,
            "n_ams_beam_off": self.n_ams_beam_off,
            "ood_box": self.ood_box,
            "train_input_groups": self.train_input_groups,
            "gate_thresholds": {
                "q0_min": Q0_MIN,
                "q0_max": Q0_MAX,
                "rax_min": RAX_MIN,
                "rax_max": RAX_MAX,
                "err_frac": ERR_FRAC,
                "pitch_valid_min_ch": PITCH_VALID_MIN_CH,
            },
            "notes": self.notes,
        }


def _in_box(ip_mean: float, ne_mean: float, box: dict) -> bool:
    ne_scaled = ne_mean / 1e19
    return (
        box["ip_min_kA"] <= ip_mean <= box["ip_max_kA"]
        and box["ne_min_1e19"] <= ne_scaled <= box["ne_max_1e19"]
    )


def build_mse_split(
    inventory_path: Path,
    regime_path: Path,
    level1_dir: Path,
    beam_on_shots: list[int],
    ood_box: dict | None = None,
) -> MseSplit:
    """Assemble the TRAIN / CAL / HELD-OUT partitions.

    Parameters
    ----------
    inventory_path, regime_path:
        Family inventory + regime-scalar manifests.
    level1_dir:
        Root of the level-1 Zarr corpus.
    beam_on_shots:
        Pre-confirmed beam-on ams shots (from the cheap probe pass).
    ood_box:
        OOD box dict; defaults to the LOCKED v0 box.
    """
    box = dict(ood_box or LOCKED_OOD_BOX_V0)
    shot_groups = load_inventory_shot_groups(inventory_path)
    regime = {int(k): v for k, v in json.loads(regime_path.read_text()).items()}

    all_ams = ams_shots(shot_groups)
    beam_on = sorted(set(beam_on_shots) & set(all_ams) | set(beam_on_shots))
    beam_on_set = set(beam_on)

    # TRAIN = Tier-1 union MINUS all beam-on ams shots (disjointness guarantee).
    tier1 = tier1_union_shots(shot_groups)
    train = sorted(set(tier1) - beam_on_set)

    # CAL / HELD-OUT split of the beam-on ams shots by the locked OOD box.
    held_out, calibration, no_regime = [], [], []
    for sid in beam_on:
        r = regime.get(sid)
        if r is None or "ip_mean" not in r or "ne_mean" not in r:
            no_regime.append(sid)
            calibration.append(sid)  # no regime coord → keep as in-distribution cal
            continue
        if _in_box(r["ip_mean"], r["ne_mean"], box):
            held_out.append(sid)
        else:
            calibration.append(sid)

    notes = [
        "TRAIN = Tier-1 union (mag∧cam∧bolo∧sxr), MSE-independent, minus all "
        "beam-on ams shots (disjointness). MSE never an input or label.",
        "CAL/HELD-OUT = beam-on ams shots, split by the LOCKED v0 OOD box "
        "(joint_p84, by-current-density) — identical to the dalpha / "
        "integrated_core v0 split artifacts.",
        f"{len(no_regime)} beam-on ams shots had no regime coord → assigned to CAL.",
        "PRIMARY truth = pitcha (clean). SECONDARY = physically-gated + "
        "error-weighted q0_kappa1.85_4pt + rax_4pt.",
        "SECONDARY note: q0_4pt/rax_4pt are MSE-pipeline 2pt/4pt fits (fixed "
        "κ=1.85), finite on a slice population that only PARTIALLY overlaps the "
        "pitch slices (~330 co-finite/shot on the studied shot). The eval "
        "harness scores the method-matched secondary on inv(predicted_pitch) "
        "and reports inv(truth_pitch)-vs-raw-q0_4pt agreement as a provisional "
        "cross-check. The gate rests on PRIMARY pitch only.",
    ]

    split = MseSplit(
        train=train,
        calibration=sorted(calibration),
        held_out=sorted(held_out),
        ood_box=box,
        train_input_groups=sorted(
            set(TIER1_MAGNETICS)
            | set(TIER1_CAMERA)
            | set(TIER1_BOLOMETER)
            | set(TIER1_SXR)
        ),
        n_ams_total=len(all_ams),
        n_ams_beam_on=len(beam_on),
        n_ams_beam_off=len(all_ams) - len(beam_on),
        notes=notes,
    )

    # Held-out floor guard: widen the OOD hole if too small (documented).
    if split.n_heldout < HELDOUT_N_FLOOR:
        split.notes.append(
            f"WARNING: held-out N={split.n_heldout} < floor {HELDOUT_N_FLOOR}. "
            "Caller should widen the OOD hole (see build_mse_split ood_box arg)."
        )
    return split


# ---------------------------------------------------------------------------
# Manifest assembly (per-shot truth + masks) for CAL ∪ HELD-OUT
# ---------------------------------------------------------------------------


def build_shot_manifest(shot: AmsShot, partition: str) -> dict:
    """Per-shot manifest entry for one beam-on ams shot.

    Times are the decimated beam-on slice times; masks are per-slice booleans.
    The truth arrays themselves are read on demand by the eval harness from
    the level-1 corpus (the manifest carries only times + masks + geometry,
    keeping it compact and the truth single-sourced).
    """
    pv = pitch_valid_mask(shot)
    q0g = q0_gated_mask(shot)
    raxg = rax_gated_mask(shot)
    return {
        "shot_id": int(shot.shot_id),
        "partition": partition,
        "model_grid_hz": MODEL_GRID_HZ,
        "beam_on_slice_times": [float(t) for t in shot.time],
        "active_channel_ids": [int(c) for c in shot.active_channel_ids],
        "active_channel_rpos": [float(r) for r in shot.active_channel_rpos],
        "pitch_valid_mask": [bool(x) for x in pv],
        "q0_gated_mask": [bool(x) for x in q0g],
        "rax_gated_mask": [bool(x) for x in raxg],
        "n_pitch_valid": int(pv.sum()),
        "n_q0_gated": int(q0g.sum()),
        "n_rax_gated": int(raxg.sum()),
        "n_secondary_cofinite": int((pv & q0g).sum()),
    }


def save_manifest(
    split: MseSplit,
    shot_entries: dict[int, dict],
    path: Path,
) -> None:
    """Persist the full per-shot manifest JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": "mse_heldout_split_v0",
        "summary": split.summary_dict(),
        "shots": {str(sid): entry for sid, entry in sorted(shot_entries.items())},
    }
    path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    logger.info(
        "MSE manifest saved to %s (cal=%d heldout=%d)",
        path,
        split.n_cal,
        split.n_heldout,
    )


def save_summary(split: MseSplit, path: Path) -> None:
    """Persist the compact split summary (counts, partitions, gates)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = split.summary_dict()
    # include the partition shot-id lists in the compact summary for auditing
    payload["train"] = [int(x) for x in split.train]
    payload["calibration"] = [int(x) for x in split.calibration]
    payload["held_out"] = [int(x) for x in split.held_out]
    path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    logger.info("MSE split summary saved to %s", path)


# ---------------------------------------------------------------------------
# Driver — cheap beam-on probe pass + manifest build
# ---------------------------------------------------------------------------


def _probe_worker(root_str: str, sid: int) -> tuple[int, bool | None]:
    from pathlib import Path as _Path  # noqa: PLC0415

    return sid, probe_beam_ok(_Path(root_str) / f"{sid}.zarr")


def find_beam_on_shots(
    candidate_shots: list[int],
    level1_dir: Path,
    max_workers: int = 8,
) -> list[int]:
    """Cheap parallel pass: which candidate ams shots are beam-on?"""
    from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: PLC0415
    from functools import partial  # noqa: PLC0415

    worker = partial(_probe_worker, str(level1_dir))
    beam_on = []
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(worker, sid): sid for sid in candidate_shots}
        for fut in as_completed(futs):
            sid, ok = fut.result()
            if ok:
                beam_on.append(sid)
    return sorted(beam_on)


def _manifest_worker(
    root_str: str, sid: int, partition: str
) -> tuple[int, dict | None]:
    from pathlib import Path as _Path  # noqa: PLC0415

    shot = read_ams_shot(_Path(root_str) / f"{sid}.zarr")
    if shot is None:
        return sid, None
    return sid, build_shot_manifest(shot, partition)


def build_manifests_parallel(
    split: MseSplit,
    level1_dir: Path,
    max_workers: int = 8,
) -> dict[int, dict]:
    """Build per-shot manifest entries for CAL ∪ HELD-OUT in parallel."""
    from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: PLC0415
    from functools import partial  # noqa: PLC0415

    worker = partial(_manifest_worker, str(level1_dir))
    jobs = [(sid, "calibration") for sid in split.calibration]
    jobs += [(sid, "held_out") for sid in split.held_out]
    entries: dict[int, dict] = {}
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(worker, sid, part): sid for sid, part in jobs}
        for n_done, fut in enumerate(as_completed(futs), start=1):
            sid, entry = fut.result()
            if entry is not None:
                entries[sid] = entry
            if n_done % 200 == 0:
                logger.info("  … %d / %d manifests built", n_done, len(jobs))
    return entries


def main(
    inventory_path: Path | None = None,
    regime_path: Path | None = None,
    level1_dir: Path | None = None,
    manifest_out: Path | None = None,
    summary_out: Path | None = None,
    max_workers: int = 8,
) -> MseSplit:
    """End-to-end build of the locked MSE held-out split + manifest."""
    from imas_ambix.data.paths import LEVEL1_DIR, MANIFEST_DIR  # noqa: PLC0415

    inventory_path = inventory_path or (
        MANIFEST_DIR / "statespace_family_inventory.json"
    )
    regime_path = regime_path or (MANIFEST_DIR / "statespace_regime_scalars.json")
    level1_dir = level1_dir or LEVEL1_DIR
    manifest_out = manifest_out or (MANIFEST_DIR / "mse_heldout_split_v0.json")
    summary_out = summary_out or (
        Path(__file__).parent / "artifacts" / "mse_split_v0.json"
    )

    shot_groups = load_inventory_shot_groups(inventory_path)
    candidate = ams_shots(shot_groups)
    logger.info("Found %d candidate ams shots; probing beam_ok…", len(candidate))
    beam_on = find_beam_on_shots(candidate, level1_dir, max_workers=max_workers)
    logger.info(
        "%d / %d ams shots are beam-on (usable MSE).", len(beam_on), len(candidate)
    )

    split = build_mse_split(inventory_path, regime_path, level1_dir, beam_on)
    split.assert_no_mse_in_train()

    entries = build_manifests_parallel(split, level1_dir, max_workers=max_workers)
    save_manifest(split, entries, manifest_out)
    save_summary(split, summary_out)

    print(
        f"[mse_split] usable-MSE (beam-on ams) shots: {split.n_ams_beam_on} "
        f"(of {split.n_ams_total} ams total; {split.n_ams_beam_off} beam-off)"
    )
    print(
        f"[mse_split] partitions: TRAIN={split.n_train}  CAL={split.n_cal}  "
        f"HELD-OUT(effective N)={split.n_heldout}"
    )
    print(f"[mse_split] manifest → {manifest_out}")
    print(f"[mse_split] summary  → {summary_out}")
    return split


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
