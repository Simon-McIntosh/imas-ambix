"""Train / held-out-shots (calibration) / held-out-regime (OOD) split generation.

Splits are defined on shot IDs and committed to a JSON artifact so they
are reproducible and immutable once locked.

Split taxonomy
--------------
train
    Shots used for model training.  May include every measured family
    (inputs + target).

calibration
    Held-out shots drawn from the SAME operating regime as train, used
    for split-conformal coverage calibration.  Disjoint from train.
    Recommendation: 10–15 % of co-available shots.

test_ood_regime
    Shots from a held-out operating-space region (Iₚ × density hole).
    Used for OOD generalisation evaluation.  Disjoint from train and
    calibration.

Regime axis
-----------
The regime axis is derived from two scalar per-shot operating-point
estimates:
    ip_mean   : plasma-on mean |Iₚ| (kA), from amc/plasma_current
    ne_mean   : plasma-on median line-integrated density (m⁻²), from ane/density

These scalars are computed by :func:`compute_regime_scalars` and stored
in the split artifact for auditing.

IMPORTANT NOTE ON REGIME-AXIS CIRCULARITY
------------------------------------------
If *magnetics* is the held-out target, the regime axis Iₚ is derived from
amc (magnetics). This means the split axis leaks target information.  The
orchestrator must weigh this; the flag is surfaced in the split artifact.
"""

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regime scalar computation — PHYSICAL plasma-on masking
# ---------------------------------------------------------------------------
#
# IMPORTANT (RCA 2026-05-29): An index-based "flat-top" mean over the FULL
# ~30k-sample amc record dilutes the operating point with off-plasma zeros —
# plasma-on occupies only ~7% of the record window (e.g. amc time spans
# [-2, 4]s but the burn is ~0.04-0.4s). The earlier middle-80% trim gave a
# diluted Iₚ p50 ≈ 119 kA against a physical MAST flat-top of ~400-900 kA.
#
# The fix: define the plasma-on window from |Iₚ| itself (contiguous span where
# |Iₚ| > threshold), take the MEAN of Iₚ over that window (physical), and take
# the MEDIAN of ne over the SAME window — selected by ne's OWN time axis, since
# amc (250 µs grid, 30k samples, [-2,4]s) and ane (~40 µs grid, 32768 samples,
# ~[-0.01,1.3]s) live on DIFFERENT time bases. Median + physical clipping
# rejects interferometer fringe-jump spikes and NaN-filled / fringe-locked
# saturated traces.
#
# UNITS (verified 2026-05-29 from the channel attrs):
#   - amc/plasma_current: peaks ~760-1122 → already in kA (NOT amps), no /1000.
#   - ane/density: units attr = '1 / m ** 2', description "integrated electron
#     density" → this is LINE-INTEGRATED density (m⁻²), NOT volumetric (m⁻³).
#     It remains a valid monotone regime axis; it is labelled m⁻² everywhere.
#     Typical physical line-integral for MAST is ~5-25e19 m⁻²; values pinned
#     near the old 50e19 clip with mostly-NaN traces (e.g. shots 15952/16000)
#     are fringe-locked / saturated artifacts → clip tightened to 30e19 m⁻².

_IP_THRESHOLD_FLOOR_KA = 50.0  # absolute floor for plasma-on detection (kA)
_IP_THRESHOLD_FRACTION = 0.2  # plasma-on = |Iₚ| > 0.2 × peak|Iₚ|
_NE_PHYSICAL_MAX = 30e19  # reject line-integrated ne above this (m⁻²) — values
# above are fringe-locked / saturated (physical MAST line-integral ≤ ~25e19 m⁻²)
_NE_PHYSICAL_MIN = 0.0  # reject negative ne (instrument DC offset / noise)
_NE_MIN_VALID_FRACTION = 0.50  # require ≥50 % of plasma-on ne samples to be
# finite-and-in-range. A trace that lost fringe lock for most of the burn
# (e.g. shot 16061: 99.4 % NaN) is a BROKEN interferometer, not a high-density
# shot; its median-of-survivors is not a trustworthy regime coordinate and
# must NOT define the OOD high-ne edge. Such shots get no ne_mean (dropped).


def _plasma_on_window(
    ip: np.ndarray,
    ip_time: np.ndarray | None,
) -> tuple[float, float, np.ndarray] | None:
    """Return (t_start, t_end, plasma_on_mask) for the plasma burn.

    The plasma-on window is the contiguous span (first→last index) where
    ``|Iₚ| > max(floor, fraction × peak|Iₚ|)``.

    Returns None if no sample exceeds the threshold (no plasma) or if the
    time axis is missing/degenerate.
    """
    if ip.ndim != 1 or ip.size < 4:
        return None
    abs_ip = np.abs(ip)
    peak = float(np.nanmax(abs_ip))
    if not np.isfinite(peak) or peak <= _IP_THRESHOLD_FLOOR_KA:
        return None
    threshold = max(_IP_THRESHOLD_FLOOR_KA, _IP_THRESHOLD_FRACTION * peak)
    on = abs_ip > threshold
    idx = np.where(on)[0]
    if idx.size == 0:
        return None
    lo, hi = int(idx[0]), int(idx[-1])
    # Contiguous span first→last (fills any brief sub-threshold dips)
    span_mask = np.zeros_like(on)
    span_mask[lo : hi + 1] = True
    if ip_time is not None and ip_time.size == ip.size:
        t_start = float(ip_time[lo])
        t_end = float(ip_time[hi])
    else:
        # No usable time axis — fall back to index bounds as pseudo-time
        t_start, t_end = float(lo), float(hi)
    return t_start, t_end, span_mask


def _compute_regime_scalars_one(
    shot_zarr_path: Path,
) -> dict[str, float]:
    """Compute physical ``ip_mean`` (kA) and ``ne_mean`` (m⁻², line-integrated).

    ``ip_mean`` is the mean of Iₚ over the plasma-on window.
    ``ne_mean`` is the median of ane/density over the SAME time window
    (selected via ane's own time axis), after rejecting non-physical values.

    Returns an empty dict if amc is missing or no plasma is detected.
    Keys are present only when their channel could be read.
    """
    import zarr  # noqa: PLC0415

    if not (shot_zarr_path / "amc").exists():
        return {}
    try:
        store = zarr.open_group(str(shot_zarr_path), mode="r")
    except Exception as e:
        logger.debug("Cannot open %s: %s", shot_zarr_path.name, e)
        return {}

    scalars: dict[str, float] = {}

    # --- Iₚ: plasma-on window + physical mean -------------------------------
    try:
        amc = store["amc"]
        ip = np.asarray(amc["plasma_current"])
        ip_time = np.asarray(amc["time"]) if "time" in amc else None
    except Exception as e:
        logger.debug(
            "Cannot read amc/plasma_current for %s: %s", shot_zarr_path.name, e
        )
        return {}

    window = _plasma_on_window(ip, ip_time)
    if window is None:
        return {}  # no plasma → no regime point
    t_start, t_end, span_mask = window
    ip_on = np.abs(ip[span_mask])
    ip_on = ip_on[np.isfinite(ip_on)]
    if ip_on.size == 0:
        return {}
    scalars["ip_mean"] = float(np.mean(ip_on))  # kA, physical flat-top

    # --- ne: median over the SAME time window (ane's own time axis) ---------
    if (shot_zarr_path / "ane").exists():
        try:
            ane = store["ane"]
            ne = np.asarray(ane["density"])
            ne_time = np.asarray(ane["time"]) if "time" in ane else None
        except Exception:
            ne = None
            ne_time = None
        if ne is not None and ne.ndim == 1 and ne.size >= 4:
            if ne_time is not None and ne_time.size == ne.size:
                ne_mask = (ne_time >= t_start) & (ne_time <= t_end)
            else:
                # No ne time axis — fall back to whole record (rare)
                ne_mask = np.ones_like(ne, dtype=bool)
            ne_window = ne[ne_mask]
            # Reject non-physical interferometer values (negative DC offsets,
            # fringe-jump spikes, NaN-filled fringe-lock losses) BEFORE the
            # median
            ne_clip = ne_window[
                np.isfinite(ne_window)
                & (ne_window >= _NE_PHYSICAL_MIN)
                & (ne_window <= _NE_PHYSICAL_MAX)
            ]
            # Valid-fraction guard: a trace that lost fringe lock for most of
            # the burn is a broken interferometer, not a high-density shot.
            # Require enough of the plasma-on window to be valid; otherwise
            # drop ne_mean entirely (the shot keeps ip_mean but is excluded
            # from the ne axis / OOD high-ne edge).
            valid_fraction = (
                ne_clip.size / ne_window.size if ne_window.size > 0 else 0.0
            )
            if ne_clip.size > 0 and valid_fraction >= _NE_MIN_VALID_FRACTION:
                scalars["ne_mean"] = float(np.median(ne_clip))  # m⁻², robust

    return scalars


def _regime_worker(root_str: str, sid: int) -> tuple[int, dict[str, float]]:
    """Module-level (picklable) worker for :func:`compute_regime_scalars`.

    Defined at module scope so it can be sent to a ProcessPoolExecutor under
    the forkserver/spawn start methods (Python 3.14+ no longer defaults to
    fork, so a closure inside ``compute_regime_scalars`` is not picklable).
    """
    from pathlib import Path as _Path  # noqa: PLC0415

    return sid, _compute_regime_scalars_one(_Path(root_str) / f"{sid}.zarr")


def compute_regime_scalars(
    shot_ids: list[int],
    level1_dir: Path,
    max_workers: int = 8,
) -> dict[int, dict[str, float]]:
    """Compute per-shot PHYSICAL operating-point scalars for regime splits.

    Uses plasma-on masking (see :func:`_plasma_on_window`):
    - ``ip_mean`` = mean |Iₚ| (kA) over the plasma-on window.
    - ``ne_mean`` = median ane/density (m⁻², line-integrated) over the same
      time window, after rejecting non-physical values.

    Parameters
    ----------
    shot_ids:
        Shots to process (typically only those with both amc and ane).
    level1_dir:
        Root of the level-1 Zarr corpus.
    max_workers:
        Worker processes for parallel Zarr reads.

    Returns
    -------
    dict mapping shot_id → {``"ip_mean"``: float (kA), ``"ne_mean"``: float (m⁻²)}
    Shots with no detectable plasma are omitted entirely.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: PLC0415
    from functools import partial  # noqa: PLC0415

    from imas_ambix.data.paths import LEVEL1_DIR  # noqa: PLC0415

    root = level1_dir or LEVEL1_DIR

    # Module-level worker (picklable under forkserver/spawn — the default
    # start method changed in Python 3.14, so a local closure can no longer
    # be sent to workers).
    worker = partial(_regime_worker, str(root))

    results: dict[int, dict[str, float]] = {}
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(worker, sid): sid for sid in shot_ids}
        for n_done, fut in enumerate(as_completed(futures), start=1):
            sid, scalars = fut.result()
            if scalars:
                results[sid] = scalars
            if n_done % 500 == 0:
                logger.info(
                    "  … %d / %d regime scalars computed", n_done, len(shot_ids)
                )

    logger.info(
        "Regime scalars: %d / %d shots with both Iₚ and ne",
        sum(1 for v in results.values() if "ip_mean" in v and "ne_mean" in v),
        len(shot_ids),
    )
    return results


# ---------------------------------------------------------------------------
# Regime-split geometry
# ---------------------------------------------------------------------------


@dataclass
class RegimeBox:
    """An axis-aligned box in (Iₚ, ne) space defining the OOD held-out region.

    Shots whose operating point falls within this box are held out as OOD.

    Attributes
    ----------
    ip_min, ip_max:
        Iₚ range in kA (MAST convention; amc/plasma_current is in kA).
    ne_min, ne_max:
        Line-integrated density range in 10^19 m⁻² (i.e. ne_mean / 1e19).
    description:
        Human-readable label (e.g. "high-current / high-density corner").
    """

    ip_min: float
    ip_max: float
    ne_min: float  # in units of 1e19 m⁻² (line-integrated)
    ne_max: float
    description: str = ""

    def contains(self, ip_mean: float, ne_mean: float) -> bool:
        """Return True if (ip_mean [kA], ne_mean [10¹⁹ m⁻²]) falls inside this box.

        Both arguments are in the units that :attr:`ip_min`/:attr:`ne_min` use:
        ip in kA, ne already divided by 10¹⁹ (i.e. ne_scaled = ne_raw / 1e19).
        """
        return (
            self.ip_min <= ip_mean <= self.ip_max
            and self.ne_min <= ne_mean <= self.ne_max
        )

    def to_dict(self) -> dict:
        return {
            "ip_min_kA": self.ip_min,
            "ip_max_kA": self.ip_max,
            "ne_min_1e19": self.ne_min,
            "ne_max_1e19": self.ne_max,
            "description": self.description,
        }


# ---------------------------------------------------------------------------
# Split dataclass
# ---------------------------------------------------------------------------


@dataclass
class ShotSplits:
    """Immutable shot-ID split assignment.

    Attributes
    ----------
    train, calibration, test_ood_regime:
        Lists of shot IDs in each split.
    regime_scalars:
        Per-shot operating-point scalars used to define the regime split.
    ood_box:
        The :class:`RegimeBox` defining the OOD held-out region.
    held_out_family:
        The held-out diagnostic family (e.g. ``"dalpha"``).
    input_groups:
        The input diagnostic groups (everything except the held-out family
        and excluded groups).
    circularity_warning:
        Non-empty string when the regime axis leaks target information.
    notes:
        Additional provenance notes.
    """

    train: list[int] = field(default_factory=list)
    calibration: list[int] = field(default_factory=list)
    test_ood_regime: list[int] = field(default_factory=list)
    regime_scalars: dict[int, dict[str, float]] = field(default_factory=dict)
    ood_box: RegimeBox | None = None
    held_out_family: str = ""
    input_groups: list[str] = field(default_factory=list)
    circularity_warning: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def n_train(self) -> int:
        return len(self.train)

    @property
    def n_cal(self) -> int:
        return len(self.calibration)

    @property
    def n_ood(self) -> int:
        return len(self.test_ood_regime)

    @property
    def n_total(self) -> int:
        return self.n_train + self.n_cal + self.n_ood

    def to_dict(self) -> dict:
        return {
            "held_out_family": self.held_out_family,
            "input_groups": self.input_groups,
            "circularity_warning": self.circularity_warning,
            "n_train": self.n_train,
            "n_calibration": self.n_cal,
            "n_test_ood_regime": self.n_ood,
            "n_total": self.n_total,
            "ood_box": self.ood_box.to_dict() if self.ood_box else None,
            "notes": self.notes,
            "train": [int(x) for x in self.train],
            "calibration": [int(x) for x in self.calibration],
            "test_ood_regime": [int(x) for x in self.test_ood_regime],
            "regime_scalars": {str(k): v for k, v in self.regime_scalars.items()},
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), separators=(",", ":")), encoding="utf-8"
        )
        logger.info(
            "Splits saved to %s (train=%d cal=%d ood=%d)",
            path,
            self.n_train,
            self.n_cal,
            self.n_ood,
        )

    @classmethod
    def load(cls, path: Path) -> ShotSplits:
        d = json.loads(path.read_text(encoding="utf-8"))
        ood_box = None
        if d.get("ood_box"):
            b = d["ood_box"]
            ood_box = RegimeBox(
                ip_min=b["ip_min_kA"],
                ip_max=b["ip_max_kA"],
                ne_min=b["ne_min_1e19"],
                ne_max=b["ne_max_1e19"],
                description=b.get("description", ""),
            )
        return cls(
            train=[int(x) for x in d.get("train", [])],
            calibration=[int(x) for x in d.get("calibration", [])],
            test_ood_regime=[int(x) for x in d.get("test_ood_regime", [])],
            regime_scalars={int(k): v for k, v in d.get("regime_scalars", {}).items()},
            ood_box=ood_box,
            held_out_family=d.get("held_out_family", ""),
            input_groups=d.get("input_groups", []),
            circularity_warning=d.get("circularity_warning", ""),
            notes=d.get("notes", []),
        )


# ---------------------------------------------------------------------------
# Split builder
# ---------------------------------------------------------------------------


def propose_ood_box(
    regime_scalars: dict[int, dict[str, float]],
    ood_fraction_target: float = 0.10,
) -> tuple[RegimeBox, list[int]]:
    """Propose a held-out OOD region based on the empirical distribution.

    Selects the high-Iₚ × high-density quadrant that contains approximately
    *ood_fraction_target* of the shots with regime scalars.  The box is
    placed in the upper-right quadrant (both Iₚ and ne above their 80th
    percentile) — this represents a distinct operating regime rarely seen
    in typical training shots.

    Returns
    -------
    (RegimeBox, ood_shot_ids)
    """
    shots_with_both = [
        (sid, v["ip_mean"], v["ne_mean"])
        for sid, v in regime_scalars.items()
        if "ip_mean" in v and "ne_mean" in v
    ]
    if not shots_with_both:
        raise ValueError("No shots with both ip_mean and ne_mean scalars")

    ips = np.array([x[1] for x in shots_with_both])
    nes = np.array([x[2] / 1e19 for x in shots_with_both])

    # Choose thresholds to capture ~ood_fraction_target of shots
    # Start with 80th percentile as the lower bound of the OOD box
    ip_pct = max(0.70, 1.0 - ood_fraction_target * 2)
    ne_pct = max(0.70, 1.0 - ood_fraction_target * 2)
    ip_thresh = float(np.percentile(ips, ip_pct * 100))
    ne_thresh = float(np.percentile(nes, ne_pct * 100))

    ip_max = float(np.max(ips)) * 1.01
    ne_max = float(np.max(nes)) * 1.01

    box = RegimeBox(
        ip_min=ip_thresh,
        ip_max=ip_max,
        ne_min=ne_thresh,
        ne_max=ne_max,
        description=(
            f"High-Iₚ (>{ip_thresh:.0f} kA) × "
            f"high-density (>{ne_thresh:.2f}×10¹⁹ m⁻²) corner"
        ),
    )

    ood_shots = [sid for sid, ip, ne in shots_with_both if box.contains(ip, ne / 1e19)]

    actual_frac = len(ood_shots) / max(len(shots_with_both), 1)
    logger.info(
        "Proposed OOD box: Iₚ>%.0f kA, ne>%.2f×1e19 m⁻² → %d shots (%.1f%%)",
        ip_thresh,
        ne_thresh,
        len(ood_shots),
        100 * actual_frac,
    )

    if actual_frac < 0.04:
        warnings.warn(
            f"OOD box captures only {100 * actual_frac:.1f}% of shots — "
            "consider relaxing thresholds.",
            stacklevel=2,
        )

    return box, ood_shots


def build_splits(
    co_available_shots: list[int],
    regime_scalars: dict[int, dict[str, float]],
    held_out_family: str,
    input_groups: list[str],
    ood_box: RegimeBox | None = None,
    cal_fraction: float = 0.12,
    seed: int = 42,
) -> ShotSplits:
    """Build train / calibration / test_ood_regime splits.

    Parameters
    ----------
    co_available_shots:
        Shot IDs with both the input and target families present.
    regime_scalars:
        Per-shot {ip_mean, ne_mean} from :func:`compute_regime_scalars`.
    held_out_family:
        Name of the held-out diagnostic family.
    input_groups:
        Zarr group names used as model inputs.
    ood_box:
        Pre-defined OOD region.  If None, :func:`propose_ood_box` is used.
    cal_fraction:
        Fraction of non-OOD shots reserved for calibration (default 12 %).
    seed:
        RNG seed for reproducibility.

    Returns
    -------
    ShotSplits
    """
    rng = np.random.default_rng(seed)
    shots = np.array(sorted(co_available_shots))

    # Check for magnetics-target circularity
    circularity_warning = ""
    if held_out_family in ("magnetics", "magnetics_raw"):
        circularity_warning = (
            "CIRCULARITY WARNING: The regime-split axis uses Iₚ from amc "
            "(magnetics). When magnetics is the held-out target, the split "
            "axis leaks target information into the OOD region definition. "
            "The orchestrator must weigh whether this is acceptable or "
            "whether an alternative split axis (e.g. campaign, shape class) "
            "is required."
        )
        logger.warning(circularity_warning)

    # Identify OOD shots
    if ood_box is None:
        shots_with_scalars = {
            sid: regime_scalars[sid] for sid in shots if sid in regime_scalars
        }
        if shots_with_scalars:
            ood_box, ood_shot_ids = propose_ood_box(shots_with_scalars)
        else:
            ood_shot_ids = []
            logger.warning("No regime scalars available — OOD split will be empty")
    else:
        ood_shot_ids = [
            sid
            for sid in shots
            if sid in regime_scalars
            and "ip_mean" in regime_scalars[sid]
            and "ne_mean" in regime_scalars[sid]
            and ood_box.contains(
                regime_scalars[sid]["ip_mean"],
                regime_scalars[sid]["ne_mean"] / 1e19,  # scale to 1e19 units
            )
        ]

    ood_set = frozenset(ood_shot_ids)
    non_ood = np.array([s for s in shots if s not in ood_set])

    # Random train/calibration split of non-OOD shots
    n_cal = max(1, int(round(len(non_ood) * cal_fraction)))
    shuffle_idx = rng.permutation(len(non_ood))
    cal_indices = shuffle_idx[:n_cal]
    train_indices = shuffle_idx[n_cal:]

    calibration_shots = sorted(non_ood[cal_indices].tolist())
    train_shots = sorted(non_ood[train_indices].tolist())
    test_ood = sorted(ood_shot_ids)

    logger.info(
        "Splits: train=%d  cal=%d  ood=%d  (total=%d)",
        len(train_shots),
        len(calibration_shots),
        len(test_ood),
        len(train_shots) + len(calibration_shots) + len(test_ood),
    )

    notes = [
        f"cal_fraction={cal_fraction}, seed={seed}",
        f"co_available input: {len(co_available_shots)} shots",
    ]
    if circularity_warning:
        notes.append(circularity_warning)

    return ShotSplits(
        train=train_shots,
        calibration=calibration_shots,
        test_ood_regime=test_ood,
        regime_scalars={
            sid: regime_scalars[sid]
            for sid in co_available_shots
            if sid in regime_scalars
        },
        ood_box=ood_box,
        held_out_family=held_out_family,
        input_groups=sorted(input_groups),
        circularity_warning=circularity_warning,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Corpus N reporting
# ---------------------------------------------------------------------------


@dataclass
class CorpusNReport:
    """Per-candidate-target corpus size report (Req #2).

    For EACH candidate held-out target AND EACH incremental input
    combination, reports:
        total co-available shots → after regime hold-out
        → train / calibration / test sizes.
    """

    rows: list[dict] = field(default_factory=list)

    def add_row(
        self,
        target: str,
        input_combo: str,
        n_total: int,
        n_ood: int,
        n_non_ood: int,
        n_train: int,
        n_cal: int,
        cal_fraction: float,
    ) -> None:
        self.rows.append(
            {
                "target": target,
                "input_combo": input_combo,
                "n_total_coavailable": n_total,
                "n_ood_regime": n_ood,
                "n_non_ood": n_non_ood,
                "n_train": n_train,
                "n_calibration": n_cal,
                "cal_fraction_used": cal_fraction,
                "cal_adequate": n_cal
                >= 200,  # heuristic: <200 cal shots → coverage unreliable
            }
        )

    def to_dict(self) -> dict:
        return {"rows": self.rows}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    def summary_table(self) -> str:
        """Return a compact ASCII table of corpus N's."""
        header = (
            f"{'target':<25} {'input_combo':<30} "
            f"{'n_total':>8} {'n_ood':>7} {'n_train':>8} {'n_cal':>8} {'cal_ok':>7}"
        )
        lines = [header, "-" * len(header)]
        for r in self.rows:
            lines.append(
                f"{r['target']:<25} {r['input_combo']:<30} "
                f"{r['n_total_coavailable']:>8} {r['n_ood_regime']:>7} "
                f"{r['n_train']:>8} {r['n_calibration']:>8} "
                f"{'yes' if r['cal_adequate'] else 'NO':>7}"
            )
        return "\n".join(lines)
