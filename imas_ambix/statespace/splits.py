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
    ip_mean   : flat-top mean Iₚ (MA), from amc/plasma_current
    ne_mean   : flat-top mean line density (m⁻³), from ane/density

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
# Regime scalar computation
# ---------------------------------------------------------------------------

_FLAT_TOP_TRIM = 0.1  # trim fraction at each end of shot for flat-top estimate


def _compute_scalar_from_zarr(
    shot_zarr_path: Path,
    group: str,
    channel: str,
    trim: float = _FLAT_TOP_TRIM,
) -> float | None:
    """Open one channel from a shot's Zarr and return the flat-top mean.

    Returns None on any read error (missing group, missing channel, etc.).
    """
    import zarr  # noqa: PLC0415

    grp_path = shot_zarr_path / group
    if not grp_path.exists():
        return None
    try:
        store = zarr.open_group(str(shot_zarr_path), mode="r")
        data = np.asarray(store[group][channel])
        if data.ndim != 1 or data.size < 4:
            return None
        # Flat-top window: middle (1-2*trim) fraction
        n = data.size
        lo = int(round(n * trim))
        hi = int(round(n * (1 - trim)))
        if lo >= hi:
            lo, hi = 0, n
        return float(np.nanmean(data[lo:hi]))
    except Exception as e:
        logger.debug("Cannot read %s/%s/%s: %s", shot_zarr_path.name, group, channel, e)
        return None


def compute_regime_scalars(
    shot_ids: list[int],
    level1_dir: Path,
    max_workers: int = 8,
) -> dict[int, dict[str, float]]:
    """Compute per-shot operating-point scalars for regime-split definition.

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
    dict mapping shot_id → {``"ip_mean"``: float (MA), ``"ne_mean"``: float (m⁻³)}
    For shots where a scalar cannot be read, the key is absent.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: PLC0415

    from imas_ambix.data.paths import LEVEL1_DIR  # noqa: PLC0415

    root = level1_dir or LEVEL1_DIR

    def _worker(sid: int) -> tuple[int, dict[str, float]]:
        shot_path = root / f"{sid}.zarr"
        scalars: dict[str, float] = {}
        ip = _compute_scalar_from_zarr(shot_path, "amc", "plasma_current")
        if ip is not None:
            # amc/plasma_current is in kA (MAST convention; range ~100–900 kA)
            scalars["ip_mean"] = abs(ip)  # kA
        ne = _compute_scalar_from_zarr(shot_path, "ane", "density")
        if ne is not None:
            scalars["ne_mean"] = float(ne)
        return sid, scalars

    results: dict[int, dict[str, float]] = {}
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_worker, sid): sid for sid in shot_ids}
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
        Line density range in 10^19 m⁻³ (i.e. ne_mean / 1e19).
    description:
        Human-readable label (e.g. "high-current / high-density corner").
    """

    ip_min: float
    ip_max: float
    ne_min: float  # in units of 1e19 m⁻³
    ne_max: float
    description: str = ""

    def contains(self, ip_mean: float, ne_mean: float) -> bool:
        """Return True if (ip_mean [kA], ne_mean [10¹⁹ m⁻³]) falls inside this box.

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
            f"high-density (>{ne_thresh:.2f}×10¹⁹ m⁻³) corner"
        ),
    )

    ood_shots = [sid for sid, ip, ne in shots_with_both if box.contains(ip, ne / 1e19)]

    actual_frac = len(ood_shots) / max(len(shots_with_both), 1)
    logger.info(
        "Proposed OOD box: Iₚ>%.0f kA, ne>%.2f×1e19 m⁻³ → %d shots (%.1f%%)",
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
