"""Training-grade shot filter for the FAIR-MAST level-2 corpus.

§4 of the data-quality plan (updated 2026-05-20) defines the acceptance gates
for shots that are suitable for world-model v0 training.  This module
implements those gates as a deterministic, re-runnable filter over a corpus
audit JSON produced by ``ambix data audit --output``.

Gates (all must pass):
    (a) magnetics complete — ``quality_flags.has_magnetics`` is True
    (b) equilibrium time-slice present — ``quality_flags.has_equilibrium``
    (c) all groups open — ``quality_flags.all_groups_open`` is True
    (d) no NaN in summary scalars — ``quality_flags.no_corrupt_nans`` True
    (e) category exclusion — shots whose groups intersect
        ``drop_categories`` are excluded (e.g. ``charge_exchange``)

Note on camera: level-2 FAIR-MAST shots do not carry camera groups (rba/rbb/
rir) — those live in the level-1 Zarr store and are handled by the frame
tokenizer separately.  The camera open-check was dropped from the level-2
acceptance gate in the 2026-05-20 plan update (§10).

Locked decision (data-quality, 2026-05-20):
    ``drop-charge-exchange → yes`` — charge_exchange is dropped entirely
    from the v0 corpus due to 12–28 orders of magnitude beyond physical
    range.  The ``drop_categories`` default captures this.

Open question (data-quality q1):
    Should charge_exchange be re-included in v1 once FAIR-MAST fixes the
    encoding?  Not resolved here — see the plan.

Usage::

    from pathlib import Path
    from imas_ambix.data.training_grade import TrainingGradeFilter

    filt = TrainingGradeFilter(audit_path=Path("/tmp/audit-full.json"))
    counts = filt.write_manifest(Path("training-grade-shots.json"))
    print(counts)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path  # noqa: TC003 — runtime: dataclass field + out_path.mkdir
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from imas_ambix.quality.audit import ShotQualityReport


# ---------------------------------------------------------------------------
# Exclusion reasons
# ---------------------------------------------------------------------------

REASON_MAGNETICS = "magnetics_incomplete"
REASON_NO_EQUILIBRIUM = "no_equilibrium"
REASON_GROUPS_NOT_OPEN = "groups_not_open"
REASON_CORRUPT_NANS = "corrupt_nans"
REASON_DROPPED_CATEGORY = "dropped_category"

# Keep REASON_NO_CAMERA for backwards compatibility (test-code compatibility)
REASON_NO_CAMERA = "no_camera_channel"


# ---------------------------------------------------------------------------
# Filter dataclass
# ---------------------------------------------------------------------------


@dataclass
class TrainingGradeFilter:
    """Apply §4 training-grade gates to a corpus audit.

    Parameters
    ----------
    audit_path:
        Path to a JSON file produced by
        ``ambix data audit --output <path>``.  The file must have a
        ``per_shot`` list of serialised
        :class:`~imas_ambix.quality.audit.ShotQualityReport` dicts
        and a top-level ``shot_ids`` list.
    drop_categories:
        Group names (Zarr sub-directory names) to treat as excluded
        regardless of other gates.  Defaults to
        ``("charge_exchange",)`` per the locked ``drop-charge-exchange``
        decision.
    """

    audit_path: Path
    drop_categories: tuple[str, ...] = ("charge_exchange",)

    # ------------------------------------------------------------------ #
    # Core gate
    # ------------------------------------------------------------------ #

    def apply(self, report: ShotQualityReport) -> tuple[bool, list[str]]:
        """Evaluate all §4 gates against a single :class:`ShotQualityReport`.

        Returns
        -------
        (passed, reasons)
            *passed* is ``True`` iff the shot is training-grade.
            *reasons* is a (possibly empty) list of exclusion-reason
            strings — populated only when *passed* is ``False``.
        """
        reasons: list[str] = []

        # Gate (a): magnetics complete
        if not report.quality_flags.get("has_magnetics", False):
            reasons.append(REASON_MAGNETICS)

        # Gate (b): equilibrium time-slice present
        if not report.quality_flags.get("has_equilibrium", False):
            reasons.append(REASON_NO_EQUILIBRIUM)

        # Gate (c): all groups open
        if not report.quality_flags.get("all_groups_open", True):
            reasons.append(REASON_GROUPS_NOT_OPEN)

        # Gate (d): no corrupt NaNs in summary scalars
        if not report.quality_flags.get("no_corrupt_nans", True):
            reasons.append(REASON_CORRUPT_NANS)

        # Gate (e): category exclusion (locked: drop-charge-exchange)
        present = set(report.groups_present)
        drop_set = set(self.drop_categories)
        if present & drop_set:
            reasons.append(REASON_DROPPED_CATEGORY)

        return len(reasons) == 0, reasons

    # ------------------------------------------------------------------ #
    # Apply to dict (audit JSON form)
    # ------------------------------------------------------------------ #

    def _apply_to_dict(self, shot_dict: dict) -> tuple[bool, list[str]]:
        """Apply gates to a raw dict from the audit JSON ``per_shot`` list."""
        reasons: list[str] = []
        flags = shot_dict.get("quality_flags", {})
        groups_present: set[str] = set(shot_dict.get("groups_present", []))

        # Gate (a): magnetics complete
        if not flags.get("has_magnetics", False):
            reasons.append(REASON_MAGNETICS)

        # Gate (b): equilibrium time-slice present
        if not flags.get("has_equilibrium", False):
            reasons.append(REASON_NO_EQUILIBRIUM)

        # Gate (c): all groups open
        if not flags.get("all_groups_open", True):
            reasons.append(REASON_GROUPS_NOT_OPEN)

        # Gate (d): no corrupt NaNs
        if not flags.get("no_corrupt_nans", True):
            reasons.append(REASON_CORRUPT_NANS)

        # Gate (e): category exclusion
        drop_set = set(self.drop_categories)
        if groups_present & drop_set:
            reasons.append(REASON_DROPPED_CATEGORY)

        return len(reasons) == 0, reasons

    # ------------------------------------------------------------------ #
    # Manifest writer
    # ------------------------------------------------------------------ #

    def write_manifest(self, out_path: Path) -> dict:
        """Load the audit, apply gates, and write ``training-grade-shots.json``.

        Parameters
        ----------
        out_path:
            Destination path for the output JSON manifest.

        Returns
        -------
        dict
            Counts summary::

                {
                    "n_total":        int,   # shots in the audit
                    "n_passed":       int,   # training-grade shots
                    "n_excluded":     int,   # shots failing ≥1 gate
                    "by_reason":      {reason: int},
                    "drop_categories": [...],
                }
        """
        payload = json.loads(self.audit_path.read_text(encoding="utf-8"))
        per_shot: list[dict] = payload.get("per_shot", [])

        passed_ids: list[int] = []
        reason_counts: dict[str, int] = {}

        for shot in per_shot:
            ok, reasons = self._apply_to_dict(shot)
            if ok:
                passed_ids.append(int(shot["shot_id"]))
            else:
                for r in reasons:
                    reason_counts[r] = reason_counts.get(r, 0) + 1

        passed_ids.sort()
        n_total = len(per_shot)
        n_passed = len(passed_ids)
        n_excluded = n_total - n_passed

        manifest = {
            "generated_at": datetime.now(UTC).isoformat(),
            "audit_path": str(self.audit_path),
            "drop_categories": list(self.drop_categories),
            "gates": [
                "magnetics_complete",
                "equilibrium_present",
                "all_groups_open",
                "no_corrupt_nans",
                "category_not_dropped",
            ],
            "n_total": n_total,
            "n_passed": n_passed,
            "n_excluded": n_excluded,
            "pass_rate": round(n_passed / n_total, 4) if n_total else 0.0,
            "by_reason": dict(sorted(reason_counts.items(), key=lambda kv: -kv[1])),
            "shot_ids": passed_ids,
        }

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return {
            "n_total": n_total,
            "n_passed": n_passed,
            "n_excluded": n_excluded,
            "by_reason": manifest["by_reason"],
            "drop_categories": list(self.drop_categories),
        }
