"""Unit tests for imas_ambix.data.training_grade.

Tests use synthetic data only — no network or SLURM access required.
Each §4 gate is tested in isolation, and the count-by-reason accounting
is verified on a mixed corpus.

Gate overview (level-2 FAIR-MAST corpus):
    (a) magnetics_complete — quality_flags.has_magnetics is True
    (b) equilibrium_present — quality_flags.has_equilibrium is True
    (c) all_groups_open — quality_flags.all_groups_open is True
    (d) no_corrupt_nans — quality_flags.no_corrupt_nans is True
    (e) category_not_dropped — groups_present does not intersect drop_categories

Note: the camera-channel gate (rba/rbb/rir) applies to level-1 Zarr
stores and was dropped from the level-2 acceptance gate in the
2026-05-20 plan update (§10).  It is NOT a gate here.
"""

from __future__ import annotations

import json
from pathlib import Path  # noqa: TC003 — runtime: _audit_json / _shot_dict functions

from imas_ambix.data.training_grade import (
    REASON_CORRUPT_NANS,
    REASON_DROPPED_CATEGORY,
    REASON_GROUPS_NOT_OPEN,
    REASON_MAGNETICS,
    REASON_NO_EQUILIBRIUM,
    TrainingGradeFilter,
)

# ---------------------------------------------------------------------------
# Synthetic audit JSON builders
# ---------------------------------------------------------------------------


def _shot_dict(
    shot_id: int = 99999,
    groups_present: list[str] | None = None,
    has_magnetics: bool = True,
    has_equilibrium: bool = True,
    all_groups_open: bool = True,
    no_corrupt_nans: bool = True,
) -> dict:
    """Build a minimal shot dict matching the audit JSON ``per_shot`` schema."""
    if groups_present is None:
        groups_present = ["magnetics", "equilibrium", "summary", "pulse_schedule"]
    return {
        "shot_id": shot_id,
        "tier": "level2",
        "groups_present": groups_present,
        "overall_severity": "info",
        "quality_flags": {
            "has_magnetics": has_magnetics,
            "has_equilibrium": has_equilibrium,
            "all_groups_open": all_groups_open,
            "no_corrupt_nans": no_corrupt_nans,
            "usable_for_training": has_magnetics
            and has_equilibrium
            and all_groups_open
            and no_corrupt_nans,
        },
        "metadata": {},
        "per_group": {},
    }


def _audit_json(shots: list[dict], tmp_path: Path) -> Path:
    """Write a minimal audit JSON to *tmp_path* and return the path."""
    payload = {
        "tier": "level2",
        "shot_ids": [s["shot_id"] for s in shots],
        "aggregate": {"n_total": len(shots)},
        "per_shot": shots,
    }
    p = tmp_path / "audit.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Gate (a): magnetics_complete
# ---------------------------------------------------------------------------


def test_gate_magnetics_missing(tmp_path):
    shot = _shot_dict(has_magnetics=False)
    audit = _audit_json([shot], tmp_path)
    filt = TrainingGradeFilter(audit_path=audit)
    out = tmp_path / "out.json"
    counts = filt.write_manifest(out)
    assert counts["n_passed"] == 0
    assert counts["n_excluded"] == 1
    assert REASON_MAGNETICS in counts["by_reason"]


def test_gate_magnetics_present(tmp_path):
    shot = _shot_dict(has_magnetics=True)
    audit = _audit_json([shot], tmp_path)
    filt = TrainingGradeFilter(audit_path=audit)
    out = tmp_path / "out.json"
    counts = filt.write_manifest(out)
    assert counts["n_passed"] == 1
    assert REASON_MAGNETICS not in counts["by_reason"]


# ---------------------------------------------------------------------------
# Gate (b): equilibrium_present
# ---------------------------------------------------------------------------


def test_gate_no_equilibrium(tmp_path):
    shot = _shot_dict(has_equilibrium=False)
    audit = _audit_json([shot], tmp_path)
    filt = TrainingGradeFilter(audit_path=audit)
    out = tmp_path / "out.json"
    counts = filt.write_manifest(out)
    assert counts["n_passed"] == 0
    assert REASON_NO_EQUILIBRIUM in counts["by_reason"]


def test_gate_equilibrium_present(tmp_path):
    shot = _shot_dict(has_equilibrium=True)
    audit = _audit_json([shot], tmp_path)
    filt = TrainingGradeFilter(audit_path=audit)
    out = tmp_path / "out.json"
    counts = filt.write_manifest(out)
    assert counts["n_passed"] == 1
    assert REASON_NO_EQUILIBRIUM not in counts["by_reason"]


# ---------------------------------------------------------------------------
# Gate (c): all_groups_open
# ---------------------------------------------------------------------------


def test_gate_groups_not_open(tmp_path):
    shot = _shot_dict(all_groups_open=False)
    audit = _audit_json([shot], tmp_path)
    filt = TrainingGradeFilter(audit_path=audit)
    out = tmp_path / "out.json"
    counts = filt.write_manifest(out)
    assert counts["n_passed"] == 0
    assert REASON_GROUPS_NOT_OPEN in counts["by_reason"]


def test_gate_all_groups_open_passes(tmp_path):
    shot = _shot_dict(all_groups_open=True)
    audit = _audit_json([shot], tmp_path)
    filt = TrainingGradeFilter(audit_path=audit)
    out = tmp_path / "out.json"
    counts = filt.write_manifest(out)
    assert counts["n_passed"] == 1
    assert REASON_GROUPS_NOT_OPEN not in counts["by_reason"]


# ---------------------------------------------------------------------------
# Gate (d): no_corrupt_nans
# ---------------------------------------------------------------------------


def test_gate_corrupt_nans(tmp_path):
    shot = _shot_dict(no_corrupt_nans=False)
    audit = _audit_json([shot], tmp_path)
    filt = TrainingGradeFilter(audit_path=audit)
    out = tmp_path / "out.json"
    counts = filt.write_manifest(out)
    assert counts["n_passed"] == 0
    assert REASON_CORRUPT_NANS in counts["by_reason"]


# ---------------------------------------------------------------------------
# Gate (e): drop_categories — locked decision: drop-charge-exchange
# ---------------------------------------------------------------------------


def test_drop_charge_exchange_default(tmp_path):
    """charge_exchange in groups_present → always excluded (default drop_categories)."""
    shot = _shot_dict(
        groups_present=["magnetics", "equilibrium", "summary", "charge_exchange"]
    )
    audit = _audit_json([shot], tmp_path)
    filt = TrainingGradeFilter(audit_path=audit)
    out = tmp_path / "out.json"
    counts = filt.write_manifest(out)
    assert counts["n_passed"] == 0
    assert REASON_DROPPED_CATEGORY in counts["by_reason"]


def test_drop_charge_exchange_absent_passes(tmp_path):
    """Shot without charge_exchange passes gate (e) even with all others present."""
    shot = _shot_dict(groups_present=["magnetics", "equilibrium", "summary"])
    audit = _audit_json([shot], tmp_path)
    filt = TrainingGradeFilter(audit_path=audit)
    out = tmp_path / "out.json"
    counts = filt.write_manifest(out)
    assert counts["n_passed"] == 1
    assert REASON_DROPPED_CATEGORY not in counts["by_reason"]


def test_drop_categories_custom(tmp_path):
    """A custom drop_categories tuple excludes shots with that group."""
    shot = _shot_dict(groups_present=["magnetics", "equilibrium", "thomson_scattering"])
    audit = _audit_json([shot], tmp_path)
    filt = TrainingGradeFilter(
        audit_path=audit, drop_categories=("thomson_scattering",)
    )
    out = tmp_path / "out.json"
    counts = filt.write_manifest(out)
    assert counts["n_passed"] == 0
    assert REASON_DROPPED_CATEGORY in counts["by_reason"]


def test_drop_categories_empty_does_not_exclude(tmp_path):
    """Empty drop_categories → no category-based exclusions."""
    shot = _shot_dict(groups_present=["magnetics", "equilibrium", "charge_exchange"])
    audit = _audit_json([shot], tmp_path)
    filt = TrainingGradeFilter(audit_path=audit, drop_categories=())
    out = tmp_path / "out.json"
    counts = filt.write_manifest(out)
    # charge_exchange is present but drop_categories is empty; passes all gates
    assert counts["n_passed"] == 1


# ---------------------------------------------------------------------------
# Level-2 corpus does NOT require camera groups
# ---------------------------------------------------------------------------


def test_no_camera_groups_still_passes(tmp_path):
    """Level-2 shots without camera groups (rba/rbb/rir) are NOT excluded.

    Camera groups live in the level-1 store and were dropped from the
    level-2 acceptance gate in the 2026-05-20 plan update (§10).
    """
    # Typical level-2 groups — no rba/rbb/rir
    shot = _shot_dict(
        groups_present=[
            "equilibrium",
            "magnetics",
            "pf_active",
            "pf_passive",
            "pulse_schedule",
            "summary",
            "wall",
        ]
    )
    audit = _audit_json([shot], tmp_path)
    filt = TrainingGradeFilter(audit_path=audit)
    out = tmp_path / "out.json"
    counts = filt.write_manifest(out)
    assert counts["n_passed"] == 1, "shot without camera groups must pass"


# ---------------------------------------------------------------------------
# Multiple-gate failures
# ---------------------------------------------------------------------------


def test_multiple_gates_all_reasons_recorded(tmp_path):
    """A shot failing multiple gates accumulates all reason strings."""
    shot = _shot_dict(
        has_magnetics=False,
        has_equilibrium=False,
        all_groups_open=False,
        no_corrupt_nans=False,
        groups_present=["charge_exchange"],
    )
    audit = _audit_json([shot], tmp_path)
    filt = TrainingGradeFilter(audit_path=audit)
    out = tmp_path / "out.json"
    counts = filt.write_manifest(out)
    assert counts["n_passed"] == 0
    br = counts["by_reason"]
    assert REASON_MAGNETICS in br
    assert REASON_NO_EQUILIBRIUM in br
    assert REASON_GROUPS_NOT_OPEN in br
    assert REASON_CORRUPT_NANS in br
    assert REASON_DROPPED_CATEGORY in br


# ---------------------------------------------------------------------------
# Counts-by-reason accuracy on a mixed corpus
# ---------------------------------------------------------------------------


def test_counts_by_reason_mixed_corpus(tmp_path):
    """Counts-by-reason are accurate on a heterogeneous synthetic corpus."""
    shots = [
        # shot 1: passes all gates
        _shot_dict(shot_id=1),
        # shot 2: fails magnetics
        _shot_dict(shot_id=2, has_magnetics=False),
        # shot 3: fails magnetics + groups not open
        _shot_dict(shot_id=3, has_magnetics=False, all_groups_open=False),
        # shot 4: fails equilibrium
        _shot_dict(shot_id=4, has_equilibrium=False),
        # shot 5: passes all gates
        _shot_dict(shot_id=5),
        # shot 6: charge_exchange present → excluded
        _shot_dict(
            shot_id=6,
            groups_present=["magnetics", "equilibrium", "charge_exchange"],
        ),
    ]
    audit = _audit_json(shots, tmp_path)
    filt = TrainingGradeFilter(audit_path=audit)
    out = tmp_path / "out.json"
    counts = filt.write_manifest(out)

    assert counts["n_total"] == 6
    assert counts["n_passed"] == 2  # shots 1 and 5
    assert counts["n_excluded"] == 4

    br = counts["by_reason"]
    assert br[REASON_MAGNETICS] == 2  # shots 2 and 3
    assert br[REASON_GROUPS_NOT_OPEN] == 1  # shot 3
    assert br[REASON_NO_EQUILIBRIUM] == 1  # shot 4
    assert br[REASON_DROPPED_CATEGORY] == 1  # shot 6

    # Verify the output JSON
    manifest = json.loads(out.read_text())
    assert sorted(manifest["shot_ids"]) == [1, 5]
    assert manifest["n_passed"] == 2
    assert manifest["n_total"] == 6
    assert 0.0 < manifest["pass_rate"] < 1.0


# ---------------------------------------------------------------------------
# Output manifest schema
# ---------------------------------------------------------------------------


def test_manifest_schema(tmp_path):
    """Output JSON has all required top-level keys."""
    shot = _shot_dict()
    audit = _audit_json([shot], tmp_path)
    filt = TrainingGradeFilter(audit_path=audit)
    out = tmp_path / "out.json"
    filt.write_manifest(out)

    m = json.loads(out.read_text())
    for key in (
        "generated_at",
        "audit_path",
        "drop_categories",
        "gates",
        "n_total",
        "n_passed",
        "n_excluded",
        "pass_rate",
        "by_reason",
        "shot_ids",
    ):
        assert key in m, f"missing key: {key!r}"

    assert isinstance(m["shot_ids"], list)
    assert m["drop_categories"] == ["charge_exchange"]
