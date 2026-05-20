"""Per-shot and corpus data-quality checks for the FAIR-MAST Zarr corpus.

Public API
----------
- :mod:`imas_ambix.quality.checks` — individual per-group check functions
- :mod:`imas_ambix.quality.audit`  — shot-level and corpus-level audit drivers
"""

from __future__ import annotations

from imas_ambix.quality.audit import (
    GroupStats,
    ShotQualityReport,
    aggregate_corpus,
    audit_corpus,
    audit_shot,
)
from imas_ambix.quality.checks import (
    CheckResult,
    check_dynamic_range,
    check_imas_label_matches_group,
    check_imas_pointer,
    check_no_all_nan,
    check_open,
    check_time_axis,
)

__all__ = [
    # checks
    "CheckResult",
    "check_open",
    "check_no_all_nan",
    "check_dynamic_range",
    "check_time_axis",
    "check_imas_pointer",
    "check_imas_label_matches_group",
    # audit
    "GroupStats",
    "ShotQualityReport",
    "audit_shot",
    "audit_corpus",
    "aggregate_corpus",
]
