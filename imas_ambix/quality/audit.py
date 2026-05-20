"""Shot-level and corpus-level quality audit drivers.

Usage
-----
::

    from imas_ambix.quality.audit import audit_shot, audit_corpus, aggregate_corpus
    from imas_ambix.data.paths import LEVEL2_DIR

    report = audit_shot(11766, tier="level2")
    reports = audit_corpus([11766, 11767, 11768], tier="level2", max_workers=4)
    summary = aggregate_corpus(reports)

Architecture
------------
- :func:`audit_shot` opens each available Zarr group for a shot, runs
  all checks, and assembles a :class:`ShotQualityReport`.
- :func:`audit_corpus` fans out over shot IDs using a thread pool.
- :func:`aggregate_corpus` reduces a list of reports to a dict of corpus-level
  statistics (campaign distribution, failure rates, plasma-current histogram,
  etc.) suitable for JSON export.

FAIR-MAST format notes (2026-05-20)
-------------------------------------
FAIR-MAST level-2 is xarray-on-Zarr v3, NOT IDS format.  Quality flags
are calibrated for this reality — no IDS-level checks (no ``ids_properties``,
no ``dd_version``, no ``homogeneous_time``).  The ``usable_for_training``
flag requires all groups to open, magnetics + pulse_schedule present, and
no hard-corruption errors.
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from imas_ambix.data.paths import LEVEL1_DIR, LEVEL2_DIR, Tier

if TYPE_CHECKING:
    from pathlib import Path
from imas_ambix.quality.checks import (
    CheckResult,
    check_dynamic_range,
    check_imas_label_matches_group,
    check_no_all_nan,
    check_open,
    check_time_axis,
    global_min_max,
    nan_fraction_of_ds,
    worst_severity,
)

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

#: Groups audited by default when auditing a level-2 shot.
DEFAULT_GROUPS = (
    "equilibrium",
    "magnetics",
    "pf_active",
    "pf_passive",
    "pulse_schedule",
    "summary",
    "soft_x_rays",
    "interferometer",
    "spectrometer_visible",
    "gas_injection",
    "wall",
)


@dataclass(frozen=True)
class GroupStats:
    """Aggregate statistics and check results for a single Zarr group.

    Parameters
    ----------
    n_variables:
        Number of data variables in the group dataset.
    n_timesteps:
        Length of the ``time`` dimension (0 if absent).
    nan_fraction:
        Overall fraction of NaN values across all floating-point variables.
    min_value:
        Global minimum finite value; ``nan`` if no finite values exist.
    max_value:
        Global maximum finite value; ``nan`` if no finite values exist.
    checks:
        All :class:`~imas_ambix.quality.checks.CheckResult` objects for
        this group, in evaluation order.
    open_ok:
        ``True`` iff ``check_open`` passed (group is readable).
    """

    n_variables: int
    n_timesteps: int
    nan_fraction: float
    min_value: float
    max_value: float
    checks: tuple[CheckResult, ...]
    open_ok: bool


@dataclass
class ShotQualityReport:
    """Quality report for a single FAIR-MAST shot.

    Parameters
    ----------
    shot_id:
        Integer shot number.
    tier:
        ``"level1"`` or ``"level2"``.
    groups_present:
        Zarr sub-groups discovered on disk for this shot.
    per_group:
        Mapping from group name → :class:`GroupStats`.
    metadata:
        Columns from the parquet index row for this shot, e.g.
        ``campaign``, ``plasma_current_max``, ``pulse_duration``.
        Empty dict when no index row was passed.
    quality_flags:
        Derived boolean flags:

        - ``usable_for_training`` — all groups open, magnetics + pulse_schedule
          present, and no error-severity checks in any group.
        - ``has_equilibrium``     — ``"equilibrium"`` group is present and opens.
        - ``has_magnetics``       — ``"magnetics"`` group is present and opens.
        - ``all_groups_open``     — every discovered group opened successfully.
        - ``no_corrupt_nans``     — no variable was all-NaN and no hard-
                                    corruption errors fired in any group.
    overall_severity:
        Maximum severity across all checks; one of ``"info"``, ``"warn"``,
        ``"error"``.
    """

    shot_id: int
    tier: str
    groups_present: tuple[str, ...]
    per_group: dict[str, GroupStats]
    metadata: dict[str, object] = field(default_factory=dict)
    quality_flags: dict[str, bool] = field(default_factory=dict)
    overall_severity: str = "info"

    def to_dict(self) -> dict:
        """Serialise to a plain dict suitable for JSON export."""
        return {
            "shot_id": self.shot_id,
            "tier": self.tier,
            "groups_present": list(self.groups_present),
            "overall_severity": self.overall_severity,
            "quality_flags": self.quality_flags,
            "metadata": {
                k: (v if not isinstance(v, float) or not math.isnan(v) else None)
                for k, v in self.metadata.items()
            },
            "per_group": {
                grp: {
                    "n_variables": gs.n_variables,
                    "n_timesteps": gs.n_timesteps,
                    "nan_fraction": (
                        None
                        if math.isnan(gs.nan_fraction)
                        else round(gs.nan_fraction, 6)
                    ),
                    "min_value": (None if math.isnan(gs.min_value) else gs.min_value),
                    "max_value": (None if math.isnan(gs.max_value) else gs.max_value),
                    "open_ok": gs.open_ok,
                    "checks": [
                        {
                            "name": cr.name,
                            "passed": cr.passed,
                            "severity": cr.severity,
                            "message": cr.message,
                            "metric": cr.metric,
                        }
                        for cr in gs.checks
                    ],
                }
                for grp, gs in self.per_group.items()
            },
        }


# ---------------------------------------------------------------------------
# Per-shot audit
# ---------------------------------------------------------------------------


def _audit_group(shot_path: Path, group: str) -> GroupStats:
    """Run all checks for a single group; return a :class:`GroupStats`."""
    import xarray as xr

    open_result = check_open(shot_path, group)
    all_checks: list[CheckResult] = [open_result]

    if not open_result.passed:
        return GroupStats(
            n_variables=0,
            n_timesteps=0,
            nan_fraction=math.nan,
            min_value=math.nan,
            max_value=math.nan,
            checks=tuple(all_checks),
            open_ok=False,
        )

    # Open with consolidated=False to silence the RuntimeWarning that FAIR-MAST
    # Zarr stores emit when consolidation metadata is absent.
    try:
        ds = xr.open_zarr(str(shot_path / group), consolidated=False)
    except Exception as exc:  # noqa: BLE001
        all_checks.append(
            CheckResult(
                name="open_dataset",
                passed=False,
                severity="error",
                message=f"second open failed: {exc!r}",
            )
        )
        return GroupStats(
            n_variables=0,
            n_timesteps=0,
            nan_fraction=math.nan,
            min_value=math.nan,
            max_value=math.nan,
            checks=tuple(all_checks),
            open_ok=False,
        )

    # --- individual checks -----------------------------------------------
    all_checks.extend(check_no_all_nan(ds))
    all_checks.extend(check_dynamic_range(ds))
    all_checks.extend(check_time_axis(ds))
    all_checks.append(check_imas_label_matches_group(ds, group))

    # --- aggregate stats --------------------------------------------------
    n_vars = len(ds.data_vars)
    n_timesteps = int(ds.sizes.get("time", 0))
    nan_frac = nan_fraction_of_ds(ds)
    gmin, gmax = global_min_max(ds)

    return GroupStats(
        n_variables=n_vars,
        n_timesteps=n_timesteps,
        nan_fraction=nan_frac,
        min_value=gmin,
        max_value=gmax,
        checks=tuple(all_checks),
        open_ok=True,
    )


def _discover_groups(shot_path: Path) -> tuple[str, ...]:
    """Return sorted sub-directory names inside *shot_path* (Zarr groups)."""
    if not shot_path.exists():
        return ()
    return tuple(
        sorted(
            p.name
            for p in shot_path.iterdir()
            if p.is_dir() and not p.name.startswith(".")
        )
    )


def audit_shot(
    shot_id: int,
    tier: Tier = "level2",
    index_row: dict | None = None,
) -> ShotQualityReport:
    """Audit a single shot by opening every discovered Zarr group.

    Parameters
    ----------
    shot_id:
        Integer shot number.
    tier:
        ``"level1"`` or ``"level2"``.
    index_row:
        Optional dict of parquet metadata columns for this shot
        (e.g. from ``load_index().loc[shot_id]``).  Keys of interest:
        ``campaign``, ``plasma_current_max``, ``pulse_duration``.
    """
    root = LEVEL1_DIR if tier == "level1" else LEVEL2_DIR
    shot_path = root / f"{shot_id}.zarr"

    groups = _discover_groups(shot_path)
    per_group: dict[str, GroupStats] = {}
    all_checks_flat: list[CheckResult] = []

    for grp in groups:
        gs = _audit_group(shot_path, grp)
        per_group[grp] = gs
        all_checks_flat.extend(gs.checks)

    overall = worst_severity(all_checks_flat) if all_checks_flat else "error"
    if not groups:
        # Shot path missing or empty — treat as error
        all_checks_flat.append(
            CheckResult(
                name="shot_exists",
                passed=False,
                severity="error",
                message=f"shot path missing or empty: {shot_path}",
            )
        )
        overall = "error"

    # --- quality flags ----------------------------------------------------
    def _group_open(name: str) -> bool:
        gs = per_group.get(name)
        return gs is not None and gs.open_ok

    def _any_check_has_error() -> bool:
        return any(
            c.severity == "error" and not c.passed for c in all_checks_flat
        )

    def _all_groups_open() -> bool:
        return all(gs.open_ok for gs in per_group.values())

    def _no_corrupt_nans() -> bool:
        """True iff no all-NaN variables and no hard-corruption errors."""
        for c in all_checks_flat:
            # Fired by check_no_all_nan or check_dynamic_range (inf/1e25)
            if (
                not c.passed
                and c.severity == "error"
                and (
                    c.name.startswith("no_all_nan:")
                    or c.name.startswith("dynamic_range:")
                )
            ):
                return False
        return True

    has_magnetics = _group_open("magnetics")
    has_pulse_schedule = _group_open("pulse_schedule")

    quality_flags = {
        "usable_for_training": (
            _all_groups_open()
            and has_magnetics
            and has_pulse_schedule
            and not _any_check_has_error()
        ),
        "has_equilibrium": _group_open("equilibrium"),
        "has_magnetics": has_magnetics,
        "all_groups_open": _all_groups_open(),
        "no_corrupt_nans": _no_corrupt_nans(),
    }

    # --- parquet metadata -------------------------------------------------
    metadata: dict[str, object] = {}
    if index_row is not None:
        for key in (
            "campaign",
            "shot_id",
            "plasma_current_max",
            "plasma_current_average",
            "pulse_duration",
            "ip_max",
        ):
            if key in index_row:
                metadata[key] = index_row[key]

    return ShotQualityReport(
        shot_id=shot_id,
        tier=tier,
        groups_present=groups,
        per_group=per_group,
        metadata=metadata,
        quality_flags=quality_flags,
        overall_severity=overall,
    )


# ---------------------------------------------------------------------------
# Corpus-level audit
# ---------------------------------------------------------------------------


def audit_corpus(
    shot_ids: list[int],
    tier: Tier = "level2",
    max_workers: int = 8,
    index_rows: dict[int, dict] | None = None,
) -> list[ShotQualityReport]:
    """Audit multiple shots in parallel.

    Parameters
    ----------
    shot_ids:
        List of shot IDs to audit.
    tier:
        ``"level1"`` or ``"level2"``.
    max_workers:
        Thread-pool concurrency.  I/O-bound Zarr opens benefit from
        parallelism even in the GIL.
    index_rows:
        Optional mapping from shot_id → parquet row dict.
    """
    reports: list[ShotQualityReport] = []
    rows = index_rows or {}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                audit_shot,
                sid,
                tier,
                rows.get(sid),
            ): sid
            for sid in shot_ids
        }
        for fut in as_completed(futures):
            sid = futures[fut]
            try:
                reports.append(fut.result())
            except Exception as exc:  # noqa: BLE001
                # Produce a minimal error report rather than dropping the shot.
                reports.append(
                    ShotQualityReport(
                        shot_id=sid,
                        tier=tier,
                        groups_present=(),
                        per_group={},
                        metadata={},
                        quality_flags={
                            "usable_for_training": False,
                            "has_equilibrium": False,
                            "has_magnetics": False,
                            "all_groups_open": False,
                            "no_corrupt_nans": False,
                        },
                        overall_severity="error",
                    )
                )
                print(f"warn: audit_shot({sid}) raised: {exc!r}")

    # Return in the same order as the input list for reproducibility.
    id_order = {sid: i for i, sid in enumerate(shot_ids)}
    reports.sort(key=lambda r: id_order.get(r.shot_id, len(shot_ids)))
    return reports


# ---------------------------------------------------------------------------
# Corpus aggregation
# ---------------------------------------------------------------------------


def aggregate_corpus(reports: list[ShotQualityReport]) -> dict:
    """Reduce a list of :class:`ShotQualityReport` objects to corpus statistics.

    Returns a dict with the following keys:

    ``n_total``
        Total shots audited.
    ``n_passed``
        Shots with ``overall_severity == "info"`` (all checks green).
    ``n_warned``
        Shots with ``overall_severity == "warn"``.
    ``n_failed``
        Shots with ``overall_severity == "error"``.
    ``pass_rate``
        Fraction of shots that passed.
    ``usable_for_training``
        Count of shots flagged usable.
    ``campaign_distribution``
        ``{campaign_name: count}`` sorted by count desc.
    ``plasma_current_deciles``
        10 equal-width decile boundaries over ``plasma_current_max``.
    ``top_failure_modes``
        Top-5 failing check names by frequency.
    ``group_open_rates``
        ``{group_name: fraction_open}`` across all shots.
    ``quality_flag_rates``
        ``{flag: fraction_true}`` for each quality flag.
    """
    n = len(reports)
    if n == 0:
        return {
            "n_total": 0,
            "n_passed": 0,
            "n_warned": 0,
            "n_failed": 0,
            "pass_rate": 0.0,
            "usable_for_training": 0,
            "campaign_distribution": {},
            "plasma_current_deciles": [],
            "top_failure_modes": [],
            "group_open_rates": {},
            "quality_flag_rates": {},
        }

    n_passed = sum(1 for r in reports if r.overall_severity == "info")
    n_warned = sum(1 for r in reports if r.overall_severity == "warn")
    n_failed = sum(1 for r in reports if r.overall_severity == "error")

    usable = sum(
        1 for r in reports if r.quality_flags.get("usable_for_training", False)
    )

    # Campaign distribution
    camp_counts: dict[str, int] = {}
    for r in reports:
        camp = str(r.metadata.get("campaign", "unknown"))
        camp_counts[camp] = camp_counts.get(camp, 0) + 1
    campaign_distribution = dict(sorted(camp_counts.items(), key=lambda kv: -kv[1]))

    # Plasma current deciles
    import math

    currents = [
        float(r.metadata["plasma_current_max"])
        for r in reports
        if "plasma_current_max" in r.metadata
        and r.metadata["plasma_current_max"] is not None
        and not math.isnan(float(r.metadata["plasma_current_max"]))
    ]
    if currents:
        import numpy as np

        deciles = [float(v) for v in np.percentile(currents, range(0, 110, 10))]
    else:
        deciles = []

    # Top failure modes
    fail_counts: dict[str, int] = {}
    for r in reports:
        for gs in r.per_group.values():
            for c in gs.checks:
                if not c.passed:
                    # Strip per-variable suffix for aggregation
                    base = c.name.split(":")[0]
                    fail_counts[base] = fail_counts.get(base, 0) + 1
    top_failures = sorted(fail_counts.items(), key=lambda kv: -kv[1])[:5]

    # Group open rates
    group_open_counts: dict[str, int] = {}
    group_total_counts: dict[str, int] = {}
    for r in reports:
        for grp, gs in r.per_group.items():
            group_total_counts[grp] = group_total_counts.get(grp, 0) + 1
            if gs.open_ok:
                group_open_counts[grp] = group_open_counts.get(grp, 0) + 1
    group_open_rates = {
        grp: group_open_counts.get(grp, 0) / group_total_counts[grp]
        for grp in group_total_counts
    }

    # Quality flag rates
    all_flags: set[str] = set()
    for r in reports:
        all_flags.update(r.quality_flags.keys())
    quality_flag_rates = {
        flag: sum(1 for r in reports if r.quality_flags.get(flag, False)) / n
        for flag in sorted(all_flags)
    }

    return {
        "n_total": n,
        "n_passed": n_passed,
        "n_warned": n_warned,
        "n_failed": n_failed,
        "pass_rate": n_passed / n,
        "usable_for_training": usable,
        "campaign_distribution": campaign_distribution,
        "plasma_current_deciles": deciles,
        "top_failure_modes": [
            {"check": name, "count": cnt} for name, cnt in top_failures
        ],
        "group_open_rates": group_open_rates,
        "quality_flag_rates": quality_flag_rates,
    }
