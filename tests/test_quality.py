"""Tests for ``imas_ambix.quality`` — checks, audit, and CLI.

Synthetic Zarr stores are written to ``tmp_path`` so tests run without
access to the live mirror at ``/work/projects/imas_gpu/mast``.

Test inventory
--------------
Individual checks (12 cases):
    test_check_open_pass               — check_open on valid group
    test_check_open_fail               — check_open on missing path
    test_check_no_all_nan_pass         — all-finite variable
    test_check_no_all_nan_fail         — all-NaN variable returns error
    test_check_dynamic_range_pass      — normal range passes
    test_check_dynamic_range_fail      — extreme values produce warn
    test_check_time_axis_pass          — monotonic time passes
    test_check_time_axis_fail_nonmono  — non-monotonic time is error
    test_check_time_axis_missing       — no time dim is warn
    test_check_homogeneous_time_flag   — imas attr present/absent
    test_check_dd_version_warn         — no version attr → warn
    test_check_dd_version_mismatch     — wrong version → error

Audit:
    test_audit_shot_synthetic_clean    — all checks pass on good shot
    test_audit_shot_all_nan            — NaN var is flagged
    test_audit_shot_missing            — absent shot path produces error report
    test_audit_corpus_three_shots      — corpus audit over 3 shots

Aggregate:
    test_aggregate_corpus_counts       — n_passed / n_warned / n_failed
    test_aggregate_corpus_campaign     — campaign distribution

CLI:
    test_audit_cmd_help                — --help smoke test
    test_audit_cmd_end_to_end          — full run with monkeypatched LEVEL2_DIR
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import patch

if TYPE_CHECKING:
    from pathlib import Path

import numpy as np
import pytest
import xarray as xr

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_zarr_group(
    shot_path: Path,
    group: str,
    *,
    time: np.ndarray | None = None,
    variables: dict[str, np.ndarray] | None = None,
    attrs: dict | None = None,
) -> Path:
    """Write a minimal Zarr group under *shot_path/group*.

    Parameters
    ----------
    shot_path: root of the shot store (e.g. ``tmp_path / '99001.zarr'``)
    group: group name (e.g. ``"magnetics"``)
    time: optional 1-D float64 array for the ``time`` coordinate
    variables: mapping name → ndarray (must have ``time`` as first dim if given)
    attrs: dataset-level attributes
    """
    shot_path.mkdir(parents=True, exist_ok=True)
    grp_path = shot_path / group
    grp_path.mkdir(parents=True, exist_ok=True)

    default_time = np.linspace(0.0, 1.0, 8) if time is None else time
    default_vars: dict[str, np.ndarray] = (
        {"signal": np.ones((len(default_time),), dtype=np.float64)}
        if variables is None
        else variables
    )

    coords: dict[str, np.ndarray] = {"time": default_time}
    data_vars_xr: dict[str, xr.DataArray] = {}
    for name, arr in default_vars.items():
        if arr.ndim == 1 and arr.shape[0] == len(default_time):
            da = xr.DataArray(arr, dims=["time"])
        elif arr.ndim >= 1:
            dims = ["time"] + [f"d{i}" for i in range(1, arr.ndim)]
            da = xr.DataArray(arr, dims=dims)
        else:
            da = xr.DataArray(arr)
        data_vars_xr[name] = da

    ds = xr.Dataset(data_vars_xr, coords=coords, attrs=attrs or {"imas": group})
    ds.to_zarr(str(grp_path), mode="w")
    return grp_path


# ---------------------------------------------------------------------------
# Individual check tests
# ---------------------------------------------------------------------------


def test_check_open_pass(tmp_path: Path) -> None:
    """check_open returns passed=True when the group is readable."""
    from imas_ambix.quality.checks import check_open

    shot = tmp_path / "99001.zarr"
    _write_zarr_group(shot, "magnetics")

    result = check_open(shot, "magnetics")
    assert result.passed is True
    assert result.severity == "info"
    assert result.metric is not None and result.metric >= 1


def test_check_open_fail(tmp_path: Path) -> None:
    """check_open returns passed=False when the group does not exist."""
    from imas_ambix.quality.checks import check_open

    shot = tmp_path / "99999.zarr"  # never written
    result = check_open(shot, "magnetics")
    assert result.passed is False
    assert result.severity == "error"


def test_check_no_all_nan_pass(tmp_path: Path) -> None:
    """check_no_all_nan passes for variables with finite values."""
    from imas_ambix.quality.checks import check_no_all_nan

    ds = xr.Dataset({"v": xr.DataArray(np.ones(5, dtype=np.float64), dims=["time"])})
    results = check_no_all_nan(ds)
    assert all(r.passed for r in results)


def test_check_no_all_nan_fail() -> None:
    """check_no_all_nan returns error for an all-NaN variable."""
    from imas_ambix.quality.checks import check_no_all_nan

    ds = xr.Dataset(
        {"v": xr.DataArray(np.full(5, np.nan, dtype=np.float64), dims=["time"])}
    )
    results = check_no_all_nan(ds)
    bad = [r for r in results if not r.passed]
    assert bad, "expected at least one failing check"
    assert bad[0].severity == "error"
    assert bad[0].metric == pytest.approx(1.0)


def test_check_dynamic_range_pass() -> None:
    """check_dynamic_range passes for physically reasonable values."""
    from imas_ambix.quality.checks import check_dynamic_range

    ds = xr.Dataset({"ip": xr.DataArray(np.linspace(-1e6, 1e6, 100), dims=["time"])})
    results = check_dynamic_range(ds)
    assert all(r.passed for r in results)


def test_check_dynamic_range_fail() -> None:
    """check_dynamic_range warns when abs_max exceeds 1e15."""
    from imas_ambix.quality.checks import check_dynamic_range

    ds = xr.Dataset(
        {"signal": xr.DataArray(np.array([0.0, 1e16], dtype=np.float64), dims=["time"])}
    )
    results = check_dynamic_range(ds)
    bad = [r for r in results if not r.passed]
    assert bad, "expected a failing dynamic range check"
    assert bad[0].severity == "warn"


def test_check_time_axis_pass() -> None:
    """check_time_axis passes for a monotonically increasing time array."""
    from imas_ambix.quality.checks import check_time_axis

    t = np.linspace(-0.1, 0.5, 50)
    ds = xr.Dataset(coords={"time": t})
    result = check_time_axis(ds)
    assert result.passed is True
    assert result.metric == pytest.approx(50.0)


def test_check_time_axis_fail_nonmono() -> None:
    """check_time_axis returns error for non-monotonic time."""
    from imas_ambix.quality.checks import check_time_axis

    t = np.array([0.0, 0.1, 0.05, 0.2])  # 0.05 < 0.1 — not monotonic
    ds = xr.Dataset(coords={"time": t})
    result = check_time_axis(ds)
    assert result.passed is False
    assert result.severity == "error"


def test_check_time_axis_missing() -> None:
    """check_time_axis returns warn when no time dimension is present."""
    from imas_ambix.quality.checks import check_time_axis

    ds = xr.Dataset({"v": xr.DataArray(np.ones(3), dims=["x"])})
    result = check_time_axis(ds)
    assert result.passed is False
    assert result.severity == "warn"


def test_check_homogeneous_time_flag_present() -> None:
    """check_homogeneous_time_flag passes when imas attr is present."""
    from imas_ambix.quality.checks import check_homogeneous_time_flag

    ds = xr.Dataset(attrs={"imas": "magnetics"})
    result = check_homogeneous_time_flag(ds)
    assert result.passed is True


def test_check_homogeneous_time_flag_absent() -> None:
    """check_homogeneous_time_flag warns when imas attr is missing."""
    from imas_ambix.quality.checks import check_homogeneous_time_flag

    ds = xr.Dataset(attrs={})
    result = check_homogeneous_time_flag(ds)
    assert result.passed is False
    assert result.severity == "warn"


def test_check_dd_version_warn() -> None:
    """check_dd_version warns when no version attribute is present."""
    from imas_ambix.quality.checks import check_dd_version

    ds = xr.Dataset(attrs={"imas": "summary"})
    result = check_dd_version(ds)
    assert result.passed is False
    assert result.severity == "warn"


def test_check_dd_version_mismatch() -> None:
    """check_dd_version returns error when stored version != expected."""
    from imas_ambix.quality.checks import check_dd_version

    ds = xr.Dataset(attrs={"dd_version": "3.39.0"})
    result = check_dd_version(ds, expected="3.40.0")
    assert result.passed is False
    assert result.severity == "error"


# ---------------------------------------------------------------------------
# Audit tests
# ---------------------------------------------------------------------------


def test_audit_shot_synthetic_clean(tmp_path: Path) -> None:
    """audit_shot on a well-formed synthetic shot returns usable_for_training=True."""
    from imas_ambix.quality.audit import audit_shot

    shots_dir = tmp_path / "level2" / "shots"
    shot_path = shots_dir / "99001.zarr"
    _write_zarr_group(shot_path, "magnetics")
    _write_zarr_group(
        shot_path,
        "equilibrium",
        variables={"magnetic_axis_r": np.linspace(0.5, 0.8, 8)},
    )

    with patch("imas_ambix.quality.audit.LEVEL2_DIR", shots_dir):
        report = audit_shot(99001, tier="level2")

    assert report.shot_id == 99001
    assert "magnetics" in report.groups_present
    assert "equilibrium" in report.groups_present
    assert report.quality_flags["usable_for_training"] is True
    assert report.quality_flags["has_equilibrium"] is True
    assert report.quality_flags["has_magnetics"] is True


def test_audit_shot_all_nan(tmp_path: Path) -> None:
    """audit_shot flags a shot with an all-NaN variable as not usable."""
    from imas_ambix.quality.audit import audit_shot

    shots_dir = tmp_path / "level2" / "shots"
    shot_path = shots_dir / "99002.zarr"
    _write_zarr_group(
        shot_path,
        "magnetics",
        variables={"signal": np.full(8, np.nan, dtype=np.float64)},
    )

    with patch("imas_ambix.quality.audit.LEVEL2_DIR", shots_dir):
        report = audit_shot(99002, tier="level2")

    assert report.quality_flags["usable_for_training"] is False
    assert report.overall_severity == "error"


def test_audit_shot_missing(tmp_path: Path) -> None:
    """audit_shot on a missing shot path returns an error-severity report."""
    from imas_ambix.quality.audit import audit_shot

    shots_dir = tmp_path / "level2" / "shots"  # never created

    with patch("imas_ambix.quality.audit.LEVEL2_DIR", shots_dir):
        report = audit_shot(99999, tier="level2")

    assert report.overall_severity == "error"
    assert report.groups_present == ()
    assert report.quality_flags.get("usable_for_training") is False


def test_audit_corpus_three_shots(tmp_path: Path) -> None:
    """audit_corpus over 3 synthetic shots returns 3 reports in input order."""
    from imas_ambix.quality.audit import audit_corpus

    shots_dir = tmp_path / "level2" / "shots"
    for sid in (10001, 10002, 10003):
        shot_path = shots_dir / f"{sid}.zarr"
        _write_zarr_group(shot_path, "magnetics")

    with patch("imas_ambix.quality.audit.LEVEL2_DIR", shots_dir):
        reports = audit_corpus([10001, 10002, 10003], tier="level2", max_workers=2)

    assert len(reports) == 3
    assert [r.shot_id for r in reports] == [10001, 10002, 10003]
    assert all(r.overall_severity in ("info", "warn") for r in reports)


# ---------------------------------------------------------------------------
# Aggregate tests
# ---------------------------------------------------------------------------


def _make_report(shot_id: int, severity: str, campaign: str = "M9") -> object:
    """Build a minimal ShotQualityReport for aggregate testing."""
    from imas_ambix.quality.audit import GroupStats, ShotQualityReport

    return ShotQualityReport(
        shot_id=shot_id,
        tier="level2",
        groups_present=("magnetics",),
        per_group={
            "magnetics": GroupStats(
                n_variables=2,
                n_timesteps=100,
                nan_fraction=0.0,
                min_value=-1.0,
                max_value=1.0,
                checks=(),
                open_ok=True,
            )
        },
        metadata={"campaign": campaign},
        quality_flags={"usable_for_training": severity != "error"},
        overall_severity=severity,
    )


def test_aggregate_corpus_counts() -> None:
    """aggregate_corpus produces correct n_passed / n_warned / n_failed counts."""
    from imas_ambix.quality.audit import aggregate_corpus

    reports = [
        _make_report(1, "info"),
        _make_report(2, "warn"),
        _make_report(3, "error"),
        _make_report(4, "info"),
    ]
    agg = aggregate_corpus(reports)  # type: ignore[arg-type]

    assert agg["n_total"] == 4
    assert agg["n_passed"] == 2
    assert agg["n_warned"] == 1
    assert agg["n_failed"] == 1
    assert agg["pass_rate"] == pytest.approx(0.5)
    assert agg["usable_for_training"] == 3


def test_aggregate_corpus_campaign() -> None:
    """aggregate_corpus correctly counts campaign distribution."""
    from imas_ambix.quality.audit import aggregate_corpus

    reports = [
        _make_report(1, "info", "M9"),
        _make_report(2, "info", "M9"),
        _make_report(3, "info", "M8"),
    ]
    agg = aggregate_corpus(reports)  # type: ignore[arg-type]
    dist = agg["campaign_distribution"]
    assert dist["M9"] == 2
    assert dist["M8"] == 1


def test_aggregate_corpus_empty() -> None:
    """aggregate_corpus on empty list returns safe zero-value dict."""
    from imas_ambix.quality.audit import aggregate_corpus

    agg = aggregate_corpus([])
    assert agg["n_total"] == 0
    assert agg["pass_rate"] == 0.0


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------


def test_audit_cmd_help() -> None:
    """audit --help exits 0 and mentions key options."""
    from click.testing import CliRunner

    from imas_ambix.data.cli import data

    runner = CliRunner()
    result = runner.invoke(data, ["audit", "--help"])
    assert result.exit_code == 0, result.output
    assert "--tier" in result.output
    assert "--shot-ids" in result.output
    assert "--output" in result.output
    assert "--workers" in result.output


def test_audit_cmd_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Full CLI run: audit --shot-ids X --output produces valid JSON."""
    from click.testing import CliRunner

    from imas_ambix.data.cli import data

    # Build synthetic shots in a temp dir that stands in for LEVEL2_DIR
    shots_dir = tmp_path / "level2" / "shots"
    for sid in (20001, 20002):
        shot_path = shots_dir / f"{sid}.zarr"
        _write_zarr_group(shot_path, "magnetics")
        _write_zarr_group(
            shot_path,
            "equilibrium",
            variables={"r": np.linspace(0.5, 0.9, 8)},
        )

    out_path = tmp_path / "audit.json"

    with (
        patch("imas_ambix.quality.audit.LEVEL2_DIR", shots_dir),
        patch("imas_ambix.data.cli._L2", shots_dir, create=True),
    ):
        runner = CliRunner()
        result = runner.invoke(
            data,
            [
                "audit",
                "--tier",
                "level2",
                "--shot-ids",
                "20001,20002",
                "--workers",
                "1",
                "--output",
                str(out_path),
            ],
        )

    assert result.exit_code == 0, result.output
    assert out_path.exists(), "JSON output file not created"

    payload = json.loads(out_path.read_text())
    assert payload["tier"] == "level2"
    assert set(payload["shot_ids"]) == {20001, 20002}
    agg = payload["aggregate"]
    assert agg["n_total"] == 2
    assert "per_shot" in payload
    assert len(payload["per_shot"]) == 2
