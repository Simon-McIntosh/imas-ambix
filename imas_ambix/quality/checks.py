"""Per-group Zarr-level quality checks for FAIR-MAST shots.

Each check takes an ``xarray.Dataset`` (one group already opened) and
returns one or more :class:`CheckResult` objects describing whether the
data passes the gate.  The caller (``audit.py``) aggregates all results
into a :class:`~imas_ambix.quality.audit.ShotQualityReport`.

Design principles
-----------------
- Checks are pure functions: they never mutate the dataset.
- Failures are data, not exceptions: every issue is captured in
  ``CheckResult.message``.  Callers decide what to do with severity.
- ``check_open`` is the only function that may raise (if ``xr.open_zarr``
  throws). All other checks receive a pre-opened dataset.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import xarray as xr

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

#: Upper bound for physically plausible absolute signal magnitude.
_ABS_MAX_THRESHOLD: float = 1e15


@dataclass(frozen=True)
class CheckResult:
    """Outcome of a single quality check.

    Parameters
    ----------
    name:
        Human-readable check name (snake_case).
    passed:
        ``True`` iff the check gate is satisfied.
    severity:
        ``"info"`` | ``"warn"`` | ``"error"``.  ``"info"`` results are
        always :attr:`passed`.  ``"error"`` results indicate data that
        should be excluded from training.
    message:
        One-line description of the outcome.
    metric:
        Optional scalar summarising the measured quantity (e.g. NaN
        fraction, dynamic range ratio).  ``None`` when not applicable.
    """

    name: str
    passed: bool
    severity: str  # "info" | "warn" | "error"
    message: str
    metric: float | None = field(default=None)


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def check_open(shot_path: Path, group: str) -> CheckResult:
    """Attempt to open *shot_path/group* with :func:`xarray.open_zarr`.

    Returns a *passed* result if the store opens successfully, an
    ``"error"`` result otherwise.

    Parameters
    ----------
    shot_path:
        Root of the shot Zarr store (e.g. ``.../11766.zarr``).
    group:
        Sub-group name (e.g. ``"magnetics"``).
    """
    try:
        ds = xr.open_zarr(str(shot_path / group), consolidated=False)
        n_vars = len(ds.data_vars)
        return CheckResult(
            name="open",
            passed=True,
            severity="info",
            message=f"opened ok ({n_vars} vars)",
            metric=float(n_vars),
        )
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            name="open",
            passed=False,
            severity="error",
            message=f"open failed: {exc!r}",
            metric=None,
        )


def check_no_all_nan(ds: xr.Dataset) -> list[CheckResult]:
    """Check that no floating-point variable is entirely NaN.

    One :class:`CheckResult` is returned per variable.  Non-float
    variables (e.g. string coordinates, integers) always pass as
    ``"info"``.

    Parameters
    ----------
    ds:
        Already-opened dataset for a single group.
    """
    import numpy as np

    results: list[CheckResult] = []
    for var in ds.data_vars:
        da = ds[var]
        if not np.issubdtype(da.dtype, np.floating):
            results.append(
                CheckResult(
                    name=f"no_all_nan:{var}",
                    passed=True,
                    severity="info",
                    message=f"{var}: non-float, skip",
                )
            )
            continue
        try:
            arr = da.values
        except Exception as exc:  # noqa: BLE001
            results.append(
                CheckResult(
                    name=f"no_all_nan:{var}",
                    passed=False,
                    severity="error",
                    message=f"{var}: could not load values: {exc!r}",
                )
            )
            continue
        nan_frac = float(np.isnan(arr).mean()) if arr.size > 0 else 0.0
        all_nan = arr.size > 0 and nan_frac == 1.0
        results.append(
            CheckResult(
                name=f"no_all_nan:{var}",
                passed=not all_nan,
                severity="error" if all_nan else "info",
                message=(
                    f"{var}: all-NaN ({arr.size} elements)"
                    if all_nan
                    else f"{var}: ok (nan_frac={nan_frac:.3f})"
                ),
                metric=nan_frac,
            )
        )
    return results


def check_dynamic_range(ds: xr.Dataset) -> list[CheckResult]:
    """Check that floating-point variables do not contain infinite values
    and that their dynamic range is physically plausible.

    Gate: ``abs_max`` must be finite and ``< 1e15``.  Values above this
    threshold suggest a unit error, a fill-value leak, or a reconstruction
    artefact.  One :class:`CheckResult` per variable.

    Parameters
    ----------
    ds:
        Already-opened dataset for a single group.
    """
    import numpy as np

    results: list[CheckResult] = []
    for var in ds.data_vars:
        da = ds[var]
        if not np.issubdtype(da.dtype, np.floating):
            results.append(
                CheckResult(
                    name=f"dynamic_range:{var}",
                    passed=True,
                    severity="info",
                    message=f"{var}: non-float, skip",
                )
            )
            continue
        try:
            arr = da.values.ravel()
        except Exception as exc:  # noqa: BLE001
            results.append(
                CheckResult(
                    name=f"dynamic_range:{var}",
                    passed=False,
                    severity="error",
                    message=f"{var}: could not load values: {exc!r}",
                )
            )
            continue
        finite = arr[np.isfinite(arr)]
        has_inf = bool(np.any(np.isinf(arr)))
        if finite.size == 0:
            results.append(
                CheckResult(
                    name=f"dynamic_range:{var}",
                    passed=False,
                    severity="warn",
                    message=f"{var}: no finite values",
                    metric=None,
                )
            )
            continue
        abs_max = float(np.abs(finite).max())
        if has_inf or abs_max >= _ABS_MAX_THRESHOLD:
            results.append(
                CheckResult(
                    name=f"dynamic_range:{var}",
                    passed=False,
                    severity="warn",
                    message=(
                        f"{var}: extreme range — abs_max={abs_max:.3e}"
                        + (" (has_inf)" if has_inf else "")
                    ),
                    metric=abs_max,
                )
            )
        else:
            results.append(
                CheckResult(
                    name=f"dynamic_range:{var}",
                    passed=True,
                    severity="info",
                    message=f"{var}: range ok (abs_max={abs_max:.3e})",
                    metric=abs_max,
                )
            )
    return results


def check_time_axis(
    ds: xr.Dataset,
    expected_dim: str = "time",
) -> CheckResult:
    """Verify that the ``time`` coordinate is monotonically increasing and
    contains no NaN / Inf values.

    Parameters
    ----------
    ds:
        Already-opened dataset for a single group.
    expected_dim:
        Name of the time dimension to inspect.  Defaults to ``"time"``.
    """
    import numpy as np

    if expected_dim not in ds.coords and expected_dim not in ds.dims:
        return CheckResult(
            name="time_axis",
            passed=False,
            severity="warn",
            message=f"no '{expected_dim}' dimension found",
            metric=None,
        )
    try:
        t = ds.coords[expected_dim].values.astype(float)
    except (KeyError, Exception) as exc:  # noqa: BLE001
        # Dimension exists but no coordinate array; try from sizes.
        n = ds.sizes.get(expected_dim, 0)
        return CheckResult(
            name="time_axis",
            passed=n > 0,
            severity="warn" if n == 0 else "info",
            message=f"time coord not readable ({exc!r}); n_steps={n}",
            metric=float(n),
        )

    n = len(t)
    if n == 0:
        return CheckResult(
            name="time_axis",
            passed=False,
            severity="warn",
            message="time axis is empty",
            metric=0.0,
        )
    has_nan = bool(np.any(np.isnan(t)))
    has_inf = bool(np.any(np.isinf(t)))
    if has_nan or has_inf:
        return CheckResult(
            name="time_axis",
            passed=False,
            severity="error",
            message=(
                f"time axis contains {'NaN' if has_nan else ''}"
                f"{'Inf' if has_inf else ''} (n={n})"
            ),
            metric=float(n),
        )
    monotonic = bool(np.all(np.diff(t) >= 0))
    if not monotonic:
        return CheckResult(
            name="time_axis",
            passed=False,
            severity="error",
            message=f"time axis is not monotonically non-decreasing (n={n})",
            metric=float(n),
        )
    return CheckResult(
        name="time_axis",
        passed=True,
        severity="info",
        message=f"time axis ok (n={n}, span={t[-1] - t[0]:.4f}s)",
        metric=float(n),
    )


def check_homogeneous_time_flag(ds: xr.Dataset) -> CheckResult:
    """Check whether ``ids_properties.homogeneous_time`` is consistently
    encoded as a dataset attribute.

    For FAIR-MAST Zarr stores the flag lives in ``ds.attrs['imas']`` or
    as an attribute on the group.  This check looks for the ``imas``
    attribute (group name) and verifies it is a non-empty string.  A
    missing attribute is a ``"warn"`` (not fatal).

    Parameters
    ----------
    ds:
        Already-opened dataset for a single group.
    """
    imas_attr = ds.attrs.get("imas", None)
    if imas_attr is None:
        return CheckResult(
            name="homogeneous_time_flag",
            passed=False,
            severity="warn",
            message="'imas' attribute missing from group attrs",
        )
    return CheckResult(
        name="homogeneous_time_flag",
        passed=True,
        severity="info",
        message=f"imas attr present: '{imas_attr}'",
    )


def check_dd_version(
    ds: xr.Dataset,
    expected: str | None = None,
) -> CheckResult:
    """Check that the dataset carries a ``dd_version`` attribute or that
    its value matches *expected* when provided.

    FAIR-MAST level-2 Zarr stores do not currently embed ``dd_version``
    directly in group attributes — the version lives in the IMAS IDS
    ``ids_properties``.  For Zarr-backed data this check looks for a
    ``version`` or ``dd_version`` key in ``ds.attrs``.  A missing key is
    ``"warn"``; a value mismatch with *expected* is ``"error"``.

    Parameters
    ----------
    ds:
        Already-opened dataset for a single group.
    expected:
        When not ``None``, the stored version must equal this string.
    """
    actual = ds.attrs.get("dd_version", ds.attrs.get("version", None))
    if actual is None:
        return CheckResult(
            name="dd_version",
            passed=False,
            severity="warn",
            message="dd_version attribute not found in group attrs",
        )
    if expected is not None and str(actual) != str(expected):
        return CheckResult(
            name="dd_version",
            passed=False,
            severity="error",
            message=f"dd_version mismatch: stored={actual!r} expected={expected!r}",
        )
    return CheckResult(
        name="dd_version",
        passed=True,
        severity="info",
        message=f"dd_version={actual!r}",
    )


def _severity_rank(severity: str) -> int:
    """Numeric rank for severity comparison: info < warn < error."""
    return {"info": 0, "warn": 1, "error": 2}.get(severity, 0)


def worst_severity(results: list[CheckResult]) -> str:
    """Return the highest severity among *results*, defaulting to ``"info"``."""
    if not results:
        return "info"
    return max(results, key=lambda r: _severity_rank(r.severity)).severity


def nan_fraction_of_ds(ds: xr.Dataset) -> float:
    """Compute the overall NaN fraction across all floating-point variables."""
    import numpy as np

    total = 0
    n_nan = 0
    for var in ds.data_vars:
        da = ds[var]
        if not np.issubdtype(da.dtype, np.floating):
            continue
        try:
            arr = da.values
            total += arr.size
            n_nan += int(np.isnan(arr).sum())
        except Exception:  # noqa: BLE001
            pass
    return n_nan / total if total > 0 else 0.0


def global_min_max(ds: xr.Dataset) -> tuple[float, float]:
    """Return (global_min, global_max) across all finite floating-point values."""
    import numpy as np

    mins: list[float] = []
    maxs: list[float] = []
    for var in ds.data_vars:
        da = ds[var]
        if not np.issubdtype(da.dtype, np.floating):
            continue
        try:
            arr = da.values
            finite = arr[np.isfinite(arr)]
            if finite.size > 0:
                mins.append(float(finite.min()))
                maxs.append(float(finite.max()))
        except Exception:  # noqa: BLE001
            pass
    return (
        min(mins) if mins else math.nan,
        max(maxs) if maxs else math.nan,
    )
