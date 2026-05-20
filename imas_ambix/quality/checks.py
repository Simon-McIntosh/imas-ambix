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

FAIR-MAST format notes (2026-05-20)
-------------------------------------
FAIR-MAST level-2 is xarray-on-Zarr v3, NOT IDS format.  Groups have
an ``imas`` string attribute (e.g. ``"magnetics"``) that is a label /
cross-reference pointer — there are no ``ids_properties``, ``version_put``,
``homogeneous_time``, or ``dd_version`` attributes anywhere in the bucket.
Time coordinates are per-group, not at root.  Dynamic ranges are physical
(1e19 – 1e22 for densities; 0–750 kA for plasma current).
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

#: Hard corruption threshold — any |value| above this is a data error,
#: not a physics outlier.  Set at 1e25 to cover all expected plasma physics
#: quantities (densities ~ 1e22, currents ~ 1e6, temperatures ~ 1e4) with
#: many orders of magnitude headroom.
_HARD_CORRUPTION_THRESHOLD: float = 1e25

#: Relative tolerance for uniform-Δt check on time coordinates.
_DT_JITTER_RTOL: float = 0.05


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
    """Check floating-point variables for hard corruption indicators.

    Gates (corpus-calibrated for FAIR-MAST physical ranges):

    * **Hard corruption** (severity ``"error"``): any ``|value| > 1e25``,
      any ``inf`` / ``-inf``, or any ``nan`` payload that decodes as a
      huge-but-finite number.
    * **Constant time-series** (severity ``"warn"``): ``max == min`` for a
      variable that has a ``time`` dimension — indicates a possibly stuck
      channel.
    * **All-zeros / all-constant static variable**: intentionally skipped
      (geometry placeholders, pre-shot baselines).

    Physical ranges that pass silently include densities ~1e19–1e22,
    particle rates ~1e22, currents 0–750 kA, temperatures 0–30 keV, and
    all other MAST-range quantities.

    One :class:`CheckResult` per variable.

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

        has_inf = bool(np.any(np.isinf(arr)))
        finite = arr[np.isfinite(arr)]

        if has_inf:
            abs_max = float(np.abs(finite).max()) if finite.size > 0 else math.nan
            results.append(
                CheckResult(
                    name=f"dynamic_range:{var}",
                    passed=False,
                    severity="error",
                    message=f"{var}: contains inf/−inf (abs_max_finite={abs_max:.3e})",
                    metric=abs_max,
                )
            )
            continue

        if finite.size == 0:
            # All-NaN — handled by check_no_all_nan; pass here as info.
            results.append(
                CheckResult(
                    name=f"dynamic_range:{var}",
                    passed=True,
                    severity="info",
                    message=f"{var}: no finite values (deferred to no_all_nan check)",
                    metric=None,
                )
            )
            continue

        abs_max = float(np.abs(finite).max())

        if abs_max > _HARD_CORRUPTION_THRESHOLD:
            results.append(
                CheckResult(
                    name=f"dynamic_range:{var}",
                    passed=False,
                    severity="error",
                    message=f"{var}: hard corruption — abs_max={abs_max:.3e} > 1e25",
                    metric=abs_max,
                )
            )
            continue

        # Constant time-series check — only relevant for time-varying variables.
        # Skip all-zero signals (pre-shot baselines, geometry placeholders).
        has_time_dim = "time" in da.dims
        val_max = float(finite.max())
        val_min = float(finite.min())
        if has_time_dim and val_max == val_min and val_max != 0.0:
            results.append(
                CheckResult(
                    name=f"dynamic_range:{var}",
                    passed=False,
                    severity="warn",
                    message=(
                        f"{var}: constant non-zero time-series"
                        f" (value={val_max:.3e})"
                    ),
                    metric=abs_max,
                )
            )
            continue

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


def check_time_axis(ds: xr.Dataset) -> list[CheckResult]:
    """Verify every time-like coordinate is monotonic and uniformly spaced.

    Enumerates **all coordinates** whose name contains ``"time"``
    (e.g. ``time``, ``time_saddle``).  For each such coordinate:

    * Returns a severity ``"error"`` result if the axis is non-monotonic
      or contains NaN / Inf.
    * Returns a severity ``"warn"`` result if the time step jitter exceeds
      5 % of the median Δt (loose uniformity check).
    * Returns a severity ``"info"`` result if the axis passes.

    If the group has **no** time-like coordinate (static geometry groups
    such as ``pf_passive``, ``wall``, ``soft_x_rays``), returns a single
    info-level ``"no time axis (static group)"`` result.

    Parameters
    ----------
    ds:
        Already-opened dataset for a single group.
    """
    import numpy as np

    time_coords = [c for c in ds.coords if "time" in str(c)]

    if not time_coords:
        return [
            CheckResult(
                name="time_axis",
                passed=True,
                severity="info",
                message="no time axis (static group)",
                metric=0.0,
            )
        ]

    results: list[CheckResult] = []
    for coord_name in time_coords:
        try:
            t = ds.coords[coord_name].values.astype(float)
        except Exception as exc:  # noqa: BLE001
            results.append(
                CheckResult(
                    name=f"time_axis:{coord_name}",
                    passed=False,
                    severity="error",
                    message=f"{coord_name}: could not read coord: {exc!r}",
                    metric=None,
                )
            )
            continue

        n = len(t)
        if n == 0:
            results.append(
                CheckResult(
                    name=f"time_axis:{coord_name}",
                    passed=False,
                    severity="warn",
                    message=f"{coord_name}: empty time axis",
                    metric=0.0,
                )
            )
            continue

        has_nan = bool(np.any(np.isnan(t)))
        has_inf = bool(np.any(np.isinf(t)))
        if has_nan or has_inf:
            results.append(
                CheckResult(
                    name=f"time_axis:{coord_name}",
                    passed=False,
                    severity="error",
                    message=(
                        f"{coord_name}: contains "
                        f"{'NaN ' if has_nan else ''}{'Inf' if has_inf else ''}"
                        f"(n={n})"
                    ),
                    metric=float(n),
                )
            )
            continue

        diffs = np.diff(t)
        monotonic = bool(np.all(diffs >= 0))
        if not monotonic:
            results.append(
                CheckResult(
                    name=f"time_axis:{coord_name}",
                    passed=False,
                    severity="error",
                    message=f"{coord_name}: non-monotonic (n={n})",
                    metric=float(n),
                )
            )
            continue

        # Uniformity check — warn if Δt jitter > 5 % of median.
        if n > 2:
            dt_median = float(np.median(diffs))
            if dt_median > 0:
                jitter = float(np.abs(diffs - dt_median).max() / dt_median)
                if jitter > _DT_JITTER_RTOL:
                    results.append(
                        CheckResult(
                            name=f"time_axis:{coord_name}",
                            passed=False,
                            severity="warn",
                            message=(
                                f"{coord_name}: Δt jitter={jitter:.1%} "
                                f"(>{_DT_JITTER_RTOL:.0%} of median "
                                f"dt={dt_median:.4g}s) n={n}"
                            ),
                            metric=float(n),
                        )
                    )
                    continue

        results.append(
            CheckResult(
                name=f"time_axis:{coord_name}",
                passed=True,
                severity="info",
                message=(
                    f"{coord_name}: ok (n={n}, "
                    f"span={t[-1] - t[0]:.4f}s)"
                ),
                metric=float(n),
            )
        )

    return results


def check_imas_pointer(ds: xr.Dataset, var: str) -> CheckResult:
    """Report whether a variable carries an ``imas`` attribute.

    FAIR-MAST stores a per-variable ``imas`` string attribute that points
    to the corresponding IDS path (e.g.
    ``"magnetics.b_field_pol_probe[:].field.data"``).  This check is
    *informational only* — absence is not a failure, because many
    variables (e.g. computed quantities, coordinate arrays) legitimately
    lack an IDS pointer.

    Parameters
    ----------
    ds:
        Already-opened dataset for a single group.
    var:
        Name of the variable to inspect.
    """
    if var not in ds.data_vars and var not in ds.coords:
        return CheckResult(
            name=f"imas_pointer:{var}",
            passed=True,
            severity="info",
            message=f"{var}: variable not in dataset",
        )
    attrs = ds[var].attrs if var in ds.data_vars else ds.coords[var].attrs
    imas_val = attrs.get("imas", None)
    if imas_val:
        return CheckResult(
            name=f"imas_pointer:{var}",
            passed=True,
            severity="info",
            message=f"{var}: imas pointer present → '{imas_val}'",
        )
    return CheckResult(
        name=f"imas_pointer:{var}",
        passed=True,
        severity="info",
        message=f"{var}: no imas pointer (expected for non-IDS vars)",
    )


def check_imas_label_matches_group(ds: xr.Dataset, group_name: str) -> CheckResult:
    """Verify that the group-level ``imas`` attribute matches the group name.

    FAIR-MAST level-2 groups carry a dataset-level ``imas`` string attribute
    equal to the group's directory name (e.g. the ``magnetics/`` group has
    ``imas: "magnetics"``).  A mismatch is informational — it may indicate
    a metadata inconsistency in the bucket but is not a training-data
    failure.

    Parameters
    ----------
    ds:
        Already-opened dataset for a single group.
    group_name:
        Expected group name (the directory component, e.g. ``"magnetics"``).
    """
    imas_attr = ds.attrs.get("imas", None)
    if imas_attr is None:
        return CheckResult(
            name="imas_label_matches_group",
            passed=True,
            severity="info",
            message=(
                f"group '{group_name}': no imas attribute"
                " (ok for non-standard groups)"
            ),
        )
    if str(imas_attr) != str(group_name):
        return CheckResult(
            name="imas_label_matches_group",
            passed=True,
            severity="info",
            message=(
                f"imas label '{imas_attr}' differs from group name '{group_name}'"
            ),
        )
    return CheckResult(
        name="imas_label_matches_group",
        passed=True,
        severity="info",
        message=f"imas label matches group name '{group_name}'",
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
