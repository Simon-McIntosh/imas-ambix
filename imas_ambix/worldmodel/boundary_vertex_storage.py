"""Inspect unfiltered boundary-vertex storage in Nova session NetCDF files."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

DEFAULT_BASELINE_ROOT = Path(
    "/work/projects/imas_gpu/sophelio/labeller_sessions/76906a29"
)
DEFAULT_REPAIRED_ROOT = Path(
    "/work/projects/imas_gpu/sophelio/labeller_sessions/"
    "boundary-repair-validation-20260907T1124Z"
)
DEFAULT_SHOTS = (21978, 22086)
DEFAULT_GROUP = "steering"
DEFAULT_OUTPUT = Path(
    "docs/figures/physics-carried-playable-plasma/label-quality/"
    "boundary-vertex-storage.json"
)
BOUNDARY_R_VARIABLE = "flux_surface_r"
BOUNDARY_Z_VARIABLE = "flux_surface_z"
SURFACE_LEVEL_VARIABLE = "flux_surface_psi_norm"


@dataclass(frozen=True, slots=True)
class _StoredBoundary:
    report: dict[str, Any]
    values: np.ndarray
    trailing_nonfinite_counts: np.ndarray


def _source_revision() -> str:
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _literal_scalar(value: float) -> float | str:
    number = float(value)
    if np.isnan(number):
        return "NaN"
    if np.isposinf(number):
        return "+Infinity"
    if np.isneginf(number):
        return "-Infinity"
    return number


def _literal_values(values: np.ndarray) -> list[Any]:
    array = np.asarray(values)
    if array.ndim == 0:
        return [_literal_scalar(float(array))]
    if array.ndim == 1:
        return [_literal_scalar(value) for value in array]
    return [_literal_values(row) for row in array]


def _trailing_nonfinite_rows(boundary: np.ndarray) -> int:
    rows = np.asarray(boundary, dtype=np.float64)
    count = 0
    for row in rows[::-1]:
        if np.isfinite(row).all():
            break
        count += 1
    return count


def _open_group(path: Path, group: str) -> Any:
    import xarray as xr  # noqa: PLC0415

    try:
        return xr.open_dataset(
            path,
            group=group,
            engine="h5netcdf",
            decode_cf=False,
            mask_and_scale=False,
        )
    except OSError as error:
        raise ValueError(f"{path} has no readable NetCDF group /{group}") from error


def _raw_root_variables(path: Path) -> list[str]:
    import xarray as xr  # noqa: PLC0415

    with xr.open_dataset(
        path, engine="h5netcdf", decode_cf=False, mask_and_scale=False
    ) as root:
        return sorted(str(name) for name in root.variables)


def _inspect_boundary_storage(path: Path, *, group: str) -> _StoredBoundary:
    session_path = Path(path)
    root_variables = _raw_root_variables(session_path)
    with _open_group(session_path, group) as source:
        if not source.variables:
            raise ValueError(f"{session_path} group /{group} contains no variables")
        required = {
            BOUNDARY_R_VARIABLE,
            BOUNDARY_Z_VARIABLE,
            SURFACE_LEVEL_VARIABLE,
            "time",
        }
        missing = required - set(source.variables)
        if missing:
            raise ValueError(f"{session_path} group /{group} lacks {sorted(missing)}")
        r_variable = source[BOUNDARY_R_VARIABLE]
        z_variable = source[BOUNDARY_Z_VARIABLE]
        if r_variable.dims != z_variable.dims or r_variable.shape != z_variable.shape:
            raise ValueError("stored boundary R and Z variables do not align")
        required_dimensions = {"n_surface", "n_theta", "time"}
        if set(r_variable.dims) != required_dimensions:
            expected = sorted(required_dimensions)
            raise ValueError(f"{BOUNDARY_R_VARIABLE} dimensions must be {expected}")
        levels = np.asarray(source[SURFACE_LEVEL_VARIABLE], dtype=np.float64)
        candidates = np.flatnonzero(np.isclose(levels, 1.0, rtol=0.0, atol=1.0e-6))
        if not candidates.size:
            raise ValueError(f"{session_path} has no stored psi_norm=1 surface")
        surface_index = int(candidates[-1])
        r_values = np.asarray(
            r_variable.isel(n_surface=surface_index).transpose("time", "n_theta")
        )
        z_values = np.asarray(
            z_variable.isel(n_surface=surface_index).transpose("time", "n_theta")
        )
        boundary = np.stack((r_values, z_values), axis=-1)
        times = np.asarray(source["time"], dtype=np.float64).reshape(-1)
        if boundary.shape[0] != times.size:
            raise ValueError("stored boundary time axis does not align with time")
        trailing_counts = np.asarray(
            [_trailing_nonfinite_rows(frame) for frame in boundary], dtype=np.int64
        )
        rows = []
        for index, (time_s, frame, trailing_count) in enumerate(
            zip(times, boundary, trailing_counts, strict=True)
        ):
            first_row = frame[0]
            last_row = frame[-1]
            first_finite = bool(np.isfinite(first_row).all())
            last_finite = bool(np.isfinite(last_row).all())
            rows.append(
                {
                    "session_index": index,
                    "time_s": float(time_s),
                    "shape": list(frame.shape),
                    "first_row": _literal_values(first_row),
                    "last_row": _literal_values(last_row),
                    "trailing_nonfinite_row_count": int(trailing_count),
                    "first_row_finite": first_finite,
                    "last_row_finite": last_finite,
                    "final_row_equals_first_row": bool(
                        first_finite
                        and last_finite
                        and np.allclose(last_row, first_row, rtol=1.0e-7, atol=1.0e-12)
                    ),
                }
            )
        coordinate_variables = {}
        for name, variable in (
            (BOUNDARY_R_VARIABLE, r_variable),
            (BOUNDARY_Z_VARIABLE, z_variable),
        ):
            raw = np.asarray(variable)
            theta_axis = variable.dims.index("n_theta")
            first_theta_row = np.take(raw, 0, axis=theta_axis)
            last_theta_row = np.take(raw, -1, axis=theta_axis)
            coordinate_variables[name] = {
                "dtype": str(variable.dtype),
                "dimensions": list(variable.dims),
                "full_shape": list(variable.shape),
                "first_n_theta_row": _literal_values(first_theta_row),
                "last_n_theta_row": _literal_values(last_theta_row),
            }

    count_frequency: dict[str, int] = {}
    for value in trailing_counts.tolist():
        key = str(int(value))
        count_frequency[key] = count_frequency.get(key, 0) + 1
    report = {
        "session_path": str(session_path.resolve()),
        "root_group": {
            "group_path": "/",
            "variable_names": root_variables,
            "has_variables": bool(root_variables),
        },
        "opened_group_path": f"/{group}",
        "boundary_variable_name": (
            f"stack({BOUNDARY_R_VARIABLE}, {BOUNDARY_Z_VARIABLE}) at "
            f"{SURFACE_LEVEL_VARIABLE}=1"
        ),
        "coordinate_variables": coordinate_variables,
        "selected_surface_index": surface_index,
        "selected_surface_psi_norm": float(levels[surface_index]),
        "boundary_array": {
            "dtype": str(boundary.dtype),
            "dimensions": ["time", "n_theta", "coordinate"],
            "full_shape": list(boundary.shape),
            "stored_vertex_count": int(boundary.shape[1]),
            "trailing_nonfinite_row_count_frequency": count_frequency,
            "slices": rows,
        },
    }
    return _StoredBoundary(
        report=report,
        values=boundary,
        trailing_nonfinite_counts=trailing_counts,
    )


def read_boundary_storage(path: Path, *, group: str = DEFAULT_GROUP) -> dict[str, Any]:
    """Read boundary storage without filtering non-finite vertex rows."""
    return _inspect_boundary_storage(Path(path), group=group).report


def compare_boundary_storage(
    baseline_path: Path,
    repaired_path: Path,
    *,
    group: str = DEFAULT_GROUP,
) -> dict[str, Any]:
    """Compare boundary storage layout and trailing-row finiteness."""
    baseline = _inspect_boundary_storage(Path(baseline_path), group=group)
    repaired = _inspect_boundary_storage(Path(repaired_path), group=group)
    baseline_shape = baseline.values.shape
    repaired_shape = repaired.values.shape
    length_differs = baseline_shape[1] != repaired_shape[1]
    trailing_finiteness_differs = not np.array_equal(
        baseline.trailing_nonfinite_counts, repaired.trailing_nonfinite_counts
    )
    same_shape = baseline_shape == repaired_shape
    arrays_equal = bool(
        same_shape and np.array_equal(baseline.values, repaired.values, equal_nan=True)
    )
    arrays_close = bool(
        same_shape
        and np.allclose(
            baseline.values,
            repaired.values,
            rtol=1.0e-12,
            atol=1.0e-12,
            equal_nan=True,
        )
    )
    maximum_absolute_difference: float | None = None
    if same_shape:
        shared_finite = np.isfinite(baseline.values) & np.isfinite(repaired.values)
        if np.any(shared_finite):
            maximum_absolute_difference = float(
                np.max(
                    np.abs(
                        repaired.values[shared_finite] - baseline.values[shared_finite]
                    )
                )
            )
    if length_differs:
        category = "stored_length_differs"
    elif trailing_finiteness_differs:
        category = "trailing_row_finiteness_differs"
    else:
        category = "identical_vertex_storage_layout"
    return {
        "baseline": baseline.report,
        "repaired": repaired.report,
        "comparison": {
            "category": category,
            "stored_vertex_count_differs": length_differs,
            "trailing_row_finiteness_differs": trailing_finiteness_differs,
            "full_boundary_shape_equal": same_shape,
            "boundary_values_bitwise_equal": arrays_equal,
            "boundary_values_equal_within_float_tolerance": arrays_close,
            "maximum_absolute_finite_value_difference": maximum_absolute_difference,
        },
    }


def build_report(
    *,
    baseline_root: Path = DEFAULT_BASELINE_ROOT,
    repaired_root: Path = DEFAULT_REPAIRED_ROOT,
    shot_ids: Sequence[int] = DEFAULT_SHOTS,
    group: str = DEFAULT_GROUP,
) -> dict[str, Any]:
    """Read matching shots from both session roots and classify their storage."""
    shots = [
        {
            "shot_id": int(shot),
            **compare_boundary_storage(
                Path(baseline_root) / f"{int(shot)}.nc",
                Path(repaired_root) / f"{int(shot)}.nc",
                group=group,
            ),
        }
        for shot in shot_ids
    ]
    categories = {shot["comparison"]["category"] for shot in shots}
    if categories == {"stored_length_differs"}:
        verdict = "The two roots differ in stored boundary-vertex length."
    elif (
        categories
        <= {
            "trailing_row_finiteness_differs",
            "identical_vertex_storage_layout",
        }
        and "trailing_row_finiteness_differs" in categories
    ):
        verdict = (
            "The roots have equal stored length and differ only in trailing-row "
            "finiteness for at least one shot."
        )
    elif categories == {"identical_vertex_storage_layout"}:
        verdict = (
            "The two roots are identical in stored boundary-vertex length and "
            "trailing-row finiteness."
        )
    else:
        verdict = "The compared shots have mixed boundary-storage categories."
    root_group_has_no_variables = all(
        not root_name["root_group"]["has_variables"]
        for shot in shots
        for root_name in (shot["baseline"], shot["repaired"])
    )
    return {
        "schema": "boundary-vertex-storage",
        "generated_at": datetime.now(UTC).isoformat(),
        "source_revision": _source_revision(),
        "reader": {
            "library": "xarray",
            "engine": "h5netcdf",
            "decode_cf": False,
            "mask_and_scale": False,
            "opened_group_path": f"/{group}",
            "absent_named_group_behavior": "raises ValueError",
        },
        "root_group_observation": (
            "Opening / yields no variables in every compared file; boundary variables "
            f"are read explicitly from /{group}."
        ),
        "root_group_has_no_variables_in_all_files": root_group_has_no_variables,
        "baseline_root": str(Path(baseline_root).resolve()),
        "repaired_root": str(Path(repaired_root).resolve()),
        "shots": shots,
        "verdict": verdict,
    }


def write_report(report: Mapping[str, Any], output: Path = DEFAULT_OUTPUT) -> Path:
    """Write the raw-storage receipt."""
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return destination


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT)
    parser.add_argument("--repaired-root", type=Path, default=DEFAULT_REPAIRED_ROOT)
    parser.add_argument("--shots", nargs="+", type=int, default=list(DEFAULT_SHOTS))
    parser.add_argument("--group", default=DEFAULT_GROUP)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the paired raw boundary-storage read."""
    args = _parser().parse_args(argv)
    report = build_report(
        baseline_root=args.baseline_root,
        repaired_root=args.repaired_root,
        shot_ids=args.shots,
        group=args.group,
    )
    output = write_report(report, args.output)
    print(json.dumps({"verdict": report["verdict"], "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
