#!/usr/bin/env python
# ruff: noqa: E501  # Markdown output keeps complete table headers and prose rows.
"""Adjudicate disputed MAST flux-loop positions from vacuum measurements.

The geometry candidates come from the level-2 nominal table and reconstruction
signal positions. Signal identity is established independently by correlating the
raw ``amb`` waveform against every experimental ``efm/silop_x`` column on two
range representatives.  Candidate fields use only measured PF/case currents
and static coil geometry through the finite-area forward operator.

The output is a reproducible Markdown receipt and two diagnostic figures.  A
candidate store is never rejected solely because consolidated metadata hides a
group: format-specific consolidated reads are followed by an unconsolidated
read, with every exclusion retained in a skip ledger.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import zarr

from imas_alambic.eddb import read_channel
from imas_alambic.machine_map import load_packaged_machine_map
from imas_alambic.signal_map import load_packaged_signal_map
from imas_ambix.data.paths import JT60SA_ROOT, LEVEL1_DIR, LEVEL2_DIR
from imas_ambix.gs.machine_geometry import MachineGeometryService
from imas_ambix.gs.operator import COIL_MODEL_VERSION, build_operator

if TYPE_CHECKING:
    from imas_ambix.gs.machine_geometry import OperatorGeometry

EARLY_REPRESENTATIVE = 11766
LATE_REPRESENTATIVE = 12417
EARLY_RANGE = "11766-12416"
LATE_RANGE = "12417-30471"
VACUUM_CURRENT_LIMIT_KA = 20.0
KNOWN_LATE_SHOTS = (27394, 27425, 27462, 27507, 27539, 25836)
MIN_LOOP_SHOTS = 6
SCORING_WINDOW_SECONDS = 0.35
MIN_SHOT_SAMPLES = 80
BOOTSTRAP_DRAWS = 20_000
POSITION_TOLERANCE_M = 1e-7

ARTIFACT_ROOT = Path("imas_ambix/latent/artifacts/patch_gate")
EARLY_COHORT_SOURCE = ARTIFACT_ROOT / "vacuum_coil_response_audit-solv4-vac.json"
LATE_CANDIDATE_SOURCE = ARTIFACT_ROOT / "ivc_vacuum_candidates.json"
FULL_SHOT_MANIFEST = Path("/work/projects/imas_gpu/mast/manifests/level1-all.json")
NOTE_PATH = Path("docs/notes/vacuum-loop-adjudication.md")
FIGURE_ROOT = Path("docs/figures/vacuum-loop-adjudication")
_GEOMETRY_SERVICE = MachineGeometryService()


@dataclass(frozen=True)
class _CandidateSensorMapping:
    """One locally adjudicated sensor position for a candidate operator."""

    amb_channel: str
    kind: str
    efm_index: int
    r: float
    z: float
    angle_deg: float | None
    residual_m: float
    flag: str


@dataclass(frozen=True)
class PositionRecord:
    """One range-qualified disagreement between the two stored positions."""

    range_name: str
    representative_shot: int
    loop: str
    nominal_r: float
    nominal_z: float
    reconstruction_r: float
    reconstruction_z: float
    reconstruction_index: int
    identity_correlation: float

    @property
    def key(self) -> str:
        return f"{self.range_name}:{self.loop}"

    @property
    def displacement_m(self) -> float:
        return float(
            np.hypot(
                self.nominal_r - self.reconstruction_r,
                self.nominal_z - self.reconstruction_z,
            )
        )


@dataclass
class DiscoveryReceipt:
    """Auditable outcome for one candidate store."""

    range_name: str
    shot: int
    metadata: str
    attempts: list[str]
    open_mode: str
    peak_ip_ka: float | None
    available_loops: int
    present_loops: tuple[str, ...]
    eligible: bool
    selected: bool
    reason: str


@dataclass
class ShotSignals:
    """Quasi-static measured and candidate-predicted signals for one shot."""

    range_name: str
    shot: int
    peak_ip_ka: float
    metadata: str
    open_mode: str
    n_samples: int
    window_start_s: float
    window_end_s: float
    normalized_activity: float
    values: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]


@dataclass(frozen=True)
class PassiveIndependenceReceipt:
    """Comparison proving that the candidate PF columns do not use amm."""

    shot: int
    metadata: str
    open_mode: str
    passive_structures_with_read: int
    passive_structures_without_read: int
    g_pf_shape: tuple[int, int]
    max_absolute_difference: float


def _metadata_flavor(path: Path) -> str:
    found: list[str] = []
    if (path / ".zmetadata").is_file():
        found.append("format 2 consolidated (.zmetadata)")
    elif (path / ".zgroup").is_file():
        found.append("format 2 unconsolidated (.zgroup)")
    if (path / "zarr.json").is_file():
        found.append("format 3 (zarr.json)")
    return "; ".join(found) if found else "metadata marker not found"


def _open_level1(
    shot: int,
    *,
    required_groups: tuple[str, ...],
    force_unconsolidated: bool = False,
) -> tuple[Any | None, str, list[str], str]:
    """Open a level-1 store with an explicit consolidated fallback receipt."""

    path = LEVEL1_DIR / f"{shot}.zarr"
    metadata = _metadata_flavor(path)
    attempts: list[str] = []
    if not path.exists():
        return None, metadata, attempts, f"store missing: {path}"

    configurations: list[tuple[int | None, bool | None, str]] = []
    if not force_unconsolidated:
        if (path / ".zmetadata").is_file() or (path / ".zgroup").is_file():
            configurations.append((2, True, "format 2 consolidated"))
        if (path / "zarr.json").is_file():
            configurations.append((3, True, "format 3 consolidated"))
        configurations.append((None, None, "reader auto-detection"))
    configurations.append((None, False, "unconsolidated fallback"))

    seen: set[tuple[int | None, bool | None]] = set()
    last_reason = "no open attempt made"
    for zarr_format, consolidated, label in configurations:
        signature = (zarr_format, consolidated)
        if signature in seen:
            continue
        seen.add(signature)
        kwargs: dict[str, object] = {
            "mode": "r",
            "use_consolidated": consolidated,
        }
        if zarr_format is not None:
            kwargs["zarr_format"] = zarr_format
        try:
            group = zarr.open_group(str(path), **kwargs)
            groups = set(group.group_keys())
        except Exception as exc:  # noqa: BLE001
            last_reason = f"{label} open failed: {type(exc).__name__}: {exc}"
            attempts.append(last_reason)
            continue
        missing = sorted(set(required_groups) - groups)
        if missing:
            last_reason = f"{label} hid/missed required groups {missing}"
            attempts.append(last_reason)
            continue
        attempts.append(f"{label} exposed required groups {list(required_groups)}")
        return group, metadata, attempts, label
    return None, metadata, attempts, last_reason


def _group_time(group: Any) -> np.ndarray | None:
    for key in ("time", "sec", "timesec"):
        if key in group:
            value = np.asarray(group[key], dtype=np.float64)
            if value.ndim == 1 and value.size >= 2:
                return value
    return None


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    good = np.isfinite(left) & np.isfinite(right)
    if int(good.sum()) < 8:
        return float("nan")
    x = left[good] - np.mean(left[good])
    y = right[good] - np.mean(right[good])
    denom = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
    return float(np.sum(x * y) / denom) if denom > 0.0 else float("nan")


def _nominal_positions(shot: int) -> dict[str, tuple[float, float]]:
    path = LEVEL2_DIR / f"{shot}.zarr"
    store = zarr.open_group(str(path), mode="r", use_consolidated=None)
    magnetics = store["magnetics"]
    names = [
        str(value).lower().replace("fl_cc010", "fl_cc10")
        for value in np.asarray(magnetics["flux_loop_geometry_channel"]).tolist()
    ]
    radius = np.asarray(magnetics["flux_loop_r"], dtype=np.float64)
    height = np.asarray(magnetics["flux_loop_z"], dtype=np.float64)
    return {
        name: (float(r), float(z))
        for name, r, z in zip(names, radius, height, strict=True)
    }


def _derive_positions(shot: int, range_name: str) -> list[PositionRecord]:
    store, _metadata, attempts, _mode = _open_level1(
        shot, required_groups=("amb", "efm")
    )
    if store is None:
        raise RuntimeError(f"representative {shot} unreadable: {'; '.join(attempts)}")
    nominal = _nominal_positions(shot)
    amb = store["amb"]
    efm = store["efm"]
    amb_time = _group_time(amb)
    efm_time = _group_time(efm)
    if amb_time is None or efm_time is None:
        raise RuntimeError(f"representative {shot} has no usable amb/efm time")
    experimental = np.asarray(efm["silop_x"], dtype=np.float64)
    reconstruction_r = np.asarray(efm["silop_r"], dtype=np.float64)
    reconstruction_z = np.asarray(efm["silop_z"], dtype=np.float64)
    records: list[PositionRecord] = []
    for loop in sorted(key for key in amb.array_keys() if key.startswith("fl_")):
        if loop not in nominal:
            continue
        signal = np.asarray(amb[loop], dtype=np.float64)
        if signal.shape != amb_time.shape:
            continue
        aligned = np.interp(efm_time, amb_time, signal)
        correlations = np.array(
            [
                _correlation(aligned, experimental[:, index])
                for index in range(experimental.shape[1])
            ]
        )
        if not np.isfinite(correlations).any():
            continue
        index = int(np.nanargmax(correlations))
        nr, nz = nominal[loop]
        rr = float(reconstruction_r[index])
        rz = float(reconstruction_z[index])
        displacement = float(np.hypot(nr - rr, nz - rz))
        if displacement <= POSITION_TOLERANCE_M:
            continue
        records.append(
            PositionRecord(
                range_name=range_name,
                representative_shot=shot,
                loop=loop,
                nominal_r=nr,
                nominal_z=nz,
                reconstruction_r=rr,
                reconstruction_z=rz,
                reconstruction_index=index,
                identity_correlation=float(correlations[index]),
            )
        )
    return records


def _peak_plasma_current(store: Any) -> float:
    amc = store["amc"]
    if "plasma_current" not in amc:
        return float("nan")
    current = np.asarray(amc["plasma_current"], dtype=np.float64)
    return (
        float(np.nanmax(np.abs(current)))
        if np.isfinite(current).any()
        else float("nan")
    )


def _inspect_candidate(
    shot: int,
    *,
    range_name: str,
    required_loops: set[str],
    force_unconsolidated: bool = False,
) -> DiscoveryReceipt:
    store, metadata, attempts, mode = _open_level1(
        shot,
        required_groups=("amb", "amc"),
        force_unconsolidated=force_unconsolidated,
    )
    if store is None:
        return DiscoveryReceipt(
            range_name,
            shot,
            metadata,
            attempts,
            "unreadable",
            None,
            0,
            (),
            False,
            False,
            attempts[-1] if attempts else "store could not be opened",
        )
    amb = store["amb"]
    available = {key for key in amb.array_keys() if key.startswith("fl_")}
    missing = sorted(required_loops - available)
    if missing and mode != "unconsolidated fallback":
        retry_store, retry_metadata, retry_attempts, retry_mode = _open_level1(
            shot,
            required_groups=("amb", "amc"),
            force_unconsolidated=True,
        )
        attempts.extend(retry_attempts)
        if retry_store is not None:
            store, metadata, mode = retry_store, retry_metadata, retry_mode
            amb = store["amb"]
            available = {key for key in amb.array_keys() if key.startswith("fl_")}
            missing = sorted(required_loops - available)
    peak_ip = _peak_plasma_current(store)
    present = tuple(sorted(required_loops & available))
    if not present:
        reason = "no disputed loop signals present"
    elif not np.isfinite(peak_ip):
        reason = "measured plasma-current channel absent or non-finite"
    elif peak_ip >= VACUUM_CURRENT_LIMIT_KA:
        reason = (
            f"measured peak |Ip|={peak_ip:.3f} kA is not below "
            f"{VACUUM_CURRENT_LIMIT_KA:.1f} kA"
        )
    else:
        reason = (
            f"eligible with {len(present)}/{len(required_loops)} disputed loop signals"
        )
    return DiscoveryReceipt(
        range_name=range_name,
        shot=shot,
        metadata=metadata,
        attempts=attempts,
        open_mode=mode,
        peak_ip_ka=peak_ip if np.isfinite(peak_ip) else None,
        available_loops=len(available),
        present_loops=present,
        eligible=reason.startswith("eligible with"),
        selected=False,
        reason=reason,
    )


def _metadata_mentions(path: Path, loops: set[str]) -> bool:
    """Cheap physical-metadata prefilter before opening an expanded candidate."""

    for marker in (path / ".zmetadata", path / "zarr.json"):
        if not marker.is_file():
            continue
        try:
            text = marker.read_text(errors="ignore")
        except OSError:
            continue
        if any(f"amb/{loop}" in text for loop in loops):
            return True
    return False


def _discover_cohorts(
    records: list[PositionRecord],
) -> tuple[list[int], list[int], list[DiscoveryReceipt]]:
    early_loops = {row.loop for row in records if row.range_name == EARLY_RANGE}
    late_loops = {row.loop for row in records if row.range_name == LATE_RANGE}
    early_source = json.loads(EARLY_COHORT_SOURCE.read_text())
    dedicated_count = int(early_source["strata"]["dedicated_vacuum"]["n_shots"])
    early_candidates = [
        int(value) for value in early_source["shots_used"][-dedicated_count:]
    ]

    receipts = [
        _inspect_candidate(
            shot,
            range_name=EARLY_RANGE,
            required_loops=early_loops,
        )
        for shot in early_candidates
    ]
    early_selected = [row.shot for row in receipts if row.eligible]

    late_source = json.loads(LATE_CANDIDATE_SOURCE.read_text())
    inventory = [int(row["shot"]) for row in late_source["candidates"]]
    late_receipts = [
        _inspect_candidate(
            shot,
            range_name=LATE_RANGE,
            required_loops=late_loops,
        )
        for shot in inventory
    ]

    known_eligible = [
        shot
        for shot in KNOWN_LATE_SHOTS
        if any(row.shot == shot and row.eligible for row in late_receipts)
    ]
    if len(known_eligible) < len(KNOWN_LATE_SHOTS):
        by_shot = {row.shot: row for row in late_receipts}
        for shot in KNOWN_LATE_SHOTS:
            if shot in known_eligible:
                continue
            retried = _inspect_candidate(
                shot,
                range_name=LATE_RANGE,
                required_loops=late_loops,
                force_unconsolidated=True,
            )
            previous = by_shot.get(shot)
            if previous is not None:
                retried.attempts = previous.attempts + retried.attempts
                late_receipts[late_receipts.index(previous)] = retried
            else:
                late_receipts.append(retried)
            by_shot[shot] = retried
        known_eligible = [
            shot
            for shot in KNOWN_LATE_SHOTS
            if by_shot.get(shot) is not None and by_shot[shot].eligible
        ]

    if len(known_eligible) < len(KNOWN_LATE_SHOTS):
        raise RuntimeError(
            "one or more of the six known late shots failed explicit unconsolidated retry"
        )

    late_selected = list(known_eligible)
    receipt_by_shot = {row.shot: row for row in late_receipts}
    coverage = Counter()
    for shot in late_selected:
        coverage.update(receipt_by_shot[shot].present_loops)

    undercovered = {loop for loop in late_loops if coverage[loop] < MIN_LOOP_SHOTS}
    expanded_receipts: list[DiscoveryReceipt] = []
    inventory_set = set(inventory)
    if undercovered:
        full_inventory = sorted(
            int(value)
            for value in json.loads(FULL_SHOT_MANIFEST.read_text())["shot_ids"]
            if LATE_REPRESENTATIVE <= int(value) <= int(LATE_RANGE.split("-")[1])
        )
        for shot in full_inventory:
            if shot in inventory_set:
                continue
            path = LEVEL1_DIR / f"{shot}.zarr"
            if not _metadata_mentions(path, undercovered):
                continue
            receipt = _inspect_candidate(
                shot,
                range_name=LATE_RANGE,
                required_loops=late_loops,
            )
            expanded_receipts.append(receipt)
            if receipt.eligible and any(
                coverage[loop] < MIN_LOOP_SHOTS for loop in receipt.present_loops
            ):
                late_selected.append(shot)
                coverage.update(receipt.present_loops)
                undercovered = {
                    loop for loop in late_loops if coverage[loop] < MIN_LOOP_SHOTS
                }
            if not undercovered:
                break
    late_receipts.extend(expanded_receipts)
    if undercovered:
        details = ", ".join(f"{loop}={coverage[loop]}" for loop in sorted(undercovered))
        raise RuntimeError(
            f"late cohort cannot provide {MIN_LOOP_SHOTS} shots per loop: {details}"
        )

    selected = {(EARLY_RANGE, shot) for shot in early_selected} | {
        (LATE_RANGE, shot) for shot in late_selected
    }
    all_receipts = receipts + late_receipts
    for row in all_receipts:
        row.selected = (row.range_name, row.shot) in selected
        if row.eligible and not row.selected:
            row.reason = (
                f"{row.reason}; not selected because per-loop coverage was already met"
            )
    return early_selected, late_selected, all_receipts


def _build_candidate_table_without_amm(shot: int) -> OperatorGeometry:
    """Build a facade projection without passive structures."""

    return replace(_GEOMETRY_SERVICE.operator(shot), passives=())


def _operator_from_table(
    table: OperatorGeometry,
    records: list[PositionRecord],
) -> Any:
    mappings: list[_CandidateSensorMapping] = []
    for row in records:
        mappings.extend(
            [
                _CandidateSensorMapping(
                    amb_channel=f"nominal:{row.loop}",
                    kind="flux_loop",
                    efm_index=-1,
                    r=row.nominal_r,
                    z=row.nominal_z,
                    angle_deg=None,
                    residual_m=0.0,
                    flag="",
                ),
                _CandidateSensorMapping(
                    amb_channel=f"reconstruction:{row.loop}",
                    kind="flux_loop",
                    efm_index=row.reconstruction_index,
                    r=row.reconstruction_r,
                    z=row.reconstruction_z,
                    angle_deg=None,
                    residual_m=0.0,
                    flag="",
                ),
            ]
        )
    candidate_table = replace(
        table,
        sensor_map=tuple(mappings),
        unmatched_channels=(),
    )
    return build_operator(candidate_table)


def _candidate_operator(
    shot: int,
    records: list[PositionRecord],
    cache: dict[tuple[object, ...], Any],
) -> Any:
    # The map declares the machine description per shot range, so the geometry is
    # the range's, not the shot's; building it from the range representative
    # keeps a cohort shot the catalogue does not map (pre-11766 MAST pulses) from
    # requesting a description the selector cannot resolve.
    geometry_shot = records[0].representative_shot
    table = _build_candidate_table_without_amm(geometry_shot)
    key = (
        records[0].range_name,
        table.identity.representation_key,
        tuple(table.available_current_channels),
    )
    if key in cache:
        return cache[key]
    operator = _operator_from_table(table, records)
    cache[key] = operator
    return operator


def _passive_independence_receipt(
    records: list[PositionRecord],
) -> PassiveIndependenceReceipt:
    store, metadata, attempts, mode = _open_level1(
        EARLY_REPRESENTATIVE,
        required_groups=("amm",),
    )
    if store is None:
        raise RuntimeError(
            f"parity shot {EARLY_REPRESENTATIVE} does not expose amm after fallbacks: "
            f"{'; '.join(attempts)}"
        )
    table_with_read = _GEOMETRY_SERVICE.operator(EARLY_REPRESENTATIVE)
    if not table_with_read.passives:
        raise RuntimeError(
            f"parity shot {EARLY_REPRESENTATIVE} carries amm but yielded no structures"
        )
    table_without_read = _build_candidate_table_without_amm(EARLY_REPRESENTATIVE)
    operator_with_read = _operator_from_table(table_with_read, records)
    operator_without_read = _operator_from_table(table_without_read, records)
    with_columns = np.asarray(operator_with_read.g_pf, dtype=np.float64)
    without_columns = np.asarray(operator_without_read.g_pf, dtype=np.float64)
    if with_columns.shape != without_columns.shape:
        raise RuntimeError(
            "candidate g_pf shape changed when the amm read was omitted: "
            f"{with_columns.shape} versus {without_columns.shape}"
        )
    difference = float(np.max(np.abs(with_columns - without_columns)))
    if not np.array_equal(with_columns, without_columns):
        raise RuntimeError(
            "candidate g_pf columns differ when the amm read is omitted: "
            f"max absolute difference {difference:.17g}"
        )
    return PassiveIndependenceReceipt(
        shot=EARLY_REPRESENTATIVE,
        metadata=metadata,
        open_mode=mode,
        passive_structures_with_read=len(table_with_read.passives),
        passive_structures_without_read=len(table_without_read.passives),
        g_pf_shape=with_columns.shape,
        max_absolute_difference=difference,
    )


def _interpolate_channel(group: Any, name: str, target_time: np.ndarray) -> np.ndarray:
    if name not in group:
        return np.zeros(target_time.shape, dtype=np.float64)
    source_time = _group_time(group)
    values = np.asarray(group[name], dtype=np.float64)
    if source_time is None or values.shape != source_time.shape:
        return np.zeros(target_time.shape, dtype=np.float64)
    finite = np.isfinite(source_time) & np.isfinite(values)
    if int(finite.sum()) < 2:
        return np.zeros(target_time.shape, dtype=np.float64)
    return np.interp(target_time, source_time[finite], values[finite])


def _load_shot_signals(
    shot: int,
    range_name: str,
    records: list[PositionRecord],
    cache: dict[tuple[object, ...], Any],
) -> ShotSignals:
    store, metadata, attempts, mode = _open_level1(shot, required_groups=("amb", "amc"))
    if store is None:
        raise RuntimeError(f"selected shot {shot} unreadable: {'; '.join(attempts)}")
    peak_ip = _peak_plasma_current(store)
    if not np.isfinite(peak_ip) or peak_ip >= VACUUM_CURRENT_LIMIT_KA:
        raise RuntimeError(f"selected shot {shot} failed measured-current receipt")

    operator = _candidate_operator(shot, records, cache)
    amb = store["amb"]
    amc = store["amc"]
    time = _group_time(amb)
    if time is None:
        raise RuntimeError(f"selected shot {shot} has no amb time")
    median_dt = float(np.nanmedian(np.diff(time)))
    if not np.isfinite(median_dt) or median_dt <= 0.0:
        raise RuntimeError(f"selected shot {shot} has a non-monotonic amb time base")
    stride = max(1, int(round((1.0 / median_dt) / 1000.0)))

    current = np.column_stack(
        [
            1000.0 * _interpolate_channel(amc, channel, time)
            for channel in operator.pf_amc_channels
        ]
    )
    derivative = np.gradient(current, time, axis=0)
    scale = np.nanpercentile(
        np.abs(current - np.nanmedian(current, axis=0)), 90, axis=0
    )
    active = scale > 1e-6
    if not np.any(active):
        raise RuntimeError(f"selected shot {shot} has no varying known-coil current")
    activity = np.nanmax(np.abs(derivative[:, active]) / scale[active], axis=1)
    finite_activity = np.isfinite(activity)
    if int(finite_activity.sum()) < MIN_SHOT_SAMPLES * stride:
        raise RuntimeError(f"selected shot {shot} has too little finite coil activity")
    target_window_samples = int(round(SCORING_WINDOW_SECONDS / median_dt))
    window_samples = min(
        time.size,
        max(MIN_SHOT_SAMPLES * stride, target_window_samples),
    )
    window_kernel = np.ones(window_samples, dtype=np.float64)
    activity_sum = np.convolve(
        np.where(finite_activity, activity, 0.0), window_kernel, mode="valid"
    )
    activity_count = np.convolve(
        finite_activity.astype(np.float64), window_kernel, mode="valid"
    )
    window_activity = np.divide(
        activity_sum,
        activity_count,
        out=np.full(activity_sum.shape, np.inf),
        where=activity_count == window_samples,
    )
    if not np.any(np.isfinite(window_activity)):
        raise RuntimeError(f"selected shot {shot} has no fully finite scoring window")
    window_start = int(np.argmin(window_activity))
    window_stop = window_start + window_samples
    selected_index = np.arange(window_start, window_stop, stride, dtype=np.int64)
    selected_activity = float(window_activity[window_start])
    prediction = current @ np.asarray(operator.g_pf, dtype=np.float64).T
    channel_row = {name: index for index, name in enumerate(operator.sensor_channels)}
    values: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for row in records:
        if row.loop not in amb:
            continue
        measured = np.asarray(amb[row.loop], dtype=np.float64)
        if measured.shape != time.shape:
            continue
        nominal = prediction[:, channel_row[f"nominal:{row.loop}"]]
        reconstruction = prediction[:, channel_row[f"reconstruction:{row.loop}"]]
        good_index = selected_index[
            np.isfinite(measured[selected_index])
            & np.isfinite(nominal[selected_index])
            & np.isfinite(reconstruction[selected_index])
        ]
        if good_index.size < MIN_SHOT_SAMPLES:
            continue
        values[row.key] = (
            measured[good_index],
            nominal[good_index],
            reconstruction[good_index],
        )
    return ShotSignals(
        range_name=range_name,
        shot=shot,
        peak_ip_ka=peak_ip,
        metadata=metadata,
        open_mode=mode,
        n_samples=int(selected_index.size),
        window_start_s=float(time[window_start]),
        window_end_s=float(time[window_stop - 1]),
        normalized_activity=selected_activity,
        values=values,
    )


def _center(values: np.ndarray) -> np.ndarray:
    return values - np.mean(values)


def _gain(rows: list[tuple[np.ndarray, np.ndarray]], held_out: int | None) -> float:
    numerator = 0.0
    denominator = 0.0
    for index, (measured, predicted) in enumerate(rows):
        if held_out is not None and index == held_out:
            continue
        x = _center(predicted)
        y = _center(measured)
        numerator += float(np.sum(x * y))
        denominator += float(np.sum(x * x))
    return numerator / denominator if denominator > 0.0 else float("nan")


def _held_out_residuals(
    rows: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, float]:
    residuals: list[float] = []
    for index, (measured, predicted) in enumerate(rows):
        gain = _gain(rows, index)
        x = _center(predicted)
        y = _center(measured)
        scale = float(np.sqrt(np.mean(y * y)))
        residuals.append(
            float(np.sqrt(np.mean((y - gain * x) ** 2)) / scale)
            if scale > 0.0 and np.isfinite(gain)
            else float("nan")
        )
    return np.asarray(residuals, dtype=np.float64), _gain(rows, None)


def _score_records(
    records: list[PositionRecord], shots: list[ShotSignals], seed: int
) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    output: list[dict[str, object]] = []
    for record in records:
        available_shots = [shot for shot in shots if record.key in shot.values]
        nominal_rows = [
            (shot.values[record.key][0], shot.values[record.key][1])
            for shot in available_shots
        ]
        reconstruction_rows = [
            (shot.values[record.key][0], shot.values[record.key][2])
            for shot in available_shots
        ]
        nominal_residuals, nominal_gain = _held_out_residuals(nominal_rows)
        reconstruction_residuals, reconstruction_gain = _held_out_residuals(
            reconstruction_rows
        )
        good = np.isfinite(nominal_residuals) & np.isfinite(reconstruction_residuals)
        if int(good.sum()) < MIN_LOOP_SHOTS:
            raise RuntimeError(
                f"{record.key} has fewer than {MIN_LOOP_SHOTS} paired shot residuals"
            )
        nominal_value = float(np.mean(nominal_residuals[good]))
        reconstruction_value = float(np.mean(reconstruction_residuals[good]))
        differences = nominal_residuals[good] - reconstruction_residuals[good]
        draws = rng.choice(
            differences, size=(BOOTSTRAP_DRAWS, differences.size), replace=True
        )
        boot = np.mean(draws, axis=1)
        margin = float(np.mean(differences))
        lower, upper = [float(value) for value in np.percentile(boot, [2.5, 97.5])]
        if lower > 0.0:
            verdict = "reconstruction"
        elif upper < 0.0:
            verdict = "nominal-table"
        else:
            verdict = "null"
        output.append(
            {
                "key": record.key,
                "range": record.range_name,
                "loop": record.loop,
                "nominal_r": record.nominal_r,
                "nominal_z": record.nominal_z,
                "reconstruction_r": record.reconstruction_r,
                "reconstruction_z": record.reconstruction_z,
                "displacement_m": record.displacement_m,
                "identity_correlation": record.identity_correlation,
                "n_shots": int(good.sum()),
                "nominal_residual": nominal_value,
                "reconstruction_residual": reconstruction_value,
                "margin": margin,
                "margin_ci": [lower, upper],
                "nominal_gain": float(nominal_gain),
                "reconstruction_gain": float(reconstruction_gain),
                "verdict": verdict,
            }
        )
    return output


def _fmt(value: float | None, digits: int = 6) -> str:
    return "n/a" if value is None or not np.isfinite(value) else f"{value:.{digits}f}"


def _write_figures(
    scores: list[dict[str, object]], shots: list[ShotSignals]
) -> list[Path]:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    labels = [f"{row['range']}  {row['loop']}" for row in scores]
    nominal = np.array([row["nominal_residual"] for row in scores], dtype=float)
    reconstruction = np.array(
        [row["reconstruction_residual"] for row in scores], dtype=float
    )
    y = np.arange(len(scores))
    fig, ax = plt.subplots(figsize=(10.5, 8.0))
    for index in y:
        ax.plot(
            [nominal[index], reconstruction[index]],
            [index, index],
            color="#b8b8b8",
            linewidth=1.0,
            zorder=1,
        )
    ax.scatter(nominal, y, s=28, color="#8c2d24", label="nominal table", zorder=2)
    ax.scatter(
        reconstruction,
        y,
        s=28,
        color="#2166ac",
        label="reconstruction",
        zorder=2,
    )
    ax.set_yticks(y, labels=labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("shot-held-out normalized residual (lower is better)")
    ax.legend(frameon=False, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    residual_path = FIGURE_ROOT / "candidate-residual-comparison.png"
    fig.savefig(residual_path, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    range_color = {EARLY_RANGE: "#4d9221", LATE_RANGE: "#c51b7d"}
    for range_name in (EARLY_RANGE, LATE_RANGE):
        selected = [row for row in shots if row.range_name == range_name]
        ax.scatter(
            [row.shot for row in selected],
            [row.peak_ip_ka for row in selected],
            s=28,
            color=range_color[range_name],
            label=f"{range_name} (n={len(selected)})",
        )
    ax.axhline(
        VACUUM_CURRENT_LIMIT_KA,
        color="#333333",
        linestyle="--",
        linewidth=1.0,
        label=f"vacuum ceiling {VACUUM_CURRENT_LIMIT_KA:g} kA",
    )
    ax.set_xlabel("shot")
    ax.set_ylabel("measured peak |plasma current| (kA)")
    ax.legend(frameon=False, ncol=3, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    current_path = FIGURE_ROOT / "cohort-plasma-current-receipts.png"
    fig.savefig(current_path, dpi=180)
    plt.close(fig)
    return [residual_path, current_path]


def _source_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _write_note(
    records: list[PositionRecord],
    scores: list[dict[str, object]],
    shots: list[ShotSignals],
    receipts: list[DiscoveryReceipt],
    passive_receipt: PassiveIndependenceReceipt,
    figures: list[Path],
) -> None:
    counts = {
        verdict: sum(row["verdict"] == verdict for row in scores)
        for verdict in ("nominal-table", "reconstruction", "null")
    }
    early_shots = [row for row in shots if row.range_name == EARLY_RANGE]
    late_shots = [row for row in shots if row.range_name == LATE_RANGE]
    skipped = [row for row in receipts if not row.selected]
    lines = [
        "# Vacuum flux-loop position adjudication",
        "",
        "## Result",
        "",
        (
            f"The vacuum measurement adjudicates **{counts['nominal-table']} nominal-table**, "
            f"**{counts['reconstruction']} reconstruction**, and **{counts['null']} null** "
            f"range-qualified loop positions; the counts sum to **{len(scores)}**. "
            "Null rows remain explicitly undecided."
        ),
        "",
        (
            f"The cohort contains **{len(early_shots)} early-range** and "
            f"**{len(late_shots)} late-range** dedicated vacuum shots. Every selected "
            f"shot has a directly measured peak `|plasma_current| < {VACUUM_CURRENT_LIMIT_KA:g} kA`; "
            f"the observed range is {_fmt(min(row.peak_ip_ka for row in shots), 3)}-"
            f"{_fmt(max(row.peak_ip_ka for row in shots), 3)} kA. This is the corpus's "
            "measured-current vacuum criterion, not an assumption that the digitized channel is identically zero."
        ),
        "",
        f"![Candidate residual comparison](/imas-ambix/figures/vacuum-loop-adjudication/{figures[0].name})",
        "",
        "The connecting segment shows the inter-candidate margin for each range-qualified loop; lower residual is better.",
        "",
        f"![Measured plasma-current receipts](/imas-ambix/figures/vacuum-loop-adjudication/{figures[1].name})",
        "",
        "## Per-loop verdicts",
        "",
        (
            "Residuals are leave-one-shot-out normalized RMSE after removing each shot's constant "
            "offset and fitting one fixed acquisition gain on the remaining shots. `margin` is "
            "nominal residual minus reconstruction residual: positive values favour reconstruction. "
            "A verdict is directional only when the paired shot-bootstrap 95% interval excludes zero."
        ),
        "",
        "| Range | Loop | Nominal `(R,Z)` m | Reconstruction `(R,Z)` m | Separation m | Identity corr. | Shots | Nominal residual | Reconstruction residual | Margin [95% interval] | Verdict |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in scores:
        lower, upper = row["margin_ci"]
        lines.append(
            "| {range} | `{loop}` | ({nr:.6f}, {nz:.6f}) | ({rr:.6f}, {rz:.6f}) | "
            "{sep:.6f} | {corr:.9f} | {shots} | {nom:.6f} | {rec:.6f} | "
            "{margin:+.6f} [{lower:+.6f}, {upper:+.6f}] | **{verdict}** |".format(
                range=row["range"],
                loop=row["loop"],
                nr=row["nominal_r"],
                nz=row["nominal_z"],
                rr=row["reconstruction_r"],
                rz=row["reconstruction_z"],
                sep=row["displacement_m"],
                corr=row["identity_correlation"],
                shots=row["n_shots"],
                nom=row["nominal_residual"],
                rec=row["reconstruction_residual"],
                margin=row["margin"],
                lower=lower,
                upper=upper,
                verdict=row["verdict"],
            )
        )

    lines.extend(
        [
            "",
            "## Measurement and model boundary",
            "",
            "- Candidate identity is position-independent: each raw `amb` loop waveform was correlated against all experimental `efm/silop_x` columns on representatives 11766 and 12417. All 19 winning identity correlations exceed 0.9999.",
            "- The nominal candidate is the named level-2 `magnetics/flux_loop_{r,z}` row. The reconstruction candidate is the signal-identified static `efm/silop_{r,z}` row.",
            "- The prediction is **PF-only**: it uses raw measured `amc` PF and case currents converted from kA-turn to A, the static EFM coil geometry, and the finite-area cylinder forward operator. No plasma reconstruction, `silop_c`, fitted EFIT current, or `amm` current enters the prediction.",
            f"- Scoring compares both positions point-by-point inside one contiguous **{SCORING_WINDOW_SECONDS:.2f} s** window per shot, chosen for the lowest mean coil-current-normalized `|dI/dt|`. This suppresses passive-vessel transients without assembling unrelated quiet samples. Neglected passive-eddy flux is common to both candidate positions at fixed time, so it cancels to first order in the inter-candidate margin. A constant per-shot sensor offset and a fixed candidate-specific acquisition gain are nuisances; the held-out residual tests field-pattern agreement rather than digitizer scale.",
            f"- `amm` independence receipt: shot **{passive_receipt.shot}** exposes `amm` through **{passive_receipt.open_mode}** ({passive_receipt.metadata}). Building the same candidate operator with the normal `amm` geometry read ({passive_receipt.passive_structures_with_read} structures) and with no `amm` read ({passive_receipt.passive_structures_without_read} structures) produced `g_pf` shape **{passive_receipt.g_pf_shape}** and maximum absolute difference **{passive_receipt.max_absolute_difference:.17g}**. Exact array equality was required before cohort scoring.",
            "- This remains a position adjudication only while coil geometry is outside the disputed set. If coil geometry becomes disputed, the problem is a joint inverse problem and these verdicts lose their arbiter status.",
            f"- The six known late vacuum shots are retained. The full level-1 manifest supplies additional measured-current-qualified shots until every late disputed loop has at least {MIN_LOOP_SHOTS} independent shot residuals; no loop is dropped to fit the convenience cohort.",
            "",
            "## Selected-shot measured-current receipts",
            "",
            "| Range | Shot | Metadata | Read mode | Peak `|Ip|` kA | Scored loops | Window s | Mean normalized `|dI/dt|` | Retained samples |",
            "|---|---:|---|---|---:|---:|---|---:|---:|",
        ]
    )
    for row in sorted(shots, key=lambda value: (value.range_name, value.shot)):
        lines.append(
            f"| {row.range_name} | {row.shot} | {row.metadata} | {row.open_mode} | "
            f"{row.peak_ip_ka:.3f} | {len(row.values)} | "
            f"{row.window_start_s:.6f}-{row.window_end_s:.6f} | "
            f"{row.normalized_activity:.6g} | {row.n_samples} |"
        )

    lines.extend(
        [
            "",
            "## Candidate skip ledger",
            "",
            (
                f"The ledger names every inspected candidate excluded from the final cohort "
                f"(**{len(skipped)} rows**). A missing group or loop was accepted only after an "
                "explicit unconsolidated read attempt; valid rows not needed after every loop reached the shot-coverage floor are also named."
            ),
            "",
            "| Range | Shot | Metadata | Attempts | Peak `|Ip|` kA | All loop channels | Disputed signals present | Exclusion reason |",
            "|---|---:|---|---|---:|---:|---|---|",
        ]
    )
    for row in sorted(skipped, key=lambda value: (value.range_name, value.shot)):
        attempts = "; ".join(row.attempts).replace("|", "/")
        reason = row.reason.replace("|", "/")
        present = ", ".join(row.present_loops) if row.present_loops else "none"
        lines.append(
            f"| {row.range_name} | {row.shot} | {row.metadata} | {attempts} | "
            f"{_fmt(row.peak_ip_ka, 3)} | {row.available_loops} | {present} | {reason} |"
        )

    lines.extend(
        [
            "",
            "## Reproducibility",
            "",
            f"- Source-tree parent at measurement time: `{_source_commit()}`. The script and generated artifacts are committed together on top of that parent.",
            f"- Command: `PYTHONPATH=$PWD python scripts/{Path(__file__).name}` using the repository environment.",
            f"- Candidate records: {len(records)}; bootstrap draws per record: {BOOTSTRAP_DRAWS}; verdict total: {sum(counts.values())}.",
            f"- Scoring window: one contiguous nominal {SCORING_WINDOW_SECONDS:.2f} s minimum-activity interval per shot; exact endpoints and activity receipts are tabulated above.",
            f"- Coil-model marker: `{COIL_MODEL_VERSION}`.",
            "- Inputs are read-only Zarr stores beneath the configured level-1 and level-2 roots. This script writes only this note and the two figure files named above.",
            "",
        ]
    )
    NOTE_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTE_PATH.write_text("\n".join(lines))


JT60SA_SHOTS: tuple[int, ...] = (100579, 100595, 100642)
JT60SA_SLOPE_TOLERANCE = 0.1
JT60SA_CORRELATION_FLOOR = 0.95
#: A driven coil is quasi-static at a sample when its slew there is at most this
#: fraction of its own peak slew across the shot.
JT60SA_SLEW_FRACTION = 0.1
JT60SA_SWAP_CONTROL: tuple[str, str] = ("magFlxLp7", "magPbTC8")
JT60SA_EVIDENCE_JSON = Path(
    "docs/evidence/fragments/jt60sa-machine-map/jtmm-vacuum-adjudication.json"
)
JT60SA_EVIDENCE_HTML = Path(
    "docs/evidence/fragments/jt60sa-machine-map/jtmm-vacuum-adjudication.html"
)
JT60SA_LOOP_CHANNELS: tuple[str, ...] = tuple(
    f"magFlxLp{index}" for index in range(1, 28)
)
JT60SA_PROBE_CHANNELS: tuple[str, ...] = tuple(
    f"magPbTC{index}" for index in range(1, 18)
)


def _jt60sa_map_entry(catalog: Any, shot: int) -> Any:
    for entry in catalog.maps:
        if entry.first_shot <= shot <= entry.last_shot:
            return entry
    raise RuntimeError(f"no JT-60SA machine map covers shot {shot}")


def _jt60sa_coil_channels() -> list[str]:
    """Ordered PF coil-current channels the DD map binds, by coil index."""

    source_map = load_packaged_signal_map("jt-60sa", "pf_active")
    rows = sorted(source_map.signals, key=lambda rule: int(rule.target_index))
    return [rule.source_array for rule in rows]


def _jt60sa_operator(shot: int, catalog: Any, channels: list[str]) -> Any:
    """Build the OP description operator with catalogue drives and probe axes.

    Both the geometry and the probe sensitive axes come through the description
    selector: ``read_geometry_table`` emits the description and fills each
    probe's ``angle_deg`` from its ``probe_angle_source`` branch, so no reader
    here opens ``magnetics.nc`` directly.  The measured coil-current channels
    the catalogue's drive topology names are bound to the geometry table so the
    operator's own per-column scale carries the deck's turn totals.
    """

    from imas_ambix.data.description_reader import read_geometry_table  # noqa: PLC0415
    from imas_ambix.gs.geometry import CircuitDrive  # noqa: PLC0415

    table = read_geometry_table(shot, machine="jt-60sa")
    entry = _jt60sa_map_entry(catalog, shot)
    topology = next(
        item for item in catalog.drive_topologies if item.name == entry.drive_topology
    )
    order = list(
        dict.fromkeys(item.circuit_identifier for item in topology.connections)
    )
    per_ampere = {
        identity: float(
            sum(
                item.turns * item.current_weight * item.direction
                for item in topology.connections
                if item.circuit_identifier == identity
            )
        )
        for identity in order
    }
    drives = [
        CircuitDrive(
            circuit=index + 1,
            channel=channels[index],
            ampere_turns_per_ampere=per_ampere[identity],
            evidence="catalogue drive topology",
            conductor=identity,
        )
        for index, identity in enumerate(order)
        if index < len(channels)
    ]
    geometry = replace(
        table,
        circuit_drives=tuple(drives),
        amc_current_channels=list(channels),
    )
    return build_operator(geometry)


def _jt60sa_quasi_static_mask(slew: np.ndarray, fraction: float) -> np.ndarray:
    """Samples where every driven coil's ``|dI/dt|`` is below a fixed fraction
    of that coil's own peak ``|dI/dt|`` across the shot.

    The vacuum operator carries no passive currents, so a fast coil ramp
    induces eddy currents in the vessel that add to a near-wall probe's
    measured signal and bias its fit against the static prediction.  A sample
    is quasi-static when every coil's slew there is small beside that same
    coil's fastest slew in the shot, which is the shot's own reference for
    "the coils are running": the comparison is per coil, so a quiet coil is
    not held to a global rate it never reaches.  A coil whose slew is
    identically zero (an undriven coil) imposes no constraint.
    """

    slew = np.asarray(slew, dtype=np.float64)
    peak = np.nanmax(np.abs(slew), axis=0)
    reference = np.where(peak > 0.0, fraction * peak, 0.0)
    return np.all(np.abs(slew) <= reference[None, :], axis=1)


def _jt60sa_shot_data(shot: int, catalog: Any, channels: list[str]) -> dict[str, Any]:
    operator = _jt60sa_operator(shot, catalog, channels)
    records = [read_channel(JT60SA_ROOT, shot, "MMSYS", name) for name in channels]
    time = np.asarray(records[0].time, dtype=np.float64)
    raw = {
        name: np.interp(
            time,
            np.asarray(record.time, dtype=np.float64),
            np.asarray(record.data, dtype=np.float64).reshape(-1),
        )
        for name, record in zip(channels, records, strict=True)
    }
    # Every coil current is assembled through the operator's own per-column
    # scale rather than by repeating that scaling here.
    currents = np.empty((time.size, len(channels)), dtype=np.float64)
    for index in range(time.size):
        currents[index] = operator.assemble_pf_currents(
            {name: raw[name][index] for name in channels}
        )
    slew = np.gradient(currents, time, axis=0)
    mask = _jt60sa_quasi_static_mask(slew, JT60SA_SLEW_FRACTION)
    selected = np.flatnonzero(mask)
    if selected.size < MIN_SHOT_SAMPLES:
        raise RuntimeError(
            f"selected shot {shot} has {selected.size} quasi-static samples, "
            f"fewer than {MIN_SHOT_SAMPLES}"
        )
    predicted = np.vstack(
        [operator.vacuum_prediction(row) for row in currents[selected]]
    )
    row_index = {name: index for index, name in enumerate(operator.sensor_channels)}
    measured: dict[str, np.ndarray] = {}
    for channel in (*JT60SA_LOOP_CHANNELS, *JT60SA_PROBE_CHANNELS):
        record = read_channel(JT60SA_ROOT, shot, "MDAC", channel)
        aligned = np.interp(
            time,
            np.asarray(record.time, dtype=np.float64),
            np.asarray(record.data, dtype=np.float64).reshape(-1),
        )
        measured[channel] = aligned[selected]
    peak_slew = np.abs(slew).max(axis=0)
    return {
        "time": time[selected],
        "predicted": predicted,
        "row_index": row_index,
        "measured": measured,
        "n_samples": int(time.size),
        "n_quasi_static": int(selected.size),
        "slew_reference": peak_slew,
    }


def _jt60sa_fit(measured: np.ndarray, predicted: np.ndarray) -> dict[str, float] | None:
    good = np.isfinite(measured) & np.isfinite(predicted)
    if int(good.sum()) < 8:
        return None
    x = predicted[good] - float(np.mean(predicted[good]))
    y = measured[good] - float(np.mean(measured[good]))
    denominator = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
    slope = (
        float(np.sum(x * y) / np.sum(x * x)) if float(np.sum(x * x)) > 0.0 else np.nan
    )
    correlation = float(np.sum(x * y) / denominator) if denominator > 0.0 else np.nan
    return {
        "slope": slope,
        "offset": float(np.mean(measured[good]) - slope * np.mean(predicted[good])),
        "pearson_r": correlation,
        "n_samples": int(good.sum()),
    }


def _jt60sa_sensor_name(channel: str) -> str:
    """Map an EDDB acquisition channel to its description sensor name."""

    if channel.startswith("magFlxLp"):
        return f"FL{channel.removeprefix('magFlxLp')}"
    if channel.startswith("magPbTC"):
        return f"MP{channel.removeprefix('magPbTC')}"
    return channel


def _jt60sa_fit_table(
    data: dict[str, Any], swap: tuple[str, str] | None = None
) -> dict[str, dict[str, float] | None]:
    measured = dict(data["measured"])
    if swap is not None:
        left, right = swap
        measured[left], measured[right] = measured[right], measured[left]
    table: dict[str, dict[str, float] | None] = {}
    for channel in (*JT60SA_LOOP_CHANNELS, *JT60SA_PROBE_CHANNELS):
        index = data["row_index"].get(_jt60sa_sensor_name(channel))
        if index is None:
            table[channel] = None
            continue
        table[channel] = _jt60sa_fit(measured[channel], data["predicted"][:, index])
    return table


def _jt60sa_median_verdict(
    fits: list[tuple[int, dict[str, float] | None]],
) -> tuple[str, str, list[int]]:
    """Median verdict rule over the shot cohort.

    A channel is ``identity`` or ``negate`` when the median slope over the
    shots is within :data:`JT60SA_SLOPE_TOLERANCE` of +1 or -1, the median
    ``|r|`` exceeds :data:`JT60SA_CORRELATION_FLOOR`, and every shot has the
    same slope sign.  Otherwise it is ``undecided`` with the reason.  The
    returned outlier list names every shot whose slope sign differs from the
    cohort, whose slope is outside the tolerance of the winning direction, or
    whose ``|r|`` is at or below the floor.
    """

    missing = sorted(shot for shot, fit in fits if fit is None)
    if missing:
        return "undecided", f"no finite quasi-static fit on shots {missing}", []
    slopes = np.asarray([fit["slope"] for _, fit in fits], dtype=np.float64)
    correlations = np.asarray(
        [abs(fit["pearson_r"]) for _, fit in fits], dtype=np.float64
    )
    median_slope = float(np.median(slopes))
    median_r = float(np.median(correlations))
    same_sign = len({bool(value > 0.0) for value in slopes}) == 1
    detail = "; ".join(
        f"{shot} slope {fit['slope']:+.3f} r {fit['pearson_r']:+.4f}"
        for shot, fit in fits
    )
    positive = median_slope > 0.0
    target = 1.0 if positive else -1.0
    outliers = [
        shot
        for shot, fit in fits
        if (float(fit["slope"]) > 0.0) != positive
        or abs(float(fit["slope"]) - target) > JT60SA_SLOPE_TOLERANCE
        or abs(float(fit["pearson_r"])) <= JT60SA_CORRELATION_FLOOR
    ]
    if not same_sign:
        return "undecided", f"shots disagree in slope sign: {detail}", outliers
    if median_r <= JT60SA_CORRELATION_FLOOR:
        return (
            "undecided",
            f"median |r| {median_r:.3f} does not exceed "
            f"{JT60SA_CORRELATION_FLOOR:.2f}: {detail}",
            outliers,
        )
    if abs(median_slope - 1.0) <= JT60SA_SLOPE_TOLERANCE:
        verdict = "identity"
    elif abs(median_slope + 1.0) <= JT60SA_SLOPE_TOLERANCE:
        verdict = "negate"
    else:
        return (
            "undecided",
            f"median slope {median_slope:+.3f} is not within "
            f"{JT60SA_SLOPE_TOLERANCE:.1f} of +-1: {detail}",
            outliers,
        )
    return (
        verdict,
        f"median slope {median_slope:+.3f}, median |r| {median_r:.3f}, all shots "
        f"same sign; outliers: {outliers or 'none'}",
        outliers,
    )


def _jt60sa_html(payload: dict[str, Any]) -> str:
    counts = payload["verdict_counts"]
    rule = payload["rule"]
    samples = rule["quasi_static_samples"]
    sample_text = ", ".join(
        f"{shot}: {samples[str(shot)]['selected']}/{samples[str(shot)]['total']}"
        for shot in payload["shots"]
    )
    lines = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '  <meta charset="utf-8">',
        '  <meta name="viewport" content="width=device-width, initial-scale=1">',
        '  <meta name="docs-project" content="imas-ambix">',
        '  <meta name="reckon-type" content="evidence">',
        '  <meta name="plan-slug" content="jt60sa-machine-map">',
        '  <meta name="plan-evidence-for" content="jt60sa-machine-map">',
        "</head>",
        "<body>",
        '  <main class="plan-doc">',
        '    <section id="jtmm-vacuum-adjudication-landing" data-reckon="evidence">',
        "      <h3>Vacuum loop and probe adjudication</h3>",
        (
            "      <p>Every bound flux loop and tangential probe is predicted from the "
            "measured MMSYS coil currents through the OP1 description operator built by "
            "build_operator, and compared per shot against the raw EDDB channel on "
            "quasi-static samples only. A sample is quasi-static when every driven "
            f"coil's |dI/dt| is at most {rule['slew_fraction']:.2f} of that coil's own "
            f"peak |dI/dt| in the shot (retained/total: {sample_text}). A channel is "
            f"identity or negate when its median slope over the shots is within "
            f"{rule['slope_tolerance']:.1f} of +-1, its median |r| exceeds "
            f"{rule['correlation_floor']:.2f}, and every shot has the same sign. "
            f"Verdicts: <strong>{counts['identity']} identity</strong>, "
            f"<strong>{counts['negate']} negate</strong>, "
            f"<strong>{counts['undecided']} undecided</strong>.</p>"
        ),
        "      <table>",
        "        <thead><tr><th>Channel</th><th>Median slope</th>"
        "<th>Median |r|</th><th>Same sign</th><th>Per-shot slope</th>"
        "<th>Verdict</th></tr></thead>",
        "        <tbody>",
    ]
    for channel in payload["channel_order"]:
        row = payload["channels"][channel]
        slopes = ", ".join(
            f"{shot} {row['shots'][shot]['slope']:+.3f}"
            if row["shots"][shot] is not None
            else f"{shot} n/a"
            for shot in row["shots"]
        )
        lines.append(
            f"          <tr><td>{channel}</td>"
            f"<td>{row['median_slope']:+.6f}</td>"
            f"<td>{row['median_abs_r']:.6f}</td>"
            f"<td>{row['same_sign']}</td><td>{slopes}</td>"
            f"<td><strong>{row['verdict']}</strong></td></tr>"
        )
    lines.extend(
        [
            "        </tbody>",
            "      </table>",
        ]
    )
    for channel in payload["channel_order"]:
        row = payload["channels"][channel]
        outliers = row["outliers"]
        lines.append(
            f'      <p id="jtmm-vacuum-adjudication-{channel}">{channel}: '
            f"<strong>{row['verdict']}</strong> &mdash; {row['reason']}"
            + (f" Outliers: {outliers}." if outliers else "")
            + "</p>"
        )
    lines.extend(
        [
            (
                '      <p id="jtmm-vacuum-adjudication-negative-control">Negative '
                f"control: swapping the measured channels of {payload['swap_control'][0]} "
                f"and {payload['swap_control'][1]} changes both their verdicts "
                f"({payload['negative_control']['control_verdicts'][payload['swap_control'][0]]['before']}"
                f" &rarr; {payload['negative_control']['control_verdicts'][payload['swap_control'][0]]['after']}"
                f" and {payload['negative_control']['control_verdicts'][payload['swap_control'][1]]['before']}"
                f" &rarr; {payload['negative_control']['control_verdicts'][payload['swap_control'][1]]['after']}), "
                f"so {payload['negative_control']['changed_verdicts']} channel verdicts "
                "in total respond to the measured channel.</p>"
            ),
            "    </section>",
            "  </main>",
            "</body>",
            "</html>",
            "",
        ]
    )
    return "\n".join(lines)


def _run_jt60sa() -> int:
    catalog = load_packaged_machine_map("jt-60sa")
    channels = _jt60sa_coil_channels()
    shot_data = {
        shot: _jt60sa_shot_data(shot, catalog, channels) for shot in JT60SA_SHOTS
    }
    fit_tables = {shot: _jt60sa_fit_table(data) for shot, data in shot_data.items()}
    swap_tables = {
        shot: _jt60sa_fit_table(data, JT60SA_SWAP_CONTROL)
        for shot, data in shot_data.items()
    }
    channel_order = [*JT60SA_LOOP_CHANNELS, *JT60SA_PROBE_CHANNELS]
    control_pair = list(JT60SA_SWAP_CONTROL)
    channels_payload: dict[str, Any] = {}
    changed_verdicts = 0
    control_verdicts: dict[str, dict[str, str]] = {}
    undecided_reasons: dict[str, str] = {}
    for channel in channel_order:
        fits = [(shot, fit_tables[shot][channel]) for shot in JT60SA_SHOTS]
        verdict, reason, outliers = _jt60sa_median_verdict(fits)
        swapped = _jt60sa_median_verdict(
            [(shot, swap_tables[shot][channel]) for shot in JT60SA_SHOTS]
        )
        if swapped[0] != verdict:
            changed_verdicts += 1
        if channel in control_pair:
            control_verdicts[channel] = {"before": verdict, "after": swapped[0]}
        if verdict == "undecided":
            undecided_reasons[channel] = reason
        slopes = [fit["slope"] for _, fit in fits if fit is not None]
        correlations = [abs(fit["pearson_r"]) for _, fit in fits if fit is not None]
        channels_payload[channel] = {
            "verdict": verdict,
            "reason": reason,
            "outliers": outliers,
            "median_slope": float(np.median(slopes)) if slopes else None,
            "median_abs_r": float(np.median(correlations)) if correlations else None,
            "same_sign": len({bool(value > 0.0) for value in slopes}) == 1,
            "shots": {str(shot): fit_tables[shot][channel] for shot in JT60SA_SHOTS},
        }
    control_changed = [
        channel
        for channel in control_pair
        if control_verdicts.get(channel, {}).get("before")
        != control_verdicts.get(channel, {}).get("after")
    ]
    if len(control_changed) != len(control_pair):
        raise RuntimeError(
            "the swap control must change both channels' verdicts, but "
            f"{control_pair} changed {control_changed}: "
            f"{control_verdicts}"
        )
    counts = {
        name: sum(row["verdict"] == name for row in channels_payload.values())
        for name in ("identity", "negate", "undecided")
    }
    payload = {
        "node": "jtmm-vacuum-verdict-rule",
        "machine": "jt-60sa",
        "shots": list(JT60SA_SHOTS),
        "coil_channels": channels,
        "rule": {
            "slope_tolerance": JT60SA_SLOPE_TOLERANCE,
            "correlation_floor": JT60SA_CORRELATION_FLOOR,
            "slew_fraction": JT60SA_SLEW_FRACTION,
            "quasi_static_samples": {
                str(shot): {
                    "total": data["n_samples"],
                    "selected": data["n_quasi_static"],
                    "peak_slew_a_per_s": [
                        float(value) for value in data["slew_reference"]
                    ],
                }
                for shot, data in shot_data.items()
            },
        },
        "channel_order": channel_order,
        "channels": channels_payload,
        "verdict_counts": counts,
        "undecided": undecided_reasons,
        "swap_control": list(JT60SA_SWAP_CONTROL),
        "negative_control": {
            "changed_verdicts": changed_verdicts,
            "control_verdicts": control_verdicts,
        },
    }
    JT60SA_EVIDENCE_JSON.parent.mkdir(parents=True, exist_ok=True)
    JT60SA_EVIDENCE_JSON.write_text(json.dumps(payload, indent=2) + "\n")
    JT60SA_EVIDENCE_HTML.write_text(_jt60sa_html(payload))
    print(
        json.dumps(
            {
                "machine": "jt-60sa",
                "shots": list(JT60SA_SHOTS),
                "verdicts": counts,
                "undecided": len(undecided_reasons),
                "changed_verdicts_under_swap": changed_verdicts,
                "json": str(JT60SA_EVIDENCE_JSON),
                "html": str(JT60SA_EVIDENCE_HTML),
            },
            indent=2,
        )
    )
    return 0


def main() -> int:
    global NOTE_PATH, FIGURE_ROOT

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--machine", choices=("mast", "jt60sa"), default="mast")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Directory to write the MAST note and figures under, mirroring their "
            "committed relative layout. The default writes them to the repository "
            "paths; a scratch root redirects them so a receipt comparison does not "
            "touch the committed artifacts."
        ),
    )
    args = parser.parse_args()

    if args.machine == "jt60sa":
        return _run_jt60sa()
    if args.output_root is not None:
        NOTE_PATH = args.output_root / NOTE_PATH
        FIGURE_ROOT = args.output_root / FIGURE_ROOT

    early_records = _derive_positions(EARLY_REPRESENTATIVE, EARLY_RANGE)
    late_records = _derive_positions(LATE_REPRESENTATIVE, LATE_RANGE)
    records = early_records + late_records
    if len(early_records) != 5 or len(late_records) != 14 or len(records) != 19:
        raise RuntimeError(
            f"expected 5 early + 14 late = 19 disputes, got "
            f"{len(early_records)} + {len(late_records)} = {len(records)}"
        )
    if min(row.identity_correlation for row in records) <= 0.9999:
        raise RuntimeError("position-independent identity correlation floor failed")

    passive_receipt = _passive_independence_receipt(early_records)

    early_shots, late_shots, receipts = _discover_cohorts(records)
    if len(late_shots) < len(KNOWN_LATE_SHOTS):
        raise RuntimeError("late vacuum cohort is smaller than six")

    cache: dict[tuple[object, ...], Any] = {}
    shot_signals: list[ShotSignals] = []
    for range_name, shot_ids, range_records in (
        (EARLY_RANGE, early_shots, early_records),
        (LATE_RANGE, late_shots, late_records),
    ):
        for shot in shot_ids:
            shot_signals.append(
                _load_shot_signals(shot, range_name, range_records, cache)
            )

    scores = _score_records(
        early_records,
        [row for row in shot_signals if row.range_name == EARLY_RANGE],
        args.seed,
    ) + _score_records(
        late_records,
        [row for row in shot_signals if row.range_name == LATE_RANGE],
        args.seed + 1,
    )
    if len(scores) != 19:
        raise RuntimeError(f"expected 19 scores, got {len(scores)}")
    verdict_total = sum(
        sum(row["verdict"] == verdict for row in scores)
        for verdict in ("nominal-table", "reconstruction", "null")
    )
    if verdict_total != 19:
        raise RuntimeError(f"verdict counts sum to {verdict_total}, not 19")
    figures = _write_figures(scores, shot_signals)
    _write_note(records, scores, shot_signals, receipts, passive_receipt, figures)
    counts = {
        verdict: sum(row["verdict"] == verdict for row in scores)
        for verdict in ("nominal-table", "reconstruction", "null")
    }
    print(
        json.dumps(
            {
                "early_cohort": len(early_shots),
                "late_cohort": len(late_shots),
                "skip_ledger": sum(not row.selected for row in receipts),
                "amm_independence_max_abs_difference": (
                    passive_receipt.max_absolute_difference
                ),
                "verdicts": counts,
                "note": str(NOTE_PATH),
                "figures": [str(path) for path in figures],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
