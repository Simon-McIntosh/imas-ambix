"""Measure storage-arm read rates using identical MAST summary payloads."""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import os
import platform
import random
import socket
import subprocess
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import zarr

from imas_ambix.bench.store_arms import (
    FIXED_CHANNELS,
    FIXED_IDS_NAME,
    IDSZarrWriter,
    payload_to_ids,
    read_dd_zarr,
    read_fixed_native_payload,
    read_imas_netcdf,
    write_imas_netcdf,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

DEFAULT_LEVEL2_ROOT = Path("/work/projects/imas_gpu/mast/level2/shots")
DEFAULT_OUTPUT_ROOT = Path("/work/projects/imas_gpu/store-bench")
DEFAULT_LIVE_LEVEL2_ROOT = "https://s3.echo.stfc.ac.uk/mast/level2/shots"
DEFAULT_BATCH_SIZE = 4
DEFAULT_SLICE_SIZE = 64
DEFAULT_TRANSPORT_PROBE_COUNT = 24
DEFAULT_PAYLOAD_SCALES = (16, 64, 256, 1024)
DEFAULT_BATCH_SIZES = (1, 4, 8)
DEFAULT_SWEEP_REPETITIONS = 8
DEFAULT_REFRESH_SHOT_COUNT = 24
LEVEL2_CORPUS_SHOT_COUNT = 11_573
DEFAULT_S3_ENDPOINT = "https://s3.echo.stfc.ac.uk"
DEFAULT_S3_LEVEL2_ROOT = "s3://mast/level2/shots"
BENCHMARK_SHOTS = (
    11766,
    11767,
    11768,
    11769,
    11771,
    11772,
    11773,
    11776,
    11777,
    11815,
    11816,
    11817,
    11818,
    11819,
    11820,
    11821,
)
ARMS = ("native_level2", "imas_netcdf", "dd_annotated_zarr")
PATTERNS = (
    "whole_description_static_load",
    "single_shot_multi_signal_slice",
    "batch_stream",
    "random_corpus_access",
)


@dataclass(frozen=True)
class BenchmarkConfig:
    """Fixed workload counts used for every storage arm."""

    shots: tuple[int, ...] = BENCHMARK_SHOTS
    batch_size: int = DEFAULT_BATCH_SIZE
    slice_size: int = DEFAULT_SLICE_SIZE
    static_repetitions: int = 8
    slice_repetitions: int = 16
    batch_repetitions: int = 4
    random_windows: int = 64
    random_seed: int = 20260813
    transport_repetitions: int = 8


@dataclass(frozen=True)
class ReadCell:
    """One directly timed storage-arm and access-pattern cell."""

    arm: str
    access_pattern: str
    wall_seconds: float
    sample_count: int
    throughput_samples_per_second: float
    checksum: float


@dataclass(frozen=True)
class TransportCell:
    """One directly timed transport route for an identical zarr chunk."""

    route: str
    wall_seconds: float
    sample_count: int
    throughput_samples_per_second: float
    checksum: int


@dataclass(frozen=True)
class TransportProbe:
    """Identity evidence for one live and mapped payload pair."""

    shot: int
    array_path: str
    live_url: str
    mapped_path: str
    live_shape: tuple[int, ...]
    mapped_shape: tuple[int, ...]
    arrays_identical: bool


@dataclass(frozen=True)
class CrossoverCell:
    """One directly timed point in the payload-scale and batch-size grid."""

    arm: str
    payload_samples_per_shot: int
    payload_scalar_count_per_shot: int
    batch_size: int
    repetitions: int
    wall_seconds: float
    sample_count: int
    throughput_samples_per_second: float
    ratio_to_native_level2: float
    checksum: float


@dataclass(frozen=True)
class MirrorShotReceipt:
    """Refresh and identity evidence for one mirrored shot store."""

    shot: int
    live_url: str
    mapped_path: str
    live_shape: tuple[int, ...]
    mapped_shape: tuple[int, ...]
    arrays_identical: bool
    mirrored_file_count: int
    mirrored_bytes: int


@dataclass(frozen=True)
class PayloadReceipt:
    """Exact-parity evidence for materialized benchmark entries."""

    shots: int
    channels: int
    scalar_values_per_arm: int
    netcdf_writer: str
    zarr_writer: str
    zarr_base_class: str
    inherited_public_methods: tuple[str, ...]
    locally_defined_public_methods: tuple[str, ...]
    private_imas_names: tuple[str, ...]


ReadFunction = Callable[[int, int | None, int | None], tuple[np.ndarray, ...]]


def _package_versions() -> dict[str, str]:
    packages = ("imas-python", "numpy", "xarray", "zarr", "netCDF4")
    return {name: importlib.metadata.version(name) for name in packages}


def _node_identity() -> dict[str, str]:
    cpu_model = platform.processor()
    if not cpu_model:
        cpuinfo = Path("/proc/cpuinfo").read_text()
        for line in cpuinfo.splitlines():
            if line.startswith("model name"):
                cpu_model = line.partition(":")[2].strip()
                break
    partition = os.environ.get("SLURM_JOB_PARTITION", "not-under-slurm")
    return {
        "node_class": f"{partition}:{cpu_model or 'unknown-cpu'}",
        "partition": partition,
        "hostname": socket.gethostname(),
        "cpu_model": cpu_model or "unknown-cpu",
    }


def _native_reader(level2_root: Path) -> ReadFunction:
    names = tuple(binding.native_name for binding in FIXED_CHANNELS)

    def read(shot: int, start: int | None, stop: int | None) -> tuple[np.ndarray, ...]:
        group = zarr.open_group(level2_root / f"{shot}.zarr", mode="r")[FIXED_IDS_NAME]
        index = slice(start, stop)
        return tuple(np.asarray(group[name][index]) for name in names)

    return read


def _netcdf_reader(payload_root: Path) -> ReadFunction:
    paths = tuple(binding.dd_path for binding in FIXED_CHANNELS)

    def read(shot: int, start: int | None, stop: int | None) -> tuple[np.ndarray, ...]:
        arrays = read_imas_netcdf(payload_root / "netcdf" / f"{shot}.nc", paths)
        index = slice(start, stop)
        return tuple(arrays[path][index] for path in paths)

    return read


def _zarr_reader(payload_root: Path) -> ReadFunction:
    paths = tuple(binding.dd_path for binding in FIXED_CHANNELS)

    def read(shot: int, start: int | None, stop: int | None) -> tuple[np.ndarray, ...]:
        group = zarr.open_group(payload_root / "dd-zarr" / f"{shot}.zarr", mode="r")
        index = slice(start, stop)
        return tuple(np.asarray(group[path.replace("/", ".")][index]) for path in paths)

    return read


def materialize_payloads(
    level2_root: Path,
    payload_root: Path,
    shots: Sequence[int],
    sample_stop: int | None = None,
) -> tuple[dict[int, int], PayloadReceipt]:
    """Write and strictly re-read both converted arms for the fixed corpus."""
    payload_root.mkdir(parents=True, exist_ok=False)
    netcdf_root = payload_root / "netcdf"
    zarr_root = payload_root / "dd-zarr"
    netcdf_root.mkdir()
    zarr_root.mkdir()

    lengths: dict[int, int] = {}
    scalar_values = 0
    netcdf_writer = ""
    zarr_writer = ""
    zarr_base_class = ""
    inherited_methods: tuple[str, ...] = ()
    local_methods: tuple[str, ...] = ()
    private_names: tuple[str, ...] = ()

    for shot in shots:
        payload = read_fixed_native_payload(
            level2_root,
            shot=shot,
            sample_stop=sample_stop,
        )
        ids = payload_to_ids(payload)
        paths = tuple(payload.arrays)

        netcdf_receipt = write_imas_netcdf(ids, netcdf_root / f"{shot}.nc")
        zarr_receipt = IDSZarrWriter(ids, paths).write(zarr_root / f"{shot}.zarr")
        netcdf_arrays = read_imas_netcdf(netcdf_root / f"{shot}.nc", paths)
        zarr_arrays = read_dd_zarr(zarr_root / f"{shot}.zarr", paths)

        for path, native in payload.arrays.items():
            if not np.array_equal(native.values, netcdf_arrays[path]):
                raise AssertionError(f"netCDF parity failed for shot {shot}, {path}")
            if not np.array_equal(native.values, zarr_arrays[path]):
                raise AssertionError(f"zarr parity failed for shot {shot}, {path}")
            scalar_values += native.values.size

        lengths[shot] = payload.sample_count
        netcdf_writer = netcdf_receipt.entrypoint
        zarr_writer = zarr_receipt.entrypoint
        zarr_base_class = zarr_receipt.base_class or ""
        inherited_methods = tuple(
            method
            for method in zarr_receipt.called_methods + zarr_receipt.supporting_methods
            if method not in zarr_receipt.locally_defined_methods
        )
        local_methods = zarr_receipt.locally_defined_methods
        private_names = zarr_receipt.private_names

    receipt = PayloadReceipt(
        shots=len(shots),
        channels=len(shots) * len(FIXED_CHANNELS),
        scalar_values_per_arm=scalar_values,
        netcdf_writer=netcdf_writer,
        zarr_writer=zarr_writer,
        zarr_base_class=zarr_base_class,
        inherited_public_methods=inherited_methods,
        locally_defined_public_methods=local_methods,
        private_imas_names=private_names,
    )
    return lengths, receipt


def _consume(arrays: Sequence[np.ndarray]) -> tuple[int, float]:
    count = sum(array.size for array in arrays)
    checksum = sum(float(np.sum(array, dtype=np.float64)) for array in arrays)
    return count, checksum


def _measure(
    arm: str,
    pattern: str,
    operation: Callable[[], tuple[np.ndarray, ...]],
    repetitions: int,
) -> ReadCell:
    operation()
    gc.collect()
    sample_count = 0
    checksum = 0.0
    start = time.perf_counter()
    for _ in range(repetitions):
        count, value = _consume(operation())
        sample_count += count
        checksum += value
    wall_seconds = time.perf_counter() - start
    return ReadCell(
        arm=arm,
        access_pattern=pattern,
        wall_seconds=wall_seconds,
        sample_count=sample_count,
        throughput_samples_per_second=sample_count / wall_seconds,
        checksum=checksum,
    )


def measure_access_patterns(
    config: BenchmarkConfig,
    readers: Mapping[str, ReadFunction],
    lengths: Mapping[int, int],
) -> list[ReadCell]:
    """Time all three arms over the four predeclared access patterns."""
    if tuple(readers) != ARMS:
        raise ValueError(f"reader order must be {ARMS!r}")
    if config.batch_size > len(config.shots):
        raise ValueError("batch size exceeds the fixed corpus")
    if any(length < config.slice_size for length in lengths.values()):
        raise ValueError("every shot must contain one complete training slice")

    fixed_shot = config.shots[0]
    batch_shots = config.shots[: config.batch_size]
    random_source = random.Random(config.random_seed)
    random_windows = tuple(
        (
            shot,
            random_source.randrange(0, lengths[shot] - config.slice_size + 1),
        )
        for shot in (
            random_source.choice(config.shots) for _ in range(config.random_windows)
        )
    )

    cells: list[ReadCell] = []
    for arm, reader in readers.items():
        cells.append(
            _measure(
                arm,
                PATTERNS[0],
                lambda reader=reader: reader(fixed_shot, None, None),
                config.static_repetitions,
            )
        )
        cells.append(
            _measure(
                arm,
                PATTERNS[1],
                lambda reader=reader: reader(fixed_shot, 0, config.slice_size),
                config.slice_repetitions,
            )
        )

        def read_batch(reader: ReadFunction = reader) -> tuple[np.ndarray, ...]:
            return tuple(
                array
                for shot in batch_shots
                for array in reader(shot, 0, config.slice_size)
            )

        cells.append(
            _measure(
                arm,
                PATTERNS[2],
                read_batch,
                config.batch_repetitions,
            )
        )

        random_index = 0

        def read_random(reader: ReadFunction = reader) -> tuple[np.ndarray, ...]:
            nonlocal random_index
            shot, start = random_windows[random_index % len(random_windows)]
            random_index += 1
            return reader(shot, start, start + config.slice_size)

        cells.append(
            _measure(
                arm,
                PATTERNS[3],
                read_random,
                config.random_windows,
            )
        )
    return cells


def measure_crossover_grid(
    shots: Sequence[int],
    payload_scales: Sequence[int],
    batch_sizes: Sequence[int],
    readers_by_scale: Mapping[int, Mapping[str, ReadFunction]],
    repetitions: int,
) -> list[CrossoverCell]:
    """Directly time every arm in a payload-scale by batch-size grid."""
    cells: list[CrossoverCell] = []
    for payload_samples in payload_scales:
        readers = readers_by_scale[payload_samples]
        if tuple(readers) != ARMS:
            raise ValueError(f"reader order must be {ARMS!r}")
        for batch_size in batch_sizes:
            if batch_size > len(shots):
                raise ValueError("batch size exceeds the fixed corpus")
            batch_shots = shots[:batch_size]
            measured: list[ReadCell] = []
            for arm, reader in readers.items():

                def read_batch(
                    reader: ReadFunction = reader,
                    batch_shots: Sequence[int] = batch_shots,
                    payload_samples: int = payload_samples,
                ) -> tuple[np.ndarray, ...]:
                    return tuple(
                        array
                        for shot in batch_shots
                        for array in reader(shot, 0, payload_samples)
                    )

                measured.append(
                    _measure(
                        arm,
                        "payload_scale_batch_grid",
                        read_batch,
                        repetitions,
                    )
                )

            native_rate = measured[0].throughput_samples_per_second
            expected_count = (
                payload_samples
                * len(FIXED_CHANNELS)
                * batch_size
                * repetitions
            )
            for cell in measured:
                if cell.sample_count != expected_count:
                    raise AssertionError(
                        f"unexpected scalar count for {cell.arm}: "
                        f"{cell.sample_count} != {expected_count}"
                    )
                cells.append(
                    CrossoverCell(
                        arm=cell.arm,
                        payload_samples_per_shot=payload_samples,
                        payload_scalar_count_per_shot=(
                            payload_samples * len(FIXED_CHANNELS)
                        ),
                        batch_size=batch_size,
                        repetitions=repetitions,
                        wall_seconds=cell.wall_seconds,
                        sample_count=cell.sample_count,
                        throughput_samples_per_second=(
                            cell.throughput_samples_per_second
                        ),
                        ratio_to_native_level2=(
                            cell.throughput_samples_per_second / native_rate
                        ),
                        checksum=cell.checksum,
                    )
                )
    return cells


def summarize_crossovers(
    cells: Sequence[CrossoverCell],
    payload_scales: Sequence[int],
    batch_sizes: Sequence[int],
) -> dict[str, Any]:
    """Report observed ranking changes along both dimensions of the grid."""

    def point(
        payload_samples: int,
        batch_size: int,
    ) -> dict[str, Any]:
        group = [
            cell
            for cell in cells
            if cell.payload_samples_per_shot == payload_samples
            and cell.batch_size == batch_size
        ]
        if {cell.arm for cell in group} != set(ARMS):
            raise ValueError("every grid point must contain all three storage arms")
        ranked = sorted(
            group,
            key=lambda cell: cell.throughput_samples_per_second,
            reverse=True,
        )
        return {
            "payload_samples_per_shot": payload_samples,
            "payload_scalar_count_per_shot": (
                payload_samples * len(FIXED_CHANNELS)
            ),
            "batch_size": batch_size,
            "ranking_fastest_first": [cell.arm for cell in ranked],
            "ratios_to_native_level2": {
                cell.arm: cell.ratio_to_native_level2 for cell in group
            },
        }

    payload_series = []
    payload_crossings = []
    for batch_size in batch_sizes:
        points = [point(scale, batch_size) for scale in payload_scales]
        payload_series.append({"batch_size": batch_size, "points": points})
        for before, after in zip(points, points[1:], strict=False):
            if before["ranking_fastest_first"] != after["ranking_fastest_first"]:
                payload_crossings.append(
                    {
                        "batch_size": batch_size,
                        "after_payload_scalar_count_per_shot": after[
                            "payload_scalar_count_per_shot"
                        ],
                        "bracket_scalar_counts_per_shot": [
                            before["payload_scalar_count_per_shot"],
                            after["payload_scalar_count_per_shot"],
                        ],
                        "ranking_before": before["ranking_fastest_first"],
                        "ranking_after": after["ranking_fastest_first"],
                    }
                )

    batch_series = []
    batch_crossings = []
    for payload_samples in payload_scales:
        points = [point(payload_samples, size) for size in batch_sizes]
        batch_series.append(
            {
                "payload_scalar_count_per_shot": (
                    payload_samples * len(FIXED_CHANNELS)
                ),
                "points": points,
            }
        )
        for before, after in zip(points, points[1:], strict=False):
            if before["ranking_fastest_first"] != after["ranking_fastest_first"]:
                batch_crossings.append(
                    {
                        "payload_scalar_count_per_shot": before[
                            "payload_scalar_count_per_shot"
                        ],
                        "after_batch_size": after["batch_size"],
                        "bracket_batch_sizes": [
                            before["batch_size"],
                            after["batch_size"],
                        ],
                        "ranking_before": before["ranking_fastest_first"],
                        "ranking_after": after["ranking_fastest_first"],
                    }
                )

    return {
        "payload_scale": {
            "crossing_count": len(payload_crossings),
            "crossings": payload_crossings,
            "series": payload_series,
        },
        "batch_size": {
            "crossing_count": len(batch_crossings),
            "crossings": batch_crossings,
            "series": batch_series,
        },
    }


def run_crossover_sweep(
    output_log: Path,
    payload_root: Path,
    level2_root: Path = DEFAULT_LEVEL2_ROOT,
    payload_scales: Sequence[int] = DEFAULT_PAYLOAD_SCALES,
    batch_sizes: Sequence[int] = DEFAULT_BATCH_SIZES,
    repetitions: int = DEFAULT_SWEEP_REPETITIONS,
) -> dict[str, Any]:
    """Materialize exact payloads and persist the full crossover sweep."""
    payload_scales = tuple(payload_scales)
    batch_sizes = tuple(batch_sizes)
    if len(payload_scales) < 4:
        raise ValueError("the crossover sweep requires at least four payload scales")
    if min(payload_scales) <= 0 or max(payload_scales) / min(payload_scales) < 10:
        raise ValueError("payload scalar counts must span at least tenfold")
    if len(batch_sizes) < 3 or min(batch_sizes) <= 0:
        raise ValueError("the crossover sweep requires at least three batch sizes")
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    if output_log.exists():
        raise FileExistsError(output_log)
    if payload_root.exists():
        raise FileExistsError(payload_root)
    output_log.parent.mkdir(parents=True, exist_ok=True)
    payload_root.mkdir(parents=True)

    shots = BENCHMARK_SHOTS[: max(batch_sizes)]
    native_reader = _native_reader(level2_root)
    readers_by_scale: dict[int, dict[str, ReadFunction]] = {}
    parity_receipts: dict[int, dict[str, Any]] = {}
    for payload_samples in payload_scales:
        scale_root = payload_root / f"samples-{payload_samples}"
        _, receipt = materialize_payloads(
            level2_root,
            scale_root,
            shots,
            sample_stop=payload_samples,
        )
        parity_receipts[payload_samples] = asdict(receipt)
        readers_by_scale[payload_samples] = {
            ARMS[0]: native_reader,
            ARMS[1]: _netcdf_reader(scale_root),
            ARMS[2]: _zarr_reader(scale_root),
        }

    cells = measure_crossover_grid(
        shots,
        payload_scales,
        batch_sizes,
        readers_by_scale,
        repetitions,
    )
    for payload_samples in payload_scales:
        for batch_size in batch_sizes:
            checksums = [
                cell.checksum
                for cell in cells
                if cell.payload_samples_per_shot == payload_samples
                and cell.batch_size == batch_size
            ]
            if not np.array_equal(checksums, np.repeat(checksums[0], len(checksums))):
                raise AssertionError(
                    "storage-arm checksums differ at "
                    f"payload_samples={payload_samples}, batch_size={batch_size}"
                )

    scalar_counts = tuple(scale * len(FIXED_CHANNELS) for scale in payload_scales)
    result = {
        "schema": "imas-ambix-store-read-crossover",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "node": _node_identity(),
        "environment": _package_versions(),
        "config": {
            "shots": list(shots),
            "payload_samples_per_shot": list(payload_scales),
            "payload_scalar_counts_per_shot": list(scalar_counts),
            "payload_scalar_range_factor": max(scalar_counts) / min(scalar_counts),
            "batch_sizes": list(batch_sizes),
            "repetitions_per_cell": repetitions,
        },
        "measurement_contract": {
            "grid": "three arms by payload scale by batch size",
            "sample_unit": "one scalar array value returned to the process",
            "throughput_formula": (
                "actual accumulated sample_count / perf_counter wall_seconds"
            ),
            "assumed_rate_or_size_used": False,
        },
        "payload_parity": parity_receipts,
        "cells": [asdict(cell) for cell in cells],
        "cell_count": len(cells),
        "crossovers": summarize_crossovers(cells, payload_scales, batch_sizes),
    }
    output_log.write_text(json.dumps(result, indent=2) + "\n")
    return result


def measure_transport(
    local_array_path: Path,
    remote_url: str,
    repetitions: int,
) -> tuple[list[TransportCell], dict[str, Any]]:
    """Compare decoded live HTTPS reads with an already-mapped zarr array."""
    local_probe = np.asarray(zarr.open_array(local_array_path, mode="r")[:])
    remote_probe = np.asarray(zarr.open_array(remote_url, mode="r")[:])
    arrays_identical = bool(
        local_probe.shape == remote_probe.shape
        and np.array_equal(local_probe, remote_probe)
    )
    identity = {
        "remote_url": remote_url,
        "local_path": str(local_array_path),
        "remote_shape": list(remote_probe.shape),
        "mapped_shape": list(local_probe.shape),
        "decoded_arrays_identical": arrays_identical,
    }
    print("TRANSPORT_IDENTITY " + json.dumps(identity, sort_keys=True), flush=True)
    if not arrays_identical:
        raise ValueError("transport timing requires numerically identical payloads")

    live_checksum = 0.0
    live_samples = 0
    start = time.perf_counter()
    for _ in range(repetitions):
        values = np.asarray(zarr.open_array(remote_url, mode="r")[:])
        live_checksum += float(np.sum(values, dtype=np.float64))
        live_samples += values.size
    live_seconds = time.perf_counter() - start

    mapped_checksum = 0.0
    mapped_samples = 0
    start = time.perf_counter()
    for _ in range(repetitions):
        values = np.asarray(zarr.open_array(local_array_path, mode="r")[:])
        mapped_checksum += float(np.sum(values, dtype=np.float64))
        mapped_samples += values.size
    mapped_seconds = time.perf_counter() - start

    cells = [
        TransportCell(
            route="live_https_transfer",
            wall_seconds=live_seconds,
            sample_count=live_samples,
            throughput_samples_per_second=live_samples / live_seconds,
            checksum=int(live_checksum),
        ),
        TransportCell(
            route="mapped_on_disk",
            wall_seconds=mapped_seconds,
            sample_count=mapped_samples,
            throughput_samples_per_second=mapped_samples / mapped_seconds,
            checksum=int(mapped_checksum),
        ),
    ]
    return cells, identity


def mapped_shots(
    level2_root: Path,
    array_path: str = "summary/ip",
) -> tuple[int, ...]:
    """List mapped shots containing the requested zarr array."""
    shots = []
    for entry in level2_root.glob("*.zarr"):
        try:
            shot = int(entry.stem)
        except ValueError:
            continue
        if (entry / array_path).is_dir():
            shots.append(shot)
    return tuple(sorted(shots))


def refresh_mapped_shots(
    mirror_root: Path,
    shots: Sequence[int],
    array_path: str = "summary/ip",
    live_root: str = DEFAULT_LIVE_LEVEL2_ROOT,
    s3_root: str = DEFAULT_S3_LEVEL2_ROOT,
    s3_endpoint: str = DEFAULT_S3_ENDPOINT,
    copy_executable: str = "s5cmd",
) -> list[MirrorShotReceipt]:
    """Download complete public shot stores and prove one array identical."""
    if mirror_root.exists():
        raise FileExistsError(mirror_root)
    mirror_root.mkdir(parents=True)
    receipts = []
    for shot in shots:
        shot_root = mirror_root / f"{shot}.zarr"
        subprocess.run(
            [
                copy_executable,
                "--endpoint-url",
                s3_endpoint,
                "--no-sign-request",
                "--numworkers",
                "32",
                "cp",
                f"{s3_root.rstrip('/')}/{shot}.zarr/*",
                f"{shot_root}/",
            ],
            check=True,
        )
        mapped_path = shot_root / array_path
        live_url = f"{live_root.rstrip('/')}/{shot}.zarr/{array_path}"
        mapped_values = np.asarray(zarr.open_array(mapped_path, mode="r")[:])
        live_values = np.asarray(zarr.open_array(live_url, mode="r")[:])
        arrays_identical = bool(
            mapped_values.shape == live_values.shape
            and np.array_equal(mapped_values, live_values)
        )
        files = tuple(path for path in shot_root.rglob("*") if path.is_file())
        receipt = MirrorShotReceipt(
            shot=shot,
            live_url=live_url,
            mapped_path=str(mapped_path),
            live_shape=tuple(live_values.shape),
            mapped_shape=tuple(mapped_values.shape),
            arrays_identical=arrays_identical,
            mirrored_file_count=len(files),
            mirrored_bytes=sum(path.stat().st_size for path in files),
        )
        receipts.append(receipt)
        print(
            "MIRROR_REFRESH " + json.dumps(asdict(receipt), sort_keys=True),
            flush=True,
        )
    return receipts


def run_mirror_refresh(
    output_log: Path,
    mirror_root: Path,
    source_mirror_root: Path = DEFAULT_LEVEL2_ROOT,
    refresh_shot_count: int = DEFAULT_REFRESH_SHOT_COUNT,
    corpus_shot_count: int = LEVEL2_CORPUS_SHOT_COUNT,
    minimum_identical_shots: int = 20,
    live_root: str = DEFAULT_LIVE_LEVEL2_ROOT,
    s3_root: str = DEFAULT_S3_LEVEL2_ROOT,
    s3_endpoint: str = DEFAULT_S3_ENDPOINT,
    copy_executable: str = "s5cmd",
) -> dict[str, Any]:
    """Persist the scope and exact identity result of a partial mirror refresh."""
    if output_log.exists():
        raise FileExistsError(output_log)
    if refresh_shot_count < minimum_identical_shots:
        raise ValueError(
            "refresh shot count must permit the minimum identical-shot evidence"
        )
    available_shots = mapped_shots(source_mirror_root)
    shots = available_shots[:refresh_shot_count]
    if len(shots) != refresh_shot_count:
        raise ValueError(
            f"requested {refresh_shot_count} shots but found {len(shots)} candidates"
        )
    print(
        "MIRROR_SCOPE "
        + json.dumps(
            {
                "selected_shot_count": len(shots),
                "corpus_shot_count": corpus_shot_count,
                "coverage_fraction": len(shots) / corpus_shot_count,
                "shots": shots,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    receipts = refresh_mapped_shots(
        mirror_root,
        shots,
        live_root=live_root,
        s3_root=s3_root,
        s3_endpoint=s3_endpoint,
        copy_executable=copy_executable,
    )
    identical_count = sum(receipt.arrays_identical for receipt in receipts)
    disagreements = [
        {
            "shot": receipt.shot,
            "live_shape": list(receipt.live_shape),
            "mapped_shape": list(receipt.mapped_shape),
        }
        for receipt in receipts
        if not receipt.arrays_identical
    ]
    if identical_count < minimum_identical_shots:
        raise AssertionError(
            f"only {identical_count} refreshed shots are numerically identical"
        )
    result = {
        "schema": "imas-ambix-level2-mirror-refresh",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "node": _node_identity(),
        "environment": _package_versions(),
        "mirror_root": str(mirror_root),
        "scope": {
            "refreshed_shot_count": len(receipts),
            "corpus_shot_count": corpus_shot_count,
            "coverage_fraction": len(receipts) / corpus_shot_count,
            "partial_refresh": len(receipts) < corpus_shot_count,
            "shots": list(shots),
        },
        "identity": {
            "array_path": "summary/ip",
            "identical_shot_count": identical_count,
            "disagreeing_shot_count": len(disagreements),
            "disagreements": disagreements,
        },
        "receipts": [asdict(receipt) for receipt in receipts],
        "mirrored_file_count": sum(
            receipt.mirrored_file_count for receipt in receipts
        ),
        "mirrored_bytes": sum(receipt.mirrored_bytes for receipt in receipts),
    }
    output_log.parent.mkdir(parents=True, exist_ok=True)
    output_log.write_text(json.dumps(result, indent=2) + "\n")
    return result


def find_matching_transport_payload(
    level2_root: Path,
    shots: Sequence[int],
    array_path: str = "summary/ip",
    live_root: str = DEFAULT_LIVE_LEVEL2_ROOT,
) -> tuple[TransportProbe | None, list[TransportProbe]]:
    """Probe live and mapped arrays in order, stopping at exact equality."""
    probes: list[TransportProbe] = []
    for shot in shots:
        mapped_path = level2_root / f"{shot}.zarr" / array_path
        live_url = f"{live_root.rstrip('/')}/{shot}.zarr/{array_path}"
        mapped_array = zarr.open_array(mapped_path, mode="r")
        live_array = zarr.open_array(live_url, mode="r")
        shapes_equal = mapped_array.shape == live_array.shape
        arrays_identical = bool(
            shapes_equal
            and np.array_equal(
                np.asarray(mapped_array[:]),
                np.asarray(live_array[:]),
            )
        )
        probe = TransportProbe(
            shot=shot,
            array_path=array_path,
            live_url=live_url,
            mapped_path=str(mapped_path),
            live_shape=tuple(live_array.shape),
            mapped_shape=tuple(mapped_array.shape),
            arrays_identical=arrays_identical,
        )
        probes.append(probe)
        print(
            "TRANSPORT_PROBE " + json.dumps(asdict(probe), sort_keys=True),
            flush=True,
        )
        if arrays_identical:
            return probe, probes
    return None, probes


def run_transport_comparison(
    output_log: Path,
    level2_root: Path = DEFAULT_LEVEL2_ROOT,
    shots: Sequence[int] | None = None,
    probe_count: int = DEFAULT_TRANSPORT_PROBE_COUNT,
    repetitions: int = 8,
    array_path: str = "summary/ip",
    live_root: str = DEFAULT_LIVE_LEVEL2_ROOT,
    minimum_negative_probes: int = 20,
) -> dict[str, Any]:
    """Persist an identity-gated live-versus-mapped transport comparison."""
    if output_log.exists():
        raise FileExistsError(output_log)
    output_log.parent.mkdir(parents=True, exist_ok=True)
    candidate_shots = tuple(shots or mapped_shots(level2_root, array_path))[
        :probe_count
    ]
    match, probes = find_matching_transport_payload(
        level2_root,
        candidate_shots,
        array_path=array_path,
        live_root=live_root,
    )
    if match is None and len(probes) < minimum_negative_probes:
        raise ValueError(
            "an unobtainable verdict requires at least "
            f"{minimum_negative_probes} completed probes"
        )

    cells: list[TransportCell] = []
    identity: dict[str, Any] | None = None
    status = "unobtainable_without_mirror_refresh"
    if match is not None:
        print(
            "TRANSPORT_TIMING_START "
            + json.dumps(
                {"shot": match.shot, "array_path": match.array_path},
                sort_keys=True,
            ),
            flush=True,
        )
        cells, identity = measure_transport(
            Path(match.mapped_path),
            match.live_url,
            repetitions,
        )
        status = "measured_identical_payload"

    result = {
        "schema": "imas-ambix-identical-payload-transport",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "node": _node_identity(),
        "environment": _package_versions(),
        "status": status,
        "identity_gate": {
            "array_path": array_path,
            "candidate_count": len(candidate_shots),
            "completed_probe_count": len(probes),
            "minimum_negative_probe_count": minimum_negative_probes,
            "matching_shot": match.shot if match else None,
            "agreement_printed_before_timing": match is not None,
        },
        "probes": [asdict(probe) for probe in probes],
        "transport_cells": [asdict(cell) for cell in cells],
        "transport": identity,
        "measurement_contract": {
            "timing_requires_shape_equality": True,
            "timing_requires_np_array_equal": True,
            "sample_unit": "one scalar array value returned to the process",
            "throughput_formula": (
                "actual accumulated sample_count / perf_counter wall_seconds"
            ),
            "assumed_rate_or_size_used": False,
        },
    }
    output_log.write_text(json.dumps(result, indent=2) + "\n")
    return result


def run_benchmark(
    output_log: Path,
    payload_root: Path,
    level2_root: Path = DEFAULT_LEVEL2_ROOT,
    config: BenchmarkConfig | None = None,
) -> dict[str, Any]:
    """Materialize, verify, measure, and persist the complete benchmark record."""
    config = config or BenchmarkConfig()
    if output_log.exists():
        raise FileExistsError(output_log)
    output_log.parent.mkdir(parents=True, exist_ok=True)
    lengths, payload_receipt = materialize_payloads(
        level2_root, payload_root, config.shots
    )
    readers = {
        ARMS[0]: _native_reader(level2_root),
        ARMS[1]: _netcdf_reader(payload_root),
        ARMS[2]: _zarr_reader(payload_root),
    }
    cells = measure_access_patterns(config, readers, lengths)

    transport_match, transport_probes = find_matching_transport_payload(
        level2_root,
        config.shots,
    )
    transport_cells: list[TransportCell] = []
    transport_evidence: dict[str, Any] = {
        "status": "unobtainable_without_mirror_refresh",
        "probes": [asdict(probe) for probe in transport_probes],
    }
    if transport_match is not None:
        transport_cells, identity = measure_transport(
            Path(transport_match.mapped_path),
            transport_match.live_url,
            config.transport_repetitions,
        )
        transport_evidence = {
            "status": "measured_identical_payload",
            **identity,
            "probes": [asdict(probe) for probe in transport_probes],
        }

    versions = _package_versions()
    base_methods = {
        name for name in dir(IDSZarrWriter.__mro__[1]) if not name.startswith("_")
    }
    direct_seam_missing = sorted(
        set(payload_receipt.locally_defined_public_methods) - base_methods
    )
    checksum_groups = {
        pattern: [cell.checksum for cell in cells if cell.access_pattern == pattern]
        for pattern in PATTERNS
    }
    if not all(
        np.array_equal(values, np.repeat(values[0], len(values)))
        for values in checksum_groups.values()
    ):
        raise AssertionError("storage-arm read checksums differ")
    result = {
        "schema": "imas-ambix-store-read-rates",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "node": _node_identity(),
        "environment": versions,
        "config": asdict(config),
        "measurement_contract": {
            "sample_unit": "one scalar array value returned to the process",
            "throughput_formula": (
                "actual accumulated sample_count / perf_counter wall_seconds"
            ),
            "assumed_rate_or_size_used": False,
            "access_patterns": {
                PATTERNS[0]: "all three full arrays for fixed shot 11766",
                PATTERNS[1]: "one 64-sample, three-signal window from shot 11766",
                PATTERNS[2]: "four shots per batch, matching training micro-batch size",
                PATTERNS[3]: "64 seeded random 64-sample windows across 16 shots",
            },
        },
        "payload_root": str(payload_root),
        "payload_parity": asdict(payload_receipt),
        "tensorizer_seam": {
            "verdict": "refuted_for_zero_wrapper_subclass",
            "resolved_imas_python": versions["imas-python"],
            "base_class": payload_receipt.zarr_base_class,
            "missing_public_base_methods": direct_seam_missing,
            "private_names_reached": list(payload_receipt.private_imas_names),
            "converted_writer_verdict": "verified_with_public_local_helpers",
        },
        "read_cells": [asdict(cell) for cell in cells],
        "read_cell_count": len(cells),
        "read_checksums_match_across_arms": True,
        "transport_cells": [asdict(cell) for cell in transport_cells],
        "transport": transport_evidence,
    }
    output_log.write_text(json.dumps(result, indent=2) + "\n")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-log", type=Path, required=True)
    parser.add_argument("--payload-root", type=Path)
    parser.add_argument("--level2-root", type=Path, default=DEFAULT_LEVEL2_ROOT)
    parser.add_argument("--transport-only", action="store_true")
    parser.add_argument("--crossover-only", action="store_true")
    parser.add_argument("--refresh-only", action="store_true")
    parser.add_argument("--mirror-root", type=Path)
    parser.add_argument(
        "--transport-probe-count",
        type=int,
        default=DEFAULT_TRANSPORT_PROBE_COUNT,
    )
    parser.add_argument("--transport-repetitions", type=int, default=8)
    parser.add_argument(
        "--sweep-repetitions",
        type=int,
        default=DEFAULT_SWEEP_REPETITIONS,
    )
    parser.add_argument(
        "--refresh-shot-count",
        type=int,
        default=DEFAULT_REFRESH_SHOT_COUNT,
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.transport_only:
        result = run_transport_comparison(
            args.output_log,
            level2_root=args.level2_root,
            probe_count=args.transport_probe_count,
            repetitions=args.transport_repetitions,
        )
    elif args.refresh_only:
        if args.mirror_root is None:
            raise SystemExit("--mirror-root is required for --refresh-only")
        result = run_mirror_refresh(
            args.output_log,
            args.mirror_root,
            source_mirror_root=args.level2_root,
            refresh_shot_count=args.refresh_shot_count,
        )
    elif args.crossover_only:
        if args.payload_root is None:
            raise SystemExit("--payload-root is required for --crossover-only")
        result = run_crossover_sweep(
            args.output_log,
            args.payload_root,
            level2_root=args.level2_root,
            repetitions=args.sweep_repetitions,
        )
    else:
        if args.payload_root is None:
            raise SystemExit(
                "--payload-root is required unless --transport-only is set"
            )
        result = run_benchmark(args.output_log, args.payload_root, args.level2_root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
