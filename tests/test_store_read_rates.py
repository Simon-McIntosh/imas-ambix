"""Tests for directly timed storage-arm benchmark measurements."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import numpy as np
import pytest
import zarr

from imas_ambix.bench.store_arms import (
    FIXED_CHANNELS,
    FIXED_MAST_SHOT,
    IDSZarrWriter,
    payload_to_ids,
    read_dd_zarr,
    read_fixed_native_payload,
    read_imas_netcdf,
    write_imas_netcdf,
)
from imas_ambix.bench.store_read_rates import (
    ARMS,
    PATTERNS,
    BenchmarkConfig,
    CrossoverCell,
    find_matching_transport_payload,
    measure_access_patterns,
    measure_crossover_grid,
    measure_transport,
    run_transport_comparison,
    summarize_crossovers,
)

LEVEL2_ROOT = Path("/work/projects/imas_gpu/mast/level2/shots")


def test_access_matrix_has_twelve_directly_timed_cells():
    config = BenchmarkConfig(
        shots=(10, 11, 12, 13),
        batch_size=4,
        slice_size=4,
        static_repetitions=2,
        slice_repetitions=2,
        batch_repetitions=2,
        random_windows=4,
    )

    def reader(shot: int, start: int | None, stop: int | None):
        values = np.arange(12, dtype=np.float64) + shot
        return tuple(values[slice(start, stop)] for _ in FIXED_CHANNELS)

    readers = {arm: reader for arm in ARMS}
    cells = measure_access_patterns(
        config, readers, {shot: 12 for shot in config.shots}
    )

    assert len(cells) == 12
    assert {(cell.arm, cell.access_pattern) for cell in cells} == {
        (arm, pattern) for arm in ARMS for pattern in PATTERNS
    }
    assert all(cell.wall_seconds > 0 for cell in cells)
    assert all(cell.sample_count > 0 for cell in cells)
    assert all(cell.throughput_samples_per_second > 0 for cell in cells)


def test_crossover_grid_measures_every_arm_scale_and_batch_cell():
    shots = tuple(range(8))
    scales = (2, 4, 8, 32)
    batch_sizes = (1, 2, 8)

    def reader(shot: int, start: int | None, stop: int | None):
        values = np.arange(32, dtype=np.float64) + shot
        return tuple(values[slice(start, stop)] for _ in FIXED_CHANNELS)

    readers = {scale: {arm: reader for arm in ARMS} for scale in scales}
    cells = measure_crossover_grid(
        shots,
        scales,
        batch_sizes,
        readers,
        repetitions=2,
    )

    assert len(cells) == 36
    assert {
        (cell.arm, cell.payload_samples_per_shot, cell.batch_size)
        for cell in cells
    } == {
        (arm, scale, batch_size)
        for arm in ARMS
        for scale in scales
        for batch_size in batch_sizes
    }
    assert all(cell.wall_seconds > 0 for cell in cells)
    assert all(cell.sample_count > 0 for cell in cells)
    assert all(cell.ratio_to_native_level2 > 0 for cell in cells)


def test_crossover_summary_reports_axis_changes_and_native_ratios():
    scales = (2, 4, 8, 32)
    batch_sizes = (1, 2, 8)
    cells = []
    for scale in scales:
        for batch_size in batch_sizes:
            rates = {
                ARMS[0]: 10.0,
                ARMS[1]: 20.0 if scale < 8 else 40.0,
                ARMS[2]: (
                    40.0
                    if scale < 8 and batch_size < 8
                    else 30.0
                    if batch_size < 8
                    else 5.0
                ),
            }
            for arm, rate in rates.items():
                cells.append(
                    CrossoverCell(
                        arm=arm,
                        payload_samples_per_shot=scale,
                        payload_scalar_count_per_shot=scale * len(FIXED_CHANNELS),
                        batch_size=batch_size,
                        repetitions=1,
                        wall_seconds=1.0,
                        sample_count=int(rate),
                        throughput_samples_per_second=rate,
                        ratio_to_native_level2=rate / rates[ARMS[0]],
                        checksum=1.0,
                    )
                )

    result = summarize_crossovers(cells, scales, batch_sizes)

    assert result["payload_scale"]["crossing_count"] == 2
    assert result["batch_size"]["crossing_count"] == 4
    for series in result["payload_scale"]["series"]:
        for point in series["points"]:
            assert point["ratios_to_native_level2"][ARMS[0]] == 1.0


def test_transport_times_only_identical_live_and_mapped_objects(
    tmp_path, capsys, monkeypatch
):
    local_array = tmp_path / "local.zarr"
    remote_array = tmp_path / "remote.zarr"
    values = np.arange(128, dtype=np.float64)

    zarr.save_array(local_array, values)
    zarr.save_array(remote_array, values)
    cells, evidence = measure_transport(
        local_array,
        str(remote_array),
        repetitions=3,
    )

    assert [cell.route for cell in cells] == [
        "live_https_transfer",
        "mapped_on_disk",
    ]
    assert all(cell.sample_count == 384 for cell in cells)
    assert all(cell.wall_seconds > 0 for cell in cells)
    assert evidence["decoded_arrays_identical"] is True
    assert evidence["remote_shape"] == [128]
    assert evidence["mapped_shape"] == [128]
    assert "TRANSPORT_IDENTITY" in capsys.readouterr().out

    zarr.save_array(remote_array, values[:-1], overwrite=True)
    monkeypatch.setattr(
        "imas_ambix.bench.store_read_rates.time.perf_counter",
        lambda: pytest.fail("timing began before payload equality was established"),
    )
    with pytest.raises(ValueError, match="numerically identical"):
        measure_transport(local_array, str(remote_array), repetitions=3)


def test_transport_probe_records_unobtainable_comparison(tmp_path, capsys):
    mapped_root = tmp_path / "mapped"
    live_root = tmp_path / "live"
    for shot, mapped_size, live_size in ((10, 8, 3), (11, 9, 4)):
        zarr.save_array(
            mapped_root / f"{shot}.zarr/summary/ip",
            np.arange(mapped_size),
        )
        zarr.save_array(
            live_root / f"{shot}.zarr/summary/ip",
            np.arange(live_size),
        )

    match, probes = find_matching_transport_payload(
        mapped_root,
        (10, 11),
        live_root=str(live_root),
    )
    assert match is None
    assert [probe.mapped_shape for probe in probes] == [(8,), (9,)]
    assert [probe.live_shape for probe in probes] == [(3,), (4,)]

    output_log = tmp_path / "transport.json"
    result = run_transport_comparison(
        output_log,
        level2_root=mapped_root,
        shots=(10, 11),
        probe_count=2,
        live_root=str(live_root),
        minimum_negative_probes=2,
    )
    assert result["status"] == "unobtainable_without_mirror_refresh"
    assert result["identity_gate"]["completed_probe_count"] == 2
    assert result["transport_cells"] == []
    assert len(result["probes"]) == 2
    assert output_log.exists()
    assert capsys.readouterr().out.count("TRANSPORT_PROBE") == 4


@pytest.mark.skipif(
    not (LEVEL2_ROOT / f"{FIXED_MAST_SHOT}.zarr").exists(),
    reason="FAIR-MAST level-2 mirror not mounted",
)
def test_project_environment_writes_and_reads_both_converted_arms(tmp_path):
    payload = read_fixed_native_payload(LEVEL2_ROOT)
    ids = payload_to_ids(payload)
    paths = tuple(payload.arrays)

    netcdf_path = tmp_path / "summary.nc"
    write_imas_netcdf(ids, netcdf_path)
    netcdf_arrays = read_imas_netcdf(netcdf_path, paths)

    zarr_path = tmp_path / "summary.zarr"
    IDSZarrWriter(ids, paths).write(zarr_path)
    zarr_arrays = read_dd_zarr(zarr_path, paths)

    for path, native in payload.arrays.items():
        assert np.array_equal(native.values, netcdf_arrays[path])
        assert np.array_equal(native.values, zarr_arrays[path])


def test_tensorizer_subclass_reaches_no_private_imas_names():
    tree = ast.parse(inspect.getsource(IDSZarrWriter))
    private_attributes = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and node.attr.startswith("_")
        and not node.attr.startswith("__")
    }
    assert private_attributes == set()
    assert IDSZarrWriter.__mro__[1].__name__ == "IDSTensorizer"
    assert {
        "get_dimensions",
        "get_attributes",
        "get_shape_dimensions",
        "get_shape_attributes",
    }.issubset(IDSZarrWriter.__dict__)
