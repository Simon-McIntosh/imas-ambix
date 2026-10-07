"""A run file appears only after every description IDS is written."""

from __future__ import annotations

import json
import os
import stat

import imas
import pytest

from imas_alambic import pulse_writer


@pytest.fixture
def synthetic_bundle(tmp_path, monkeypatch):
    root = tmp_path / "bundle"
    root.mkdir()
    machine = "synthetic"
    (root / "bundle.json").write_text(
        json.dumps(
            {
                "name": "synthetic",
                "version": "1.0.0",
                "machine": machine,
                "store_roots": {"description": "descriptions"},
            }
        )
    )
    catalog = {
        "schema_version": "1.0.0",
        "dd_version": "4.1.1",
        "source": "synthetic",
        "source_revision": "0",
        "source_cocos": 0,
        "description_store_format": "netcdf",
        "description_store_root": "description",
        "description_store_layout": "static-over-map",
        "probe_angle_source": "description",
        "binding_sets": [
            {
                "name": "synthetic",
                "bindings": [
                    {
                        "name": "ip",
                        "source_group": "amc",
                        "source_array": "plasma_current",
                        "source_rank": 0,
                        "source_role": "value",
                        "source_location": "file:///synthetic/amc",
                        "dd_path": "magnetics/ip/data",
                        "source_unit": "kA",
                        "target_unit": "A",
                        "sign_convention": "identity",
                        "evidence": "receipt sha256:synthetic",
                    }
                ],
            }
        ],
        "maps": [
            {
                "name": "phase",
                "machine": machine,
                "first_shot": 900001,
                "last_shot": 900001,
                "transition": None,
                "binding_set": "synthetic",
                "drive_topology": None,
                "description_supplement": None,
                "validation_state": "corpus-validated",
            }
        ],
        "validation_gaps": [],
        "source_qualifications": [],
        "sensor_identity_rules": [],
        "identity_qualifications": [],
        "flux_loop_position_declarations": [],
        "circuit_current_joins": [],
        "drive_topologies": [],
        "structure_assemblies": [],
        "acquisition_declarations": [],
        "description_supplements": [],
    }
    (root / "machine_map.json").write_text(json.dumps(catalog))
    descriptions = root / "descriptions" / "phase"
    descriptions.mkdir(parents=True)
    factory = imas.IDSFactory("4.1.1")
    for name in pulse_writer.IDS_NAMES:
        ids = factory.new(name)
        ids.ids_properties.homogeneous_time = 2
        with imas.DBEntry(
            descriptions / f"{name}.nc", "w", dd_version="4.1.1"
        ) as entry:
            entry.put(ids)
    cache = tmp_path / "cache"
    (cache / "900001.zarr").mkdir(parents=True)
    monkeypatch.setattr("imas_alambic.machine_map.entry_points", lambda group: [])
    monkeypatch.setattr(pulse_writer, "_SYSTEM_FOR_IDS", {})
    monkeypatch.setenv("IMAS_ALAMBIC_MAP_PATH", str(root))
    return machine, cache


def test_failure_on_third_ids_leaves_no_run_file_or_temporary(
    tmp_path, monkeypatch, synthetic_bundle
):
    machine, cache = synthetic_bundle
    out = tmp_path / "ids"
    original = pulse_writer._read_description
    seen = []

    def fail_on_third(path, ids_name, dd_version):
        seen.append(ids_name)
        if len(seen) == 3:
            raise RuntimeError("third IDS failed")
        return original(path, ids_name, dd_version)

    monkeypatch.setattr(pulse_writer, "_read_description", fail_on_third)
    with pytest.raises(RuntimeError, match="third IDS failed"):
        pulse_writer.write_pulse(machine, 900001, out, cache=str(cache))

    assert seen == list(pulse_writer.IDS_NAMES[:3])
    assert list(out.iterdir()) == []


def test_complete_run_is_readable_and_refuses_a_second_write(
    tmp_path, synthetic_bundle
):
    machine, cache = synthetic_bundle
    out = tmp_path / "ids"
    old_umask = os.umask(0o022)
    try:
        receipt = pulse_writer.write_pulse(machine, 900001, out, cache=str(cache))
    finally:
        os.umask(old_umask)

    run_file = out / "900001_0.nc"
    assert receipt.path == str(run_file)
    assert [path.name for path in out.iterdir()] == [run_file.name]
    assert stat.S_IMODE(run_file.stat().st_mode) == 0o644
    with imas.DBEntry(run_file, "r", dd_version="4.1.1") as entry:
        for name in pulse_writer.IDS_NAMES:
            assert entry.get(name, autoconvert=False) is not None
    with pytest.raises(pulse_writer.PulseWriteError, match="already exists"):
        pulse_writer.write_pulse(machine, 900001, out, cache=str(cache))
    assert [path.name for path in out.iterdir()] == [run_file.name]
