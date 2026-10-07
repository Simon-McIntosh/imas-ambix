"""Write fetches a pulse it has not cached before writing it.

At the facility the EDDB cache is empty, so the writer has to land the channels
its signal maps declare before it can read them.  These tests drive a write
against a synthetic flat bundle and the EDDB stand-in over the local transport:
a first write on a pulse with no cache fetches the mapped channels and writes
the run file, a second write with ``--overwrite`` makes no transport call
because every channel is cached, and a channel EDDB refuses is reported on the
receipt without stopping the write.
"""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING

import imas
from eddb_standin import write_canned_wrapper

from imas_alambic import pulse_writer
from imas_alambic.eddb import is_cached
from imas_alambic.eddb_remote import RemoteEddbExtractor, SubprocessTransport
from imas_alambic.signal_map import MAP_SCHEMA_VERSION, SignalMap, SignalRule

if TYPE_CHECKING:
    from pathlib import Path

_DD_VERSION = "4.1.1"


class _CountingTransport(SubprocessTransport):
    """A subprocess transport that counts the batches it is asked to run."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def run(self, argv, payload):  # type: ignore[override]
        self.calls += 1
        return super().run(argv, payload)


def _local_extractor(api_dir: Path, transport: SubprocessTransport):
    """A factory for the local transport, standing in for ``extractor_for_host``."""

    def build(_host: str | None) -> RemoteEddbExtractor:
        return RemoteEddbExtractor(
            ssh_command=(),
            remote_python=sys.executable,
            api_path=str(api_dir),
            nice=False,
            transport=transport,
        )

    return build


def _signal(semantic_id: str, source_array: str, target_path: str) -> SignalRule:
    return SignalRule(
        semantic_id=semantic_id,
        source_group="amc",
        source_array=source_array,
        source_unit="kA",
        target_path=target_path,
        target_unit="A",
        target_index=0,
        transformation="ip_like",
        source_cocos=3,
        unit_factor=1000.0,
        channel_factor=1.0,
        standard_name=None,
        evidence="receipt sha256:synthetic",
        validation_state="corpus-validated",
    )


def _catalog(machine: str) -> dict:
    return {
        "schema_version": "1.0.0",
        "dd_version": _DD_VERSION,
        "source": "synthetic",
        "source_revision": "0",
        "source_cocos": 0,
        "description_store_format": "netcdf",
        "description_store_root": "description",
        "description_store_layout": "static-over-map",
        "probe_angle_source": "description",
        "binding_sets": [
            {
                "name": "bs",
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
                "binding_set": "bs",
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


def _write_descriptions(root: Path) -> None:
    descriptions = root / "machine_description" / "phase"
    descriptions.mkdir(parents=True)
    factory = imas.IDSFactory(_DD_VERSION)
    for name in pulse_writer.IDS_NAMES:
        ids = factory.new(name)
        ids.ids_properties.homogeneous_time = 2
        with imas.DBEntry(
            descriptions / f"{name}.nc", "w", dd_version=_DD_VERSION
        ) as entry:
            entry.put(ids)


def _flat_bundle(root: Path, machine: str, signals: tuple[SignalRule, ...]) -> Path:
    root.mkdir(parents=True)
    (root / "bundle.json").write_text(
        json.dumps(
            {
                "name": "facility",
                "version": "1.0.0",
                "machine": machine,
                "store_roots": {"description": "machine_description"},
            }
        )
    )
    (root / "machine_map.json").write_text(json.dumps(_catalog(machine)))
    payload = SignalMap.create(
        schema_version=MAP_SCHEMA_VERSION,
        set_version="0.1.0",
        machine=machine,
        system="magnetics",
        source_dataset="synthetic",
        target_dd_version=_DD_VERSION,
        target_cocos=17,
        discovery_producer="imas-codex",
        discovery_receipt="sha256:discovery",
        signals=signals,
        calibrations=(),
        blocked=(),
    ).as_dict()
    (root / "maps").mkdir()
    (root / "maps" / "magnetics.json").write_text(json.dumps(payload))
    _write_descriptions(root)
    return root


def _prepare(tmp_path, monkeypatch, machine, signals):
    bundle = _flat_bundle(tmp_path / "bundle", machine, signals)
    api = tmp_path / "eddb"
    api.mkdir()
    write_canned_wrapper(api)
    transport = _CountingTransport()
    monkeypatch.setattr(
        pulse_writer, "extractor_for_host", _local_extractor(api, transport)
    )
    monkeypatch.setattr(pulse_writer, "_SYSTEM_FOR_IDS", {"magnetics": "magnetics"})
    monkeypatch.setattr("imas_alambic.machine_map.entry_points", lambda group: [])
    monkeypatch.setenv("IMAS_ALAMBIC_MAP_PATH", str(bundle))
    return bundle, transport


def test_write_fetches_an_uncached_pulse_then_a_second_write_makes_no_call(
    tmp_path, monkeypatch
):
    machine = "synth-machine"
    bundle, transport = _prepare(
        tmp_path,
        monkeypatch,
        machine,
        (_signal("ip", "plasma_current", "magnetics/ip/data"),),
    )
    cache = tmp_path / "cache"
    out = tmp_path / "ids"
    assert not (cache / "900001.zarr").is_dir()

    receipt = pulse_writer.write_pulse(
        machine,
        900001,
        out,
        maps=str(bundle),
        cache=str(cache),
        eddb_host="local",
    )

    run_file = out / "900001_0.nc"
    assert receipt.path == str(run_file)
    assert run_file.is_file()
    assert is_cached(cache, 900001, "amc", "plasma_current")
    assert any(leaf.target_path == "magnetics/ip/data" for leaf in receipt.leaves)
    assert receipt.refused == ()
    calls_after_first = transport.calls
    assert calls_after_first == 1

    second = pulse_writer.write_pulse(
        machine,
        900001,
        out,
        maps=str(bundle),
        cache=str(cache),
        eddb_host="local",
        overwrite=True,
    )

    assert transport.calls == calls_after_first
    assert second.path == str(run_file)
    with imas.DBEntry(run_file, "r", dd_version=_DD_VERSION) as entry:
        assert entry.get("magnetics", autoconvert=False) is not None


def test_a_refused_channel_is_reported_without_stopping_the_write(
    tmp_path, monkeypatch
):
    machine = "synth-machine"
    bundle, _transport = _prepare(
        tmp_path,
        monkeypatch,
        machine,
        (
            _signal("ip", "plasma_current", "magnetics/ip/data"),
            _signal("refused", "REFUSED", "magnetics/refused/data"),
        ),
    )
    cache = tmp_path / "cache"
    out = tmp_path / "ids"

    receipt = pulse_writer.write_pulse(
        machine,
        900001,
        out,
        maps=str(bundle),
        cache=str(cache),
        eddb_host="local",
    )

    assert (out / "900001_0.nc").is_file()
    assert [refusal.dname for refusal in receipt.refused] == ["REFUSED"]
    assert receipt.refused[0].code == 1015
    assert any(leaf.target_path == "magnetics/ip/data" for leaf in receipt.leaves)
    assert not is_cached(cache, 900001, "amc", "REFUSED")
