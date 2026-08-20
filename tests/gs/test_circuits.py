"""Tests for source-declared active circuits and circuit drives.

The geometry table is the circuit authority.  These fixtures stay pure logic:
they model the declaration carried by a resolved table without reading a shot
store or maintaining a second machine-description module.
"""

from __future__ import annotations

from types import SimpleNamespace

from imas_ambix.gs import operator as op


def _filament(r: float, z: float, circuit: int) -> SimpleNamespace:
    return SimpleNamespace(
        r=r,
        z=z,
        turns=1.0,
        width=0.01,
        height=0.01,
        circuit=circuit,
        xmult=1.0,
    )


def _declared_table() -> SimpleNamespace:
    active_drives = [
        SimpleNamespace(
            circuit=index,
            channel=channel,
            ampere_turns_per_ampere=1.0,
            evidence="declared fixture",
            conductor=label,
        )
        for index, (label, channel) in enumerate(op._PF_COIL_AMC.items(), start=1)
    ]
    case_channels = (
        "p2u_case_current",
        "p2l_case_current",
        "p3u_case_current",
        "p3l_case_current",
        "p4u_case_current",
        "p4l_case_current",
        "p5u_case_current",
        "p5l_case_current",
    )
    case_drives = [
        SimpleNamespace(
            circuit=index,
            channel=channel,
            ampere_turns_per_ampere=1.0,
            evidence="declared fixture",
            conductor=channel.removesuffix("_current"),
        )
        for index, channel in enumerate(case_channels, start=14)
    ]
    filaments = [
        _filament(*op._PF_COIL_CENTROID[drive.conductor], drive.circuit)
        for drive in active_drives
    ]
    filaments.extend(
        _filament(1.0 + 0.01 * drive.circuit, 0.0, drive.circuit)
        for drive in case_drives
    )
    return SimpleNamespace(
        flux_loops=[],
        pf_filaments=filaments,
        limiter_r=[0.3, 1.6, 1.6, 0.3],
        limiter_z=[-1.0, -1.0, 1.0, 1.0],
        sensor_map=[],
        passive_structures=[],
        amc_current_channels=[drive.channel for drive in active_drives + case_drives],
        unmatched_amb=[],
        active_circuits=[drive.circuit for drive in active_drives],
        circuit_drives=active_drives + case_drives,
    )


def test_declared_table_carries_thirteen_active_and_twenty_one_driven_circuits():
    table = _declared_table()

    assert len(table.active_circuits) == 13
    assert len(table.circuit_drives) == 21
    assert len({drive.circuit for drive in table.circuit_drives}) == 21
    assert set(table.active_circuits) <= {
        drive.circuit for drive in table.circuit_drives
    }


def test_declared_membership_decides_winding_and_structural_roles():
    table = _declared_table()

    classes = op.classify_circuits(
        table.pf_filaments,
        table.amc_current_channels,
        table.active_circuits,
        table.circuit_drives,
    )
    by_circuit = {entry.circuit: entry for entry in classes}

    assert {by_circuit[circuit].role for circuit in table.active_circuits} == {
        "known_pf"
    }
    case_ids = {
        drive.circuit
        for drive in table.circuit_drives
        if drive.circuit not in table.active_circuits
    }
    assert len(case_ids) == 8
    assert {by_circuit[circuit].role for circuit in case_ids} == {"known_case"}
    assert {
        by_circuit[drive.circuit].amc_channel for drive in table.circuit_drives
    } == {drive.channel for drive in table.circuit_drives}


def test_role_does_not_depend_on_channel_spelling():
    filaments = [_filament(1.0, 0.0, 4), _filament(1.1, 0.0, 19)]
    drives = [
        SimpleNamespace(
            circuit=4,
            channel="winding_case_current",
            ampere_turns_per_ampere=1.0,
            conductor="winding",
            evidence="",
        ),
        SimpleNamespace(
            circuit=19,
            channel="structure_coil_current",
            ampere_turns_per_ampere=1.0,
            conductor="structure",
            evidence="",
        ),
    ]

    classes = op.classify_circuits(
        filaments,
        [drive.channel for drive in drives],
        active_circuits=[4],
        circuit_drives=drives,
    )

    assert [entry.role for entry in classes] == ["known_pf", "known_case"]


def test_missing_declared_channel_stays_inferred():
    filament = _filament(1.0, 0.0, 4)
    drive = SimpleNamespace(
        circuit=4,
        channel="winding_current",
        ampere_turns_per_ampere=1.0,
        evidence="declared fixture",
        conductor="winding",
    )

    classes = op.classify_circuits(
        [filament],
        ["other_current"],
        active_circuits=[4],
        circuit_drives=[drive],
    )

    assert classes[0].role == "inferred_passive"
    assert classes[0].amc_channel == ""
    assert "does not publish that channel" in classes[0].flag
