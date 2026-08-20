"""Tests for the flat per-channel sensor-geometry export."""

from __future__ import annotations

import numpy as np
import pytest

from imas_ambix.gs import geometry as gsg
from imas_ambix.gs import geometry_export
from imas_ambix.gs.geometry_export import (
    GEOMETRY_FEATURE_NAMES,
    KIND_BPOL_PROBE,
    KIND_COIL,
    KIND_FLUX_LOOP,
    KIND_INTERFEROMETER_CHORD,
    KIND_SCALAR,
    KIND_SXR_CHORD,
    N_GEOMETRY_FEATURES,
    build_geometry_fields_from_table,
)
from imas_ambix.gs.machine_geometry import (
    GeometryIdentity,
    SensorGeometry,
    _project_operator_geometry,
    _project_sensor_features,
)


def _synthetic_table() -> gsg.GeometryTable:
    """A small but realistic GeometryTable built from synthetic efm geometry.

    Reuses the same synthetic geometry shape the gs.geometry tests pin: a
    co-located radial/vertical B-probe pair, a clean flux loop, a small PF set.
    """
    geom = {
        "magpr_r": np.array([0.18, 1.85, 1.85, 1.44]),
        "magpr_z": np.array([1.0, 0.3, 0.3, -1.2]),
        "magpr_ang": np.array([-90.0, -90.0, 0.0, 0.0]),
        "magpr_len": np.array([0.025, 0.025, 0.025, 0.025]),
        "silop_r": np.array([0.178, 1.163]),
        "silop_z": np.array([1.235, 1.083]),
        "fcoil_r": np.array([0.12, 0.13, 1.9, 1.92]),
        "fcoil_z": np.array([-1.5, -1.4, 1.4, 1.5]),
        "fcoil_turns": np.array([10.0, 10.0, 5.0, 5.0]),
        "fcoil_width": np.full(4, 0.01),
        "fcoil_height": np.full(4, 0.02),
        "fcoil_circ": np.array([1.0, 1.0, 2.0, 2.0]),  # two circuits, 2 filaments each
        "fcoil_xmult": np.full(4, 0.5),
        "limiterr": np.array([1.9, 1.55, 1.40]),
        "limiterz": np.array([0.4, 0.4, 0.82]),
    }
    sig = gsg.SetupSignature(
        n_bprobe=4,
        n_fluxloop=2,
        n_pf_filament=4,
        n_limiter=3,
        digest=gsg.round_geometry_hash(
            [
                geom["magpr_r"],
                geom["magpr_z"],
                geom["magpr_ang"],
                geom["silop_r"],
                geom["silop_z"],
                geom["fcoil_r"],
                geom["fcoil_z"],
                geom["fcoil_turns"],
                geom["limiterr"],
                geom["limiterz"],
            ]
        ),
    )
    mr, mz, mang, mlen = (
        geom["magpr_r"],
        geom["magpr_z"],
        geom["magpr_ang"],
        geom["magpr_len"],
    )
    b_probes = [
        gsg.BProbe(
            index=i,
            r=float(mr[i]),
            z=float(mz[i]),
            angle_deg=float(mang[i]),
            length=float(mlen[i]),
        )
        for i in range(mr.size)
    ]
    flux_loops = [
        gsg.FluxLoop(index=i, r=float(geom["silop_r"][i]), z=float(geom["silop_z"][i]))
        for i in range(2)
    ]
    fr, fz, ft, fc = (
        geom["fcoil_r"],
        geom["fcoil_z"],
        geom["fcoil_turns"],
        geom["fcoil_circ"],
    )
    pf = [
        gsg.PFFilament(
            r=float(fr[i]),
            z=float(fz[i]),
            turns=float(ft[i]),
            width=0.01,
            height=0.02,
            circuit=int(fc[i]),
            xmult=0.5,
        )
        for i in range(fr.size)
    ]
    amb = [
        ("obv06", "Outer coil r=1.850, z=0.300"),
        ("obr06", "Outer coil r=1.850, z=0.300"),
        ("ccbv01", "Centre Column Vertical r=0.180, z=1.000"),
        ("fl_cc01", "Flux Loop r=0.178, z=1.235"),
    ]
    sensor_map, unmatched = gsg.map_amb_sensors(geom, amb)
    return gsg.GeometryTable(
        signature=sig,
        shots=[12345],
        b_probes=b_probes,
        flux_loops=flux_loops,
        pf_filaments=pf,
        limiter_r=geom["limiterr"].tolist(),
        limiter_z=geom["limiterz"].tolist(),
        sensor_map=sensor_map,
        passive_structures=[],
        amc_current_channels=["p3u", "p4l", "solenoid"],
        unmatched_amb=unmatched,
    )


# --- schema ----------------------------------------------------------------


def test_feature_schema_is_stable():
    assert len(GEOMETRY_FEATURE_NAMES) == N_GEOMETRY_FEATURES
    assert GEOMETRY_FEATURE_NAMES[:3] == ("r", "z", "phi")
    assert "chord_r1" in GEOMETRY_FEATURE_NAMES
    assert "chord_z2" in GEOMETRY_FEATURE_NAMES


# --- B-probe / flux-loop coordinates cross-checked vs gs.geometry ----------


def test_bprobe_rows_carry_correct_rz_angle_and_normal():
    fields = build_geometry_fields_from_table(_synthetic_table())

    # obv06 (vertical, DD ang=-90) and obr06 (radial, ang=0) are co-located.
    obv = fields.get("obv06")
    obr = fields.get("obr06")
    assert obv is not None and obr is not None
    assert obv.sensor_kind == KIND_BPOL_PROBE
    assert obr.sensor_kind == KIND_BPOL_PROBE

    # R,Z come from efm (1.85, 0.3) for both
    assert obv.r == 1.85 and obv.z == 0.3
    assert obr.r == 1.85 and obr.z == 0.3

    # DDv4 orientation: vertical +Bz=-90, radial +Br=0.
    assert obv.angle_deg == -90.0
    assert obr.angle_deg == 0.0

    # DD projection coefficients are (cos(theta), -sin(theta)).
    assert np.isclose(obv.normal_r, 0.0, atol=1e-9)
    assert np.isclose(obv.normal_z, -1.0, atol=1e-9)
    assert np.isclose(obr.normal_r, 1.0, atol=1e-9)
    assert np.isclose(obr.normal_z, 0.0, atol=1e-9)


def test_bprobe_row_matches_source_bprobe_values():
    """Cross-check the flat row's R/Z/angle against the gs.geometry BProbe."""
    table = _synthetic_table()
    fields = build_geometry_fields_from_table(table)
    ccbv = fields.get("ccbv01")
    # ccbv01 maps to magpr index 0 (R=0.18, Z=1.0, ang=-90)
    src = next(m for m in table.sensor_map if m.amb_channel == "ccbv01")
    assert ccbv.r == src.r == 0.18
    assert ccbv.z == src.z == 1.0
    assert ccbv.angle_deg == src.angle_deg == -90.0


def test_flux_loop_is_point_sensor_nan_angle_and_chord():
    fields = build_geometry_fields_from_table(_synthetic_table())
    fl = fields.get("fl_cc01")
    assert fl is not None
    assert fl.sensor_kind == KIND_FLUX_LOOP
    # point sensor: real R/Z, NaN orientation, NaN chord endpoints
    assert fl.r == 0.178 and fl.z == 1.235
    assert np.isnan(fl.angle_deg)
    assert np.isnan(fl.normal_r) and np.isnan(fl.normal_z)
    for col in ("chord_r1", "chord_z1", "chord_r2", "chord_z2"):
        assert np.isnan(getattr(fl, col))


# --- line-integrated diagnostics get the chord kind ------------------------


def test_line_integrated_diagnostics_get_chord_kind_nan_endpoints():
    fields = build_geometry_fields_from_table(
        _synthetic_table(),
        extra_channel_names=["interferometer_03", "sxr_t01", "nbar_core"],
    )
    interf = fields.get("interferometer_03")
    sxr = fields.get("sxr_t01")
    nbar = fields.get("nbar_core")
    assert interf.sensor_kind == KIND_INTERFEROMETER_CHORD
    assert nbar.sensor_kind == KIND_INTERFEROMETER_CHORD
    assert sxr.sensor_kind == KIND_SXR_CHORD
    # chord endpoints are NaN (schema present, values not yet tabulated)
    for cg in (interf, sxr, nbar):
        for col in ("chord_r1", "chord_z1", "chord_r2", "chord_z2"):
            assert np.isnan(getattr(cg, col))


# --- scalars present + explicit, never dropped -----------------------------


def test_pure_scalar_channels_present_with_nan_coords():
    fields = build_geometry_fields_from_table(
        _synthetic_table(),
        extra_channel_names=["ip", "gas_valve_setpoint", "density_unknown"],
    )
    ip = fields.get("ip")
    gas = fields.get("gas_valve_setpoint")
    assert ip is not None and gas is not None  # present, never dropped
    # ip is prefix 'ip' -> classified coil-ish per the actuator prefixes; the
    # gas-valve setpoint is a pure scalar with NaN coords.
    assert gas.sensor_kind == KIND_SCALAR
    assert np.isnan(gas.r) and np.isnan(gas.z)


def test_coil_channels_kinded_coil():
    fields = build_geometry_fields_from_table(_synthetic_table())
    # amc current channels become explicit coil rows
    p3u = fields.get("p3u")
    sol = fields.get("solenoid")
    assert p3u is not None and p3u.sensor_kind == KIND_COIL
    assert sol is not None and sol.sensor_kind == KIND_COIL


# --- separator-insensitive matching ----------------------------------------


def test_channel_lookup_is_separator_insensitive():
    fields = build_geometry_fields_from_table(_synthetic_table())
    # the amb sensor was 'ccbv01'; a re-encoded store names it 'ccbv_01'
    assert fields.get("ccbv_01") is not None
    assert fields.get("ccbv_01").r == fields.get("ccbv01").r


# --- machine-geometry block ------------------------------------------------


def test_machine_block_carries_limiter_pf_and_constants():
    table = _synthetic_table()
    fields = build_geometry_fields_from_table(table)
    m = fields.machine
    assert list(m.limiter_r) == table.limiter_r
    assert list(m.limiter_z) == table.limiter_z
    assert m.r0 == gsg.MAST_R0
    assert m.minor_radius == gsg.MAST_A
    # two PF circuits -> two coil centroids; the inner circuit's turns-weighted
    # R is between its two filaments' R (0.12, 0.13)
    assert len(m.pf_coil_r) == 2
    assert 0.12 <= m.pf_coil_r[0] <= 0.13


# --- dense feature matrix aligned to channel names -------------------------


def test_feature_matrix_aligns_to_channel_names_with_nan_fill():
    fields = build_geometry_fields_from_table(_synthetic_table())
    names = ["ccbv01", "fl_cc01", "totally_unknown_channel", "obr06"]
    feats, kinds = fields.feature_matrix(names)
    assert feats.shape == (4, N_GEOMETRY_FEATURES)
    assert feats.dtype == np.float32
    # ccbv01 row carries finite R
    assert feats[0, GEOMETRY_FEATURE_NAMES.index("r")] == np.float32(0.18)
    # the unknown channel is all-NaN and scalar-kinded (present, not dropped)
    assert np.all(np.isnan(feats[2]))
    assert kinds[2] == KIND_SCALAR
    assert kinds[0] == KIND_BPOL_PROBE
    assert kinds[1] == KIND_FLUX_LOOP


def test_to_dict_is_json_serialisable():
    import json

    fields = build_geometry_fields_from_table(_synthetic_table())
    d = fields.to_dict()
    json.dumps(d)  # must not raise
    assert d["feature_names"] == list(GEOMETRY_FEATURE_NAMES)
    assert "ccbv01" in d["channels"]


def test_shot_builder_consumes_operator_and_sensor_projections(
    monkeypatch: pytest.MonkeyPatch,
):
    table = _synthetic_table()
    identity = GeometryIdentity(
        representation_key=table.signature.key,
        representation_digest=table.signature.digest,
        derivation_id="declared-probe-angle",
        physical_digest="machine-digest",
        registry_digest="registry-digest",
    )
    operator = _project_operator_geometry(table, identity=identity)
    requested = tuple(
        [mapping.amb_channel for mapping in table.sensor_map]
        + table.amc_current_channels
        + ["gas_valve_setpoint"]
    )
    matrix, kinds = _project_sensor_features(table, requested)
    sensors = SensorGeometry(
        identity=identity,
        channels=requested,
        feature_names=GEOMETRY_FEATURE_NAMES,
        sensor_kinds=kinds,
        feature_matrix=matrix,
    )
    calls: list[tuple[str, int]] = []

    class StubService:
        def operator(self, shot: int):
            calls.append(("operator", shot))
            return operator

        def sensors(self, shot: int, channels):
            calls.append(("sensors", shot))
            assert tuple(channels) == requested
            return sensors

    monkeypatch.setattr(geometry_export, "MachineGeometryService", StubService)
    fields = geometry_export.build_geometry_table(
        12_345, extra_channel_names=["gas_valve_setpoint"]
    )

    assert calls == [("operator", 12_345), ("sensors", 12_345)]
    assert fields.signature_key == identity.representation_key
    assert fields.physical_digest == "machine-digest"
    assert fields.shots == [12_345]
    assert fields.get("ccbv01").r == pytest.approx(0.18)
    assert fields.get("gas_valve_setpoint").sensor_kind == KIND_SCALAR
