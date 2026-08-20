"""Tests for the flat per-channel sensor-geometry export."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

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
    _fields_from_projections,
)
from imas_ambix.gs.machine_geometry import (
    GeometryIdentity,
    OperatorGeometry,
    SensorGeometry,
)


def _sensor_row(
    *, r: float = np.nan, z: float = np.nan, angle_deg: float = np.nan
) -> np.ndarray:
    row = np.full(N_GEOMETRY_FEATURES, np.nan, dtype=np.float32)
    if np.isfinite(r) or np.isfinite(z) or np.isfinite(angle_deg):
        row[GEOMETRY_FEATURE_NAMES.index("phi")] = 0.0
    row[GEOMETRY_FEATURE_NAMES.index("r")] = r
    row[GEOMETRY_FEATURE_NAMES.index("z")] = z
    row[GEOMETRY_FEATURE_NAMES.index("angle_deg")] = angle_deg
    if np.isfinite(angle_deg):
        radians = np.deg2rad(angle_deg)
        row[GEOMETRY_FEATURE_NAMES.index("normal_r")] = np.cos(radians)
        row[GEOMETRY_FEATURE_NAMES.index("normal_z")] = np.sin(radians)
    return row


def _synthetic_projections(
    extra_channel_names: tuple[str, ...] = (),
) -> tuple[OperatorGeometry, SensorGeometry]:
    """Build realistic public projections without private geometry records."""
    identity = GeometryIdentity(
        representation_key="mp4-fl2-fc4-lim3-synthetic",
        representation_digest="synthetic-geometry",
        derivation_id="declared-probe-angle",
        physical_digest="",
        registry_digest="",
    )
    sensor_map = (
        SimpleNamespace(
            amb_channel="obv06",
            kind="b_probe",
            efm_index=1,
            r=1.85,
            z=0.3,
            angle_deg=-90.0,
            flag="",
        ),
        SimpleNamespace(
            amb_channel="obr06",
            kind="b_probe",
            efm_index=2,
            r=1.85,
            z=0.3,
            angle_deg=0.0,
            flag="",
        ),
        SimpleNamespace(
            amb_channel="ccbv01",
            kind="b_probe",
            efm_index=0,
            r=0.18,
            z=1.0,
            angle_deg=-90.0,
            flag="",
        ),
        SimpleNamespace(
            amb_channel="fl_cc01",
            kind="flux_loop",
            efm_index=0,
            r=0.178,
            z=1.235,
            angle_deg=None,
            flag="",
        ),
    )
    conductors = tuple(
        SimpleNamespace(r=r, z=z, turns=turns, circuit=circuit)
        for r, z, turns, circuit in (
            (0.12, -1.5, 10.0, 1),
            (0.13, -1.4, 10.0, 1),
            (1.9, 1.4, 5.0, 2),
            (1.92, 1.5, 5.0, 2),
        )
    )
    operator = OperatorGeometry(
        identity=identity,
        probes=(),
        loops=(),
        conductors=conductors,
        passives=(),
        limiter_r=(1.9, 1.55, 1.4),
        limiter_z=(0.4, 0.4, 0.82),
        polygon_sections=(),
        drive_map=(),
        sensor_map=sensor_map,
        unmatched_channels=(),
        active_circuits=(1, 2),
        available_current_channels=("p3u", "p4l", "solenoid"),
        r0=0.85,
        minor_radius=0.65,
        unresolved_turns={},
        coil_channels=(),
        coil_column_matrix=np.empty((len(sensor_map), 0)),
    )
    mapped = {
        "obv06": (KIND_BPOL_PROBE, _sensor_row(r=1.85, z=0.3, angle_deg=-90.0)),
        "obr06": (KIND_BPOL_PROBE, _sensor_row(r=1.85, z=0.3, angle_deg=0.0)),
        "ccbv01": (KIND_BPOL_PROBE, _sensor_row(r=0.18, z=1.0, angle_deg=-90.0)),
        "fl_cc01": (KIND_FLUX_LOOP, _sensor_row(r=0.178, z=1.235)),
        "p3u": (KIND_COIL, _sensor_row()),
        "p4l": (KIND_COIL, _sensor_row()),
        "solenoid": (KIND_COIL, _sensor_row()),
    }
    channels = tuple(mapped) + tuple(extra_channel_names)
    kinds: list[str] = []
    rows: list[np.ndarray] = []
    for channel in channels:
        if channel in mapped:
            kind, row = mapped[channel]
        elif channel.startswith(("interferometer", "nbar")):
            kind, row = KIND_INTERFEROMETER_CHORD, _sensor_row()
        elif channel.startswith("sxr"):
            kind, row = KIND_SXR_CHORD, _sensor_row()
        elif channel == "ip":
            kind, row = KIND_COIL, _sensor_row()
        else:
            kind, row = KIND_SCALAR, _sensor_row()
        kinds.append(kind)
        rows.append(row)
    sensors = SensorGeometry(
        identity=identity,
        channels=channels,
        feature_names=GEOMETRY_FEATURE_NAMES,
        sensor_kinds=tuple(kinds),
        feature_matrix=np.stack(rows),
    )
    return operator, sensors


def _synthetic_fields(*extra_channel_names: str):
    operator, sensors = _synthetic_projections(tuple(extra_channel_names))
    return _fields_from_projections(operator, sensors, (12_345,))


# --- schema ----------------------------------------------------------------


def test_feature_schema_is_stable():
    assert len(GEOMETRY_FEATURE_NAMES) == N_GEOMETRY_FEATURES
    assert GEOMETRY_FEATURE_NAMES[:3] == ("r", "z", "phi")
    assert "chord_r1" in GEOMETRY_FEATURE_NAMES
    assert "chord_z2" in GEOMETRY_FEATURE_NAMES


# --- B-probe / flux-loop coordinates cross-checked vs gs.geometry ----------


def test_bprobe_rows_carry_correct_rz_angle_and_normal():
    fields = _synthetic_fields()

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
    """Cross-check the flat row against the operator projection."""
    operator, sensors = _synthetic_projections()
    fields = _fields_from_projections(operator, sensors, (12_345,))
    ccbv = fields.get("ccbv01")
    src = next(m for m in operator.sensor_map if m.amb_channel == "ccbv01")
    assert ccbv.r == src.r == 0.18
    assert ccbv.z == src.z == 1.0
    assert ccbv.angle_deg == src.angle_deg == -90.0


def test_flux_loop_is_point_sensor_nan_angle_and_chord():
    fields = _synthetic_fields()
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
    fields = _synthetic_fields("interferometer_03", "sxr_t01", "nbar_core")
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
    fields = _synthetic_fields("ip", "gas_valve_setpoint", "density_unknown")
    ip = fields.get("ip")
    gas = fields.get("gas_valve_setpoint")
    assert ip is not None and gas is not None  # present, never dropped
    # ip is prefix 'ip' -> classified coil-ish per the actuator prefixes; the
    # gas-valve setpoint is a pure scalar with NaN coords.
    assert gas.sensor_kind == KIND_SCALAR
    assert np.isnan(gas.r) and np.isnan(gas.z)


def test_coil_channels_kinded_coil():
    fields = _synthetic_fields()
    # amc current channels become explicit coil rows
    p3u = fields.get("p3u")
    sol = fields.get("solenoid")
    assert p3u is not None and p3u.sensor_kind == KIND_COIL
    assert sol is not None and sol.sensor_kind == KIND_COIL


# --- separator-insensitive matching ----------------------------------------


def test_channel_lookup_is_separator_insensitive():
    fields = _synthetic_fields()
    # the amb sensor was 'ccbv01'; a re-encoded store names it 'ccbv_01'
    assert fields.get("ccbv_01") is not None
    assert fields.get("ccbv_01").r == fields.get("ccbv01").r


# --- machine-geometry block ------------------------------------------------


def test_machine_block_carries_limiter_pf_and_constants():
    operator, sensors = _synthetic_projections()
    fields = _fields_from_projections(operator, sensors, (12_345,))
    m = fields.machine
    assert m.limiter_r == operator.limiter_r
    assert m.limiter_z == operator.limiter_z
    assert m.r0 == operator.r0
    assert m.minor_radius == operator.minor_radius
    # two PF circuits -> two coil centroids; the inner circuit's turns-weighted
    # R is between its two filaments' R (0.12, 0.13)
    assert len(m.pf_coil_r) == 2
    assert 0.12 <= m.pf_coil_r[0] <= 0.13


# --- dense feature matrix aligned to channel names -------------------------


def test_feature_matrix_aligns_to_channel_names_with_nan_fill():
    fields = _synthetic_fields()
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

    fields = _synthetic_fields()
    d = fields.to_dict()
    json.dumps(d)  # must not raise
    assert d["feature_names"] == list(GEOMETRY_FEATURE_NAMES)
    assert "ccbv01" in d["channels"]


def test_shot_builder_consumes_operator_and_sensor_projections(
    monkeypatch: pytest.MonkeyPatch,
):
    operator, _ = _synthetic_projections()
    identity = GeometryIdentity(
        representation_key=operator.identity.representation_key,
        representation_digest=operator.identity.representation_digest,
        derivation_id=operator.identity.derivation_id,
        physical_digest="machine-digest",
        registry_digest="registry-digest",
    )
    operator = OperatorGeometry(**{**operator.__dict__, "identity": identity})
    requested = tuple(
        [mapping.amb_channel for mapping in operator.sensor_map]
        + list(operator.available_current_channels)
        + ["gas_valve_setpoint"]
    )
    _, sensors = _synthetic_projections(("gas_valve_setpoint",))
    sensors = SensorGeometry(**{**sensors.__dict__, "identity": identity})
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
