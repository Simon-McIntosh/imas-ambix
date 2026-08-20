"""Regression tests for case-circuit classification.

Eight of MAST's ten case circuits sit within 8 cm of an active-coil centroid.
Geometry alone therefore cannot tell an active winding apart from its
co-located, separately supplied case circuit.  Case circuits must retain their
own Green's-function columns and measured case currents.

Classification consumes the geometry table's source-declared active circuits
and circuit drives.  A structural circuit keeps its own measured channel when
declared, and remains inferred when the source declares no drive.

These tests are pure-logic (synthetic geometry, no mirror needed).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from imas_ambix.gs import operator as op


class _Filament(SimpleNamespace):
    def __init__(self, r, z, turns, width, height, circuit, xmult):
        super().__init__(
            r=r,
            z=z,
            turns=turns,
            width=width,
            height=height,
            circuit=circuit,
            xmult=xmult,
        )


class _CircuitDrive(SimpleNamespace):
    def __init__(
        self, circuit, channel, ampere_turns_per_ampere, evidence="", conductor=""
    ):
        super().__init__(
            circuit=circuit,
            channel=channel,
            ampere_turns_per_ampere=ampere_turns_per_ampere,
            evidence=evidence,
            conductor=conductor,
        )


class _Signature(SimpleNamespace):
    @property
    def key(self):
        counts = (
            f"mp{self.n_bprobe}-fl{self.n_fluxloop}-fc{self.n_pf_filament}"
            f"-lim{self.n_limiter}"
        )
        return f"{counts}-{self.digest}"


class _GeometryFixture(SimpleNamespace):
    def __init__(self, **values):
        values.setdefault("polygon_sections", [])
        values.setdefault("circuit_drives", [])
        values.setdefault("active_circuits", [])
        values.setdefault("provenance_flags", [])
        values.setdefault("r0", 0.85)
        values.setdefault("minor_radius", 0.65)
        super().__init__(**values)


class _BProbe(SimpleNamespace):
    pass


class _SensorMapping(SimpleNamespace):
    def __init__(self, amb_channel, kind, efm_index, r, z, angle_deg, residual_m, flag):
        super().__init__(
            amb_channel=amb_channel,
            kind=kind,
            efm_index=efm_index,
            r=r,
            z=z,
            angle_deg=angle_deg,
            residual_m=residual_m,
            flag=flag,
        )


_fixtures = SimpleNamespace(
    BProbe=_BProbe,
    CircuitDrive=_CircuitDrive,
    GeometryTable=_GeometryFixture,
    PFFilament=_Filament,
    SensorMapping=_SensorMapping,
    SetupSignature=_Signature,
)

# --- coil-model version marker ---------------------------------------------


def test_coil_model_marker_identifies_source_stated_drives():
    """A downstream cache keying on COIL_MODEL_VERSION must invalidate across
    changes to circuit assignment rules."""
    assert op.COIL_MODEL_VERSION not in ("", "case-circuits-v1", "case-circuits-v2")
    assert op.COIL_MODEL_VERSION == "source-stated-drives"


def test_operator_summary_reports_coil_model_version(tmp_path):
    table = _table([_filament(_P4U_R, _P4U_Z, circuit=8)], ["p4u_coil_current"])
    operators = op.build_all_operators({table.signature.key: table})
    out = op.write_operator_summary(operators, out_path=tmp_path / "summary.json")
    import json

    payload = json.loads(out.read_text())
    assert payload["coil_model_version"] == op.COIL_MODEL_VERSION


# --- source-declared case identities --------------------------------------


def test_p4u_case_drive_is_declared_separately_from_the_winding():
    table = _table(
        [_filament(_P4U_R, _P4U_Z, 8), _filament(_P4U_CASE_R, _P4U_CASE_Z, 18)],
        ["p4u_coil_current", "p4u_case_current"],
    )

    assert table.active_circuits == [8]
    assert [(drive.circuit, drive.channel) for drive in table.circuit_drives] == [
        (8, "p4u_coil_current"),
        (18, "p4u_case_current"),
    ]


def test_p6_case_has_no_declared_drive():
    table = _table(
        [_filament(*op._PF_COIL_CENTROID["p6u"], 12), _filament(1.45, 0.9, 22)],
        ["p6u_current"],
    )

    assert table.active_circuits == [12]
    assert [(drive.circuit, drive.channel) for drive in table.circuit_drives] == [
        (12, "p6u_current")
    ]


# --- a minimal fixture builder ---------------------------------------------


def _filament(
    r: float, z: float, circuit: int, xmult: float = 1.0
) -> _fixtures.PFFilament:
    return _fixtures.PFFilament(
        r=r, z=z, turns=1.0, width=0.01, height=0.01, circuit=circuit, xmult=xmult
    )


def _table(
    filaments: list[_fixtures.PFFilament], amc_channels: list[str]
) -> _fixtures.GeometryTable:
    """A single-vertical-probe geometry table around whatever filaments are given."""
    bp = _fixtures.BProbe(index=0, r=1.3, z=0.0, angle_deg=-90.0, length=0.025)
    sig = _fixtures.SetupSignature(
        n_bprobe=1,
        n_fluxloop=0,
        n_pf_filament=len(filaments),
        n_limiter=4,
        digest="c4se0000c1rcu17s",
    )
    channel_by_circuit = {
        1: "sol_current",
        8: "p4u_coil_current",
        12: "p6u_current",
        13: "p6l_current",
        18: "p4u_case_current",
    }
    conductor_by_circuit = {
        1: "sol",
        8: "p4u",
        12: "p6u",
        13: "p6l",
        18: "p4u_case",
    }
    represented = {filament.circuit for filament in filaments}
    drives = [
        _fixtures.CircuitDrive(
            circuit=circuit,
            channel=channel,
            ampere_turns_per_ampere=1.0,
            evidence="declared fixture",
            conductor=conductor_by_circuit[circuit],
        )
        for circuit, channel in channel_by_circuit.items()
        if circuit in represented
    ]
    return _fixtures.GeometryTable(
        signature=sig,
        shots=[99999],
        b_probes=[bp],
        flux_loops=[],
        pf_filaments=filaments,
        limiter_r=[0.3, 1.6, 1.6, 0.3],
        limiter_z=[-1.0, -1.0, 1.0, 1.0],
        sensor_map=[
            _fixtures.SensorMapping("obv01", "b_probe", 0, 1.3, 0.0, -90.0, 0.001, ""),
        ],
        passive_structures=[],
        amc_current_channels=amc_channels,
        unmatched_amb=[],
        active_circuits=[drive.circuit for drive in drives if drive.circuit <= 13],
        circuit_drives=drives,
    )


def _classify(
    filaments: list[_fixtures.PFFilament], amc_channels: list[str]
) -> list[op.CircuitClass]:
    table = _table(filaments, amc_channels)
    return op.classify_circuits(
        table.pf_filaments,
        table.amc_current_channels,
        table.active_circuits,
        table.circuit_drives,
    )


# --- the mis-assignment fixture: case gets its OWN channel -----------------

# P4U centroid per operator._PF_COIL_CENTROID; the case sits 2 cm away, well
# inside _COIL_MATCH_M (8 cm) -- exactly the geometric confusion the audit found.
_P4U_R, _P4U_Z = op._PF_COIL_CENTROID["p4u"]
_P4U_CASE_R, _P4U_CASE_Z = _P4U_R + 0.02, _P4U_Z


def test_case_filament_2cm_from_active_centroid_gets_its_own_channel():
    filaments = [
        _filament(_P4U_R, _P4U_Z, circuit=8),  # declared P4U winding
        _filament(_P4U_CASE_R, _P4U_CASE_Z, circuit=18),  # declared P4U case
    ]
    classes = _classify(filaments, ["p4u_coil_current", "p4u_case_current"])
    by_circ = {c.circuit: c for c in classes}

    assert by_circ[8].role == "known_pf"
    assert by_circ[8].coil_label == "p4u"
    assert by_circ[8].amc_channel == "p4u_coil_current"

    # the case circuit must NOT be folded into p4u's channel/role.
    assert by_circ[18].role == "known_case"
    assert by_circ[18].coil_label == "p4u_case"
    assert by_circ[18].amc_channel == "p4u_case_current"
    assert by_circ[18].amc_channel != by_circ[8].amc_channel


def test_case_filament_without_its_channel_falls_back_to_inferred():
    """A campaign missing the case-current channel must INFER, never borrow
    the active coil's channel (verify-and-flag, never fabricate)."""
    filaments = [
        _filament(_P4U_R, _P4U_Z, circuit=8),
        _filament(_P4U_CASE_R, _P4U_CASE_Z, circuit=18),
    ]
    classes = _classify(filaments, ["p4u_coil_current"])
    by_circ = {c.circuit: c for c in classes}

    assert by_circ[18].role == "inferred_passive"
    assert by_circ[18].amc_channel == ""
    assert "does not publish that channel" in by_circ[18].flag
    # the active circuit is unaffected by its case's channel being absent.
    assert by_circ[8].role == "known_pf"
    assert by_circ[8].amc_channel == "p4u_coil_current"


def test_p6_case_is_zero_passive_even_if_a_channel_is_injected():
    """A structural circuit with no declared drive must stay inferred passive."""
    p6u_r, p6u_z = op._PF_COIL_CENTROID["p6u"]
    filaments = [
        _filament(p6u_r, p6u_z, circuit=12),  # P6U active (id 12)
        _filament(p6u_r + 0.02, p6u_z, circuit=22),  # P6U case (id 22)
    ]
    # inject a hypothetical "p6u_case_current" -- should still be ignored.
    classes = _classify(filaments, ["p6u_current", "p6u_case_current"])
    by_circ = {c.circuit: c for c in classes}

    assert by_circ[12].role == "known_pf"
    assert by_circ[12].amc_channel == "p6u_current"

    assert by_circ[22].role == "inferred_passive"
    assert by_circ[22].amc_channel == ""
    assert by_circ[22].flag == ""


def test_p6_case_without_any_channel_is_also_inferred_passive():
    p6l_r, p6l_z = op._PF_COIL_CENTROID["p6l"]
    filaments = [
        _filament(p6l_r, p6l_z, circuit=13),  # P6L active (id 13)
        _filament(p6l_r - 0.02, p6l_z, circuit=23),  # P6L case (id 23)
    ]
    classes = _classify(filaments, ["p6l_current"])
    by_circ = {c.circuit: c for c in classes}
    assert by_circ[13].role == "known_pf"
    assert by_circ[23].role == "inferred_passive"


# --- active mapping unchanged for all 13 active circuits -------------------


def test_all_13_active_circuits_still_classify_known_pf():
    """Every active circuit at its real centroid classifies as known PF."""
    filaments = []
    amc_channels = []
    drives = []
    for circuit, (label, channel) in enumerate(op._PF_COIL_AMC.items(), start=1):
        r, z = op._PF_COIL_CENTROID[label]
        filaments.append(_filament(r, z, circuit=circuit))
        amc_channels.append(channel)
        drives.append(_fixtures.CircuitDrive(circuit, channel, 1.0, conductor=label))

    classes = op.classify_circuits(filaments, amc_channels, list(range(1, 14)), drives)
    by_circ = {c.circuit: c for c in classes}
    assert len(by_circ) == 13
    for drive in drives:
        c = by_circ[drive.circuit]
        assert c.role == "known_pf"
        assert c.coil_label == drive.conductor
        assert c.amc_channel == drive.channel


# --- build_operator: case circuit gets its OWN G_pf column, never merged ---


def test_build_operator_case_circuit_is_a_separate_g_pf_column():
    filaments = [
        _filament(_P4U_R, _P4U_Z, circuit=8),
        _filament(_P4U_CASE_R, _P4U_CASE_Z, circuit=18),
    ]
    table = _table(filaments, ["p4u_coil_current", "p4u_case_current"])
    forward = op.build_operator(table)

    assert sorted(forward.pf_amc_channels) == ["p4u_case_current", "p4u_coil_current"]
    assert forward.g_pf.shape == (1, 2)  # two SEPARATE columns, not merged
    for circs in forward.pf_merged_circuits:
        assert len(circs) == 1  # neither column is an average of >1 circuit


def test_p6_case_drops_out_of_g_pf_into_g_passive():
    p6u_r, p6u_z = op._PF_COIL_CENTROID["p6u"]
    filaments = [
        _filament(p6u_r, p6u_z, circuit=12),
        _filament(p6u_r + 0.02, p6u_z, circuit=22),
    ]
    table = _table(filaments, ["p6u_current"])
    forward = op.build_operator(table)
    assert forward.pf_amc_channels == ["p6u_current"]
    assert forward.g_pf.shape[1] == 1
    # the P6U case circuit becomes one INFERRED passive column.
    assert forward.g_passive.shape[1] == 1
    assert np.allclose(forward.passive_rz[0], [p6u_r + 0.02, p6u_z])


# --- separate case-current columns versus an invalid merged model ----------


def test_separate_case_current_differs_from_wrong_merged_prediction():
    """The separate-column model must differ from a merged-column model.

    The comparison is independent of ``classify_circuits``: it averages the
    active and case Green's-function columns and drives that invalid average
    only by the active-coil current.  The exact delta pins both the sign and
    magnitude needed to catch any silent re-merging of the circuits.
    """
    filaments = [
        _filament(_P4U_R, _P4U_Z, circuit=8),
        _filament(_P4U_CASE_R, _P4U_CASE_Z, circuit=18),
    ]
    table = _table(filaments, ["p4u_coil_current", "p4u_case_current"])
    forward = op.build_operator(table)

    amc_active_ka, amc_case_ka = 100.0, 5.0  # kA*turn, realistic order-of-magnitude
    amc = {"p4u_coil_current": amc_active_ka, "p4u_case_current": amc_case_ka}
    i_pf = forward.assemble_pf_currents(amc)
    pred_separate = forward.vacuum_prediction(i_pf)

    # independently recompute the two raw Green's-function columns (never via
    # classify_circuits / build_operator) to build the invalid merged model.
    is_flux = np.array([False])
    sensor_r = np.array([1.3])
    sensor_z = np.array([0.0])
    sensor_ang = np.array([-90.0])
    col_active = op._green_columns(
        np.array([_P4U_R]),
        np.array([_P4U_Z]),
        np.array([1.0]),
        sensor_r,
        sensor_z,
        sensor_ang,
        is_flux,
    )
    col_case = op._green_columns(
        np.array([_P4U_CASE_R]),
        np.array([_P4U_CASE_Z]),
        np.array([1.0]),
        sensor_r,
        sensor_z,
        sensor_ang,
        is_flux,
    )
    i_active_a = amc_active_ka * op._KA_TURN_TO_A
    i_case_a = amc_case_ka * op._KA_TURN_TO_A

    # Each column is driven by its own current and matches the operator output.
    pred_separate_hand = col_active * i_active_a + col_case * i_case_a
    assert np.allclose(pred_separate, pred_separate_hand, rtol=1e-12)

    # The invalid model merges both circuits and drops the measured case current.
    pred_merged_hand = 0.5 * (col_active + col_case) * i_active_a

    delta = pred_separate - pred_merged_hand
    expected_delta = 0.5 * i_active_a * (col_active - col_case) + i_case_a * col_case
    assert np.allclose(delta, expected_delta, rtol=1e-12)
    # the correction must be non-trivial for this fixture (case current is
    # 1/20th of active but the case sits at a different, non-degenerate
    # location -- both terms of expected_delta are non-zero).
    assert np.all(np.abs(delta) > 0)
    assert not np.allclose(pred_separate, pred_merged_hand)


def test_write_case_circuit_impact_artifact_schema():
    """The measurement artifact retains the fields needed by its impact report."""
    import json
    from pathlib import Path

    art = (
        Path(__file__).resolve().parents[2]
        / "imas_ambix"
        / "gs"
        / "artifacts"
        / "case_circuit_impact.json"
    )
    if not art.exists():
        pytest.skip("case_circuit_impact.json not generated in this checkout")
    payload = json.loads(art.read_text())
    assert payload["schema"] == "case-circuit-impact-v0"
    assert payload["per_probe"]
    for row in payload["per_probe"]:
        assert {"channel", "max_abs_delta_T", "frac_of_sensor_scale"} <= row.keys()


# --- a source that states which conductors are supplied --------------------


def test_a_declared_active_set_is_what_decides_a_known_role():
    """A structural circuit inside the match radius is not promoted by position.

    A source that resolves the structure around a coil into its own circuits
    puts several of them closer to the winding than the centroid radius, so the
    geometric pass alone would drive each of them with the winding's measured
    current.  When such a source states which circuits it supplies, only those
    may take a known role, however near the others sit.
    """
    filaments = [
        _filament(_P4U_R, _P4U_Z, circuit=0),
        _filament(_P4U_CASE_R, _P4U_CASE_Z, circuit=1),
        _filament(_P4U_R, _P4U_Z + 0.03, circuit=2),
    ]

    classes = op.classify_circuits(
        filaments, ["p4u_coil_current", "p4u_case_current"], active_circuits=[0]
    )

    by_circ = {c.circuit: c for c in classes}
    assert by_circ[0].role == "known_pf"
    assert by_circ[0].amc_channel == "p4u_coil_current"
    assert by_circ[1].role == "inferred_passive"
    assert by_circ[2].role == "inferred_passive"
    assert by_circ[1].amc_channel == ""
    assert by_circ[2].amc_channel == ""


def test_declared_membership_is_independent_of_another_sources_numbering():
    """Circuit identifiers only carry meaning within their declaring source."""
    filaments = [_filament(_P4U_R, _P4U_Z, circuit=18)]

    classes = op.classify_circuits(
        filaments, ["p4u_coil_current", "p4u_case_current"], active_circuits=[18]
    )

    assert classes[0].role == "known_pf"
    assert classes[0].coil_label == "p4u"
    assert classes[0].amc_channel == "p4u_coil_current"


def test_a_source_that_states_nothing_keeps_only_geometric_classification():
    """Without declarations, nearby structure cannot be distinguished by role."""
    filaments = [
        _filament(_P4U_R, _P4U_Z, circuit=8),
        _filament(_P4U_CASE_R, _P4U_CASE_Z, circuit=18),
    ]
    channels = ["p4u_coil_current", "p4u_case_current"]

    stated = op.classify_circuits(filaments, channels, active_circuits=())
    inherited = op.classify_circuits(filaments, channels)

    assert stated == inherited
    assert [c.role for c in inherited] == ["known_pf", "known_pf"]
    assert not any(c.source_stated_weight for c in inherited)


def test_a_declared_drive_displaces_every_positional_rule():
    """A drive names the channel outright, displacing positional reconstruction."""
    filaments = [
        _filament(_P4U_R, _P4U_Z, circuit=8),
        _filament(_P4U_CASE_R, _P4U_CASE_Z, circuit=18),
    ]
    channels = ["p4u_coil_current", "p4u_case_current"]
    # deliberately CROSSED against what position would conclude
    drives = [
        _fixtures.CircuitDrive(
            circuit=8,
            channel="p4u_case_current",
            ampere_turns_per_ampere=1.0,
            evidence="generated",
        ),
        _fixtures.CircuitDrive(
            circuit=18,
            channel="p4u_coil_current",
            ampere_turns_per_ampere=1.0,
            evidence="measured",
        ),
    ]

    declared = op.classify_circuits(filaments, channels, [18], drives)

    assert [c.amc_channel for c in declared] == [
        "p4u_case_current",
        "p4u_coil_current",
    ]
    assert [c.role for c in declared] == ["known_case", "known_pf"]
    assert all(c.source_stated_weight for c in declared)


def test_a_declared_channel_this_campaign_lacks_stays_inferred():
    """A conductor the source says is supplied, that nothing measured here,
    keeps an induced current rather than borrowing another channel."""
    filaments = [_filament(_P4U_R, _P4U_Z, circuit=8)]
    drives = [
        _fixtures.CircuitDrive(
            circuit=8,
            channel="p4u_coil_current",
            ampere_turns_per_ampere=1.0,
            evidence="measured",
        )
    ]

    declared = op.classify_circuits(filaments, ["p3u_coil_current"], [8], drives)

    assert [c.role for c in declared] == ["inferred_passive"]
    assert "does not publish that channel" in declared[0].flag


def test_a_stated_weight_withholds_the_fitted_solenoid_correction():
    """The correction and a stated weight are the same claim; both is twice."""
    sol_r, sol_z = op._PF_COIL_CENTROID["sol"]
    filaments = [_filament(sol_r, sol_z, circuit=1)]
    table_inferred = _table(filaments, ["sol_current"])
    table_inferred.active_circuits = []
    table_inferred.circuit_drives = []
    table_stated = _table(filaments, ["sol_current"])
    table_stated.circuit_drives = [
        _fixtures.CircuitDrive(
            circuit=1,
            channel="sol_current",
            ampere_turns_per_ampere=1.0,
            evidence="fitted",
        )
    ]
    table_stated.active_circuits = [1]

    inferred = op.build_operator(table_inferred).g_pf[:, 0]
    stated = op.build_operator(table_stated).g_pf[:, 0]

    assert inferred == pytest.approx(stated * op.SOLENOID_RESPONSE_SCALE)
