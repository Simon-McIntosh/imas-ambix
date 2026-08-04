"""Regression tests for the case-circuit fix in ``classify_circuits``.

Before this fix, ``classify_circuits`` labelled any ``efm`` fcoil circuit
within 8 cm of a known PF-coil centroid as "known_pf" for that coil, with no
way to tell an ACTIVE coil apart from its co-located, separately-supplied
CASE circuit (docs/mast-coil-circuits.html §6).  8 of MAST's 10 case circuits
sit within that 8 cm radius of their active coil and were silently averaged
into the active coil's G_pf column, driven by the ACTIVE coil's (much larger)
amc current instead of their own small measured case current.

The fix uses the authoritative ``pfSystems.xml`` id correspondence
(:mod:`imas_ambix.gs.circuits`, measured to agree 1:1 with the ``efm`` circuit
numbering for ids 1-23 on real data, both ``fcoil`` signatures) to recognise a
case circuit BY ID once geometry has already flagged it as a neighbour of an
active coil, and drive it by its own channel (or drop it to INFERRED passive
if that channel is absent / the case is constrained to zero, e.g. P6U/P6L).

These tests are pure-logic (synthetic geometry, no mirror needed).
"""

from __future__ import annotations

import numpy as np
import pytest

from imas_ambix.gs import circuits as pfc
from imas_ambix.gs import geometry as gsg
from imas_ambix.gs import operator as op

# --- coil-model version marker ---------------------------------------------


def test_coil_model_version_bumped_for_the_case_circuit_fix():
    """A downstream cache keying on COIL_MODEL_VERSION must invalidate across
    coil-model fixes -- pin that the marker exists and is neither the implicit
    pre-fix baseline nor any retired intermediate."""
    assert op.COIL_MODEL_VERSION not in ("", "case-circuits-v1", "case-circuits-v2")
    assert op.COIL_MODEL_VERSION == "cylinder-sensors-v5"


def test_operator_summary_reports_coil_model_version(tmp_path):
    table = _table([_filament(_P4U_R, _P4U_Z, circuit=8)], ["p4u_coil_current"])
    operators = op.build_all_operators({table.signature.key: table})
    out = op.write_operator_summary(operators, out_path=tmp_path / "summary.json")
    import json

    payload = json.loads(out.read_text())
    assert payload["coil_model_version"] == op.COIL_MODEL_VERSION


# --- circuits.py cross-check: the ids this fix hinges on ------------------


def test_p4u_case_circuit_id_is_18_confusable_with_p4u():
    """Sanity-pin the fixture's chosen ids against circuits.py's own table."""
    case = pfc.case_circuit_for_active_coil("p4u")
    assert case is not None
    assert case.circuit_id == 18
    assert case.l1_case_channel == "p4u_case_current"
    assert case.constrained_zero is False


def test_p6u_case_circuit_id_is_22_constrained_zero():
    case = pfc.case_circuit_for_active_coil("p6u")
    assert case is not None
    assert case.circuit_id == 22
    assert case.constrained_zero is True
    assert case.l1_case_channel is None


# --- a minimal fixture builder ---------------------------------------------


def _filament(r: float, z: float, circuit: int, xmult: float = 1.0) -> gsg.PFFilament:
    return gsg.PFFilament(
        r=r, z=z, turns=1.0, width=0.01, height=0.01, circuit=circuit, xmult=xmult
    )


def _table(
    filaments: list[gsg.PFFilament], amc_channels: list[str]
) -> gsg.GeometryTable:
    """A single-vertical-probe geometry table around whatever filaments are given."""
    bp = gsg.BProbe(index=0, r=1.3, z=0.0, angle_deg=90.0, length=0.025)
    sig = gsg.SetupSignature(
        n_bprobe=1,
        n_fluxloop=0,
        n_pf_filament=len(filaments),
        n_limiter=4,
        digest="c4se0000c1rcu17s",
    )
    return gsg.GeometryTable(
        signature=sig,
        shots=[99999],
        b_probes=[bp],
        flux_loops=[],
        pf_filaments=filaments,
        limiter_r=[0.3, 1.6, 1.6, 0.3],
        limiter_z=[-1.0, -1.0, 1.0, 1.0],
        sensor_map=[
            gsg.SensorMapping("obv01", "b_probe", 0, 1.3, 0.0, 90.0, 0.001, ""),
        ],
        passive_structures=[],
        amc_current_channels=amc_channels,
        unmatched_amb=[],
    )


# --- the mis-assignment fixture: case gets its OWN channel -----------------

# P4U centroid per operator._PF_COIL_CENTROID; the case sits 2 cm away, well
# inside _COIL_MATCH_M (8 cm) -- exactly the geometric confusion the audit found.
_P4U_R, _P4U_Z = op._PF_COIL_CENTROID["p4u"]
_P4U_CASE_R, _P4U_CASE_Z = _P4U_R + 0.02, _P4U_Z


def test_case_filament_2cm_from_active_centroid_gets_its_own_channel():
    filaments = [
        _filament(_P4U_R, _P4U_Z, circuit=8),  # P4U active (pfSystems.xml id 8)
        _filament(_P4U_CASE_R, _P4U_CASE_Z, circuit=18),  # P4U case (id 18)
    ]
    classes = op.classify_circuits(filaments, ["p4u_coil_current", "p4u_case_current"])
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
    classes = op.classify_circuits(filaments, ["p4u_coil_current"])  # no case channel
    by_circ = {c.circuit: c for c in classes}

    assert by_circ[18].role == "inferred_passive"
    assert by_circ[18].amc_channel == ""
    assert "absent from this campaign" in by_circ[18].flag
    # the active circuit is unaffected by its case's channel being absent.
    assert by_circ[8].role == "known_pf"
    assert by_circ[8].amc_channel == "p4u_coil_current"


def test_p6_case_is_zero_passive_even_if_a_channel_is_injected():
    """P6U/P6L cases are constrained to zero by pfSystems.xml -- they must
    stay INFERRED passive even if a (spurious) case-current channel is
    present, never become "known_case" (docs/mast-coil-circuits.html §3)."""
    p6u_r, p6u_z = op._PF_COIL_CENTROID["p6u"]
    filaments = [
        _filament(p6u_r, p6u_z, circuit=12),  # P6U active (id 12)
        _filament(p6u_r + 0.02, p6u_z, circuit=22),  # P6U case (id 22)
    ]
    # inject a hypothetical "p6u_case_current" -- should still be ignored.
    classes = op.classify_circuits(filaments, ["p6u_current", "p6u_case_current"])
    by_circ = {c.circuit: c for c in classes}

    assert by_circ[12].role == "known_pf"
    assert by_circ[12].amc_channel == "p6u_current"

    assert by_circ[22].role == "inferred_passive"
    assert by_circ[22].amc_channel == ""
    assert "constrained to zero" in by_circ[22].flag


def test_p6_case_without_any_channel_is_also_inferred_passive():
    p6l_r, p6l_z = op._PF_COIL_CENTROID["p6l"]
    filaments = [
        _filament(p6l_r, p6l_z, circuit=13),  # P6L active (id 13)
        _filament(p6l_r - 0.02, p6l_z, circuit=23),  # P6L case (id 23)
    ]
    classes = op.classify_circuits(filaments, ["p6l_current"])
    by_circ = {c.circuit: c for c in classes}
    assert by_circ[13].role == "known_pf"
    assert by_circ[23].role == "inferred_passive"


# --- active mapping unchanged for all 13 active circuits -------------------


def test_all_13_active_circuits_still_classify_known_pf():
    """The fix touches ONLY case-circuit ids (14-23); every active circuit
    (1-13), placed at its real centroid with no case circuit present, must
    classify exactly as it did before the fix."""
    filaments = []
    amc_channels = []
    for active in pfc.active_circuits():
        r, z = op._PF_COIL_CENTROID[active.coil_label]
        filaments.append(_filament(r, z, circuit=active.circuit_id))
        amc_channels.append(active.preferred_current_channel())

    classes = op.classify_circuits(filaments, amc_channels)
    by_circ = {c.circuit: c for c in classes}
    assert len(by_circ) == 13
    for active in pfc.active_circuits():
        c = by_circ[active.circuit_id]
        assert c.role == "known_pf"
        assert c.coil_label == active.coil_label
        assert c.amc_channel == active.preferred_current_channel()


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


# --- regression pin: the corrected G_pf assembly changes the vacuum -------
# --- prediction by exactly the hand-computed delta on a mini fixture ------


def test_regression_pin_vacuum_prediction_delta_vs_pre_fix_merge():
    """Recomputes, independently of ``classify_circuits``, what the PRE-FIX
    merge-by-amc-channel behaviour would have predicted (average the active +
    case Green's-function columns, drive the average by ONLY the active
    coil's current) and asserts the FIXED operator's prediction differs from
    it by exactly that hand-computed delta -- pinning both the sign and the
    magnitude of the correction so a regression silently re-merging the two
    circuits is caught."""
    filaments = [
        _filament(_P4U_R, _P4U_Z, circuit=8),
        _filament(_P4U_CASE_R, _P4U_CASE_Z, circuit=18),
    ]
    table = _table(filaments, ["p4u_coil_current", "p4u_case_current"])
    forward = op.build_operator(table)

    amc_active_ka, amc_case_ka = 100.0, 5.0  # kA*turn, realistic order-of-magnitude
    amc = {"p4u_coil_current": amc_active_ka, "p4u_case_current": amc_case_ka}
    i_pf = forward.assemble_pf_currents(amc)
    pred_after = forward.vacuum_prediction(i_pf)

    # independently recompute the two raw Green's-function columns (never via
    # classify_circuits / build_operator) to build the PRE-FIX prediction.
    is_flux = np.array([False])
    sensor_r, sensor_z, sensor_ang = np.array([1.3]), np.array([0.0]), np.array([90.0])
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

    # FIXED (after): each column driven by its own current -- must match the
    # operator's actual output exactly.
    pred_after_hand = col_active * i_active_a + col_case * i_case_a
    assert np.allclose(pred_after, pred_after_hand, rtol=1e-12)

    # PRE-FIX (before): both circuits merged (averaged) into ONE column keyed
    # on the active coil's channel, driven ONLY by the active current -- the
    # case current never entered the prediction at all.
    pred_before_hand = 0.5 * (col_active + col_case) * i_active_a

    delta = pred_after - pred_before_hand
    expected_delta = 0.5 * i_active_a * (col_active - col_case) + i_case_a * col_case
    assert np.allclose(delta, expected_delta, rtol=1e-12)
    # the correction must be non-trivial for this fixture (case current is
    # 1/20th of active but the case sits at a different, non-degenerate
    # location -- both terms of expected_delta are non-zero).
    assert np.all(np.abs(delta) > 0)
    assert not np.allclose(pred_after, pred_before_hand)


def test_write_case_circuit_impact_artifact_schema():
    """Smoke-check the shape of the committed measurement artifact so a
    future re-measurement can't silently drop the fields this fix's impact
    report depends on."""
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


def test_the_efm_case_id_table_is_not_applied_to_another_sources_numbering():
    """Circuit ids only mean something within the numbering that assigned them.

    ``_CASE_BY_CIRCUIT_ID`` is keyed by ``efm``'s circuit numbering, so reading
    it against a source with its own numbering would label whichever circuit
    happened to land on id 18 a P4U case.  A declared active set is the signal
    that the ids came from elsewhere, so the table is not consulted: circuit 18
    here is the supplied winding and keeps the winding's own channel.
    """
    filaments = [_filament(_P4U_R, _P4U_Z, circuit=18)]

    classes = op.classify_circuits(
        filaments, ["p4u_coil_current", "p4u_case_current"], active_circuits=[18]
    )

    assert classes[0].role == "known_pf"
    assert classes[0].coil_label == "p4u"
    assert classes[0].amc_channel == "p4u_coil_current"


def test_a_source_that_states_nothing_keeps_the_geometric_classification():
    """The declaration is opt-in: an empty one must leave ``efm`` untouched."""
    filaments = [
        _filament(_P4U_R, _P4U_Z, circuit=8),
        _filament(_P4U_CASE_R, _P4U_CASE_Z, circuit=18),
    ]
    channels = ["p4u_coil_current", "p4u_case_current"]

    stated = op.classify_circuits(filaments, channels, active_circuits=())
    inherited = op.classify_circuits(filaments, channels)

    assert stated == inherited
    assert [c.role for c in inherited] == ["known_pf", "known_case"]
