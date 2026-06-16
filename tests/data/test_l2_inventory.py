"""Coherence of the L2 inventory manifest structures + committed artifact.

These run CPU-only and do not read the corpus (the heavy walk is done by
``python -m imas_ambix.data.l2_inventory``). They pin:

* the static name-collision / actuator-pair tables resolve through the
  same classifier the guard uses;
* the committed manifest artifact (if present) honours the
  reconstruction-vs-plan principle field-by-field.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from imas_ambix.data.l2_inventory import (
    ACTUATOR_PAIRS,
    NAME_COLLISIONS,
)
from imas_ambix.data.provenance import classify_l2_field

MANIFEST = (
    Path(__file__).resolve().parents[2]
    / "imas_ambix"
    / "data"
    / "artifacts"
    / "l2_inventory.json"
)


def test_name_collision_plasma_current_is_three_distinct_provenances():
    members = NAME_COLLISIONS["plasma_current"]
    assert ("magnetics", "ip") in members
    assert ("summary", "ip") in members
    assert ("pulse_schedule", "i_plasma") in members


def test_name_collision_line_density_spans_measured_planned_derived():
    members = NAME_COLLISIONS["line_density"]
    assert ("interferometer", "n_e_line") in members  # measured
    assert ("pulse_schedule", "n_e_line") in members  # planned
    assert ("summary", "line_average_n_e") in members  # reconstruction-derived


def test_actuator_pairs_realised_is_measured_planned_is_action():
    # Each pair must couple a realised (measured) field with a planned
    # (pulse-schedule) one — the world model needs both.
    uda = {
        ("pf_active", "coil_current"): "AMC_P2IL FEED CURRENT",
        ("pf_active", "coil_voltage"): "XDC_PF_F_P1",
        ("gas_injection", "inboard_total"): "AGA_INBOARD_TOTAL",
        ("gas_injection", "valve_voltage"): "XDC_GAS_F_G1",
        ("magnetics", "ip"): "AMC_PLASMA CURRENT",
        ("pulse_schedule", "i_plasma"): "XDC_IP_T_IPREF",
        ("interferometer", "n_e_line"): "ANE_DENSITY",
        ("pulse_schedule", "n_e_line"): "XDC_DENSITY_T_NELREF",
    }
    for pair in ACTUATOR_PAIRS:
        rg, rv = pair["realised"]
        pg, pv = pair["planned"]
        r = classify_l2_field(rg, rv, uda[(rg, rv)])
        p = classify_l2_field(pg, pv, uda[(pg, pv)])
        assert r.classification == "input", pair
        assert p.classification == "planned-action", pair


@pytest.mark.skipif(not MANIFEST.exists(), reason="manifest not built yet")
def test_committed_manifest_honours_the_principle():
    m = json.loads(MANIFEST.read_text())
    assert m["schema"].startswith("imas-ambix.l2-inventory")
    # Re-classify every field from its recorded (group, var, uda_name) and
    # confirm the stored classification matches — the artifact cannot drift
    # from the classifier.
    for f in m["fields"]:
        fc = classify_l2_field(f["group"], f["var"], f["uda_name"])
        assert fc.classification == f["classification"], f
    # Equilibrium is entirely banned; the derived summary scalars too.
    eq = [f for f in m["fields"] if f["group"] == "equilibrium"]
    assert eq and all(f["classification"] == "banned" for f in eq)
    derived = {
        (f["group"], f["var"]): f["classification"]
        for f in m["fields"]
        if (f["group"], f["var"])
        in {("summary", "line_average_n_e"), ("summary", "greenwald_density")}
    }
    assert derived and all(c == "banned" for c in derived.values())
    # No field silently left for review.
    assert m["review_fields"] == []
