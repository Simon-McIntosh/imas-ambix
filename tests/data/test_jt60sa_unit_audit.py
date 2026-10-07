"""Unit audit for the packaged JT-60SA signal maps and machine-map catalogue.

Every ``source_unit`` and ``target_unit`` declared by the JT-60SA packaged
signal maps and the machine-map catalogue must resolve through the Data
Dictionary unit vocabulary of the catalogue's own DD version, and every
``target_unit`` must agree with the unit the Data Dictionary declares on the
bound leaf.  The factor that carries a source unit into its target unit is
computed from a small SI prefix, base-unit and compound-product table and
asserted against one hand-computed value per distinct unit pair, and the
equilibrium half's one dimensional factor (the reference radius that turns F =
R . BT into b0) is asserted against its own hand-computed value, so the audit
fails on any unit the vocabulary does not carry.

The Data Dictionary metadata is read from ``imas-python``'s bundled dictionaries
(``imas.dd_zip.dd_etree``) at the version each catalogue names, never from a
hand-maintained list.
"""

from __future__ import annotations

import collections
from typing import Any

import pytest
from imas.dd_zip import dd_etree

from imas_alambic.machine_map import load_packaged_machine_map
from imas_alambic.signal_map import load_packaged_signal_map
from tests.jt60sa_bundle import BUNDLE, SKIP_REASON

pytestmark = pytest.mark.skipif(BUNDLE is None, reason=SKIP_REASON)

MACHINE = "jt-60sa"
SYSTEMS = ("magnetics", "pf_active", "tf", "equilibrium")

# A name, index or structural identifier leaf carries no physical unit; the
# catalogues spell its dimensionless target as "1", which the Data Dictionary
# also uses for dimensionless numeric leaves.
DIMENSIONLESS = "1"

# One hand-computed factor per distinct (source, target) unit pair either
# catalogue declares.  Every pair here is a scale-preserving pair, so each
# factor is one unit of the source equal to one unit of the target; a
# dimensional difference a rule carries (the reference radius that turns the
# FAME F = R . BT in T.m into b0 in T) is carried by the rule's channel_factor
# and does not appear here, because this table converts units and not physics.
HAND_FACTORS: dict[tuple[str, str], float] = {
    ("A", "A"): 1.0,
    ("T", "T"): 1.0,
    ("Wb", "Wb"): 1.0,
    ("1", "1"): 1.0,
    ("m", "m"): 1.0,
    ("rad", "rad"): 1.0,
    ("T.m", "T"): 1.0,
}

# The reference major radius the catalogue declares, and the BTV -> b0 channel
# factor it fixes: FAME's BTV carries F = R . BT in T.m, so B0 = BTV / R on the
# declared radius.  The value is hand-computed here beside the other hand-checked
# factors, so a shifted radius reddens this audit as well as the map test.
REFERENCE_RADIUS_M = 3.0
BTV_TO_B0_FACTOR = 1.0 / REFERENCE_RADIUS_M

# The numeric value of one unit in SI base units of its own dimension.  Only the
# units the catalogues name appear here; the prefix parser below covers a
# prefixed spelling such as mWb without widening the base table, and the
# compound parser covers a product spelling such as T.m by multiplying the
# value of each dotted component.
_SI_BASE = {"1": 1.0, "m": 1.0, "A": 1.0, "T": 1.0, "Wb": 1.0, "rad": 1.0}
_SI_PREFIX = {
    "Y": 1.0e24,
    "Z": 1.0e21,
    "E": 1.0e18,
    "P": 1.0e15,
    "T": 1.0e12,
    "G": 1.0e9,
    "M": 1.0e6,
    "k": 1.0e3,
    "h": 1.0e2,
    "d": 1.0e-1,
    "c": 1.0e-2,
    "m": 1.0e-3,
    "u": 1.0e-6,
    "n": 1.0e-9,
    "p": 1.0e-12,
}

_MISSING = object()


def _si_value(unit: str) -> float:
    """The numeric value of one ``unit`` in SI base units of its dimension."""
    if "." in unit:
        value = 1.0
        for component in unit.split("."):
            value *= _si_value(component)
        return value
    if unit in _SI_BASE:
        return _SI_BASE[unit]
    for prefix in sorted(_SI_PREFIX, key=len, reverse=True):
        rest = unit[len(prefix) :]
        if unit.startswith(prefix) and rest in _SI_BASE:
            return _SI_PREFIX[prefix] * _SI_BASE[rest]
    raise KeyError(f"unit {unit!r} is not in the SI base or prefix table")


def _unit_factor(source_unit: str, target_unit: str) -> float:
    """The factor that carries a value from ``source_unit`` into ``target_unit``."""
    return _si_value(source_unit) / _si_value(target_unit)


def _dd_unit_tables(version: str) -> tuple[set[str], dict[str, str | None]]:
    """The DD unit vocabulary and per-leaf unit for a bundled DD ``version``."""
    root = dd_etree(version).getroot()
    vocabulary: set[str] = set()
    leaf_units: dict[str, str | None] = {}

    def walk(element: Any, prefix: str) -> None:
        for child in element:
            name = child.get("name")
            if name is None:
                walk(child, prefix)
                continue
            if child.tag == "IDS":
                walk(child, name)
                continue
            path = f"{prefix}/{name}" if prefix else name
            units = child.get("units")
            leaf_units[path] = units
            if units is not None:
                vocabulary.add(units)
            walk(child, path)

    walk(root, "")
    return vocabulary, leaf_units


def _signal_rules():
    for system in SYSTEMS:
        source_map = load_packaged_signal_map(MACHINE, system)
        for rule in source_map.signals:
            yield system, rule.source_unit, rule.target_unit, rule.target_path, rule


def _machine_bindings():
    catalog = load_packaged_machine_map(MACHINE)
    for binding_set, bindings in catalog.binding_sets.items():
        for binding in bindings:
            yield (
                binding_set,
                binding.source_unit,
                binding.target_unit,
                binding.dd_path,
                binding,
            )


def _tables_for_signals() -> tuple[set[str], dict[str, str | None]]:
    version = load_packaged_signal_map(MACHINE, SYSTEMS[0]).target_dd_version
    return _dd_unit_tables(version)


def _tables_for_machine_map() -> tuple[set[str], dict[str, str | None]]:
    return _dd_unit_tables(load_packaged_machine_map(MACHINE).dd_version)


def test_signal_unit_vocabulary_resolves():
    """Every signal-map source and target unit is in the catalogue's DD vocabulary."""
    vocabulary, _ = _tables_for_signals()
    assert vocabulary, "the DD vocabulary is empty"
    for system, source_unit, target_unit, _, _ in _signal_rules():
        assert source_unit in vocabulary, (system, "source_unit", source_unit)
        assert target_unit in vocabulary, (system, "target_unit", target_unit)


def test_machine_map_unit_vocabulary_resolves():
    """Every machine-map source and target unit is in the catalogue's DD vocabulary."""
    vocabulary, _ = _tables_for_machine_map()
    assert vocabulary, "the DD vocabulary is empty"
    for binding_set, source_unit, target_unit, _, _ in _machine_bindings():
        assert source_unit in vocabulary, (binding_set, "source_unit", source_unit)
        assert target_unit in vocabulary, (binding_set, "target_unit", target_unit)


def test_signal_target_unit_equals_the_dd_leaf_unit():
    """Each signal target unit equals the unit the DD declares on its leaf."""
    _, leaf_units = _tables_for_signals()
    for system, _, target_unit, target_path, _ in _signal_rules():
        declared = leaf_units.get(target_path, _MISSING)
        assert declared is not _MISSING, (system, "unresolved path", target_path)
        expected = DIMENSIONLESS if declared is None else declared
        assert target_unit == expected, (system, target_path, target_unit, expected)


def test_machine_map_target_unit_equals_the_dd_leaf_unit():
    """Each machine-map target unit equals the DD leaf unit on its dd_path."""
    _, leaf_units = _tables_for_machine_map()
    for binding_set, _, target_unit, dd_path, _ in _machine_bindings():
        declared = leaf_units.get(dd_path, _MISSING)
        assert declared is not _MISSING, (binding_set, "unresolved path", dd_path)
        expected = DIMENSIONLESS if declared is None else declared
        assert target_unit == expected, (binding_set, dd_path, target_unit, expected)


def test_every_declared_pair_has_one_hand_factor_and_it_derives():
    """Every distinct unit pair carries one hand value the derivation reproduces."""
    pairs = collections.Counter()
    for _, source_unit, target_unit, _, _ in _machine_bindings():
        pairs[(source_unit, target_unit)] += 1
    for _, source_unit, target_unit, _, _ in _signal_rules():
        pairs[(source_unit, target_unit)] += 1

    assert pairs, "no unit pairs were audited"
    for pair in pairs:
        assert pair in HAND_FACTORS, f"unit pair {pair} has no hand-computed value"
        derived = _unit_factor(*pair)
        assert derived == pytest.approx(HAND_FACTORS[pair]), (pair, derived)


def test_signal_derived_unit_factor_matches_the_hand_value():
    """The declared signal unit factor equals the hand value for its pair."""
    for system, source_unit, target_unit, _, rule in _signal_rules():
        hand = HAND_FACTORS[(source_unit, target_unit)]
        assert rule.unit_factor == pytest.approx(hand), (
            system,
            rule.semantic_id,
            rule.unit_factor,
            hand,
        )


def test_equilibrium_b0_channel_factor_is_the_hand_computed_radius_reciprocal():
    """The BTV -> b0 factor is the reciprocal of the declared reference radius.

    FAME's BTV carries F = R . BT in T.m, and b0 is the toroidal field on the
    catalogue's declared reference major radius, so the factor carrying the
    source into its target is 1/R.  The radius is hand-computed to 3.0 m, the
    catalogue's own ``reference_radius``, so a shifted radius reddens this audit.
    """
    rules = {
        rule.semantic_id: rule
        for rule in load_packaged_signal_map(MACHINE, "equilibrium").signals
    }
    rule = rules["equilibrium_toroidal_field_b0"]
    assert rule.source_unit == "T.m"
    assert rule.target_unit == "T"
    assert rule.unit_factor == pytest.approx(1.0)
    assert rule.channel_factor == pytest.approx(BTV_TO_B0_FACTOR)


def test_audit_reports_the_declared_unit_pair_inventory():
    """Both catalogues are reached and their distinct pairs are the audited set."""
    pairs = collections.Counter()
    for _, source_unit, target_unit, _, _ in _machine_bindings():
        pairs[(source_unit, target_unit)] += 1
    for _, source_unit, target_unit, _, _ in _signal_rules():
        pairs[(source_unit, target_unit)] += 1
    assert sum(pairs.values()) == 230
    assert set(pairs) == set(HAND_FACTORS)
