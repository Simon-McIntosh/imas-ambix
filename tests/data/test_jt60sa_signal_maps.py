"""JT-60SA first-pass current signal maps over the on-demand EDDB cache.

The three packaged maps bind the commissioning-shot channels whose names are
already known: the MMSYS coil currents (two chains), the MMSYS TF currents and
the PSRC plasma current.  The maps are read by
:func:`imas_alambic.signal_map.load_packaged_signal_map` and applied by
:class:`imas_alambic.virtual_zarr.VirtualZarrView` directly over the
``{shot}.zarr/{category}/{dname}`` cache.

The two measurement chains of every coil are recorded as a ranked pair: one is
served and the other is blocked, and the test prints the measured disagreement
so the served chain's choice is auditable.  Cache units are Amperes as recorded
by the EDDB census, so the plasma-current binding applies no scale factor.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from imas_alambic.machine_map import load_packaged_machine_map
from imas_alambic.signal_map import load_packaged_signal_map
from imas_alambic.virtual_zarr import VirtualZarrError, VirtualZarrView
from imas_ambix.data.paths import JT60SA_ROOT
from tests.jt60sa_bundle import BUNDLE, SKIP_REASON

pytestmark = pytest.mark.skipif(BUNDLE is None, reason=SKIP_REASON)

SHOTS = (101154, 101173, 60033)
COILS = ("CS1", "CS2", "CS3", "CS4", "EF1", "EF2", "EF3", "EF4", "EF5", "EF6")
SYSTEMS = ("pf_active", "tf", "magnetics")
SOURCE_DATASET = "jt-60sa-eddb-cache"
PF_CURRENT_PATH = "pf_active/coil/current/data"
TF_CURRENT_PATH = "tf/coil/current/data"
FLUX_LOOP_PATH = "magnetics/flux_loop/flux/data"
PROBE_FIELD_PATH = "magnetics/b_field_pol_probe/field/data"
PLASMA_CURRENT_PATH = "magnetics/ip/data"
UNIT_TABLE_PATH = BUNDLE.root / "eddb_units.json" if BUNDLE else Path()
STORE_ROOT = BUNDLE.store_roots["description"] / "OP1" if BUNDLE else Path()
DESCRIPTION_STORE = (
    BUNDLE.store_roots["description"] / "OP1" / "magnetics.nc" if BUNDLE else Path()
)
DD_VERSION = "4.1.1"
# The vacuum shots the magnetics sign verdicts rest on.
VACUUM_SHOTS = (100579, 100595, 100642)
_IDENTITY_PROBES = tuple(f"magPbTC{i}" for i in range(1, 17))
#: Loops 1-6 and 8-11 track reference 7 only on E100642, so their convention is
#: not fixed and they stay source-only.  Reference loop 7 and loops 12-27 track on
#: every shot the fits cover, so those are corpus-validated.
_SOURCE_ONLY_LOOPS = tuple(f"magFlxLp{i}" for i in (1, 2, 3, 4, 5, 6, 8, 9, 10, 11))
_CORPUS_VALIDATED_LOOPS = (7, *range(12, 28))
REFERENCE_FRAGMENT = "jtmm-loop-reference-rca.html"
#: loop -> ((shot, slope, r), ...) for the vacuum shots on which that loop's
#: per-shot differential fit (loop L against reference 7) falls outside the
#: threshold (|slope + 1| <= 0.1 and |r| > 0.95), transcribed from the per-shot
#: table in REFERENCE_FRAGMENT.  The median rule still fixes the loop's sign, so
#: each rule stays corpus-validated, but the failing shot is recorded beside the
#: verdict.
_LOOP_PER_SHOT_OUTLIERS: dict[int, tuple[tuple[int, float, float], ...]] = {
    12: ((100595, -0.341, -0.5920),),
    13: ((100595, -0.513, -0.5578),),
    14: ((100595, -0.416, -0.4192),),
    15: ((100579, -1.156, -0.9940), (100595, -0.890, -0.9170)),
    16: ((100579, -1.185, -0.9924), (100595, -0.929, -0.9497)),
    17: ((100579, -1.127, -0.9875),),
    27: ((100595, -0.856, -0.9017),),
}
_MEDIAN_SLOPE = re.compile(r"median slope [+-]\d+\.\d{3}")
_MEDIAN_ABS_R = re.compile(r"median \|r\| \d+\.\d{3}")
# A corroboration verb not negated by the immediately preceding "not": the
# negated form ("do not corroborate") is the legitimate way a reason says the
# other shots do NOT support the choice.
_CORROBORATES = re.compile(r"(?<!not )corroborat", re.IGNORECASE)
# A blocked reason grounds the chain choice on E101154 as a single-shot basis.
# The claim is expressed either by the word "alone" or by saying the choice
# rests on that shot; either way it contradicts a reason that also cites a
# second shot as corroborating the same choice.
_RESTS_ON_E101154 = re.compile(
    r"E101154 alone|\brests?\b[^.;]*?\bon E101154\b", re.IGNORECASE
)
# Bound-rule channels the map declares but the E101154 cache does not carry, so
# their unit spelling cannot be checked against it. Every bound rule must be
# either present in the cache or named here, so a channel that goes absent fails
# the unit-table test instead of being skipped. The census finds every bound
# channel present on E101154, so the list is empty.
_CHANNELS_ABSENT_FROM_E101154: frozenset[tuple[str, str, str]] = frozenset()

# The equilibrium sign cohort's four FAME scalars.  B0 is the toroidal field on
# the catalogue's declared reference major radius, so BTV, which carries
# F = R . BT in T.m, converts by dividing by that radius; the radius is read
# from machine_map.json rather than restated here, so the catalogue stays its
# one declaration.
EQ_SYSTEM = "equilibrium"
EQ_FRAGMENT = (
    "docs/evidence/fragments/jt60sa-machine-map/"
    "jtmm-equilibrium-source-leads.html"
)
_FAME_BTV_SEMANTIC_ID = "equilibrium_toroidal_field_b0"
# semantic_id -> (source_group, source_array, source_unit, target_path,
#                 target_unit).  The three identity scalars carry channel_factor
# one; the BTV entry's channel factor is the reciprocal of the catalogue's
# declared reference radius and is asserted in the b0 test.
_EQ_RULES = {
    "equilibrium_q_axis": (
        "FAME",
        "QAXIS",
        "1",
        "equilibrium/time_slice/global_quantities/q_axis",
        "1",
    ),
    "equilibrium_q_95": (
        "FAME",
        "Q95",
        "1",
        "equilibrium/time_slice/global_quantities/q_95",
        "1",
    ),
    "equilibrium_plasma_current": (
        "FAME",
        "TTCU",
        "A",
        "equilibrium/time_slice/global_quantities/ip",
        "A",
    ),
    "equilibrium_toroidal_field_b0": (
        "FAME",
        "BTV",
        "T.m",
        "equilibrium/vacuum_toroidal_field/b0",
        "T",
    ),
}

_cache_missing = not (JT60SA_ROOT / "101154.zarr").is_dir()
needs_cache = pytest.mark.skipif(
    _cache_missing, reason="JT-60SA EDDB cache is not mounted"
)
needs_store = pytest.mark.skipif(
    not (STORE_ROOT / "pf_active.nc").is_file(),
    reason="JT-60SA machine-description store is not mounted",
)


def _zarr():
    import zarr

    return zarr


def _group(shot: int):
    return _zarr().open_group(str(JT60SA_ROOT / f"{shot}.zarr"), mode="r")


def _channel(shot: int, group: str, array: str) -> np.ndarray | None:
    store = _group(shot)
    if group not in store:
        return None
    group_obj = store[group]
    if array not in group_obj:
        return None
    return np.asarray(group_obj[array][:], dtype=float).ravel()


def _coil_names(ids_name: str) -> list[str]:
    import imas

    with imas.DBEntry(
        str(STORE_ROOT / f"{ids_name}.nc"), "r", dd_version=DD_VERSION
    ) as entry:
        ids = entry.get(ids_name, autoconvert=False)
        return [str(coil.name) for coil in ids.coil]


def _magnetics():
    return load_packaged_signal_map("jt-60sa", "magnetics")


def _rules_for(target_path: str):
    return [rule for rule in _magnetics().signals if rule.target_path == target_path]


def _rule_index():
    return {rule.source_array: rule for rule in _magnetics().signals}


def _blocked_index():
    return {(row.source_group, row.source_array): row for row in _magnetics().blocked}


def _differential_pairs():
    """Map each type-6 flux-loop entry index to the loop pair it names.

    The store holds its entries in Data Dictionary order, so the pairs are read
    from it rather than assumed: a non-differential (type-1) entry carries the
    int32 fill in place of a pair and is omitted.
    """

    import imas

    pairs = {}
    with imas.DBEntry(DESCRIPTION_STORE, "r") as entry:
        flux_loop = entry.get("magnetics").flux_loop
        for index, loop in enumerate(flux_loop):
            values = np.asarray(
                getattr(loop, "indices_differential", np.empty(0))
            ).reshape(-1)
            if values.size == 2 and not np.any(values == -2147483647):
                pairs[index] = (int(values[0]), int(values[1]))
    return pairs


def _pearson(first: np.ndarray, second: np.ndarray) -> float:
    # Both channels are sampled on the shared 0.25 ms grid, so the truncated
    # common window is a common time base.
    length = min(len(first), len(second))
    return float(np.corrcoef(first[:length], second[:length])[0, 1])


def test_packaged_maps_load_and_name_the_eddb_cache():
    for system in SYSTEMS:
        source_map = load_packaged_signal_map("jt-60sa", system)
        assert source_map.machine == "jt-60sa"
        assert source_map.system == system
        assert source_map.source_dataset == SOURCE_DATASET
        assert source_map.set_version == "0.1.0"
        assert source_map.target_cocos == 17
        assert source_map.target_dd_version == "4.1.1"


def test_every_jt60sa_signal_rule_declares_a_known_validation_state():
    for system in SYSTEMS:
        source_map = load_packaged_signal_map("jt-60sa", system)
        assert source_map.signals, f"jt-60sa/{system} serves no signal rule"
        for rule in source_map.signals:
            assert rule.validation_state in ("source-only", "corpus-validated"), (
                f"{system}:{rule.semantic_id} declares {rule.validation_state}"
            )
            if system != "magnetics":
                assert rule.validation_state == "source-only", (
                    f"{system}:{rule.semantic_id} declares {rule.validation_state}"
                )


@needs_store
def test_pf_active_binds_one_chain_per_coil_at_the_store_index():
    source_map = load_packaged_signal_map("jt-60sa", "pf_active")
    names = _coil_names("pf_active")
    assert names[: len(COILS)] == list(COILS)

    assert len(source_map.signals) == len(COILS)
    for index, coil in enumerate(COILS):
        matches = [
            signal
            for signal in source_map.signals
            if signal.source_array in (f"cur{coil}HiTe", f"cur{coil}LKAT")
        ]
        assert len(matches) == 1, f"coil {coil} serves {len(matches)} chains"
        signal = matches[0]
        assert signal.source_group == "MMSYS"
        assert signal.source_array.startswith(f"cur{coil}")
        assert signal.source_unit == "A"
        assert signal.target_path == PF_CURRENT_PATH
        assert signal.target_index == index

        other = "LKAT" if signal.source_array.endswith("HiTe") else "HiTe"
        blocked = [
            row for row in source_map.blocked if row.source_array == f"cur{coil}{other}"
        ]
        assert len(blocked) == 1, f"coil {coil} blocks {len(blocked)} chains"
        # The blocked row names the unserved chain structurally rather than by
        # the served semantic id, so it does not imply the two chains agree.
        assert (
            "the other measurement chain of the same coil current" in blocked[0].reason
        )
        assert blocked[0].reason.startswith("the other measurement chain")


@needs_cache
def test_pf_active_chains_agree_on_the_cached_shots(capsys):
    """Print max|HiTe-LKAT| / max|LKAT| for every coil on every cached shot."""
    for shot in SHOTS:
        for coil in COILS:
            hi = _channel(shot, "MMSYS", f"cur{coil}HiTe")
            lk = _channel(shot, "MMSYS", f"cur{coil}LKAT")
            assert hi is not None and lk is not None
            denominator = float(np.max(np.abs(lk)))
            relative = float(np.max(np.abs(hi - lk))) / denominator
            print(f"shot {shot} {coil}: max|HiTe-LKAT|/max|LKAT| = {relative:.6f}")

    # On the plasma shot, where every circuit carries current, the two chains
    # track each other; EF6 is the sole circuit beyond a 15% agreement bound and
    # is the one that will be served from HiTe.
    relative_on_plasma = {
        coil: float(
            np.max(
                np.abs(
                    _channel(101154, "MMSYS", f"cur{coil}HiTe")
                    - _channel(101154, "MMSYS", f"cur{coil}LKAT")
                )
            )
        )
        / float(np.max(np.abs(_channel(101154, "MMSYS", f"cur{coil}LKAT"))))
        for coil in COILS
    }
    assert max(relative_on_plasma, key=relative_on_plasma.get) == "EF6"
    assert relative_on_plasma["EF6"] > 0.15
    assert all(
        value <= 0.15 for coil, value in relative_on_plasma.items() if coil != "EF6"
    )
    captured = capsys.readouterr().out
    assert "shot 101154 CS1" in captured


@needs_store
def test_tf_binds_the_first_chain_and_blocks_the_second():
    source_map = load_packaged_signal_map("jt-60sa", "tf")
    assert _coil_names("tf") == ["TF1"]
    assert len(source_map.signals) == 1
    signal = source_map.signals[0]
    assert signal.source_group == "MMSYS"
    assert signal.source_array == "cur1TFLKAT"
    assert signal.source_unit == "A"
    assert signal.target_path == "tf/coil/current/data"
    assert signal.target_index == 0
    assert [row.source_array for row in source_map.blocked] == ["cur2TFLKAT"]
    assert signal.semantic_id in source_map.blocked[0].reason


@needs_cache
def test_tf_cur1_is_the_energised_chain():
    shot = 60033
    cur1 = _channel(shot, "MMSYS", "cur1TFLKAT")
    cur2 = _channel(shot, "MMSYS", "cur2TFLKAT")
    assert cur1 is not None and cur2 is not None
    assert np.max(np.abs(cur1)) > 1.0e4
    assert np.max(np.abs(cur2)) == 0.0


@needs_cache
def test_tf_blocked_row_states_the_two_measured_readings():
    """cur2TFLKAT is not a duplicate of cur1TFLKAT.

    Re-derive both readings the blocked row quotes straight from the cache, then
    require the row's reason to carry them, so the text cannot drift from the
    data: on E060033 every cur2TFLKAT sample is exactly 0.0 while cur1TFLKAT
    carries ~25.7 kA, and on E101154 the two channels peak near 23.3 kA but
    correlate only 0.646 over the cached window.
    """
    reason = load_packaged_signal_map("jt-60sa", "tf").blocked[0].reason

    dead_cur1 = _channel(60033, "MMSYS", "cur1TFLKAT")
    dead_cur2 = _channel(60033, "MMSYS", "cur2TFLKAT")
    assert np.all(dead_cur2 == 0.0)
    assert np.max(np.abs(dead_cur1)) > 2.5e4

    live_cur1 = _channel(101154, "MMSYS", "cur1TFLKAT")
    live_cur2 = _channel(101154, "MMSYS", "cur2TFLKAT")
    assert np.max(np.abs(live_cur1)) > 2.0e4
    assert np.max(np.abs(live_cur2)) > 2.0e4
    correlation = float(np.corrcoef(live_cur1, live_cur2)[0, 1])
    assert correlation < 0.9

    assert "duplicate" not in reason.lower()
    assert str(60033) in reason
    assert "0.0" in reason
    assert f"{correlation:.3f}" in reason


@needs_store
def test_pf_chain_choice_names_the_shot_it_rests_on():
    """Every served rule and blocked row says the choice rests on E101154."""
    source_map = load_packaged_signal_map("jt-60sa", "pf_active")
    for signal in source_map.signals:
        assert "E101154" in signal.evidence
    for row in source_map.blocked:
        assert "E101154" in row.reason


def test_no_blocked_reason_calls_the_other_chain_a_duplicate():
    """A blocked chain is a chain, not a second measurement of the same one.

    The two measurement chains of a coil (and of the TF) decorrelate at the
    noise floor on the shots where the current is too small to compare, so
    calling the unserved chain a duplicate would assert agreement the data
    does not show.
    """
    for system in ("pf_active", "tf"):
        source_map = load_packaged_signal_map("jt-60sa", system)
        for row in source_map.blocked:
            assert "duplicate" not in row.reason.lower(), (system, row.source_array)


def test_magnetics_binds_psrc_ip_in_amperes():
    source_map = load_packaged_signal_map("jt-60sa", "magnetics")
    matches = [
        rule for rule in source_map.signals if rule.target_path == PLASMA_CURRENT_PATH
    ]
    assert len(matches) == 1
    signal = matches[0]
    assert signal.source_group == "PSRC"
    assert signal.source_array == "Ip"
    assert signal.source_unit == "A"
    assert signal.target_path == "magnetics/ip/data"
    assert signal.target_unit == "A"
    assert signal.unit_factor == 1.0
    assert signal.channel_factor == 1.0
    assert "unknown-unvalidated" in signal.evidence


def test_magnetics_binds_every_raw_mdac_flux_loop_in_order():
    loops = _rules_for(FLUX_LOOP_PATH)
    assert len(loops) == 27
    by_loop = {int(rule.source_array.removeprefix("magFlxLp")): rule for rule in loops}
    assert sorted(by_loop) == list(range(1, 28))
    for loop, rule in by_loop.items():
        assert rule.source_group == "MDAC"
        assert rule.source_array == f"magFlxLp{loop}"
        assert rule.source_unit == "Wb"
        assert rule.target_unit == "Wb"
        assert rule.unit_factor == 1.0
        assert rule.channel_factor == -1.0
        assert rule.validation_state in ("source-only", "corpus-validated")


def test_magnetics_retargets_every_loop_but_the_reference_onto_its_differential_entry():
    """Each loop rule but loop 7 targets its type-6 ``[7, L]`` entry.

    Once the store carries the differential pairs, loop ``L``'s own type-1 entry
    holds geometry and no flux, so the rule must resolve to the type-6 entry
    naming ``[7, L]`` -- the difference of loop ``L`` against reference loop 7.
    Loop 7 is the reference and keeps its absolute type-1 entry.
    """

    by_loop = {
        int(rule.source_array.removeprefix("magFlxLp")): rule
        for rule in _rules_for(FLUX_LOOP_PATH)
    }
    pairs = _differential_pairs()
    assert len(pairs) == 26

    assert by_loop[7].target_index == 6
    assert by_loop[7].target_index not in pairs
    for loop, rule in by_loop.items():
        if loop == 7:
            continue
        assert pairs[rule.target_index] == (7, loop), (loop, rule.target_index)


def test_magnetics_marks_loop_seven_and_loops_12_to_27_corpus_validated():
    """The loops that track reference 7 on every shot carry a proven sign.

    Loops 12-27 fit the reference on E100579/E100595/E100642 or on E100642
    alone with loop 7 as reference; loops 1-6 and 8-11 track only on E100642 and
    stay source-only.
    """

    by_loop = {
        int(rule.source_array.removeprefix("magFlxLp")): rule
        for rule in _rules_for(FLUX_LOOP_PATH)
    }
    proven = sorted(
        loop
        for loop, rule in by_loop.items()
        if rule.validation_state == "corpus-validated"
    )
    assert proven == list(_CORPUS_VALIDATED_LOOPS)

    for loop in _CORPUS_VALIDATED_LOOPS:
        if loop != 7:
            assert REFERENCE_FRAGMENT in by_loop[loop].evidence, loop
    for loop in (1, 2, 3, 4, 5, 6, 8, 9, 10, 11):
        assert by_loop[loop].validation_state == "source-only", loop
        assert "E100642" in by_loop[loop].evidence, loop


def test_magnetics_binds_every_raw_mdac_probe_in_order():
    probes = _rules_for(PROBE_FIELD_PATH)
    assert len(probes) == 17
    assert sorted(rule.target_index for rule in probes) == list(range(17))
    for rule in probes:
        index = rule.target_index + 1
        assert rule.source_group == "MDAC"
        assert rule.source_array == f"magPbTC{index}", (
            rule.target_index,
            rule.source_array,
        )
        assert rule.source_unit == "T"
        assert rule.target_unit == "T"
        assert rule.unit_factor == 1.0
        assert rule.validation_state in ("source-only", "corpus-validated")


def test_magnetics_authors_the_vacuum_sign_verdicts():
    """The 16 identity probes and the negate loop 7 carry their measured sign.

    A rule's stated sign must be traceable to the adjudication, so each
    corpus-validated rule's evidence cites the median slope, the median |r| and
    the three vacuum shots the verdict rests on.
    """
    index = _rule_index()
    for name in _IDENTITY_PROBES:
        rule = index[name]
        assert rule.validation_state == "corpus-validated", name
        assert rule.channel_factor == 1.0, name
        assert _MEDIAN_SLOPE.search(rule.evidence), name
        assert _MEDIAN_ABS_R.search(rule.evidence), name
        for shot in VACUUM_SHOTS:
            assert str(shot) in rule.evidence, (name, shot)

    loop7 = index["magFlxLp7"]
    assert loop7.validation_state == "corpus-validated"
    assert loop7.channel_factor == -1.0
    assert _MEDIAN_SLOPE.search(loop7.evidence)
    assert _MEDIAN_ABS_R.search(loop7.evidence)
    for shot in VACUUM_SHOTS:
        assert str(shot) in loop7.evidence, shot

    # The 16 identity probes and loop 7 carry a vacuum-adjudicated sign; the
    # differential loops 12-27 carry a sign proven against the reference fits,
    # so the measured set is 16 + 1 + 16.
    measured = [
        rule
        for rule in _magnetics().signals
        if rule.validation_state == "corpus-validated"
    ]
    assert len(measured) == 33
    assert index["magPbTC17"].validation_state == "source-only"


def test_magnetics_leaves_the_unmatched_loops_source_only():
    """Loops 1-6 and 8-11 and probe 17 keep their sign unproven.

    Each source-only loop still negates its differential channel (the type-6
    entry stores -raw_L), but its convention is not fixed: it tracks reference
    7 only on E100642, so its evidence names that single shot rather than a
    three-shot verdict.
    """

    index = _rule_index()
    source_only = list(_SOURCE_ONLY_LOOPS) + ["magPbTC17"]
    assert len(source_only) == 11
    for name in source_only:
        rule = index[name]
        assert rule.validation_state == "source-only", name
    for name in _SOURCE_ONLY_LOOPS:
        assert index[name].channel_factor == -1.0, name
        assert "E100642" in index[name].evidence, name
    assert index["magPbTC17"].channel_factor == 1.0


def test_magnetics_records_the_differential_loop_per_shot_outliers():
    """A loop whose per-shot fit fails names the failing shot, slope and r.

    A differential loop rule keeps the reference-proven sign from the median
    rule, so it stays corpus-validated.  A shot whose per-shot fit falls outside
    the threshold is an outlier, and the rule records it beside the verdict so a
    reader sees which shot the verdict rests less securely on.
    """

    by_loop = {
        int(rule.source_array.removeprefix("magFlxLp")): rule
        for rule in _rules_for(FLUX_LOOP_PATH)
    }
    for loop, outliers in _LOOP_PER_SHOT_OUTLIERS.items():
        rule = by_loop[loop]
        assert rule.validation_state == "corpus-validated", loop
        for shot, slope, r in outliers:
            assert f"E{shot}" in rule.evidence, (loop, shot)
            assert f"{slope:+.3f}" in rule.evidence, (loop, shot, slope)
            assert f"{r:+.4f}" in rule.evidence, (loop, shot, r)


def test_magnetics_blocks_each_processed_psrc_alternate_naming_the_raw():
    mapping = _blocked_index()
    for i in range(1, 28):
        row = mapping[("PSRC", f"magFluxLp{i}")]
        assert f"raw MDAC magFlxLp{i}" in row.reason, (i, row.reason)
    for j in range(1, 18):
        row = mapping[("PSRC", f"magPbTC{j}")]
        assert f"raw MDAC magPbTC{j}" in row.reason, (j, row.reason)


def test_magnetics_blocks_the_loops_and_probes_beyond_selene():
    mapping = _blocked_index()
    for i in range(28, 35):
        row = mapping[("MDAC", f"magFlxLp{i}")]
        assert "27" in row.reason and "SELENE" in row.reason
    for i in range(18, 24):
        row = mapping[("MDAC", f"magPbTC{i}")]
        assert "17" in row.reason and "SELENE" in row.reason


@needs_cache
def test_eddb_unit_spellings_resolve_to_each_bound_rule():
    table = json.loads(UNIT_TABLE_PATH.read_text(encoding="utf-8"))["units"]
    store = _group(101154)
    bound: set[tuple[str, str, str]] = set()
    checked: set[tuple[str, str, str]] = set()
    for system in SYSTEMS:
        for rule in load_packaged_signal_map("jt-60sa", system).signals:
            key = (system, rule.source_group, rule.source_array)
            bound.add(key)
            present = (
                rule.source_group in store
                and rule.source_array in store[rule.source_group]
            )
            if not present:
                # The channel is absent, so its spelling cannot be checked; it
                # must be one the map already knows to be absent, otherwise the
                # rule is silently unverified.
                assert key in _CHANNELS_ABSENT_FROM_E101154, key
                continue
            group = store[rule.source_group]
            spelling = group[rule.source_array].attrs.get("units")
            assert spelling in table, (system, rule.source_array, spelling)
            assert table[spelling] == rule.source_unit, (
                system,
                rule.source_array,
                spelling,
                rule.source_unit,
            )
            checked.add(key)
    # Every bound rule is either checked or an explicitly excused absence, so a
    # newly absent channel fails here rather than being skipped under a bare
    # "checked > 0".
    assert checked == bound - _CHANNELS_ABSENT_FROM_E101154, (
        bound - _CHANNELS_ABSENT_FROM_E101154 - checked,
    )
    assert checked


@needs_cache
def test_magnetics_raw_flux_loops_pin_to_the_processed_channels():
    for rule in _rules_for(FLUX_LOOP_PATH):
        index = int(rule.source_array.removeprefix("magFlxLp"))
        raw = _channel(101154, "MDAC", rule.source_array)
        processed = _channel(101154, "PSRC", f"magFluxLp{index}")
        assert raw is not None and processed is not None
        correlation = _pearson(raw, processed)
        assert correlation > 0.999, (index, correlation)


@needs_cache
def test_magnetics_raw_probes_pin_to_the_processed_channels():
    for rule in _rules_for(PROBE_FIELD_PATH):
        index = rule.target_index + 1
        assert rule.source_array == f"magPbTC{index}", (
            rule.target_index,
            rule.source_array,
        )
        raw = _channel(101154, "MDAC", rule.source_array)
        processed = _channel(101154, "PSRC", f"magPbTC{index}")
        assert raw is not None and processed is not None
        correlation = _pearson(raw, processed)
        assert correlation > 0.999, (index, correlation)


def test_no_blocked_reason_rests_on_e101154_alone_while_citing_corroboration():
    """A choice may rest on E101154 alone only when no shot corroborates it.

    The other commissioning shots sit at the noise floor, so a reason that both
    grounds the choice on E101154 as a single-shot basis and cites a shot as
    corroborating it is self-contradictory; the negated form ("do not
    corroborate") is legitimate. The single-shot basis is caught by either the
    word "alone" or a claim that the choice rests on E101154, so a reason that
    avoids the word "alone" but repeats the contradiction still fails.
    """
    for system in SYSTEMS:
        for row in load_packaged_signal_map("jt-60sa", system).blocked:
            if not _RESTS_ON_E101154.search(row.reason):
                continue
            assert not _CORROBORATES.search(row.reason), (
                system,
                row.source_array,
                row.reason,
            )


@needs_cache
@pytest.mark.parametrize("shot", SHOTS)
def test_virtual_zarr_reads_every_rule_with_a_source_array(shot):
    served = 0
    for system in ("pf_active", "tf", "magnetics"):
        source_map = load_packaged_signal_map("jt-60sa", system)
        view = VirtualZarrView.open(
            str(JT60SA_ROOT / f"{shot}.zarr"), source_map, shot=shot
        )
        for signal in source_map.signals:
            present = (
                _channel(shot, signal.source_group, signal.source_array) is not None
            )
            if not present:
                with pytest.raises(VirtualZarrError):
                    view[signal.semantic_id]
                continue
            values = np.asarray(view[signal.semantic_id][:]).ravel()
            assert values.ndim == 1 and values.size > 0
            assert np.isfinite(values).all()
            served += 1
    assert served > 0


@needs_cache
def test_psrc_ip_is_absent_where_the_shot_has_no_psrc():
    source_map = load_packaged_signal_map("jt-60sa", "magnetics")
    for shot in (101173, 60033):
        assert _channel(shot, "PSRC", "Ip") is None
        view = VirtualZarrView.open(
            str(JT60SA_ROOT / f"{shot}.zarr"), source_map, shot=shot
        )
        with pytest.raises(VirtualZarrError, match="absent"):
            view["magnetics_plasma_current"]


@needs_cache
def test_psrc_ip_peak_on_the_plasma_shot_exceeds_a_megaampere_scale():
    source_map = load_packaged_signal_map("jt-60sa", "magnetics")
    view = VirtualZarrView.open(
        str(JT60SA_ROOT / "101154.zarr"), source_map, shot=101154
    )
    served = np.asarray(view["magnetics_plasma_current"][:]).ravel()
    raw = _channel(101154, "PSRC", "Ip")
    assert np.max(np.abs(served)) > 1.0e4
    # The cache already carries Amperes, so the served value equals the source.
    assert np.max(np.abs(served)) == pytest.approx(float(np.max(np.abs(raw))))


def _equilibrium():
    return load_packaged_signal_map("jt-60sa", EQ_SYSTEM)


def _equilibrium_index():
    return {rule.semantic_id: rule for rule in _equilibrium().signals}


def _reference_radius_m() -> float:
    """The reference major radius the catalogue declares in machine_map.json."""
    catalog = load_packaged_machine_map("jt-60sa")
    return catalog.description_supplements[0].reference_radius


def _channel_factor(semantic_id: str) -> float:
    """The channel factor a rule is expected to declare.

    The three identity scalars carry one; BTV carries F = R . BT in T.m and is
    divided by the catalogue's declared reference radius to give b0 in T.
    """
    if semantic_id == _FAME_BTV_SEMANTIC_ID:
        return 1.0 / _reference_radius_m()
    return 1.0


# Plausible FAME scalars for one shot; the equilibrium rules bind these source
# arrays and the transform must not touch the raw values.
_FAME_SERIES = {
    "QAXIS": (0.98, 1.00, 1.02, 1.01, 0.99, 1.03, 1.00, 0.97),
    "Q95": (30.0, 29.5, 31.0, 32.4, 28.7, 29.9, 30.5, 33.1),
    "TTCU": (4.0e5, 5.0e5, 6.5e5, 7.2e5, 6.1e5, 5.3e5, 4.4e5, 3.8e5),
    "BTV": (6.02, 6.01, 6.00, 5.99, 5.985, 5.995, 6.005, 6.023),
}


def _fame_store(tmp_path):
    """Build a FAME zarr store under ``tmp_path`` and return its path."""

    import zarr

    path = tmp_path / "101154.zarr"
    root = zarr.open_group(path, mode="w")
    fame = root.create_group("FAME")
    length = len(next(iter(_FAME_SERIES.values())))
    time = np.linspace(0.0, 7.0, length)
    for name, values in _FAME_SERIES.items():
        fame.create_array(name, data=np.asarray(values, dtype=np.float64))
        fame.create_array(f"{name}_time", data=time)
    return path


def test_equilibrium_map_binds_the_four_fame_scalars():
    """Each FAME scalar binds its DD leaf, source-only, citing the survey."""
    source_map = _equilibrium()
    assert source_map.machine == "jt-60sa"
    assert source_map.system == EQ_SYSTEM
    assert source_map.source_dataset == SOURCE_DATASET
    assert source_map.set_version == "0.1.0"
    assert source_map.target_cocos == 17
    assert source_map.target_dd_version == "4.1.1"

    index = _equilibrium_index()
    assert set(index) == set(_EQ_RULES)
    for semantic_id, (
        group,
        array,
        source_unit,
        target_path,
        target_unit,
    ) in _EQ_RULES.items():
        rule = index[semantic_id]
        assert rule.source_group == group
        assert rule.source_array == array
        assert rule.source_unit == source_unit
        assert rule.target_path == target_path
        assert rule.target_unit == target_unit
        assert rule.target_index is None
        assert rule.transformation == "one_like"
        assert rule.source_cocos is None
        assert rule.unit_factor == 1.0
        assert rule.channel_factor == pytest.approx(_channel_factor(semantic_id))
        assert rule.validation_state == "source-only"
        assert EQ_FRAGMENT in rule.evidence


def test_equilibrium_map_compiles_and_reads_each_raw_series(tmp_path):
    """Compile the map, then read every rule's raw series over the FAME store."""
    source_map = _equilibrium()
    compiled = source_map.compile(101154)
    assert sorted(signal.rule.semantic_id for signal in compiled) == sorted(_EQ_RULES)

    path = _fame_store(tmp_path)
    view = VirtualZarrView.open(str(path), source_map, shot=101154)
    time = np.linspace(0.0, 7.0, len(_FAME_SERIES["QAXIS"]))
    for semantic_id in _EQ_RULES:
        values, times = view.raw_series(semantic_id)
        array = _EQ_RULES[semantic_id][1]
        assert values == pytest.approx(np.asarray(_FAME_SERIES[array], dtype=float))
        assert times == pytest.approx(time)


def test_equilibrium_b0_divides_btimes_r_by_the_reference_radius(tmp_path):
    """The served b0 is BTV over the catalogue's declared reference radius."""
    radius = _reference_radius_m()
    rule = _equilibrium_index()[_FAME_BTV_SEMANTIC_ID]
    assert rule.channel_factor == pytest.approx(1.0 / radius)

    path = _fame_store(tmp_path)
    view = VirtualZarrView.open(str(path), _equilibrium(), shot=101154)

    raw_btv, _ = view.raw_series(_FAME_BTV_SEMANTIC_ID)
    served = np.asarray(view[_FAME_BTV_SEMANTIC_ID][:]).ravel()

    assert served == pytest.approx(raw_btv / radius)
    # The rule must actually divide by the radius, so the served value differs
    # from the raw F = R . BT it was read from.
    assert not np.allclose(served, raw_btv)
