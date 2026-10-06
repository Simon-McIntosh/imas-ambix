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

import numpy as np
import pytest

from imas_alambic.signal_map import load_packaged_signal_map
from imas_alambic.virtual_zarr import VirtualZarrError, VirtualZarrView
from imas_ambix.data.paths import JT60SA_ROOT, PACKAGED_MACHINE_MAP_ROOT

SHOTS = (101154, 101173, 60033)
COILS = ("CS1", "CS2", "CS3", "CS4", "EF1", "EF2", "EF3", "EF4", "EF5", "EF6")
SYSTEMS = ("pf_active", "tf", "magnetics")
SOURCE_DATASET = "jt-60sa-eddb-cache"
PF_CURRENT_PATH = "pf_active/coil/current/data"
TF_CURRENT_PATH = "tf/coil/current/data"
FLUX_LOOP_PATH = "magnetics/flux_loop/flux/data"
PROBE_FIELD_PATH = "magnetics/b_field_pol_probe/field/data"
PLASMA_CURRENT_PATH = "magnetics/ip/data"
UNIT_TABLE_PATH = PACKAGED_MACHINE_MAP_ROOT / "jt-60sa-eddb-units.json"
STORE_ROOT = JT60SA_ROOT / "machine_description" / "OP1"
DD_VERSION = "4.1.1"
# A corroboration verb not negated by the immediately preceding "not": the
# negated form ("do not corroborate") is the legitimate way a reason says the
# other shots do NOT support the choice.
_CORROBORATES = re.compile(r"(?<!not )corroborat", re.IGNORECASE)

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


def _blocked_index():
    return {
        (row.source_group, row.source_array): row for row in _magnetics().blocked
    }


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


def test_every_jt60sa_signal_rule_is_source_only():
    for system in SYSTEMS:
        source_map = load_packaged_signal_map("jt-60sa", system)
        assert source_map.signals, f"jt-60sa/{system} serves no signal rule"
        for rule in source_map.signals:
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
            "the other measurement chain of the same coil current"
            in blocked[0].reason
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
    assert sorted(rule.target_index for rule in loops) == list(range(27))
    for rule in loops:
        index = rule.target_index + 1
        assert rule.source_group == "MDAC"
        assert rule.source_array == f"magFlxLp{index}", (
            rule.target_index,
            rule.source_array,
        )
        assert rule.source_unit == "Wb"
        assert rule.target_unit == "Wb"
        assert rule.unit_factor == 1.0
        assert rule.validation_state == "source-only"


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
        assert rule.validation_state == "source-only"


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
    checked = 0
    for system in SYSTEMS:
        for rule in load_packaged_signal_map("jt-60sa", system).signals:
            if rule.source_group not in store:
                continue
            group = store[rule.source_group]
            if rule.source_array not in group:
                continue
            spelling = group[rule.source_array].attrs.get("units")
            assert spelling in table, (system, rule.source_array, spelling)
            assert table[spelling] == rule.source_unit, (
                system,
                rule.source_array,
                spelling,
                rule.source_unit,
            )
            checked += 1
    assert checked > 0


@needs_cache
def test_magnetics_raw_flux_loops_pin_to_the_processed_channels():
    for rule in _rules_for(FLUX_LOOP_PATH):
        index = rule.target_index + 1
        assert rule.source_array == f"magFlxLp{index}", (
            rule.target_index,
            rule.source_array,
        )
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
    claims the choice rests on E101154 alone and cites a shot as corroborating it
    is self-contradictory; the negated form ("do not corroborate") is legitimate.
    """
    for system in SYSTEMS:
        for row in load_packaged_signal_map("jt-60sa", system).blocked:
            if "E101154 alone" not in row.reason:
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
