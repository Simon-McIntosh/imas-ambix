"""JT-60SA first-pass current signal maps over the on-demand EDDB cache.

The three packaged maps bind the commissioning-shot channels whose names are
already known: the MMSYS coil currents (two chains), the MMSYS TF currents and
the PSRC plasma current.  The maps are read by
:func:`imas_ambix.data.signal_map.load_packaged_signal_map` and applied by
:class:`imas_ambix.data.virtual_zarr.VirtualZarrView` directly over the
``{shot}.zarr/{category}/{dname}`` cache.

The two measurement chains of every coil are recorded as a ranked pair: one is
served and the other is blocked, and the test prints the measured disagreement
so the served chain's choice is auditable.  Cache units are Amperes as recorded
by the EDDB census, so the plasma-current binding applies no scale factor.
"""

from __future__ import annotations

import numpy as np
import pytest

from imas_ambix.data.paths import JT60SA_ROOT
from imas_ambix.data.signal_map import load_packaged_signal_map
from imas_ambix.data.virtual_zarr import VirtualZarrError, VirtualZarrView

SHOTS = (101154, 101173, 60033)
COILS = ("CS1", "CS2", "CS3", "CS4", "EF1", "EF2", "EF3", "EF4", "EF5", "EF6")
SYSTEMS = ("pf_active", "tf", "magnetics")
SOURCE_DATASET = "jt-60sa-eddb-cache"
PF_CURRENT_PATH = "pf_active/coil/current/data"
TF_CURRENT_PATH = "tf/coil/current/data"
STORE_ROOT = JT60SA_ROOT / "machine_description" / "OP1"
DD_VERSION = "4.1.1"

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


def test_packaged_maps_load_and_name_the_eddb_cache():
    for system in SYSTEMS:
        source_map = load_packaged_signal_map("jt-60sa", system)
        assert source_map.machine == "jt-60sa"
        assert source_map.system == system
        assert source_map.source_dataset == SOURCE_DATASET
        assert source_map.set_version == "0.1.0"
        assert source_map.target_cocos == 17
        assert source_map.target_dd_version == "4.1.1"


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
        assert signal.semantic_id in blocked[0].reason


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


def test_magnetics_binds_psrc_ip_in_amperes():
    source_map = load_packaged_signal_map("jt-60sa", "magnetics")
    assert len(source_map.signals) == 1
    signal = source_map.signals[0]
    assert signal.source_group == "PSRC"
    assert signal.source_array == "Ip"
    assert signal.source_unit == "A"
    assert signal.target_path == "magnetics/ip/data"
    assert signal.target_unit == "A"
    assert signal.unit_factor == 1.0
    assert signal.channel_factor == 1.0
    assert "unknown-unvalidated" in signal.evidence


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
