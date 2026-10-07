"""Write a JT-60SA pulse: the description IDSs carry the served signals.

The writer reads the phase description whole from the machine-description store
and the dynamic values from the EDDB cache, and writes one DD 4.1.1 netCDF file
per description IDS.  This test writes E101154 into a temporary directory and
checks the done-when of the pulse-write step: every description IDS is written
and reads back through :class:`imas.DBEntry`, the ten coils the ``pf_active``
map serves carry the compiled map applied to their cached channels sample for
sample, both FPPC coils carry no current, the TF coil current equals the served
TF chain, ``magnetics/ip`` peaks near 1 MA, and the written IDSs differ from the
OP1 description in no path outside ``ids_properties`` other than the
time-dependent leaves the receipt names.  A shot with no cache refuses and
names the missing cache.

Writing the same pulse once and re-reading is cheap; the pulse is written once
per module into ``tmp_path_factory``.
"""

from __future__ import annotations

import imas
import numpy as np
import pytest
from imas.ids_struct_array import IDSStructArray
from imas.util import idsdiffgen

from imas_alambic.eddb import read_channel
from imas_alambic.pulse_writer import IDS_NAMES, PulseWriteError, write_pulse
from imas_alambic.signal_map import load_packaged_signal_map
from imas_ambix.data.paths import JT60SA_ROOT

SHOT_TOKEN = "E101154"
SHOT_INT = 101154
DD_VERSION = "4.1.1"
MACHINE = "jt-60sa"

PF_COILS = ("CS1", "CS2", "CS3", "CS4", "EF1", "EF2", "EF3", "EF4", "EF5", "EF6")
FPPC_COIL_INDICES = (10, 11)

DESCRIPTION_ROOT = JT60SA_ROOT / "machine_description" / "OP1"
_cache_missing = not (JT60SA_ROOT / f"{SHOT_INT}.zarr").is_dir()
_store_missing = not (DESCRIPTION_ROOT / "pf_active.nc").is_file()
pytestmark = pytest.mark.skipif(
    _cache_missing or _store_missing,
    reason="the JT-60SA EDDB cache or machine-description store is not mounted",
)


@pytest.fixture(scope="module")
def receipt(tmp_path_factory):
    out = tmp_path_factory.mktemp("write")
    return write_pulse(MACHINE, SHOT_TOKEN, out)


def _read(receipt, ids_name: str):
    with imas.DBEntry(receipt.files[ids_name], "r", dd_version=DD_VERSION) as entry:
        return entry.get(ids_name, autoconvert=False)


def _read_description(ids_name: str):
    path = DESCRIPTION_ROOT / f"{ids_name}.nc"
    with imas.DBEntry(path, "r", dd_version=DD_VERSION) as entry:
        return entry.get(ids_name, autoconvert=False)


def _relative(path: str, ids_name: str) -> str:
    prefix = f"{ids_name}/"
    assert path.startswith(prefix), path
    return path[len(prefix) :]


def _leaf_paths(receipt, ids_name: str) -> set[str]:
    paths: set[str] = set()
    for leaf in receipt.leaves:
        if leaf.ids != ids_name:
            continue
        paths.add(_relative(leaf.target_path, ids_name))
        paths.add(_relative(leaf.time_path, ids_name))
    return paths


def _is_named(diff_path: str, leaf_paths: set[str]) -> bool:
    """A diff path is named when it is a leaf or an ancestor of one.

    ``magnetics/ip`` is a struct array the description leaves empty; filling
    its first element names ``ip/data`` and sizes ``ip``, so the diff is
    reported at the array.  Either path is the same leaf.
    """

    return any(
        diff_path == leaf or leaf.startswith(f"{diff_path}/") for leaf in leaf_paths
    )


def _leaves_of(receipt, ids_name: str):
    return [leaf for leaf in receipt.leaves if leaf.ids == ids_name]


def _expected_homogeneous_time(receipt, ids_name: str) -> int:
    """The homogeneous_time the writer's own receipt implies for one IDS.

    1 when every leaf the receipt names for the IDS carries its time base at
    the IDS-level ``time`` path; 0 when the leaves carry their own time
    vectors; 2 when the IDS has no time-dependent leaf.
    """

    leaves = _leaves_of(receipt, ids_name)
    if not leaves:
        return 2
    if all(leaf.time_path == f"{ids_name}/time" for leaf in leaves):
        return 1
    return 0


def _nodes_at(ids, relative: str):
    """Every node a receipt path reaches, expanding struct arrays as it goes.

    A receipt path names the struct array, not the element (``flux_loop`` and
    not ``flux_loop[3]``), so one receipt leaf stands for every element the
    writer filled; each element's time vector is checked against the leaf.
    """

    nodes = [ids]
    for component in relative.split("/"):
        children = []
        for node in nodes:
            child = getattr(node, component)
            if isinstance(child, IDSStructArray):
                children.extend(child)
            else:
                children.append(child)
        nodes = children
    return nodes


def test_write_covers_every_description_ids(receipt):
    assert receipt.ids_written == IDS_NAMES
    assert receipt.phase == "OP1"
    # The expected value comes from the receipt, not a fixed set: an IDS whose
    # served leaves all share the IDS-level time path is homogeneous; one whose
    # leaves carry their own time vectors is not; one the map serves nothing
    # into is written whole from the description and keeps its unset convention.
    for ids_name in IDS_NAMES:
        assert ids_name in receipt.files
        read = _read(receipt, ids_name)
        expected = _expected_homogeneous_time(receipt, ids_name)
        assert read.ids_properties.homogeneous_time == expected, ids_name
        if expected != 0:
            continue
        for leaf in _leaves_of(receipt, ids_name):
            times = _nodes_at(read, _relative(leaf.time_path, ids_name))
            assert times, leaf.time_path
            for time in times:
                assert len(time) == leaf.samples, (ids_name, leaf.target_path)


def test_served_coil_currents_match_the_compiled_map(receipt):
    """The ten served coils carry the compiled map, sample for sample."""

    compiled = load_packaged_signal_map(MACHINE, "pf_active").compile(SHOT_INT)
    pf_active = _read(receipt, "pf_active")

    names = [str(coil.name) for coil in pf_active.coil]
    for index, coil_name in enumerate(PF_COILS):
        assert names[index] == coil_name, (index, names[index])
        rule = compiled[f"pf_active_coil_{coil_name}_current"].rule
        raw = read_channel(
            JT60SA_ROOT, SHOT_TOKEN, rule.source_group, rule.source_array
        )
        expected = compiled.apply(rule.semantic_id, raw.data[0])
        written = np.asarray(pf_active.coil[index].current.data)
        assert written.shape[-1] == expected.shape[-1]
        assert np.array_equal(written, expected), coil_name
        assert np.array_equal(np.asarray(pf_active.time), raw.time), coil_name


def test_fppc_coils_carry_no_current(receipt):
    pf_active = _read(receipt, "pf_active")
    for index in FPPC_COIL_INDICES:
        assert str(pf_active.coil[index].name) in ("FPPC_UP", "FPPC_DOWN")
        assert np.asarray(pf_active.coil[index].current.data).size == 0


def test_tf_current_equals_the_served_chain(receipt):
    compiled = load_packaged_signal_map(MACHINE, "tf").compile(SHOT_INT)
    rule = compiled["tf_coil_TF1_current"].rule
    raw = read_channel(JT60SA_ROOT, SHOT_TOKEN, rule.source_group, rule.source_array)
    expected = compiled.apply(rule.semantic_id, raw.data[0])
    tf = _read(receipt, "tf")
    written = np.asarray(tf.coil[0].current.data)
    assert np.array_equal(written, expected)
    assert np.array_equal(np.asarray(tf.time), raw.time)


def test_magnetics_ip_peaks_near_one_ma(receipt):
    magnetics = _read(receipt, "magnetics")
    ip = np.asarray(magnetics.ip[0].data)
    assert ip.size > 0
    assert 0.8e6 <= float(np.nanmax(ip)) <= 1.2e6


def test_static_content_is_the_description_unchanged(receipt):
    for ids_name in IDS_NAMES:
        written = _read(receipt, ids_name)
        original = _read_description(ids_name)
        leaf_paths = _leaf_paths(receipt, ids_name)
        for path in {entry[0] for entry in idsdiffgen(written, original)}:
            if path == "ids_properties" or path.startswith("ids_properties/"):
                continue
            assert _is_named(path, leaf_paths), (ids_name, path, sorted(leaf_paths))


def test_shot_without_cache_refuses_and_names_the_cache(tmp_path):
    with pytest.raises(PulseWriteError) as error:
        write_pulse(MACHINE, 999999, tmp_path)
    message = str(error.value)
    assert "999999" in message
    assert "no EDDB cache" in message
    assert str(JT60SA_ROOT / "999999.zarr") in message


def test_write_command_writes_the_pulse(tmp_path):
    from click.testing import CliRunner

    from imas_alambic.cli import main

    result = CliRunner().invoke(
        main,
        ["write", "--machine", MACHINE, "--shot", SHOT_TOKEN, "--out", str(tmp_path)],
    )
    assert result.exit_code == 0, result.output
    for ids_name in IDS_NAMES:
        assert (tmp_path / str(SHOT_INT) / f"{ids_name}.nc").is_file()
