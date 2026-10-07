"""Write a JT-60SA pulse: one run file holds every description IDS.

The writer reads the phase description whole from the machine-description store
and the dynamic values from the EDDB cache, and writes one DD 4.1.1 netCDF file
per pulse and run holding every description IDS.  This test writes E101154 into
a temporary directory and checks the done-when of the pulse-write step: the run
file holds every description IDS and reads back through :class:`imas.DBEntry`,
the ten coils the ``pf_active`` map serves carry the compiled map applied to
their cached channels sample for sample, both FPPC coils carry no current, the
TF coil current equals the served TF chain, ``magnetics/ip`` peaks near 1 MA,
and the written IDSs differ from the description in no path outside
``ids_properties`` other than the time-dependent leaves the receipt names.  A
shot with no cache refuses and names the missing cache, an existing run file is
refused without ``--overwrite``, and the ``write`` command names the run file
through :func:`pulse_path`.

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
from imas_alambic.machine_map import MachineMapError, bundle_for_machine
from imas_alambic.pulse_writer import (
    IDS_NAMES,
    PulseWriteError,
    pulse_path,
    write_pulse,
)
from imas_alambic.signal_map import load_packaged_signal_map
from imas_ambix.data.paths import JT60SA_ROOT

SHOT_TOKEN = "E101154"
SHOT_INT = 101154
DD_VERSION = "4.1.1"
MACHINE = "jt-60sa"

PF_COILS = ("CS1", "CS2", "CS3", "CS4", "EF1", "EF2", "EF3", "EF4", "EF5", "EF6")
FPPC_COIL_INDICES = (10, 11)

try:
    BUNDLE = bundle_for_machine(MACHINE)
except MachineMapError:
    BUNDLE = None
DESCRIPTION_ROOT = BUNDLE.store_roots["description"] / "OP1" if BUNDLE else None
_cache_missing = not (JT60SA_ROOT / f"{SHOT_INT}.zarr").is_dir()
_store_missing = (
    DESCRIPTION_ROOT is None or not (DESCRIPTION_ROOT / "pf_active.nc").is_file()
)
pytestmark = pytest.mark.skipif(
    BUNDLE is None or _cache_missing or _store_missing,
    reason="JT-60SA bundle, EDDB cache, or description store is unavailable; "
    "set IMAS_ALAMBIC_MAP_PATH",
)


@pytest.fixture(scope="module")
def written(tmp_path_factory):
    out = tmp_path_factory.mktemp("write")
    return write_pulse(MACHINE, SHOT_TOKEN, out)


def _read(written, ids_name: str):
    with imas.DBEntry(written.path, "r", dd_version=DD_VERSION) as entry:
        return entry.get(ids_name, autoconvert=False)


def _read_description(ids_name: str):
    path = DESCRIPTION_ROOT / f"{ids_name}.nc"
    with imas.DBEntry(path, "r", dd_version=DD_VERSION) as entry:
        return entry.get(ids_name, autoconvert=False)


def _relative(path: str, ids_name: str) -> str:
    prefix = f"{ids_name}/"
    assert path.startswith(prefix), path
    return path[len(prefix) :]


def _leaf_paths(written, ids_name: str) -> set[str]:
    paths: set[str] = set()
    for leaf in written.leaves:
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


def _leaves_of(written, ids_name: str):
    return [leaf for leaf in written.leaves if leaf.ids == ids_name]


def _expected_homogeneous_time(written, ids_name: str) -> int:
    """The homogeneous_time the writer's own receipt implies for one IDS.

    1 when every leaf the receipt names for the IDS carries its time base at
    the IDS-level ``ids/time`` path; 0 when the leaves carry their own time
    vectors; 2 when the IDS has no time-dependent leaf.
    """

    leaves = _leaves_of(written, ids_name)
    if not leaves:
        return 2
    if all(leaf.time_path == f"{ids_name}/time" for leaf in leaves):
        return 1
    return 0


def _nodes_at(ids, relative: str):
    """Every node a receipt path reaches, expanding struct arrays as it goes."""

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


def test_write_covers_every_description_ids_in_one_file(written):
    assert written.ids_written == IDS_NAMES
    assert written.phase == "OP1"
    assert written.path == str(pulse_path(written.out_dir, SHOT_INT, 0))
    assert written.path.endswith(f"{SHOT_INT}_0.nc")
    # The expected value comes from the receipt, not a fixed set: an IDS whose
    # served leaves all share the IDS-level time path is homogeneous; one whose
    # leaves carry their own time vectors is not; one the map serves nothing
    # into is written whole from the description and keeps its unset convention.
    for ids_name in IDS_NAMES:
        assert written.files[ids_name] == written.path
        read = _read(written, ids_name)
        expected = _expected_homogeneous_time(written, ids_name)
        assert read.ids_properties.homogeneous_time == expected, ids_name
        if expected != 0:
            continue
        for leaf in _leaves_of(written, ids_name):
            times = _nodes_at(read, _relative(leaf.time_path, ids_name))
            assert times, leaf.time_path
            for time in times:
                assert len(time) == leaf.samples, (ids_name, leaf.target_path)


def test_served_coil_currents_match_the_compiled_map(written):
    """The ten served coils carry the compiled map, sample for sample."""

    compiled = load_packaged_signal_map(MACHINE, "pf_active").compile(SHOT_INT)
    pf_active = _read(written, "pf_active")

    names = [str(coil.name) for coil in pf_active.coil]
    for index, coil_name in enumerate(PF_COILS):
        assert names[index] == coil_name, (index, names[index])
        rule = compiled[f"pf_active_coil_{coil_name}_current"].rule
        raw = read_channel(
            JT60SA_ROOT, SHOT_TOKEN, rule.source_group, rule.source_array
        )
        expected = compiled.apply(rule.semantic_id, raw.data[0])
        value = np.asarray(pf_active.coil[index].current.data)
        assert value.shape[-1] == expected.shape[-1]
        assert np.array_equal(value, expected), coil_name
        assert np.array_equal(np.asarray(pf_active.time), raw.time), coil_name


def test_fppc_coils_carry_no_current(written):
    pf_active = _read(written, "pf_active")
    for index in FPPC_COIL_INDICES:
        assert str(pf_active.coil[index].name) in ("FPPC_UP", "FPPC_DOWN")
        assert np.asarray(pf_active.coil[index].current.data).size == 0


def test_tf_current_equals_the_served_chain(written):
    compiled = load_packaged_signal_map(MACHINE, "tf").compile(SHOT_INT)
    rule = compiled["tf_coil_TF1_current"].rule
    raw = read_channel(JT60SA_ROOT, SHOT_TOKEN, rule.source_group, rule.source_array)
    expected = compiled.apply(rule.semantic_id, raw.data[0])
    tf = _read(written, "tf")
    value = np.asarray(tf.coil[0].current.data)
    assert np.array_equal(value, expected)
    assert np.array_equal(np.asarray(tf.time), raw.time)


def test_magnetics_ip_peaks_near_one_ma(written):
    magnetics = _read(written, "magnetics")
    ip = np.asarray(magnetics.ip[0].data)
    assert ip.size > 0
    assert 0.8e6 <= float(np.nanmax(ip)) <= 1.2e6


def test_static_content_is_the_description_unchanged(written):
    for ids_name in IDS_NAMES:
        produced = _read(written, ids_name)
        original = _read_description(ids_name)
        leaf_paths = _leaf_paths(written, ids_name)
        for path in {entry[0] for entry in idsdiffgen(produced, original)}:
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


def test_an_existing_run_file_is_refused_without_overwrite(tmp_path):
    first = write_pulse(MACHINE, SHOT_TOKEN, tmp_path)
    with pytest.raises(PulseWriteError) as error:
        write_pulse(MACHINE, SHOT_TOKEN, tmp_path)
    assert "already exists" in str(error.value)
    assert "--overwrite" in str(error.value)
    replaced = write_pulse(MACHINE, SHOT_TOKEN, tmp_path, overwrite=True)
    assert replaced.path == first.path


def test_write_command_writes_the_run_file(tmp_path):
    from click.testing import CliRunner

    from imas_alambic.cli import main

    result = CliRunner().invoke(
        main, ["write", SHOT_TOKEN, "--machine", MACHINE, "--out", str(tmp_path)]
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / f"{SHOT_INT}_0.nc").is_file()
