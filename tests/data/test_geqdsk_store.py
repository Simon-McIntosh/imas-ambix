"""Bind E101011's G-EQDSK flux into the psi-shaped equilibrium target.

The G-EQDSK is a 2-D poloidal-flux grid, so it cannot be fetched as an EDDB
channel.  :func:`imas_ambix.data.geqdsk_store.write_geqdsk_store` stages it into
a per-shot Zarr group laid out as the cache writer lays a fetched channel, and
the packaged equilibrium map binds that array to
``equilibrium/time_slice/profiles_2d/psi`` with ``source_cocos`` 1.

The store holds the file's own COCOS 1 (Wb/rad) flux, so the COCOS-17 psi the
engine mints carries the rule's ``psi_like`` convention factor -2 pi and agrees
with the G-EQDSK psi times -2 pi on the file's own (Z, R) grid.  Flip the rule's
``source_cocos`` to 17 and the factor becomes one, so the missing -2 pi leaves
the psi an order of magnitude away.  The store is built under ``tmp_path``; the
G-EQDSK under ``/work`` is read and never written.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import zarr

from imas_alambic.eddb import time_array_name
from imas_alambic.signal_map import MAP_SCHEMA_VERSION, SignalMap, load_packaged_signal_map
from imas_alambic.virtual_zarr import VirtualZarrView
from imas_ambix.challenge.loader import load_geqdsk
from imas_ambix.data.geqdsk_store import (
    GEQDSK_GROUP,
    PSI_ARRAY,
    R_ARRAY,
    Z_ARRAY,
    default_geqdsk_path,
    read_geqdsk_grid,
    write_geqdsk_store,
)
from imas_ambix.data.paths import JT60SA_ROOT
from tests.jt60sa_bundle import BUNDLE, SKIP_REASON

pytestmark = pytest.mark.skipif(
    BUNDLE is None or not default_geqdsk_path("E101011").is_file(),
    reason="JT-60SA G-EQDSK or map bundle is unavailable; set IMAS_ALAMBIC_MAP_PATH",
)

SHOT_TOKEN = "E101011"
SHOT_INT = 101011
MACHINE = "jt-60sa"
PSI_SEMANTIC_ID = "equilibrium_psi"

#: The mount the read-only G-EQDSK lives under; nothing here is written.
GEQDSK_PATH = JT60SA_ROOT / "101011.geqdsk"


@pytest.fixture(scope="module")
def staged(tmp_path_factory):
    root = tmp_path_factory.mktemp("geqdsk-store")
    write_geqdsk_store(root, SHOT_TOKEN)
    return root


def _labels():
    return load_geqdsk(GEQDSK_PATH)


def _engine_psi(root, source_map):
    view = VirtualZarrView.open(
        str(root / f"{SHOT_INT}.zarr"), source_map, shot=SHOT_INT
    )
    return np.asarray(view[PSI_SEMANTIC_ID][:], dtype=np.float64)


def _normalised_rms(served: np.ndarray, expected: np.ndarray) -> float:
    """RMS difference between two flux grids, as a fraction of the expected range."""

    difference = np.sqrt(np.mean((served - expected) ** 2))
    return float(difference / np.ptp(expected))


def test_store_writes_the_flux_and_grid_under_the_cache_layout(staged):
    store = zarr.open_group(str(staged / f"{SHOT_INT}.zarr"), mode="r")
    assert GEQDSK_GROUP in store
    group = store[GEQDSK_GROUP]
    assert PSI_ARRAY in group
    assert R_ARRAY in group and Z_ARRAY in group

    labels = _labels()
    time = np.asarray(group[time_array_name(PSI_ARRAY)][...], dtype=np.float64)
    assert time.shape == (1,)
    assert time[0] == pytest.approx(float(labels.time_ms[0]) / 1000.0)

    radial, vertical = read_geqdsk_grid(staged, SHOT_TOKEN)
    assert radial == pytest.approx(labels.grid_r_m)
    assert vertical == pytest.approx(labels.grid_z_m)


def test_engine_psi_matches_the_geqdsk_flux_times_minus_two_pi(staged):
    labels = _labels()
    source_map = load_packaged_signal_map(MACHINE, "equilibrium")
    rule = next(
        row for row in source_map.signals if row.semantic_id == PSI_SEMANTIC_ID
    )
    assert rule.source_cocos == 1
    assert rule.transformation == "psi_like"
    assert rule.convention_factor == pytest.approx(-2.0 * np.pi)

    expected = np.asarray(labels.psirz[0], dtype=np.float64) * (-2.0 * np.pi)
    served = _engine_psi(staged, source_map).reshape(expected.shape)

    assert _normalised_rms(served, expected) < 0.01


def test_flipping_the_rule_sign_exceeds_the_tolerance(staged):
    """The negative control: source_cocos 17 drops the -2 pi and leaves the psi."""

    labels = _labels()
    source_map = load_packaged_signal_map(MACHINE, "equilibrium")
    rule = next(
        row for row in source_map.signals if row.semantic_id == PSI_SEMANTIC_ID
    )
    flipped_map = SignalMap.create(
        schema_version=MAP_SCHEMA_VERSION,
        set_version=source_map.set_version,
        machine=source_map.machine,
        system=source_map.system,
        source_dataset=source_map.source_dataset,
        target_dd_version=source_map.target_dd_version,
        target_cocos=source_map.target_cocos,
        discovery_producer=source_map.discovery_producer,
        discovery_receipt=source_map.discovery_receipt,
        signals=(replace(rule, source_cocos=17),),
    )

    raw = np.asarray(labels.psirz[0], dtype=np.float64)
    expected = raw * (-2.0 * np.pi)
    served = _engine_psi(staged, flipped_map).reshape(expected.shape)

    assert _normalised_rms(served, expected) > 0.01