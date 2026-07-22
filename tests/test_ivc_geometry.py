"""Tests for the era-keyed non-axisymmetric coil geometry and toroidal sensors.

Covers :mod:`imas_ambix.gs.ivc_geometry` (era-dependent ELM/RMP complement, the
EFCC picture-frame pairs, loop topology) and :mod:`imas_ambix.gs.sensor_toroidal`
(the L2 toroidal-position reader, guarded by data availability).  The physics
pins reuse the validated filament kernels: a complete toroidal loop must reject
the n != 0 EFCC field to (near) machine precision — the geometry counterpart of
the measured flux-loop immunity.
"""

from __future__ import annotations

import numpy as np
import pytest

from imas_ambix.gs import filaments3d as f3d
from imas_ambix.gs import ivc_geometry as ivc

# ------------------------------------------------------------- era keying


def test_elm_complement_is_era_keyed():
    """0 in-vessel coils before install, 12 in the lower era, 18 with the upper row."""
    assert ivc.elm_coils_for_era(ivc.ELM_LOWER_FIRST_SHOT - 1) == ()
    assert len(ivc.elm_coils_for_era(ivc.ELM_LOWER_FIRST_SHOT)) == 12
    assert len(ivc.elm_coils_for_era(ivc.ELM_UPPER_FIRST_SHOT - 1)) == 12
    assert len(ivc.elm_coils_for_era(ivc.ELM_UPPER_FIRST_SHOT)) == 18
    assert len(ivc.elm_coils_for_era(ivc.ELM_UPPER_FIRST_SHOT + 5000)) == 18


def test_upper_row_is_odd_sectors_only():
    """The 6 upper coils sit on odd sectors; the 12 lower cover every sector."""
    coils = ivc.elm_coils_for_era(ivc.ELM_UPPER_FIRST_SHOT)
    upper = [c for c in coils if "_u_" in c.name]
    lower = [c for c in coils if "_l_" in c.name]
    assert len(upper) == 6
    assert len(lower) == 12
    # upper coils sit above the midplane, lower below
    assert all(c.z_lo > 0 for c in upper)
    assert all(c.z_hi < 0 for c in lower)


def test_coil_set_summary():
    cs = ivc.coil_set_for_shot(ivc.ELM_UPPER_FIRST_SHOT + 1)
    s = cs.summary()
    assert s["n_efcc_pairs"] == 2
    assert s["n_efcc_coils"] == 4
    assert s["n_elm_coils"] == 18
    assert "18 in-vessel" in cs.era


# ------------------------------------------------------------- EFCC geometry


def test_efcc_pairs_shape():
    pairs = ivc.efcc_pairs()
    assert set(pairs) == {"error_field_02", "error_field_05"}
    for p in pairs.values():
        built = p.build()
        assert len(built) == 2  # two coils per pair
        signs = [s for _, s in built]
        assert signs[0] == -signs[1]  # opposite in series
        for poly, _ in built:
            # closed picture frame
            assert np.allclose(poly[0], poly[-1])


def test_efcc_zhalf_override_moves_the_frame():
    lo = ivc.efcc_pairs(z_half=0.8)["error_field_02"].coils[0]
    hi = ivc.efcc_pairs(z_half=2.0)["error_field_02"].coils[0]
    assert hi.z_hi > lo.z_hi
    assert lo.z_hi == pytest.approx(0.8)
    assert hi.z_hi == pytest.approx(2.0)


def test_full_loop_rejects_efcc_field():
    """A complete toroidal loop links ~0 net flux from an n=1 EFCC pair.

    The reference loop is offset from the midplane (z=0.5) so a single coil links
    an appreciable, unambiguous flux — a midplane loop links ~0 from a single
    coil by up-down symmetry, which makes the ratio a 0/0 non-test.  The
    opposite-in-series pair (n=1) then cancels that linkage to the numeric floor.
    """
    pair = ivc.efcc_pairs()["error_field_02"].build()
    loop = f3d.circle(radius=1.5, z=0.5, n=720)
    current = 15e3  # A-turn
    phi_pair = sum(
        f3d.flux_through_loop(poly, sgn * current, loop) for poly, sgn in pair
    )
    phi_single = abs(f3d.flux_through_loop(pair[0][0], current, loop))
    assert phi_single > 1e-6  # single coil links real flux (well-posed reference)
    # the paired (opposite-series) field cancels around the full loop
    assert abs(phi_pair) < 1e-9 * phi_single


# ------------------------------------------------------------- loop topology


def test_loop_topology_classes():
    assert ivc.loop_topology("fl_p5l_1") == "full_loop"
    assert ivc.loop_topology("sad_out_03") == "partial_loop"
    assert ivc.loop_topology("obv09") == "probe"
    assert ivc.is_immune_to_energised_field("fl_cc01") is True
    assert ivc.is_immune_to_energised_field("sad_out_01") is False
    assert ivc.is_immune_to_energised_field("obr06") is False


# ------------------------------------------------------------- toroidal reader
#
# Data-availability guarded: only runs where an L2 shot is present.


def _first_available_l2_shot():
    from imas_ambix.data.paths import LEVEL2_DIR

    if not LEVEL2_DIR.is_dir():
        return None
    for p in sorted(LEVEL2_DIR.glob("*.zarr")):
        try:
            return int(p.stem)
        except ValueError:
            continue
    return None


def test_toroidal_reader_returns_probe_banks():
    from imas_ambix.gs import sensor_toroidal as st

    shot = _first_available_l2_shot()
    if shot is None:
        pytest.skip("no L2 magnetics data available")
    geo = st.read_toroidal_geometry(shot)
    # obv/obr families should be present with the two toroidal banks (150/330)
    for fam in ("obv", "obr"):
        if fam not in geo.probes:
            pytest.skip(f"family {fam} absent in this L2 shot")
        p = geo.probes[fam]
        assert p.r.size == p.z.size == len(p.geometry_channels)
        assert any(140 <= b <= 160 for b in p.banks_deg)
        assert any(320 <= b <= 340 for b in p.banks_deg)
        # a named channel resolves to a row
        idx = p.channel_index(p.geometry_channels[0])
        assert idx == 0
