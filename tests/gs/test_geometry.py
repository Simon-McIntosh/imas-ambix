"""Tests for the GS machine-geometry extractor.

Two layers:

* **Pure-logic** tests build synthetic efm geometry dicts so they run anywhere
  (no mirror, no network) — they pin the orientation-resolution, signature, and
  mapping invariants that the locked ``gs-geometry-source`` decision depends on.
* **Integration** tests read a real level-1 shot from the mirror and are
  skipped when the mirror is absent (CI).
"""

from __future__ import annotations

import numpy as np
import pytest

from imas_ambix.data.paths import local_shot_path
from imas_ambix.gs import geometry as gsg

# --- fixtures: synthetic efm static geometry --------------------------


def _synthetic_geom(*, n_pf: int = 8, magpr_z_shift: float = 0.0) -> dict:
    """A minimal but realistic efm static-geometry dict.

    Builds a co-located radial/vertical B-probe pair at one (R, Z) — the exact
    degeneracy that plain nearest-neighbour mishandles — plus a clean flux loop
    and a small PF/limiter set.  ``magpr_z_shift`` simulates the ~1 cm
    per-campaign drift.
    """
    # 4 B-probes: 2 vertical (ang=90) + 2 radial (ang=0); one v/r pair colocated
    magpr_r = np.array([0.18, 1.85, 1.85, 1.44])
    magpr_z = np.array([1.0, 0.3, 0.3, -1.2]) + magpr_z_shift
    magpr_ang = np.array([90.0, 90.0, 0.0, 0.0])  # idx1=vertical, idx2=radial colocated
    magpr_len = np.array([0.025, 0.025, 0.025, 0.025])
    silop_r = np.array([0.178, 1.163])
    silop_z = np.array([1.235, 1.083])
    fcoil_r = np.linspace(0.12, 1.9, n_pf)
    fcoil_z = np.linspace(-1.5, 1.5, n_pf)
    fcoil_turns = np.ones(n_pf)
    fcoil_width = np.full(n_pf, 0.01)
    fcoil_height = np.full(n_pf, 0.02)
    fcoil_circ = np.arange(1, n_pf + 1, dtype=float)
    fcoil_xmult = np.full(n_pf, 0.5)
    limiterr = np.array([1.9, 1.55, 1.40])
    limiterz = np.array([0.4, 0.4, 0.82])
    return {
        "magpr_r": magpr_r,
        "magpr_z": magpr_z,
        "magpr_ang": magpr_ang,
        "magpr_len": magpr_len,
        "silop_r": silop_r,
        "silop_z": silop_z,
        "fcoil_r": fcoil_r,
        "fcoil_z": fcoil_z,
        "fcoil_turns": fcoil_turns,
        "fcoil_width": fcoil_width,
        "fcoil_height": fcoil_height,
        "fcoil_circ": fcoil_circ,
        "fcoil_xmult": fcoil_xmult,
        "limiterr": limiterr,
        "limiterz": limiterz,
    }


# --- read-list audit: the strict boundary -----------------------------


def test_read_list_is_geometry_only():
    """The hardcoded read-list must contain ONLY static geometry arrays."""
    forbidden = {
        "psirz",
        "pprime",
        "ffprime",
        "lcfs_r",
        "lcfs_z",
        "qpsi_c",
        "plasma_current_c",
        "plasma_current_x",
        "plasma_current_rz",
        "magnetic_axis_r",
        "magnetic_axis_z",
        "betap",
        "li",
        "magpr_c",
        "magpr_x",
        "silop_c",
        "silop_x",
        "fcoil_c",
        "fcoil_x",
    }
    assert forbidden.isdisjoint(set(gsg.EFM_GEOMETRY_ARRAYS))
    # every read-list entry is one of the known static prefixes
    for name in gsg.EFM_GEOMETRY_ARRAYS:
        assert name.startswith(("magpr_", "silop_", "fcoil_", "limiter"))
    # the fitted/experimental current neighbours must NOT be present
    assert "magpr_c" not in gsg.EFM_GEOMETRY_ARRAYS
    assert "magpr_x" not in gsg.EFM_GEOMETRY_ARRAYS


# --- orientation resolution: the crux of the locked decision ----------


def test_orientation_from_magpr_ang_resolves_colocated_pair():
    """Co-located radial/vertical probes must each get the CORRECT orientation.

    Plain NN on (R, Z) is degenerate here; the ang-constrained mapping must
    send the vertical channel to the ang=90 magpr and the radial to ang=0.
    """
    geom = _synthetic_geom()
    # idx1 (ang=90) and idx2 (ang=0) are co-located at (1.85, 0.3)
    amb = [
        ("obv06", "Outer coil r=1.850, z=0.300"),  # name says vertical
        ("obr06", "Outer coil r=1.850, z=0.300"),  # name says radial
    ]
    mappings, unmatched = gsg.map_amb_sensors(geom, amb)
    by_ch = {m.amb_channel: m for m in mappings}
    assert by_ch["obv06"].angle_deg == 90.0
    assert by_ch["obr06"].angle_deg == 0.0
    # they must map to DIFFERENT efm indices (no collision)
    assert by_ch["obv06"].efm_index != by_ch["obr06"].efm_index
    assert not by_ch["obv06"].flag
    assert not by_ch["obr06"].flag


def test_name_says_radial_but_ang_says_vertical_is_authoritative():
    """If a probe NAME implies vertical but efm ang is the only truth, ang wins.

    Mirrors the real amb obv* copy-paste bug ("Br" in description, ang=90).
    Here we feed a ccbv-named channel whose ONLY vertical candidate is ang=90;
    the stored angle must come from efm, never the name/description.
    """
    geom = _synthetic_geom()
    amb = [("ccbv01", "Centre Column Vertical Bv r=0.180, z=1.000")]
    mappings, _ = gsg.map_amb_sensors(geom, amb)
    assert mappings[0].angle_deg == 90.0  # from magpr_ang, not parsed from name
    # the stored R,Z come from efm (0.18), not the amb-desc rounding (0.180 here)
    assert mappings[0].r == pytest.approx(0.18)


def test_amb_description_rz_is_lookup_key_only():
    """The stored (R, Z) must equal efm's, even when amb-desc is offset."""
    geom = _synthetic_geom()
    # amb desc says 0.186 (rounded) but efm magpr_r[0] is 0.18
    amb = [("ccbv01", "Bv r=0.186, z=1.004")]
    mappings, _ = gsg.map_amb_sensors(geom, amb)
    assert mappings[0].r == pytest.approx(0.18)  # efm value, not 0.186
    assert mappings[0].residual_m > 0  # there WAS a small offset
    assert mappings[0].residual_m < gsg._MAX_RESIDUAL_M


def test_flux_loop_collision_is_flagged_not_forced():
    """Two amb FL channels claiming one silop index must both be flagged."""
    geom = _synthetic_geom()
    amb = [
        ("fl_cc01", "Flux Loop r=0.178, z=1.235"),
        ("fl_cc02", "Flux Loop r=0.178, z=1.235"),  # duplicate description
    ]
    mappings, _ = gsg.map_amb_sensors(geom, amb)
    assert len(mappings) == 2
    assert all("non-unique" in m.flag for m in mappings)
    assert all(m.angle_deg is None for m in mappings)  # flux loops carry no angle


def test_unparseable_or_far_channel_is_unmatched():
    """A channel with no parseable R,Z, or one far from any sensor, is unmatched."""
    geom = _synthetic_geom()
    amb = [
        ("ccbv99", "no coordinates here"),
        ("obr99", "r=99.0, z=99.0"),  # nowhere near any probe
    ]
    mappings, unmatched = gsg.map_amb_sensors(geom, amb)
    assert "ccbv99" in unmatched
    assert "obr99" in unmatched
    assert mappings == []


def test_time_columns_are_excluded_not_unmatched():
    """time / timesec / status are not sensors and must be silently dropped."""
    geom = _synthetic_geom()
    amb = [
        ("time", ""),
        ("timesec", ""),
        ("status", ""),
        ("ccbv01", "Bv r=0.18, z=1.0"),
    ]
    mappings, unmatched = gsg.map_amb_sensors(geom, amb)
    assert unmatched == []  # none of the time columns leak into unmatched
    assert [m.amb_channel for m in mappings] == ["ccbv01"]


# --- per-campaign signature -------------------------------------------


def test_signature_differs_on_pf_count():
    """A different fcoil discretisation must yield a different signature."""
    g1 = _synthetic_geom(n_pf=8)
    g2 = _synthetic_geom(n_pf=10)
    s1, s2 = gsg.setup_signature(g1), gsg.setup_signature(g2)
    assert s1.n_pf_filament != s2.n_pf_filament
    assert s1.key != s2.key


def test_signature_differs_on_position_drift():
    """A ~1 cm magpr_z drift (same counts) must yield a different signature."""
    g1 = _synthetic_geom(magpr_z_shift=0.0)
    g2 = _synthetic_geom(magpr_z_shift=0.012)  # 12 mm, the observed drift
    s1, s2 = gsg.setup_signature(g1), gsg.setup_signature(g2)
    assert s1.n_pf_filament == s2.n_pf_filament  # counts identical
    assert s1.digest != s2.digest  # but positions differ → different key
    assert s1.key != s2.key


def test_signature_stable_under_repeat():
    """Identical geometry must hash to the identical signature."""
    g = _synthetic_geom()
    assert gsg.setup_signature(g).key == gsg.setup_signature(g).key


def test_signature_ignores_nan_padded_silop():
    """silop padded with trailing NaN must not change the valid-count or digest."""
    g1 = _synthetic_geom()
    g2 = _synthetic_geom()
    g2["silop_r"] = np.concatenate([g2["silop_r"], [np.nan, np.nan]])
    g2["silop_z"] = np.concatenate([g2["silop_z"], [np.nan, np.nan]])
    s1, s2 = gsg.setup_signature(g1), gsg.setup_signature(g2)
    assert s1.n_fluxloop == s2.n_fluxloop == 2
    assert s1.digest == s2.digest


# --- amm passive R,Z parsing ------------------------------------------


def test_amm_rz_parser():
    """The amm 'R=..  Z=..' description form must parse (distinct from amb)."""
    matches = gsg._AMM_RZ_RE.findall("Centre column inconnel: R=0.18  Z=1.80")
    assert matches == [("0.18", "1.80")]
    # multi-line description → one (R,Z) per line
    multi = gsg._AMM_RZ_RE.findall("Wall a: R=0.29  Z=-2.21\nWall b: R=0.30  Z=2.30")
    assert len(multi) == 2


# --- integration: a real mirror shot (skipped in CI) ------------------

_REP_SHOT = 11766
_HAVE_MIRROR = local_shot_path(_REP_SHOT, tier="level1").exists()
_skip_no_mirror = pytest.mark.skipif(
    not _HAVE_MIRROR, reason="level-1 mirror not available (CI)"
)


@_skip_no_mirror
def test_read_efm_geometry_only_reads_geometry_arrays():
    geom = gsg.read_efm_geometry(_REP_SHOT)
    # every key read is in the auditable read-list — nothing else
    assert set(geom).issubset(set(gsg.EFM_GEOMETRY_ARRAYS))
    assert "magpr_r" in geom and "fcoil_r" in geom and "limiterr" in geom


@_skip_no_mirror
def test_build_table_for_real_shot_maps_all_bprobes_cleanly():
    table = gsg.build_table_for_shot(_REP_SHOT)
    cov = table.coverage()
    assert cov["n_bprobe"] == 78
    assert cov["n_fluxloop"] == 46
    assert cov["n_limiter"] == 37
    # every mapped B-probe agrees name-orientation ↔ stored magpr_ang
    for m in table.sensor_map:
        if m.kind == "b_probe":
            exp = gsg._expected_angle(m.amb_channel)
            if exp is not None:
                assert m.angle_deg == exp, f"{m.amb_channel}: {m.angle_deg} != {exp}"
    # all flagged channels are flux loops (the known amb-desc data-quality issue)
    for m in table.sensor_map:
        if m.flag:
            assert m.kind == "flux_loop"


@_skip_no_mirror
def test_real_shot_signature_is_one_of_known_campaigns():
    table = gsg.build_table_for_shot(_REP_SHOT)
    # 11766 is in the earliest fc1004 campaign
    assert table.signature.n_pf_filament in (938, 1004)
    assert table.signature.n_bprobe == 78
    assert table.signature.n_fluxloop == 46


# --- reader-interface refactor: MAST keys must stay byte-identical -----


@_skip_no_mirror
def test_mast_zarr_reader_is_a_byte_identical_adapter():
    """MastZarrGeometryReader must reproduce build_table_for_shot exactly.

    The reader-interface refactor (machine-tagged SetupSignature) MUST NOT
    change a single MAST key: existing g_pg caches and trained checkpoints
    resolve through it.
    """
    direct = gsg.build_table_for_shot(_REP_SHOT)
    via_reader = gsg.MastZarrGeometryReader(shot_id=_REP_SHOT).read()
    assert via_reader.signature.key == direct.signature.key
    assert via_reader.signature.key == direct.to_dict()["signature_key"]
    assert not via_reader.signature.key.startswith("mast-")
    assert via_reader.signature.machine == "mast"
    assert isinstance(via_reader, type(direct))
