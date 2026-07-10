"""Tests for the GS Green's-function forward operator (T2).

Three layers:

* **Physics** tests pin the Green's-function correctness absolutely — the
  on-axis field against the textbook circular-loop formula, the
  ``ellipk``/``ellipe`` ``m = k²`` parameter convention, the orientation
  projection (Bz at 90°, Br at 0°), and a finite-difference ``∂ψ`` ↔ ``B``
  consistency check.  No mirror / network needed.
* **Operator-assembly** tests build a :class:`ForwardOperator` from a synthetic
  geometry table and pin the three-block structure, the KNOWN-PF assembly from
  raw amc (units + xmult split), the excluded/flagged channel handling, and the
  vacuum round-trip.
* **Integration** tests build operators from the real per-campaign tables and
  are skipped when the mirror is absent (CI).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import ellipk

from imas_ambix.data.paths import MANIFEST_DIR
from imas_ambix.gs import geometry as gsg
from imas_ambix.gs import operator as op

# --- physics: Green's-function correctness ----------------------------


def test_on_axis_field_matches_textbook():
    """On-axis Bz of a unit loop must equal the textbook closed form.

    Bz(R→0) = μ0 a² / (2 (a² + Δz²)^{3/2}); Br→0.  This validates the operator
    absolutely (no EFIT reference) and pins the SI scale (μ0 in, current in A).
    """
    a, z0, zobs = 0.5, 0.0, 0.3
    r = np.array([1e-8])
    z = np.array([zobs])
    bz, br = op.greens_bz_br(r, z, a, z0)
    textbook = op.MU0 * a**2 / (2.0 * (a**2 + (zobs - z0) ** 2) ** 1.5)
    assert bz[0] == pytest.approx(textbook, rel=1e-6)
    assert abs(br[0]) < 1e-10


def test_ellipk_parameter_convention():
    """scipy ellipk takes m = k² (parameter), not k (modulus) — pin it.

    ellipk(0) == π/2 exactly; a slip to modulus would still give π/2 at 0 but
    diverge elsewhere, so we also check a known value: ellipk(0.5) ≈ 1.8540747.
    """
    assert ellipk(0.0) == pytest.approx(np.pi / 2)
    assert ellipk(0.5) == pytest.approx(1.8540746773, rel=1e-8)


def test_orientation_projection_picks_component():
    """B-probe projection: 90° reads Bz, 0° reads Br, oblique mixes."""
    bz = np.array([2.0, 2.0, 2.0])
    br = np.array([3.0, 3.0, 3.0])
    ang = np.array([90.0, 0.0, 45.0])
    proj = op._project_bprobe(bz, br, ang)
    assert proj[0] == pytest.approx(2.0)  # vertical → Bz
    assert proj[1] == pytest.approx(3.0)  # radial → Br
    assert proj[2] == pytest.approx((3.0 + 2.0) / np.sqrt(2.0))  # 45° mix


def test_psi_and_b_finite_difference_consistency():
    """B must be the curl of the poloidal flux: Bz = (1/R) ∂ψ/∂R, BR = -(1/R) ∂ψ/∂Z.

    A finite-difference of ψ at an off-axis point must match the analytic B to
    a few × the step error — a self-contained correctness check that ties the
    flux and field Green's functions together (no EFIT reference).
    """
    a, z0 = 0.6, 0.1
    r0, zc = 1.0, 0.4
    h = 1e-5

    def psi(rr: float, zz: float) -> float:
        return float(op.greens_psi(np.array([rr]), np.array([zz]), a, z0)[0])

    dpsi_dr = (psi(r0 + h, zc) - psi(r0 - h, zc)) / (2 * h)
    dpsi_dz = (psi(r0, zc + h) - psi(r0, zc - h)) / (2 * h)
    bz_fd = dpsi_dr / (2 * np.pi * r0)
    br_fd = -dpsi_dz / (2 * np.pi * r0)
    bz, br = op.greens_bz_br(np.array([r0]), np.array([zc]), a, z0)
    assert bz[0] == pytest.approx(bz_fd, rel=1e-3)
    assert br[0] == pytest.approx(br_fd, rel=1e-3)


def test_field_falls_off_with_distance():
    """A unit loop's |B| must decrease monotonically moving away from it."""
    a, z0 = 0.5, 0.0
    zs = np.array([0.2, 0.5, 1.0, 2.0])
    bz, _ = op.greens_bz_br(np.full(zs.shape, 1e-8), zs, a, z0)
    assert np.all(np.diff(np.abs(bz)) < 0)


# --- synthetic geometry table for operator assembly -------------------


def _synthetic_table() -> gsg.GeometryTable:
    """A minimal campaign table: 1 vertical + 1 radial probe, 1 flux loop,
    one KNOWN PF coil (P4U-like centroid) + one passive structural circuit,
    one unmatched + one flagged amb channel."""
    # sensors
    bp_v = gsg.BProbe(index=0, r=1.5, z=0.0, angle_deg=90.0, length=0.025)
    bp_r = gsg.BProbe(index=1, r=1.5, z=0.0, angle_deg=0.0, length=0.025)
    fl = gsg.FluxLoop(index=0, r=1.3, z=0.5)
    # KNOWN PF coil — filaments near the P4U centroid (1.50, 1.10), Σxmult=1
    pf_known = [
        gsg.PFFilament(
            r=1.50, z=1.10, turns=1.0, width=0.01, height=0.01, circuit=1, xmult=0.5
        ),
        gsg.PFFilament(
            r=1.50, z=1.10, turns=1.0, width=0.01, height=0.01, circuit=1, xmult=0.5
        ),
    ]
    # passive structural circuit — far from any coil centroid → INFERRED
    pf_passive = [
        gsg.PFFilament(
            r=2.0, z=0.0, turns=1.0, width=0.01, height=0.01, circuit=2, xmult=1.0
        ),
    ]
    sig = gsg.SetupSignature(
        n_bprobe=2,
        n_fluxloop=1,
        n_pf_filament=3,
        n_limiter=4,
        digest="deadbeef00000000",
    )
    sensor_map = [
        gsg.SensorMapping("obv01", "b_probe", 0, 1.5, 0.0, 90.0, 0.001, ""),
        gsg.SensorMapping("obr01", "b_probe", 1, 1.5, 0.0, 0.0, 0.001, ""),
        gsg.SensorMapping("fl_p4u_1", "flux_loop", 0, 1.3, 0.5, None, 0.001, ""),
        gsg.SensorMapping(
            "fl_cc01",
            "flux_loop",
            0,
            1.3,
            0.5,
            None,
            0.02,
            "non-unique: silop[0] claimed by ['fl_cc01','fl_cc02']",
        ),
    ]
    return gsg.GeometryTable(
        signature=sig,
        shots=[12345],
        b_probes=[bp_v, bp_r],
        flux_loops=[fl],
        pf_filaments=pf_known + pf_passive,
        limiter_r=[0.3, 1.6, 1.6, 0.3],
        limiter_z=[-1.0, -1.0, 1.0, 1.0],
        sensor_map=sensor_map,
        passive_structures=[
            gsg.PassiveStructure(name="wall_a", r=2.0, z=0.0, obsolete=False)
        ],
        amc_current_channels=["p4u_coil_current", "p4u_current", "plasma_current"],
        unmatched_amb=["fl_p2u_1"],
    )


# --- circuit classification (verify-and-flag) -------------------------


def test_classify_circuits_splits_known_pf_from_inferred_passive():
    table = _synthetic_table()
    classes = op.classify_circuits(table.pf_filaments, table.amc_current_channels)
    by_circ = {c.circuit: c for c in classes}
    # circuit 1 sits at the P4U centroid → KNOWN, mapped to a real amc channel
    assert by_circ[1].role == "known_pf"
    assert by_circ[1].coil_label == "p4u"
    assert by_circ[1].amc_channel == "p4u_coil_current"
    # circuit 2 is far from any coil → INFERRED passive, no amc channel
    assert by_circ[2].role == "inferred_passive"
    assert by_circ[2].amc_channel == ""


def test_unknown_coil_without_amc_channel_is_inferred_not_guessed():
    """A circuit at a coil centroid whose amc channel is ABSENT must be flagged
    + INFERRED, never force-mapped to a fabricated channel."""
    table = _synthetic_table()
    # strip the P4U amc channels → the geometric match has no signal source
    classes = op.classify_circuits(table.pf_filaments, ["plasma_current"])
    by_circ = {c.circuit: c for c in classes}
    assert by_circ[1].role == "inferred_passive"
    assert by_circ[1].amc_channel == ""
    assert "no amc channel present" in by_circ[1].flag


# --- operator assembly + shapes ---------------------------------------


def test_build_operator_three_blocks_and_rows():
    table = _synthetic_table()
    operator = op.build_operator(table)
    n_sensor = len(operator.sensor_channels)
    # 2 B-probes + 2 flux-loop rows (one clean fl_p4u_1, one flagged fl_cc01)
    assert n_sensor == 4
    assert operator.g_pf.shape == (n_sensor, 1)  # one KNOWN PF circuit
    assert operator.g_passive.shape[0] == n_sensor
    assert operator.g_passive.shape[1] == 1  # one passive circuit
    assert operator.g_plasma.shape[0] == n_sensor
    assert operator.g_plasma.shape[1] > 0  # limiter-masked plasma nodes
    assert operator.pf_circuits == [1]
    assert operator.pf_amc_channels == ["p4u_coil_current"]


def test_excluded_and_flagged_channels_handled():
    table = _synthetic_table()
    operator = op.build_operator(table)
    # the T1-unmatched fl_p2u_1 is EXCLUDED from the prediction rows
    assert "fl_p2u_1" not in operator.sensor_channels
    assert operator.excluded_channels == ["fl_p2u_1"]
    # the non-unique fl_cc01 IS a predicted row but listed as flagged
    assert "fl_cc01" in operator.sensor_channels
    assert "fl_cc01" in operator.flagged_channels


def test_assemble_pf_currents_units_and_xmult():
    """KNOWN PF current = amc[kA·turn] × 1000; xmult split folded into G_pf."""
    table = _synthetic_table()
    operator = op.build_operator(table)
    amc = {"p4u_coil_current": 100.0, "plasma_current": 500.0}  # kA·turn
    i_pf = operator.assemble_pf_currents(amc)
    assert i_pf.shape == (1,)
    assert i_pf[0] == pytest.approx(100.0 * 1000.0)  # kA·turn → A
    # plasma_current is NOT a PF column (inferred) → not in the known term
    assert operator.pf_amc_channels == ["p4u_coil_current"]


def test_g_pf_folds_xmult_split():
    """The two xmult=0.5 filaments of circuit 1 must sum into one column whose
    response equals a single unit-weight filament at that (R, Z)."""
    table = _synthetic_table()
    operator = op.build_operator(table)
    # rebuild the same single-source column directly at the coil (R,Z) with w=1
    channels, kinds, sr, sz, sang, _, _ = op._sensor_rows(table)
    is_flux = np.array([k == "flux_loop" for k in kinds])
    single = op._green_columns(
        np.array([1.50]), np.array([1.10]), np.array([1.0]), sr, sz, sang, is_flux
    )
    assert np.allclose(operator.g_pf[:, 0], single, rtol=1e-12)


def test_redundant_circuits_merge_no_double_count():
    """Two circuits at one coil sharing an amc channel must merge into ONE G_pf
    column (averaged) — applying the coil current once, not twice.

    This is the confirmed double-count: EFIT represents each MAST PF coil with a
    fine + a coarse fcoil circuit, EACH normalised to the full coil current
    (Σxmult=1).  Without the merge, the coil current is applied per circuit → ~2×.
    """
    table = _synthetic_table()
    # add a SECOND P4U circuit (coarse 2-filament representation, Σxmult=1) at
    # the same coil location, mapped to the same amc channel.
    table.pf_filaments = list(table.pf_filaments) + [
        gsg.PFFilament(
            r=1.49, z=1.10, turns=1.0, width=0.01, height=0.01, circuit=3, xmult=0.5
        ),
        gsg.PFFilament(
            r=1.51, z=1.10, turns=1.0, width=0.01, height=0.01, circuit=3, xmult=0.5
        ),
    ]
    operator = op.build_operator(table)
    # circuits 1 and 3 are the SAME coil (p4u) → ONE column, two merged circuits
    assert operator.pf_amc_channels.count("p4u_coil_current") == 1
    col_idx = operator.pf_amc_channels.index("p4u_coil_current")
    assert sorted(operator.pf_merged_circuits[col_idx]) == [1, 3]
    # the merged column is the AVERAGE of the two (near-identical) circuit cols
    channels, kinds, sr, sz, sang, _, _ = op._sensor_rows(table)
    is_flux = np.array([k == "flux_loop" for k in kinds])
    c1 = op._green_columns(
        np.array([1.50, 1.50]),
        np.array([1.10, 1.10]),
        np.array([0.5, 0.5]),
        sr,
        sz,
        sang,
        is_flux,
    )
    c3 = op._green_columns(
        np.array([1.49, 1.51]),
        np.array([1.10, 1.10]),
        np.array([0.5, 0.5]),
        sr,
        sz,
        sang,
        is_flux,
    )
    assert np.allclose(operator.g_pf[:, col_idx], 0.5 * (c1 + c3), rtol=1e-12)
    # assemble_pf_currents has ONE entry for the coil (current applied once)
    i_pf = operator.assemble_pf_currents({"p4u_coil_current": 100.0})
    assert i_pf.shape == (operator.g_pf.shape[1],)
    assert i_pf[col_idx] == pytest.approx(100.0 * 1000.0)  # once, not 2×


# --- the vacuum round-trip --------------------------------------------


def test_vacuum_round_trip_single_coil():
    """A single KNOWN PF-coil current produces the expected vacuum-field
    signature at the probes — no plasma term.  This is the T2 sanity check."""
    table = _synthetic_table()
    operator = op.build_operator(table)
    amc = {"p4u_coil_current": 50.0}  # 50 kA·turn
    i_pf = operator.assemble_pf_currents(amc)
    pred = operator.vacuum_prediction(i_pf)
    assert pred.shape == (len(operator.sensor_channels),)
    assert np.all(np.isfinite(pred))
    # the field at the probes must be non-trivial and scale linearly with current
    pred2 = operator.vacuum_prediction(
        operator.assemble_pf_currents({"p4u_coil_current": 100.0})
    )
    assert np.allclose(pred2, 2.0 * pred, rtol=1e-10)
    # vacuum prediction (c_plasma=None) must equal predict with explicit zeros
    z_plasma = np.zeros(operator.g_plasma.shape[1])
    z_passive = np.zeros(operator.g_passive.shape[1])
    pred_explicit = operator.predict(i_pf, z_plasma, z_passive)
    assert np.allclose(pred, pred_explicit)


def test_vacuum_field_sign_and_magnitude_physical():
    """The vacuum Bz at a probe below an upper coil must have the right sign +
    a physically-plausible magnitude for a ~kA·turn MAST coil."""
    table = _synthetic_table()
    operator = op.build_operator(table)
    # coil at (1.50, 1.10), vertical probe (obv01) at (1.5, 0.0) below it.
    amc = {"p4u_coil_current": 100.0}  # 100 kA·turn → 1e5 A
    i_pf = operator.assemble_pf_currents(amc)
    pred = operator.vacuum_prediction(i_pf)
    iv = operator.sensor_channels.index("obv01")
    # 1e5 A loop ~0.4 m below the probe → Bz of order 0.01–1 T (MAST-realistic)
    assert 1e-3 < abs(pred[iv]) < 5.0


def test_plasma_term_adds_when_nonzero():
    """A non-zero plasma amplitude must change the prediction (block is wired)."""
    table = _synthetic_table()
    operator = op.build_operator(table)
    i_pf = operator.assemble_pf_currents({"p4u_coil_current": 10.0})
    base = operator.vacuum_prediction(i_pf)
    c_plasma = np.ones(operator.g_plasma.shape[1]) * 1e4  # 10 kA per node
    withp = operator.predict(i_pf, c_plasma=c_plasma)
    assert not np.allclose(base, withp)


# --- integration: real campaign tables (skipped in CI) ----------------

_HAVE_TABLES = (MANIFEST_DIR / "gs_geometry_tables.json").exists()
_skip_no_tables = pytest.mark.skipif(
    not _HAVE_TABLES, reason="geometry tables artifact not available (CI)"
)


def _load_real_tables() -> dict[str, gsg.GeometryTable]:
    """Reconstruct GeometryTable objects from the committed full-tables JSON."""
    import json as _json  # noqa: PLC0415

    raw = _json.loads((MANIFEST_DIR / "gs_geometry_tables.json").read_text())
    tables: dict[str, gsg.GeometryTable] = {}
    for key, t in raw["campaigns"].items():
        sig = gsg.SetupSignature(**t["signature"])
        tables[key] = gsg.GeometryTable(
            signature=sig,
            shots=t["shots"],
            b_probes=[gsg.BProbe(**b) for b in t["b_probes"]],
            flux_loops=[gsg.FluxLoop(**f) for f in t["flux_loops"]],
            pf_filaments=[gsg.PFFilament(**p) for p in t["pf_filaments"]],
            limiter_r=t["limiter_r"],
            limiter_z=t["limiter_z"],
            sensor_map=[gsg.SensorMapping(**m) for m in t["sensor_map"]],
            passive_structures=[
                gsg.PassiveStructure(**p) for p in t["passive_structures"]
            ],
            amc_current_channels=t["amc_current_channels"],
            unmatched_amb=t["unmatched_amb"],
            r0=t["r0"],
            minor_radius=t["minor_radius"],
        )
    return tables


@_skip_no_tables
def test_real_operators_build_for_all_campaigns():
    tables = _load_real_tables()
    operators = op.build_all_operators(tables)
    assert len(operators) == len(tables)
    for operator in operators.values():
        s = operator.shapes()
        # every campaign predicts the B-probes + the clean flux loops
        assert s["n_b_probe"] >= 60  # ~69 amb B-probes
        assert s["n_known_coil"] >= 8  # the P2–P6 + solenoid family
        assert s["n_plasma_node"] > 0
        # the 8 fl_p2* unmatched loops are excluded
        assert s["n_excluded_channel"] >= 2
        assert operator.g_pf.shape[0] == s["n_sensor"]
        # one G_pf column per physical coil (== per distinct amc channel)
        assert operator.g_pf.shape[1] == len(set(operator.pf_amc_channels))
        assert len(operator.pf_amc_channels) == len(set(operator.pf_amc_channels))


@_skip_no_tables
def test_real_operator_known_pf_maps_solenoid_and_pf_coils():
    tables = _load_real_tables()
    operators = op.build_all_operators(tables)
    # pick the dominant fc938 campaign
    key = next(k for k in operators if "fc938" in k)
    operator = operators[key]
    labels = {c.coil_label for c in operator.circuit_classes if c.role == "known_pf"}
    # the central solenoid + the main PF coils must be identified
    assert "sol" in labels
    assert any(lbl.startswith("p4") for lbl in labels)
    # every KNOWN circuit has a real amc channel (never fabricated)
    for c in operator.circuit_classes:
        if c.role == "known_pf":
            assert c.amc_channel in operator.pf_amc_channels
            assert c.amc_channel != ""


@_skip_no_tables
def test_real_vacuum_round_trip_matches_raw_amb_at_near_vacuum_slice():
    """DONE-WHEN #2: at a NEAR-VACUUM slice (|Ip|≈0, coils sizable) the PF-only
    vacuum prediction must match raw ``amb`` per-probe to ~unity.

    This is the ABSOLUTE SI / flux-convention / orientation anchor — it pins
    that the operator's physical units are right end-to-end (raw amc → A,
    Green's functions → Wb/T, orientation projection), validated against real
    raw magnetics with no EFIT reference.  Its sharpest catch is a 2π-class
    flux-convention slip (stream-function vs total flux sends the median to
    ~6.7 or ~0.16) or a unit/μ0 error (gross) — both trip the [0.5, 1.5] band.
    The PF-only prediction tracks raw amb here because, with no plasma, the
    field is vacuum + a small eddy term (the residual T3 will infer).

    NOTE: this is NOT the double-count guard — the only clean near-vacuum
    slices are solenoid-dominated (a singleton circuit the merge does not
    touch), so a reverted merge only nudges the median.  The deterministic
    double-count guard is the one-G_pf-column-per-coil structural assertion in
    ``test_real_operators_build_for_all_campaigns`` + the merge-math invariant
    in ``test_redundant_circuits_merge_no_double_count``.
    """
    import zarr  # noqa: PLC0415

    from imas_ambix.data.paths import local_shot_path  # noqa: PLC0415

    tables = _load_real_tables()
    operators = op.build_all_operators(tables)
    key = next(k for k in operators if "fc938" in k)
    operator = operators[key]
    shot = tables[key].shots[0]
    if not local_shot_path(shot, tier="level1").exists():
        pytest.skip("representative shot not in mirror")

    store = zarr.open(str(local_shot_path(shot, tier="level1")), mode="r")
    amc, amb = store["amc"], store["amb"]
    ip = np.asarray(amc["plasma_current"][:])
    amc_t = np.asarray(amc["time"][:])
    sol = np.asarray(amc["sol_current"][:])
    fin = np.isfinite(ip) & np.isfinite(sol)
    # near-vacuum: |Ip| < 3 kA but solenoid sizable (>5 kA)
    mask = fin & (np.abs(ip) < 3.0) & (np.abs(sol) > 5.0)
    if not mask.any():
        pytest.skip("no near-vacuum slice in representative shot")
    cand = np.where(mask)[0]
    ti = int(cand[np.argmax(np.abs(sol[cand]))])
    t0 = float(amc_t[ti])

    amc_vals = {
        k: (v if np.isfinite(v) else 0.0)
        for k, v in op.read_amc_currents_at_index(shot, ti).items()
    }
    i_pf = operator.assemble_pf_currents(amc_vals)
    pred = operator.vacuum_prediction(i_pf)
    assert pred.shape[0] == len(operator.sensor_channels)
    assert np.all(np.isfinite(pred))

    amb_t = np.asarray(amb["time"][:])
    ai = int(np.argmin(np.abs(amb_t - t0)))
    ratios = []
    for idx, ch in enumerate(operator.sensor_channels):
        if operator.sensor_kind[idx] != "b_probe" or ch not in amb:
            continue
        rawv = float(np.asarray(amb[ch][:])[ai])
        if abs(rawv) > 0.01:  # only probes with real signal
            ratios.append(pred[idx] / rawv)
    ratios = np.array(ratios)
    assert ratios.size >= 10  # enough probes to be meaningful
    med = float(np.median(ratios))
    # [0.5, 1.5] band is the absolute SI/flux/orientation anchor — a 2π flux
    # slip → ~6.7 or ~0.16, a unit/μ0 error → gross; both trip this.
    assert 0.5 < med < 1.5, f"median pred/raw={med:.3f} (SI/flux/orientation off?)"


@_skip_no_tables
def test_passive_amm_coincidence_documented():
    tables = _load_real_tables()
    key = next(k for k in operators_keys(tables) if "fc1004" in k)
    coin = op.passive_amm_coincidence(tables[key])
    assert coin["n_inferred_passive_circuit"] > 0
    assert coin["n_amm_passive_structure"] == 76
    # the fc1004 singletons are known to overlap amm geometry
    assert coin["n_coincident_within_tol"] > 0


def operators_keys(tables: dict) -> list[str]:
    return list(tables.keys())


def test_pf_columns_use_finite_area_kernel_near_packs():
    """PF-circuit sensor columns must be the finite-area cylinder response:
    identical to the point filament far from the pack, different (smooth)
    at a sensor adjacent to a large winding pack."""
    import numpy as np

    from imas_ambix.gs.cylinder import hybrid_greens
    from imas_ambix.gs.operator import _green_columns, greens_psi

    # one fat solenoid-like pack, one far flux loop + one adjacent flux loop
    src_r = np.array([0.12])
    src_z = np.array([0.0])
    w = np.array([1.0])
    dr = np.array([0.10])
    dz = np.array([0.30])
    sens_r = np.array([1.5, 0.19])  # far, adjacent
    sens_z = np.array([0.0, 0.05])
    ang = np.array([90.0, 90.0])
    is_flux = np.array([True, True])

    col = _green_columns(
        src_r, src_z, w, sens_r, sens_z, ang, is_flux, src_dr=dr, src_dz=dz
    )
    point = greens_psi(sens_r, sens_z, 0.12, 0.0)
    cyl, _br, _bz = hybrid_greens(sens_r, sens_z, 0.12, 0.0, 0.10, 0.30)
    np.testing.assert_allclose(col, cyl, rtol=1e-12)
    np.testing.assert_allclose(col[0], point[0], rtol=1e-9)  # far: identical
    assert abs(col[1] - point[1]) / abs(point[1]) > 1e-4  # near: finite-area
