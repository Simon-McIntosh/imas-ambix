"""Tests for the opt-in saddle-robust LCFS boundary read.

On free (non-force-balanced) current, the innermost-ψ boundary pick can lock
onto a SPURIOUS saddle in the current's sensor-null-space concentration — a
saddle a hair's-breadth from the axis, not the genuine separatrix that bounds
the whole confined region — which under-sizes the read LCFS by tens of cm.
These tests pin the guard on a KNOWN synthetic field: a circular paraboloid
(exact circular boundary at the limiter contact, per
``test_lcfs_radii_on_circular_boundary_equal_radius``) with a small local
double-well perturbation added near the axis, which manufactures exactly one
extra O-point/X-point pair — the spurious saddle — well inside the true
boundary radius.
"""

from __future__ import annotations

import numpy as np
import pytest

from imas_ambix.latent import topology as topo
from imas_ambix.latent.gs_solve import (
    EquilibriumGrid,
    _read_axis,
    _read_boundary_psi,
    _read_boundary_psi_robust,
)
from scripts.boundary_moment_gate_eval import build_parser
from scripts.patch_gate_eval import (
    count_saddles,
    lcfs_offset_cm_stats,
    saddle_excess_stats,
    score,
)

# --- isolated topology.boundary_flux_robust tests --------------------------


def test_boundary_flux_robust_reproduces_naive_at_zero_guard():
    """min_axis_dist=0.0 (default) must reproduce boundary_flux exactly."""
    axis = (1.0, 0.0)
    axis_psi = 0.0
    cp = topo.CriticalPoints(
        o_points=np.zeros((0, 2)),
        o_psi=np.zeros((0,)),
        x_points=np.array([[1.05, 0.0], [1.35, 0.0]]),
        x_psi=np.array([0.02, 0.15]),
    )
    naive = topo.boundary_flux(cp, axis, axis_psi)
    reproduced = topo.boundary_flux_robust(cp, axis, axis_psi, min_axis_dist=0.0)
    assert reproduced == naive


def test_boundary_flux_robust_rejects_near_axis_spurious_saddle():
    """A spurious saddle a hair's-breadth from the axis must not steal the
    boundary read from a genuine, more distant bounding saddle."""
    axis = (1.0, 0.0)
    axis_psi = 0.0
    cp = topo.CriticalPoints(
        o_points=np.zeros((0, 2)),
        o_psi=np.zeros((0,)),
        x_points=np.array([[1.05, 0.0], [1.35, 0.0]]),  # spurious, genuine
        x_psi=np.array([0.02, 0.15]),
    )
    naive = topo.boundary_flux(cp, axis, axis_psi)
    assert naive == 0.02  # the pathology: closest-ψ picks the spurious saddle

    robust = topo.boundary_flux_robust(cp, axis, axis_psi, min_axis_dist=0.1)
    assert robust == 0.15  # falls through to the genuine, distant saddle


def test_boundary_flux_robust_returns_none_when_all_saddles_rejected():
    """If every candidate is within the guard radius, fall through to None
    (the caller's limiter fallback), never a spurious pick."""
    axis = (1.0, 0.0)
    axis_psi = 0.0
    cp = topo.CriticalPoints(
        o_points=np.zeros((0, 2)),
        o_psi=np.zeros((0,)),
        x_points=np.array([[1.05, 0.0]]),
        x_psi=np.array([0.02]),
    )
    assert topo.boundary_flux_robust(cp, axis, axis_psi, min_axis_dist=0.5) is None


# --- full-grid integration: gs_solve._read_boundary_psi(_robust) ----------


def _spurious_saddle_grid_and_psi(
    r0: float = 1.0,
    rb: float = 0.4,
    dip_offset: float = 0.15,
    dip_amplitude: float = 0.02,
    dip_width: float = 0.04,
):
    """A circular-paraboloid ψ field with one manufactured spurious saddle.

    ``psi = (R-R0)^2 + Z^2`` alone has NO critical point but the axis, and the
    exact circular limiter contact bounds it at radius ``rb`` (matches
    ``test_lcfs_radii_on_circular_boundary_equal_radius``).  Adding a small,
    spatially LOCALISED Gaussian dip offset from the axis by ``dip_offset``
    manufactures exactly one extra O-point / X-point pair — a spurious saddle
    well inside ``rb`` whose ψ is close to the axis flux — while decaying to
    ~0 well before the limiter, so the true circular boundary is undisturbed
    (an earlier, non-localised polynomial perturbation contaminated the whole
    far field and had to be replaced by this Gaussian form).  The dip is
    shallow enough that its own local minimum stays higher than the true
    axis, so the axis itself is unaffected.
    """
    nr, nz = 161, 161
    rg = np.linspace(r0 - rb - 0.2, r0 + rb + 0.2, nr)
    zg = np.linspace(-(rb + 0.2), rb + 0.2, nz)
    theta = np.linspace(0.0, 2.0 * np.pi, 128, endpoint=False)
    limiter_r = r0 + rb * np.cos(theta)
    limiter_z = rb * np.sin(theta)
    grid = EquilibriumGrid(
        rg=rg,
        zg=zg,
        limiter_r=limiter_r,
        limiter_z=limiter_z,
        coil_psi_columns=np.zeros((rg.size * zg.size, 0)),
        r0=r0,
    )
    rr, zz = grid.mesh_r, grid.mesh_z
    dip = -dip_amplitude * np.exp(
        -(((rr - (r0 + dip_offset)) ** 2 + zz**2) / dip_width**2)
    )
    psi2d = (rr - r0) ** 2 + zz**2 + dip
    return grid, psi2d


def test_naive_boundary_read_under_sizes_lcfs_on_spurious_saddle():
    """Pins the pathology: the naive read locks onto the spurious saddle and
    under-sizes the known circular boundary by roughly 5x."""
    r0, rb = 1.0, 0.4
    grid, psi2d = _spurious_saddle_grid_and_psi(r0=r0, rb=rb)
    axis, axis_psi = _read_axis(psi2d, grid, -1.0)
    np.testing.assert_allclose(axis, (r0, 0.0), atol=0.01)

    naive_bnd = _read_boundary_psi(psi2d, grid, axis_psi)
    radii = topo.lcfs_radii(psi2d, grid.rg, grid.zg, axis, naive_bnd)
    # the spurious saddle sits at distance ~0.09 m from the axis; the naive
    # read's median radius lands well short of the true 0.4 m boundary
    assert np.nanmedian(radii) < 0.3 * rb


def test_robust_boundary_read_recovers_circular_boundary():
    """With the min-axis-distance guard the read rejects the spurious saddle
    and falls back to the (correct) limiter-contact flux, recovering the
    known circular boundary radius."""
    r0, rb = 1.0, 0.4
    grid, psi2d = _spurious_saddle_grid_and_psi(r0=r0, rb=rb)
    axis, axis_psi = _read_axis(psi2d, grid, -1.0)

    robust_bnd = _read_boundary_psi_robust(
        psi2d, grid, axis, axis_psi, min_axis_dist=0.12
    )
    radii = topo.lcfs_radii(psi2d, grid.rg, grid.zg, axis, robust_bnd)
    np.testing.assert_allclose(radii, rb, atol=0.01)


def test_robust_boundary_read_default_matches_naive():
    """smooth_sigma=0, min_axis_dist=0 must reproduce _read_boundary_psi
    exactly — the new readout path is opt-in, not a silent default change."""
    r0, rb = 1.0, 0.4
    grid, psi2d = _spurious_saddle_grid_and_psi(r0=r0, rb=rb)
    axis, axis_psi = _read_axis(psi2d, grid, -1.0)

    naive_bnd = _read_boundary_psi(psi2d, grid, axis_psi)
    reproduced = _read_boundary_psi_robust(psi2d, grid, axis, axis_psi)
    assert reproduced == naive_bnd


def test_smoothing_also_suppresses_the_spurious_saddle():
    """The smoothing arm alone (no distance guard) also recovers the known
    boundary: the small-scale spurious saddle washes out under a light
    Gaussian blur while the genuine circular structure survives."""
    r0, rb = 1.0, 0.4
    grid, psi2d = _spurious_saddle_grid_and_psi(r0=r0, rb=rb)
    axis, axis_psi = _read_axis(psi2d, grid, -1.0)

    robust_bnd = _read_boundary_psi_robust(
        psi2d, grid, axis, axis_psi, smooth_sigma=3.0
    )
    radii = topo.lcfs_radii(psi2d, grid.rg, grid.zg, axis, robust_bnd)
    np.testing.assert_allclose(radii, rb, atol=0.01)


# --- lcfs_offset_cm_stats (the flat-top-only cm summary requested for the
# side-by-side comparison against the flux-map report's 31.3 cm baseline) --


def test_lcfs_offset_cm_stats_matches_hand_computed_median():
    """Two slices, known 8-radii offsets; the flat-top mask selects one."""
    target = np.zeros((2, 14))
    ref = np.zeros((2, 14))
    # slice 0 (flat-top): all 8 lcfs radii off by 0.10 m = 10 cm
    target[0, 6:14] = 0.10
    # slice 1 (not flat-top): all 8 lcfs radii off by 0.30 m = 30 cm
    target[1, 6:14] = 0.30
    flattop_mask = np.array([True, False])

    stats = lcfs_offset_cm_stats(target, ref, flattop_mask)
    assert stats["lcfs_offset_median_cm_all"] == 20.0  # median(10, 30)
    assert stats["lcfs_offset_median_cm_flattop"] == 10.0  # only slice 0
    assert stats["n_flattop_slices"] == 1


def test_lcfs_offset_cm_stats_empty_flattop_mask_returns_none():
    """No flat-top slice selected -> the flat-top field is None, not a crash."""
    target = np.zeros((1, 14))
    ref = np.zeros((1, 14))
    target[0, 6:14] = 0.05
    flattop_mask = np.array([False])

    stats = lcfs_offset_cm_stats(target, ref, flattop_mask)
    assert stats["lcfs_offset_median_cm_all"] == 5.0
    assert stats["lcfs_offset_median_cm_flattop"] is None
    assert stats["n_flattop_slices"] == 0


# --- score() paired-bootstrap CIs over shots --------------------------------


def _make_score_fixture(n_shots=6, rows_per_shot=4, seed=0):
    """Deterministic small fixture: a decent model beating a poor baseline,
    no X-points present (isolates the axis/LCFS CI machinery from the
    permutation-matching path, which is exercised separately below)."""
    rng = np.random.default_rng(seed)
    shot_ids = np.repeat(np.arange(n_shots), rows_per_shot)
    n = shot_ids.size
    ref = np.zeros((n, 14))
    ref[:, 0] = rng.normal(1.0, 0.05, n)
    ref[:, 1] = rng.normal(0.0, 0.05, n)
    ref[:, 2:6] = np.nan
    ref[:, 6:14] = rng.normal(0.5, 0.02, (n, 8))
    model = ref + rng.normal(0.0, 0.005, ref.shape)  # close to the referee
    # columns 2:6 (X-points) are all-NaN in this fixture, so their baseline
    # value is never scored (matched_xpoint_error requires a finite ref too)
    finite_cols = np.isfinite(ref).any(axis=0)
    baseline_vec = np.zeros(ref.shape[1])
    baseline_vec[finite_cols] = np.nanmean(ref[:, finite_cols], axis=0) + 0.2
    return model, ref, baseline_vec, shot_ids


def test_score_without_shot_ids_omits_ci_fields():
    """Backward compatible: callers that have not threaded shot identity
    through get the same skill dict as before, no CI keys."""
    model, ref, baseline_vec, _ = _make_score_fixture()
    out = score(model, ref, baseline_vec)
    assert "axis_skill_ci" not in out
    assert "lcfs_skill_ci" not in out
    assert "per_quantity_skill_ci" not in out


def test_score_with_shot_ids_returns_bootstrap_ci_for_every_skill():
    model, ref, baseline_vec, shot_ids = _make_score_fixture()
    out = score(model, ref, baseline_vec, shot_ids=shot_ids, n_boot=200, ci_seed=0)

    for key in ("axis_skill_ci", "lcfs_skill_ci", "xpoint_set_skill_ci"):
        assert key in out
        lo, hi = out[key]
        assert lo is None or (np.isfinite(lo) and np.isfinite(hi) and lo <= hi)

    assert "per_quantity_skill_ci" in out
    assert set(out["per_quantity_skill_ci"]) == set(out["per_quantity_skill"])
    lo, hi = out["per_quantity_skill_ci"]["axis_R"]
    assert np.isfinite(lo) and np.isfinite(hi) and lo <= hi

    assert out["ci_n_boot"] == 200
    assert out["ci_seed"] == 0


def test_score_ci_is_deterministic_given_seed():
    model, ref, baseline_vec, shot_ids = _make_score_fixture()
    out1 = score(model, ref, baseline_vec, shot_ids=shot_ids, n_boot=150, ci_seed=7)
    out2 = score(model, ref, baseline_vec, shot_ids=shot_ids, n_boot=150, ci_seed=7)
    assert out1["axis_skill_ci"] == out2["axis_skill_ci"]
    assert out1["lcfs_skill_ci"] == out2["lcfs_skill_ci"]


def test_score_ci_degenerate_single_shot_returns_none_not_crash():
    """Fewer than 2 unique shots cannot be resampled meaningfully — the CI
    fields must be [None, None], never raise."""
    model, ref, baseline_vec, _ = _make_score_fixture(n_shots=1, rows_per_shot=5)
    shot_ids = np.zeros(5, dtype=int)
    out = score(model, ref, baseline_vec, shot_ids=shot_ids, n_boot=50, ci_seed=0)
    assert out["axis_skill_ci"] == [None, None]


# --- double-null-correct saddle-excess metric -------------------------------


def test_naive_saddle_free_rule_mislabels_double_null():
    """Documents the bug the excess metric fixes: a naive 'saddles <= 1'
    rule flags a genuine MAST double-null (2 real referee X-points) as NOT
    saddle-free, when the model's read is in fact clean."""
    saddle_counts = np.array([2])
    naive_saddle_free = saddle_counts <= 1
    assert not naive_saddle_free[0]  # mislabelled by the old rule

    ref = np.zeros((1, 14))
    ref[:, 2:6] = 1.0  # both X-point slots present -> referee sees a double-null
    out = saddle_excess_stats(saddle_counts, ref)
    assert out["referee_xpoint_count_mean"] == 2.0
    assert out["saddle_excess_mean"] == 0.0
    assert out["saddle_clean_fraction"] == 1.0


def test_saddle_excess_stats_mixed_cohort():
    # slice 0: single-null referee (1 X-point), model finds 1 -> excess 0
    # slice 1: single-null referee, model finds 2 -> excess 1 (genuinely spurious)
    # slice 2: double-null referee (2 X-points), model finds 3 -> excess 1
    ref = np.zeros((3, 14))
    ref[0, 2:4] = 1.0
    ref[0, 4:6] = np.nan
    ref[1, 2:4] = 1.0
    ref[1, 4:6] = np.nan
    ref[2, 2:6] = 1.0
    saddle_counts = np.array([1, 2, 3])

    out = saddle_excess_stats(saddle_counts, ref)
    assert out["referee_xpoint_count_mean"] == pytest.approx((1 + 1 + 2) / 3)
    assert out["saddle_excess_mean"] == pytest.approx((0 + 1 + 1) / 3)
    assert out["saddle_clean_fraction"] == pytest.approx(1 / 3)


def test_saddle_excess_stats_empty_returns_none():
    out = saddle_excess_stats(np.array([]), np.zeros((0, 14)))
    assert out == {
        "saddle_excess_mean": None,
        "saddle_excess_median": None,
        "saddle_clean_fraction": None,
        "referee_xpoint_count_mean": None,
    }


def test_count_saddles_finds_the_manufactured_spurious_saddle():
    """The shared in-limiter saddle counter (reused by both the free-current
    and current-moment arms) counts the one manufactured X-point in the
    known synthetic field from the boundary-read robustness fixture above."""
    r0, rb = 1.0, 0.4
    grid, psi2d = _spurious_saddle_grid_and_psi(r0=r0, rb=rb)
    assert count_saddles(psi2d, grid) == 1


# --- origin-controlled LCFS read: --axis-source patch is the scored default -


def test_axis_source_default_is_patch():
    """The scored default must be 'patch' (origin-controlled); 'centroid'
    remains available as the fast smoke arm only, never silently scored."""
    args = build_parser().parse_args([])
    assert args.axis_source == "patch"
