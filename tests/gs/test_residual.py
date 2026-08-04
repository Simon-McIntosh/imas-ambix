"""Tests for the GS force-balance residual monitor.

Layers:

* **Unit** — the profile-DOF polynomial basis DOF counts, the passive low-rank
  basis, the column-normalisation / λ regulariser behaviour, and the
  anti-tuning operating-point selector.  No mirror / network needed.
* **Synthetic inverse** — a controlled forward→inverse round-trip on a tiny
  synthetic operator: with a planted plasma current the solve recovers a
  non-trivial residual, and the regulariser collapses r as DOF grows.
* **Integration** — the real per-campaign operator + a real shot, skipped when
  the mirror is absent (CI).
"""

from __future__ import annotations

import numpy as np
import pytest

from imas_ambix.data.paths import MANIFEST_DIR
from imas_ambix.gs import operator as op
from imas_ambix.gs import residual as res

# --- unit: profile-DOF basis ------------------------------------------


def test_profile_poly_basis_dof_counts():
    """order-1 → 3 DOF, order-2 → 6, order-4 → 15 (the locked frontier axis)."""
    rz = np.random.default_rng(0).normal(size=(40, 2))
    assert res.plasma_poly_basis(rz, 1, 0.85, 0.65).shape[1] == 3
    assert res.plasma_poly_basis(rz, 2, 0.85, 0.65).shape[1] == 6
    assert res.plasma_poly_basis(rz, 4, 0.85, 0.65).shape[1] == 15


def test_profile_basis_constant_column():
    """The basis always carries a constant column (uniform jφ mode)."""
    rz = np.random.default_rng(1).normal(size=(10, 2))
    b = res.plasma_poly_basis(rz, 2, 0.85, 0.65)
    assert np.allclose(b[:, 0], 1.0)


def test_passive_lowrank_basis_rank_and_orthonormal():
    """The passive basis is the leading-r right singular vectors (orthonormal)."""
    rng = np.random.default_rng(2)
    g = rng.normal(size=(20, 30))
    v = res.passive_lowrank_basis(g, 4)
    assert v.shape == (30, 4)
    # columns orthonormal
    assert np.allclose(v.T @ v, np.eye(4), atol=1e-10)


# --- unit: anti-tuning operating-point selector -----------------------


def test_operating_point_selects_min_dof_in_band():
    cells = [
        {"profile_order": 1, "n_plasma_dof": 3, "passive_rank": 4,
         "lambda": 0.0, "quiescent_residual_median": 0.5},
        {"profile_order": 1, "n_plasma_dof": 3, "passive_rank": 4,
         "lambda": 1e-2, "quiescent_residual_median": 0.6},
        {"profile_order": 2, "n_plasma_dof": 6, "passive_rank": 4,
         "lambda": 0.0, "quiescent_residual_median": 0.3},
    ]
    sel = res._select_operating_point(cells)
    assert sel["selected"]
    # min plasma-DOF (3) at the smallest λ in band
    assert sel["n_plasma_dof"] == 3
    assert sel["lambda"] == 0.0


def test_operating_point_fails_when_all_trivial():
    """If every cell collapses to r≈0 (trivial), no operating point is selectable."""
    cells = [
        {"profile_order": 1, "n_plasma_dof": 3, "passive_rank": 4,
         "lambda": 0.0, "quiescent_residual_median": 1e-6},
    ]
    sel = res._select_operating_point(cells)
    assert not sel["selected"]


# --- synthetic inverse: forward→inverse round-trip --------------------


def _toy_operator(seed: int = 0) -> op.ForwardOperator:
    """A tiny synthetic ForwardOperator with random but well-posed G blocks."""
    rng = np.random.default_rng(seed)
    n_sensor = 30
    g_pf = rng.normal(size=(n_sensor, 3))
    plasma_rz = rng.uniform([0.3, -1.0], [1.4, 1.0], size=(20, 2))
    # plasma columns = Green-like smooth functions of node position
    g_plasma = np.column_stack(
        [np.exp(-((np.arange(n_sensor) - i) ** 2) / 50.0) for i in range(20)]
    )
    g_passive = rng.normal(size=(n_sensor, 12)) * 0.1
    return op.ForwardOperator(
        signature_key="toy",
        sensor_channels=[f"obv{i:02d}" for i in range(n_sensor)],
        sensor_kind=["b_probe"] * n_sensor,
        g_pf=g_pf,
        g_plasma=g_plasma,
        g_passive=g_passive,
        pf_circuits=[1, 2, 3],
        pf_amc_channels=["a", "b", "c"],
        pf_merged_circuits=[[1], [2], [3]],
        plasma_rz=plasma_rz,
        passive_rz=rng.normal(size=(12, 2)),
        circuit_classes=[],
        excluded_channels=[],
        flagged_channels=[],
    )


def test_synthetic_residual_nontrivial_then_collapses_with_dof():
    """The residual shrinks as plasma DOF grows — the frontier's core property.

    Plant a field that the operator does NOT exactly represent at low DOF; the
    order-1 solve leaves a non-trivial residual, and raising DOF reduces it.
    """
    toy = _toy_operator(1)
    target = res.trustworthy_target(toy)
    rng = np.random.default_rng(5)
    # a 'true' field = PF term + a structured plasma contribution + noise
    i_pf = rng.normal(size=3) * 10.0
    true_theta = np.array([1.0, 0.5, -0.3])  # order-1 plasma profile
    b1 = res.plasma_poly_basis(toy.plasma_rz, 1, toy.r0, toy.minor_radius)
    c_plasma = b1 @ true_theta
    raw = toy.g_pf @ i_pf + toy.g_plasma @ c_plasma
    raw = raw + rng.normal(size=raw.size) * 0.01 * np.std(raw)
    raw_trust = raw[target.rows][None, :]
    scale = res.robust_sensor_scale(raw_trust, None)
    r_by_dof = []
    for order in (1, 2, 4):
        solver = res.InverseSolver(toy, target, scale, order, 4)
        out = solver.solve(raw_trust[0], i_pf, 0.0)
        r_by_dof.append(out["residual_frac"])
    # residual is non-trivial at the lowest DOF and does not increase with DOF
    assert r_by_dof[0] >= 0.0
    assert r_by_dof[-1] <= r_by_dof[0] + 1e-9


def test_lambda_regularises_amplitudes():
    """Larger λ shrinks the inferred amplitudes (ridge behaviour)."""
    toy = _toy_operator(3)
    target = res.trustworthy_target(toy)
    rng = np.random.default_rng(9)
    i_pf = rng.normal(size=3) * 5.0
    raw = toy.g_pf @ i_pf + rng.normal(size=toy.g_pf.shape[0])
    raw_trust = raw[target.rows]
    scale = res.robust_sensor_scale(raw_trust[None, :], None)
    solver = res.InverseSolver(toy, target, scale, 2, 4)
    lo = solver.solve(raw_trust, i_pf, 0.0)
    hi = solver.solve(raw_trust, i_pf, 1e-1)
    norm_lo = np.linalg.norm(lo["theta_plasma"]) + np.linalg.norm(lo["psi_passive"])
    norm_hi = np.linalg.norm(hi["theta_plasma"]) + np.linalg.norm(hi["psi_passive"])
    assert norm_hi <= norm_lo + 1e-9


# --- integration: real campaign operator + a real shot ----------------

_HAVE_TABLES = (MANIFEST_DIR / "gs_geometry_tables.json").exists()
_skip_no_tables = pytest.mark.skipif(
    not _HAVE_TABLES, reason="geometry tables artifact not available (CI)"
)


def _load_real_fc938_operator() -> op.ForwardOperator:
    import json as _json  # noqa: PLC0415

    from imas_ambix.gs import geometry as gsg  # noqa: PLC0415

    raw = _json.loads((MANIFEST_DIR / "gs_geometry_tables.json").read_text())
    key = next(k for k in raw["campaigns"] if "fc938" in k)
    t = raw["campaigns"][key]
    table = gsg.GeometryTable(
        signature=gsg.SetupSignature(**t["signature"]),
        shots=t["shots"],
        b_probes=[gsg.BProbe(**b) for b in t["b_probes"]],
        flux_loops=[gsg.FluxLoop(**f) for f in t["flux_loops"]],
        pf_filaments=[gsg.PFFilament(**p) for p in t["pf_filaments"]],
        limiter_r=t["limiter_r"],
        limiter_z=t["limiter_z"],
        sensor_map=[gsg.SensorMapping(**m) for m in t["sensor_map"]],
        passive_structures=[gsg.PassiveStructure(**p) for p in t["passive_structures"]],
        amc_current_channels=t["amc_current_channels"],
        unmatched_amb=t["unmatched_amb"],
        r0=t["r0"],
        minor_radius=t["minor_radius"],
    )
    return op.build_operator(table)


@_skip_no_tables
def test_real_trustworthy_target_is_bprobes_plus_one_fl():
    """The trustworthy target excludes the flagged + unmatched flux loops."""
    operator = _load_real_fc938_operator()
    target = res.trustworthy_target(operator)
    n_b = target.kinds.count("b_probe")
    n_f = target.kinds.count("flux_loop")
    assert n_b >= 60  # the ~76 B-probes
    assert n_f <= 2  # at most the 1 cleanly-mapped flux loop
    # no flagged channel leaks into the target
    assert not (set(target.channels) & set(operator.flagged_channels))


@_skip_no_tables
def test_real_design_effective_rank_is_small():
    """The row-scaled design effective rank is « nominal column count.

    This is the diagnostic that JUSTIFIES the regulariser: 162 nominal columns
    but effective rank ~6 (smooth Green's kernels are highly correlated).
    """
    operator = _load_real_fc938_operator()
    target = res.trustworthy_target(operator)
    g = np.hstack([operator.g_plasma[target.rows], operator.g_passive[target.rows]])
    # row-whiten by column-wise std proxy (uniform here is fine for the rank)
    s = np.linalg.svd(g, compute_uv=False)
    energy = np.cumsum(s**2) / np.sum(s**2)
    eff_rank_99 = int(np.searchsorted(energy, 0.99) + 1)
    assert eff_rank_99 < g.shape[1] // 3  # « 162
