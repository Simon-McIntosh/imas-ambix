"""Unit tests for imas_ambix.gs.integrated_inventory (T9 feasibility scoping).

All tests run on a tiny SYNTHETIC ``InventoryResult`` + synthetic regime
scalars + a synthetic OOD box — no GPFS, no Zarr open, no per-shot read.
They check the load-bearing invariants of the feasibility builder:

* monotonicity (a richer input combo's co-available corpus is a SUBSET of a
  poorer one's, so N is non-increasing as inputs are added);
* the GS-envelope restriction is a subset (n_gs ≤ n_total) and applies the
  efm + shot-range proxy;
* the conformal cal-viability flag fires at the 200 floor;
* the LOCKED OOD box is applied identically to every combo;
* the report serialises to a compact dict and round-trips through ``save``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from imas_ambix.gs import integrated_inventory as ii
from imas_ambix.statespace.inventory import InventoryResult
from imas_ambix.statespace.splits import RegimeBox

# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------


def _synthetic_inventory() -> InventoryResult:
    """A small inventory exercising every combo branch.

    Shot IDs chosen so some fall outside the GS envelope and some lack efm.
    """
    base_mag_ane_xim = ["ama", "amb", "amc", "ane", "xim", "efm"]
    shot_groups: dict[int, tuple[str, ...]] = {
        # in-envelope, full internal core + emission + camera
        20001: tuple(sorted([*base_mag_ane_xim, "ams", "ayc", "abm", "xsx", "rbb"])),
        20002: tuple(sorted([*base_mag_ane_xim, "ams", "atm", "abm", "xsx"])),
        # MSE but no Thomson
        20003: tuple(sorted([*base_mag_ane_xim, "ams"])),
        # mag+ane+xim only (v0 baseline)
        20004: tuple(sorted(base_mag_ane_xim)),
        20005: tuple(sorted(base_mag_ane_xim)),
        # in-envelope MSE+TS but NO efm → dropped by the GS-envelope proxy
        20006: tuple(sorted(["ama", "amb", "amc", "ane", "xim", "ams", "aye"])),
        # OUTSIDE the shot-range envelope (sid > GS_CAMPAIGN_SHOT_MAX)
        40001: tuple(sorted([*base_mag_ane_xim, "ams", "ayc"])),
        # no target (xim) → never co-available
        20007: tuple(sorted(["ama", "amb", "amc", "ane", "ams", "efm"])),
        # IR-only-ish shot (sparse) to populate ir_absence
        20008: tuple(sorted([*base_mag_ane_xim, "rir"])),
    }
    all_groups = sorted({g for grps in shot_groups.values() for g in grps})
    return InventoryResult(
        shot_groups=shot_groups,
        all_groups=all_groups,
        n_shots=len(shot_groups),
    )


def _synthetic_scalars() -> dict[int, dict[str, float]]:
    """Give every shot an in-distribution operating point (none inside the box)."""
    return {
        sid: {"ip_mean": 400.0, "ne_mean": 5.0e19}
        for sid in (
            20001,
            20002,
            20003,
            20004,
            20005,
            20006,
            40001,
            20007,
            20008,
        )
    }


def _synthetic_box() -> RegimeBox:
    """A high-corner OOD box that NO synthetic shot falls into (clean split)."""
    return RegimeBox(
        ip_min=667.0,
        ip_max=986.0,
        ne_min=13.4,
        ne_max=27.4,
        description="synthetic locked joint_p84",
    )


@pytest.fixture
def report() -> ii.IntegratedFeasibilityReport:
    return ii.build_feasibility(
        inventory=_synthetic_inventory(),
        regime_scalars=_synthetic_scalars(),
        ood_box=_synthetic_box(),
    )


# ---------------------------------------------------------------------------
# Combo shot-counting
# ---------------------------------------------------------------------------


def test_combo_shots_target_required() -> None:
    """A shot without the Dα target (xim) is never co-available."""
    inv = _synthetic_inventory()
    combo = ii.InputCombo(key="v0", label="v0", require_groups=(*ii.MAG_GROUPS, "ane"))
    shots = ii._combo_shots(inv, combo)
    assert 20007 not in shots  # has mag+ane+ams but no xim


def test_combo_shots_any_of_semantics() -> None:
    """TS-any requires at least one Thomson system present."""
    inv = _synthetic_inventory()
    combo = ii.InputCombo(
        key="core",
        label="core",
        require_groups=(*ii.MAG_GROUPS, "ane", "ams"),
        any_of=("ayc", "atm", "aye"),
    )
    shots = ii._combo_shots(inv, combo)
    # 20001 (ayc), 20002 (atm), 20006 (aye) qualify; 20003 (MSE no TS) does not
    assert 20003 not in shots
    assert {20001, 20002, 20006}.issubset(set(shots))


def test_gs_envelope_is_subset_and_filters() -> None:
    """GS-envelope restriction drops out-of-range and efm-less shots."""
    inv = _synthetic_inventory()
    combo = ii.InputCombo(
        key="core",
        label="core",
        require_groups=(*ii.MAG_GROUPS, "ane", "ams"),
        any_of=("ayc", "atm", "aye"),
    )
    total = set(ii._combo_shots(inv, combo, restrict_gs_envelope=False))
    gs = set(ii._combo_shots(inv, combo, restrict_gs_envelope=True))
    assert gs <= total  # subset
    assert 40001 in total and 40001 not in gs  # out of shot-range envelope
    assert 20006 in total and 20006 not in gs  # no efm


# ---------------------------------------------------------------------------
# Matrix invariants
# ---------------------------------------------------------------------------


def test_matrix_monotone_non_increasing(
    report: ii.IntegratedFeasibilityReport,
) -> None:
    """Adding inputs never increases the co-available corpus."""
    by_key = {r["key"]: r["n_total_coavailable"] for r in report.coavailability_matrix}
    assert by_key["v0_baseline"] >= by_key["core_mse"]
    assert by_key["core_mse"] >= by_key["core_internal"]
    assert by_key["core_internal"] >= by_key["core_plus_emission"]
    assert by_key["core_plus_emission"] >= by_key["full"]


def test_matrix_gs_le_total(report: ii.IntegratedFeasibilityReport) -> None:
    for r in report.coavailability_matrix:
        assert r["n_gs_envelope"] <= r["n_total_coavailable"]
        # train+cal+ood reconcile against the GS-envelope corpus
        assert (
            r["n_train"] + r["n_calibration"] + r["n_test_ood_regime"]
            == (r["n_gs_envelope"])
        )


def test_locked_box_applied_identically(
    report: ii.IntegratedFeasibilityReport,
) -> None:
    """Every combo reports against the same locked OOD box (in meta)."""
    assert report.meta["ood_box"]["ip_min_kA"] == pytest.approx(667.0)
    assert "by-current-density" in report.meta["regime_split_axis"]


def test_cal_viability_flag() -> None:
    """The cal_adequate flag fires exactly at the 200-shot floor."""
    assert ii.CAL_VIABILITY_FLOOR == 200
    # construct a combo whose GS corpus is large enough that cal >= 200
    n = 2000
    inv_groups = {
        sid: tuple(sorted(["ama", "amb", "amc", "ane", "xim", "efm"]))
        for sid in range(20000, 20000 + n)
    }
    inv = InventoryResult(
        shot_groups=inv_groups,
        all_groups=sorted({g for v in inv_groups.values() for g in v}),
        n_shots=n,
    )
    scalars = {sid: {"ip_mean": 400.0, "ne_mean": 5.0e19} for sid in inv_groups}
    rep = ii.build_feasibility(
        inventory=inv, regime_scalars=scalars, ood_box=_synthetic_box()
    )
    v0 = next(r for r in rep.coavailability_matrix if r["key"] == "v0_baseline")
    # 2000 shots × 0.12 cal_fraction ≈ 240 > 200 → adequate
    assert v0["n_calibration"] >= 200
    assert v0["cal_adequate"] is True


# ---------------------------------------------------------------------------
# IR absence + camera plan + recommendation + serialisation
# ---------------------------------------------------------------------------


def test_ir_excluded(report: ii.IntegratedFeasibilityReport) -> None:
    assert report.ir_absence["decision"] == "EXCLUDED"
    assert report.ir_absence["groups"]["rir"] >= 1  # 20008 carries rir
    assert report.ir_absence["groups"]["rit"] == 0


def test_camera_feature_plan_no_raw_pixels(
    report: ii.IntegratedFeasibilityReport,
) -> None:
    plan = report.camera_feature_plan
    assert "raw" in plan["principle"].lower()
    keys = {o["name"] for o in plan["options"]}
    assert {"camera_boundary_edge", "magvit2_token_pool"} <= keys


def test_recommendation_has_options(
    report: ii.IntegratedFeasibilityReport,
) -> None:
    rec = report.recommendation
    assert 2 <= len(rec["options"]) <= 3
    # the recommended option is the internal-profile core (MSE + Thomson)
    assert rec["options"][0]["combo_key"] == "core_internal"
    assert "MSE" in rec["headline"]


def test_alignment_plan_flags_sparse_thomson(
    report: ii.IntegratedFeasibilityReport,
) -> None:
    """Thomson/CXRS (slower than the grid) get a hold/sparse-likelihood note."""
    by_diag = {a["diagnostic"]: a for a in report.alignment_plan}
    ts = by_diag["Thomson any (ayc|atm|aye)"]
    assert ts["strategy"] == "hold_at_native_cadence"
    assert ts["native_hz"] < report.meta["engine_grid_hz"]
    sxr = by_diag["soft-xray (xsx)"]
    assert "aggregate" in sxr["strategy"]


def test_serialisation_roundtrip(
    report: ii.IntegratedFeasibilityReport, tmp_path: Path
) -> None:
    out = tmp_path / "feasibility.json"
    report.save(out)
    d = json.loads(out.read_text(encoding="utf-8"))
    assert d["schema"] == "gs_integrated_feasibility_v0"
    assert len(d["coavailability_matrix"]) == len(ii.default_combos())
    assert "recommendation" in d and "options" in d["recommendation"]
