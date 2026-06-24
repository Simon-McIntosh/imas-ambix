"""Unit tests for the screened eval-only gate cohort.

These cover the pure-numpy / filesystem parts — the train-disjoint candidate
enumeration, the training-shot extraction, the binding LEAKAGE GUARD, and the
brightness/motion SCREEN LOGIC (scored on synthetic decoded gray stacks, no VQ
decode) — so the cohort builder is green BEFORE a GPU decode runs.  The actual
GT-decode subprocess needs the VQ stack and is exercised on a compute node.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from imas_ambix.worldmodel.gate_cohort import (
    CohortShotScreen,
    DriveabilityThresholds,
    ScreenThresholds,
    _apply_driveability,
    _centroid_path_length,
    _command_change_profile,
    _gt_window_stats,
    _score_gt,
    _select_cohort,
    _select_driveable_cohort,
    _temporal_association,
    assert_disjoint,
    build_driveable_cohort,
    build_screened_cohort,
    driveability_score,
    enumerate_candidate_shots,
    load_cohort,
    training_shot_ids,
)

# ---------------------------------------------------------------------------
# training_shot_ids — read the manifest's shot set
# ---------------------------------------------------------------------------


def test_training_shot_ids_reads_dict_windows(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "windows": [
                    {"shot_id": 100, "camera_id": "rbb"},
                    {"shot_id": 100, "camera_id": "rco"},  # dup shot, one id
                    {"shot_id": 101, "camera_id": "rbb"},
                ],
                "held_out": [18502, 18503],
            }
        )
    )
    ids = training_shot_ids(manifest)
    assert ids == {100, 101, 18502, 18503}


def test_training_shot_ids_reads_list_schema_windows(tmp_path):
    # the list-row schema: shot_id is column 0.
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": ["shot_id", "camera_id", "start_frame"],
                "windows": [[200, "rbb", 0], [201, "rbb", 5]],
            }
        )
    )
    assert training_shot_ids(manifest) == {200, 201}


# ---------------------------------------------------------------------------
# leakage guard
# ---------------------------------------------------------------------------


def test_assert_disjoint_passes_when_no_overlap():
    assert_disjoint([1, 2, 3], {4, 5, 6})  # no raise


def test_assert_disjoint_raises_on_leak():
    with pytest.raises(AssertionError) as exc:
        assert_disjoint([1, 2, 7], {7, 8, 9})
    assert "7" in str(exc.value)
    assert "leakage" in str(exc.value).lower()


# ---------------------------------------------------------------------------
# candidate enumeration — train-disjoint + has the camera recording
# ---------------------------------------------------------------------------


def _mk_token_tree(root: Path, shots, *, camera="rbb", with_mag=None):
    """Fake the token store layout: frames/<shot>/<cam>.zarr + signals-magnetics."""
    with_mag = shots if with_mag is None else with_mag
    frames = root / "v1" / "frames"
    mag = root / "v1" / "signals-magnetics" / "magnetics"
    for s in shots:
        (frames / str(s) / f"{camera}.zarr").mkdir(parents=True)
    for s in with_mag:
        (mag / str(s) / "magnetics.zarr").mkdir(parents=True)


def test_enumerate_excludes_training_shots(tmp_path):
    _mk_token_tree(tmp_path, [10, 11, 12, 13])
    cand = enumerate_candidate_shots(
        token_root=tmp_path, camera="rbb", train_ids={11, 13}, cap=10
    )
    assert cand == [10, 12]  # train shots dropped, sorted ascending


def test_enumerate_requires_camera_and_magnetics(tmp_path):
    # shot 20 has rbb + mag; 21 has rbb but NO mag; 22 has neither (no rbb dir).
    _mk_token_tree(tmp_path, [20, 21], with_mag=[20])
    (tmp_path / "v1" / "frames" / "22").mkdir(parents=True)  # dir but no rbb.zarr
    cand = enumerate_candidate_shots(
        token_root=tmp_path, camera="rbb", train_ids=set(), cap=10
    )
    assert cand == [20]  # 21 dropped (no mag), 22 dropped (no rbb recording)
    # without the magnetics requirement, 21 is admitted too.
    cand2 = enumerate_candidate_shots(
        token_root=tmp_path,
        camera="rbb",
        train_ids=set(),
        cap=10,
        require_magnetics=False,
    )
    assert cand2 == [20, 21]


def test_enumerate_caps_candidate_count(tmp_path):
    _mk_token_tree(tmp_path, list(range(30, 50)))
    cand = enumerate_candidate_shots(
        token_root=tmp_path, camera="rbb", train_ids=set(), cap=5
    )
    assert len(cand) == 5
    assert cand == [30, 31, 32, 33, 34]  # deterministic ascending


# ---------------------------------------------------------------------------
# load_cohort
# ---------------------------------------------------------------------------


def test_load_cohort_reads_ids(tmp_path):
    p = tmp_path / "cohort.json"
    p.write_text(json.dumps({"cohort": [101, 202, 303], "summary": {}}))
    assert load_cohort(p) == [101, 202, 303]


def test_screen_thresholds_defaults_kill_dark_and_static():
    """The default thresholds reject the known degenerate gate shots by metric."""
    t = ScreenThresholds()
    # 18504 is dark (mean 2.7) and near-static (motion 2.3) -> both fail.
    assert t.min_brightness > 2.7
    assert t.min_transient_motion > 2.3
    # 18502 is bright (17) and moving (8.2) -> both pass.
    assert t.min_brightness <= 17.0
    assert t.min_transient_motion <= 8.2


# ---------------------------------------------------------------------------
# brightness/motion SCREEN LOGIC — scored on synthetic decoded gray stacks
# (the same gate the GPU builder applies, exercised with no VQ decode)
# ---------------------------------------------------------------------------


def _bright_moving_stack(n_frames=24, hw=256) -> np.ndarray:
    """A decoded GT gray stack that is BRIGHT and MOVING (a fair probe).

    The whole frame's brightness sweeps low<->high each frame, so the mean
    forecast-window brightness is well above the gate AND the frame-to-frame
    pixel-L1 (transient motion) is large — a fair controllability probe.
    """
    base = np.zeros((n_frames, hw, hw), dtype=np.float64)
    for f in range(n_frames):
        # alternate the whole field between bright and very-bright -> big L1.
        base[f] = 60.0 if (f % 2 == 0) else 180.0
    return base


def _dark_static_stack(n_frames=24, hw=256) -> np.ndarray:
    """A decoded GT gray stack that is DARK and STATIC (a degenerate probe)."""
    return np.full((n_frames, hw, hw), 2.0)


def _rec(shot_id=900, *, context_frames=8, plan_variation=1.0, n_streams=12):
    return CohortShotScreen(
        shot_id=shot_id,
        assemblable=True,
        passed=False,
        context_frames=context_frames,
        plan_variation=plan_variation,
        n_streams=n_streams,
    )


def test_screen_passes_bright_moving_candidate():
    rec = _score_gt(_rec(), _bright_moving_stack(), ScreenThresholds())
    assert rec.passed, rec.reason
    assert rec.mean_brightness >= ScreenThresholds().min_brightness
    assert rec.transient_motion >= ScreenThresholds().min_transient_motion


def test_screen_rejects_dark_static_candidate():
    rec = _score_gt(_rec(), _dark_static_stack(), ScreenThresholds())
    assert not rec.passed
    assert "brightness" in rec.reason
    assert "motion" in rec.reason


def test_screen_only_scores_forecast_window():
    """Brightness/motion are measured on frames AFTER context_frames only."""
    # bright + moving in context, dark + static in the forecast window.
    stack = _dark_static_stack()
    stack[:8] = _bright_moving_stack()[:8]
    rec = _score_gt(_rec(context_frames=8), stack, ScreenThresholds())
    assert not rec.passed  # the forecast window (frames 8:) is dark+static


def test_screen_rejects_low_plan_variation_and_low_streams():
    t = ScreenThresholds()
    rec = _score_gt(_rec(plan_variation=0.0, n_streams=1), _bright_moving_stack(), t)
    assert not rec.passed
    assert "plan_var" in rec.reason
    assert "streams" in rec.reason


def test_gt_window_stats_matches_manual():
    stack = _bright_moving_stack()
    mean_bri, p99_bri, motion = _gt_window_stats(stack, 8)
    fwin = stack[8:]
    assert mean_bri == pytest.approx(float(fwin.mean()))
    assert motion == pytest.approx(float(np.abs(np.diff(fwin, axis=0)).mean()))
    assert p99_bri >= mean_bri


# ---------------------------------------------------------------------------
# cohort selection + threshold relaxation (re-scores cached stats; no decode)
# ---------------------------------------------------------------------------


def _stat_rec(shot_id, mean_bri, motion, *, ctx=8, plan_variation=1.0, n_streams=12):
    r = _rec(
        shot_id=shot_id,
        context_frames=ctx,
        plan_variation=plan_variation,
        n_streams=n_streams,
    )
    r.mean_brightness = mean_bri
    r.p99_brightness = mean_bri
    r.transient_motion = motion
    return r


def test_select_cohort_keeps_passers_up_to_target():
    recs = [
        _stat_rec(1, 30.0, 8.0),  # pass
        _stat_rec(2, 30.0, 8.0),  # pass
        _stat_rec(3, 2.0, 1.0),  # dark+static fail
        _stat_rec(4, 30.0, 8.0),  # pass
    ]
    kept = _select_cohort(recs, ScreenThresholds(), target_size=10)
    assert kept == [1, 2, 4]
    # target cap is honoured.
    assert _select_cohort(recs, ScreenThresholds(), target_size=2) == [1, 2]


def test_relax_step_lowers_brightness_and_motion_toward_floor():
    t = ScreenThresholds(min_brightness=10.0, min_transient_motion=4.0)
    r1 = t.relax_step()
    assert r1.min_brightness < t.min_brightness
    assert r1.min_transient_motion < t.min_transient_motion
    assert r1.min_brightness >= r1.min_brightness_floor
    # plan-variation + stream gates are NEVER relaxed.
    assert r1.min_plan_variation == t.min_plan_variation
    assert r1.min_streams == t.min_streams
    # repeated relaxation reaches the floor.
    cur = t
    for _ in range(50):
        cur = cur.relax_step()
    assert cur.at_floor()


def test_relaxation_recovers_borderline_shots():
    """A shot that fails the default motion gate (4.0) passes after relaxation."""
    # motion 3.7 < default 4.0 but >= the first relax step (floor 3.0 + 0.5*1.0 = 3.5).
    recs = [_stat_rec(1, 30.0, 3.7)]
    assert _select_cohort(recs, ScreenThresholds(), target_size=5) == []
    relaxed = ScreenThresholds().relax_step()  # motion gate -> 3.5
    assert relaxed.min_transient_motion == pytest.approx(3.5)
    assert _select_cohort(recs, relaxed, target_size=5) == [1]


# ---------------------------------------------------------------------------
# public signature of build_screened_cohort is preserved (eval imports it)
# ---------------------------------------------------------------------------


def test_build_screened_cohort_signature_preserved():
    sig = inspect.signature(build_screened_cohort)
    params = sig.parameters
    # the eval's CLI call binds these by name — they must not be renamed/removed.
    assert "cfg" in params
    for kw in (
        "camera",
        "token_root",
        "manifest_path",
        "device",
        "out_json",
        "thresholds",
        "candidate_cap",
        "target_size",
        "work_dir",
    ):
        assert kw in params, f"missing build_screened_cohort kwarg: {kw}"
    # cfg is positional; the rest are keyword-only.
    assert params["cfg"].kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.POSITIONAL_ONLY,
    )
    for kw in ("camera", "token_root", "manifest_path", "device"):
        assert params[kw].kind == inspect.Parameter.KEYWORD_ONLY


# ===========================================================================
# DRIVEABLE-enriched cohort — additive, model-free driveability screen
# (default OFF: build_screened_cohort is unchanged; this is a sibling builder)
# ===========================================================================


def _moving_blob_stack(n_frames=24, hw=64, step=2.0, r0=10.0) -> np.ndarray:
    """A decoded GT gray stack with a bright blob that TRANSLATES across frames.

    The emission centroid travels along a diagonal at ``step`` px/frame, so the
    decoded-centroid path length over the forecast window is large — a genuinely
    MOVING (driveable-response) plasma.
    """
    yy, xx = np.mgrid[0:hw, 0:hw].astype(np.float64)
    out = np.zeros((n_frames, hw, hw), dtype=np.float64)
    for f in range(n_frames):
        cy = r0 + step * f
        cx = r0 + step * f
        out[f] = 220.0 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 4.0**2))
    return np.clip(out, 0.0, 255.0)


def _static_blob_stack(n_frames=24, hw=64, r0=32.0) -> np.ndarray:
    """A bright blob fixed at the centre — bright but the centroid does NOT travel."""
    yy, xx = np.mgrid[0:hw, 0:hw].astype(np.float64)
    blob = 220.0 * np.exp(-((yy - r0) ** 2 + (xx - r0) ** 2) / (2 * 4.0**2))
    return np.clip(np.broadcast_to(blob, (n_frames, hw, hw)).copy(), 0.0, 255.0)


# --- centroid path length (reuses controllable_eval.decoded_centroid) ---


def test_centroid_path_long_for_moving_blob():
    path = _centroid_path_length(_moving_blob_stack(step=2.0), context_frames=8)
    # 15 forecast steps * sqrt(2)*2 px/step ~ 42 px; assert it is clearly nonzero.
    assert path > 10.0


def test_centroid_path_near_zero_for_static_blob():
    path = _centroid_path_length(_static_blob_stack(), context_frames=8)
    assert path < 1.0


# --- command-change profile (demanded actuator plan only; model-free) ---


def test_command_change_profile_zero_for_flat_plan():
    vals = np.ones((8, 3), dtype=np.float64)  # flat-top: no command change
    miss = np.zeros((8, 3), dtype=np.float64)
    prof = _command_change_profile(vals, miss)
    assert prof.shape == (7,)
    assert float(prof.sum()) == pytest.approx(0.0)


def test_command_change_profile_tracks_present_channels_only():
    vals = np.zeros((4, 2), dtype=np.float64)
    vals[:, 0] = [0.0, 1.0, 1.0, 3.0]  # present, changes
    vals[:, 1] = [0.0, 5.0, 0.0, 9.0]  # MISSING — must be ignored
    miss = np.zeros((4, 2), dtype=np.float64)
    miss[:, 1] = 1.0  # channel 1 missing for the whole window
    prof = _command_change_profile(vals, miss)
    # only channel 0 contributes: |1-0|+|1-1|+|3-1| = 1+0+2 = 3.
    assert float(prof.sum()) == pytest.approx(3.0)


# --- temporal-association proxy (when commands change vs when plasma moves) ---


def test_temporal_association_high_when_aligned():
    prof = np.array([0.0, 5.0, 0.0, 0.0], dtype=np.float64)
    assert _temporal_association(prof, prof.copy()) == pytest.approx(1.0, abs=1e-6)


def test_temporal_association_low_when_misaligned():
    cmd = np.array([5.0, 0.0, 0.0, 0.0], dtype=np.float64)
    cen = np.array([0.0, 0.0, 0.0, 5.0], dtype=np.float64)
    assert _temporal_association(cmd, cen) < 0.2


def test_temporal_association_zero_for_empty_profile():
    assert _temporal_association(np.zeros((0,)), np.array([1.0, 2.0])) == 0.0


# --- combined driveability score: anti-circular, model-free ranking signal ---


def test_driveability_high_for_moving_commanded_plasma():
    """Commands move AND GT plasma moves + centroid travels + aligned -> HIGH."""
    s = driveability_score(
        plan_variation=5.0,
        transient_motion=12.0,
        centroid_path_px=40.0,
        temporal_association=0.95,
    )
    assert s > 0.4


def test_driveability_low_for_flat_command_plasma():
    """Plasma moves but NO command variation -> not driveable (nothing to drive)."""
    s = driveability_score(
        plan_variation=0.0,
        transient_motion=12.0,
        centroid_path_px=40.0,
        temporal_association=0.95,
    )
    assert s < 0.05


def test_driveability_low_for_static_plasma():
    """Commands move but the GT plasma is static -> not driveable (no response)."""
    s = driveability_score(
        plan_variation=5.0,
        transient_motion=0.2,
        centroid_path_px=0.1,
        temporal_association=0.95,
    )
    assert s < 0.05


def test_driveability_moving_beats_flat_and_static():
    moving = driveability_score(
        plan_variation=5.0,
        transient_motion=12.0,
        centroid_path_px=40.0,
        temporal_association=0.95,
    )
    flat = driveability_score(
        plan_variation=1e-4,
        transient_motion=12.0,
        centroid_path_px=40.0,
        temporal_association=0.95,
    )
    static = driveability_score(
        plan_variation=5.0,
        transient_motion=0.1,
        centroid_path_px=0.05,
        temporal_association=0.95,
    )
    assert moving > flat
    assert moving > static


def test_apply_driveability_fills_components_from_profiles():
    rec = CohortShotScreen(
        shot_id=7,
        assemblable=True,
        passed=False,
        plan_variation=5.0,
        transient_motion=12.0,
        n_streams=12,
    )
    rec.centroid_path_px = 40.0
    cmd = np.array([0.0, 4.0, 0.0, 0.0])
    cen = np.array([0.0, 4.0, 0.0, 0.0])
    _apply_driveability(rec, command_profile=cmd, centroid_profile=cen)
    assert rec.command_change_total == pytest.approx(4.0)
    assert rec.temporal_association == pytest.approx(1.0, abs=1e-6)
    assert rec.driveability_score > 0.4


# --- eligibility gates + top-N ranking selection ---


def _drive_rec(shot_id, *, plan_var, motion, cen_path, bri=30.0, n_streams=12):
    r = CohortShotScreen(
        shot_id=shot_id,
        assemblable=True,
        passed=False,
        plan_variation=plan_var,
        transient_motion=motion,
        n_streams=n_streams,
    )
    r.mean_brightness = bri
    r.centroid_path_px = cen_path
    r.temporal_association = 0.9
    _apply_driveability(r)
    return r


def test_eligibility_rejects_dark_flat_static_and_few_streams():
    t = DriveabilityThresholds()
    # dark
    assert "brightness" in "; ".join(
        t.eligibility_fails(_drive_rec(1, plan_var=5, motion=8, cen_path=20, bri=2.0))
    )
    # flat command
    assert "plan_var" in "; ".join(
        t.eligibility_fails(_drive_rec(2, plan_var=0.0, motion=8, cen_path=20))
    )
    # static plasma (no centroid travel)
    assert "centroid_path" in "; ".join(
        t.eligibility_fails(_drive_rec(3, plan_var=5, motion=8, cen_path=0.0))
    )
    # too few streams
    assert "streams" in "; ".join(
        t.eligibility_fails(
            _drive_rec(4, plan_var=5, motion=8, cen_path=20, n_streams=1)
        )
    )
    # a driveable candidate is eligible.
    assert t.eligibility_fails(_drive_rec(5, plan_var=5, motion=8, cen_path=20)) == []


def test_select_driveable_keeps_top_scoring_eligible():
    recs = [
        _drive_rec(10, plan_var=5.0, motion=12.0, cen_path=40.0),  # high score
        _drive_rec(11, plan_var=0.0, motion=12.0, cen_path=40.0),  # flat -> ineligible
        _drive_rec(12, plan_var=2.0, motion=6.0, cen_path=10.0),  # mid score
        _drive_rec(13, plan_var=5.0, motion=12.0, cen_path=35.0),  # high score
        _drive_rec(14, plan_var=5.0, motion=0.1, cen_path=0.0),  # static -> ineligible
    ]
    kept = _select_driveable_cohort(recs, DriveabilityThresholds(), target_size=2)
    # the two highest-driveability ELIGIBLE shots, flat/static excluded.
    assert set(kept) == {10, 13}
    # target cap honoured + ranked (highest first).
    kept3 = _select_driveable_cohort(recs, DriveabilityThresholds(), target_size=3)
    assert kept3[0] in (10, 13) and kept3[-1] == 12
    assert 11 not in kept3 and 14 not in kept3


def test_select_driveable_is_deterministic_tie_break():
    # identical scores -> ascending shot-id tie-break.
    recs = [
        _drive_rec(30, plan_var=5.0, motion=12.0, cen_path=40.0),
        _drive_rec(20, plan_var=5.0, motion=12.0, cen_path=40.0),
    ]
    kept = _select_driveable_cohort(recs, DriveabilityThresholds(), target_size=2)
    assert kept == [20, 30]


# --- back-compat: the screened builder + its API are UNCHANGED (default OFF) ---


def test_screened_cohort_default_path_unchanged():
    """The driveable builder writes a SEPARATE file; the screened default stands."""
    from imas_ambix.worldmodel.gate_cohort import (
        DEFAULT_COHORT_PATH,
        DEFAULT_DRIVEABLE_COHORT_PATH,
    )

    assert DEFAULT_COHORT_PATH != DEFAULT_DRIVEABLE_COHORT_PATH
    assert DEFAULT_COHORT_PATH.name == "gate_cohort.json"
    assert DEFAULT_DRIVEABLE_COHORT_PATH.name == "gate_cohort_driveable.json"


def test_build_driveable_cohort_signature():
    sig = inspect.signature(build_driveable_cohort)
    params = sig.parameters
    assert "cfg" in params
    for kw in (
        "camera",
        "token_root",
        "manifest_path",
        "device",
        "out_json",
        "thresholds",
        "candidate_cap",
        "target_size",
        "work_dir",
    ):
        assert kw in params, f"missing build_driveable_cohort kwarg: {kw}"
    # out_json defaults to the NEW driveable path (never the screened cohort).
    from imas_ambix.worldmodel.gate_cohort import DEFAULT_DRIVEABLE_COHORT_PATH

    assert params["out_json"].default == DEFAULT_DRIVEABLE_COHORT_PATH


def test_screened_scoring_unaffected_by_driveable_fields():
    """CohortShotScreen with driveability fields still screens on brightness/motion."""
    rec = _score_gt(_rec(), _bright_moving_stack(), ScreenThresholds())
    assert rec.passed
    # the new fields default to 0.0 and do not affect the brightness/motion screen.
    assert rec.driveability_score == 0.0
    assert rec.centroid_path_px == 0.0
