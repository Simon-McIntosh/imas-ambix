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
    ScreenThresholds,
    _gt_window_stats,
    _score_gt,
    _select_cohort,
    assert_disjoint,
    build_screened_cohort,
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
