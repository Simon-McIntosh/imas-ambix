"""Tests for the model-independent excitation-window selector + excited cohort.

These prove the identifiability device the selector exists to provide — all on
a synthetic plan/coil stream, CPU-only, no ``/work``:

(a) a high-``|dI/dt|`` ramping window scores ABOVE a flat-top window (the
    excitation discriminator is level-invariant);
(b) the selector picks the excited windows and orders them most-excited first;
(c) the leakage audit RAISES when an excited shot leaks into the training set and
    PASSES (returns a disjoint cohort) when train/eval are disjoint;
(d) the locked held-out family 18502-18505 is retained as held-out (and the audit
    raises if it leaks into training);
(e) the plan-change excitation measure separates a stepped plan from a flat one.
"""

from __future__ import annotations

import numpy as np
import pytest

from imas_ambix.eval.excitation_selector import (
    DEFAULT_EXCITATION_THRESHOLD,
    LOCKED_HELD_OUT,
    ScoredWindow,
    assemble_excited_cohort,
    coil_ramp_profile,
    is_excited,
    plan_change_excitation,
    score_windows_from_streams,
    select_excited_windows,
    window_excitation_score,
)

# ---------------------------------------------------------------------------
# Synthetic streams
# ---------------------------------------------------------------------------


def _uniform_time(n: int, dt: float = 1e-3) -> np.ndarray:
    """A uniform frame-time axis (s)."""
    return np.arange(n, dtype=np.float64) * dt


def _ramp_then_flat(n_ramp: int, n_flat: int, *, slope: float = 1.0e5) -> np.ndarray:
    """A single coil channel: a steep ramp then a high but FLAT hold.

    The flat hold sits at a LARGE current level — the test that ramp > flat must
    therefore rely on the score being level-invariant (``|dI/dt|``), not on the
    absolute current value.
    """
    ramp = np.arange(n_ramp, dtype=np.float64) * slope * 1e-3  # rises over the ramp
    hold = np.full(n_flat, ramp[-1] if n_ramp else 0.0, dtype=np.float64)
    return np.concatenate([ramp, hold])[:, None]


# ---------------------------------------------------------------------------
# (a) ramping window scores above a flat-top window
# ---------------------------------------------------------------------------


def test_ramp_window_scores_above_flat_top():
    n_ramp, n_flat = 20, 20
    coil = _ramp_then_flat(n_ramp, n_flat)
    ftime = _uniform_time(coil.shape[0])
    ramp = coil_ramp_profile(coil, ftime)

    ramp_score = window_excitation_score(ramp[:n_ramp])
    flat_score = window_excitation_score(ramp[n_ramp:])

    assert ramp_score > flat_score
    # the flat HOLD is at a large current but its |dI/dt| ~ 0 -> not excited.
    assert flat_score < ramp_score
    assert is_excited(ramp_score)  # the ramp clears the default excited threshold
    assert not is_excited(flat_score)


def test_excitation_is_level_invariant():
    """A flat current at a HUGE level still scores ~0 (level-invariant)."""
    ftime = _uniform_time(16)
    huge_flat = np.full((16, 1), 1.0e7, dtype=np.float64)
    ramp = coil_ramp_profile(huge_flat, ftime)
    assert window_excitation_score(ramp) == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# (b) the selector picks the excited windows, most-excited first
# ---------------------------------------------------------------------------


def test_selector_picks_excited_windows():
    # ramp (frames 0..19) then flat (20..39); span 10, stride 5.
    coil = _ramp_then_flat(20, 20)
    ftime = _uniform_time(coil.shape[0])
    scored = score_windows_from_streams(coil, ftime, span=10, shot_id=42, stride=5)
    assert scored, "expected sliding windows"

    excited = select_excited_windows(scored)
    # at least the early ramping windows are kept; the late flat windows are not.
    assert excited, "expected some excited windows"
    starts = {w.start_frame for w in excited}
    assert 0 in starts  # the first ramp window is excited
    # the deep flat-top window (start 30, all-flat hold) is NOT excited.
    flat_only = [w for w in scored if w.start_frame == 30]
    if flat_only:
        assert not flat_only[0].excited

    # most-excited first ordering.
    scores = [w.excitation_score for w in excited]
    assert scores == sorted(scores, reverse=True)


def test_selector_limit_keeps_top_n():
    coil = _ramp_then_flat(30, 10)
    ftime = _uniform_time(coil.shape[0])
    scored = score_windows_from_streams(coil, ftime, span=8, shot_id=7, stride=4)
    top2 = select_excited_windows(scored, limit=2)
    assert len(top2) <= 2
    if len(top2) == 2:
        assert top2[0].excitation_score >= top2[1].excitation_score


# ---------------------------------------------------------------------------
# (c) leakage audit: raises on leak, passes on disjoint
# ---------------------------------------------------------------------------


def test_leakage_audit_passes_when_disjoint():
    # held-out candidate scores: 3 excited, 1 flat-top.
    big = 2.0 * DEFAULT_EXCITATION_THRESHOLD
    candidate_scores = {
        20001: big,
        20002: big * 1.5,
        20003: big * 0.6,
        20004: 0.1 * DEFAULT_EXCITATION_THRESHOLD,  # flat -> rejected
    }
    train_ids = {30001, 30002, 30003}  # fully disjoint
    cohort = assemble_excited_cohort(candidate_scores, train_ids)

    assert cohort.disjoint is True
    assert set(cohort.shot_ids) == {20001, 20002, 20003}
    assert cohort.rejected_flat == [20004]
    # most-excited first.
    assert cohort.shot_ids[0] == 20002
    assert cohort.n_train_shots == 3
    dist = cohort.score_distribution_summary()
    assert dist["n_excited"] == 3
    assert dist["n_flat"] == 1


def test_leakage_audit_raises_when_excited_shot_leaks():
    big = 2.0 * DEFAULT_EXCITATION_THRESHOLD
    candidate_scores = {20001: big, 20002: big}
    # 20002 is ALSO a training shot -> leak.
    train_ids = {30001, 20002}
    with pytest.raises(AssertionError, match="leakage"):
        assemble_excited_cohort(candidate_scores, train_ids)


def test_flat_top_shot_excluded_from_cohort():
    candidate_scores = {
        20001: 0.0,  # flat
        20002: 0.5 * DEFAULT_EXCITATION_THRESHOLD,  # below threshold -> flat
    }
    cohort = assemble_excited_cohort(candidate_scores, train_ids={9999})
    assert cohort.shot_ids == []
    assert sorted(cohort.rejected_flat) == [20001, 20002]


# ---------------------------------------------------------------------------
# (d) locked held-out family 18502-18505 retained as held-out
# ---------------------------------------------------------------------------


def test_locked_family_is_18502_to_18505():
    assert LOCKED_HELD_OUT == (18502, 18503, 18504, 18505)


def test_locked_family_retained_as_held_out_when_excited():
    big = 2.0 * DEFAULT_EXCITATION_THRESHOLD
    # every locked held-out shot is excited and the train set is disjoint.
    candidate_scores = {s: big for s in LOCKED_HELD_OUT}
    train_ids = {30001, 30002}
    cohort = assemble_excited_cohort(candidate_scores, train_ids)
    assert set(cohort.shot_ids) == set(LOCKED_HELD_OUT)
    assert cohort.locked_held_out == LOCKED_HELD_OUT
    # none of the locked family appears in the training set.
    assert not (set(LOCKED_HELD_OUT) & set(train_ids))


def test_audit_raises_if_locked_family_leaks_into_training():
    candidate_scores = {20001: 2.0 * DEFAULT_EXCITATION_THRESHOLD}
    # 18503 (a locked held-out shot) wrongly placed in the training set.
    train_ids = {30001, 18503}
    with pytest.raises(AssertionError, match="locked held-out"):
        assemble_excited_cohort(candidate_scores, train_ids)


# ---------------------------------------------------------------------------
# (e) plan-change excitation separates stepped from flat plans
# ---------------------------------------------------------------------------


def test_plan_change_excitation_separates_stepped_from_flat():
    n_plan, n_ch = 8, 3
    flat = np.ones((n_plan, n_ch), dtype=np.float64)
    stepped = flat.copy()
    stepped[n_plan // 2 :, 0] += 5.0  # a step in channel 0 mid-window
    miss = np.zeros((n_plan, n_ch), dtype=np.float64)  # all present

    flat_ex = plan_change_excitation(flat, miss)
    step_ex = plan_change_excitation(stepped, miss)
    assert step_ex > flat_ex
    assert flat_ex == pytest.approx(0.0, abs=1e-12)


def test_plan_change_excitation_ignores_missing_channels():
    n_plan, n_ch = 6, 2
    vals = np.zeros((n_plan, n_ch), dtype=np.float64)
    vals[3:, 1] += 10.0  # the only varying channel...
    miss = np.zeros((n_plan, n_ch), dtype=np.float64)
    miss[:, 1] = 1.0  # ...is fully missing -> contributes nothing
    assert plan_change_excitation(vals, miss) == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# ScoredWindow.excited convenience
# ---------------------------------------------------------------------------


def test_scored_window_excited_flag():
    hot = ScoredWindow(shot_id=1, start_frame=0, excitation_score=1e4)
    cold = ScoredWindow(shot_id=1, start_frame=10, excitation_score=1.0)
    assert hot.excited is True
    assert cold.excited is False
