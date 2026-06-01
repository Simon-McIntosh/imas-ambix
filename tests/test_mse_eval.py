"""Unit tests for the D1 held-out-MSE eval harness (S9 gate).

The core tests use SYNTHETIC data only — no GPFS / network — so they run in
CI.  One integration test (the full smoke on the real level-1 corpus) is
guarded by the presence of the GPFS mirror and skipped otherwise.

What is tested
--------------
* :func:`pitch_from_current_profile` — the shared forward observation model:
  monotonicity on the low-field side, the signed zero crossing at the axis,
  time-batching, and the ``kind='iota'`` path.
* :func:`invert_pitch_to_q0rax` — round-trips a synthetic profile: rax recovers
  the magnetic axis.
* :class:`ShotPrediction` shape validation.
* :func:`score` — finite metrics + correct nested structure on a synthetic
  manifest + truth, with the persistence baseline.
* The GATE assertion: a TRAIN partition that overlaps eval / contains MSE
  fails ``assert_no_mse_in_train``; a clean one passes.
"""

# Uppercase physics-symbol / matrix-dimension names (G, K, C, R0) + short module
# aliases (E, M) are intentional in these tests.
# ruff: noqa: N803, N812
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from imas_ambix.statespace import mse_eval as E
from imas_ambix.statespace import mse_split as M

# ---------------------------------------------------------------------------
# Shared forward observation model
# ---------------------------------------------------------------------------


def _parabolic_profile(G=64, a=0.6, j0=1e6):
    rho = np.linspace(0.0, a, G)
    j = np.clip(j0 * (1.0 - (rho / a) ** 2), 0.0, None)
    return rho, j


def test_forward_monotone_on_lfs():
    rho, j = _parabolic_profile()
    sight = np.linspace(0.86, 1.30, 15)  # all outboard of R0=0.85
    pitch = E.pitch_from_current_profile(
        j, rho, sight, R0=0.85, B_tor0_or_Ip=0.5, kind="j"
    )
    assert pitch.shape == (15,)
    # peaked current → enclosed current grows with r → B_pol grows → pitch grows
    assert np.all(np.diff(pitch) >= -1e-6)
    assert np.all(pitch > 0)  # outboard side, positive pitch


def test_forward_signed_zero_crossing_at_axis():
    rho, j = _parabolic_profile()
    R0 = 0.85
    sight = np.linspace(0.70, 1.30, 15)  # spans the axis
    pitch = E.pitch_from_current_profile(
        j, rho, sight, R0=R0, B_tor0_or_Ip=0.5, kind="j"
    )
    # inboard (R<R0) pitch negative, outboard positive → crosses zero at axis
    assert pitch[sight < R0].max() < 0.05
    assert pitch[sight > R0].min() > -0.05
    assert np.any(pitch < 0) and np.any(pitch > 0)


def test_forward_time_batched():
    rho, j = _parabolic_profile()
    sight = np.linspace(0.70, 1.30, 12)
    jt = np.stack([j, 0.8 * j, 1.2 * j])  # (3, G)
    pt = E.pitch_from_current_profile(
        jt, rho, sight, R0=0.85, B_tor0_or_Ip=0.5, kind="j"
    )
    assert pt.shape == (3, 12)
    # higher current → steeper pitch
    assert np.nanmax(np.abs(pt[2])) >= np.nanmax(np.abs(pt[0]))


def test_forward_iota_kind():
    G = 64
    rho = np.linspace(0.0, 0.6, G)
    iota = 0.3 + 0.7 * rho / 0.6
    sight = np.linspace(0.70, 1.30, 12)
    pit = E.pitch_from_current_profile(
        iota, rho, sight, R0=0.85, B_tor0_or_Ip=0.5, kind="iota"
    )
    assert pit.shape == (12,)
    assert np.all(np.isfinite(pit))


def test_forward_rejects_bad_kind():
    rho, j = _parabolic_profile()
    with pytest.raises(ValueError):
        E.pitch_from_current_profile(j, rho, np.array([1.0]), 0.85, 0.5, kind="bogus")


# ---------------------------------------------------------------------------
# Inversion round-trip
# ---------------------------------------------------------------------------


def test_inversion_recovers_axis():
    rho, j = _parabolic_profile()
    R0 = 0.85
    sight = np.linspace(0.70, 1.30, 15)
    pitch = E.pitch_from_current_profile(
        j, rho, sight, R0=R0, B_tor0_or_Ip=0.5, kind="j"
    )
    q0, rax = E.invert_pitch_to_q0rax(
        pitch[np.newaxis, :], {"rpos": sight, "R0": R0, "Bt0": 0.5}
    )
    assert q0.shape == (1,) and rax.shape == (1,)
    assert np.isfinite(rax[0])
    assert abs(rax[0] - R0) < 0.06  # recovers the magnetic axis
    # q0 within the physical gate
    assert M.Q0_MIN <= q0[0] <= M.Q0_MAX


def test_inversion_all_nan_when_no_crossing():
    # all-positive pitch (no sign change) → no crossing → NaN
    pitch = np.array([[0.1, 0.2, 0.3, 0.4, 0.5]])
    sight = np.linspace(0.9, 1.3, 5)
    q0, rax = E.invert_pitch_to_q0rax(pitch, {"rpos": sight, "R0": 0.85, "Bt0": 0.5})
    assert np.isnan(q0[0]) and np.isnan(rax[0])


# ---------------------------------------------------------------------------
# PRIMARY pitch physical gate (rail + error)
# ---------------------------------------------------------------------------


def test_pitch_point_gate_drops_rails_and_high_error():
    pitch = np.array([[0.1, 1.6, 0.2, np.nan, 0.3]])  # idx1 railed, idx3 NaN
    perr = np.array([[0.02, 0.02, 0.5, 0.02, 0.1]])  # idx2 high-error
    gate = M.pitch_point_gate(pitch, perr)
    np.testing.assert_array_equal(gate, np.array([[True, False, False, False, True]]))


def test_pitch_point_gate_missing_error_passes_error_gate():
    pitch = np.array([[0.1, 0.2, 1.7]])  # idx2 railed
    perr = np.array([[np.nan, np.nan, np.nan]])  # no error → error gate passes
    gate = M.pitch_point_gate(pitch, perr)
    # rail gate still drops idx2; finite non-railed pass even w/o error
    np.testing.assert_array_equal(gate, np.array([[True, True, False]]))


def test_pitch_point_gate_custom_thresholds():
    pitch = np.array([[0.9, 1.2]])
    perr = np.array([[0.4, 0.1]])
    # default err_thresh=0.3 drops idx0; rail_thresh=1.5 keeps both
    np.testing.assert_array_equal(
        M.pitch_point_gate(pitch, perr), np.array([[False, True]])
    )
    # relax error, tighten rail → idx0 passes (err ok), idx1 dropped (railed)
    np.testing.assert_array_equal(
        M.pitch_point_gate(pitch, perr, err_thresh=0.5, rail_thresh=1.1),
        np.array([[True, False]]),
    )


def test_score_gate_drops_railed_truth_points():
    """A railed truth channel must not influence the primary pitch RMSE."""
    shots = {300: _make_synthetic_shot(300, seed=3)}
    # inject a rail-pinned, low-error truth point that the predictor gets wrong
    tr = shots[300]
    tr.pitch[0, 0] = 1.6  # rail
    tr.pitch_error[0, 0] = 0.01  # low error → would dominate if not gated
    manifest = _make_manifest(shots)
    truth = _SyntheticTruth(shots)
    entry = manifest["shots"]["300"]
    t = np.asarray(entry["beam_on_slice_times"])
    # predictor returns truth EXCEPT it is wrong at the railed point
    pm = tr.pitch.copy()
    pm[0, 0] = 0.0  # large error at the railed (gated-out) point
    pred = {
        300: E.ShotPrediction(
            t=t, pitch_mean=pm, pitch_std=np.full_like(tr.pitch, 0.05)
        )
    }
    result = E.score(pred, manifest, truth)
    # railed point is gated out → RMSE stays ~0 despite the injected error
    assert result["primary"]["pitch"]["rmse"] < 1e-3


# ---------------------------------------------------------------------------
# Contract validation
# ---------------------------------------------------------------------------


def test_shot_prediction_validate_ok():
    K, C = 10, 6
    p = E.ShotPrediction(
        t=np.arange(K, dtype=float),
        pitch_mean=np.zeros((K, C)),
        pitch_std=np.ones((K, C)),
    )
    p.validate(C)  # no raise


def test_shot_prediction_validate_bad_shape():
    K, C = 10, 6
    p = E.ShotPrediction(
        t=np.arange(K, dtype=float),
        pitch_mean=np.zeros((K, C + 1)),
        pitch_std=np.ones((K, C + 1)),
    )
    with pytest.raises(ValueError):
        p.validate(C)


# ---------------------------------------------------------------------------
# GATE: no-MSE-in-train assertion
# ---------------------------------------------------------------------------


def test_gate_train_disjoint_passes():
    split = M.MseSplit(
        train=[1, 2, 3],
        calibration=[10, 11],
        held_out=[20, 21],
        train_input_groups=["ama", "amc", "rbb", "abm", "xsx"],
    )
    split.assert_no_mse_in_train()  # disjoint + no ams → no raise


def test_gate_train_overlap_fails():
    split = M.MseSplit(
        train=[1, 2, 20],  # 20 is also held-out
        calibration=[10],
        held_out=[20],
        train_input_groups=["ama", "rbb"],
    )
    with pytest.raises(AssertionError):
        split.assert_no_mse_in_train()


def test_gate_train_has_mse_fails():
    split = M.MseSplit(
        train=[1, 2],
        calibration=[10],
        held_out=[20],
        train_input_groups=["ama", "ams", "rbb"],  # ams = MSE leak!
    )
    with pytest.raises(AssertionError):
        split.assert_no_mse_in_train()


# ---------------------------------------------------------------------------
# score() on a fully synthetic manifest + truth
# ---------------------------------------------------------------------------


class _SyntheticTruth:
    """Minimal MseTruth stand-in returning AmsShot-like objects."""

    def __init__(self, shots):
        self._shots = shots

    def get(self, sid):
        return self._shots.get(sid)


def _make_synthetic_shot(sid, K=40, C=12, R0=0.85, seed=0):
    rng = np.random.default_rng(seed)
    rho, j = _parabolic_profile()
    sight = np.linspace(0.70, 1.30, C)
    base = E.pitch_from_current_profile(
        j, rho, sight, R0=R0, B_tor0_or_Ip=0.5, kind="j"
    )
    # slowly varying truth pitch over time + small noise
    drift = np.linspace(1.0, 1.3, K)[:, None]
    pitch = base[None, :] * drift + 0.01 * rng.standard_normal((K, C))
    pitch_err = np.full((K, C), 0.05)
    q0 = np.full(K, 1.2) + 0.05 * rng.standard_normal(K)
    rax = np.full(K, R0) + 0.01 * rng.standard_normal(K)
    return M.AmsShot(
        shot_id=sid,
        beam_ok=True,
        time=np.linspace(0.0, 0.04, K),
        active_channel_ids=np.arange(C),
        active_channel_rpos=sight,
        pitch=pitch,
        pitch_error=pitch_err,
        gamma=pitch.copy(),
        gamma_error=pitch_err.copy(),
        q0=q0,
        q0_error=np.full(K, 0.1),
        rax=rax,
        rax_error=np.full(K, 0.01),
    )


def _make_manifest(shots):
    entries = {}
    for sid, shot in shots.items():
        entries[str(sid)] = M.build_shot_manifest(shot, "held_out")
    return {"version": "synthetic", "summary": {}, "shots": entries}


def test_score_structure_and_finiteness():
    shots = {100 + i: _make_synthetic_shot(100 + i, seed=i) for i in range(3)}
    manifest = _make_manifest(shots)
    truth = _SyntheticTruth(shots)

    preds = E.PersistencePredictor().predict(manifest, truth)
    assert len(preds) == 3
    result = E.score(preds, manifest, truth)

    # nested structure exactly per the contract
    assert "primary" in result and "pitch" in result["primary"]
    pp = result["primary"]["pitch"]
    for key in ("rmse", "crps", "nll", "cov90", "by_window"):
        assert key in pp
    assert "quiescent" in pp["by_window"] and "transient" in pp["by_window"]
    assert "secondary" in result
    assert "q0" in result["secondary"] and "rax" in result["secondary"]
    assert "meta" in result and result["meta"]["n_shots"] == 3

    # primary pitch metrics finite + coverage in [0,1]
    assert np.isfinite(pp["rmse"]) and pp["rmse"] >= 0
    assert np.isfinite(pp["crps"]) and pp["crps"] >= 0
    assert np.isfinite(pp["nll"])
    assert 0.0 <= pp["cov90"] <= 1.0


def test_score_perfect_predictor_low_rmse():
    """A predictor that returns the truth pitch should have ~0 RMSE."""
    shots = {200: _make_synthetic_shot(200, seed=7)}
    manifest = _make_manifest(shots)
    truth = _SyntheticTruth(shots)
    entry = manifest["shots"]["200"]
    tr = shots[200]
    t = np.asarray(entry["beam_on_slice_times"])
    pred = {
        200: E.ShotPrediction(
            t=t,
            pitch_mean=tr.pitch.copy(),
            pitch_std=np.full_like(tr.pitch, 0.05),
        )
    }
    result = E.score(pred, manifest, truth)
    assert result["primary"]["pitch"]["rmse"] < 1e-6


# ---------------------------------------------------------------------------
# Integration: full smoke on the real corpus (GPFS-gated)
# ---------------------------------------------------------------------------


_HAS_CORPUS = Path("/work/projects/imas_gpu/mast/level1/shots").exists()


@pytest.mark.skipif(not _HAS_CORPUS, reason="level-1 GPFS corpus not mounted")
def test_full_smoke_on_real_corpus():
    assert E._smoke() == 0


if __name__ == "__main__":
    # Allow `uv run python tests/test_mse_eval.py` as a quick check.
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
