"""Unit tests for the raw-held-out prediction bar wrapper.

These tests exercise the composition logic — manifest -> persistence leg ->
scored bar, EnKF leg hoisted from a reference artifact, and the Bar accessors —
on SYNTHETIC ShotPrediction / MseTruth / manifest objects.  No /work, no TORAX,
no GPU: the truth is synthesised in-process and the EnKF reference is a tiny
temp JSON.  The heavy live EnKF leg is NOT exercised here (that needs the
sun_debug partition); it is covered by the locked reference artifact path.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

from imas_ambix.eval import prediction_bar as pb
from imas_ambix.statespace import mse_eval

# ---------------------------------------------------------------------------
# Synthetic truth + manifest builders (no corpus, no solver)
# ---------------------------------------------------------------------------


@dataclass
class _SynthShot:
    """Minimal stand-in for mse_split.AmsShot — only the fields score() reads."""

    pitch: np.ndarray  # (K, C)
    pitch_error: np.ndarray  # (K, C)
    q0: np.ndarray  # (K,)
    q0_error: np.ndarray  # (K,)
    rax: np.ndarray  # (K,)
    rax_error: np.ndarray  # (K,)


@dataclass
class _SynthTruth:
    """A truth provider compatible with mse_eval.MseTruth.get()."""

    shots: dict = field(default_factory=dict)

    def get(self, shot_id: int):
        return self.shots.get(int(shot_id))


def _build_synthetic(
    *,
    shot_id: int = 9001,
    n_slices: int = 8,
    n_channels: int = 8,
    drift: float = 0.0,
    seed: int = 0,
):
    """Build a (manifest, truth) pair for one held-out shot.

    The truth pitch ramps linearly across channels (a sensible radial profile
    with a zero crossing) and drifts in time by ``drift`` rad/slice, so a
    persistence predictor (freeze at slice 0) accrues error proportional to
    ``drift``.  All channels are finite, non-railed, low-error -> they pass the
    shared pitch point gate, so every slice is pitch-valid.
    """
    rng = np.random.default_rng(seed)
    rpos = np.linspace(0.7, 1.0, n_channels)  # major radii (m), radial order
    R0 = 0.85
    # base radial pitch profile: crosses zero near R0, |pitch| well under the rail
    base = np.tanh((rpos - R0) * 5.0) * 0.4  # (C,) in ~[-0.4, 0.4]
    pitch = np.empty((n_slices, n_channels))
    for k in range(n_slices):
        pitch[k] = base + drift * k + rng.normal(0, 1e-3, n_channels)
    pitch_error = np.full((n_slices, n_channels), 0.02)  # low, reliable
    # secondary q0/rax: inside the gate bands, low error
    q0 = np.full(n_slices, 1.2)
    q0_error = np.full(n_slices, 0.1)
    rax = np.full(n_slices, 0.9)
    rax_error = np.full(n_slices, 0.01)

    truth = _SynthTruth(
        shots={
            shot_id: _SynthShot(
                pitch=pitch,
                pitch_error=pitch_error,
                q0=q0,
                q0_error=q0_error,
                rax=rax,
                rax_error=rax_error,
            )
        }
    )

    # All slices pitch-valid; secondary gates open.
    t = np.linspace(0.1, 0.1 + 0.001 * (n_slices - 1), n_slices)
    manifest = {
        "version": "synthetic-test",
        "shots": {
            str(shot_id): {
                "shot_id": shot_id,
                "partition": "held_out",
                "model_grid_hz": 1000,
                "beam_on_slice_times": [float(x) for x in t],
                "active_channel_ids": list(range(n_channels)),
                "active_channel_rpos": [float(r) for r in rpos],
                "pitch_valid_mask": [True] * n_slices,
                "q0_gated_mask": [True] * n_slices,
                "rax_gated_mask": [True] * n_slices,
            }
        },
    }
    return manifest, truth


# ---------------------------------------------------------------------------
# persistence leg
# ---------------------------------------------------------------------------


def test_persistence_leg_scores_live_and_has_keys():
    manifest, truth = _build_synthetic(drift=0.05, n_slices=6)
    leg = pb.persistence_bar(manifest, truth)
    assert leg.name == "persistence"
    assert leg.source == "live"
    assert leg.n_shots == 1
    # the headline metrics are finite
    for v in (leg.pitch_rmse, leg.pitch_crps, leg.pitch_nll, leg.pitch_cov90):
        assert np.isfinite(v)
    # the full nested scored dict is preserved
    assert "primary" in leg.scored
    assert "pitch" in leg.scored["primary"]
    summ = leg.summary()
    assert set(summ) >= {
        "name",
        "source",
        "pitch_rmse",
        "pitch_crps",
        "pitch_nll",
        "pitch_cov90",
        "n_shots",
    }


def test_persistence_is_perfect_on_flat_data():
    """No drift -> frozen-slice-0 pitch == every slice -> ~zero RMSE."""
    manifest, truth = _build_synthetic(drift=0.0, n_slices=6, seed=1)
    leg = pb.persistence_bar(manifest, truth)
    assert leg.pitch_rmse < 1e-2  # only the tiny per-slice noise remains


def test_persistence_error_grows_with_drift():
    """Persistence accrues error proportional to the temporal drift."""
    m_lo, t_lo = _build_synthetic(drift=0.01, n_slices=8, seed=2)
    m_hi, t_hi = _build_synthetic(drift=0.10, n_slices=8, seed=2)
    rmse_lo = pb.persistence_bar(m_lo, t_lo).pitch_rmse
    rmse_hi = pb.persistence_bar(m_hi, t_hi).pitch_rmse
    assert rmse_hi > rmse_lo


# ---------------------------------------------------------------------------
# EnKF leg from a reference artifact
# ---------------------------------------------------------------------------


def _write_reference_artifact(tmp_path: Path) -> Path:
    """A tiny stand-in for the locked enkf_baseline_metrics_v0.json."""
    payload = {
        "schema": "enkf-baseline-metrics-v0",
        "n_shots_scored": 112,
        "metrics_analysis_arm": {
            "n_shots": 112,
            "pitch_rmse": {"mean": 0.225, "ci_lo": 0.199, "ci_hi": 0.259, "n": 112},
            "pitch_crps": {"mean": 0.150, "ci_lo": 0.132, "ci_hi": 0.174, "n": 112},
            "pitch_nll": {"mean": -0.5, "ci_lo": -0.7, "ci_hi": -0.3, "n": 112},
            "pitch_cov90": {"mean": 0.465, "ci_lo": 0.423, "ci_hi": 0.506, "n": 112},
        },
        "metrics_forecast_arm_NONVACUITY_CONTROL": {
            "n_shots": 112,
            "pitch_rmse": {"mean": 0.230, "ci_lo": 0.20, "ci_hi": 0.26, "n": 112},
            "pitch_crps": {"mean": 0.152, "ci_lo": 0.13, "ci_hi": 0.18, "n": 112},
            "pitch_nll": {"mean": -0.4, "ci_lo": -0.6, "ci_hi": -0.2, "n": 112},
            "pitch_cov90": {"mean": 0.47, "ci_lo": 0.43, "ci_hi": 0.51, "n": 112},
        },
    }
    p = tmp_path / "enkf_ref.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


def test_enkf_leg_from_reference(tmp_path):
    ref = _write_reference_artifact(tmp_path)
    leg = pb.enkf_leg_from_reference(ref)
    assert leg.name == "enkf"
    assert leg.source == "from_reference"
    assert leg.n_shots == 112
    assert leg.pitch_rmse == pytest.approx(0.225)
    assert leg.pitch_cov90 == pytest.approx(0.465)
    assert leg.pitch_rmse_ci == pytest.approx((0.199, 0.259))


def test_enkf_leg_forecast_arm(tmp_path):
    ref = _write_reference_artifact(tmp_path)
    leg = pb.enkf_leg_from_reference(ref, arm="forecast")
    assert leg.pitch_rmse == pytest.approx(0.230)


def test_real_reference_artifact_reproduces_locked_numbers():
    """The shipped artifact reproduces the locked sequential-da-v1 bar."""
    if not pb._REFERENCE_ARTIFACT.exists():
        pytest.skip("locked reference artifact not present in this checkout")
    leg = pb.enkf_leg_from_reference()
    # locked: EnKF sightline pitch RMSE ~= 0.225 [0.199, 0.259]
    assert leg.pitch_rmse == pytest.approx(0.225, abs=0.02)
    assert leg.n_shots > 0


# ---------------------------------------------------------------------------
# the assembled bar
# ---------------------------------------------------------------------------


def test_prediction_bar_assembles_both_legs(tmp_path):
    manifest, truth = _build_synthetic(drift=0.08, n_slices=8)
    ref = _write_reference_artifact(tmp_path)
    bar = pb.prediction_bar(manifest, truth=truth, reference_artifact=ref)
    assert set(bar.legs) == {"persistence", "enkf"}
    assert bar.legs["persistence"].source == "live"
    assert bar.legs["enkf"].source == "from_reference"
    # target = EnKF RMSE; floor = persistence RMSE
    assert bar.target == pytest.approx(0.225)
    assert bar.floor == bar.legs["persistence"].pitch_rmse
    # the to_dict has the expected top-level structure
    d = bar.to_dict()
    assert d["schema"] == "raw-held-out-prediction-bar-v0"
    assert d["target_pitch_rmse"] == pytest.approx(0.225)
    assert "coverage_gate" in d
    assert "reference" in d


def test_bar_beats_logic(tmp_path):
    manifest, truth = _build_synthetic(drift=0.05, n_slices=6)
    ref = _write_reference_artifact(tmp_path)
    bar = pb.prediction_bar(manifest, truth=truth, reference_artifact=ref)
    # a candidate below the EnKF RMSE beats the bar; above does not
    assert bar.beats(0.10, against="enkf") is True
    assert bar.beats(0.50, against="enkf") is False
    # a missing leg -> beats() is False (not a crash)
    assert bar.beats(0.01, against="does_not_exist") is False


def test_held_out_shot_ids_filters_partition():
    manifest, _ = _build_synthetic(shot_id=12345)
    # add a non-held-out entry that must be excluded
    manifest["shots"]["999"] = {"partition": "calibration"}
    ids = pb.held_out_shot_ids(manifest)
    assert ids == [12345]


def test_coverage_gate_matches_harness():
    """The bar's coverage gate equals the harness's pre-registered band."""
    assert pb.REFERENCE["coverage_gate"] == (
        mse_eval.COVERAGE_GATE_LO,
        mse_eval.COVERAGE_GATE_HI,
    )
