"""Tests for the forward-horizon forecasting table.

Pure-numpy helpers (frame decimation, matched-stride logic, per-window
horizon scoring, aggregation) are exercised directly on synthetic arrays;
the full GPU-free end-to-end path runs against the synthetic corpus with
tiny CPU checkpoints (matched baseline / dynamics arms), mirroring the
checkpoint-building pattern in ``test_train.test_arm_compare_paired_verdict``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from imas_ambix.camdyn.dataset import discover_token_shots
from imas_ambix.camdyn.horizon_eval import (
    _agg_horizon,
    _bit_map_pred,
    _forward_dt_1d,
    decimate_window,
    matched_stride_for,
    score_window_horizons,
)
from imas_ambix.camdyn.metrics import HORIZON_MS
from imas_ambix.camdyn.model import N_COND_CHANNELS, CamdynConfig
from imas_ambix.camdyn.splits import CamdynSplit
from imas_ambix.camdyn.train import TrainConfig


def _write_split(tmp_path: Path, sc) -> Path:
    ids = sc["shot_ids"]
    split = CamdynSplit(
        train=[ids[0]],
        val=[ids[1]],
        held_out=[ids[0], ids[1]],  # the two longer shots reach a short horizon
        n_token_shots=len(ids),
    )
    path = tmp_path / "horizon_split.json"
    path.write_text(json.dumps(split.to_dict()), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Pure-numpy helpers
# ---------------------------------------------------------------------------


def test_forward_dt_1d_matches_diff():
    ft = np.array([0.0, 1.0, 3.0, 6.0])
    dt = _forward_dt_1d(ft)
    assert dt.shape == ft.shape
    np.testing.assert_allclose(dt, [1.0, 2.0, 3.0, 3.0])  # last repeated
    assert _forward_dt_1d(np.array([0.0])).tolist() == [0.0]


def test_matched_stride_reaches_or_reports_unreachable():
    # 16-frame window, native dt 1 ms → spans 15 ms.  Decimation widens the
    # spacing so a 200 ms horizon becomes reachable; ``ok`` is True exactly
    # when the (n_frames-1)*stride*dt span covers the horizon.
    ft = 1e-3 * np.arange(16, dtype=np.float64)
    stride, ok = matched_stride_for(ft, 16, 200.0)
    assert stride >= 1
    reach = (16 - 1) * stride * 1e-3 * 1000.0
    assert ok == (reach >= 200.0)
    assert ok is True  # 16 frames can be decimated to span 200 ms
    # a degenerate time base (single sample / non-positive dt) is unreachable
    _, ok_single = matched_stride_for(np.array([0.0]), 16, 200.0)
    assert ok_single is False
    _, ok_flat = matched_stride_for(np.zeros(16), 16, 200.0)
    assert ok_flat is False


def test_decimate_window_subsamples_frame_axis_and_recomputes_dt():
    nf = 12
    arr = {
        "tokens": np.zeros((2, nf, 16, 16), dtype=np.int64),
        "frame_time": np.tile(1e-3 * np.arange(nf), (2, 1)),
        "cond_values": np.zeros((2, nf, N_COND_CHANNELS), dtype=np.float32),
        "shot_id": np.array([1, 2]),  # non-frame axis: passed through
    }
    out = decimate_window(arr, stride=3)
    assert out["tokens"].shape == (2, 4, 16, 16)
    assert out["frame_time"].shape == (2, 4)
    assert out["shot_id"].tolist() == [1, 2]  # untouched
    # dt recomputed as forward diff of the decimated time base (3 ms spacing)
    np.testing.assert_allclose(out["dt"][0], [3e-3, 3e-3, 3e-3, 3e-3], atol=1e-9)
    # stride <= 1 is a passthrough
    assert decimate_window(arr, stride=1) is arr


def test_bit_map_pred_round_trips_token_ids():
    bits = 18
    rng = np.random.default_rng(0)
    ids = rng.integers(0, 1 << bits, size=(3, 4, 4), dtype=np.int64)
    # build logits whose per-bit sign encodes each id's bits exactly
    shifts = np.arange(bits)
    tgt_bits = ((ids[..., None] >> shifts) & 1).astype(np.float64)
    logits = np.where(tgt_bits > 0.5, 5.0, -5.0)
    np.testing.assert_array_equal(_bit_map_pred(logits), ids)


def test_score_window_horizons_pairs_predictors_and_flags_unreachable():
    nf, bits = 8, 18
    rng = np.random.default_rng(1)
    tokens = rng.integers(0, 1 << bits, size=(nf, 16, 16), dtype=np.int64)
    # 1 ms cadence → 10 ms horizon is offset 10 (out of an 8-frame window)
    frame_time = 1e-3 * np.arange(nf, dtype=np.float64)
    valid = np.ones(nf, dtype=bool)
    # perfect-prediction logits for the dynamics arm
    shifts = np.arange(bits)
    tgt_bits = ((tokens[..., None] >> shifts) & 1).astype(np.float64)
    bit_logits = np.where(tgt_bits > 0.5, 8.0, -8.0)
    rec = score_window_horizons(bit_logits, tokens, frame_time, valid, frontier_frame=2)
    # all locked horizons present; at 1 ms cadence none of 10/50/200 ms is
    # in an 8-frame window → all flagged unreachable
    assert set(rec) == set(HORIZON_MS)
    for h in HORIZON_MS:
        assert rec[h]["valid"] == 0


def test_score_window_horizons_perfect_dynamics_beats_persistence():
    # short horizon reachable: dt = 5 ms so 10 ms = offset 2
    nf, bits = 8, 18
    rng = np.random.default_rng(2)
    tokens = rng.integers(0, 1 << bits, size=(nf, 16, 16), dtype=np.int64)
    frame_time = 5e-3 * np.arange(nf, dtype=np.float64)
    valid = np.ones(nf, dtype=bool)
    shifts = np.arange(bits)
    tgt_bits = ((tokens[..., None] >> shifts) & 1).astype(np.float64)
    bit_logits = np.where(tgt_bits > 0.5, 8.0, -8.0)
    rec = score_window_horizons(
        bit_logits, tokens, frame_time, valid, frontier_frame=2, horizons_ms=(10.0,)
    )
    cell = rec[10.0]
    assert cell["valid"] == 1
    assert cell["target_frame"] == 4  # frontier 2 + offset 2
    # perfect dynamics → top1 all 1; persistence (copy frame 1) is random
    assert float(np.mean(cell["dyn_top1"])) == pytest.approx(1.0)
    assert float(np.mean(cell["persist_top1"])) < 0.5


def test_agg_horizon_builds_paired_cis_for_valid_horizons():
    # two windows, dynamics perfect, persistence/baseline poor at h=10ms
    records = []
    for _ in range(2):
        dyn = {
            10.0: {
                "valid": 1,
                "target_frame": 4,
                "dyn_top1": np.ones(256),
                "dyn_nll": np.zeros(256),
                "persist_top1": np.zeros(256),
                "n_cells": 256,
            }
        }
        base = {
            10.0: {
                "valid": 1,
                "target_frame": 4,
                "dyn_top1": np.zeros(256),
                "dyn_nll": np.full(256, 5.0),
                "persist_top1": np.zeros(256),
                "n_cells": 256,
            }
        }
        records.append({"dyn": dyn, "base": base})
    table = _agg_horizon(records, (10.0,))
    cell = table[10.0]
    assert cell["valid_windows"] == 2
    assert cell["n_cells"] == 512
    assert cell["dynamics"]["top1"] == pytest.approx(1.0)
    # dynamics beats both persistence and baseline (lower CI bound > 0)
    assert cell["dynamics_vs_persistence_top1"]["favours_dynamics"] is True
    assert cell["dynamics_vs_baseline_top1"]["favours_dynamics"] is True
    assert cell["dynamics_vs_baseline_nll"]["favours_dynamics"] is True


def test_score_window_horizons_short_decimated_window_all_invalid():
    """A window with fewer frames than the frontier (the matched-regime edge
    case) is honestly invalid, not an IndexError."""
    nf, bits = 2, 18
    tokens = np.zeros((nf, 16, 16), dtype=np.int64)
    bit_logits = np.zeros((nf, 16, 16, bits))
    valid = np.ones(nf, dtype=bool)
    frame_time = 1e-3 * np.arange(nf, dtype=np.float64)
    rec = score_window_horizons(bit_logits, tokens, frame_time, valid, frontier_frame=8)
    for h in HORIZON_MS:
        assert rec[h]["valid"] == 0


def test_agg_horizon_marks_no_valid_windows():
    table = _agg_horizon([], (10.0, 50.0))
    for h in (10.0, 50.0):
        assert table[h]["valid_windows"] == 0
        assert "note" in table[h]


# ---------------------------------------------------------------------------
# End-to-end (CPU, tiny matched arms against the synthetic corpus)
# ---------------------------------------------------------------------------


def test_horizon_table_end_to_end_cpu(synthetic_corpus, tmp_path, monkeypatch):
    """The CLI path runs end-to-end on tiny CPU ckpts and writes a structurally
    complete artifact (native + matched regimes, three predictors, verdict)."""
    torch = pytest.importorskip("torch")
    from imas_ambix.camdyn import horizon_eval as he
    from imas_ambix.camdyn.model import CamdynModel

    sc = synthetic_corpus
    split_path = _write_split(tmp_path, sc)

    import imas_ambix.camdyn.train as trainmod

    real_discover = discover_token_shots

    def _patched_discover(*, shot_ids=None, read_n_frames=False, **_kw):
        return real_discover(
            token_root=sc["token_root"],
            level1_dir=sc["level1_dir"],
            shot_ids=shot_ids,
            read_n_frames=read_n_frames,
        )

    monkeypatch.setattr(trainmod, "discover_token_shots", _patched_discover)

    def _cfg(temporal):
        return TrainConfig(
            model=CamdynConfig(
                temporal_attention=temporal,
                dim=32,
                n_layers=2,
                n_heads=4,
                mlp_ratio=2.0,
                n_frames=6,
                cond_channels=N_COND_CHANNELS,
            ),
            n_frames=6,
            stride=4,
            batch_size=2,
            num_workers=0,
            eval_windows=4,
            max_heldout_shots=None,
            seed=0,
            split_path=str(split_path),
            device="cpu",
        )

    cond_stats = [[0.0] * N_COND_CHANNELS, [1.0] * N_COND_CHANNELS]
    ckpts = {}
    for arm, temporal in (("baseline", False), ("dynamics", True)):
        cfg = _cfg(temporal)
        model = CamdynModel.from_config(cfg.model)
        p = tmp_path / f"{arm}.pt"
        torch.save(
            {
                "config": cfg.to_dict(),
                "model_state": model.module.state_dict(),
                "cond_stats": cond_stats,
            },
            p,
        )
        ckpts[arm] = p

    out_path = tmp_path / "horizon.json"
    rc = he.main(
        [
            "--baseline",
            str(ckpts["baseline"]),
            "--dynamics",
            str(ckpts["dynamics"]),
            "--out",
            str(out_path),
            "--device",
            "cpu",
            "--split-path",
            str(split_path),
            "--frontier",
            "3",
        ]
    )
    assert rc == 0
    art = json.loads(out_path.read_text())
    assert art["horizons_ms"] == list(HORIZON_MS)
    assert art["frontier_frame"] == 3
    # both cadence regimes scored, each with the full horizon table
    for regime in ("native", "matched"):
        assert regime in art
        assert set(art[regime]["table"]) == {str(h) for h in HORIZON_MS}
    # the matched verdict has an entry per horizon (reachable flag honest)
    assert set(art["verdict_matched"]) == {str(h) for h in HORIZON_MS}
    for h in HORIZON_MS:
        cell = art["matched"]["table"][str(h)]
        # every cell is either populated (valid_windows>0) or honestly noted
        if cell.get("valid_windows"):
            assert "dynamics" in cell and "persistence" in cell and "baseline" in cell
        else:
            assert "note" in cell
