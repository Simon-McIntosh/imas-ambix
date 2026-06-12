"""Tests for the gas-puff attribution probe.

Covers the aga channel-mask + index lookup, the inboard visibility mask,
the decoder-free predicted-change activity score, the puff-shot selector,
and the full end-to-end probe on a tiny CPU dynamics checkpoint against the
synthetic corpus (whose ``aga/inboard_total`` trace varies, so the shots
qualify as puff shots).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from imas_ambix.camdyn.conditioning import CONDITIONING_CHANNELS
from imas_ambix.camdyn.dataset import discover_token_shots
from imas_ambix.camdyn.model import N_COND_CHANNELS, CamdynConfig
from imas_ambix.camdyn.puff_attribution import (
    DEFAULT_INBOARD_COLS,
    _bit_map_pred,
    _pearson,
    aga_channel_mask,
    gas_inboard_total_index,
    inboard_activity_score,
    inboard_visibility_mask,
    select_puff_shots,
)
from imas_ambix.camdyn.splits import CamdynSplit
from imas_ambix.camdyn.train import TrainConfig


def _write_split(tmp_path: Path, sc) -> Path:
    ids = sc["shot_ids"]
    split = CamdynSplit(
        train=[ids[0]],
        val=[ids[1]],
        held_out=[ids[0], ids[1]],
        n_token_shots=len(ids),
    )
    path = tmp_path / "puff_split.json"
    path.write_text(json.dumps(split.to_dict()), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Channel lookups
# ---------------------------------------------------------------------------


def test_aga_channel_mask_selects_only_gas_channels():
    mask = aga_channel_mask()
    assert mask.shape == (N_COND_CHANNELS,)
    sel_keys = {CONDITIONING_CHANNELS[i].key for i in np.where(mask)[0]}
    assert sel_keys == {
        "gas_inboard_total",
        "gas_inboard_upper",
        "gas_inboard_lower",
        "gas_outboard_total",
    }
    assert all(CONDITIONING_CHANNELS[i].source == "aga" for i in np.where(mask)[0])


def test_gas_inboard_total_index_points_at_the_right_channel():
    idx = gas_inboard_total_index()
    assert CONDITIONING_CHANNELS[idx].key == "gas_inboard_total"


# ---------------------------------------------------------------------------
# Masks + activity score
# ---------------------------------------------------------------------------


def test_inboard_visibility_mask_hides_only_the_inboard_band():
    vis = inboard_visibility_mask(5, DEFAULT_INBOARD_COLS, grid=(16, 16))
    assert vis.shape == (5, 16, 16)
    c0, c1 = DEFAULT_INBOARD_COLS
    assert not vis[:, :, c0:c1].any()  # inboard band masked
    # everything outside the band visible
    assert vis[:, :, :c0].all()
    assert vis[:, :, c1:].all()


def test_inboard_activity_score_counts_predicted_change():
    nf, bits = 4, 18
    rng = np.random.default_rng(0)
    tokens = rng.integers(0, 1 << bits, size=(nf, 16, 16), dtype=np.int64)
    valid = np.ones(nf, dtype=bool)
    # logits that predict the SAME token as the previous frame in the inboard
    # band → zero predicted change there
    shifts = np.arange(bits)
    prev_shift = np.zeros_like(tokens)
    prev_shift[1:] = tokens[:-1]
    tgt_bits = ((prev_shift[..., None] >> shifts) & 1).astype(np.float64)
    bit_logits = np.where(tgt_bits > 0.5, 8.0, -8.0)
    scores, vf = inboard_activity_score(bit_logits, tokens, DEFAULT_INBOARD_COLS, valid)
    assert vf[0] == False  # noqa: E712 — frame 0 has no previous frame
    assert vf[1:].all()
    # predicted == previous-frame token → activity 0 in the inboard band
    np.testing.assert_allclose(scores[1:], 0.0, atol=1e-9)


def test_inboard_activity_full_change_when_prediction_differs():
    nf, bits = 3, 18
    tokens = np.zeros((nf, 16, 16), dtype=np.int64)  # all zeros
    valid = np.ones(nf, dtype=bool)
    # predict token id 1 everywhere → differs from the all-zero previous frame
    bit_logits = np.full((nf, 16, 16, bits), -8.0)
    bit_logits[..., 0] = 8.0  # bit 0 set → predicted id 1
    scores, vf = inboard_activity_score(bit_logits, tokens, DEFAULT_INBOARD_COLS, valid)
    np.testing.assert_allclose(scores[1:], 1.0, atol=1e-9)


def test_pearson_handles_degenerate_inputs():
    assert np.isnan(_pearson([1.0], [1.0]))  # too few
    assert np.isnan(_pearson([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]))  # zero variance
    r = _pearson([1.0, 2.0, 3.0], [2.0, 4.0, 6.0])
    assert r == pytest.approx(1.0)


def test_bit_map_pred_round_trips():
    bits = 18
    ids = np.random.default_rng(0).integers(0, 1 << bits, size=(2, 4, 4))
    shifts = np.arange(bits)
    tgt_bits = ((ids[..., None] >> shifts) & 1).astype(np.float64)
    logits = np.where(tgt_bits > 0.5, 5.0, -5.0)
    np.testing.assert_array_equal(_bit_map_pred(logits), ids)


def test_select_puff_shots_picks_varying_command():
    gas_idx = gas_inboard_total_index()
    nf = 6
    # one shot with varying puff (qualifies), one constant (does not)
    arr_vary = {
        "cond_values": np.zeros((1, nf, N_COND_CHANNELS), dtype=np.float32),
        "cond_missing": np.zeros((1, nf, N_COND_CHANNELS), dtype=np.float32),
        "shot_id": np.array([101]),
    }
    arr_vary["cond_values"][0, :, gas_idx] = np.linspace(0, 10, nf)
    arr_const = {
        "cond_values": np.zeros((1, nf, N_COND_CHANNELS), dtype=np.float32),
        "cond_missing": np.zeros((1, nf, N_COND_CHANNELS), dtype=np.float32),
        "shot_id": np.array([202]),
    }
    arr_const["cond_values"][0, :, gas_idx] = 5.0  # constant → std 0
    qualifying = select_puff_shots([arr_vary, arr_const], gas_idx)
    assert 101 in qualifying
    assert 202 not in qualifying


# ---------------------------------------------------------------------------
# End-to-end probe (CPU, tiny dynamics arm)
# ---------------------------------------------------------------------------


def test_probe_puff_attribution_end_to_end_cpu(synthetic_corpus, tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from imas_ambix.camdyn import puff_attribution as pa
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

    cfg = TrainConfig(
        model=CamdynConfig(
            temporal_attention=True,
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
        eval_windows=8,
        max_heldout_shots=None,
        seed=0,
        split_path=str(split_path),
        device="cpu",
    )
    cond_stats = [[0.0] * N_COND_CHANNELS, [1.0] * N_COND_CHANNELS]
    model = CamdynModel.from_config(cfg.model)
    ckpt = tmp_path / "dynamics.pt"
    torch.save(
        {
            "config": cfg.to_dict(),
            "model_state": model.module.state_dict(),
            "cond_stats": cond_stats,
        },
        ckpt,
    )

    out_path = tmp_path / "puff.json"
    rc = pa.main(
        [
            "--dynamics",
            str(ckpt),
            "--out",
            str(out_path),
            "--device",
            "cpu",
            "--split-path",
            str(split_path),
        ]
    )
    assert rc == 0
    art = json.loads(out_path.read_text())
    assert art["inboard_cols"] == list(DEFAULT_INBOARD_COLS)
    assert "verdict" in art
    # the synthetic aga trace varies → at least one puff shot is selected and a
    # pooled result + counterfactual are produced (attribution in {positive,
    # partial, null} — an honest null is acceptable)
    assert art["n_puff_shots"] >= 1
    assert "pooled" in art
    assert "counterfactual_delta_mean" in art["pooled"]
    assert art["verdict"]["attribution"] in {"positive", "partial", "null"}
