"""Split manifest invariants: 112 MSE ⊂ held-out, disjoint partitions."""

from __future__ import annotations

import json

import pytest

from imas_ambix.camdyn.splits import (
    DEFAULT_SPLIT_OUT,
    CamdynSplit,
    build_camdyn_split,
    load_mse_heldout_shots,
)


def test_load_mse_heldout_returns_112():
    mse = load_mse_heldout_shots()
    assert len(mse) == 112
    assert mse == sorted(mse)


def test_build_forces_mse_into_held_out_synthetic():
    # synthetic universe: 1000 token shots, of which 50 are "MSE held-out"
    token_shots = list(range(1000))
    mse = list(range(900, 960))  # 60 MSE, 50 of which have tokens (900..949 ⊂ 0..999)
    split = build_camdyn_split(
        token_shots, mse_heldout=mse, val_fraction=0.1, held_out_fraction=0.1, seed=1
    )
    split.assert_invariants()
    forced = set(split.mse_heldout_forced)
    # MSE shots with tokens are 900..959 ∩ 0..999 = 900..959 → all 60 have tokens
    assert forced == set(mse)
    assert forced <= set(split.held_out)
    # disjoint partitions
    assert not (set(split.train) & set(split.val))
    assert not (set(split.train) & set(split.held_out))
    assert not (set(split.val) & set(split.held_out))


def test_mse_shots_without_tokens_are_surfaced():
    token_shots = list(range(100))
    mse = [50, 51, 999, 1000]  # 999/1000 lack tokens
    split = build_camdyn_split(token_shots, mse_heldout=mse, seed=1)
    assert set(split.mse_heldout_without_tokens) == {999, 1000}
    assert set(split.mse_heldout_forced) == {50, 51}
    assert {50, 51} <= set(split.held_out)


def test_split_partitions_cover_all_token_shots():
    token_shots = list(range(500))
    mse = [10, 20, 30]
    split = build_camdyn_split(token_shots, mse_heldout=mse, seed=2)
    union = set(split.train) | set(split.val) | set(split.held_out)
    assert union == set(token_shots)


def test_reproducible_with_seed():
    token_shots = list(range(300))
    mse = [5, 6, 7]
    a = build_camdyn_split(token_shots, mse_heldout=mse, seed=99)
    b = build_camdyn_split(token_shots, mse_heldout=mse, seed=99)
    assert a.train == b.train and a.val == b.val and a.held_out == b.held_out


def test_roundtrip_save_load(tmp_path):
    token_shots = list(range(200))
    mse = [1, 2, 3]
    split = build_camdyn_split(token_shots, mse_heldout=mse, seed=4)
    out = split.save(tmp_path / "s.json")
    loaded = CamdynSplit.load(out)
    assert loaded.train == split.train
    assert loaded.held_out == split.held_out
    assert loaded.mse_heldout_forced == split.mse_heldout_forced


# --- the committed manifest (real corpus) ---


@pytest.mark.skipif(
    not DEFAULT_SPLIT_OUT.exists(), reason="committed camdyn manifest absent"
)
def test_committed_manifest_invariants():
    split = CamdynSplit.load(DEFAULT_SPLIT_OUT)
    split.assert_invariants()
    mse = set(load_mse_heldout_shots())
    ho = set(split.held_out)
    # every MSE held-out shot is either in held_out or recorded as token-less
    accounted = ho | set(split.mse_heldout_without_tokens)
    assert mse <= accounted
    # the forced set is exactly the MSE shots that carry tokens
    assert set(split.mse_heldout_forced) == (mse & ho)
    # all 112 accounted for: forced + without-tokens == 112
    assert len(split.mse_heldout_forced) + len(split.mse_heldout_without_tokens) == 112


@pytest.mark.skipif(
    not DEFAULT_SPLIT_OUT.exists(), reason="committed camdyn manifest absent"
)
def test_committed_manifest_has_expected_shape():
    d = json.loads(DEFAULT_SPLIT_OUT.read_text())
    assert d["version"] == "camdyn_split_v0"
    assert d["unit"] == "shot"
    assert d["n_train"] + d["n_val"] + d["n_held_out"] == d["n_token_shots"]
    assert d["n_mse_heldout_forced"] + d["n_mse_heldout_without_tokens"] == 112
