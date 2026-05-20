"""Tests for :mod:`imas_ambix.data.persist`.

All tests use ``monkeypatch`` to redirect ``TOKEN_ROOT`` to ``tmp_path``
so they never touch the real file-system under
``/work/projects/imas_gpu/mast-tokens``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from imas_ambix.tokenizer.base import EncodedFrames, EncodedSignals

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_encoded_frames(
    shape: tuple[int, ...] = (4, 8, 10),
    tokenizer_name: str = "test_frames_v1",
    metadata: dict | None = None,
) -> EncodedFrames:
    """Return a synthetic :class:`EncodedFrames` for testing."""
    rng = np.random.default_rng(0)
    token_ids = rng.integers(0, 1000, size=shape, dtype=np.int32)
    return EncodedFrames(
        token_ids=token_ids,
        shape=shape,
        tokenizer_name=tokenizer_name,
        metadata=metadata or {"input_shape": list(shape)},
    )


def _make_encoded_signals(
    shape: tuple[int, int] = (16, 3),
    channel_names: tuple[str, ...] = ("ip", "ne", "beta"),
    tokenizer_name: str = "test_signals_v1",
    metadata: dict | None = None,
) -> EncodedSignals:
    """Return a synthetic :class:`EncodedSignals` for testing."""
    rng = np.random.default_rng(1)
    token_ids = rng.integers(0, 512, size=shape, dtype=np.int32)
    return EncodedSignals(
        token_ids=token_ids,
        channel_names=channel_names,
        tokenizer_name=tokenizer_name,
        metadata=metadata or {"n_bins": 512},
    )


# ---------------------------------------------------------------------------
# Path format tests
# ---------------------------------------------------------------------------


def test_token_paths_format():
    """frames/signals_token_path return paths matching the documented layout."""
    from imas_ambix.data.persist import frames_token_path, signals_token_path

    fpath = frames_token_path(30001, "rbb", vocab_version="v1")
    assert fpath.parts[-1] == "rbb.zarr"
    assert fpath.parts[-2] == "30001"
    assert fpath.parts[-3] == "frames"
    assert fpath.parts[-4] == "v1"

    spath = signals_token_path(30001, "magnetics", vocab_version="v1")
    assert spath.parts[-1] == "magnetics.zarr"
    assert spath.parts[-2] == "30001"
    assert spath.parts[-3] == "signals"
    assert spath.parts[-4] == "v1"


def test_token_paths_rooted_under_token_root():
    """Both path helpers are rooted under TOKEN_ROOT."""
    from imas_ambix.data.paths import TOKEN_ROOT
    from imas_ambix.data.persist import frames_token_path, signals_token_path

    fp = frames_token_path(12345, "rba")
    sp = signals_token_path(12345, "equilibrium")
    assert str(fp).startswith(str(TOKEN_ROOT))
    assert str(sp).startswith(str(TOKEN_ROOT))


# ---------------------------------------------------------------------------
# Round-trip tests: frames
# ---------------------------------------------------------------------------


def test_save_load_frame_tokens_roundtrip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Write a synthetic EncodedFrames, load back and verify array + metadata."""
    import imas_ambix.data.persist as persist_mod

    monkeypatch.setattr(persist_mod, "TOKEN_ROOT", tmp_path)

    encoded = _make_encoded_frames(shape=(4, 8, 10), tokenizer_name="frames_test")

    out_path = persist_mod.save_frame_tokens(
        shot_id=99001,
        camera="rbb",
        encoded=encoded,
        vocab_version="v1",
    )
    assert out_path.exists()

    loaded = persist_mod.load_frame_tokens(
        shot_id=99001, camera="rbb", vocab_version="v1"
    )

    np.testing.assert_array_equal(loaded.token_ids, encoded.token_ids)
    assert loaded.shape == encoded.shape
    assert loaded.tokenizer_name == encoded.tokenizer_name
    assert loaded.metadata == encoded.metadata


def test_load_frame_tokens_missing_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """load_frame_tokens raises FileNotFoundError when the Zarr is absent."""
    import imas_ambix.data.persist as persist_mod

    monkeypatch.setattr(persist_mod, "TOKEN_ROOT", tmp_path)

    with pytest.raises(FileNotFoundError):
        persist_mod.load_frame_tokens(shot_id=0, camera="rbb")


# ---------------------------------------------------------------------------
# Round-trip tests: signals
# ---------------------------------------------------------------------------


def test_save_load_signal_tokens_roundtrip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Write a synthetic EncodedSignals, load back and verify array + metadata."""
    import imas_ambix.data.persist as persist_mod

    monkeypatch.setattr(persist_mod, "TOKEN_ROOT", tmp_path)

    encoded = _make_encoded_signals(
        shape=(16, 3),
        channel_names=("ip", "ne", "beta"),
        tokenizer_name="signals_test",
    )

    out_path = persist_mod.save_signal_tokens(
        shot_id=99002,
        group="magnetics",
        encoded=encoded,
        vocab_version="v1",
    )
    assert out_path.exists()

    loaded = persist_mod.load_signal_tokens(
        shot_id=99002, group="magnetics", vocab_version="v1"
    )

    np.testing.assert_array_equal(loaded.token_ids, encoded.token_ids)
    assert loaded.channel_names == encoded.channel_names
    assert loaded.tokenizer_name == encoded.tokenizer_name
    assert loaded.metadata == encoded.metadata


def test_load_signal_tokens_missing_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """load_signal_tokens raises FileNotFoundError when the Zarr is absent."""
    import imas_ambix.data.persist as persist_mod

    monkeypatch.setattr(persist_mod, "TOKEN_ROOT", tmp_path)

    with pytest.raises(FileNotFoundError):
        persist_mod.load_signal_tokens(shot_id=0, group="magnetics")


# ---------------------------------------------------------------------------
# list_persisted_shots
# ---------------------------------------------------------------------------


def test_list_persisted_shots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Write tokens for 3 shots, assert list returns sorted shot ids."""
    import imas_ambix.data.persist as persist_mod

    monkeypatch.setattr(persist_mod, "TOKEN_ROOT", tmp_path)

    shot_ids = [30003, 30001, 30002]
    for sid in shot_ids:
        encoded = _make_encoded_frames(shape=(2, 4, 4))
        persist_mod.save_frame_tokens(
            shot_id=sid,
            camera="rbb",
            encoded=encoded,
            vocab_version="v1",
        )

    result = persist_mod.list_persisted_shots(modality="frames", vocab_version="v1")
    assert result == sorted(shot_ids)


def test_list_persisted_shots_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """list_persisted_shots returns an empty list when no tokens exist."""
    import imas_ambix.data.persist as persist_mod

    monkeypatch.setattr(persist_mod, "TOKEN_ROOT", tmp_path)

    result = persist_mod.list_persisted_shots(modality="frames", vocab_version="v1")
    assert result == []


def test_list_persisted_signals_shots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """list_persisted_shots works for the signals modality."""
    import imas_ambix.data.persist as persist_mod

    monkeypatch.setattr(persist_mod, "TOKEN_ROOT", tmp_path)

    for sid in [40001, 40002]:
        encoded = _make_encoded_signals(shape=(8, 2))
        persist_mod.save_signal_tokens(
            shot_id=sid,
            group="magnetics",
            encoded=encoded,
            vocab_version="v1",
        )

    result = persist_mod.list_persisted_shots(modality="signals", vocab_version="v1")
    assert result == [40001, 40002]
