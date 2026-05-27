"""Parity tests for the in-process continuous-batching stream encoder.

The whole point of :mod:`imas_ambix.data.stream_encode` is byte-identical
tokens vs the live per-shot path. These tests prove the *reassembly*
correctness on CPU with a deterministic stub model: that flattening shots
into one stream, encoding in fixed-size batches that fall mid-shot and
across shot boundaries, and reassembling per shot yields exactly the same
per-shot token arrays as encoding each shot on its own.

The GPU byte-identity vs the live daemon is checked separately by
``scripts``-driven spot-checks (see the agent report); here we lock the
host-side stream logic, registry offset, and Zarr persistence.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from imas_ambix.data import stream_encode as se  # noqa: E402


# A deterministic per-frame "encode": maps each (H,W,3) uint8 frame to a
# (16,16) int64 token grid using only that frame's bytes — exactly the
# property the real model has (no cross-frame state on the encode path).
# We derive a stable per-frame fingerprint and tile it, so identical frames
# always produce identical tokens regardless of which batch they land in.
def _stub_encode(model, frames_u8_rgb_batch: np.ndarray) -> np.ndarray:
    b = frames_u8_rgb_batch.shape[0]
    out = np.empty((b, se.TOKEN_HW, se.TOKEN_HW), dtype=np.int64)
    flat = frames_u8_rgb_batch.reshape(b, -1).astype(np.int64)
    # Per-frame deterministic value in [0, 2^18) — within the magvit2 vocab.
    fp = (flat.sum(axis=1) * 2654435761) % se.VOCAB_SIZE
    for i in range(b):
        # Vary spatially so reshape/transpose bugs surface, but stay a pure
        # function of the frame contents.
        grid = (fp[i] + np.arange(se.TOKEN_HW * se.TOKEN_HW)) % se.VOCAB_SIZE
        out[i] = grid.reshape(se.TOKEN_HW, se.TOKEN_HW)
    return out


def _make_frames(shot_id: int, n_frames: int, rng: np.random.Generator) -> np.ndarray:
    """A fake L1 raw frame array (T,H,W) uint16, distinct per shot."""
    h, w = 40, 56  # non-square, non-256 → exercises the resize/normalise path
    base = rng.integers(0, 4000, size=(n_frames, h, w), dtype=np.uint16)
    return base + shot_id  # shift so shots differ


def _per_shot_reference(
    shot_ids, frame_arrays, camera
) -> dict[int, np.ndarray]:
    """Encode each shot independently with the stub → global int32 tokens.

    Mirrors the live single-shot path: normalise→RGB (presized=False) then
    per-frame stub encode, then registry shift (+offset, int32).
    """
    ref: dict[int, np.ndarray] = {}
    for sid in shot_ids:
        raw = frame_arrays[sid]
        u8_rgb = se.frames_to_rgb_uint8(raw, presized=False)
        toks = _stub_encode(None, u8_rgb)  # (T,16,16) int64
        ref[sid] = (toks.astype(np.int64) + se.REGISTRY_OFFSET).astype(np.int32)
    return ref


class _DictDataset(se.ShotFrameDataset):
    """ShotFrameDataset variant that serves from an in-memory dict of raw
    frame arrays — no zarr/xarray needed for the CPU parity test."""

    def __init__(self, shot_ids, frame_arrays, camera):
        super().__init__(shot_ids, camera, l1_root=Path("/nonexistent"))
        self._frame_arrays = frame_arrays

    def __getitem__(self, i):
        sid = self.shot_ids[i]
        raw = self._frame_arrays[sid]
        u8_rgb = se.frames_to_rgb_uint8(raw, presized=False)
        return (
            sid,
            u8_rgb,
            tuple(int(x) for x in raw.shape),
            (int(u8_rgb.shape[1]), int(u8_rgb.shape[2])),
            None,
        )


@pytest.mark.parametrize("batch_frames", [1, 3, 7, 13, 50, 1000])
def test_stream_reassembly_matches_per_shot(tmp_path, batch_frames):
    """Stream-batched + reassembled tokens == per-shot tokens, for shots with
    DIFFERENT frame counts so batches fall mid-shot and across boundaries."""
    rng = np.random.default_rng(0)
    camera = "rbb"
    # Deliberately varied, mutually-coprime-ish frame counts so that for
    # several batch sizes a batch boundary lands inside a shot AND a batch
    # spans two shots.
    counts = {101: 5, 102: 1, 103: 9, 104: 12, 105: 2, 106: 17, 107: 4}
    shot_ids = list(counts)
    frame_arrays = {sid: _make_frames(sid, counts[sid], rng) for sid in shot_ids}

    reference = _per_shot_reference(shot_ids, frame_arrays, camera)

    dataset = _DictDataset(shot_ids, frame_arrays, camera)
    stream_root = tmp_path / "frames-stream"

    # Monkeypatch the dataset constructor used inside stream_encode so it
    # serves from our in-memory dataset, and run num_workers=0 (direct iter).
    orig_ctor = se.ShotFrameDataset
    se.ShotFrameDataset = lambda *a, **k: dataset  # type: ignore[assignment]
    try:
        stats = se.stream_encode(
            shot_ids,
            camera,
            model=None,
            device="cpu",
            stream_root=stream_root,
            batch_frames=batch_frames,
            num_workers=0,
            encode_fn=_stub_encode,
        )
    finally:
        se.ShotFrameDataset = orig_ctor  # type: ignore[assignment]

    assert stats.shots_fail == 0, stats.load_errors
    assert stats.shots_ok == len(shot_ids)
    assert stats.frames_encoded == sum(counts.values())
    assert stats.write_errors == []

    # Read back every shot's Zarr and compare byte-for-byte to the reference.
    import zarr

    for sid in shot_ids:
        path = se.stream_frames_token_path(sid, camera, stream_root)
        assert path.exists(), f"missing output for shot {sid}"
        store = zarr.open_group(str(path), mode="r")
        got = np.asarray(store["tokens"], dtype=np.int32)
        ref = reference[sid]
        assert got.shape == ref.shape, (sid, got.shape, ref.shape)
        assert got.dtype == np.int32
        np.testing.assert_array_equal(got, ref)


def test_registry_offset_matches_live_registry():
    """The hardcoded REGISTRY_OFFSET must equal the live registry's offset for
    the magvit2 frame block, or stream tokens diverge from the live path."""
    from imas_ambix.tokenizer.registry import registry

    start, end = registry.allocate(se.TOKENIZER_NAME, se.VOCAB_SIZE)
    assert start == se.REGISTRY_OFFSET, (start, se.REGISTRY_OFFSET)
    assert end - start == se.VOCAB_SIZE


def test_zarr_layout_matches_save_frame_tokens(tmp_path):
    """A Zarr written by save_stream_frame_tokens must carry the same arrays
    and attrs (modulo root dir) as the live persist.save_frame_tokens."""
    import json

    import zarr

    toks = np.arange(2 * 16 * 16, dtype=np.int32).reshape(2, 16, 16) + 4
    path = se.save_stream_frame_tokens(
        999,
        "rbb",
        toks,
        input_shape=(2, 40, 56),
        original_hw=(40, 56),
        stream_root=tmp_path,
    )
    store = zarr.open_group(str(path), mode="r")
    assert np.asarray(store["tokens"], dtype=np.int32).shape == (2, 16, 16)
    attrs = dict(store.attrs)
    assert attrs["shot_id"] == 999
    assert attrs["camera"] == "rbb"
    assert attrs["vocab_version"] == "v1"
    assert attrs["tokenizer_name"] == se.TOKENIZER_NAME
    assert attrs["shape"] == [2, 16, 16]
    md = json.loads(attrs["metadata"])
    assert md["model_image_size"] == 256
    assert md["spatial_compression"] == 16
    assert md["temporal_compression"] == 1
    assert md["original_hw"] == [40, 56]
    assert md["ckpt"] == "imagenet_256_L.ckpt"


def test_normalise_byte_identical_to_live():
    """stream_encode.normalise_frames_to_uint8 must equal the live frames.py
    implementation byte-for-byte."""
    from imas_ambix.tokenizer.frames import _normalise_frames_to_uint8

    rng = np.random.default_rng(1)
    raw = rng.integers(0, 65535, size=(4, 30, 30), dtype=np.uint16)
    np.testing.assert_array_equal(
        se.normalise_frames_to_uint8(raw), _normalise_frames_to_uint8(raw)
    )
    # uint8 passthrough and degenerate (hi<=lo) branches.
    u8 = rng.integers(0, 256, size=(2, 5, 5), dtype=np.uint8)
    np.testing.assert_array_equal(
        se.normalise_frames_to_uint8(u8), _normalise_frames_to_uint8(u8)
    )
    flat = np.full((2, 5, 5), 7, dtype=np.uint16)
    np.testing.assert_array_equal(
        se.normalise_frames_to_uint8(flat), _normalise_frames_to_uint8(flat)
    )
