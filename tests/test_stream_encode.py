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

# --- Stub prepare + encode --------------------------------------------------
#
# The real pipeline resizes each shot's native-resolution frames to 256²
# *per-shot, before the cross-shot buffer* (so frames from shots with
# different native (H,W) stack cleanly). The CPU parity stub mirrors that
# split: `_stub_prepare` does a cheap deterministic per-shot "resize" to a
# uniform shape, and `_stub_encode` fingerprints the prepared frame. Both the
# stream path and the per-shot reference apply the SAME prepare, so the parity
# assertion proves the reassembly is byte-identical irrespective of native
# resolution.

# Fixed uniform "prepared" frame size for the stub (analogue of 256²). Small,
# so the test stays fast, but >1 so resize-by-mean actually mixes pixels.
STUB_SIZE = 8


def _stub_prepare(shot_u8_rgb: np.ndarray) -> np.ndarray:
    """Deterministic per-shot 'resize' (T,H,W,3) uint8 -> (T,STUB_SIZE,STUB_SIZE,3).

    Mirrors the structural role of ``frames_to_input``: applied per-shot where
    the frames are uniform, producing fixed-shape slices that stack across
    shots regardless of native (H,W). It is a pure, per-frame function (block
    mean down to a fixed grid), so splitting a shot's frames across batches and
    reassembling is byte-identical to processing the shot whole.
    """
    arr = np.asarray(shot_u8_rgb)
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(f"expected (T,H,W,3), got {arr.shape}")
    t, h, w, c = arr.shape
    out = np.empty((t, STUB_SIZE, STUB_SIZE, c), dtype=np.uint8)
    # Block-mean each frame into a STUB_SIZE×STUB_SIZE grid (a per-frame,
    # deterministic reduction — the stub analogue of a bilinear downsample).
    ys = np.linspace(0, h, STUB_SIZE + 1).astype(int)
    xs = np.linspace(0, w, STUB_SIZE + 1).astype(int)
    for fi in range(t):
        for yi in range(STUB_SIZE):
            for xi in range(STUB_SIZE):
                block = arr[fi, ys[yi] : ys[yi + 1], xs[xi] : xs[xi + 1], :]
                out[fi, yi, xi, :] = block.reshape(-1, c).mean(axis=0).astype(np.uint8)
    return out


# A deterministic per-frame "encode": maps each prepared (uniform-shape) frame
# to a (16,16) int64 token grid using only that frame's bytes — exactly the
# property the real model has (no cross-frame state on the encode path).
# We derive a stable per-frame fingerprint and tile it, so identical prepared
# frames always produce identical tokens regardless of which batch they land in.
def _stub_encode(model, prepared_batch: np.ndarray) -> np.ndarray:
    b = prepared_batch.shape[0]
    out = np.empty((b, se.TOKEN_HW, se.TOKEN_HW), dtype=np.int64)
    flat = np.asarray(prepared_batch).reshape(b, -1).astype(np.int64)
    # Per-frame deterministic value in [0, 2^18) — within the magvit2 vocab.
    fp = (flat.sum(axis=1) * 2654435761) % se.VOCAB_SIZE
    for i in range(b):
        # Vary spatially so reshape/transpose bugs surface, but stay a pure
        # function of the frame contents.
        grid = (fp[i] + np.arange(se.TOKEN_HW * se.TOKEN_HW)) % se.VOCAB_SIZE
        out[i] = grid.reshape(se.TOKEN_HW, se.TOKEN_HW)
    return out


def _make_frames(
    shot_id: int,
    n_frames: int,
    rng: np.random.Generator,
    hw: tuple[int, int] = (40, 56),
) -> np.ndarray:
    """A fake L1 raw frame array (T,H,W) uint16, distinct per shot.

    *hw* defaults to a non-square, non-256 shape so the resize/normalise path
    is exercised; pass different *hw* per shot to mix native resolutions
    across the cross-shot stream (the bug the post-stack resize design hit).
    """
    h, w = hw
    base = rng.integers(0, 4000, size=(n_frames, h, w), dtype=np.uint16)
    return base + shot_id  # shift so shots differ


def _per_shot_reference(shot_ids, frame_arrays, camera) -> dict[int, np.ndarray]:
    """Encode each shot independently with the stub → global int32 tokens.

    Mirrors the live single-shot path: normalise→RGB (presized=False) then
    per-frame stub encode, then registry shift (+offset, int32).
    """
    ref: dict[int, np.ndarray] = {}
    for sid in shot_ids:
        raw = frame_arrays[sid]
        u8_rgb = se.frames_to_rgb_uint8(raw, presized=False)
        prepared = _stub_prepare(u8_rgb)  # per-shot resize, exactly as stream
        toks = _stub_encode(None, prepared)  # (T,16,16) int64
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
            prepare_fn=_stub_prepare,
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


@pytest.mark.parametrize("batch_frames", [1, 7, 32, 256])
def test_stream_reassembly_mixed_native_resolutions(tmp_path, batch_frames):
    """Cross-shot batches that MIX shots of DIFFERENT native (H,W) must still
    produce byte-identical per-shot tokens.

    This is the regression for the byte-diff failure (jobs 1208239 / 1208235):
    the old design stacked native-resolution frames across shots *before*
    resizing, so a batch spanning shots of different (H,W) hit
    ``ValueError: all input arrays must have the same shape``. The fix resizes
    per-shot before the cross-shot buffer; this test exercises real-world
    geometries (536×560, 402×512, 1024×512) with frame counts chosen so that
    for several batch sizes a single batch contains frames from two shots of
    DIFFERENT native resolution.
    """
    rng = np.random.default_rng(7)
    camera = "rbb"
    # (frame_count, (H, W)) per shot — geometries lifted from the live path.
    spec = {
        301: (50, (536, 560)),
        302: (30, (402, 512)),
        303: (40, (1024, 512)),
        304: (5, (200, 248)),
        305: (17, (536, 560)),
    }
    shot_ids = list(spec)
    counts = {sid: spec[sid][0] for sid in shot_ids}
    frame_arrays = {
        sid: _make_frames(sid, spec[sid][0], rng, hw=spec[sid][1]) for sid in shot_ids
    }

    reference = _per_shot_reference(shot_ids, frame_arrays, camera)

    dataset = _DictDataset(shot_ids, frame_arrays, camera)
    stream_root = tmp_path / "frames-stream"

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
            prepare_fn=_stub_prepare,
        )
    finally:
        se.ShotFrameDataset = orig_ctor  # type: ignore[assignment]

    assert stats.shots_fail == 0, stats.load_errors
    assert stats.shots_ok == len(shot_ids)
    assert stats.frames_encoded == sum(counts.values())
    assert stats.write_errors == []

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
    """The hardcoded REGISTRY_OFFSET must equal the offset the magvit2 frame
    block gets as the first non-control allocation, or stream tokens diverge
    from the live path.

    Uses a fresh :class:`TokenRegistry` rather than the process-global
    singleton: in a real encode process magvit2 is the first tokenizer to
    allocate, so its start is exactly the control-range end. The shared
    singleton is polluted by other tokenizers allocated earlier in the test
    session, which would shift the offset and make this order-dependent.
    """
    from imas_ambix.tokenizer.registry import TokenRegistry

    reg = TokenRegistry()
    start, end = reg.allocate(se.TOKENIZER_NAME, se.VOCAB_SIZE)
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


def test_encode_batch_indices_subchunks_at_model_forward_batch():
    """encode_batch_indices must feed model.encode in MODEL_FORWARD_BATCH-sized
    chunks regardless of the (larger, cross-shot) input batch.

    The Open-MAGVIT2 VQModel forward is NOT batch-size invariant (GPU byte-diff
    2026-05-27: batch-4 vs batch-256 of the same frames differ ~12%, even with
    cuDNN determinism forced). The live daemon runs the forward in chunks of
    OpenMagvit2Tokenizer.batch_size; the stream must match that chunk size to
    stay byte-identical. This stub records the per-call batch sizes to prove
    the sub-chunking, and returns batch-position-independent tokens to prove
    the reassembly preserves frame order.
    """
    import torch

    class _RecordingModel:
        use_ema = False

        def __init__(self):
            self.seen_batch_sizes = []

        def encode(self, x):
            b = x.shape[0]
            self.seen_batch_sizes.append(b)
            # Per-frame deterministic ids from the frame's mean (pure function
            # of the frame, independent of batch position).
            flat = x.reshape(b, -1).to(torch.float64)
            base = (flat.mean(dim=1).abs() * 1000).to(torch.int64) % se.VOCAB_SIZE
            idx = (
                base[:, None] + torch.arange(se.TOKEN_HW * se.TOKEN_HW)
            ) % se.VOCAB_SIZE
            return None, None, idx.reshape(-1), None

    rng = np.random.default_rng(11)
    frames = rng.integers(0, 256, size=(10, 40, 56, 3), dtype=np.uint8)
    images = se.frames_to_input(frames, dtype=torch.float32)  # (10,3,256,256)

    model = _RecordingModel()
    out = se.encode_batch_indices(model, images, "cpu")
    assert out.shape == (10, se.TOKEN_HW, se.TOKEN_HW)
    # 10 frames at MODEL_FORWARD_BATCH=4 -> chunks [4,4,2].
    assert model.seen_batch_sizes == [4, 4, 2], model.seen_batch_sizes
    assert all(b <= se.MODEL_FORWARD_BATCH for b in model.seen_batch_sizes)

    # The reassembled output must be independent of how the caller batches:
    # encoding the whole 10 in one call equals encoding each frame's chunk.
    model2 = _RecordingModel()
    per_frame = np.concatenate(
        [
            se.encode_batch_indices(model2, images[i : i + 1], "cpu")
            for i in range(10)
        ],
        axis=0,
    )
    np.testing.assert_array_equal(out, per_frame)
    assert se.MODEL_FORWARD_BATCH == 4  # must track the live daemon default


# ===========================================================================
# Hardening tests (graceful SIGTERM, per-batch watchdog, CPU-runnable load).
# Motivated by docs/rca-node-drain-2026-05-27.md: a hung GPU process that
# cannot be reaped within SLURM's UnkillableStepTimeout auto-drains the node.
# ===========================================================================


def _two_shot_dict():
    """Two distinct shots with a handful of frames each, served in-memory."""
    rng = np.random.default_rng(42)
    counts = {201: 6, 202: 8}
    frame_arrays = {sid: _make_frames(sid, counts[sid], rng) for sid in counts}
    return list(counts), counts, frame_arrays


def test_frames_to_input_cpu_uses_float32():
    """CPU model load casts the model to float32, so the input cast must be
    float32 (not bf16) or model.encode raises a dtype mismatch. bf16 stays the
    GPU default to preserve byte-identity."""
    import torch

    rng = np.random.default_rng(3)
    frames = rng.integers(0, 256, size=(2, 40, 56, 3), dtype=np.uint8)

    # Default (no dtype) preserves historical GPU behaviour: bf16.
    assert se.frames_to_input(frames).dtype == torch.bfloat16
    # Explicit float32 for the CPU path.
    assert se.frames_to_input(frames, dtype=torch.float32).dtype == torch.float32
    # The real CPU code path inside stream_encode must feed a float32-castable
    # model: a float32 stub conv proves the dtype lines up end-to-end.

    class _F32Conv:
        """Minimal stand-in for VQModel: a real float32 conv2d so a bf16 input
        would raise the exact RCA dtype mismatch."""

        use_ema = False

        def __init__(self):
            self.conv = torch.nn.Conv2d(3, 3, 1).to(torch.float32).eval()

        def encode(self, x):
            self.conv(x)  # raises if x is bf16 and weight is f32
            b = x.shape[0]
            idx = torch.zeros(b * se.TOKEN_HW * se.TOKEN_HW, dtype=torch.int64)
            return None, None, idx, None

    # device='cpu' => stream_encode selects float32 input dtype; the conv runs.
    model = _F32Conv()
    out = se.encode_batch_indices(
        model, se.frames_to_input(frames, dtype=torch.float32), "cpu"
    )
    assert out.shape == (2, se.TOKEN_HW, se.TOKEN_HW)


def test_sigterm_handler_sets_flag_and_loop_breaks(tmp_path, monkeypatch):
    """A SIGTERM (simulated by setting STOP between shots) makes the encode
    loop stop pulling work; already-encoded shots are persisted by the writer,
    and the partially-buffered tail is NOT emitted (no truncated shots)."""
    import zarr

    shot_ids, counts, frame_arrays = _two_shot_dict()
    camera = "rbb"
    stream_root = tmp_path / "frames-stream"

    dataset = _DictDataset(shot_ids, frame_arrays, camera)
    monkeypatch.setattr(se, "ShotFrameDataset", lambda *a, **k: dataset)

    # Simulate a signal arriving mid-run: the stub sets STOP once shot 201's
    # frames are all encoded. batch_frames=1 makes each call one frame so we
    # can count deterministically. The between-shot STOP check then prevents
    # shot 202 from being pulled/emitted.
    seen = {"frames": 0}

    def _encode_one(model, batch):
        seen["frames"] += batch.shape[0]
        out = _stub_encode(model, batch)
        # After shot 201 (first 6 frames) fully encoded, request stop.
        if seen["frames"] >= counts[201]:
            se.STOP.set()
        return out

    se.STOP.clear()
    stats = se.stream_encode(
        shot_ids,
        camera,
        model=None,
        device="cpu",
        stream_root=stream_root,
        batch_frames=1,
        num_workers=0,
        encode_fn=_encode_one,
        prepare_fn=_stub_prepare,
    )

    assert stats.aborted is True
    # Shot 201 was fully encoded before STOP -> persisted.
    p201 = se.stream_frames_token_path(201, camera, stream_root)
    assert p201.exists(), "fully-encoded shot must be persisted on graceful stop"
    store = zarr.open_group(str(p201), mode="r")
    assert np.asarray(store["tokens"]).shape[0] == counts[201]
    # Shot 202 was only partially buffered -> NOT persisted (no truncated shot).
    p202 = se.stream_frames_token_path(202, camera, stream_root)
    assert not p202.exists(), "partially-streamed shot must NOT be persisted"
    # STOP left set is fine; next run clears it. Confirm a fresh run clears it.
    se.STOP.clear()


def test_watchdog_fires_on_slow_batch_and_stops(tmp_path, monkeypatch):
    """A deliberately-slow stub encode that exceeds the batch timeout triggers
    the watchdog, which sets STOP -> the run aborts cleanly instead of
    hanging. We use a tiny explicit batch_timeout_s so the test is fast."""
    shot_ids, counts, frame_arrays = _two_shot_dict()
    camera = "rbb"
    stream_root = tmp_path / "frames-stream"

    dataset = _DictDataset(shot_ids, frame_arrays, camera)
    monkeypatch.setattr(se, "ShotFrameDataset", lambda *a, **k: dataset)

    import time as _time

    calls = {"n": 0}

    def _slow_encode(model, batch):
        calls["n"] += 1
        # First batch is fast (so a median can form / watchdog arms cleanly);
        # the second batch hangs past the 0.3 s budget -> watchdog fires.
        if calls["n"] >= 2:
            # Sleep longer than the timeout; the watchdog should set STOP and
            # _encode raises StreamAborted on return, unwinding the loop.
            _time.sleep(1.5)
        return _stub_encode(model, batch)

    se.STOP.clear()
    t0 = _time.monotonic()
    stats = se.stream_encode(
        shot_ids,
        camera,
        model=None,
        device="cpu",
        stream_root=stream_root,
        batch_frames=4,  # multiple batches across the two shots
        num_workers=0,
        encode_fn=_slow_encode,
        prepare_fn=_stub_prepare,
        batch_timeout_s=0.3,
    )
    elapsed = _time.monotonic() - t0

    assert stats.aborted is True, "watchdog must abort the run"
    # The run must not hang far beyond the slow batch's sleep — it returns once
    # the slow batch finishes and the post-batch STOP check unwinds.
    assert elapsed < 5.0, f"run took {elapsed:.1f}s — watchdog did not stop it"
    se.STOP.clear()


def test_clean_run_does_not_abort(tmp_path, monkeypatch):
    """A normal run (no signal, no slow batch) completes with aborted=False and
    a generous watchdog budget that never fires — guards against the watchdog
    firing spuriously on healthy runs."""
    shot_ids, counts, frame_arrays = _two_shot_dict()
    camera = "rbb"
    stream_root = tmp_path / "frames-stream"

    dataset = _DictDataset(shot_ids, frame_arrays, camera)
    monkeypatch.setattr(se, "ShotFrameDataset", lambda *a, **k: dataset)

    se.STOP.clear()
    stats = se.stream_encode(
        shot_ids,
        camera,
        model=None,
        device="cpu",
        stream_root=stream_root,
        batch_frames=4,
        num_workers=0,
        encode_fn=_stub_encode,
        prepare_fn=_stub_prepare,
        batch_timeout_s=30.0,
    )
    assert stats.aborted is False
    assert stats.shots_ok == len(shot_ids)
    assert stats.shots_fail == 0
    assert se.STOP.is_set() is False


def test_signal_handler_installs_and_sets_stop():
    """_install_signal_handlers wires SIGTERM to set the module STOP flag.
    We install, fire SIGTERM at our own process, and confirm STOP is set, then
    restore default handlers so we don't perturb the test runner."""
    import os
    import signal as _signal

    prev_term = _signal.getsignal(_signal.SIGTERM)
    prev_int = _signal.getsignal(_signal.SIGINT)
    try:
        se.STOP.clear()
        se._install_signal_handlers()
        os.kill(os.getpid(), _signal.SIGTERM)
        # Signal delivery is synchronous on the main thread between bytecodes;
        # a short spin lets the handler run.
        import time as _t

        for _ in range(100):
            if se.STOP.is_set():
                break
            _t.sleep(0.01)
        assert se.STOP.is_set(), "SIGTERM did not set STOP"
    finally:
        _signal.signal(_signal.SIGTERM, prev_term)
        _signal.signal(_signal.SIGINT, prev_int)
        se.STOP.clear()
