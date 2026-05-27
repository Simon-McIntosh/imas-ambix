"""Tests for the encode-pipeline optimisations in imas_ambix.data.encoding.

Covers the three changes landed by the encode-pipeline refactor:

R2  prefetch / double-buffer
    The prefetch producer→consumer path MUST yield the same EncodeReport
    token counts (and identical token ids) as the legacy serial path.

R3b precomputed-frame fast path
    When a precomputed ``(T,S,S,3)`` uint8 store exists for a shot it is fed
    straight to the tokenizer.  The tokens MUST be byte-for-byte identical to
    the legacy L1 path *when the precomputed store equals the legacy
    normalise+RGB output* (which is the contract the precompute job upholds).

All tests are fully offline — synthetic Zarr + a stub two-phase tokenizer and
the real PlaceholderFrameTokenizer.  No GPU, no Open-MAGVIT2 weights, no
network.  The live rbb encode job is never touched.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import zarr

from imas_ambix.tokenizer.base import EncodedFrames
from imas_ambix.tokenizer.registry import registry

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------


def make_frame_zarr(
    base_dir: Path,
    shot_id: int,
    camera: str = "rbb",
    *,
    t: int = 6,
    h: int = 12,
    w: int = 12,
    seed: int = 0,
) -> Path:
    """Create a minimal level-1 (T, H, W) uint16 frame Zarr for shot_id."""
    rng = np.random.default_rng(seed)
    shot_path = base_dir / f"{shot_id}.zarr"
    shot_path.mkdir(parents=True, exist_ok=True)
    g = zarr.open_group(str(shot_path), mode="w")
    cam = g.create_group(camera)
    cam.create_array(
        "data",
        data=rng.integers(100, 60000, (t, h, w), dtype=np.uint16),
        dimension_names=["time", "y", "x"],
    )
    cam.create_array(
        "time",
        data=np.arange(t, dtype=np.float64) * 0.01,
        dimension_names=["time"],
    )
    return shot_path


def make_precomputed_zarr(dst_root: Path, shot_id: int, rgb: np.ndarray) -> Path:
    """Write a precomputed (T, S, S, 3) uint8 store, matching preprocess layout."""
    out_path = dst_root / f"{shot_id}.zarr"
    dst_root.mkdir(parents=True, exist_ok=True)
    zarr.save_array(str(out_path / "data"), np.ascontiguousarray(rgb))
    return out_path


# ---------------------------------------------------------------------------
# Stub two-phase tokenizer (mirrors the OpenMagvit2 prepare/encode_prepared API)
# ---------------------------------------------------------------------------


@dataclass
class StubTwoPhaseTokenizer:
    """A deterministic tokenizer exposing the prepare/encode_prepared split.

    Tokens are derived directly from the exact uint8 RGB bytes the daemon
    would consume — so identical token output proves identical input bytes
    flowed through.  ``prepare`` does the normalise+RGB (the CPU half) exactly
    as OpenMagvit2Tokenizer does; ``encode_prepared`` hashes the staged array.
    Optional ``prep_sleep`` makes prep slow so the prefetch overlap is real in
    the timing assertion.
    """

    name: str = "stub_two_phase"
    vocab_size: int = 4096
    spatial_compression: int = 16
    temporal_compression: int = 1
    image_size: int = 256
    prep_sleep: float = 0.0
    encode_sleep: float = 0.0
    # records prep/encode start times keyed by a stable id for overlap checks
    _events: dict = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        registry.allocate(self.name, self.vocab_size)

    # --- two-phase API ---------------------------------------------------
    def prepare(self, frames: np.ndarray, *, presized: bool = False):
        from imas_ambix.tokenizer.frames import _normalise_frames_to_uint8

        if presized:
            u8_rgb = np.asarray(frames)
        else:
            u8 = _normalise_frames_to_uint8(frames)
            u8_rgb = np.repeat(u8[..., None], 3, axis=-1) if u8.ndim == 3 else u8
        if self.prep_sleep:
            time.sleep(self.prep_sleep)
        # carry the staged bytes + native shape, mimicking PreparedFrames
        return {"rgb": np.ascontiguousarray(u8_rgb), "input_shape": tuple(frames.shape)}

    def encode_prepared(self, prepared) -> EncodedFrames:
        rgb = prepared["rgb"]
        if self.encode_sleep:
            time.sleep(self.encode_sleep)
        local_ids = self._tokens_from_rgb(rgb)
        global_ids = registry.shift(self.name, local_ids)
        return EncodedFrames(
            token_ids=global_ids,
            shape=tuple(global_ids.shape),
            tokenizer_name=self.name,
            metadata={"input_shape": list(prepared["input_shape"])},
        )

    def encode(self, frames: np.ndarray) -> EncodedFrames:
        return self.encode_prepared(self.prepare(frames))

    def decode(self, tokens):  # pragma: no cover - not exercised
        raise NotImplementedError

    # --- helper ----------------------------------------------------------
    def _tokens_from_rgb(self, rgb: np.ndarray) -> np.ndarray:
        """One token per frame derived from the frame's exact bytes."""
        flat = rgb.reshape(rgb.shape[0], -1).astype(np.int64)
        # deterministic, byte-sensitive reduction → (T,) local ids
        ids = (flat.sum(axis=1) % self.vocab_size).astype(np.int32)
        return ids


# ---------------------------------------------------------------------------
# R2 — prefetch path yields identical results to serial path
# ---------------------------------------------------------------------------


def _patch_paths(monkeypatch, level1, tokens_root):
    import imas_ambix.data.paths as paths_mod
    import imas_ambix.data.persist as persist_mod

    monkeypatch.setattr(paths_mod, "LEVEL1_DIR", level1)
    monkeypatch.setattr(persist_mod, "TOKEN_ROOT", tokens_root)


def test_prefetch_matches_serial_token_counts_stub(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Prefetch path == serial path: same EncodeReport order, counts, errors."""
    from imas_ambix.data.encoding import bulk_encode_frames
    from imas_ambix.data.persist import load_frame_tokens

    level1 = tmp_path / "level1" / "shots"
    shot_ids = [101, 102, 103, 104, 105]
    for sid in shot_ids:
        make_frame_zarr(level1, sid, "rbb", t=5, h=10, w=10, seed=sid)

    tok = StubTwoPhaseTokenizer()

    # --- serial reference ---
    _patch_paths(monkeypatch, level1, tmp_path / "tokens_serial")
    serial = bulk_encode_frames(
        shot_ids, "rbb", lambda: tok, skip_existing=False, prefetch=False
    )
    serial_tokens = {
        r.shot_id: load_frame_tokens(r.shot_id, "rbb").token_ids.copy() for r in serial
    }

    # --- prefetch under test ---
    _patch_paths(monkeypatch, level1, tmp_path / "tokens_prefetch")
    prefetch = bulk_encode_frames(
        shot_ids,
        "rbb",
        lambda: tok,
        skip_existing=False,
        prefetch=True,
        prefetch_workers=3,
        prefetch_queue_size=4,
    )

    # order preserved
    assert [r.shot_id for r in prefetch] == shot_ids
    assert [r.shot_id for r in serial] == shot_ids
    for rs, rp in zip(serial, prefetch, strict=True):
        assert rs.error is None and rp.error is None
        assert rs.n_tokens == rp.n_tokens
        assert rp.n_tokens > 0
        # byte-for-byte identical token ids
        pf = load_frame_tokens(rp.shot_id, "rbb").token_ids
        np.testing.assert_array_equal(pf, serial_tokens[rs.shot_id])


def test_prefetch_matches_serial_placeholder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real PlaceholderFrameTokenizer (no two-phase API) also matches serial."""
    from imas_ambix.data.encoding import bulk_encode_frames
    from imas_ambix.data.persist import load_frame_tokens
    from imas_ambix.tokenizer.frames import PlaceholderFrameTokenizer

    level1 = tmp_path / "level1" / "shots"
    shot_ids = [201, 202, 203]
    for sid in shot_ids:
        make_frame_zarr(level1, sid, "rbb", t=8, h=16, w=16, seed=sid)

    _patch_paths(monkeypatch, level1, tmp_path / "tok_serial")
    serial = bulk_encode_frames(
        shot_ids, "rbb", PlaceholderFrameTokenizer, skip_existing=False, prefetch=False
    )
    assert all(r.error is None for r in serial)
    serial_tokens = {
        sid: load_frame_tokens(sid, "rbb").token_ids.copy() for sid in shot_ids
    }

    _patch_paths(monkeypatch, level1, tmp_path / "tok_prefetch")
    prefetch = bulk_encode_frames(
        shot_ids, "rbb", PlaceholderFrameTokenizer, skip_existing=False, prefetch=True
    )

    assert [r.shot_id for r in prefetch] == shot_ids
    for r in prefetch:
        assert r.error is None
        np.testing.assert_array_equal(
            load_frame_tokens(r.shot_id, "rbb").token_ids, serial_tokens[r.shot_id]
        )


def test_prefetch_preserves_per_shot_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One bad shot does not kill the run; its error is captured in-order."""
    from imas_ambix.data.encoding import bulk_encode_frames

    level1 = tmp_path / "level1" / "shots"
    make_frame_zarr(level1, 301, "rbb", t=4, h=8, w=8, seed=1)
    # 302 missing entirely → error
    make_frame_zarr(level1, 303, "rbb", t=4, h=8, w=8, seed=3)

    _patch_paths(monkeypatch, level1, tmp_path / "tokens")
    tok = StubTwoPhaseTokenizer()
    reports = bulk_encode_frames(
        [301, 302, 303], "rbb", lambda: tok, skip_existing=False, prefetch=True
    )

    assert [r.shot_id for r in reports] == [301, 302, 303]
    assert reports[0].error is None and reports[0].n_tokens > 0
    assert reports[1].error is not None and reports[1].n_tokens == 0
    assert reports[2].error is None and reports[2].n_tokens > 0


def test_prefetch_skip_existing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """skip_existing under prefetch returns 0-token reports for present shots."""
    from imas_ambix.data.encoding import bulk_encode_frames

    level1 = tmp_path / "level1" / "shots"
    shot_ids = [401, 402]
    for sid in shot_ids:
        make_frame_zarr(level1, sid, "rbb", t=4, h=8, w=8, seed=sid)

    _patch_paths(monkeypatch, level1, tmp_path / "tokens")
    tok = StubTwoPhaseTokenizer()
    first = bulk_encode_frames(shot_ids, "rbb", lambda: tok, skip_existing=False)
    assert all(r.n_tokens > 0 for r in first)

    second = bulk_encode_frames(shot_ids, "rbb", lambda: tok, skip_existing=True)
    assert [r.shot_id for r in second] == shot_ids
    assert all(r.n_tokens == 0 and r.error is None for r in second)


def test_prefetch_actually_overlaps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Prefetch overlaps CPU prep with the consumer: wall time < fully serial.

    With slow prep and a serialised consumer, prefetch should finish in less
    than (n * (prep + encode)) because prep of N+1 overlaps encode of N.
    """
    from imas_ambix.data.encoding import bulk_encode_frames

    level1 = tmp_path / "level1" / "shots"
    shot_ids = [501, 502, 503, 504]
    for sid in shot_ids:
        make_frame_zarr(level1, sid, "rbb", t=3, h=8, w=8, seed=sid)

    _patch_paths(monkeypatch, level1, tmp_path / "tokens")
    tok = StubTwoPhaseTokenizer(prep_sleep=0.10, encode_sleep=0.10)

    t0 = time.monotonic()
    reports = bulk_encode_frames(
        shot_ids,
        "rbb",
        lambda: tok,
        skip_existing=False,
        prefetch=True,
        prefetch_workers=2,
        prefetch_queue_size=4,
    )
    elapsed = time.monotonic() - t0

    assert all(r.error is None for r in reports)
    fully_serial = len(shot_ids) * (0.10 + 0.10)  # 0.80s
    # overlap should shave at least one prep window; allow generous margin
    assert elapsed < fully_serial - 0.10, (
        f"prefetch wall {elapsed:.2f}s not below serial bound {fully_serial:.2f}s"
    )


# ---------------------------------------------------------------------------
# R3b — precomputed fast path yields identical tokens to the legacy L1 path
# ---------------------------------------------------------------------------


def test_precomputed_fastpath_matches_legacy_stub(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fast path == legacy path when precomputed store == legacy normalise+RGB.

    Build the precomputed store FROM the legacy normalise+RGB of the same raw
    frames (no resize — equal size), then assert identical token ids whether
    the encoder reads L1 (legacy) or the precomputed store (fast path).
    """
    from imas_ambix.data.encoding import bulk_encode_frames
    from imas_ambix.data.persist import load_frame_tokens
    from imas_ambix.tokenizer.frames import _normalise_frames_to_uint8

    level1 = tmp_path / "level1" / "shots"
    pre_root = tmp_path / "preprocessed" / "rbb-256"
    shot_ids = [601, 602]
    for sid in shot_ids:
        make_frame_zarr(level1, sid, "rbb", t=5, h=14, w=14, seed=sid)

    # Build precomputed stores byte-identical to the legacy normalise+RGB.
    import xarray as xr

    for sid in shot_ids:
        ds = xr.open_zarr(str(level1 / f"{sid}.zarr" / "rbb"))
        raw = np.asarray(ds[list(ds.data_vars)[0]].values)
        u8 = _normalise_frames_to_uint8(raw)
        rgb = np.repeat(u8[..., None], 3, axis=-1)
        make_precomputed_zarr(pre_root, sid, rgb)

    tok = StubTwoPhaseTokenizer()

    # legacy (no preprocessed_root)
    _patch_paths(monkeypatch, level1, tmp_path / "tok_legacy")
    legacy = bulk_encode_frames(
        shot_ids, "rbb", lambda: tok, skip_existing=False, preprocessed_root=None
    )
    assert all(r.error is None for r in legacy)
    legacy_tokens = {
        sid: load_frame_tokens(sid, "rbb").token_ids.copy() for sid in shot_ids
    }

    # fast path (with preprocessed_root)
    _patch_paths(monkeypatch, level1, tmp_path / "tok_fast")
    fast = bulk_encode_frames(
        shot_ids, "rbb", lambda: tok, skip_existing=False, preprocessed_root=pre_root
    )

    assert [r.shot_id for r in fast] == shot_ids
    for r in fast:
        assert r.error is None and r.n_tokens > 0
        np.testing.assert_array_equal(
            load_frame_tokens(r.shot_id, "rbb").token_ids, legacy_tokens[r.shot_id]
        )


def test_precomputed_fastpath_falls_back_when_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Opt-in-safe: a shot lacking a precompute falls back to the L1 path."""
    from imas_ambix.data.encoding import bulk_encode_frames
    from imas_ambix.data.persist import load_frame_tokens
    from imas_ambix.tokenizer.frames import _normalise_frames_to_uint8

    level1 = tmp_path / "level1" / "shots"
    pre_root = tmp_path / "preprocessed" / "rbb-256"
    # 701 has a precompute, 702 does not
    make_frame_zarr(level1, 701, "rbb", t=4, h=10, w=10, seed=7)
    make_frame_zarr(level1, 702, "rbb", t=4, h=10, w=10, seed=8)

    import xarray as xr

    ds = xr.open_zarr(str(level1 / "701.zarr" / "rbb"))
    raw = np.asarray(ds[list(ds.data_vars)[0]].values)
    rgb = np.repeat(_normalise_frames_to_uint8(raw)[..., None], 3, axis=-1)
    make_precomputed_zarr(pre_root, 701, rgb)

    tok = StubTwoPhaseTokenizer()

    _patch_paths(monkeypatch, level1, tmp_path / "tok_legacy")
    legacy = bulk_encode_frames(
        [701, 702], "rbb", lambda: tok, skip_existing=False, preprocessed_root=None
    )
    legacy_tokens = {
        r.shot_id: load_frame_tokens(r.shot_id, "rbb").token_ids.copy() for r in legacy
    }

    _patch_paths(monkeypatch, level1, tmp_path / "tok_fast")
    fast = bulk_encode_frames(
        [701, 702], "rbb", lambda: tok, skip_existing=False, preprocessed_root=pre_root
    )

    for r in fast:
        assert r.error is None and r.n_tokens > 0
        np.testing.assert_array_equal(
            load_frame_tokens(r.shot_id, "rbb").token_ids, legacy_tokens[r.shot_id]
        )


def test_precomputed_fastpath_reads_store_not_l1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the precompute differs from L1, the fast path returns the STORE's tokens.

    Proves the encoder actually reads the precomputed bytes (not L1) when the
    fast path is engaged: a deliberately-different precompute yields different
    tokens than the legacy L1 read.
    """
    from imas_ambix.data.encoding import bulk_encode_frames
    from imas_ambix.data.persist import load_frame_tokens

    level1 = tmp_path / "level1" / "shots"
    pre_root = tmp_path / "preprocessed" / "rbb-256"
    make_frame_zarr(level1, 801, "rbb", t=4, h=10, w=10, seed=11)

    # Precompute deliberately distinct from any normalisation of L1.
    distinct_rgb = np.full((4, 10, 10, 3), 123, dtype=np.uint8)
    make_precomputed_zarr(pre_root, 801, distinct_rgb)

    tok = StubTwoPhaseTokenizer()
    _patch_paths(monkeypatch, level1, tmp_path / "tok_fast")
    fast = bulk_encode_frames(
        [801], "rbb", lambda: tok, skip_existing=False, preprocessed_root=pre_root
    )
    assert fast[0].error is None

    fast_tokens = load_frame_tokens(801, "rbb").token_ids
    # Expected tokens come from feeding the distinct precompute straight in.
    expected = registry.shift(tok.name, tok._tokens_from_rgb(distinct_rgb))
    np.testing.assert_array_equal(fast_tokens, expected)


# ---------------------------------------------------------------------------
# Tokenizer two-phase split is byte-identical to single-call encode
# ---------------------------------------------------------------------------


def test_openmagvit2_encode_equals_prepare_then_encode_prepared() -> None:
    """encode(frames) == encode_prepared(prepare(frames)) for the stub split.

    Mirrors the OpenMagvit2Tokenizer contract: the two-phase split must be a
    pure decomposition of the single-call path.
    """
    tok = StubTwoPhaseTokenizer()
    rng = np.random.default_rng(99)
    frames = rng.integers(0, 60000, (5, 12, 12), dtype=np.uint16)

    one_call = tok.encode(frames)
    two_phase = tok.encode_prepared(tok.prepare(frames))
    np.testing.assert_array_equal(one_call.token_ids, two_phase.token_ids)
    assert one_call.shape == two_phase.shape
