"""Frame tokenizer wrappers.

Two implementations:

- :class:`PlaceholderFrameTokenizer` — a deterministic bit-packing
  scheme that works without any external dependency. Used for tests
  and end-to-end plumbing checks before Open-MAGVIT2 weights are
  downloaded.
- :class:`OpenMagvit2Tokenizer` — the real Apache-2.0 Open-MAGVIT2
  model from TencentARC, driven via an isolated venv at
  ``/work/projects/imas_gpu/mast-tokens/v1/open-magvit2/``.

Both honour the :class:`FrameTokenizer` protocol and the global
:class:`TokenRegistry`.
"""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

from imas_ambix.tokenizer.base import EncodedFrames
from imas_ambix.tokenizer.registry import registry


@dataclass
class PreparedFrames:
    """CPU-side staged frame input ready for the GPU encode step.

    Produced by :meth:`OpenMagvit2Tokenizer.prepare` (the CPU/IO half of an
    encode: load + normalise + RGB-replicate + write ``.npy``) and consumed
    by :meth:`OpenMagvit2Tokenizer.encode_prepared` (the GPU half: worker
    subprocess request + registry shift).

    ``cleanup`` releases any temp resources (the staged ``.npy`` dir).  It is
    idempotent and called by ``encode_prepared``; callers that abandon a
    prepared item (e.g. on error) must call it themselves.
    """

    input_path: Path
    input_h: int
    input_w: int
    input_shape: tuple[int, ...]
    _tmpdir: Any = field(default=None, repr=False, compare=False)

    def cleanup(self) -> None:
        """Release the staged temp directory (idempotent)."""
        if self._tmpdir is not None:
            self._tmpdir.cleanup()
            self._tmpdir = None


def _normalise_frames_to_uint8(frames: np.ndarray) -> np.ndarray:
    """Map any-dtype frames into uint8 in [0, 255] for tokenizer ingestion.

    Camera raw is uint16; image tokenizers expect uint8 RGB or grayscale.
    We collapse the upper 8 bits of dynamic range using per-shot
    min/max — adequate for v0, more careful normalisation can come
    later from the camera attrs.
    """
    import numpy as np

    if frames.dtype == np.uint8:
        return frames
    f = frames.astype(np.float32)
    lo = float(f.min())
    hi = float(f.max())
    if hi <= lo:
        return np.zeros_like(f, dtype=np.uint8)
    return ((f - lo) * 255.0 / (hi - lo)).clip(0, 255).astype(np.uint8)


@dataclass
class PlaceholderFrameTokenizer:
    """A simple downsample-and-quantise frame tokenizer.

    Each frame is downsampled by ``spatial_compression`` and the pixel
    intensity is quantised to ``intensity_levels`` bins, giving a token
    per spatial cell. Temporal compression groups ``temporal_compression``
    frames by majority vote (the median).

    The output token field is a faithful local id in ``[0, vocab_size)``
    that decodes back to a low-resolution version of the input. This
    tokenizer is not a research-grade image tokenizer — it exists so the
    rest of the pipeline (registry, multi-modal aggregator, model loader,
    training loop) can be exercised end-to-end before Open-MAGVIT2 is
    plumbed in.
    """

    name: str = "frames_placeholder_v1"
    spatial_compression: int = 8
    temporal_compression: int = 4
    intensity_levels: int = 256  # → vocab_size = 256

    def __post_init__(self) -> None:
        self.vocab_size = self.intensity_levels
        # Allocate on import via the shared registry (idempotent).
        registry.allocate(self.name, self.vocab_size)

    def encode(self, frames: np.ndarray) -> EncodedFrames:
        """Encode `(T, H, W)` or `(T, H, W, C)` frames into global ids."""
        import numpy as np

        if frames.ndim == 4:
            # `(T, H, W, C)` — collapse channels by mean
            frames = frames.mean(axis=-1)
        if frames.ndim != 3:
            raise ValueError(
                f"frames must be (T,H,W) or (T,H,W,C), got shape {frames.shape}"
            )

        u8 = _normalise_frames_to_uint8(frames)
        t, h, w = u8.shape

        # Temporal compression: group by `temporal_compression` and take median
        tc = self.temporal_compression
        t_keep = (t // tc) * tc
        u8 = u8[:t_keep]
        u8 = u8.reshape(t_keep // tc, tc, h, w).astype(np.uint16).mean(axis=1)

        # Spatial compression: block-average
        sc = self.spatial_compression
        h_keep = (h // sc) * sc
        w_keep = (w // sc) * sc
        u8 = u8[:, :h_keep, :w_keep]
        u8 = u8.reshape(u8.shape[0], h_keep // sc, sc, w_keep // sc, sc)
        compressed = u8.mean(axis=(2, 4))  # `(T_c, h_c, w_c)` float

        # Quantise to intensity_levels bins → local id
        bin_size = 256 // self.intensity_levels
        local_ids = (compressed.astype(np.int32) // max(bin_size, 1)).clip(
            0, self.intensity_levels - 1
        )

        global_ids = registry.shift(self.name, local_ids)
        return EncodedFrames(
            token_ids=global_ids,
            shape=tuple(global_ids.shape),
            tokenizer_name=self.name,
            metadata={
                "input_shape": list(frames.shape),
                "spatial_compression": self.spatial_compression,
                "temporal_compression": self.temporal_compression,
                "intensity_levels": self.intensity_levels,
            },
        )

    def decode(self, tokens: EncodedFrames) -> np.ndarray:
        """Decode global ids back to a coarse approximation of the input."""
        import numpy as np

        start, _ = registry.allocate(self.name, self.vocab_size)
        local = np.asarray(tokens.token_ids, dtype=np.int64) - start
        bin_size = 256 // self.intensity_levels
        # Recover the bin midpoint as the decoded intensity
        intensity = (local * bin_size + bin_size // 2).clip(0, 255).astype(np.uint8)

        # Upsample by spatial_compression along H and W to approximate input
        sc = self.spatial_compression
        tc = self.temporal_compression
        t_c, h_c, w_c = intensity.shape
        # Repeat spatially
        out = intensity.repeat(sc, axis=1).repeat(sc, axis=2)
        # Repeat temporally
        out = out.repeat(tc, axis=0)
        return out


DEFAULT_MAGVIT2_ROOT = Path("/work/projects/imas_gpu/mast-tokens/v1/open-magvit2")


class OpenMagvit2UnavailableError(RuntimeError):
    """Raised when the Open-MAGVIT2 staging dir / venv / weights are missing."""


@dataclass
class OpenMagvit2Tokenizer:
    """Open-MAGVIT2 wrapper that drives the model via an isolated venv.

    Open-MAGVIT2 requires torch 2.1.1 + lightning 2.2.0 + transformers
    4.37.2 — pins that conflict with the ambix main venv (torch ≥ 2.6).
    The model therefore runs inside its own venv at
    ``/work/projects/imas_gpu/mast-tokens/v1/open-magvit2/.venv`` and we
    bridge over via the ``worker.py`` script under the same root, passing
    numpy arrays through ``.npy`` temp files.

    Each ``encode``/``decode`` call spawns a one-shot worker subprocess that
    loads the checkpoint (~5-10 s on CPU, sub-second once warm on GPU), so
    batch every shot's frames into a single call to amortise. This wrapper
    is for smoke tests, single-shot CLI use and calibration; high-throughput
    corpus encoding runs through the in-process streaming encoder in
    :mod:`imas_ambix.data.stream_encode`.

    **Device split:**

    - ``device="cpu"`` (default) — runs on any login or compute node; does
      not require GPU hardware.  Suitable for smoke tests and single-frame
      checks.  Per-frame encode/decode is ~30 s due to CPU-only torch.
    - ``device="cuda"`` — must be run inside a SLURM allocation on the
      ``betelgeuse`` partition (``--reservation=gpu_0003_grpA``).  The GPU
      node has no outbound network, but the venv and weights already live on
      GPFS.  Worker auto-selects batch size 8 when CUDA is used (vs. 4 for
      CPU).  If ``torch.cuda.is_available()`` is ``False`` on the target
      node, the worker exits immediately with a clear error naming the host.

    - Source: <https://github.com/TencentARC/Open-MAGVIT2> @ c1544ef (Apache-2.0)
    - Checkpoint: ``imagenet_256_L.ckpt`` (2^18 LFQ codebook, rFID 1.17 on ImageNet)
    - Compression: 16× spatial (256×256 → 16×16 tokens), no temporal compression
    """

    name: str = "frames_open_magvit2_v1"
    root: Path = DEFAULT_MAGVIT2_ROOT
    image_size: int = 256
    spatial_compression: int = 16
    temporal_compression: int = 1
    vocab_size: int = 1 << 18  # 262144
    device: str = "cpu"
    batch_size: int = 4

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        if not self.root.is_dir():
            raise OpenMagvit2UnavailableError(
                f"Open-MAGVIT2 staging directory not found at {self.root}. "
                "Clone github.com/TencentARC/Open-MAGVIT2 there and run "
                "`uv venv` + `uv pip install -r requirements.txt`."
            )
        self._python = self.root / ".venv" / "bin" / "python"
        self._worker = self.root / "worker.py"
        self._ckpt = self.root / "weights" / "imagenet_256_L.ckpt"
        for p in (self._python, self._worker, self._ckpt):
            if not p.exists():
                raise OpenMagvit2UnavailableError(f"missing {p}")
        registry.allocate(self.name, self.vocab_size)

    def prepare(self, frames: np.ndarray, *, presized: bool = False) -> PreparedFrames:
        """Stage the CPU/IO half of an encode and return a :class:`PreparedFrames`.

        This does normalise → RGB-replicate → write a temp ``.npy`` — every
        byte the daemon consumes — but issues no GPU work.  Pair it with
        :meth:`encode_prepared`.  ``encode`` is exactly
        ``encode_prepared(prepare(frames))`` so the single-call path is
        byte-for-byte unchanged.

        Parameters
        ----------
        frames:
            ``(T, H, W)`` or ``(T, H, W, 3)`` array.  When *presized* is
            ``True`` the input is taken to be already-normalised RGB uint8
            (e.g. read from a precomputed ``rbb-256`` store) and is written
            straight through without re-normalising — the daemon's resize to
            ``image_size`` is a near-identity on already-``image_size``²
            input.  When ``False`` (default) the legacy normalise+RGB path
            runs, preserving the running job's behaviour exactly.
        """
        import numpy as np

        if presized:
            u8_rgb = np.asarray(frames)
            if u8_rgb.dtype != np.uint8 or u8_rgb.ndim != 4 or u8_rgb.shape[-1] != 3:
                raise ValueError(
                    "presized frames must be (T,H,W,3) uint8, got "
                    f"shape={u8_rgb.shape} dtype={u8_rgb.dtype}"
                )
        else:
            u8 = _normalise_frames_to_uint8(frames)
            # The worker expects (T, H, W, 3) uint8
            if u8.ndim == 3:
                u8_rgb = np.repeat(u8[..., None], 3, axis=-1)
            elif u8.ndim == 4 and u8.shape[-1] == 3:
                u8_rgb = u8
            else:
                raise ValueError(f"frames must be (T,H,W) or (T,H,W,3), got {u8.shape}")

        input_h, input_w = u8_rgb.shape[1], u8_rgb.shape[2]
        tmp = tempfile.TemporaryDirectory(prefix="magvit2-enc-")
        in_path = Path(tmp.name) / "frames.npy"
        np.save(in_path, u8_rgb)
        return PreparedFrames(
            input_path=in_path,
            input_h=int(input_h),
            input_w=int(input_w),
            input_shape=tuple(int(x) for x in frames.shape),
            _tmpdir=tmp,
        )

    def encode_prepared(self, prepared: PreparedFrames) -> EncodedFrames:
        """Run the GPU half of an encode on a :class:`PreparedFrames`.

        Issues the daemon encode request on the pre-staged ``.npy`` and
        applies the registry shift.  Always releases the prepared item's
        temp dir before returning (including on error).
        """
        import numpy as np

        try:
            in_path = prepared.input_path
            out_path = in_path.parent / "tokens.npy"
            self._run_worker(
                "encode",
                ["--input", str(in_path), "--output", str(out_path)],
            )
            local_ids = np.load(out_path)
        finally:
            prepared.cleanup()

        global_ids = registry.shift(self.name, local_ids)
        return EncodedFrames(
            token_ids=global_ids,
            shape=tuple(global_ids.shape),
            tokenizer_name=self.name,
            metadata={
                "input_shape": list(prepared.input_shape),
                "model_image_size": self.image_size,
                "spatial_compression": self.spatial_compression,
                "temporal_compression": self.temporal_compression,
                "original_hw": [int(prepared.input_h), int(prepared.input_w)],
                "ckpt": "imagenet_256_L.ckpt",
            },
        )

    def encode(self, frames: np.ndarray) -> EncodedFrames:
        """Encode ``(T, H, W)`` or ``(T, H, W, C)`` frames into global token ids."""
        return self.encode_prepared(self.prepare(frames))

    def decode(self, tokens: EncodedFrames) -> np.ndarray:
        """Decode global token ids back to ``(T, H, W, 3)`` uint8 frames."""
        import numpy as np

        start, _ = registry.allocate(self.name, self.vocab_size)
        local = (np.asarray(tokens.token_ids, dtype=np.int64) - start).clip(
            0, self.vocab_size - 1
        )
        orig_hw = tokens.metadata.get("original_hw", [self.image_size, self.image_size])
        target = f"{int(orig_hw[0])},{int(orig_hw[1])}"

        with tempfile.TemporaryDirectory(prefix="magvit2-dec-") as tmp:
            in_path = Path(tmp) / "tokens.npy"
            out_path = Path(tmp) / "recon.npy"
            np.save(in_path, local.astype(np.int64))
            self._run_worker(
                "decode",
                [
                    "--input",
                    str(in_path),
                    "--output",
                    str(out_path),
                    "--target-hw",
                    target,
                ],
            )
            return np.load(out_path)

    def _run_worker(self, mode: str, extra: list[str]) -> None:
        """Invoke the worker subprocess in the isolated venv."""
        cmd = [
            str(self._python),
            str(self._worker),
            mode,
            "--device",
            self.device,
            "--image-size",
            str(self.image_size),
            "--batch-size",
            str(self.batch_size),
            "--ckpt",
            str(self._ckpt),
            *extra,
        ]
        proc = subprocess.run(
            cmd, capture_output=True, text=True, check=False, timeout=3600
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"Open-MAGVIT2 worker {mode!r} failed (exit {proc.returncode}):\n"
                f"{proc.stderr[-2000:]}"
            )
