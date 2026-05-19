"""Frame tokenizer wrappers.

Two implementations:

- :class:`PlaceholderFrameTokenizer` — a deterministic bit-packing
  scheme that works without any external dependency. Used for tests
  and end-to-end plumbing checks before Open-MAGVIT2 weights are
  downloaded.
- :class:`OpenMagvit2Tokenizer` (planned, not yet implemented) — the
  real Apache-2.0 Open-MAGVIT2 model from TencentARC. Weights live at
  ``/work/projects/imas_gpu/mast-tokens/v1/open-magvit2/``.

Both honour the :class:`FrameTokenizer` protocol and the global
:class:`TokenRegistry`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

from imas_ambix.tokenizer.base import EncodedFrames
from imas_ambix.tokenizer.registry import registry


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


@dataclass
class OpenMagvit2Tokenizer:
    """Open-MAGVIT2 wrapper — not yet implemented.

    See ``plans/tokenizers.md`` §2 for the rationale. Implementation
    notes:

    - Apache-2.0, source repo: https://github.com/TencentARC/Open-MAGVIT2
    - Compression: 8 (spatial) × 4 (temporal)
    - Codebook: 2^18 LFQ tokens (vocab_size = 262144)
    - Weights to be staged at
      ``/work/projects/imas_gpu/mast-tokens/v1/open-magvit2/``
    - First-use will fine-tune the **decoder** on plasma imagery if the
      ImageNet checkpoint's rFID is > 5; the encoder + codebook stay
      frozen so the registry allocation never moves.

    Raising :class:`NotImplementedError` at construction makes us notice
    at module-load time if downstream code accidentally picks this
    instead of the placeholder.
    """

    name: str = "frames_open_magvit2_v1"
    spatial_compression: int = 8
    temporal_compression: int = 4
    vocab_size: int = 1 << 18

    def __post_init__(self) -> None:
        raise NotImplementedError(
            "OpenMagvit2Tokenizer is not yet wired up — see "
            "plans/tokenizers.md §2 for the rollout plan"
        )

    def encode(self, frames: np.ndarray) -> EncodedFrames:  # pragma: no cover
        raise NotImplementedError

    def decode(self, tokens: EncodedFrames) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError
