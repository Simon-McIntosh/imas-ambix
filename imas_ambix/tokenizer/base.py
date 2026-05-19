"""Tokenizer interfaces shared by the frame and signal implementations.

Every concrete tokenizer (Open-MAGVIT2 frame tokenizer, Chronos signal
tokenizer, PatchTST patch tokenizer) implements one of the protocols in
this module. The world-model training loop consumes only these
protocols and never needs to know the underlying implementation.

Two specific protocols extend the generic :class:`Tokenizer`:

- :class:`FrameTokenizer` — input shape ``(T, H, W)`` (single camera)
  or ``(T, H, W, C)`` (multi-channel), output ``EncodedFrames``.
- :class:`SignalTokenizer` — input is an ``xarray.Dataset`` (or a single
  ``DataArray``) at the model time grid, output ``EncodedSignals``.

The output dataclasses carry the **global** token ids
(post-:func:`TokenRegistry.shift`) so the multi-modal aggregator can
concatenate them without further bookkeeping.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np
    import xarray as xr


@dataclass(frozen=True)
class EncodedFrames:
    """Output of a :class:`FrameTokenizer.encode` call.

    ``token_ids`` are **global** ids — i.e. shifted into the registry's
    namespace. ``shape`` is the encoder's native output shape *before*
    flattening to 1-D for the model stream (kept for round-trip).
    """

    token_ids: np.ndarray  # int32, shape (T_compressed, h, w)
    shape: tuple[int, ...]
    tokenizer_name: str
    metadata: dict[str, object]


@dataclass(frozen=True)
class EncodedSignals:
    """Output of a :class:`SignalTokenizer.encode` call.

    ``token_ids`` is shape ``(T, n_channels)`` — one token per channel
    per model timestep. ``channel_names`` records the per-column meaning
    so the model knows what each id position represents.
    """

    token_ids: np.ndarray  # int32, shape (T, n_channels)
    channel_names: tuple[str, ...]
    tokenizer_name: str
    metadata: dict[str, object]


@runtime_checkable
class Tokenizer(Protocol):
    """The minimal contract every tokenizer must satisfy."""

    name: str
    """Stable identifier used by :class:`TokenRegistry` for namespacing."""

    vocab_size: int
    """Number of distinct **local** token ids this tokenizer emits."""

    def encode(self, data: object) -> object:
        """Map raw data to **global** token ids."""

    def decode(self, tokens: object) -> object:
        """Inverse-encode global tokens to (an approximation of) the input."""


@runtime_checkable
class FrameTokenizer(Tokenizer, Protocol):
    """A tokenizer for video / frame sequences."""

    spatial_compression: int
    """Per-axis spatial down-sample, e.g. 8 for Open-MAGVIT2."""

    temporal_compression: int
    """Frame-axis down-sample, e.g. 4."""

    def encode(self, frames: np.ndarray) -> EncodedFrames:  # type: ignore[override]
        """Encode a ``(T, H, W)`` or ``(T, H, W, C)`` frame tensor."""

    def decode(self, tokens: EncodedFrames) -> np.ndarray:  # type: ignore[override]
        """Decode back to the original frame shape (approximately)."""


@runtime_checkable
class SignalTokenizer(Tokenizer, Protocol):
    """A tokenizer for 1-D signal channels at the model time grid."""

    patch_size: int
    """How many native samples one output token aggregates."""

    def encode(self, ds: xr.Dataset) -> EncodedSignals:  # type: ignore[override]
        """Encode an xarray Dataset of 1-D signals into per-channel tokens."""

    def decode(self, tokens: EncodedSignals) -> xr.Dataset:  # type: ignore[override]
        """Decode back to (an approximation of) the input dataset."""
