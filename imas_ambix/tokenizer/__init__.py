"""Multi-modal tokenizers for FAIR-MAST data.

The Fusion World Model consumes a unified 1-D integer token stream that
interleaves camera frame tokens (from level-1 rbb/rba/rir) with signal
tokens (from level-2 magnetics / equilibrium / pf_active / summary). The
encode → train → decode pipeline must agree on:

1. A **global token vocabulary** (:mod:`registry`) so the model never
   sees tokenizer-local id collisions.
2. A **common time grid** (:mod:`alignment`) so cross-modal alignment
   is deterministic regardless of native sample rate.
3. Per-modality **encoders / decoders** (:mod:`frames`, :mod:`signals`)
   that share a base :class:`Tokenizer` interface.

See ``plans/tokenizers.md`` for the design rationale.
"""

from __future__ import annotations

from imas_ambix.tokenizer.base import (
    BlockKind,
    EncodedFrames,
    EncodedSignals,
    FrameTokenizer,
    SignalTokenizer,
    Tokenizer,
)
from imas_ambix.tokenizer.registry import TokenRegistry, registry

__all__ = [
    "BlockKind",
    "EncodedFrames",
    "EncodedSignals",
    "FrameTokenizer",
    "SignalTokenizer",
    "TokenRegistry",
    "Tokenizer",
    "registry",
]
