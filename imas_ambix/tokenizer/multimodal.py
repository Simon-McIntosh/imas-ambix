"""Top-level per-shot tokenizer that ties frames + signals together.

Given a shot's level-1 camera and level-2 diagnostic Zarrs, the
:class:`ShotTokenizer` produces a single interleaved 1-D ``int32``
token array suitable for the world-model training loop. The layout
matches ``plans/world-model-v0.md`` §2:

```
<step_start>
<frame_tokens for this step>
<signal_tokens for this step>
<step_end>
```

…repeated per timestep on the model time grid.

A parallel ``block_kind`` array (``uint8``, same length) labels each
position with its :class:`~imas_ambix.tokenizer.base.BlockKind` code so
the training loop can weight the cross-entropy loss per block.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import xarray as xr

from imas_ambix.tokenizer.alignment import MODEL_HZ_DEFAULT
from imas_ambix.tokenizer.base import BlockKind
from imas_ambix.tokenizer.registry import CONTROL_TOKENS

if TYPE_CHECKING:
    from imas_ambix.tokenizer.base import FrameTokenizer, SignalTokenizer


@dataclass
class ShotTokenizer:
    """Aggregates per-modality tokenizers into a single shot stream."""

    frame_tokenizer: FrameTokenizer
    signal_tokenizer: SignalTokenizer
    model_hz: float = MODEL_HZ_DEFAULT
    sep_token: int = field(default=CONTROL_TOKENS["sep"])
    bos_token: int = field(default=CONTROL_TOKENS["bos"])
    eos_token: int = field(default=CONTROL_TOKENS["eos"])

    def encode_shot(
        self,
        frames: np.ndarray,
        signals: xr.Dataset,
        *,
        return_block_kind: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Encode one shot end-to-end into a 1-D int32 token stream.

        Args:
            frames: ``(T, H, W)`` or ``(T, H, W, C)`` array. The frame
                tokenizer compresses the time axis by its own factor.
            signals: ``xr.Dataset`` aligned to the model time grid.
            return_block_kind: When ``True``, return a
                ``(tokens, block_kind)`` tuple instead of only ``tokens``.
                The ``block_kind`` array is ``uint8``, same length as
                ``tokens``, with values from
                :class:`~imas_ambix.tokenizer.base.BlockKind`.

        Returns:
            1-D ``int32`` array of global token ids:
            ``[bos, (sep, frame_block, signal_block)*N, eos]``.
            When *return_block_kind* is ``True``, returns a tuple
            ``(tokens, block_kind)`` of the same length.
        """
        tokens, kinds = self._encode_shot_impl(frames, signals)
        if return_block_kind:
            return tokens, kinds
        return tokens

    def encode_shot_with_block_kind(
        self,
        frames: np.ndarray,
        signals: xr.Dataset,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Encode one shot and always return ``(tokens, block_kind)``.

        Preferred over ``encode_shot(return_block_kind=True)`` for new
        code — the return type is unambiguous.

        Args:
            frames: ``(T, H, W)`` or ``(T, H, W, C)`` array.
            signals: ``xr.Dataset`` aligned to the model time grid.

        Returns:
            ``(tokens, block_kind)`` — both 1-D, same length.
            ``tokens`` is ``int32``; ``block_kind`` is ``uint8`` with
            :class:`~imas_ambix.tokenizer.base.BlockKind` codes.
        """
        return self._encode_shot_impl(frames, signals)

    # ------------------------------------------------------------------
    # Internal implementation
    # ------------------------------------------------------------------

    def _encode_shot_impl(
        self,
        frames: np.ndarray,
        signals: xr.Dataset,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build the token stream and parallel block_kind array."""
        import numpy as np

        frame_enc = self.frame_tokenizer.encode(frames)
        signal_enc = self.signal_tokenizer.encode(signals)

        # The frame tokenizer compresses time; the signal tokenizer
        # emits one token-per-channel per timestep at the model grid.
        # For v0 we pad the shorter axis with sep tokens; alignment is
        # tracked in metadata so the model loader can resync.
        n_frame_steps = frame_enc.token_ids.shape[0]
        n_signal_steps = signal_enc.token_ids.shape[0]
        n_steps = min(n_frame_steps, n_signal_steps)

        sep = np.array([self.sep_token], dtype=np.int32)

        stream: list[np.ndarray] = [np.array([self.bos_token], dtype=np.int32)]
        kinds: list[np.ndarray] = [np.array([BlockKind.CONTROL], dtype=np.uint8)]

        for step in range(n_steps):
            # <sep>
            stream.append(sep)
            kinds.append(np.array([BlockKind.CONTROL], dtype=np.uint8))

            # frame block
            frame_flat = frame_enc.token_ids[step].reshape(-1).astype(np.int32)
            stream.append(frame_flat)
            kinds.append(np.full(frame_flat.shape[0], BlockKind.FRAME, dtype=np.uint8))

            # signal block
            signal_flat = signal_enc.token_ids[step].reshape(-1).astype(np.int32)
            stream.append(signal_flat)
            kinds.append(
                np.full(signal_flat.shape[0], BlockKind.SIGNAL, dtype=np.uint8)
            )

        # <eos>
        stream.append(np.array([self.eos_token], dtype=np.int32))
        kinds.append(np.array([BlockKind.CONTROL], dtype=np.uint8))

        return np.concatenate(stream), np.concatenate(kinds)
