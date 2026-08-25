"""Top-level per-shot tokenizer that ties frames + signals together.

Given a shot's level-1 camera and level-2 diagnostic Zarrs, the
:class:`ShotTokenizer` produces a single interleaved 1-D ``int32``
token array suitable for the world-model training loop. The layout is:

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
    enforce_alignment: bool = True
    """When ``True`` (default), resample ``signals`` onto the model time
    grid and sub-sample ``frames`` to match before encoding.  Set to
    ``False`` for backward-compatible behaviour (shorter axis wins)."""

    def encode_shot(
        self,
        frames: np.ndarray | None,
        signals: xr.Dataset,
        *,
        return_block_kind: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Encode one shot end-to-end into a 1-D int32 token stream.

        Args:
            frames: ``(T, H, W)`` or ``(T, H, W, C)`` array, or ``None``
                if this shot has no camera data.
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
        frames: np.ndarray | None,
        signals: xr.Dataset,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Encode one shot and always return ``(tokens, block_kind)``.

        Preferred over ``encode_shot(return_block_kind=True)`` for new
        code — the return type is unambiguous.

        Args:
            frames: ``(T, H, W)`` or ``(T, H, W, C)`` array, or ``None``.
            signals: ``xr.Dataset`` aligned to the model time grid.

        Returns:
            ``(tokens, block_kind)`` — both 1-D, same length.
            ``tokens`` is ``int32``; ``block_kind`` is ``uint8`` with
            :class:`~imas_ambix.tokenizer.base.BlockKind` codes.
        """
        return self._encode_shot_impl(frames, signals)

    def encode_shot_signal_only(
        self,
        signals: xr.Dataset,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Encode a shot that has no camera frames.

        Emits ``<pad>`` blocks for frame positions so signal-only shots
        can enter the training set with zero frame-loss weight.

        Args:
            signals: ``xr.Dataset`` containing signal channels.

        Returns:
            ``(tokens, block_kind)`` — frame positions carry
            ``CONTROL_TOKENS["pad"]`` with ``BlockKind.CONTROL``.
        """
        return self._encode_shot_impl(None, signals)

    def encode_shot_frames_only(
        self,
        frames: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Encode a shot that has no signal data.

        Emits ``<pad>`` blocks for signal positions so frame-only shots
        can enter the training set with zero signal-loss weight.

        Args:
            frames: ``(T, H, W)`` or ``(T, H, W, C)`` array.

        Returns:
            ``(tokens, block_kind)`` — signal positions carry
            ``CONTROL_TOKENS["pad"]`` with ``BlockKind.CONTROL``.
        """
        import xarray as xr

        empty_signals: xr.Dataset = xr.Dataset()
        return self._encode_shot_impl(frames, empty_signals)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _frame_block_len_per_step(self) -> int:
        """Estimate the per-step frame token count for pad-block sizing.

        Uses ``spatial_compression`` and ``image_size`` when available on
        the frame tokenizer; falls back to 0 (signal-only stream) if
        neither is set.
        """
        ft = self.frame_tokenizer
        sc = getattr(ft, "spatial_compression", 0)
        image_size = getattr(ft, "image_size", None)
        if sc and image_size:
            if isinstance(image_size, (list, tuple)) and len(image_size) >= 2:
                h_tok = image_size[0] // sc
                w_tok = image_size[1] // sc
                return h_tok * w_tok
            elif isinstance(image_size, int):
                h_tok = image_size // sc
                return h_tok * h_tok
        # Default: 0 means signal-only (no pad block emitted)
        return 0

    # ------------------------------------------------------------------
    # Internal implementation
    # ------------------------------------------------------------------

    def _encode_shot_impl(
        self,
        frames: np.ndarray | None,
        signals: xr.Dataset,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build the token stream and parallel block_kind array."""
        import numpy as np

        has_frames = frames is not None
        has_signals = bool(signals.data_vars)

        if self.enforce_alignment and (has_frames or has_signals):
            from imas_ambix.tokenizer.alignment import align_frames_signals

            frames, signals = align_frames_signals(
                frames, signals, self.model_hz
            )
            # Recompute after potential resampling
            has_signals = bool(signals.data_vars)

        # --- Encode each modality -------------------------------------
        frame_enc = self.frame_tokenizer.encode(frames) if has_frames else None
        signal_enc = self.signal_tokenizer.encode(signals) if has_signals else None

        # Determine n_steps
        if frame_enc is not None and signal_enc is not None:
            if self.enforce_alignment:
                # After alignment both axes should agree; trust frame enc
                n_steps = min(
                    frame_enc.token_ids.shape[0],
                    signal_enc.token_ids.shape[0],
                )
            else:
                n_steps = min(
                    frame_enc.token_ids.shape[0],
                    signal_enc.token_ids.shape[0],
                )
        elif frame_enc is not None:
            n_steps = frame_enc.token_ids.shape[0]
        elif signal_enc is not None:
            n_steps = signal_enc.token_ids.shape[0]
        else:
            n_steps = 0

        pad_token = np.array([CONTROL_TOKENS["pad"]], dtype=np.int32)
        sep = np.array([self.sep_token], dtype=np.int32)

        # Per-step frame pad block length (for missing-frames case)
        frame_pad_len = self._frame_block_len_per_step() if not has_frames else 0

        stream: list[np.ndarray] = [np.array([self.bos_token], dtype=np.int32)]
        kinds: list[np.ndarray] = [np.array([BlockKind.CONTROL], dtype=np.uint8)]

        for step in range(n_steps):
            # <sep>
            stream.append(sep)
            kinds.append(np.array([BlockKind.CONTROL], dtype=np.uint8))

            # frame block (real or pad)
            if frame_enc is not None:
                frame_flat = frame_enc.token_ids[step].reshape(-1).astype(np.int32)
                stream.append(frame_flat)
                kinds.append(
                    np.full(frame_flat.shape[0], BlockKind.FRAME, dtype=np.uint8)
                )
            elif frame_pad_len > 0:
                pad_block = np.full(
                    frame_pad_len, CONTROL_TOKENS["pad"], dtype=np.int32
                )
                stream.append(pad_block)
                kinds.append(
                    np.full(frame_pad_len, BlockKind.CONTROL, dtype=np.uint8)
                )
            # else: signal-only stream, no frame block emitted

            # signal block (real or pad)
            if signal_enc is not None:
                signal_flat = signal_enc.token_ids[step].reshape(-1).astype(np.int32)
                stream.append(signal_flat)
                kinds.append(
                    np.full(signal_flat.shape[0], BlockKind.SIGNAL, dtype=np.uint8)
                )
            else:
                # Emit a single pad token for the signal position
                stream.append(pad_token)
                kinds.append(np.array([BlockKind.CONTROL], dtype=np.uint8))

        # <eos>
        stream.append(np.array([self.eos_token], dtype=np.int32))
        kinds.append(np.array([BlockKind.CONTROL], dtype=np.uint8))

        return np.concatenate(stream), np.concatenate(kinds)
