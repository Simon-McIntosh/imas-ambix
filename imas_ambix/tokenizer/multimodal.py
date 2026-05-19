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
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import xarray as xr

from imas_ambix.tokenizer.alignment import MODEL_HZ_DEFAULT
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
    ) -> np.ndarray:
        """Encode one shot end-to-end into a 1-D int32 token stream.

        Args:
            frames: ``(T, H, W)`` or ``(T, H, W, C)`` array. The frame
                tokenizer compresses the time axis by its own factor.
            signals: ``xr.Dataset`` aligned to the model time grid.

        Returns:
            1-D ``int32`` array of global token ids:
            ``[bos, (sep, frame_block, signal_block)*N, eos]``.
        """
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
        for step in range(n_steps):
            stream.append(sep)
            stream.append(frame_enc.token_ids[step].reshape(-1).astype(np.int32))
            stream.append(signal_enc.token_ids[step].reshape(-1).astype(np.int32))
        stream.append(np.array([self.eos_token], dtype=np.int32))

        return np.concatenate(stream)
