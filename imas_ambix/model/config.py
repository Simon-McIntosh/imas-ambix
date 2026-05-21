"""WhamConfig — hyperparameter dataclass for the v0 WHAM-style transformer.

Maps one-to-one onto ``transformers.LlamaConfig``. Two named variants are
provided:

- :meth:`WhamConfig.variant_125m` — 125 M-parameter baseline (default values).
- :meth:`WhamConfig.variant_500m` — 500 M-parameter curriculum step.

All heavy imports (``transformers``) are deferred to the method that needs
them so that ``from imas_ambix.model import WhamConfig`` works without
torch/transformers installed.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class WhamConfig:
    """v0 WHAM-style Llama-class decoder-only AR transformer config.

    Maps to transformers.LlamaConfig. Defaults: 125M variant.
    """

    # ------------------------------------------------------------------ #
    # Architecture                                                         #
    # ------------------------------------------------------------------ #
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072  # ~4x hidden
    max_position_embeddings: int = 16384  # 16K context
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5
    tie_word_embeddings: bool = True

    # ------------------------------------------------------------------ #
    # Vocabulary (must match TokenRegistry.total_vocab_size())            #
    # ------------------------------------------------------------------ #
    # control + Open-MAGVIT2 (262144) + Chronos (4096) + headroom
    vocab_size: int = 280_000

    # ------------------------------------------------------------------ #
    # Loss weighting per block (see plans/world-model-v0.md §4.1)        #
    # ------------------------------------------------------------------ #
    w_frame: float = 1.0
    w_signal: float = 0.3
    w_action: float = 0.1
    w_control: float = 0.0

    # ------------------------------------------------------------------ #

    def __post_init__(self) -> None:
        if self.hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {self.hidden_size}")
        if self.num_attention_heads <= 0:
            raise ValueError(
                f"num_attention_heads must be positive, got {self.num_attention_heads}"
            )
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads})"
            )
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")
        if self.num_hidden_layers <= 0:
            raise ValueError(
                f"num_hidden_layers must be positive, got {self.num_hidden_layers}"
            )
        if self.intermediate_size <= 0:
            raise ValueError(
                f"intermediate_size must be positive, got {self.intermediate_size}"
            )
        if self.max_position_embeddings <= 0:
            raise ValueError(
                "max_position_embeddings must be positive, "
                f"got {self.max_position_embeddings}"
            )

    @classmethod
    def variant_125m(cls) -> WhamConfig:
        """Return the default 125 M parameter configuration."""
        return cls()  # defaults

    @classmethod
    def variant_500m(cls) -> WhamConfig:
        """Return the 500 M parameter configuration."""
        return cls(
            hidden_size=1024,
            num_hidden_layers=24,
            num_attention_heads=16,
            intermediate_size=4096,
        )

    def estimate_params(self) -> int:
        """Quick parameter count estimate (non-embedding params + embeddings).

        Uses the standard Llama-style formula:
        - Token embeddings: vocab_size * hidden_size (shared input/output if tied)
        - Per layer: attention (4 weight matrices) + FFN (3 weight matrices for
          SwiGLU: gate, up, down) + 2 RMSNorm
        - Final RMSNorm

        Returns the total rounded to the nearest integer (not millions).
        """
        h = self.hidden_size
        n = self.num_hidden_layers
        v = self.vocab_size
        ffn = self.intermediate_size

        # Embedding table (input; output is tied so counted once)
        embed_params = v * h

        # Per-layer parameter count
        # Attention: Q, K, V, O projections — each h×h
        attn_params = 4 * h * h
        # FFN SwiGLU: gate_proj (h→ffn), up_proj (h→ffn), down_proj (ffn→h)
        ffn_params = 3 * h * ffn
        # Two RMSNorm per layer (pre-attn, pre-ffn), each of size h
        norm_params = 2 * h

        layer_params = attn_params + ffn_params + norm_params

        # Final RMSNorm before LM head
        final_norm = h

        total = embed_params + n * layer_params + final_norm
        return int(total)

    def to_llama_config(self) -> object:
        """Return a ``transformers.LlamaConfig`` with our settings.

        Heavy import deferred so the dataclass is importable without
        ``transformers`` installed.
        """
        from transformers import LlamaConfig  # type: ignore[import-untyped]

        return LlamaConfig(
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            max_position_embeddings=self.max_position_embeddings,
            rope_theta=self.rope_theta,
            rms_norm_eps=self.rms_norm_eps,
            tie_word_embeddings=self.tie_word_embeddings,
            vocab_size=self.vocab_size,
            # Llama defaults we rely on
            hidden_act="silu",
        )
