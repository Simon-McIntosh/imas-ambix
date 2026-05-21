"""WhamModel — thin wrapper over transformers.LlamaForCausalLM.

Adds block-weighted loss masking via the ``loss_mask`` argument in
:meth:`WhamModel.forward`. All torch and transformers imports are deferred
to method bodies so that the class can be imported without a GPU environment.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from imas_ambix.model.config import WhamConfig

if TYPE_CHECKING:
    from pathlib import Path

    import torch


class WhamModel:
    """v0 WHAM decoder-only transformer.

    Thin wrapper over ``transformers.LlamaForCausalLM`` whose construction
    parameters come from :class:`WhamConfig`. Adds block-weighted loss
    masking via the ``loss_mask`` passed in the forward signature.
    """

    def __init__(self, config: WhamConfig, _model: object) -> None:
        self._config = config
        self._model = _model  # transformers.LlamaForCausalLM

    @classmethod
    def from_config(cls, config: WhamConfig) -> WhamModel:
        """Instantiate a randomly initialised WhamModel from *config*."""
        from transformers import LlamaForCausalLM  # type: ignore[import-untyped]

        llama_cfg = config.to_llama_config()
        model = LlamaForCausalLM(llama_cfg)
        return cls(config, model)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor | None = None,
        labels: torch.LongTensor | None = None,
        loss_mask: torch.FloatTensor | None = None,
    ) -> dict:
        """Standard CausalLM forward with optional per-position weighted CE.

        Parameters
        ----------
        input_ids:
            ``(B, L)`` token ids.
        attention_mask:
            ``(B, L)`` boolean / 0-1 mask; optional.
        labels:
            ``(B, L)`` token ids for teacher-forcing. Positions with
            ``labels == -100`` are ignored by cross-entropy (HF convention).
        loss_mask:
            ``(B, L)`` float weights in ``[0, 1]``. When provided, the
            per-position CE loss is multiplied element-wise by these weights
            and renormalised by the sum of non-zero weights. If ``None``, the
            plain mean CE (HF default) is returned.

        Returns
        -------
        dict with keys:
            ``"loss"`` — scalar ``Tensor`` (or ``None`` if ``labels`` is ``None``).
            ``"logits"`` — ``(B, L, V)`` raw logit ``Tensor``.
        """
        import torch.nn.functional as F  # noqa: N812

        if loss_mask is None or labels is None:
            # Delegate entirely to the HF implementation
            out = self._model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            return {"loss": out.loss, "logits": out.logits}

        # --- Weighted CE path ---
        # Get logits without HF's own loss computation
        out = self._model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits: torch.Tensor = out.logits  # (B, L, V)

        # Shift for causal LM: predict tokens [1..L] from inputs [0..L-1]
        shift_logits = logits[:, :-1, :].contiguous()  # (B, L-1, V)
        shift_labels = labels[:, 1:].contiguous()  # (B, L-1)
        shift_mask = loss_mask[:, 1:].contiguous()  # (B, L-1)

        # Per-token CE (reduction="none")
        b, l_minus_1, v = shift_logits.shape
        per_token_loss = F.cross_entropy(
            shift_logits.view(b * l_minus_1, v),
            shift_labels.view(b * l_minus_1),
            ignore_index=-100,
            reduction="none",
        ).view(b, l_minus_1)  # (B, L-1)

        # Apply the weight mask
        weighted = per_token_loss * shift_mask

        # Renormalise by sum of active (non-zero) weights, ignoring -100 labels
        valid = (shift_labels != -100).float() * shift_mask
        denom = valid.sum().clamp(min=1.0)
        loss = weighted.sum() / denom

        return {"loss": loss, "logits": logits}

    def num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self._model.parameters() if p.requires_grad)

    def save_pretrained(self, path: str | Path) -> None:
        """Save model weights and config to *path* via HF safetensors."""
        self._model.save_pretrained(str(path))

    @classmethod
    def from_pretrained(cls, path: str | Path) -> WhamModel:
        """Load a previously saved WhamModel from *path*."""
        from transformers import (  # type: ignore[import-untyped]
            LlamaConfig,
            LlamaForCausalLM,
        )

        llama_cfg = LlamaConfig.from_pretrained(str(path))
        hf_model = LlamaForCausalLM.from_pretrained(str(path))

        # Reconstruct a WhamConfig from the LlamaConfig fields we set
        wham_cfg = WhamConfig(
            hidden_size=llama_cfg.hidden_size,
            num_hidden_layers=llama_cfg.num_hidden_layers,
            num_attention_heads=llama_cfg.num_attention_heads,
            intermediate_size=llama_cfg.intermediate_size,
            max_position_embeddings=llama_cfg.max_position_embeddings,
            rope_theta=llama_cfg.rope_theta,
            rms_norm_eps=llama_cfg.rms_norm_eps,
            tie_word_embeddings=llama_cfg.tie_word_embeddings,
            vocab_size=llama_cfg.vocab_size,
        )
        return cls(wham_cfg, hf_model)
