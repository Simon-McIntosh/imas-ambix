"""Factorized space/time transformer + LFQ bit-head — the camdyn model family.

This is the SHARED architecture for the camera-dynamics world model.  D1
(this deliverable) is the per-frame **spatial-inpainting baseline**: the
temporal-attention path is DISABLED so every frame is reconstructed from
its own visible (clipped) tokens + the per-frame conditioning alone, with
no information flowing between frames.  D2 will be **the same network**
with the temporal path ENABLED — the only difference is a single boolean
toggle (:attr:`CamdynConfig.temporal_attention`).  Nothing else about the
capacity, parameter count, or training tokens changes, so the W1
comparison isolates *exactly* the value of dynamics (matched-arm).

Architecture
------------
Each token grid cell is embedded; per-frame spatial position (row, col)
and per-window temporal position (frame index) get learned embeddings.
A masked grid cell (clipped away — ``visible == False``) is replaced by a
shared learned ``[MASK]`` embedding so the model never sees the hidden
token id.  The conditioning vector + Δt + missing-flags for each frame
are projected to the model width and ADDED to every cell of that frame
(FiLM-free additive conditioning — simple, matched across arms).

The trunk is a stack of factorized blocks.  Each block does:

  1. **Spatial** self-attention — full (non-causal) attention over the
     256 cells WITHIN one frame.  Active in both D1 and D2.
  2. **Temporal** self-attention — causal attention ACROSS frames at a
     fixed grid position.  This is the dynamics path.  When
     ``temporal_attention`` is False (D1) the temporal sub-layer is still
     PRESENT (so the parameter count is identical to D2) but is forced to
     attend only to the current frame via a diagonal attention mask — it
     becomes an identity-in-information op (a per-frame token-mixing MLP
     over a length-1 sequence), contributing zero cross-frame signal.

Forcing D1's temporal attention to a diagonal mask (rather than deleting
the layer) is the matched-arm guarantee: D1 and D2 have byte-identical
parameter tensors; flipping the toggle only widens the temporal
receptive field from 1 frame to the full causal past.

Vocab head — factorized LFQ bit-head
-------------------------------------
The OMAG2 LFQ tokenizer encodes each grid cell as one of ``2^18`` ids,
where the id is the integer formed by 18 independent sign bits of the
quantised latent.  Rather than a 262 144-way softmax we predict the **18
binary bits** directly: the head emits ``(..., 18)`` logits, one per bit,
each the logit of "bit b is 1".  Bit ``b`` of token id ``v`` is
``(v >> b) & 1``.

The D0 metrics (:mod:`imas_ambix.camdyn.metrics`) are vocab-agnostic:
they take ``logits[..., V]`` and index the true token id.  Materialising
a dense ``(..., 2^18)`` logit tensor is infeasible, so
:func:`bit_logits_to_token_logits` provides the documented adapter for a
*restricted* vocabulary (the union of candidate ids present in the
scored window), under the bit-independence assumption:

    log p(v) = Σ_b log σ( s_b · z_b ),   s_b = +1 if bit b set else −1

i.e. the per-id score is the sum over bits of the log-sigmoid of the
signed bit logit.  This is exactly the bitwise factorised log-likelihood;
the argmax over the restricted id set and the NLL of the true id under
this score are what the metrics consume.  The training loss is the
bitwise binary cross-entropy over masked cells (equivalent — it is the
same factorised likelihood, summed over the 18 bits).

All torch imports are deferred to method bodies so the module imports
(and the config + the pure-numpy bit adapter are usable) without a GPU /
torch environment.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from imas_ambix.camdyn.conditioning import CONDITIONING_CHANNELS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GRID_H, GRID_W = 16, 16
N_CELLS = GRID_H * GRID_W  # 256 spatial positions per frame
LFQ_BITS = 18  # 2^18 vocabulary → 18 independent sign bits
VOCAB_SIZE = 1 << LFQ_BITS

# Number of physical conditioning channels (full actuator vector).
N_COND_CHANNELS = len(CONDITIONING_CHANNELS)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class CamdynConfig:
    """Configuration for the shared factorized ST-transformer.

    The ``temporal_attention`` flag is the ONLY difference between the D1
    baseline (False) and the D2 dynamics arm (True); every other field is
    shared so the two arms are matched on parameters and capacity.

    Attributes
    ----------
    temporal_attention:
        D1 baseline → ``False`` (temporal sub-layers forced to a diagonal
        / current-frame-only mask: no cross-frame information).
        D2 dynamics → ``True`` (causal temporal attention across frames).
    dim:
        Model width.
    n_layers:
        Number of factorized (spatial + temporal) blocks.
    n_heads:
        Attention heads (shared between the spatial and temporal sub-layers).
    mlp_ratio:
        Feed-forward expansion factor.
    dropout:
        Dropout probability.
    n_frames:
        Maximum temporal context (window length) — sizes the temporal
        position embedding.
    cond_channels:
        Number of physical conditioning channels fed per frame.
    grid:
        Spatial token-grid shape.
    bits:
        LFQ bit count (head width).
    """

    temporal_attention: bool = False
    dim: int = 256
    n_layers: int = 6
    n_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    n_frames: int = 16
    cond_channels: int = N_COND_CHANNELS
    grid: tuple[int, int] = (GRID_H, GRID_W)
    bits: int = LFQ_BITS

    @property
    def n_cells(self) -> int:
        h, w = self.grid
        return h * w

    def to_dict(self) -> dict:
        return {
            "temporal_attention": self.temporal_attention,
            "dim": self.dim,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "mlp_ratio": self.mlp_ratio,
            "dropout": self.dropout,
            "n_frames": self.n_frames,
            "cond_channels": self.cond_channels,
            "grid": list(self.grid),
            "bits": self.bits,
        }

    @classmethod
    def from_dict(cls, d: dict) -> CamdynConfig:
        d = dict(d)
        if "grid" in d:
            d["grid"] = tuple(d["grid"])
        known = {f for f in cls.__dataclass_fields__}  # noqa: PLC0206
        return cls(**{k: v for k, v in d.items() if k in known})


# ---------------------------------------------------------------------------
# Pure-numpy bit utilities (usable without torch — feed the D0 metrics)
# ---------------------------------------------------------------------------


def token_ids_to_bits(ids: np.ndarray, bits: int = LFQ_BITS) -> np.ndarray:
    """Expand integer token ids to their ``bits`` binary digits (LSB first).

    Returns a float32 array of shape ``(*ids.shape, bits)`` with values in
    ``{0.0, 1.0}``; bit ``b`` is ``(id >> b) & 1``.
    """
    ids = np.asarray(ids, dtype=np.int64)
    shifts = np.arange(bits, dtype=np.int64)
    out = (ids[..., None] >> shifts) & 1
    return out.astype(np.float32)


def bit_logits_to_token_logits(
    bit_logits: np.ndarray,
    candidate_ids: np.ndarray,
) -> np.ndarray:
    """Adapter: per-bit logits → per-token-id scores over a RESTRICTED vocab.

    The metrics interface wants ``logits[..., v]``; a dense ``2^18``-wide
    tensor is infeasible, so we score only ``candidate_ids`` (the union of
    ids appearing in the scored window — typically the masked target ids
    plus, for accuracy, every id the head could plausibly pick).  Under
    bit-independence the log-likelihood of id ``v`` is

        log p(v) = Σ_b log σ( s_b(v) · z_b ),   s_b(v) = +1 if bit b of v
                                                  set, else −1

    where ``z_b`` is the per-bit logit.  We return these summed
    log-sigmoid scores (an unnormalised but monotone-in-probability score
    vector); the metrics apply their own log-sum-exp over the candidate
    axis, so the NLL and argmax are exact for the restricted id set.

    Parameters
    ----------
    bit_logits:
        ``(..., bits)`` per-bit logits (logit of "bit is 1").
    candidate_ids:
        ``(K,)`` int token ids to score.

    Returns
    -------
    ``(..., K)`` float — per-candidate summed log-sigmoid scores.  Column
    ``k`` corresponds to ``candidate_ids[k]``; to use with the metrics,
    remap each true token id to its column index.
    """
    bit_logits = np.asarray(bit_logits, dtype=np.float64)
    bits = bit_logits.shape[-1]
    cand = np.asarray(candidate_ids, dtype=np.int64).reshape(-1)
    # signed-bit table for the candidates: (K, bits) in {-1, +1}
    cand_bits = token_ids_to_bits(cand, bits=bits)  # (K, bits)
    signs = 2.0 * cand_bits - 1.0  # (K, bits) ∈ {-1, +1}
    # log σ(x) = -softplus(-x); score_k = Σ_b log σ(s_{k,b} · z_b)
    # bit_logits: (..., bits); signs: (K, bits) → broadcast over leading dims.
    z = bit_logits[..., None, :]  # (..., 1, bits)
    s = signs[(None,) * (bit_logits.ndim - 1)]  # (..., K, bits) via broadcast
    signed = s * z  # (..., K, bits)
    log_sig = -np.logaddexp(0.0, -signed)  # log σ
    return log_sig.sum(axis=-1)  # (..., K)


def restricted_vocab_logits(
    bit_logits: np.ndarray,
    target_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a (restricted) dense logit tensor + remapped targets for metrics.

    Convenience wrapper used by the eval path: the candidate vocabulary is
    the set of UNIQUE true target ids in the scored window.  Top-1
    accuracy under this restriction is the model's ability to pick the
    right token *among the ids that actually occur* — a strictly fair,
    well-defined accuracy for both arms (a 262 144-way argmax over a
    sparse bit-head is not meaningfully different and is intractable to
    materialise).  NLL is the exact bitwise-factorised NLL renormalised
    over the candidate set.

    Returns ``(dense_logits, remapped_targets)`` where ``dense_logits`` is
    ``(*target_ids.shape, K)`` and ``remapped_targets`` indexes the last
    axis (column of the true id in ``candidate_ids``).
    """
    target_ids = np.asarray(target_ids, dtype=np.int64)
    candidates, inverse = np.unique(target_ids, return_inverse=True)
    dense = bit_logits_to_token_logits(bit_logits, candidates)
    remapped = inverse.reshape(target_ids.shape)
    return dense, remapped


def bitwise_nll(bit_logits: np.ndarray, target_ids: np.ndarray) -> np.ndarray:
    """Exact per-cell bitwise-factorised NLL (nats), summed over 18 bits.

    This is the loss the model is trained on and the cleanest full-vocab
    NLL: ``-Σ_b log p(bit_b = target_bit_b)``.  Equivalent to the NLL of
    the true id under the bit-independent token distribution over the FULL
    ``2^18`` vocabulary (no restriction).  Returns an array shaped like
    ``target_ids``.
    """
    bit_logits = np.asarray(bit_logits, dtype=np.float64)
    bits = bit_logits.shape[-1]
    tgt_bits = token_ids_to_bits(target_ids, bits=bits)  # (..., bits) ∈ {0,1}
    # BCE-with-logits per bit: softplus(z) - z*target  (= -log p(target bit))
    sp = np.logaddexp(0.0, bit_logits)  # softplus(z)
    per_bit = sp - bit_logits * tgt_bits
    return per_bit.sum(axis=-1)


# ---------------------------------------------------------------------------
# torch model (deferred imports)
# ---------------------------------------------------------------------------


def _make_block(cfg: CamdynConfig):
    import torch.nn as nn  # noqa: PLC0415

    class _SelfAttention(nn.Module):
        """Multi-head self-attention over the last sequence axis."""

        def __init__(self, dim: int, n_heads: int, dropout: float) -> None:
            super().__init__()
            self.attn = nn.MultiheadAttention(
                dim, n_heads, dropout=dropout, batch_first=True
            )

        def forward(self, x, attn_mask=None):
            # x: (N, S, D); attn_mask: (S, S) bool — True = DISALLOWED.
            out, _ = self.attn(x, x, x, attn_mask=attn_mask, need_weights=False)
            return out

    class _FactorizedBlock(nn.Module):
        """One factorized spatial→temporal→MLP block (pre-norm residual)."""

        def __init__(self, cfg: CamdynConfig) -> None:
            super().__init__()
            import torch.nn as nn  # noqa: PLC0415

            d = cfg.dim
            self.ln_s = nn.LayerNorm(d)
            self.spatial = _SelfAttention(d, cfg.n_heads, cfg.dropout)
            self.ln_t = nn.LayerNorm(d)
            self.temporal = _SelfAttention(d, cfg.n_heads, cfg.dropout)
            self.ln_m = nn.LayerNorm(d)
            hidden = int(d * cfg.mlp_ratio)
            self.mlp = nn.Sequential(
                nn.Linear(d, hidden),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(hidden, d),
            )

        def forward(self, x, n_frames, n_cells, temporal_mask):
            # x: (B, F, C, D)

            b, f, c, d = x.shape
            # --- spatial: attend over C cells within each frame ---
            xs = x.reshape(b * f, c, d)
            xs = xs + self.spatial(self.ln_s(xs))
            x = xs.reshape(b, f, c, d)
            # --- temporal: attend over F frames at each cell ---
            # reshape to (B*C, F, D); causal/diagonal mask supplied by caller.
            xt = x.permute(0, 2, 1, 3).reshape(b * c, f, d)
            xt = xt + self.temporal(self.ln_t(xt), attn_mask=temporal_mask)
            x = xt.reshape(b, c, f, d).permute(0, 2, 1, 3)
            # --- MLP ---
            x = x + self.mlp(self.ln_m(x))
            return x

    return _FactorizedBlock(cfg)


class CamdynModel:
    """Shared factorized ST-transformer with an LFQ 18-bit head.

    Construct via :meth:`from_config`.  The underlying ``torch.nn.Module``
    is :attr:`module`.  All torch imports are deferred so this file is
    importable (and the config + numpy adapters usable) without torch.
    """

    def __init__(self, config: CamdynConfig, module) -> None:
        self._config = config
        self.module = module

    @property
    def config(self) -> CamdynConfig:
        return self._config

    @classmethod
    def from_config(cls, config: CamdynConfig) -> CamdynModel:
        import torch  # noqa: PLC0415
        import torch.nn as nn  # noqa: PLC0415

        cfg = config
        d = cfg.dim
        h, w = cfg.grid
        n_cells = h * w

        class _Module(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                # token embedding factorised through the 18 bits: each cell's
                # input is the sum of per-bit embeddings (a compact, vocab-free
                # input encoding matched to the bit-head output).  A masked
                # cell uses a single shared [MASK] embedding instead.
                self.bit_embed = nn.Parameter(torch.zeros(cfg.bits, d))  # (bits, D)
                self.mask_embed = nn.Parameter(torch.zeros(d))
                self.spatial_pos = nn.Parameter(torch.zeros(n_cells, d))
                self.temporal_pos = nn.Parameter(torch.zeros(cfg.n_frames, d))
                # conditioning: [values | missing-flags | dt] → D, added per frame
                self.cond_proj = nn.Linear(2 * cfg.cond_channels + 1, d)
                self.in_norm = nn.LayerNorm(d)
                self.blocks = nn.ModuleList(
                    [_make_block(cfg) for _ in range(cfg.n_layers)]
                )
                self.out_norm = nn.LayerNorm(d)
                self.head = nn.Linear(d, cfg.bits)
                self._init_weights()

            def _init_weights(self) -> None:
                nn.init.normal_(self.bit_embed, std=0.02)
                nn.init.normal_(self.mask_embed, std=0.02)
                nn.init.normal_(self.spatial_pos, std=0.02)
                nn.init.normal_(self.temporal_pos, std=0.02)
                for m in self.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.normal_(m.weight, std=0.02)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)

            def _temporal_mask(self, f: int, device, dtype):
                """Bool (F, F) additive-style mask: True = DISALLOWED.

                D2 (temporal ON): strict causal — frame i may attend to
                frames ≤ i.  D1 (temporal OFF): diagonal only — frame i may
                attend ONLY to itself (no cross-frame information), so the
                temporal sub-layer carries identical parameters but zero
                dynamics signal (matched-arm guarantee).
                """
                idx = torch.arange(f, device=device)
                if cfg.temporal_attention:
                    # disallow attending to the future: j > i
                    disallow = idx[None, :] > idx[:, None]
                else:
                    # allow only the diagonal: disallow j != i
                    disallow = idx[None, :] != idx[:, None]
                return disallow

            def encode_inputs(self, tokens, visible, cond_values, cond_missing, dt):
                """Build the (B, F, C, D) input embedding tensor."""
                b, f, h2, w2 = tokens.shape
                c = h2 * w2
                tok_flat = tokens.reshape(b, f, c).long()
                vis_flat = visible.reshape(b, f, c).bool()
                # per-bit input embedding: sum of bit embeddings present in the id
                bits = torch.arange(cfg.bits, device=tokens.device)
                tok_bits = ((tok_flat[..., None] >> bits) & 1).float()  # (B,F,C,bits)
                tok_emb = tok_bits @ self.bit_embed  # (B,F,C,D)
                # replace masked (clipped-away) cells with the shared [MASK] emb
                mask_emb = self.mask_embed.view(1, 1, 1, d).expand(b, f, c, d)
                vis = vis_flat[..., None].float()
                x = vis * tok_emb + (1.0 - vis) * mask_emb
                # positions
                x = x + self.spatial_pos.view(1, 1, c, d)
                x = x + self.temporal_pos[:f].view(1, f, 1, d)
                # conditioning (per frame, added to every cell)
                cond_in = torch.cat(
                    [cond_values, cond_missing, dt[..., None]], dim=-1
                )  # (B, F, 2C+1)
                cond_emb = self.cond_proj(cond_in)  # (B, F, D)
                x = x + cond_emb[:, :, None, :]
                return self.in_norm(x)

            def forward(self, tokens, visible, cond_values, cond_missing, dt):
                """Return per-bit logits ``(B, F, H, W, bits)``."""
                b, f, h2, w2 = tokens.shape
                c = h2 * w2
                x = self.encode_inputs(tokens, visible, cond_values, cond_missing, dt)
                tmask = self._temporal_mask(f, x.device, x.dtype)
                for blk in self.blocks:
                    x = blk(x, f, c, tmask)
                x = self.out_norm(x)
                logits = self.head(x)  # (B, F, C, bits)
                return logits.reshape(b, f, h2, w2, cfg.bits)

        module = _Module()
        return cls(cfg, module)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.module.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Loss — bitwise BCE over masked (clipped-away) cells
# ---------------------------------------------------------------------------


def masked_bit_bce(
    bit_logits,
    target_tokens,
    loss_mask,
    valid_frames=None,
    bits: int = LFQ_BITS,
):
    """Mean bitwise BCE over MASKED cells (the reconstruction loss).

    Parameters
    ----------
    bit_logits:
        ``(B, F, H, W, bits)`` per-bit logits (torch.Tensor).
    target_tokens:
        ``(B, F, H, W)`` int token ids.
    loss_mask:
        ``(B, F, H, W)`` bool/float — True where the cell was MASKED
        (clipped away), i.e. the scored reconstruction set.  Visible cells
        do not contribute to the loss.
    valid_frames:
        Optional ``(B, F)`` bool — False for zero-padded frames (short
        shots); padded frames are excluded from the loss.
    """
    import torch  # noqa: PLC0415
    import torch.nn.functional as F  # noqa: N812, PLC0415

    b, f, h, w, nb = bit_logits.shape
    device = bit_logits.device
    bit_idx = torch.arange(bits, device=device)
    tgt_bits = ((target_tokens[..., None].long() >> bit_idx) & 1).float()
    per_bit = F.binary_cross_entropy_with_logits(
        bit_logits, tgt_bits, reduction="none"
    )  # (B,F,H,W,bits)
    per_cell = per_bit.sum(dim=-1)  # (B,F,H,W) — summed over 18 bits

    m = loss_mask.to(device).float()
    if valid_frames is not None:
        vf = valid_frames.to(device).float().view(b, f, 1, 1)
        m = m * vf
    denom = m.sum().clamp(min=1.0)
    return (per_cell * m).sum() / denom


# ---------------------------------------------------------------------------
# Eval adapter — score a window with the D0 vocab-agnostic metrics
# ---------------------------------------------------------------------------


@dataclass
class WindowScore:
    """Per-window masked-token scores (feeds the W1 bootstrap CI)."""

    nll_per_token: np.ndarray = field(default_factory=lambda: np.array([]))
    acc_per_token: np.ndarray = field(default_factory=lambda: np.array([]))

    @property
    def n(self) -> int:
        return int(self.nll_per_token.size)


def score_window_bits(
    bit_logits: np.ndarray,
    target_tokens: np.ndarray,
    loss_mask: np.ndarray,
) -> WindowScore:
    """Score one window's masked cells via the bit-head → metric adapter.

    NLL is the EXACT full-vocab bitwise-factorised NLL (no restriction):
    :func:`bitwise_nll` over masked cells.  Top-1 accuracy is computed
    under the restricted-candidate adapter (argmax over the ids that
    occur in the masked target set), the well-defined fair accuracy
    documented in :func:`restricted_vocab_logits`.  Returns per-token
    arrays for the masked positions — the paired inputs to
    :func:`imas_ambix.camdyn.metrics.bootstrap_ci`.
    """
    from imas_ambix.camdyn.metrics import (  # noqa: PLC0415
        masked_top1_accuracy,
    )

    mask = np.asarray(loss_mask, dtype=bool)
    if not mask.any():
        return WindowScore()

    # --- exact full-vocab NLL over masked cells ---
    nll_grid = bitwise_nll(bit_logits, target_tokens)  # (F,H,W)
    nll_sel = nll_grid[mask]

    # --- restricted-vocab top-1 accuracy over masked cells ---
    tgt_masked = np.asarray(target_tokens)[mask]
    bl_masked = np.asarray(bit_logits)[mask]  # (M, bits)
    dense, remapped = restricted_vocab_logits(bl_masked, tgt_masked)
    full = np.ones(remapped.shape, dtype=bool)
    acc_sel = masked_top1_accuracy(dense, remapped, full, reduce="none")

    return WindowScore(nll_per_token=nll_sel, acc_per_token=acc_sel)
