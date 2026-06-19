"""Spatiotemporal autoregressive transformer over camera frame tokens.

Why this exists (and why the bag-of-channels model failed)
----------------------------------------------------------
The earlier world model fused a frame's 256 spatial tokens into ONE vector per
timestep (``emb.sum(dim=2)``, a bag-of-channels) and asked a single head to
regenerate 256 spatial positions from that summed vector.  Summing destroys
spatial structure: the transformer only ever attended over TIME, never over
SPACE, so even an overfit reconstruction was incoherent mush.  This model keeps
EVERY token's identity end to end — there is no spatial sum anywhere.

Architecture
------------
Decoder-only, operating on a sequence of ``T`` frames × ``256`` raster tokens.

* **Embeddings (per token, never summed across space):**
  - a value embedding of the token's LFQ id (vocab ``2**18``);
  - a SPATIAL position embedding factorised as row(0..15) + col(0..15) learned
    tables (so a token at grid cell (r, c) is identifiable);
  - a TEMPORAL frame-position embedding (which frame in the window).

* **Factorised space-time attention** (the affordability choice).  A naive
  full-sequence causal attention over ``256·T`` tokens costs ``(256·T)²``.
  Instead each block does two cheap attentions in sequence (axial / VideoPoet
  style):
    1. **Spatial** — FULL (bidirectional) self-attention among the 256 tokens
       WITHIN each frame independently.  Cost ``T · 256²``.
    2. **Temporal** — CAUSAL self-attention across frames at each FIXED spatial
       position (token (r, c) of frame t attends to token (r, c) of frames
       ≤ t).  Cost ``256 · T²``.
  Total ``T·256² + 256·T²`` ≪ ``(256·T)²``.  Causality is enforced ONLY on the
  temporal axis (a frame may use its whole self spatially — that is teacher
  forcing within the frame's known tokens at training; at generation we decode
  a frame's tokens in one shot from the previous frames, see the rollout), so a
  prediction of frame ``t`` never sees frame ``> t``.

* **Plan conditioning.**  The pulse-schedule plan is embedded and prepended as
  a small set of conditioning frames (each plan step a "frame" of plan-channel
  tokens, spatially attended within itself, and visible CAUSALLY to every real
  frame via the temporal axis).  The plan frames sit at temporal positions
  before frame 0, so every predicted camera token can attend back to the whole
  plan — "this pulse schedule → this video".

* **Output head** over ``2**18`` vocab, applied per token.  The full
  ``B · T · 256 · 2**18`` logit tensor is never materialised: training uses a
  CHUNKED cross-entropy over the flattened (frame, position) token axis, and
  generation a chunked argmax — peak head memory is ``chunk · 2**18`` only.

Prediction target
------------------
NEXT-FRAME prediction: the model predicts frame ``t``'s 256 tokens from frames
``< t`` (+ the plan).  At a token's own position the spatial attention is
bidirectional, but a frame's hidden states are read at the temporal position of
the PREVIOUS frame, so frame ``t`` is generated WITHOUT having seen any of its
own tokens (full-frame parallel decode).  This is the standard next-frame
factorisation used by token video models and keeps generation one forward pass
per frame rather than 256.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class SpacetimeConfig:
    """Hyper-parameters for the spatiotemporal camera transformer.

    Attributes
    ----------
    vocab_size:
        Camera LFQ codebook size (the per-token output classes), ``2**18``.
    grid_h, grid_w:
        Frame token grid (16 × 16); ``n_spatial = grid_h * grid_w`` tokens/frame.
    max_frames:
        Largest temporal position budget (real frames + plan frames).
    plan_vocab, plan_channels:
        Plan (pulse-schedule) local vocab + per-plan-step channel count.  When
        ``plan_channels == 0`` the model runs unconditioned (no plan prefix).
    d_model, n_layers, n_heads, d_ff:
        Backbone width / depth / heads / feed-forward width.  The block applies
        a spatial AND a temporal attention, so a "layer" here is heavier than a
        vanilla transformer layer (two attentions + one MLP).
    dropout:
        Dropout probability.
    """

    vocab_size: int = 1 << 18
    grid_h: int = 16
    grid_w: int = 16
    max_frames: int = 64
    plan_vocab: int = 257
    plan_channels: int = 2
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    d_ff: int = 2048
    dropout: float = 0.0

    @property
    def n_spatial(self) -> int:
        return int(self.grid_h * self.grid_w)


# ---------------------------------------------------------------------------
# Attention blocks (torch SDPA / flash)
# ---------------------------------------------------------------------------


class _MHA(nn.Module):
    """Multi-head self-attention via torch SDPA (flash on CUDA bf16).

    A single ``is_causal`` flag selects a causal vs full attention; the caller
    folds the batch / parallel axis so SDPA sees ``(B*, n_heads, L, d_head)``.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model {d_model} not divisible by n_heads {n_heads}")
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, *, is_causal: bool) -> torch.Tensor:
        bstar, length, _ = x.shape
        qkv = self.qkv(x).view(bstar, length, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)  # each (B*, L, H, d_head)
        q = q.transpose(1, 2)  # (B*, H, L, d_head)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )  # (B*, H, L, d_head)
        out = out.transpose(1, 2).reshape(bstar, length, self.n_heads * self.d_head)
        return self.proj(out)


class _SpaceTimeBlock(nn.Module):
    """One factorised block: spatial (full) then temporal (causal) then MLP.

    Operates on ``x`` of shape ``(B, T, S, d)`` — B batch, T frames, S spatial
    tokens per frame.  Pre-norm residual throughout.

    * Spatial attention: reshape to ``(B*T, S, d)`` and attend FULLY among the S
      tokens of each frame (no mask — a frame may use all its own positions).
    * Temporal attention: reshape to ``(B*S, T, d)`` and attend CAUSALLY across
      frames at each fixed spatial position (frame t sees frames ≤ t).
    """

    def __init__(self, cfg: SpacetimeConfig) -> None:
        super().__init__()
        d = cfg.d_model
        self.ln_s = nn.LayerNorm(d)
        self.attn_s = _MHA(d, cfg.n_heads, cfg.dropout)
        self.ln_t = nn.LayerNorm(d)
        self.attn_t = _MHA(d, cfg.n_heads, cfg.dropout)
        self.ln_m = nn.LayerNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, cfg.d_ff),
            nn.GELU(),
            nn.Linear(cfg.d_ff, d),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, s, d = x.shape
        # ── spatial: full attention within each frame ──
        xs = self.ln_s(x).reshape(b * t, s, d)
        xs = self.attn_s(xs, is_causal=False).reshape(b, t, s, d)
        x = x + xs
        # ── temporal: causal attention across frames at each spatial position ──
        xt = self.ln_t(x).permute(0, 2, 1, 3).reshape(b * s, t, d)
        xt = self.attn_t(xt, is_causal=True).reshape(b, s, t, d).permute(0, 2, 1, 3)
        x = x + xt
        # ── MLP ──
        x = x + self.mlp(self.ln_m(x))
        return x


# ---------------------------------------------------------------------------
# Nucleus (top-p) logit masking — shared by the sampling decode
# ---------------------------------------------------------------------------


def _nucleus_mask_logits(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Mask all but the smallest top-``p`` probability nucleus to ``-inf``.

    ``logits`` is ``(rows, vocab)``.  For each row the tokens are sorted by
    descending probability and the smallest prefix whose cumulative probability
    first reaches ``top_p`` is kept; every other token's logit is set to
    ``-inf`` so it has zero probability after the subsequent softmax.  The single
    most-likely token is always kept (the ``> top_p`` test is shifted by one
    position), so a tiny ``top_p`` degrades to argmax rather than an empty set.
    """
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    cum = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    # remove tokens once the cumulative mass has PASSED top_p; shift right by one
    # so the first token that crosses the threshold (and the top-1) are kept.
    remove_sorted = cum > top_p
    remove_sorted[..., 1:] = remove_sorted[..., :-1].clone()
    remove_sorted[..., 0] = False
    remove = torch.zeros_like(remove_sorted)
    remove.scatter_(dim=-1, index=sorted_idx, src=remove_sorted)
    return logits.masked_fill(remove, float("-inf"))


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------


@dataclass
class SpacetimeOutput:
    """Forward output.

    Attributes
    ----------
    hidden:
        ``(B, T, S, d_model)`` per-token hidden states over the CAMERA frames
        (plan frames stripped).  ``hidden[:, t]`` is the state that predicts
        frame ``t+1``'s tokens (next-frame factorisation) — the chunked head /
        chunked argmax consume this without ever building the full logit tensor.
    logits:
        ``(B, T, S, vocab)`` per-token logits — ONLY populated when
        ``return_logits=True`` (tests / small cases); the corpus path leaves it
        ``None`` and uses :meth:`chunked_nll`.
    """

    hidden: torch.Tensor
    logits: torch.Tensor | None = None


class SpacetimeTransformer(nn.Module):
    """Factorised spatiotemporal autoregressive transformer over frame tokens.

    See the module docstring for the architecture and the next-frame
    factorisation.
    """

    def __init__(self, cfg: SpacetimeConfig) -> None:
        super().__init__()
        self.config = cfg
        d = cfg.d_model

        # ── token value embeddings ──
        self.token_embed = nn.Embedding(cfg.vocab_size, d)
        # ── factorised spatial position (row + col), shared across frames ──
        self.row_embed = nn.Embedding(cfg.grid_h, d)
        self.col_embed = nn.Embedding(cfg.grid_w, d)
        # ── temporal frame position (real frames + the plan prefix frames) ──
        self.frame_embed = nn.Embedding(cfg.max_frames, d)

        # ── plan conditioning (optional) ──
        self.has_plan = cfg.plan_channels > 0
        if self.has_plan:
            self.plan_embed = nn.Embedding(cfg.plan_vocab, d)
            # a plan step has its own channel positions; a learned per-channel
            # spatial slot lets the plan tokens occupy distinct spatial lanes so
            # the spatial attention inside a plan frame can tell them apart.
            self.plan_channel_embed = nn.Parameter(torch.zeros(cfg.plan_channels, d))
            # a learned marker so a plan frame is distinguishable from a camera
            # frame (both ride the same temporal axis).
            self.plan_marker = nn.Parameter(torch.zeros(d))
            self.cam_marker = nn.Parameter(torch.zeros(d))

        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([_SpaceTimeBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, cfg.vocab_size, bias=False)
        # weight tying: the head shares the token value-embedding weight.  At
        # 2^18 × d this table is the single largest parameter block; tying halves
        # it and is the standard LM choice — it also keeps the value space and
        # the prediction space aligned.
        self.head.weight = self.token_embed.weight

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    # -- introspection -----------------------------------------------------

    def num_parameters(self, *, trainable_only: bool = True) -> int:
        return sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad or not trainable_only
        )

    # -- embedding ---------------------------------------------------------

    def _embed_camera(self, frames: torch.Tensor) -> torch.Tensor:
        """``(B, T, S) long -> (B, T, S, d)`` camera-token embeddings.

        Value embedding + factorised (row + col) spatial position; the temporal
        frame-position embedding is added by :meth:`_forward_tokens` after the
        plan prefix is concatenated (so the frame index is the position in the
        full plan+camera sequence).
        """
        b, t, s = frames.shape
        cfg = self.config
        emb = self.token_embed(frames)  # (B, T, S, d)
        # factorised spatial position over the 16×16 raster grid.
        rows = torch.arange(cfg.grid_h, device=frames.device).repeat_interleave(
            cfg.grid_w
        )
        cols = torch.arange(cfg.grid_w, device=frames.device).repeat(cfg.grid_h)
        spatial = self.row_embed(rows) + self.col_embed(cols)  # (S, d)
        emb = emb + spatial.view(1, 1, s, cfg.d_model)
        if self.has_plan:
            emb = emb + self.cam_marker
        return emb

    def _embed_plan(self, plan: torch.Tensor) -> torch.Tensor:
        """``(B, P, C) long -> (B, P, S, d)`` plan-frame embeddings (or None).

        The plan has ``C`` channels per step; they are placed in the FIRST C
        spatial lanes (with a learned per-channel slot) and the remaining
        ``S - C`` lanes are zero-filled value embeddings so a plan "frame" is
        the same spatial width ``S`` as a camera frame (lets the same spatial
        attention + the temporal axis treat them uniformly).  Returns ``(B, P,
        S, d)`` or ``None`` when there is no plan.
        """
        if not self.has_plan or plan.numel() == 0 or plan.shape[1] == 0:
            return None
        b, p, c = plan.shape
        cfg = self.config
        s = cfg.n_spatial
        c = min(c, cfg.plan_channels)
        val = self.plan_embed(plan[:, :, :c])  # (B, P, c, d)
        val = val + self.plan_channel_embed[:c].view(1, 1, c, cfg.d_model)
        # pad channel lanes up to S with zeros (no token signal), so a plan frame
        # is spatially S-wide like a camera frame.
        if c < s:
            pad = val.new_zeros((b, p, s - c, cfg.d_model))
            val = torch.cat([val, pad], dim=2)
        # plan frames also get the shared spatial (row+col) position so the
        # spatial attention has a consistent positional basis across plan + cam.
        rows = torch.arange(cfg.grid_h, device=plan.device).repeat_interleave(
            cfg.grid_w
        )
        cols = torch.arange(cfg.grid_w, device=plan.device).repeat(cfg.grid_h)
        spatial = self.row_embed(rows) + self.col_embed(cols)  # (S, d)
        val = val + spatial.view(1, 1, s, cfg.d_model) + self.plan_marker
        return val  # (B, P, S, d)

    def _forward_tokens(
        self, frames: torch.Tensor, plan: torch.Tensor | None
    ) -> torch.Tensor:
        """Run the backbone; return camera-frame hidden states ``(B, T, S, d)``.

        Builds ``[plan_frames | camera_frames]`` along the temporal axis, adds
        the absolute frame-position embedding over the full sequence, runs the
        factorised space-time blocks, and STRIPS the plan frames from the output
        (the plan is conditioning, never a prediction target).
        """
        cfg = self.config
        cam = self._embed_camera(frames)  # (B, T, S, d)
        b, t, s, d = cam.shape
        plan_emb = self._embed_plan(plan) if plan is not None else None
        if plan_emb is not None:
            p = plan_emb.shape[1]
            x = torch.cat([plan_emb, cam], dim=1)  # (B, P+T, S, d)
        else:
            p = 0
            x = cam
            if self.has_plan:
                # No plan in THIS batch, but the model is plan-capable.  Touch the
                # plan parameters with a zero-magnitude contribution so they stay
                # in the autograd graph every step.  Under DDP this is essential:
                # the reducer requires every rank to use the SAME parameter set on
                # each step — a rank whose shard happens to be all-plan-less would
                # otherwise leave the plan params grad-less and desynchronise the
                # ring (hang).  Zero contribution => no change to the prediction.
                zero = (
                    self.plan_embed.weight.sum()
                    + self.plan_channel_embed.sum()
                    + self.plan_marker.sum()
                    + self.cam_marker.sum()
                ) * 0.0
                x = x + zero
        total_t = x.shape[1]
        if total_t > cfg.max_frames:
            raise ValueError(
                f"sequence has {total_t} frames (plan {p} + cam {t}) > max_frames "
                f"{cfg.max_frames}"
            )
        # absolute temporal position over the full plan+camera sequence.
        fpos = torch.arange(total_t, device=frames.device)
        x = x + self.frame_embed(fpos).view(1, total_t, 1, d)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return x[:, p:]  # (B, T, S, d) — camera frames only

    # -- forward / loss ----------------------------------------------------

    def forward(
        self,
        batch: dict,
        *,
        return_logits: bool = False,
        loss_spec: dict | None = None,
    ) -> SpacetimeOutput | torch.Tensor:
        """Teacher-forced forward.

        ``batch`` is ``{"frames": (B, T, S) long, "plan": (B, P, C) long}``.

        * ``loss_spec`` given — return the scalar CHUNKED next-frame NLL
          (computed inside forward so a DDP wrapper sees the full graph).
          ``loss_spec = {"chunk": int, "context_frames": int|None}``; when
          ``context_frames`` is set only forecast-window frames are scored.
        * ``return_logits`` True — also materialise ``(B, T, S, vocab)`` logits
          (tests / tiny cases only).
        * else — return only the hidden states in a :class:`SpacetimeOutput`.
        """
        frames = batch["frames"]
        plan = batch.get("plan")
        hidden = self._forward_tokens(frames, plan)  # (B, T, S, d)

        if loss_spec is not None:
            return self.chunked_nll(
                hidden,
                frames,
                chunk=int(loss_spec.get("chunk", 4096)),
                context_frames=loss_spec.get("context_frames"),
            )

        if return_logits:
            logits = self.head(hidden)  # (B, T, S, vocab) — small cases only
            return SpacetimeOutput(hidden=hidden, logits=logits)
        return SpacetimeOutput(hidden=hidden, logits=None)

    def chunked_nll(
        self,
        hidden: torch.Tensor,
        frames: torch.Tensor,
        *,
        chunk: int = 4096,
        context_frames: int | None = None,
    ) -> torch.Tensor:
        """Next-frame cross-entropy, flattened over (frame, position), chunked.

        ``hidden`` is ``(B, T, S, d)`` from the backbone; ``hidden[:, t]``
        predicts frame ``t+1``.  The target for prediction position ``t`` is
        ``frames[:, t+1]`` (its 256 tokens).  The (prediction, target) pairs are
        flattened over ``(B, T-1, S)`` and cross-entropy is accumulated CHUNK
        rows at a time so the head logits never exceed ``chunk · vocab``.

        ``context_frames`` (optional): when set, only PREDICTIONS that land in
        the forecast window (target frame index ``>= context_frames``) are
        scored — the forecasting objective.  ``None`` scores every next-frame
        prediction (the standard teacher-forced LM loss the overfit gate uses).
        """
        b, t, s, d = hidden.shape
        if t < 2:
            raise ValueError("need >= 2 frames to form a next-frame target")
        pred_h = hidden[:, : t - 1]  # (B, T-1, S, d) predicts frames 1..T-1
        target = frames[:, 1:t]  # (B, T-1, S)
        # forecasting mask over the TARGET frame index (1..T-1)
        if context_frames is not None:
            tgt_frame_idx = torch.arange(1, t, device=hidden.device)
            keep = tgt_frame_idx >= int(context_frames)  # (T-1,)
            if not bool(keep.any()):
                keep = torch.ones_like(keep)  # window too short — score all
            pred_h = pred_h[:, keep]
            target = target[:, keep]
        flat_h = pred_h.reshape(-1, d)  # (N, d)
        flat_tgt = target.reshape(-1)  # (N,)
        n = flat_h.shape[0]
        total = hidden.new_zeros(())
        for start in range(0, n, chunk):
            stop = min(start + chunk, n)
            logits = self.head(flat_h[start:stop])  # (chunk, vocab)
            total = total + F.cross_entropy(
                logits, flat_tgt[start:stop], reduction="sum"
            )
            del logits
        return total / max(n, 1)

    @torch.no_grad()
    def chunked_argmax_frame(
        self, hidden_at_prev: torch.Tensor, *, chunk: int = 4096
    ) -> torch.Tensor:
        """Argmax-decode one frame's 256 tokens from a predecessor hidden state.

        ``hidden_at_prev`` is ``(B, S, d)`` — the backbone hidden at the frame
        BEFORE the one being generated.  Returns ``(B, S)`` long predicted LOCAL
        token ids, computed chunk rows at a time over the flattened (B·S) axis so
        the head logits never exceed ``chunk · vocab``.
        """
        b, s, d = hidden_at_prev.shape
        flat = hidden_at_prev.reshape(-1, d)
        out = flat.new_zeros(flat.shape[0], dtype=torch.long)
        for start in range(0, flat.shape[0], chunk):
            stop = min(start + chunk, flat.shape[0])
            logits = self.head(flat[start:stop])  # (chunk, vocab)
            out[start:stop] = logits.argmax(dim=-1)
            del logits
        return out.reshape(b, s)

    @torch.no_grad()
    def chunked_sample_frame(
        self,
        hidden_at_prev: torch.Tensor,
        *,
        temperature: float = 1.0,
        top_p: float = 1.0,
        chunk: int = 4096,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Temperature + nucleus (top-p) sample one frame's tokens per position.

        The stochastic counterpart of :meth:`chunked_argmax_frame`.  Argmax is
        mode-seeking — under an autoregressive rollout it collapses the camera
        forecast onto the single most-likely token at every position, which is
        empirically the persistence (frozen-last-frame) solution.  Sampling from
        the (temperature-scaled, nucleus-truncated) per-token categorical instead
        draws a *coherent realisation* from the predictive distribution rather
        than its mode, which is the right object to score against persistence with
        a distributional metric.

        The output head is a single softmax over the full LFQ codebook (the LFQ
        bits were fused into one ``vocab_size`` id at tokenisation), so the
        per-token distribution is the softmax of the ``vocab_size`` logits at that
        position; there is no per-bit factorisation to sample independently.

        Parameters
        ----------
        hidden_at_prev:
            ``(B, S, d)`` backbone hidden at the frame BEFORE the one generated.
        temperature:
            Softmax temperature.  ``> 0`` divides the logits before the softmax
            (``< 1`` sharpens toward the mode, ``> 1`` flattens).  ``<= 0`` falls
            back to a deterministic argmax (so a caller can pass ``temperature=0``
            for the greedy baseline through the same entry point).
        top_p:
            Nucleus mass in ``(0, 1]``.  Only the smallest set of tokens whose
            cumulative probability first reaches ``top_p`` is kept; the rest are
            masked to zero probability before renormalising and sampling.  The
            single most-likely token is ALWAYS kept, so ``top_p`` near 0 degrades
            gracefully to argmax rather than emptying the support.
        chunk:
            Rows of the flattened ``(B·S)`` axis processed per head call so the
            head logits never exceed ``chunk · vocab`` (same budget as argmax).
        generator:
            Optional ``torch.Generator`` for a reproducible draw.

        Returns
        -------
        ``(B, S)`` long sampled LOCAL token ids.
        """
        b, s, d = hidden_at_prev.shape
        if temperature is None or temperature <= 0.0:
            return self.chunked_argmax_frame(hidden_at_prev, chunk=chunk)
        top_p = float(top_p)
        if not 0.0 < top_p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1]; got {top_p}")
        flat = hidden_at_prev.reshape(-1, d)
        out = flat.new_zeros(flat.shape[0], dtype=torch.long)
        for start in range(0, flat.shape[0], chunk):
            stop = min(start + chunk, flat.shape[0])
            logits = self.head(flat[start:stop]).float()  # (rows, vocab)
            logits = logits / float(temperature)
            if top_p < 1.0:
                logits = _nucleus_mask_logits(logits, top_p)
            probs = torch.softmax(logits, dim=-1)
            idx = torch.multinomial(probs, num_samples=1, generator=generator)
            out[start:stop] = idx.squeeze(-1)
            del logits, probs, idx
        return out.reshape(b, s)
