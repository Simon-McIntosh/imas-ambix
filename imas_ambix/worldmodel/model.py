"""Plan-conditioned decoder-only Transformer world model (§6 piece 2).

Architecture (the "GPT" pattern)
--------------------------------
A stack of causal self-attention layers that predicts the next token from all
previous tokens.  Three prototype-specific design points implement §3:

1. **Per-group-local token embeddings.**  The substrate's locked decision is
   that token ids are PER-GROUP-LOCAL — an id is meaningless without the group
   it came from, because independently-encoded groups overlap in global id
   space.  So each modality (``summary``, ``pf_active``, ``xma``, ``camera``,
   ...) gets its OWN ``nn.Embedding`` table over its own local vocabulary, and
   its OWN next-token head.  A per-step, per-modality token is embedded by that
   modality's table; the per-modality channel embeddings at one grid step are
   summed (a bag-of-channels) into a single step embedding, plus a learned
   per-modality "which modality" embedding so the model can tell streams apart.

2. **Pulse-schedule injected as PREPENDED conditioning tokens** (the simplest
   mechanism; locked §-decision "prepend-tokens").  The plan (the
   ``pulse_schedule`` modality) is embedded the same way and the WHOLE plan
   sequence is prepended to the observation sequence as a prefix the causal
   attention can read at every step.  Because the prefix sits before step 0,
   every predicted observation token can attend to the entire plan — exactly
   "the same initial state under two different plans yields two different
   futures".

3. **Per-channel next-token head.**  Each non-conditioning modality has a
   linear head producing logits over its local vocabulary, applied to every
   channel of that modality at every step.  Training is teacher-forced
   next-token NLL summed over channels and modalities (see
   :mod:`imas_ambix.worldmodel.train`).

Sequence layout (one shot)
--------------------------
The token sequence the Transformer sees is::

    [ plan_0 plan_1 ... plan_{P-1} | obs_0 obs_1 ... obs_{N-1} ]
      <----- conditioning prefix --->  <----- observations ----->

where each ``plan_t`` / ``obs_t`` is ONE fused step embedding (the sum over
that step's modality-channel embeddings).  Position embeddings are learned and
absolute over the full prefix+obs length.  The model predicts ``obs_{t}`` from
the prefix + ``obs_{<t}`` (causal); the prefix is never a prediction target.

Prototype size + context length (documented contract)
-----------------------------------------------------
The default :class:`WorldModelConfig` is intentionally small, but the modality
set is the FULL tokenised substrate (see
:func:`~imas_ambix.worldmodel.dataset.default_modalities`): the conditioning
plan, the five measured L2 light-path groups (``summary``, ``pf_active``,
``interferometer``, ``gas_injection``, ``soft_x_rays``), the three L1
high-frequency streams (``xma``, ``xim``, ``xsx``), and ALL FIVE MAST cameras
(``rbb``, ``rba``, ``rco``, ``rgb``, ``rgc``).

The per-modality embedding tables dominate the param count once the cameras are
included.  Each camera gets its OWN ``nn.Embedding`` + next-token ``nn.Linear``
head over the 2^18 LFQ vocabulary, so EACH camera contributes ``2^18 * d_model``
embedding params + ``d_model * 2^18`` head params — the five cameras together
are the overwhelming driver.  The xim head (vocab 12806) and xsx head (vocab
1030) add a further, much smaller, vocab-sized contribution; the L2 groups
(vocab 257) and xma (vocab 8) are negligible.  Concretely the five cameras are
~1.0B params at ``d_model=384`` (~0.2B per camera) and ~0.34B at ``d_model=128``
— and the whole model (all modalities, the five SEPARATE camera tables/heads,
bf16 weights + AdamW state) fits on ONE H200 (140 GB) with very large headroom:
the five-camera tables are ~16 GB even in the worst-case full-fp32 AdamW
accounting (~2 GB bf16 weights + ~12-14 GB AdamW state at d_model=384), so NO
shared-codebook compression is needed and the cameras stay per-camera to keep
per-camera identity exact.  Per shot only the cameras actually present receive a
gradient (an absent camera's block is all-PAD + masked), and the single-device
trainer (:func:`~imas_ambix.worldmodel.train.train_corpus`) loads the whole
model once on its one GPU.  If a much wider future run ever needed it, the five
cameras COULD share one embedding (they share the LFQ codebook) — the wiring is
unchanged; only ``WorldModel`` would route the camera names to a common
embedding key while keeping per-camera heads + ``modality_embed`` identity — but
at the documented scales this is unnecessary.
Context length is
``plan_steps + n_steps`` fused steps (default ``64 + 64 = 128`` positions).
:meth:`WorldModel.num_parameters` returns the live count and
:meth:`WorldModel.context_length` the live position budget so the actual
numbers are never guessed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from collections.abc import Sequence

    from imas_ambix.worldmodel.dataset import ModalitySpec


@dataclass
class ModalityHeadSpec:
    """The model-side view of one modality (decoupled from the dataset spec).

    Attributes
    ----------
    name:
        Modality key (matches the dataset sample's token-dict key).
    vocab_size:
        Local vocabulary size (embedding rows + head outputs).
    n_channels:
        Number of token channels this modality contributes per step.
    is_conditioning:
        True for the plan (prepended, no prediction head).
    """

    name: str
    vocab_size: int
    n_channels: int
    is_conditioning: bool = False


@dataclass
class WorldModelConfig:
    """Hyper-parameters for the plan-conditioned world model.

    Attributes
    ----------
    modalities:
        Per-modality head specs (built from the dataset modality set via
        :meth:`WorldModelConfig.from_modalities`).
    d_model, n_layers, n_heads, d_ff:
        Transformer width / depth / attention heads / feed-forward width.
    dropout:
        Dropout probability.
    plan_steps, obs_steps:
        Prefix (plan) length and observation length, in fused steps — together
        the absolute position budget.
    """

    modalities: list[ModalityHeadSpec]
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 256
    dropout: float = 0.0
    plan_steps: int = 64
    obs_steps: int = 64

    @classmethod
    def from_modalities(
        cls,
        modalities: Sequence[ModalitySpec],
        channels: dict[str, int],
        *,
        plan_steps: int,
        obs_steps: int,
        **kwargs: object,
    ) -> WorldModelConfig:
        """Build a config from dataset :class:`ModalitySpec`s + channel counts.

        A head + embedding table is built for EVERY declared modality —
        UNCONDITIONALLY, never filtered by which modalities a probe happened to
        see.  Cameras/HF streams live in high shot-ids, so a probe over the
        first few (low-id) shots would otherwise drop them and collapse the
        intended ~1B-param all-streams model to a tiny one; building from the
        DECLARED set keeps every camera/HF/L2 head present.

        Each modality's FIXED per-step channel width is resolved robustly, in
        priority order:

        1. the probed count ``channels[name]`` (the MAX seen across probe shots
           that actually carried it) — preferred when the modality was probed;
        2. otherwise the spec's :meth:`ModalitySpec.fixed_channel_width` — a
           structural constant for cameras (16×16 frame grid sub-sampled at the
           camera's stride) and for any modality that pins ``n_channels``;
        3. otherwise ``1`` — a degenerate but non-zero block so the table
           exists; a shot carrying the modality is pad/truncated to this width
           (collate) and a shot lacking it is the all-PAD masked block.

        ``channels`` maps modality name -> probed channel count.
        """
        heads: list[ModalityHeadSpec] = []
        for m in modalities:
            width = channels.get(m.name)
            if width is None:
                width = m.fixed_channel_width()
            if width is None or int(width) < 1:
                width = 1
            heads.append(
                ModalityHeadSpec(
                    name=m.name,
                    vocab_size=m.vocab_size,
                    n_channels=int(width),
                    is_conditioning=m.is_conditioning,
                )
            )
        return cls(
            modalities=heads,
            plan_steps=plan_steps,
            obs_steps=obs_steps,
            **kwargs,  # type: ignore[arg-type]
        )

    @property
    def context_length(self) -> int:
        return self.plan_steps + self.obs_steps


# ---------------------------------------------------------------------------
# Transformer building blocks (minimal, self-contained, CPU-friendly)
# ---------------------------------------------------------------------------


class _CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention.

    The first ``prefix_len`` positions (the plan prefix) are fully visible to
    every position (non-causal among themselves and visible to all obs steps)
    via the additive mask; the observation positions are causal w.r.t. each
    other.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model {d_model} not divisible by n_heads {n_heads}")
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        qkv = self.qkv(x).view(b, t, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)
        # -> (B, n_heads, T, d_head)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        scores = scores + attn_mask  # (1, 1, T, T) additive -inf mask
        weights = torch.softmax(scores, dim=-1)
        weights = self.drop(weights)
        out = weights @ v  # (B, n_heads, T, d_head)
        out = out.transpose(1, 2).reshape(b, t, c)
        return self.proj(out)


class _Block(nn.Module):
    """Pre-norm Transformer block (attention + feed-forward)."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = _CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), attn_mask)
        x = x + self.mlp(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# The world model
# ---------------------------------------------------------------------------


@dataclass
class WorldModelOutput:
    """Forward-pass output.

    Attributes
    ----------
    logits:
        ``{modality_name: (B, obs_steps, n_channels, vocab_size)}`` next-token
        logits for every NON-conditioning modality.  Position ``t`` predicts
        the observation token at grid step ``t`` from the plan prefix +
        observation steps ``< t``.  EMPTY when the forward was run with
        ``return_logits=False`` (the memory-safe full-resolution path, which
        keeps only ``obs_hidden`` and never materialises the all-channel logit
        tensor — see :meth:`WorldModel.encode`).
    obs_hidden:
        ``(B, obs_steps, d_model)`` per-step observation hidden states (the
        shared step state BEFORE the per-channel query + head).  This is the
        cheap, channel-count-independent quantity the chunked cross-entropy /
        chunked argmax consume so the full-resolution camera head never
        materialises ``(B, T, n_channels, vocab)`` at once.
    """

    logits: dict[str, torch.Tensor] = field(default_factory=dict)
    obs_hidden: torch.Tensor | None = None


class WorldModel(nn.Module):
    """Plan-conditioned decoder-only Transformer over multi-modal tokens.

    See the module docstring for the architecture and sequence layout.
    """

    def __init__(self, config: WorldModelConfig) -> None:
        super().__init__()
        self.config = config
        d = config.d_model

        # Per-group-local embedding tables (one per modality, including plan).
        self.token_embed = nn.ModuleDict(
            {m.name: nn.Embedding(m.vocab_size, d) for m in config.modalities}
        )
        # "which modality" embedding so the fused step knows its streams apart.
        self.modality_embed = nn.ParameterDict(
            {m.name: nn.Parameter(torch.zeros(d)) for m in config.modalities}
        )
        # Learned absolute position embedding over the full prefix+obs length.
        self.pos_embed = nn.Embedding(config.context_length, d)
        # A learned marker distinguishing the plan prefix from observations.
        self.segment_embed = nn.Embedding(2, d)  # 0 = plan, 1 = obs

        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [
                _Block(d, config.n_heads, config.d_ff, config.dropout)
                for _ in range(config.n_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(d)

        # Per-channel next-token heads for every NON-conditioning modality.
        # The head is a Linear over (fused step state + a learned per-channel
        # query), so channels of the same modality share head weights but get
        # DISTINCT logits — the model can fit per-channel token sequences
        # (e.g. the 15 distinct pf_active coil currents at one grid step).
        self.heads = nn.ModuleDict(
            {
                m.name: nn.Linear(d, m.vocab_size)
                for m in config.modalities
                if not m.is_conditioning
            }
        )
        self.channel_query = nn.ParameterDict(
            {
                m.name: nn.Parameter(torch.zeros(max(m.n_channels, 1), d))
                for m in config.modalities
                if not m.is_conditioning
            }
        )

        self._plan_names = [m.name for m in config.modalities if m.is_conditioning]
        self._obs_names = [m.name for m in config.modalities if not m.is_conditioning]
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
        """Live parameter count (documented contract: never guessed)."""
        return sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad or not trainable_only
        )

    def context_length(self) -> int:
        """Live absolute position budget (plan prefix + observation steps)."""
        return self.config.context_length

    # -- step fusion -------------------------------------------------------

    def _fuse_step_embeddings(
        self,
        tokens: dict[str, torch.Tensor],
        names: Sequence[str],
        *,
        t_slice: slice | None = None,
    ) -> torch.Tensor:
        """Sum each step's per-modality channel embeddings into one embedding.

        ``tokens[name]`` is ``(B, T, n_channels)`` local ids.  Returns
        ``(B, T_sel, d_model)`` where each step is the sum over modalities and
        their channels (a bag-of-channels), plus the per-modality marker.
        """
        fused: torch.Tensor | None = None
        for name in names:
            tok = tokens[name]
            if t_slice is not None:
                tok = tok[:, t_slice]
            emb = self.token_embed[name](tok)  # (B, T, n_ch, d)
            step = emb.sum(dim=2) + self.modality_embed[name]  # (B, T, d)
            fused = step if fused is None else fused + step
        if fused is None:
            raise ValueError("no modalities to fuse")
        return fused

    def _build_attn_mask(
        self, prefix_len: int, obs_len: int, device: torch.device
    ) -> torch.Tensor:
        """Additive attention mask: plan prefix fully visible, obs causal.

        Shape ``(1, 1, T, T)`` with 0 where attention is allowed and -inf where
        blocked.  ``T = prefix_len + obs_len``.  A query may attend to: any
        prefix key (the whole plan), and any obs key at position <= itself.
        """
        t = prefix_len + obs_len
        allowed = torch.zeros(t, t, dtype=torch.bool, device=device)
        # everyone may attend to the prefix
        allowed[:, :prefix_len] = True
        # obs queries: causal among obs (and prefix already allowed)
        obs_idx = torch.arange(obs_len, device=device)
        causal = obs_idx[:, None] >= obs_idx[None, :]  # (obs, obs)
        allowed[prefix_len:, prefix_len:] = causal
        # prefix queries: may attend among the whole prefix (bidirectional)
        # (already True from the column rule above)
        mask = torch.zeros(t, t, dtype=torch.float32, device=device)
        mask.masked_fill_(~allowed, float("-inf"))
        return mask.view(1, 1, t, t)

    # -- forward -----------------------------------------------------------

    def encode(self, batch: dict) -> torch.Tensor:
        """Run the backbone and return the per-step observation hidden states.

        ``batch["tokens"][name]`` is ``(B, n_steps, n_channels)`` LONG local
        ids for every modality.  The first :attr:`config.plan_steps` rows of
        the conditioning modality are the plan prefix; the leading
        ``obs_steps`` rows of the non-conditioning modalities are the
        observation sequence.

        Returns ``obs_hidden`` ``(B, obs_len, d_model)`` — the shared step
        state at every observation position, BEFORE the per-channel query +
        head.  This is the cheap, channel-count-INDEPENDENT quantity from which
        both the full per-channel logits (small modalities / tests) and the
        chunked cross-entropy / chunked argmax (the full-resolution camera) are
        derived, so the all-channel ``(B, T, n_channels, vocab)`` tensor never
        has to exist for a wide head.
        """
        tokens = batch["tokens"]
        device = next(self.parameters()).device

        # ground the lengths in the actual data, capped by the position budget
        any_obs = tokens[self._obs_names[0]]
        _b, n_steps, _ = any_obs.shape
        obs_len = min(n_steps, self.config.obs_steps)
        plan_len = min(self.config.plan_steps, tokens[self._plan_names[0]].shape[1])

        # Prefix embedding (the plan).  Use the leading plan_len plan steps.
        prefix = self._fuse_step_embeddings(
            tokens, self._plan_names, t_slice=slice(0, plan_len)
        )  # (B, plan_len, d)
        # Observation embedding (everything the model predicts).
        obs = self._fuse_step_embeddings(
            tokens, self._obs_names, t_slice=slice(0, obs_len)
        )  # (B, obs_len, d)

        # segment + position embeddings
        seg_plan = self.segment_embed(
            torch.zeros(plan_len, dtype=torch.long, device=device)
        )
        seg_obs = self.segment_embed(
            torch.ones(obs_len, dtype=torch.long, device=device)
        )
        prefix = prefix + seg_plan
        obs = obs + seg_obs

        x = torch.cat([prefix, obs], dim=1)  # (B, plan_len+obs_len, d)
        t_total = x.shape[1]
        pos = torch.arange(t_total, device=device)
        x = x + self.pos_embed(pos)
        x = self.drop(x)

        attn_mask = self._build_attn_mask(plan_len, obs_len, device)
        for block in self.blocks:
            x = block(x, attn_mask)
        x = self.ln_f(x)

        return x[:, plan_len:]  # (B, obs_len, d) — observation hidden states

    def channel_logits(
        self,
        obs_hidden: torch.Tensor,
        name: str,
        *,
        ch_start: int = 0,
        ch_stop: int | None = None,
    ) -> torch.Tensor:
        """Per-channel next-token logits for a slice of one modality's channels.

        ``obs_hidden`` is ``(B, obs_len, d)`` from :meth:`encode`.  Returns
        ``(B, obs_len, n_ch_slice, vocab)`` logits for channels
        ``[ch_start, ch_stop)`` of modality ``name`` — the per-channel query
        added to the shared step state then the modality head.  Slicing the
        channel range is what makes the full-resolution head fit memory: a
        caller materialises at most ``chunk × vocab`` logits at a time instead
        of ``n_channels × vocab`` (the ~50 GB wall at 256 channels × 2^18
        vocab).  Numerically identical to the full computation per channel.
        """
        query_all = self.channel_query[name]  # (fixed_ch, d)
        fixed_ch = query_all.shape[0]
        if ch_stop is None:
            ch_stop = fixed_ch
        ch_stop = min(int(ch_stop), fixed_ch)
        ch_start = max(int(ch_start), 0)
        query = query_all[ch_start:ch_stop]  # (n_ch_slice, d)
        # (B, obs_len, 1, d) + (n_ch_slice, d) -> (B, obs_len, n_ch_slice, d)
        per_ch = obs_hidden.unsqueeze(2) + query
        return self.heads[name](per_ch)  # (B, obs_len, n_ch_slice, vocab)

    def chunked_nll(
        self,
        obs_hidden: torch.Tensor,
        batch: dict,
        obs_names: Sequence[str],
        *,
        target_only: bool = False,
        chunk_channels: int = 16,
    ) -> torch.Tensor:
        """Masked teacher-forced next-token NLL, computed channel-chunk at a time.

        Numerically EQUAL to building the full per-channel logits and taking the
        per-modality ``cross_entropy(reduction="mean")`` over the valid
        (position, channel) pairs, then the mean over modalities — but the
        per-channel logits are built one channel-chunk at a time (via
        :meth:`channel_logits`) and freed before the next chunk, so peak head
        memory is ``~chunk_channels × vocab`` not ``n_channels × vocab`` (the
        ~50 GB wall at the full-resolution camera's 256 channels × 2^18 vocab).

        ``obs_hidden`` is ``(B, obs_len, d)`` from :meth:`encode`.  The
        mean-over-valid identity is preserved exactly: per modality the
        per-element NLL is accumulated with ``reduction="sum"`` over chunks plus
        the total valid count, then divided once.  This lives ON the model so
        the loss is produced INSIDE :meth:`forward` (``loss_spec``), which is
        what lets a ``DistributedDataParallel`` wrapper see the full
        backbone+head autograd graph and all-reduce every gradient.
        """
        ce = nn.functional.cross_entropy
        ctx = int(batch["context_steps"])
        total = torch.zeros((), dtype=torch.float32, device=obs_hidden.device)
        n_terms = 0
        obs_len = obs_hidden.shape[1]
        for name in obs_names:
            tgt = batch["tokens"][name]  # (B, T, C)
            val = batch["valid"][name]  # (B, T, C)
            fixed_ch = int(self.channel_query[name].shape[0])
            in_ch = int(tgt.shape[2])
            n_ch = min(in_ch, fixed_ch)
            if n_ch < 1:
                continue
            t = min(obs_len, int(tgt.shape[1]))
            if t < 2:
                continue
            target_all = tgt[:, 1:t, :n_ch]  # (B, T-1, n_ch)
            tvalid_all = val[:, 1:t, :n_ch]  # (B, T-1, n_ch)
            if target_only:
                step_idx = torch.arange(1, t, device=obs_hidden.device)
                in_target = (step_idx >= ctx).view(1, -1, 1)
                tvalid_all = tvalid_all & in_target
            n_valid = int(tvalid_all.sum())
            if n_valid == 0:
                continue
            hidden_pred = obs_hidden[:, : t - 1]  # (B, T-1, d)
            mod_sum = torch.zeros((), dtype=torch.float32, device=obs_hidden.device)
            chunk = max(1, int(chunk_channels))
            for cs in range(0, n_ch, chunk):
                ce_stop = min(cs + chunk, n_ch)
                lg = self.channel_logits(
                    hidden_pred, name, ch_start=cs, ch_stop=ce_stop
                )
                v = lg.shape[-1]
                tgt_c = target_all[:, :, cs:ce_stop]
                val_c = tvalid_all[:, :, cs:ce_stop]
                flat_pred = lg.reshape(-1, v)
                flat_tgt = tgt_c.reshape(-1)
                flat_mask = val_c.reshape(-1)
                if flat_mask.any():
                    mod_sum = mod_sum + ce(
                        flat_pred[flat_mask], flat_tgt[flat_mask], reduction="sum"
                    )
                del lg
            total = total + mod_sum / n_valid
            n_terms += 1
        if n_terms == 0:
            raise ValueError("no valid target positions in batch — cannot compute NLL")
        return total / n_terms

    def forward(
        self,
        batch: dict,
        *,
        return_logits: bool = True,
        loss_spec: dict | None = None,
    ) -> WorldModelOutput | torch.Tensor:
        """Teacher-forced forward pass.

        Three modes, all running the backbone (:meth:`encode`) once:

        * ``loss_spec`` given — return the scalar CHUNKED next-token NLL
          (:meth:`chunked_nll`) computed INSIDE this forward.  This is the DDP
          path: a ``DistributedDataParallel`` wrapper drives ``forward``, so
          producing the loss here lets DDP's reducer see the full
          backbone+head graph and all-reduce every gradient.  ``loss_spec`` is
          ``{"obs_names": [...], "target_only": bool, "chunk_channels": int}``.
        * ``return_logits`` True (default — small modalities, tests, the eval
          skill path) — materialise the full per-channel next-token logits for
          every observation modality and return them in a
          :class:`WorldModelOutput`.
        * ``return_logits`` False — return only ``obs_hidden`` (no all-channel
          logits), so a caller can apply the head channel-chunk at a time.
        """
        tokens = batch["tokens"]
        obs_hidden = self.encode(batch)

        if loss_spec is not None:
            return self.chunked_nll(
                obs_hidden,
                batch,
                loss_spec["obs_names"],
                target_only=bool(loss_spec.get("target_only", False)),
                chunk_channels=int(loss_spec.get("chunk_channels", 16)),
            )

        if not return_logits:
            return WorldModelOutput(logits={}, obs_hidden=obs_hidden)

        logits: dict[str, torch.Tensor] = {}
        for name in self._obs_names:
            # Emit logits at the modality's FIXED head/channel_query width — the
            # width the head + channel_query were built to — NOT the incoming
            # token width.  Callers (collate) pad/truncate tokens to this fixed
            # width (``pad_collate_batch``); clamping here makes the model
            # self-consistent so a wider-than-model input can never produce
            # logits of a width that mismatches its targets downstream (the
            # blocker-2 crash, which surfaced not in forward but in the loss /
            # skill score when logits width != target width).
            fixed_ch = self.channel_query[name].shape[0]
            in_ch = tokens[name].shape[2]
            n_ch = min(in_ch, fixed_ch)
            logits[name] = self.channel_logits(
                obs_hidden, name, ch_start=0, ch_stop=n_ch
            )
        return WorldModelOutput(logits=logits, obs_hidden=obs_hidden)
