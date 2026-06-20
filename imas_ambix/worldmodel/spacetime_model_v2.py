"""Signal-conditioned spatiotemporal camera transformer (the scaled variant).

What this adds over :mod:`imas_ambix.worldmodel.spacetime_model`
---------------------------------------------------------------
The v1 model conditions the camera-frame prediction on the pulse-schedule plan
ONLY.  The plan is the *programmed demand* — what the operator asked the machine
to do — but the machine's actual response (the plasma state) is observed by the
measured diagnostics.  v1 therefore had to GENERATE the response from the demand
with no measured feedback, which is why it produced plausible-looking frames but
LOST to persistence at forecasting: the demand under-determines the next frame.

This model adds the MEASURED signal streams (magnetics ``xma``, density
``interferometer``, ``soft_x_rays``, and the L2 measured groups ``summary`` /
``pf_active`` / ``gas_injection``) as additional CONDITIONING context, attended
exactly like the plan.  The hypothesis is that the measured plasma state lets the
model FORECAST (track the true evolution) rather than merely dream a coherent
clip.  The camera frame prediction stays the only target — signals are context,
never predicted.

How a signal stream is conditioned
----------------------------------
Each measured stream is sub-sampled to a small set of conditioning "frames"
(steps) spanning the camera window, mirroring how the plan is prepended:

* **per-stream value embedding** — each stream has its OWN ``nn.Embedding`` sized
  to its own local vocabulary (the L2 groups share a 257-id vocab; ``xma`` is 8,
  ``xim`` 12806, ``xsx`` 1030).  Per-group-local ids are meaningless across
  groups, so a shared table would be wrong — each stream gets its own.
* **per-channel spatial lane** — a stream's ``C`` channels occupy the first ``C``
  spatial lanes of a conditioning frame (a learned per-stream channel slot lets
  the within-frame spatial attention tell channels apart); the remaining lanes
  are zero-filled, so a signal frame is the same spatial width ``S`` as a camera
  frame and rides the shared spatial + temporal positional basis.
* **stream-type embedding** — a learned per-stream marker so the model knows
  WHICH diagnostic a conditioning frame carries (magnetics vs density vs …), and
  a single ``signal_marker`` distinguishing a signal frame from a plan frame and
  from a camera frame.

All conditioning frames (plan + every present signal stream) are concatenated on
the temporal axis BEFORE frame 0, so every predicted camera token attends back
causally to the whole conditioning context.  A stream absent for a shot simply
contributes no frames (its block is omitted) — but, mirroring the plan's
zero-touch, the model touches every signal parameter with a zero-magnitude
contribution on a signal-less batch so DDP's reducer sees a uniform parameter set
on every rank (no ring desync / hang).

The backbone, the factorised space-time attention, the chunked cross-entropy
head, and the next-frame factorisation are all inherited unchanged from v1 — only
the conditioning prefix construction and the embedding tables differ.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from imas_ambix.worldmodel.spacetime_model import (
    SpacetimeConfig,
    SpacetimeOutput,
    SpacetimeTransformer,
    _SpaceTimeBlock,
)

# ---------------------------------------------------------------------------
# Signal-stream specification (model side)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SignalStreamSpec:
    """One measured conditioning stream the model embeds.

    Attributes
    ----------
    name:
        Stable stream key (matches the dataset modality name, e.g. ``"xma"``,
        ``"interferometer"``).  Also the embedding-table key.
    vocab:
        Local vocabulary size for this stream's value-embedding table (the L2
        groups are 257; ``xma`` 8; ``xim`` 12806; ``xsx`` 1030).  Always sized so
        every rebased local id and the PAD id (0) fit.
    channels:
        Number of channels this stream contributes per conditioning step.  A
        stream's channels occupy the first ``channels`` spatial lanes of a signal
        frame; ``channels`` must be ``<= n_spatial``.
    """

    name: str
    vocab: int
    channels: int


@dataclass
class SignalSpacetimeConfig(SpacetimeConfig):
    """:class:`SpacetimeConfig` + the measured-signal conditioning streams.

    ``signal_streams`` lists the measured streams the model embeds as
    conditioning frames; ``n_signal_steps`` is the number of conditioning steps
    each present stream is sub-sampled to (mirrors the plan's ``n_plan``).  When
    ``signal_streams`` is empty the model is byte-equivalent to v1 (plan-only).

    ``corruption_levels`` adds the anti-drift history-corruption conditioning: a
    learned per-level embedding (this many rows) is added to the CONTEXT camera
    frames so the model knows how corrupt its history is and can correct it.  0
    or 1 disables it (the model is then byte-equivalent to the un-corrupted v2),
    so a checkpoint trained without it loads cleanly.

    ``max_frames`` must cover plan steps + every present stream's signal steps +
    the camera frames; the builder sizes it.
    """

    signal_streams: tuple[SignalStreamSpec, ...] = field(default_factory=tuple)
    n_signal_steps: int = 4
    corruption_levels: int = 0

    @property
    def has_signals(self) -> bool:
        return len(self.signal_streams) > 0

    @property
    def has_corruption(self) -> bool:
        return int(self.corruption_levels) > 1


# ---------------------------------------------------------------------------
# The signal-conditioned model
# ---------------------------------------------------------------------------


class SignalSpacetimeTransformer(SpacetimeTransformer):
    """v1 backbone + measured-signal conditioning frames on the temporal axis.

    The plan-conditioning, the factorised space-time blocks, the chunked NLL /
    argmax head, and the next-frame factorisation are inherited verbatim from
    :class:`SpacetimeTransformer`.  This subclass adds per-stream signal
    embeddings and overrides only the prefix construction so signal frames are
    prepended alongside the plan, with the camera frames stripped from the output
    exactly as before.
    """

    config: SignalSpacetimeConfig

    def __init__(self, cfg: SignalSpacetimeConfig) -> None:
        # Build the v1 parameters (token/spatial/temporal/plan embeddings, blocks,
        # head, weight tying) via the parent constructor, then add the signal
        # parameters.  The parent's ``self.apply(_init_weights)`` runs over the v1
        # params; we init the new signal params explicitly afterwards so they get
        # the same std=0.02 normal init.
        super().__init__(cfg)
        self.config = cfg  # narrow the type for the signal-aware paths

        self.has_signals = cfg.has_signals
        d = cfg.d_model
        s = cfg.n_spatial
        # Per-stream value embeddings (each its own local vocab) + a learned
        # per-channel spatial slot so a stream's channels occupy distinct lanes,
        # and a learned per-stream-type marker.  Held in ModuleDict / ParameterDict
        # keyed by stream name so a checkpoint records WHICH streams were trained.
        self.signal_embed = nn.ModuleDict()
        self.signal_channel_embed = nn.ParameterDict()
        self.signal_type_embed = nn.ParameterDict()
        for stream in cfg.signal_streams:
            if stream.channels > s:
                raise ValueError(
                    f"signal stream {stream.name!r} has {stream.channels} channels "
                    f"> n_spatial {s}; reduce channels or grow the grid"
                )
            self.signal_embed[stream.name] = nn.Embedding(int(stream.vocab), d)
            self.signal_channel_embed[stream.name] = nn.Parameter(
                torch.zeros(int(stream.channels), d)
            )
            self.signal_type_embed[stream.name] = nn.Parameter(torch.zeros(d))
        if self.has_signals:
            # one marker distinguishing ANY signal frame from plan / camera frames
            # (both v1 markers already exist on the parent).
            self.signal_marker = nn.Parameter(torch.zeros(d))
            # init the new value-embedding tables to match the v1 std.
            for emb in self.signal_embed.values():
                nn.init.normal_(emb.weight, std=0.02)

        # Anti-drift history-corruption conditioning: a learned per-level
        # embedding added to the CONTEXT camera frames so the model conditions on
        # how corrupt its fed-back history is.  Row 0 = clean (rate 0) is
        # zero-initialised so a model that has never seen corruption (or runs at
        # inference with level 0) is unchanged from the un-corrupted v2.
        self.has_corruption = cfg.has_corruption
        if self.has_corruption:
            self.corruption_embed = nn.Embedding(int(cfg.corruption_levels), d)
            nn.init.zeros_(self.corruption_embed.weight)

    # -- embedding ---------------------------------------------------------

    def _embed_signal_stream(
        self, name: str, tokens: torch.Tensor, vocab: int
    ) -> torch.Tensor | None:
        """``(B, P, C) long -> (B, P, S, d)`` conditioning frames for one stream.

        Mirrors :meth:`SpacetimeTransformer._embed_plan`: the stream's ``C``
        channels occupy the first ``C`` spatial lanes (with the learned per-channel
        slot), the rest are zero-filled, and the shared spatial (row+col) position
        + the per-stream-type marker + the signal marker are added so a signal
        frame rides the same positional basis as a camera/plan frame.  Returns
        ``None`` when the stream contributes no steps for this batch.
        """
        if tokens.numel() == 0 or tokens.ndim != 3 or tokens.shape[1] == 0:
            return None
        cfg = self.config
        b, p, c = tokens.shape
        s = cfg.n_spatial
        d = cfg.d_model
        c = min(c, int(self.signal_channel_embed[name].shape[0]))
        # clamp ids into the stream's vocab so an out-of-range id can never index
        # past the table (defensive — the dataset already clamps + masks).
        ids = tokens[:, :, :c].clamp_(0, vocab - 1)
        val = self.signal_embed[name](ids)  # (B, P, c, d)
        val = val + self.signal_channel_embed[name][:c].view(1, 1, c, d)
        if c < s:
            pad = val.new_zeros((b, p, s - c, d))
            val = torch.cat([val, pad], dim=2)
        rows = torch.arange(cfg.grid_h, device=tokens.device).repeat_interleave(
            cfg.grid_w
        )
        cols = torch.arange(cfg.grid_w, device=tokens.device).repeat(cfg.grid_h)
        spatial = self.row_embed(rows) + self.col_embed(cols)  # (S, d)
        val = (
            val
            + spatial.view(1, 1, s, d)
            + self.signal_type_embed[name]
            + self.signal_marker
        )
        return val  # (B, P, S, d)

    def _signal_zero_touch(self) -> torch.Tensor:
        """A zero-magnitude sum over every signal parameter (DDP-uniform graph).

        Mirrors the plan zero-touch in :meth:`SpacetimeTransformer._forward_tokens`:
        when a batch carries NO signals (every stream empty), the signal params
        would otherwise be grad-less and a DDP rank whose shard is signal-less
        would desync the ring.  Touching them with a ``*0.0`` contribution keeps
        every signal param in the autograd graph with zero effect on the output.
        """
        acc = self.signal_marker.sum()
        for emb in self.signal_embed.values():
            acc = acc + emb.weight.sum()
        for slot in self.signal_channel_embed.values():
            acc = acc + slot.sum()
        for marker in self.signal_type_embed.values():
            acc = acc + marker.sum()
        return acc * 0.0

    def _forward_tokens(
        self,
        frames: torch.Tensor,
        plan: torch.Tensor | None,
        signals: dict[str, torch.Tensor] | None = None,
        corruption_level: torch.Tensor | None = None,
        *,
        context_frames: int | None = None,
    ) -> torch.Tensor:
        """Run the backbone with plan + signal conditioning; return camera hidden.

        Builds ``[signal_frames | plan_frames | camera_frames]`` on the temporal
        axis (signals first, then plan, then the real frames — every conditioning
        frame sits before frame 0 so every camera token attends back to all of
        it), adds the absolute frame-position embedding over the full sequence,
        runs the factorised space-time blocks, and STRIPS every conditioning frame
        from the output (only camera frames are a prediction target).

        ``signals`` maps stream name -> ``(B, P, C)`` long local token ids; a
        stream absent for the batch is omitted.  When no signals are present the
        model still touches every signal param with a zero contribution so DDP's
        reducer sees a uniform parameter set on each rank.

        ``corruption_level`` is an optional ``(B,)`` long per-sample bin index
        into the corruption-level embedding; when the model is corruption-capable
        the level embedding is added to the CONTEXT camera frames (frames
        ``< context_frames``) so the model conditions on how corrupt its fed-back
        history is.  ``None`` defaults to bin 0 (the CLEAN rate-0 case the model
        saw on its clean-fraction training samples) — so an inference forward with
        no level supplied adds the level-0 embedding, matching how clean inputs
        were trained, rather than silently skipping the conditioning.
        """
        cfg = self.config
        cam = self._embed_camera(frames)  # (B, T, S, d)
        b, t, s, d = cam.shape

        # Anti-drift history-corruption conditioning: add the per-sample level
        # embedding to the CONTEXT frames (the ones the rollout re-feeds itself).
        if self.has_corruption:
            ctx = int(context_frames if context_frames is not None else t)
            ctx = max(0, min(ctx, t))
            if ctx > 0:
                if corruption_level is None:
                    # inference / clean default: bin 0 (the rate-0 row).
                    lvl = torch.zeros(b, dtype=torch.long, device=frames.device)
                else:
                    lvl = (
                        corruption_level.to(frames.device)
                        .long()
                        .clamp(0, int(cfg.corruption_levels) - 1)
                    )
                lvl_emb = self.corruption_embed(lvl).view(b, 1, 1, d)  # (B,1,1,d)
                cam[:, :ctx] = cam[:, :ctx] + lvl_emb
            else:
                # no context frames to condition — keep the embedding in the
                # autograd graph (DDP-uniform) with a zero-magnitude touch.
                cam = cam + self.corruption_embed.weight.sum() * 0.0

        prefix: list[torch.Tensor] = []
        # signals first (closest to the temporal front), then the plan.
        signal_frame_count = 0
        if self.has_signals and signals:
            vocab_by_name = {st.name: st.vocab for st in cfg.signal_streams}
            for stream in cfg.signal_streams:  # deterministic order
                tok = signals.get(stream.name)
                if tok is None:
                    continue
                emb = self._embed_signal_stream(
                    stream.name, tok, vocab_by_name[stream.name]
                )
                if emb is not None:
                    prefix.append(emb)
                    signal_frame_count += emb.shape[1]

        plan_emb = self._embed_plan(plan) if plan is not None else None
        if plan_emb is not None:
            prefix.append(plan_emb)

        x = torch.cat([*prefix, cam], dim=1) if prefix else cam  # (B, P+T, S, d)

        n_prefix = x.shape[1] - t

        # Keep plan + signal params in the autograd graph on a prefix-less / partial
        # batch (DDP uniformity).  The plan zero-touch is in the parent; replicate
        # the SAME guard here for the plan (the parent's _forward_tokens is not
        # used on this path), and add the signal zero-touch.
        zero = frames.new_zeros((), dtype=x.dtype)
        if self.has_plan and plan_emb is None:
            zero = (
                zero
                + (
                    self.plan_embed.weight.sum()
                    + self.plan_channel_embed.sum()
                    + self.plan_marker.sum()
                    + self.cam_marker.sum()
                )
                * 0.0
            )
        if self.has_signals and signal_frame_count == 0:
            zero = zero + self._signal_zero_touch()
        x = x + zero

        total_t = x.shape[1]
        if total_t > cfg.max_frames:
            raise ValueError(
                f"sequence has {total_t} frames (prefix {n_prefix} + cam {t}) > "
                f"max_frames {cfg.max_frames}"
            )
        fpos = torch.arange(total_t, device=frames.device)
        x = x + self.frame_embed(fpos).view(1, total_t, 1, d)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return x[:, n_prefix:]  # (B, T, S, d) — camera frames only

    # -- forward / loss ----------------------------------------------------

    def forward(
        self,
        batch: dict,
        *,
        return_logits: bool = False,
        loss_spec: dict | None = None,
    ) -> SpacetimeOutput | torch.Tensor:
        """Teacher-forced forward with plan + measured-signal conditioning.

        ``batch`` is ``{"frames": (B, T, S) long, "plan": (B, P, C) long,
        "signals": {name: (B, P_s, C_s) long}}``.  ``signals`` may be absent or
        empty (the model then conditions on the plan only, with the signal
        zero-touch keeping DDP uniform).  An optional ``corruption_level`` ``(B,)``
        long in the batch conditions the anti-drift history-corruption embedding;
        the loss context-frame count (``loss_spec["context_frames"]``) bounds which
        frames the level embedding is added to.  An optional ``target_frames``
        ``(B, T, S)`` long supplies the CLEAN frames the loss is scored against
        when ``frames`` carries a corrupted history — keeping the prediction
        target uncorrupted independently of the loss mask.  ``loss_spec`` /
        ``return_logits`` behave exactly as in v1.
        """
        frames = batch["frames"]
        plan = batch.get("plan")
        signals = batch.get("signals")
        corruption_level = batch.get("corruption_level")
        # the loss target is the CLEAN frames when a corrupted history is fed.
        target = batch.get("target_frames")
        if target is None:
            target = frames
        context_frames = (loss_spec or {}).get("context_frames")
        hidden = self._forward_tokens(
            frames,
            plan,
            signals,
            corruption_level,
            context_frames=context_frames,
        )  # (B, T, S, d)

        if loss_spec is not None:
            return self.chunked_nll(
                hidden,
                target,
                chunk=int(loss_spec.get("chunk", 4096)),
                context_frames=loss_spec.get("context_frames"),
            )
        if return_logits:
            logits = self.head(hidden)
            return SpacetimeOutput(hidden=hidden, logits=logits)
        return SpacetimeOutput(hidden=hidden, logits=None)


# Re-export the block so a reader sees the v2 module is the full model surface.
__all__ = [
    "SignalSpacetimeConfig",
    "SignalSpacetimeTransformer",
    "SignalStreamSpec",
    "_SpaceTimeBlock",
]
