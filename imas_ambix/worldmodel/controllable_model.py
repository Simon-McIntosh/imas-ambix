"""Actuator-PLAN-conditioned spatiotemporal camera transformer (the driveable model).

What this adds over :mod:`imas_ambix.worldmodel.spacetime_model_v2`
------------------------------------------------------------------
The v2 model conditions the camera prediction on the MEASURED diagnostic streams
(magnetics, interferometer, soft_x_rays, gas-puff flow, …).  Those measured
streams are mutually REDUNDANT — the realised plasma state is written into all of
them — so the controllability falsification (M3) showed they are at best weakly
load-bearing.  The model FORECASTS well but is not DRIVEABLE.

This model makes the DEMANDED actuator PLAN load-bearing.  The first attempt
PREPENDED the plan as conditioning tokens; the M4 gate showed the model IGNORED
it (true-vs-zeroed margin ~0).  The control-conditioning survey
(``docs/control-conditioning-survey.html``) traced that to three converging
causes and prescribes three matched fixes, applied here TOGETHER:

* **AdaLN-Zero per-layer plan conditioning** — the prepended plan tokens are
  REPLACED by per-block affine modulation.  A small MLP maps a pooled summary of
  the continuous actuator plan to per-transformer-block ``(gamma, beta, alpha)``
  for each of the block's three pre-norm sub-layers (spatial attn, temporal attn,
  MLP); the modulated sub-layer is ``alpha * sublayer((1+gamma)*LN(x) + beta)``,
  with ``alpha`` ZERO-INIT so the model starts as the unconditioned forecaster
  and the plan EARNS influence.  This is the DiT-ablation-best inject
  (AdaLN-Zero > cross-attn > tokens) and is structurally load-bearing: the plan
  modulates EVERY block rather than competing for attention at one injection
  point.

* **camera-history bottleneck** (:mod:`imas_ambix.worldmodel.history_bottleneck`)
  — independent per-frame corruption of the PAST FRAME EMBEDDINGS reaching the
  dynamics head, so the predictable history no longer suffices and the plan must
  carry what it cannot (breaks the latent-action collapse).  Wired into
  :meth:`_forward_tokens` and driven by the training loop.

* **inverse-dynamics auxiliary head** — predicts the continuous actuator plan
  from consecutive camera latents ``(z_t, z_{t+1})``.  We have 100% plan labels,
  so this forces the action into the latent (Schmidt & Jiang Prop 4.4) cheaply.
  Exposed via :meth:`inverse_dynamics_loss`; the training loop adds it to the
  objective.

The measured-signal streams + the tokenised pulse-schedule plan are inherited
from v2 and remain OPTIONAL context (high-dropout); only the actuator drive moves
from prepended tokens to AdaLN.  The factorised space-time attention, the chunked
NLL / argmax head, the next-frame factorisation, and the history-corruption
LEVEL embedding are inherited verbatim — but the transformer BLOCKS are rebuilt
as AdaLN-capable variants that reuse the v2 block's submodule weights by name (so
a v2 / forecaster checkpoint loads with ``strict=False``: the attention + MLP
submodules load, the new AdaLN MLP + inverse-dynamics head start at init, and the
zero-init gate makes that a no-op for the forecaster).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from imas_ambix.worldmodel.history_bottleneck import (
    HistoryBottleneckConfig,
    bottleneck_history_embeddings,
)
from imas_ambix.worldmodel.spacetime_model import _MHA
from imas_ambix.worldmodel.spacetime_model_v2 import (
    SignalSpacetimeConfig,
    SignalSpacetimeTransformer,
    SignalStreamSpec,
)


@dataclass
class ControllableSpacetimeConfig(SignalSpacetimeConfig):
    """:class:`SignalSpacetimeConfig` + the AdaLN-Zero actuator-plan drive surface.

    ``actuator_channels`` is the width of the continuous actuator vector (the
    drive surface).  When it is 0 the model is byte-equivalent to the v2
    signal-conditioned model (no actuator path, plain blocks), so a v2 checkpoint
    loads cleanly.  ``n_act_steps`` is the number of plan steps the actuator plan
    is sub-sampled to (the AdaLN summary pools over them).

    ``adaln_hidden`` is the hidden width of the plan-summary -> per-block
    modulation MLP.  ``inverse_dynamics`` enables the inverse-dynamics auxiliary
    head (predict the plan from consecutive latents); ``inv_dyn_hidden`` is its
    hidden width.

    Because the actuator plan is now injected via per-block AdaLN (NOT prepended
    as temporal frames), ``max_frames`` no longer needs to budget actuator frames
    — only the tokenised plan + signal steps + camera frames.
    """

    actuator_channels: int = 0
    n_act_steps: int = 8
    adaln_hidden: int = 256
    inverse_dynamics: bool = True
    inv_dyn_hidden: int = 256
    #: Column indices into the actuator vector to ZERO before conditioning — the
    #: measured STATES (plasma_current, ne_line_integrated) + the quasi-static
    #: tf_current.  These are NOT commands: an always-on Ip context is nearly a
    #: readout of the plasma state, which lets the model reproduce the camera by
    #: reading the state instead of deriving it from the commands (the
    #: persistence / ignore-the-actuators failure mode).  Masking them at the
    #: single conditioning entry point (:meth:`_plan_summary`) forces drive-from-
    #: commands and keeps train / gate / inference consistent.  Empty = condition
    #: on the full vector (states are kept — the v2-equivalent / debug path).  Ip
    #: + density remain available as the v2 OBSERVATION streams regardless.
    masked_command_indices: tuple[int, ...] = ()

    @property
    def has_actuator(self) -> bool:
        return int(self.actuator_channels) > 0


# ---------------------------------------------------------------------------
# AdaLN-Zero space-time block (reuses the v2 block's submodule weights by name)
# ---------------------------------------------------------------------------


class _AdaLNSpaceTimeBlock(nn.Module):
    """Factorised space-time block with per-sub-layer AdaLN-Zero modulation.

    Mirrors :class:`imas_ambix.worldmodel.spacetime_model._SpaceTimeBlock` — same
    submodule names (``ln_s``/``attn_s``, ``ln_t``/``attn_t``, ``ln_m``/``mlp``),
    so a v2 / forecaster checkpoint's ``blocks.N.*`` weights load by name — but its
    :meth:`forward` takes a per-block modulation tensor ``mod`` of shape
    ``(B, 9, d)`` carrying ``(gamma, beta, alpha)`` for each of the three pre-norm
    sub-layers, and applies

        h = h + alpha * sublayer((1 + gamma) * LN(h) + beta)

    The LayerNorms are made affine-free (``elementwise_affine=False``) because the
    AdaLN ``gamma``/``beta`` ARE the affine — a v2 checkpoint's ``ln_*.weight`` /
    ``ln_*.bias`` are therefore intentionally NOT loaded into these blocks (they
    fold into the data-independent component of the AdaLN MLP at init via the
    ``1 +`` identity).  When the modulation is the AdaLN-Zero identity
    (gamma=beta=0, alpha=1) the block reduces EXACTLY to a standard pre-norm
    residual block, so a zero-init plan gate starts as the plain forecaster.
    """

    def __init__(self, cfg: ControllableSpacetimeConfig) -> None:
        super().__init__()
        d = cfg.d_model
        # affine-free norms — the AdaLN gamma/beta supply the affine per sub-layer.
        self.ln_s = nn.LayerNorm(d, elementwise_affine=False)
        self.attn_s = _MHA(d, cfg.n_heads, cfg.dropout)
        self.ln_t = nn.LayerNorm(d, elementwise_affine=False)
        self.attn_t = _MHA(d, cfg.n_heads, cfg.dropout)
        self.ln_m = nn.LayerNorm(d, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(d, cfg.d_ff),
            nn.GELU(),
            nn.Linear(cfg.d_ff, d),
            nn.Dropout(cfg.dropout),
        )

    @staticmethod
    def _modulate(
        x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor
    ) -> torch.Tensor:
        # x: (B, T, S, d); gamma/beta: (B, d) -> broadcast over (T, S).
        return x * (1.0 + gamma[:, None, None, :]) + beta[:, None, None, :]

    def forward(self, x: torch.Tensor, mod: torch.Tensor | None = None) -> torch.Tensor:
        b, t, s, d = x.shape
        if mod is None:
            # plain pre-norm residual block (AdaLN identity): used when no plan.
            xs = self.ln_s(x).reshape(b * t, s, d)
            xs = self.attn_s(xs, is_causal=False).reshape(b, t, s, d)
            x = x + xs
            xt = self.ln_t(x).permute(0, 2, 1, 3).reshape(b * s, t, d)
            xt = self.attn_t(xt, is_causal=True).reshape(b, s, t, d).permute(0, 2, 1, 3)
            x = x + xt
            x = x + self.mlp(self.ln_m(x))
            return x
        # mod: (B, 9, d) -> three (gamma, beta, alpha) triples.
        g_s, b_s, a_s, g_t, b_t, a_t, g_m, b_m, a_m = mod.unbind(dim=1)
        # ── spatial: full attention within each frame ──
        h = self._modulate(self.ln_s(x), g_s, b_s).reshape(b * t, s, d)
        h = self.attn_s(h, is_causal=False).reshape(b, t, s, d)
        x = x + a_s[:, None, None, :] * h
        # ── temporal: causal attention across frames at each spatial position ──
        h = (
            self._modulate(self.ln_t(x), g_t, b_t)
            .permute(0, 2, 1, 3)
            .reshape(b * s, t, d)
        )
        h = self.attn_t(h, is_causal=True).reshape(b, s, t, d).permute(0, 2, 1, 3)
        x = x + a_t[:, None, None, :] * h
        # ── MLP ──
        h = self.mlp(self._modulate(self.ln_m(x), g_m, b_m))
        x = x + a_m[:, None, None, :] * h
        return x


# ---------------------------------------------------------------------------
# The driveable model
# ---------------------------------------------------------------------------


class ControllableSpacetimeTransformer(SignalSpacetimeTransformer):
    """v2 backbone + AdaLN-Zero actuator-plan conditioning + inverse-dynamics aux.

    The tokenised-plan conditioning, the measured-signal conditioning, the chunked
    head, the next-frame factorisation, and the history-corruption LEVEL embedding
    are inherited from :class:`SignalSpacetimeTransformer`.  This subclass:

    * REBUILDS the transformer blocks as :class:`_AdaLNSpaceTimeBlock` (same
      submodule names, so v2 weights load by name) and
    * adds a continuous-actuator-plan ENCODER + a plan-summary -> per-block
      ``(gamma, beta, alpha)`` AdaLN MLP (alpha zero-init), and
    * adds an inverse-dynamics head predicting the plan from consecutive latents,
    * and bottlenecks the camera-history embeddings in :meth:`_forward_tokens`.

    The actuator plan is NO LONGER prepended as temporal frames — it modulates
    every block — so the prefix is just the v2 tokenised-plan + signal frames.
    """

    config: ControllableSpacetimeConfig

    def __init__(self, cfg: ControllableSpacetimeConfig) -> None:
        # Build the v2 parameters (token / spatial / temporal / plan / signal
        # embeddings, plain blocks, head, weight tying) via the parent, then swap
        # the blocks for AdaLN-capable ones and add the actuator path.
        super().__init__(cfg)
        self.config = cfg  # narrow the type for the actuator-aware paths

        d = cfg.d_model
        # Rebuild the blocks as AdaLN-capable variants.  Their submodule names
        # match the v2 block so a v2 checkpoint's blocks.N.{attn_s,attn_t,mlp}
        # weights load by name (strict=False); the affine-free norms drop the v2
        # ln_*.weight/bias (folded into the AdaLN identity).
        self.blocks = nn.ModuleList(
            [_AdaLNSpaceTimeBlock(cfg) for _ in range(cfg.n_layers)]
        )
        for blk in self.blocks:
            blk.apply(self._init_weights)

        self.has_actuator = cfg.has_actuator
        self.has_inverse_dynamics = bool(cfg.inverse_dynamics and cfg.has_actuator)
        if self.has_actuator:
            c_act = int(cfg.actuator_channels)
            # Plan-summary encoder: the per-step [values | missing] vector (2*C)
            # is projected and POOLED over the n_act_steps into a single summary
            # vector that conditions every block.  A small MLP then produces the
            # per-block, per-sub-layer (gamma, beta, alpha).  The summary pools the
            # whole plan (mean over steps) so a change ANYWHERE in the plan moves
            # the conditioning; AdaLN does not need per-step temporal placement
            # (the temporal dynamics live in the camera frames it modulates).
            self.actuator_in = nn.Linear(2 * c_act, cfg.adaln_hidden)
            n_blocks = cfg.n_layers
            # one shared trunk + a per-block (gamma, beta, alpha) x 3 sub-layers.
            self.adaln_trunk = nn.Sequential(
                nn.SiLU(),
                nn.Linear(cfg.adaln_hidden, cfg.adaln_hidden),
                nn.SiLU(),
            )
            # gamma/beta head (data-dependent affine) and a SEPARATE alpha (gate)
            # head that is ZERO-INIT so every block starts as the AdaLN identity
            # (the plain forecaster) and the plan earns influence.
            self.adaln_gammabeta = nn.Linear(cfg.adaln_hidden, n_blocks * 6 * d)
            self.adaln_alpha = nn.Linear(cfg.adaln_hidden, n_blocks * 3 * d)
            nn.init.normal_(self.actuator_in.weight, std=0.02)
            nn.init.zeros_(self.actuator_in.bias)
            for m in self.adaln_trunk:
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.02)
                    nn.init.zeros_(m.bias)
            # gamma/beta start at 0 (identity affine: (1+0)*x + 0 = x).
            nn.init.zeros_(self.adaln_gammabeta.weight)
            nn.init.zeros_(self.adaln_gammabeta.bias)
            # alpha (gate) starts at 0 — zero residual contribution => identity.
            nn.init.zeros_(self.adaln_alpha.weight)
            nn.init.zeros_(self.adaln_alpha.bias)

        if self.has_inverse_dynamics:
            c_act = int(cfg.actuator_channels)
            # Inverse dynamics: predict the (mean over steps) continuous plan from
            # a pooled pair of consecutive camera latents (z_t, z_{t+1}).  A
            # regression head (the plan is continuous, normalised).  Pools each
            # latent over the spatial axis (mean) before concatenation.
            self.inv_dyn = nn.Sequential(
                nn.Linear(2 * d, cfg.inv_dyn_hidden),
                nn.SiLU(),
                nn.Linear(cfg.inv_dyn_hidden, cfg.inv_dyn_hidden),
                nn.SiLU(),
                nn.Linear(cfg.inv_dyn_hidden, c_act),
            )
            for m in self.inv_dyn:
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.02)
                    nn.init.zeros_(m.bias)

    # -- actuator-plan AdaLN conditioning ----------------------------------

    def _plan_summary(
        self, values: torch.Tensor, missing: torch.Tensor | None
    ) -> torch.Tensor | None:
        """``(B, P, C) + (B, P, C) -> (B, adaln_hidden)`` pooled plan summary.

        Concatenates ``[values | missing]`` per step, projects, and mean-pools
        over the plan steps into one summary vector per sample.  Returns ``None``
        when the plan contributes no steps (the model then runs unconditioned).

        The measured STATE columns (``masked_command_indices`` — Ip, density, tf)
        are ZEROED in BOTH ``values`` and ``missing`` here, the single conditioning
        entry point, so the model conditions ONLY on the commands (drive-from-
        commands; it cannot read the plasma state off an always-on Ip context).
        Zeroing ``missing`` too (to PRESENT-but-zero rather than absent) keeps the
        per-column contribution a constant the encoder can absorb, so the masked
        columns add no information regardless of the raw plan.
        """
        if not self.has_actuator or values is None or values.numel() == 0:
            return None
        if values.ndim != 3 or values.shape[1] == 0:
            return None
        b, p, c = values.shape
        miss = missing
        if miss is None or miss.shape != values.shape:
            miss = torch.zeros_like(values)
        mask_idx = getattr(self.config, "masked_command_indices", ()) or ()
        if mask_idx:
            cols = [int(i) for i in mask_idx if 0 <= int(i) < c]
            if cols:
                values = values.clone()
                miss = miss.clone()
                values[:, :, cols] = 0.0
                miss[:, :, cols] = 0.0
        feat = torch.cat([values, miss], dim=-1).to(self.actuator_in.weight.dtype)
        proj = self.actuator_in(feat)  # (B, P, adaln_hidden)
        return proj.mean(dim=1)  # (B, adaln_hidden) — pool over plan steps

    def _block_modulations(self, summary: torch.Tensor) -> torch.Tensor:
        """``(B, adaln_hidden) -> (B, n_blocks, 9, d)`` per-block AdaLN params.

        The 9 = 3 sub-layers x (gamma, beta, alpha).  ``gamma``/``beta`` come from
        the (zero-init) gamma/beta head; ``alpha`` from the SEPARATE zero-init gate
        head, interleaved as ``(g_s, b_s, a_s, g_t, b_t, a_t, g_m, b_m, a_m)`` so
        the block can ``unbind`` them in order.
        """
        cfg = self.config
        d = cfg.d_model
        nb = cfg.n_layers
        h = self.adaln_trunk(summary)  # (B, adaln_hidden)
        gb = self.adaln_gammabeta(h).view(
            -1, nb, 3, 2, d
        )  # (B, nb, sublayer, {g,b}, d)
        al = self.adaln_alpha(h).view(-1, nb, 3, 1, d)  # (B, nb, sublayer, {a}, d)
        # concat to (B, nb, sublayer, 3, d) ordered (gamma, beta, alpha) then
        # flatten the (sublayer, 3) axes to the 9 the block unbinds.
        mod = torch.cat([gb, al], dim=3)  # (B, nb, 3, 3, d)
        return mod.reshape(-1, nb, 9, d)

    def _actuator_zero_touch(self) -> torch.Tensor:
        """A zero-magnitude sum over every actuator/AdaLN param (DDP-uniform graph).

        A batch with no actuator plan would leave the actuator-encoder + AdaLN MLP
        params grad-less and desync a DDP rank.  Touching them with a ``*0.0``
        contribution keeps every actuator param in the autograd graph with zero
        effect on the output.
        """
        acc = (
            self.actuator_in.weight.sum()
            + self.actuator_in.bias.sum()
            + self.adaln_gammabeta.weight.sum()
            + self.adaln_gammabeta.bias.sum()
            + self.adaln_alpha.weight.sum()
            + self.adaln_alpha.bias.sum()
        )
        for m in self.adaln_trunk:
            if isinstance(m, nn.Linear):
                acc = acc + m.weight.sum() + m.bias.sum()
        if self.has_inverse_dynamics:
            for m in self.inv_dyn:
                if isinstance(m, nn.Linear):
                    acc = acc + m.weight.sum() + m.bias.sum()
        return acc * 0.0

    # -- forward backbone --------------------------------------------------

    def _forward_tokens(
        self,
        frames: torch.Tensor,
        plan: torch.Tensor | None,
        signals: dict[str, torch.Tensor] | None = None,
        corruption_level: torch.Tensor | None = None,
        *,
        actuator: dict[str, torch.Tensor] | None = None,
        context_frames: int | None = None,
        history_bottleneck: HistoryBottleneckConfig | None = None,
        history_strengths: torch.Tensor | None = None,
        history_generator: torch.Generator | None = None,
        return_latents: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Run the backbone with AdaLN actuator conditioning + a history bottleneck.

        Builds ``[signals | plan | camera]`` on the temporal axis (the v2 prefix —
        the actuator plan is NO LONGER a prefix; it modulates every block via
        AdaLN), optionally bottlenecks the camera-history EMBEDDINGS, adds the
        absolute frame-position embedding, runs the AdaLN space-time blocks
        modulated by the actuator-plan summary, and STRIPS every conditioning
        frame from the output.

        ``actuator`` is ``{"values": (B, P, C) float, "missing": (B, P, C)
        float}`` (the demanded plan); ``None`` / empty omits it (the model then
        runs the unconditioned forecaster, with the actuator zero-touch keeping
        DDP uniform).  ``history_bottleneck`` + ``history_strengths`` corrupt the
        context-frame embeddings (the corrected controllability lever); when
        ``history_strengths`` is None the bottleneck is skipped (clean history —
        the inference / eval default).

        With ``return_latents`` the camera latents are also returned (for the
        inverse-dynamics auxiliary).
        """
        cfg = self.config
        cam = self._embed_camera(frames)  # (B, T, S, d)
        b, t, s, d = cam.shape

        # Camera-HISTORY bottleneck — the corrected controllability lever.  Corrupt
        # the context-frame EMBEDDINGS (independent per frame) BEFORE the level
        # embedding + the blocks so the predictable history no longer suffices.
        if (
            history_bottleneck is not None
            and history_strengths is not None
            and history_bottleneck.enabled
        ):
            ctx_b = int(context_frames if context_frames is not None else t)
            cam = bottleneck_history_embeddings(
                cam,
                history_strengths,
                history_bottleneck,
                context_frames=ctx_b,
                generator=history_generator,
            )

        # Anti-drift history-corruption LEVEL conditioning (inherited semantics):
        # add the per-sample level embedding to the CONTEXT frames so the model
        # knows how corrupt its history is.  Distinct from the embedding
        # bottleneck above — this is the learned scalar the M2 recipe conditions
        # on; here it can also carry the bottleneck strength bin.
        if self.has_corruption:
            ctx = int(context_frames if context_frames is not None else t)
            ctx = max(0, min(ctx, t))
            if ctx > 0:
                if corruption_level is None:
                    lvl = torch.zeros(b, dtype=torch.long, device=frames.device)
                else:
                    lvl = (
                        corruption_level.to(frames.device)
                        .long()
                        .clamp(0, int(cfg.corruption_levels) - 1)
                    )
                lvl_emb = self.corruption_embed(lvl).view(b, 1, 1, d)
                cam[:, :ctx] = cam[:, :ctx] + lvl_emb
            else:
                cam = cam + self.corruption_embed.weight.sum() * 0.0

        # v2 prefix: measured signals first, then the tokenised pulse-schedule plan
        # (the actuator plan is NOT prefixed — it modulates the blocks).
        prefix: list[torch.Tensor] = []
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

        # actuator-plan AdaLN modulation (None when no plan -> plain blocks).
        act_vals = actuator.get("values") if actuator else None
        act_miss = actuator.get("missing") if actuator else None
        summary = self._plan_summary(act_vals, act_miss) if self.has_actuator else None
        block_mods = self._block_modulations(summary) if summary is not None else None

        # DDP uniformity: keep every conditioning param in the autograd graph on a
        # partial batch (plan + signal zero-touch inherited + the actuator/AdaLN
        # zero-touch).
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
        if self.has_actuator and summary is None:
            zero = zero + self._actuator_zero_touch()
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
        for li, block in enumerate(self.blocks):
            mod = block_mods[:, li] if block_mods is not None else None
            x = block(x, mod)
        x = self.ln_f(x)
        latents = x[:, n_prefix:]  # (B, T, S, d) — camera frames only
        if return_latents:
            return latents, latents
        return latents

    # -- inverse-dynamics auxiliary ----------------------------------------

    def inverse_dynamics_loss(
        self,
        latents: torch.Tensor,
        actuator: dict[str, torch.Tensor],
        *,
        context_frames: int | None = None,
    ) -> torch.Tensor:
        """Predict the (pooled) actuator plan from consecutive camera latents.

        ``latents`` is ``(B, T, S, d)`` camera hiddens.  For each consecutive pair
        ``(z_t, z_{t+1})`` (spatially mean-pooled) the inverse-dynamics head
        regresses the per-sample mean (over plan steps) NORMALISED actuator vector;
        the loss is the MSE over present (non-missing) channels, averaged over the
        forecast-window pairs.  Forces the action into the latent (Schmidt & Jiang
        Prop 4.4) — we have 100% plan labels.

        Returns a scalar.  When the head is disabled / no plan, returns a
        zero-magnitude touch over the head params (DDP-uniform) so it never
        desyncs a rank.
        """
        if not self.has_inverse_dynamics:
            return latents.new_zeros(())
        vals = actuator.get("values") if actuator else None
        miss = actuator.get("missing") if actuator else None
        if vals is None or vals.numel() == 0 or vals.ndim != 3:
            # keep the head in the graph with a zero touch.
            acc = latents.new_zeros((), dtype=latents.dtype)
            for m in self.inv_dyn:
                if isinstance(m, nn.Linear):
                    acc = acc + m.weight.sum() + m.bias.sum()
            return acc * 0.0
        b, t, s, d = latents.shape
        if t < 2:
            return latents.new_zeros(())
        pooled = latents.mean(dim=2)  # (B, T, d) spatial mean-pool
        z_t = pooled[:, : t - 1]  # (B, T-1, d)
        z_n = pooled[:, 1:t]
        pair = torch.cat([z_t, z_n], dim=-1)  # (B, T-1, 2d)
        # restrict to forecast-window pairs (target frame index t+1 >= ctx) when
        # given — the inverse map is only meaningful where the plan drives a
        # transition the model is being asked to forecast.
        if context_frames is not None:
            tgt_idx = torch.arange(1, t, device=latents.device)
            keep = tgt_idx >= int(context_frames)
            if bool(keep.any()):
                pair = pair[:, keep]
        pred = self.inv_dyn(pair)  # (B, K, C)
        # target: the per-sample mean (over plan steps) normalised actuator vector,
        # broadcast over the K pairs.  present-channel mask from missing.
        tgt_vec = vals.to(pred.dtype).mean(dim=1)  # (B, C)
        present = (
            (miss.to(pred.dtype).mean(dim=1) < 1.0).to(pred.dtype)
            if miss is not None
            else torch.ones_like(tgt_vec)
        )
        # also drop the MASKED (non-command) columns from the target: the inverse
        # map should force the COMMANDS into the latent, not the masked states
        # (Ip/density/tf) which the model never conditions on.
        mask_idx = getattr(self.config, "masked_command_indices", ()) or ()
        if mask_idx:
            cols = [int(i) for i in mask_idx if 0 <= int(i) < present.shape[-1]]
            if cols:
                present = present.clone()
                present[:, cols] = 0.0
        tgt = tgt_vec[:, None, :].expand_as(pred)
        wmask = present[:, None, :].expand_as(pred)
        sq = (pred - tgt) ** 2 * wmask
        denom = wmask.sum().clamp_min(1.0)
        return sq.sum() / denom

    # -- forward / loss ----------------------------------------------------

    def forward(
        self,
        batch: dict,
        *,
        return_logits: bool = False,
        loss_spec: dict | None = None,
    ):
        """Teacher-forced forward with AdaLN actuator + plan + measured-signal cond.

        ``batch`` is ``{"frames": (B, T, S) long, "plan": (B, P, C) long,
        "signals": {name: (B, P_s, C_s) long}, "actuator": {"values": (B, P_a,
        C_a) float, "missing": (B, P_a, C_a) float}}``.  An optional
        ``corruption_level`` ``(B,)`` long, ``target_frames`` ``(B, T, S)`` long,
        ``history_bottleneck`` (:class:`HistoryBottleneckConfig`),
        ``history_strengths`` ``(B, ctx)`` float, and ``history_generator`` drive
        the camera-history bottleneck.  ``loss_spec`` may carry
        ``inverse_dynamics_weight`` to add the inverse-dynamics auxiliary loss to
        the returned scalar (training).  ``return_logits`` behaves as in v1.
        """
        frames = batch["frames"]
        plan = batch.get("plan")
        signals = batch.get("signals")
        actuator = batch.get("actuator")
        corruption_level = batch.get("corruption_level")
        target = batch.get("target_frames")
        if target is None:
            target = frames
        context_frames = (loss_spec or {}).get("context_frames")
        hb = batch.get("history_bottleneck")
        hs = batch.get("history_strengths")
        hg = batch.get("history_generator")
        need_latents = bool(
            loss_spec is not None
            and self.has_inverse_dynamics
            and float((loss_spec or {}).get("inverse_dynamics_weight", 0.0)) > 0.0
        )
        out = self._forward_tokens(
            frames,
            plan,
            signals,
            corruption_level,
            actuator=actuator,
            context_frames=context_frames,
            history_bottleneck=hb,
            history_strengths=hs,
            history_generator=hg,
            return_latents=need_latents,
        )
        if need_latents:
            hidden, latents = out
        else:
            hidden, latents = out, out

        if loss_spec is not None:
            nll = self.chunked_nll(
                hidden,
                target,
                chunk=int(loss_spec.get("chunk", 4096)),
                context_frames=loss_spec.get("context_frames"),
            )
            w_inv = float(loss_spec.get("inverse_dynamics_weight", 0.0))
            if self.has_inverse_dynamics and w_inv > 0.0 and actuator is not None:
                inv = self.inverse_dynamics_loss(
                    latents, actuator, context_frames=context_frames
                )
                return nll + w_inv * inv
            return nll
        from imas_ambix.worldmodel.spacetime_model import (
            SpacetimeOutput,  # noqa: PLC0415
        )

        if return_logits:
            logits = self.head(hidden)
            return SpacetimeOutput(hidden=hidden, logits=logits)
        return SpacetimeOutput(hidden=hidden, logits=None)


__all__ = [
    "ControllableSpacetimeConfig",
    "ControllableSpacetimeTransformer",
    "SignalStreamSpec",
]
