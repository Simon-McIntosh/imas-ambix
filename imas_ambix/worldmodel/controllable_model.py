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
from imas_ambix.worldmodel.timescale_conditioning import (
    CAMERA_IDS,
    REFERENCE_CAMERA_INDEX,
    TimescaleEncoder,
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
    #: Per-frame timescale (Δt) conditioning.  When True the model is told each
    #: camera frame's inter-frame interval (log-Δt) so the SAME token sequence at
    #: a slow (~6 ms, coil/position regime) vs fast (~50 µs, MHD regime) cadence
    #: is interpreted differently — a log-Δt scalar → MLP → added to the per-frame
    #: temporal embedding (:class:`imas_ambix.worldmodel.timescale_conditioning.
    #: TimescaleEncoder`, zero-init output → identity at init).  False (default)
    #: is byte-identical to the cadence-blind model; a checkpoint without the Δt
    #: head loads cleanly (the head stays at its fresh zero init).
    timescale_conditioning: bool = False
    timescale_hidden: int = 64
    #: Per-camera (view) conditioning.  When True a learned per-camera embedding
    #: (rbb/rco/rgb/rgc/rba/rbc — :data:`imas_ambix.worldmodel.
    #: timescale_conditioning.CAMERA_IDS`) is ADDED to the camera-frame token
    #: embeddings so the model knows WHICH view (FOV / optics / colour) it is
    #: predicting; the table is zero-init (identity at init) and an unknown /
    #: missing camera falls back to the reference camera (``rbb``).  False
    #: (default) is byte-identical; a prior checkpoint loads cleanly.
    camera_conditioning: bool = False
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
    #: Joint generation: when True (and the model has measured-signal streams) the
    #: model grows a per-stream PREDICTION head over each stream's vocab and learns
    #: to forecast the NEXT-step diagnostic tokens (a cross-entropy on the same
    #: tokens it conditions on), so it dreams the cameras AND the diagnostics — a
    #: joint state model, not a camera predictor that merely reads the diagnostics.
    #: The heads are built LAST in ``__init__`` so the backbone RNG stream is
    #: unperturbed: with diagnostics OFF, or warm-started from a camera-only
    #: checkpoint, the forecaster is byte-identical (the heads are simply new /
    #: absent).  The diagnostic loss weight is a TRAINING hyperparameter (passed
    #: via ``loss_spec["diagnostic_weight"]``, mirroring ``inverse_dynamics_weight``)
    #: — this flag only decides whether the heads + objective EXIST.  OFF is the
    #: ablation switch (no heads built, no objective).
    generate_diagnostics: bool = True

    @property
    def has_actuator(self) -> bool:
        return int(self.actuator_channels) > 0

    @property
    def has_timescale(self) -> bool:
        return bool(self.timescale_conditioning)

    @property
    def has_camera(self) -> bool:
        return bool(self.camera_conditioning)

    @property
    def has_diagnostics(self) -> bool:
        """The model generates diagnostics (heads + objective) iff ON and signalled."""
        return bool(self.generate_diagnostics) and self.has_signals


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

        # Per-frame timescale (Δt) head — log-Δt scalar → d_model offset added to
        # the camera frames' temporal embedding.  Zero-init output (inside
        # TimescaleEncoder) → identity at init, so an OFF / fresh head is a no-op
        # and a prior checkpoint loads cleanly (the head is simply new).
        self.has_timescale = cfg.has_timescale
        if self.has_timescale:
            self.timescale_encoder = TimescaleEncoder(d, hidden=cfg.timescale_hidden)

        # Per-camera (view) embedding — added to the camera-frame token embeddings
        # so the model knows which view it predicts.  Zero-init → identity at init.
        self.has_camera = cfg.has_camera
        if self.has_camera:
            self.camera_embed = nn.Embedding(len(CAMERA_IDS), d)
            nn.init.zeros_(self.camera_embed.weight)

        # Joint generation: per-stream diagnostic-prediction heads.  Built LAST so
        # every backbone parameter above was drawn from the SAME RNG stream a
        # diagnostics-OFF model uses — a model warm-started from a camera-only
        # checkpoint loads the backbone byte-for-byte and only these heads start
        # fresh.  Each head maps a signal-frame hidden (d) to that stream's own
        # local vocabulary, decoding the stream's NEXT-step tokens (the heads are
        # NOT weight-tied across streams — the per-group-local ids are meaningless
        # across streams, exactly as the per-stream value-embeddings are separate).
        self.has_diagnostics = cfg.has_diagnostics
        if self.has_diagnostics:
            self.diagnostic_heads = nn.ModuleDict()
            for stream in cfg.signal_streams:
                head = nn.Linear(d, int(stream.vocab))
                nn.init.normal_(head.weight, std=0.02)
                nn.init.zeros_(head.bias)
                self.diagnostic_heads[stream.name] = head

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
        frame_log_dt: torch.Tensor | None = None,
        camera_id: torch.Tensor | None = None,
        return_latents: bool = False,
        return_signal_latents: bool = False,
    ) -> (
        torch.Tensor
        | tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, dict[str, torch.Tensor]]
        | tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]
    ):
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

        ``frame_log_dt`` is an optional ``(B, T)`` per-camera-frame log-Δt offset
        (centred on the reference cadence) — when the model is timescale-capable
        it is encoded and ADDED to the camera frames' temporal embedding so the
        same token sequence at a different cadence is interpreted differently;
        ``None`` falls back to the reference cadence (offset 0).  ``camera_id`` is
        an optional ``(B,)`` long per-sample camera index — when the model is
        camera-capable the matching learned view embedding is added to the camera
        token embeddings; ``None`` falls back to the reference camera (``rbb``).
        """
        cfg = self.config
        cam = self._embed_camera(frames)  # (B, T, S, d)
        b, t, s, d = cam.shape

        # Per-camera (view) conditioning: add the learned per-camera embedding to
        # the camera token embeddings so the model knows which view it predicts.
        # Zero-init table → identity at init; None / unknown → reference camera.
        if self.has_camera:
            if camera_id is None:
                cam_idx = torch.full(
                    (b,), REFERENCE_CAMERA_INDEX, dtype=torch.long, device=frames.device
                )
            else:
                cam_idx = (
                    camera_id.to(frames.device)
                    .long()
                    .clamp(0, len(CAMERA_IDS) - 1)
                )
            cam = cam + self.camera_embed(cam_idx).view(b, 1, 1, d)

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
        # Track each present stream's slice in the (signals-first) prefix so the
        # diagnostic heads can read back its post-transformer hiddens: (name,
        # offset, n_steps, n_channels).  Signals are prepended BEFORE the plan, so
        # a stream's offset is its running signal-frame count (independent of the
        # plan that follows).
        sig_layout: list[tuple[str, int, int, int]] = []
        if self.has_signals and signals:
            vocab_by_name = {st.name: st.vocab for st in cfg.signal_streams}
            chan_by_name = {st.name: st.channels for st in cfg.signal_streams}
            for stream in cfg.signal_streams:  # deterministic order
                tok = signals.get(stream.name)
                if tok is None:
                    continue
                emb = self._embed_signal_stream(
                    stream.name, tok, vocab_by_name[stream.name]
                )
                if emb is not None:
                    c_used = min(int(tok.shape[2]), int(chan_by_name[stream.name]))
                    sig_layout.append(
                        (stream.name, signal_frame_count, int(emb.shape[1]), c_used)
                    )
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

        # Per-frame timescale (Δt) conditioning: encode each CAMERA frame's
        # log-Δt offset and ADD it to that frame's temporal embedding, so the same
        # token sequence at a different cadence is interpreted differently.  Only
        # the camera frames (the trailing t) carry a cadence — the conditioning
        # prefix frames get the reference (zero) offset.  Always run when the model
        # is timescale-capable (reference offset when none supplied) so the encoder
        # stays in the autograd graph every step (DDP-uniform), zero-init → no-op.
        if self.has_timescale:
            if frame_log_dt is None:
                dt = frames.new_zeros((b, t), dtype=torch.float32)
            else:
                dt = frame_log_dt.to(frames.device, torch.float32)
                if dt.ndim == 1:
                    dt = dt.unsqueeze(0).expand(b, -1)
                # align to the camera-frame count: pad/truncate to t.
                if dt.shape[1] != t:
                    fixed = frames.new_zeros((b, t), dtype=torch.float32)
                    k = min(dt.shape[1], t)
                    fixed[:, :k] = dt[:, :k]
                    dt = fixed
            dt_off = self.timescale_encoder(dt)  # (B, t, d)
            x = torch.cat(
                [x[:, :n_prefix], x[:, n_prefix:] + dt_off.unsqueeze(2)], dim=1
            )

        x = self.drop(x)
        for li, block in enumerate(self.blocks):
            mod = block_mods[:, li] if block_mods is not None else None
            x = block(x, mod)
        x = self.ln_f(x)
        latents = x[:, n_prefix:]  # (B, T, S, d) — camera frames only
        if return_signal_latents:
            # Per-stream signal-frame hiddens for the diagnostic heads.  A stream's
            # C channels occupy spatial lanes [0:C]; the signal frames sit at the
            # FRONT of the sequence (offset within [0, signal_frame_count)).
            signal_latents = {
                name: x[:, off : off + p, :c, :] for (name, off, p, c) in sig_layout
            }
            if return_latents:
                return latents, latents, signal_latents
            return latents, signal_latents
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

    # -- joint generation: diagnostic-prediction heads ---------------------

    def _diagnostic_zero_touch(self) -> torch.Tensor:
        """A zero-magnitude sum over every diagnostic-head param (DDP-uniform graph).

        A batch where a stream is absent (or where the diagnostic weight is 0)
        would leave that head grad-less and desync a DDP rank.  Touching every
        head with a ``*0.0`` contribution keeps them all in the autograd graph with
        zero effect on the output — mirrors :meth:`_actuator_zero_touch`.
        """
        acc: torch.Tensor | None = None
        for head in self.diagnostic_heads.values():
            s = head.weight.sum() + head.bias.sum()
            acc = s if acc is None else acc + s
        if acc is None:
            return next(self.parameters()).new_zeros(())
        return acc * 0.0

    def diagnostic_logits(
        self, signal_latents: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """``{name: (B, P, C, d)} -> {name: (B, P, C, vocab)}`` per-stream logits.

        Decodes the post-transformer signal-frame hiddens to next-step token
        logits with each stream's own head.  A stream with no head (not generated)
        is skipped.  Used by the eval's dreamt-vs-real diagnostic-match metric.
        """
        out: dict[str, torch.Tensor] = {}
        if not self.has_diagnostics:
            return out
        # nn.ModuleDict has no .get() — guard with membership + indexing.
        for name, lat in signal_latents.items():
            if name not in self.diagnostic_heads:
                continue
            out[name] = self.diagnostic_heads[name](lat)
        return out

    def diagnostic_loss(
        self,
        signal_latents: dict[str, torch.Tensor],
        signals: dict[str, torch.Tensor] | None,
    ) -> torch.Tensor:
        """Next-step cross-entropy on the measured-signal tokens (joint generation).

        ``signal_latents`` is ``{name: (B, P, C, d)}`` (post-transformer signal-frame
        hiddens, from :meth:`_forward_tokens` with ``return_signal_latents``);
        ``signals`` is ``{name: (B, P, C) long}`` the REAL local token ids — which
        are ALSO the target: signal frame ``j`` predicts frame ``j+1``.  For each
        stream the head maps ``d -> vocab`` and the loss is the CE over present
        (non-PAD) next-step targets, summed across streams and averaged over the
        scored positions.  PAD (id 0 = an absent/sub-sampled-empty step) is the
        ``ignore_index`` so an absent reading is never a target.

        Returns a scalar.  When NO stream contributes a scored position (all PAD /
        signal-less batch) it returns a zero-magnitude touch over every head param
        (DDP-uniform) so a rank never desyncs.
        """
        from torch.nn import functional as F  # noqa: PLC0415

        from imas_ambix.worldmodel.dataset import PAD_LOCAL_ID  # noqa: PLC0415

        if not self.has_diagnostics:
            return next(self.parameters()).new_zeros(())
        pad = int(PAD_LOCAL_ID)
        total_ce: torch.Tensor | None = None
        total_count = 0
        for name in self.diagnostic_heads:
            lat = signal_latents.get(name) if signal_latents else None
            tok = signals.get(name) if signals else None
            if lat is None or tok is None:
                continue
            p = min(int(lat.shape[1]), int(tok.shape[1]))
            c = min(int(lat.shape[2]), int(tok.shape[2]))
            if p < 2 or c < 1:
                continue
            logits = self.diagnostic_heads[name](lat[:, :p, :c, :])  # (B, p, c, V)
            pred = logits[:, : p - 1].reshape(-1, logits.shape[-1])  # frames 0..p-2
            target = tok[:, 1:p, :c].reshape(-1).to(pred.device).long()  # next step
            ce = F.cross_entropy(pred, target, ignore_index=pad, reduction="sum")
            cnt = int((target != pad).sum())
            if cnt > 0:
                total_ce = ce if total_ce is None else total_ce + ce
                total_count += cnt
        zt = self._diagnostic_zero_touch()
        if total_ce is None or total_count == 0:
            return zt
        return total_ce / float(total_count) + zt

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
        the camera-history bottleneck.  An optional ``frame_log_dt`` ``(B, T)``
        float (per-camera-frame log-Δt offset) and ``camera_id`` ``(B,)`` long
        drive the timescale + camera conditioning when the model is capable of
        them.  ``loss_spec`` may carry ``inverse_dynamics_weight`` to add the
        inverse-dynamics auxiliary loss, and ``diagnostic_weight`` to add the
        per-stream next-step diagnostic cross-entropy (joint generation) — both to
        the returned scalar (training).  With ``loss_spec["return_components"]`` a
        dict ``{loss, camera_nll, diagnostic_ce, inv_dyn}`` is returned instead of
        the bare scalar (for per-component logging); the camera/diagnostic/inv-dyn
        terms are detached, only ``loss`` carries grad.  ``return_logits`` behaves
        as in v1.
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
        frame_log_dt = batch.get("frame_log_dt")
        camera_id = batch.get("camera_id")
        need_latents = bool(
            loss_spec is not None
            and self.has_inverse_dynamics
            and float((loss_spec or {}).get("inverse_dynamics_weight", 0.0)) > 0.0
        )
        need_diag = bool(
            loss_spec is not None
            and self.has_diagnostics
            and float((loss_spec or {}).get("diagnostic_weight", 0.0)) > 0.0
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
            frame_log_dt=frame_log_dt,
            camera_id=camera_id,
            return_latents=need_latents,
            return_signal_latents=need_diag,
        )
        signal_latents: dict[str, torch.Tensor] | None = None
        if need_latents and need_diag:
            hidden, latents, signal_latents = out
        elif need_diag:
            hidden, signal_latents = out
            latents = hidden
        elif need_latents:
            hidden, latents = out
        else:
            hidden = latents = out

        if loss_spec is not None:
            nll = self.chunked_nll(
                hidden,
                target,
                chunk=int(loss_spec.get("chunk", 4096)),
                context_frames=loss_spec.get("context_frames"),
                frame_weights=loss_spec.get("frame_weights"),
            )
            total = nll
            inv = None
            w_inv = float(loss_spec.get("inverse_dynamics_weight", 0.0))
            if self.has_inverse_dynamics and w_inv > 0.0 and actuator is not None:
                inv = self.inverse_dynamics_loss(
                    latents, actuator, context_frames=context_frames
                )
                total = total + w_inv * inv
            diag = None
            w_diag = float(loss_spec.get("diagnostic_weight", 0.0))
            if need_diag:
                diag = self.diagnostic_loss(signal_latents, signals)
                total = total + w_diag * diag
            elif self.has_diagnostics:
                # DDP-uniform: keep the heads in the graph even when the diagnostic
                # weight is 0 (a weight-ablation with the heads still built).
                total = total + self._diagnostic_zero_touch()
            if loss_spec.get("return_components"):
                zero = nll.new_zeros(())
                return {
                    "loss": total,
                    "camera_nll": nll.detach(),
                    "diagnostic_ce": diag.detach() if diag is not None else zero,
                    "inv_dyn": inv.detach() if inv is not None else zero,
                }
            return total
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
