"""Actuator-PLAN-conditioned spatiotemporal camera transformer (the driveable model).

What this adds over :mod:`imas_ambix.worldmodel.spacetime_model_v2`
------------------------------------------------------------------
The v2 model conditions the camera prediction on the MEASURED diagnostic streams
(magnetics, interferometer, soft_x_rays, gas-puff flow, …).  Those measured
streams are mutually REDUNDANT — the realised plasma state is written into all of
them — so the controllability falsification (M3) showed they are at best weakly
load-bearing: zeroing one barely moves the dream and classifier-free guidance
has nothing causal to amplify.  The model FORECASTS well but is not DRIVEABLE.

This model adds the DEMANDED actuator PLAN as the ALWAYS-ON drive surface (the
locked ``control-surface = actuator-vector`` decision):

* **actuator-plan path (always on).**  The continuous physical actuator vector
  (``amc`` coil/solenoid/TF currents + plasma current, ``anb`` NBI powers,
  ``aga`` gas-puff flows, ``ane`` density — see
  :mod:`imas_ambix.worldmodel.actuator_plan`) is sub-sampled to ``n_act_steps``
  steps spanning the window and projected by a LINEAR encoder (not a vocabulary
  embedding — it is continuous) into conditioning frames prepended on the
  temporal axis, exactly like the tokenised plan / signal frames.  A per-channel
  ``missing`` flag is concatenated so the encoder can ignore an absent actuator.
  This is the surface the operator edits to "play" the plasma.

* **measured observations become OPTIONAL.**  The v2 measured-signal streams are
  retained but conditioned with HIGH dropout at training (``observation_dropout``)
  so the model cannot shortcut the control→camera map by reading the redundant
  realised observations — it must learn to drive the camera from the actuator
  PLAN.  At inference the observations can be supplied (TRACK mode) or omitted
  (pure PLAY from the plan).

* **classifier-free guidance still works.**  Control-dropout zeroes the whole
  conditioning (plan + actuator + observations) on a fraction of steps so CFG can
  amplify the actuator plan's influence at inference
  (:mod:`imas_ambix.worldmodel.control_guidance`).

Everything else — the factorised space-time attention, the chunked NLL / argmax
head, the next-frame factorisation, the tokenised pulse-schedule plan prefix, the
measured-signal embedding tables, and the history-corruption conditioning — is
inherited verbatim from :class:`SignalSpacetimeTransformer`.  This subclass adds
only the actuator-plan encoder + its prefix construction.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from imas_ambix.worldmodel.spacetime_model_v2 import (
    SignalSpacetimeConfig,
    SignalSpacetimeTransformer,
    SignalStreamSpec,
)


@dataclass
class ControllableSpacetimeConfig(SignalSpacetimeConfig):
    """:class:`SignalSpacetimeConfig` + the demanded actuator-plan drive surface.

    ``actuator_channels`` is the width of the continuous actuator vector (the
    drive surface).  When it is 0 the model is byte-equivalent to the v2
    signal-conditioned model (no actuator path), so a v2 checkpoint loads
    cleanly.  ``n_act_steps`` is the number of conditioning steps the actuator
    plan is sub-sampled to (mirrors the plan's ``n_plan`` / the signals'
    ``n_signal_steps``).

    ``max_frames`` must cover the tokenised plan steps + the actuator steps +
    every present signal stream's steps + the camera frames; the builder sizes
    it.
    """

    actuator_channels: int = 0
    n_act_steps: int = 8

    @property
    def has_actuator(self) -> bool:
        return int(self.actuator_channels) > 0


class ControllableSpacetimeTransformer(SignalSpacetimeTransformer):
    """v2 backbone + a continuous actuator-PLAN drive surface on the temporal axis.

    The tokenised-plan conditioning, the measured-signal conditioning, the
    factorised space-time blocks, the chunked head, the next-frame factorisation,
    and the history-corruption embedding are all inherited verbatim from
    :class:`SignalSpacetimeTransformer`.  This subclass adds a LINEAR actuator
    encoder and overrides the prefix construction so actuator-plan frames are
    prepended alongside the tokenised plan + the measured signals, with the
    camera frames stripped from the output exactly as before.
    """

    config: ControllableSpacetimeConfig

    def __init__(self, cfg: ControllableSpacetimeConfig) -> None:
        # Build the v2 parameters (token / spatial / temporal / plan / signal
        # embeddings, blocks, head, weight tying) via the parent constructor,
        # then add the actuator parameters and init them to the same std=0.02.
        super().__init__(cfg)
        self.config = cfg  # narrow the type for the actuator-aware paths

        self.has_actuator = cfg.has_actuator
        d = cfg.d_model
        if self.has_actuator:
            c_act = int(cfg.actuator_channels)
            # A linear encoder per actuator step: the per-step input is the
            # actuator vector PLUS its per-channel missing flag (2*C_act), so the
            # encoder can learn to ignore an absent actuator.  It projects to one
            # vector per (step, spatial-lane): the actuator plan occupies the
            # FIRST ``n_act_lanes`` spatial lanes of a conditioning frame (with a
            # learned per-lane slot), the rest zero-filled — mirroring how the
            # tokenised plan + signals occupy lanes.  Spreading the projection
            # over several lanes (not one) gives the within-frame spatial
            # attention room to read different facets of the drive vector.
            self.actuator_lanes = min(int(cfg.actuator_channels), cfg.n_spatial)
            self.actuator_encoder = nn.Linear(2 * c_act, self.actuator_lanes * d)
            self.actuator_lane_embed = nn.Parameter(torch.zeros(self.actuator_lanes, d))
            # a learned marker so an actuator-plan frame is distinguishable from a
            # tokenised-plan frame, a signal frame, and a camera frame.
            self.actuator_marker = nn.Parameter(torch.zeros(d))
            nn.init.normal_(self.actuator_encoder.weight, std=0.02)
            nn.init.zeros_(self.actuator_encoder.bias)

    # -- embedding ---------------------------------------------------------

    def _embed_actuator(
        self,
        values: torch.Tensor,
        missing: torch.Tensor,
    ) -> torch.Tensor | None:
        """``(B, P, C) float + (B, P, C) float -> (B, P, S, d)`` actuator frames.

        Mirrors :meth:`SpacetimeTransformer._embed_plan` but for the CONTINUOUS
        actuator vector: the per-step ``[values | missing]`` vector is linearly
        projected to ``actuator_lanes`` lane vectors (with a learned per-lane
        slot), occupying the first ``actuator_lanes`` spatial lanes of a frame;
        the rest are zero-filled, and the shared spatial (row+col) position + the
        actuator marker are added so an actuator frame rides the same positional
        basis as a camera / plan / signal frame.  Returns ``None`` when the plan
        contributes no steps for the batch.
        """
        if values.numel() == 0 or values.ndim != 3 or values.shape[1] == 0:
            return None
        cfg = self.config
        b, p, c = values.shape
        s = cfg.n_spatial
        d = cfg.d_model
        miss = missing
        if miss is None or miss.shape != values.shape:
            miss = torch.zeros_like(values)
        # [values | missing] -> (B, P, 2C) -> (B, P, lanes*d) -> (B, P, lanes, d).
        # The shared spatial (row+col) embedding is the reference dtype: under a
        # bf16 autocast the Linear would emit bf16 while the embeddings stay
        # float32, so cast the encoded drive back to the embedding dtype before
        # adding — otherwise the later cat([prefix..., camera]) mixes dtypes.
        rows = torch.arange(cfg.grid_h, device=values.device).repeat_interleave(
            cfg.grid_w
        )
        cols = torch.arange(cfg.grid_w, device=values.device).repeat(cfg.grid_h)
        spatial = self.row_embed(rows) + self.col_embed(cols)  # (S, d)
        feat = torch.cat([values, miss], dim=-1).to(self.actuator_encoder.weight.dtype)
        proj = self.actuator_encoder(feat).to(spatial.dtype)  # (B, P, lanes*d)
        nl = self.actuator_lanes
        val = proj.view(b, p, nl, d)
        val = val + self.actuator_lane_embed.view(1, 1, nl, d)
        if nl < s:
            pad = val.new_zeros((b, p, s - nl, d))
            val = torch.cat([val, pad], dim=2)
        val = val + spatial.view(1, 1, s, d) + self.actuator_marker
        return val  # (B, P, S, d)

    def _actuator_zero_touch(self) -> torch.Tensor:
        """A zero-magnitude sum over every actuator param (DDP-uniform graph).

        Mirrors the signal zero-touch: a batch with no actuator plan would leave
        the actuator params grad-less and desync a DDP rank.  Touching them with
        a ``*0.0`` contribution keeps every actuator param in the autograd graph
        with zero effect on the output.
        """
        acc = (
            self.actuator_encoder.weight.sum()
            + self.actuator_encoder.bias.sum()
            + self.actuator_lane_embed.sum()
            + self.actuator_marker.sum()
        )
        return acc * 0.0

    def _forward_tokens(
        self,
        frames: torch.Tensor,
        plan: torch.Tensor | None,
        signals: dict[str, torch.Tensor] | None = None,
        corruption_level: torch.Tensor | None = None,
        *,
        actuator: dict[str, torch.Tensor] | None = None,
        context_frames: int | None = None,
    ) -> torch.Tensor:
        """Run the backbone with actuator-plan + tokenised-plan + signal conditioning.

        Builds ``[actuator | signals | plan | camera]`` on the temporal axis (the
        actuator drive surface first, closest to the temporal front, then the
        measured signals, then the tokenised plan, then the real frames — every
        conditioning frame sits before frame 0 so every camera token attends back
        to all of it), adds the absolute frame-position embedding, runs the
        factorised space-time blocks, and STRIPS every conditioning frame from the
        output.

        ``actuator`` is ``{"values": (B, P, C) float, "missing": (B, P, C)
        float}`` (the demanded actuator plan); ``None`` / empty omits it (the
        model then conditions on the plan + signals only, with the actuator
        zero-touch keeping DDP uniform).  ``signals`` / ``corruption_level`` /
        ``context_frames`` behave exactly as in v2.

        The history-corruption conditioning (inherited from v2) is applied to the
        CONTEXT camera frames here too — replicated rather than delegated because
        this overrides the v2 ``_forward_tokens`` (the parent's is not on this
        path).
        """
        cfg = self.config
        cam = self._embed_camera(frames)  # (B, T, S, d)
        b, t, s, d = cam.shape

        # Anti-drift history-corruption conditioning (inherited semantics).
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

        prefix: list[torch.Tensor] = []

        # actuator plan first (the drive surface, closest to the temporal front).
        actuator_frame_count = 0
        act_emb = None
        if self.has_actuator and actuator:
            vals = actuator.get("values")
            miss = actuator.get("missing")
            if vals is not None:
                act_emb = self._embed_actuator(vals, miss)
                if act_emb is not None:
                    prefix.append(act_emb)
                    actuator_frame_count += act_emb.shape[1]

        # measured signals next (optional, high-dropout context).
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

        # tokenised pulse-schedule plan last in the prefix.
        plan_emb = self._embed_plan(plan) if plan is not None else None
        if plan_emb is not None:
            prefix.append(plan_emb)

        x = torch.cat([*prefix, cam], dim=1) if prefix else cam  # (B, P+T, S, d)
        n_prefix = x.shape[1] - t

        # Keep every conditioning param in the autograd graph on a partial batch
        # (DDP uniformity): the plan + signal zero-touch (inherited) and the
        # actuator zero-touch.
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
        if self.has_actuator and actuator_frame_count == 0:
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
    ):
        """Teacher-forced forward with actuator-plan + plan + measured-signal cond.

        ``batch`` is ``{"frames": (B, T, S) long, "plan": (B, P, C) long,
        "signals": {name: (B, P_s, C_s) long}, "actuator": {"values": (B, P_a,
        C_a) float, "missing": (B, P_a, C_a) float}}``.  ``actuator`` may be
        absent / empty (the actuator zero-touch keeps DDP uniform).  An optional
        ``corruption_level`` ``(B,)`` long and ``target_frames`` ``(B, T, S)``
        long behave exactly as in v2.  ``loss_spec`` / ``return_logits`` behave
        exactly as in v1.
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
        hidden = self._forward_tokens(
            frames,
            plan,
            signals,
            corruption_level,
            actuator=actuator,
            context_frames=context_frames,
        )  # (B, T, S, d)

        if loss_spec is not None:
            return self.chunked_nll(
                hidden,
                target,
                chunk=int(loss_spec.get("chunk", 4096)),
                context_frames=loss_spec.get("context_frames"),
            )
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
