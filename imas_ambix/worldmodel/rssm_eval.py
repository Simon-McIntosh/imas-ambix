"""Held-out controllability EVAL for the command-conditioned RSSM world model.

This is the Phase-2 verdict.  The token backbone injected the demanded actuator
plan as an AdaLN *side*-input and the powered ΔN-M gate confirmed it could ignore
the plan (true-plan rollouts ≈ random-plan rollouts on the screened cohort).  The
RSSM (:mod:`imas_ambix.worldmodel.rssm`) puts the command INSIDE the recurrent
transition, so a different plan is a different recurrence and therefore a
different rollout — controllable by construction.  This module MEASURES that on
the SAME powered gate the token backbone failed:

* the noise-floor-NORMALISED, collapse-rejecting ΔN-M ratio over the screened
  25-shot cohort at ``n_random=10``, with the IDENTICAL summary schema (mean /
  median normalised ratio, pass/n, bootstrap CI, variance decomposition, per-shot
  ratios) as :func:`imas_ambix.worldmodel.controllable_eval.multi_shot_delta_nm`
  — so the RSSM number is directly comparable to the token-backbone numbers
  (baseline 0.85 / joint-gen 0.91 at n_random=10);
* the dreamt-vs-real next-step DIAGNOSTIC-MATCH (teacher-forced per-stream token
  accuracy + CE), the same schema as
  :func:`imas_ambix.worldmodel.controllable_eval.multi_shot_diagnostic_match`;
* an interpretable DREAM artifact — the true-plan PRIOR rollout decoded to a
  GIF/PNG for one cohort shot ("play the plasma").

The ONLY difference from the token gate is the ROLLOUT: instead of the token
transformer's autoregressive argmax rollout, each plan is rolled through the
RSSM's :meth:`RSSMWorldModel.rollout_prior` (warm the recurrent state on the GT
context frames, then roll the PRIOR forward under the commands, NO observations).
A per-rollout full window is assembled as ``[GT context frames | RSSM prior
forecast frames]`` so it plugs straight into the controllable_eval divergence +
summary machinery (which scores the forecast window, frames ``>= ctx``).

This file IMPORTS the powered-gate machinery from
:mod:`imas_ambix.worldmodel.controllable_eval` and
:mod:`imas_ambix.worldmodel.gate_cohort` — it does not duplicate or edit them.

GPU-safety (AGENTS.md §2b): the RSSM is loaded ONCE; every plan's rollout reuses
it; the VQ decode runs in the frozen-VQ subprocess
(:func:`imas_ambix.worldmodel.control_falsification.decode_roles`); SIGTERM-clean;
``try/finally`` release.  Everything except the decode is decoder-free and
CPU-testable (``--no-decode`` / the unit tests).
"""

from __future__ import annotations

import contextlib
import json
import logging
import signal as _signal
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

#: A cooperative-stop flag the SIGTERM/SIGINT handler sets so a long cohort eval
#: tears down cleanly (the per-shot loop checks it).  AGENTS.md §2b.
_STOP = {"flag": False}


def _install_stop_handler() -> None:
    def _on_signal(signum, _frame):  # noqa: ANN001
        logger.warning(
            "RSSM eval received signal %s — stopping after this shot", signum
        )
        _STOP["flag"] = True

    for sig in (_signal.SIGTERM, _signal.SIGINT):
        # not the main thread / unsupported platform -> leave the default handler.
        with contextlib.suppress(ValueError, OSError):
            _signal.signal(sig, _on_signal)


# ---------------------------------------------------------------------------
# Load a trained RSSM checkpoint (reconstruct the config + load the weights)
# ---------------------------------------------------------------------------


def load_rssm_from_checkpoint(path, *, map_location: str = "cpu"):
    """Rebuild an :class:`RSSMWorldModel` from a checkpoint + load its weights.

    Mirrors
    :func:`imas_ambix.worldmodel.controllable_train.load_controllable_model_from_checkpoint`:
    reconstructs the FULL :class:`imas_ambix.worldmodel.rssm.RSSMConfig` from the
    saved ``model_config`` (signal streams + actuator channels + the masked-command
    set) so the eval model conditions EXACTLY as the trained one, then loads the
    state dict (``strict=False`` — a camera-only checkpoint has no
    ``diagnostic_heads.*`` and a DDP checkpoint may carry a ``module.`` prefix).

    Accepts the RSSM trainer's payload (``{"model_config": {...},
    "model_state_dict": ...}`` with optional ``extra.stream_names``).  Returns
    ``(model, payload)``.
    """
    import torch  # noqa: PLC0415

    from imas_ambix.worldmodel.rssm import (  # noqa: PLC0415
        RSSMConfig,
        RSSMWorldModel,
        SignalStreamSpec,
    )

    payload = torch.load(str(path), map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or "model_config" not in payload:
        raise ValueError(
            f"checkpoint {path!r} has no 'model_config' — not an RSSM checkpoint"
        )
    d = dict(payload["model_config"])
    streams = tuple(
        SignalStreamSpec(
            name=str(s["name"]), vocab=int(s["vocab"]), channels=int(s["channels"])
        )
        for s in d.get("signal_streams", [])
    )
    scalar = {
        k: d[k]
        for k in d
        if k in RSSMConfig.__dataclass_fields__
        and k not in ("signal_streams", "masked_command_indices")
    }
    cfg = RSSMConfig(
        signal_streams=streams,
        masked_command_indices=tuple(d.get("masked_command_indices", ())),
        **scalar,
    )
    model = RSSMWorldModel(cfg)

    # find the state dict under any of the trainer's conventional keys.
    state = None
    for key in ("model_state_dict", "model_state", "model", "state_dict"):
        if isinstance(payload.get(key), dict):
            state = payload[key]
            break
    if state is None:
        raise ValueError(f"checkpoint {path!r} carries no state dict")
    # strip a DDP ``module.`` prefix if present.
    state = {
        (k[len("module.") :] if k.startswith("module.") else k): v
        for k, v in state.items()
    }
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        logger.info(
            "loaded RSSM checkpoint with strict=False: %d missing, %d unexpected keys",
            len(missing),
            len(unexpected),
        )
    # re-tie the head to the (loaded) token embed.
    model.head.weight = model.token_embed.weight
    model.to(map_location)
    return model, payload


# ---------------------------------------------------------------------------
# RSSM rollout -> a full (T, S) token window (GT context + prior forecast)
# ---------------------------------------------------------------------------


def _rssm_full_rollout(model, sample, plan, device, *, chunk: int = 4096) -> np.ndarray:
    """A FULL-window ``(T, S)`` token rollout for ``plan`` under the RSSM.

    The RSSM warms its recurrent state on the GT context frames, then rolls the
    PRIOR forward under the commands (no observations) for the forecast window.
    The returned array is ``[GT context frames | RSSM prior forecast frames]`` in
    LOCAL camera-token ids — the SAME layout as
    :func:`imas_ambix.worldmodel.controllable_train._argmax_token_rollout`, so it
    plugs straight into the controllable_eval divergence scoring (which slices
    ``[ctx:]``).  ``sample=False`` (the prior MEAN) is the cleanest, deterministic
    controllability probe.
    """
    import torch  # noqa: PLC0415

    from imas_ambix.worldmodel.controllable_train import (  # noqa: PLC0415
        _actuator_batch_from_plan,
    )
    from imas_ambix.worldmodel.spacetime_train import _AutocastCtx  # noqa: PLC0415

    ctx = int(sample.context_frames)
    frames = np.asarray(sample.frames, dtype=np.int64)  # (T, S) LOCAL ids
    t_total = int(frames.shape[0])
    n_steps = t_total - ctx
    out = frames.copy()
    if n_steps <= 0:
        return out

    context = torch.as_tensor(frames[:ctx][None], dtype=torch.long, device=device)
    actuator = _actuator_batch_from_plan(plan, device)
    with torch.no_grad(), _AutocastCtx(device):
        rollout = model.rollout_prior(
            context, actuator, n_steps, chunk=chunk, sample=False
        )
    forecast = rollout.frames[0].cpu().numpy().astype(np.int64)  # (n_steps, S)
    out[ctx:] = forecast[:n_steps]
    return out


# ---------------------------------------------------------------------------
# Multi-shot held-out ΔN-M — the powered robust gate, RSSM rollouts
# ---------------------------------------------------------------------------


def multi_shot_delta_nm_rssm(
    model,
    *,
    config,
    camera: str,
    token_root: Path | None,
    device: str,
    out_json: Path,
    work_dir: Path | None = None,
    decode: bool = True,
) -> dict:
    """Decoded-pixel ΔN-M over the held-out cohort — the RSSM controllability verdict.

    Identical to
    :func:`imas_ambix.worldmodel.controllable_eval.multi_shot_delta_nm` except that
    each plan is rolled through the RSSM PRIOR (:func:`_rssm_full_rollout`) instead
    of the token transformer's argmax rollout.  For each cohort shot: roll the TRUE
    plan + ``n_random`` BOUNDED coil counterfactuals
    (:func:`imas_ambix.worldmodel.controllable_train._random_actuator_like`), decode
    all rollouts through the frozen Open-MAGVIT2 VQ, and score the forecast-window
    decoded-pixel L1 of true-vs-random against the (collapse-rejected) pairwise
    random-vs-random noise floor — REUSING the controllable_eval verdict,
    divergence, within-shot bootstrap, and summary helpers so the output JSON
    schema is IDENTICAL to the token gate (directly comparable).

    ``decode=False`` scores in token space (a decoder-free lower bound) for the CPU
    smoke / when the VQ stack is unavailable.  Writes the verdict JSON and returns
    the summary dict.
    """
    import torch  # noqa: PLC0415

    from imas_ambix.worldmodel.control_falsification import (  # noqa: PLC0415
        decode_roles,
    )
    from imas_ambix.worldmodel.controllable_eval import (  # noqa: PLC0415
        HeldoutDeltaNMVerdict,
        _assemble_heldout,
        _decoded_divergences,
        _plan_variation,
        _summarise,
        _token_divergences,
        _within_shot_ratio_std,
    )
    from imas_ambix.worldmodel.controllable_train import (  # noqa: PLC0415
        _random_actuator_like,
    )
    from imas_ambix.worldmodel.spacetime_dataset import (  # noqa: PLC0415
        GRID_H,
        GRID_W,
        local_to_store,
    )

    dev = torch.device(device)
    model.eval()
    samples = []
    for sid in config.held_out:
        if _STOP["flag"]:
            break
        try:
            samples.append(
                _assemble_heldout(sid, config, camera=camera, token_root=token_root)
            )
        except (ValueError, FileNotFoundError, KeyError) as exc:
            logger.warning("held-out shot %s unavailable (%r) — skipped", sid, exc)
    if not samples:
        raise ValueError("no held-out shot could be assembled")

    verdicts: list[HeldoutDeltaNMVerdict] = []
    work = Path(work_dir) if work_dir else None

    for s in samples:
        if _STOP["flag"]:
            break
        ctx = int(s.context_frames)
        plan_var = _plan_variation(s)
        is_transient = bool(plan_var >= config.transient_threshold)
        rng = np.random.default_rng((int(s.shot_id) * 1_000_003) ^ (config.seed * 31))

        # TRUE-plan RSSM rollout (full-window token ids).
        true_tok = _rssm_full_rollout(model, s, s.actuator, dev, chunk=config.chunk)
        # N bounded coil-counterfactual RSSM rollouts.
        rand_toks: list[np.ndarray] = []
        for _ in range(int(config.n_random)):
            rplan = _random_actuator_like(
                s.actuator, rng=rng, perturb_scale=config.perturb_scale
            )
            rand_toks.append(
                _rssm_full_rollout(model, s, rplan, dev, chunk=config.chunk)
            )

        n_collapsed = 0
        n_kept = int(config.n_random)
        if decode:
            tvr, rvr, n_collapsed, n_kept, tvr_samples, rvr_samples = (
                _decoded_divergences(
                    true_tok,
                    rand_toks,
                    ctx,
                    device=device,
                    work_dir=work,
                    shot_id=int(s.shot_id),
                    grid_hw=(GRID_H, GRID_W),
                    local_to_store=local_to_store,
                    decode_roles=decode_roles,
                    reject_collapsed=config.reject_collapsed,
                )
            )
        else:
            tvr, rvr, tvr_samples, rvr_samples = _token_divergences(
                true_tok, rand_toks, ctx
            )

        margin = tvr - rvr
        ratio = float("inf") if rvr == 0.0 else tvr / rvr
        ratio_within_std = _within_shot_ratio_std(
            tvr_samples,
            rvr_samples,
            n_boot=config.n_within_bootstrap,
            seed=(int(s.shot_id) * 7919) ^ (config.seed * 31),
        )
        if config.robust_gate:
            passed = bool(
                is_transient
                and (
                    (rvr == 0.0 and tvr > config.margin_threshold)
                    or ratio > config.ratio_threshold
                )
            )
        else:
            passed = bool(
                is_transient
                and margin > config.margin_threshold
                and (
                    (rvr == 0.0 and tvr > config.margin_threshold)
                    or ratio > config.floor_ratio
                )
            )
        verdicts.append(
            HeldoutDeltaNMVerdict(
                shot_id=int(s.shot_id),
                is_transient=is_transient,
                plan_variation=plan_var,
                true_vs_random=tvr,
                random_vs_random=rvr,
                margin=margin,
                ratio=ratio,
                n_random=int(config.n_random),
                n_random_collapsed=int(n_collapsed),
                n_random_kept=int(n_kept),
                true_vs_random_samples=[float(x) for x in tvr_samples],
                random_vs_random_samples=[float(x) for x in rvr_samples],
                ratio_within_std=float(ratio_within_std),
                passed=passed,
            )
        )

    summary = _summarise(verdicts, config, decode=decode)
    summary["rollout"] = "rssm_prior"  # provenance: which rollout produced the gate
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(
            {"per_shot": [v.to_dict() for v in verdicts], "summary": summary},
            indent=2,
            default=str,
        )
    )
    logger.info("RSSM held-out ΔN-M verdict -> %s : %s", out_json, summary["verdict"])
    return summary


# ---------------------------------------------------------------------------
# Dreamt-vs-real next-step diagnostic-match (the RSSM teacher-forced axis)
# ---------------------------------------------------------------------------


def diagnostic_match_rssm(model, sample, *, device, chunk: int = 4096) -> dict:
    """How well the RSSM's NEXT-step dreamt diagnostic tokens match the REAL ones.

    The RSSM grows per-stream diagnostic heads (joint generation).  This is the
    teacher-forced held-out axis for them, the SAME schema as
    :func:`imas_ambix.worldmodel.controllable_eval.diagnostic_match`: a single
    teacher-forced forward on the real frames + real signals, the per-frame latent
    decoded through the per-stream heads, scored next-step against the REAL
    next-step tokens (PAD id 0 ignored).  Per stream: top-1 accuracy over masked
    positions + a continuous next-step CE.  A model with no diagnostic heads
    reports ``diagnostics_generated=False`` and zeros — an honest read.

    Returns ``{"per_stream": {name: {"accuracy", "ce", "n"}}, "mean_accuracy",
    "mean_ce", "diagnostics_generated"}``.
    """
    import torch  # noqa: PLC0415
    from torch.nn import functional as F  # noqa: PLC0415, N812

    from imas_ambix.worldmodel.controllable_train import (  # noqa: PLC0415
        _actuator_batch_from_plan,
    )

    empty = {
        "per_stream": {},
        "mean_accuracy": 0.0,
        "mean_ce": 0.0,
        "diagnostics_generated": bool(getattr(model.config, "has_diagnostics", False)),
    }
    if not bool(getattr(model.config, "has_diagnostics", False)):
        return empty

    dev = torch.device(device)
    model.eval()
    frames = torch.as_tensor(
        np.asarray(sample.frames, dtype=np.int64)[None], dtype=torch.long, device=dev
    )
    # signals: each stream's (Ps, Cs) -> (1, Ps, Cs) long.
    signals: dict[str, torch.Tensor] = {}
    for name, tok in sample.signals.items():
        arr = np.asarray(tok, dtype=np.int64)
        if arr.ndim == 2:
            signals[name] = torch.as_tensor(arr[None], dtype=torch.long, device=dev)

    # teacher-forced forward (under the TRUE actuator plan — the command lives in
    # the GRU transition, so a model with an actuator path needs it) to get the
    # per-frame latent z, then the per-frame diagnostic logits — the RSSM dreams a
    # diagnostic token from EACH latent.
    actuator = (
        _actuator_batch_from_plan(sample.actuator, dev)
        if getattr(model.config, "has_actuator", False)
        else None
    )
    with torch.no_grad():
        out = model(
            {
                "frames": frames,
                "actuator": actuator,
                "signals": signals,
            },
            chunk=chunk,
        )
        z = torch.cat([out.h, out.s], dim=-1)  # (1, T, z_dim)
        logits = model.diagnostic_logits(z)  # {name: (1, T, C, vocab)}

    per_stream: dict[str, dict] = {}
    accs: list[float] = []
    ces: list[float] = []
    for name, lg in logits.items():
        tok = signals.get(name)
        if tok is None:
            continue
        # lg: (1, T, C, V); tok: (1, Ps, Cs) — the latent at frame j predicts the
        # stream token at step j+1 (next-step), over the steps the stream supplies.
        p = min(int(lg.shape[1]), int(tok.shape[1]))
        c = min(int(lg.shape[2]), int(tok.shape[2]))
        if p < 2 or c < 1:
            continue
        pred_logits = lg[:, : p - 1, :c, :].float()  # (1, p-1, C, V)
        pred = pred_logits.argmax(dim=-1)  # (1, p-1, C)
        target = tok[:, 1:p, :c].to(dev).long()  # (1, p-1, C)
        mask = target != 0  # PAD id 0 = absent step
        n = int(mask.sum())
        if n == 0:
            continue
        acc = float((pred[mask] == target[mask]).float().mean())
        ce = float(
            F.cross_entropy(
                pred_logits.reshape(-1, pred_logits.shape[-1]),
                target.reshape(-1),
                ignore_index=0,
            )
        )
        per_stream[name] = {"accuracy": acc, "ce": ce, "n": n}
        accs.append(acc)
        ces.append(ce)

    return {
        "per_stream": per_stream,
        "mean_accuracy": float(np.mean(accs)) if accs else 0.0,
        "mean_ce": float(np.mean(ces)) if ces else 0.0,
        "diagnostics_generated": True,
    }


def multi_shot_diagnostic_match_rssm(
    model,
    *,
    config,
    camera: str,
    token_root: Path | None,
    device: str,
    out_json: Path,
) -> dict:
    """Dreamt-vs-real diagnostic-match over the held-out cohort (RSSM) + JSON summary.

    Mirrors
    :func:`imas_ambix.worldmodel.controllable_eval.multi_shot_diagnostic_match`:
    assembles each held-out window, scores its next-step diagnostic-match via the
    RSSM (:func:`diagnostic_match_rssm`), and aggregates the same summary schema
    (mean accuracy + mean CE across shots + a per-stream macro breakdown).  Writes
    ``out_json`` and returns the summary.
    """
    import torch  # noqa: PLC0415

    from imas_ambix.worldmodel.controllable_eval import (  # noqa: PLC0415
        _assemble_heldout,
    )

    dev = torch.device(device)
    samples = []
    for sid in config.held_out:
        if _STOP["flag"]:
            break
        try:
            samples.append(
                _assemble_heldout(sid, config, camera=camera, token_root=token_root)
            )
        except (ValueError, FileNotFoundError, KeyError) as exc:
            logger.warning("held-out shot %s unavailable (%r) — skipped", sid, exc)
    if not samples:
        raise ValueError("no held-out shot could be assembled")

    per_shot: list[dict] = []
    stream_acc: dict[str, list[float]] = {}
    stream_ce: dict[str, list[float]] = {}
    for s in samples:
        if _STOP["flag"]:
            break
        res = diagnostic_match_rssm(model, s, device=str(dev), chunk=config.chunk)
        res["shot_id"] = int(s.shot_id)
        per_shot.append(res)
        for name, d in res["per_stream"].items():
            stream_acc.setdefault(name, []).append(float(d["accuracy"]))
            stream_ce.setdefault(name, []).append(float(d["ce"]))

    scored = [r for r in per_shot if r["per_stream"]]
    mean_acc = float(np.mean([r["mean_accuracy"] for r in scored])) if scored else 0.0
    mean_ce = float(np.mean([r["mean_ce"] for r in scored])) if scored else 0.0
    per_stream_macro = {
        name: {
            "accuracy": float(np.mean(stream_acc[name])),
            "ce": float(np.mean(stream_ce[name])),
            "n_shots": len(stream_acc[name]),
        }
        for name in stream_acc
    }
    diagnostics_generated = bool(getattr(model.config, "has_diagnostics", False))
    summary = {
        "metric": "diagnostic_match_next_step",
        "diagnostics_generated": diagnostics_generated,
        "n_samples": len(per_shot),
        "n_scored": len(scored),
        "mean_accuracy": mean_acc,
        "mean_ce": mean_ce,
        "per_stream": per_stream_macro,
        "rollout": "rssm_teacher_forced",
    }
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps({"per_shot": per_shot, "summary": summary}, indent=2, default=str)
    )
    logger.info(
        "RSSM held-out diagnostic-match -> %s : mean_acc=%.4f mean_ce=%.4f (gen=%s)",
        out_json,
        mean_acc,
        mean_ce,
        diagnostics_generated,
    )
    return summary


# ---------------------------------------------------------------------------
# Interpretable dream artifact (the true-plan prior rollout, decoded)
# ---------------------------------------------------------------------------


def dream_rollout(
    model,
    *,
    config,
    camera: str,
    token_root: Path | None,
    device: str,
    out_dir: Path,
    shot_id: int | None = None,
    fps: int = 8,
) -> dict:
    """Render the "play the plasma" artifact: the RSSM true-plan PRIOR dream.

    Picks one cohort shot, rolls the TRUE plan through the RSSM prior
    (:func:`_rssm_full_rollout`), decodes the full window through the frozen VQ, and
    writes a GIF (the dreamt clip, GT context | prior forecast) + a centroid-trace
    PNG + a small JSON.  The analogue of
    :func:`imas_ambix.worldmodel.controllable_eval.coil_edit_dream` for the RSSM —
    a single interpretable rollout rather than a side-by-side coil edit.  Requires
    the decode stack (GPU + the frozen VQ); raises if unavailable.
    """
    import torch  # noqa: PLC0415

    from imas_ambix.worldmodel.control_falsification import (  # noqa: PLC0415
        decode_roles,
    )
    from imas_ambix.worldmodel.controllable_eval import (  # noqa: PLC0415
        _assemble_heldout,
        decoded_centroid,
    )
    from imas_ambix.worldmodel.spacetime_dataset import (  # noqa: PLC0415
        GRID_H,
        GRID_W,
        local_to_store,
    )

    dev = torch.device(device)
    model.eval()
    sid = int(shot_id if shot_id is not None else config.held_out[0])
    sample = _assemble_heldout(sid, config, camera=camera, token_root=token_root)
    ctx = int(sample.context_frames)

    # the dreamt clip under the TRUE plan + the recorded ground truth, decoded
    # together in ONE VQ pass.
    dream_tok = _rssm_full_rollout(
        model, sample, sample.actuator, dev, chunk=config.chunk
    )
    gt_tok = np.asarray(sample.frames, dtype=np.int64)
    n = int(gt_tok.shape[0])
    grids = {
        "dream": local_to_store(dream_tok.reshape(-1, GRID_H, GRID_W)),
        "gt": local_to_store(gt_tok.reshape(-1, GRID_H, GRID_W)),
    }
    roles = [{"role": "dream"}, {"role": "gt"}]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    decoded = decode_roles(grids, roles, work_dir=out_dir / "_decode", device=device)
    dream_px = decoded["dream"]
    gt_px = decoded["gt"]

    paths = _render_dream_artifacts(
        gt_px,
        dream_px,
        decoded_centroid(gt_px),
        decoded_centroid(dream_px),
        ctx=ctx,
        out_dir=out_dir,
        shot_id=sid,
        fps=fps,
    )
    result = {"shot_id": sid, "n_frames": n, "context_frames": ctx, **paths}
    (out_dir / f"rssm_dream_shot{sid}.json").write_text(
        json.dumps(result, indent=2, default=str)
    )
    logger.info("RSSM dream artifact shot %s -> %s", sid, paths.get("gif_path"))
    return result


def _render_dream_artifacts(
    gt_px, dream_px, gt_cen, dream_cen, *, ctx, out_dir, shot_id, fps
):
    """Write the GT|dream side-by-side GIF + the centroid-trace PNG."""
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    from imas_ambix.worldmodel.control_guidance import _to_gray_f64  # noqa: PLC0415
    from imas_ambix.worldmodel.dream_gifs import (  # noqa: PLC0415
        _panel_frame,
        _save_gif,
    )

    out_dir = Path(out_dir)
    gif_path = out_dir / f"rssm_dream_shot{shot_id}.gif"
    n = min(gt_px.shape[0], dream_px.shape[0])
    gt = _to_gray_f64(gt_px)
    dr = _to_gray_f64(dream_px)
    frames = []
    for i in range(n):
        banner = f"shot {shot_id} — play the plasma: RSSM prior dream (frame {i})"
        frames.append(
            _panel_frame(
                gt[i],
                dr[i],
                left_title="ground truth",
                right_title="RSSM dream",
                banner=banner,
                in_target=(i >= ctx),
            )
        )
    _save_gif(frames, gif_path, fps=fps)

    png_path = out_dir / f"rssm_dream_shot{shot_id}_centroid.png"
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2), dpi=110)
    t = np.arange(gt_cen.shape[0])
    dims = ((axes[0], 0, "row (vertical)"), (axes[1], 1, "col (radial)"))
    for ax, dim, name in dims:
        ax.plot(t, gt_cen[:, dim], "-o", ms=3, label="ground truth", color="#1f77b4")
        ax.plot(t, dream_cen[:, dim], "-o", ms=3, label="RSSM dream", color="#d62728")
        ax.axvline(ctx - 0.5, ls="--", color="#888", lw=1)
        ax.set_title(f"centroid {name}", fontsize=10)
        ax.set_xlabel("frame")
        ax.set_ylabel("pixel")
        ax.legend(fontsize=8)
    fig.suptitle(
        f"shot {shot_id}: RSSM prior dream vs ground truth centroid "
        f"(dashed = forecast start)",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(png_path)
    plt.close(fig)
    return {"gif_path": str(gif_path), "centroid_png_path": str(png_path)}


# ---------------------------------------------------------------------------
# CLI — run the instant the overnight RSSM train writes best.pt
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    import argparse  # noqa: PLC0415
    import logging as _logging  # noqa: PLC0415

    import torch  # noqa: PLC0415

    from imas_ambix.worldmodel.controllable_eval import (  # noqa: PLC0415
        EvalConfig,
        _resolve_eval_modalities,
    )
    from imas_ambix.worldmodel.spacetime_dataset import (  # noqa: PLC0415
        SpacetimeWindowConfig,
    )

    p = argparse.ArgumentParser(
        description="Held-out RSSM controllability eval (powered ΔN-M gate)."
    )
    p.add_argument("--checkpoint", required=True, help="the RSSM train best.pt")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--camera", default="rbb")
    p.add_argument("--token-root", default=None)
    p.add_argument("--held-out", default="18502,18503,18504,18505")
    p.add_argument(
        "--held-out-cohort",
        default=None,
        help="path to a screened-cohort JSON (gate_cohort.build_screened_cohort); "
        "its shot ids REPLACE --held-out for the robust gate.  The 25-shot cohort "
        "lives at /work/projects/imas_gpu/worldmodel/gate_cohort.json.",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--n-random",
        type=int,
        default=10,
        help="bounded coil counterfactuals rolled per shot (default 10 — matches "
        "the token-backbone gate so the RSSM ratio is directly comparable).",
    )
    p.add_argument("--perturb-scale", type=float, default=0.3)
    p.add_argument("--margin-threshold", type=float, default=1.0)
    p.add_argument("--floor-ratio", type=float, default=1.5)
    p.add_argument("--ratio-threshold", type=float, default=1.5)
    p.add_argument("--n-bootstrap", type=int, default=2000)
    p.add_argument("--n-frames", type=int, default=24)
    p.add_argument("--n-plan", type=int, default=8)
    p.add_argument("--context-frames", type=int, default=8)
    p.add_argument("--frame-stride", type=int, default=1)
    p.add_argument(
        "--target-horizon-s",
        type=float,
        default=0.25,
        help="physical seconds the window spans — MUST match the training run.",
    )
    p.add_argument("--n-signal-steps", type=int, default=4)
    p.add_argument("--n-act-steps", type=int, default=8)
    p.add_argument("--chunk", type=int, default=8192)
    p.add_argument(
        "--signal-modalities",
        choices=("auto", "default", "extended"),
        default="auto",
        help="which measured-signal streams to CONDITION + score on (auto reads the "
        "trained set from the checkpoint's extra.stream_names).",
    )
    p.add_argument(
        "--no-decode",
        action="store_true",
        help="score the ΔN-M in TOKEN space (decoder-free lower bound); skips the "
        "dream GIF.  Used when the VQ stack is unavailable.",
    )
    p.add_argument(
        "--no-dream",
        action="store_true",
        help="skip the dream GIF (run only the ΔN-M verdict).",
    )
    p.add_argument(
        "--no-diagnostic-match",
        action="store_true",
        help="skip the dreamt-vs-real next-step diagnostic-match metric.",
    )
    p.add_argument(
        "--no-reject-collapsed",
        dest="reject_collapsed",
        action="store_false",
        default=True,
        help="keep collapsed random dreams in the noise floor (debug).",
    )
    p.add_argument(
        "--no-robust-gate",
        dest="robust_gate",
        action="store_false",
        default=True,
        help="legacy fixed-shot ABSOLUTE-margin gate (back-comparison).",
    )
    args = p.parse_args(argv)

    _logging.basicConfig(
        level=_logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    _install_stop_handler()
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA unavailable — CPU (token-space only)")
        device = "cpu"
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if device == "cuda":
        torch.set_float32_matmul_precision("high")

    out_dir = Path(args.out_dir)
    model = None
    try:
        model, payload = load_rssm_from_checkpoint(
            Path(args.checkpoint), map_location=device
        )
        model.eval()
        modalities = _resolve_eval_modalities(args.signal_modalities, payload)
        window = SpacetimeWindowConfig(
            n_frames=args.n_frames,
            n_plan=args.n_plan,
            context_frames=args.context_frames,
            frame_stride=args.frame_stride,
            target_horizon_s=args.target_horizon_s,
        )
        token_root = Path(args.token_root) if args.token_root else None

        held_out = tuple(int(s) for s in args.held_out.split(",") if s.strip())
        if args.held_out_cohort:
            from imas_ambix.worldmodel.gate_cohort import (  # noqa: PLC0415
                load_cohort,
            )

            held_out = tuple(load_cohort(args.held_out_cohort))
            logger.info(
                "cohort loaded from %s: %d shots", args.held_out_cohort, len(held_out)
            )
        if not held_out:
            raise ValueError("empty held-out cohort — nothing to evaluate")

        cfg = EvalConfig(
            held_out=held_out,
            n_random=args.n_random,
            perturb_scale=args.perturb_scale,
            margin_threshold=args.margin_threshold,
            floor_ratio=args.floor_ratio,
            chunk=args.chunk,
            n_signal_steps=args.n_signal_steps,
            n_act_steps=args.n_act_steps,
            modalities=modalities,
            robust_gate=args.robust_gate,
            ratio_threshold=args.ratio_threshold,
            n_bootstrap=args.n_bootstrap,
            reject_collapsed=args.reject_collapsed,
            window=window,
        )
        summary = multi_shot_delta_nm_rssm(
            model,
            config=cfg,
            camera=args.camera,
            token_root=token_root,
            device=device,
            out_json=out_dir / "heldout_delta_nm.json",
            work_dir=out_dir / "_dnm",
            decode=not args.no_decode,
        )
        logger.info("RSSM HELD-OUT ΔN-M: %s", summary)
        if not args.no_diagnostic_match:
            try:
                dmatch = multi_shot_diagnostic_match_rssm(
                    model,
                    config=cfg,
                    camera=args.camera,
                    token_root=token_root,
                    device=device,
                    out_json=out_dir / "heldout_diagnostic_match.json",
                )
                logger.info("RSSM HELD-OUT diagnostic-match: %s", dmatch)
            except ValueError as exc:
                logger.warning("diagnostic-match skipped (%r)", exc)
        if not args.no_dream and not args.no_decode and not _STOP["flag"]:
            dream = dream_rollout(
                model,
                config=cfg,
                camera=args.camera,
                token_root=token_root,
                device=device,
                out_dir=out_dir,
            )
            logger.info("RSSM DREAM artifact: %s", dream.get("gif_path"))
    finally:
        try:
            del model
            if device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001
            logger.warning("model release note: %r", exc)
    return 0


__all__ = [
    "diagnostic_match_rssm",
    "dream_rollout",
    "load_rssm_from_checkpoint",
    "main",
    "multi_shot_delta_nm_rssm",
    "multi_shot_diagnostic_match_rssm",
]


if __name__ == "__main__":
    raise SystemExit(main())
