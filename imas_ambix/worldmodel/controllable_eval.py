"""Post-train controllability EVAL on the HELD-OUT excited shots (the real test).

The overfit gate de-risked the architecture (1-step conditioning load-bearing);
the corpus re-train then has to make the actuator plan steer a FREE-RUNNING
rollout on shots the model never trained on.  This module is the verdict + the
headline artifact for that, runnable the instant the re-train writes ``best.pt``:

1. :func:`multi_shot_delta_nm` — the multi-shot held-out ΔN-M.  For each held-out
   shot it autoregressively rolls the model under the TRUE coil(+gas+nbi) plan and
   under several BOUNDED, in-distribution coil counterfactuals
   (:func:`imas_ambix.worldmodel.controllable_train._random_actuator_like`),
   DECODES the rollouts through the frozen Open-MAGVIT2 VQ, and scores the
   forecast-window decoded-pixel divergence true-vs-counterfactual against the
   counterfactual-vs-counterfactual NOISE FLOOR.  PASS = true-vs-random clears the
   floor by a clear margin (the plan moves the dream more than a different plan
   moves it from another).  Emits a verdict JSON.

2. :func:`coil_edit_dream` — the interpretable "play the plasma" artifact: pick
   the PF POSITION coil whose bounded edit most moves the decoded plasma centroid,
   then render the TRUE-plan dream beside the EDITED-plan dream (the plasma
   visibly shifting) + the decoded-centroid trace.

3. :func:`diagnostic_match` / :func:`multi_shot_diagnostic_match` — the dreamt-vs-
   real diagnostic-match: the joint world-model grows per-stream heads that predict
   the NEXT-step measured-signal tokens, so this scores how well those dreamt tokens
   match the REAL next-step tokens on the held-out shots (per-stream top-1 token
   accuracy + a continuous next-step CE), the new quantitative eval axis beside the
   ΔN-M controllability gate.  A camera-only baseline (no / untrained heads) reports
   ``diagnostics_generated=False`` and zeros — an honest read, not a near-chance
   score.

The model conditions ONLY on the commands (it masks Ip/density/tf internally), so
the held-out windows are assembled with the DEFAULT full actuator vector — no
special channel list needed; the model masks consistently with training.

GPU-safety (AGENTS.md §2b): the model is loaded ONCE; the decode runs in the
frozen-VQ subprocess (:func:`decode_roles`); SIGTERM-clean; ``try/finally``
release.  Everything except the decode is decoder-free and CPU-testable with a
dummy checkpoint (see ``scripts``/the unit tests).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

#: The held-out excited shots (disjoint from the curated train corpus).
DEFAULT_HELD_OUT: tuple[int, ...] = (18502, 18503, 18504, 18505)


# ---------------------------------------------------------------------------
# Decoded-pixel helpers (centroid + divergence)
# ---------------------------------------------------------------------------


def decoded_centroid(frames: np.ndarray, *, threshold_frac: float = 0.5) -> np.ndarray:
    """Per-frame plasma emission centroid ``(row, col)`` in pixel coordinates.

    ``frames`` is a decoded ``(F, H, W[, C])`` stack.  For each frame the centroid
    is the luminance-weighted mean position over the pixels brighter than
    ``threshold_frac`` of that frame's max (a simple, robust emission centroid —
    the bright plasma blob).  Returns ``(F, 2)`` float ``[row, col]``; a dark frame
    (no pixel over threshold) yields the frame-centre so the trace stays finite.
    """
    from imas_ambix.worldmodel.control_guidance import _to_gray_f64  # noqa: PLC0415

    g = _to_gray_f64(frames)  # (F, H, W)
    f, h, w = g.shape
    rows = np.arange(h, dtype=np.float64)[:, None]
    cols = np.arange(w, dtype=np.float64)[None, :]
    out = np.zeros((f, 2), dtype=np.float64)
    for i in range(f):
        frame = g[i]
        thr = threshold_frac * float(frame.max())
        m = np.where(frame >= thr, frame, 0.0)
        tot = float(m.sum())
        if tot <= 0.0:
            out[i] = [(h - 1) / 2.0, (w - 1) / 2.0]
            continue
        out[i, 0] = float((m * rows).sum() / tot)
        out[i, 1] = float((m * cols).sum() / tot)
    return out


def _forecast_pixel_l1(a: np.ndarray, b: np.ndarray, ctx: int) -> float:
    """Mean abs decoded-pixel difference over the forecast window (frames >= ctx)."""
    from imas_ambix.worldmodel.control_guidance import _to_gray_f64  # noqa: PLC0415

    ga, gb = _to_gray_f64(a), _to_gray_f64(b)
    n = ga.shape[0]
    if n <= ctx:
        return 0.0
    return float(np.abs(ga[ctx:] - gb[ctx:]).mean())


#: A decoded rollout is COLLAPSED when its forecast frames carry almost no
#: spatial structure (a near-uniform / washed-out dream) — its variation then
#: reflects degenerate dream noise, not plan-driven plasma motion, so it is
#: excluded from the random-vs-random NOISE FLOOR.
COLLAPSE_MIN_STD = 1.5  # 0-255 luminance: forecast-frame spatial std floor
COLLAPSE_MIN_BRIGHTNESS_FRAC = 0.15  # fraction of the GT-scale brightness floor


def _is_collapsed_rollout(
    rollout_px: np.ndarray,
    ctx: int,
    *,
    gt_brightness: float | None = None,
    min_std: float = COLLAPSE_MIN_STD,
    min_brightness_frac: float = COLLAPSE_MIN_BRIGHTNESS_FRAC,
) -> bool:
    """Is this decoded rollout a COLLAPSED dream (near-uniform / near-black)?

    A counterfactual rollout whose decoded forecast frames are near-uniform (low
    spatial std) or far dimmer than the GT scale is a degenerate dream — the model
    gave up and emitted a washed-out / black frame regardless of plan.  Such a
    rollout inflates the random-vs-random floor with dream noise rather than real
    plan-driven variation, so it is dropped from the floor estimate (the collapse
    test).  Returns ``True`` when the rollout collapsed.

    The test is: mean per-forecast-frame SPATIAL std below ``min_std`` (the dream
    has no structure), OR — when a GT brightness scale is supplied — forecast mean
    brightness below ``min_brightness_frac`` of it (the dream is near-black on a
    bright shot).
    """
    from imas_ambix.worldmodel.control_guidance import _to_gray_f64  # noqa: PLC0415

    g = _to_gray_f64(rollout_px)
    if g.shape[0] <= ctx:
        return False
    fwin = g[ctx:]
    spatial_std = float(fwin.reshape(fwin.shape[0], -1).std(axis=1).mean())
    if spatial_std < float(min_std):
        return True
    if gt_brightness is not None and gt_brightness > 0.0:
        mean_bri = float(fwin.mean())
        if mean_bri < float(min_brightness_frac) * float(gt_brightness):
            return True
    return False


# ---------------------------------------------------------------------------
# Verdict containers
# ---------------------------------------------------------------------------


@dataclass
class HeldoutDeltaNMVerdict:
    shot_id: int
    is_transient: bool
    plan_variation: float
    true_vs_random: float  # mean decoded-pixel L1, true plan vs random plans
    random_vs_random: float  # mean pairwise pixel L1 among NON-collapsed randoms
    margin: float
    ratio: float
    n_random: int
    passed: bool
    n_random_collapsed: int = 0  # randoms dropped from the floor (collapse test)
    n_random_kept: int = 0  # randoms kept in the floor estimate

    def to_dict(self) -> dict:
        return {
            "shot_id": self.shot_id,
            "is_transient": self.is_transient,
            "plan_variation": self.plan_variation,
            "true_vs_random": self.true_vs_random,
            "random_vs_random": self.random_vs_random,
            "margin": self.margin,
            "ratio": self.ratio,
            "n_random": self.n_random,
            "n_random_collapsed": self.n_random_collapsed,
            "n_random_kept": self.n_random_kept,
            "passed": self.passed,
        }


@dataclass
class EvalConfig:
    held_out: tuple[int, ...] = DEFAULT_HELD_OUT
    n_random: int = 3
    perturb_scale: float = 0.3
    margin_threshold: float = 1.0  # decoded-pixel L1 (0-255 luminance) units
    floor_ratio: float = 1.5
    transient_threshold: float = 1e-3
    chunk: int = 8192
    n_signal_steps: int = 4
    n_act_steps: int = 8
    seed: int = 0
    window: object = None  # SpacetimeWindowConfig; set by the driver
    modalities: list = field(default_factory=list)
    # --- robust-gate knobs (default = robust cohort + normalised metric) ---
    #: When True (default) the gate normalises by the noise floor (ratio-based,
    #: collapse-rejecting, bootstrap-CI'd) instead of the absolute margin>1.0
    #: pass.  The legacy fixed-shot absolute-margin path is kept behind
    #: ``robust_gate=False`` for back-comparison.
    robust_gate: bool = True
    #: Per-shot ratio (true_vs_random / random_vs_random) a shot must clear.
    ratio_threshold: float = 1.5
    #: Bootstrap resamples for the cohort mean-ratio CI.
    n_bootstrap: int = 2000
    #: Cohort-level CI percentiles (lower, upper).
    ci_pct: tuple[float, float] = (2.5, 97.5)
    #: Reject collapsed random rollouts from the noise floor (the collapse test).
    reject_collapsed: bool = True


# ---------------------------------------------------------------------------
# Assemble a held-out sample (full actuator vector — the model masks itself)
# ---------------------------------------------------------------------------


def _resolve_eval_modalities(mode, payload):
    """Pick the measured-signal stream set the eval conditions + scores on.

    'auto' reads the TRAINED stream set from the checkpoint's
    ``extra.stream_names`` and selects the matching :class:`SignalModalitySpec`
    list from ``extended_signal_modalities`` (a superset of ``default``), so the
    eval conditions on EXACTLY the streams the model trained with — the model was
    starved of magnetics/Dα when this silently defaulted to the 6-stream set.
    'default'/'extended' force the 6- or 13-stream list.  Falls back to the
    checkpoint's model_config ``signal_streams`` names, then to ``extended`` when
    no stream record is present.
    """
    from imas_ambix.worldmodel.spacetime_dataset_v2 import (  # noqa: PLC0415
        default_signal_modalities,
        extended_signal_modalities,
    )

    if mode == "default":
        return default_signal_modalities()
    if mode == "extended":
        return extended_signal_modalities()
    # auto: match the trained stream set recorded in the checkpoint.
    names: list[str] = []
    extra = (payload or {}).get("extra") or {}
    if isinstance(extra.get("stream_names"), (list, tuple)):
        names = [str(n) for n in extra["stream_names"]]
    if not names:
        streams = (payload or {}).get("model_config", {}).get("signal_streams") or []
        names = [str(s.get("name")) for s in streams if isinstance(s, dict)]
    if not names:
        logger.warning(
            "checkpoint records no trained stream set — eval falls back to the "
            "EXTENDED modality list (may mismatch a default-trained model)"
        )
        return extended_signal_modalities()
    want = set(names)
    selected = [m for m in extended_signal_modalities() if m.name in want]
    missing = want - {m.name for m in selected}
    if missing:
        logger.warning(
            "trained streams not in the modality registry: %s", sorted(missing)
        )
    logger.info(
        "eval modalities (auto from checkpoint): %d streams %s",
        len(selected),
        [m.name for m in selected],
    )
    return selected


def _assemble_heldout(shot_id, cfg, *, camera, token_root):
    """Assemble one held-out controllable window at its EXCITED, horizon-spanning span.

    Held-out shots have no curated manifest window, so the eval FINDS the excited
    region itself (find_transient_window) over the per-shot HORIZON span (so the
    scan covers ~target_horizon_s, not a ~15ms slice), then assembles the
    ~target_horizon_s window there.  The per-shot stride comes from the same
    cfg.window.target_horizon_s path assemble_window uses for training, so the eval
    horizon matches training.  Full actuator vector — the model masks Ip/density/tf.
    """
    from imas_ambix.worldmodel.actuator_plan import (  # noqa: PLC0415
        find_transient_window,
    )
    from imas_ambix.worldmodel.controllable_dataset import (  # noqa: PLC0415
        assemble_controllable_window,
    )
    from imas_ambix.worldmodel.spacetime_dataset import (  # noqa: PLC0415
        window_span_for_shot,
    )
    from imas_ambix.worldmodel.spacetime_dataset_v2 import (  # noqa: PLC0415
        default_signal_modalities,
    )

    modalities = cfg.modalities or default_signal_modalities()
    # find the excited window over the HORIZON span (per-shot stride), so the eval
    # scores where the plan actually moves.
    span = window_span_for_shot(
        int(shot_id), cfg.window, camera=camera, token_root=token_root
    )
    start_frame = None
    try:
        sf, _score = find_transient_window(
            int(shot_id), span, camera=camera, token_root=token_root
        )
        start_frame = sf
    except (ValueError, FileNotFoundError, KeyError):
        start_frame = None
    return assemble_controllable_window(
        int(shot_id),
        cfg.window,
        modalities,
        cfg.n_signal_steps,
        cfg.n_act_steps,
        camera=camera,
        token_root=token_root,
        start_frame=start_frame,
    )


def _plan_variation(sample) -> float:
    miss = np.asarray(sample.actuator.missing, dtype=np.float32)
    present = miss.mean(axis=0) < 1.0
    vals = np.asarray(sample.actuator.values, dtype=np.float64)
    if not bool(present.any()) or vals.shape[0] <= 1:
        return 0.0
    return float(np.std(vals[:, present], axis=0).sum())


# ---------------------------------------------------------------------------
# Multi-shot held-out ΔN-M (decoded-pixel)
# ---------------------------------------------------------------------------


def multi_shot_delta_nm(
    model,
    *,
    config: EvalConfig,
    camera: str,
    token_root: Path | None,
    device: str,
    out_json: Path,
    work_dir: Path | None = None,
    decode: bool = True,
) -> dict:
    """Decoded-pixel ΔN-M over the held-out cohort — the real controllability verdict.

    For each cohort shot: roll the model under the TRUE plan + ``n_random``
    BOUNDED coil counterfactuals, decode all rollouts, and score the
    forecast-window decoded-pixel L1 of true-vs-each-random (the action signal)
    against the mean pairwise random-vs-random L1 (the noise floor).

    Under the ROBUST gate (``config.robust_gate=True``, the default) the floor
    EXCLUDES collapsed random dreams (see :func:`_is_collapsed_rollout`) so it
    reflects real plan-driven variation, a shot passes on the noise-floor
    NORMALISED ratio (``ratio > ratio_threshold``), and the cohort gate passes
    when a majority of shots clear the ratio AND the bootstrap CI of the cohort
    mean ratio is clear of 1.0 — so a single GOOD shot can no longer carry the
    gate.  Under ``robust_gate=False`` the legacy absolute-margin pass is used
    (margin>threshold AND ratio>floor_ratio).

    ``decode=False`` scores in TOKEN space instead (a decoder-free lower bound) —
    used by the CPU smoke / when the VQ stack is unavailable.  Writes the verdict
    JSON and returns the summary dict.
    """
    import torch  # noqa: PLC0415

    from imas_ambix.worldmodel.control_falsification import (
        decode_roles,  # noqa: PLC0415
    )
    from imas_ambix.worldmodel.controllable_train import (  # noqa: PLC0415
        _actuator_batch_from_plan,
        _argmax_token_rollout,
        _random_actuator_like,
    )
    from imas_ambix.worldmodel.spacetime_dataset import (  # noqa: PLC0415
        GRID_H,
        GRID_W,
        local_to_store,
    )

    dev = torch.device(device)
    samples = []
    for sid in config.held_out:
        try:
            samples.append(
                _assemble_heldout(sid, config, camera=camera, token_root=token_root)
            )
        except (ValueError, FileNotFoundError, KeyError) as exc:
            logger.warning("held-out shot %s unavailable (%r) — skipped", sid, exc)
    if not samples:
        raise ValueError("no held-out shot could be assembled")

    stream_names = list(samples[0].signals.keys())
    verdicts: list[HeldoutDeltaNMVerdict] = []
    work = Path(work_dir) if work_dir else None

    for s in samples:
        ctx = int(s.context_frames)
        plan_var = _plan_variation(s)
        is_transient = bool(plan_var >= config.transient_threshold)
        rng = np.random.default_rng((int(s.shot_id) * 1_000_003) ^ (config.seed * 31))
        # TRUE-plan rollout (token ids).
        true_tok = _argmax_token_rollout(
            model,
            s,
            stream_names,
            _actuator_batch_from_plan(s.actuator, dev),
            dev,
            chunk=config.chunk,
        )
        # N bounded coil-counterfactual rollouts.
        rand_toks: list[np.ndarray] = []
        for _ in range(int(config.n_random)):
            rplan = _random_actuator_like(
                s.actuator, rng=rng, perturb_scale=config.perturb_scale
            )
            rand_toks.append(
                _argmax_token_rollout(
                    model,
                    s,
                    stream_names,
                    _actuator_batch_from_plan(rplan, dev),
                    dev,
                    chunk=config.chunk,
                )
            )

        n_collapsed = 0
        n_kept = int(config.n_random)
        if decode:
            tvr, rvr, n_collapsed, n_kept = _decoded_divergences(
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
        else:
            tvr, rvr = _token_divergences(true_tok, rand_toks, ctx)

        margin = tvr - rvr
        ratio = float("inf") if rvr == 0.0 else tvr / rvr
        if config.robust_gate:
            # noise-floor-NORMALISED pass: true-vs-random must clear the
            # (collapse-rejected) floor by the ratio threshold.  rvr==0 is only a
            # pass when the true plan actually moved the dream (tvr>0); a flat
            # 0/0 shot is not a controllability win.
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
                passed=passed,
            )
        )

    summary = _summarise(verdicts, config, decode=decode)
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(
            {"per_shot": [v.to_dict() for v in verdicts], "summary": summary},
            indent=2,
            default=str,
        )
    )
    logger.info("held-out ΔN-M verdict -> %s : %s", out_json, summary["verdict"])
    return summary


def _token_divergences(true_tok, rand_toks, ctx):
    """Forecast-window token-mismatch true-vs-random + random-vs-random floor."""

    def fc(a, b):
        if a.shape[0] <= ctx:
            return 0.0
        return float((a[ctx:] != b[ctx:]).mean())

    tvr = float(np.mean([fc(true_tok, r) for r in rand_toks])) if rand_toks else 0.0
    pair = [
        fc(rand_toks[i], rand_toks[j])
        for i in range(len(rand_toks))
        for j in range(i + 1, len(rand_toks))
    ]
    rvr = float(np.mean(pair)) if pair else 0.0
    return tvr, rvr


def _decoded_divergences(
    true_tok,
    rand_toks,
    ctx,
    *,
    device,
    work_dir,
    shot_id,
    grid_hw,
    local_to_store,
    decode_roles,
    reject_collapsed: bool = True,
):
    """Decode the true + random rollouts in ONE VQ pass and score pixel-L1.

    Also decodes the GROUND-TRUTH camera tokens (role ``gt``) so the collapse
    test can compare each random dream's brightness to the real GT scale.  Returns
    ``(tvr, rvr, n_collapsed, n_kept)``: ``tvr`` is the mean forecast-window
    pixel-L1 of the TRUE plan vs the NON-collapsed randoms; ``rvr`` is the mean
    pairwise pixel-L1 among the NON-collapsed randoms (the noise floor).  A random
    rollout whose decoded forecast collapses (near-uniform / near-black, see
    :func:`_is_collapsed_rollout`) is excluded from BOTH so the floor reflects
    real plan-driven variation, not degenerate-dream variation.
    """
    import tempfile  # noqa: PLC0415

    gh, gw = grid_hw
    grids = {"true": local_to_store(true_tok.reshape(-1, gh, gw))}
    roles = [{"role": "true"}]
    for k, rt in enumerate(rand_toks):
        grids[f"rand{k}"] = local_to_store(rt.reshape(-1, gh, gw))
        roles.append({"role": f"rand{k}"})
    wd = Path(work_dir or tempfile.mkdtemp(prefix="m4-heldout-dnm-")) / f"shot{shot_id}"
    decoded = decode_roles(grids, roles, work_dir=wd, device=device)
    true_px = decoded["true"]
    rand_px = [decoded[f"rand{k}"] for k in range(len(rand_toks))]

    # the TRUE-plan dream's brightness is the GT scale proxy here (the true plan
    # tracks the recorded shot); near-black randoms on a bright true dream collapse.
    from imas_ambix.worldmodel.control_guidance import _to_gray_f64  # noqa: PLC0415

    tg = _to_gray_f64(true_px)
    gt_brightness = float(tg[ctx:].mean()) if tg.shape[0] > ctx else None

    kept_idx = list(range(len(rand_px)))
    if reject_collapsed and rand_px:
        kept_idx = [
            i
            for i, r in enumerate(rand_px)
            if not _is_collapsed_rollout(r, ctx, gt_brightness=gt_brightness)
        ]
    n_collapsed = len(rand_px) - len(kept_idx)
    n_kept = len(kept_idx)
    kept = [rand_px[i] for i in kept_idx]

    tvr = (
        float(np.mean([_forecast_pixel_l1(true_px, r, ctx) for r in kept]))
        if kept
        else 0.0
    )
    pair = [
        _forecast_pixel_l1(kept[i], kept[j], ctx)
        for i in range(len(kept))
        for j in range(i + 1, len(kept))
    ]
    rvr = float(np.mean(pair)) if pair else 0.0
    return tvr, rvr, n_collapsed, n_kept


def _bootstrap_mean_ratio_ci(ratios, *, n_boot, ci_pct, seed=0):
    """Bootstrap CI for the cohort MEAN ratio (true_vs_random / floor).

    Resamples the per-shot finite ratios with replacement ``n_boot`` times,
    returns ``(mean, lo, hi)`` of the resampled means at ``ci_pct``.  Infinite
    ratios (a 0.0 floor with a positive true signal) are excluded from the CI math
    but counted separately by the caller — they are unambiguous passes, not a
    finite statistic to bootstrap.
    """
    finite = np.asarray([r for r in ratios if np.isfinite(r)], dtype=np.float64)
    if finite.size == 0:
        return 0.0, 0.0, 0.0
    if finite.size == 1:
        m = float(finite[0])
        return m, m, m
    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, finite.size, size=(int(n_boot), finite.size))
    boot_means = finite[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, list(ci_pct))
    return float(finite.mean()), float(lo), float(hi)


def _summarise(verdicts, config, *, decode):
    transient = [v for v in verdicts if v.is_transient]
    score_set = transient or verdicts
    n_transient = len(transient)
    n_pass = sum(1 for v in score_set if v.passed)
    mean_margin = float(np.mean([v.margin for v in score_set])) if score_set else 0.0
    mean_tvr = (
        float(np.mean([v.true_vs_random for v in score_set])) if score_set else 0.0
    )
    mean_rvr = (
        float(np.mean([v.random_vs_random for v in score_set])) if score_set else 0.0
    )
    pass_fraction = float(n_pass / len(score_set)) if score_set else 0.0
    n_collapsed = sum(int(v.n_random_collapsed) for v in score_set)

    # cohort mean normalised ratio + bootstrap CI (the robust statistic).
    ratios = [v.ratio for v in score_set]
    n_inf = sum(1 for r in ratios if not np.isfinite(r))
    mean_ratio, ci_lo, ci_hi = _bootstrap_mean_ratio_ci(
        ratios, n_boot=config.n_bootstrap, ci_pct=config.ci_pct, seed=config.seed
    )

    if config.robust_gate:
        # ROBUST gate: a majority of cohort shots clear the ratio AND the cohort
        # mean-ratio bootstrap CI lower bound is clear of the noise floor (1.0),
        # so the controllability win is not a 1-shot artifact.
        gate_pass = bool(n_transient > 0 and pass_fraction >= 0.5 and ci_lo > 1.0)
    else:
        gate_pass = bool(
            n_transient > 0
            and n_pass >= max(1, len(score_set) // 2 + 1)
            and mean_margin > config.margin_threshold
        )
    metric = "delta_nm_decoded_pixel" if decode else "delta_nm_token_lowerbound"
    if config.robust_gate:
        metric += "_robust"
    return {
        "metric": metric,
        "robust_gate": bool(config.robust_gate),
        "n_samples": len(verdicts),
        "n_transient": n_transient,
        "n_pass": n_pass,
        "pass_fraction": pass_fraction,
        "mean_true_vs_random": mean_tvr,
        "mean_random_vs_random_noise_floor": mean_rvr,
        "mean_margin": mean_margin,
        "mean_normalised_ratio": mean_ratio,
        "ratio_ci_lo": ci_lo,
        "ratio_ci_hi": ci_hi,
        "ci_pct": list(config.ci_pct),
        "n_bootstrap": int(config.n_bootstrap),
        "n_ratio_infinite": n_inf,
        "n_random_collapsed_total": n_collapsed,
        "reject_collapsed": bool(config.reject_collapsed),
        "margin_threshold": config.margin_threshold,
        "ratio_threshold": config.ratio_threshold,
        "floor_ratio": config.floor_ratio,
        "n_random": int(config.n_random),
        "perturb_scale": config.perturb_scale,
        "decoded": bool(decode),
        "gate_testable": bool(n_transient > 0),
        "gate_pass": gate_pass,
        "verdict": "PASS" if gate_pass else "FAIL",
    }


# ---------------------------------------------------------------------------
# Dreamt-vs-real diagnostic-match (the joint-generation quantitative axis)
# ---------------------------------------------------------------------------


def _stream_diagnostic_logits(model, signal_latents: dict) -> dict:
    """``{name: (B, P, C, d)} -> {name: (B, P, C, vocab)}`` per-stream logits.

    Prefers the model's :meth:`diagnostic_logits`; falls back to applying each
    stream's head directly when the model's helper is unavailable / raises (so the
    eval does not depend on a particular ``nn.ModuleDict`` API — e.g. ``.get`` is
    absent on some torch builds).  Streams with no head are skipped.
    """
    heads = getattr(model, "diagnostic_heads", None)
    if heads is None or not bool(getattr(model, "has_diagnostics", False)):
        return {}
    try:
        return model.diagnostic_logits(signal_latents)
    except (AttributeError, TypeError):
        out: dict = {}
        for name, lat in signal_latents.items():
            if name in heads:
                out[name] = heads[name](lat)
        return out


def diagnostic_match(
    model,
    sample,
    stream_names,
    *,
    device,
    chunk: int = 4096,
) -> dict:
    """How well the model's NEXT-step dreamt diagnostic tokens match the REAL ones.

    The joint world-model grows a per-stream head that predicts the next-step
    measured-signal tokens (it dreams the diagnostics, not just the cameras).  This
    is the quantitative held-out axis for that: a SINGLE teacher-forced forward on
    the real frames + real signals, decoded through the per-stream diagnostic heads,
    scored next-step against the REAL next-step tokens.

    For signal frame ``j`` (0..P-2) the head's logits predict ``signals[name][:,
    j+1]``; PAD (id 0 = an absent / sub-sampled-empty step) targets are masked.  Per
    stream we report the top-1 token accuracy over masked positions and a continuous
    next-step cross-entropy (``ignore_index=0``).  A stream with no scored (non-PAD)
    position is skipped.

    Returns ``{"per_stream": {name: {"accuracy", "ce", "n"}}, "mean_accuracy"
    (macro mean over streams with n>0), "mean_ce", "diagnostics_generated"
    (``model.has_diagnostics``)}``.  When the model generates no diagnostics (a
    camera-only baseline — no / untrained heads) the per-stream map is empty and the
    means are 0.0, with ``diagnostics_generated=False`` so the eval reads the result
    honestly rather than scoring a near-chance head.
    """
    import torch  # noqa: PLC0415
    from torch.nn import functional as F  # noqa: PLC0415, N812

    from imas_ambix.worldmodel.controllable_train import (  # noqa: PLC0415
        _actuator_batch_from_plan,
        _batch_to,
        collate_controllable_windows,
    )
    from imas_ambix.worldmodel.spacetime_train import _AutocastCtx  # noqa: PLC0415

    empty = {
        "per_stream": {},
        "mean_accuracy": 0.0,
        "mean_ce": 0.0,
        "diagnostics_generated": bool(getattr(model, "has_diagnostics", False)),
    }
    if not bool(getattr(model, "has_diagnostics", False)):
        return empty

    dev = torch.device(device)
    model.eval()
    names = list(stream_names)
    batch = _batch_to(collate_controllable_windows([sample], stream_names=names), dev)
    frames = batch["frames"]
    plan = batch.get("plan")
    signals = batch.get("signals") or {}
    actuator = _actuator_batch_from_plan(sample.actuator, dev)
    ctx = int(sample.context_frames)

    with torch.no_grad(), _AutocastCtx(dev):
        out = model._forward_tokens(
            frames,
            plan,
            signals,
            actuator=actuator,
            context_frames=ctx,
            return_signal_latents=True,
        )
        # (cam, sig) — return_latents is off, so the second element is the
        # per-stream signal latents dict.
        _cam, sig_latents = out
        logits = _stream_diagnostic_logits(model, sig_latents)

    per_stream: dict[str, dict] = {}
    accs: list[float] = []
    ces: list[float] = []
    for name, lg in logits.items():
        tok = signals.get(name)
        if tok is None:
            continue
        # lg: (B, P, C, V); tok: (B, P, C) long local ids.
        p = min(int(lg.shape[1]), int(tok.shape[1]))
        c = min(int(lg.shape[2]), int(tok.shape[2]))
        if p < 2 or c < 1:
            continue
        # frame j (0..P-2) predicts frame j+1.
        pred_logits = lg[:, : p - 1, :c, :].float()  # (B, P-1, C, V)
        pred = pred_logits.argmax(dim=-1)  # (B, P-1, C)
        target = tok[:, 1:p, :c].to(dev).long()  # (B, P-1, C)
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


def multi_shot_diagnostic_match(
    model,
    *,
    config: EvalConfig,
    camera: str,
    token_root: Path | None,
    device: str,
    out_json: Path,
) -> dict:
    """Dreamt-vs-real diagnostic-match over the held-out shots + a JSON summary.

    Mirrors :func:`multi_shot_delta_nm`: assembles each held-out window, scores its
    next-step diagnostic-match (:func:`diagnostic_match`), and aggregates a summary
    (mean accuracy + mean CE across shots, and a per-stream macro breakdown over the
    shots that scored each stream).  Writes ``out_json`` and returns the summary.
    """
    import torch  # noqa: PLC0415

    dev = torch.device(device)
    samples = []
    for sid in config.held_out:
        try:
            samples.append(
                _assemble_heldout(sid, config, camera=camera, token_root=token_root)
            )
        except (ValueError, FileNotFoundError, KeyError) as exc:
            logger.warning("held-out shot %s unavailable (%r) — skipped", sid, exc)
    if not samples:
        raise ValueError("no held-out shot could be assembled")

    stream_names = list(samples[0].signals.keys())
    per_shot: list[dict] = []
    # accumulate per-stream accuracy/CE across shots for a macro breakdown.
    stream_acc: dict[str, list[float]] = {}
    stream_ce: dict[str, list[float]] = {}
    for s in samples:
        res = diagnostic_match(
            model, s, stream_names, device=str(dev), chunk=config.chunk
        )
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
    diagnostics_generated = bool(getattr(model, "has_diagnostics", False))
    summary = {
        "metric": "diagnostic_match_next_step",
        "diagnostics_generated": diagnostics_generated,
        "n_samples": len(per_shot),
        "n_scored": len(scored),
        "mean_accuracy": mean_acc,
        "mean_ce": mean_ce,
        "per_stream": per_stream_macro,
    }
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps({"per_shot": per_shot, "summary": summary}, indent=2, default=str)
    )
    logger.info(
        "held-out diagnostic-match -> %s : mean_acc=%.4f mean_ce=%.4f (generated=%s)",
        out_json,
        mean_acc,
        mean_ce,
        diagnostics_generated,
    )
    return summary


# ---------------------------------------------------------------------------
# Interpretable coil-edit dream (the "play the plasma" artifact)
# ---------------------------------------------------------------------------


def _bounded_coil_edit(plan, coil_col: int, *, frac: float):
    """A copy of ``plan`` with ONE coil's command bounded-edited by ``frac``.

    Scales the coil's RAW trajectory to ``(1 + frac)`` (a position coil pushed
    harder/softer), preserving its temporal shape — an interpretable, physically
    plausible single-actuator intervention.  ``missing`` is preserved.
    """
    from imas_ambix.worldmodel.actuator_plan import (  # noqa: PLC0415
        ActuatorPlan,
        normalise_actuator_values,
    )

    raw = np.asarray(plan.raw_values, dtype=np.float64).copy()
    if 0 <= coil_col < raw.shape[1]:
        raw[:, coil_col] = raw[:, coil_col] * (1.0 + float(frac))
    return ActuatorPlan(
        values=normalise_actuator_values(raw),
        missing=plan.missing.copy(),
        channel_keys=list(plan.channel_keys),
        raw_values=raw.astype(np.float32),
    )


def _position_coil_columns(channel_keys: Sequence[str]) -> list[int]:
    """Columns of the PF POSITION coils — the p4/p5/p6 vertical/radial-field set.

    These shape/position coils are the most VISUALLY interpretable to edit (they
    move the plasma up/down / in/out).  Falls back to all coil columns if none of
    the p4/p5/p6 keys are present.  KEY-based against the plan's channel_keys.
    """
    from imas_ambix.worldmodel.actuator_plan import (  # noqa: PLC0415
        ACTUATOR_CHANNEL_KEYS,
        coil_current_channel_indices,
    )

    def _is_coil(key: str) -> bool:
        return "coil" in key or key.endswith("_current")

    pos = [
        i
        for i, k in enumerate(channel_keys)
        if _is_coil(k) and any(tag in k for tag in ("p4", "p5", "p6"))
    ]
    if pos:
        return pos
    # fallback: any coil column present (by the real coil-channel key set).
    coil_keys = {
        ACTUATOR_CHANNEL_KEYS[i]
        for i in coil_current_channel_indices()
        if i < len(ACTUATOR_CHANNEL_KEYS)
    }
    return [i for i, k in enumerate(channel_keys) if k in coil_keys]


def coil_edit_dream(
    model,
    *,
    config: EvalConfig,
    camera: str,
    token_root: Path | None,
    device: str,
    out_dir: Path,
    shot_id: int | None = None,
    edit_frac: float = 0.3,
    fps: int = 8,
) -> dict:
    """Render the headline "play the plasma" artifact: an interpretable coil edit.

    Picks the PF POSITION coil whose bounded edit most moves the decoded plasma
    CENTROID on the chosen held-out shot, then writes a side-by-side GIF (TRUE
    plan dream | edited-coil dream — the plasma visibly shifting) + a centroid-
    trace PNG + a small JSON with the per-coil centroid displacements.  Returns the
    artifact paths + the winning coil + its centroid shift.

    All candidate coil edits + the true plan are decoded in ONE VQ pass.  Requires
    the decode stack (GPU + the frozen VQ); raises if unavailable.
    """
    import torch  # noqa: PLC0415

    from imas_ambix.worldmodel.actuator_plan import (  # noqa: PLC0415
        ACTUATOR_CHANNEL_KEYS,
    )
    from imas_ambix.worldmodel.control_falsification import (
        decode_roles,  # noqa: PLC0415
    )
    from imas_ambix.worldmodel.controllable_train import (  # noqa: PLC0415
        _actuator_batch_from_plan,
        _argmax_token_rollout,
    )
    from imas_ambix.worldmodel.spacetime_dataset import (  # noqa: PLC0415
        GRID_H,
        GRID_W,
        local_to_store,
    )

    dev = torch.device(device)
    sid = int(shot_id if shot_id is not None else config.held_out[0])
    sample = _assemble_heldout(sid, config, camera=camera, token_root=token_root)
    stream_names = list(sample.signals.keys())
    ctx = int(sample.context_frames)

    coil_cols = _position_coil_columns(list(sample.actuator.channel_keys))
    present = np.asarray(sample.actuator.missing, dtype=np.float32).mean(axis=0) < 1.0
    coil_cols = [c for c in coil_cols if c < present.shape[0] and present[c]]
    if not coil_cols:
        raise ValueError(f"no present PF position coil on shot {sid}")

    # roll out the TRUE plan + one edit per candidate coil; decode all in one pass.
    true_tok = _argmax_token_rollout(
        model,
        sample,
        stream_names,
        _actuator_batch_from_plan(sample.actuator, dev),
        dev,
        chunk=config.chunk,
    )
    grids = {"true": local_to_store(true_tok.reshape(-1, GRID_H, GRID_W))}
    roles = [{"role": "true"}]
    edit_tok: dict[int, np.ndarray] = {}
    for c in coil_cols:
        et = _argmax_token_rollout(
            model,
            sample,
            stream_names,
            _actuator_batch_from_plan(
                _bounded_coil_edit(sample.actuator, c, frac=edit_frac), dev
            ),
            dev,
            chunk=config.chunk,
        )
        edit_tok[c] = et
        grids[f"coil{c}"] = local_to_store(et.reshape(-1, GRID_H, GRID_W))
        roles.append({"role": f"coil{c}"})

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    decoded = decode_roles(grids, roles, work_dir=out_dir / "_decode", device=device)

    true_px = decoded["true"]
    true_cen = decoded_centroid(true_px)
    # pick the coil whose edit shifts the forecast-window centroid most.
    shifts: dict[int, float] = {}
    for c in coil_cols:
        cen = decoded_centroid(decoded[f"coil{c}"])
        d = float(np.linalg.norm(cen[ctx:] - true_cen[ctx:], axis=1).mean())
        shifts[c] = d
    best = max(shifts, key=shifts.get)
    best_key = (
        ACTUATOR_CHANNEL_KEYS[best]
        if best < len(ACTUATOR_CHANNEL_KEYS)
        else f"col{best}"
    )
    logger.info(
        "coil-edit dream shot %s: best coil %s (col %d) centroid shift %.2f px",
        sid,
        best_key,
        best,
        shifts[best],
    )

    paths = _render_coil_edit_artifacts(
        true_px,
        decoded[f"coil{best}"],
        true_cen,
        decoded_centroid(decoded[f"coil{best}"]),
        ctx=ctx,
        out_dir=out_dir,
        shot_id=sid,
        coil_key=best_key,
        edit_frac=edit_frac,
        fps=fps,
    )
    result = {
        "shot_id": sid,
        "best_coil_key": best_key,
        "best_coil_col": int(best),
        "best_centroid_shift_px": shifts[best],
        "per_coil_centroid_shift_px": {
            ACTUATOR_CHANNEL_KEYS[c] if c < len(ACTUATOR_CHANNEL_KEYS) else f"col{c}": v
            for c, v in shifts.items()
        },
        "edit_frac": edit_frac,
        **paths,
    }
    (out_dir / f"coil_edit_shot{sid}.json").write_text(
        json.dumps(result, indent=2, default=str)
    )
    return result


def _render_coil_edit_artifacts(
    true_px,
    edit_px,
    true_cen,
    edit_cen,
    *,
    ctx,
    out_dir,
    shot_id,
    coil_key,
    edit_frac,
    fps,
):
    """Write the side-by-side dream GIF + the centroid-trace PNG."""
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    from imas_ambix.worldmodel.control_guidance import _to_gray_f64  # noqa: PLC0415
    from imas_ambix.worldmodel.dream_gifs import (  # noqa: PLC0415
        _panel_frame,
        _save_gif,
    )

    out_dir = Path(out_dir)
    gif_path = out_dir / f"coil_edit_shot{shot_id}_{coil_key}.gif"
    n = min(true_px.shape[0], edit_px.shape[0])
    frames = []
    gt = _to_gray_f64(true_px)
    ed = _to_gray_f64(edit_px)
    for i in range(n):
        frames.append(
            _panel_frame(
                gt[i],
                ed[i],
                left_title="true plan",
                right_title=f"{coil_key} x{1 + edit_frac:.2f}",
                banner=f"shot {shot_id} — play the plasma: edit {coil_key} (frame {i})",
                in_target=(i >= ctx),
            )
        )
    _save_gif(frames, gif_path, fps=fps)

    # centroid trace: row/col of the plasma centroid, true vs edited, over forecast.
    png_path = out_dir / f"coil_edit_shot{shot_id}_{coil_key}_centroid.png"
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2), dpi=110)
    t = np.arange(true_cen.shape[0])
    dims = ((axes[0], 0, "row (vertical)"), (axes[1], 1, "col (radial)"))
    for ax, dim, name in dims:
        ax.plot(t, true_cen[:, dim], "-o", ms=3, label="true plan", color="#1f77b4")
        ax.plot(
            t, edit_cen[:, dim], "-o", ms=3, label=f"{coil_key} edit", color="#d62728"
        )
        ax.axvline(ctx - 0.5, ls="--", color="#888", lw=1)
        ax.set_title(f"centroid {name}", fontsize=10)
        ax.set_xlabel("frame")
        ax.set_ylabel("pixel")
        ax.legend(fontsize=8)
    fig.suptitle(
        f"shot {shot_id}: plasma centroid responds to a {coil_key} edit "
        f"(dashed = forecast start)",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(png_path)
    plt.close(fig)
    return {"gif_path": str(gif_path), "centroid_png_path": str(png_path)}


# ---------------------------------------------------------------------------
# CLI — run the instant the re-train writes best.pt
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    import argparse  # noqa: PLC0415
    import logging as _logging  # noqa: PLC0415

    import torch  # noqa: PLC0415

    from imas_ambix.worldmodel.controllable_train import (  # noqa: PLC0415
        load_controllable_model_from_checkpoint,
    )
    from imas_ambix.worldmodel.spacetime_dataset import (  # noqa: PLC0415
        SpacetimeWindowConfig,
    )

    p = argparse.ArgumentParser(description="Held-out controllability eval.")
    p.add_argument("--checkpoint", required=True, help="the re-train best.pt")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--camera", default="rbb")
    p.add_argument("--token-root", default=None)
    p.add_argument("--held-out", default="18502,18503,18504,18505")
    p.add_argument("--device", default="cuda")
    p.add_argument("--n-random", type=int, default=3)
    p.add_argument("--perturb-scale", type=float, default=0.3)
    p.add_argument("--margin-threshold", type=float, default=1.0)
    p.add_argument("--floor-ratio", type=float, default=1.5)
    p.add_argument("--n-frames", type=int, default=24)
    p.add_argument("--n-plan", type=int, default=8)
    p.add_argument("--context-frames", type=int, default=8)
    p.add_argument("--frame-stride", type=int, default=1)
    p.add_argument(
        "--target-horizon-s",
        type=float,
        default=0.25,
        help="physical seconds the window spans — MUST match the training run so "
        "the eval conditions on the same horizon",
    )
    p.add_argument("--n-signal-steps", type=int, default=4)
    p.add_argument("--n-act-steps", type=int, default=8)
    p.add_argument("--chunk", type=int, default=8192)
    p.add_argument("--edit-frac", type=float, default=0.3)
    p.add_argument(
        "--no-decode",
        action="store_true",
        help="score the ΔN-M in TOKEN space (decoder-free lower bound); skips the "
        "dream GIF.  Used when the VQ stack is unavailable.",
    )
    p.add_argument(
        "--no-dream",
        action="store_true",
        help="skip the coil-edit dream GIF (run only the ΔN-M verdict)",
    )
    p.add_argument(
        "--no-diagnostic-match",
        action="store_true",
        help="skip the dreamt-vs-real next-step diagnostic-match metric",
    )
    p.add_argument(
        "--signal-modalities",
        choices=("auto", "default", "extended"),
        default="auto",
        help="which measured-signal streams to CONDITION + score on. 'auto' "
        "(default) reads the trained stream set from the checkpoint's "
        "extra.stream_names so the eval matches training EXACTLY (the model "
        "was starved of magnetics/Dα when this defaulted to the 6-stream set); "
        "'default'/'extended' force the 6- or 13-stream list.",
    )
    # --- robust-gate / screened-cohort flags (default = robust gate) ---
    g = p.add_mutually_exclusive_group()
    g.add_argument(
        "--robust-gate",
        dest="robust_gate",
        action="store_true",
        default=True,
        help="(default) noise-floor-NORMALISED, collapse-rejecting gate over a "
        "screened cohort: per-shot ratio>ratio-threshold, cohort gate needs a "
        "majority pass AND a bootstrap-CI lower bound clear of 1.0.",
    )
    g.add_argument(
        "--no-robust-gate",
        dest="robust_gate",
        action="store_false",
        help="legacy fixed-shot ABSOLUTE-margin gate (margin>threshold AND "
        "ratio>floor-ratio) for back-comparison.",
    )
    p.add_argument("--ratio-threshold", type=float, default=1.5)
    p.add_argument("--n-bootstrap", type=int, default=2000)
    p.add_argument(
        "--no-reject-collapsed",
        dest="reject_collapsed",
        action="store_false",
        default=True,
        help="keep collapsed random dreams in the noise floor (debug; the floor "
        "then reflects degenerate-dream noise, not plan-driven variation).",
    )
    p.add_argument(
        "--held-out-cohort",
        default=None,
        help="path to a screened-cohort JSON (built by gate_cohort.build_screened_"
        "cohort); its shot ids REPLACE --held-out for the robust gate.",
    )
    p.add_argument(
        "--build-cohort",
        action="store_true",
        help="screen + write a fresh train-disjoint eval-only cohort (needs "
        "--manifest) before the eval, then run the gate on it.",
    )
    p.add_argument(
        "--manifest",
        default="/work/projects/imas_gpu/agents/excitation-corpus/"
        "curated_windows_unified_6cam.json",
        help="training manifest the cohort must be DISJOINT from (--build-cohort).",
    )
    p.add_argument("--cohort-cap", type=int, default=60)
    p.add_argument("--cohort-target", type=int, default=30)
    p.add_argument(
        "--cohort-out",
        default="/work/projects/imas_gpu/worldmodel/gate_cohort.json",
        help="where --build-cohort writes the screened cohort JSON.",
    )
    args = p.parse_args(argv)

    _logging.basicConfig(
        level=_logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
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
        model, _payload = load_controllable_model_from_checkpoint(
            Path(args.checkpoint), map_location=device
        )
        model.eval()
        modalities = _resolve_eval_modalities(args.signal_modalities, _payload)
        window = SpacetimeWindowConfig(
            n_frames=args.n_frames,
            n_plan=args.n_plan,
            context_frames=args.context_frames,
            frame_stride=args.frame_stride,
            target_horizon_s=args.target_horizon_s,
        )
        token_root = Path(args.token_root) if args.token_root else None

        # resolve the cohort: build a fresh screened cohort, load one, or fall
        # back to the legacy fixed --held-out list.
        held_out = tuple(int(s) for s in args.held_out.split(",") if s.strip())
        if args.build_cohort:
            from imas_ambix.worldmodel.gate_cohort import (  # noqa: PLC0415
                build_screened_cohort,
                load_cohort,
            )

            screen_cfg = EvalConfig(
                n_signal_steps=args.n_signal_steps,
                n_act_steps=args.n_act_steps,
                modalities=modalities,
                window=window,
            )
            build_screened_cohort(
                screen_cfg,
                camera=args.camera,
                token_root=token_root or Path("/work/projects/imas_gpu/mast-tokens"),
                manifest_path=args.manifest,
                device=device,
                out_json=args.cohort_out,
                candidate_cap=args.cohort_cap,
                target_size=args.cohort_target,
                work_dir=out_dir / "_cohort_screen",
            )
            held_out = tuple(load_cohort(args.cohort_out))
            logger.info("cohort built: %d shots %s", len(held_out), list(held_out))
        elif args.held_out_cohort:
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
        summary = multi_shot_delta_nm(
            model,
            config=cfg,
            camera=args.camera,
            token_root=token_root,
            device=device,
            out_json=out_dir / "heldout_delta_nm.json",
            work_dir=out_dir / "_dnm",
            decode=not args.no_decode,
        )
        logger.info("HELD-OUT ΔN-M: %s", summary)
        if not args.no_diagnostic_match:
            try:
                dmatch = multi_shot_diagnostic_match(
                    model,
                    config=cfg,
                    camera=args.camera,
                    token_root=token_root,
                    device=device,
                    out_json=out_dir / "heldout_diagnostic_match.json",
                )
                logger.info("HELD-OUT diagnostic-match: %s", dmatch)
            except ValueError as exc:
                logger.warning("diagnostic-match skipped (%r)", exc)
        if not args.no_dream and not args.no_decode:
            dream = coil_edit_dream(
                model,
                config=cfg,
                camera=args.camera,
                token_root=token_root,
                device=device,
                out_dir=out_dir,
                edit_frac=args.edit_frac,
            )
            logger.info("DREAM artifact: %s", dream.get("gif_path"))
    finally:
        try:
            del model
            if device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001
            logger.warning("model release note: %r", exc)
    return 0


__all__ = [
    "COLLAPSE_MIN_BRIGHTNESS_FRAC",
    "COLLAPSE_MIN_STD",
    "DEFAULT_HELD_OUT",
    "EvalConfig",
    "HeldoutDeltaNMVerdict",
    "coil_edit_dream",
    "decoded_centroid",
    "diagnostic_match",
    "main",
    "multi_shot_delta_nm",
    "multi_shot_diagnostic_match",
]


if __name__ == "__main__":
    raise SystemExit(main())
