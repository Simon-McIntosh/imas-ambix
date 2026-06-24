"""Decoder-free DREAMED-MAGNETICS command-sensitivity (the go/no-go gate).

This is the DECISIVE, cheap go/no-go that gates a multi-hour absolute
re-tokenise + retrain: does an existing world model's DREAMED MAGNETICS respond
to the actuator command?  It is measured ENTIRELY in diagnostic-token space —
no camera decode, no VQ stack, so it is fast and CPU-runnable.

The camera-pixel ΔN-M gate (controllable_eval / rssm_eval) found the camera
DREAM washed out the command signal.  But both models grow per-stream
DIAGNOSTIC heads (joint generation): they dream the magnetics / interferometer /
soft_x_rays / Dα tokens too.  The hypothesis is that those diagnostic heads may
PRESERVE the command-sensitive latent even where the camera decoder lost it.
MAGNETICS is the physics-gate decision driver; the others are reported for
context.

For each gate-cohort shot, this rolls the world model under the TRUE actuator
plan and N WRONG / random plans (the same bounded coil counterfactuals the
camera gate uses — :func:`controllable_train._random_actuator_like`, which
HOLDS the masked state columns 13/14/22), extracts the DREAMED per-stream
diagnostic tokens (argmax of each per-stream diagnostic head) over the FORECAST
window, and computes a diagnostic-token-space ΔN-M:

* ``true_vs_random`` — mean forecast-window token-mismatch fraction between the
  TRUE-plan dreamed stream tokens and each random-plan one (the action signal),
* ``random_vs_random`` — mean PAIRWISE token-mismatch among the random-plan
  dreamed stream tokens (the noise floor),
* ``ratio = true_vs_random / random_vs_random`` per shot; the cohort reports the
  mean + per-shot ratios.

``ratio`` meaningfully > 1 for MAGNETICS ⇒ the dreamed magnetics are
command-sensitive ⇒ GO (the physics-gate path is worth the retrain).  ``ratio``
~ 1 ⇒ the dream ignores the command ⇒ NO-GO.

Two rollout paths, one per model family:

* ``rssm`` — :meth:`RSSMWorldModel.rollout_prior` returns ``.diagnostics``
  (``{name: (B, n_steps, C, vocab)}``) directly; the command lives inside the
  recurrent transition so a different plan IS a different rollout.
* ``token`` — the joint-gen ControllableSpacetime transformer: roll the camera
  frames forward with :func:`controllable_train._argmax_token_rollout` under the
  plan, then a single forward over the ROLLED window with
  ``return_signal_latents=True`` → :meth:`diagnostic_logits` → argmax gives the
  dreamed diagnostic tokens for the forecast frames (the command flows through
  the rolled camera frames AND the AdaLN actuator modulation).

GPU-safety (AGENTS.md §2b): each model is loaded ONCE; SIGTERM-clean; the work
is decoder-free; ``try/finally`` releases the model + empties the CUDA cache.
This script imports from the existing eval / train modules and does NOT edit
them.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import signal as _signal
from pathlib import Path

import numpy as np

logger = logging.getLogger("diag_space_controllability")

#: cooperative-stop flag a SIGTERM/SIGINT handler sets (AGENTS.md §2b).
_STOP = {"flag": False}

#: the magnetics stream is the physics-gate decision driver; the others are
#: reported for context (interferometer / soft x-rays / the three Dα families).
DECISION_STREAM = "magnetics"
CONTEXT_STREAMS = ("interferometer", "soft_x_rays", "ada", "adg", "aim")


def _install_stop_handler() -> None:
    def _on_signal(signum, _frame):  # noqa: ANN001
        logger.warning("received signal %s — stopping after this shot", signum)
        _STOP["flag"] = True

    for sig in (_signal.SIGTERM, _signal.SIGINT):
        with contextlib.suppress(ValueError, OSError):
            _signal.signal(sig, _on_signal)


# ---------------------------------------------------------------------------
# Token-mismatch divergence over the forecast window (decoder-free)
# ---------------------------------------------------------------------------


def _forecast_token_mismatch(a: np.ndarray, b: np.ndarray) -> float:
    """Mean element-wise token-mismatch fraction between two ``(P, C)`` dreams.

    Both arrays are already restricted to the FORECAST steps and aligned in
    shape; PAD-id-0 positions in EITHER are excluded (an absent / sub-sampled
    step is not a dreamt prediction).  Returns 0.0 when there is no comparable
    position.
    """
    if a.size == 0 or b.size == 0:
        return 0.0
    p = min(a.shape[0], b.shape[0])
    c = min(a.shape[1], b.shape[1]) if a.ndim == 2 else 1
    aa = a[:p, :c].reshape(p, -1)
    bb = b[:p, :c].reshape(p, -1)
    valid = (aa != 0) & (bb != 0)
    n = int(valid.sum())
    if n == 0:
        return 0.0
    return float((aa[valid] != bb[valid]).mean())


def _stream_delta_nm(true_tok: np.ndarray, rand_toks: list[np.ndarray]) -> dict:
    """diagnostic-token-space ΔN-M for ONE stream on ONE shot.

    ``true_tok`` / each ``rand_toks[i]`` is the dreamed ``(P, C)`` forecast-window
    stream tokens under the TRUE / a random plan.  Returns the per-shot
    true_vs_random, random_vs_random floor, margin, ratio + the sample lists.
    """
    tvr_samples = [_forecast_token_mismatch(true_tok, r) for r in rand_toks]
    rvr_samples = [
        _forecast_token_mismatch(rand_toks[i], rand_toks[j])
        for i in range(len(rand_toks))
        for j in range(i + 1, len(rand_toks))
    ]
    tvr = float(np.mean(tvr_samples)) if tvr_samples else 0.0
    rvr = float(np.mean(rvr_samples)) if rvr_samples else 0.0
    ratio = float("inf") if rvr == 0.0 else tvr / rvr
    return {
        "true_vs_random": tvr,
        "random_vs_random": rvr,
        "margin": tvr - rvr,
        "ratio": ratio,
        "true_vs_random_samples": [float(x) for x in tvr_samples],
        "random_vs_random_samples": [float(x) for x in rvr_samples],
    }


def _cohort_summary(per_shot_ratios: list[float]) -> dict:
    """Cohort mean / median / pass-fraction over per-shot ratios (inf-safe)."""
    finite = [r for r in per_shot_ratios if np.isfinite(r)]
    n_inf = sum(1 for r in per_shot_ratios if not np.isfinite(r))
    mean_ratio = float(np.mean(finite)) if finite else float("nan")
    median_ratio = float(np.median(finite)) if finite else float("nan")
    # a per-shot ratio > 1.0 means the true plan moved the dream more than a
    # different plan moved it from another — command-sensitive on that shot.
    n_pass = sum(
        1 for r in per_shot_ratios if (np.isfinite(r) and r > 1.0) or not np.isfinite(r)
    )
    n_scored = len(per_shot_ratios)
    return {
        "n_scored": n_scored,
        "n_ratio_infinite": n_inf,
        "mean_ratio": mean_ratio,
        "median_ratio": median_ratio,
        "n_pass_ratio_gt_1": n_pass,
        "pass_fraction": float(n_pass / n_scored) if n_scored else 0.0,
        "per_shot_ratios": [float(r) for r in per_shot_ratios],
    }


# ---------------------------------------------------------------------------
# Per-model dreamed-diagnostic extraction (decoder-free)
# ---------------------------------------------------------------------------


def _rssm_dream_diag_tokens(model, sample, plan, device, *, chunk: int) -> dict:
    """Dreamed per-stream FORECAST diagnostic tokens under ``plan`` (RSSM).

    Warms the recurrent state on the GT context frames, rolls the PRIOR forward
    under the commands, and argmaxes the ``rollout_prior(...).diagnostics``
    logits ``{name: (B, n_steps, C, vocab)}`` to ``{name: (n_steps, C)}`` LOCAL
    token ids — the dreamt diagnostics for the FORECAST window only (no
    observations consumed past the context).
    """
    import torch  # noqa: PLC0415

    from imas_ambix.worldmodel.controllable_train import (  # noqa: PLC0415
        _actuator_batch_from_plan,
    )
    from imas_ambix.worldmodel.spacetime_train import _AutocastCtx  # noqa: PLC0415

    ctx = int(sample.context_frames)
    frames = np.asarray(sample.frames, dtype=np.int64)
    n_steps = int(frames.shape[0]) - ctx
    if n_steps <= 0:
        return {}
    context = torch.as_tensor(frames[:ctx][None], dtype=torch.long, device=device)
    actuator = _actuator_batch_from_plan(plan, device)
    with torch.no_grad(), _AutocastCtx(device):
        rollout = model.rollout_prior(
            context, actuator, n_steps, chunk=chunk, sample=False
        )
    out: dict[str, np.ndarray] = {}
    for name, lg in rollout.diagnostics.items():
        # lg: (1, n_steps, C, vocab) -> (n_steps, C) argmax local ids.
        tok = lg[0].argmax(dim=-1).cpu().numpy().astype(np.int64)
        out[name] = tok
    return out


def _token_dream_diag_tokens(
    model, sample, stream_names, plan, device, *, chunk: int
) -> dict:
    """Dreamed per-stream FORECAST diagnostic tokens under ``plan`` (token model).

    Rolls the CAMERA frames forward autoregressively under ``plan`` (
    :func:`_argmax_token_rollout`), then runs ONE forward over the rolled window
    with ``return_signal_latents=True`` and argmaxes :meth:`diagnostic_logits`.
    The command flows through (a) the rolled camera frames differing per plan and
    (b) the AdaLN actuator modulation.  The REAL signals are held as the head
    input (we score the head's PREDICTIONS, not a re-rolled signal stream).
    Returns ``{name: (P_fore, C)}`` LOCAL token ids over the FORECAST signal
    frames (signal frame ``j`` predicts ``j+1``; we keep frames that fall in the
    camera forecast window).
    """
    import torch  # noqa: PLC0415

    from imas_ambix.worldmodel.controllable_train import (  # noqa: PLC0415
        _actuator_batch_from_plan,
        _argmax_token_rollout,
        _batch_to,
        collate_controllable_windows,
    )
    from imas_ambix.worldmodel.spacetime_train import _AutocastCtx  # noqa: PLC0415

    dev = torch.device(device)
    ctx = int(sample.context_frames)
    names = list(stream_names)
    act_batch = _actuator_batch_from_plan(plan, dev)

    # roll the camera frames forward under THIS plan (decoder-free token ids).
    rolled = _argmax_token_rollout(model, sample, names, act_batch, dev, chunk=chunk)
    rolled_t = torch.as_tensor(rolled[None], dtype=torch.long, device=dev)

    batch = _batch_to(collate_controllable_windows([sample], stream_names=names), dev)
    plan_tok = batch.get("plan")
    signals = batch.get("signals") or {}

    with torch.no_grad(), _AutocastCtx(dev):
        out = model._forward_tokens(
            rolled_t,
            plan_tok,
            signals,
            actuator=act_batch,
            context_frames=ctx,
            return_signal_latents=True,
        )
        _cam, sig_latents = out
        logits = model.diagnostic_logits(sig_latents)

    # signal frames map onto the camera-frame axis at n_signal_steps cadence; the
    # head at signal-frame j predicts j+1.  We keep the next-step predictions and
    # restrict to the FORECAST portion of the signal axis (signal frames whose
    # next-step lands at or after the camera context boundary).  The signal step
    # count is small (n_signal_steps), so we keep all next-step predictions whose
    # SOURCE frame is in the back half (forecast-aligned) of the signal window.
    res: dict[str, np.ndarray] = {}
    for name, lg in logits.items():
        # lg: (1, P, C, V).  next-step prediction = argmax over the head outputs at
        # frames 0..P-2 (predicting 1..P-1).
        p = int(lg.shape[1])
        if p < 2:
            continue
        pred = lg[0, : p - 1].argmax(dim=-1).cpu().numpy().astype(np.int64)  # (P-1, C)
        # forecast portion of the signal axis: the signal frames are sub-sampled
        # uniformly across the camera window, so the back (P-1)*(forecast/total)
        # signal next-steps fall in the camera forecast window.  Keep the back
        # half (the smallest robust forecast-aligned slice for a short signal
        # axis); if P-1 == 1 keep it.
        t_total = int(np.asarray(sample.frames).shape[0])
        keep = max(1, int(round((p - 1) * (t_total - ctx) / max(t_total, 1))))
        res[name] = pred[-keep:]
    return res


# ---------------------------------------------------------------------------
# Per-model cohort sweep
# ---------------------------------------------------------------------------


def _run_model(
    kind: str,
    checkpoint: Path,
    *,
    cohort: list[int],
    camera: str,
    token_root: Path | None,
    device: str,
    n_random: int,
    perturb_scale: float,
    chunk: int,
    window_kwargs: dict,
    out_dir: Path,
) -> dict:
    """Sweep one checkpoint over the cohort; return the per-stream ΔN-M summary."""
    import torch  # noqa: PLC0415

    from imas_ambix.worldmodel.controllable_eval import (  # noqa: PLC0415
        EvalConfig,
        _assemble_heldout,
        _plan_variation,
        _resolve_eval_modalities,
    )
    from imas_ambix.worldmodel.controllable_train import (  # noqa: PLC0415
        _random_actuator_like,
    )
    from imas_ambix.worldmodel.spacetime_dataset import (  # noqa: PLC0415
        SpacetimeWindowConfig,
    )

    dev = torch.device(device)
    model = None
    try:
        if kind == "rssm":
            from imas_ambix.worldmodel.rssm_eval import (  # noqa: PLC0415
                load_rssm_from_checkpoint,
            )

            model, payload = load_rssm_from_checkpoint(checkpoint, map_location=device)
            has_diag = bool(getattr(model.config, "has_diagnostics", False))
        else:
            from imas_ambix.worldmodel.controllable_train import (  # noqa: PLC0415
                load_controllable_model_from_checkpoint,
            )

            model, payload = load_controllable_model_from_checkpoint(
                checkpoint, map_location=device
            )
            has_diag = bool(getattr(model, "has_diagnostics", False))
        model.eval()
        logger.info("[%s] loaded %s (has_diagnostics=%s)", kind, checkpoint, has_diag)

        modalities = _resolve_eval_modalities("auto", payload)
        window = SpacetimeWindowConfig(**window_kwargs)
        cfg = EvalConfig(
            held_out=tuple(cohort),
            n_random=n_random,
            perturb_scale=perturb_scale,
            chunk=chunk,
            n_signal_steps=4,
            n_act_steps=8,
            modalities=modalities,
            window=window,
        )

        # accumulate per-stream per-shot ratios + records.
        per_shot: list[dict] = []
        stream_ratios: dict[str, list[float]] = {}
        stream_names_ref: list[str] | None = None

        for sid in cohort:
            if _STOP["flag"]:
                break
            try:
                sample = _assemble_heldout(
                    sid, cfg, camera=camera, token_root=token_root
                )
            except (ValueError, FileNotFoundError, KeyError) as exc:
                logger.warning("shot %s unavailable (%r) — skipped", sid, exc)
                continue
            if stream_names_ref is None:
                stream_names_ref = list(sample.signals.keys())

            plan_var = _plan_variation(sample)
            rng = np.random.default_rng(int(sid) * 1_000_003)
            rand_plans = [
                _random_actuator_like(
                    sample.actuator, rng=rng, perturb_scale=perturb_scale
                )
                for _ in range(int(n_random))
            ]

            # dream the diagnostics under the TRUE + each random plan.
            if kind == "rssm":
                true_d = _rssm_dream_diag_tokens(
                    model, sample, sample.actuator, dev, chunk=chunk
                )
                rand_d = [
                    _rssm_dream_diag_tokens(model, sample, rp, dev, chunk=chunk)
                    for rp in rand_plans
                ]
            else:
                names = stream_names_ref
                true_d = _token_dream_diag_tokens(
                    model, sample, names, sample.actuator, dev, chunk=chunk
                )
                rand_d = [
                    _token_dream_diag_tokens(model, sample, names, rp, dev, chunk=chunk)
                    for rp in rand_plans
                ]

            shot_rec: dict = {
                "shot_id": int(sid),
                "plan_variation": float(plan_var),
                "per_stream": {},
            }
            present_streams = set(true_d.keys())
            for name in sorted(present_streams):
                rand_for_stream = [d[name] for d in rand_d if name in d]
                if true_d.get(name) is None or len(rand_for_stream) < 2:
                    continue
                dnm = _stream_delta_nm(true_d[name], rand_for_stream)
                shot_rec["per_stream"][name] = dnm
                stream_ratios.setdefault(name, []).append(dnm["ratio"])
            per_shot.append(shot_rec)
            mag = shot_rec["per_stream"].get(DECISION_STREAM, {})
            logger.info(
                "[%s] shot %s magnetics ratio=%.3f (tvr=%.4f rvr=%.4f)",
                kind,
                sid,
                mag.get("ratio", float("nan")),
                mag.get("true_vs_random", float("nan")),
                mag.get("random_vs_random", float("nan")),
            )

        per_stream_summary = {
            name: _cohort_summary(ratios) for name, ratios in stream_ratios.items()
        }
        result = {
            "model_kind": kind,
            "checkpoint": str(checkpoint),
            "has_diagnostics": has_diag,
            "n_random": int(n_random),
            "perturb_scale": float(perturb_scale),
            "n_shots": len(per_shot),
            "decision_stream": DECISION_STREAM,
            "per_stream_summary": per_stream_summary,
            "per_shot": per_shot,
        }
        mag_sum = per_stream_summary.get(DECISION_STREAM, {})
        logger.info(
            "[%s] COHORT magnetics: mean_ratio=%.3f median=%.3f pass_frac=%.2f (n=%d)",
            kind,
            mag_sum.get("mean_ratio", float("nan")),
            mag_sum.get("median_ratio", float("nan")),
            mag_sum.get("pass_fraction", float("nan")),
            mag_sum.get("n_scored", 0),
        )
        return result
    finally:
        try:
            del model
            if device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001
            logger.warning("[%s] model release note: %r", kind, exc)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def _render_figure(report: dict, out_png: Path) -> None:
    """A grouped bar of the cohort-mean diagnostic-token ΔN-M ratio per stream."""
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    streams = [DECISION_STREAM, *CONTEXT_STREAMS]
    models = [m for m in report.get("models", []) if m.get("per_stream_summary")]
    if not models:
        return
    colors = {"rssm": "#1f77b4", "token": "#d62728"}
    x = np.arange(len(streams))
    n_models = len(models)
    width = 0.8 / max(n_models, 1)

    fig, ax = plt.subplots(figsize=(9.0, 4.2), dpi=120)
    for mi, m in enumerate(models):
        ps = m["per_stream_summary"]
        vals = []
        for s in streams:
            r = ps.get(s, {}).get("mean_ratio", float("nan"))
            vals.append(r if np.isfinite(r) else 0.0)
        offs = (mi - (n_models - 1) / 2.0) * width
        bars = ax.bar(
            x + offs,
            vals,
            width,
            label=m["model_kind"],
            color=colors.get(m["model_kind"]),
        )
        for b, v in zip(bars, vals, strict=False):
            ax.annotate(
                f"{v:.2f}",
                (b.get_x() + b.get_width() / 2.0, v),
                ha="center",
                va="bottom",
                fontsize=7,
            )
    ax.axhline(1.0, ls="--", color="#444", lw=1, label="noise floor (ratio=1)")
    ax.set_xticks(x)
    ax.set_xticklabels(streams, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("cohort-mean ΔN-M ratio (dreamed diagnostic tokens)")
    ax.set_title(
        "Decoder-free dreamed-diagnostic command-sensitivity\n"
        "(true-plan vs random-plan dreamed tokens / random-vs-random floor)",
        fontsize=10,
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)
    logger.info("figure -> %s", out_png)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    import torch  # noqa: PLC0415

    from imas_ambix.worldmodel.gate_cohort import load_cohort  # noqa: PLC0415

    p = argparse.ArgumentParser(
        description="Decoder-free dreamed-MAGNETICS command-sensitivity gate."
    )
    p.add_argument(
        "--rssm-checkpoint",
        default="/work/projects/imas_gpu/worldmodel/ckpt/rssm-1222278/latest.pt",
    )
    p.add_argument(
        "--token-checkpoint",
        default="/work/projects/imas_gpu/worldmodel/ckpt/controllable-1221834/latest.pt",
    )
    p.add_argument(
        "--cohort",
        default="/work/projects/imas_gpu/worldmodel/gate_cohort.json",
    )
    p.add_argument(
        "--out-dir",
        default="/work/projects/imas_gpu/worldmodel/diag_space_controllability",
    )
    p.add_argument(
        "--figure",
        default="docs/figures/joint-multimodal-plasma-wm/diag_space_controllability.png",
    )
    p.add_argument(
        "--token-root",
        default="/work/projects/imas_gpu/worldmodel/curated-token-view",
    )
    p.add_argument("--camera", default="rbb")
    p.add_argument("--device", default="cuda")
    p.add_argument("--n-random", type=int, default=8)
    p.add_argument("--perturb-scale", type=float, default=0.3)
    p.add_argument("--chunk", type=int, default=8192)
    p.add_argument("--n-frames", type=int, default=24)
    p.add_argument("--n-plan", type=int, default=8)
    p.add_argument("--context-frames", type=int, default=8)
    p.add_argument("--frame-stride", type=int, default=1)
    p.add_argument("--target-horizon-s", type=float, default=0.25)
    p.add_argument("--skip-rssm", action="store_true", help="skip the RSSM checkpoint")
    p.add_argument(
        "--skip-token", action="store_true", help="skip the token checkpoint"
    )
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    _install_stop_handler()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA unavailable — falling back to CPU")
        device = "cpu"
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if device == "cuda":
        torch.set_float32_matmul_precision("high")

    cohort = list(load_cohort(args.cohort))
    logger.info("cohort: %d shots from %s", len(cohort), args.cohort)
    token_root = Path(args.token_root) if args.token_root else None
    window_kwargs = dict(
        n_frames=args.n_frames,
        n_plan=args.n_plan,
        context_frames=args.context_frames,
        frame_stride=args.frame_stride,
        target_horizon_s=args.target_horizon_s,
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    models: list[dict] = []
    plan_steps = [
        ("rssm", Path(args.rssm_checkpoint), args.skip_rssm),
        ("token", Path(args.token_checkpoint), args.skip_token),
    ]
    for kind, ckpt, skip in plan_steps:
        if skip:
            logger.info("[%s] skipped by flag", kind)
            continue
        if not ckpt.exists():
            logger.warning("[%s] checkpoint missing: %s — skipped", kind, ckpt)
            continue
        if _STOP["flag"]:
            break
        try:
            res = _run_model(
                kind,
                ckpt,
                cohort=cohort,
                camera=args.camera,
                token_root=token_root,
                device=device,
                n_random=args.n_random,
                perturb_scale=args.perturb_scale,
                chunk=args.chunk,
                window_kwargs=window_kwargs,
                out_dir=out_dir,
            )
            models.append(res)
        except Exception as exc:  # noqa: BLE001
            logger.exception("[%s] failed: %r", kind, exc)

    # decision: are the dreamed MAGNETICS command-sensitive?
    decision: dict = {}
    for m in models:
        mag = m.get("per_stream_summary", {}).get(DECISION_STREAM, {})
        decision[m["model_kind"]] = {
            "mean_ratio": mag.get("mean_ratio", float("nan")),
            "median_ratio": mag.get("median_ratio", float("nan")),
            "pass_fraction": mag.get("pass_fraction", float("nan")),
            "n_scored": mag.get("n_scored", 0),
        }
    # GO if EITHER model's dreamed magnetics clears the floor by a clear cohort
    # margin (mean ratio > 1.1 AND a majority of shots pass).
    go = any(
        np.isfinite(d.get("mean_ratio", float("nan")))
        and d["mean_ratio"] > 1.1
        and d.get("pass_fraction", 0.0) >= 0.5
        for d in decision.values()
    )
    verdict = "GO" if go else "NO-GO"

    report = {
        "metric": "dreamed_diagnostic_token_delta_nm",
        "decision_stream": DECISION_STREAM,
        "context_streams": list(CONTEXT_STREAMS),
        "cohort_size": len(cohort),
        "n_random": args.n_random,
        "perturb_scale": args.perturb_scale,
        "magnetics_decision": decision,
        "verdict": verdict,
        "models": models,
    }
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    logger.info("report -> %s", report_path)

    try:
        _render_figure(report, Path(args.figure))
    except Exception as exc:  # noqa: BLE001
        logger.warning("figure render skipped (%r)", exc)

    logger.info("=" * 70)
    logger.info("DREAMED-MAGNETICS COMMAND-SENSITIVITY: %s", verdict)
    for kind, d in decision.items():
        logger.info(
            "  %s magnetics: mean_ratio=%.3f median=%.3f pass_frac=%.2f (n=%d)",
            kind,
            d["mean_ratio"],
            d["median_ratio"],
            d["pass_fraction"],
            d["n_scored"],
        )
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
