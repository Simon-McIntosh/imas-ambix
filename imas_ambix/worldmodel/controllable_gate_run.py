"""M4 controllability GATE driver: overfit on the actuator PLAN, then test it.

The cheap de-risking gate for the playable-plasma PLAY bridge.  M3 proved the
measured-signal conditioning is NOT controllable (redundant observations); this
asks the go/no-go question for the 6-GPU re-train: does conditioning the camera
model on the demanded actuator PLAN make the controls causally LOAD-BEARING?

Pipeline (1 GPU, minutes):

1. Overfit a handful of shots conditioned on the actuator plan, with HIGH
   observation-dropout (the model must drive from the PLAN, not the redundant
   observations) + control-dropout (so classifier-free guidance works).
2. Token-space controllability gate (:func:`controllable_train.controllability_gate`):
   vary the actuator plan (silence the whole drive, scale / silence the gas-puff
   command, scale the NBI command) and measure whether the predicted next-frame
   tokens change — the true-vs-zeroed causal margin, vs the redundant-observation
   baseline.  Decoder-free → fast, a strict lower bound on the pixel response.
3. (optional) DECODE a true-plan vs silenced-plan rollout pair on one shot and
   re-use M3's pixel-space metrics (counterfactual delta + control divergence)
   to confirm the token response shows up in decoded pixels.

Writes a verdict JSON with a clear PASS/FAIL + numbers.  GPU-safe (AGENTS.md
§2b): model loaded once, SIGTERM STOP flag, try/finally release + empty_cache.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import signal
import tempfile
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_STOP = {"flag": False}


def _install_signal_handler() -> None:
    def _handler(signum, _frame):  # noqa: ANN001
        logger.warning("signal %s received — stopping after the current phase", signum)
        _STOP["flag"] = True

    for sig in (signal.SIGTERM, signal.SIGINT):
        with contextlib.suppress(ValueError, OSError):
            signal.signal(sig, _handler)


# ---------------------------------------------------------------------------
# Optional pixel-space confirmation (reuses the frozen Open-MAGVIT2 decode + M3)
# ---------------------------------------------------------------------------


def _argmax_actuator_rollout(
    model,
    sample,
    stream_names,
    actuator_batch,
    device,
    *,
    chunk: int,
):
    """Autoregressive argmax rollout under a FIXED actuator plan (token ids).

    Keeps the context frames as truth, then rolls forward consuming its own
    predicted frames while conditioning on the plan + signals + the SUPPLIED
    actuator drive at every step.  Returns ``(T, S)`` local token ids.
    """
    import torch  # noqa: PLC0415

    from imas_ambix.worldmodel.controllable_train import (  # noqa: PLC0415
        _batch_to,
        collate_controllable_windows,
    )

    model.eval()
    ctx = int(sample.context_frames)
    t_total = int(sample.frames.shape[0])
    batch = _batch_to(
        collate_controllable_windows([sample], stream_names=list(stream_names)), device
    )
    plan = batch.get("plan")
    signals = batch.get("signals")
    gen = np.asarray(sample.frames, dtype=np.int64).copy()
    with torch.no_grad():
        for ti in range(ctx, t_total):
            cur = torch.as_tensor(gen[:ti][None], dtype=torch.long, device=device)
            hidden = model._forward_tokens(cur, plan, signals, actuator=actuator_batch)
            pred = model.chunked_argmax_frame(hidden[:, ti - 1], chunk=chunk)
            gen[ti] = pred[0].cpu().numpy().astype(np.int64)
    return gen


def confirm_in_pixels(
    model,
    sample,
    stream_names,
    *,
    device: str,
    chunk: int = 8192,
    work_dir: Path | None = None,
) -> dict | None:
    """Decode a true-plan vs RANDOM-plan rollout and score with M3 metrics.

    The pixel-space confirmation of the ΔN-M gate: reuses the frozen Open-MAGVIT2
    decode + the M3 pixel-space metrics (counterfactual delta + inboard-band L1)
    on a TRUE-plan vs RANDOM-plan rollout pair (the random plan matches the true
    plan's marginal scale — :func:`_random_actuator_like`), so a positive decoded
    delta confirms the token-space ΔN-M divergence shows up in pixels.  Returns the
    scored dict, or ``None`` if the decode stack is unavailable.
    """
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    from imas_ambix.worldmodel.control_falsification import decode_roles
    from imas_ambix.worldmodel.control_guidance import (
        counterfactual_delta,
        frame_l1,
        inboard_emission_series,
    )
    from imas_ambix.worldmodel.controllable_train import (
        _actuator_batch_from_plan,
        _random_actuator_like,
    )
    from imas_ambix.worldmodel.spacetime_dataset import GRID_H, GRID_W, local_to_store

    dev = torch.device(device)
    rng = np.random.default_rng(int(sample.shot_id) * 1_000_003)
    full_act = _actuator_batch_from_plan(sample.actuator, dev)
    rand_act = _actuator_batch_from_plan(
        _random_actuator_like(sample.actuator, rng=rng), dev
    )

    true_tok = _argmax_actuator_rollout(
        model, sample, stream_names, full_act, dev, chunk=chunk
    ).reshape(-1, GRID_H, GRID_W)
    rand_tok = _argmax_actuator_rollout(
        model, sample, stream_names, rand_act, dev, chunk=chunk
    ).reshape(-1, GRID_H, GRID_W)

    ctx = int(sample.context_frames)
    try:
        wd = Path(work_dir or tempfile.mkdtemp(prefix="m4-gate-pixels-"))
        decoded = decode_roles(
            {
                "true_plan": local_to_store(true_tok),
                "rand_plan": local_to_store(rand_tok),
            },
            [{"role": "true_plan"}, {"role": "rand_plan"}],
            work_dir=wd,
            device=device,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("pixel-space confirmation decode unavailable: %r", exc)
        return None

    cf = counterfactual_delta(decoded["true_plan"], decoded["rand_plan"], ctx)
    pixel_l1 = frame_l1(decoded["true_plan"], decoded["rand_plan"], ctx)
    return {
        "shot_id": int(sample.shot_id),
        "context_frames": ctx,
        "true_plan_inboard_mean": float(
            inboard_emission_series(decoded["true_plan"])[ctx:].mean()
        ),
        "zero_plan_inboard_mean": float(
            inboard_emission_series(decoded["rand_plan"])[ctx:].mean()
        ),
        "counterfactual_delta": cf["counterfactual_delta"],
        "true_vs_zero_pixel_l1": pixel_l1,
        "token_mismatch_true_vs_random": float(
            (true_tok[ctx:] != rand_tok[ctx:]).mean()
        ),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _run_ablation(shot_ids, make_config, args, *, device, token_root, init_checkpoint):
    """Re-overfit + ΔN-M-gate with each fixable leg turned off, to attribute it.

    Toggles the two cheaply-disableable legs in turn (the history bottleneck and
    the inverse-dynamics auxiliary) plus an all-legs-off baseline, and reports the
    ΔN-M margin under each.  The AdaLN-vs-prepended-tokens leg cannot be toggled by
    a flag (it is the architecture), so it is not in the ablation — the comparison
    point for AdaLN is the PRIOR gate result (prepended tokens, margin ~0).  A leg
    whose removal collapses the ΔN-M margin toward the prior ~0 is the binding
    contributor.
    """
    from imas_ambix.worldmodel.controllable_train import (
        delta_nm_gate,
        overfit_controllable,
    )

    legs = {
        "full": dict(
            hb_noise_std=args.hb_noise_std,
            hb_mask_prob=args.hb_mask_prob,
            hb_max_strength=args.hb_max_strength,
            inv_dyn_weight=args.inverse_dynamics_weight,
        ),
        "no_history_bottleneck": dict(
            hb_noise_std=0.0,
            hb_mask_prob=0.0,
            hb_max_strength=0.0,
            inv_dyn_weight=args.inverse_dynamics_weight,
        ),
        "no_inverse_dynamics": dict(
            hb_noise_std=args.hb_noise_std,
            hb_mask_prob=args.hb_mask_prob,
            hb_max_strength=args.hb_max_strength,
            inv_dyn_weight=0.0,
        ),
        "neither": dict(
            hb_noise_std=0.0,
            hb_mask_prob=0.0,
            hb_max_strength=0.0,
            inv_dyn_weight=0.0,
        ),
    }
    out: dict[str, dict] = {}
    for name, kw in legs.items():
        if _STOP["flag"]:
            break
        logger.info("==== M4 GATE ablation leg: %s ====", name)
        cfg = make_config(**kw)
        res, model, samples, stream_names = overfit_controllable(
            shot_ids,
            camera=args.camera,
            config=cfg,
            token_root=token_root,
            device=device,
            init_checkpoint=init_checkpoint,
        )
        _verdicts, summary = delta_nm_gate(
            model,
            samples,
            stream_names,
            device=device,
            chunk=args.chunk,
            n_random=args.n_random,
            margin_threshold=args.margin_threshold,
            floor_ratio=args.floor_ratio,
            transient_threshold=args.transient_threshold,
        )
        out[name] = {
            "final_loss": res.final_loss,
            "mean_true_vs_random": summary["mean_true_vs_random"],
            "mean_random_vs_random_noise_floor": summary[
                "mean_random_vs_random_noise_floor"
            ],
            "mean_margin": summary["mean_margin"],
            "verdict": summary["verdict"],
        }
        import torch  # noqa: PLC0415

        del model
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--shots",
        default="18502,18503,18504,18505",
        help="comma-separated shots to overfit + gate on",
    )
    p.add_argument("--out-json", required=True)
    p.add_argument("--camera", default="rbb")
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--n-frames", type=int, default=24)
    p.add_argument("--n-plan", type=int, default=8)
    p.add_argument("--context-frames", type=int, default=8)
    p.add_argument("--frame-stride", type=int, default=1)
    p.add_argument("--n-signal-steps", type=int, default=4)
    p.add_argument("--n-act-steps", type=int, default=8)
    p.add_argument("--observation-dropout", type=float, default=0.8)
    p.add_argument("--control-dropout", type=float, default=0.15)
    # ── camera-history bottleneck (the corrected lever) ──
    p.add_argument(
        "--hb-noise-std",
        type=float,
        default=1.0,
        help="full-strength additive-Gaussian std (in embedding-RMS units) of the "
        "camera-history bottleneck; 0 disables the noise leg",
    )
    p.add_argument(
        "--hb-mask-prob",
        type=float,
        default=0.5,
        help="full-strength probability of masking a whole context-frame's "
        "embedding (the strongest bottleneck); 0 disables the mask leg",
    )
    p.add_argument(
        "--hb-max-strength",
        type=float,
        default=1.0,
        help="max per-frame bottleneck strength (0 disables the whole bottleneck "
        "— the prior failed recipe)",
    )
    p.add_argument(
        "--hb-clean-fraction",
        type=float,
        default=0.2,
        help="fraction of samples whose history is left fully clean (keeps the "
        "clean/inference regime well-trained)",
    )
    p.add_argument(
        "--hb-independent-per-frame",
        type=int,
        default=1,
        help="1 = Diffusion-Forcing independent per-frame strength; 0 = one "
        "shared strength per sample (M2 single-level recipe)",
    )
    # ── inverse-dynamics auxiliary + AdaLN width ──
    p.add_argument(
        "--inverse-dynamics-weight",
        type=float,
        default=1.0,
        help="weight of the inverse-dynamics auxiliary loss (predict the plan "
        "from consecutive latents); 0 disables it",
    )
    p.add_argument("--adaln-hidden", type=int, default=256)
    # ── ΔN-M gate ──
    p.add_argument(
        "--n-random",
        type=int,
        default=4,
        help="number of random plans for the ΔN-M action-sensitivity gate (the "
        "random-vs-random noise floor needs >= 2)",
    )
    p.add_argument(
        "--floor-ratio",
        type=float,
        default=1.5,
        help="ΔN-M PASS also requires true-vs-random > floor_ratio * the "
        "random-vs-random noise floor",
    )
    p.add_argument("--gas-scale", type=float, default=3.0)
    p.add_argument("--nbi-scale", type=float, default=3.0)
    p.add_argument("--margin-threshold", type=float, default=0.02)
    p.add_argument("--chunk", type=int, default=4096)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-layers", type=int, default=6)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--d-ff", type=int, default=1024)
    p.add_argument("--device", default="cuda")
    p.add_argument("--token-root", default=None)
    p.add_argument(
        "--init-checkpoint",
        default=None,
        help="warm-start the backbone from the M2 forecaster checkpoint "
        "(strict=False; AdaLN MLP + inverse-dynamics head start at init). Omit to "
        "overfit from scratch.",
    )
    p.add_argument(
        "--confirm-pixels",
        action="store_true",
        help="also DECODE a true-plan vs random-plan rollout and score with "
        "M3 pixel-space metrics (slower — loads the frozen VQ)",
    )
    p.add_argument(
        "--ablate",
        action="store_true",
        help="run a per-leg ablation: re-overfit + ΔN-M-gate with each of the "
        "three legs (history-bottleneck / AdaLN-via-actuator / inverse-dynamics) "
        "turned off in turn, to attribute which moves the needle (slower)",
    )
    p.add_argument(
        "--no-transient-windows",
        action="store_true",
        help="overfit the CENTRED (flat-top) window instead of the actuator-plan "
        "transient window (debug — the gate is unfair on a flat-top window where "
        "the plan does not vary)",
    )
    p.add_argument(
        "--transient-threshold",
        type=float,
        default=1e-3,
        help="min summed per-channel std of the normalised actuator drive for a "
        "window to count as transient (fairly testable)",
    )
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    _install_signal_handler()

    import torch

    from imas_ambix.worldmodel.controllable_train import (
        OverfitControllableConfig,
        controllability_gate,
        delta_nm_gate,
        overfit_controllable,
    )
    from imas_ambix.worldmodel.history_bottleneck import HistoryBottleneckConfig
    from imas_ambix.worldmodel.spacetime_dataset import SpacetimeWindowConfig

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA unavailable — falling back to CPU (slow)")
        device = "cpu"
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if device == "cuda":
        torch.set_float32_matmul_precision("high")

    shot_ids = [int(s) for s in args.shots.split(",") if s.strip()]
    window = SpacetimeWindowConfig(
        n_frames=args.n_frames,
        n_plan=args.n_plan,
        context_frames=args.context_frames,
        frame_stride=args.frame_stride,
    )
    model_kwargs = {
        "d_model": args.d_model,
        "n_layers": args.n_layers,
        "n_heads": args.n_heads,
        "d_ff": args.d_ff,
        "adaln_hidden": args.adaln_hidden,
    }

    def _make_config(
        *,
        hb_noise_std: float,
        hb_mask_prob: float,
        hb_max_strength: float,
        inv_dyn_weight: float,
    ) -> OverfitControllableConfig:
        return OverfitControllableConfig(
            steps=args.steps,
            lr=args.lr,
            n_signal_steps=args.n_signal_steps,
            n_act_steps=args.n_act_steps,
            observation_dropout=args.observation_dropout,
            control_dropout=args.control_dropout,
            history_bottleneck=HistoryBottleneckConfig(
                noise_std=hb_noise_std,
                mask_prob=hb_mask_prob,
                max_strength=hb_max_strength,
                clean_fraction=args.hb_clean_fraction,
                independent_per_frame=bool(args.hb_independent_per_frame),
            ),
            inverse_dynamics_weight=inv_dyn_weight,
            transient_windows=not args.no_transient_windows,
            window=window,
            model_kwargs=model_kwargs,
        )

    cfg = _make_config(
        hb_noise_std=args.hb_noise_std,
        hb_mask_prob=args.hb_mask_prob,
        hb_max_strength=args.hb_max_strength,
        inv_dyn_weight=args.inverse_dynamics_weight,
    )

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    result = None
    model = None
    try:
        logger.info(
            "==== M4 GATE: overfit on the actuator PLAN (%d shots) ====",
            len(shot_ids),
        )
        init_ckpt = Path(args.init_checkpoint) if args.init_checkpoint else None
        result, model, samples, stream_names = overfit_controllable(
            shot_ids,
            camera=args.camera,
            config=cfg,
            token_root=Path(args.token_root) if args.token_root else None,
            device=device,
            init_checkpoint=init_ckpt,
        )
        logger.info(
            "overfit done: initial=%.4f final=%.4f (drop %.1f%%) params=%d",
            result.initial_loss,
            result.final_loss,
            100.0 * (1.0 - result.final_loss / max(result.initial_loss, 1e-9)),
            result.n_parameters,
        )

        # ── PRIMARY gate: ΔN-M action-sensitivity (autoregressive token rollout,
        # true-vs-random plan vs a random-vs-random noise floor) ──
        logger.info("==== M4 GATE: ΔN-M action-sensitivity (true vs random plan) ====")
        dnm_verdicts, dnm_summary = delta_nm_gate(
            model,
            samples,
            stream_names,
            device=device,
            chunk=args.chunk,
            n_random=args.n_random,
            margin_threshold=args.margin_threshold,
            floor_ratio=args.floor_ratio,
            transient_threshold=args.transient_threshold,
        )

        # ── SECONDARY gate: teacher-forced token mismatch (fast lower bound) ──
        logger.info("==== M4 GATE: teacher-forced token-mismatch (secondary) ====")
        verdicts, summary = controllability_gate(
            model,
            samples,
            stream_names,
            device=device,
            chunk=args.chunk,
            gas_scale=args.gas_scale,
            nbi_scale=args.nbi_scale,
            margin_threshold=args.margin_threshold,
            transient_threshold=args.transient_threshold,
        )

        pixel_confirm = None
        if args.confirm_pixels and not _STOP["flag"]:
            logger.info("==== M4 GATE: pixel-space confirmation (1 shot) ====")
            with contextlib.suppress(Exception):
                pixel_confirm = confirm_in_pixels(
                    model, samples[0], stream_names, device=device, chunk=8192
                )

        # ── optional per-leg ablation (which fix moves the needle) ──
        ablation = None
        if args.ablate and not _STOP["flag"]:
            ablation = _run_ablation(
                shot_ids,
                _make_config,
                args,
                device=device,
                token_root=Path(args.token_root) if args.token_root else None,
                init_checkpoint=init_ckpt,
            )

        payload = {
            "overfit": {
                "initial_loss": result.initial_loss,
                "final_loss": result.final_loss,
                "n_parameters": result.n_parameters,
                "shot_ids": result.shot_ids,
                "steps": args.steps,
                "observation_dropout": args.observation_dropout,
                "control_dropout": args.control_dropout,
                "hb_noise_std": args.hb_noise_std,
                "hb_mask_prob": args.hb_mask_prob,
                "hb_max_strength": args.hb_max_strength,
                "inverse_dynamics_weight": args.inverse_dynamics_weight,
            },
            "delta_nm_gate": {
                "per_shot": [v.to_dict() for v in dnm_verdicts],
                "summary": dnm_summary,
            },
            "secondary_teacher_forced_gate": {
                "per_shot": [v.to_dict() for v in verdicts],
                "summary": summary,
            },
            "ablation": ablation,
            "pixel_confirmation": pixel_confirm,
            "config": {
                "n_frames": args.n_frames,
                "n_plan": args.n_plan,
                "context_frames": args.context_frames,
                "n_signal_steps": args.n_signal_steps,
                "n_act_steps": args.n_act_steps,
                "n_random": args.n_random,
                "floor_ratio": args.floor_ratio,
                "gas_scale": args.gas_scale,
                "nbi_scale": args.nbi_scale,
                "margin_threshold": args.margin_threshold,
                "model_kwargs": model_kwargs,
            },
            # the headline verdict is the ΔN-M gate.
            "summary": dnm_summary,
        }
        out_json.write_text(json.dumps(payload, indent=2, default=str))
        _print_verdict(payload)
    finally:
        try:
            del model
            if device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001
            logger.warning("model release note: %r", exc)
    return 0


def _print_verdict(payload: dict) -> None:
    of = payload["overfit"]
    dnm = payload["delta_nm_gate"]["summary"]
    print("\n=== M4 actuator-PLAN controllability GATE VERDICT ===")
    print(
        f"overfit: initial={of['initial_loss']:.4f} -> final={of['final_loss']:.4f} "
        f"(params={of['n_parameters']})"
    )
    print(
        f"fixes: history-bottleneck(std={of['hb_noise_std']}, "
        f"mask={of['hb_mask_prob']}, "
        f"max={of['hb_max_strength']}) + AdaLN-Zero plan conditioning + "
        f"inverse-dynamics(w={of['inverse_dynamics_weight']})"
    )
    print(
        "\nPRIMARY METRIC = ΔN-M action-sensitivity: forecast-window token "
        "divergence of an AUTOREGRESSIVE rollout under the TRUE plan vs a RANDOM "
        "plan, scored against a RANDOM-vs-RANDOM noise floor. PASS = true-vs-random "
        "clears the floor by a clear margin (the plan steers the dream)."
    )
    hdr = (
        f"{'shot':>6} {'plan_var':>8} {'tr':>3} | "
        f"{'true_v_rnd':>10} {'rnd_v_rnd':>10} {'margin':>8} {'ratio':>7} {'pass':>5}"
    )
    print(hdr)
    print("-" * len(hdr))
    for row in payload["delta_nm_gate"]["per_shot"]:
        ratio = row.get("ratio", float("nan"))
        ratio_s = "inf" if ratio == float("inf") else f"{ratio:.2f}"
        print(
            f"{row['shot_id']:>6} "
            f"{row.get('plan_variation', float('nan')):>8.3f} "
            f"{str(row.get('is_transient'))[0]:>3} | "
            f"{row.get('true_vs_random', float('nan')):>10.4f} "
            f"{row.get('random_vs_random', float('nan')):>10.4f} "
            f"{row.get('margin', float('nan')):>8.4f} "
            f"{ratio_s:>7} "
            f"{str(row.get('passed'))[0]:>5}"
        )
    print(
        f"\ntransient windows (plan actually varies): "
        f"{dnm.get('n_transient')}/{dnm['n_samples']}"
    )
    print(
        f"DECISION: mean true-vs-random margin: {dnm['mean_margin']:.4f} "
        f"(threshold {dnm['margin_threshold']}); "
        f"mean true-vs-random {dnm['mean_true_vs_random']:.4f} vs "
        f"noise floor {dnm['mean_random_vs_random_noise_floor']:.4f}"
    )

    # secondary gate (teacher-forced, fast lower bound)
    sec = payload.get("secondary_teacher_forced_gate", {}).get("summary")
    if sec:
        cc = sec.get("mean_cc_true_vs_zeroed_mismatch", float("nan"))
        print(
            f"\nSECONDARY (teacher-forced, corrupted-context true-vs-zeroed): "
            f"mean cc margin {cc:.4f} -> {sec.get('verdict')}"
        )

    abl = payload.get("ablation")
    if abl:
        print("\n--- per-leg ablation (ΔN-M mean margin) ---")
        for name, d in abl.items():
            print(
                f"  {name:>22}: margin={d['mean_margin']:>8.4f} "
                f"(true_v_rnd={d['mean_true_vs_random']:.4f}, "
                f"floor={d['mean_random_vs_random_noise_floor']:.4f}, "
                f"loss={d['final_loss']:.4f}) -> {d['verdict']}"
            )

    pc = payload.get("pixel_confirmation")
    if pc:
        print(
            f"\npixel confirmation (shot {pc['shot_id']}): "
            f"true-plan inboard={pc['true_plan_inboard_mean']:.2f} "
            f"alt-plan inboard={pc.get('zero_plan_inboard_mean', float('nan')):.2f} "
            f"counterfactual delta={pc['counterfactual_delta']:.4f} "
            f"pixel L1={pc['true_vs_zero_pixel_l1']:.4f}"
        )

    n_score = dnm.get("n_transient") or dnm["n_samples"]
    print(f"\nGATE: {dnm['n_pass']}/{n_score} transient shots pass")
    if not dnm.get("gate_testable", True):
        print(
            "WARNING: NO transient window found on any shot — the gate could not "
            "fairly test controllability (flat-top windows only)."
        )
    print(f"M4 GATE VERDICT: {dnm['verdict']}")
    print(
        "(PASS => the actuator PLAN is causally load-bearing => the 6-GPU "
        "re-train is justified)"
        if dnm["gate_pass"]
        else "(FAIL => plan conditioning did NOT make controls load-bearing on "
        "the overfit gate => re-train NOT justified by this evidence)"
    )


if __name__ == "__main__":
    raise SystemExit(main())
