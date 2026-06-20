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
    """Decode a true-plan vs silenced-plan rollout and score with M3 metrics.

    Reuses the frozen Open-MAGVIT2 decode + the M3 pixel-space metrics
    (counterfactual delta + control divergence on the inboard band).  Returns the
    scored dict, or ``None`` if the decode stack is unavailable.
    """
    import torch  # noqa: PLC0415

    from imas_ambix.worldmodel.actuator_plan import zero_plan
    from imas_ambix.worldmodel.control_falsification import decode_roles
    from imas_ambix.worldmodel.control_guidance import (
        counterfactual_delta,
        frame_l1,
        inboard_emission_series,
    )
    from imas_ambix.worldmodel.controllable_train import _actuator_batch_from_plan
    from imas_ambix.worldmodel.spacetime_dataset import GRID_H, GRID_W, local_to_store

    dev = torch.device(device)
    full_act = _actuator_batch_from_plan(sample.actuator, dev)
    zero_act = _actuator_batch_from_plan(zero_plan(sample.actuator), dev)

    true_tok = _argmax_actuator_rollout(
        model, sample, stream_names, full_act, dev, chunk=chunk
    ).reshape(-1, GRID_H, GRID_W)
    zero_tok = _argmax_actuator_rollout(
        model, sample, stream_names, zero_act, dev, chunk=chunk
    ).reshape(-1, GRID_H, GRID_W)

    ctx = int(sample.context_frames)
    try:
        wd = Path(work_dir or tempfile.mkdtemp(prefix="m4-gate-pixels-"))
        decoded = decode_roles(
            {
                "true_plan": local_to_store(true_tok),
                "zero_plan": local_to_store(zero_tok),
            },
            [{"role": "true_plan"}, {"role": "zero_plan"}],
            work_dir=wd,
            device=device,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("pixel-space confirmation decode unavailable: %r", exc)
        return None

    cf = counterfactual_delta(decoded["true_plan"], decoded["zero_plan"], ctx)
    pixel_l1 = frame_l1(decoded["true_plan"], decoded["zero_plan"], ctx)
    return {
        "shot_id": int(sample.shot_id),
        "context_frames": ctx,
        "true_plan_inboard_mean": float(
            inboard_emission_series(decoded["true_plan"])[ctx:].mean()
        ),
        "zero_plan_inboard_mean": float(
            inboard_emission_series(decoded["zero_plan"])[ctx:].mean()
        ),
        "counterfactual_delta": cf["counterfactual_delta"],
        "true_vs_zero_pixel_l1": pixel_l1,
        "token_mismatch_true_vs_zero": float((true_tok[ctx:] != zero_tok[ctx:]).mean()),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


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
    p.add_argument(
        "--context-corruption-rate",
        type=float,
        default=0.5,
        help="fraction of context-frame tokens noised during overfit so the "
        "model must lean on the conditioning (M2 recipe); 0 disables it",
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
        "--confirm-pixels",
        action="store_true",
        help="also DECODE a true-plan vs silenced-plan rollout and score with "
        "M3 pixel-space metrics (slower — loads the frozen VQ)",
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
        overfit_controllable,
    )
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
    }
    cfg = OverfitControllableConfig(
        steps=args.steps,
        lr=args.lr,
        n_signal_steps=args.n_signal_steps,
        n_act_steps=args.n_act_steps,
        observation_dropout=args.observation_dropout,
        control_dropout=args.control_dropout,
        context_corruption_rate=args.context_corruption_rate,
        transient_windows=not args.no_transient_windows,
        window=window,
        model_kwargs=model_kwargs,
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
        result, model, samples, stream_names = overfit_controllable(
            shot_ids,
            camera=args.camera,
            config=cfg,
            token_root=Path(args.token_root) if args.token_root else None,
            device=device,
        )
        logger.info(
            "overfit done: initial=%.4f final=%.4f (drop %.1f%%) params=%d",
            result.initial_loss,
            result.final_loss,
            100.0 * (1.0 - result.final_loss / max(result.initial_loss, 1e-9)),
            result.n_parameters,
        )

        logger.info("==== M4 GATE: token-space controllability test ====")
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

        payload = {
            "overfit": {
                "initial_loss": result.initial_loss,
                "final_loss": result.final_loss,
                "n_parameters": result.n_parameters,
                "shot_ids": result.shot_ids,
                "steps": args.steps,
                "observation_dropout": args.observation_dropout,
                "control_dropout": args.control_dropout,
                "context_corruption_rate": args.context_corruption_rate,
            },
            "per_shot": [v.to_dict() for v in verdicts],
            "summary": summary,
            "pixel_confirmation": pixel_confirm,
            "config": {
                "n_frames": args.n_frames,
                "n_plan": args.n_plan,
                "context_frames": args.context_frames,
                "n_signal_steps": args.n_signal_steps,
                "n_act_steps": args.n_act_steps,
                "gas_scale": args.gas_scale,
                "nbi_scale": args.nbi_scale,
                "margin_threshold": args.margin_threshold,
                "model_kwargs": model_kwargs,
            },
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
    summary = payload["summary"]
    print("\n=== M4 actuator-PLAN controllability GATE VERDICT ===")
    of = payload["overfit"]
    print(
        f"overfit: initial={of['initial_loss']:.4f} -> final={of['final_loss']:.4f} "
        f"(params={of['n_parameters']}, obs_dropout={of['observation_dropout']})"
    )
    print(
        "Decision metric = CORRUPTED-context true-vs-zeroed margin (cc_*). The "
        "clean-context margins read ~0 on a memorised overfit regardless of "
        "controllability, so the gate decides on the corrupted-context reading "
        "where the camera lookup is removed and only the conditioning can move "
        "the prediction."
    )
    hdr = (
        f"{'shot':>6} {'plan_var':>8} {'cam_chg':>7} {'tr':>3} | "
        f"{'cc_true_v0':>10} {'cc_gas':>7} {'cc_nbi':>7} {'cc_obs':>7} | "
        f"{'clean_t0':>8} {'pass':>5}"
    )
    print(hdr)
    print("-" * len(hdr))
    for row in payload["per_shot"]:
        print(
            f"{row['shot_id']:>6} "
            f"{row.get('plan_variation', float('nan')):>8.3f} "
            f"{row.get('camera_change_fraction', float('nan')):>7.3f} "
            f"{str(row.get('is_transient'))[0]:>3} | "
            f"{row.get('cc_true_vs_zeroed_mismatch', float('nan')):>10.4f} "
            f"{row.get('cc_gas_scale_mismatch', float('nan')):>7.4f} "
            f"{row.get('cc_nbi_scale_mismatch', float('nan')):>7.4f} "
            f"{row.get('cc_observation_mismatch', float('nan')):>7.4f} | "
            f"{row['true_vs_zeroed_mismatch']:>8.4f} "
            f"{str(row['passed'])[0]:>5}"
        )
    print(
        f"\ntransient windows (plan actually varies): "
        f"{summary.get('n_transient')}/{summary['n_samples']} "
        f"(mean plan_variation {summary.get('mean_plan_variation', float('nan')):.3f})"
    )
    print(
        f"DECISION: mean cc true-vs-zeroed margin: "
        f"{summary.get('mean_cc_true_vs_zeroed_mismatch', float('nan')):.4f} "
        f"(threshold {summary['margin_threshold']})"
    )
    print(
        f"mean cc gas-scale margin: "
        f"{summary.get('mean_cc_gas_scale_mismatch', float('nan')):.4f}; "
        f"mean cc NBI-scale margin: "
        f"{summary.get('mean_cc_nbi_scale_mismatch', float('nan')):.4f}; "
        f"mean cc observation margin (redundancy baseline): "
        f"{summary.get('mean_cc_observation_mismatch', float('nan')):.4f}"
    )
    print(
        f"(clean-context mean true-vs-zeroed margin "
        f"{summary['mean_true_vs_zeroed_mismatch']:.4f} — expected ~0 on a "
        f"memorised overfit, NOT the decision metric)"
    )
    pc = payload.get("pixel_confirmation")
    if pc:
        print(
            f"\npixel confirmation (shot {pc['shot_id']}): "
            f"true-plan inboard={pc['true_plan_inboard_mean']:.2f} "
            f"zero-plan inboard={pc['zero_plan_inboard_mean']:.2f} "
            f"counterfactual delta={pc['counterfactual_delta']:.4f} "
            f"pixel L1={pc['true_vs_zero_pixel_l1']:.4f}"
        )
    n_score = summary.get("n_transient") or summary["n_samples"]
    print(f"\nGATE: {summary['n_pass']}/{n_score} transient shots pass")
    if not summary.get("gate_testable", True):
        print(
            "WARNING: NO transient window found on any shot — the gate could not "
            "fairly test controllability (flat-top windows only)."
        )
    print(f"M4 GATE VERDICT: {summary['verdict']}")
    print(
        "(PASS => the actuator PLAN is causally load-bearing => the 6-GPU "
        "re-train is justified)"
        if summary["gate_pass"]
        else "(FAIL => plan conditioning did NOT make controls load-bearing on "
        "the overfit gate => re-train NOT justified by this evidence)"
    )


if __name__ == "__main__":
    raise SystemExit(main())
