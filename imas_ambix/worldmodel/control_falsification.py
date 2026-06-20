"""W1 driver — CFG sweep + gas-puff falsification on held-out shots.

Answers the W1 controllability bar (``docs/playable-plasma-wm-v0.html`` §7) on the
M2 context-corruption checkpoint:

* **CFG at inference** — for each held-out shot, roll the dream out at guidance
  weights ``w in {1.0, 1.5, 2.0}`` (1.0 is the plain conditioned model; > 1
  amplifies the controls) and decode every rollout.
* **Gas-puff falsification** — decode three conditionings: (a) the TRUE
  ``gas_injection``, (b) ``gas_injection`` ZEROED (counterfactual no-puff), (c)
  CFG amplifying it.  Measure the decoded emission in the inboard pixel band
  (left of the centre column) and report the timing correlation against the puff
  command + the counterfactual ``(a) - (b)`` delta.
* **Control-divergence** — from one fixed seed, true-puff vs zeroed-puff pixel L1
  vs the same-conditioning sample spread (different seeds).

Performance (AGENTS.md §2b): the transformer AND the frozen Open-MAGVIT2 VQModel
are each loaded ONCE.  Per shot every rollout (all CFG weights, the counterfactual,
the spread members, the GT) is packed into a SINGLE token bundle decoded in ONE VQ
pass.  A SIGTERM handler sets a STOP flag so a cancellation flushes the partial
verdict losslessly.  All rollouts are sampled (the M1/M2 lesson: argmax collapses)
at a fixed ``(temperature, top_p)``; CFG composes with sampling on the guided
logits.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import signal
import tempfile
import time
from pathlib import Path

import numpy as np

from imas_ambix.worldmodel.control_guidance import (
    GAS_STREAM,
    INBOARD_COLS,
    cfg_guided_dream,
    control_divergence,
    counterfactual_delta,
    gas_command_per_frame,
    inboard_emission_series,
    puff_timing_correlation,
)
from imas_ambix.worldmodel.spacetime_dataset import (
    GRID_H,
    GRID_W,
    REFERENCE_CAMERA,
    SpacetimeWindowConfig,
    local_to_store,
)
from imas_ambix.worldmodel.spacetime_dataset_v2 import (
    SignalModalitySpec,
    assemble_signal_window,
    default_signal_modalities,
)

logger = logging.getLogger(__name__)

_STOP = {"flag": False}

#: CFG weights swept.  1.0 = plain conditioned model; > 1 amplifies the controls.
DEFAULT_GUIDANCE = (1.0, 1.5, 2.0)


def _install_signal_handler() -> None:
    def _handler(signum, _frame):  # noqa: ANN001
        logger.warning(
            "signal %s received — finishing the current shot then stopping", signum
        )
        _STOP["flag"] = True

    for sig in (signal.SIGTERM, signal.SIGINT):
        with contextlib.suppress(ValueError, OSError):
            signal.signal(sig, _handler)


# ---------------------------------------------------------------------------
# Per-shot rollouts (model loaded ONCE by the caller)
# ---------------------------------------------------------------------------


def _roll(
    model,
    sample,
    *,
    stream_names,
    device,
    guidance: float,
    temperature: float,
    top_p: float,
    seed: int,
    chunk: int,
    zero_streams=None,
):
    """One CFG rollout (local token ids ``(F, S)``) at a reproducible seed."""
    import torch

    gen = torch.Generator(device=torch.device(device)).manual_seed(int(seed))
    return cfg_guided_dream(
        model,
        sample,
        stream_names=stream_names,
        guidance=float(guidance),
        temperature=float(temperature),
        top_p=float(top_p),
        chunk=chunk,
        device=device,
        generator=gen,
        zero_streams=zero_streams,
    ).reshape(-1, GRID_H, GRID_W)


def generate_rollouts(
    *,
    model,
    payload: dict,
    checkpoint: Path,
    shot_id: int,
    window: SpacetimeWindowConfig,
    guidance_weights,
    spread_members: int,
    temperature: float,
    top_p: float,
    modalities: list[SignalModalitySpec] | None = None,
    n_signal_steps: int = 4,
    camera: str = REFERENCE_CAMERA,
    device: str = "cuda",
    token_root: Path | None = None,
    chunk: int = 8192,
    puff_window: bool = True,
) -> tuple[dict, list, dict]:
    """Build GT + per-w CFG rollouts + the zeroed-puff counterfactual + spread.

    Returns ``(grids_by_role, roles_index, meta)`` ready to decode in one VQ pass.
    Roles:
      * ``gt``
      * ``cfg_w{w}`` for each guidance weight (true conditioning, seed 0)
      * ``nopuff_w1`` — gas_injection zeroed, w=1, seed 0 (the counterfactual (b))
      * ``spread_m{m}`` — true conditioning, w=1, DIFFERENT seeds (the noise floor)

    ``puff_window`` (default True): select the camera window where the inboard puff
    TRANSITIONS most (:func:`...control_guidance.find_puff_window`) rather than the
    centred default — the falsification can only fire where the puff actually fires.
    Falls back to the centred window (and records ``puff_window_found=False``) when
    the command is flat everywhere for the shot.
    """
    from imas_ambix.worldmodel.control_guidance import find_puff_window

    modalities = modalities or default_signal_modalities()
    step = int((payload or {}).get("step", -1))
    model_streams = [st.name for st in model.config.signal_streams]

    span = (window.n_frames - 1) * window.frame_stride + 1
    start_frame = None
    puff_std = 0.0
    puff_found = False
    if puff_window:
        start_frame, puff_std = find_puff_window(
            int(shot_id), span, camera=camera, token_root=token_root
        )
        puff_found = start_frame is not None
        logger.info(
            "shot %s puff-window: start_frame=%s command_std=%.3f (found=%s)",
            shot_id,
            start_frame,
            puff_std,
            puff_found,
        )

    sample = assemble_signal_window(
        int(shot_id),
        window,
        modalities,
        int(n_signal_steps),
        camera=camera,
        token_root=token_root,
        start_frame=start_frame,
    )
    present = sorted(sample.signals.keys())
    ctx = int(sample.context_frames)
    n_frames = int(sample.frames.shape[0])
    has_gas = GAS_STREAM in sample.signals

    gt_local = np.asarray(sample.frames, dtype=np.int64).reshape(
        n_frames, GRID_H, GRID_W
    )
    grids_by_role: dict[str, np.ndarray] = {"gt": local_to_store(gt_local)}
    roles_index: list[dict] = [{"role": "gt"}]

    base_seed = (int(shot_id) * 100003) ^ 0x5A5A

    # CFG sweep — true conditioning, fixed seed so divergence vs spread is fair.
    for w in guidance_weights:
        t0 = time.time()
        roll = _roll(
            model,
            sample,
            stream_names=model_streams,
            device=device,
            guidance=float(w),
            temperature=temperature,
            top_p=top_p,
            seed=base_seed,
            chunk=chunk,
        )
        key = f"cfg_w{w}"
        grids_by_role[key] = local_to_store(roll)
        roles_index.append({"role": key})
        logger.info("shot %s CFG w=%.2f rollout in %.1fs", shot_id, w, time.time() - t0)
        if _STOP["flag"]:
            break

    # Counterfactual (b): gas_injection zeroed in the conditioned pass, w=1.
    if has_gas and not _STOP["flag"]:
        t0 = time.time()
        roll = _roll(
            model,
            sample,
            stream_names=model_streams,
            device=device,
            guidance=1.0,
            temperature=temperature,
            top_p=top_p,
            seed=base_seed,
            chunk=chunk,
            zero_streams=[GAS_STREAM],
        )
        grids_by_role["nopuff_w1"] = local_to_store(roll)
        roles_index.append({"role": "nopuff_w1"})
        logger.info(
            "shot %s no-puff counterfactual in %.1fs", shot_id, time.time() - t0
        )

    # Same-conditioning spread (noise floor): w=1, different seeds.
    for m in range(int(spread_members)):
        if _STOP["flag"]:
            break
        t0 = time.time()
        roll = _roll(
            model,
            sample,
            stream_names=model_streams,
            device=device,
            guidance=1.0,
            temperature=temperature,
            top_p=top_p,
            seed=base_seed ^ ((m + 1) * 2654435761),
            chunk=chunk,
        )
        grids_by_role[f"spread_m{m}"] = local_to_store(roll)
        roles_index.append({"role": f"spread_m{m}"})
        logger.info("shot %s spread member %d in %.1fs", shot_id, m, time.time() - t0)

    gas_cmd = gas_command_per_frame(sample).tolist()
    meta = {
        "shot_id": int(shot_id),
        "checkpoint": str(checkpoint),
        "checkpoint_step": step,
        "camera": camera,
        "n_frames": n_frames,
        "context_frames": ctx,
        "grid_hw": [GRID_H, GRID_W],
        "signal_streams": model_streams,
        "present_streams": present,
        "has_gas": bool(has_gas),
        "gas_command_per_frame": gas_cmd,
        "start_frame": int(sample.base.start_frame),
        "puff_window_found": bool(puff_found),
        "puff_window_command_std": float(puff_std),
        "guidance_weights": [float(w) for w in guidance_weights],
        "spread_members": int(spread_members),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "inboard_token_cols": list(INBOARD_COLS),
        "frame_time": np.asarray(sample.frame_time, dtype=np.float64).tolist(),
    }
    return grids_by_role, roles_index, meta


# ---------------------------------------------------------------------------
# Decode (one VQ pass) + score
# ---------------------------------------------------------------------------


def decode_roles(
    grids_by_role: dict[str, np.ndarray],
    roles_index: list[dict],
    *,
    work_dir: Path,
    device: str = "cuda",
) -> dict[str, np.ndarray]:
    """Decode every role's token grid to ``(F,256,256,3)`` in ONE VQ pass."""
    from imas_ambix.camdyn.reconstruction_demo import run_decode_subprocess

    work_dir.mkdir(parents=True, exist_ok=True)
    token_bundle = work_dir / "tokens.npz"
    image_bundle = work_dir / "images.npz"

    keys = [e["role"] for e in roles_index]
    grids = np.stack([grids_by_role[k] for k in keys]).astype(np.int64)
    index = [{"role": k, "slot": i} for i, k in enumerate(keys)]
    np.savez_compressed(
        token_bundle,
        grids=grids,
        index=json.dumps(index),
        meta=json.dumps({"format": "reconstruction_demo", "grid_hw": [GRID_H, GRID_W]}),
    )
    run_decode_subprocess(token_bundle, image_bundle, device)
    if not image_bundle.exists():
        raise RuntimeError(f"decode produced no image bundle at {image_bundle}")
    data = np.load(str(image_bundle), allow_pickle=True)
    images = np.asarray(data["images"], dtype=np.uint8)
    idx = json.loads(str(data["index"]))
    slot = {e["role"]: e["slot"] for e in idx}
    return {k: images[slot[k]] for k in keys}


def score_shot(decoded: dict[str, np.ndarray], meta: dict) -> dict:
    """Score the CFG sweep + the gas-puff falsification + control-divergence."""
    ctx = int(meta["context_frames"])
    cmd = np.asarray(meta["gas_command_per_frame"], dtype=np.float64)
    weights = meta["guidance_weights"]

    out: dict = {
        "shot_id": meta["shot_id"],
        "checkpoint": meta["checkpoint"],
        "checkpoint_step": meta["checkpoint_step"],
        "context_frames": ctx,
        "n_frames": meta["n_frames"],
        "present_streams": meta["present_streams"],
        "has_gas": meta["has_gas"],
        "temperature": meta["temperature"],
        "top_p": meta["top_p"],
        "inboard_token_cols": meta["inboard_token_cols"],
        "start_frame": meta.get("start_frame"),
        "puff_window_found": meta.get("puff_window_found"),
        "puff_window_command_std": meta.get("puff_window_command_std"),
        "gas_command_forecast_std": float(np.std(cmd[ctx:]) if cmd.size > ctx else 0.0),
    }

    # CFG sweep — inboard emission + timing correlation per weight.
    cfg: dict[str, dict] = {}
    for w in weights:
        key = f"cfg_w{w}"
        if key not in decoded:
            continue
        timing = puff_timing_correlation(decoded[key], cmd, ctx)
        emission = inboard_emission_series(decoded[key]).tolist()
        cfg[str(w)] = {
            "guidance": float(w),
            **timing,
            "inboard_emission_per_frame": emission,
        }
    out["cfg_sweep"] = cfg

    # Gas-puff falsification — counterfactual (a)-(b) at w=1 + CFG-amplified.
    if meta["has_gas"] and "nopuff_w1" in decoded and "cfg_w1.0" in decoded:
        cf_w1 = counterfactual_delta(decoded["cfg_w1.0"], decoded["nopuff_w1"], ctx)
        out["counterfactual"] = {
            "w1.0": cf_w1,
            "nopuff_timing": puff_timing_correlation(decoded["nopuff_w1"], cmd, ctx),
        }
        # CFG amplification of the counterfactual: does (a)-(b) grow with w?
        amp: dict[str, dict] = {}
        for w in weights:
            key = f"cfg_w{w}"
            if key in decoded:
                amp[str(w)] = counterfactual_delta(
                    decoded[key], decoded["nopuff_w1"], ctx
                )
        out["counterfactual"]["amplified_by_w"] = amp

    # Control-divergence — true-puff vs zeroed-puff vs same-conditioning spread.
    spread = [decoded[k] for k in decoded if k.startswith("spread_m")]
    if meta["has_gas"] and "nopuff_w1" in decoded and "cfg_w1.0" in decoded and spread:
        out["control_divergence"] = control_divergence(
            decoded["cfg_w1.0"], decoded["nopuff_w1"], spread, ctx
        )
    return out


# ---------------------------------------------------------------------------
# Figures (optional — a per-shot inboard-emission + counterfactual panel)
# ---------------------------------------------------------------------------


def assemble_figure(decoded: dict, scored: dict, meta: dict, out_dir: Path) -> dict:
    """A per-shot panel: inboard emission vs puff command + counterfactual GIF.

    Decoder-free metrics are the verdict; this panel makes the W1 result legible —
    the inboard emission series for true-puff / no-puff / CFG against the command,
    plus a contact sheet of the inboard region for true vs no-puff.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ctx = int(meta["context_frames"])
    cmd = np.asarray(meta["gas_command_per_frame"], dtype=np.float64)
    sid = meta["shot_id"]
    frames = np.arange(meta["n_frames"])

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(7.0, 6.0), dpi=120)
    # top: inboard emission series for each conditioning + the command (twin axis).
    for key, lbl, col in (
        ("cfg_w1.0", "true puff (w=1)", "#0a7d33"),
        ("nopuff_w1", "no puff (zeroed)", "#b00020"),
        ("cfg_w2.0", "CFG w=2.0", "#1f6feb"),
    ):
        if key in decoded:
            em = inboard_emission_series(decoded[key])
            ax0.plot(frames, em, color=col, lw=1.6, label=lbl)
    ax0.axvline(ctx - 0.5, color="#888", ls="--", lw=1.0)
    ax0.set_ylabel("inboard emission (mean lum)")
    axc = ax0.twinx()
    axc.plot(frames, cmd, color="#b08000", lw=1.2, ls=":", label="puff command")
    axc.set_ylabel("gas command (token proxy)", color="#b08000")
    ax0.set_title(f"shot {sid}: inboard emission vs gas-puff command")
    ax0.legend(loc="upper left", fontsize=8)
    ax0.set_xlabel("frame")

    # bottom: per-frame counterfactual delta (true - no-puff) over the forecast.
    if "cfg_w1.0" in decoded and "nopuff_w1" in decoded:
        a = inboard_emission_series(decoded["cfg_w1.0"])
        b = inboard_emission_series(decoded["nopuff_w1"])
        ax1.bar(frames, a - b, color="#0a7d33", alpha=0.8)
        ax1.axvline(ctx - 0.5, color="#888", ls="--", lw=1.0)
        ax1.axhline(0.0, color="#222", lw=0.8)
        ax1.set_title("counterfactual delta  (true puff − no puff)  inboard emission")
        ax1.set_xlabel("frame")
        ax1.set_ylabel("Δ emission")
    fig.tight_layout()
    panel = out_dir / f"shot-{sid}-control.png"
    fig.savefig(str(panel))
    plt.close(fig)
    return {"control_panel": str(panel)}


# ---------------------------------------------------------------------------
# Aggregate verdict
# ---------------------------------------------------------------------------


def summarise(per_shot: list[dict]) -> dict:
    """Aggregate W1: does emission track the puff + does CFG strengthen it?"""
    bright = {18502, 18505}
    rows = []
    n_timing_pos = n_cf_pos = n_div_exceeds = 0
    n_gas = 0
    n_puff_fired = 0
    for s in per_shot:
        sid = s["shot_id"]
        cfg = s.get("cfg_sweep", {})
        c_w1 = cfg.get("1.0", {})
        timing_w1 = c_w1.get("pearson_emission_vs_command", float("nan"))
        cf = s.get("counterfactual", {}).get("w1.0", {})
        cf_delta = cf.get("counterfactual_delta", float("nan"))
        div = s.get("control_divergence", {})
        cmd_std = s.get("gas_command_forecast_std", 0.0) or 0.0
        # the puff "fired" in-window if the conditioning command actually moved.
        puff_fired = bool(s.get("has_gas") and cmd_std > 0.5)
        row = {
            "shot_id": sid,
            "is_bright": sid in bright,
            "has_gas": s.get("has_gas"),
            "puff_window_found": s.get("puff_window_found"),
            "gas_command_forecast_std": cmd_std,
            "puff_fired_in_window": puff_fired,
            "timing_corr_w1": timing_w1,
            "counterfactual_delta_w1": cf_delta,
            "control_divergence_l1": div.get("control_divergence_l1"),
            "same_conditioning_spread_l1": div.get("same_conditioning_spread_l1"),
            "divergence_over_spread_ratio": div.get("divergence_over_spread_ratio"),
            "control_exceeds_spread": div.get("control_exceeds_spread"),
        }
        # does CFG strengthen the counterfactual delta vs w=1?
        amp = s.get("counterfactual", {}).get("amplified_by_w", {})
        if amp:
            deltas = {w: r.get("counterfactual_delta") for w, r in amp.items()}
            row["counterfactual_delta_by_w"] = deltas
            d1 = deltas.get("1.0")
            d_hi = deltas.get("2.0")
            row["cfg_strengthens_counterfactual"] = bool(
                d1 is not None and d_hi is not None and d_hi > d1
            )
        if s.get("has_gas"):
            n_gas += 1
            if puff_fired:
                n_puff_fired += 1
            if np.isfinite(timing_w1) and timing_w1 > 0.1:
                n_timing_pos += 1
            if np.isfinite(cf_delta) and cf_delta > 0.0:
                n_cf_pos += 1
            if div.get("control_exceeds_spread"):
                n_div_exceeds += 1
        rows.append(row)
    # W1 is only fairly TESTABLE on shots whose puff actually fires in-window; the
    # verdict requires the controllable signal to appear AND beat the noise floor.
    return {
        "per_shot": rows,
        "n_shots": len(per_shot),
        "n_gas_shots": n_gas,
        "n_puff_fired_in_window": n_puff_fired,
        "n_timing_positive": n_timing_pos,
        "n_counterfactual_positive": n_cf_pos,
        "n_control_exceeds_spread": n_div_exceeds,
        "bright_shots": sorted(bright),
        "w1_testable": bool(n_puff_fired > 0),
        "w1_met": bool(
            n_puff_fired > 0
            and n_timing_pos >= 1
            and n_cf_pos >= 1
            and n_div_exceeds >= 1
        ),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _dump(out_json: Path, per_shot: list[dict], summary: dict | None = None) -> None:
    payload = {"per_shot": per_shot}
    if summary is not None:
        payload["summary"] = summary
    out_json.write_text(json.dumps(payload, indent=2, default=str))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--shots", default="18502,18503,18504,18505")
    p.add_argument("--out-json", required=True)
    p.add_argument("--figure-dir", default=None)
    p.add_argument("--camera", default=REFERENCE_CAMERA)
    p.add_argument("--n-frames", type=int, default=24)
    p.add_argument("--n-plan", type=int, default=8)
    p.add_argument("--context-frames", type=int, default=8)
    p.add_argument("--frame-stride", type=int, default=1)
    p.add_argument("--n-signal-steps", type=int, default=4)
    p.add_argument(
        "--guidance", default="1.0,1.5,2.0", help="comma-separated CFG weights"
    )
    p.add_argument("--spread-members", type=int, default=3)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--chunk", type=int, default=8192)
    p.add_argument("--device", default="cuda")
    p.add_argument("--token-root", default=None)
    p.add_argument("--work-dir", default=None)
    p.add_argument(
        "--no-puff-window",
        action="store_true",
        help="score the CENTRED window instead of the puff-transition window "
        "(debug — the centred window is often quasi-static so the puff cannot fire)",
    )
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    _install_signal_handler()

    window = SpacetimeWindowConfig(
        n_frames=args.n_frames,
        n_plan=args.n_plan,
        context_frames=args.context_frames,
        frame_stride=args.frame_stride,
    )
    shot_ids = [int(s) for s in args.shots.split(",") if s.strip()]
    guidance_weights = [float(x) for x in args.guidance.split(",") if x.strip()]
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    base_work = Path(
        args.work_dir
        or tempfile.mkdtemp(prefix="w1-cfg-", dir=os.environ.get("TMPDIR", "/tmp"))
    )

    import torch

    from imas_ambix.worldmodel.spacetime_train_v2 import (
        load_signal_model_from_checkpoint,
    )

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA unavailable — falling back to CPU (slow)")
        device = "cpu"
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if device == "cuda":
        torch.set_float32_matmul_precision("high")
    logger.info("loading v2 checkpoint ONCE on %s: %s", device, args.checkpoint)
    model, payload = load_signal_model_from_checkpoint(
        Path(args.checkpoint), map_location=device
    )
    model.eval()
    logger.info(
        "checkpoint step=%s has_corruption=%s streams=%s",
        payload.get("step"),
        getattr(model, "has_corruption", False),
        [st.name for st in model.config.signal_streams],
    )

    per_shot: list[dict] = []
    try:
        for sid in shot_ids:
            if _STOP["flag"]:
                logger.warning("stop flag set — skipping remaining shots")
                break
            logger.info("==== W1 falsification shot %s ====", sid)
            grids_by_role, roles_index, meta = generate_rollouts(
                model=model,
                payload=payload,
                checkpoint=Path(args.checkpoint),
                shot_id=int(sid),
                window=window,
                guidance_weights=guidance_weights,
                spread_members=args.spread_members,
                temperature=args.temperature,
                top_p=args.top_p,
                n_signal_steps=args.n_signal_steps,
                camera=args.camera,
                device=device,
                token_root=Path(args.token_root) if args.token_root else None,
                chunk=args.chunk,
                puff_window=not args.no_puff_window,
            )
            decoded = decode_roles(
                grids_by_role,
                roles_index,
                work_dir=base_work / f"shot-{sid}",
                device=device,
            )
            scored = score_shot(decoded, meta)
            if args.figure_dir:
                with contextlib.suppress(Exception):
                    scored.update(
                        assemble_figure(decoded, scored, meta, Path(args.figure_dir))
                    )
            per_shot.append(scored)
            _dump(out_json, per_shot)
    finally:
        try:
            del model
            if device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001
            logger.warning("model release note: %r", exc)

    summary = summarise(per_shot)
    _dump(out_json, per_shot, summary)
    _print_verdict(per_shot, summary)
    return 0


def _print_verdict(per_shot: list[dict], summary: dict) -> None:
    print("\n=== W1 CFG + gas-puff falsification VERDICT ===")
    if per_shot:
        print(
            f"checkpoint: {per_shot[0]['checkpoint']} "
            f"(step {per_shot[0]['checkpoint_step']})"
        )
    hdr = (
        f"{'shot':>6} {'bright':>6} {'fired':>6} {'cmd_std':>8} | "
        f"{'timing_w1':>10} {'cf_delta_w1':>12} {'div_l1':>9} {'spread_l1':>10} "
        f"{'div/spread':>10} {'exceeds':>8}"
    )
    print(hdr)
    print("-" * len(hdr))
    for row in summary["per_shot"]:

        def _f(x, fmt="{:>10.3f}"):
            return (
                fmt.format(x)
                if isinstance(x, (int, float)) and x == x
                else f"{x!s:>10}"
            )

        print(
            f"{row['shot_id']:>6} {str(row['is_bright']):>6} "
            f"{str(row.get('puff_fired_in_window')):>6} "
            f"{_f(row.get('gas_command_forecast_std'), '{:>8.3f}'):>8} | "
            f"{_f(row['timing_corr_w1']):>10} "
            f"{_f(row['counterfactual_delta_w1'], '{:>12.4f}'):>12} "
            f"{_f(row['control_divergence_l1'], '{:>9.3f}'):>9} "
            f"{_f(row['same_conditioning_spread_l1'], '{:>10.3f}'):>10} "
            f"{_f(row['divergence_over_spread_ratio'], '{:>10.3f}'):>10} "
            f"{str(row['control_exceeds_spread']):>8}"
        )
    print(
        f"\ngas shots: {summary['n_gas_shots']}; "
        f"puff-fired-in-window: {summary.get('n_puff_fired_in_window')}; "
        f"timing-positive (>0.1): {summary['n_timing_positive']}; "
        f"counterfactual-positive: {summary['n_counterfactual_positive']}; "
        f"control>spread: {summary['n_control_exceeds_spread']}"
    )
    print(f"\nW1 testable (puff fired in a window): {summary.get('w1_testable')}")
    print(f"W1 (responds-to-controls) MET: {summary['w1_met']}")


if __name__ == "__main__":
    raise SystemExit(main())
