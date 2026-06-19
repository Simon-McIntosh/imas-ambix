"""Re-score argmax vs temperature/top-p sampling rollouts of the v2 camera model.

The camera world model collapses to persistence under the greedy *argmax*
rollout.  This driver re-runs the held-out forecast with the SAME signal-
conditioned checkpoint and decode pipeline as
:mod:`imas_ambix.worldmodel.spacetime_dream_v2`, but compares the deterministic
argmax baseline against a small grid of temperature + nucleus (top-p) sampling
settings, scoring each with the distributional / motion / luminance-fair harness
in :mod:`imas_ambix.worldmodel.playability_metrics` (NOT the persistence-favouring
pixel-MAE).  It answers the M1 question: does sampling alone reduce the collapse?

Performance (repo AGENTS.md §2b): the transformer AND the frozen Open-MAGVIT2
VQModel are each loaded ONCE.  Per shot the driver generates the argmax rollout +
an ENSEMBLE of sampled rollouts for every grid setting, packs ALL of them (plus
the GT) into a SINGLE token bundle, and decodes that whole bundle in one VQ pass
(the model-load cost is paid once, the decode batched).  A SIGTERM handler sets a
STOP flag so a cancellation flushes the partial verdict losslessly.

The verdict JSON it writes carries, per shot:
* ``argmax`` — the greedy baseline's pixel-MAE (the prior committed metric, for
  continuity), CRPS, motion, and SSIM reports;
* ``sampling`` — one entry per ``(temperature, top_p)`` grid point, each with the
  ensemble CRPS, the per-member mean motion + the ensemble-mean-frame motion, and
  the SSIM sanity of the first member.
The aggregate verdict (does sampling reduce collapse, and where) is computed by
:func:`summarise` and printed.
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

from imas_ambix.worldmodel.playability_metrics import (
    control_divergence_stub,
    ensemble_crps,
    motion_report,
    ssim_report,
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
from imas_ambix.worldmodel.spacetime_dream import forecast_pixel_errors

logger = logging.getLogger(__name__)

#: Stop flag set by the SIGTERM/SIGINT handler — checked between shots so a
#: cancellation flushes the partial verdict rather than losing it.
_STOP = {"flag": False}


def _install_signal_handler() -> None:
    def _handler(signum, _frame):  # noqa: ANN001
        logger.warning(
            "signal %s received — finishing the current shot then stopping", signum
        )
        _STOP["flag"] = True

    for sig in (signal.SIGTERM, signal.SIGINT):
        # not in main thread / signal unsupported on this platform — ignore.
        with contextlib.suppress(ValueError, OSError):
            signal.signal(sig, _handler)


# ---------------------------------------------------------------------------
# Grid of sampling settings
# ---------------------------------------------------------------------------


def default_grid() -> list[tuple[float, float]]:
    """The small ``(temperature, top_p)`` sweep scored against argmax."""
    temps = (0.7, 0.9, 1.0)
    top_ps = (0.9, 0.95)
    return [(float(t), float(p)) for t in temps for p in top_ps]


# ---------------------------------------------------------------------------
# Per-shot rollout generation (token space) — model loaded ONCE by the caller
# ---------------------------------------------------------------------------


def _roll_local(
    model,
    sample,
    *,
    stream_names,
    device,
    temperature: float,
    top_p: float,
    seed: int,
    chunk: int,
):
    """One autoregressive rollout (local token ids ``(F, S)``).

    ``temperature <= 0`` is the greedy argmax baseline; a positive temperature
    draws a nucleus sample seeded by ``seed`` for reproducibility.
    """
    import torch

    from imas_ambix.worldmodel.spacetime_train_v2 import autoregressive_signal_dream

    gen = None
    if temperature and temperature > 0.0:
        # the generator MUST live on the same device as the sampled probabilities
        # (torch.multinomial rejects a cpu generator for cuda probs); build it on
        # the rollout device so the draw is reproducible AND device-correct.
        gen = torch.Generator(device=torch.device(device)).manual_seed(int(seed))
    return autoregressive_signal_dream(
        model,
        sample,
        stream_names=stream_names,
        device=device,
        temperature=float(temperature),
        top_p=float(top_p),
        generator=gen,
        chunk=chunk,
    ).reshape(-1, GRID_H, GRID_W)


def generate_rollouts(
    *,
    checkpoint: Path,
    shot_id: int,
    window: SpacetimeWindowConfig,
    grid: list[tuple[float, float]],
    ensemble: int,
    modalities: list[SignalModalitySpec] | None = None,
    n_signal_steps: int = 4,
    camera: str = REFERENCE_CAMERA,
    device: str = "cuda",
    token_root: Path | None = None,
    chunk: int = 8192,
    model=None,
    payload: dict | None = None,
) -> tuple[dict, dict, list, dict]:
    """Build GT + argmax + per-setting sampled rollouts for one shot.

    Returns ``(grids_by_role, roles_index, packing, meta)`` where
    ``grids_by_role`` maps a role key -> ``(F, 16, 16)`` STORE-id grid, ready to
    pack into a decode bundle.  ``model``/``payload`` may be passed in so the
    caller loads the (12 GB) checkpoint ONCE across shots; if ``None`` they are
    loaded here.
    """
    import torch

    from imas_ambix.worldmodel.spacetime_train_v2 import (
        load_signal_model_from_checkpoint,
    )

    modalities = modalities or default_signal_modalities()
    if model is None:
        torch.manual_seed(0)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if device == "cuda" and torch.cuda.is_available():
            torch.set_float32_matmul_precision("high")
        model, payload = load_signal_model_from_checkpoint(
            checkpoint, map_location=device
        )
        model.eval()

    step = int((payload or {}).get("step", -1))
    model_streams = [st.name for st in model.config.signal_streams]
    dev = torch.device(device)

    sample = assemble_signal_window(
        int(shot_id),
        window,
        modalities,
        int(n_signal_steps),
        camera=camera,
        token_root=token_root,
    )
    present = sorted(sample.signals.keys())
    ctx = int(sample.context_frames)
    n_frames = int(sample.frames.shape[0])

    gt_local = np.asarray(sample.frames, dtype=np.int64).reshape(
        n_frames, GRID_H, GRID_W
    )

    grids_by_role: dict[str, np.ndarray] = {"gt": local_to_store(gt_local)}
    roles_index: list[dict] = [{"role": "gt"}]
    packing: list[dict] = []  # describes how to regroup decoded roles into scores

    # argmax baseline.
    t0 = time.time()
    argmax_local = _roll_local(
        model,
        sample,
        stream_names=model_streams,
        device=dev,
        temperature=0.0,
        top_p=1.0,
        seed=0,
        chunk=chunk,
    )
    grids_by_role["argmax"] = local_to_store(argmax_local)
    roles_index.append({"role": "argmax"})
    packing.append(
        {"setting": "argmax", "temperature": 0.0, "top_p": 1.0, "members": ["argmax"]}
    )
    logger.info("shot %s argmax rollout in %.1fs", shot_id, time.time() - t0)

    # sampling grid — an ensemble per setting.
    for ti, (temp, top_p) in enumerate(grid):
        members: list[str] = []
        t0 = time.time()
        for m in range(int(ensemble)):
            # distinct, reproducible seed per (setting, member, shot).
            seed = (int(shot_id) * 100003) ^ (ti * 1009) ^ (m * 31) ^ 0x5A5A
            roll = _roll_local(
                model,
                sample,
                stream_names=model_streams,
                device=dev,
                temperature=temp,
                top_p=top_p,
                seed=seed,
                chunk=chunk,
            )
            key = f"s_t{temp}_p{top_p}_m{m}"
            grids_by_role[key] = local_to_store(roll)
            roles_index.append({"role": key})
            members.append(key)
        packing.append(
            {
                "setting": f"T={temp}_p={top_p}",
                "temperature": temp,
                "top_p": top_p,
                "members": members,
            }
        )
        logger.info(
            "shot %s sampling T=%.2f p=%.2f x%d in %.1fs",
            shot_id,
            temp,
            top_p,
            ensemble,
            time.time() - t0,
        )
        if _STOP["flag"]:
            logger.warning(
                "stop flag set — truncating sampling grid for shot %s", shot_id
            )
            break

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
        "n_signal_steps": int(n_signal_steps),
        "ensemble": int(ensemble),
        "frame_time": np.asarray(sample.frame_time, dtype=np.float64).tolist(),
    }
    return grids_by_role, roles_index, packing, meta


# ---------------------------------------------------------------------------
# Decode (one VQ pass over all roles) + score
# ---------------------------------------------------------------------------


def decode_roles(
    grids_by_role: dict[str, np.ndarray],
    roles_index: list[dict],
    *,
    work_dir: Path,
    device: str = "cuda",
) -> dict[str, np.ndarray]:
    """Decode every role's token grid to ``(F,256,256,3)`` in ONE VQ pass.

    Packs all roles into a single ``(N,F,16,16)`` bundle, calls the frozen
    Open-MAGVIT2 decode subprocess once, and returns ``{role: image_stack}``.
    """
    from imas_ambix.camdyn.reconstruction_demo import run_decode_subprocess

    work_dir.mkdir(parents=True, exist_ok=True)
    token_bundle = work_dir / "tokens.npz"
    image_bundle = work_dir / "images.npz"

    keys = [e["role"] for e in roles_index]
    grids = np.stack([grids_by_role[k] for k in keys]).astype(np.int64)  # (N,F,16,16)
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
    images = np.asarray(data["images"], dtype=np.uint8)  # (N,F,256,256,3)
    idx = json.loads(str(data["index"]))
    slot = {e["role"]: e["slot"] for e in idx}
    return {k: images[slot[k]] for k in keys}


def score_shot(
    decoded: dict[str, np.ndarray],
    packing: list,
    meta: dict,
) -> dict:
    """Score argmax + every sampling setting for one shot from decoded frames."""
    ctx = int(meta["context_frames"])
    gt = decoded["gt"]

    out: dict = {
        "shot_id": meta["shot_id"],
        "checkpoint": meta["checkpoint"],
        "checkpoint_step": meta["checkpoint_step"],
        "context_frames": ctx,
        "n_frames": meta["n_frames"],
        "present_streams": meta["present_streams"],
        "ensemble": meta["ensemble"],
        "control_divergence": control_divergence_stub(),
    }

    def _setting_scores(members: list[str]) -> dict:
        ens = np.stack([decoded[m] for m in members], axis=0)  # (M,F,256,256,3)
        crps = ensemble_crps(gt, ens, ctx)
        # motion: per-member mean change-fraction + collapse ratio of member 0.
        per_member = [motion_report(gt, decoded[m], ctx) for m in members]
        mean_pred_cf = float(np.mean([r["pred_change_fraction"] for r in per_member]))
        gt_cf = per_member[0]["gt_change_fraction"]
        pers_cf = per_member[0]["persistence_change_fraction"]
        collapse_ratio = float("inf") if gt_cf == 0.0 else mean_pred_cf / gt_cf
        # pixel-MAE (prior metric) of member 0 + SSIM sanity of member 0.
        px = forecast_pixel_errors(gt, decoded[members[0]], ctx)
        ssim = ssim_report(gt, decoded[members[0]], ctx)
        return {
            "crps": crps,
            "motion": {
                "gt_change_fraction": gt_cf,
                "persistence_change_fraction": pers_cf,
                "mean_pred_change_fraction": mean_pred_cf,
                "collapse_ratio": collapse_ratio,
            },
            "pixel_mae_member0": px,
            "ssim_sanity_member0": ssim,
        }

    for entry in packing:
        scores = _setting_scores(entry["members"])
        rec = {
            "temperature": entry["temperature"],
            "top_p": entry["top_p"],
            **scores,
        }
        if entry["setting"] == "argmax":
            out["argmax"] = rec
        else:
            out.setdefault("sampling", {})[entry["setting"]] = rec
    return out


# ---------------------------------------------------------------------------
# Aggregate verdict
# ---------------------------------------------------------------------------


def summarise(per_shot: list[dict]) -> dict:
    """Aggregate: does sampling reduce the collapse, and where?

    For each shot picks the BEST sampling setting by CRPS ratio and compares it to
    the argmax baseline on (a) the distributional CRPS-vs-persistence ratio and (b)
    the motion collapse-ratio.  Reports how many shots improve, split by the
    bright/high-motion vs dim sets.
    """
    bright = {18502, 18505}
    summary = {"per_shot": [], "n_shots": len(per_shot)}
    crps_better = motion_better = crps_beats_pers = 0
    for s in per_shot:
        sid = s["shot_id"]
        am = s.get("argmax", {})
        am_crps = am.get("crps", {}).get("ratio", float("nan"))
        am_collapse = am.get("motion", {}).get("collapse_ratio", float("nan"))
        sampling = s.get("sampling", {})
        # best sampling by CRPS ratio (lower is better).
        best_name, best = None, None
        for name, rec in sampling.items():
            r = rec["crps"]["ratio"]
            if best is None or r < best["crps"]["ratio"]:
                best_name, best = name, rec
        row = {
            "shot_id": sid,
            "is_bright": sid in bright,
            "argmax_crps_ratio": am_crps,
            "argmax_collapse_ratio": am_collapse,
            "best_sampling_setting": best_name,
            "best_sampling_crps_ratio": best["crps"]["ratio"] if best else None,
            "best_sampling_collapse_ratio": (
                best["motion"]["collapse_ratio"] if best else None
            ),
            "best_sampling_crps_beats_persistence": (
                best["crps"]["model_beats_persistence"] if best else None
            ),
        }
        if best is not None:
            if best["crps"]["ratio"] < am_crps:
                crps_better += 1
            # closer to GT motion == |collapse_ratio - 1| smaller (and not over).
            if abs(best["motion"]["collapse_ratio"] - 1.0) < abs(am_collapse - 1.0):
                motion_better += 1
            if best["crps"]["model_beats_persistence"]:
                crps_beats_pers += 1
        summary["per_shot"].append(row)

    summary["sampling_improves_crps_over_argmax"] = crps_better
    summary["sampling_motion_closer_to_gt_than_argmax"] = motion_better
    summary["sampling_crps_beats_persistence"] = crps_beats_pers
    summary["bright_shots"] = sorted(bright)
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--shots", default="18502,18503,18504,18505")
    p.add_argument("--out-json", required=True)
    p.add_argument("--camera", default=REFERENCE_CAMERA)
    p.add_argument("--n-frames", type=int, default=24)
    p.add_argument("--n-plan", type=int, default=8)
    p.add_argument("--context-frames", type=int, default=8)
    p.add_argument("--frame-stride", type=int, default=1)
    p.add_argument("--n-signal-steps", type=int, default=4)
    p.add_argument("--ensemble", type=int, default=4)
    p.add_argument("--chunk", type=int, default=8192)
    p.add_argument("--device", default="cuda")
    p.add_argument("--token-root", default=None)
    p.add_argument("--work-dir", default=None)
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
    grid = default_grid()
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    base_work = Path(
        args.work_dir
        or tempfile.mkdtemp(prefix="m1-rescore-", dir=os.environ.get("TMPDIR", "/tmp"))
    )

    # Load the model ONCE for all shots (12 GB checkpoint; AGENTS.md §2b).
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

    per_shot: list[dict] = []
    try:
        for sid in shot_ids:
            if _STOP["flag"]:
                logger.warning("stop flag set — skipping remaining shots")
                break
            logger.info("==== re-score shot %s ====", sid)
            grids_by_role, roles_index, packing, meta = generate_rollouts(
                checkpoint=Path(args.checkpoint),
                shot_id=int(sid),
                window=window,
                grid=grid,
                ensemble=args.ensemble,
                n_signal_steps=args.n_signal_steps,
                camera=args.camera,
                device=device,
                token_root=Path(args.token_root) if args.token_root else None,
                chunk=args.chunk,
                model=model,
                payload=payload,
            )
            decoded = decode_roles(
                grids_by_role,
                roles_index,
                work_dir=base_work / f"shot-{sid}",
                device=device,
            )
            scored = score_shot(decoded, packing, meta)
            per_shot.append(scored)
            # flush incrementally so a SIGTERM keeps what we have.
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


def _dump(out_json: Path, per_shot: list[dict], summary: dict | None = None) -> None:
    payload = {"per_shot": per_shot}
    if summary is not None:
        payload["summary"] = summary
    out_json.write_text(json.dumps(payload, indent=2, default=str))


def _print_verdict(per_shot: list[dict], summary: dict) -> None:
    print("\n=== M1 argmax-vs-sampling re-score VERDICT ===")
    if per_shot:
        print(
            f"checkpoint: {per_shot[0]['checkpoint']} "
            f"(step {per_shot[0]['checkpoint_step']})"
        )
    hdr = (
        f"{'shot':>6} {'bright':>6} | {'argmax_CRPSr':>12} {'argmax_clps':>11} | "
        f"{'best_set':>14} {'samp_CRPSr':>10} {'samp_clps':>9} {'beats_pers':>10}"
    )
    print(hdr)
    print("-" * len(hdr))
    for row in summary["per_shot"]:
        am_r = row["argmax_crps_ratio"]
        am_c = row["argmax_collapse_ratio"]
        s_r = row["best_sampling_crps_ratio"] or float("nan")
        s_c = row["best_sampling_collapse_ratio"] or float("nan")
        print(
            f"{row['shot_id']:>6} {str(row['is_bright']):>6} | "
            f"{am_r:>12.3f} {am_c:>11.3f} | "
            f"{str(row['best_sampling_setting']):>14} {s_r:>10.3f} {s_c:>9.3f} "
            f"{str(row['best_sampling_crps_beats_persistence']):>10}"
        )
    n = summary["n_shots"]
    n_crps = summary["sampling_improves_crps_over_argmax"]
    n_motion = summary["sampling_motion_closer_to_gt_than_argmax"]
    n_beats = summary["sampling_crps_beats_persistence"]
    print(
        f"\nsampling improves CRPS-ratio over argmax on {n_crps}/{n} shots; "
        f"motion closer to GT on {n_motion}/{n}; "
        f"CRPS beats persistence on {n_beats}/{n}."
    )


if __name__ == "__main__":
    raise SystemExit(main())
