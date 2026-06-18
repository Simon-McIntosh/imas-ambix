"""Predict-vs-reality artifact bundle for the plan-conditioned world model.

This builds the self-describing artifact bundle a downstream figure-rendering
step consumes to show, on GENUINELY HELD-OUT shots, what the converged world
model forecasts versus reality — token forward-prediction skill vs persistence
per modality, the camera 4x4 subgrid prediction-vs-truth token arrays, and the
full-resolution ground-truth camera frames decoded back to images.

Honesty contract (binding)
--------------------------
* Eval runs ONLY on held-out shots — shots discovered at indices ``>= 4000``,
  beyond the ``limit=4000`` training discovery cap (training fit the first
  3996; 4 held out).  The chosen shots are LOGGED with their discovery index so
  a reader can confirm they were never trained on.
* Skill is reported warts-and-all, per modality, for every scored modality and
  every chosen shot — including NEGATIVE skill vs persistence.  Persistence is a
  very strong baseline on quasi-stationary signals at short horizons; the model
  losing to it on some modalities is expected and is reported, not hidden.
* The camera modality the model predicts is a COARSE 4x4=16-token subsample of
  the 16x16 frame grid (``camera_grid_stride=4``).  The bundle saves the
  predicted-vs-truth SUBGRID token arrays (for a token-grid heatmap) — it never
  decodes the model's coarse prediction to a full-res image.  The full-res
  decode is GROUND TRUTH ONLY (the on-disk 256-token frames), so a decoded
  image is never mislabelled as "the model's prediction".

Two phases (mirrors :mod:`imas_ambix.camdyn.reconstruction_demo`)
-----------------------------------------------------------------
PHASE A — WM venv, GPU.  Load the converged checkpoint ONCE, loop over the
held-out shots, roll out, score skill, dump the signal/skill/param bundle plus a
ground-truth camera token-grid bundle.

PHASE B — Open-MAGVIT2 venv, GPU.  Decode the ground-truth camera token grids to
256x256 images by re-using
:func:`imas_ambix.camdyn.reconstruction_demo.run_decode_subprocess` (the
established two-venv decode handoff).  If the MAGVIT2 interpreter/weights are
absent, the bundle records a clear "camera decode unavailable" note and the
build continues (the doc then uses the token-grid heatmap only).

Run (single GPU, betelgeuse; keep the neighbour's LLM server up)::

    .venv/bin/python -m imas_ambix.worldmodel.demo_artifacts
"""

from __future__ import annotations

import json
import logging
import signal
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from imas_ambix.worldmodel.dataset import CAMERA_IDS as CAMERA_NAMES

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fixed locations (the converged run + the on-disk corpus).
# ---------------------------------------------------------------------------

#: The converged checkpoint (step 40000, loss 1.69, 1.034B params).
CHECKPOINT = Path("/work/projects/imas_gpu/worldmodel/ckpt/1219352/latest.pt")

#: Where the self-describing artifact bundle lands (data, NOT git-tracked).
OUTPUT_DIR = Path("/work/projects/imas_gpu/worldmodel/demo")

#: Full-resolution camera frame token stores (16x16 = 256-token GLOBAL ids).
FRAMES_ROOT = Path("/work/projects/imas_gpu/mast-tokens/v1/frames")

#: The window used at training time — must match (model token context = 128 =
#: plan_steps 64 + obs_steps 64; the common-grid window is n_steps=64 with the
#: first context_steps=16 given as the initial-condition context).
WINDOW_N_STEPS = 64
WINDOW_CONTEXT_STEPS = 16

#: Training discovered shots with ``limit=4000`` and trained on the first 3996.
#: A held-out shot is one whose discovery index is ``>= TRAIN_DISCOVERY_LIMIT``,
#: so we re-discover with a LARGER limit and keep only the tail.
TRAIN_DISCOVERY_LIMIT = 4000
DISCOVERY_LIMIT = 4200

#: How many held-out, rbb-bearing-and-core shots to score (target 6-8).
N_HELDOUT_TARGET = 8
N_HELDOUT_MIN = 6

#: The reference camera whose full-res frames we decode for the qualitative row.
REFERENCE_CAMERA = "rbb"

#: Contiguous run of full-res frames to decode per shot (spanning the window).
GT_FRAME_RUN = 48


# ---------------------------------------------------------------------------
# Clean-cancellation stop flag (repo §2b GPU-safety pattern).
# ---------------------------------------------------------------------------


class _StopFlag:
    """SIGTERM/SIGINT-set stop flag the per-shot loop polls (clean cancel)."""

    def __init__(self) -> None:
        self.stop = False

    def install(self) -> None:
        def _handler(signum, _frame):  # noqa: ANN001
            logger.warning("received signal %s — setting STOP flag", signum)
            self.stop = True

        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                signal.signal(sig, _handler)
            except (ValueError, OSError):
                logger.debug("could not install handler for %s", sig)


# ---------------------------------------------------------------------------
# Held-out shot selection (CRITICAL for eval honesty)
# ---------------------------------------------------------------------------


@dataclass
class HeldoutPick:
    """One held-out shot kept for the eval, with its discovery provenance."""

    shot_id: int
    discovery_index: int  # position in the (seed-stable) discovery ordering


def select_heldout_shots(
    modalities,
    *,
    token_root: Path | None = None,
    limit: int = DISCOVERY_LIMIT,
    train_cap: int = TRAIN_DISCOVERY_LIMIT,
    n_target: int = N_HELDOUT_TARGET,
) -> tuple[list[HeldoutPick], int]:
    """Pick held-out shots beyond the training discovery cap that carry rbb+core.

    Re-runs the EXACT discovery the trainer used
    (``discover_worldmodel_shots(default_modalities(), limit=...)`` —
    seed-stable ``camera_first`` sampling), takes only shots at indices
    ``>= train_cap`` (guaranteed NOT in the training set), and keeps those that
    carry BOTH the rbb camera store AND assemble cleanly with the core signals.

    Returns ``(picks, n_heldout_rbb_candidates)`` — the kept picks and how many
    held-out rbb-bearing shots existed in the tail (so the caller can report the
    pool size if fewer than the target qualify).
    """
    from imas_ambix.worldmodel.dataset import (
        _modality_store_present,
        discover_worldmodel_shots,
    )

    # Reproduce the trainer's discovery ordering EXACTLY (default seed + sampling
    # mode), just with a larger limit so the tail beyond the training cap is
    # exposed.  Same modality set, same seed -> the first ``train_cap`` entries
    # are byte-identical to what training saw; the entries at index >= train_cap
    # are the never-trained tail.
    ordered = discover_worldmodel_shots(modalities, token_root=token_root, limit=limit)
    if len(ordered) <= train_cap:
        raise RuntimeError(
            f"discovery returned only {len(ordered)} shots at limit={limit}; "
            f"need > {train_cap} to expose a held-out tail"
        )
    tail = ordered[train_cap:]  # discovery indices >= train_cap — never trained

    root = Path(token_root) if token_root is not None else None
    cam_spec = next((m for m in modalities if m.name == REFERENCE_CAMERA), None)
    if cam_spec is None:
        raise RuntimeError(f"no {REFERENCE_CAMERA!r} modality in the declared set")

    # Held-out shots that carry the rbb camera store on disk.
    rbb_tail: list[HeldoutPick] = []
    for offset, sid in enumerate(tail):
        if _modality_store_present(int(sid), cam_spec, root):
            rbb_tail.append(HeldoutPick(int(sid), train_cap + offset))

    logger.info(
        "held-out tail: %d shots beyond train cap %d; %d carry the %s camera",
        len(tail),
        train_cap,
        len(rbb_tail),
        REFERENCE_CAMERA,
    )

    # Keep up to n_target that ALSO assemble cleanly (core signals present +
    # windows overlap).  Assembly failures are logged, not hidden.
    picks: list[HeldoutPick] = []
    from imas_ambix.worldmodel.dataset import WorldModelWindowConfig, build_shot_sample

    window = WorldModelWindowConfig(
        n_steps=WINDOW_N_STEPS, context_steps=WINDOW_CONTEXT_STEPS
    )
    for cand in rbb_tail:
        if len(picks) >= n_target:
            break
        try:
            sample = build_shot_sample(
                cand.shot_id, modalities, window, token_root=token_root
            )
        except (ValueError, FileNotFoundError, KeyError) as exc:
            logger.info("held-out candidate %s NOT assemblable: %r", cand.shot_id, exc)
            continue
        if REFERENCE_CAMERA not in sample.tokens:
            logger.info(
                "held-out candidate %s assembled WITHOUT %s tokens — skipping",
                cand.shot_id,
                REFERENCE_CAMERA,
            )
            continue
        picks.append(cand)
    return picks, len(rbb_tail)


# ---------------------------------------------------------------------------
# Parameter breakdown by component
# ---------------------------------------------------------------------------


def param_breakdown(model, camera_names) -> dict[str, int]:
    """Group ``model.named_parameters()`` counts into reportable components.

    Groups:

    * ``camera_embed_head`` — the per-camera embedding tables + next-token heads
      + channel queries + modality markers (the 2^18-vocab driver);
    * ``transformer_backbone`` — attention + FFN + layer-norms (the ``blocks``
      stack and the final norm);
    * ``other_modality_embed_head`` — the non-camera modality embedding tables,
      heads, channel queries, modality markers (plan + L2 + HF streams);
    * ``other`` — everything else (position + segment embeddings).

    Returns ``{group: param_count}``; the caller sanity-checks the sum against
    ``model.num_parameters()``.
    """
    cams = set(camera_names)
    groups: dict[str, int] = {
        "camera_embed_head": 0,
        "transformer_backbone": 0,
        "other_modality_embed_head": 0,
        "other": 0,
    }
    # the per-modality top-level containers whose 2nd path component is a
    # modality name (camera vs other-modality routing).
    modality_containers = {
        "token_embed",
        "modality_embed",
        "heads",
        "channel_query",
    }
    for name, p in model.named_parameters():
        n = int(p.numel())
        top = name.split(".")[0]
        if top in ("blocks", "ln_f"):
            groups["transformer_backbone"] += n
        elif top in modality_containers:
            parts = name.split(".")
            mod_name = parts[1] if len(parts) > 1 else ""
            if mod_name in cams:
                groups["camera_embed_head"] += n
            else:
                groups["other_modality_embed_head"] += n
        else:
            # pos_embed, segment_embed, any stray top-level parameter
            groups["other"] += n
    return groups


# ---------------------------------------------------------------------------
# Phase A — predict + score + build the signal/camera bundles
# ---------------------------------------------------------------------------


def _persistence_tokens(tokens: np.ndarray, context_steps: int) -> np.ndarray:
    """Persistence baseline: repeat the last context token through the window.

    ``tokens`` is ``(n_steps, n_channels)``.  Returns the SAME shape with the
    target window (steps ``>= context_steps``) overwritten by the last context
    step's tokens — the strong quasi-stationary baseline the skill is measured
    against.
    """
    out = np.asarray(tokens).copy()
    last_ctx = out[context_steps - 1]
    out[context_steps:] = last_ctx[None, :]
    return out


def _load_gt_camera_grids(
    shot_id: int, camera: str, n_run: int
) -> tuple[np.ndarray, dict] | None:
    """Load a contiguous run of full-res (16x16) GT camera token grids.

    Reads the on-disk frame-token store directly (the established
    reconstruction-demo bundle format), which holds GLOBAL ids (REGISTRY_OFFSET
    already applied).  Returns ``(grids (F,16,16) int64, info)`` where ``info``
    records the shot/camera/start/frame-count, or ``None`` if the store is
    missing/unreadable.  The run is centred so it spans the eval window's middle
    rather than the dark ramp-up.
    """
    import zarr  # noqa: PLC0415

    store_path = FRAMES_ROOT / str(shot_id) / f"{camera}.zarr"
    if not store_path.exists():
        logger.info("GT camera store missing for shot %s: %s", shot_id, store_path)
        return None
    try:
        grp = zarr.open_group(str(store_path), mode="r")
        grid = np.asarray(grp["tokens"], dtype=np.int64)  # (T,16,16) GLOBAL ids
    except Exception as exc:  # noqa: BLE001 — a flaky store is just skipped
        logger.warning("GT camera store unreadable for shot %s: %r", shot_id, exc)
        return None
    n_total = int(grid.shape[0])
    if n_total == 0:
        return None
    take = min(n_run, n_total)
    # centre the contiguous run on the middle of the recording (the established
    # plasma, not the dark ramp-up / aborted tail).
    start = max(0, (n_total - take) // 2)
    sub = grid[start : start + take]
    info = {
        "shot_id": int(shot_id),
        "camera": camera,
        "start_frame": int(start),
        "n_frames": int(take),
        "n_frames_total": int(n_total),
    }
    return np.asarray(sub, dtype=np.int64), info


def run_phase_a(
    *,
    checkpoint: Path = CHECKPOINT,
    out_dir: Path = OUTPUT_DIR,
    token_root: Path | None = None,
) -> dict:
    """Load the converged model ONCE, eval the held-out shots, dump the bundles.

    Returns a summary dict (held-out picks + per-shot/averaged skill + the param
    breakdown) so the caller can print/report it without re-reading the bundle.
    """
    import torch

    from imas_ambix.worldmodel.dataset import (
        WorldModelWindowConfig,
        build_shot_sample,
        default_modalities,
    )
    from imas_ambix.worldmodel.eval import (
        _model_obs_plan_names,
        rollout,
        score_skill,
    )
    from imas_ambix.worldmodel.train import load_model_from_checkpoint

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stop = _StopFlag()
    stop.install()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # determinism (repo §2b) — matches the training / encode flags.
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if device == "cuda":
        torch.set_float32_matmul_precision("high")

    modalities = default_modalities()
    window = WorldModelWindowConfig(
        n_steps=WINDOW_N_STEPS, context_steps=WINDOW_CONTEXT_STEPS
    )

    # ── held-out shot selection (honesty gate) ──────────────────────────────
    picks, n_rbb_pool = select_heldout_shots(modalities, token_root=token_root)
    if not picks:
        raise RuntimeError(
            f"no held-out rbb-bearing shots qualified (pool of "
            f"{n_rbb_pool} rbb-bearing held-out shots, none assembled)"
        )
    logger.info(
        "held-out picks (discovery index >= %d): %s",
        TRAIN_DISCOVERY_LIMIT,
        [(p.shot_id, p.discovery_index) for p in picks],
    )
    if len(picks) < N_HELDOUT_MIN:
        logger.warning(
            "only %d held-out rbb-bearing shots qualified (target %d); "
            "rbb-bearing held-out pool size was %d",
            len(picks),
            N_HELDOUT_MIN,
            n_rbb_pool,
        )

    # ── load the converged model ONCE (in-process; §2b) ─────────────────────
    # Loaded on the GPU (the prescribed map_location) so the 12 GB checkpoint
    # restore + the param-breakdown introspection run on-device.  The rollout
    # itself runs on CPU: ``eval.rollout`` assembles its batch tensors on CPU
    # (``pad_collate_batch``) and the model forward picks its device from the
    # parameters, so the model is moved to CPU for the per-shot rollouts — the
    # ESTABLISHED CPU-eval path (``train._run_periodic_eval`` does exactly this).
    # A mixed CPU-input / CUDA-weight forward is the device-mismatch crash.
    logger.info("loading converged checkpoint on %s: %s", device, checkpoint)
    model, payload = load_model_from_checkpoint(checkpoint, map_location=device)
    model.eval()
    obs_names, plan_names = _model_obs_plan_names(model)
    cam_names = [m.name for m in model.config.modalities if m.name in CAMERA_NAMES]

    summary: dict = {
        "checkpoint": str(checkpoint),
        "checkpoint_step": int(payload.get("step", -1)),
        "n_parameters": int(model.num_parameters()),
        "context_length": int(model.context_length()),
        "window": {"n_steps": WINDOW_N_STEPS, "context_steps": WINDOW_CONTEXT_STEPS},
        "obs_names": list(obs_names),
        "plan_names": list(plan_names),
        "heldout_picks": [
            {"shot_id": p.shot_id, "discovery_index": p.discovery_index} for p in picks
        ],
        "heldout_rbb_pool_size": int(n_rbb_pool),
        "train_discovery_limit": TRAIN_DISCOVERY_LIMIT,
        "per_shot_skill": {},
        "param_breakdown": {},
    }

    # ── parameter breakdown (component-grouped) ─────────────────────────────
    breakdown = param_breakdown(model, cam_names)
    total = int(model.num_parameters())
    breakdown_sum = int(sum(breakdown.values()))
    summary["param_breakdown"] = {
        "groups": breakdown,
        "sum": breakdown_sum,
        "model_num_parameters": total,
        "matches_total": bool(breakdown_sum == total),
        "matches_logged_1_034_027_033": bool(total == 1_034_027_033),
    }
    logger.info(
        "param breakdown: %s | sum=%d num_parameters=%d match=%s",
        breakdown,
        breakdown_sum,
        total,
        breakdown_sum == total,
    )

    # Move the model to CPU for the rollouts — ``eval.rollout`` builds CPU batch
    # tensors and the model reads its device from the parameters, so a CUDA model
    # would device-mismatch the CPU inputs.  This frees the GPU during the
    # CPU-bound rollout, leaving the neighbour's LLM server its cards.
    if device == "cuda":
        model.to("cpu")
        torch.cuda.empty_cache()
        logger.info("moved model to CPU for the rollout (frees the GPU)")

    # ── per-shot eval loop (model loaded ONCE; loop over shots) ─────────────
    # Per-modality arrays over the FULL window are stored flat under
    # shot-prefixed keys; the bundle is self-describing via the JSON index.
    signal_arrays: dict[str, np.ndarray] = {}
    bundle_index: list[dict] = []
    gt_grids: list[np.ndarray] = []
    gt_index: list[dict] = []

    watchdog_budget = 600.0  # s per shot (auto-relaxed below if needed)
    for pick in picks:
        if stop.stop:
            logger.warning("STOP flag set — ending eval loop early")
            break
        sid = pick.shot_id
        t0 = time.time()
        try:
            sample = build_shot_sample(sid, modalities, window, token_root=token_root)
        except (ValueError, FileNotFoundError, KeyError) as exc:
            logger.warning("shot %s NOT assemblable at eval time: %r", sid, exc)
            continue

        with torch.no_grad():
            predicted = rollout(model, sample, obs_names, plan_names)
        skill = score_skill(sample, predicted, obs_names)

        # record per-modality skill (warts-and-all: every scored modality)
        shot_skill: dict[str, dict] = {}
        for name, s in skill.items():
            shot_skill[name] = {
                "model_error": float(s.model_error),
                "persistence_error": float(s.persistence_error),
                "skill": float(s.skill),
                "n_scored": int(s.n_scored),
            }
        summary["per_shot_skill"][str(sid)] = shot_skill

        # save per-modality FULL-window arrays: truth / valid / prediction /
        # persistence baseline.  Only modalities the shot carries AND the
        # rollout predicted are saved (the scorable set).
        ctx = int(sample.context_steps)
        for name in obs_names:
            if name not in sample.tokens or name not in predicted:
                continue
            c = int(predicted[name].shape[1])
            truth = np.asarray(sample.tokens[name], dtype=np.int64)[:, :c]
            valid = np.asarray(sample.valid[name], dtype=bool)[:, :c]
            pred = np.asarray(predicted[name], dtype=np.int64)  # (n_steps, c)
            persist = _persistence_tokens(truth, ctx)
            key = f"{sid}::{name}"
            signal_arrays[f"{key}::truth"] = truth
            signal_arrays[f"{key}::valid"] = valid
            signal_arrays[f"{key}::pred"] = pred
            signal_arrays[f"{key}::persist"] = persist
            bundle_index.append(
                {
                    "shot_id": int(sid),
                    "modality": name,
                    "key": key,
                    "n_steps": int(truth.shape[0]),
                    "n_channels": int(c),
                    "context_steps": ctx,
                    "is_camera": bool(name in CAMERA_NAMES),
                }
            )

        # camera predicted-vs-truth 4x4 subgrid token arrays over time (for a
        # token-grid heatmap — NO decode).  These live in the same signal bundle
        # under the camera modality keys above; we additionally tag them so the
        # renderer can find the camera subgrids quickly.
        # (the rbb subgrid is already saved above as f"{sid}::rbb::{truth,pred}")

        # full-res GT camera token grids (256 tokens = 16x16) for the decode
        # phase — the ESTABLISHED reconstruction-demo bundle format.
        loaded = _load_gt_camera_grids(sid, REFERENCE_CAMERA, GT_FRAME_RUN)
        if loaded is not None:
            grid, info = loaded
            info["slot"] = len(gt_grids)
            gt_grids.append(grid)
            gt_index.append(info)

        dt = time.time() - t0
        logger.info("shot %s scored in %.1fs (modalities=%d)", sid, dt, len(shot_skill))
        if dt > watchdog_budget:
            logger.warning(
                "shot %s took %.0fs (> watchdog %0.f s) — continuing but slow",
                sid,
                dt,
                watchdog_budget,
            )

    # ── averaged skill across the scored shots (per modality + overall) ─────
    summary["averaged_skill"] = _average_skill(summary["per_shot_skill"])

    # ── release the model (repo §2b: try/finally happens in main) ───────────
    try:
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    except Exception as exc:  # noqa: BLE001
        logger.warning("model release note: %r", exc)

    # ── write the two bundles ────────────────────────────────────────────────
    eval_bundle_path = out_dir / "eval_bundle.npz"
    np.savez_compressed(
        eval_bundle_path,
        index=json.dumps(bundle_index),
        summary=json.dumps(summary),
        **signal_arrays,
    )
    logger.info("wrote eval bundle -> %s", eval_bundle_path)

    gt_bundle_path = out_dir / "gt_camera_tokens.npz"
    if gt_grids:
        # ragged frame counts per shot would break a single stack; the
        # reconstruction-demo decode bundle wants ONE (N,F,16,16) stack, so we
        # pad every shot's run to the max frame count and record the true
        # n_frames per slot in the index (the decode phase decodes the padded
        # stack; the renderer trims to n_frames).
        max_f = max(g.shape[0] for g in gt_grids)
        padded: list[np.ndarray] = []
        for g, info in zip(gt_grids, gt_index, strict=True):
            f = g.shape[0]
            if f < max_f:
                pad = np.full((max_f - f, 16, 16), -1, dtype=np.int64)  # -1 = black
                g = np.concatenate([g, pad], axis=0)
            padded.append(g)
            info["n_frames_padded"] = int(max_f)
        grids_stack = np.stack(padded).astype(np.int64)  # (N,F,16,16) GLOBAL ids
        np.savez_compressed(
            gt_bundle_path,
            grids=grids_stack,
            index=json.dumps(gt_index),
            meta=json.dumps(
                {
                    "format": "reconstruction_demo",
                    "camera": REFERENCE_CAMERA,
                    "id_space": "global (REGISTRY_OFFSET applied)",
                    "grid_hw": [16, 16],
                }
            ),
        )
        logger.info(
            "wrote GT camera token bundle -> %s (%d shots, %d frames each)",
            gt_bundle_path,
            grids_stack.shape[0],
            grids_stack.shape[1],
        )
    else:
        logger.warning("no GT camera grids loaded — gt_camera_tokens.npz NOT written")

    summary["artifact_paths"] = {
        "eval_bundle": str(eval_bundle_path),
        "gt_camera_tokens": str(gt_bundle_path) if gt_grids else None,
    }
    return summary


def _average_skill(per_shot_skill: dict) -> dict:
    """Average per-modality skill across shots (weighted by n_scored).

    Returns ``{modality: {model_error, persistence_error, skill, n_scored,
    n_shots}}`` plus an ``__overall__`` entry — the n_scored-weighted mean of the
    per-modality errors so a modality with more valid tokens dominates honestly,
    and skill recomputed from the pooled errors.
    """
    # pool the (model_err, persist_err) weighted by n_scored, per modality.
    pooled: dict[str, dict] = {}
    for shot_skill in per_shot_skill.values():
        for name, s in shot_skill.items():
            n = int(s["n_scored"])
            if n <= 0:
                continue
            acc = pooled.setdefault(
                name, {"merr": 0.0, "perr": 0.0, "n": 0, "shots": 0}
            )
            acc["merr"] += float(s["model_error"]) * n
            acc["perr"] += float(s["persistence_error"]) * n
            acc["n"] += n
            acc["shots"] += 1
    out: dict[str, dict] = {}
    tot_merr = tot_perr = 0.0
    tot_n = 0
    for name, acc in pooled.items():
        n = acc["n"]
        merr = acc["merr"] / n if n else 0.0
        perr = acc["perr"] / n if n else 0.0
        skill = 1.0 - merr / perr if perr > 0 else 0.0
        out[name] = {
            "model_error": merr,
            "persistence_error": perr,
            "skill": skill,
            "n_scored": int(n),
            "n_shots": int(acc["shots"]),
        }
        tot_merr += acc["merr"]
        tot_perr += acc["perr"]
        tot_n += n
    if tot_n:
        omerr = tot_merr / tot_n
        operr = tot_perr / tot_n
        out["__overall__"] = {
            "model_error": omerr,
            "persistence_error": operr,
            "skill": (1.0 - omerr / operr) if operr > 0 else 0.0,
            "n_scored": int(tot_n),
            "n_shots": len(per_shot_skill),
        }
    return out


# ---------------------------------------------------------------------------
# Phase B — decode the GT camera token grids to images (MAGVIT2 venv)
# ---------------------------------------------------------------------------


def run_phase_b(*, out_dir: Path = OUTPUT_DIR, device: str = "cuda") -> dict:
    """Decode the GT camera token bundle to 256x256 images via the MAGVIT2 venv.

    Re-uses :func:`imas_ambix.camdyn.reconstruction_demo.run_decode_subprocess`
    (the established two-venv decode handoff: it re-invokes ``decode_phase``
    under the MAGVIT2 interpreter, which subtracts REGISTRY_OFFSET and decodes
    the full ``(N,F,16,16)`` stack in one batched pass to ``(N,F,256,256,3)``
    uint8).  If the MAGVIT2 interpreter/weights are absent, records a clear
    "camera decode unavailable" note and returns — never fakes an image.

    Returns ``{"decoded": bool, "image_bundle": path|None, "note": str|None}``.
    """
    from imas_ambix.camdyn.reconstruction_demo import (
        MAGVIT2_PYTHON,
        MAGVIT2_ROOT,
        run_decode_subprocess,
    )

    out_dir = Path(out_dir)
    token_bundle = out_dir / "gt_camera_tokens.npz"
    image_bundle = out_dir / "gt_camera_images.npz"

    if not token_bundle.exists():
        note = f"camera decode unavailable: token bundle {token_bundle} not written"
        logger.warning(note)
        return {"decoded": False, "image_bundle": None, "note": note}
    if not MAGVIT2_PYTHON.exists():
        note = (
            f"camera decode unavailable: Open-MAGVIT2 interpreter not found at "
            f"{MAGVIT2_PYTHON}"
        )
        logger.warning(note)
        return {"decoded": False, "image_bundle": None, "note": note}
    if not MAGVIT2_ROOT.exists():
        note = (
            f"camera decode unavailable: Open-MAGVIT2 root not found at {MAGVIT2_ROOT}"
        )
        logger.warning(note)
        return {"decoded": False, "image_bundle": None, "note": note}

    try:
        run_decode_subprocess(token_bundle, image_bundle, device)
    except Exception as exc:  # noqa: BLE001 — decode failure is recorded, not faked
        note = f"camera decode unavailable: decode subprocess failed: {exc!r}"
        logger.warning(note)
        return {"decoded": False, "image_bundle": None, "note": note}

    if not image_bundle.exists():
        note = "camera decode unavailable: decode subprocess produced no image bundle"
        logger.warning(note)
        return {"decoded": False, "image_bundle": None, "note": note}

    logger.info("decoded GT camera images -> %s", image_bundle)
    return {"decoded": True, "image_bundle": str(image_bundle), "note": None}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def build_artifacts(
    *,
    checkpoint: Path = CHECKPOINT,
    out_dir: Path = OUTPUT_DIR,
    token_root: Path | None = None,
    device: str = "cuda",
    decode_camera: bool = True,
) -> dict:
    """Run phase A (predict + score + bundle) then phase B (GT camera decode).

    Returns the full summary dict (also persisted into ``eval_bundle.npz`` under
    the ``summary`` key, and re-written after phase B with the decode outcome).
    """
    import torch

    out_dir = Path(out_dir)
    try:
        summary = run_phase_a(
            checkpoint=checkpoint, out_dir=out_dir, token_root=token_root
        )
    finally:
        # belt-and-braces release even if phase A raised mid-loop (§2b).
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if decode_camera:
        decode = run_phase_b(out_dir=out_dir, device=device)
    else:
        decode = {
            "decoded": False,
            "image_bundle": None,
            "note": "camera decode skipped (decode_camera=False)",
        }
    summary["camera_decode"] = decode

    # re-write the eval bundle's summary with the decode outcome folded in, so
    # the single bundle is fully self-describing.
    eval_bundle_path = out_dir / "eval_bundle.npz"
    if eval_bundle_path.exists():
        data = np.load(str(eval_bundle_path), allow_pickle=True)
        arrays = {k: data[k] for k in data.files if k not in ("summary",)}
        np.savez_compressed(eval_bundle_path, summary=json.dumps(summary), **arrays)
        logger.info("re-wrote eval bundle summary with decode outcome")
    return summary


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default=str(CHECKPOINT))
    p.add_argument("--out-dir", default=str(OUTPUT_DIR))
    p.add_argument("--token-root", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--no-decode-camera",
        action="store_true",
        help="skip phase B (GT camera decode); only build the token bundles",
    )
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    token_root = Path(args.token_root) if args.token_root else None
    summary = build_artifacts(
        checkpoint=Path(args.checkpoint),
        out_dir=Path(args.out_dir),
        token_root=token_root,
        device=args.device,
        decode_camera=not args.no_decode_camera,
    )

    # human-readable digest to stdout (the full record is in the bundle).
    print("\n=== predict-vs-reality demo artifact summary ===")
    print(f"checkpoint: {summary['checkpoint']} (step {summary['checkpoint_step']})")
    print(f"params: {summary['n_parameters']:,} | ctx_len {summary['context_length']}")
    print(
        "held-out picks (shot, discovery_index, >= "
        f"{summary['train_discovery_limit']}): "
        f"{[(d['shot_id'], d['discovery_index']) for d in summary['heldout_picks']]}"
    )
    pb = summary["param_breakdown"]
    print(f"param breakdown: {pb['groups']}")
    print(
        f"  sum={pb['sum']:,} num_parameters={pb['model_num_parameters']:,} "
        f"match={pb['matches_total']} "
        f"matches_1.034B={pb['matches_logged_1_034_027_033']}"
    )
    print("averaged skill (n_scored-weighted, per modality):")
    for name, s in summary.get("averaged_skill", {}).items():
        print(
            f"  {name:16s} skill={s['skill']:+.4f} "
            f"(model_err={s['model_error']:.4f} "
            f"persist_err={s['persistence_error']:.4f} "
            f"n={s['n_scored']} shots={s['n_shots']})"
        )
    print(f"camera decode: {summary.get('camera_decode')}")
    print(f"artifact paths: {summary.get('artifact_paths')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
