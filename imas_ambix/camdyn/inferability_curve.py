"""Held-out partial-view reconstruction quality versus visible clip area.

The evaluator loads the matched temporal and per-frame checkpoints once,
materialises one deterministic held-out window sample, and reuses that exact
sample at every clip size.  Each window gets a compact, static, axis-aligned
clip at a seeded random position.  The two learned arms and the causal
carry-forward reference therefore see identical tokens and masks.

Confidence intervals resample held-out shots, the independent experimental
unit, after aggregating all sampled windows and masked cells within each shot.
The carry-forward likelihood is an explicit factorised-bit distribution: an
unseen cell has 0.5 probability per bit, while a previously observed token is
carried with a fixed error floor.  This makes its NLL defined without granting
the reference access to hidden full-frame truth.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

from imas_ambix.camdyn.arm_compare import _carry_forward_pred, _load_arm
from imas_ambix.camdyn.dataset import FrameWindowConfig
from imas_ambix.camdyn.metrics import bootstrap_ci
from imas_ambix.camdyn.model import GRID_H, GRID_W, LFQ_BITS
from imas_ambix.camdyn.train import (
    TrainConfig,
    Trainer,
    _normalise_conditioning,
    _specs_for_shots,
    score_window_bits,
)

logger = logging.getLogger(__name__)

DEFAULT_VISIBLE_FRACTIONS = (0.05, 0.10, 0.20, 0.35, 0.50)
DEFAULT_BASELINE = Path(
    "/work/projects/imas_gpu/mast-checkpoints/camdyn/cap_v1_baseline/final.pt"
)
DEFAULT_DYNAMICS = Path(
    "/work/projects/imas_gpu/mast-checkpoints/camdyn/cap_v1_dynamics/final.pt"
)
DEFAULT_ARTIFACT = Path(__file__).resolve().parent / "artifacts/inferability_curve.json"
DEFAULT_FIGURE = (
    Path(__file__).resolve().parents[2]
    / "docs/figures/camera-dynamics-wm/fig-cdw-inferability-curve.png"
)


def compact_clip_shape(
    fraction: float, grid: tuple[int, int] = (GRID_H, GRID_W)
) -> tuple[int, int]:
    """Return the near-square integer rectangle closest to ``fraction``.

    The token grid is discrete, so requested and realised fractions can differ.
    Area error is primary; a mild aspect penalty breaks near-ties away from
    implausibly thin strips.
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError("visible fraction must lie strictly between zero and one")
    gh, gw = grid
    candidates = []
    for height in range(1, gh + 1):
        for width in range(1, gw + 1):
            realised = height * width / float(gh * gw)
            aspect_penalty = 0.03 * abs(np.log(height / width))
            candidates.append(
                (
                    abs(realised - fraction) + aspect_penalty,
                    abs(realised - fraction),
                    height,
                    width,
                )
            )
    _, _, height, width = min(candidates)
    return height, width


def clip_masks_for_batch(
    shot_ids: np.ndarray,
    n_frames: int,
    fraction: float,
    *,
    seed: int,
    sample_offset: int,
    grid: tuple[int, int] = (GRID_H, GRID_W),
) -> np.ndarray:
    """Build reproducible randomly positioned static clips for one batch."""
    gh, gw = grid
    height, width = compact_clip_shape(fraction, grid)
    masks = np.zeros((len(shot_ids), n_frames, gh, gw), dtype=bool)
    for index, shot_id in enumerate(np.asarray(shot_ids, dtype=np.int64)):
        rng = np.random.default_rng(
            np.random.SeedSequence([seed, sample_offset + index, int(shot_id)])
        )
        centre_row, centre_col = rng.random(2)
        row0 = int(round(centre_row * max(0, gh - height)))
        col0 = int(round(centre_col * max(0, gw - width)))
        masks[index, :, row0 : row0 + height, col0 : col0 + width] = True
    return masks


def mean_bootstrap_ci(
    values: np.ndarray, *, n_boot: int = 10_000, seed: int = 0
) -> dict:
    """Percentile CI for an equal-shot mean."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return {"mean": 0.0, "lo": 0.0, "hi": 0.0, "n_shots": 0}
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float64)
    chunk = 1_000
    for start in range(0, n_boot, chunk):
        stop = min(n_boot, start + chunk)
        indices = rng.integers(0, values.size, size=(stop - start, values.size))
        means[start:stop] = values[indices].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "lo": float(np.quantile(means, 0.025)),
        "hi": float(np.quantile(means, 0.975)),
        "std_between_shots": float(values.std()),
        "n_shots": int(values.size),
    }


def carry_forward_scores(
    tokens: np.ndarray,
    visible: np.ndarray,
    loss_mask: np.ndarray,
    valid: np.ndarray,
    *,
    error_floor: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Return factorised-bit NLL and exact-token top-1 for carry-forward.

    Cells without a causal observation use a uniform distribution over token
    bits.  Seen cells carry the latest observed token with ``error_floor`` per
    bit, keeping the deterministic reference's likelihood finite and explicit.
    """
    tokens = np.asarray(tokens, dtype=np.int64)
    visible = np.asarray(visible, dtype=bool)
    scored = (
        np.asarray(loss_mask, dtype=bool) & np.asarray(valid, dtype=bool)[:, None, None]
    )
    pred = _carry_forward_pred(tokens, visible)
    pred_seen = pred >= 0
    top1 = (pred == tokens).astype(np.float64)[scored]

    shifts = np.arange(LFQ_BITS, dtype=np.int64)
    true_bits = ((tokens[..., None] >> shifts) & 1).astype(bool)
    safe_pred = np.maximum(pred, 0)
    pred_bits = ((safe_pred[..., None] >> shifts) & 1).astype(bool)
    bit_match = true_bits == pred_bits
    seen_nll = np.where(bit_match, -np.log1p(-error_floor), -np.log(error_floor)).sum(
        axis=-1
    )
    unseen_nll = float(LFQ_BITS * np.log(2.0))
    nll = np.where(pred_seen, seen_nll, unseen_nll)[scored]
    return nll.astype(np.float64), top1


def _add_scores(store, shot_id: int, nll: np.ndarray, top1: np.ndarray) -> None:
    rec = store[int(shot_id)]
    rec["nll_sum"] += float(np.asarray(nll).sum())
    rec["top1_sum"] += float(np.asarray(top1).sum())
    rec["n"] += int(np.asarray(nll).size)


def _score_model(model, batches, masks, trainer, torch, device):
    by_shot = defaultdict(lambda: {"nll_sum": 0.0, "top1_sum": 0.0, "n": 0})
    with torch.no_grad():
        for arr, visible in zip(batches, masks, strict=True):
            normalised = _normalise_conditioning(
                arr["cond_values"], trainer._cond_stats
            )
            t_tokens = torch.from_numpy(np.ascontiguousarray(arr["tokens"])).to(device)
            t_visible = torch.from_numpy(np.ascontiguousarray(visible)).to(device)
            t_values = torch.from_numpy(normalised.astype(np.float32)).to(device)
            t_missing = torch.from_numpy(arr["cond_missing"].astype(np.float32)).to(
                device
            )
            t_dt = torch.from_numpy(arr["dt"].astype(np.float32)).to(device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=(device.type == "cuda"),
            ):
                logits = model.module(t_tokens, t_visible, t_values, t_missing, t_dt)
            logits_np = logits.float().cpu().numpy()
            for index, shot_id in enumerate(arr["shot_id"]):
                scored = (~visible[index]) & arr["valid"][index, :, None, None]
                score = score_window_bits(
                    logits_np[index], arr["tokens"][index], scored
                )
                if score.n:
                    _add_scores(
                        by_shot, int(shot_id), score.nll_per_token, score.acc_per_token
                    )
    return by_shot


def _score_carry_forward_by_shot(batches, masks):
    by_shot = defaultdict(lambda: {"nll_sum": 0.0, "top1_sum": 0.0, "n": 0})
    for arr, visible in zip(batches, masks, strict=True):
        for index, shot_id in enumerate(arr["shot_id"]):
            loss_mask = ~visible[index]
            nll, top1 = carry_forward_scores(
                arr["tokens"][index],
                visible[index],
                loss_mask,
                arr["valid"][index],
            )
            if nll.size:
                _add_scores(by_shot, int(shot_id), nll, top1)
    return by_shot


def _summarise(by_shot, *, n_boot: int, seed: int) -> dict:
    shot_ids = sorted(shot_id for shot_id, row in by_shot.items() if row["n"])
    nll = np.asarray([by_shot[s]["nll_sum"] / by_shot[s]["n"] for s in shot_ids])
    top1 = np.asarray([by_shot[s]["top1_sum"] / by_shot[s]["n"] for s in shot_ids])
    return {
        "masked_nll": mean_bootstrap_ci(nll, n_boot=n_boot, seed=seed),
        "masked_top1": mean_bootstrap_ci(top1, n_boot=n_boot, seed=seed + 1),
        "n_masked_tokens": int(sum(by_shot[s]["n"] for s in shot_ids)),
        "shot_values": {
            str(s): {"masked_nll": float(vn), "masked_top1": float(va)}
            for s, vn, va in zip(shot_ids, nll, top1, strict=True)
        },
    }


def _paired_summary(reference: dict, dynamics: dict, *, n_boot: int) -> dict:
    ref = reference["shot_values"]
    dyn = dynamics["shot_values"]
    common = sorted(set(ref) & set(dyn), key=int)
    ref_nll = np.asarray([ref[s]["masked_nll"] for s in common])
    dyn_nll = np.asarray([dyn[s]["masked_nll"] for s in common])
    ref_acc = np.asarray([ref[s]["masked_top1"] for s in common])
    dyn_acc = np.asarray([dyn[s]["masked_top1"] for s in common])
    nll = bootstrap_ci(ref_nll - dyn_nll, n_boot=n_boot)
    acc = bootstrap_ci(dyn_acc - ref_acc, n_boot=n_boot)
    nll["n_shots"] = len(common)
    acc["n_shots"] = len(common)
    return {"masked_nll": nll, "masked_top1": acc}


def _public_summary(summary: dict) -> dict:
    return {key: value for key, value in summary.items() if key != "shot_values"}


def evaluate_inferability(
    baseline_ckpt,
    dynamics_ckpt,
    *,
    fractions=DEFAULT_VISIBLE_FRACTIONS,
    split_path=None,
    device="cuda",
    eval_seed=24680,
    max_windows=256,
    n_boot=10_000,
) -> dict:
    """Evaluate both trained arms and carry-forward on paired held-out clips."""
    import torch  # noqa: PLC0415

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    baseline, baseline_cfg, baseline_stats = _load_arm(baseline_ckpt, torch, dev)
    dynamics, _dynamics_cfg, _dynamics_stats = _load_arm(dynamics_ckpt, torch, dev)
    cfg = TrainConfig.from_dict(baseline_cfg)
    if split_path is not None:
        cfg.split_path = str(split_path)
    cfg.device = str(dev)
    cfg.num_workers = 0
    cfg.eval_windows = int(max_windows)
    trainer = Trainer(cfg)
    trainer._cond_stats = baseline_stats
    split = trainer._load_split()
    specs = _specs_for_shots(split.held_out, max_shots=cfg.max_heldout_shots)
    frame_cfg = FrameWindowConfig(
        n_frames=cfg.n_frames, stride=cfg.stride, seed=cfg.seed
    )
    logger.info("materialising at most %d held-out windows", max_windows)
    batches = trainer._materialize_eval(
        specs, frame_cfg, max_windows=max_windows, seed=eval_seed
    )
    if not batches:
        raise RuntimeError("held-out materialisation produced no windows")

    rows = []
    for fraction_index, requested in enumerate(fractions):
        height, width = compact_clip_shape(float(requested))
        masks = []
        sample_offset = 0
        for arr in batches:
            masks.append(
                clip_masks_for_batch(
                    arr["shot_id"],
                    arr["tokens"].shape[1],
                    float(requested),
                    seed=eval_seed,
                    sample_offset=sample_offset,
                )
            )
            sample_offset += arr["tokens"].shape[0]
        logger.info(
            "scoring requested fraction %.3f (realised %.4f; %dx%d)",
            requested,
            height * width / float(GRID_H * GRID_W),
            height,
            width,
        )
        base = _summarise(
            _score_model(baseline, batches, masks, trainer, torch, dev),
            n_boot=n_boot,
            seed=100 + fraction_index,
        )
        dyn = _summarise(
            _score_model(dynamics, batches, masks, trainer, torch, dev),
            n_boot=n_boot,
            seed=200 + fraction_index,
        )
        carry = _summarise(
            _score_carry_forward_by_shot(batches, masks),
            n_boot=n_boot,
            seed=300 + fraction_index,
        )
        rows.append(
            {
                "requested_visible_fraction": float(requested),
                "realised_visible_fraction": height * width / float(GRID_H * GRID_W),
                "clip_shape_tokens": [height, width],
                "dynamics": _public_summary(dyn),
                "per_frame_baseline": _public_summary(base),
                "carry_forward": _public_summary(carry),
                "dynamics_vs_per_frame_baseline": _paired_summary(
                    base, dyn, n_boot=n_boot
                ),
                "dynamics_vs_carry_forward": _paired_summary(carry, dyn, n_boot=n_boot),
            }
        )

    n_windows = int(sum(arr["tokens"].shape[0] for arr in batches))
    n_shots = len({int(s) for arr in batches for s in arr["shot_id"]})
    return {
        "evaluation": "held-out partial-to-full reconstruction inferability curve",
        "mask_protocol": (
            "seeded random-position static compact rectangular clips; "
            "paired across predictors"
        ),
        "bootstrap": {
            "unit": "held-out shot",
            "confidence": 0.95,
            "resamples": int(n_boot),
            "equal_shot_weight": True,
        },
        "carry_forward_likelihood": {
            "tokenisation": f"{LFQ_BITS} independent bits",
            "unseen_cell_bit_probability": 0.5,
            "seen_cell_error_floor": 1e-6,
            "causal": True,
        },
        "baseline_ckpt": str(baseline_ckpt),
        "dynamics_ckpt": str(dynamics_ckpt),
        "baseline_params": int(baseline.num_parameters()),
        "dynamics_params": int(dynamics.num_parameters()),
        "n_materialised_windows": n_windows,
        "n_heldout_shots_sampled": n_shots,
        "fractions": rows,
    }


def plot_curve(artifact: dict, path: Path) -> None:
    """Render the two decision metrics with direct uncertainty bands."""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    rows = artifact["fractions"]
    x = 100.0 * np.asarray([row["realised_visible_fraction"] for row in rows])
    labels = (
        ("dynamics", "Dynamics", "#2369a8"),
        ("per_frame_baseline", "Per-frame", "#c44e52"),
        ("carry_forward", "Carry-forward", "#555555"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1), constrained_layout=True)
    for key, label, colour in labels:
        for axis, metric in zip(axes, ("masked_nll", "masked_top1"), strict=True):
            means = np.asarray([row[key][metric]["mean"] for row in rows])
            lo = np.asarray([row[key][metric]["lo"] for row in rows])
            hi = np.asarray([row[key][metric]["hi"] for row in rows])
            axis.plot(x, means, marker="o", color=colour, label=label)
            axis.fill_between(x, lo, hi, color=colour, alpha=0.16, linewidth=0)
    axes[0].set_ylabel("Masked-token NLL (lower is better)")
    axes[1].set_ylabel("Masked-token top-1 (higher is better)")
    for axis in axes:
        axis.set_xlabel("Visible clip area (%)")
        axis.grid(alpha=0.22, linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False)
    fig.suptitle("Held-out partial-to-full reconstruction")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--dynamics", type=Path, default=DEFAULT_DYNAMICS)
    parser.add_argument("--out", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--split-path", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-windows", type=int, default=256)
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--eval-seed", type=int, default=24680)
    parser.add_argument(
        "--fractions", type=float, nargs="+", default=DEFAULT_VISIBLE_FRACTIONS
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    if args.smoke:
        args.max_windows = min(args.max_windows, 4)
        args.n_boot = min(args.n_boot, 200)
    artifact = evaluate_inferability(
        args.baseline,
        args.dynamics,
        fractions=args.fractions,
        split_path=args.split_path,
        device=args.device,
        eval_seed=args.eval_seed,
        max_windows=args.max_windows,
        n_boot=args.n_boot,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    plot_curve(artifact, args.figure)
    logger.info("wrote %s and %s", args.out, args.figure)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
