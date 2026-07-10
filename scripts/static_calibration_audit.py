#!/usr/bin/env python
"""Static per-channel calibration audit — is the halo a coil/instrument error?

The free patch inverse carries a ~0.75·Ip anti-parallel halo that is
PHASE-INDEPENDENT (flat-top = rampup), so the sensor signal it absorbs is a
static error, not vessel eddy current.  This script fits the simplest static
instrument model — one affine nuisance (gain, offset) per magnetics channel —
and freezes it for a held-out test:

fit (``--stage fit``, train shots only)
    Invert train slices with the physics-prior arm (unidirectional softplus +
    support consistency at the frozen tune winner) so the plasma current is
    physically constrained and the per-channel residual isolates the static
    error.  Per channel, robust-fit measured ≈ gain·predicted + offset over
    all (slice, shot) pairs; bootstrap over shots for CIs.  Writes the FROZEN
    calibration JSON + a per-channel figure.

apply (held-out)
    ``scripts/patch_gate_eval.py --calibration <json>`` corrects the raw
    payloads (measured' = (measured − offset)/gain, scale' = scale/|gain|).
    The honest metric is the held-out FREE-arm halo: if the calibration is
    real, the anti-parallel fraction drops without any prior active.

Firewall: raw magnetics + our own forward model only — no EFIT anywhere.
Leakage: fitted on train shots, frozen, applied held-out.

Artifacts: imas_ambix/latent/artifacts/patch_gate/static_calibration.json
Figure:    docs/figures/plasma-current-priors-hardening/fig-static-calibration.png
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from imas_ambix.latent.data import read_split_shot_lists
from imas_ambix.latent.patch_inverse import InverseConfig, invert_slices
from scripts.patch_gate_eval import shot_payloads

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("static_calibration_audit")

ARTIFACTS = Path("imas_ambix/latent/artifacts/patch_gate")
FIGURES = Path("docs/figures/plasma-current-priors-hardening")


def _prior_config(grid) -> InverseConfig:
    """The frozen tune-winner physics-prior arm (constrained plasma current)."""
    return InverseConfig(
        policy="discrepancy",
        lambda_fb=3.0,
        misfit_ratio=1.5,
        lambda_max=100.0,
        iters=800,
        connectivity="locality",
        sign_prior="softplus",
        support_prior=True,
        support_weight=2000.0,
        halo_budget=0.03,
        limiter_r=np.asarray(grid.limiter_r, dtype=np.float64),
        limiter_z=np.asarray(grid.limiter_z, dtype=np.float64),
    )


def _robust_affine(pred: np.ndarray, meas: np.ndarray) -> tuple[float, float]:
    """Least-squares gain/offset after one 3-sigma residual clip."""
    a = np.polyfit(pred, meas, 1)
    res = meas - np.polyval(a, pred)
    keep = np.abs(res - np.median(res)) <= 3.0 * (np.std(res) + 1e-30)
    if keep.sum() >= 8:
        a = np.polyfit(pred[keep], meas[keep], 1)
    return float(a[0]), float(a[1])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-fit-shots", type=int, default=12)
    ap.add_argument("--n-baseline-shots", type=int, default=10)
    ap.add_argument("--device", type=str, default="auto")
    args = ap.parse_args()
    device = (
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else args.device
    )
    FIGURES.mkdir(parents=True, exist_ok=True)

    train_shots, _held = read_split_shot_lists(40, 8)
    # skip the baseline-vector shots so the nuisance fit and the skill
    # baseline never share slices
    fit_shots = train_shots[
        args.n_baseline_shots : args.n_baseline_shots + args.n_fit_shots
    ]
    logger.info("device=%s  fit shots: %s", device, fit_shots)

    per_shot: list[dict] = []  # per shot: channels x (pred, meas) row stacks
    channels: list[str] | None = None
    scale_med: np.ndarray | None = None

    for shot in fit_shots:
        try:
            payload = shot_payloads(
                shot, nr=65, nz=97, max_slices=20, min_ip_ka=300.0, split="train"
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("shot %s failed to load: %s", shot, exc)
            continue
        if payload is None:
            continue
        grid, basis, payloads = payload["grid"], payload["basis"], payload["payloads"]
        if channels is None:
            channels = list(basis.sensor_channels)
            scale_med = np.median(np.stack([p.scale for p in payloads]), axis=0)
        inv = invert_slices(basis, payloads, _prior_config(grid), device=device)
        m_sens = basis.m_sens.detach().cpu().numpy().astype(np.float64)
        preds, meass, masks = [], [], []
        for r, p in zip(inv, payloads, strict=True):
            preds.append(p.vacuum + m_sens @ r.i_cell)
            meass.append(np.nan_to_num(p.measured))
            masks.append(p.mask)
        per_shot.append(
            {
                "shot": int(shot),
                "pred": np.stack(preds),
                "meas": np.stack(meass),
                "mask": np.stack(masks),
            }
        )
        logger.info("shot %s: %d slices inverted", shot, len(payloads))

    assert channels is not None and per_shot, "no fit shots loaded"
    n_ch = len(channels)

    def fit_over(shot_rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
        gain = np.ones(n_ch)
        offset = np.zeros(n_ch)
        for s in range(n_ch):
            pred = np.concatenate([d["pred"][d["mask"][:, s], s] for d in shot_rows])
            meas = np.concatenate([d["meas"][d["mask"][:, s], s] for d in shot_rows])
            if pred.size < 12 or np.ptp(pred) < 1e-12:
                continue  # unmeasured / degenerate channel: identity
            gain[s], offset[s] = _robust_affine(pred, meas)
        return gain, offset

    gain, offset = fit_over(per_shot)

    rng = np.random.default_rng(0)
    boots_g, boots_o = [], []
    for _ in range(200):
        rows = [per_shot[i] for i in rng.integers(0, len(per_shot), len(per_shot))]
        g, o = fit_over(rows)
        boots_g.append(g)
        boots_o.append(o)
    g_lo, g_hi = np.percentile(boots_g, [2.5, 97.5], axis=0)
    o_lo, o_hi = np.percentile(boots_o, [2.5, 97.5], axis=0)

    off_sigma = offset / (scale_med + 1e-30)
    out = {
        "fit_shots": [int(s) for s in fit_shots],
        "n_fit_shots": len(per_shot),
        "arm": "sign-softplus+support-hb0.03-sw2000 (frozen tune winner)",
        "channels": channels,
        "gain": gain.tolist(),
        "gain_ci_lo": g_lo.tolist(),
        "gain_ci_hi": g_hi.tolist(),
        "offset": offset.tolist(),
        "offset_ci_lo": o_lo.tolist(),
        "offset_ci_hi": o_hi.tolist(),
        "offset_over_sigma": off_sigma.tolist(),
        "gain_median": float(np.median(gain)),
        "gain_p10_p90": [
            float(np.percentile(gain, 10)),
            float(np.percentile(gain, 90)),
        ],
        "offset_over_sigma_median_abs": float(np.median(np.abs(off_sigma))),
    }
    (ARTIFACTS / "static_calibration.json").write_text(json.dumps(out, indent=2))
    logger.info(
        "calibration frozen: gain median %.3f [p10 %.3f, p90 %.3f], "
        "|offset|/sigma median %.2f",
        out["gain_median"],
        *out["gain_p10_p90"],
        out["offset_over_sigma_median_abs"],
    )

    x = np.arange(n_ch)
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    axes[0].errorbar(
        x,
        gain - 1.0,
        yerr=[gain - g_lo, g_hi - gain],
        fmt="o",
        ms=3,
        color="#1565c0",
        ecolor="#9bb8d9",
        elinewidth=0.8,
    )
    axes[0].axhline(0, color="k", lw=0.8)
    axes[0].set_ylabel("gain − 1")
    axes[0].set_title(
        "Static per-channel calibration fitted against the physics-prior arm "
        f"({len(per_shot)} train shots) — 95% CI over shots"
    )
    axes[1].errorbar(
        x,
        off_sigma,
        yerr=[
            (offset - o_lo) / (scale_med + 1e-30),
            (o_hi - offset) / (scale_med + 1e-30),
        ],
        fmt="o",
        ms=3,
        color="#8a3324",
        ecolor="#d3a493",
        elinewidth=0.8,
    )
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set_ylabel("offset / channel σ")
    axes[1].set_xlabel("channel index")
    step = max(1, n_ch // 40)
    axes[1].set_xticks(x[::step])
    axes[1].set_xticklabels(
        [c.split("/")[-1] for c in channels[::step]], rotation=90, fontsize=6
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "fig-static-calibration.png", dpi=140)
    logger.info("wrote %s", FIGURES / "fig-static-calibration.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
