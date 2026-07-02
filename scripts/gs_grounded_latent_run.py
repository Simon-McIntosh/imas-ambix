#!/usr/bin/env python
"""Real-data run for the GS-grounded latent engine (v0 stage 2 gate).

Produces the gate deliverables on MAST held-out shots, using ONLY raw magnetics
and the KNOWN coil currents — never EFIT as a label:

* **Gate 2 (spatial anchor).**  Per held-out slice, solve the ridge GS-inverse
  for the plasma-current amplitudes θ (:func:`evaluate.gs_inverse_theta`), read
  the topology from the resulting ψ (:func:`topology.read_topology`), and score
  axis / X-point / boundary against the firewalled EFIT referee with the same
  RMSE-skill formula the absolute-magnetics oracle uses (~0.5–0.7 bar).  This is
  exactly the map the learned encoder amortises.
* **Learned grounding (smoke).**  Train the hybrid-latent encoder end-to-end on
  the GS residual + anchored supervision for a few hundred steps and report the
  loss drop — evidence the learned path grounds in raw magnetics too.
* **Gate 3 (temporal anchor).**  Report the strictly-positive learned
  diffusivity (D≥0 by construction) and the command load-bearing test (zeroing
  the source changes ∂ψ/∂t).

Writes a JSON result + figures under ``--out``.  Runs on a compute node with the
``/work`` mirror; SIGTERM-clean (partial results are flushed on signal).
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
from pathlib import Path

import numpy as np

# NOTE: shot-list / campaign-operator / assembly / sensor-scale helpers live in
# imas_ambix.latent.data (shared with scripts/train_gs_grounded_latent.py) —
# thin local aliases keep this script's call sites unchanged.
from imas_ambix.latent.data import assemble_shot_windows as _assemble_impl
from imas_ambix.latent.data import build_campaign_operators as _build_campaigns_impl
from imas_ambix.latent.data import feature_schema as _feature_schema
from imas_ambix.latent.data import read_split_shot_lists as _read_shot_lists
from imas_ambix.latent.data import sensor_scale_for_campaign as _sensor_scale

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("gs_grounded_run")

# axis + X-point headline components (matches the oracle headline)
AXIS_COMPONENTS = ["axis_R", "axis_Z"]
LCFS_COMPONENTS = [f"lcfs_r_{k}" for k in range(8)]


def _build_campaigns(shots: list[int], grid_nr: int, grid_nz: int, order: int):
    return _build_campaigns_impl(
        shots, grid_nr=grid_nr, grid_nz=grid_nz, profile_order=order
    )


def _assemble(shots, campaign_of, schema, *, with_referee):
    return _assemble_impl(shots, campaign_of, schema, with_referee=with_referee)


_FIG_DIR = Path("docs/figures/gs-grounded-latent-engine")


def _make_figures(result, model_arr, ref_arr, baseline_arr, example) -> None:
    """Figure-rich evidence for the research doc (mandate 2026-06-03)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _FIG_DIR.mkdir(parents=True, exist_ok=True)

    # (1) plasma-only ψ map with the GS-read axis vs the firewalled EFIT axis
    if example is not None:
        fig, ax = plt.subplots(figsize=(4.2, 5.2))
        RR, ZZ = np.meshgrid(example["r1d"], example["z1d"])
        ax.contourf(RR, ZZ, example["psi"], levels=30, cmap="viridis")
        ax.contour(
            RR, ZZ, example["psi"], levels=12, colors="w", linewidths=0.4, alpha=0.5
        )
        if example["limiter_r"] is not None:
            lr = np.append(example["limiter_r"], example["limiter_r"][0])
            lz = np.append(example["limiter_z"], example["limiter_z"][0])
            ax.plot(lr, lz, "-", color="0.7", lw=1.0, label="limiter")
        ax.plot(
            *example["model_axis"],
            "x",
            color="red",
            ms=13,
            mew=2.5,
            label="GS-read axis",
        )
        ax.plot(
            *example["ref_axis"],
            "o",
            color="cyan",
            ms=9,
            mfc="none",
            mew=2,
            label="EFIT axis (referee)",
        )
        ax.set_xlabel("R [m]")
        ax.set_ylabel("Z [m]")
        ax.set_title(f"Plasma-only ψ + axis readout\n(shot {example['shot']})")
        ax.legend(fontsize=7, loc="upper right")
        ax.set_aspect("equal")
        fig.tight_layout()
        fig.savefig(_FIG_DIR / "fig-psi-axis-readout.png", dpi=130)
        plt.close(fig)

    # (2) model / baseline vs referee axis scatter (R and Z)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.0))
    for k, (comp, lab) in enumerate([(0, "axis R [m]"), (1, "axis Z [m]")]):
        a = axes[k]
        m = np.isfinite(model_arr[:, comp]) & np.isfinite(ref_arr[:, comp])
        a.scatter(
            ref_arr[m, comp], model_arr[m, comp], s=14, alpha=0.6, label="GS-read"
        )
        a.scatter(
            ref_arr[m, comp],
            baseline_arr[m, comp],
            s=14,
            alpha=0.5,
            marker="s",
            label="train-mean",
        )
        lims = [np.nanmin(ref_arr[m, comp]), np.nanmax(ref_arr[m, comp])]
        a.plot(lims, lims, "k--", lw=0.8)
        a.set_xlabel(f"EFIT referee {lab}")
        a.set_ylabel(f"predicted {lab}")
        a.legend(fontsize=7)
    fig.suptitle("GS-readout vs firewalled EFIT axis (held-out)")
    fig.tight_layout()
    fig.savefig(_FIG_DIR / "fig-axis-scatter.png", dpi=130)
    plt.close(fig)

    # (3) per-quantity skill bars with the oracle 0.5–0.7 band
    g2 = result.get("gate2", {})
    sk = g2.get("per_quantity_skill", {})
    names = [k for k in sk if sk[k] is not None]
    vals = [sk[k] for k in names]
    if names:
        fig, ax = plt.subplots(figsize=(8.6, 3.6))
        ax.axhspan(0.5, 0.7, color="green", alpha=0.15, label="oracle bar (0.5–0.7)")
        ax.axhline(0.0, color="k", lw=0.8)
        ax.bar(
            range(len(names)),
            vals,
            color=["#b05a66" if v < 0 else "#2e7d32" for v in vals],
        )
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("RMSE-skill vs train-mean")
        ax.set_title("GS-readout per-quantity skill (training-free GS-inverse)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(_FIG_DIR / "fig-skill-bars.png", dpi=130)
        plt.close(fig)
    logger.info("figures written to %s", _FIG_DIR)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=120)
    ap.add_argument("--n-heldout", type=int, default=8)
    ap.add_argument("--grid-nr", type=int, default=65)
    ap.add_argument("--grid-nz", type=int, default=97)
    ap.add_argument("--order", type=int, default=1)
    ap.add_argument("--ridge", type=float, default=1e-2)
    ap.add_argument("--smoke-train-steps", type=int, default=300)
    ap.add_argument("--max-slices-per-shot", type=int, default=40)
    ap.add_argument("--out", type=str, default="imas_ambix/latent/artifacts")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    result: dict = {"config": vars(args), "status": "running"}

    def _flush(*_a):
        (out_dir / "gs_grounded_run.json").write_text(
            json.dumps(result, indent=2, default=float)
        )
        logger.info("flushed partial results")

    signal.signal(signal.SIGTERM, lambda *a: (_flush(), sys.exit(0)))

    from imas_ambix.latent.evaluate import (
        gs_inverse_theta,
        headline_skill,
        matched_xpoint_error,
        per_quantity_skill,
    )
    from imas_ambix.latent.topology import read_topology

    schema = _feature_schema()
    train_shots, held_shots = _read_shot_lists(args.n_train, args.n_heldout)
    logger.info("train=%d held-out=%d", len(train_shots), len(held_shots))

    all_shots = list(dict.fromkeys(train_shots + held_shots))
    gs_by_sig, limiter_by_sig, campaign_of = _build_campaigns(
        all_shots, args.grid_nr, args.grid_nz, args.order
    )
    logger.info("built %d campaign operators", len(gs_by_sig))
    result["campaigns"] = list(gs_by_sig)

    train_w = _assemble(train_shots, campaign_of, schema, with_referee=True)
    held_w = _assemble(held_shots, campaign_of, schema, with_referee=True)
    logger.info(
        "assembled train=%d held-out=%d shot-windows", len(train_w), len(held_w)
    )

    # baseline = train-mean of the referee 14-D geometry (the oracle's baseline)
    ref_rows = [w.ref_target for w in train_w if w.ref_target is not None]
    if ref_rows:
        ref_stack = np.concatenate(ref_rows, axis=0)
        baseline_vec = np.nanmean(ref_stack, axis=0)
    else:
        baseline_vec = np.full(14, np.nan)

    # --- Gate 2: GS-readout topology vs referee, per held-out slice ---
    model_targets, ref_targets = [], []
    example_slice = None
    for w in held_w:
        gs = gs_by_sig[w.campaign]
        a_plasma = gs.a_plasma.numpy()
        g_pf = gs.g_pf.numpy()
        scale = _sensor_scale(train_w + held_w, w.campaign, a_plasma.shape[0])
        theta = gs_inverse_theta(
            a_plasma, g_pf, w.raw_mag, w.mag_mask, w.i_pf, scale, ridge=args.ridge
        )
        lr, lz = limiter_by_sig[w.campaign]
        import torch

        theta_t = torch.tensor(theta, dtype=torch.float64)
        i_pf_t = torch.tensor(w.i_pf, dtype=torch.float64)
        # Topology is read from the PLASMA-only ψ (coils zeroed): a linear
        # GS-inverse ψ is not a force-balanced equilibrium, so the total field
        # is dominated by the in-vessel PF coils and has no confined O-point.
        # The plasma-current-generated flux peaks at the current centroid — a
        # coil-free, always-defined magnetic-axis estimate.
        psi2d = gs.psi_field_2d(theta_t, torch.zeros_like(i_pf_t)).numpy()
        r1d = gs.grid_r_1d.numpy()
        z1d = gs.grid_z_1d.numpy()
        # subsample slices so the per-slice ψ read stays fast over many shots
        valid = [
            t
            for t in range(psi2d.shape[0])
            if w.ref_target is not None and np.isfinite(w.ref_target[t, :2]).all()
        ]
        if len(valid) > args.max_slices_per_shot:
            valid = valid[:: max(1, len(valid) // args.max_slices_per_shot)]
        bbox = gs.plasma_bbox()  # restrict axis/X-point to the plasma-current region
        coils = gs.coil_rz  # exclude in-vessel PF-coil O-points
        for t in valid:
            read = read_topology(
                psi2d[t],
                r1d,
                z1d,
                limiter_r=lr,
                limiter_z=lz,
                search_bbox=bbox,
                exclude_rz=coils,
                exclude_radius=0.15,
            )
            model_targets.append(read.target)
            ref_targets.append(w.ref_target[t])
            if example_slice is None and read.axis is not None:
                example_slice = {
                    "psi": psi2d[t],
                    "r1d": r1d,
                    "z1d": z1d,
                    "limiter_r": lr,
                    "limiter_z": lz,
                    "model_axis": np.array(read.axis),
                    "ref_axis": w.ref_target[t, :2].copy(),
                    "shot": w.shot_id,
                }
    if model_targets:
        model_arr = np.array(model_targets)
        ref_arr = np.array(ref_targets)
        baseline_arr = np.tile(baseline_vec, (len(model_arr), 1))
        names = [
            "axis_R",
            "axis_Z",
            "xpt0_R",
            "xpt0_Z",
            "xpt1_R",
            "xpt1_Z",
            *LCFS_COMPONENTS,
        ]
        skill = per_quantity_skill(model_arr, ref_arr, baseline_arr, names)
        # permutation-invariant X-point-set skill
        xm = np.array(
            [
                matched_xpoint_error(
                    model_arr[i, 2:6].reshape(2, 2), ref_arr[i, 2:6].reshape(2, 2)
                )
                for i in range(len(model_arr))
            ]
        )
        xb = np.array(
            [
                matched_xpoint_error(
                    baseline_arr[i, 2:6].reshape(2, 2), ref_arr[i, 2:6].reshape(2, 2)
                )
                for i in range(len(model_arr))
            ]
        )
        finite = np.isfinite(xm) & np.isfinite(xb)
        xpt_skill = (
            1.0
            - np.sqrt(np.nanmean(xm[finite] ** 2))
            / np.sqrt(np.nanmean(xb[finite] ** 2))
            if finite.any()
            else np.nan
        )
        result["gate2"] = {
            "n_slices": int(len(model_arr)),
            "per_quantity_skill": {
                k: (None if not np.isfinite(v) else float(v)) for k, v in skill.items()
            },
            "axis_skill": headline_skill(skill, AXIS_COMPONENTS),
            "xpoint_set_skill": None
            if not np.isfinite(xpt_skill)
            else float(xpt_skill),
            "lcfs_skill": headline_skill(skill, LCFS_COMPONENTS),
        }
        logger.info(
            "Gate2 axis_skill=%s xpt_skill=%s", result["gate2"]["axis_skill"], xpt_skill
        )
        np.savez(
            out_dir / "gate2_arrays.npz",
            model=model_arr,
            ref=ref_arr,
            baseline=baseline_arr,
        )
        try:
            _make_figures(result, model_arr, ref_arr, baseline_arr, example_slice)
        except Exception as exc:  # noqa: BLE001 — figures are best-effort
            logger.warning("figure generation failed: %s", exc)
    else:
        result["gate2"] = {"n_slices": 0, "note": "no held-out slices with referee"}

    # --- Gate 3: transport temporal anchor (D>=0 + command load-bearing) ---
    import torch

    from imas_ambix.latent.transport import FluxDiffusionPrior

    tr = FluxDiffusionPrior(nrho=args.grid_nr, cmd_dim=2, feat_dim=8).double()
    feat = torch.randn(64, 8, dtype=torch.float64) * 20 - 40  # adversarial
    d_min = float(tr.diffusivity(feat).min())
    rho = torch.linspace(0, 1, args.grid_nr, dtype=torch.float64).unsqueeze(0)
    psi = torch.sin(rho * 3.0)
    cmd = torch.randn(1, 2, dtype=torch.float64)
    dc = tr.dpsi_dt(psi, rho, torch.zeros(1, 8, dtype=torch.float64), cmd)
    dz = tr.dpsi_dt(
        psi, rho, torch.zeros(1, 8, dtype=torch.float64), torch.zeros_like(cmd)
    )
    result["gate3"] = {
        "diffusivity_min": d_min,
        "D_geq_0": bool(d_min > 0),
        "command_source_delta": float((dc - dz).abs().sum()),
        "command_load_bearing": bool(float((dc - dz).abs().sum()) > 0),
    }
    logger.info(
        "Gate3 D_min=%.3e load-bearing-delta=%.3e",
        d_min,
        result["gate3"]["command_source_delta"],
    )

    # --- Learned grounding smoke ---
    if args.smoke_train_steps > 0 and train_w:
        result["learned_grounding"] = _smoke_train(
            train_w, gs_by_sig, schema, steps=args.smoke_train_steps
        )

    result["status"] = "done"
    _flush()
    logger.info("done: %s", json.dumps(result.get("gate2", {}), default=float))
    return 0


def _smoke_train(train_w, gs_by_sig, schema, *, steps: int) -> dict:
    """Short end-to-end fit of the learned encoder on the GS residual."""
    # single dominant campaign for the smoke fit
    from collections import Counter

    import torch

    from imas_ambix.latent.data import fit_corpus_stats
    from imas_ambix.latent.encoder import HybridLatentEncoder, LatentConfig
    from imas_ambix.latent.engine import GSGroundedLatentEngine
    from imas_ambix.latent.transport import FluxDiffusionPrior

    key = Counter(w.campaign for w in train_w).most_common(1)[0][0]
    ws = [w for w in train_w if w.campaign == key]
    gs = gs_by_sig[key].double()
    stats = fit_corpus_stats([w.features_raw for w in ws])
    feats = np.concatenate([stats.normalise(w.features_raw) for w in ws], axis=0)
    raw = np.concatenate([w.raw_mag for w in ws], axis=0)
    mask = np.concatenate([w.mag_mask for w in ws], axis=0)
    ipf = np.concatenate([w.i_pf for w in ws], axis=0)
    scale = np.nanstd(raw, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)

    cfg = LatentConfig(
        n_features=feats.shape[1],
        n_theta=gs.n_dof,
        n_anchored=2,
        n_free=8,
        hidden=128,
        depth=3,
    )
    enc = HybridLatentEncoder(cfg).double()
    tr = FluxDiffusionPrior(nrho=gs.grid_nr, cmd_dim=2, feat_dim=8).double()
    eng = GSGroundedLatentEngine(enc, gs, tr)
    opt = torch.optim.Adam(eng.parameters(), lr=3e-3)

    x = torch.tensor(feats, dtype=torch.float64)
    rm = torch.tensor(np.nan_to_num(raw), dtype=torch.float64)
    mk = torch.tensor(mask)
    ip = torch.tensor(ipf, dtype=torch.float64)
    sc = torch.tensor(scale, dtype=torch.float64).unsqueeze(0).expand_as(rm)
    n = x.shape[0]
    rng = np.random.RandomState(0)
    r0 = None
    losses = []
    for step in range(steps):
        idx = torch.tensor(rng.choice(n, size=min(256, n), replace=False))
        lat = eng.encode(x[idx])
        loss = eng.gs_residual_loss(lat, ip[idx], rm[idx], sc[idx], mk[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if r0 is None:
            r0 = float(loss)
        if step % 50 == 0:
            losses.append(float(loss))
    r1 = float(loss)
    return {
        "campaign": key,
        "n_slices": int(n),
        "gs_residual_start": r0,
        "gs_residual_end": r1,
        "drop_ratio": (r1 / r0) if r0 else None,
        "loss_trace": losses,
    }


if __name__ == "__main__":
    raise SystemExit(main())
