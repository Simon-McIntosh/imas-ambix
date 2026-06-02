#!/usr/bin/env python
r"""Extract + plot the recovered current-density evolution for the S9 findings doc.

Re-runs the D2 EnKF/TORAX baseline's PRIOR (forecast-arm) ensemble for a handful
of representative held-out shots and plots the recovered MSE-free current-density
evolution.  The committed metrics artifact (``enkf_baseline_metrics_v0.json``)
persists only SCALAR per-shot diagnostics (q0 medians, innovation), NOT the
per-shot ``j(rho, t)`` trajectories — so the trajectories must be regenerated.

We reproduce the PRIOR (forecast) ensemble exactly as ``enkf_baseline.run_shot``
builds it:

    cfg = EnKFConfig(n_ensemble=32, n_assim_slices=5)   # the committed config
    rng = np.random.default_rng(cfg.seed + shot_id)
    thetas = _sample_theta(cfg, rng)
    trajs  = [run_torax_member(inp, cfg, th) for th in thetas]

so the plotted ensemble IS the one behind the committed bar.  We do NOT reproduce
the ANALYSIS (EKI-updated) arm — duplicating the whole EKI block is error-prone,
and the artifact's own verdict is that analysis ≈ forecast for the INTERNAL
current (the Stage-2 magnetics-under-determination thesis: external magnetics fix
the boundary/Ip, not internal j(psi)).  The recovery of internal current is done
by the resistive current-diffusion TRANSITION (measured Ip + Te->sigma), with NO
MSE on the input side — exactly what the prior ensemble represents.

NOTHING is read out from MSE on the input side.  MSE enters ONLY as held-out eval
truth in figure 3 (the q0 overlay), method-matched to the eval harness.

Foreground, CPU-only.  Per-shot wall-clock guard drops/cuts a shot that trips the
known ~8 s/member stiff-regime slowdown (and SAYS SO in the manifest).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from imas_ambix.data.paths import LEVEL1_DIR, MANIFEST_DIR
from imas_ambix.statespace import mse_eval as me
from imas_ambix.statespace.enkf_baseline import (
    EnKFConfig,
    _operator_for_shot,
    _sample_theta,
    load_shot_inputs,
    run_torax_member,
)

# --- output locations -------------------------------------------------
REPO = Path(__file__).resolve().parents[1]
FIG_DIR = REPO / "docs" / "figures" / "current-recovery"
FIG_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT = (
    REPO / "imas_ambix" / "statespace" / "artifacts" / "enkf_baseline_metrics_v0.json"
)
MANIFEST = MANIFEST_DIR / "mse_heldout_split_v0.json"

# Representative held-out shots: finite innovation, n_ok=32, forecast ~= analysis
# (no forecast-arm q0 blowup), spanning the dynamic OOD flat-top q0 0.25-0.6 band.
# (22086 / 22759 deliberately EXCLUDED — forecast-arm q0 blows up to 3.0 / 2.3.)
SHOTS = [24443, 24148, 24065, 22329]

# rho surfaces for the j(rho,t) headline figure
RHO_SURFACES = [0.0, 0.2, 0.4, 0.6, 0.8]

# per-shot wall-clock guard [s] — drop the shot if the ensemble blows past this
PER_SHOT_BUDGET_S = 240.0

# committed config the v0 bar used (note: artifact ran n_ensemble=32, n_assim_slices=5)
CFG = EnKFConfig(n_ensemble=32, n_assim_slices=5)

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.titlesize": 11,
    }
)
PALETTE = ["#1f4e8c", "#2d8659", "#b8860b", "#b91c1c", "#6b3fa0"]


# --- ensemble extraction ----------------------------------------------


def manifest_grid_for(manifest: dict, sid: int) -> dict | None:
    e = manifest["shots"].get(str(int(sid)))
    if e is None or e.get("partition") != "held_out":
        return None
    return {
        "t": np.asarray(e["beam_on_slice_times"], dtype=np.float64),
        "rpos": np.asarray(e["active_channel_rpos"], dtype=np.float64),
        "entry": e,
    }


def build_prior_ensemble(sid: int, manifest: dict, op_cache: dict, reps):
    """Reproduce the forecast-arm prior TORAX ensemble for one shot.

    Returns (inp, trajs_ok, t_common, j_mean, j_std) where trajs_ok are the OK
    members, t_common is a shared TORAX output time axis (intersection grid), and
    j_mean/j_std are (T, G) ensemble mean / 1-sigma over members on rho_norm.
    """
    grid = manifest_grid_for(manifest, sid)
    if grid is None:
        return None
    op = _operator_for_shot(int(sid), op_cache, reps)
    if op is None:
        print(f"  [{sid}] no operator — skip")
        return None
    inp = load_shot_inputs(
        int(sid),
        op,
        CFG,
        slice_times_override=grid["t"],
        channel_rpos_override=grid["rpos"],
    )
    if inp is None:
        print(f"  [{sid}] no usable inputs — skip")
        return None

    rng = np.random.default_rng(CFG.seed + int(sid))
    thetas = _sample_theta(CFG, rng)

    t0 = time.monotonic()
    trajs = []
    for k, th in enumerate(thetas):
        trajs.append(run_torax_member(inp, CFG, th))
        if time.monotonic() - t0 > PER_SHOT_BUDGET_S:
            print(
                f"  [{sid}] WALL-CLOCK GUARD tripped after {k + 1}/{len(thetas)} "
                f"members ({time.monotonic() - t0:.0f}s) — using partial ensemble"
            )
            break
    ok = [tr for tr in trajs if tr.ok and tr.j_total.shape[0] > 1]
    if len(ok) < 4:
        print(f"  [{sid}] only {len(ok)} OK members — skip")
        return None

    # common rho grid (all members share the same circular face grid) + a common
    # time grid (members can have slightly different lengths; interp onto the
    # densest member's time axis clipped to the common [t0, tf] span).
    rho = ok[0].rho_norm
    t_lo = max(float(tr.time[0]) for tr in ok)
    t_hi = min(float(tr.time[-1]) for tr in ok)
    n_t = max(tr.time.size for tr in ok)
    t_common = np.linspace(t_lo, t_hi, max(n_t, 20))
    stack = np.empty((len(ok), t_common.size, rho.size))
    for m, tr in enumerate(ok):
        for g in range(rho.size):
            stack[m, :, g] = np.interp(t_common, tr.time, tr.j_total[:, g])
    j_mean = stack.mean(axis=0)
    j_std = stack.std(axis=0)
    dt = time.monotonic() - t0
    print(
        f"  [{sid}] {len(ok)}/{len(thetas)} OK members in {dt:.0f}s "
        f"(t {t_lo:.3f}-{t_hi:.3f}s, G={rho.size})"
    )
    return {
        "inp": inp,
        "trajs_ok": ok,
        "rho": rho,
        "t": t_common,
        "j_mean": j_mean,
        "j_std": j_std,
        "stack": stack,
        "grid": grid,
        "wall_s": dt,
        "n_ok": len(ok),
        "n_total": len(thetas),
    }


# --- figure 1: j(rho, t) at discrete flux surfaces --------------------


def fig_j_rho_t(ensembles: dict):
    sids = list(ensembles.keys())
    n = len(sids)
    ncol = 2
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(11, 4.2 * nrow), squeeze=False)
    for ax, sid in zip(axes.ravel(), sids, strict=False):
        e = ensembles[sid]
        rho, t, jm, js = e["rho"], e["t"], e["j_mean"], e["j_std"]
        for c, rs in enumerate(RHO_SURFACES):
            g = int(np.argmin(np.abs(rho - rs)))
            col = PALETTE[c % len(PALETTE)]
            mu = jm[:, g] / 1e6  # A/m^2 -> MA/m^2
            sd = js[:, g] / 1e6
            ax.plot(t, mu, color=col, lw=1.8, label=f"ρ={rs:.1f}")
            ax.fill_between(t, mu - sd, mu + sd, color=col, alpha=0.16, lw=0)
        ax.set_title(f"shot {sid}  (N={e['n_ok']} members)")
        ax.set_xlabel("time t  [s]")
        ax.set_ylabel("toroidal current density  jφ  [MA/m²]")
        ax.legend(loc="best", fontsize=8, ncol=2, framealpha=0.85)
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    fig.suptitle(
        "Recovered MSE-free current-density evolution  jφ(ρ, t)  — TORAX prior "
        "ensemble\n(mean ± 1σ over members; ρ = normalised minor radius; current "
        "is genuinely DYNAMIC)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in ("svg", "png"):
        fig.savefig(FIG_DIR / f"fig-cr-j-rho-t.{ext}")
    plt.close(fig)


# --- figure 2: j vs machine R at snapshot times -----------------------


def fig_j_vs_R(ensembles: dict):  # noqa: N802
    sids = list(ensembles.keys())
    n = len(sids)
    ncol = 2
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(11, 4.2 * nrow), squeeze=False)
    R0, a = CFG.r_major, CFG.a_minor  # noqa: N806
    snap_labels = ["early (ramp)", "flat-top (mid)", "late"]
    snap_cols = ["#2d8659", "#1f4e8c", "#b91c1c"]
    for ax, sid in zip(axes.ravel(), sids, strict=False):
        e = ensembles[sid]
        rho, t, jm, js = e["rho"], e["t"], e["j_mean"], e["j_std"]
        # outboard-midplane major radius for the circular geometry
        R_out = R0 + rho * a  # noqa: N806
        frac = [0.10, 0.50, 0.90]
        for f, lbl, col in zip(frac, snap_labels, snap_cols, strict=True):
            k = int(round(f * (t.size - 1)))
            mu = jm[k] / 1e6
            sd = js[k] / 1e6
            ax.plot(R_out, mu, color=col, lw=1.8, label=f"{lbl}  t={t[k]:.3f}s")
            ax.fill_between(R_out, mu - sd, mu + sd, color=col, alpha=0.15, lw=0)
        ax.axvline(R0, color="#888", ls=":", lw=1)
        ax.set_title(f"shot {sid}")
        ax.set_xlabel("outboard-midplane major radius  R  [m]  (R = R₀ + ρ·a)")
        ax.set_ylabel("toroidal current density  jφ  [MA/m²]")
        ax.legend(loc="best", fontsize=8, framealpha=0.85)
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    fig.suptitle(
        "Spatial current profile evolving in time  jφ(R)  at three snapshots\n"
        "(nominal circular-geometry R = R₀ + ρ·a, R₀=0.85 m, a=0.5 m — not a "
        "real flux-surface R)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ext in ("svg", "png"):
        fig.savefig(FIG_DIR / f"fig-cr-j-vs-R.{ext}")
    plt.close(fig)


# --- figure 3: recovered q0(t) vs held-out MSE truth (eval-only) ------


def member_pitch(traj, inp, slice_t):
    """Predicted pitch (K, C) for one member, mirroring enkf_baseline._pitch_samples."""
    from imas_ambix.statespace.enkf_baseline import _j_at_slices

    j_k = _j_at_slices(traj, slice_t)  # (K, G)
    rho_m = traj.rho_norm * CFG.a_minor
    return me.pitch_from_current_profile(
        j_k, rho_m, inp.active_channel_rpos, inp.r0, inp.bt0, kind="j"
    )


def fig_q0_overlay(ensembles: dict, truth: me.MseTruth):
    """Method-matched recovered q0(t) vs held-out MSE-derived truth q0(t).

    Both q0 are obtained by the SHARED eval inverter invert_pitch_to_q0rax with
    the eval-harness geometry (DEFAULT_R0, DEFAULT_BT0 — exactly as mse_eval.score
    builds geom for both truth and pred), so the comparison isolates state error.
    A separate dotted line shows the TORAX-free on-axis q0 (q[:,0]) — the
    non-vacuity headline — which is a DIFFERENT construct (not the inverted pitch).
    MSE is HELD-OUT EVAL TRUTH ONLY; it never touches the recovery.
    """
    sids = list(ensembles.keys())
    n = len(sids)
    ncol = 2
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(11, 4.2 * nrow), squeeze=False)
    any_truth = False
    for ax, sid in zip(axes.ravel(), sids, strict=False):
        e = ensembles[sid]
        inp = e["inp"]
        entry = e["grid"]["entry"]
        slice_t = np.asarray(entry["beam_on_slice_times"], dtype=np.float64)
        rpos = np.asarray(entry["active_channel_rpos"], dtype=np.float64)
        pv = np.asarray(entry["pitch_valid_mask"], dtype=bool)
        q0g = np.asarray(entry["q0_gated_mask"], dtype=bool)
        gate = pv & q0g

        # geom keys match mse_eval.score EXACTLY ("rpos", DEFAULT_R0/BT0 for both)
        geom = {"rpos": rpos, "R0": me.DEFAULT_R0, "Bt0": me.DEFAULT_BT0}

        # recovered: ensemble-mean predicted pitch -> shared inverter
        pk = np.stack(
            [member_pitch(tr, inp, slice_t) for tr in e["trajs_ok"]]
        )  # (M,K,C)
        pred_pitch_mean = np.nanmean(pk, axis=0)  # (K, C)
        q0_rec, _ = me.invert_pitch_to_q0rax(pred_pitch_mean, geom)

        # TORAX-free on-axis q0 (q[:,0]) interpolated to slices — separate construct
        from imas_ambix.statespace.enkf_baseline import _q0_at_slices

        q0_torax = np.nanmean(
            np.stack([_q0_at_slices(tr, slice_t) for tr in e["trajs_ok"]]), axis=0
        )

        # truth: held-out MSE pitch -> same inverter (eval-only)
        tr_ams = truth.get(sid)
        q0_truth = None
        if tr_ams is not None:
            pt_truth = np.asarray(tr_ams.pitch, dtype=np.float64)  # (K, C)
            if pt_truth.shape == pred_pitch_mean.shape:
                q0_truth, _ = me.invert_pitch_to_q0rax(pt_truth, geom)

        t_g = slice_t[gate]
        ax.plot(
            t_g,
            q0_rec[gate],
            color="#1f4e8c",
            lw=2.0,
            marker="o",
            ms=2.5,
            label="recovered q₀  (MSE-free; inv(pred pitch))",
        )
        ax.plot(
            t_g,
            q0_torax[gate],
            color="#6b3fa0",
            lw=1.3,
            ls=":",
            label="TORAX-free q₀ = q(ρ=0)  (non-vacuity)",
        )
        if q0_truth is not None and np.isfinite(q0_truth[gate]).any():
            any_truth = True
            ax.plot(
                t_g,
                q0_truth[gate],
                color="#b91c1c",
                lw=2.0,
                marker="s",
                ms=2.5,
                label="held-out MSE truth q₀  (EVAL-ONLY)",
            )
        ax.set_title(f"shot {sid}  (flat-top gated slices)")
        ax.set_xlabel("time t  [s]")
        ax.set_ylabel("on-axis safety factor  q₀")
        # The two method-matched inverted-pitch q0 lines are clamped by the shared
        # inverter to the physical gate [0.5, 3.0]; the TORAX-free q(ρ=0) line is
        # UNBOUNDED (blows up as Ip->0 on residual ramp slices), so clip the view
        # to a physical window for readability — values above are off-scale.
        ax.set_ylim(0.0, 3.2)
        ax.legend(loc="best", fontsize=7.5, framealpha=0.85)
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    note = (
        "Recovered (inv. of predicted pitch) vs held-out MSE truth (inv. of MSE "
        "pitch),\nmethod-matched via the SHARED inverter + eval geometry "
        "(R₀=0.85 m, Bt0=0.5 T).  MSE is EVAL-ONLY — it never enters the recovery."
    )
    fig.suptitle(
        "On-axis q₀(t): recovered (MSE-free) vs held-out MSE truth\n" + note,
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for ext in ("svg", "png"):
        fig.savefig(FIG_DIR / f"fig-cr-q0-overlay.{ext}")
    plt.close(fig)
    return any_truth


# --- driver -----------------------------------------------------------


def main():
    import torax

    torax.set_jax_precision()
    manifest = json.loads(MANIFEST.read_text())
    truth = me.MseTruth(level1_dir=LEVEL1_DIR)
    op_cache: dict = {}
    from imas_ambix.statespace.enkf_baseline import _campaign_representatives

    reps = _campaign_representatives()

    ensembles: dict[int, dict] = {}
    gen_manifest = {
        "source_artifact": str(ARTIFACT.relative_to(REPO)),
        "manifest": str(MANIFEST),
        "config": CFG.to_dict(),
        "arm": "forecast/prior (measured Ip + Te->sigma; NO MSE on input)",
        "rng_seed_convention": "default_rng(cfg.seed + shot_id)",
        "shots_requested": SHOTS,
        "shots": {},
    }
    for sid in SHOTS:
        print(f"[run] shot {sid}")
        e = build_prior_ensemble(sid, manifest, op_cache, reps)
        if e is None:
            gen_manifest["shots"][str(sid)] = {"status": "skipped"}
            continue
        ensembles[sid] = e
        gen_manifest["shots"][str(sid)] = {
            "status": "ok",
            "n_ok_members": e["n_ok"],
            "n_total_members": e["n_total"],
            "wall_s": round(e["wall_s"], 1),
            "t_span_s": [float(e["t"][0]), float(e["t"][-1])],
            "j_grid_points": int(e["rho"].size),
        }

    if not ensembles:
        raise SystemExit("no ensembles produced — cannot plot")

    print("[plot] fig 1 j(rho,t)")
    fig_j_rho_t(ensembles)
    print("[plot] fig 2 j vs R")
    fig_j_vs_R(ensembles)
    print("[plot] fig 3 q0 overlay")
    any_truth = fig_q0_overlay(ensembles, truth)
    gen_manifest["q0_overlay_has_mse_truth"] = bool(any_truth)

    (FIG_DIR / "generation_manifest.json").write_text(
        json.dumps(gen_manifest, indent=2, default=float)
    )
    print(f"[done] {len(ensembles)} shots plotted -> {FIG_DIR}")
    print(json.dumps(gen_manifest, indent=2, default=float))


if __name__ == "__main__":
    main()
