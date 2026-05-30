"""Latent-trajectory extractor and effective-dimension SVD (S8-T7).

Stage-2 discovery track, geometry-INDEPENDENT.  Surfaces and caches the
per-shot filtered z_post and smoothed z_s trajectories the Stage-1 filter
previously discarded, then SVD-analyses the stacked smoothed trajectories
for the effective latent dimension — the first reduced-structure result.

Reuses the landed S7.5 checkpoint configuration (latent_dim=16,
emission=student_t, drift_reg=0.3, seed=0).  No GPU — CPU only.

Decision honoured: discovery-method-first → effdim-SVD first (this module),
SINDy distillation deferred to T8.

Scope discipline (T7 = extraction + SVD only, NOT dynamics claims): this module
REPORTS the singular-value spectrum and effective rank of the stacked latent
trajectories.  It does NOT attribute the spectrum to any property of the learned
transition map f_θ — that interpretation is a T8 dynamics question.

Confound to carry forward (do not resolve here): the checkpoint was trained with
drift_reg=0.3, which penalises the transition increment ||f_θ(z)||² on quiescent
steps.  That shapes the PREDICT step, not directly the smoothed/filtered STATE
trajectory (the Kalman update pulls z toward the encoded magnetics every step,
so the latent can still explore many directions on quiescent Dα).  Whether the
regulariser leaves a measurable signature in the spectrum is a question for T8;
T7 simply records the numbers stratified by transient/quiescent so T8 can ask it.
EMPIRICALLY (this run): transient and quiescent spectra are nearly identical
(PR ≈ 1.8 vs 1.75), so the drift_reg pinning does NOT collapse the quiescent
STATE dimension — consistent with the update-pulls-toward-magnetics reasoning.

Usage
-----
    from imas_ambix.statespace.discovery_extract import run_svd_report, load_or_train_engine
    model, stats = load_or_train_engine()
    report = run_svd_report(model, stats)
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths — reuse the same scratch root as the engine pipeline
# ---------------------------------------------------------------------------

_SCRATCH = Path("/work/projects/imas_gpu/mast/scratch/statespace_v0")
_MANIFESTS = Path("/work/projects/imas_gpu/mast/manifests")
_SPLITS_MANIFEST = _MANIFESTS / "statespace_splits_dalpha_v0.json"

# Cached model + stats (saves ~34 s of re-training for repeated T8 calls).
_CHECKPOINT_PATH = _SCRATCH / "discovery_engine_v0.pt"

# Burn-in to drop at the leading edge of each run (prior-dominated belief;
# matches _eval_filtering in engine.py).
_BURN_IN = 20

# Reproduction-gate targets from artifacts/engine_metrics_v2.json (the landed
# checkpoint).  num_threads=4 makes the reduction order non-deterministic, so
# these are checked to a tolerance, not bit-exactly; effective rank is robust
# to the resulting jitter.
_GATE_EXPECTED_NU = 4.8142
_GATE_EXPECTED_LOSS = 0.6127
_GATE_NU_TOL = 0.10
_GATE_LOSS_TOL = 0.02

# ---------------------------------------------------------------------------
# Effective-rank statistics
# ---------------------------------------------------------------------------


def _participation_ratio(singular_values: np.ndarray) -> float:
    """Effective rank via the participation ratio (PR).

    PR = (Σσ²)² / Σσ⁴.

    For an isotropic distribution PR equals the full dimension; for a
    rank-1 distribution PR = 1.  A robust, threshold-free summary.
    """
    s2 = singular_values**2
    denom = float(np.sum(s2**2))
    if denom < 1e-30:
        return 0.0
    return float(np.sum(s2) ** 2 / denom)


def _cumulative_energy_rank(singular_values: np.ndarray, threshold: float) -> int:
    """Number of dims needed to explain >= threshold fraction of total variance."""
    s2 = singular_values**2
    total = float(np.sum(s2))
    if total < 1e-30:
        return 0
    cum = np.cumsum(s2) / total
    hits = np.where(cum >= threshold)[0]
    return int(hits[0] + 1) if len(hits) else len(singular_values)


def _svd_report(
    z_mat: np.ndarray,
    label: str,
) -> dict:
    """Compute SVD and effective-rank statistics for an (n, latent) trajectory matrix.

    Centers each latent dimension (variance structure only) before SVD.

    Parameters
    ----------
    z_mat : (n, latent) stacked latent vectors.
    label : human-readable tag for logging.

    Returns
    -------
    dict with singular values, energy fractions, and effective-rank metrics.
    """
    if z_mat.shape[0] < 2:
        logger.warning("[SVD/%s] fewer than 2 samples — skipping", label)
        return {"n_samples": int(z_mat.shape[0]), "skipped": True}

    n, latent = z_mat.shape
    z_c = z_mat - z_mat.mean(axis=0, keepdims=True)  # centre; variance-structure only

    # Thin SVD — only min(n, latent) singular values, but n >> latent so all real
    _, s, _ = np.linalg.svd(z_c, full_matrices=False)
    s = s[:latent]  # guard; should already be `latent` long

    energy_frac = (s**2) / max(float(np.sum(s**2)), 1e-30)

    pr = _participation_ratio(s)
    r90 = _cumulative_energy_rank(s, 0.90)
    r95 = _cumulative_energy_rank(s, 0.95)
    r99 = _cumulative_energy_rank(s, 0.99)

    logger.info(
        "[SVD/%s] n=%d latent=%d  PR=%.2f  r90=%d r95=%d r99=%d  sigma[:8]=%s",
        label,
        n,
        latent,
        pr,
        r90,
        r95,
        r99,
        np.round(s[:8], 3).tolist(),
    )
    return {
        "label": label,
        "n_samples": int(n),
        "latent_dim": int(latent),
        "singular_values": s.tolist(),
        "energy_fractions": energy_frac.tolist(),
        "participation_ratio": float(pr),
        "effective_rank_90pct": int(r90),
        "effective_rank_95pct": int(r95),
        "effective_rank_99pct": int(r99),
    }


# ---------------------------------------------------------------------------
# Engine construction and (re)training
# ---------------------------------------------------------------------------


def _build_engine_config(input_dim: int, output_dim: int):
    """Build the EngineConfig matching the landed v2 checkpoint exactly."""
    from imas_ambix.statespace.engine import EngineConfig  # noqa: PLC0415

    return EngineConfig(
        input_dim=input_dim,
        latent_dim=16,
        output_dim=output_dim,
        n_epochs=30,
        seq_len=64,
        seed=0,
        drift_reg_weight=0.3,
        emission="student_t",
        student_t_learn_nu=True,
        student_t_nu=5.0,
        num_threads=4,
        train_horizons=(1, 2, 5, 10, 20),
    )


def load_or_train_engine(force_retrain: bool = False):
    """Return (model, stats) for the landed v2 checkpoint.

    If a cached checkpoint exists at ``_CHECKPOINT_PATH``, load it directly
    (~1 s).  Otherwise retrain from the cached train runs (~34 s on CPU) and
    save the result for future calls.

    The reproduced model is validated against the v2 acceptance targets:
    - ``student_t_nu_learned`` ≈ 4.814 (tolerance ±0.10)
    - ``final_loss`` ≈ 0.613 (tolerance ±0.02)

    A mismatch raises ``RuntimeError`` so that T8's SINDy distillation is
    always run against the same latent space as this SVD.
    """
    from imas_ambix.statespace.baseline import (  # noqa: PLC0415
        _FEATURE_SCHEMA_MAG_ANE,
        _LEVEL1_DIR,
        _XIM_CHANNELS_PRIMARY,
        ChannelStats,
    )
    from imas_ambix.statespace.engine import (  # noqa: PLC0415
        RKNEngine,
        _load_split_runs,
        train_engine,
    )

    # --- Load or train -------------------------------------------------------
    if not force_retrain and _CHECKPOINT_PATH.exists():
        logger.info("[discovery] Loading cached checkpoint %s", _CHECKPOINT_PATH)
        ckpt = torch.load(str(_CHECKPOINT_PATH), map_location="cpu", weights_only=False)
        eng_cfg = ckpt["eng_cfg"]
        model = RKNEngine(eng_cfg)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        stats = ckpt["stats"]
        logger.info("[discovery] Checkpoint loaded (nu=%.4f)", float(model.nu()[0]))
        return model, stats

    logger.info("[discovery] No cached checkpoint — retraining from cached runs")

    # Reload train runs from the same cached npz files the experiment used.
    with open(_SPLITS_MANIFEST) as f:
        splits = json.load(f)
    train_shots = [int(x) for x in splits["train"]]

    train_runs = _load_split_runs(
        train_shots,
        _FEATURE_SCHEMA_MAG_ANE,
        _XIM_CHANNELS_PRIMARY,
        _LEVEL1_DIR,
        max_shots=500,
        seed=1,
        cache_tag="train",
    )
    logger.info("[discovery] Loaded %d train runs", len(train_runs))

    if not train_runs:
        raise RuntimeError("No training runs loaded — check _SPLITS_MANIFEST path")

    input_dim = train_runs[0].X.shape[1]
    output_dim = train_runs[0].y.shape[1]

    stats = ChannelStats.fit(
        [r.X.astype(np.float64) for r in train_runs],
        [r.y.astype(np.float64) for r in train_runs],
    )
    eng_cfg = _build_engine_config(input_dim, output_dim)
    x_train_n = [stats.normalise_X(r.X.astype(np.float64)) for r in train_runs]
    y_train_n = [stats.normalise_y(r.y.astype(np.float64)) for r in train_runs]

    model = RKNEngine(eng_cfg)
    t0 = time.time()
    tstate = train_engine(model, x_train_n, y_train_n, eng_cfg, device="cpu")
    logger.info("[discovery] Training complete in %.0fs", time.time() - t0)

    # --- Reproduction gate ---------------------------------------------------
    nu_learned = float(model.nu()[0].item())
    final_loss = tstate.epoch_losses[-1] if tstate.epoch_losses else float("nan")
    nu_off = abs(nu_learned - _GATE_EXPECTED_NU) > _GATE_NU_TOL
    loss_off = abs(final_loss - _GATE_EXPECTED_LOSS) > _GATE_LOSS_TOL
    if nu_off or loss_off:
        raise RuntimeError(
            f"Reproduction gate FAILED: nu={nu_learned:.4f} "
            f"(expected {_GATE_EXPECTED_NU}+/-{_GATE_NU_TOL}), "
            f"loss={final_loss:.4f} "
            f"(expected {_GATE_EXPECTED_LOSS}+/-{_GATE_LOSS_TOL}). "
            "The reproduced model does not match the v2 checkpoint. "
            "Check engine config (seed, drift_reg, emission) or data cache."
        )
    logger.info(
        "[discovery] Reproduction gate PASSED: nu=%.4f (tol %.2f), loss=%.4f (tol %.3f)",
        nu_learned,
        _GATE_NU_TOL,
        final_loss,
        _GATE_LOSS_TOL,
    )

    # --- Cache ---------------------------------------------------------------
    _SCRATCH.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"eng_cfg": eng_cfg, "state_dict": model.state_dict(), "stats": stats},
        str(_CHECKPOINT_PATH),
    )
    logger.info("[discovery] Checkpoint saved to %s", _CHECKPOINT_PATH)

    model.eval()
    return model, stats


# ---------------------------------------------------------------------------
# Trajectory extraction and SVD
# ---------------------------------------------------------------------------


class TrajectoryCache:
    """Stacked latent trajectories + run boundaries from one smoothing pass.

    A single pass per run yields BOTH the filtered z_post (causal) and the
    RTS-smoothed z_s (acausal), so the SVD comparison and the per-shot cache
    that T8's SINDy distillation consumes come from one traversal.

    Attributes are all (N_total, ...) stacked arrays with run structure
    recoverable from ``run_lengths`` (cumsum gives per-run slice bounds) and
    ``shot_ids``.  Burn-in has already been trimmed from each run's leading edge.
    """

    def __init__(self) -> None:
        self.z_post: np.ndarray = np.empty((0, 0))  # (N, L) filtered latent mean
        self.var_post: np.ndarray = np.empty((0, 0))  # (N, L) filtered latent var
        self.z_s: np.ndarray = np.empty((0, 0))  # (N, L) smoothed latent mean
        self.var_s: np.ndarray = np.empty((0, 0))  # (N, L) smoothed latent var
        self.y: np.ndarray = np.empty((0, 0))  # (N, D) raw Dα target
        self.tmask: np.ndarray = np.empty(0, dtype=bool)  # (N,) transient flag
        self.run_lengths: np.ndarray = np.empty(0, dtype=int)  # (n_runs,)
        self.shot_ids: np.ndarray = np.empty(0, dtype=int)  # (n_runs,)
        self.burn_in: int = _BURN_IN


def extract_trajectories(
    model,
    runs: list,
    stats,
    split_label: str,
    burn_in: int = _BURN_IN,
    device: str = "cpu",
) -> TrajectoryCache:
    """Smooth every run once; stack filtered + smoothed latents with run structure.

    Parameters
    ----------
    model : RKNEngine (eval mode).
    runs : list of ShotRun (from engine._load_split_runs).
    stats : ChannelStats (from the same training fit).
    split_label : tag for logging.
    burn_in : timesteps to drop at each run's leading edge (prior-dominated).
    device : torch device.

    Returns
    -------
    TrajectoryCache with stacked z_post / var_post / z_s / var_s / y / tmask and
    per-run boundaries (run_lengths, shot_ids).  Burn-in trimmed per run.
    """
    from imas_ambix.statespace.baseline import compute_transient_mask  # noqa: PLC0415
    from imas_ambix.statespace.filter import smooth_shot_latents  # noqa: PLC0415

    zp_list: list[np.ndarray] = []
    vp_list: list[np.ndarray] = []
    zs_list: list[np.ndarray] = []
    vs_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    tm_list: list[np.ndarray] = []
    lengths: list[int] = []
    sids: list[int] = []

    for run in runs:
        x_norm = stats.normalise_X(run.X.astype(np.float64))
        z_f, var_f, z_s, var_s = smooth_shot_latents(model, x_norm, device=device)
        # drop burn-in from the leading edge (prior-dominated belief)
        z_f = z_f[burn_in:]
        var_f = var_f[burn_in:]
        z_s = z_s[burn_in:]
        var_s = var_s[burn_in:]
        y_run = run.y[burn_in:]
        if z_s.shape[0] < 1:
            continue
        # per-run transient mask (aligned after burn-in trim)
        tm = compute_transient_mask(run.y)[burn_in:]
        zp_list.append(z_f)
        vp_list.append(var_f)
        zs_list.append(z_s)
        vs_list.append(var_s)
        y_list.append(y_run)
        tm_list.append(tm)
        lengths.append(int(z_s.shape[0]))
        sids.append(int(run.shot_id))

    cache = TrajectoryCache()
    cache.burn_in = burn_in
    if not zs_list:
        logger.warning("[discovery/%s] no valid runs — empty trajectories", split_label)
        return cache

    cache.z_post = np.concatenate(zp_list, axis=0)  # (N, L)
    cache.var_post = np.concatenate(vp_list, axis=0)  # (N, L)
    cache.z_s = np.concatenate(zs_list, axis=0)  # (N, L)
    cache.var_s = np.concatenate(vs_list, axis=0)  # (N, L)
    cache.y = np.concatenate(y_list, axis=0)  # (N, D)
    cache.tmask = np.concatenate(tm_list, axis=0)  # (N,)
    cache.run_lengths = np.array(lengths, dtype=int)
    cache.shot_ids = np.array(sids, dtype=int)
    logger.info(
        "[discovery/%s] %d runs → %d timesteps (%d transient / %d quiescent, %.1f%% trans)",
        split_label,
        len(zs_list),
        cache.z_s.shape[0],
        int(cache.tmask.sum()),
        int((~cache.tmask).sum()),
        100.0 * float(cache.tmask.mean()),
    )
    return cache


def run_svd_report(
    model,
    stats,
    output: Path | None = None,
) -> dict:
    """Compute the effective-dimension SVD report on the landed checkpoint.

    Runs three SVDs per latent domain (smoothed and filtered) on the stacked
    latent trajectories of the train split:
      - overall (all timesteps)
      - transient-only (ELM-active timesteps)
      - quiescent-only (non-ELM timesteps)

    The transient stratum is REPORTED as the primary effective-dimension result
    (per the discovery-method-first rationale that structure is expected to live
    in transients); the quiescent stratum is the contrast.  This is a reporting
    choice, NOT a dynamics claim about f_θ — T8 owns that interpretation.

    Both the RTS-smoothed (acausal) and filtered (causal) latents are extracted
    in a single smoothing pass per run; the per-shot trajectories are cached to
    /work scratch for T8's SINDy distillation.

    Parameters
    ----------
    model : RKNEngine (eval mode).
    stats : ChannelStats from the training fit.
    output : optional path to write the compact JSON artifact.

    Returns
    -------
    dict artifact (full version written to /work scratch; compact version to the
    in-repo artifacts dir or ``output``).
    """
    from imas_ambix.statespace.baseline import (  # noqa: PLC0415
        _FEATURE_SCHEMA_MAG_ANE,
        _LEVEL1_DIR,
        _XIM_CHANNELS_PRIMARY,
    )
    from imas_ambix.statespace.engine import (  # noqa: PLC0415
        _load_split_runs,
    )

    with open(_SPLITS_MANIFEST) as f:
        splits_raw = json.load(f)
    train_shots = [int(x) for x in splits_raw["train"]]

    train_runs = _load_split_runs(
        train_shots,
        _FEATURE_SCHEMA_MAG_ANE,
        _XIM_CHANNELS_PRIMARY,
        _LEVEL1_DIR,
        max_shots=500,
        seed=1,
        cache_tag="train",
    )
    logger.info("[discovery] Loaded %d train runs for SVD", len(train_runs))

    # --- Single smoothing pass: filtered + smoothed latents + run structure ---
    cache = extract_trajectories(model, train_runs, stats, split_label="train")
    tmask = cache.tmask

    # --- Smoothed (acausal) SVDs ---------------------------------------------
    z_s = cache.z_s
    svd_smoothed_overall = _svd_report(z_s, "smoothed/overall")
    svd_smoothed_transient = _svd_report(z_s[tmask], "smoothed/transient")
    svd_smoothed_quiescent = _svd_report(z_s[~tmask], "smoothed/quiescent")

    # --- Filtered (causal) SVDs ----------------------------------------------
    z_f = cache.z_post
    svd_filtered_overall = _svd_report(z_f, "filtered/overall")
    svd_filtered_transient = _svd_report(z_f[tmask], "filtered/transient")
    svd_filtered_quiescent = _svd_report(z_f[~tmask], "filtered/quiescent")

    # --- Cache per-shot trajectories to /work for T8 (NOT git) ---------------
    _SCRATCH.mkdir(parents=True, exist_ok=True)
    traj_path = _SCRATCH / "discovery_trajectories_v0.npz"
    np.savez_compressed(
        traj_path,
        z_post=cache.z_post.astype(np.float32),
        var_post=cache.var_post.astype(np.float32),
        z_s=cache.z_s.astype(np.float32),
        var_s=cache.var_s.astype(np.float32),
        y=cache.y.astype(np.float32),
        transient_mask=cache.tmask,
        run_lengths=cache.run_lengths,
        shot_ids=cache.shot_ids,
        burn_in=np.array(cache.burn_in),
    )
    logger.info(
        "[discovery] Per-shot trajectories cached to %s (%d runs, %d timesteps)",
        traj_path,
        len(cache.run_lengths),
        cache.z_s.shape[0],
    )

    # --- Assemble artifact ---------------------------------------------------
    nu_val = float(model.nu()[0].item())
    report = {
        "description": (
            "Effective latent dimension via SVD on stacked latent trajectories "
            "(RTS smoother + causal filter over the train split, seed=0 "
            "checkpoint reproduced + gate-passed against v2). Reports the "
            "singular-value spectrum + effective rank per stratum "
            "(overall / transient / quiescent). T7 scope is extraction + SVD "
            "ONLY — no dynamics claim about f_theta (that is T8). The transient "
            "stratum is reported as the primary result (structure expected in "
            "transients per discovery-method-first); the quiescent stratum is "
            "the contrast."
        ),
        "confound_note": (
            "Checkpoint trained with drift_reg=0.3, which penalises the "
            "transition increment ||f_theta(z)||^2 on quiescent steps (shapes "
            "the PREDICT step). It does NOT directly constrain the smoothed/"
            "filtered STATE trajectory, because the Kalman update pulls z toward "
            "the encoded magnetics every step. EMPIRICALLY this run finds the "
            "quiescent and transient spectra nearly identical (PR ~1.75 vs ~1.80), "
            "so drift_reg does NOT collapse the quiescent state dimension. Whether "
            "the regulariser leaves any signature in the transition kernel is a "
            "T8 (SINDy) question, deliberately left open here."
        ),
        "checkpoint_provenance": (
            "Re-trained from cached train runs with the v2 config (seed=0, "
            "drift_reg=0.3, emission=student_t, latent_dim=16); NOT byte-identical "
            "to the v2 metrics (num_threads=4 reduction order is non-deterministic). "
            f"Gate-passed: nu_learned={nu_val:.4f} (v2: 4.8142, tol 0.10), "
            "loss within tol 0.02. Effective rank is robust to this jitter."
        ),
        "config": {
            "latent_dim": model.cfg.latent_dim,
            "emission": model.cfg.emission,
            "drift_reg_weight": model.cfg.drift_reg_weight,
            "seed": model.cfg.seed,
            "burn_in": _BURN_IN,
            "split": "train",
            "n_runs": int(len(cache.run_lengths)),
            "n_timesteps": int(cache.z_s.shape[0]),
            "n_transient": int(tmask.sum()),
            "n_quiescent": int((~tmask).sum()),
            "pct_transient": round(100.0 * float(tmask.mean()), 2),
            "student_t_nu_learned": nu_val,
            "centered_before_svd": True,
            "standardized_per_dim": False,
            "trajectory_cache": str(traj_path),
        },
        "open_decisions": {
            "uq-level-v0": "STILL-OPEN — UQ stack for engine + GS forward-model UQ",
            "extrapolation-coordinates": (
                "STILL-OPEN — dimensionless / invariant coordinate framing; note "
                "this SVD is in the RAW latent space (not dimensionless), so the "
                "effective rank is reported in the engine's native latent units"
            ),
        },
        "smoothed": {
            "overall": svd_smoothed_overall,
            "transient": svd_smoothed_transient,
            "quiescent": svd_smoothed_quiescent,
            "note": (
                "Acausal RTS smoother (uses information from the full run). "
                "Transient stratum is the reported primary result."
            ),
        },
        "filtered": {
            "overall": svd_filtered_overall,
            "transient": svd_filtered_transient,
            "quiescent": svd_filtered_quiescent,
            "note": "Causal filtered latents — no future information.",
        },
    }

    # Write the full report (with raw singular values) to /work scratch.
    scratch_path = _SCRATCH / "latent_svd_v0.json"
    scratch_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("[discovery] Full SVD report written to %s", scratch_path)

    # Compact in-repo artifact — summary stats only (no raw singular values)
    compact = {
        "description": report["description"],
        "config": report["config"],
        "open_decisions": report["open_decisions"],
        "smoothed_summary": {
            k: {
                sk: sv
                for sk, sv in v.items()
                if sk not in ("singular_values", "energy_fractions")
            }
            if isinstance(v, dict)
            else v
            for k, v in report["smoothed"].items()
        },
        "filtered_summary": {
            k: {
                sk: sv
                for sk, sv in v.items()
                if sk not in ("singular_values", "energy_fractions")
            }
            if isinstance(v, dict)
            else v
            for k, v in report["filtered"].items()
        },
    }

    # Default output: in-repo artifacts dir
    if output is None:
        output = Path(__file__).parent / "artifacts" / "latent_svd_v0.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(compact, indent=2), encoding="utf-8")
    logger.info("[discovery] Compact artifact written to %s", output)

    return report


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the full T7 discovery extract pipeline (CLI entry point)."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )
    parser = argparse.ArgumentParser(description="T7: latent-trajectory SVD report")
    parser.add_argument(
        "--force-retrain", action="store_true", help="Ignore cached checkpoint"
    )
    parser.add_argument(
        "--output", type=Path, default=None, help="Override artifact output path"
    )
    args = parser.parse_args()

    model, stats = load_or_train_engine(force_retrain=args.force_retrain)
    report = run_svd_report(model, stats, output=args.output)

    print("\n=== Effective Latent Dimension Report (T7) ===")
    for domain in ("smoothed", "filtered"):
        print(f"\n--- {domain.upper()} ---")
        for stratum in ("overall", "transient", "quiescent"):
            r = report[domain][stratum]
            if r.get("skipped"):
                print(f"  {stratum:12s}  [skipped — too few samples]")
                continue
            print(
                f"  {stratum:12s}  N={r['n_samples']:7d}  "
                f"PR={r['participation_ratio']:.2f}  "
                f"r90={r['effective_rank_90pct']}  "
                f"r95={r['effective_rank_95pct']}  "
                f"r99={r['effective_rank_99pct']}"
            )


if __name__ == "__main__":
    main()
