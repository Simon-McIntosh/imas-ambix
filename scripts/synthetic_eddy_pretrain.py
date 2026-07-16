#!/usr/bin/env python
"""Synthetic pretraining + degenerate-truth scoring for the temporal operator.

Two jobs, both on manufactured equilibria where the degenerate quantities are
KNOWN exactly (:mod:`imas_ambix.latent.synthetic_truth` — no EFIT anywhere):

``pretrain`` — known-eddy sequences.  Time-coupled synthetic sequences whose
vessel-eddy mode amplitudes evolve under the physical L/R ODE driven by a
ramping coil trajectory, with the TRUE decay times perturbed per sequence
(resistivity ×[0.7, 1.4]) so the physically-integrated feature is informative
but wrong — the eddy pathway must correct it from sensor evidence.  The eddy
head is supervised directly on the known mode amplitudes (plus the sensor
loss); recovery is scored on held-out synthetic sequences per mode.  The
checkpoint warm-starts real-corpus training.

``split`` — p′/FF′ split recovery.  Sequences of manufactured equilibria with
KNOWN profile-basis coefficients (the spine's own edge-capable basis) evolving
smoothly in time.  Each slice is fit by the frozen classical ladder (the
static spine, passive sidecar and all), then the temporal operator corrects
the sequence.  The split observable is the p′-group current fraction; the
score is |split − truth| for the static spine vs the temporal operator.  The
real-data split gate stays data-gated on Thomson T_e — this synthetic score is
the exact-truth bar the plan requires.

Artifacts: imas_ambix/latent/artifacts/temporal_operator/synthetic_pretrain_report.json
           (+ pretrained checkpoint temporal_operator_synthetic.pt)
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from imas_ambix.latent.gs_solve import build_passive_sidecar
from imas_ambix.latent.profile_greens_decoder import ProfileGreensDecoder
from imas_ambix.latent.residual_operator import slice_globals, slice_tokens
from imas_ambix.latent.synthetic_truth import build_campaign, manufacture
from imas_ambix.latent.temporal_operator import (
    TemporalOperator,
    build_passive_eigenbasis,
    load_eigenbasis,
    physical_eddy_history,
    save_checkpoint,
    save_eigenbasis,
)
from scripts.spine_label_factory import frozen_spine_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("synthetic_eddy_pretrain")

ARTIFACTS = Path("imas_ambix/latent/artifacts/temporal_operator")
CAMPAIGN_SHOT = 11766  # train-split shot: geometry + noise floor only

_CAMPAIGN_CACHE: dict[int, tuple] = {}  # per-process (fork workers rebuild once)


def _campaign_and_eigen(k_modes: int):
    if k_modes in _CAMPAIGN_CACHE:
        return _CAMPAIGN_CACHE[k_modes]
    campaign = build_campaign(CAMPAIGN_SHOT, nr=65, nz=97)
    key = campaign.table.signature.key
    cache = ARTIFACTS / f"eigenbasis-{key}-k{k_modes}.npz"
    if cache.exists():
        eigen = load_eigenbasis(cache)
    else:
        eigen = build_passive_eigenbasis(
            campaign.table, campaign.grid, sensor_scale=campaign.scale, k=k_modes
        )
        cache.parent.mkdir(parents=True, exist_ok=True)
        save_eigenbasis(cache, eigen)
    _CAMPAIGN_CACHE[k_modes] = (campaign, eigen)
    return campaign, eigen


def _coil_trajectory(campaign, rng, n_steps: int):
    """A ramping confining-coil trajectory (P4/P5/P6 ramp + wobble) [A]."""
    from imas_ambix.latent.synthetic_truth import build_confining_i_pf

    base = build_confining_i_pf(campaign.fwd, 6.0e4)
    ramp = np.linspace(rng.uniform(0.55, 0.8), 1.0, n_steps)
    wobble = 1.0 + 0.05 * np.cumsum(rng.normal(0, 0.2, size=n_steps))
    return np.outer(ramp * wobble, base)  # (T, C)


def _coeff_trajectory(rng, n_steps: int, n_dof: int = 6):
    """Smooth non-negative profile-coefficient trajectory (random-walk blend)."""
    c0 = rng.uniform(0.2, 1.0, size=n_dof)
    c1 = rng.uniform(0.2, 1.0, size=n_dof)
    w = np.linspace(0.0, 1.0, n_steps)[:, np.newaxis]
    traj = (1 - w) * c0 + w * c1
    traj *= 1.0 + 0.03 * np.cumsum(rng.normal(0, 0.3, size=(n_steps, n_dof)), axis=0)
    return np.clip(traj, 0.05, None)


def generate_sequence(job: tuple) -> dict | None:
    """One synthetic shot: known eddy history + known coefficients, chained."""
    seed, k_modes, n_steps, ip_amperes = job
    rng = np.random.default_rng(seed)
    campaign, eigen = _campaign_and_eigen(k_modes)
    grid = campaign.grid

    # per-sequence resistivity error: the TRUE decays differ from the nominal
    # feature integrator's — the model must correct the feature from sensors
    rho_factor = rng.uniform(0.7, 1.4)
    tau_true = eigen.tau / rho_factor

    times = 0.05 + np.cumsum(rng.uniform(0.012, 0.03, size=n_steps))
    i_pf_seq = _coil_trajectory(campaign, rng, n_steps)
    coeffs_seq = _coeff_trajectory(rng, n_steps)

    rows = []
    a_true = np.zeros(eigen.n_modes)
    warm = None
    prev_ic = np.zeros(eigen.m_cells.shape[1])
    prev_psi_m = i_pf_seq[0] @ eigen.m_coil.T + prev_ic @ eigen.m_cells.T
    for t in range(n_steps):
        psi_m = i_pf_seq[t] @ eigen.m_coil.T + prev_ic @ eigen.m_cells.T
        if t > 0:
            dt = times[t] - times[t - 1]
            decay = np.exp(-dt / tau_true)
            coeff = tau_true / dt * (1.0 - decay)
            a_true = decay * a_true + coeff * (-(psi_m - prev_psi_m))
        prev_psi_m = psi_m
        truth = manufacture(
            campaign,
            coeffs=coeffs_seq[t],
            n_p=3,
            n_f=3,
            nonneg_basis=True,
            passive_amplitudes=eigen.v @ a_true,
            i_pf=i_pf_seq[t],
            ip_amperes=ip_amperes,
            seed=int(seed * 1000 + t),
            warm_jphi=warm,
            continuation=warm is None,
        )
        if not truth.confined:
            logger.warning("seq %d step %d not confined — dropped", seed, t)
            return None
        warm = np.zeros(grid.flat_r.size)
        warm[grid.cells] = truth.cell_currents / (grid.dr * grid.dz)
        prev_ic = truth.cell_currents
        rows.append({"truth": truth, "a_true": a_true.copy(), "time_s": times[t]})
    return {"seed": seed, "rho_factor": rho_factor, "rows": rows}


def _sequence_features(seq, campaign, eigen, *, i_cell_source="truth"):
    """Model inputs for one synthetic sequence (tokens about the given arm)."""
    from imas_ambix.latent.boundary_disc import sensor_signature_arrays

    sr, sz, sang, is_flux = sensor_signature_arrays(campaign.table)
    rows = seq["rows"]
    times = np.array([r["time_s"] for r in rows])
    i_pf_seq = np.stack([r["truth"].i_pf for r in rows])
    i_cell_seq = np.stack(
        [
            r["truth"].cell_currents if i_cell_source == "truth" else r["i_cell_fit"]
            for r in rows
        ]
    )
    a_phys, u_drive = physical_eddy_history(eigen, times, i_pf_seq, i_cell_seq)
    m_sens = campaign.g_sens
    tokens, masks, gl = [], [], []
    for j, r in enumerate(rows):
        truth = r["truth"]
        pred = m_sens @ i_cell_seq[j] + truth.vacuum
        tk, m = slice_tokens(
            truth.measured,
            truth.vacuum,
            pred,
            truth.scale,
            truth.mask,
            sr,
            sz,
            sang,
            is_flux,
        )
        tokens.append(tk)
        masks.append(m)
        gl.append(slice_globals(truth.ip_amperes, float("nan")))
    dt = np.diff(times, prepend=times[0])
    dt[0] = float(np.median(dt[1:]))
    return {
        "tokens": np.stack(tokens),
        "masks": np.stack(masks),
        "globals": np.stack(gl),
        "dt": dt,
        "a_phys": a_phys,
        "u_drive": u_drive,
        "a_true": np.stack([r["a_true"] for r in rows]),
        "measured": np.stack([np.nan_to_num(r["truth"].measured) for r in rows]),
        "vacuum": np.stack([r["truth"].vacuum for r in rows]),
        "scale": np.stack([r["truth"].scale for r in rows]),
        "mask": np.stack([r["truth"].mask for r in rows]),
        "i_cell": i_cell_seq,
    }


def _to_tensors(feat, device="cpu"):
    return (
        torch.tensor(feat["tokens"], dtype=torch.float32, device=device).unsqueeze(0),
        torch.tensor(feat["masks"], device=device).unsqueeze(0),
        torch.tensor(feat["globals"], dtype=torch.float32, device=device).unsqueeze(0),
        torch.tensor(feat["dt"], dtype=torch.float32, device=device).unsqueeze(0),
        torch.tensor(feat["a_phys"], dtype=torch.float32, device=device).unsqueeze(0),
        torch.tensor(feat["u_drive"], dtype=torch.float32, device=device).unsqueeze(0),
    )


def run_pretrain(args) -> dict:
    campaign, eigen = _campaign_and_eigen(args.k_modes)
    jobs = [
        (seed, args.k_modes, args.n_steps, 6.0e5)
        for seed in range(args.n_sequences + args.n_val_sequences)
    ]
    t0 = time.perf_counter()
    if args.workers > 1:
        ctx = __import__("multiprocessing").get_context("fork")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
            seqs = [s for s in pool.map(generate_sequence, jobs) if s is not None]
    else:
        seqs = [s for s in map(generate_sequence, jobs) if s is not None]
    logger.info(
        "generated %d/%d sequences in %.0f s",
        len(seqs),
        len(jobs),
        time.perf_counter() - t0,
    )
    feats = [_sequence_features(s, campaign, eigen) for s in seqs]
    n_val = max(1, min(args.n_val_sequences, len(feats) // 4))
    train_f, val_f = feats[:-n_val], feats[-n_val:]

    a_all = np.concatenate([f["a_phys"] for f in train_f])
    u_all = np.concatenate([f["u_drive"] for f in train_f])

    def rstd(x):
        med = np.median(x, axis=0)
        return np.clip(1.4826 * np.median(np.abs(x - med), axis=0), 1e-30, None)

    eddy_std, drive_std = rstd(a_all), rstd(u_all)
    model = TemporalOperator(
        6, eigen.tau, eddy_std, drive_std, d_model=args.d_model, n_layers=args.n_layers
    )
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    a_sens_t = torch.tensor(eigen.a_sens, dtype=torch.float64)
    std_t = torch.tensor(eddy_std, dtype=torch.float64)

    def losses(f):
        dc, da = model(*_to_tensors(f))
        da = da[0].double()
        a_true = torch.tensor(f["a_true"], dtype=torch.float64)
        sup = (((da - a_true) / std_t) ** 2).mean()
        pred = (
            torch.tensor(f["i_cell"] @ campaign.g_sens.T + f["vacuum"])
            + da @ a_sens_t.T
        )
        w = torch.tensor(f["mask"], dtype=torch.float64)
        r = (pred - torch.tensor(f["measured"])) / torch.tensor(f["scale"])
        misfit = ((r**2) * w).sum() / w.sum().clamp(min=1.0)
        leash = (dc[0] ** 2).mean().double()
        return sup, misfit, leash

    best_val, best_state, patience = float("inf"), None, 0
    for epoch in range(args.epochs):
        model.train()
        order = np.random.default_rng(epoch).permutation(len(train_f))
        for i in order:
            sup, misfit, leash = losses(train_f[i])
            loss = sup + 0.1 * misfit + 10.0 * leash
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = float(np.mean([float(losses(f)[0]) for f in val_f]))
        if v < best_val - 1e-6:
            best_val, patience = v, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            patience += 1
        if patience >= 12:
            break
    model.load_state_dict(best_state)
    model.eval()

    # recovery score on held-out synthetic sequences
    da_pred, a_true_all, a_feat_all = [], [], []
    with torch.no_grad():
        for f in val_f:
            _dc, da = model(*_to_tensors(f))
            da_pred.append(da[0].double().numpy())
            a_true_all.append(f["a_true"])
            a_feat_all.append(f["a_phys"])
    da_pred = np.concatenate(da_pred)
    a_true_all = np.concatenate(a_true_all)
    a_feat_all = np.concatenate(a_feat_all)

    def r2(pred, true):
        ss_res = np.sum((pred - true) ** 2, axis=0)
        ss_tot = np.sum((true - true.mean(axis=0)) ** 2, axis=0).clip(min=1e-30)
        return 1.0 - ss_res / ss_tot

    r2_model = r2(da_pred, a_true_all)
    r2_feature = r2(a_feat_all, a_true_all)  # the uncorrected physical feature
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        ARTIFACTS / "temporal_operator_synthetic.pt",
        model,
        {
            "eddy_std": eddy_std,
            "drive_std": drive_std,
            "tau_init": eigen.tau,
            "pretrain_val_loss": best_val,
        },
    )
    out = {
        "n_sequences": len(train_f),
        "n_val_sequences": len(val_f),
        "n_steps": args.n_steps,
        "epochs_best": best_val,
        "eddy_recovery_r2_per_mode": np.round(r2_model, 4).tolist(),
        "eddy_recovery_r2_median": float(np.median(r2_model)),
        "feature_r2_per_mode": np.round(r2_feature, 4).tolist(),
        "feature_r2_median": float(np.median(r2_feature)),
        "checkpoint": str(ARTIFACTS / "temporal_operator_synthetic.pt"),
    }
    logger.info("pretrain: %s", {k: v for k, v in out.items() if "per_mode" not in k})
    return out


# ---------------------------------------------------------------------------
# p′/FF′ split recovery on exact synthetic truth
# ---------------------------------------------------------------------------
def split_fraction(coeffs, psi_n_cells, r_cells, r0) -> float:
    """p′-group current fraction of a K-coefficient profile at a ψ_N map."""
    from imas_ambix.latent.gs_solve import profile_basis

    images = profile_basis(
        psi_n_cells, r_cells, r0=r0, n_p=3, n_f=3, kind="monomial-nonneg"
    )
    cur = images * np.asarray(coeffs)[np.newaxis, :]
    tot = np.abs(cur).sum()
    if tot <= 0:
        return float("nan")
    return float(np.abs(cur[:, :3]).sum() / tot)


def operator_split_fraction(
    c_fit, dc, gross_col, gross_raw, psi_n_cells, r_cells, r0, s_true
) -> tuple[float, bool]:
    """Operator-corrected p′-group current fraction, robust to any checkpoint.

    The operator's correction ``dc`` lives in the Ip-normalised column space;
    re-express it on the raw ladder coefficients through the per-column
    gross-current scale, clip to the non-negative-basis sign, and read the
    split.  A checkpoint whose corrections are large enough to drive EVERY
    profile DOF to zero yields a currentless (non-physical) profile — a state
    the decoder's Ip renormalisation never actually emits, so it is a genuine
    operator FAILURE for the slice, not a valid answer.  Rather than let the
    resulting NaN poison ``np.median`` over the sequence (or silently drop the
    slice, which would hide the failure), score the collapse at the worst
    attainable fraction error — the valid fraction farthest from truth.

    Returns ``(split_fraction, degenerate)`` where ``degenerate`` flags the
    collapse so the caller can report how many slices the operator annihilated.
    """
    gross_raw = np.clip(np.asarray(gross_raw), 1e-30, None)
    c_op = np.clip(c_fit + dc * gross_col / gross_raw, 0.0, None)
    s_op = split_fraction(c_op, psi_n_cells, r_cells, r0)
    if np.isfinite(s_op):
        return s_op, False
    # farthest valid fraction from truth (max |s_op - s_true|, s_op ∈ [0, 1])
    return (0.0 if s_true >= 0.5 else 1.0), True


def fit_sequence_spine(job):
    """Frozen classical ladder fit for each slice of one synthetic sequence.

    Runs in a fork worker: rebuilds the (cached per-process) campaign and the
    passive sidecar locally so only the sequence itself crosses the pipe.
    """
    seq, k_modes = job
    from scripts.closure_gate_eval import fit_and_read_slice

    campaign, _eigen = _campaign_and_eigen(k_modes)
    spine, _sha = frozen_spine_config()
    sidecar = build_passive_sidecar(
        campaign.table,
        campaign.grid,
        g_passive=campaign.passive_g_sens,
        sensor_scale=campaign.scale,
        k=int(spine["interior_solve"]["passive_k"]),
    )
    isolve = spine["interior_solve"]
    spc = dict(spine["soft_priors"])
    grid = campaign.grid
    cell_area = grid.dr * grid.dz
    warm = None
    for r in seq["rows"]:
        payload = r["truth"].to_payload()
        f = fit_and_read_slice(
            grid,
            campaign.table,
            payload,
            beta0_grid=(0.5,),
            alpha_grid=(1.0,),
            cost_limit=float("inf"),
            convergence_limit=5e-3,
            retry_max_iterations=160,
            fit_mode="ladder",
            n_p=int(isolve["n_p"]),
            n_f=int(isolve["n_f"]),
            smoothness=float(isolve["smoothness"]),
            nonneg=True,
            passive=sidecar,
            passive_ridge=1.0,
            warm_jphi=warm,
            reseed_axis_r_max=float(isolve["reseed_axis_r_max"]),
            keep_psi=True,
            keep_jphi=True,
            basis=campaign.basis,
            meta={},
            soft_prior_cfg=spc,
            boundary_read=isolve["boundary_read_scoring"],
        )
        if not f.scored:
            r["fit"] = None
            continue
        if f.converged:
            warm = f.jphi_flat
        from scripts.closure_gate_eval import geometry_target_pushout

        _t, psi_ax, psi_b = geometry_target_pushout(f.psi, grid)
        psi_n = np.clip(
            (f.psi.ravel()[grid.cells] - psi_ax) / (psi_b - psi_ax), 0.0, 1.5
        )
        r["fit"] = f
        r["i_cell_fit"] = f.jphi_flat[grid.cells] * cell_area
        r["psi_n_fit"] = psi_n
    return seq


def run_split(args) -> dict:
    from imas_ambix.latent.temporal_operator import load_checkpoint

    campaign, eigen = _campaign_and_eigen(args.k_modes)
    jobs = [
        (1000 + seed, args.k_modes, args.n_split_steps, 6.0e5)
        for seed in range(args.n_split_sequences)
    ]
    ctx = __import__("multiprocessing").get_context("fork")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
        seqs = [s for s in pool.map(generate_sequence, jobs) if s is not None]
    logger.info("split: %d sequences generated; ladder-fitting", len(seqs))
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
        seqs = list(pool.map(fit_sequence_spine, [(s, args.k_modes) for s in seqs]))

    model, ckpt = load_checkpoint(args.checkpoint)
    dec = ProfileGreensDecoder(campaign.basis, n_p=3, n_f=3, kind="monomial-nonneg")
    grid = campaign.grid
    r_cells = grid.flat_r[grid.cells]

    err_spine, err_op, n_fit, n_degenerate = [], [], 0, 0
    for seq in seqs:
        rows = [r for r in seq["rows"] if r.get("fit") is not None]
        if len(rows) < 3:
            continue
        seq_f = {"rows": rows}
        feat = _sequence_features(seq_f, campaign, eigen, i_cell_source="fit")
        with torch.no_grad():
            dc, _da = model(*_to_tensors(feat))
            dc = dc[0].double().numpy()
        for j, r in enumerate(rows):
            n_fit += 1
            truth = r["truth"]
            true_psin = (
                (truth.psi.ravel()[grid.cells] - truth.axis_psi)
                / (truth.boundary_psi - truth.axis_psi)
            ).clip(0.0, 1.5)
            s_true = split_fraction(truth.coeffs_true, true_psin, r_cells, grid.r0)
            c_fit = np.asarray(r["fit"].coeffs, dtype=np.float64)
            s_spine = split_fraction(c_fit, r["psi_n_fit"], r_cells, grid.r0)
            # the operator's correction lives in the Ip-normalised column
            # basis; re-express it on the raw ladder coefficients through the
            # per-column gross-current scale before reading the split
            cols = dec.profile_columns(
                torch.tensor(r["psi_n_fit"]).unsqueeze(0),
                torch.tensor([truth.ip_amperes], dtype=torch.float64),
            )[0].numpy()
            from imas_ambix.latent.gs_solve import profile_basis

            images = profile_basis(
                r["psi_n_fit"],
                r_cells,
                r0=grid.r0,
                n_p=3,
                n_f=3,
                kind="monomial-nonneg",
            )
            gross_raw = np.abs(images).sum(axis=0).clip(min=1e-30)
            gross_col = np.abs(cols).sum(axis=0)
            s_op, degenerate = operator_split_fraction(
                c_fit,
                dc[j],
                gross_col,
                gross_raw,
                r["psi_n_fit"],
                r_cells,
                grid.r0,
                s_true,
            )
            n_degenerate += int(degenerate)
            err_spine.append(abs(s_spine - s_true))
            err_op.append(abs(s_op - s_true))
    out = {
        "n_split_slices": n_fit,
        "operator_degenerate_slices": n_degenerate,
        "split_abs_err_spine_median": float(np.median(err_spine)),
        "split_abs_err_operator_median": float(np.median(err_op)),
        "split_abs_err_spine_mean": float(np.mean(err_spine)),
        "split_abs_err_operator_mean": float(np.mean(err_op)),
        "operator_beats_spine": bool(np.median(err_op) < np.median(err_spine)),
    }
    logger.info("split recovery: %s", out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("pretrain", "split", "both"), default="both")
    ap.add_argument("--n-sequences", type=int, default=24)
    ap.add_argument("--n-val-sequences", type=int, default=8)
    ap.add_argument("--n-steps", type=int, default=20)
    ap.add_argument("--n-split-sequences", type=int, default=8)
    ap.add_argument("--n-split-steps", type=int, default=12)
    ap.add_argument("--k-modes", type=int, default=12)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument(
        "--checkpoint",
        type=str,
        default="imas_ambix/latent/artifacts/temporal_operator/temporal_operator.pt",
        help="operator checkpoint the split score evaluates",
    )
    args = ap.parse_args()

    report_path = ARTIFACTS / "synthetic_pretrain_report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    if args.mode in ("pretrain", "both"):
        report["pretrain"] = run_pretrain(args)
    if args.mode in ("split", "both"):
        report["split_recovery"] = run_split(args)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    logger.info("report -> %s", report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
