#!/usr/bin/env python
"""Held-out gate evaluation of a trained GS-grounded latent-engine checkpoint.

Closes the f-malwm-02 done-when #2 measurement debt — "GS-readout axis /
X-point / boundary skill matches the absolute-magnetics oracle (~0.5-0.7)" —
on a LEARNED hybrid latent (:mod:`scripts.train_grounded_latent_engine`),
never measured before.  Reuses the 160-slice gate-2 protocol VERBATIM from
``scripts/patch_gate_eval.py`` (same held-out shots, slice selection,
train-mean baseline, per-quantity skill formulas — imported, never copied) but
replaces the per-slice variational inverse with a single forward pass of the
trained :class:`~imas_ambix.latent.encoder.HybridLatentEncoder`, exactly as
``scripts/patch_encoder_gate_eval.py`` does for the amortised patch-current
encoder.

Five measurements, per the plan:

(a) per-quantity axis / X-point / LCFS skill vs the train-mean baseline and
    the 0.5-0.7 supervised-oracle bar (:mod:`imas_ambix.eval.magnetics_oracle`);
(b) grounding — mean whitened sensor misfit vs a paired shuffled-current
    control, in-signature and cross-signature (the >=5.53x discrimination bar
    from ``scripts/patch_encoder_gate_eval.py``);
(c) 160/160 availability (n_scored / n_candidate against the 8 x 20-slice
    theoretical ceiling);
(d) the CLOSURE GATE — the closure-coordinate head's ``(a_k, b_k)`` readout
    vs a per-slice regression fit on the TRAINING-FREE variational inverse's
    currents (:func:`imas_ambix.latent.patch_inverse.invert_slices` under the
    locked discrepancy policy lambda0=3, ratio=1.5, lambda_max=100) on
    held-out FLAT-TOP slices, reporting the fraction within the regression's
    own 1-sigma per bin and pooled, plus the F^2>=0 integrability rate;
(e) D>=0 + command-load-bearing structural verification of the trained
    transport prior.

Artifact: ``imas_ambix/latent/artifacts/grounded_latent/gate_<run>.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

try:  # very recent addition (sensor-channel-set determinism fix) -- an older
    # checkpoint or checkout predates it; label honestly rather than crash
    from imas_ambix.gs.geometry import GEOMETRY_TABLE_VERSION
except ImportError:
    GEOMETRY_TABLE_VERSION = None
from imas_ambix.gs.operator import COIL_MODEL_VERSION, build_operator
from imas_ambix.latent.data import (
    feature_schema,
    load_shot_windows,
    read_split_shot_lists,
)
from imas_ambix.latent.encoder import HybridLatentEncoder, LatentConfig
from imas_ambix.latent.patch_inverse import InverseConfig, invert_slices
from imas_ambix.latent.structure_residual import fit_flux_functions, integrate_closures
from imas_ambix.latent.transport import FluxDiffusionPrior

# reuse the gate-2 protocol verbatim
from scripts.patch_gate_eval import (
    TARGET_NAMES,  # noqa: F401  (kept for parity / downstream imports)
    geometry_target,
    score,
    shot_payloads,
    train_mean_baseline,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("grounded_latent_gate_eval")

ARTIFACTS = Path("imas_ambix/latent/artifacts/grounded_latent")

#: the supervised-oracle skill range the f-malwm-02 done-when references
#: (imas_ambix/eval/magnetics_oracle.py: +0.59 / +0.82 on real data)
ORACLE_BAR = (0.5, 0.7)

#: the locked bounded-discrepancy weight policy (patch-current-force-balance
#: P3) — lambda0=3, ratio=1.5, lambda_max=100
CLOSURE_INVERSE_CONFIG = InverseConfig(
    policy="discrepancy",
    lambda_fb=3.0,
    misfit_ratio=1.5,
    lambda_max=100.0,
    connectivity="locality",
)

#: TF-field x R0 product used for the F^2 vacuum reference — same constant
#: ``scripts/patch_gate_eval.py`` uses for its own closure integrability check
F_VAC = 0.85 * 0.55

#: shuffled-current grounding discrimination bar (patch_encoder_gate_eval.py)
GROUNDING_BAR = 5.53


def _i_cell_from_latent(
    closure_x: torch.Tensor, ip: torch.Tensor, candidate_mask
) -> torch.Tensor:
    """``I = x . Ip / n_cells . candidate_mask`` — mirrors
    :meth:`~imas_ambix.latent.engine.GSGroundedLatentEngine.i_cell_from_latent`
    exactly, applied directly to the encoder output (no engine object needed
    for the per-shot geometry read — the transport prior it also owns plays
    no part here)."""
    n = closure_x.shape[-1]
    mask = candidate_mask.to(dtype=closure_x.dtype, device=closure_x.device)
    return closure_x * (ip[:, None] / n) * mask[None, :]


def _whitened_misfit(basis, i_cell: np.ndarray, payload) -> float:
    """Masked whitened mean-square sensor misfit of ``i_cell`` for one slice."""
    m_sens = basis.m_sens.to(torch.float64).cpu().numpy()
    pred = np.asarray(payload.vacuum) + m_sens @ np.asarray(i_cell, dtype=np.float64)
    resid = np.where(payload.mask, (pred - payload.measured) / payload.scale, 0.0)
    denom = max(float(payload.mask.sum()), 1.0)
    return float(np.sum(resid**2) / denom)


def _grounding(enc_misfits: np.ndarray, shuf_misfits: np.ndarray) -> dict:
    enc_m = float(np.mean(enc_misfits)) if len(enc_misfits) else float("nan")
    shuf_m = float(np.mean(shuf_misfits)) if len(shuf_misfits) else float("nan")
    ratio = shuf_m / enc_m if enc_m > 0 else float("inf")
    return {
        "encoder_misfit_mean": enc_m,
        "shuffled_misfit_mean": shuf_m,
        "ratio": ratio,
        "bar": GROUNDING_BAR,
        "pass": bool(ratio >= GROUNDING_BAR),
    }


def _bundle(model, ref, baseline_vec, enc_arr, shuf_arr, sel) -> dict | None:
    if int(sel.sum()) == 0:
        return None
    sc = score(model[sel], ref[sel], baseline_vec)
    sc.pop("axis_errors")
    sc["n_scored"] = int(sel.sum())
    sc["grounding"] = _grounding(enc_arr[sel], shuf_arr[sel])
    sc["oracle_bar"] = list(ORACLE_BAR)
    axis_skill = sc.get("axis_skill")
    sc["meets_oracle_bar"] = (
        bool(axis_skill is not None and axis_skill >= ORACLE_BAR[0])
        if axis_skill is not None
        else None
    )
    return sc


def _flattop_rows(w, payloads, frac: float = 0.5) -> list[int]:
    """Indices into ``payloads`` whose |dIp/dt| sits in the flatter half of
    the shot's own gate slices — a simple, documented flat-top proxy (no
    dedicated flat-top label exists upstream)."""
    ip = np.abs(np.asarray(w.anchored[:, 0], dtype=np.float64))
    dip = np.abs(np.gradient(ip, w.times))
    t_index = np.array([p.t_index for p in payloads])
    dip_at = dip[t_index]
    finite = np.isfinite(dip_at)
    if not finite.any():
        return []
    thresh = np.nanpercentile(dip_at[finite], 100 * frac)
    return [i for i in range(len(payloads)) if finite[i] and dip_at[i] <= thresh]


def _closure_gate_for_shot(
    basis, grid, payloads, flat_idx, pred_a, pred_b, n_closure_bins, device
) -> list[dict]:
    """Per-flat-top-slice closure comparison: head prediction vs the
    training-free variational-inverse's own regression fit + its 1-sigma."""
    rows = []
    if not flat_idx:
        return rows
    sel_payloads = [payloads[i] for i in flat_idx]
    inversions = invert_slices(
        basis, sel_payloads, CLOSURE_INVERSE_CONFIG, device=device
    )
    r_c = basis.r_cells.to(torch.float64)
    z_c = basis.z_cells.to(torch.float64)
    for local_k, i in enumerate(flat_idx):
        inv = inversions[local_k]
        p = payloads[i]
        psi_c = basis.psi_cells_np(inv.i_cell, p.i_pf)
        jphi = inv.i_cell / float(basis.cell_area)
        fit = fit_flux_functions(
            torch.as_tensor(psi_c, dtype=torch.float64),
            r_c,
            torch.as_tensor(jphi, dtype=torch.float64),
            n_bins=n_closure_bins,
            z_c=z_c,
            connectivity="locality",
        )
        psi2d = basis.psi_grid_2d_np(inv.i_cell, p.i_pf)
        _target, psi_ax, psi_b = geometry_target(psi2d, grid)
        mass = fit.weight_mass.numpy()
        keep = mass > 1e-3 * max(float(mass.max()), 1e-30)
        a_k, b_k = fit.a_k.numpy(), fit.b_k.numpy()
        a_err, b_err = fit.a_err.numpy(), fit.b_err.numpy()
        pa, pb = pred_a[i], pred_b[i]
        within_a = np.abs(pa - a_k) <= a_err
        within_b = np.abs(pb - b_k) <= b_err
        integ = integrate_closures(
            fit, psi_axis=psi_ax, psi_boundary=psi_b, f_vac=F_VAC
        )
        f2_ok = (
            bool(np.all(integ["f_squared"] >= -1e-12))
            if integ["f_squared"].size
            else None
        )
        rows.append(
            {
                "shot": p.shot,
                "t_index": p.t_index,
                "keep": keep.tolist(),
                "within_1sigma_a": within_a.tolist(),
                "within_1sigma_b": within_b.tolist(),
                "f2_min": float(np.min(integ["f_squared"]))
                if integ["f_squared"].size
                else None,
                "f2_nonneg": f2_ok,
            }
        )
    return rows


def _transport_structural_checks(
    payload_transport: dict, n_free: int, per_sig_n_coil: dict
) -> dict:
    """D>=0 + command-load-bearing verification, per trained transport."""
    out = {}
    torch.manual_seed(0)
    for key, state in payload_transport.items():
        n_coil = max(int(per_sig_n_coil.get(key, 1)), 1)
        nrho = int(state["noninductive.weight"].shape[0])
        tr = FluxDiffusionPrior(nrho=nrho, cmd_dim=n_coil, feat_dim=n_free).double()
        tr.load_state_dict(state)
        tr.eval()
        b = 8
        psi = torch.randn(b, nrho, dtype=torch.float64)
        rho = torch.linspace(0.1, 1.0, nrho, dtype=torch.float64)
        feat = torch.randn(b, n_free, dtype=torch.float64)
        cmd = torch.randn(b, n_coil, dtype=torch.float64)
        with torch.no_grad():
            d_cmd = tr.dpsi_dt(psi, rho, feat, cmd)
            d_zero = tr.dpsi_dt(psi, rho, feat, torch.zeros_like(cmd))
            diffusivity_min = float(tr.diffusivity(feat).min().item())
        out[key] = {
            "diffusivity_min": diffusivity_min,
            "d_geq_0": bool(diffusivity_min > 0.0),
            "command_load_bearing": bool((d_cmd - d_zero).abs().sum().item() > 0.0),
        }
    return out


def main() -> int:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--n-train", type=int, default=40)
    ap.add_argument("--n-heldout", type=int, default=8)
    ap.add_argument("--nr", type=int, default=65)
    ap.add_argument("--nz", type=int, default=97)
    ap.add_argument("--max-slices-per-shot", type=int, default=20)
    ap.add_argument("--min-ip-ka", type=float, default=300.0)
    ap.add_argument("--n-baseline-shots", type=int, default=10)
    ap.add_argument("--flattop-frac", type=float, default=0.5)
    ap.add_argument("--run", type=str, default="direct")
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = (
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else args.device
    )
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    payload = torch.load(Path(args.checkpoint), map_location=device, weights_only=False)
    extra = payload["extra"]
    ckpt_coil_version = extra.get("coil_model_version")
    if ckpt_coil_version != COIL_MODEL_VERSION:
        raise RuntimeError(
            f"checkpoint trained under coil_model_version={ckpt_coil_version!r} "
            f"but the installed operator is {COIL_MODEL_VERSION!r} — retrain "
            "before gating (physics mismatch)"
        )
    # geometry_table_version is not gate-critical (documented as a single
    # flux-loop-channel effect on one signature) -- label honestly, never fail
    ckpt_geometry_version = extra.get("geometry_table_version")
    geometry_version_label = ckpt_geometry_version or "pre-fix (absent from checkpoint)"
    if ckpt_geometry_version != GEOMETRY_TABLE_VERSION:
        logger.warning(
            "checkpoint geometry_table_version=%r != installed %r — gate "
            "numbers below are honest but predate the sensor-channel-set fix",
            geometry_version_label,
            GEOMETRY_TABLE_VERSION,
        )
    cfg_train = extra["config"]
    feature_stats = extra["feature_stats"]
    n_cells = int(extra["n_cells"])
    ref_signature = extra.get("reference_signature")
    per_sig_n_coil = extra.get("per_signature_n_coil", {})
    n_closure_bins = int(cfg_train.get("n_closure_bins", 0))

    encoder = HybridLatentEncoder(
        LatentConfig(
            n_features=int(feature_stats.mean.shape[0]),
            n_theta=1,
            n_anchored=2,
            n_free=int(cfg_train["n_free"]),
            n_cells=n_cells,
            n_closure_bins=n_closure_bins,
            hidden=int(cfg_train["hidden"]),
            depth=int(cfg_train["depth"]),
        )
    ).double()
    encoder.load_state_dict(payload["encoder"])
    encoder.eval()
    encoder.to(device)

    _train_shots, held_shots = read_split_shot_lists(args.n_train, args.n_heldout)
    baseline_vec = train_mean_baseline(
        args.n_train, args.n_baseline_shots, args.min_ip_ka
    )
    logger.info(
        "baseline (train-mean) axis: (%.3f, %.3f)", baseline_vec[0], baseline_vec[1]
    )

    schema = feature_schema()
    model_rows, ref_rows, is_ref_rows, sig_rows = [], [], [], []
    enc_misfits, shuf_misfits = [], []
    closure_rows_all = []
    per_shot_signature: dict[str, str] = {}
    n_candidate_theoretical = args.n_heldout * args.max_slices_per_shot
    n_loaded, n_scored = 0, 0

    for s in held_shots:
        try:
            gp = shot_payloads(
                s,
                nr=args.nr,
                nz=args.nz,
                max_slices=args.max_slices_per_shot,
                min_ip_ka=args.min_ip_ka,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("shot %s failed to load: %s", s, exc)
            continue
        if gp is None:
            continue
        n_loaded += len(gp["payloads"])
        table, grid, basis = gp["table"], gp["grid"], gp["basis"]
        payloads, refs = gp["payloads"], gp["refs"]
        sig_key = table.signature.key
        per_shot_signature[str(int(s))] = sig_key
        is_ref = sig_key == ref_signature

        if int(basis.r_cells.shape[0]) != n_cells:
            logger.warning(
                "shot %s (sig %s): n_cells %d != checkpoint %d — skipped",
                s,
                sig_key,
                int(basis.r_cells.shape[0]),
                n_cells,
            )
            continue

        fwd = build_operator(table)
        w = load_shot_windows(int(s), fwd, "eval", schema, with_referee=False)
        x_rows = np.stack([w.features_raw[p.t_index] for p in payloads])
        x_norm = np.nan_to_num(feature_stats.normalise(x_rows), nan=0.0)
        ip = np.array([p.ip_amperes for p in payloads])

        basis_d = basis.double().to(device)
        with torch.no_grad():
            lat = encoder(torch.as_tensor(x_norm, dtype=torch.float64, device=device))
            if lat.i_cell_x is None:
                raise RuntimeError("checkpoint's encoder carries no patch-current head")
            i_cell = _i_cell_from_latent(
                lat.i_cell_x,
                torch.as_tensor(ip, dtype=torch.float64, device=device),
                basis_d.candidate_mask,
            )
        i_cell_np = i_cell.detach().cpu().numpy()
        pred_a = pred_b = None
        if lat.closure is not None:
            closure_np = lat.closure.detach().cpu().numpy()
            pred_a, pred_b = closure_np[:, :, 0], closure_np[:, :, 1]

        perm = rng.permutation(len(payloads))
        for k, p in enumerate(payloads):
            psi2d = basis_d.psi_grid_2d_np(i_cell_np[k], p.i_pf)
            target, _pax, _pb = geometry_target(psi2d, grid)
            model_rows.append(target)
            ref_rows.append(refs[k])
            is_ref_rows.append(is_ref)
            sig_rows.append(sig_key)
            enc_misfits.append(_whitened_misfit(basis_d, i_cell_np[k], p))
            shuf_misfits.append(_whitened_misfit(basis_d, i_cell_np[perm[k]], p))
            n_scored += 1

        if pred_a is not None:
            flat_idx = _flattop_rows(w, payloads, frac=args.flattop_frac)
            closure_rows_all.extend(
                _closure_gate_for_shot(
                    basis_d,
                    grid,
                    payloads,
                    flat_idx,
                    pred_a,
                    pred_b,
                    n_closure_bins,
                    device,
                )
            )
        logger.info(
            "shot %s (%s): %d/%d slices scored",
            s,
            "in-sig" if is_ref else "cross-sig",
            len(payloads),
            len(payloads),
        )

    if not model_rows:
        logger.error("no slices scored — aborting")
        return 1

    model = np.asarray(model_rows)
    ref = np.asarray(ref_rows)
    is_ref_arr = np.asarray(is_ref_rows, dtype=bool)
    enc_arr = np.asarray(enc_misfits)
    shuf_arr = np.asarray(shuf_misfits)
    axis_errors = np.hypot(model[:, 0] - ref[:, 0], model[:, 1] - ref[:, 1])

    overall = _bundle(
        model, ref, baseline_vec, enc_arr, shuf_arr, np.ones(len(model), dtype=bool)
    )

    # ---- closure gate summary ----
    closure_summary = None
    if closure_rows_all:
        keep = np.array(
            [r["keep"] for r in closure_rows_all], dtype=bool
        )  # (N, n_bins)
        within_a = np.array(
            [r["within_1sigma_a"] for r in closure_rows_all], dtype=bool
        )
        within_b = np.array(
            [r["within_1sigma_b"] for r in closure_rows_all], dtype=bool
        )
        per_bin_frac_a = [
            float(within_a[:, j][keep[:, j]].mean()) if keep[:, j].any() else None
            for j in range(keep.shape[1])
        ]
        per_bin_frac_b = [
            float(within_b[:, j][keep[:, j]].mean()) if keep[:, j].any() else None
            for j in range(keep.shape[1])
        ]
        pooled_a = float(within_a[keep].mean()) if keep.any() else None
        pooled_b = float(within_b[keep].mean()) if keep.any() else None
        f2_flags = [
            r["f2_nonneg"] for r in closure_rows_all if r["f2_nonneg"] is not None
        ]
        closure_summary = {
            "n_flattop_slices": len(closure_rows_all),
            "per_bin_within_1sigma_a": per_bin_frac_a,
            "per_bin_within_1sigma_b": per_bin_frac_b,
            "pooled_within_1sigma_a": pooled_a,
            "pooled_within_1sigma_b": pooled_b,
            "f2_nonneg_rate": float(np.mean(f2_flags)) if f2_flags else None,
        }
    else:
        logger.warning(
            "no closure comparisons produced (no closure head or no flat-top slices)"
        )

    transport_checks = _transport_structural_checks(
        payload.get("transport", {}), int(cfg_train["n_free"]), per_sig_n_coil
    )

    result = {
        "checkpoint": str(args.checkpoint),
        "device": device,
        "coil_model_version": COIL_MODEL_VERSION,
        "geometry_table_version": geometry_version_label,
        "geometry_table_version_installed": GEOMETRY_TABLE_VERSION,
        "reference_signature": ref_signature,
        "per_shot_signature": per_shot_signature,
        "n_scored": int(len(model)),
        "n_candidate_loaded": int(n_loaded),
        "n_candidate_theoretical": int(n_candidate_theoretical),
        "scored_fraction_of_theoretical": float(
            len(model) / max(n_candidate_theoretical, 1)
        ),
        "baseline_axis": [float(baseline_vec[0]), float(baseline_vec[1])],
        **overall,
        "in_signature": _bundle(
            model, ref, baseline_vec, enc_arr, shuf_arr, is_ref_arr
        ),
        "cross_signature": _bundle(
            model, ref, baseline_vec, enc_arr, shuf_arr, ~is_ref_arr
        ),
        "closure_gate": closure_summary,
        "transport_structural_checks": transport_checks,
    }
    out_path = Path(args.out) if args.out else (ARTIFACTS / f"gate_{args.run}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    np.savez(
        out_path.with_suffix("").with_name(out_path.stem + "_arrays.npz"),
        model=model,
        ref=ref,
        baseline=np.tile(baseline_vec, (len(model), 1)),
        axis_errors=axis_errors,
        is_reference_signature=is_ref_arr,
        signature=np.asarray(sig_rows),
    )
    logger.info(
        "[grounded-latent gate] scored %d/%d (theoretical)  axis_skill=%s  "
        "median %.3f m  grounding %.2fx  oracle_bar=%s",
        len(model),
        n_candidate_theoretical,
        overall["axis_skill"],
        overall["axis_error_median_m"],
        overall["grounding"]["ratio"],
        ORACLE_BAR,
    )
    if closure_summary is not None:
        logger.info(
            "  closure gate: %d flat-top slices, pooled within-1sigma a=%s b=%s, "
            "F^2>=0 rate=%s",
            closure_summary["n_flattop_slices"],
            closure_summary["pooled_within_1sigma_a"],
            closure_summary["pooled_within_1sigma_b"],
            closure_summary["f2_nonneg_rate"],
        )
    logger.info("  transport structural checks: %s", transport_checks)
    logger.info("gate artifact -> %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
