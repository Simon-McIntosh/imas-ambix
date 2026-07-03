#!/usr/bin/env python
"""Held-out gate-2 evaluation of the trained amortised patch-current encoder.

Runs the EXACT gate-2 protocol of ``scripts/patch_gate_eval.py`` (same held-out
shots, slice selection, train-mean baseline, and per-quantity skill formulas —
imported verbatim) but replaces the per-slice variational inverse with a SINGLE
encoder forward per slice.  For every gate slice, a 12-step token window is built
around the slice time exactly as in training, standardised with the checkpoint's
per-channel corpus stats, and pushed through the encoder to a patch-current
vector; the assembled ψ is read for axis / X-point / LCFS geometry with the same
``geometry_target`` used by the Picard and variational gates.

Also reports the GROUNDING ratio: the mean whitened sensor misfit of the
encoder's prediction against a paired SHUFFLED-current control (each slice scored
with its own measured / vacuum / mask / scale) — the ≥5.53× discrimination bar.

Artifacts: imas_ambix/latent/artifacts/patch_gate/encoder_gate.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from imas_ambix.gs.geometry import build_table_for_shot
from imas_ambix.gs.operator import build_operator
from imas_ambix.latent.data import (
    feature_schema,
    load_shot_windows,
    read_split_shot_lists,
)
from imas_ambix.latent.patch_encoder import PatchCurrentEncoder, PatchEncoderConfig

# reuse the gate-2 protocol verbatim
from scripts.patch_gate_eval import (
    TARGET_NAMES,  # noqa: F401  (kept for parity / downstream imports)
    geometry_target,
    score,
    shot_payloads,
    train_mean_baseline,
)
from scripts.train_patch_encoder import (
    _basis_alignment,
    _standardise_values,
    _window_indices,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("patch_encoder_gate_eval")

ARTIFACTS = Path("imas_ambix/latent/artifacts/patch_gate")


def _load_encoder(ckpt_path: Path, device: str):
    """Rebuild the encoder from its checkpoint (config + reference geometry)."""
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    extra = payload["extra"]
    cfg = PatchEncoderConfig(**extra["encoder_config"])
    encoder = PatchCurrentEncoder(
        cfg,
        sensor_geometry=np.asarray(extra["sensor_geometry"]),
        coil_centroids=np.asarray(extra["coil_centroids"]),
        candidate_mask=np.asarray(extra["candidate_mask"]),
    ).to(device)
    encoder.load_state_dict(payload["encoder"])
    encoder.eval()
    return encoder, extra


def _build_slice_windows(shot, payloads, channels, extra):
    """12-step token windows for the gate slices of one shot (basis channel order).

    Windows are centred on each gate slice's time using the same shot's full
    magnetics stream — identical slice selection to ``shot_payloads`` (the
    payloads carry ``t_index`` / ``time_s``).  Returns (values_std, finite,
    valid) where ``valid`` masks slices whose full 0.25 s window does not fit.
    """
    fwd = build_operator(build_table_for_shot(int(shot)))
    w = load_shot_windows(int(shot), fwd, "eval", feature_schema(), with_referee=False)
    ch_rows, present = _basis_alignment(fwd, channels)
    clip_rows = np.clip(ch_rows, 0, None)
    raw_b = np.where(present, w.raw_mag[:, clip_rows], np.nan)
    mask_b = present & w.mag_mask[:, clip_rows]
    times = np.asarray(w.times, dtype=np.float64)
    t_steps = int(extra.get("t_steps", 12))

    values, finite, valid = [], [], []
    for p in payloads:
        # locate the payload's slice on this shot's stream by its recorded time
        c = int(np.argmin(np.abs(times - p.time_s)))
        idx = _window_indices(times, float(times[c]), t_steps)
        if idx is None:
            values.append(np.zeros((t_steps, len(channels))))
            finite.append(np.zeros((t_steps, len(channels)), dtype=bool))
            valid.append(False)
            continue
        values.append(raw_b[idx])
        finite.append(mask_b[idx])
        valid.append(True)
    values = np.asarray(values, dtype=np.float64)
    finite = np.asarray(finite, dtype=bool)
    values_std = _standardise_values(
        values,
        finite,
        np.asarray(extra["channel_mean"]),
        np.asarray(extra["channel_std"]),
    )
    return values_std, finite, np.asarray(valid, dtype=bool)


def _whitened_misfit(basis, i_cell, payload):
    """Masked whitened mean-square sensor misfit of ``i_cell`` for one slice."""
    m_sens = basis.m_sens.to(torch.float64).cpu().numpy()
    pred = np.asarray(payload.vacuum) + m_sens @ np.asarray(i_cell, dtype=np.float64)
    resid = np.where(payload.mask, (pred - payload.measured) / payload.scale, 0.0)
    denom = max(float(payload.mask.sum()), 1.0)
    return float(np.sum(resid**2) / denom)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--n-train", type=int, default=40)
    ap.add_argument("--n-heldout", type=int, default=8)
    ap.add_argument("--nr", type=int, default=65)
    ap.add_argument("--nz", type=int, default=97)
    ap.add_argument("--max-slices-per-shot", type=int, default=20)
    ap.add_argument("--min-ip-ka", type=float, default=300.0)
    ap.add_argument("--n-baseline-shots", type=int, default=10)
    ap.add_argument("--out", type=str, default=str(ARTIFACTS / "encoder_gate.json"))
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

    encoder, extra = _load_encoder(Path(args.checkpoint), device)
    ref_channels = list(extra["sensor_channels"])
    ipf_mean = np.asarray(extra["ipf_mean"])
    ipf_std = np.asarray(extra["ipf_std"])

    _train_shots, held_shots = read_split_shot_lists(args.n_train, args.n_heldout)
    baseline_vec = train_mean_baseline(
        args.n_train, args.n_baseline_shots, args.min_ip_ka
    )
    logger.info(
        "baseline (train-mean) axis: (%.3f, %.3f)", baseline_vec[0], baseline_vec[1]
    )

    model_rows, ref_rows = [], []
    enc_misfits, shuf_misfits = [], []
    n_candidate = 0
    for s in held_shots:
        try:
            payload = shot_payloads(
                s,
                nr=args.nr,
                nz=args.nz,
                max_slices=args.max_slices_per_shot,
                min_ip_ka=args.min_ip_ka,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("shot %s failed to load: %s", s, exc)
            continue
        if payload is None:
            continue
        basis = payload["basis"]
        grid = payload["grid"]
        payloads = payload["payloads"]
        refs = payload["refs"]
        # basis channels must match the encoder's reference geometry
        if basis.sensor_channels != ref_channels or (
            int(basis.candidate_mask.shape[0]) != int(extra["n_cells"])
        ):
            logger.warning(
                "shot %s geometry mismatch vs checkpoint reference — skipped", s
            )
            continue
        n_candidate += len(payloads)

        values_std, finite, valid = _build_slice_windows(
            s, payloads, ref_channels, extra
        )
        i_pf = np.asarray([p.i_pf for p in payloads], dtype=np.float64)
        i_pf_std = (i_pf - ipf_mean[None, :]) / ipf_std[None, :]
        ip = np.asarray([p.ip_amperes for p in payloads], dtype=np.float64)

        with torch.no_grad():
            i_cell = encoder(
                torch.as_tensor(values_std, dtype=torch.float32, device=device),
                torch.as_tensor(finite, dtype=torch.bool, device=device),
                torch.as_tensor(i_pf_std, dtype=torch.float32, device=device),
                torch.as_tensor(ip, dtype=torch.float32, device=device),
            )
        i_cell = np.asarray(i_cell.detach().cpu().numpy(), dtype=np.float64)

        # paired shuffled-current control (permute the slice→current assignment)
        perm = rng.permutation(len(payloads))
        for k, p in enumerate(payloads):
            if not valid[k]:
                continue
            psi2d = basis.psi_grid_2d_np(i_cell[k], p.i_pf)
            target, _pax, _pb = geometry_target(psi2d, grid)
            model_rows.append(target)
            ref_rows.append(refs[k])
            enc_misfits.append(_whitened_misfit(basis, i_cell[k], p))
            shuf_misfits.append(_whitened_misfit(basis, i_cell[perm[k]], p))
        logger.info("shot %s: %d/%d slices scored", s, int(valid.sum()), len(payloads))

    if not model_rows:
        logger.error("no slices scored — aborting")
        return 1

    model = np.asarray(model_rows)
    ref = np.asarray(ref_rows)
    sc = score(model, ref, baseline_vec)
    axis_errors = sc.pop("axis_errors")
    enc_m = float(np.mean(enc_misfits))
    shuf_m = float(np.mean(shuf_misfits))
    grounding_ratio = shuf_m / enc_m if enc_m > 0 else float("inf")

    result = {
        "checkpoint": str(args.checkpoint),
        "device": device,
        "head": extra["config"].get("head"),
        "reference_signature": extra.get("reference_signature"),
        "n_scored": int(len(model)),
        "n_candidate": int(n_candidate),
        "scored_fraction": float(len(model) / max(n_candidate, 1)),
        "baseline_axis": [float(baseline_vec[0]), float(baseline_vec[1])],
        **sc,
        "grounding": {
            "encoder_misfit_mean": enc_m,
            "shuffled_misfit_mean": shuf_m,
            "ratio": grounding_ratio,
            "bar": 5.53,
            "pass": bool(grounding_ratio >= 5.53),
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    np.savez(
        ARTIFACTS / "encoder_gate_arrays.npz",
        model=model,
        ref=ref,
        baseline=np.tile(baseline_vec, (len(model), 1)),
        axis_errors=axis_errors,
    )
    logger.info(
        "[encoder gate] scored %d/%d  axis_skill=%s  median %.3f m  grounding %.2fx",
        len(model),
        n_candidate,
        result["axis_skill"],
        result["axis_error_median_m"],
        grounding_ratio,
    )
    logger.info("gate artifact -> %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
