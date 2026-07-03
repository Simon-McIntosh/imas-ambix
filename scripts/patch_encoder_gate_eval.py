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

Cross-signature evaluation is a REQUIRED capability (machine-agnostic North
Star): held-out shots from a campaign whose sensor set differs from the training
reference are scored, not skipped.  The encoder is set-based and consumes each
sensor's geometry as an additive positional encoding, so a fresh encoder is
built with the eval signature's own sensor/coil geometry and the trained learned
weights are loaded into it (the plasma patch substrate — ``n_cells`` and the
per-cell head — is identical across signatures).  Token values are standardised
per channel BY NAME using the checkpoint's stored per-channel stats, with a
per-sensor-kind median fallback for channels the checkpoint never saw.  Each
scored slice is labelled with its signature so in-signature and cross-signature
skills are reported separately.

Edge slices whose full 0.25 s window overhangs the recorded stream are PADDED
(masked-absent steps), so every gate slice is scored — no coverage loss.

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
    coil_centroids_array,
    sensor_geometry_array,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("patch_encoder_gate_eval")

ARTIFACTS = Path("imas_ambix/latent/artifacts/patch_gate")

#: geometry buffers rebuilt per eval signature — never loaded from the checkpoint
#: (they carry the TRAINING reference's geometry; the learned weights are what
#: transfer across signatures).
_GEOMETRY_BUFFERS = frozenset(
    {"sensor_geom", "sensor_kind", "coil_geom", "coil_kind", "candidate_mask"}
)


def _load_checkpoint(ckpt_path: Path, device: str):
    """Return the trained ``state_dict`` and the checkpoint ``extra`` block."""
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    return payload["encoder"], payload["extra"]


def _resolve_channel_stats(channels, sensor_geom_eval, extra):
    """Per-channel (mean, std) for ``channels``, aligned BY NAME to the checkpoint.

    Channels the checkpoint saw reuse their stored standardisation stat; channels
    it never saw (a novel probe in a cross-signature campaign) fall back to the
    median stat of the training channels of the SAME sensor kind (a global median
    if the kind is unseen too), so a novel flux loop is standardised on a
    flux-loop scale rather than a B-probe scale.  Returns (mean, std, n_fallback).
    """
    ref_channels = list(extra["sensor_channels"])
    ref_mean = np.asarray(extra["channel_mean"], dtype=np.float64)
    ref_std = np.asarray(extra["channel_std"], dtype=np.float64)
    ref_geom = np.asarray(extra["sensor_geometry"], dtype=np.float64)
    ref_kind = (
        ref_geom[:, 4].astype(int)
        if ref_geom.shape[1] > 4
        else np.zeros(len(ref_channels), dtype=int)
    )
    by_name = {ch: (ref_mean[i], ref_std[i]) for i, ch in enumerate(ref_channels)}
    kind_mean, kind_std = {}, {}
    for k in np.unique(ref_kind):
        sel = ref_kind == k
        kind_mean[int(k)] = float(np.median(ref_mean[sel]))
        kind_std[int(k)] = float(np.median(ref_std[sel]))
    glob_mean = float(np.median(ref_mean))
    glob_std = float(np.median(ref_std))
    ev_kind = np.asarray(sensor_geom_eval, dtype=np.float64)[:, 4].astype(int)

    means, stds, n_fallback = [], [], 0
    for j, ch in enumerate(channels):
        if ch in by_name:
            m, s = by_name[ch]
        else:
            k = int(ev_kind[j])
            m = kind_mean.get(k, glob_mean)
            s = kind_std.get(k, glob_std)
            n_fallback += 1
        means.append(float(m))
        stds.append(float(s) if s > 0 else 1.0)
    return np.asarray(means), np.asarray(stds), n_fallback


def _encoder_for_signature(
    state_dict, extra, table, fwd, channels, candidate_mask, device
):
    """Build an encoder on this signature's geometry, load the trained weights.

    The geometry buffers are this signature's own (sensor + coil positions); the
    learned weights (value/geometry/kind embeddings, transformer, pool, per-cell
    head) come from the checkpoint.  ``n_cells`` is identical across signatures,
    so the head loads without shape mismatch; a real mismatch would raise.
    """
    cfg = PatchEncoderConfig(**extra["encoder_config"])
    sensor_geom = sensor_geometry_array(table, list(channels))
    encoder = PatchCurrentEncoder(
        cfg,
        sensor_geometry=sensor_geom,
        coil_centroids=coil_centroids_array(table, fwd),
        candidate_mask=np.asarray(candidate_mask, dtype=np.float64),
    ).to(device)
    learned = {k: v for k, v in state_dict.items() if k not in _GEOMETRY_BUFFERS}
    missing, unexpected = encoder.load_state_dict(learned, strict=False)
    missing = [m for m in missing if m not in _GEOMETRY_BUFFERS]
    if missing or unexpected:
        raise RuntimeError(
            f"encoder weight load mismatch: missing={missing} unexpected={unexpected}"
        )
    encoder.eval()
    ch_mean, ch_std, n_fallback = _resolve_channel_stats(channels, sensor_geom, extra)
    return encoder, ch_mean, ch_std, n_fallback


def _build_slice_windows(shot, fwd, payloads, channels, ch_mean, ch_std, extra):
    """12-step token windows for the gate slices of one shot (basis channel order).

    Windows are centred on each gate slice's time using the same shot's full
    magnetics stream — identical slice selection to ``shot_payloads``.  Edge
    slices whose window overhangs the stream are PADDED (masked-absent steps), so
    every payload yields a window.  Values are standardised with the per-channel
    stats resolved for this shot's signature.
    """
    w = load_shot_windows(int(shot), fwd, "eval", feature_schema(), with_referee=False)
    ch_rows, present = _basis_alignment(fwd, channels)
    clip_rows = np.clip(ch_rows, 0, None)
    raw_b = np.where(present, w.raw_mag[:, clip_rows], np.nan)
    mask_b = present & w.mag_mask[:, clip_rows]
    times = np.asarray(w.times, dtype=np.float64)
    t_steps = int(extra.get("t_steps", 12))

    values, finite = [], []
    for p in payloads:
        # locate the payload's slice on this shot's stream by its recorded time
        c = int(np.argmin(np.abs(times - p.time_s)))
        res = _window_indices(times, float(times[c]), t_steps, min_real=1)
        if res is None:  # the centre itself is in-stream, so this cannot happen
            raise RuntimeError(f"shot {shot}: slice {p.t_index} has no in-stream steps")
        idx, real = res
        real2 = real[:, None]
        values.append(np.where(real2, raw_b[idx], 0.0))
        finite.append(mask_b[idx] & real2)
    values = np.asarray(values, dtype=np.float64)
    finite = np.asarray(finite, dtype=bool)
    values_std = _standardise_values(values, finite, ch_mean, ch_std)
    return values_std, finite


def _whitened_misfit(basis, i_cell, payload):
    """Masked whitened mean-square sensor misfit of ``i_cell`` for one slice."""
    m_sens = basis.m_sens.to(torch.float64).cpu().numpy()
    pred = np.asarray(payload.vacuum) + m_sens @ np.asarray(i_cell, dtype=np.float64)
    resid = np.where(payload.mask, (pred - payload.measured) / payload.scale, 0.0)
    denom = max(float(payload.mask.sum()), 1.0)
    return float(np.sum(resid**2) / denom)


def _grounding(enc_misfits, shuf_misfits):
    enc_m = float(np.mean(enc_misfits))
    shuf_m = float(np.mean(shuf_misfits))
    ratio = shuf_m / enc_m if enc_m > 0 else float("inf")
    return {
        "encoder_misfit_mean": enc_m,
        "shuffled_misfit_mean": shuf_m,
        "ratio": ratio,
        "bar": 5.53,
        "pass": bool(ratio >= 5.53),
    }


def _bundle(model, ref, baseline_vec, enc_arr, shuf_arr, sel):
    """Skill + grounding for the subset ``sel`` (None if the subset is empty)."""
    if int(sel.sum()) == 0:
        return None
    sc = score(model[sel], ref[sel], baseline_vec)
    sc.pop("axis_errors")
    sc["n_scored"] = int(sel.sum())
    sc["grounding"] = _grounding(enc_arr[sel], shuf_arr[sel])
    return sc


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

    state_dict, extra = _load_checkpoint(Path(args.checkpoint), device)
    ref_signature = extra.get("reference_signature")
    ipf_mean = np.asarray(extra["ipf_mean"])
    ipf_std = np.asarray(extra["ipf_std"])

    _train_shots, held_shots = read_split_shot_lists(args.n_train, args.n_heldout)
    baseline_vec = train_mean_baseline(
        args.n_train, args.n_baseline_shots, args.min_ip_ka
    )
    logger.info(
        "baseline (train-mean) axis: (%.3f, %.3f)", baseline_vec[0], baseline_vec[1]
    )

    # one encoder per signature (geometry rebuilt, learned weights loaded)
    enc_cache: dict = {}
    model_rows, ref_rows, is_ref_rows, sig_rows = [], [], [], []
    enc_misfits, shuf_misfits = [], []
    per_shot_signature: dict[str, str] = {}
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

        table = build_table_for_shot(int(s))
        fwd = build_operator(table)
        sig_key = table.signature.key
        is_ref = sig_key == ref_signature
        per_shot_signature[str(int(s))] = sig_key
        channels = list(basis.sensor_channels)
        candidate_mask = np.asarray(
            basis.candidate_mask.detach().cpu().numpy(), dtype=np.float64
        )
        if int(candidate_mask.shape[0]) != int(extra["n_cells"]):
            logger.warning(
                "shot %s n_cells %d != checkpoint %d — skipped (incompatible head)",
                s,
                int(candidate_mask.shape[0]),
                int(extra["n_cells"]),
            )
            continue

        if sig_key not in enc_cache:
            enc_cache[sig_key] = _encoder_for_signature(
                state_dict, extra, table, fwd, channels, candidate_mask, device
            )
            logger.info(
                "signature %s (%s): encoder built, S=%d, %d channels use a "
                "kind-median fallback stat",
                sig_key[-16:],
                "reference" if is_ref else "cross",
                len(channels),
                enc_cache[sig_key][3],
            )
        encoder, ch_mean, ch_std, _nf = enc_cache[sig_key]
        n_candidate += len(payloads)

        values_std, finite = _build_slice_windows(
            s, fwd, payloads, channels, ch_mean, ch_std, extra
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
            psi2d = basis.psi_grid_2d_np(i_cell[k], p.i_pf)
            target, _pax, _pb = geometry_target(psi2d, grid)
            model_rows.append(target)
            ref_rows.append(refs[k])
            is_ref_rows.append(is_ref)
            sig_rows.append(sig_key)
            enc_misfits.append(_whitened_misfit(basis, i_cell[k], p))
            shuf_misfits.append(_whitened_misfit(basis, i_cell[perm[k]], p))
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

    overall = _bundle(
        model, ref, baseline_vec, enc_arr, shuf_arr, np.ones(len(model), dtype=bool)
    )
    axis_errors = np.hypot(model[:, 0] - ref[:, 0], model[:, 1] - ref[:, 1])

    result = {
        "checkpoint": str(args.checkpoint),
        "device": device,
        "head": extra["config"].get("head"),
        "reference_signature": ref_signature,
        "per_shot_signature": per_shot_signature,
        "n_scored": int(len(model)),
        "n_candidate": int(n_candidate),
        "scored_fraction": float(len(model) / max(n_candidate, 1)),
        "baseline_axis": [float(baseline_vec[0]), float(baseline_vec[1])],
        **overall,
        "in_signature": _bundle(
            model, ref, baseline_vec, enc_arr, shuf_arr, is_ref_arr
        ),
        "cross_signature": _bundle(
            model, ref, baseline_vec, enc_arr, shuf_arr, ~is_ref_arr
        ),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    np.savez(
        ARTIFACTS / "encoder_gate_arrays.npz",
        model=model,
        ref=ref,
        baseline=np.tile(baseline_vec, (len(model), 1)),
        axis_errors=axis_errors,
        is_reference_signature=is_ref_arr,
        signature=np.asarray(sig_rows),
    )
    logger.info(
        "[encoder gate] scored %d/%d  axis_skill=%s  median %.3f m  grounding %.2fx",
        len(model),
        n_candidate,
        overall["axis_skill"],
        overall["axis_error_median_m"],
        overall["grounding"]["ratio"],
    )
    for name, block in (
        ("in-signature", result["in_signature"]),
        ("cross-signature", result["cross_signature"]),
    ):
        if block is not None:
            logger.info(
                "  %s: %d slices  axis_skill=%s  median %.3f m  grounding %.2fx",
                name,
                block["n_scored"],
                block["axis_skill"],
                block["axis_error_median_m"],
                block["grounding"]["ratio"],
            )
    logger.info("gate artifact -> %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
