#!/usr/bin/env python
"""Per-quantity calibration of the Gaussian ("gaussian-direct") patch head.

Runs the SAME held-out shots, slice selection, and encoder-eval machinery as
``scripts/patch_encoder_gate_eval.py`` (imported, never duplicated) but reads
the checkpoint's variance arm alongside its mean and asks the distributional
question the deterministic gate cannot: is the encoder's uncertainty honest?

Three measurements:

(a) SENSOR coverage — exact, no sampling.  The per-cell current Gaussian
    (mean ``i_mean``, diagonal variance ``i_var``) propagates linearly through
    the sensor forward (``imas_ambix.latent.patch_encoder.amortised_losses``'s
    ``pred_var = i_var @ (m_sens**2).T``), so the 90% interval on each held
    sensor is analytic.  Coverage (the fraction of masked sensors whose
    MEASURED value lands inside its own predicted interval) is reported per
    sensor KIND (b-probe / flux-loop / ...).

(b) TOPOLOGY coverage — sampled.  ``axis_R``/``axis_Z``/the X-point cannot be
    read from a sensor-space interval (the ψ-field topology read is
    nonlinear), so ``K`` current draws ``~ N(i_mean, diag(i_var))`` per slice
    are pushed through the SAME ``geometry_target`` the deterministic gate
    uses; the empirical [5th, 95th] percentile band over the draws is checked
    against the firewalled EFIT referee.  The referee is read ONLY here, for
    SCORING — never inside a gradient, identical discipline to
    ``patch_encoder_gate_eval.py``.

(c) MEAN-READOUT regression — the encoder's mean (``i_mean`` alone, exactly
    the value a "direct" head would emit) is scored with the deterministic
    gate's own skill formula (``scripts.patch_gate_eval.score``) and compared
    against a ``--baseline-gate`` JSON from a deterministic run on the SAME
    corpus.  The Gaussian head must not cost mean-readout skill.

Gate: per-quantity / per-sensor-kind coverage in [0.88, 0.92]; no mean-skill
regression beyond ``--regression-tolerance``.

Artifacts: imas_ambix/latent/artifacts/patch_gate/calibration_<run>.json
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
from imas_ambix.latent.data import read_split_shot_lists
from imas_ambix.latent.patch_basis import PatchBasis
from imas_ambix.latent.patch_encoder import PATCH_SENSOR_KINDS

from scripts.patch_encoder_gate_eval import (
    _build_slice_windows,
    _encoder_for_signature,
    _load_checkpoint,
)
from scripts.patch_gate_eval import (
    TARGET_NAMES,
    geometry_target,
    score,
    shot_payloads,
    train_mean_baseline,
)
from scripts.train_patch_encoder import sensor_geometry_array

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("patch_encoder_calibration_eval")

ARTIFACTS = Path("imas_ambix/latent/artifacts/patch_gate")

#: 90% two-sided normal coverage z-score (Φ⁻¹(0.95)) — the exact sensor arm.
Z90 = 1.6448536269514722

#: topology quantities scored by sampling (a subset of
#: scripts.patch_gate_eval.TARGET_NAMES — axis + the primary X-point).
TOPOLOGY_QUANTITIES: tuple[str, ...] = ("axis_R", "axis_Z", "xpt0_R", "xpt0_Z")

#: coverage gate every quantity/sensor-kind must land inside.
COVERAGE_GATE: tuple[float, float] = (0.88, 0.92)


# --------------------------------------------------------------------------- #
#  (a) exact sensor coverage                                                   #
# --------------------------------------------------------------------------- #
def sensor_coverage_counts(
    basis: PatchBasis,
    i_mean: np.ndarray,  # (n,) current mean [A]
    i_var: np.ndarray,  # (n,) current variance [A^2]
    payload,  # SlicePayload (imas_ambix.latent.patch_inverse)
    sensor_kind: np.ndarray,  # (S,) int — PATCH_SENSOR_KINDS index per basis column
    *,
    z: float = Z90,
) -> dict[int, tuple[int, int]]:
    """Per-sensor-kind ``(n_covered, n_masked)`` for ONE slice's held sensors.

    Coverage is EXACT — no sampling.  The predicted sensor mean/variance
    propagate linearly from the diagonal current Gaussian exactly as
    ``amortised_losses`` computes them; the 90% interval is
    ``pred ± z·sqrt(pred_var)`` in the slice's own (unwhitened) units, which
    is equivalent to whitening both sides by the per-sensor scale — coverage
    is scale-invariant.
    """
    m_sens = basis.m_sens.to(torch.float64).cpu().numpy()  # (S, n)
    pred = np.asarray(payload.vacuum, dtype=np.float64) + m_sens @ i_mean
    pred_var = (m_sens**2) @ i_var
    half_width = z * np.sqrt(np.clip(pred_var, 0.0, None))
    lo, hi = pred - half_width, pred + half_width
    measured = np.asarray(payload.measured, dtype=np.float64)
    covered = (measured >= lo) & (measured <= hi)
    mask = np.asarray(payload.mask, dtype=bool)

    out: dict[int, tuple[int, int]] = {}
    for k in np.unique(sensor_kind):
        sel = mask & (sensor_kind == k)
        n = int(sel.sum())
        if n == 0:
            continue
        out[int(k)] = (int(covered[sel].sum()), n)
    return out


# --------------------------------------------------------------------------- #
#  (b) sampled topology coverage                                               #
# --------------------------------------------------------------------------- #
def sample_topology_targets(
    basis: PatchBasis,
    grid,
    i_mean: np.ndarray,
    i_var: np.ndarray,
    i_pf: np.ndarray,
    *,
    k_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """``(K, 14)`` geometry targets read from ``K`` current draws ``~
    N(i_mean, diag(i_var))`` — the honest sampled-current topology coverage a
    mean-only readout cannot provide (axis/X-point read nonlinearly off ψ)."""
    n = i_mean.shape[0]
    std = np.sqrt(np.clip(i_var, 0.0, None))
    draws = i_mean[None, :] + std[None, :] * rng.standard_normal((k_samples, n))
    targets = np.empty((k_samples, 14), dtype=np.float64)
    for j in range(k_samples):
        psi2d = basis.psi_grid_2d_np(draws[j], i_pf)
        targets[j], _pax, _pb = geometry_target(psi2d, grid)
    return targets


def topology_coverage(
    samples: np.ndarray,  # (K, 14)
    ref: np.ndarray,  # (14,)
    quantity_idx: dict[str, int],
    *,
    lo_q: float = 0.05,
    hi_q: float = 0.95,
    min_finite_frac: float = 0.5,
) -> dict[str, bool | None]:
    """Per-quantity: does the empirical ``[lo_q, hi_q]`` band over the ``K``
    samples bracket the referee reference?  ``None`` (excluded from the
    coverage denominator) when the referee is non-finite for this slice or
    too many samples failed to place a critical point."""
    out: dict[str, bool | None] = {}
    k = samples.shape[0]
    for name, idx in quantity_idx.items():
        col = samples[:, idx]
        r = ref[idx]
        finite = np.isfinite(col)
        if not np.isfinite(r) or finite.sum() < max(2, int(min_finite_frac * k)):
            out[name] = None
            continue
        lo, hi = np.quantile(col[finite], [lo_q, hi_q])
        out[name] = bool(lo <= r <= hi)
    return out


def coverage_fraction(counts: tuple[int, int]) -> float | None:
    covered, n = counts
    return (covered / n) if n > 0 else None


def verdict(frac: float | None, gate: tuple[float, float] = COVERAGE_GATE) -> str:
    if frac is None:
        return "no-data"
    lo, hi = gate
    if frac < lo:
        return "under-covered"
    if frac > hi:
        return "over-covered"
    return "pass"


# --------------------------------------------------------------------------- #
#  driver                                                                      #
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument(
        "--baseline-gate",
        type=str,
        default="",
        help="scripts/patch_encoder_gate_eval.py JSON from a deterministic "
        "('direct' head) run on the SAME corpus — the (c) regression check. "
        "Omit to skip the regression check.",
    )
    ap.add_argument("--n-train", type=int, default=40)
    ap.add_argument("--n-heldout", type=int, default=8)
    ap.add_argument("--nr", type=int, default=65)
    ap.add_argument("--nz", type=int, default=97)
    ap.add_argument("--max-slices-per-shot", type=int, default=20)
    ap.add_argument("--min-ip-ka", type=float, default=300.0)
    ap.add_argument("--n-baseline-shots", type=int, default=10)
    ap.add_argument("--k-samples", type=int, default=100)
    ap.add_argument(
        "--regression-tolerance",
        type=float,
        default=0.02,
        help="max allowed drop in axis_skill vs --baseline-gate before FAIL",
    )
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = (
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else args.device
    )
    rng = np.random.default_rng(args.seed)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(args.checkpoint)
    state_dict, extra = _load_checkpoint(ckpt_path, device)
    head = extra["config"].get("head")
    if head != "gaussian-direct":
        raise ValueError(
            f"calibration requires a gaussian-direct checkpoint; got head={head!r}"
        )
    ipf_mean = np.asarray(extra["ipf_mean"])
    ipf_std = np.asarray(extra["ipf_std"])
    quantity_idx = {
        name: i for i, name in enumerate(TARGET_NAMES) if name in TOPOLOGY_QUANTITIES
    }

    _train_shots, held_shots = read_split_shot_lists(args.n_train, args.n_heldout)
    baseline_vec = train_mean_baseline(
        args.n_train, args.n_baseline_shots, args.min_ip_ka
    )

    enc_cache: dict = {}
    sensor_counts: dict[int, list[int]] = {}
    topo_counts: dict[str, list[int]] = {q: [0, 0] for q in TOPOLOGY_QUANTITIES}
    model_rows, ref_rows = [], []
    n_slices = 0

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
        channels = list(basis.sensor_channels)
        candidate_mask = np.asarray(
            basis.candidate_mask.detach().cpu().numpy(), dtype=np.float64
        )
        if int(candidate_mask.shape[0]) != int(extra["n_cells"]):
            logger.warning(
                "shot %s n_cells %d != checkpoint %d — skipped",
                s,
                int(candidate_mask.shape[0]),
                int(extra["n_cells"]),
            )
            continue
        if sig_key not in enc_cache:
            enc_cache[sig_key] = _encoder_for_signature(
                state_dict, extra, table, fwd, channels, candidate_mask, device
            )
        encoder, ch_mean, ch_std, _nf = enc_cache[sig_key]
        sensor_kind = sensor_geometry_array(table, channels)[:, 4].astype(int)

        values_std, finite = _build_slice_windows(
            s, fwd, payloads, channels, ch_mean, ch_std, extra
        )
        i_pf = np.asarray([p.i_pf for p in payloads], dtype=np.float64)
        i_pf_std = (i_pf - ipf_mean[None, :]) / ipf_std[None, :]
        ip = np.asarray([p.ip_amperes for p in payloads], dtype=np.float64)

        with torch.no_grad():
            i_mean_t, i_var_t = encoder(
                torch.as_tensor(values_std, dtype=torch.float32, device=device),
                torch.as_tensor(finite, dtype=torch.bool, device=device),
                torch.as_tensor(i_pf_std, dtype=torch.float32, device=device),
                torch.as_tensor(ip, dtype=torch.float32, device=device),
                return_variance=True,
            )
        i_mean = np.asarray(i_mean_t.detach().cpu().numpy(), dtype=np.float64)
        i_var = np.asarray(i_var_t.detach().cpu().numpy(), dtype=np.float64)

        for k_i, p in enumerate(payloads):
            n_slices += 1
            for kind, (c, n) in sensor_coverage_counts(
                basis, i_mean[k_i], i_var[k_i], p, sensor_kind
            ).items():
                acc = sensor_counts.setdefault(kind, [0, 0])
                acc[0] += c
                acc[1] += n

            samples = sample_topology_targets(
                basis,
                grid,
                i_mean[k_i],
                i_var[k_i],
                p.i_pf,
                k_samples=args.k_samples,
                rng=rng,
            )
            for q, ok in topology_coverage(samples, refs[k_i], quantity_idx).items():
                if ok is None:
                    continue
                topo_counts[q][0] += int(ok)
                topo_counts[q][1] += 1

            # (c) the mean-only readout, scored with the deterministic gate's
            # own formula — identical to what a "direct" head would emit
            psi2d = basis.psi_grid_2d_np(i_mean[k_i], p.i_pf)
            target, _pax, _pb = geometry_target(psi2d, grid)
            model_rows.append(target)
            ref_rows.append(refs[k_i])

        logger.info("shot %s: %d slices scored", s, len(payloads))

    if not model_rows:
        logger.error("no slices scored — aborting")
        return 1

    sensor_report = {
        PATCH_SENSOR_KINDS[k] if 0 <= k < len(PATCH_SENSOR_KINDS) else str(k): {
            "n_covered": c,
            "n_masked": n,
            "coverage": coverage_fraction((c, n)),
            "verdict": verdict(coverage_fraction((c, n))),
        }
        for k, (c, n) in sensor_counts.items()
    }
    topology_report = {
        q: {
            "n_covered": c,
            "n_scored": n,
            "coverage": coverage_fraction((c, n)),
            "verdict": verdict(coverage_fraction((c, n))),
        }
        for q, (c, n) in topo_counts.items()
    }

    mean_skill = score(np.asarray(model_rows), np.asarray(ref_rows), baseline_vec)
    mean_skill.pop("axis_errors")

    regression = None
    if args.baseline_gate:
        baseline = json.loads(Path(args.baseline_gate).read_text())
        base_axis_skill = baseline.get("axis_skill")
        this_axis_skill = mean_skill.get("axis_skill")
        if base_axis_skill is not None and this_axis_skill is not None:
            delta = float(this_axis_skill) - float(base_axis_skill)
            regression = {
                "baseline_gate": args.baseline_gate,
                "baseline_axis_skill": float(base_axis_skill),
                "gaussian_mean_axis_skill": float(this_axis_skill),
                "delta": delta,
                "tolerance": args.regression_tolerance,
                "pass": bool(delta >= -args.regression_tolerance),
            }
        else:
            logger.warning(
                "baseline-gate or this run missing axis_skill — regression "
                "check skipped"
            )

    # "no-data" (zero observations — e.g. no X-point ever resolved in the
    # held-out cohort) is excluded from the roll-up: it is not evidence of
    # miscalibration, just an untested quantity, and must not force a
    # permanent FAIL for a quantity the corpus never exercises.
    all_verdicts = [
        v["verdict"]
        for v in list(sensor_report.values()) + list(topology_report.values())
        if v["verdict"] != "no-data"
    ]
    overall_pass = (
        bool(all_verdicts)
        and all(v == "pass" for v in all_verdicts)
        and (regression is None or regression["pass"])
    )

    result = {
        "checkpoint": str(ckpt_path),
        "device": device,
        "n_slices_scored": n_slices,
        "coverage_gate": list(COVERAGE_GATE),
        "sensor_coverage": sensor_report,
        "topology_coverage": topology_report,
        "mean_readout_skill": mean_skill,
        "regression_check": regression,
        "overall_pass": overall_pass,
    }

    if args.out:
        out_path = Path(args.out)
    else:
        run_name = ckpt_path.parent.name or "run"
        out_path = ARTIFACTS / f"calibration_{run_name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))

    logger.info(
        "[calibration] %d slices  overall_pass=%s  sensor=%s  topology=%s",
        n_slices,
        overall_pass,
        {k: v["verdict"] for k, v in sensor_report.items()},
        {k: v["verdict"] for k, v in topology_report.items()},
    )
    logger.info("calibration artifact -> %s", out_path)
    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
