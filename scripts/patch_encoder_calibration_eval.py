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

Optional ``--temperature-scale``: a post-hoc, per-SENSOR-KIND σ-temperature
(closed-form quantile fit on TRAIN-shot sensor coverage, frozen before any
held-out read — leakage-free).  A per-kind correction MUST live in sensor
space: every current cell contributes to every sensor with a different
weight, so there is no per-cell (current-space) scaling that independently
widens one sensor kind's coverage while tightening another's.  Consequently
this refinement has NO effect on the (b) topology sampling, which draws
directly from the current-space ``i_var`` — an honest limitation, not a gap
in the implementation (see ``fit_kind_temperature``).

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
from imas_ambix.latent.patch_encoder import (
    PATCH_SENSOR_KINDS,
    PatchCurrentEncoder,
    PatchEncoderConfig,
)

from scripts.patch_encoder_gate_eval import (
    _GEOMETRY_BUFFERS,
    _build_slice_windows,
    _encoder_for_signature,
    _load_checkpoint,
    _resolve_channel_stats,
)
from scripts.patch_gate_eval import (
    TARGET_NAMES,
    geometry_target,
    score,
    shot_payloads,
    train_mean_baseline,
)
from scripts.train_patch_encoder import (
    _cache_root,
    _standardise_values,
    assemble_corpus_cached,
    sensor_geometry_array,
)

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
    kind_temperature: dict[int, float] | None = None,
) -> dict[int, tuple[int, int]]:
    """Per-sensor-kind ``(n_covered, n_masked)`` for ONE slice's held sensors.

    Coverage is EXACT — no sampling.  The predicted sensor mean/variance
    propagate linearly from the diagonal current Gaussian exactly as
    ``amortised_losses`` computes them; the 90% interval is
    ``pred ± z·sqrt(pred_var)`` in the slice's own (unwhitened) units, which
    is equivalent to whitening both sides by the per-sensor scale — coverage
    is scale-invariant.

    ``kind_temperature`` (optional): a per-sensor-KIND multiplicative scalar
    on ``pred_var`` (i.e. std scales by ``sqrt(temp)``) — see
    :func:`fit_kind_temperature`.  Applied ONLY here, in sensor space; it does
    not touch ``i_var`` and therefore has no bearing on topology sampling.
    """
    m_sens = basis.m_sens.to(torch.float64).cpu().numpy()  # (S, n)
    pred = np.asarray(payload.vacuum, dtype=np.float64) + m_sens @ i_mean
    pred_var = (m_sens**2) @ i_var
    if kind_temperature:
        pred_var = pred_var.copy()
        for k, temp in kind_temperature.items():
            sel = sensor_kind == k
            pred_var[sel] = pred_var[sel] * float(temp)
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
#  optional: per-sensor-kind post-hoc σ-temperature (leakage-free, TRAIN-only) #
# --------------------------------------------------------------------------- #
def _encoder_for_corpus_signature(state_dict, extra, corp, device):
    """Build+load an encoder from a CACHED :class:`SignatureCorpus`'s own
    geometry — no table/operator rebuild needed, the corpus cache already
    carries ``sensor_geometry``/``coil_centroids``/``candidate_mask`` for this
    signature (mirrors :func:`scripts.patch_encoder_gate_eval._encoder_for_signature`,
    but reads geometry from the corpus rather than re-deriving it from a
    freshly-built ``GeometryTable``)."""
    cfg = PatchEncoderConfig(**extra["encoder_config"])
    encoder = PatchCurrentEncoder(
        cfg,
        sensor_geometry=corp.sensor_geometry,
        coil_centroids=corp.coil_centroids,
        candidate_mask=np.asarray(corp.candidate_mask, dtype=np.float64),
    ).to(device)
    learned = {k: v for k, v in state_dict.items() if k not in _GEOMETRY_BUFFERS}
    missing, unexpected = encoder.load_state_dict(learned, strict=False)
    missing = [m for m in missing if m not in _GEOMETRY_BUFFERS]
    if missing or unexpected:
        raise RuntimeError(
            f"encoder weight load mismatch: missing={missing} unexpected={unexpected}"
        )
    encoder.eval()
    ch_mean, ch_std, _n_fallback = _resolve_channel_stats(
        corp.sensor_channels, corp.sensor_geometry, extra
    )
    return encoder, ch_mean, ch_std


def fit_kind_temperature(
    state_dict,
    extra,
    corpora,  # dict[str, SignatureCorpus] — TRAIN shots only
    ipf_mean: np.ndarray,
    ipf_std: np.ndarray,
    device,
    *,
    target: float = 0.90,
    z: float = Z90,
    batch_size: int = 512,
    max_examples_per_sig: int = 4000,
    seed: int = 0,
) -> dict[int, float]:
    """Per-sensor-kind post-hoc σ-temperature, fit on TRAIN-shot coverage.

    CLOSED FORM — no search.  Let ``z_i = (measured_i - pred_i)/pred_std_i``
    be a kind's masked, whitened residuals over the TRAIN corpus; the
    temperature (on VARIANCE) that makes exactly ``target`` of them fall
    inside the ``z``-sigma interval is ``(quantile_target(|z_i|) / z)**2``.
    Leakage-free: ``corpora`` must be assembled from TRAIN shots only (the
    SAME split ``read_split_shot_lists`` excluded from the held-out cohort);
    held-out data is never read here.

    A PER-KIND correction is necessarily a SENSOR-space operation: every
    current cell contributes to every sensor with a different weight (via
    ``m_sens``), so no single per-cell (current-space) rescale can
    independently widen one kind's coverage while tightening another's. This
    fit therefore returns sensor-space variance multipliers ONLY — it does
    not, and cannot, feed back into ``i_var``/topology sampling.
    """
    rng = np.random.default_rng(seed)
    zs_by_kind: dict[int, list[np.ndarray]] = {}
    for corp in corpora.values():
        encoder, ch_mean, ch_std = _encoder_for_corpus_signature(
            state_dict, extra, corp, device
        )
        sensor_kind = np.asarray(corp.sensor_geometry, dtype=np.float64)[:, 4].astype(
            int
        )
        m_sens = corp.basis.m_sens.to(torch.float64).cpu().numpy()
        n = corp.values.shape[0]
        idx = rng.permutation(n)[: min(n, max_examples_per_sig)]
        for start in range(0, len(idx), batch_size):
            rows = idx[start : start + batch_size]
            v = _standardise_values(corp.values[rows], corp.finite[rows], ch_mean, ch_std)
            i_pf = corp.i_pf[rows]
            i_pf_std = (i_pf - ipf_mean[None, :]) / ipf_std[None, :]
            with torch.no_grad():
                i_mean_t, i_var_t = encoder(
                    torch.as_tensor(v, dtype=torch.float32, device=device),
                    torch.as_tensor(corp.finite[rows], dtype=torch.bool, device=device),
                    torch.as_tensor(i_pf_std, dtype=torch.float32, device=device),
                    torch.as_tensor(corp.ip[rows], dtype=torch.float32, device=device),
                    return_variance=True,
                )
            i_mean = np.asarray(i_mean_t.detach().cpu().numpy(), dtype=np.float64)
            i_var = np.asarray(i_var_t.detach().cpu().numpy(), dtype=np.float64)
            pred = corp.vacuum[rows] + i_mean @ m_sens.T
            pred_var = i_var @ (m_sens**2).T
            std = np.sqrt(np.clip(pred_var, 0.0, None))
            resid = corp.measured[rows] - pred
            z_scores = np.divide(
                resid, std, out=np.full_like(resid, np.nan), where=std > 0
            )
            mask = corp.mask[rows].astype(bool)
            for k in np.unique(sensor_kind):
                sel = mask & (sensor_kind[None, :] == k)
                vals = np.abs(z_scores[sel])
                vals = vals[np.isfinite(vals)]
                if vals.size:
                    zs_by_kind.setdefault(int(k), []).append(vals)

    temps: dict[int, float] = {}
    for k, chunks in zs_by_kind.items():
        allz = np.concatenate(chunks)
        q = float(np.quantile(allz, target))
        temps[k] = (q / z) ** 2
    return temps


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
    ap.add_argument(
        "--temperature-scale",
        action="store_true",
        help="fit a per-sensor-kind post-hoc σ-temperature on TRAIN-shot "
        "coverage (leakage-free, closed-form, target 0.90) and report a "
        "second 'sensor_coverage_tempscaled' block. Does not affect topology "
        "sampling (a per-kind correction is sensor-space only — see "
        "fit_kind_temperature). overall_pass rolls up the tempscaled sensor "
        "block (not the raw one) when this flag is set.",
    )
    ap.add_argument(
        "--temp-target",
        type=float,
        default=0.90,
        help="target coverage the per-kind temperature fit solves for",
    )
    ap.add_argument(
        "--temp-max-examples-per-sig",
        type=int,
        default=4000,
        help="cap on TRAIN-corpus examples per signature used for the fit "
        "(random subsample; the fit is a single quantile, doesn't need all)",
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
    # per-slice records, kept for a second (tempscaled) sensor-coverage pass
    # without re-running the encoder — (basis, i_mean, i_var, payload, sensor_kind)
    slice_records: list[tuple] = []

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
            slice_records.append((basis, i_mean[k_i], i_var[k_i], p, sensor_kind))
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
        base_median = baseline.get("axis_error_median_m")
        this_median = mean_skill.get("axis_error_median_m")
        base_mean = baseline.get("axis_error_mean_m")
        this_mean = mean_skill.get("axis_error_mean_m")
        regression = {"baseline_gate": args.baseline_gate}

        if base_axis_skill is not None and this_axis_skill is not None:
            delta = float(this_axis_skill) - float(base_axis_skill)
            regression["rmse_skill"] = {
                "baseline_axis_skill": float(base_axis_skill),
                "gaussian_mean_axis_skill": float(this_axis_skill),
                "delta": delta,
                "tolerance": args.regression_tolerance,
                "pass": bool(delta >= -args.regression_tolerance),
                "caveat": "RMSE-based skill can be dominated by a handful of "
                "outlier slices — see median_axis_error for a robust read.",
            }
        # median/mean axis error: reported for human judgement, NOT folded
        # into overall_pass (no established tolerance for it) — the RMSE
        # skill above can look fine while the median tells a different story
        # (metre-scale outliers in one run vs a systematically-worse but
        # outlier-free run in the other are NOT the same failure mode).
        if base_median is not None and this_median is not None and base_median > 0:
            regression["median_axis_error"] = {
                "baseline_median_m": float(base_median),
                "gaussian_median_m": float(this_median),
                "ratio": float(this_median) / float(base_median),
            }
        if base_mean is not None and this_mean is not None and base_mean > 0:
            regression["mean_axis_error"] = {
                "baseline_mean_m": float(base_mean),
                "gaussian_mean_m": float(this_mean),
                "ratio": float(this_mean) / float(base_mean),
            }
        if len(regression) == 1:  # only baseline_gate — nothing comparable found
            regression = None
            logger.warning(
                "baseline-gate or this run missing axis_skill/axis_error_*"
                " — regression check skipped"
            )
        else:
            regression["pass"] = regression.get("rmse_skill", {}).get("pass", True)

    kind_temperature: dict[int, float] | None = None
    sensor_report_tempscaled = None
    if args.temperature_scale:
        train_shots, _held_unused = read_split_shot_lists(
            int(extra["config"]["n_train"]), int(extra["config"]["n_heldout"])
        )
        cache_dir = extra["config"].get("cache_dir") or ""
        corpora_train = assemble_corpus_cached(
            train_shots,
            nr=int(extra["nr"]),
            nz=int(extra["nz"]),
            t_steps=int(extra["t_steps"]),
            stride_s=float(extra["config"]["stride_ms"]) / 1000.0,
            min_ip_ka=float(extra["config"]["min_ip_ka"]),
            cache_root=_cache_root(cache_dir) if cache_dir else None,
        )
        kind_temperature = fit_kind_temperature(
            state_dict,
            extra,
            corpora_train,
            ipf_mean,
            ipf_std,
            device,
            target=args.temp_target,
            max_examples_per_sig=args.temp_max_examples_per_sig,
            seed=args.seed,
        )
        sensor_counts_scaled: dict[int, list[int]] = {}
        for basis_r, i_mean_r, i_var_r, payload_r, sensor_kind_r in slice_records:
            for kind, (c, n) in sensor_coverage_counts(
                basis_r,
                i_mean_r,
                i_var_r,
                payload_r,
                sensor_kind_r,
                kind_temperature=kind_temperature,
            ).items():
                acc = sensor_counts_scaled.setdefault(kind, [0, 0])
                acc[0] += c
                acc[1] += n
        sensor_report_tempscaled = {
            PATCH_SENSOR_KINDS[k] if 0 <= k < len(PATCH_SENSOR_KINDS) else str(k): {
                "n_covered": c,
                "n_masked": n,
                "coverage": coverage_fraction((c, n)),
                "verdict": verdict(coverage_fraction((c, n))),
            }
            for k, (c, n) in sensor_counts_scaled.items()
        }
        logger.info(
            "[temperature-scale] frozen per-kind variance multiplier: %s",
            {
                (
                    PATCH_SENSOR_KINDS[k] if 0 <= k < len(PATCH_SENSOR_KINDS) else str(k)
                ): round(t, 4)
                for k, t in kind_temperature.items()
            },
        )

    # "no-data" (zero observations — e.g. no X-point ever resolved in the
    # held-out cohort) is excluded from the roll-up: it is not evidence of
    # miscalibration, just an untested quantity, and must not force a
    # permanent FAIL for a quantity the corpus never exercises.  Rolls up the
    # TEMPSCALED sensor block (not the raw one) when --temperature-scale ran.
    roll_up_sensor = (
        sensor_report_tempscaled if sensor_report_tempscaled is not None else sensor_report
    )
    all_verdicts = [
        v["verdict"]
        for v in list(roll_up_sensor.values()) + list(topology_report.values())
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
        "sensor_coverage_tempscaled": sensor_report_tempscaled,
        "kind_temperature": (
            {
                (
                    PATCH_SENSOR_KINDS[k] if 0 <= k < len(PATCH_SENSOR_KINDS) else str(k)
                ): t
                for k, t in kind_temperature.items()
            }
            if kind_temperature is not None
            else None
        ),
        "topology_coverage": topology_report,
        "topology_note": (
            "unaffected by --temperature-scale: a per-sensor-kind correction "
            "is necessarily sensor-space (see fit_kind_temperature's "
            "docstring) — it does not touch i_var, which is what topology "
            "sampling draws from, so these numbers are identical to a "
            "run without --temperature-scale."
            if args.temperature_scale
            else None
        ),
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
        "[calibration] %d slices  overall_pass=%s  sensor=%s  sensor_tempscaled=%s  "
        "topology=%s",
        n_slices,
        overall_pass,
        {k: v["verdict"] for k, v in sensor_report.items()},
        (
            {k: v["verdict"] for k, v in sensor_report_tempscaled.items()}
            if sensor_report_tempscaled is not None
            else None
        ),
        {k: v["verdict"] for k, v in topology_report.items()},
    )
    logger.info("calibration artifact -> %s", out_path)
    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
