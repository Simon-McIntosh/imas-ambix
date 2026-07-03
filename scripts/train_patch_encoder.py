#!/usr/bin/env python
"""Amortised patch-current encoder: corpus trainer over the MAST train split.

Trains ONE shared transformer that reads a temporal window of RAW magnetics
(plus the KNOWN PF-coil currents and Rogowski Ip) and emits the in-limiter
patch-current vector in a single forward pass — the learned amortisation of the
per-slice variational inverse (``scripts/patch_gate_eval.py``).  The objective is
the SAME force-balanced loss the inverse minimises: masked whitened sensor
misfit against the RAW magnetics + a Rogowski Ip anchor + a per-example
discrepancy-scheduled Grad-Shafranov structure residual.  The referee is never
read here — training stays behind the firewall by construction.

Every physical input the loss consumes (measured magnetics, the vacuum coil
prediction, the whitening scale) is assembled exactly as the gate assembles it
(:func:`scripts.patch_gate_eval.shot_payloads`): raw magnetics aligned BY NAME to
the patch basis' sensor channels, per-shot ``nanstd`` scale, masks taken pre-fill.
Only the encoder's INPUT tokens are standardised (per-channel corpus mean/std,
saved with the checkpoint); the loss path uses the raw values + per-shot scale
unchanged.

The corpus spans two campaign signatures (differing only in PF-filament
discretisation — the plasma patch substrate ``g_pg``/``g_cc``/``m_sens`` and the
limiter cell set are identical, only the coil-coupling matrices differ), so a
single shared encoder trains against BOTH bases; batches never cross a signature
(each carries one basis for the loss).

In-process, SIGTERM-clean (atomic checkpoint on signal), resume-safe (--resume
restores optimiser + step + epoch + per-example λ state exactly).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import signal
import time
from dataclasses import asdict, dataclass, field
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
from imas_ambix.latent.gs_solve import EquilibriumGrid
from imas_ambix.latent.patch_basis import PatchBasis
from imas_ambix.latent.patch_encoder import (
    DiscrepancyLambda,
    PatchCurrentEncoder,
    PatchEncoderConfig,
    amortised_losses,
    sensor_geometry_from_records,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train_patch_encoder")

DEFAULT_ARTIFACT_ROOT = Path("/work/projects/imas_gpu/latent/patch_encoder")
FALLBACK_ARTIFACT_ROOT = Path("imas_ambix/latent/artifacts/patch_encoder")
WINDOW_HORIZON_S = 0.25  # physical span of a 12-step token window


# --------------------------------------------------------------------------- #
#  Per-campaign geometry: sensor + coil positional features                    #
# --------------------------------------------------------------------------- #
def sensor_geometry_array(table, channels: list[str]) -> np.ndarray:
    """(S, 5) [R, Z, sin θ, cos θ, kind-index] per basis sensor channel, by name.

    Built via the encoder's :func:`sensor_geometry_from_records` so the angle is
    resolved to the seam-continuous sin/cos pair the tokeniser expects.  Channels
    absent from ``table.sensor_map`` (unmatched) get non-finite geometry, which
    the encoder's has-geometry flag handles (their tokens are also always masked
    non-finite, so the value never enters the trunk).
    """
    by_name = {m.amb_channel: m for m in table.sensor_map}
    r = np.full(len(channels), np.nan)
    z = np.full(len(channels), np.nan)
    angle = np.full(len(channels), np.nan)
    kinds: list[str] = []
    for i, ch in enumerate(channels):
        m = by_name.get(ch)
        if m is None:
            kinds.append("unknown")
            continue
        r[i] = float(m.r)
        z[i] = float(m.z)
        angle[i] = np.nan if m.angle_deg is None else float(m.angle_deg)
        kinds.append(m.kind)
    return sensor_geometry_from_records(r, z, angle, kinds)


def coil_centroids_array(table, fwd) -> np.ndarray:
    """(C, 2) [r, z] centroid of each merged PF circuit, in ``i_pf`` column order.

    A column's centroid is the xmult-weighted mean filament position over every
    circuit merged into it (``fwd.pf_merged_circuits`` order == the ``i_pf`` /
    ``coil_psi`` order used everywhere else).
    """
    by_circ: dict[int, list] = {}
    for f in table.pf_filaments:
        by_circ.setdefault(f.circuit, []).append(f)
    out = np.zeros((len(fwd.pf_merged_circuits), 2), dtype=np.float64)
    for j, circs in enumerate(fwd.pf_merged_circuits):
        rs, zs, ws = [], [], []
        for c in circs:
            for f in by_circ.get(c, []):
                w = abs(float(f.xmult)) + 1e-9
                rs.append(float(f.r) * w)
                zs.append(float(f.z) * w)
                ws.append(w)
        if ws:
            out[j, 0] = float(np.sum(rs) / np.sum(ws))
            out[j, 1] = float(np.sum(zs) / np.sum(ws))
    return out


# --------------------------------------------------------------------------- #
#  Corpus assembly                                                             #
# --------------------------------------------------------------------------- #
@dataclass
class SignatureCorpus:
    """One campaign signature's examples + geometry (all fp arrays)."""

    key: str
    basis: PatchBasis
    sensor_channels: list[str]  # basis order
    sensor_geometry: np.ndarray  # (S, 4)
    coil_centroids: np.ndarray  # (C, 2)
    n_cells: int
    candidate_mask: np.ndarray  # (n_cells,)
    # example arrays (N = number of examples)
    values: np.ndarray = field(default=None)  # (N, T, S) raw magnetics, basis order
    finite: np.ndarray = field(default=None)  # (N, T, S) bool
    measured: np.ndarray = field(default=None)  # (N, S) centre raw magnetics
    vacuum: np.ndarray = field(default=None)  # (N, S) centre coil prediction
    mask: np.ndarray = field(default=None)  # (N, S) centre mask
    scale: np.ndarray = field(default=None)  # (N, S) per-shot whitening scale
    i_pf: np.ndarray = field(default=None)  # (N, C) centre known coil currents [A]
    ip: np.ndarray = field(default=None)  # (N,) centre plasma current [A]
    ids: np.ndarray = field(default=None)  # (N,) global example ids


def _basis_alignment(fwd, channels: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """(ch_rows, present): operator rows feeding each basis channel, by name."""
    row_of = {ch: i for i, ch in enumerate(fwd.sensor_channels)}
    ch_rows = np.array([row_of.get(ch, -1) for ch in channels], dtype=np.int64)
    return ch_rows, ch_rows >= 0


def _window_indices(times: np.ndarray, centre_t: float, t_steps: int, *, min_real=1):
    """t_steps sample indices spanning WINDOW_HORIZON_S centred at ``centre_t``.

    Returns ``(idx, real)`` where ``idx`` are per-step nearest-sample indices and
    ``real`` flags the steps whose sample time falls inside the shot's recorded
    stream.  Steps that fall outside ``[times[0], times[-1]]`` (an edge centre
    whose ±0.125 s window overhangs the stream) are PADDED: their index clamps to
    the nearest in-stream sample and ``real`` is False so the caller zeros the
    value and clears the has-value flag — a padded step is never a fabricated
    measurement, only a masked-absent token.  Returns None only when fewer than
    ``min_real`` of the ``t_steps`` steps land inside the stream.
    """
    lo = centre_t - WINDOW_HORIZON_S / 2
    hi = centre_t + WINDOW_HORIZON_S / 2
    sample_t = np.linspace(lo, hi, t_steps)
    real = (sample_t >= times[0]) & (sample_t <= times[-1])
    if int(real.sum()) < min_real:
        return None
    idx = np.clip(
        np.array(
            [int(np.argmin(np.abs(times - st))) for st in sample_t], dtype=np.int64
        ),
        0,
        len(times) - 1,
    )
    return idx, real


def _assemble_shot_examples(
    shot: int,
    *,
    nr: int,
    nz: int,
    t_steps: int,
    stride_s: float,
    min_ip_ka: float,
    basis_cache: dict,
    schema,
):
    """Windowed examples for one shot, or None. Populates ``basis_cache`` by sig."""
    table = build_table_for_shot(int(shot))
    fwd = build_operator(table)
    key = table.signature.key
    if key not in basis_cache:
        grid = EquilibriumGrid.from_table(table, nr=nr, nz=nz)
        basis = PatchBasis.from_table(table, nr=nr, nz=nz)
        _g, channels = grid.sensor_greens(table)
        basis_cache[key] = {
            "basis": basis,
            "channels": list(channels),
            "sensor_geometry": sensor_geometry_array(table, list(channels)),
            "coil_centroids": coil_centroids_array(table, fwd),
            "n_cells": int(basis.candidate_mask.shape[0]),
            "candidate_mask": np.asarray(
                basis.candidate_mask.detach().cpu().numpy(), dtype=np.float64
            ),
        }
    channels = basis_cache[key]["channels"]
    ch_rows, present = _basis_alignment(fwd, channels)
    clip_rows = np.clip(ch_rows, 0, None)

    w = load_shot_windows(int(shot), fwd, key, schema, with_referee=False)
    if w is None:
        return None, key
    times = np.asarray(w.times, dtype=np.float64)
    if times.size < t_steps:
        return None, key

    # raw magnetics + mask aligned to the basis sensor channels (gate convention)
    raw_b = np.where(present, w.raw_mag[:, clip_rows], np.nan)  # (T_all, S)
    mask_b = present & w.mag_mask[:, clip_rows]
    scale = np.nanstd(w.raw_mag, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)
    scale_b = np.where(present, scale[clip_rows], 1.0)  # (S,)

    values, finite = [], []
    measured, vacuum, mask, ip_list, i_pf_list = [], [], [], [], []
    # centres span the whole recorded stream: an edge centre whose window
    # overhangs the stream is admitted with a PADDED partial window (masked
    # steps) provided at least half its steps are real (min_real below).
    centres = np.arange(times[0], times[-1] + 1e-9, stride_s)
    min_real = max(1, t_steps // 2)
    for ct in centres:
        c = int(np.argmin(np.abs(times - ct)))
        if abs(float(w.anchored[c, 0])) <= min_ip_ka:
            continue
        res = _window_indices(times, float(times[c]), t_steps, min_real=min_real)
        if res is None:
            continue
        idx, real = res
        real2 = real[:, None]
        values.append(np.where(real2, raw_b[idx], 0.0))
        finite.append(mask_b[idx] & real2)
        vac = fwd.vacuum_prediction(w.i_pf[c])
        measured.append(raw_b[c])
        vacuum.append(np.where(present, vac[clip_rows], 0.0))
        mask.append(mask_b[c])
        ip_list.append(float(abs(w.anchored[c, 0])) * 1e3)
        i_pf_list.append(w.i_pf[c])
    if not values:
        return None, key
    n = len(values)
    ex = {
        "values": np.asarray(values, dtype=np.float64),
        "finite": np.asarray(finite, dtype=bool),
        "measured": np.asarray(measured, dtype=np.float64),
        "vacuum": np.asarray(vacuum, dtype=np.float64),
        "mask": np.asarray(mask, dtype=bool),
        "scale": np.tile(scale_b, (n, 1)),
        "i_pf": np.asarray(i_pf_list, dtype=np.float64),
        "ip": np.asarray(ip_list, dtype=np.float64),
    }
    return ex, key


def assemble_corpus(
    shots: list[int],
    *,
    nr: int,
    nz: int,
    t_steps: int,
    stride_s: float,
    min_ip_ka: float,
    max_populated_shots: int | None = None,
) -> dict[str, SignatureCorpus]:
    """Build per-signature :class:`SignatureCorpus` bundles for the train split.

    ``max_populated_shots`` stops once that many shots have contributed examples
    (a shot may yield none — short plasma, sub-threshold Ip); used by --dry-run
    to bound the assembly regardless of where the empty shots fall in the list.
    """
    schema = feature_schema()
    basis_cache: dict = {}
    per_sig: dict[str, list[dict]] = {}
    n_populated = 0
    for s in shots:
        try:
            ex, key = _assemble_shot_examples(
                s,
                nr=nr,
                nz=nz,
                t_steps=t_steps,
                stride_s=stride_s,
                min_ip_ka=min_ip_ka,
                basis_cache=basis_cache,
                schema=schema,
            )
        except Exception as exc:  # noqa: BLE001 — a shot w/o geometry is skipped
            logger.warning("shot %s skipped: %s", s, exc)
            continue
        if ex is not None:
            per_sig.setdefault(key, []).append(ex)
            n_populated += 1
            if max_populated_shots is not None and n_populated >= max_populated_shots:
                break

    corpora: dict[str, SignatureCorpus] = {}
    gid = 0
    for key, ex_list in per_sig.items():
        meta = basis_cache[key]
        cat = {
            k: np.concatenate([e[k] for e in ex_list], axis=0)
            for k in (
                "values",
                "finite",
                "measured",
                "vacuum",
                "mask",
                "scale",
                "i_pf",
                "ip",
            )
        }
        n = cat["values"].shape[0]
        corpora[key] = SignatureCorpus(
            key=key,
            basis=meta["basis"],
            sensor_channels=meta["channels"],
            sensor_geometry=meta["sensor_geometry"],
            coil_centroids=meta["coil_centroids"],
            n_cells=meta["n_cells"],
            candidate_mask=meta["candidate_mask"],
            ids=np.arange(gid, gid + n, dtype=np.int64),
            **cat,
        )
        gid += n
        logger.info(
            "signature %s: %d examples, S=%d, n_cells=%d",
            key,
            n,
            len(meta["channels"]),
            meta["n_cells"],
        )
    return corpora


def token_channel_stats(
    corpora: dict[str, SignatureCorpus],
) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel corpus mean/std of the token values (finite entries only).

    Assumes every signature shares the reference channel set (asserted by the
    caller); NaN/absent entries never contribute (masked before the reduction).
    """
    ref = next(iter(corpora.values()))
    s = len(ref.sensor_channels)
    sums = np.zeros(s)
    sqs = np.zeros(s)
    counts = np.zeros(s)
    for corp in corpora.values():
        v = np.where(corp.finite, corp.values, np.nan).reshape(-1, s)
        finite = np.isfinite(v)
        vv = np.where(finite, v, 0.0)
        sums += vv.sum(0)
        sqs += (vv**2).sum(0)
        counts += finite.sum(0)
    counts = np.clip(counts, 1.0, None)
    mean = sums / counts
    var = np.clip(sqs / counts - mean**2, 0.0, None)
    std = np.sqrt(var)
    std = np.where(std > 0, std, 1.0)
    return mean, std


# --------------------------------------------------------------------------- #
#  Batch tensors                                                               #
# --------------------------------------------------------------------------- #
def _standardise_values(values, finite, ch_mean, ch_std):
    """(values - mean)/std per channel; non-finite tokens set to 0."""
    std = (values - ch_mean[None, None, :]) / ch_std[None, None, :]
    return np.where(finite, std, 0.0)


def _make_batch(corp, rows, ch_mean, ch_std, ipf_mean, ipf_std, device):
    """Assemble the encoder inputs + loss payload for one within-signature batch."""
    v = _standardise_values(corp.values[rows], corp.finite[rows], ch_mean, ch_std)
    i_pf = corp.i_pf[rows]
    i_pf_std = (i_pf - ipf_mean[None, :]) / ipf_std[None, :]

    def t32(a):
        return torch.as_tensor(np.asarray(a), dtype=torch.float32, device=device)

    def t64(a):
        return torch.as_tensor(np.asarray(a), dtype=torch.float64, device=device)

    enc_in = {
        "values": t32(v),
        "finite": torch.as_tensor(corp.finite[rows], dtype=torch.bool, device=device),
        "i_pf_std": t32(i_pf_std),
        "ip": t32(corp.ip[rows]),
    }
    payload = {
        "measured": t64(corp.measured[rows]),
        "vacuum": t64(corp.vacuum[rows]),
        "mask": t64(corp.mask[rows].astype(np.float64)),
        "scale": t64(corp.scale[rows]),
        "i_pf_amperes": t64(i_pf),
        "ip": t64(corp.ip[rows]),
    }
    return enc_in, payload


def _epoch_batches(corpora, batch_size, rng):
    """List of (sig_key, row_indices) batches; each batch stays within one sig."""
    batches: list[tuple[str, np.ndarray]] = []
    for key, corp in corpora.items():
        n = corp.values.shape[0]
        order = rng.permutation(n)
        for i in range(0, n, batch_size):
            batches.append((key, order[i : i + batch_size]))
    rng.shuffle(batches)
    return batches


# --------------------------------------------------------------------------- #
#  Training                                                                    #
# --------------------------------------------------------------------------- #
def _lr_lambda(total_steps, warmup_frac, floor_frac):
    warmup = max(1, int(warmup_frac * total_steps))

    def fn(step):
        if step < warmup:
            return step / warmup
        prog = (step - warmup) / max(1, total_steps - warmup)
        cos = 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))
        return floor_frac + (1 - floor_frac) * cos

    return fn


def _disc_state(disc) -> dict:
    """Serialisable per-example λ schedule state (the module class has no I/O)."""
    return {
        "lam": disc.lam.detach().cpu(),
        "target": disc.target.detach().cpu(),
        "warm_misfit": disc._warm_misfit.detach().cpu(),
        "epoch": int(disc._epoch),
    }


def _restore_disc(disc, state) -> None:
    disc.lam = state["lam"].to(disc.device, disc.dtype)
    disc.target = state["target"].to(disc.device, disc.dtype)
    disc._warm_misfit = state["warm_misfit"].to(disc.device, disc.dtype)
    disc._epoch = int(state["epoch"])


def _save_checkpoint(path, encoder, optimizer, scheduler, disc, step, epoch, extra):
    payload = {
        "step": int(step),
        "epoch": int(epoch),
        "encoder": encoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "discrepancy": _disc_state(disc),
        "extra": extra,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)  # atomic on the same filesystem — SIGTERM-safe


def _train_eval_split(corpora, n_holdback_shots_frac=0.05):
    """Reserve the last few examples of the dominant signature for a cheap eval.

    Returns (eval_rows_by_sig): a small held-back slice used only for a
    misfit/fb read each epoch (no referee, no scoring).
    """
    ref_key = max(corpora, key=lambda k: corpora[k].values.shape[0])
    n = corpora[ref_key].values.shape[0]
    n_eval = max(1, int(n_holdback_shots_frac * n))
    return {ref_key: np.arange(n - n_eval, n)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-train", type=int, default=200)
    ap.add_argument("--n-heldout", type=int, default=8)
    ap.add_argument("--nr", type=int, default=65)
    ap.add_argument("--nz", type=int, default=97)
    ap.add_argument("--t-steps", type=int, default=12)
    ap.add_argument("--stride-ms", type=float, default=25.0)
    ap.add_argument("--min-ip-ka", type=float, default=300.0)
    ap.add_argument("--head", type=str, default="direct", choices=("direct", "lowrank"))
    ap.add_argument("--d-model", type=int, default=160)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--warmup-frac", type=float, default=0.10)
    ap.add_argument("--lr-floor-frac", type=float, default=0.03)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--lam0", type=float, default=3.0)
    ap.add_argument("--lambda-ratio", type=float, default=1.5)
    ap.add_argument("--lam-max", type=float, default=100.0)
    ap.add_argument("--lambda-warmup-epochs", type=int, default=5)
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--run", type=str, default="direct")
    ap.add_argument("--artifact-root", type=str, default="")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="assemble <=2 shots, one tiny CPU forward+loss step, print the loss dict",
    )
    args = ap.parse_args()

    device = (
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else args.device
    )
    if args.dry_run:
        device = "cpu"
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    train_shots, _held = read_split_shot_lists(args.n_train, args.n_heldout)
    if args.dry_run:
        # scan a modest candidate window; stop after 2 shots actually populate
        # (the first shots in the split can legitimately yield no usable slices)
        train_shots = train_shots[:16]
    logger.info("assembling corpus over up to %d train shots", len(train_shots))

    corpora = assemble_corpus(
        train_shots,
        nr=args.nr,
        nz=args.nz,
        t_steps=args.t_steps,
        stride_s=args.stride_ms / 1000.0,
        min_ip_ka=args.min_ip_ka,
        max_populated_shots=2 if args.dry_run else None,
    )
    if not corpora:
        logger.error("no usable training examples assembled — aborting")
        return 1

    # one shared encoder is bound to the reference (dominant) signature's
    # geometry; the plasma substrate is identical across signatures, so the same
    # n_cells / sensor set serves every basis (asserted below)
    ref_key = max(corpora, key=lambda k: corpora[k].values.shape[0])
    ref = corpora[ref_key]
    for key, corp in list(corpora.items()):
        if corp.sensor_channels != ref.sensor_channels or corp.n_cells != ref.n_cells:
            logger.warning(
                "signature %s geometry differs from reference %s (S=%d/%d, "
                "n_cells=%d/%d) — dropping its %d examples",
                key,
                ref_key,
                len(corp.sensor_channels),
                len(ref.sensor_channels),
                corp.n_cells,
                ref.n_cells,
                corp.values.shape[0],
            )
            del corpora[key]
    n_total = sum(c.values.shape[0] for c in corpora.values())
    logger.info("corpus: %d examples across %d signatures", n_total, len(corpora))

    ch_mean, ch_std = token_channel_stats(corpora)
    all_ipf = np.concatenate([c.i_pf for c in corpora.values()], axis=0)
    ipf_mean = all_ipf.mean(0)
    ipf_std = np.where(all_ipf.std(0) > 0, all_ipf.std(0), 1.0)

    cfg = PatchEncoderConfig(
        head=args.head,
        d_model=args.d_model if not args.dry_run else 32,
        n_layers=args.n_layers if not args.dry_run else 1,
        n_heads=args.n_heads if not args.dry_run else 2,
        dim_feedforward=640 if not args.dry_run else 64,
        dropout=args.dropout,
        n_time=args.t_steps,
    )
    encoder = PatchCurrentEncoder(
        cfg,
        sensor_geometry=ref.sensor_geometry,
        coil_centroids=ref.coil_centroids,
        candidate_mask=ref.candidate_mask,
    ).to(device)

    optimizer = torch.optim.AdamW(
        encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    total_steps = max(1, args.epochs * math.ceil(n_total / args.batch_size))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, _lr_lambda(total_steps, args.warmup_frac, args.lr_floor_frac)
    )
    disc = DiscrepancyLambda(
        n_total,
        lam0=args.lam0,
        ratio=args.lambda_ratio,
        lam_max=args.lam_max,
        warmup_epochs=args.lambda_warmup_epochs,
        device=device,
    )

    root = (
        Path(args.artifact_root)
        if args.artifact_root
        else (
            DEFAULT_ARTIFACT_ROOT
            if DEFAULT_ARTIFACT_ROOT.parent.exists()
            else FALLBACK_ARTIFACT_ROOT
        )
    )
    run_dir = root / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "patch_encoder.pt"
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

    ckpt_extra = {
        "config": vars(args),
        "encoder_config": asdict(cfg),
        "sensor_channels": ref.sensor_channels,
        "sensor_geometry": ref.sensor_geometry,
        "coil_centroids": ref.coil_centroids,
        "n_cells": ref.n_cells,
        "candidate_mask": ref.candidate_mask,
        "channel_mean": ch_mean,
        "channel_std": ch_std,
        "ipf_mean": ipf_mean,
        "ipf_std": ipf_std,
        "nr": args.nr,
        "nz": args.nz,
        "t_steps": args.t_steps,
        "reference_signature": ref_key,
    }

    start_epoch = 0
    global_step = 0
    if args.resume and ckpt_path.exists():
        payload = torch.load(ckpt_path, map_location=device, weights_only=False)
        encoder.load_state_dict(payload["encoder"])
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        if payload.get("discrepancy") is not None:
            _restore_disc(disc, payload["discrepancy"])
        start_epoch = int(payload["epoch"])
        global_step = int(payload["step"])
        logger.info("resumed at epoch %d, step %d", start_epoch, global_step)

    stop = {"flag": False}

    def _on_sigterm(*_a):
        logger.warning("SIGTERM received — checkpointing and exiting")
        stop["flag"] = True

    signal.signal(signal.SIGTERM, _on_sigterm)

    eval_rows = _train_eval_split(corpora)

    if args.dry_run:
        key, corp = next(iter(corpora.items()))
        rows = np.arange(min(args.batch_size, corp.values.shape[0]))
        enc_in, payload = _make_batch(
            corp, rows, ch_mean, ch_std, ipf_mean, ipf_std, device
        )
        lam = disc.get(torch.as_tensor(corp.ids[rows], device=device))
        i_cell = encoder(
            enc_in["values"], enc_in["finite"], enc_in["i_pf_std"], enc_in["ip"]
        )
        losses = amortised_losses(corp.basis, i_cell, lam=lam, **payload)
        printable = {
            k: (float(v.mean()) if torch.is_tensor(v) else float(v))
            for k, v in losses.items()
        }
        logger.info("DRY-RUN loss dict: %s", printable)
        print("DRY_RUN_LOSS", json.dumps(printable))
        return 0

    t0 = time.time()
    for epoch in range(start_epoch, args.epochs):
        if stop["flag"]:
            break
        encoder.train()
        batches = _epoch_batches(corpora, args.batch_size, rng)
        for key, rows in batches:
            if stop["flag"]:
                break
            corp = corpora[key]
            enc_in, payload = _make_batch(
                corp, rows, ch_mean, ch_std, ipf_mean, ipf_std, device
            )
            ids = torch.as_tensor(corp.ids[rows], device=device)
            lam = disc.get(ids)
            optimizer.zero_grad()
            i_cell = encoder(
                enc_in["values"], enc_in["finite"], enc_in["i_pf_std"], enc_in["ip"]
            )
            losses = amortised_losses(corp.basis, i_cell, lam=lam, **payload)
            loss = losses["total"].mean() if losses["total"].dim() else losses["total"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            with torch.no_grad():
                misfit = losses.get("misfit")
                if misfit is not None:
                    disc.update(ids, misfit.detach(), epoch)
            if global_step % args.log_every == 0:
                terms = {
                    k: round(float(v.mean()), 5)
                    for k, v in losses.items()
                    if torch.is_tensor(v)
                }
                logger.info(
                    "epoch %d step %d lr=%.2e %s (%.0fs)",
                    epoch,
                    global_step,
                    scheduler.get_last_lr()[0],
                    terms,
                    time.time() - t0,
                )
            if global_step > 0 and global_step % args.ckpt_every == 0:
                _save_checkpoint(
                    ckpt_path,
                    encoder,
                    optimizer,
                    scheduler,
                    disc,
                    global_step,
                    epoch,
                    ckpt_extra,
                )
            global_step += 1

        # cheap held-back eval (misfit / fb only, no referee)
        encoder.eval()
        with torch.no_grad():
            for key, rows in eval_rows.items():
                corp = corpora[key]
                enc_in, payload = _make_batch(
                    corp, rows, ch_mean, ch_std, ipf_mean, ipf_std, device
                )
                lam = torch.zeros(len(rows), device=device)
                i_cell = encoder(
                    enc_in["values"],
                    enc_in["finite"],
                    enc_in["i_pf_std"],
                    enc_in["ip"],
                )
                losses = amortised_losses(corp.basis, i_cell, lam=lam, **payload)
                report = {
                    k: round(float(v.mean()), 5)
                    for k, v in losses.items()
                    if torch.is_tensor(v)
                }
                logger.info("epoch %d HELD-BACK eval %s", epoch, report)

    _save_checkpoint(
        ckpt_path,
        encoder,
        optimizer,
        scheduler,
        disc,
        global_step,
        args.epochs,
        ckpt_extra,
    )
    logger.info("final checkpoint -> %s (step %d)", ckpt_path, global_step)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
