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
import hashlib
import json
import logging
import math
import signal
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from imas_ambix.gs.geometry import (
    GEOMETRY_TABLE_VERSION,
    build_table_for_shot,
    canonical_amb_channels,
    discover_signatures,
)
from imas_ambix.gs.operator import (
    COIL_MODEL_VERSION,
    build_operator,
    write_operator_summary,
)
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
    kind_index,
    sensor_geometry_from_records,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train_patch_encoder")

DEFAULT_ARTIFACT_ROOT = Path("/work/projects/imas_gpu/latent/patch_encoder")
FALLBACK_ARTIFACT_ROOT = Path("imas_ambix/latent/artifacts/patch_encoder")
DEFAULT_CACHE_ROOT = Path("/work/projects/imas_gpu/latent/patch_encoder/corpus_cache")
FALLBACK_CACHE_ROOT = Path("imas_ambix/latent/artifacts/patch_encoder/corpus_cache")
WINDOW_HORIZON_S = 0.25  # physical span of a 12-step token window

CORPUS_ASSEMBLY_VERSION = "geometry-shots-v1"
"""Bump whenever ``assemble_corpus``'s ASSEMBLY SEMANTICS change in a way that
changes its output for the SAME (shots, nr, nz, t_steps, stride, min_ip,
COIL_MODEL_VERSION, GEOMETRY_TABLE_VERSION) — i.e. a behavioural fix to this
module itself, not to anything it calls into.  Those upstream constants only
cover changes in imas_ambix.gs.operator / .geometry; a change HERE (e.g.
"geometry-shots-v1": canonical_amb_channels now scans every shot sharing a
signature, not just a shard's own local slice) would otherwise silently keep
serving an already-cached corpus assembled under the OLD semantics, since none
of the other cache-key inputs changed. ``--force-rebuild-cache`` / ``FORCE=1``
remain the manual override for a one-off rebuild without bumping this.
"""

#: encoder buffers that carry per-CAMPAIGN geometry, never the learned trunk —
#: swapped per batch by :func:`_bind_signature` and excluded from any
#: checkpoint load/save identity check (mirrors
#: ``scripts/patch_encoder_gate_eval.py``'s ``_GEOMETRY_BUFFERS``).
_GEOMETRY_BUFFERS = frozenset(
    {"sensor_geom", "sensor_kind", "coil_geom", "coil_kind", "candidate_mask"}
)


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
    table,  # GeometryTable — the signature's CANONICAL table (see assemble_corpus)
    fwd,  # ForwardOperator built from that same canonical table
    key: str,
    basis_cache: dict,
    schema,
    operator_out: dict | None = None,
):
    """Windowed examples for one shot, or None. Populates ``basis_cache`` by sig.

    ``table``/``fwd`` are the signature's canonical geometry (built ONCE by
    ``assemble_corpus`` from :func:`imas_ambix.gs.geometry.canonical_amb_channels`
    over every shot sharing the signature — not rebuilt per shot) so the
    resulting sensor channel SET is geometry-determined, never an artifact of
    which shot happened to be processed first (a per-shot ``amb`` schema gap —
    e.g. a flux loop recorded for most but not all shots of a campaign — used
    to silently change ``S`` depending on shot order; see
    ``imas_ambix/gs/geometry.py::canonical_amb_channels``).  Per-shot data that
    is genuinely absent still comes back correctly masked below (``present``/
    ``mask_b``), by construction — this only fixes which channels EXIST in the
    corpus, not whether a given shot measured them.

    ``operator_out`` (optional) collects one ``(table, operator)`` pair per
    first-seen signature — a free side-product of the assembly pass, used to
    regenerate ``imas_ambix/gs/artifacts/gs_operator_summary.json`` without a
    separate shot scan (see ``regenerate_operator_summary`` below).
    """
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
        if operator_out is not None:
            operator_out[key] = (table, fwd)
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
    operator_out: dict | None = None,
    geometry_shots: list[int] | None = None,
) -> dict[str, SignatureCorpus]:
    """Build per-signature :class:`SignatureCorpus` bundles for the train split.

    ``max_populated_shots`` stops once that many shots have contributed examples
    (a shot may yield none — short plasma, sub-threshold Ip); used by --dry-run
    to bound the assembly regardless of where the empty shots fall in the list.

    ``operator_out`` (optional): pass an empty dict to collect one
    ``(table, operator)`` pair per signature this assembly encounters — see
    :func:`regenerate_operator_summary`.

    ``geometry_shots`` (optional): the shot list ``canonical_amb_channels`` is
    unioned over, if it should be WIDER than ``shots`` itself — the sharded
    assembly fleet's reason to pass this: a channel present on only a sparse
    subset of a signature's shots can appear in one shard's own local slice
    and not another's, so resolving the canonical schema per-shard-slice can
    still disagree ACROSS shards.  Pass the FULL, un-sliced corpus shot list
    here so every shard converges to the identical canonical schema by
    construction, independent of which shots landed in which shard.  Defaults
    to ``shots`` (correct for an unsharded call, where they're the same list).

    Geometry is resolved in two phases so the sensor channel SET is
    geometry-determined rather than an artifact of shot order (see
    ``imas_ambix/gs/geometry.py::canonical_amb_channels``): (1) group
    ``geometry_shots`` by :class:`~imas_ambix.gs.geometry.SetupSignature` and
    build ONE canonical table per signature from the union of EVERY one of its
    shots' amb schema; (2) walk ``shots`` (this call's own, possibly sharded,
    slice) reusing that canonical ``(table, fwd)`` pair — never rebuilt per
    shot — for the per-shot windowed-example assembly.
    """
    schema = feature_schema()
    groups = discover_signatures(shots if geometry_shots is None else geometry_shots)
    sig_geometry: dict[str, tuple] = {}
    shot_to_key: dict[int, str] = {}
    for key, (_sig, sig_shots) in groups.items():
        canonical_amb = canonical_amb_channels(sig_shots)  # unbounded: every shot
        table = None
        for rep in sig_shots:
            try:
                table = build_table_for_shot(rep, amb_channels=canonical_amb)
                break
            except Exception as exc:  # noqa: BLE001 — try the next candidate shot
                logger.warning(
                    "signature %s: candidate shot %s failed (%s) — trying next",
                    key,
                    rep,
                    exc,
                )
                continue
        if table is None:
            logger.warning(
                "signature %s: no candidate shot could build a table (%d tried) "
                "— its %d shots are skipped",
                key,
                len(sig_shots),
                len(sig_shots),
            )
            continue
        fwd = build_operator(table)
        sig_geometry[key] = (table, fwd)
        for s in sig_shots:
            shot_to_key[int(s)] = key

    basis_cache: dict = {}
    per_sig: dict[str, list[dict]] = {}
    n_populated = 0
    for s in shots:
        key = shot_to_key.get(int(s))
        if key is None:
            continue  # discovery skipped this shot (unreadable efm geometry)
        table, fwd = sig_geometry[key]
        try:
            ex, key = _assemble_shot_examples(
                s,
                nr=nr,
                nz=nz,
                t_steps=t_steps,
                stride_s=stride_s,
                min_ip_ka=min_ip_ka,
                table=table,
                fwd=fwd,
                key=key,
                basis_cache=basis_cache,
                schema=schema,
                operator_out=operator_out,
            )
        except Exception as exc:  # noqa: BLE001 — a shot w/o usable windows is skipped
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


def regenerate_operator_summary(
    operator_out: dict, *, out_path: Path | None = None
) -> Path:
    """Refresh ``imas_ambix/gs/artifacts/gs_operator_summary.json`` from the
    ``(table, operator)`` pairs an assembly pass already collected.

    A free side-product of a corpus assembly rather than a separate shot scan:
    ``assemble_corpus(..., operator_out=out)`` populates ``out`` with one entry
    per campaign signature it encounters.  Call this after an assembly whose
    shot list is broad enough to have seen every signature you care about
    (the committed summary is stale after any coil-model change — e.g. the
    case-circuit fix, ``COIL_MODEL_VERSION``).
    """
    if not operator_out:
        raise ValueError(
            "operator_out is empty — assemble with operator_out={} passed "
            "through, over a shot list that actually reaches every signature"
        )
    tables = {k: t for k, (t, _fwd) in operator_out.items()}
    operators = {k: fwd for k, (_t, fwd) in operator_out.items()}
    path = write_operator_summary(operators, tables, out_path=out_path)
    logger.info(
        "operator summary regenerated -> %s (%d signatures, coil_model_version=%s)",
        path,
        len(operators),
        COIL_MODEL_VERSION,
    )
    return path


def token_channel_stats_by_name(
    corpora: dict[str, SignatureCorpus],
) -> dict[str, tuple[float, float]]:
    """Per-channel-NAME corpus mean/std of token values across ALL signatures.

    Signatures may carry different sensor sets / orderings (the S=81-style
    campaign no longer gets dropped — see :func:`main`), so channels are
    accumulated BY NAME rather than by column position: a channel present in
    two signatures pools its finite observations from both.  A channel with
    zero finite observations anywhere (should not happen in practice, but is
    handled rather than dividing by zero) falls back to the median stat of
    channels sharing its sensor KIND (the kind index carried in column 4 of
    each signature's ``sensor_geometry``), or the global median if the kind
    itself has no observed channel either.
    """
    sums: dict[str, float] = {}
    sqs: dict[str, float] = {}
    counts: dict[str, float] = {}
    kind_of: dict[str, int] = {}
    for corp in corpora.values():
        s = len(corp.sensor_channels)
        v = np.where(corp.finite, corp.values, np.nan).reshape(-1, s)
        finite = np.isfinite(v)
        vv = np.where(finite, v, 0.0)
        csum = vv.sum(0)
        csq = (vv**2).sum(0)
        ccount = finite.sum(0)
        geom = np.asarray(corp.sensor_geometry, dtype=np.float64)
        kinds = geom[:, 4].astype(int) if geom.shape[1] > 4 else np.zeros(s, dtype=int)
        for j, ch in enumerate(corp.sensor_channels):
            sums[ch] = sums.get(ch, 0.0) + float(csum[j])
            sqs[ch] = sqs.get(ch, 0.0) + float(csq[j])
            counts[ch] = counts.get(ch, 0.0) + float(ccount[j])
            kind_of.setdefault(ch, int(kinds[j]))

    stats: dict[str, tuple[float, float]] = {}
    for ch, cnt in counts.items():
        if cnt > 0:
            mean = sums[ch] / cnt
            var = max(sqs[ch] / cnt - mean**2, 0.0)
            stats[ch] = (mean, float(np.sqrt(var)) if var > 0 else 1.0)

    by_kind: dict[int, list[tuple[float, float]]] = {}
    for ch, (m, sd) in stats.items():
        by_kind.setdefault(kind_of[ch], []).append((m, sd))
    kind_median = {
        k: (float(np.median([m for m, _ in v])), float(np.median([sd for _, sd in v])))
        for k, v in by_kind.items()
    }
    glob_median = (
        (float(np.median([m for m, _ in stats.values()])), 1.0) if stats else (0.0, 1.0)
    )
    for ch, cnt in counts.items():
        if cnt == 0:
            stats[ch] = kind_median.get(kind_of[ch], glob_median)
    return stats


def channel_stats_for_signature(
    channels: list[str], stats_by_name: dict[str, tuple[float, float]]
) -> tuple[np.ndarray, np.ndarray]:
    """This signature's own ``(mean, std)`` arrays, looked up BY NAME."""
    means, stds = [], []
    for ch in channels:
        m, sd = stats_by_name.get(ch, (0.0, 1.0))
        means.append(float(m))
        stds.append(float(sd) if sd > 0 else 1.0)
    return np.asarray(means), np.asarray(stds)


def _bind_signature(
    encoder: PatchCurrentEncoder, corp: SignatureCorpus, device
) -> None:
    """Rebind ``encoder``'s geometry buffers to ``corp``'s campaign in place.

    The trunk (value/geometry/kind/flag/temporal embeddings, transformer, pool,
    per-cell head) is the SAME module instance and is never touched — only the
    ``_GEOMETRY_BUFFERS`` (plain tensors the forward pass reads at call time,
    per ``imas_ambix/latent/patch_encoder.py``) plus the plain ``n_sensor`` /
    ``n_coil`` bookkeeping ints are rebound, mirroring the buffer-swap already
    used at eval (``scripts/patch_encoder_gate_eval.py::_encoder_for_signature``).
    One shared encoder therefore trains against every campaign signature in the
    corpus — no signature is dropped for a differing sensor/coil count.

    ``n_cells`` (the per-cell head's output width) is NOT swappable — it is
    fixed by the trunk's construction — so a signature whose candidate mask
    disagrees in length is a hard error, not a silent drop.
    """
    if int(corp.n_cells) != int(encoder.n_cells):
        raise ValueError(
            f"signature {corp.key}: n_cells={corp.n_cells} != encoder "
            f"n_cells={encoder.n_cells} — the plasma patch substrate must be "
            "identical across every campaign signature the encoder trains on"
        )
    dtype = encoder.sensor_geom.dtype
    sg = np.asarray(corp.sensor_geometry, dtype=np.float64)
    geom = sg[:, :4]
    if sg.shape[1] > 4:
        kind = sg[:, 4].astype(np.int64)
    else:
        kind = np.zeros(sg.shape[0], dtype=np.int64)
    cc = np.asarray(corp.coil_centroids, dtype=np.float64).reshape(-1, 2)
    coil_geom = np.zeros((cc.shape[0], 4), dtype=np.float64)
    coil_geom[:, :2] = cc
    coil_kind = np.full(cc.shape[0], kind_index("coil"), dtype=np.int64)

    encoder.sensor_geom = torch.as_tensor(geom, dtype=dtype, device=device)
    encoder.sensor_kind = torch.as_tensor(kind, dtype=torch.long, device=device)
    encoder.coil_geom = torch.as_tensor(coil_geom, dtype=dtype, device=device)
    encoder.coil_kind = torch.as_tensor(coil_kind, dtype=torch.long, device=device)
    encoder.candidate_mask = torch.as_tensor(
        np.asarray(corp.candidate_mask, dtype=np.float64), dtype=dtype, device=device
    )
    encoder.n_sensor = int(sg.shape[0])
    encoder.n_coil = int(cc.shape[0])


# --------------------------------------------------------------------------- #
#  Corpus cache: per-signature example arrays + a fully self-contained         #
#  PatchBasis (no IMAS re-read needed on load — every PatchBasis constructor   #
#  argument is a plain geometry-derived numpy array; see patch_basis.py).      #
# --------------------------------------------------------------------------- #
def _config_hash(
    shots: list[int],
    *,
    t_steps: int,
    stride_s: float,
    min_ip_ka: float,
    nr: int,
    nz: int,
) -> str:
    """Stable short digest identifying an assembly configuration.

    Includes ``COIL_MODEL_VERSION`` (imas_ambix.gs.operator) so a corpus
    assembled under one coil-current model (the vacuum-field prediction the
    loss trains against) can never collide with one assembled after a coil
    model fix; ``GEOMETRY_TABLE_VERSION`` (imas_ambix.gs.geometry) so the same
    holds for a change in how a :class:`GeometryTable` — its sensor channel
    SET in particular — is derived from a fixed signature digest; and
    ``CORPUS_ASSEMBLY_VERSION`` (this module) so the same holds for a change
    in how THIS module turns a shot list into a corpus (e.g. which shots
    canonical_amb_channels scans) even when neither upstream constant moved.
    Every one of the three busts the cache key automatically the moment its
    constant changes, with no separate migration step.
    """
    payload = {
        "shots": sorted(int(s) for s in shots),
        "t_steps": int(t_steps),
        "stride_s": round(float(stride_s), 9),
        "min_ip_ka": round(float(min_ip_ka), 6),
        "nr": int(nr),
        "nz": int(nz),
        "coil_model_version": COIL_MODEL_VERSION,
        "geometry_table_version": GEOMETRY_TABLE_VERSION,
        "corpus_assembly_version": CORPUS_ASSEMBLY_VERSION,
    }
    blob = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def _cache_root(explicit: str | Path | None) -> Path:
    if explicit:
        return Path(explicit)
    if DEFAULT_CACHE_ROOT.parent.exists():
        return DEFAULT_CACHE_ROOT
    return FALLBACK_CACHE_ROOT


def _patch_basis_kwargs(basis: PatchBasis) -> dict:
    """``PatchBasis`` constructor kwargs as plain numpy — every field is a pure
    geometry-derived matrix, so this round-trips through ``PatchBasis(**kw)``
    with no IMAS access at all."""
    return {
        "g_pg": basis._g_pg_np,
        "g_cc": basis._g_cc_np,
        "m_sens": np.asarray(basis.m_sens.detach().cpu().numpy(), dtype=np.float64),
        "m_coil": np.asarray(basis.m_coil.detach().cpu().numpy(), dtype=np.float64),
        "psi_coil_grid": basis._psi_coil_grid_np,
        "psi_coil_cells": basis._psi_coil_cells_np,
        "r_cells": np.asarray(basis.r_cells.detach().cpu().numpy(), dtype=np.float64),
        "z_cells": np.asarray(basis.z_cells.detach().cpu().numpy(), dtype=np.float64),
        "grid_r": np.asarray(basis.grid_r.detach().cpu().numpy(), dtype=np.float64),
        "grid_z": np.asarray(basis.grid_z.detach().cpu().numpy(), dtype=np.float64),
        "nr": int(basis.nr),
        "nz": int(basis.nz),
        "cell_area": float(basis.cell_area),
        "r0": float(basis.r0),
        "sensor_channels": list(basis.sensor_channels),
    }


def _save_signature_npz(
    path: Path,
    corp: SignatureCorpus,
    *,
    shots: list[int],
    t_steps: int,
    stride_s: float,
    min_ip_ka: float,
    nr: int,
    nz: int,
    config_hash: str,
) -> None:
    bk = _patch_basis_kwargs(corp.basis)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / (path.stem + ".tmp.npz")
    np.savez(
        tmp,
        key=np.asarray(corp.key),
        sensor_channels=np.asarray(corp.sensor_channels),
        sensor_geometry=corp.sensor_geometry,
        coil_centroids=corp.coil_centroids,
        n_cells=np.asarray(corp.n_cells),
        candidate_mask=corp.candidate_mask,
        values=corp.values,
        finite=corp.finite,
        measured=corp.measured,
        vacuum=corp.vacuum,
        mask=corp.mask,
        scale=corp.scale,
        i_pf=corp.i_pf,
        ip=corp.ip,
        ids=corp.ids,
        basis_g_pg=bk["g_pg"],
        basis_g_cc=bk["g_cc"],
        basis_m_sens=bk["m_sens"],
        basis_m_coil=bk["m_coil"],
        basis_psi_coil_grid=bk["psi_coil_grid"],
        basis_psi_coil_cells=bk["psi_coil_cells"],
        basis_r_cells=bk["r_cells"],
        basis_z_cells=bk["z_cells"],
        basis_grid_r=bk["grid_r"],
        basis_grid_z=bk["grid_z"],
        basis_nr=np.asarray(bk["nr"]),
        basis_nz=np.asarray(bk["nz"]),
        basis_cell_area=np.asarray(bk["cell_area"]),
        basis_r0=np.asarray(bk["r0"]),
        basis_sensor_channels=np.asarray(bk["sensor_channels"]),
        config_shots=np.asarray(sorted({int(s) for s in shots}), dtype=np.int64),
        config_t_steps=np.asarray(t_steps),
        config_stride_s=np.asarray(stride_s),
        config_min_ip_ka=np.asarray(min_ip_ka),
        config_nr=np.asarray(nr),
        config_nz=np.asarray(nz),
        config_hash=np.asarray(config_hash),
    )
    tmp.replace(path)  # atomic on the same filesystem — matches _save_checkpoint


def _load_signature_npz(path: Path) -> SignatureCorpus:
    d = np.load(path, allow_pickle=False)
    basis = PatchBasis(
        g_pg=d["basis_g_pg"],
        g_cc=d["basis_g_cc"],
        m_sens=d["basis_m_sens"],
        m_coil=d["basis_m_coil"],
        psi_coil_grid=d["basis_psi_coil_grid"],
        psi_coil_cells=d["basis_psi_coil_cells"],
        r_cells=d["basis_r_cells"],
        z_cells=d["basis_z_cells"],
        candidate_mask=d["candidate_mask"],
        grid_r=d["basis_grid_r"],
        grid_z=d["basis_grid_z"],
        nr=int(d["basis_nr"]),
        nz=int(d["basis_nz"]),
        cell_area=float(d["basis_cell_area"]),
        r0=float(d["basis_r0"]),
        sensor_channels=[str(c) for c in d["basis_sensor_channels"]],
    )
    return SignatureCorpus(
        key=str(d["key"]),
        basis=basis,
        sensor_channels=[str(c) for c in d["sensor_channels"]],
        sensor_geometry=d["sensor_geometry"],
        coil_centroids=d["coil_centroids"],
        n_cells=int(d["n_cells"]),
        candidate_mask=d["candidate_mask"],
        values=d["values"],
        finite=d["finite"],
        measured=d["measured"],
        vacuum=d["vacuum"],
        mask=d["mask"],
        scale=d["scale"],
        i_pf=d["i_pf"],
        ip=d["ip"],
        ids=d["ids"],
    )


_LIGHT_EXAMPLE_FIELDS = (
    "values",
    "finite",
    "measured",
    "vacuum",
    "mask",
    "scale",
    "i_pf",
    "ip",
)


def _load_signature_npz_light(path: Path) -> dict:
    """Load ONLY a signature npz's example arrays + small geometry-identity
    fields — never the ``basis_*`` keys, so the ``g_pg`` matrix (the dominant
    memory cost: ``O(grid_points x n_cells)``, ~hundreds of MB per signature)
    is never decompressed.  ``np.load`` on an npz is lazy per-key, so simply
    not touching ``basis_*`` is enough to skip reading it.

    Used to merge MANY shard files per signature without reconstructing a
    full :class:`PatchBasis` once per shard (only once per signature is ever
    needed — see :func:`_merge_corpus_dirs`).
    """
    d = np.load(path, allow_pickle=False)
    out = {
        "key": str(d["key"]),
        "sensor_channels": [str(c) for c in d["sensor_channels"]],
        "n_cells": int(d["n_cells"]),
    }
    out.update({k: d[k] for k in _LIGHT_EXAMPLE_FIELDS})
    return out


def _corpus_dir_complete(dir_path: Path) -> bool:
    return (dir_path / "_DONE").exists()


def _save_corpus_dir(
    dir_path: Path,
    corpora: dict[str, SignatureCorpus],
    *,
    shots: list[int],
    t_steps: int,
    stride_s: float,
    min_ip_ka: float,
    nr: int,
    nz: int,
    config_hash: str,
) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    for key, corp in corpora.items():
        _save_signature_npz(
            dir_path / f"{key}.npz",
            corp,
            shots=shots,
            t_steps=t_steps,
            stride_s=stride_s,
            min_ip_ka=min_ip_ka,
            nr=nr,
            nz=nz,
            config_hash=config_hash,
        )
    meta = {
        "config_hash": config_hash,
        "coil_model_version": COIL_MODEL_VERSION,
        "geometry_table_version": GEOMETRY_TABLE_VERSION,
        "corpus_assembly_version": CORPUS_ASSEMBLY_VERSION,
        "n_shots": len({int(s) for s in shots}),
        "t_steps": t_steps,
        "stride_s": stride_s,
        "min_ip_ka": min_ip_ka,
        "nr": nr,
        "nz": nz,
        "signatures": list(corpora.keys()),
        "n_examples": {k: int(c.values.shape[0]) for k, c in corpora.items()},
    }
    (dir_path / "meta.json").write_text(json.dumps(meta, indent=2))
    (dir_path / "_DONE").write_text("1")  # written last: marks the dir load-safe


def _load_corpus_dir(dir_path: Path) -> dict[str, SignatureCorpus]:
    corpora = {}
    for p in sorted(dir_path.glob("*.npz")):
        corp = _load_signature_npz(p)
        corpora[corp.key] = corp
    return corpora


def _merge_corpus_dirs(shard_dirs: list[Path]) -> dict[str, SignatureCorpus]:
    """Concatenate every shard's per-signature examples, in shard order.

    Geometry/basis for a signature is reconstructed ONCE, from the FIRST
    shard file that carries it (all shards describing the same signature key
    share identical geometry by construction) — every OTHER shard file of
    that signature is read via the light loader (example arrays only, no
    ``PatchBasis``/``g_pg`` reconstruction).  A full corpus can span dozens of
    shard files; reconstructing the (large) basis once per shard rather than
    once per signature is what OOM'd the first version of this merge.  A
    geometry mismatch across shards is a hard error, never a silent drop.
    Example ids are renumbered contiguously over the merged set, in sorted
    signature-key order, so the result is deterministic regardless of shard
    scan order.
    """
    by_key_light: dict[str, list[dict]] = {}
    ref_path_by_key: dict[str, Path] = {}
    for d in shard_dirs:
        for p in sorted(d.glob("*.npz")):
            light = _load_signature_npz_light(p)
            key = light["key"]
            by_key_light.setdefault(key, []).append(light)
            ref_path_by_key.setdefault(key, p)  # first shard file wins as basis source

    merged: dict[str, SignatureCorpus] = {}
    gid = 0
    for key in sorted(by_key_light):
        parts = by_key_light[key]
        ref_light = parts[0]
        for p in parts[1:]:
            if (
                p["sensor_channels"] != ref_light["sensor_channels"]
                or p["n_cells"] != ref_light["n_cells"]
            ):
                s_p, s_ref = (
                    len(p["sensor_channels"]),
                    len(ref_light["sensor_channels"]),
                )
                raise ValueError(
                    f"signature {key}: geometry differs across shards "
                    f"(S={s_p}/{s_ref}, n_cells={p['n_cells']}/{ref_light['n_cells']}) "
                    "— shards of the same signature must share identical "
                    "geometry by construction"
                )
        ref = _load_signature_npz(ref_path_by_key[key])  # ONE full basis reconstruction
        cat = {
            k: np.concatenate([part[k] for part in parts], axis=0)
            for k in _LIGHT_EXAMPLE_FIELDS
        }
        n = cat["values"].shape[0]
        merged[key] = SignatureCorpus(
            key=key,
            basis=ref.basis,
            sensor_channels=ref.sensor_channels,
            sensor_geometry=ref.sensor_geometry,
            coil_centroids=ref.coil_centroids,
            n_cells=ref.n_cells,
            candidate_mask=ref.candidate_mask,
            ids=np.arange(gid, gid + n, dtype=np.int64),
            **cat,
        )
        gid += n
    return merged


def assemble_corpus_cached(
    shots: list[int],
    *,
    nr: int,
    nz: int,
    t_steps: int,
    stride_s: float,
    min_ip_ka: float,
    cache_root: Path | None = None,
    shard: tuple[int, int] | None = None,
    max_populated_shots: int | None = None,
    force: bool = False,
    operator_out: dict | None = None,
) -> dict[str, SignatureCorpus]:
    """Assemble a corpus, transparently caching to / loading from ``cache_root``.

    ``shard = (i, n)`` restricts assembly to ``shots[i::n]`` and caches that
    slice's partial corpora under its own shard directory (for a CPU-partition
    assembly fleet).  The full (unsharded) cache is produced either directly, or
    — the first time it is requested — by merging every shard directory for the
    same ``(config_hash, n)`` once all ``n`` shards report complete; the merged
    result is itself cached so later loads are a single read.

    ``operator_out`` (optional, see :func:`regenerate_operator_summary`) is
    only populated on a CACHE MISS (an actual ``assemble_corpus`` call) — a
    cache hit reuses previously-collected examples with no fresh
    ``(table, operator)`` build, so it stays empty on that path.
    """
    root = cache_root or _cache_root(None)
    key = _config_hash(
        shots, t_steps=t_steps, stride_s=stride_s, min_ip_ka=min_ip_ka, nr=nr, nz=nz
    )
    base = root / key

    if shard is not None:
        i, n = shard
        shard_shots = list(shots)[i::n]
        shard_dir = base / f"shards_{n}" / f"shard_{i:03d}"
        if not force and _corpus_dir_complete(shard_dir):
            logger.info("CACHE HIT (shard %d/%d, %s): %s", i, n, key, shard_dir)
            return _load_corpus_dir(shard_dir)
        t0 = time.time()
        corpora = assemble_corpus(
            shard_shots,
            nr=nr,
            nz=nz,
            t_steps=t_steps,
            stride_s=stride_s,
            min_ip_ka=min_ip_ka,
            max_populated_shots=max_populated_shots,
            operator_out=operator_out,
            geometry_shots=list(shots),  # the FULL list — see assemble_corpus docstring
        )
        dt = time.time() - t0
        n_ex = sum(c.values.shape[0] for c in corpora.values())
        logger.info(
            "CACHE MISS (shard %d/%d, %s): %d shots -> %d examples in %.1fs "
            "(%.3fs/shot)",
            i,
            n,
            key,
            len(shard_shots),
            n_ex,
            dt,
            dt / max(1, len(shard_shots)),
        )
        _save_corpus_dir(
            shard_dir,
            corpora,
            shots=shard_shots,
            t_steps=t_steps,
            stride_s=stride_s,
            min_ip_ka=min_ip_ka,
            nr=nr,
            nz=nz,
            config_hash=key,
        )
        return corpora

    final_dir = base / "full"
    if not force and _corpus_dir_complete(final_dir):
        logger.info("CACHE HIT (full, %s): %s", key, final_dir)
        return _load_corpus_dir(final_dir)

    if base.exists():
        for shards_dir in sorted(base.glob("shards_*")):
            try:
                n = int(shards_dir.name.split("_", 1)[1])
            except (IndexError, ValueError):
                continue
            shard_dirs = [shards_dir / f"shard_{i:03d}" for i in range(n)]
            if shard_dirs and all(_corpus_dir_complete(d) for d in shard_dirs):
                logger.info("CACHE HIT (merging %d shards, %s): %s", n, key, shards_dir)
                merged = _merge_corpus_dirs(shard_dirs)
                _save_corpus_dir(
                    final_dir,
                    merged,
                    shots=shots,
                    t_steps=t_steps,
                    stride_s=stride_s,
                    min_ip_ka=min_ip_ka,
                    nr=nr,
                    nz=nz,
                    config_hash=key,
                )
                return merged

    t0 = time.time()
    corpora = assemble_corpus(
        shots,
        nr=nr,
        nz=nz,
        t_steps=t_steps,
        stride_s=stride_s,
        min_ip_ka=min_ip_ka,
        max_populated_shots=max_populated_shots,
        operator_out=operator_out,
    )
    dt = time.time() - t0
    n_ex = sum(c.values.shape[0] for c in corpora.values())
    logger.info(
        "CACHE MISS (full, %s): %d shots -> %d examples in %.1fs (%.3fs/shot)",
        key,
        len(shots),
        n_ex,
        dt,
        dt / max(1, len(shots)),
    )
    _save_corpus_dir(
        final_dir,
        corpora,
        shots=shots,
        t_steps=t_steps,
        stride_s=stride_s,
        min_ip_ka=min_ip_ka,
        nr=nr,
        nz=nz,
        config_hash=key,
    )
    return corpora


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


def _encoder_forward(encoder, enc_in, head: str):
    """``(i_cell, i_var)`` — ``i_var`` is ``None`` unless ``head`` is Gaussian,
    in which case the encoder's variance arm is read alongside the mean."""
    if head == "gaussian-direct":
        return encoder(
            enc_in["values"],
            enc_in["finite"],
            enc_in["i_pf_std"],
            enc_in["ip"],
            return_variance=True,
        )
    i_cell = encoder(
        enc_in["values"], enc_in["finite"], enc_in["i_pf_std"], enc_in["ip"]
    )
    return i_cell, None


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
    ap.add_argument(
        "--head",
        type=str,
        default="direct",
        choices=("direct", "lowrank", "gaussian-direct"),
    )
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
    ap.add_argument(
        "--assemble-only",
        action="store_true",
        help="assemble (or load) the corpus cache and exit — no training",
    )
    ap.add_argument(
        "--shot-shard",
        type=str,
        default="",
        help="i/N: assemble only shots[i::N] into their own shard cache entry "
        "(a CPU-partition assembly fleet; shards merge automatically on the "
        "first unsharded load once all N report complete)",
    )
    ap.add_argument(
        "--cache-dir",
        type=str,
        default="",
        help="corpus cache root (default: DEFAULT_CACHE_ROOT, fallback in-repo)",
    )
    ap.add_argument(
        "--no-cache",
        action="store_true",
        help="bypass the corpus cache entirely (always assemble fresh)",
    )
    ap.add_argument(
        "--force-rebuild-cache",
        action="store_true",
        help="ignore any existing cache entry for this config and reassemble",
    )
    ap.add_argument(
        "--regenerate-operator-summary",
        action="store_true",
        help="on a cache MISS, also refresh imas_ambix/gs/artifacts/"
        "gs_operator_summary.json from every signature this assembly builds "
        "(a free side-product — see regenerate_operator_summary)",
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

    shard = None
    if args.shot_shard:
        i_s, n_s = args.shot_shard.split("/")
        shard = (int(i_s), int(n_s))

    operator_out: dict | None = {} if args.regenerate_operator_summary else None
    if args.dry_run or args.no_cache:
        corpora = assemble_corpus(
            train_shots,
            nr=args.nr,
            nz=args.nz,
            t_steps=args.t_steps,
            stride_s=args.stride_ms / 1000.0,
            min_ip_ka=args.min_ip_ka,
            max_populated_shots=2 if args.dry_run else None,
            operator_out=operator_out,
        )
    else:
        corpora = assemble_corpus_cached(
            train_shots,
            nr=args.nr,
            nz=args.nz,
            t_steps=args.t_steps,
            stride_s=args.stride_ms / 1000.0,
            min_ip_ka=args.min_ip_ka,
            cache_root=_cache_root(args.cache_dir) if args.cache_dir else None,
            shard=shard,
            force=args.force_rebuild_cache,
            operator_out=operator_out,
        )
    if not corpora:
        logger.error("no usable training examples assembled — aborting")
        return 1

    if args.regenerate_operator_summary:
        if operator_out:
            regenerate_operator_summary(operator_out)
        else:
            logger.warning(
                "--regenerate-operator-summary requested but this assembly "
                "was a cache HIT (no fresh table/operator built) — rerun with "
                "--force-rebuild-cache to actually regenerate"
            )

    if args.assemble_only:
        n_total = sum(c.values.shape[0] for c in corpora.values())
        logger.info(
            "assemble-only: %d examples across %d signatures", n_total, len(corpora)
        )
        for key, corp in corpora.items():
            logger.info(
                "  signature %s: %d examples, S=%d, n_cells=%d",
                key,
                corp.values.shape[0],
                len(corp.sensor_channels),
                corp.n_cells,
            )
        return 0

    # one shared encoder trains against EVERY campaign signature in the corpus
    # (per-batch geometry binding — see _bind_signature); no signature is
    # dropped for a differing sensor/coil count.  "reference" below only labels
    # the dominant signature for checkpoint bookkeeping + in/cross-signature
    # eval reporting — it plays no other privileged role.
    ref_key = max(corpora, key=lambda k: corpora[k].values.shape[0])
    ref = corpora[ref_key]
    coil_widths = {key: int(corp.i_pf.shape[1]) for key, corp in corpora.items()}
    if len(set(coil_widths.values())) > 1:
        raise ValueError(
            f"PF coil count differs across signatures: {coil_widths} — the "
            "corpus assumes identical PF circuit topology across campaign "
            "signatures (only the coil-coupling matrices differ), so a single "
            "i_pf standardisation cannot serve a mismatched coil count"
        )
    # renumber example ids contiguously (a defensive no-op for a fresh
    # assemble_corpus() call, which already numbers contiguously; load-bearing
    # after a cache merge/resume path where a future change might not)
    gid = 0
    for corp in corpora.values():
        n = corp.values.shape[0]
        corp.ids = np.arange(gid, gid + n, dtype=np.int64)
        gid += n
    n_total = sum(c.values.shape[0] for c in corpora.values())
    logger.info("corpus: %d examples across %d signatures", n_total, len(corpora))

    stats_by_name = token_channel_stats_by_name(corpora)
    per_sig_stats = {
        key: channel_stats_for_signature(corp.sensor_channels, stats_by_name)
        for key, corp in corpora.items()
    }
    ch_mean, ch_std = per_sig_stats[ref_key]
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
        # audit-only additions (the eval script keys above are unchanged and
        # remain sufficient on their own): the full by-name channel stat table
        # and every signature's own geometry, so a signature that trained but
        # is not the checkpoint's "reference" is still fully recoverable.
        "channel_stats_by_name": {k: list(v) for k, v in stats_by_name.items()},
        "per_signature_geometry": {
            key: {
                "sensor_channels": corp.sensor_channels,
                "sensor_geometry": corp.sensor_geometry,
                "coil_centroids": corp.coil_centroids,
                "n_cells": corp.n_cells,
                "candidate_mask": corp.candidate_mask,
                "n_examples": int(corp.values.shape[0]),
            }
            for key, corp in corpora.items()
        },
    }

    start_epoch = 0
    global_step = 0
    if args.resume and ckpt_path.exists():
        payload = torch.load(ckpt_path, map_location=device, weights_only=False)
        # geometry buffers are rebound per batch (see _bind_signature) and are
        # never part of the learned identity — exclude them from the strict
        # check exactly as the gate-eval script does on load.
        learned = {
            k: v for k, v in payload["encoder"].items() if k not in _GEOMETRY_BUFFERS
        }
        missing, unexpected = encoder.load_state_dict(learned, strict=False)
        missing = [m for m in missing if m not in _GEOMETRY_BUFFERS]
        if missing or unexpected:
            raise RuntimeError(
                f"resume weight load mismatch: missing={missing} "
                f"unexpected={unexpected}"
            )
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
        _bind_signature(encoder, corp, device)
        sig_ch_mean, sig_ch_std = per_sig_stats[key]
        rows = np.arange(min(args.batch_size, corp.values.shape[0]))
        enc_in, payload = _make_batch(
            corp, rows, sig_ch_mean, sig_ch_std, ipf_mean, ipf_std, device
        )
        lam = disc.get(torch.as_tensor(corp.ids[rows], device=device))
        i_cell, i_var = _encoder_forward(encoder, enc_in, args.head)
        losses = amortised_losses(corp.basis, i_cell, lam=lam, i_var=i_var, **payload)
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
            _bind_signature(encoder, corp, device)
            sig_ch_mean, sig_ch_std = per_sig_stats[key]
            enc_in, payload = _make_batch(
                corp, rows, sig_ch_mean, sig_ch_std, ipf_mean, ipf_std, device
            )
            ids = torch.as_tensor(corp.ids[rows], device=device)
            lam = disc.get(ids)
            optimizer.zero_grad()
            i_cell, i_var = _encoder_forward(encoder, enc_in, args.head)
            losses = amortised_losses(corp.basis, i_cell, lam=lam, i_var=i_var, **payload)
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
                # rebind to the reference signature first so the saved buffers
                # (excluded from any load anyway — see _GEOMETRY_BUFFERS) are
                # deterministic rather than whatever batch happened to run last
                _bind_signature(encoder, ref, device)
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
                _bind_signature(encoder, corp, device)
                sig_ch_mean, sig_ch_std = per_sig_stats[key]
                enc_in, payload = _make_batch(
                    corp, rows, sig_ch_mean, sig_ch_std, ipf_mean, ipf_std, device
                )
                lam = torch.zeros(len(rows), device=device)
                i_cell, i_var = _encoder_forward(encoder, enc_in, args.head)
                losses = amortised_losses(
                    corp.basis, i_cell, lam=lam, i_var=i_var, **payload
                )
                report = {
                    k: round(float(v.mean()), 5)
                    for k, v in losses.items()
                    if torch.is_tensor(v)
                }
                logger.info("epoch %d HELD-BACK eval %s", epoch, report)

    _bind_signature(encoder, ref, device)
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
