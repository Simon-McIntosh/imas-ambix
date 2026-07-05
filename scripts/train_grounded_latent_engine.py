#!/usr/bin/env python
"""Corpus-train the GS-grounded hybrid latent on the patch-current substrate.

Trains ONE shared :class:`~imas_ambix.latent.encoder.HybridLatentEncoder`
across every campaign signature's
:class:`~imas_ambix.latent.engine.GSGroundedLatentEngine`
(:class:`~imas_ambix.latent.training.CorpusTrainer`), on the composite
raw-signal objective the engine defines: whitened magnetics misfit + a
Rogowski Ip anchor + a per-example bounded-discrepancy-weighted GS structure
residual + flux-diffusion transport guard-rails + anchored/dimensionless/KL
regularisers + the self-supervised closure-coordinate readout — using ONLY
training shots (the firewalled EFIT referee is never read here).

This is the patch-current-substrate successor of the deleted
``train_gs_grounded_latent.py`` (which trained against the low-DOF polynomial-
theta :class:`~imas_ambix.latent.gs_observation.GSObservation` carrier — see
``git show 40d30ff^:scripts/train_gs_grounded_latent.py``): every physical
input the loss consumes is assembled the same way
``scripts/train_patch_encoder.py`` assembles it (raw magnetics aligned BY NAME
to the basis' sensor channels, per-shot ``nanstd`` scale, masks taken
pre-fill), but the training EXAMPLE here is a consecutive ``(t, t+1)`` slice
pair (:func:`imas_ambix.latent.training.consecutive_pairs`) rather than a
single centred window, because the engine's transport prior needs both ends
of a transition.

The command channel driving the transport prior's inductive source is the
per-coil current derivative ``dI_pf/dt`` (a loop-voltage proxy) -- a
documented modelling choice carried over unchanged from the deleted script.

Only campaign signatures whose patch-current cell count matches the corpus's
most populous signature are trained (the shared encoder's per-cell head has
one fixed output width -- a real mismatch is a config error worth surfacing
via the corpus-assembly log, not a silent drop of the whole run).

In-process, SIGTERM-clean (atomic checkpoint on signal), resume-safe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import signal
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from imas_ambix.eval.excitation_selector import coil_ramp_profile
from imas_ambix.gs.geometry import discover_signatures, extract_campaign_tables

try:  # GEOMETRY_TABLE_VERSION is a very recent addition (sensor-channel-set
    # determinism fix) -- degrade to an honest "absent" label rather than a
    # hard import error if an older checkout doesn't have it yet.  The
    # discover_signatures / extract_campaign_tables functions above landed in
    # the SAME commit, so their absence would already have failed the import
    # above -- only the version STRING (used for labelling/cache-keying) needs
    # this softer fallback.
    from imas_ambix.gs.geometry import GEOMETRY_TABLE_VERSION
except ImportError:
    GEOMETRY_TABLE_VERSION = None
from imas_ambix.gs.operator import COIL_MODEL_VERSION, build_operator
from imas_ambix.latent.data import (
    ANCHORED_NAMES,
    CHANNEL_SCALE_KIND_FLOOR_REL,
    CorpusStats,
    feature_schema,
    fit_corpus_stats,
    load_shot_windows,
    read_split_shot_lists,
    robust_channel_scale,
)
from imas_ambix.latent.encoder import HybridLatentEncoder, LatentConfig
from imas_ambix.latent.engine import GSGroundedLatentEngine, LossWeights
from imas_ambix.latent.patch_basis import PatchBasis
from imas_ambix.latent.patch_encoder import DiscrepancyLambda
from imas_ambix.latent.training import CorpusTrainer, consecutive_pairs
from imas_ambix.latent.transport import FluxDiffusionPrior

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train_grounded_latent_engine")

DEFAULT_ARTIFACT_ROOT = Path("/work/projects/imas_gpu/latent/grounded_latent_engine")
FALLBACK_ARTIFACT_ROOT = Path("imas_ambix/latent/artifacts/grounded_latent")
DEFAULT_CACHE_ROOT = DEFAULT_ARTIFACT_ROOT / "corpus_cache"
FALLBACK_CACHE_ROOT = FALLBACK_ARTIFACT_ROOT / "corpus_cache"

_PAIR_ARRAY_KEYS = (
    "x_t",
    "x_tp1",
    "i_pf_t",
    "i_pf_tp1",
    "ip_t",
    "ip_tp1",
    "raw_mag_t",
    "mag_mask_t",
    "sensor_scale",
    "anchored_t",
    "cmd_t",
    "dt",
    "weight",
)


# --------------------------------------------------------------------------- #
#  Corpus assembly: per-campaign PatchBasis + consecutive-pair examples       #
# --------------------------------------------------------------------------- #
@dataclass
class SignatureCorpus:
    """One campaign signature's consecutive-pair examples + its PatchBasis."""

    key: str
    basis: PatchBasis
    n_coil: int
    x_t: np.ndarray = field(default=None)  # (N, n_feat)
    x_tp1: np.ndarray = field(default=None)  # (N, n_feat)
    i_pf_t: np.ndarray = field(default=None)  # (N, n_coil)
    i_pf_tp1: np.ndarray = field(default=None)  # (N, n_coil)
    ip_t: np.ndarray = field(default=None)  # (N,) [A]
    ip_tp1: np.ndarray = field(default=None)  # (N,) [A]
    raw_mag_t: np.ndarray = field(default=None)  # (N, n_sensor)
    mag_mask_t: np.ndarray = field(default=None)  # (N, n_sensor) bool
    sensor_scale: np.ndarray = field(default=None)  # (N, n_sensor)
    anchored_t: np.ndarray = field(default=None)  # (N, n_anchored) raw
    cmd_t: np.ndarray = field(default=None)  # (N, n_coil) raw dI/dt
    dt: np.ndarray = field(default=None)  # (N,)
    weight: np.ndarray = field(default=None)  # (N,) excitation sample weight
    ids: np.ndarray = field(default=None)  # (N,) contiguous global ids

    @property
    def n_cells(self) -> int:
        return int(self.basis.r_cells.shape[0])

    @property
    def n_examples(self) -> int:
        return 0 if self.x_t is None else int(self.x_t.shape[0])


def _shot_pairs_for_operator(
    shot: int,
    fwd,
    key: str,
    schema: dict[str, list[str]],
    *,
    min_ip_ka: float,
):
    """Consecutive-pair example arrays for one shot against a signature's
    ALREADY-BUILT canonical operator, or ``None``.

    ``fwd`` (and the ``key`` it was built from) is shared across every shot of
    the signature — built once, in :func:`assemble_corpus`, from a
    :func:`~imas_ambix.gs.geometry.extract_campaign_tables` table whose sensor
    channel SET is the canonical union over every shot of that signature
    (:data:`~imas_ambix.gs.geometry.GEOMETRY_TABLE_VERSION`).  A per-shot
    ``build_table_for_shot(shot)`` call here would silently reintroduce the
    single-shot indeterminism the union fixes (its ``amb_channels`` argument
    defaults to that ONE shot's own schema) — this is why the operator is
    passed in rather than rebuilt.

    The cached ``sensor_scale`` here is the RAW per-shot ``nanstd`` (with only
    the pre-existing isfinite/positive -> 1.0 fallback) — the kind-median
    floor (:func:`~imas_ambix.latent.data.robust_channel_scale`) is applied
    downstream at batch-construction time (:func:`_build_batch`), matching
    ``scripts/train_patch_encoder.py``'s convention: the cache stays valid
    across a floor-formula change, since no floor is baked into it.
    """
    w = load_shot_windows(int(shot), fwd, key, schema, with_referee=False)
    if w is None or w.times.size < 2:
        return None
    pairs = consecutive_pairs(w.times)
    if not pairs:
        return None

    scale_row = np.nanstd(w.raw_mag, axis=0)
    scale_row = np.where(np.isfinite(scale_row) & (scale_row > 0), scale_row, 1.0)
    # excitation weighting: |dI/dt| at the pair's start slice (the transport
    # prior's command is most informative on ramps) — carried over from the
    # deleted train_gs_grounded_latent.py's excitation-weighted sampling.
    ramp = coil_ramp_profile(w.i_pf, w.times)

    x_t, x_tp1, i_pf_t, i_pf_tp1 = [], [], [], []
    ip_t, ip_tp1, raw_mag_t, mag_mask_t = [], [], [], []
    anchored_t, cmd_t, dts, weight = [], [], [], []
    for a, b, dt_val in pairs:
        if abs(float(w.anchored[a, 0])) <= min_ip_ka:
            continue
        x_t.append(w.features_raw[a])
        x_tp1.append(w.features_raw[b])
        i_pf_t.append(w.i_pf[a])
        i_pf_tp1.append(w.i_pf[b])
        ip_t.append(abs(float(w.anchored[a, 0])) * 1e3)
        ip_tp1.append(abs(float(w.anchored[b, 0])) * 1e3)
        raw_mag_t.append(w.raw_mag[a])
        mag_mask_t.append(w.mag_mask[a])
        anchored_t.append(w.anchored[a])
        cmd_t.append((w.i_pf[b] - w.i_pf[a]) / max(dt_val, 1e-9))
        dts.append(dt_val)
        weight.append(float(ramp[a]))
    if not x_t:
        return None

    weight_arr = np.asarray(weight, dtype=np.float64)
    weight_arr = weight_arr + 1e-6 * weight_arr.max() + 1e-9  # never fully zero
    ex = {
        "x_t": np.asarray(x_t, dtype=np.float64),
        "x_tp1": np.asarray(x_tp1, dtype=np.float64),
        "i_pf_t": np.asarray(i_pf_t, dtype=np.float64),
        "i_pf_tp1": np.asarray(i_pf_tp1, dtype=np.float64),
        "ip_t": np.asarray(ip_t, dtype=np.float64),
        "ip_tp1": np.asarray(ip_tp1, dtype=np.float64),
        "raw_mag_t": np.asarray(raw_mag_t, dtype=np.float64),
        "mag_mask_t": np.asarray(mag_mask_t, dtype=bool),
        "sensor_scale": np.tile(scale_row, (len(x_t), 1)),
        "anchored_t": np.asarray(anchored_t, dtype=np.float64),
        "cmd_t": np.asarray(cmd_t, dtype=np.float64),
        "dt": np.asarray(dts, dtype=np.float64),
        "weight": weight_arr,
    }
    return ex


def assemble_corpus(
    shots: list[int],
    *,
    nr: int,
    nz: int,
    min_ip_ka: float,
    max_populated_shots: int | None = None,
) -> dict[str, SignatureCorpus]:
    """Build per-signature :class:`SignatureCorpus` bundles for ``shots``.

    One :class:`~imas_ambix.gs.geometry.GeometryTable` (hence one
    :class:`~imas_ambix.gs.operator.ForwardOperator` and one
    :class:`~imas_ambix.latent.patch_basis.PatchBasis`) is built PER SIGNATURE
    via :func:`~imas_ambix.gs.geometry.extract_campaign_tables`, which unions
    the sensor channel set across every shot of that signature — never per
    shot.  A per-shot ``build_table_for_shot(shot)`` call (the original,
    pre-fix shape of this function) silently falls back to that ONE shot's own
    amb schema and reintroduces the sensor-channel-set indeterminism the union
    exists to close; two real corpus-assembly crashes (jobs 1225204, 1225258)
    were exactly this bug.
    """
    schema = feature_schema()
    groups = discover_signatures(shots)  # {key: (sig, [shot_ids])}
    tables = extract_campaign_tables(shots)  # {key: canonical GeometryTable}
    basis_cache: dict[str, PatchBasis] = {}
    per_sig: dict[str, list[dict]] = {}
    n_pf: dict[str, int] = {}
    n_populated = 0
    for key, (_sig, shot_list) in groups.items():
        table = tables.get(key)
        if table is None:
            logger.warning("signature %s: no buildable representative — skipped", key)
            continue
        try:
            fwd = build_operator(table)
            basis_cache[key] = PatchBasis.from_table(table, nr=nr, nz=nz)
        except Exception as exc:  # noqa: BLE001 — a signature w/o a valid forward
            logger.warning("signature %s: operator/basis build failed: %s", key, exc)
            continue
        for s in shot_list:
            try:
                ex = _shot_pairs_for_operator(s, fwd, key, schema, min_ip_ka=min_ip_ka)
            except Exception as exc:  # noqa: BLE001 — a shot w/o usable data
                logger.warning("shot %s skipped: %s", s, exc)
                continue
            if ex is not None:
                per_sig.setdefault(key, []).append(ex)
                n_pf.setdefault(key, ex["i_pf_t"].shape[1])
                n_populated += 1
                if (
                    max_populated_shots is not None
                    and n_populated >= max_populated_shots
                ):
                    break
        if max_populated_shots is not None and n_populated >= max_populated_shots:
            break

    corpora: dict[str, SignatureCorpus] = {}
    for key, ex_list in per_sig.items():
        cat = {
            k: np.concatenate([e[k] for e in ex_list], axis=0) for k in _PAIR_ARRAY_KEYS
        }
        n = cat["x_t"].shape[0]
        # LOCAL 0..n-1 ids, not a corpus-wide running counter: each signature
        # gets its OWN DiscrepancyLambda buffer sized to exactly this n (see
        # main()), and that buffer is indexed by these ids directly. A shared
        # global counter across signatures (the pre-fix shape of this line)
        # produced ids >= n for every signature after the first, which
        # silently indexed past a smaller buffer -- a CUDA device-side assert
        # on GPU, invisible until job 1225426's 37-minute run hit it.
        corpora[key] = SignatureCorpus(
            key=key,
            basis=basis_cache[key],
            n_coil=n_pf[key],
            ids=np.arange(n, dtype=np.int64),
            **cat,
        )
        logger.info(
            "signature %s: %d pairs, n_cells=%d, n_coil=%d",
            key,
            n,
            corpora[key].n_cells,
            n_pf[key],
        )
    return corpora


# --------------------------------------------------------------------------- #
#  Corpus cache (single directory, no sharding — see module docstring)        #
# --------------------------------------------------------------------------- #
def _config_hash(shots: list[int], *, nr: int, nz: int, min_ip_ka: float) -> str:
    payload = {
        "shots": sorted(int(s) for s in shots),
        "nr": int(nr),
        "nz": int(nz),
        "min_ip_ka": round(float(min_ip_ka), 6),
        "coil_model_version": COIL_MODEL_VERSION,
        # a sensor-channel-set fix on a FIXED signature digest (fc938's
        # per-shot flux-loop presence) — folded in so a cache built before the
        # fix can never be silently reused after it; None (pre-fix checkout)
        # still busts a post-fix cache since it differs from the real string
        "geometry_table_version": GEOMETRY_TABLE_VERSION,
        # NOTE: the sensor-scale kind-median floor (robust_channel_scale) is
        # NOT part of this key -- it is applied at batch-construction time
        # against the cached RAW scale, so a floor-formula change picks up
        # automatically without invalidating an existing corpus cache.
    }
    blob = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def _cache_root(explicit: str | Path | None) -> Path:
    if explicit:
        return Path(explicit)
    if DEFAULT_CACHE_ROOT.parent.parent.exists():
        return DEFAULT_CACHE_ROOT
    return FALLBACK_CACHE_ROOT


def _patch_basis_kwargs(basis: PatchBasis) -> dict:
    """``PatchBasis`` constructor kwargs as plain numpy — round-trips with no
    IMAS access (every field is a pure geometry-derived matrix)."""
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


def _save_signature_npz(path: Path, corp: SignatureCorpus) -> None:
    bk = _patch_basis_kwargs(corp.basis)
    candidate_mask = np.asarray(
        corp.basis.candidate_mask.detach().cpu().numpy(), dtype=np.float64
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / (path.stem + ".tmp.npz")
    np.savez(
        tmp,
        key=np.asarray(corp.key),
        n_coil=np.asarray(corp.n_coil),
        ids=corp.ids,
        **{k: getattr(corp, k) for k in _PAIR_ARRAY_KEYS},
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
        basis_candidate_mask=candidate_mask,
    )
    tmp.replace(path)  # atomic on the same filesystem


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
        candidate_mask=d["basis_candidate_mask"],
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
        n_coil=int(d["n_coil"]),
        ids=d["ids"],
        **{k: d[k] for k in _PAIR_ARRAY_KEYS},
    )


def _corpus_dir_complete(dir_path: Path) -> bool:
    return (dir_path / "_DONE").exists()


def _save_corpus_dir(
    dir_path: Path, corpora: dict[str, SignatureCorpus], *, config_hash: str
) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    for key, corp in corpora.items():
        _save_signature_npz(dir_path / f"{key}.npz", corp)
    meta = {
        "config_hash": config_hash,
        "coil_model_version": COIL_MODEL_VERSION,
        "geometry_table_version": GEOMETRY_TABLE_VERSION,
        # cached scale is RAW nanstd -- robust_channel_scale is applied later,
        # at batch-construction time (see ckpt_extra's sensor_scale_floor)
        "signatures": list(corpora.keys()),
        "n_examples": {k: c.n_examples for k, c in corpora.items()},
    }
    (dir_path / "meta.json").write_text(json.dumps(meta, indent=2))
    (dir_path / "_DONE").write_text("1")  # written last: marks the dir load-safe


def _load_corpus_dir(dir_path: Path) -> dict[str, SignatureCorpus]:
    return {
        (c := _load_signature_npz(p)).key: c for p in sorted(dir_path.glob("*.npz"))
    }


def assemble_corpus_cached(
    shots: list[int],
    *,
    nr: int,
    nz: int,
    min_ip_ka: float,
    cache_root: Path | None = None,
    force: bool = False,
) -> dict[str, SignatureCorpus]:
    root = cache_root or _cache_root(None)
    key = _config_hash(shots, nr=nr, nz=nz, min_ip_ka=min_ip_ka)
    final_dir = root / key / "full"
    if not force and _corpus_dir_complete(final_dir):
        logger.info("CACHE HIT (%s): %s", key, final_dir)
        return _load_corpus_dir(final_dir)

    t0 = time.time()
    corpora = assemble_corpus(shots, nr=nr, nz=nz, min_ip_ka=min_ip_ka)
    dt = time.time() - t0
    n_ex = sum(c.n_examples for c in corpora.values())
    logger.info(
        "CACHE MISS (%s): %d shots -> %d pairs in %.1fs (%.3fs/shot)",
        key,
        len(shots),
        n_ex,
        dt,
        dt / max(1, len(shots)),
    )
    _save_corpus_dir(final_dir, corpora, config_hash=key)
    return corpora


# --------------------------------------------------------------------------- #
#  Batch sampling                                                             #
# --------------------------------------------------------------------------- #
@dataclass
class _CampaignState:
    corp: SignatureCorpus
    engine: GSGroundedLatentEngine
    cmd_stats: CorpusStats
    sample_p: np.ndarray  # normalised excitation sampling weights (held_idx zeroed)
    steps_per_epoch: int
    disc: DiscrepancyLambda
    held_idx: np.ndarray  # (n_held,) rows never sampled for training


def _held_back_split(
    n_examples: int, frac: float, rng: np.random.Generator
) -> np.ndarray:
    """A FIXED subset of row indices, held out of training entirely (never
    fed to :func:`_build_batch`'s sampler) so a periodic evaluation on them
    is a genuine held-back check, not just a re-read of trained-on rows."""
    n_held = max(1, min(n_examples - 1, int(round(frac * n_examples))))
    return rng.choice(n_examples, size=n_held, replace=False)


def _build_batch(
    state: _CampaignState,
    feature_stats: CorpusStats,
    anchored_stats: CorpusStats,
    batch_size: int,
    rng: np.random.Generator,
    device: torch.device,
    *,
    rows: np.ndarray | None = None,
) -> tuple[dict, np.ndarray]:
    """One (t, t+1) minibatch for one campaign; returns ``(batch, row_idx)``.

    ``rows`` (optional): use these EXACT rows instead of sampling -- the
    held-back evaluation path (:func:`_held_back_physics_misfit`) passes its
    fixed, never-trained-on indices here.
    """
    corp = state.corp
    n = corp.n_examples
    if rows is None:
        # a campaign with fewer pairs than batch_size still fills a FULL batch
        # (with-replacement); a larger campaign samples without replacement so
        # a batch never repeats a row
        rows = rng.choice(n, size=batch_size, p=state.sample_p, replace=n < batch_size)

    x_t = np.nan_to_num(feature_stats.normalise(corp.x_t[rows]), nan=0.0)
    x_tp1 = np.nan_to_num(feature_stats.normalise(corp.x_tp1[rows]), nan=0.0)
    anchored_arr = anchored_stats.normalise(corp.anchored_t[rows])
    anchored_mask = np.isfinite(anchored_arr)
    anchored_arr = np.nan_to_num(anchored_arr, nan=0.0)
    cmd_arr = state.cmd_stats.normalise(corp.cmd_t[rows])
    mag_mask = corp.mag_mask_t[rows]
    ids = corp.ids[rows]
    lam = state.disc.get(ids).to(device=device)
    # kind-median floor on the RAW cached scale (never baked into the cache
    # itself -- see _shot_pairs_for_operator) so a modest vacuum/measured
    # mismatch on an occasionally-near-zero-variance channel cannot whiten to
    # thousands of sigma and dominate the batch loss.
    scale_floored = robust_channel_scale(
        corp.sensor_scale[rows], corp.basis.sensor_channels
    )

    def t(a, dtype=torch.float64):
        return torch.as_tensor(np.asarray(a), dtype=dtype, device=device)

    batch = {
        "x_t": t(x_t),
        "x_tp1": t(x_tp1),
        "i_pf_t": t(corp.i_pf_t[rows]),
        "i_pf_tp1": t(corp.i_pf_tp1[rows]),
        "ip_t": t(corp.ip_t[rows]),
        "ip_tp1": t(corp.ip_tp1[rows]),
        "raw_mag_t": t(np.nan_to_num(corp.raw_mag_t[rows], nan=0.0)),
        "sensor_scale": t(scale_floored),
        "mag_mask": t(mag_mask, dtype=torch.bool),
        "structure_lam": lam,
        "cmd_t": t(cmd_arr),
        "anchored_target": t(anchored_arr),
        "anchored_mask": t(anchored_mask, dtype=torch.bool),
        "dt": float(np.mean(corp.dt[rows])),
    }
    return batch, rows


def _per_example_magnetics_misfit(
    engine: GSGroundedLatentEngine, batch: dict
) -> torch.Tensor:
    """Per-example whitened magnetics misfit (post-update), for the lambda
    schedule — :meth:`GSGroundedLatentEngine.magnetics_loss` reduces over the
    batch, so this is computed directly rather than by editing the engine."""
    with torch.no_grad():
        lat = engine.encode(batch["x_t"])
        i_cell = engine.i_cell_from_latent(lat, batch["ip_t"])
        pred = engine.predict_magnetics(i_cell, batch["i_pf_t"])
        raw = torch.nan_to_num(batch["raw_mag_t"], nan=0.0)
        resid = (pred - raw) / batch["sensor_scale"].clamp_min(1e-12)
        m = batch["mag_mask"].to(resid.dtype)
        return (resid**2 * m).sum(-1) / m.sum(-1).clamp_min(1.0)


def _held_back_physics_misfit(
    state: _CampaignState,
    feature_stats: CorpusStats,
    anchored_stats: CorpusStats,
    device: torch.device,
) -> float:
    """Mean (magnetics + anchored) loss on rows NEVER fed to the sampler --
    the "physics we care about" signal the f-malwm-02 gate reads, deliberately
    EXCLUDING structure_residual/closure so a lambda-ramp or an auxiliary-loss
    divergence (job 1225447) cannot masquerade as (or hide) genuine physics
    progress in the checkpoint-selection metric."""
    batch, _rows = _build_batch(
        state,
        feature_stats,
        anchored_stats,
        batch_size=0,
        rng=np.random.default_rng(0),
        device=device,
        rows=state.held_idx,
    )
    with torch.no_grad():
        out = state.engine.losses(batch)
        return float(out["magnetics"] + out["anchored"])


# --------------------------------------------------------------------------- #
#  Checkpointing                                                              #
# --------------------------------------------------------------------------- #
def _disc_state(disc) -> dict:
    return {
        "lam": disc.lam.detach().cpu(),
        "target": disc.target.detach().cpu(),
        "warm_misfit": disc._warm_misfit.detach().cpu(),
        "epoch": int(disc._epoch),
    }


def _restore_disc(disc, state: dict) -> None:
    disc.lam = state["lam"].to(disc.device, disc.dtype)
    disc.target = state["target"].to(disc.device, disc.dtype)
    disc._warm_misfit = state["warm_misfit"].to(disc.device, disc.dtype)
    disc._epoch = int(state["epoch"])


def _lr_lambda(total_steps: int, warmup_frac: float, floor_frac: float):
    """Warmup-then-cosine LR multiplier (matches train_patch_encoder.py's
    schedule) -- cheap insurance against a constant, non-decaying LR
    sustaining a divergence once one starts, per job 1225447's post-mortem."""
    warmup = max(1, int(warmup_frac * total_steps))

    def fn(step: int) -> float:
        if step < warmup:
            return step / warmup
        prog = (step - warmup) / max(1, total_steps - warmup)
        cos = 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))
        return floor_frac + (1 - floor_frac) * cos

    return fn


def main() -> int:  # noqa: PLR0915 — a single-file training driver
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-train", type=int, default=1000)
    ap.add_argument("--n-heldout", type=int, default=8)
    ap.add_argument("--nr", type=int, default=65)
    ap.add_argument("--nz", type=int, default=97)
    ap.add_argument("--min-ip-ka", type=float, default=100.0)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument(
        "--warmup-frac",
        type=float,
        default=0.05,
        help="LR warmup fraction of --steps before the cosine decay begins",
    )
    ap.add_argument(
        "--lr-floor-frac",
        type=float,
        default=0.1,
        help="LR floor as a fraction of the peak, at the end of the cosine decay",
    )
    ap.add_argument(
        "--held-back-frac",
        type=float,
        default=0.02,
        help="per-campaign fraction of examples NEVER sampled for training, "
        "reserved for the periodic best-checkpoint physics-misfit check",
    )
    ap.add_argument("--n-free", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument(
        "--n-closure-bins",
        type=int,
        default=24,
        help="closure-coordinate head width (0 disables it)",
    )
    ap.add_argument("--closure-weight", type=float, default=0.05)
    ap.add_argument("--lam0", type=float, default=3.0)
    ap.add_argument("--lambda-ratio", type=float, default=1.5)
    ap.add_argument("--lam-max", type=float, default=100.0)
    ap.add_argument("--lambda-warmup-epochs", type=int, default=3)
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--run", type=str, default="direct")
    ap.add_argument("--artifact-root", type=str, default="")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache-dir", type=str, default="")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--force-rebuild-cache", action="store_true")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="assemble a few shots, run one CPU step, print the loss dict",
    )
    ap.add_argument(
        "--assemble-only",
        action="store_true",
        help="assemble (or load) the corpus cache and exit — no training",
    )
    args = ap.parse_args()

    device = torch.device("cpu" if args.dry_run else args.device)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    train_shots, _held = read_split_shot_lists(args.n_train, args.n_heldout)
    if args.dry_run:
        train_shots = train_shots[:16]
    logger.info(
        "training on %d shots (held-out excluded, referee never read)", len(train_shots)
    )

    if args.dry_run or args.no_cache:
        corpora = assemble_corpus(
            train_shots,
            nr=args.nr,
            nz=args.nz,
            min_ip_ka=args.min_ip_ka,
            max_populated_shots=4 if args.dry_run else None,
        )
    else:
        corpora = assemble_corpus_cached(
            train_shots,
            nr=args.nr,
            nz=args.nz,
            min_ip_ka=args.min_ip_ka,
            cache_root=_cache_root(args.cache_dir) if args.cache_dir else None,
            force=args.force_rebuild_cache,
        )
    corpora = {k: c for k, c in corpora.items() if c.n_examples}
    if not corpora:
        logger.error("no campaign has usable training pairs — aborting")
        return 1
    # self-heal: a corpus cache built before the local-id fix (job 1225426's
    # crash) carries a corpus-WIDE running id count, not this signature's own
    # 0..n-1 range that its DiscrepancyLambda buffer is sized to -- renumber
    # unconditionally rather than trusting whatever is on disk.
    for corp in corpora.values():
        corp.ids = np.arange(corp.n_examples, dtype=np.int64)

    # the shared encoder's per-cell head has ONE fixed output width — only
    # signatures matching the dominant (most populous) signature's cell count
    # can train against it (see the module docstring).
    ref_key = max(corpora, key=lambda k: corpora[k].n_examples)
    ref_n_cells = corpora[ref_key].n_cells
    dropped = {k: c.n_cells for k, c in corpora.items() if c.n_cells != ref_n_cells}
    if dropped:
        logger.warning(
            "dropping %d signature(s) with n_cells != reference (%d): %s",
            len(dropped),
            ref_n_cells,
            dropped,
        )
    corpora = {k: c for k, c in corpora.items() if c.n_cells == ref_n_cells}

    coil_widths = {k: c.n_coil for k, c in corpora.items()}
    if len(set(coil_widths.values())) > 1:
        logger.warning(
            "PF coil count differs across signatures %s — the shared "
            "cmd_stats is still fit per campaign (this only affects transport "
            "cmd_dim, which is per-engine already)",
            coil_widths,
        )

    if args.assemble_only:
        for key, corp in corpora.items():
            logger.info(
                "  signature %s: %d pairs, n_cells=%d, n_coil=%d",
                key,
                corp.n_examples,
                corp.n_cells,
                corp.n_coil,
            )
        return 0

    n_features = next(iter(corpora.values())).x_t.shape[1]
    feature_stats = fit_corpus_stats(
        [c.x_t for c in corpora.values()] + [c.x_tp1 for c in corpora.values()]
    )
    anchored_stats = fit_corpus_stats([c.anchored_t for c in corpora.values()])

    encoder = HybridLatentEncoder(
        LatentConfig(
            n_features=n_features,
            n_theta=1,  # unused by the patch-current carrier; kept for shape parity
            n_anchored=len(ANCHORED_NAMES),
            n_free=args.n_free,
            n_cells=ref_n_cells,
            n_closure_bins=args.n_closure_bins,
            hidden=args.hidden,
            depth=args.depth,
        )
    ).double()

    campaign_states: dict[str, _CampaignState] = {}
    engines: dict[str, GSGroundedLatentEngine] = {}
    for key, corp in corpora.items():
        basis = corp.basis.double()
        transport = FluxDiffusionPrior(
            nrho=basis.nr, cmd_dim=max(corp.n_coil, 1), feat_dim=args.n_free
        ).double()
        weights = LossWeights(closure=args.closure_weight)
        engine = GSGroundedLatentEngine(encoder, basis, transport, weights=weights)
        engines[key] = engine
        cmd_stats = fit_corpus_stats([corp.cmd_t])
        held_idx = _held_back_split(corp.n_examples, args.held_back_frac, rng)
        weight = corp.weight.copy()
        weight[held_idx] = 0.0  # held-back rows are NEVER sampled for training
        sample_p = weight / weight.sum()
        steps_per_epoch = max(1, int(round(corp.n_examples / args.batch_size)))
        # fail loudly on CPU (job 1225426's mismatch was a CUDA device-side
        # assert 37 minutes into a run) -- DiscrepancyLambda's buffer below is
        # sized to exactly corp.n_examples, indexed by corp.ids directly.
        if corp.ids.min() < 0 or corp.ids.max() >= corp.n_examples:
            raise ValueError(
                f"signature {key}: ids range [{corp.ids.min()}, {corp.ids.max()}] "
                f"is not contained in [0, {corp.n_examples}) -- the "
                "DiscrepancyLambda buffer below would be indexed out of bounds"
            )
        disc = DiscrepancyLambda(
            corp.n_examples,
            lam0=args.lam0,
            ratio=args.lambda_ratio,
            lam_max=args.lam_max,
            warmup_epochs=args.lambda_warmup_epochs,
            device=device,
        )
        campaign_states[key] = _CampaignState(
            corp=corp,
            engine=engine,
            cmd_stats=cmd_stats,
            sample_p=sample_p,
            steps_per_epoch=steps_per_epoch,
            disc=disc,
            held_idx=held_idx,
        )
        logger.info(
            "signature %s: %d/%d examples held back from training",
            key,
            len(held_idx),
            corp.n_examples,
        )

    trainer = CorpusTrainer(
        encoder, engines, lr=args.lr, weight_decay=args.weight_decay
    )
    trainer.to(device)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        trainer.optimizer, _lr_lambda(args.steps, args.warmup_frac, args.lr_floor_frac)
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
    run_dir = root / "checkpoints" / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "grounded_latent_engine.pt"  # "latest" -- resume reads this
    best_path = run_dir / "best.pt"  # lowest held-back physics misfit seen so far
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2))
    best_state = {"misfit": float("inf"), "step": -1}

    ckpt_extra = {
        "config": vars(args),
        "feature_stats": feature_stats,
        "anchored_stats": anchored_stats,
        "cmd_stats": {k: s.cmd_stats for k, s in campaign_states.items()},
        "reference_signature": ref_key,
        "n_cells": ref_n_cells,
        "coil_model_version": COIL_MODEL_VERSION,
        "geometry_table_version": GEOMETRY_TABLE_VERSION,
        "sensor_scale_floor": {
            "fn": "imas_ambix.latent.data.robust_channel_scale",
            "rel_floor": CHANNEL_SCALE_KIND_FLOOR_REL,
        },
        "per_signature_n_coil": {k: c.n_coil for k, c in corpora.items()},
        "n_examples": {k: c.n_examples for k, c in corpora.items()},
        "held_back_idx": {k: s.held_idx for k, s in campaign_states.items()},
    }

    start_step = 0
    if args.resume and ckpt_path.exists():
        start_step, extra = trainer.load(
            ckpt_path, map_location=str(device), return_extra=True
        )
        disc_states = extra.get("discrepancy", {})
        for key, state in campaign_states.items():
            if key in disc_states:
                _restore_disc(state.disc, disc_states[key])
        if "scheduler" in extra:
            scheduler.load_state_dict(extra["scheduler"])
        if "best" in extra:
            best_state.update(extra["best"])
        logger.info(
            "resumed from step %d (best held-back misfit %.4f @ step %d)",
            start_step,
            best_state["misfit"],
            best_state["step"],
        )

    if args.dry_run:
        key, state = next(iter(campaign_states.items()))
        batch, rows = _build_batch(
            state, feature_stats, anchored_stats, args.batch_size, rng, device
        )
        out = state.engine.losses(batch)
        printable = {k: round(float(v), 6) for k, v in out.items()}
        logger.info("DRY-RUN loss dict (%s): %s", key, printable)
        print("DRY_RUN_LOSS", json.dumps(printable))
        return 0

    stop = {"flag": False}

    def _on_sigterm(*_a):
        logger.warning("SIGTERM received — checkpointing and exiting")
        stop["flag"] = True

    signal.signal(signal.SIGTERM, _on_sigterm)

    def _link_or_copy(src: Path, dst: Path) -> None:
        """Hard-link ``dst`` to ``src`` (no extra disk space; a later atomic
        replace of ``src`` cannot affect ``dst``'s already-linked content), or
        fall back to a plain copy if the filesystem cannot hard-link."""
        if dst.exists():
            dst.unlink()
        try:
            dst.hardlink_to(src)
        except OSError:
            import shutil

            shutil.copy2(src, dst)

    def _checkpoint(step: int) -> float:
        """Save the "latest" (resume) path + a NUMBERED snapshot, evaluate the
        held-back physics misfit, and update best.pt if it improved -- job
        1225447 left only one, already-diverged, overwritten checkpoint behind;
        this is the fix ("never again leave only a diverged state")."""
        held_back = {
            key: _held_back_physics_misfit(state, feature_stats, anchored_stats, device)
            for key, state in campaign_states.items()
        }
        mean_misfit = float(np.mean(list(held_back.values())))
        extra = {
            **ckpt_extra,
            "discrepancy": {k: _disc_state(s.disc) for k, s in campaign_states.items()},
            "scheduler": scheduler.state_dict(),
            "held_back_misfit": held_back,
            "best": best_state,
        }
        trainer.save(ckpt_path, step=step, extra=extra)
        numbered = run_dir / f"step_{step:06d}.pt"
        _link_or_copy(ckpt_path, numbered)
        logger.info(
            "checkpoint saved at step %d -> %s (held-back misfit %.4f, per-sig %s)",
            step,
            numbered,
            mean_misfit,
            {k: round(v, 4) for k, v in held_back.items()},
        )
        if mean_misfit < best_state["misfit"]:
            best_state["misfit"] = mean_misfit
            best_state["step"] = step
            _link_or_copy(numbered, best_path)
            logger.info(
                "new best held-back misfit %.4f @ step %d -> %s",
                mean_misfit,
                step,
                best_path,
            )
        return mean_misfit

    t0 = time.time()
    for step in range(start_step, args.steps):
        if stop["flag"]:
            break
        prebuilt: dict[str, tuple[dict, np.ndarray]] = {}
        for key, state in campaign_states.items():
            prebuilt[key] = _build_batch(
                state, feature_stats, anchored_stats, args.batch_size, rng, device
            )
        batch_fns = {key: (lambda b=b: b) for key, (b, _rows) in prebuilt.items()}
        totals = trainer.step(batch_fns)
        scheduler.step()

        for key, (batch, rows) in prebuilt.items():
            state = campaign_states[key]
            misfit = _per_example_magnetics_misfit(state.engine, batch)
            epoch = step // state.steps_per_epoch
            state.disc.update(state.corp.ids[rows], misfit, epoch)

        if step % args.log_every == 0:
            elapsed = time.time() - t0
            logger.info(
                "step %d/%d  lr=%.2e  totals=%s  (%.1fs elapsed)",
                step,
                args.steps,
                scheduler.get_last_lr()[0],
                {k: round(v, 4) for k, v in totals.items()},
                elapsed,
            )
        if step > 0 and step % args.ckpt_every == 0:
            _checkpoint(step)

    _checkpoint(trainer.step_count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
