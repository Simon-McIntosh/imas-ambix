"""In-process trainer + codebook-decision + encoder for HF signal tokens.

Drives the phase-aware patch-transformer tokenizer end-to-end on a single
GPU (repo §2b in-process pattern: model loaded once, many shots streamed
through the same process — no subprocess-per-shot, no file-IPC daemon).

Three phases:

``train``
    Fit the patch-transformer autoencoder on a set of shots for ONE
    bottleneck variant.  Saves the trained weights.
``decide``
    Train all three bottlenecks (FSQ, VQ, continuous) on the SAME train
    shots, measure round-trip phase fidelity on a held-out shot set, and
    write a decision report (reconstruction CRPS + spectral phase error +
    mode-number recovery per variant).  The continuous bottleneck is the
    fidelity-ceiling control: if quantisation phase error is far above it,
    prefer continuous.
``encode``
    Encode a list of shots into the v2 ``signals_hf`` store with the chosen
    tokenizer — native cadence, per-token time, per-channel validity.

A per-shot watchdog and a SIGTERM/SIGINT handler make a long encode lossless
and cancellation-safe (repo §2a good-practice hang protection).
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from imas_ambix.data.paths import LEVEL1_DIR
from imas_ambix.statespace.fast_features import XSX_STUCK_CHANNEL
from imas_ambix.statespace.fast_loader import (
    read_xim_shot,
    read_xma_shot,
    read_xsx_shot,
)
from imas_ambix.tokenizer.patch_transformer import (
    PatchTokenizerConfig,
    PatchTransformerTokenizer,
    mode_decomposition,
)
from imas_ambix.tokenizer.registry import (
    BLOCK_XIM_PATCH,
    BLOCK_XMA_MODE,
    BLOCK_XMA_PATCH,
    BLOCK_XSX_PATCH,
    BLOCK_XSX_PROFILE,
    VOCAB_VERSION,
    registry,
)
from imas_ambix.tokenizer.store_v2 import (
    StoreV2Attrs,
    save_signal_hf_tokens,
    signal_hf_token_path,
)

logger = logging.getLogger("signal_hf_encode")

# Cooperative-cancellation flag (set by the SIGTERM/SIGINT handler).
_STOP = {"flag": False}


def _install_signal_handler() -> None:
    def _handler(signum, _frame):
        logger.warning(
            "signal %d received — finishing current shot then stopping", signum
        )
        _STOP["flag"] = True

    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)


# Per-modality config: which loader, which channels are the "array" carrying
# cross-channel mode structure, and the registry block name for the codes.
N_XMA_COILS = 40
N_MODES = 4

# Radians of spectral phase error above the continuous (no-quantisation)
# ceiling that we tolerate before declaring quantisation phase-destructive.
PHASE_TOLERANCE_RAD = 0.20


@dataclass
class ModalitySpec:
    group: str
    patch_block: str
    is_coil_array: bool
    mode_block: str | None  # cross-channel mode-number tokens, if applicable


XMA_SPEC = ModalitySpec(
    group="xma",
    patch_block=BLOCK_XMA_PATCH,
    is_coil_array=True,
    mode_block=BLOCK_XMA_MODE,
)
XIM_SPEC = ModalitySpec(
    group="xim",
    patch_block=BLOCK_XIM_PATCH,
    is_coil_array=False,
    mode_block=None,
)
# xsx soft-X-ray chord array.  Like xma it is a spatially-distributed array
# (the horizontal-camera chords sample a line-integral across the plasma), so
# the cross-channel decomposition fires — but the relevant cross-chord
# structure is the radial *emission profile*, not a periodic poloidal mode.
# The same spatial-DFT block (mode_decomposition) gives a compact, phase-
# preserving radial-profile latent; the block name records it as a profile.
XSX_SPEC = ModalitySpec(
    group="xsx",
    patch_block=BLOCK_XSX_PATCH,
    is_coil_array=True,
    mode_block=BLOCK_XSX_PROFILE,
)
SPECS = {"xma": XMA_SPEC, "xim": XIM_SPEC, "xsx": XSX_SPEC}


# ---------------------------------------------------------------------------
# Shot loading → (C, T) window + per-channel validity + native rate
# ---------------------------------------------------------------------------


def _canonical_ccbv(name: str) -> str | None:
    """Map a ccbv coil name (modern ``ccbv_NN`` / legacy ``ccbvNN``) to ``ccbv_NN``.

    Returns None if ``name`` is not a ccbv Mirnov coil.  Canonicalising on the
    coil INDEX lets modern and legacy campaigns share a common channel set so
    cross-shot training and the spatial-DFT mode decomposition line up.
    """
    if not name.startswith("ccbv"):
        return None
    rest = name[len("ccbv") :].lstrip("_")
    if rest.isdigit():
        return f"ccbv_{int(rest):02d}"
    return None


def load_shot_window(shot_id: int, group: str):
    """Load one shot's (C, T) signal window for ``group`` (xma|xim).

    Returns ``(data, channel_names, valid, native_rate_hz, window)`` or None.
    ``data`` is ``(C, T)``; ``valid`` is ``(C,)`` per-channel availability
    expanded to per-token at encode time.
    """
    path = LEVEL1_DIR / f"{shot_id}.zarr"
    if not path.exists():
        return None
    if group == "xma":
        shot = read_xma_shot(path)
        if shot is None:
            return None
        # Restrict to the ccbv Mirnov coils for the coil-array mode decomposition.
        # Canonicalise the coil name to ``ccbv_NN`` (a coil INDEX) so modern
        # (ccbv_01) and legacy (ccbv01) campaigns share a common channel set —
        # the intersection across shots is otherwise empty.
        names = shot.channel_names
        keep, chan = [], []
        for i, n in enumerate(names):
            canon = _canonical_ccbv(n)
            if canon is not None:
                keep.append(i)
                chan.append(canon)
        if not keep:
            return None
        data = shot.data[:, keep].T  # (C, T)
    elif group == "xim":
        shot = read_xim_shot(path)
        if shot is None:
            return None
        data = shot.data.T  # (C, T)
        chan = list(shot.channel_names)
    elif group == "xsx":
        shot = read_xsx_shot(path)
        if shot is None:
            return None
        # Stack the lower horizontal camera (always present — read_xsx_shot
        # returns None otherwise) and the upper camera when operational.  Each
        # chord becomes one channel of the per-chord patch tokenizer; the
        # cross-chord profile latent then captures the radial emission shape.
        blocks: list[np.ndarray] = [shot.hcam_l]  # (18, T)
        chan = [f"hcam_l_{i:02d}" for i in range(shot.hcam_l.shape[0])]
        if shot.hcam_u is not None:
            blocks.append(shot.hcam_u)
            chan += [f"hcam_u_{i:02d}" for i in range(shot.hcam_u.shape[0])]
        data = np.concatenate(blocks, axis=0)  # (C, T)
        valid = np.isfinite(data).any(axis=1)  # (C,)
        # Mark the known stuck chord (constant value, no fluctuation content)
        # invalid so a consumer honours the mask rather than learning a flat
        # signal as real structure.  The channel stays in the set to keep the
        # per-chord channel layout stable across shots.
        stuck = []
        if shot.hcam_l.shape[0] > XSX_STUCK_CHANNEL:
            stuck.append(XSX_STUCK_CHANNEL)  # in hcam_l
        if shot.hcam_u is not None and shot.hcam_u.shape[0] > XSX_STUCK_CHANNEL:
            stuck.append(shot.hcam_l.shape[0] + XSX_STUCK_CHANNEL)  # in hcam_u
        for s in stuck:
            valid[s] = False
        window = (float(shot.time[0]), float(shot.time[-1]))
        return data.astype(np.float32), chan, valid, float(shot.rate_hz), window
    else:
        raise ValueError(f"unknown group {group!r}")

    valid = np.isfinite(data).any(axis=1)  # (C,)
    window = (float(shot.time[0]), float(shot.time[-1]))
    return data.astype(np.float32), chan, valid, float(shot.rate_hz), window


# ---------------------------------------------------------------------------
# Codebook decision — train all three, measure phase fidelity on a holdout
# ---------------------------------------------------------------------------


def _collect_windows(shot_ids: list[int], group: str):
    """Load + intersect channels across shots → uniform (C, T) windows.

    Channels are intersected to a common set so the per-coil model sees a
    fixed channel count; shorter shots define T per window (the model is
    length-agnostic — sequence length varies per window).
    """
    loaded = []
    for sid in shot_ids:
        w = load_shot_window(sid, group)
        if w is not None:
            loaded.append((sid, w))
    if not loaded:
        return [], []
    common = set(loaded[0][1][1])
    for _, (_, chan, *_rest) in loaded:
        common &= set(chan)
    common_order = [c for c in loaded[0][1][1] if c in common]
    windows = []
    kept_ids = []
    for sid, (data, chan, _valid, _rate, _win) in loaded:
        idx = [chan.index(c) for c in common_order]
        windows.append(data[idx])
        kept_ids.append(sid)
    return windows, common_order


def decide_codebook(
    group: str,
    train_ids: list[int],
    holdout_ids: list[int],
    *,
    device: str,
    epochs: int,
    patch_size: int,
    out_path: Path,
    corpus_calibration=None,
) -> dict:
    """Train FSQ/VQ/continuous; measure phase fidelity on the holdout.

    Returns the decision report and writes it to ``out_path``.  The winner is
    the quantised variant whose phase error is within a tolerance of the
    continuous ceiling AND has the best reconstruction CRPS; if no quantised
    variant comes within tolerance, ``continuous`` wins (quantisation
    destroys phase → prefer continuous+mask).
    """
    spec = SPECS[group]
    train_windows, channels = _collect_windows(train_ids, group)
    hold_windows, hold_channels = _collect_windows(holdout_ids, group)
    if not train_windows or not hold_windows:
        raise RuntimeError(f"no usable {group} windows (train/holdout)")
    # rate from the first train shot (uniform within a group/campaign era).
    w0 = load_shot_window(train_ids[0], group)
    native_rate = w0[3] if w0 else 50_000.0
    dt = 1.0 / native_rate

    report: dict = {
        "group": group,
        "n_channels": len(channels),
        "channels": channels,
        "native_rate_hz": native_rate,
        "n_train": len(train_windows),
        "n_holdout": len(hold_windows),
        "calibration_mode": (
            "absolute" if corpus_calibration is not None else "per_window"
        ),
        "variants": {},
    }

    for bottleneck in ("continuous", "fsq", "vq"):
        cfg = PatchTokenizerConfig(
            patch_size=patch_size, bottleneck=bottleneck, use_stft=True
        )
        tok = PatchTransformerTokenizer(cfg=cfg, name=spec.patch_block, device=device)
        t0 = time.time()
        hist = tok.fit(
            train_windows,
            epochs=epochs,
            logger=logger,
            channel_names=channels,
            corpus_calibration=corpus_calibration,
        )
        # Aggregate phase-fidelity over the holdout shots.
        crps, perr, corr, mperr, active = [], [], [], [], []
        for hw in hold_windows:
            m = tok.roundtrip_metrics(
                hw,
                dt=dt,
                is_coil_array=spec.is_coil_array,
                channel_names=hold_channels,
                corpus_calibration=corpus_calibration,
            )
            crps.append(m["recon_crps"])
            perr.append(m["phase_err"])
            active.append(m["n_active_codes"])
            if spec.is_coil_array:
                corr.append(m.get("mean_complex_corr", np.nan))
                mperr.append(m.get("mean_mode_phase_err", np.nan))
        report["variants"][bottleneck] = {
            "train_seconds": round(time.time() - t0, 1),
            "final_recon_loss": hist["recon"][-1],
            "recon_crps": float(np.nanmean(crps)),
            "phase_err_rad": float(np.nanmean(perr)),
            "n_active_codes": int(np.nanmax(active)) if active else 0,
            "mode_complex_corr": float(np.nanmean(corr)) if corr else None,
            "mode_phase_err_rad": float(np.nanmean(mperr)) if mperr else None,
        }
        logger.info("[%s/%s] %s", group, bottleneck, report["variants"][bottleneck])

    # Decision: continuous is the phase-fidelity ceiling.  A quantised variant
    # wins only if its phase error is within PHASE_TOLERANCE_RAD of the ceiling.
    cont = report["variants"]["continuous"]["phase_err_rad"]
    candidates = []
    for bn in ("fsq", "vq"):
        v = report["variants"][bn]
        if (
            np.isfinite(v["phase_err_rad"])
            and v["phase_err_rad"] <= cont + PHASE_TOLERANCE_RAD
        ):
            candidates.append((v["recon_crps"], bn))
    if candidates:
        winner = min(candidates)[1]
        rationale = (
            f"{winner} phase error within {PHASE_TOLERANCE_RAD} rad of the "
            f"continuous ceiling ({cont:.3f}); best reconstruction CRPS among "
            f"in-tolerance quantised variants"
        )
    else:
        winner = "continuous"
        rationale = (
            "no quantised variant came within "
            f"{PHASE_TOLERANCE_RAD} rad of the continuous phase-error ceiling "
            f"({cont:.3f}) — quantisation destroys phase; prefer continuous+mask"
        )
    report["decision"] = {"winner": winner, "rationale": rationale}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    logger.info("DECISION [%s]: %s — %s", group, winner, rationale)
    return report


# ---------------------------------------------------------------------------
# Encode → v2 store
# ---------------------------------------------------------------------------


class ShotWindowDataset:
    """torch ``Dataset`` that reads one shot's signal window per item.

    The corpus encode is overwhelmingly IO-bound — the patch-transformer
    inference is < 1 % GPU utilisation, while the cost is the GPFS read of the
    MHz-rate raw signals (measured: xsx 500 kHz × 300k × 36 chords).  Reading
    each shot in a torch ``DataLoader`` worker subprocess overlaps the next
    shot's GPFS read with the current shot's inference + store write, the
    canonical in-process IO-overlap pattern (repo §2b: a bounded DataLoader
    worker pool, NOT a prefetch producer/consumer thread or a file-IPC daemon).

    Each item is ``(shot_id, window_or_None, reason)`` where ``window`` is the
    ``load_shot_window`` tuple or ``None`` (with a skip ``reason``) so a
    group-absent / unreadable shot is reported rather than crashing a worker.
    """

    def __init__(self, shot_ids: list[int], group: str) -> None:
        self.shot_ids = list(shot_ids)
        self.group = group

    def __len__(self) -> int:
        return len(self.shot_ids)

    def __getitem__(self, i: int):
        sid = self.shot_ids[i]
        if not group_present(sid, self.group):
            return sid, None, "group_absent"
        try:
            w = load_shot_window(sid, self.group)
        except Exception as exc:  # noqa: BLE001 — corpus robustness
            return sid, None, f"load_error:{exc}"
        if w is None:
            return sid, None, "no_data"
        return sid, w, None


def _passthrough_collate(batch):
    """Identity collate — keep variable-length windows as a python list.

    Shots have different ``T`` and channel inventories, so they cannot stack
    into a tensor batch; ``batch_size=1`` + this collate yields the single
    item unchanged.
    """
    return batch[0]


def encode_shots(
    group: str,
    shot_ids: list[int],
    tokenizer: PatchTransformerTokenizer,
    *,
    watchdog_s: float = 120.0,
    skip_existing: bool = True,
    num_workers: int = 0,
    corpus_calibration=None,
    attach_geometry: bool = True,
) -> dict:
    """Encode shots into the v2 signals_hf store with ``tokenizer``.

    Emits per-coil patch codes, per-token time, and per-channel validity.
    For a coil/chord array, additionally emits cross-channel mode/profile
    tokens as a sibling group ``{group}_mode``.  Returns a summary dict.

    ``skip_existing`` (default True) makes a long corpus encode resumable: a
    shot whose v2 store already exists is skipped, so re-running with the same
    ENCODE_IDS picks up where a cancelled/expired job left off.

    ``num_workers > 0`` reads shot windows in a torch ``DataLoader`` worker
    pool so the GPFS read of shot ``N+1`` overlaps the inference + store write
    of shot ``N`` — the IO-overlap that matters for this IO-bound encode (repo
    §2b).  ``num_workers == 0`` reads inline (used by unit tests, which
    monkeypatch ``load_shot_window``).

    ``corpus_calibration`` (``dict[str, ChannelCalibration] | None``): when
    supplied, each channel is standardised against its CORPUS mean/std so
    absolute magnitude survives tokenisation (the patch-transformer codebook
    must be retrained under absolute normalisation for the codes to be valid).
    Default ``None`` keeps the existing per-window behaviour byte-for-byte.
    """
    spec = SPECS[group]
    summary = {"group": group, "encoded": [], "skipped": [], "n_tokens_total": 0}
    cfg = tokenizer.cfg

    # Allocate registry blocks for the patch codes (+ mode codes if a coil array).
    patch_vocab = max(getattr(tokenizer._model.bottleneck, "codebook_size", 1), 1)
    registry.allocate(spec.patch_block, patch_vocab)
    if spec.mode_block is not None:
        # mode tokens reuse the patch codebook conceptually; allocate a 1-id
        # placeholder block so mode token positions occupy a distinct namespace
        # (the real values are the complex mode amplitudes carried in metadata).
        registry.allocate(spec.mode_block, 1)

    # Resume + presence filter up front so DataLoader workers never read a shot
    # that is already done (cheap path-exists check, not a decode).
    todo: list[int] = []
    for sid in shot_ids:
        if skip_existing and already_encoded(sid, group):
            summary["skipped"].append({"shot": sid, "reason": "already_encoded"})
        else:
            todo.append(sid)

    feed = _shot_window_feed(todo, group, num_workers)
    try:
        _drain_feed(
            feed,
            group,
            tokenizer,
            spec,
            cfg,
            patch_vocab,
            summary,
            watchdog_s,
            corpus_calibration=corpus_calibration,
            attach_geometry=attach_geometry,
        )
    finally:
        _close_feed(feed)
    return summary


# --- per-channel geometry attach (campaign-cached) -------------------------


def _geometry_features_for(
    shot_id: int,
    channel_names: list[str],
    cache: dict,
) -> tuple[np.ndarray | None, tuple[str, ...], tuple[str, ...]]:
    """Build the ``(n_channels, N_GEOMETRY_FEATURES)`` geometry array for a shot.

    Geometry is campaign-constant (static efm sensor/coil positions — NO
    equilibrium / psi / boundary, firewall-safe), so the per-shot
    ``GeometryFields`` is cached by the shot's setup signature and reused across
    every shot of the same campaign.  Best-effort: a shot whose static geometry
    cannot be read still encodes (geometry omitted + a one-line warning) — the
    corpus encode never crashes on a geometry read.

    Returns ``(features, feature_names, sensor_kinds)`` or ``(None, (), ())``.
    """
    from imas_ambix.gs.geometry_export import (
        GEOMETRY_FEATURE_NAMES,
        build_geometry_fields_from_table,
    )

    try:
        from imas_ambix.gs.geometry import build_table_for_shot

        # Per-shot static-geometry table is the cheap thing; the signature keys
        # the (expensive-to-flatten) GeometryFields cache.
        table = build_table_for_shot(shot_id)
        sig = table.signature.key
        fields = cache.get(sig)
        if fields is None:
            fields = build_geometry_fields_from_table(
                table, extra_channel_names=channel_names
            )
            cache[sig] = fields
        feats, kinds = fields.feature_matrix(channel_names)
        return feats, tuple(GEOMETRY_FEATURE_NAMES), tuple(kinds)
    except Exception as exc:  # noqa: BLE001 — corpus robustness; geometry best-effort
        logger.warning(
            "shot %d: geometry unavailable (%s) — encoding without geometry",
            shot_id,
            exc,
        )
        return None, (), ()


def _inline_feed(shot_ids: list[int], group: str):
    """Read shot windows inline (no worker subprocesses)."""
    for sid in shot_ids:
        if not group_present(sid, group):
            yield sid, None, "group_absent"
            continue
        w = load_shot_window(sid, group)
        yield sid, w, (None if w is not None else "no_data")


def _shot_window_feed(shot_ids: list[int], group: str, num_workers: int):
    """Yield ``(shot_id, window_or_None, reason)`` — DataLoader-fed if workers.

    Worker subprocesses read each shot's signal off GPFS in parallel with the
    main-process inference + store write (the IO-overlap that matters for this
    IO-bound encode).  Uses an explicit ``fork`` context: the workers do pure
    CPU numpy / Zarr reads (no CUDA in the child), and fork avoids the
    forkserver handshake that is restricted on some shared nodes.  If worker
    startup fails for any environmental reason, fall back to inline reads so a
    long corpus encode never dies on a multiprocessing hiccup.
    """
    if num_workers <= 0 or not shot_ids:
        yield from _inline_feed(shot_ids, group)
        return

    import multiprocessing as mp

    from torch.utils.data import DataLoader

    ds = ShotWindowDataset(shot_ids, group)
    try:
        ctx = mp.get_context("fork")
        loader = DataLoader(
            ds,
            batch_size=1,
            num_workers=num_workers,
            prefetch_factor=2,
            collate_fn=_passthrough_collate,
            shuffle=False,
            persistent_workers=False,
            multiprocessing_context=ctx,
        )
        it = iter(loader)
        # Pull the FIRST item inside the try so a worker-startup failure (the
        # forkserver/fork handshake can fail before any batch arrives) is caught
        # here, before anything is yielded — then a clean inline fallback runs
        # over the WHOLE list with no double-encoding.
        first = next(it)
    except StopIteration:
        return
    except Exception as exc:  # noqa: BLE001 — environmental worker-start failure
        logger.warning(
            "DataLoader workers unavailable (%s); falling back to inline reads",
            exc,
        )
        yield from _inline_feed(shot_ids, group)
        return
    yield first
    yield from it


def _close_feed(feed) -> None:
    """Best-effort generator close so DataLoader workers tear down cleanly."""
    import contextlib

    # Teardown is hang protection, not a drain guard — never raise from here.
    with contextlib.suppress(Exception):
        feed.close()


def _drain_feed(
    feed,
    group,
    tokenizer,
    spec,
    cfg,
    patch_vocab,
    summary,
    watchdog_s,
    *,
    corpus_calibration=None,
    attach_geometry: bool = True,
) -> None:
    summary["calibration"] = (
        "absolute" if corpus_calibration is not None else "per_window"
    )
    # Campaign-keyed geometry cache: each distinct setup signature reads static
    # geometry once, then every shot of that campaign reuses the flattened table.
    geom_cache: dict = {}
    summary.setdefault("geometry_attached", 0)
    for sid, w, reason in feed:
        sid = int(sid)
        if _STOP["flag"]:
            summary["skipped"].append({"shot": sid, "reason": "stop_requested"})
            continue
        if w is None:
            summary["skipped"].append({"shot": sid, "reason": reason or "no_data"})
            continue
        t0 = time.time()
        data, chan, valid_ch, native_rate, window = w
        try:
            ids, latent, _recon = tokenizer.encode_window(
                data,
                channel_names=chan,
                corpus_calibration=corpus_calibration,
            )  # (C,P),(C,P,d),(C,T)
        except Exception as exc:  # pragma: no cover - corpus robustness
            summary["skipped"].append({"shot": sid, "reason": f"encode_error:{exc}"})
            continue
        if time.time() - t0 > watchdog_s:
            logger.warning("shot %d exceeded watchdog (%.0fs)", sid, watchdog_s)

        n_patches = ids.shape[1]
        # Per-token time: centre of each patch on the native axis.
        token_time = (
            window[0] + (np.arange(n_patches) + 0.5) * cfg.patch_size / native_rate
        )
        # tokens: (n_patches, n_channels) — shift to global ids.
        local = ids.T.astype(np.int64)  # (P, C)
        global_ids = registry.shift(spec.patch_block, local)
        # per-token-per-channel validity: a channel valid for the shot is valid
        # at every token; an absent channel is invalid at every token.
        token_valid = np.broadcast_to(valid_ch[None, :], (n_patches, len(chan)))

        geom_feats = None
        geom_feature_names: tuple[str, ...] = ()
        geom_kinds: tuple[str, ...] = ()
        if attach_geometry:
            geom_feats, geom_feature_names, geom_kinds = _geometry_features_for(
                sid, list(chan), geom_cache
            )
            if geom_feats is not None:
                summary["geometry_attached"] += 1

        attrs = StoreV2Attrs(
            tokenizer_name=spec.patch_block,
            vocab_version=VOCAB_VERSION,
            native_rate_hz=native_rate,
            token_rate_hz=cfg.token_rate_hz(native_rate),
            n_channels=len(chan),
            channel_names=tuple(chan),
            phase_preserving=True,
            original_window=window,
            calibration_mode=(
                "absolute" if corpus_calibration is not None else "per_shot"
            ),
            geometry_feature_names=geom_feature_names,
            geometry_sensor_kinds=geom_kinds,
            metadata={
                "bottleneck": cfg.bottleneck,
                "patch_size": cfg.patch_size,
                "use_stft": cfg.use_stft,
                "codebook_size": patch_vocab,
                "embed_dim": int(latent.shape[-1]),
                "calibration": "absolute"
                if corpus_calibration is not None
                else "per_window",
            },
        )
        # For a continuous bottleneck the discrete ids are vestigial — persist
        # the per-token-per-channel latent (P, C, d) as the phase-preserving
        # payload.  For a quantised bottleneck the ids carry the information,
        # so the embedding is omitted to keep the store compact.
        embedding = (
            np.transpose(latent, (1, 0, 2))  # (C,P,d) → (P,C,d)
            if cfg.bottleneck == "continuous"
            else None
        )
        save_signal_hf_tokens(
            sid,
            group,
            global_ids,
            token_time,
            token_valid.copy(),
            attrs,
            embedding=embedding,
            geometry=geom_feats,
        )

        # Cross-channel mode-number tokens for a coil array.
        if spec.mode_block is not None and len(chan) >= 8:
            # Mode amplitudes at each patch centre (nearest native sample).
            centre_idx = np.clip(
                ((token_time - window[0]) * native_rate).astype(int),
                0,
                data.shape[1] - 1,
            )
            snap = data[:, centre_idx].T  # (P, C)
            modes = mode_decomposition(snap, n_modes=N_MODES)  # (P, 2*N_MODES)
            # tokens for the mode block are placeholders (id 0); the complex
            # mode amplitudes are the payload, carried in metadata + token_time.
            mode_local = np.zeros((n_patches, 2 * N_MODES), dtype=np.int64)
            mode_global = registry.shift(spec.mode_block, mode_local)
            mode_valid = np.ones((n_patches, 2 * N_MODES), dtype=bool)
            mode_attrs = StoreV2Attrs(
                tokenizer_name=spec.mode_block,
                vocab_version=VOCAB_VERSION,
                native_rate_hz=native_rate,
                token_rate_hz=cfg.token_rate_hz(native_rate),
                n_channels=2 * N_MODES,
                channel_names=tuple(
                    [f"mode{m}_re" for m in range(N_MODES)]
                    + [f"mode{m}_im" for m in range(N_MODES)]
                ),
                phase_preserving=True,
                original_window=window,
                calibration_mode=(
                    "absolute" if corpus_calibration is not None else "per_shot"
                ),
                metadata={
                    "kind": "spatial_dft_mode_amplitudes",
                    "n_modes": N_MODES,
                    "mode_values": modes.tolist(),
                },
            )
            save_signal_hf_tokens(
                sid,
                f"{group}_mode",
                mode_global,
                token_time,
                mode_valid,
                mode_attrs,
            )

        summary["encoded"].append(
            {"shot": sid, "n_patches": int(n_patches), "n_channels": len(chan)}
        )
        summary["n_tokens_total"] += int(n_patches * len(chan))
        logger.info(
            "encoded %d: %d patches × %d ch (%.1fs)",
            sid,
            n_patches,
            len(chan),
            time.time() - t0,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_ids(text: str) -> list[int]:
    if Path(text).exists():
        return [int(x) for x in Path(text).read_text().split()]
    return [int(x) for x in text.replace(",", " ").split()]


def corpus_shot_ids() -> list[int]:
    """Every shot id present in the level-1 corpus, ascending.

    Built from presence on disk (``LEVEL1_DIR/{id}.zarr``); per-group presence
    is resolved later by ``load_shot_window`` returning None, so the shotlist
    is the union and a group-absent shot is skipped (not an error).
    """
    ids: list[int] = []
    for p in LEVEL1_DIR.glob("*.zarr"):
        stem = p.stem
        if stem.isdigit():
            ids.append(int(stem))
    return sorted(ids)


def group_present(shot_id: int, group: str) -> bool:
    """Cheap on-disk presence check for ``group`` in shot ``shot_id``.

    Avoids decoding the whole shot just to discover the group is absent — the
    encode loop's authoritative guard is still ``load_shot_window`` returning
    None (which catches present-but-empty groups), but this filters the bulk of
    group-absent shots before any decode work.
    """
    return (LEVEL1_DIR / f"{shot_id}.zarr" / group).exists()


def already_encoded(shot_id: int, group: str) -> bool:
    """True if this shot/group has a **complete** v2 signals_hf token store.

    Lets a long corpus encode checkpoint/resume: re-running with the same
    ENCODE_IDS skips shots that are already done.  Completeness is validated,
    not just path-existence — the store writer creates the arrays first and
    flushes the required ``.attrs`` last, so a shot whose encode was killed
    mid-write (e.g. SIGTERM at the SLURM time limit) leaves arrays with no
    attrs.  A path-only check would skip that truncated shot forever; reading
    the attrs back means a partial shot is re-encoded on resume instead.
    """
    path = signal_hf_token_path(shot_id, group)
    if not path.exists():
        return False
    try:
        import zarr

        from imas_ambix.tokenizer.store_v2 import REQUIRED_ATTRS

        store = zarr.open_group(str(path), mode="r")
        attrs = dict(store.attrs)
        return all(k in attrs for k in REQUIRED_ATTRS)
    except Exception:  # noqa: BLE001 — unreadable/partial store ⇒ re-encode
        return False


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--group", choices=["xma", "xim", "xsx"], required=True)
    common.add_argument("--device", default="cuda")
    common.add_argument("--patch-size", type=int, default=64)

    # The bottleneck is trained under the SAME normalisation it will see at
    # encode time.  --absolute trains against the persisted CORPUS calibration
    # (SI magnitude survives tokenisation); a per-window codebook is invalid for
    # absolute-mode encode, so the retrain MUST pass this flag.
    _abs_help = (
        "train/decide against the persisted CORPUS calibration (absolute / SI "
        "magnitude) instead of per-window z-scoring; fails loud if the group "
        "has no calibration JSON yet"
    )

    pd = sub.add_parser("decide", parents=[common])
    pd.add_argument("--train", required=True, help="shot ids or a file")
    pd.add_argument("--holdout", required=True, help="shot ids or a file")
    pd.add_argument("--epochs", type=int, default=40)
    pd.add_argument("--out", type=Path, required=True)
    pd.add_argument("--absolute", action="store_true", help=_abs_help)

    pt = sub.add_parser("train", parents=[common])
    pt.add_argument("--train", required=True)
    pt.add_argument("--bottleneck", default="fsq")
    pt.add_argument("--epochs", type=int, default=40)
    pt.add_argument("--out", type=Path, required=True, help="checkpoint path")
    pt.add_argument("--absolute", action="store_true", help=_abs_help)

    pe = sub.add_parser("encode", parents=[common])
    pe.add_argument(
        "--shots",
        required=True,
        help="shot ids, a file, or the literal 'all' for the full corpus",
    )
    pe.add_argument("--ckpt", type=Path, required=True)
    pe.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="re-encode shots that already have a v2 store (default: skip them)",
    )
    pe.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="write the encode summary (counts, skips, totals) here as JSON",
    )
    pe.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="DataLoader worker subprocesses reading shot windows off GPFS in "
        "parallel with inference (this encode is IO-bound; 0 = inline reads)",
    )
    pe.add_argument(
        "--absolute",
        action="store_true",
        help="standardise each channel against the persisted CORPUS calibration "
        "(absolute / SI magnitude survives tokenisation) instead of per-window "
        "z-scoring; requires a codebook retrained under absolute normalisation",
    )

    args = p.parse_args(argv)
    _install_signal_handler()
    _torch_perf_setup(args.device)

    def _load_calibration_or_die(group: str):
        """Load the corpus calibration for ``group`` or exit loud (--absolute)."""
        from imas_ambix.calibration.corpus_compute import load_group_calibration

        cal = load_group_calibration(group)
        if cal is None:
            raise SystemExit(
                f"--absolute requested but no corpus calibration found for "
                f"group {group!r}; run `python -m "
                f"imas_ambix.calibration.corpus_compute --group {group}` first"
            )
        logger.info(
            "absolute mode: loaded calibration for %d channels (group %r)",
            len(cal),
            group,
        )
        return cal

    if args.cmd == "decide":
        corpus_calibration = (
            _load_calibration_or_die(args.group) if args.absolute else None
        )
        decide_codebook(
            args.group,
            _parse_ids(args.train),
            _parse_ids(args.holdout),
            device=args.device,
            epochs=args.epochs,
            patch_size=args.patch_size,
            out_path=args.out,
            corpus_calibration=corpus_calibration,
        )
    elif args.cmd == "train":
        corpus_calibration = (
            _load_calibration_or_die(args.group) if args.absolute else None
        )
        windows, chan = _collect_windows(_parse_ids(args.train), args.group)
        cfg = PatchTokenizerConfig(
            patch_size=args.patch_size, bottleneck=args.bottleneck, use_stft=True
        )
        spec = SPECS[args.group]
        tok = PatchTransformerTokenizer(
            cfg=cfg, name=spec.patch_block, device=args.device
        )
        tok.fit(
            windows,
            epochs=args.epochs,
            logger=logger,
            channel_names=chan,
            corpus_calibration=corpus_calibration,
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        tok.save(args.out)
        logger.info(
            "saved tokenizer → %s (calibration_mode=%s)",
            args.out,
            "absolute" if corpus_calibration is not None else "per_window",
        )
    elif args.cmd == "encode":
        tok = PatchTransformerTokenizer(device=args.device)
        tok.load(args.ckpt)
        # Fail loud on a normalisation mismatch: a per-window codebook is
        # invalid for an --absolute encode (and vice versa) — the trained input
        # distribution must match the encode-time normalisation.
        ckpt_mode = getattr(tok, "calibration_mode", "per_window")
        want_mode = "absolute" if args.absolute else "per_window"
        if ckpt_mode != want_mode:
            raise SystemExit(
                f"checkpoint {args.ckpt} was trained under "
                f"calibration_mode={ckpt_mode!r} but the encode requested "
                f"{want_mode!r}; retrain the bottleneck with "
                f"`signal_hf_encode train --group {args.group} "
                f"{'--absolute' if args.absolute else ''}` or drop the flag mismatch"
            )
        if args.shots.strip().lower() == "all":
            shot_ids = corpus_shot_ids()
            logger.info("corpus shotlist: %d shots on disk", len(shot_ids))
        else:
            shot_ids = _parse_ids(args.shots)
        corpus_calibration = None
        if args.absolute:
            from imas_ambix.calibration.corpus_compute import load_group_calibration

            corpus_calibration = load_group_calibration(args.group)
            if corpus_calibration is None:
                raise SystemExit(
                    f"--absolute requested but no corpus calibration found for "
                    f"group {args.group!r}; run "
                    f"`python -m imas_ambix.calibration.corpus_compute "
                    f"--group {args.group}` first"
                )
            logger.info(
                "absolute mode: loaded calibration for %d channels (group %r)",
                len(corpus_calibration),
                args.group,
            )
        summary = encode_shots(
            args.group,
            shot_ids,
            tok,
            skip_existing=not args.no_skip_existing,
            num_workers=args.num_workers,
            corpus_calibration=corpus_calibration,
        )
        logger.info(
            "encode summary: %d encoded, %d skipped, %d tokens",
            len(summary["encoded"]),
            len(summary["skipped"]),
            summary["n_tokens_total"],
        )
        if args.manifest is not None:
            args.manifest.parent.mkdir(parents=True, exist_ok=True)
            args.manifest.write_text(json.dumps(_manifest_payload(summary), indent=2))
            logger.info("wrote manifest → %s", args.manifest)
    return 0


def _manifest_payload(summary: dict) -> dict:
    """Condense an encode summary into a JSON manifest.

    Keeps the per-shot encoded list (shot, n_patches, n_channels) and the
    aggregate counts; collapses the skip list to per-reason counts so the
    manifest stays small for a full-corpus run.
    """
    skip_reasons: dict[str, int] = {}
    for s in summary["skipped"]:
        skip_reasons[s["reason"]] = skip_reasons.get(s["reason"], 0) + 1
    return {
        "group": summary["group"],
        "n_encoded": len(summary["encoded"]),
        "n_skipped": len(summary["skipped"]),
        "skip_reasons": skip_reasons,
        "n_tokens_total": summary["n_tokens_total"],
        "encoded": summary["encoded"],
    }


def _torch_perf_setup(device: str) -> None:
    """Reproducibility + H200 tensor-core settings (repo §2b)."""
    try:
        import torch

        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.set_float32_matmul_precision("high")
    except Exception:  # pragma: no cover
        pass


if __name__ == "__main__":
    raise SystemExit(main())
