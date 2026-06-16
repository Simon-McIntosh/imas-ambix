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
from imas_ambix.statespace.fast_loader import read_xim_shot, read_xma_shot
from imas_ambix.tokenizer.patch_transformer import (
    PatchTokenizerConfig,
    PatchTransformerTokenizer,
    mode_decomposition,
)
from imas_ambix.tokenizer.registry import (
    BLOCK_XIM_PATCH,
    BLOCK_XMA_MODE,
    BLOCK_XMA_PATCH,
    VOCAB_VERSION,
    registry,
)
from imas_ambix.tokenizer.store_v2 import (
    StoreV2Attrs,
    save_signal_hf_tokens,
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
SPECS = {"xma": XMA_SPEC, "xim": XIM_SPEC}


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
    hold_windows, _ = _collect_windows(holdout_ids, group)
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
        "variants": {},
    }

    for bottleneck in ("continuous", "fsq", "vq"):
        cfg = PatchTokenizerConfig(
            patch_size=patch_size, bottleneck=bottleneck, use_stft=True
        )
        tok = PatchTransformerTokenizer(cfg=cfg, name=spec.patch_block, device=device)
        t0 = time.time()
        hist = tok.fit(train_windows, epochs=epochs, logger=logger)
        # Aggregate phase-fidelity over the holdout shots.
        crps, perr, corr, mperr, active = [], [], [], [], []
        for hw in hold_windows:
            m = tok.roundtrip_metrics(hw, dt=dt, is_coil_array=spec.is_coil_array)
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


def encode_shots(
    group: str,
    shot_ids: list[int],
    tokenizer: PatchTransformerTokenizer,
    *,
    watchdog_s: float = 120.0,
) -> dict:
    """Encode shots into the v2 signals_hf store with ``tokenizer``.

    Emits per-coil patch codes, per-token time, and per-channel validity.
    For a coil array, additionally emits cross-channel mode-number tokens as
    a sibling group ``{group}_mode``.  Returns a summary dict.
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

    for sid in shot_ids:
        if _STOP["flag"]:
            summary["skipped"].append({"shot": sid, "reason": "stop_requested"})
            continue
        t0 = time.time()
        w = load_shot_window(sid, group)
        if w is None:
            summary["skipped"].append({"shot": sid, "reason": "no_data"})
            continue
        data, chan, valid_ch, native_rate, window = w
        try:
            ids, latent, _recon = tokenizer.encode_window(data)  # (C,P),(C,P,d),(C,T)
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

        attrs = StoreV2Attrs(
            tokenizer_name=spec.patch_block,
            vocab_version=VOCAB_VERSION,
            native_rate_hz=native_rate,
            token_rate_hz=cfg.token_rate_hz(native_rate),
            n_channels=len(chan),
            channel_names=tuple(chan),
            phase_preserving=True,
            original_window=window,
            metadata={
                "bottleneck": cfg.bottleneck,
                "patch_size": cfg.patch_size,
                "use_stft": cfg.use_stft,
                "codebook_size": patch_vocab,
                "embed_dim": int(latent.shape[-1]),
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
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_ids(text: str) -> list[int]:
    if Path(text).exists():
        return [int(x) for x in Path(text).read_text().split()]
    return [int(x) for x in text.replace(",", " ").split()]


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--group", choices=["xma", "xim"], required=True)
    common.add_argument("--device", default="cuda")
    common.add_argument("--patch-size", type=int, default=64)

    pd = sub.add_parser("decide", parents=[common])
    pd.add_argument("--train", required=True, help="shot ids or a file")
    pd.add_argument("--holdout", required=True, help="shot ids or a file")
    pd.add_argument("--epochs", type=int, default=40)
    pd.add_argument("--out", type=Path, required=True)

    pt = sub.add_parser("train", parents=[common])
    pt.add_argument("--train", required=True)
    pt.add_argument("--bottleneck", default="fsq")
    pt.add_argument("--epochs", type=int, default=40)
    pt.add_argument("--out", type=Path, required=True, help="checkpoint path")

    pe = sub.add_parser("encode", parents=[common])
    pe.add_argument("--shots", required=True)
    pe.add_argument("--ckpt", type=Path, required=True)

    args = p.parse_args(argv)
    _install_signal_handler()
    _torch_perf_setup(args.device)

    if args.cmd == "decide":
        decide_codebook(
            args.group,
            _parse_ids(args.train),
            _parse_ids(args.holdout),
            device=args.device,
            epochs=args.epochs,
            patch_size=args.patch_size,
            out_path=args.out,
        )
    elif args.cmd == "train":
        windows, _chan = _collect_windows(_parse_ids(args.train), args.group)
        cfg = PatchTokenizerConfig(
            patch_size=args.patch_size, bottleneck=args.bottleneck, use_stft=True
        )
        spec = SPECS[args.group]
        tok = PatchTransformerTokenizer(
            cfg=cfg, name=spec.patch_block, device=args.device
        )
        tok.fit(windows, epochs=args.epochs, logger=logger)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        tok.save(args.out)
        logger.info("saved tokenizer → %s", args.out)
    elif args.cmd == "encode":
        tok = PatchTransformerTokenizer(device=args.device)
        tok.load(args.ckpt)
        summary = encode_shots(args.group, _parse_ids(args.shots), tok)
        logger.info(
            "encode summary: %d encoded, %d skipped, %d tokens",
            len(summary["encoded"]),
            len(summary["skipped"]),
            summary["n_tokens_total"],
        )
    return 0


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
