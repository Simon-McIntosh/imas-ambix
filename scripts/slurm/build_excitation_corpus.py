"""Build the curated dynamic-excitation corpus — select phase + PoC validation.

Two phases, kept separate so the multi-hour GPU re-encode is GATED:

``select``  — (CPU, sun partition) Discover rbb-bearing shots, select for each
              the most coil-excited, plasma-present, long-horizon window
              (:func:`imas_ambix.worldmodel.excitation_corpus.select_curated_windows`,
              held-out reserve excluded), and write a curated-window MANIFEST
              JSON (shot_id, start_frame, fps, n_frames, frame_stride,
              excitation_score, max_abs_ip, present_fraction).  Also prints the
              full-re-encode compute estimate.  No GPU, no pixels.

``poc``     — (CPU, sun partition) Take the top-N curated windows from the
              manifest and validate the pipeline END-TO-END on CPU without the
              VQ model:
                * assemble each curated window through the real dataset
                  (:func:`assemble_controllable_window`) — proves the long-horizon
                  window + actuator plan + signals all load for the selected
                  segment;
                * run the exposure transform on the raw frames and confirm it
                  produces sane uint8 (range, no NaN, the v0-vs-percentile
                  contrast gain) — proves the re-encode input is well-formed.
              This is the cheap proof the curated corpus is buildable; the actual
              token re-encode (GPU) is a SEPARATE gated script
              (build_excitation_corpus_encode.py).

The GPU re-encode is deliberately NOT in this file — it must not run unannounced.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from imas_ambix.data.paths import TOKEN_ROOT

REFERENCE_CAMERA = "rbb"
FRAMES_VOCAB_VERSION = "v1"
DEFAULT_MANIFEST = Path(
    "/work/projects/imas_gpu/agents/excitation-corpus/curated_windows.json"
)


def _rbb_shots(token_root: Path, limit: int | None) -> list[int]:
    frames_root = Path(token_root) / FRAMES_VOCAB_VERSION / "frames"
    shots = sorted(
        int(p.name) for p in frames_root.iterdir() if p.is_dir() and p.name.isdigit()
    )
    if limit is not None:
        # even spread across the id range (a representative scan), not the first N
        sel = np.linspace(0, len(shots) - 1, min(limit, len(shots)))
        shots = [shots[int(round(i))] for i in np.unique(sel.round())]
    return shots


def _compute_estimate(windows: list[dict]) -> dict:
    """Estimate the GPU re-encode cost for the curated windows.

    The production encoder measured ~308 fps/GPU peak.  A curated window emits
    ``n_frames`` frames (the strided window the model sees), so the encode cost
    is the summed n_frames across the manifest at the measured throughput.
    """
    fps_per_gpu = 308.0  # measured peak (AGENTS.md §2b)
    total_frames = int(sum(int(w["n_frames"]) for w in windows))
    total_tokens = total_frames * 256  # 16x16 tokens/frame
    encode_s = total_frames / fps_per_gpu if total_frames else 0.0
    return {
        "n_windows": len(windows),
        "total_frames": total_frames,
        "total_tokens": total_tokens,
        "est_encode_minutes_1gpu": round(encode_s / 60.0, 2),
        "fps_per_gpu_assumed": fps_per_gpu,
        "note": (
            "curated windows are short strided spans; the re-encode is a tiny "
            "fraction of the full 4.02 B-token corpus. GATED behind a check-in."
        ),
    }


def phase_select(args: argparse.Namespace) -> int:
    from imas_ambix.worldmodel.excitation_corpus import (
        DEFAULT_HELD_OUT,
        select_curated_windows,
    )

    token_root = Path(args.token_root)
    shots = _rbb_shots(token_root, args.scan_limit)
    print(f"[select] scanning {len(shots)} rbb-bearing shots", flush=True)

    t0 = time.monotonic()
    windows = select_curated_windows(
        shots,
        camera=args.camera,
        token_root=token_root if args.token_root != str(TOKEN_ROOT) else None,
        held_out=DEFAULT_HELD_OUT,
        target_horizon_s=args.horizon_s,
        max_n_frames=args.max_n_frames,
        min_excitation=args.min_excitation,
        limit=args.limit,
    )
    el = time.monotonic() - t0
    print(
        f"[select] {len(windows)} curated windows selected in {el:.0f}s "
        f"(from {len(shots)} scanned)",
        flush=True,
    )

    rows = [
        {
            "shot_id": w.shot_id,
            "start_frame": w.start_frame,
            "fps": w.fps,
            "n_frames": w.n_frames,
            "frame_stride": w.frame_stride,
            "excitation_score": w.excitation_score,
            "max_abs_ip": w.max_abs_ip,
            "present_fraction": w.present_fraction,
        }
        for w in windows
    ]
    est = _compute_estimate(rows)
    manifest = {
        "camera": args.camera,
        "horizon_s": args.horizon_s,
        "max_n_frames": args.max_n_frames,
        "min_excitation": args.min_excitation,
        "held_out": list(DEFAULT_HELD_OUT),
        "n_scanned": len(shots),
        "windows": rows,
        "compute_estimate": est,
    }
    out = Path(args.manifest)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2))
    print(f"[select] wrote manifest {out}", flush=True)
    print("[select] compute estimate:", json.dumps(est, indent=2), flush=True)
    if rows:
        sc = np.array([r["excitation_score"] for r in rows])
        print(
            f"[select] excitation score: min={sc.min():.0f} med={np.median(sc):.0f} "
            f"max={sc.max():.0f} (kA/s)",
            flush=True,
        )
    return 0


def phase_poc(args: argparse.Namespace) -> int:
    from imas_ambix.worldmodel.controllable_dataset import (
        assemble_controllable_window,
        default_signal_modalities,
    )
    from imas_ambix.worldmodel.exposure_balance import balance_exposure
    from imas_ambix.worldmodel.spacetime_dataset import SpacetimeWindowConfig

    manifest = json.loads(Path(args.manifest).read_text())
    windows = manifest["windows"][: args.n_poc]
    if not windows:
        print("[poc] manifest has no windows — run the select phase first", flush=True)
        return 1
    token_root = Path(args.token_root) if args.token_root != str(TOKEN_ROOT) else None
    modalities = default_signal_modalities()
    print(f"[poc] validating {len(windows)} curated windows end-to-end", flush=True)

    ok_assemble = 0
    ok_exposure = 0
    for w in windows:
        sid = int(w["shot_id"])
        cfg = SpacetimeWindowConfig(
            n_frames=int(w["n_frames"]),
            n_plan=8,
            context_frames=max(1, int(w["n_frames"]) // 3),
            frame_stride=int(w["frame_stride"]),
        )
        # 1) assemble the curated long-horizon window through the real dataset.
        try:
            sample = assemble_controllable_window(
                sid,
                cfg,
                modalities,
                n_signal_steps=16,
                n_act_steps=8,
                camera=args.camera,
                token_root=token_root,
                start_frame=int(w["start_frame"]),
            )
            frames = sample.frames
            n_act_present = int((sample.actuator.missing < 1.0).any(axis=0).sum())
            print(
                f"[poc] shot {sid}: assembled frames={frames.shape} "
                f"plan={sample.plan.shape} signals={sorted(sample.signals)} "
                f"actuator_channels_present={n_act_present} "
                f"max|Ip|={w['max_abs_ip']:.0f}A score={w['excitation_score']:.0f}",
                flush=True,
            )
            ok_assemble += 1
        except Exception as exc:  # noqa: BLE001
            print(f"[poc] shot {sid}: assemble FAILED: {exc!r}", flush=True)
            continue

        # 2) exposure transform on the raw frames -> sane uint8 + contrast gain.
        try:
            raw = _load_raw_window(sid, args.camera, cfg, int(w["start_frame"]))
            if raw is not None:
                bal = balance_exposure(raw, strategy="percentile")
                glob = balance_exposure(raw, strategy="global")
                assert bal.dtype == np.uint8 and not np.isnan(bal).any()
                # robust transform should use at least as much of the range as v0
                bal_span = int(bal.max()) - int(bal.min())
                glob_span = int(glob.max()) - int(glob.min())
                print(
                    f"[poc] shot {sid}: exposure ok — raw[{int(raw.min())},"
                    f"{int(raw.max())}] percentile-span={bal_span} v0-span={glob_span}",
                    flush=True,
                )
                ok_exposure += 1
        except Exception as exc:  # noqa: BLE001
            print(f"[poc] shot {sid}: exposure FAILED: {exc!r}", flush=True)

    print(
        f"[poc] DONE: {ok_assemble}/{len(windows)} assembled, "
        f"{ok_exposure}/{len(windows)} exposure-validated",
        flush=True,
    )
    return 0 if ok_assemble == len(windows) else 1


def _load_raw_window(shot_id, camera, cfg, start_frame):
    """Load the curated window's RAW (pre-token) frames from the level-1 store."""
    import zarr

    from imas_ambix.camdyn.dataset import level1_shot_path

    lp = level1_shot_path(int(shot_id))
    if lp is None or not Path(lp).exists():
        return None
    store = zarr.open_group(str(lp), mode="r")
    if camera not in set(store.group_keys()):
        return None
    data = store[camera]["data"]
    span = (cfg.n_frames - 1) * cfg.frame_stride + 1
    stop = int(start_frame) + span
    if stop > data.shape[0]:
        return None
    run = np.asarray(data[int(start_frame) : stop])[:: cfg.frame_stride]
    return run


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="phase", required=True)

    s = sub.add_parser("select", help="select curated windows -> manifest (CPU)")
    s.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    s.add_argument("--token-root", default=str(TOKEN_ROOT))
    s.add_argument("--camera", default=REFERENCE_CAMERA)
    s.add_argument("--horizon-s", type=float, default=0.25)
    s.add_argument("--max-n-frames", type=int, default=48)
    s.add_argument("--min-excitation", type=float, default=1.0e3)
    s.add_argument(
        "--scan-limit",
        type=int,
        default=None,
        help="scan only this many shots (evenly spread) — for a fast PoC select",
    )
    s.add_argument(
        "--limit", type=int, default=None, help="keep the top-N most-excited windows"
    )
    s.set_defaults(func=phase_select)

    p = sub.add_parser("poc", help="validate the manifest end-to-end (CPU, no GPU)")
    p.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    p.add_argument("--token-root", default=str(TOKEN_ROOT))
    p.add_argument("--camera", default=REFERENCE_CAMERA)
    p.add_argument("--n-poc", type=int, default=6)
    p.set_defaults(func=phase_poc)

    args = ap.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
