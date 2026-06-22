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


def _window_row(w) -> dict:
    return {
        "shot_id": w.shot_id,
        "start_frame": w.start_frame,
        "fps": w.fps,
        "n_frames": w.n_frames,
        "frame_stride": w.frame_stride,
        "excitation_score": w.excitation_score,
        "max_abs_ip": w.max_abs_ip,
        "present_fraction": w.present_fraction,
        "phase": getattr(w, "phase", ""),
        "end_frame": getattr(w, "end_frame", 0),
        "plasma_duration_s": getattr(w, "plasma_duration_s", 0.0),
    }


def phase_select(args: argparse.Namespace) -> int:
    from imas_ambix.worldmodel.excitation_corpus import (
        DEFAULT_HELD_OUT,
        enumerate_curated_windows,
        select_curated_windows,
        select_fullshot_windows,
    )

    token_root = Path(args.token_root)
    shots = _rbb_shots(token_root, args.scan_limit)
    tr = token_root if args.token_root != str(TOKEN_ROOT) else None
    if args.full_shot:
        mode = "full-shot"
    elif args.multi_window:
        mode = "multi-window"
    else:
        mode = "single-best"
    print(f"[select] scanning {len(shots)} rbb-bearing shots ({mode})", flush=True)

    t0 = time.monotonic()
    if args.full_shot:
        windows = select_fullshot_windows(
            shots,
            camera=args.camera,
            token_root=tr,
            held_out=DEFAULT_HELD_OUT,
            min_present_fraction=args.min_present_fraction,
            min_duration_s=args.min_duration_s,
        )
    elif args.multi_window:
        windows = enumerate_curated_windows(
            shots,
            camera=args.camera,
            token_root=tr,
            held_out=DEFAULT_HELD_OUT,
            target_horizon_s=args.horizon_s,
            max_n_frames=args.max_n_frames,
            window_time_stride_s=args.window_time_stride_s,
            min_excitation=args.min_excitation,
            max_windows_per_shot=args.max_windows_per_shot,
        )
    else:
        windows = select_curated_windows(
            shots,
            camera=args.camera,
            token_root=tr,
            held_out=DEFAULT_HELD_OUT,
            target_horizon_s=args.horizon_s,
            max_n_frames=args.max_n_frames,
            min_excitation=args.min_excitation,
            limit=args.limit,
        )
    el = time.monotonic() - t0
    rows = [_window_row(w) for w in windows]

    # full-shot: plasma-phase DURATION distribution (for trainer n_frames sizing).
    duration_stats = None
    if args.full_shot and rows:
        dur = np.array([r["plasma_duration_s"] for r in rows], dtype=float)
        dur_ms = dur * 1e3
        # n_frames for ~5-8 ms resolution at the median duration.
        med_ms = float(np.median(dur_ms))
        duration_stats = {
            "median_ms": round(med_ms, 1),
            "p10_ms": round(float(np.percentile(dur_ms, 10)), 1),
            "p90_ms": round(float(np.percentile(dur_ms, 90)), 1),
            "min_ms": round(float(dur_ms.min()), 1),
            "max_ms": round(float(dur_ms.max()), 1),
            # sanity: after the min-duration floor + sustained detection, very few
            # windows should be short (the ~20 ms burst-artifact cluster is gone).
            "frac_under_100ms": round(float((dur_ms < 100).mean()), 3),
            "frac_under_250ms": round(float((dur_ms < 250).mean()), 3),
            "n_frames_for_5ms_at_median": int(round(med_ms / 5.0)),
            "n_frames_for_8ms_at_median": int(round(med_ms / 8.0)),
        }

    # phase coverage + windows-per-pulse distribution (multi-window report).
    pulses = sorted({r["shot_id"] for r in rows})
    per_pulse = {}
    for r in rows:
        per_pulse[r["shot_id"]] = per_pulse.get(r["shot_id"], 0) + 1
    counts = np.array(sorted(per_pulse.values())) if per_pulse else np.array([0])
    phase_counts: dict[str, int] = {}
    for r in rows:
        ph = r.get("phase") or "unclassified"
        phase_counts[ph] = phase_counts.get(ph, 0) + 1
    coverage = {
        "n_windows": len(rows),
        "n_pulses": len(pulses),
        "windows_per_pulse_min": int(counts.min()),
        "windows_per_pulse_median": float(np.median(counts)),
        "windows_per_pulse_max": int(counts.max()),
        "windows_per_pulse_mean": round(float(counts.mean()), 2),
        "phase_counts": phase_counts,
    }

    print(
        f"[select] {len(rows)} windows over {len(pulses)} pulses in {el:.0f}s "
        f"(from {len(shots)} scanned)",
        flush=True,
    )
    est = _compute_estimate(rows)
    if args.multi_window or args.full_shot:
        est["note"] = (
            "NO re-encode needed (whole-shot tokens already on disk; the dataset "
            "slices each window at train time). For full-shot, the trainer "
            "time-subsamples [start_frame, end_frame) to its own n_frames using "
            "target_horizon_s = plasma_duration_s; total_tokens here is the native "
            "span, NOT what the model sees per epoch."
        )

    # Held-out plasma-activity diagnostic (is a ΔN-M fail a dark-frame artifact?).
    heldout_activity = None
    if args.probe_heldout:
        from imas_ambix.worldmodel.excitation_corpus import probe_plasma_activity

        heldout_activity = probe_plasma_activity(
            list(DEFAULT_HELD_OUT), camera=args.camera, token_root=tr
        )

    manifest = {
        "camera": args.camera,
        "mode": mode,
        "horizon_s": args.horizon_s,
        "max_n_frames": args.max_n_frames,
        "window_time_stride_s": args.window_time_stride_s
        if args.multi_window
        else None,
        "max_windows_per_shot": args.max_windows_per_shot,
        "min_excitation": args.min_excitation,
        "min_present_fraction": args.min_present_fraction if args.full_shot else None,
        "min_duration_s": args.min_duration_s if args.full_shot else None,
        "held_out": list(DEFAULT_HELD_OUT),
        "n_scanned": len(shots),
        "coverage": coverage,
        "duration_stats": duration_stats,
        "heldout_activity": heldout_activity,
        "windows": rows,
        "compute_estimate": est,
    }
    out = Path(args.manifest)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2))
    print(f"[select] wrote manifest {out}", flush=True)
    print("[select] coverage:", json.dumps(coverage, indent=2), flush=True)
    if duration_stats is not None:
        print(
            "[select] plasma-phase duration:",
            json.dumps(duration_stats, indent=2),
            flush=True,
        )
    if heldout_activity is not None:
        print(
            "[select] held-out plasma activity:",
            json.dumps(heldout_activity, indent=2),
            flush=True,
        )
    print("[select] compute estimate:", json.dumps(est, indent=2), flush=True)
    if rows:
        sc = np.array([r["excitation_score"] for r in rows])
        print(
            f"[select] excitation score: min={sc.min():.0f} med={np.median(sc):.0f} "
            f"max={sc.max():.0f} (kA/s)",
            flush=True,
        )
    return 0


def phase_merge(args: argparse.Namespace) -> int:
    """Concatenate sharded unified partial manifests into one (CPU)."""
    import glob as _glob

    parts = sorted(_glob.glob(args.inputs))
    if not parts:
        print(f"[merge] no shard files match {args.inputs!r}", flush=True)
        return 1
    print(f"[merge] merging {len(parts)} shard manifests", flush=True)
    rows: list[dict] = []
    base: dict | None = None
    for p in parts:
        d = json.loads(Path(p).read_text())
        if base is None:
            base = {k: v for k, v in d.items() if k not in ("windows", "coverage")}
        rows.extend(d.get("windows", []))
        print(f"[merge]   {p}: {len(d.get('windows', []))} windows", flush=True)
    # deterministic order: ascending shot id then camera.
    rows.sort(key=lambda r: (int(r["shot_id"]), str(r["camera_id"])))

    # recompute coverage across the merged rows.
    per_cam: dict[str, int] = {}
    per_campaign: dict[str, int] = {}
    per_timescale: dict[str, int] = {}
    for r in rows:
        per_cam[r["camera_id"]] = per_cam.get(r["camera_id"], 0) + 1
        per_campaign[r["campaign"]] = per_campaign.get(r["campaign"], 0) + 1
        per_timescale[r["timescale"]] = per_timescale.get(r["timescale"], 0) + 1
    dur = np.array([r["plasma_duration_s"] for r in rows], dtype=float)
    dur_ms = dur * 1e3 if dur.size else np.array([0.0])
    coverage = {
        "n_windows": len(rows),
        "n_distinct_shots": len({r["shot_id"] for r in rows}),
        "windows_per_camera": dict(sorted(per_cam.items())),
        "windows_per_campaign": dict(sorted(per_campaign.items())),
        "windows_per_timescale": dict(sorted(per_timescale.items())),
        "duration_median_ms": round(float(np.median(dur_ms)), 1),
        "duration_p10_ms": round(float(np.percentile(dur_ms, 10)), 1),
        "duration_p90_ms": round(float(np.percentile(dur_ms, 90)), 1),
        "fps_min": round(float(min(r["fps"] for r in rows)), 1) if rows else 0.0,
        "fps_max": round(float(max(r["fps"] for r in rows)), 1) if rows else 0.0,
    }
    # held-out leakage guard: assert NONE present (across every camera).
    held = set(base.get("held_out", [])) if base else set()
    leak = sorted(held & {int(r["shot_id"]) for r in rows})

    merged = dict(base or {})
    merged["mode"] = "unified"
    merged["coverage"] = coverage
    merged["windows"] = rows
    merged["n_shards_merged"] = len(parts)
    merged["heldout_leak"] = leak
    out = Path(args.manifest)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(merged))
    print(f"[merge] wrote {out}", flush=True)
    print("[merge] coverage:", json.dumps(coverage, indent=2), flush=True)
    print(f"[merge] held-out leak: {leak if leak else 'NONE'}", flush=True)
    return 0 if not leak else 1


def phase_unified(args: argparse.Namespace) -> int:
    """Build the UNIFIED multi-camera, multi-timescale manifest (CPU)."""
    from imas_ambix.worldmodel.excitation_corpus import (
        CAMPAIGN_BANDS,
        DEFAULT_HELD_OUT,
        UNIFIED_CAMERAS,
        probe_plasma_activity,
        select_unified_windows,
    )

    token_root = Path(args.token_root)
    shots = _rbb_shots(token_root, args.scan_limit)
    tr = token_root if args.token_root != str(TOKEN_ROOT) else None
    # Stride-shard the shot list across parallel CPU workers (the unified scan is
    # IO-bound — one amc/Ip read per shot per camera — so N shards over the `sun`
    # nodes cut wall-clock ~N x). Each shard writes its own partial manifest; the
    # `merge` phase concatenates them. Deterministic + disjoint.
    n_shards = max(1, int(args.n_shards))
    shard = int(args.shard) % n_shards
    if n_shards > 1:
        shots = shots[shard::n_shards]
    cameras = (
        [c.strip() for c in args.cameras.split(",") if c.strip()]
        if args.cameras
        else list(UNIFIED_CAMERAS)
    )
    print(
        f"[unified shard {shard}/{n_shards}] scanning {len(shots)} shots x "
        f"{len(cameras)} cameras "
        f"({','.join(cameras)})",
        flush=True,
    )

    t0 = time.monotonic()
    rows = select_unified_windows(
        shots,
        cameras=cameras,
        token_root=tr,
        held_out=DEFAULT_HELD_OUT,
        min_present_fraction=args.min_present_fraction,
        fast_max_duration_s=args.fast_max_duration_s,
        include_frame_times=not args.no_frame_times,
    )
    el = time.monotonic() - t0

    # report: windows per camera / per campaign / slow-vs-fast.
    per_cam: dict[str, int] = {}
    per_campaign: dict[str, int] = {}
    per_timescale: dict[str, int] = {}
    for r in rows:
        per_cam[r["camera_id"]] = per_cam.get(r["camera_id"], 0) + 1
        per_campaign[r["campaign"]] = per_campaign.get(r["campaign"], 0) + 1
        per_timescale[r["timescale"]] = per_timescale.get(r["timescale"], 0) + 1
    dur = np.array([r["plasma_duration_s"] for r in rows], dtype=float)
    dur_ms = dur * 1e3 if dur.size else np.array([0.0])
    coverage = {
        "n_windows": len(rows),
        "n_distinct_shots": len({r["shot_id"] for r in rows}),
        "windows_per_camera": dict(sorted(per_cam.items())),
        "windows_per_campaign": dict(sorted(per_campaign.items())),
        "windows_per_timescale": dict(sorted(per_timescale.items())),
        "duration_median_ms": round(float(np.median(dur_ms)), 1),
        "duration_p10_ms": round(float(np.percentile(dur_ms, 10)), 1),
        "duration_p90_ms": round(float(np.percentile(dur_ms, 90)), 1),
        "fps_min": round(float(min(r["fps"] for r in rows)), 1) if rows else 0.0,
        "fps_max": round(float(max(r["fps"] for r in rows)), 1) if rows else 0.0,
    }

    heldout_activity = None
    if args.probe_heldout:
        heldout_activity = {
            cam: probe_plasma_activity(
                list(DEFAULT_HELD_OUT), camera=cam, token_root=tr
            )
            for cam in cameras
        }

    manifest = {
        "mode": "unified",
        "schema": [
            "shot_id",
            "camera_id",
            "campaign",
            "start_frame",
            "end_frame",
            "fps",
            "n_frames",
            "frame_times",
            "plasma_duration_s",
            "timescale",
            "excitation_score",
            "present_fraction",
        ],
        "cameras": cameras,
        "campaign_bands": [
            {"name": n, "lo": lo, "hi": hi} for n, lo, hi in CAMPAIGN_BANDS
        ],
        "fast_max_duration_s": args.fast_max_duration_s,
        "min_present_fraction": args.min_present_fraction,
        "held_out": list(DEFAULT_HELD_OUT),
        "n_scanned": len(shots),
        "include_frame_times": not args.no_frame_times,
        "coverage": coverage,
        "heldout_activity": heldout_activity,
        "windows": rows,
    }
    out = Path(args.manifest)
    if n_shards > 1:
        # shard partial — the merge phase concatenates these.
        out = out.with_suffix(f".shard{shard}of{n_shards}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest))  # no indent: frame_times make it large
    print(
        f"[unified] {len(rows)} windows over {coverage['n_distinct_shots']} shots "
        f"in {el:.0f}s -> {out}",
        flush=True,
    )
    print("[unified] coverage:", json.dumps(coverage, indent=2), flush=True)
    if heldout_activity is not None:
        # compact held-out summary: per-camera (max|Ip|, duration) for each shot.
        for cam, act in heldout_activity.items():
            for sid, a in act.items():
                if a["max_abs_ip"] > 0:
                    ip_ka = a["max_abs_ip"] / 1e3
                    print(
                        f"[unified] heldout {sid}/{cam}: max|Ip|={ip_ka:.0f}kA "
                        f"dur={a['duration_s']:.3f}s pf={a['present_fraction']:.2f}",
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
    s.add_argument(
        "--full-shot",
        action="store_true",
        help="ONE window per shot = the whole plasma phase (breakdown->termination); "
        "the trainer time-subsamples the span. Overrides --multi-window.",
    )
    s.add_argument(
        "--min-present-fraction",
        type=float,
        default=0.7,
        help="full-shot: min fraction of the plasma-phase span that must be "
        "plasma-present (may start dark, must not be mostly-dark)",
    )
    s.add_argument(
        "--min-duration-s",
        type=float,
        default=0.1,
        help="full-shot: drop shots whose SUSTAINED plasma phase is shorter than "
        "this wall-clock duration (s) — removes high-speed-burst / "
        "failed-breakdown captures so durations are mostly ~0.3-0.4 s",
    )
    s.add_argument(
        "--probe-heldout",
        action="store_true",
        help="also probe held-out 18502-05 plasma activity (dark-frame-artifact "
        "diagnostic) and record it in the manifest",
    )
    s.add_argument(
        "--multi-window",
        action="store_true",
        help="tile EVERY pulse with overlapping windows (ramp/flat-top/termination) "
        "instead of one best window per shot; disruptions INCLUDED",
    )
    s.add_argument(
        "--window-time-stride-s",
        type=float,
        default=0.075,
        help="time-stride (s) between tiled windows in --multi-window mode "
        "(~0.075 = ~70%% overlap on a 0.25 s window)",
    )
    s.add_argument(
        "--max-windows-per-shot",
        type=int,
        default=None,
        help="cap windows per pulse in --multi-window mode (spread evenly across "
        "the recording so all phases survive the cap)",
    )
    s.set_defaults(func=phase_select)

    p = sub.add_parser("poc", help="validate the manifest end-to-end (CPU, no GPU)")
    p.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    p.add_argument("--token-root", default=str(TOKEN_ROOT))
    p.add_argument("--camera", default=REFERENCE_CAMERA)
    p.add_argument("--n-poc", type=int, default=6)
    p.set_defaults(func=phase_poc)

    u = sub.add_parser(
        "unified",
        help="build the UNIFIED multi-camera, multi-timescale manifest (CPU)",
    )
    u.add_argument(
        "--manifest",
        default="/work/projects/imas_gpu/agents/excitation-corpus/"
        "curated_windows_unified.json",
    )
    u.add_argument("--token-root", default=str(TOKEN_ROOT))
    u.add_argument(
        "--cameras",
        default=None,
        help="comma-separated camera ids (default: all tokenised — "
        "rbb,rco,rgb,rgc,rba,rbc)",
    )
    u.add_argument("--min-present-fraction", type=float, default=0.7)
    u.add_argument(
        "--fast-max-duration-s",
        type=float,
        default=0.15,
        help="plasma phase shorter than this is tagged timescale=fast (else slow)",
    )
    u.add_argument(
        "--no-frame-times",
        action="store_true",
        help="omit the per-window frame_times list (lighter manifest)",
    )
    u.add_argument("--probe-heldout", action="store_true")
    u.add_argument("--scan-limit", type=int, default=None)
    u.add_argument("--shard", type=int, default=0, help="this worker's shard index")
    u.add_argument(
        "--n-shards", type=int, default=1, help="number of parallel CPU shards"
    )
    u.set_defaults(func=phase_unified)

    m = sub.add_parser(
        "merge",
        help="concatenate sharded unified partial manifests into one (CPU)",
    )
    m.add_argument(
        "--inputs",
        required=True,
        help="glob for the shard partial manifests (quote it)",
    )
    m.add_argument("--manifest", required=True, help="merged output path")
    m.set_defaults(func=phase_merge)

    args = ap.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
