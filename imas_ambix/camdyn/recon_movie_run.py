"""GPU orchestration for the headline demonstration GIFs / panels / forecast.

This is the predict→decode→render driver for :mod:`recon_movie`.  It is the
heavy, GPU-bound, corpus-bound entry point (no unit tests — the pure helpers
it calls live in :mod:`recon_movie` and :mod:`reconstruction_demo` and ARE
tested).  Two phases, exactly like :mod:`reconstruction_demo`:

  1. predict (this venv, GPU): load BOTH trained arms once, materialise the
     selected held-out windows, run each arm forward under the per-scenario
     mask, and dump token grids to a single ``.npz`` bundle;
  2. decode (the Open-MAGVIT2 venv): reuse
     :func:`reconstruction_demo.run_decode_subprocess` to decode every grid to
     a 256² image through the frozen VQModel.

The render step then lays out the GIFs / panels (per-frame normalised) and
computes the forecast sweep.  All model loads happen ONCE outside the
per-window loop (repo §2b in-process performance rule).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
from pathlib import Path

import numpy as np

from imas_ambix.camdyn import recon_movie as mv
from imas_ambix.camdyn import reconstruction_demo as rd

logger = logging.getLogger(__name__)

# Flat-top high-activity held-out shots (the report's reference shot first).
FLATTOP_SHOTS = (24065, 24446, 23937)
# A bright single shot for the GIFs (small files: one shot, ~16-24 frames).
GIF_SHOT = 24065
# Conditioning frontier frame (frames < this are observed, rest forecast).
DEFAULT_FRONTIER = 8


# ---------------------------------------------------------------------------
# Generic token-bundle builder (decode_phase-compatible: grids/index/meta)
# ---------------------------------------------------------------------------


class BundleBuilder:
    """Accumulate ``(F,16,16)`` global-id grids into a decode_phase bundle."""

    def __init__(self):
        self._grids: list[np.ndarray] = []
        self._index: list[dict] = []
        self._meta: list[dict] = []

    def add_window(self, meta_entry: dict) -> int:
        wi = len(self._meta)
        self._meta.append(meta_entry)
        return wi

    def add_grid(self, grid, window: int, scenario: str, role: str):
        self._index.append(
            {
                "window": window,
                "scenario": scenario,
                "role": role,
                "slot": len(self._grids),
            }
        )
        self._grids.append(np.asarray(grid, dtype=np.int64))

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            grids=np.stack(self._grids).astype(np.int64),
            index=json.dumps(self._index),
            meta=json.dumps(self._meta),
        )


def _meta_for(win: rd.DemoWindow, scenarios) -> dict:
    return {
        "shot_id": int(win.shot_id),
        "start": int(win.start),
        "frame_time": np.asarray(win.frame_time).tolist(),
        "dt": np.asarray(win.dt).tolist(),
        "valid": np.asarray(win.valid).astype(bool).tolist(),
        "motion_fraction": float(win.motion_fraction),
        "scenarios": list(scenarios),
    }


def run(out_dir: Path, artifact_path: Path, *, device: str = "cuda"):
    """Full predict→decode→render pipeline for every deliverable."""
    import torch

    from imas_ambix.camdyn.arm_compare import _load_arm

    out_dir = Path(out_dir)
    work_dir = Path(
        tempfile.mkdtemp(prefix="camdyn-movie-", dir=os.environ.get("TMPDIR", "/tmp"))
    )
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    logger.info("[movie] loading dynamics + baseline arms on %s", dev)
    dyn_model, _dc, dyn_stats = _load_arm(rd.DYNAMICS_CKPT, torch, dev)
    base_model, _bc, base_stats = _load_arm(rd.BASELINE_CKPT, torch, dev)

    try:
        _run_inner(
            out_dir,
            artifact_path,
            work_dir,
            torch,
            dev,
            dyn_model,
            dyn_stats,
            base_model,
            base_stats,
        )
    finally:
        import contextlib

        for m in (dyn_model, base_model):
            with contextlib.suppress(Exception):
                del m
        if dev.type == "cuda":
            torch.cuda.empty_cache()


def _arm_pred(model, stats, torch, dev, win, scenario, frontier):
    _vis, pred = rd.predict_window_arm(
        model, torch, dev, win, stats, scenario, frontier
    )
    return pred


def _run_inner(
    out_dir,
    artifact_path,
    work_dir,
    torch,
    dev,
    dyn_model,
    dyn_stats,
    base_model,
    base_stats,
):
    bb = BundleBuilder()
    plan: list[dict] = []

    def _register(win, scenario, tag, *, roles, frontier=DEFAULT_FRONTIER, extra=None):
        """Add a window + its requested role grids to the decode bundle.

        ``roles`` is any of ``{"baseline", "dynamics", "persistence"}``.
        ``baseline``/``dynamics`` run the matching arm forward under the
        scenario mask; ``persistence`` freezes the last OBSERVED frame's
        tokens across the forecast window (the honest static forecast
        comparator — the per-frame baseline has no temporal mechanism in
        frontier mode).
        """
        wi = bb.add_window(_meta_for(win, [scenario]))
        bb.add_grid(win.true_tokens, wi, scenario, "true")
        if "baseline" in roles:
            bb.add_grid(
                _arm_pred(base_model, base_stats, torch, dev, win, scenario, frontier),
                wi,
                scenario,
                "baseline",
            )
        if "dynamics" in roles:
            bb.add_grid(
                _arm_pred(dyn_model, dyn_stats, torch, dev, win, scenario, frontier),
                wi,
                scenario,
                "dynamics",
            )
        if "persistence" in roles:
            bb.add_grid(
                mv.persistence_tokens(win.true_tokens, frontier),
                wi,
                scenario,
                "persistence",
            )
        item = {"win": win, "scenario": scenario, "tag": tag, "window_index": wi}
        if extra:
            item.update(extra)
        plan.append(item)
        return item

    recon_roles = ("baseline", "dynamics")

    # ---- window selection ---------------------------------------------------
    flattop = rd.select_windows(
        list(FLATTOP_SHOTS), n_frames=16, stride=8, windows_per_shot=1
    )
    gif_win = next((w for w in flattop if w.shot_id == GIF_SHOT), flattop[0])
    rampup = _select_rampup_window(n_frames=16)
    elm = _select_elm_window(n_frames=16)
    forecast = _select_forecast_window()
    elm_forecast = _select_elm_window(n_frames=16, decimate_ms=50.0)

    # ---- ELM (TOP PRIORITY) -------------------------------------------------
    if elm is not None:
        ew, e_peak, e_dt = elm
        _register(
            ew,
            "clipped",
            "elm_recon",
            roles=recon_roles,
            extra={"peak": e_peak, "dalpha_t": e_dt},
        )
    if elm_forecast is not None:
        efw, ef_peak, ef_dt = elm_forecast
        _register(
            efw,
            "frontier",
            "elm_forecast",
            roles=("dynamics", "persistence"),
            extra={"peak": ef_peak, "dalpha_t": ef_dt},
        )

    # ---- reconstruction GIFs + static panels (flat-top reference) ----------
    _register(gif_win, "clipped", "recon_window", roles=recon_roles)
    _register(gif_win, "signals_only", "recon_signals", roles=recon_roles)

    # ---- ramp-up (both modes on the rising-current window) -----------------
    if rampup is not None:
        _register(rampup, "clipped", "rampup_clipped", roles=recon_roles)
        _register(rampup, "signals_only", "rampup_signals", roles=recon_roles)

    # ---- forecast rollout (wide decimated FRONTIER window) -----------------
    if forecast is not None:
        fw, f_h = forecast
        _register(
            fw,
            "frontier",
            "forecast_rollout",
            roles=("dynamics", "persistence"),
            extra={"horizon_ms": f_h},
        )

    # ---- decode every registered grid in ONE batched pass ------------------
    token_bundle = work_dir / "tokens.npz"
    image_bundle = work_dir / "images.npz"
    bb.save(token_bundle)
    rd.run_decode_subprocess(token_bundle, image_bundle, "cuda")

    data = np.load(str(image_bundle), allow_pickle=True)
    images = np.asarray(data["images"], dtype=np.uint8)
    index = json.loads(str(data["index"]))
    meta = json.loads(str(data["meta"]))
    slot = {(e["window"], e["scenario"], e["role"]): e["slot"] for e in index}

    raw_cache: dict[tuple[int, int], np.ndarray | None] = {}

    def _raw(win):
        key = (int(win.shot_id), int(win.start))
        if key not in raw_cache:
            raw_cache[key] = rd.load_raw_frames(
                win.shot_id, win.start, win.true_tokens.shape[0]
            )
        return raw_cache[key]

    # ---- render: ELM first (commit-order priority) -------------------------
    elm_item = _find(plan, "elm_recon")
    if elm_item is not None:
        mv.assemble_three_row_panel(
            "clipped",
            meta[elm_item["window_index"]],
            images,
            slot,
            elm_item["window_index"],
            _raw(elm_item["win"]),
            out_path=out_dir / "fig-cdw-elm-recon.png",
            title_extra="ELM edge-burst",
            highlight_frame=elm_item.get("peak"),
        )
        _render_recon_gif(
            elm_item,
            "clipped",
            meta,
            images,
            slot,
            _raw,
            out_dir / "elm-reconstruction.gif",
            middle_role="baseline",
            middle_label="baseline",
            highlight_frame=elm_item.get("peak"),
        )

    elmf_item = _find(plan, "elm_forecast")
    if elmf_item is not None:
        _render_forecast_gif(
            elmf_item,
            meta,
            images,
            slot,
            _raw,
            out_dir / "elm-forecast.gif",
            frontier=DEFAULT_FRONTIER,
            highlight_frame=elmf_item.get("peak"),
        )

    # ---- reconstruction GIFs ------------------------------------------------
    _render_recon_gif(
        _find(plan, "recon_window"),
        "clipped",
        meta,
        images,
        slot,
        _raw,
        out_dir / "recon-from-window.gif",
        middle_role="baseline",
        middle_label="baseline",
    )
    _render_recon_gif(
        _find(plan, "recon_signals"),
        "signals_only",
        meta,
        images,
        slot,
        _raw,
        out_dir / "recon-from-signals.gif",
        middle_role="baseline",
        middle_label="baseline",
    )

    # ---- static recon panels ------------------------------------------------
    rw = _find(plan, "recon_window")
    if rw is not None:
        mv.assemble_three_row_panel(
            "clipped",
            meta[rw["window_index"]],
            images,
            slot,
            rw["window_index"],
            _raw(rw["win"]),
            out_path=out_dir / "fig-cdw-recon-window.png",
            title_extra="flat-top high-activity",
        )
    rs = _find(plan, "recon_signals")
    if rs is not None:
        mv.assemble_three_row_panel(
            "signals_only",
            meta[rs["window_index"]],
            images,
            slot,
            rs["window_index"],
            _raw(rs["win"]),
            out_path=out_dir / "fig-cdw-recon-signals.png",
            title_extra="flat-top high-activity",
        )

    # ---- ramp-up combined panel --------------------------------------------
    _render_rampup(plan, meta, images, slot, _raw, out_dir)

    # ---- forecast rollout GIF + sweep --------------------------------------
    fc_item = _find(plan, "forecast_rollout")
    if fc_item is not None:
        _render_forecast_gif(
            fc_item,
            meta,
            images,
            slot,
            _raw,
            out_dir / "forecast-rollout.gif",
            frontier=DEFAULT_FRONTIER,
        )

    _forecast_sweep(
        torch, dev, dyn_model, dyn_stats, base_model, base_stats, artifact_path, out_dir
    )


def _find(plan, tag):
    return next((p for p in plan if p["tag"] == tag), None)


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------


def _gif_columns(n_frames: int, max_cols: int = 24) -> list[int]:
    if n_frames <= max_cols:
        return list(range(n_frames))
    return list(np.linspace(0, n_frames - 1, max_cols).round().astype(int))


def _render_recon_gif(
    item,
    scenario,
    meta,
    images,
    slot,
    raw_fn,
    out_path,
    *,
    middle_role,
    middle_label,
    highlight_frame=None,
):
    """3-panel reconstruction GIF: GT | static comparator | dynamics."""
    if item is None:
        logger.warning("[movie] no window for recon GIF %s", out_path.name)
        return
    wi = item["window_index"]
    win = item["win"]
    m = meta[wi]
    ft = np.asarray(m["frame_time"], dtype=float)
    raw = raw_fn(win)
    mid = images[slot[(wi, scenario, middle_role)]]
    dyn = images[slot[(wi, scenario, "dynamics")]]
    box = rd.clip_box(scenario)
    cols = _gif_columns(ft.shape[0])

    frames = []
    for fi in cols:
        gt_src = (
            raw[fi].astype(np.float64)
            if (raw is not None and fi < raw.shape[0])
            else rd._to_aspect(dyn[fi]).astype(np.float64)
        )
        vmin, vmax = rd.display_limits(gt_src)
        gt_u = mv.normalise_for_display(gt_src, vmin, vmax)
        mv._draw_clip_box(gt_u, box, value=255)
        mid_u = mv.normalise_for_display(rd._to_aspect(mid[fi]), vmin, vmax)
        dyn_u = mv.normalise_for_display(rd._to_aspect(dyn[fi]), vmin, vmax)
        dt_ms = (ft[fi] - ft[0]) * 1e3
        burst = (
            " ELM" if (highlight_frame is not None and fi == highlight_frame) else ""
        )
        frame = mv.panel_strip(
            [gt_u, mid_u, dyn_u],
            ["ground truth", middle_label, "dynamics"],
            scale=3,
            counter=f"f{fi} t{dt_ms:+.1f}ms{burst}",
        )
        frames.append(frame)
    mv.write_gif(frames, out_path, duration_ms=140)
    logger.info(
        "[movie] %s: shot %d %s %.0f-%.0f ms, %d frames",
        out_path.name,
        m["shot_id"],
        scenario,
        ft[0] * 1e3,
        ft[-1] * 1e3,
        len(frames),
    )


def _render_forecast_gif(
    item, meta, images, slot, raw_fn, out_path, *, frontier, highlight_frame=None
):
    """3-panel forecast GIF: GT | persistence (frozen) | dynamics."""
    if item is None:
        logger.warning("[movie] no window for forecast GIF %s", out_path.name)
        return
    wi = item["window_index"]
    win = item["win"]
    m = meta[wi]
    ft = np.asarray(m["frame_time"], dtype=float)
    n_frames = ft.shape[0]
    raw = raw_fn(win)
    per = images[slot[(wi, "frontier", "persistence")]]
    dyn = images[slot[(wi, "frontier", "dynamics")]]
    frames = []
    for fi in range(n_frames):
        gt_src = (
            raw[fi].astype(np.float64)
            if (raw is not None and fi < raw.shape[0])
            else rd._to_aspect(dyn[fi]).astype(np.float64)
        )
        vmin, vmax = rd.display_limits(gt_src)
        gt_u = mv.normalise_for_display(gt_src, vmin, vmax)
        per_u = mv.normalise_for_display(rd._to_aspect(per[fi]), vmin, vmax)
        dyn_u = mv.normalise_for_display(rd._to_aspect(dyn[fi]), vmin, vmax)
        phase = "observed" if fi < frontier else "FORECAST"
        burst = (
            " ELM" if (highlight_frame is not None and fi == highlight_frame) else ""
        )
        dt_ms = (ft[fi] - ft[frontier]) * 1e3
        frame = mv.panel_strip(
            [gt_u, per_u, dyn_u],
            ["ground truth", "persistence", f"dynamics ({phase})"],
            scale=3,
            counter=f"f{fi} t{dt_ms:+.1f}ms{burst}",
        )
        frames.append(frame)
    mv.write_gif(frames, out_path, duration_ms=180)
    logger.info(
        "[movie] %s: shot %d frontier@f%d horizon~%.0f ms",
        out_path.name,
        m["shot_id"],
        frontier,
        item.get("horizon_ms", float("nan")),
    )


def _render_rampup(plan, meta, images, slot, raw_fn, out_dir):
    clip = _find(plan, "rampup_clipped")
    sig = _find(plan, "rampup_signals")
    if clip is None or sig is None:
        logger.warning("[movie] no ramp-up window selected — skipping ramp-up panel")
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    win = clip["win"]
    raw = raw_fn(win)
    m = meta[clip["window_index"]]
    ft = np.asarray(m["frame_time"], dtype=float)
    n_frames = ft.shape[0]
    cols = list(np.linspace(0, n_frames - 1, 5).round().astype(int))

    # 6 rows: GT / baseline / dynamics, twice (clipped block, signals block)
    blocks = [("clipped", clip), ("signals_only", sig)]
    fig, axes = plt.subplots(
        6,
        len(cols),
        figsize=(1.7 * len(cols) + 1.6, 1.5 * 6 + 0.8),
        squeeze=False,
        constrained_layout=True,
    )
    row_names = ["truth (raw)", "baseline", "dynamics"]
    for bk, (scenario, item) in enumerate(blocks):
        wi = item["window_index"]
        base = images[slot[(wi, scenario, "baseline")]]
        dyn = images[slot[(wi, scenario, "dynamics")]]
        box = rd.clip_box(scenario)
        for ci, fi in enumerate(cols):
            gt = (
                raw[fi].astype(np.float64)
                if (raw is not None and fi < raw.shape[0])
                else rd._to_aspect(base[fi]).astype(np.float64)
            )
            vmin, vmax = rd.display_limits(gt)
            triples = [gt, rd._to_aspect(base[fi]), rd._to_aspect(dyn[fi])]
            for ri in range(3):
                ax = axes[bk * 3 + ri][ci]
                rd._imshow_cam(ax, triples[ri], vmin=vmin, vmax=vmax)
                if box is not None:
                    r0, r1, c0, c1 = box
                    ax.add_patch(
                        mpatches.Rectangle(
                            (
                                c0 / mv.GRID_W * mv.ORIGINAL_HW[1] - 0.5,
                                r0 / mv.GRID_H * mv.ORIGINAL_HW[0] - 0.5,
                            ),
                            (c1 - c0) / mv.GRID_W * mv.ORIGINAL_HW[1],
                            (r1 - r0) / mv.GRID_H * mv.ORIGINAL_HW[0],
                            fill=False,
                            edgecolor=mv.ACCENT,
                            linewidth=1.2,
                        )
                    )
                if bk == 0 and ri == 0:
                    ax.set_title(f"{(ft[fi] - ft[0]) * 1e3:+.1f} ms", fontsize=8)
                if ci == 0:
                    label = f"{scenario.split('_')[0]}\n{row_names[ri]}"
                    ax.set_ylabel(label, fontsize=7)
    fig.suptitle(
        "camera-dynamics-wm — ramp-up inference (rising Ip / brightness) | "
        f"shot {m['shot_id']} | {ft[0] * 1e3:.0f}-{ft[-1] * 1e3:.0f} ms | "
        "clipped (top 3 rows) + signals-only (bottom 3) | "
        "per-column GT 1/99-pct norm | frozen decoder",
        fontsize=10,
    )
    out_path = out_dir / "fig-cdw-recon-rampup.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("[movie] wrote %s", out_path)


# ---------------------------------------------------------------------------
# Window selection helpers (ramp-up + forecast)
# ---------------------------------------------------------------------------


def _select_rampup_window(*, n_frames: int) -> rd.DemoWindow | None:
    """Find a rising-current / rising-brightness window on a held-out shot.

    Scans early windows of the flat-top demo shots (early in the shot is where
    the ramp-up lives) and ranks by :func:`recon_movie.rampup_score` using the
    amc plasma_current conditioning + raw-frame brightness.
    """
    from imas_ambix.camdyn.conditioning import CONDITIONING_CHANNELS, load_conditioning
    from imas_ambix.camdyn.dataset import (
        FrameTokenDataset,
        FrameWindowConfig,
        discover_token_shots,
    )

    ip_idx = next(
        (i for i, c in enumerate(CONDITIONING_CHANNELS) if c.key == "plasma_current"),
        None,
    )
    best = None
    best_score = 0.0
    for sid in FLATTOP_SHOTS:
        specs = discover_token_shots(shot_ids=[sid], read_n_frames=True)
        if not specs:
            continue
        ds = FrameTokenDataset(
            specs, FrameWindowConfig(n_frames=n_frames, stride=n_frames, seed=0)
        )
        n = len(ds)
        if n == 0:
            continue
        # scan the first third of the shot (ramp-up region)
        for idx in range(min(n, max(3, n // 3))):
            win = ds[idx]
            level1_path = specs[0].level1_path
            cond = load_conditioning(
                level1_path,
                np.asarray(win.frame_time),
                sid,
                channels=CONDITIONING_CHANNELS,
            )
            ip = cond.values[:, ip_idx] if ip_idx is not None else np.zeros(n_frames)
            raw = rd.load_raw_frames(sid, int(win.start), n_frames)
            bright = (
                raw.reshape(raw.shape[0], -1).mean(axis=1)
                if raw is not None
                else np.zeros(n_frames)
            )
            score = mv.rampup_score(ip, bright)
            if score > best_score:
                best_score = score
                best = rd.DemoWindow(
                    shot_id=int(win.shot_id),
                    start=int(win.start),
                    frame_time=np.asarray(win.frame_time, dtype=np.float64),
                    dt=np.asarray(win.dt, dtype=np.float64),
                    valid=np.asarray(win.valid_frames, dtype=bool),
                    true_tokens=np.asarray(win.tokens, dtype=np.int64),
                    motion_fraction=0.0,
                )
    if best is not None:
        logger.info(
            "[movie] ramp-up window: shot %d start %d score=%.3f t=%.1f-%.1f ms",
            best.shot_id,
            best.start,
            best_score,
            best.frame_time[0] * 1e3,
            best.frame_time[-1] * 1e3,
        )
    return best


def _select_forecast_window(
    *, n_frames: int = 16, wide_factor: int = 16, horizon_ms: float = 100.0
) -> tuple[rd.DemoWindow, float] | None:
    """A wide native window decimated to ``n_frames`` spanning ``horizon_ms``.

    The rollout GIF needs the in-window future to actually reach a meaningful
    physical lead-time; a contiguous 16-frame window spans <1 ms at MAST
    cadence.  We read a wide native window of the brightest flat-top shot and
    decimate it (Δt-conditioned) so the frontier→end span covers ~``horizon_ms``.
    """
    from imas_ambix.camdyn.dataset import (
        FrameTokenDataset,
        FrameWindowConfig,
        discover_token_shots,
    )

    sid = GIF_SHOT
    specs = discover_token_shots(shot_ids=[sid], read_n_frames=True)
    if not specs:
        return None
    wide_n = n_frames * wide_factor
    ds = FrameTokenDataset(
        specs, FrameWindowConfig(n_frames=wide_n, stride=wide_n, seed=0)
    )
    if len(ds) == 0:
        return None
    # pick the brightest wide window
    starts = [ds._windows[i][1] for i in range(len(ds))]
    bright = rd._window_brightness(sid, starts, wide_n)
    pick = int(np.argmax(bright)) if bright is not None else 0
    win = ds[pick]
    ft = np.asarray(win.frame_time, dtype=np.float64)
    dt_med = float(np.median(np.diff(ft)))
    idx = mv.decimated_indices(ft.shape[0], n_frames, dt_med, horizon_ms)
    tok = np.asarray(win.tokens, dtype=np.int64)[idx]
    ft_d = ft[idx]
    dt_d = (
        np.concatenate([np.diff(ft_d), np.diff(ft_d)[-1:]])
        if ft_d.size > 1
        else (np.zeros_like(ft_d))
    )
    valid = np.asarray(win.valid_frames, dtype=bool)[idx]
    span_ms = float((ft_d[-1] - ft_d[len(ft_d) // 2]) * 1e3)
    dwin = rd.DemoWindow(
        shot_id=int(win.shot_id),
        start=int(win.start),
        frame_time=ft_d,
        dt=dt_d.astype(np.float64),
        valid=valid,
        true_tokens=tok,
        motion_fraction=0.0,
    )
    logger.info(
        "[movie] forecast window: shot %d decimated to %d frames, "
        "frontier→end span ~%.1f ms",
        sid,
        len(idx),
        span_ms,
    )
    return dwin, span_ms


# ---------------------------------------------------------------------------
# ELM window finder (transient Dα spike — edge-localized-mode signature)
# ---------------------------------------------------------------------------


#: Held-out shots scanned for ELMy windows.  The flat-top demo shots first
#: (already known bright + structured), then a bounded sample of the held-out
#: split so a clearly ELMy shot can be found without scanning all 1050.
def _elm_candidate_shots(n_sample: int = 60) -> list[int]:
    import json as _json

    shots = list(FLATTOP_SHOTS)
    try:
        split = _json.loads(
            (
                Path(__file__).resolve().parent / "artifacts" / "camdyn_split_v0.json"
            ).read_text()
        )
        ho = [int(s) for s in split.get("held_out", [])]
        # deterministic stride sample across the held-out range
        if ho:
            step = max(1, len(ho) // n_sample)
            shots += ho[::step][:n_sample]
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[movie] ELM candidate-shot list fell back to demo shots: %s", exc
        )
    # de-dup, keep order
    seen: set[int] = set()
    out: list[int] = []
    for s in shots:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


#: Verified ELM windows handed in by the orchestrator scan (shot, frame_start).
#: When set (via --elm-shot / --elm-start or env), selection targets these
#: exact camera windows instead of re-scanning.  Empty → auto-scan the
#: candidate shots by raw-camera transient brightness.
ELM_CANDIDATES: tuple[tuple[int, int], ...] = ()


def _window_at(sid: int, start: int, read_n: int) -> rd.DemoWindow | None:
    """Materialise the native token window ``[start, start+read_n)`` for a shot."""
    from imas_ambix.camdyn.dataset import (
        FrameTokenDataset,
        FrameWindowConfig,
        discover_token_shots,
    )

    specs = discover_token_shots(shot_ids=[sid], read_n_frames=True)
    if not specs:
        return None
    ds = FrameTokenDataset(
        specs,
        FrameWindowConfig(n_frames=read_n, stride=read_n, seed=0, drop_short=False),
    )
    win = ds._materialise(0, int(start))
    return rd.DemoWindow(
        shot_id=int(win.shot_id),
        start=int(win.start),
        frame_time=np.asarray(win.frame_time, dtype=np.float64),
        dt=np.asarray(win.dt, dtype=np.float64),
        valid=np.asarray(win.valid_frames, dtype=bool),
        true_tokens=np.asarray(win.tokens, dtype=np.int64),
        motion_fraction=0.0,
    )


def _select_elm_window(
    *, n_frames: int, decimate_ms: float | None = None, wide_factor: int = 16
):
    """Find a window with a sharp transient CAMERA brightness burst (an ELM).

    ELMs are detected DIRECTLY in the raw rbb camera: the per-frame
    edge/divertor brightness is high-passed and ranked by
    :func:`recon_movie.camera_elm_score` (no Dα / cross-diagnostic alignment —
    the burst frame maps 1:1 to the token-window frame).  If
    :data:`ELM_CANDIDATES` is populated (orchestrator-verified ``(shot,
    start)`` windows) those exact windows are scored instead of auto-scanning.
    Returns ``(window, peak_frame, burst_time_ms)`` or None.

    When ``decimate_ms`` is set, a WIDE native window is decimated so the burst
    lands in the post-frontier (forecast) half — the ELM-forecast variant: the
    model observes pre-burst frames and must predict the brightening.
    """
    frontier = n_frames // 2
    read_n = (n_frames * wide_factor) if decimate_ms else n_frames

    # candidate (shot, start, native-window) tuples
    if ELM_CANDIDATES:
        cand = [(sid, st) for sid, st in ELM_CANDIDATES]
    else:
        # auto-scan: enumerate read_n-windows of the candidate shots
        from imas_ambix.camdyn.dataset import (
            FrameTokenDataset,
            FrameWindowConfig,
            discover_token_shots,
        )

        cand = []
        for sid in _elm_candidate_shots():
            specs = discover_token_shots(shot_ids=[sid], read_n_frames=True)
            if not specs or specs[0].level1_path is None:
                continue
            ds = FrameTokenDataset(
                specs, FrameWindowConfig(n_frames=read_n, stride=read_n, seed=0)
            )
            for wi in range(min(len(ds), 12)):
                cand.append((sid, int(ds._windows[wi][1])))

    best = None
    best_score = 0.0
    for sid, start in cand:
        win = _window_at(sid, start, read_n)
        if win is None:
            continue
        raw = rd.load_raw_frames(sid, start, read_n)
        if raw is None or raw.shape[0] < 3:
            continue
        ft = np.asarray(win.frame_time, dtype=np.float64)
        tok = np.asarray(win.true_tokens, dtype=np.int64)
        valid = np.asarray(win.valid, dtype=bool)
        rawf = raw.astype(np.float64)
        if decimate_ms:
            dt_med = float(np.median(np.diff(ft)))
            idx = mv.decimated_indices(ft.shape[0], n_frames, dt_med, decimate_ms)
            idx = idx[idx < rawf.shape[0]]
            tok, ft, valid, rawf = tok[idx], ft[idx], valid[idx], rawf[idx]
        score, peak = mv.camera_elm_score(rawf)
        # forecast variant: require the burst to land AFTER the frontier
        if decimate_ms and peak < frontier:
            score *= 0.1
        if score > best_score:
            dt = (
                np.concatenate([np.diff(ft), np.diff(ft)[-1:]])
                if ft.size > 1
                else np.zeros_like(ft)
            )
            best_score = score
            best = (
                rd.DemoWindow(
                    shot_id=int(sid),
                    start=int(start),
                    frame_time=ft,
                    dt=dt.astype(np.float64),
                    valid=valid,
                    true_tokens=tok,
                    motion_fraction=0.0,
                ),
                int(peak),
                float(ft[peak] * 1e3),
            )
    if best is not None:
        w, peak, t_ms = best
        logger.info(
            "[movie] ELM window%s: shot %d start %d camera-burst@f%d t=%.1f ms "
            "score=%.2f window %.1f-%.1f ms",
            " (forecast)" if decimate_ms else "",
            w.shot_id,
            w.start,
            peak,
            t_ms,
            best_score,
            w.frame_time[0] * 1e3,
            w.frame_time[-1] * 1e3,
        )
    else:
        logger.warning("[movie] no ELM window found in the candidate shots")
    return best


# ---------------------------------------------------------------------------
# Forecast sweep (token-space top-1 + NLL) over a dense horizon grid
# ---------------------------------------------------------------------------


def _forecast_sweep(
    torch, dev, dyn_model, dyn_stats, base_model, base_stats, artifact_path, out_dir
):
    """Dense horizon sweep: dynamics vs persistence vs baseline, paired.

    Reuses the locked :func:`horizon_eval.score_window_horizons` /
    :func:`horizon_eval.decimate_to_n` machinery but over a finer horizon grid
    so the persistence crossover ("how far forward") is visible.  Top-1 and
    NLL are token-space (no decode needed — exact and cheap).
    """
    from imas_ambix.camdyn import horizon_eval as he
    from imas_ambix.camdyn.dataset import FrameWindowConfig
    from imas_ambix.camdyn.metrics import bootstrap_ci
    from imas_ambix.camdyn.train import TrainConfig, Trainer, _specs_for_shots

    horizons = mv.SWEEP_HORIZONS_MS
    base_full_cfg = torch.load(
        str(rd.BASELINE_CKPT), map_location="cpu", weights_only=False
    )["config"]
    tcfg = TrainConfig.from_dict(base_full_cfg)
    tcfg.device = str(dev)
    tcfg.num_workers = 0
    tr = Trainer(tcfg)
    tr._cond_stats = base_stats
    split = tr._load_split()
    ho_specs = _specs_for_shots(split.held_out, max_shots=tcfg.max_heldout_shots)
    nf = tcfg.n_frames
    frontier = nf // 2
    max_h = float(max(horizons))

    wide_factor = 16
    wide_n = nf * wide_factor
    wide_cfg = FrameWindowConfig(n_frames=wide_n, stride=wide_n, seed=tcfg.seed)
    wide_batches = tr._materialize_eval(
        ho_specs, wide_cfg, max_windows=tcfg.eval_windows, seed=999
    )
    logger.info("[sweep] %d wide held-out batches", len(wide_batches))

    # per-horizon paired arrays
    agg = {
        h: {
            "dyn_top1": [],
            "dyn_nll": [],
            "persist_top1": [],
            "base_top1": [],
            "base_nll": [],
            "n_windows": 0,
        }
        for h in horizons
    }
    for arr in wide_batches:
        barr = he.decimate_to_n(arr, nf, max_h)
        dyn_bl = he._forward_batch(dyn_model, barr, torch, dev, frontier, dyn_stats)
        base_bl = he._forward_batch(base_model, barr, torch, dev, frontier, base_stats)
        for b in range(barr["tokens"].shape[0]):
            dyn_rec = he.score_window_horizons(
                dyn_bl[b],
                barr["tokens"][b],
                barr["frame_time"][b],
                barr["valid"][b],
                frontier,
                horizons_ms=horizons,
            )
            base_rec = he.score_window_horizons(
                base_bl[b],
                barr["tokens"][b],
                barr["frame_time"][b],
                barr["valid"][b],
                frontier,
                horizons_ms=horizons,
            )
            for h in horizons:
                d = dyn_rec[h]
                if not d.get("valid"):
                    continue
                bcell = base_rec[h]
                agg[h]["dyn_top1"].append(d["dyn_top1"])
                agg[h]["dyn_nll"].append(d["dyn_nll"])
                agg[h]["persist_top1"].append(d["persist_top1"])
                agg[h]["base_top1"].append(bcell["dyn_top1"])
                agg[h]["base_nll"].append(bcell["dyn_nll"])
                agg[h]["n_windows"] += 1

    table = {}
    for h in horizons:
        a = agg[h]
        if a["n_windows"] == 0:
            table[h] = {"valid_windows": 0}
            continue
        dyn_t = np.concatenate(a["dyn_top1"])
        per_t = np.concatenate(a["persist_top1"])
        base_t = np.concatenate(a["base_top1"])
        dvp = bootstrap_ci(dyn_t - per_t)
        dvb = bootstrap_ci(dyn_t - base_t)
        table[h] = {
            "valid_windows": int(a["n_windows"]),
            "n_cells": int(dyn_t.size),
            "dynamics_top1": float(dyn_t.mean()),
            "persistence_top1": float(per_t.mean()),
            "baseline_top1": float(base_t.mean()),
            "dynamics_nll": float(np.concatenate(a["dyn_nll"]).mean()),
            "baseline_nll": float(np.concatenate(a["base_nll"]).mean()),
            "dynamics_vs_persistence_top1": dvp,
            "dynamics_vs_baseline_top1": dvb,
            "beats_persistence": bool(dvp["favours_dynamics"]),
            "beats_baseline": bool(dvb["favours_dynamics"]),
        }

    # crossover: the largest horizon where dynamics still beats persistence
    crossover = None
    for h in sorted(horizons):
        c = table[h]
        if c.get("valid_windows") and c.get("beats_persistence"):
            crossover = h
    out = {
        "task": "forward-forecast horizon sweep: dynamics vs persistence vs baseline",
        "metric": "token-space top-1 (bit-head MAP) + bitwise NLL, paired bootstrap",
        "regime": "matched (wide native window decimated to n_frames, Dt-conditioned)",
        "frontier_frame": int(frontier),
        "n_frames": int(nf),
        "wide_window_frames": int(wide_n),
        "horizons_ms": list(horizons),
        "crossover_ms": crossover,
        "crossover_note": (
            "largest horizon where dynamics top-1 still significantly beats "
            "persistence (paired bootstrap lower bound > 0); beyond it the "
            "dynamics forecast no longer improves on copy-the-last-frame"
        ),
        "table": {str(h): table[h] for h in horizons},
        "baseline_ckpt": str(rd.BASELINE_CKPT),
        "dynamics_ckpt": str(rd.DYNAMICS_CKPT),
    }
    artifact_path = Path(artifact_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    logger.info("[sweep] wrote %s (crossover=%s ms)", artifact_path, crossover)
    _plot_forecast_sweep(out, out_dir / "fig-cdw-forecast-sweep.png")
    return out


def _plot_forecast_sweep(sweep: dict, out_path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    horizons = sweep["horizons_ms"]
    hs, dyn, per, base, ndyn, nbase = [], [], [], [], [], []
    for h in horizons:
        c = sweep["table"][str(h)]
        if not c.get("valid_windows"):
            continue
        hs.append(h)
        dyn.append(c["dynamics_top1"])
        per.append(c["persistence_top1"])
        base.append(c["baseline_top1"])
        ndyn.append(c["dynamics_nll"])
        nbase.append(c["baseline_nll"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    ax1.plot(hs, dyn, "o-", color="#2ca02c", label="dynamics")
    ax1.plot(hs, per, "s--", color="#7f7f7f", label="persistence")
    ax1.plot(hs, base, "^:", color="#1f77b4", label="per-frame baseline")
    cx = sweep.get("crossover_ms")
    if cx is not None:
        ax1.axvline(cx, color="#d62728", lw=1.2, ls="-", label=f"crossover {cx:.0f} ms")
    ax1.set_xscale("log")
    ax1.set_xlabel("forecast horizon (ms, log)")
    ax1.set_ylabel("top-1 token accuracy")
    ax1.set_title("how far forward — top-1 vs horizon")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.plot(hs, ndyn, "o-", color="#2ca02c", label="dynamics")
    ax2.plot(hs, nbase, "^:", color="#1f77b4", label="per-frame baseline")
    ax2.set_xscale("log")
    ax2.set_xlabel("forecast horizon (ms, log)")
    ax2.set_ylabel("bitwise NLL (nats, lower better)")
    ax2.set_title("forecast NLL vs horizon")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    ntxt = ", ".join(
        f"{h:.0f}ms:n={sweep['table'][str(h)].get('valid_windows', 0)}"
        for h in horizons
    )
    fig.suptitle(
        "camera-dynamics-wm — forward-forecast sweep (FRONTIER mask, matched "
        f"cadence) | valid windows {ntxt}",
        fontsize=10,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("[movie] wrote %s", out_path)


def main(argv=None) -> int:
    global ELM_CANDIDATES
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="docs/figures/camera-dynamics-wm/")
    p.add_argument(
        "--artifact", default="imas_ambix/camdyn/artifacts/forecast_sweep.json"
    )
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--elm-window",
        action="append",
        default=None,
        metavar="SHOT:START",
        help=(
            "verified ELM camera window as SHOT:FRAME_START (repeatable); "
            "targets that exact window instead of auto-scanning"
        ),
    )
    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    if args.elm_window:
        ELM_CANDIDATES = tuple(
            (int(s.split(":")[0]), int(s.split(":")[1])) for s in args.elm_window
        )
        logger.info("[movie] targeting verified ELM windows: %s", ELM_CANDIDATES)
    run(Path(args.out), Path(args.artifact), device=args.device)
    return 0
