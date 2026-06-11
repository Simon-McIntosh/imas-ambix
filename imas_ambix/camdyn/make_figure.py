"""Render the D0 end-to-end batch figure on a real shot.

Proves the pipeline runs end-to-end on real corpus data: a sampled
window's full token grid, the clip-visibility mask, and the actuator
conditioning trace aligned to the frame times.

Run::

    uv run python -m imas_ambix.camdyn.make_figure

Writes ``docs/figures/camera-dynamics-wm/fig-cdw-d0-batch.png``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from imas_ambix.camdyn.conditioning import load_conditioning  # noqa: E402
from imas_ambix.camdyn.dataset import (  # noqa: E402
    FrameTokenDataset,
    FrameWindowConfig,
    discover_token_shots,
)
from imas_ambix.camdyn.masking import (  # noqa: E402
    ClipMaskConfig,
    MaskMode,
    named_geometry_mask,
    sample_clip_mask,
)
from imas_ambix.camdyn.metrics import motion_weighted_subset  # noqa: E402

FIG_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "docs"
    / "figures"
    / "camera-dynamics-wm"
)
REF_SHOT = 24065
N_FRAMES = 12


def main(shot: int = REF_SHOT, out: Path | None = None) -> Path:
    specs = discover_token_shots(shot_ids=[shot], read_n_frames=True)
    if not specs:
        raise RuntimeError(f"shot {shot} has no rbb tokens on disk")

    # Sample a longer window so the 6 displayed frames span real dynamics,
    # placed early in the record (plasma-on / ramp, where the camera is busy).
    cfg = FrameWindowConfig(n_frames=60, stride=512, seed=0)
    ds = FrameTokenDataset(specs, cfg)
    win = ds[2]  # an early window (plasma forming)
    # display 6 frames evenly spread across the window so motion is visible
    show_idx = np.linspace(0, win.tokens.shape[0] - 1, 6).round().astype(int)

    nfr = int(win.tokens.shape[0])
    # masks: a random clip + the frozen named geometries
    rng = np.random.default_rng(0)
    rand_mask, rand_meta = sample_clip_mask(
        nfr, ClipMaskConfig(), rng, mode=MaskMode.RANDOM
    )
    frontier_mask = named_geometry_mask("frontier_half", nfr)
    pan_mask = named_geometry_mask("standard_pan", nfr)

    # conditioning held to the frame times
    cond = load_conditioning(specs[0].level1_path, win.frame_time, shot)
    moving = motion_weighted_subset(win.tokens, win.frame_time)

    # ----- layout -----
    fig = plt.figure(figsize=(15, 10), constrained_layout=True)
    gs = fig.add_gridspec(4, 6, height_ratios=[1.0, 1.0, 1.0, 1.2])

    # Row 0: full token grid for 6 frames spread across the window
    for k, fi in enumerate(show_idx):
        ax = fig.add_subplot(gs[0, k])
        ax.imshow(win.tokens[fi], cmap="turbo", interpolation="nearest")
        ax.set_title(f"frame {fi}  t={win.frame_time[fi] * 1e3:.1f} ms", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        if k == 0:
            ax.set_ylabel("FULL tokens\n(16×16 grid)", fontsize=9)

    # Row 1: random clip visibility (token grid masked) for the same 6 frames
    for k, fi in enumerate(show_idx):
        ax = fig.add_subplot(gs[1, k])
        masked = np.ma.masked_where(~rand_mask[fi], win.tokens[fi])
        ax.imshow(np.zeros_like(win.tokens[fi]), cmap="gray", vmin=0, vmax=1)
        ax.imshow(masked, cmap="turbo", interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        if k == 0:
            ax.set_ylabel(
                f"RANDOM clip\n(visible {rand_mask.mean() * 100:.0f}%)", fontsize=9
            )

    # Row 2: the named-geometry masks + motion subset (binary masks, frame 0/mid)
    panels = [
        ("§2 fixed", named_geometry_mask("fixed_section2", nfr)[0]),
        ("divertor", named_geometry_mask("divertor_only", nfr)[0]),
        ("centre strip", named_geometry_mask("centre_column_strip", nfr)[0]),
        (f"pan (f={nfr // 2})", pan_mask[nfr // 2]),
        (f"frontier (f={nfr // 2})", frontier_mask[nfr // 2]),
        (
            f"moving tokens\n(mean over window: {moving.mean() * 100:.0f}%)",
            moving.mean(axis=0),
        ),
    ]
    for k, (title, m) in enumerate(panels):
        ax = fig.add_subplot(gs[2, k])
        ax.imshow(
            m.astype(float), cmap="Greens", vmin=0, vmax=1, interpolation="nearest"
        )
        ax.set_title(title, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        if k == 0:
            ax.set_ylabel("masks\n(green=visible)", fontsize=9)

    # Row 3: actuator conditioning traces aligned to the frame times
    ax_ip = fig.add_subplot(gs[3, 0:2])
    keys = cond.channel_keys
    t_ms = cond.frame_time * 1e3

    def _trace(ax, key, color, label, scale=1.0, unit=""):
        if key not in keys:
            return
        j = keys.index(key)
        present = cond.missing[:, j] < 0.5
        ax.plot(t_ms, cond.values[:, j] * scale, color=color, label=f"{label} {unit}")
        if (~present).any():
            ax.scatter(
                t_ms[~present],
                cond.values[~present, j] * scale,
                c="red",
                s=8,
                zorder=5,
                label="missing",
            )

    _trace(ax_ip, "plasma_current", "C0", "Ip", scale=1e-3, unit="(kA)")
    ax_ip.set_title("plasma current (held to frame times)", fontsize=9)
    ax_ip.set_xlabel("frame time (ms)")
    ax_ip.legend(fontsize=7)

    ax_nbi = fig.add_subplot(gs[3, 2:4])
    _trace(ax_nbi, "nbi_tot_sum_power", "C1", "NBI total", unit="(MW)")
    ax_nbi.set_title("NBI power", fontsize=9)
    ax_nbi.set_xlabel("frame time (ms)")
    ax_nbi.legend(fontsize=7)

    ax_gas = fig.add_subplot(gs[3, 4:6])
    _trace(ax_gas, "gas_inboard_total", "C2", "inboard puff", unit="(e⁻/s)")
    _trace(ax_gas, "ne_line_integrated", "C3", "∫ne dl", unit="(m⁻²)")
    ax_gas.set_title("inboard gas puff + line-integrated density", fontsize=9)
    ax_gas.set_xlabel("frame time (ms)")
    ax_gas.legend(fontsize=7)

    fig.suptitle(
        f"camera-dynamics-wm D0 — end-to-end batch on shot {shot} "
        f"(frame-grid tokens → clip mask → frame-aligned conditioning)",
        fontsize=12,
    )

    out = out or (FIG_DIR / "fig-cdw-d0-batch.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"[fig] wrote {out}")
    return out


if __name__ == "__main__":
    main()
