"""Figures and the pass-time projection for the 8xB300 serving study.

Reads census-1280470.json (written by serve_census.py from the four-card
DeepSeek-V4.1-Flash serve log and the router receipts) and writes:

* decode-by-width.svg -- aggregate and per-request decode rate against the
  engine's running-request count, measured on 4xH200 since the gate went to
  12, beside the projected 8xB300 range;
* hourly-load.svg -- hourly mean running requests and output rate across the
  serve's life, with the gate change marked;
* projection.json -- the fitted pass-time terms and the projected figures the
  study quotes.

Pass time is decode tokens per pass divided by rate: a pass emits the accept
length for every running request, so t(w) = w * accept / aggregate(w). It is
fitted as a fixed term plus a per-request term, t(w) = a + b*w. The 8xB300
projection divides the fixed term by k_a (bandwidth and collective latency on
twice the cards over NVSwitch) and the per-request term by k_b (attention and
interleaved prefill, which move with compute and bandwidth together).
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
H200 = "#3d6fa5"  # measured four-card H200 serve
B300 = "#c06a35"  # projected eight-card B300 serve
GREY = "#7a7a7a"

#: Bounds on the B300 speed-up of each pass-time term. Low: nameplate HBM
#: ratio 3.33x at today's 20% efficiency diluted by an unchanged latency
#: floor (k_a 1.7) and a per-request term that gains only the bandwidth ratio
#: (k_b 3.3). High: the block-scale tax removed (k_a 2.5) and the per-request
#: term gaining the dense-compute ratio for prefill (k_b 5.0).
K_LOW = (1.7, 3.3)
K_HIGH = (2.5, 5.0)
WIDTHS = np.arange(1, 97)


def load():
    census = json.loads((HERE / "census-1280470.json").read_text())
    table = census["width"]["post_gate12"]
    rows = sorted((int(w), v) for w, v in table.items() if int(w) <= 12)
    return census, rows


def fit(rows):
    w = np.array([r[0] for r in rows], float)
    agg = np.array([r[1]["agg_median"] for r in rows])
    acc = np.array([r[1]["accept_len"] for r in rows])
    t_ms = 1000.0 * w * acc / agg
    b, a = np.polyfit(w, t_ms, 1)
    resid = t_ms - (a + b * w)
    return a, b, float(np.mean(acc)), float(np.sqrt(np.mean(resid**2))), t_ms


def project(a, b, acc, k):
    t = a / k[0] + b * WIDTHS / k[1]
    per_req = 1000.0 * acc / t
    return per_req, per_req * WIDTHS


def width_at(a, b, k, acc, per_request):
    """Concurrent requests at which the projected per-request rate falls to a floor."""
    t_ms = 1000.0 * acc / per_request
    return (t_ms - a / k[0]) / (b / k[1])


def decode_figure(rows, proj):
    w = np.array([r[0] for r in rows])
    med = np.array([r[1]["agg_median"] for r in rows])
    p25 = np.array([r[1]["agg_p25"] for r in rows])
    p75 = np.array([r[1]["agg_p75"] for r in rows])
    per = np.array([r[1]["per_req_median"] for r in rows])
    (lo_r, lo_a), (hi_r, hi_a) = proj

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.6))
    ax1.fill_between(WIDTHS, lo_a, hi_a, color=B300, alpha=0.22, linewidth=0)
    ax1.plot(WIDTHS, lo_a, color=B300, lw=2.6)
    ax1.plot(WIDTHS, hi_a, color=B300, lw=2.6)
    ax1.errorbar(
        w,
        med,
        yerr=[med - p25, p75 - med],
        color=H200,
        lw=3.0,
        elinewidth=1.2,
        capsize=0,
        marker="o",
        ms=6,
    )
    ax1.text(14, med[-1], "4×H200 measured", color=H200, va="center")
    ax1.text(96, hi_a[-1], "8×B300 projected", color=B300, ha="right", va="bottom")
    ax1.set_ylabel("aggregate decode [tok/s]")

    ax2.fill_between(WIDTHS, lo_r, hi_r, color=B300, alpha=0.22, linewidth=0)
    ax2.plot(WIDTHS, lo_r, color=B300, lw=2.6)
    ax2.plot(WIDTHS, hi_r, color=B300, lw=2.6)
    ax2.plot(w, per, color=H200, lw=3.0, marker="o", ms=6)
    ax2.axhline(45, color=GREY, lw=1.2, ls=":")
    ax2.text(1.05, 33, "45 tok/s at today's gate", color=GREY, va="top")
    ax2.text(1.05, per[0] + 8, "4×H200", color=H200, va="bottom")
    ax2.text(20, hi_r[19] + 6, "8×B300", color=B300)
    ax2.set_ylabel("per-request decode [tok/s]")

    for ax in (ax1, ax2):
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8, 12, 32, 64, 96])
        ax.set_xticklabels(["1", "2", "4", "8", "12", "32", "64", "96"])
        ax.set_xlabel("concurrent requests")
        ax.set_ylim(bottom=0)
        ax.axvline(12, color=GREY, lw=1.2, ls=":")
    fig.tight_layout()
    fig.savefig(HERE / "decode-by-width.svg")
    plt.close(fig)


def hourly_figure(census):
    import datetime as dt

    rows = [h for h in census["hourly"] if h["observed_s"] > 0]
    t = [dt.datetime.fromtimestamp(h["t"], dt.UTC) for h in rows]
    hours = {h["t"]: h for h in rows}
    t0, t1 = min(hours), max(hours)
    grid = range(t0, t1 + 3600, 3600)
    tt = [dt.datetime.fromtimestamp(x, dt.UTC) for x in grid]
    running = [hours[x]["running"] if x in hours else 0.0 for x in grid]
    gen = [hours[x]["gen_tok_s"] if x in hours else 0.0 for x in grid]
    gate = dt.datetime(2026, 10, 8, 4, 38, tzinfo=dt.UTC)
    del t

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6.4), sharex=True)
    ax1.plot(tt, running, color=H200, lw=2.6, drawstyle="steps-post")
    ax1.set_ylabel("running requests\n(hourly mean)")
    ax2.plot(tt, gen, color=H200, lw=2.6, drawstyle="steps-post")
    ax2.set_ylabel("output [tok/s]\n(hourly mean)")
    for ax in (ax1, ax2):
        ax.axvline(gate, color=GREY, lw=1.2, ls=":")
        ax.set_ylim(bottom=0)
    ax1.text(gate, max(running) * 0.95, " gate 16 → 12", color=GREY, va="top")
    import matplotlib.dates as mdates

    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax2.set_xlabel("date [UTC]")
    fig.tight_layout()
    fig.savefig(HERE / "hourly-load.svg")
    plt.close(fig)


def main():
    plt.style.use("data-ink")
    census, rows = load()
    a, b, acc, rms, t_ms = fit(rows)
    proj = [project(a, b, acc, k) for k in (K_LOW, K_HIGH)]
    decode_figure(rows, proj)
    hourly_figure(census)
    out = {
        "fit": {
            "floor_ms": a,
            "per_request_ms": b,
            "accept_len": acc,
            "residual_rms_ms": rms,
            "pass_ms_by_width": dict(zip(range(1, 13), t_ms.tolist(), strict=True)),
        },
        "k_low": K_LOW,
        "k_high": K_HIGH,
        "b300": {},
    }
    for name, k in (("low", K_LOW), ("high", K_HIGH)):
        r1, a1 = project(a, b, acc, k)
        entry = {
            "single_stream": float(r1[0]),
            "per_request_at_12": float(r1[11]),
            "aggregate_at_12": float(a1[11]),
        }
        for floor in (45, 30):
            wf = width_at(a, b, k, acc, floor)
            entry[f"width_at_{floor}"] = float(wf)
            entry[f"aggregate_at_{floor}"] = float(wf * floor)
        out["b300"][name] = entry
    (HERE / "projection.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
