"""Figure for the telemetry store: what each tier keeps of each leaf kind."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from imas_ambix.agent import telemetry_store as store  # noqa: E402

CADENCE = 5
START = datetime(2026, 9, 1, 0, 0, tzinfo=UTC)
N = 25 * 60  # 25 minutes of 5 s samples
raw = [
    {
        "timestamp": (START + timedelta(seconds=i * CADENCE)).isoformat(),
        "prefix_cache_queries_total": 1000 + 7 * i,
        "kv_cache_usage_perc": 40.0 + 25.0 * math.sin(i / 90.0),
    }
    for i in range(N)
]

minute = store.compact_rows(raw, tier=store.TIER_MINUTE)
hour = store.compact_rows(minute, tier=store.TIER_HOUR)

raw_t = [datetime.fromisoformat(r["timestamp"]) for r in raw]
raw_counter = [r["prefix_cache_queries_total"] for r in raw]
raw_kv = [r["kv_cache_usage_perc"] for r in raw]

min_t = [datetime.fromisoformat(r["timestamp"]) for r in minute]
min_counter = [r["prefix_cache_queries_total"] for r in minute]
# What a mean would have kept instead of the endpoint.
min_mean = []
for row in minute:
    bucket = int(datetime.fromisoformat(row["window_start"]).timestamp())
    window = [
        r["prefix_cache_queries_total"]
        for r in raw
        if math.floor(datetime.fromisoformat(r["timestamp"]).timestamp() / 60) * 60
        == bucket
    ]
    min_mean.append(sum(window) / len(window))

min_kv = [r["kv_cache_usage_perc"] for r in minute]
hour_kv = hour[0]["kv_cache_usage_perc"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))

ax1.plot(raw_t, raw_counter, color="#bbbbbb", lw=1.6, label="raw 5 s samples")
ax1.plot(
    min_t,
    min_counter,
    "o-",
    color="#0b6e4f",
    ms=4,
    lw=1.4,
    label="minute tier: counter endpoint (kept)",
)
ax1.plot(
    min_t,
    min_mean,
    "x--",
    color="#c0392b",
    ms=5,
    label="what a mean would keep (forbidden)",
)
ax1.set_title("Cumulative counter: the endpoint is kept, never averaged")
ax1.set_ylabel("prefix_cache_queries_total")
ax1.legend(fontsize=8, loc="upper left")
ax1.tick_params(axis="x", labelsize=8)

step = [0, (N - 1) * CADENCE]
# Draw the gauge at both compactions side by side on a shared time base.
ax2.plot(raw_t, raw_kv, color="#bbbbbb", lw=1.6, label="raw 5 s samples")
ax2.step(
    min_t,
    min_kv,
    where="post",
    color="#0b6e4f",
    lw=1.3,
    label=f"minute tier: time-weighted mean ({minute[0]['obs']['samples']} samples/row)",
)
ax2.hlines(
    hour_kv,
    raw_t[0],
    raw_t[-1],
    color="#1f6feb",
    lw=1.6,
    linestyle="-.",
    label=f"hour tier: {hour[0]['obs']['samples']} samples/row",
)
ax2.set_title("Gauge: a time-weighted mean that carries its weight")
ax2.set_ylabel("kv_cache_usage_perc")
ax2.legend(fontsize=8, loc="upper right")
ax2.tick_params(axis="x", labelsize=8)
del step

fig.suptitle(
    "One raw record, two compacted resolutions — endpoints for counters, "
    "weighted means for gauges",
    fontsize=10,
)
fig.tight_layout(rect=(0, 0, 1, 0.95))
out = Path("docs/figures/telemetry-store/compaction.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=140)
print(f"wrote {out} ({out.stat().st_size} bytes)")