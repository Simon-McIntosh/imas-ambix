"""Negative control for the telemetry store's counter-endpoint rule.

MUTATION: compact cumulative counters by their time-weighted mean instead of
their endpoint (patch _is_counter so every leaf classifies as a gauge).

With the mutation applied the load-bearing assertion -- a counter difference
across a tier boundary equals the raw difference -- must fail, which is what
proves the assertion can see the defect the store exists to prevent.
"""

from __future__ import annotations

import math
import sys
from datetime import UTC, datetime, timedelta

sys.path.insert(0, ".")

from imas_ambix.agent import telemetry_store as store

MUTATION = (
    "compact cumulative counters by their time-weighted mean instead of their "
    "endpoint (make _is_counter report every leaf as a gauge)"
)
print(MUTATION)

store._is_counter = lambda path: False  # noqa: SLF001 - the declared mutation

CADENCE = 20
START = datetime(2026, 9, 1, tzinfo=UTC)
rows = [
    {
        "timestamp": (START + timedelta(seconds=i * CADENCE)).isoformat(),
        "prefix_cache_queries_total": 1000 + 7 * i,
    }
    for i in range(3, 3 * 24 * 60 * 60 // CADENCE)
]

minute = store.compact_rows(rows, tier=store.TIER_MINUTE)
endpoints: dict[int, int] = {}
for row in rows:
    stamp = datetime.fromisoformat(row["timestamp"])
    endpoints[math.floor(stamp.timestamp() / 60) * 60] = row[
        "prefix_cache_queries_total"
    ]

failures = 0
checked = 0
for row in minute:
    bucket = int(datetime.fromisoformat(row["window_start"]).timestamp())
    checked += 1
    if row["prefix_cache_queries_total"] != endpoints[bucket]:
        if failures == 0:
            print(
                "RED: compacted counter is not the raw endpoint "
                f"at {row['window_start']}: compacted="
                f"{row['prefix_cache_queries_total']} endpoint={endpoints[bucket]}"
            )
        failures += 1

print(f"checked={checked} endpoint_mismatches={failures}")
sys.exit(1 if failures else 0)