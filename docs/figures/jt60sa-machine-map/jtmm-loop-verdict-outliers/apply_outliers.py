#!/usr/bin/env python3
"""Append each corpus-validated differential loop rule's per-shot outliers.

Reads maps/magnetics.json, and for every rule whose loop fails the per-shot
differential fit on at least one vacuum shot appends a clause naming that shot
and the fit's slope and r to the rule's ``evidence`` string.  Only evidence text
changes; every other field is asserted unchanged after the write.

The failing (loop, shot, slope, r) tuples are transcribed from the per-shot
table in docs/evidence/fragments/jt60sa-machine-map/jtmm-loop-reference-rca.html
(the ``L-7`` columns): a shot fails when |slope + 1| > 0.1 or |r| <= 0.95.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# loop -> ((shot, slope, r), ...) for shots the per-shot fit fails.
OUTLIERS: dict[int, tuple[tuple[int, float, float], ...]] = {
    12: ((100595, -0.341, -0.5920),),
    13: ((100595, -0.513, -0.5578),),
    14: ((100595, -0.416, -0.4192),),
    15: ((100579, -1.156, -0.9940), (100595, -0.890, -0.9170)),
    16: ((100579, -1.185, -0.9924), (100595, -0.929, -0.9497)),
    17: ((100579, -1.127, -0.9875),),
    27: ((100595, -0.856, -0.9017),),
}


def _clause(outliers: tuple[tuple[int, float, float], ...]) -> str:
    parts = [
        f"E{shot} at slope {slope:+.3f} and r {r:+.4f}"
        for shot, slope, r in outliers
    ]
    joined = parts[0] if len(parts) == 1 else " and ".join(parts)
    return (
        f" Against reference loop 7 the per-shot fit leaves {joined} outside the "
        "threshold; the median rule above still fixes the sign."
    )


def main() -> int:
    map_path = Path(sys.argv[1])
    text = map_path.read_text()
    before = json.loads(text)

    loops = {
        int(s["source_array"].removeprefix("magFlxLp")): s
        for s in before["signals"]
        if str(s.get("source_array", "")).startswith("magFlxLp")
        and s["semantic_id"].startswith("magnetics_flux_loop_")
    }

    for loop, outliers in OUTLIERS.items():
        signal = loops[loop]
        assert signal["validation_state"] == "corpus-validated", loop
        old = signal["evidence"]
        # A differential loop's evidence names the [7, L] pair; guard the target.
        token = f'"evidence": "{old}"'
        assert text.count(token) == 1, loop
        text = text.replace(token, f'"evidence": "{old}{_clause(outliers)}"')

    map_path.write_text(text)

    after = json.loads(map_path.read_text())
    assert len(after["signals"]) == len(before["signals"])
    for old_sig, new_sig in zip(before["signals"], after["signals"], strict=True):
        for key in old_sig:
            if key == "evidence":
                continue
            assert old_sig[key] == new_sig[key], (old_sig.get("source_array"), key)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())