"""Retarget the JT-60SA differential flux-loop rules onto their type-6 entries.

The 53-loop description store holds 27 type-1 flux-loop entries (``FL1`` ..
``FL27``) followed by 26 type-6 differential entries, each naming a pair of
loops through ``indices_differential``.  A type-6 entry whose pair is
``[7, L]`` stores the difference of loop ``L`` against the reference loop 7.
Once the store carries those entries, every loop rule for a loop other than 7
must resolve to its ``[7, L]`` entry rather than to loop ``L``'s type-1 entry,
and must carry ``channel_factor`` ``-1`` because the stored value is
``-raw_L`` up to the global sign.

This script performs that retarget mechanically -- the rules are never edited
by hand.  It copies the prior map aside first and prints both digests, so the
caller can record them.  It reads the pair-to-entry assignment from the
description store itself, so the entry order is never assumed.

Loops 12-27 are ``corpus-validated`` against the loop-reference fits; loops
1-6 and 8-11 stay ``source-only`` because they track only on E100642.  Loop 7's
rule is left untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

REFERENCE_LOOP = 7
CORPUS_VALIDATED_FROM = 12
REFERENCE_EVIDENCE = "docs/evidence/fragments/jt60sa-machine-map/jtmm-loop-reference-rca.html"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _differential_entries(store: Path) -> dict[int, int]:
    """Map loop number to the flux_loop index whose pair is ``[7, L]``."""

    import imas

    pairs: dict[int, int] = {}
    with imas.DBEntry(store, "r") as entry:
        flux_loop = entry.get("magnetics").flux_loop
        for index, loop in enumerate(flux_loop):
            raw = getattr(loop, "indices_differential", None)
            if raw is None:
                continue
            values = np.asarray(raw).reshape(-1)
            if values.size != 2:
                raise SystemExit(
                    f"flux_loop entry {index} names {values.size} differential "
                    "indices; a type-6 entry names exactly two"
                )
            first, second = int(values[0]), int(values[1])
            if first == REFERENCE_LOOP and second != REFERENCE_LOOP:
                if second in pairs:
                    raise SystemExit(
                        f"loop {second} is named by more than one differential entry"
                    )
                pairs[second] = index
    if not pairs:
        raise SystemExit("the store carries no differential flux-loop entry")
    return pairs


def _rewrite_rule(rule: dict, index: int) -> dict:
    loop = int(rule["semantic_id"].rsplit("_", 2)[1])
    if loop == REFERENCE_LOOP:
        return rule
    if loop >= CORPUS_VALIDATED_FROM:
        state = "corpus-validated"
        evidence = (
            f"Raw MDAC magFlxLp{loop} is SELENE flux loop {loop}, whose type-6 "
            f"differential entry in the 53-loop store names [7, {loop}] and so "
            f"stores loop {loop} minus reference loop 7; the entry carries no "
            "flux of its own. The loop-reference fits ("
            f"{REFERENCE_EVIDENCE}) place loop {loop} among the loops that "
            "track reference 7 against their own absolute prediction on "
            "E100579/E100595/E100642, so the difference this entry stores "
            "carries loop 7's proven convention. Because the entry stores "
            "-raw_L up to the global sign, the rule negates: channel_factor -1 "
            "in this map's target COCOS 17."
        )
    else:
        state = "source-only"
        evidence = (
            f"Raw MDAC magFlxLp{loop} is SELENE flux loop {loop}, whose type-6 "
            f"differential entry in the 53-loop store names [7, {loop}] and so "
            f"stores loop {loop} minus reference loop 7; the entry carries no "
            "flux of its own. The loop-reference fits ("
            f"{REFERENCE_EVIDENCE}) show loop {loop} reaching slope -1 only on "
            "E100642, so it stays source-only: its convention is not fixed. "
            "Because the entry stores -raw_L up to the global sign, the rule "
            "negates: channel_factor -1 in this map's target COCOS 17."
        )
    rewritten = dict(rule)
    rewritten["target_index"] = index
    rewritten["channel_factor"] = -1.0
    rewritten["validation_state"] = state
    rewritten["evidence"] = evidence
    return rewritten


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle",
        type=Path,
        required=True,
        help="the JT-60SA bundle directory, e.g. maps/jt-60sa",
    )
    parser.add_argument(
        "--store",
        type=Path,
        default=None,
        help="the 53-loop description store (default: <bundle>/machine_description/OP1/magnetics.nc)",
    )
    args = parser.parse_args(argv)

    bundle: Path = args.bundle
    live = bundle / "maps" / "magnetics.json"
    prior = bundle / "superseded" / "pre-differential-loops" / "maps" / "magnetics.json"
    store = args.store or bundle / "machine_description" / "OP1" / "magnetics.nc"

    prior.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(live, prior)

    pairs = _differential_entries(store)

    document = json.loads(live.read_text())
    changed = 0
    for position, rule in enumerate(document["signals"]):
        if rule.get("target_path") != "magnetics/flux_loop/flux/data":
            continue
        loop = int(rule["semantic_id"].rsplit("_", 2)[1])
        if loop == REFERENCE_LOOP:
            continue
        if loop not in pairs:
            raise SystemExit(f"no differential entry names loop {loop}")
        document["signals"][position] = _rewrite_rule(rule, pairs[loop])
        changed += 1

    live.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")

    print(f"retargeted {changed} loop rules")
    print(f"prior   {prior}: {_digest(prior)}")
    print(f"live    {live}: {_digest(live)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())