"""Build explicit shotlists for the PREPARED (held) GPU frame encodes.

Three new frame/signal encodes are PREPARED but GATED on a signalled
free card (the unified corpus + current jobs hold 8/8 GPUs).  The two CAMERA
encodes (rbc; the M5/M6 rbb backfill) reuse the production frame encoder
(``scripts/slurm/stream_encode_rbb.sbatch``) verbatim — they differ only in the
CAMERA and the shotlist — so this script just writes the explicit shotlists they
consume.  (The ait divertor heat-flux is a SIGNAL stream, not frames — handled by
``encode_ait_signal.py``, not here.)

Phases:
  rbc      — shots whose L1 carries an ``rbc`` camera group but have NO rbc token
             store yet (untokenised visible).
  backfill — shots with id < 15085 (the M5/M6 early campaigns, below the current
             tokenised floor) whose L1 carries an ``rbb`` camera and have no rbb
             token store yet.

Writes a plain shotlist JSON ``{"camera", "shot_ids", "n"}`` per phase.  CPU/sun.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import zarr

from imas_ambix.camdyn.dataset import frames_token_path, level1_shot_path
from imas_ambix.data.paths import LEVEL1_DIR

TOKENISED_FLOOR = 15085  # current tokenised corpus min shot id


def _l1_has_camera(sid: int, camera: str) -> bool:
    try:
        g = zarr.open_group(str(level1_shot_path(int(sid))), mode="r")
        if camera not in set(g.group_keys()):
            return False
        grp = g[camera]
        ak = set(grp.array_keys())
        return "data" in ak and grp["data"].shape and grp["data"].shape[0] > 0
    except Exception:  # noqa: BLE001
        return False


def _already_tokenised(sid: int, camera: str) -> bool:
    try:
        p = frames_token_path(int(sid), camera, "v1", token_root=None)
        if not p.exists():
            return False
        st = zarr.open_group(str(p), mode="r")
        import numpy as np  # noqa: PLC0415

        return int(np.asarray(st["tokens"]).shape[0]) > 0
    except Exception:  # noqa: BLE001
        return False


def _all_l1_shots() -> list[int]:
    root = Path(LEVEL1_DIR)
    return sorted(
        int(p.name[:-5]) for p in root.glob("*.zarr") if p.name[:-5].isdigit()
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", choices=["rbc", "backfill"], required=True)
    ap.add_argument("--out", required=True, help="output shotlist JSON path")
    ap.add_argument(
        "--max-shots", type=int, default=None, help="cap (for a quick PoC encode)"
    )
    args = ap.parse_args(argv)

    all_shots = _all_l1_shots()
    print(f"[shotlist] {len(all_shots)} L1 shots on disk", flush=True)

    if args.phase == "rbc":
        camera = "rbc"
        cand = all_shots
    else:  # backfill: M5/M6 early campaigns below the tokenised floor
        camera = "rbb"
        cand = [s for s in all_shots if s < TOKENISED_FLOOR]
    print(
        f"[shotlist] phase={args.phase} camera={camera} candidates={len(cand)}",
        flush=True,
    )

    keep: list[int] = []
    for i, sid in enumerate(cand):
        if _l1_has_camera(sid, camera) and not _already_tokenised(sid, camera):
            keep.append(sid)
        if (i + 1) % 500 == 0:
            print(
                f"[shotlist]   scanned {i + 1}/{len(cand)}, {len(keep)} kept",
                flush=True,
            )
        if args.max_shots and len(keep) >= args.max_shots:
            break

    out = {"camera": camera, "phase": args.phase, "n": len(keep), "shot_ids": keep}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out))
    print(
        f"[shotlist] phase={args.phase}: {len(keep)} shots need {camera} encode "
        f"-> {args.out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
