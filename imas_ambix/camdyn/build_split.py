"""Build and commit the camera-dynamics shot-level split manifest.

Scans the rbb token corpus, forces the 112 MSE held-out shots into the
held-out partition (those that carry tokens), and writes the manifest to
``imas_ambix/camdyn/artifacts/camdyn_split_v0.json``.

Run::

    uv run python -m imas_ambix.camdyn.build_split

CPU-only, cheap (directory scan + a seeded shuffle — no Zarr opens of
the token payload).
"""

from __future__ import annotations

import logging

from imas_ambix.camdyn.dataset import list_token_shot_ids
from imas_ambix.camdyn.splits import DEFAULT_SPLIT_OUT, build_camdyn_split


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    token_shots = list_token_shot_ids()
    split = build_camdyn_split(token_shots)
    out = split.save(DEFAULT_SPLIT_OUT)
    print(
        f"[camdyn_split] token shots: {split.n_token_shots}  "
        f"train={split.n_train}  val={split.n_val}  held_out={split.n_held_out}"
    )
    print(
        f"[camdyn_split] MSE held-out forced into held_out: "
        f"{len(split.mse_heldout_forced)} / 112 "
        f"({len(split.mse_heldout_without_tokens)} lack rbb tokens)"
    )
    print(f"[camdyn_split] manifest → {out}")


if __name__ == "__main__":
    main()
