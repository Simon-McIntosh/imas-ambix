"""Shot-level split manifest for the camera-dynamics corpus.

The split is over the shots that actually carry rbb camera tokens on
disk (a subset of the level-1 corpus).  Shot = independent unit
(cross-shot generalisation), reusing the spirit of
:mod:`imas_ambix.statespace.splits` — a held-out set of *whole shots*,
never a within-shot temporal split, so a held-out evaluation is honest.

Hard invariant
--------------
The **112 MSE held-out shots** (the S9 oracle set, from
``imas_ambix/statespace/artifacts/mse_split_v0.json``) MUST land in this
split's held-out partition whenever they carry rbb tokens — so the
camera-dynamics W3 probe and the stretch oracle metric are scored on the
SAME shots as the S9/S12 oracles.  Any of the 112 lacking rbb tokens is
recorded in ``mse_heldout_without_tokens`` (cannot be forced into a
token split if there is no token stream) and surfaced in the manifest.

The remaining token shots are split train / val / held-out by a seeded
shuffle; the forced MSE shots are unioned into held-out and removed from
train/val so the partitions stay disjoint.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Default location of the S9 MSE split artifact carrying the 112 held-out shots.
MSE_SPLIT_ARTIFACT = (
    Path(__file__).resolve().parent.parent
    / "statespace"
    / "artifacts"
    / "mse_split_v0.json"
)

# Default output location for the committed camdyn split manifest.
DEFAULT_SPLIT_OUT = (
    Path(__file__).resolve().parent / "artifacts" / "camdyn_split_v0.json"
)


def load_mse_heldout_shots(artifact_path: Path | None = None) -> list[int]:
    """Return the 112 MSE held-out shot IDs from the S9 split artifact."""
    path = artifact_path or MSE_SPLIT_ARTIFACT
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return sorted(int(s) for s in payload.get("held_out", []))


@dataclass
class CamdynSplit:
    """Shot-level train / val / held-out split for the camera corpus.

    Attributes
    ----------
    train, val, held_out:
        Disjoint lists of shot IDs.
    mse_heldout_forced:
        The MSE held-out shots that were forced into ``held_out`` (those
        that carry rbb tokens).
    mse_heldout_without_tokens:
        MSE held-out shots that lack rbb tokens (cannot be in a token
        split) — surfaced, not silently dropped.
    n_token_shots:
        Total shots with rbb tokens considered.
    val_fraction, seed:
        Provenance of the random train/val split.
    notes:
        Provenance / invariant statements.
    """

    train: list[int] = field(default_factory=list)
    val: list[int] = field(default_factory=list)
    held_out: list[int] = field(default_factory=list)
    mse_heldout_forced: list[int] = field(default_factory=list)
    mse_heldout_without_tokens: list[int] = field(default_factory=list)
    n_token_shots: int = 0
    val_fraction: float = 0.1
    seed: int = 42
    notes: list[str] = field(default_factory=list)

    @property
    def n_train(self) -> int:
        return len(self.train)

    @property
    def n_val(self) -> int:
        return len(self.val)

    @property
    def n_held_out(self) -> int:
        return len(self.held_out)

    def assert_invariants(self) -> None:
        """Hard gates: disjoint partitions + all MSE-with-tokens held out."""
        s_tr, s_va, s_ho = set(self.train), set(self.val), set(self.held_out)
        assert not (s_tr & s_va), f"train ∩ val: {sorted(s_tr & s_va)[:10]}"
        assert not (s_tr & s_ho), f"train ∩ held_out: {sorted(s_tr & s_ho)[:10]}"
        assert not (s_va & s_ho), f"val ∩ held_out: {sorted(s_va & s_ho)[:10]}"
        forced = set(self.mse_heldout_forced)
        missing = forced - s_ho
        assert not missing, f"forced MSE shots not in held_out: {sorted(missing)[:10]}"

    def to_dict(self) -> dict:
        return {
            "version": "camdyn_split_v0",
            "unit": "shot",
            "camera": "rbb",
            "vocab_version": "v1",
            "n_token_shots": self.n_token_shots,
            "n_train": self.n_train,
            "n_val": self.n_val,
            "n_held_out": self.n_held_out,
            "n_mse_heldout_forced": len(self.mse_heldout_forced),
            "n_mse_heldout_without_tokens": len(self.mse_heldout_without_tokens),
            "val_fraction": self.val_fraction,
            "seed": self.seed,
            "notes": self.notes,
            "mse_heldout_forced": [int(x) for x in self.mse_heldout_forced],
            "mse_heldout_without_tokens": [
                int(x) for x in self.mse_heldout_without_tokens
            ],
            "train": [int(x) for x in self.train],
            "val": [int(x) for x in self.val],
            "held_out": [int(x) for x in self.held_out],
        }

    def save(self, path: Path | None = None) -> Path:
        out = path or DEFAULT_SPLIT_OUT
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(self.to_dict(), separators=(",", ":")), encoding="utf-8"
        )
        logger.info(
            "camdyn split saved to %s (train=%d val=%d held_out=%d)",
            out,
            self.n_train,
            self.n_val,
            self.n_held_out,
        )
        return out

    @classmethod
    def load(cls, path: Path) -> CamdynSplit:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            train=[int(x) for x in d.get("train", [])],
            val=[int(x) for x in d.get("val", [])],
            held_out=[int(x) for x in d.get("held_out", [])],
            mse_heldout_forced=[int(x) for x in d.get("mse_heldout_forced", [])],
            mse_heldout_without_tokens=[
                int(x) for x in d.get("mse_heldout_without_tokens", [])
            ],
            n_token_shots=int(d.get("n_token_shots", 0)),
            val_fraction=float(d.get("val_fraction", 0.1)),
            seed=int(d.get("seed", 42)),
            notes=list(d.get("notes", [])),
        )


def build_camdyn_split(
    token_shot_ids: list[int],
    *,
    mse_heldout: list[int] | None = None,
    val_fraction: float = 0.1,
    held_out_fraction: float = 0.1,
    seed: int = 42,
    mse_artifact_path: Path | None = None,
) -> CamdynSplit:
    """Build the shot-level split, forcing the 112 MSE shots into held-out.

    Parameters
    ----------
    token_shot_ids:
        Shots that carry rbb tokens on disk (the corpus universe).
    mse_heldout:
        The MSE held-out shot IDs; loaded from the S9 artifact when None.
    val_fraction, held_out_fraction:
        Fractions of the NON-MSE token shots placed in val / held-out
        (the forced MSE shots are added to held-out on top of this).
    seed:
        RNG seed (recorded in the manifest for reproducibility).

    Returns
    -------
    CamdynSplit (invariants asserted before return).
    """
    token_set = set(int(s) for s in token_shot_ids)
    if mse_heldout is None:
        mse_heldout = load_mse_heldout_shots(mse_artifact_path)
    mse_set = set(int(s) for s in mse_heldout)

    forced = sorted(mse_set & token_set)  # MSE shots that have tokens
    without_tokens = sorted(mse_set - token_set)  # MSE shots lacking tokens

    # Remaining (non-MSE) token shots, deterministically shuffled.
    remaining = np.array(sorted(token_set - mse_set))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(remaining))
    remaining = remaining[perm]

    n_rem = len(remaining)
    n_val = int(round(n_rem * val_fraction))
    n_ho = int(round(n_rem * held_out_fraction))
    val_shots = sorted(int(x) for x in remaining[:n_val])
    extra_ho = sorted(int(x) for x in remaining[n_val : n_val + n_ho])
    train_shots = sorted(int(x) for x in remaining[n_val + n_ho :])

    held_out = sorted(set(forced) | set(extra_ho))

    notes = [
        f"unit=shot; seed={seed}; val_fraction={val_fraction}; "
        f"held_out_fraction={held_out_fraction} (of non-MSE token shots).",
        f"{len(forced)} of the 112 MSE held-out shots carry rbb tokens and "
        "were FORCED into held_out (S9/S12 oracle comparability).",
        f"{len(without_tokens)} MSE held-out shots lack rbb tokens and cannot "
        "join a token split — surfaced in mse_heldout_without_tokens.",
        "EFIT (efm/esm) and pulse-schedule (xdc) signals are banned as "
        "inputs/conditioning (leakage) — enforced in the conditioning loader, "
        "not the split.",
    ]

    split = CamdynSplit(
        train=train_shots,
        val=val_shots,
        held_out=held_out,
        mse_heldout_forced=forced,
        mse_heldout_without_tokens=without_tokens,
        n_token_shots=len(token_set),
        val_fraction=val_fraction,
        seed=seed,
        notes=notes,
    )
    split.assert_invariants()
    return split
