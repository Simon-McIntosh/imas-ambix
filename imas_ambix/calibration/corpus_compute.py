"""Compute and persist corpus-wide (absolute / SI) signal calibration.

This is the engine that supplies the per-channel CORPUS mean/std the signal
token encoders use so that a given physical value maps to the SAME token in
every shot and on every machine (absolute magnitude survives tokenisation).
Without it the encoders standardise PER-SHOT, which destroys absolute
magnitude and collapses the diagnostics->equilibrium probe skill.

Two source paths, one per group family:

- **L2 light-path groups** (``summary``, ``pf_active``, ``interferometer``,
  ``gas_injection``, ``soft_x_rays``, ``magnetics``) are calibrated directly
  over their level-2 Zarr group via
  :func:`~imas_ambix.calibration.signals.compute_signal_calibration`.
- **v2 signal groups** (``xma``, ``xim``, ``xsx``) are calibrated over the
  SAME ``(C, T)`` window the encoder sees — the
  :func:`~imas_ambix.tokenizer.signal_hf_encode.load_shot_window` output —
  because their channel set is loader-derived (e.g. coil-index normalised),
  not a raw Zarr group.  Per-channel statistics accumulate with the same
  streaming Welford aggregator used by the Zarr path.

Persisted layout (v2 namespace, the v1 path is preserved untouched)::

    /work/projects/imas_gpu/mast-tokens/v2/calibration/signals/{group}.json

Usage::

    # one group
    uv run python -m imas_ambix.calibration.corpus_compute --group xma
    # every known signal group
    uv run python -m imas_ambix.calibration.corpus_compute --group all
    # shard across CPU nodes (IO-bound GPFS reads)
    uv run python -m imas_ambix.calibration.corpus_compute --group all --shard 0/8
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from imas_ambix.calibration.persistence import (
    load_signal_calibration,
    save_calibration,
)
from imas_ambix.calibration.signals import (
    ChannelCalibration,
    _WelfordAccumulator,
    compute_signal_calibration,
)
from imas_ambix.data.paths import LEVEL1_DIR, LEVEL2_DIR, TOKEN_ROOT

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# v2 calibration root — the v2 store generation.  Kept SEPARATE from the v1
# CALIBRATION_ROOT so re-generating absolute calibration never clobbers the
# v1 path.  Resolved here (persistence.py owns only the v1 constant).
# ---------------------------------------------------------------------------

CALIBRATION_V2_ROOT = TOKEN_ROOT / "v2" / "calibration"
"""Root for v2-generation persisted calibration files.

Layout::

    {CALIBRATION_V2_ROOT}/signals/{group}.json
"""


def calibration_path(group: str, *, root: Path | None = None) -> Path:
    """Path of the persisted v2 calibration JSON for ``group``."""
    base = root if root is not None else CALIBRATION_V2_ROOT
    return base / "signals" / f"{group}.json"


# ---------------------------------------------------------------------------
# Group source map.  Each entry says where the group's source data lives and
# how to read its per-channel time series.
#   - "zarr": open {tier}/{shot}.zarr group=<imas_group>, all 1-D time vars.
#   - "window": load via the signal_hf encode loader ((C,T) + channel names).
# ---------------------------------------------------------------------------

# v2 signal groups read through the encode-time window loader.
WINDOW_GROUPS: tuple[str, ...] = ("xma", "xim", "xsx")

# L2 light-path groups read directly off their level-2 Zarr group.  The Zarr
# group name matches the group key for these IMAS groups.
ZARR_GROUPS: tuple[str, ...] = (
    "summary",
    "pf_active",
    "interferometer",
    "gas_injection",
    "soft_x_rays",
    "magnetics",
)

KNOWN_GROUPS: tuple[str, ...] = WINDOW_GROUPS + ZARR_GROUPS


# ---------------------------------------------------------------------------
# Shot enumeration
# ---------------------------------------------------------------------------


def _enumerate_shots(source_dir: Path) -> list[int]:
    """Every shot id present on disk under ``source_dir`` (``{id}.zarr``)."""
    out: list[int] = []
    for p in sorted(Path(source_dir).glob("*.zarr")):
        stem = p.name[: -len(".zarr")]
        if stem.isdigit():
            out.append(int(stem))
    return out


def _source_dir_for_group(group: str) -> Path:
    """Level-1 for the v2 signal groups, level-2 for the light-path groups."""
    return LEVEL1_DIR if group in WINDOW_GROUPS else LEVEL2_DIR


def _shard(shot_ids: list[int], shard: tuple[int, int] | None) -> list[int]:
    """Return the ``i``-th of ``n`` contiguous-stride shards of ``shot_ids``."""
    if shard is None:
        return shot_ids
    i, n = shard
    return shot_ids[i::n]


# ---------------------------------------------------------------------------
# Window-loader calibration (xma / xim / xsx)
# ---------------------------------------------------------------------------


def _compute_window_calibration(
    group: str, shot_ids: list[int]
) -> dict[str, ChannelCalibration]:
    """Accumulate per-channel corpus stats from the encode-time window loader.

    Reads each shot's ``(C, T)`` window via the signal_hf loader (the exact
    array the encoder normalises), so the calibration matches the encode-time
    channel set one-for-one.  A bad / absent shot is skipped, not fatal.
    """
    import numpy as np

    from imas_ambix.tokenizer.signal_hf_encode import (
        group_present,
        load_shot_window,
    )

    welford: dict[str, _WelfordAccumulator] = {}
    gmin: dict[str, float] = {}
    gmax: dict[str, float] = {}
    q01: dict[str, list[tuple[float, int]]] = {}
    q50: dict[str, list[tuple[float, int]]] = {}
    q99: dict[str, list[tuple[float, int]]] = {}
    n_shots: dict[str, int] = {}

    for sid in shot_ids:
        try:
            if not group_present(sid, group):
                continue
            w = load_shot_window(sid, group)
        except Exception:  # noqa: BLE001 — corpus robustness
            logger.debug("shot %s group %s: window load failed", sid, group)
            continue
        if w is None:
            continue
        data, chan, _valid, _rate, _window = w
        data = np.asarray(data, dtype=np.float64)  # (C, T)
        for row, name in enumerate(chan):
            arr = data[row]
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                continue
            if name not in welford:
                welford[name] = _WelfordAccumulator()
                gmin[name] = float("inf")
                gmax[name] = float("-inf")
                q01[name] = []
                q50[name] = []
                q99[name] = []
                n_shots[name] = 0
            welford[name].update(arr)
            gmin[name] = min(gmin[name], float(arr.min()))
            gmax[name] = max(gmax[name], float(arr.max()))
            n = int(arr.size)
            q01[name].append((float(np.percentile(arr, 1)), n))
            q50[name].append((float(np.percentile(arr, 50)), n))
            q99[name].append((float(np.percentile(arr, 99)), n))
            n_shots[name] += 1

    def _wq(pairs: list[tuple[float, int]]) -> float:
        if not pairs:
            return float("nan")
        total = sum(n for _, n in pairs)
        if total == 0:
            return float("nan")
        return sum(q * n for q, n in pairs) / total

    out: dict[str, ChannelCalibration] = {}
    for name, acc in welford.items():
        out[name] = ChannelCalibration(
            name=name,
            mean=acc.mean,
            std=acc.std,
            min_value=gmin.get(name, float("nan")),
            max_value=gmax.get(name, float("nan")),
            q01=_wq(q01.get(name, [])),
            q50=_wq(q50.get(name, [])),
            q99=_wq(q99.get(name, [])),
            n_samples=acc.n,
            n_shots=n_shots.get(name, 0),
        )
    return out


# ---------------------------------------------------------------------------
# Per-group driver
# ---------------------------------------------------------------------------


def compute_group_calibration(
    group: str,
    *,
    shot_ids: list[int] | None = None,
    shard: tuple[int, int] | None = None,
    max_workers: int = 4,
) -> dict[str, ChannelCalibration]:
    """Compute the corpus calibration for one signal ``group``.

    ``shot_ids`` defaults to every ``{id}.zarr`` under the group's source
    mirror (level-1 for the v2 signal groups, level-2 for the light-path
    groups).  ``shard=(i, n)`` keeps only the ``i``-th contiguous-stride
    shard so the work fans cleanly across CPU nodes.
    """
    if group not in KNOWN_GROUPS:
        raise ValueError(f"unknown group {group!r}; known: {', '.join(KNOWN_GROUPS)}")

    if shot_ids is None:
        shot_ids = _enumerate_shots(_source_dir_for_group(group))
    shot_ids = _shard(shot_ids, shard)

    logger.info(
        "computing calibration for group %r over %d shots%s",
        group,
        len(shot_ids),
        f" (shard {shard[0]}/{shard[1]})" if shard else "",
    )

    if group in WINDOW_GROUPS:
        return _compute_window_calibration(group, shot_ids)

    source_dir = _source_dir_for_group(group)
    paths = [source_dir / f"{sid}.zarr" for sid in shot_ids]
    return compute_signal_calibration(paths, group, max_workers=max_workers)


def _merge(
    a: dict[str, ChannelCalibration], b: dict[str, ChannelCalibration]
) -> dict[str, ChannelCalibration]:
    """Combine two shard calibrations channel-wise (sample-count weighted).

    Used to reduce shard outputs into a single per-group calibration.  Welford
    moments are merged exactly (Chan's parallel form); quantiles take the
    larger-shot-count estimate as a pragmatic approximation.
    """
    out: dict[str, ChannelCalibration] = dict(a)
    for name, cb in b.items():
        ca = out.get(name)
        if ca is None:
            out[name] = cb
            continue
        na, nb = ca.n_samples, cb.n_samples
        nab = na + nb
        if nab == 0:
            continue
        delta = cb.mean - ca.mean
        mean = (na * ca.mean + nb * cb.mean) / nab
        # m2 = std^2 * n  for each, then Chan-merge.
        m2a = (ca.std**2) * na
        m2b = (cb.std**2) * nb
        m2 = m2a + m2b + delta**2 * na * nb / nab
        std = (m2 / nab) ** 0.5
        bigger = ca if ca.n_shots >= cb.n_shots else cb
        out[name] = ChannelCalibration(
            name=name,
            mean=mean,
            std=std,
            min_value=min(ca.min_value, cb.min_value),
            max_value=max(ca.max_value, cb.max_value),
            q01=bigger.q01,
            q50=bigger.q50,
            q99=bigger.q99,
            n_samples=nab,
            n_shots=ca.n_shots + cb.n_shots,
        )
    return out


# ---------------------------------------------------------------------------
# Load helper for the encoders
# ---------------------------------------------------------------------------


def load_group_calibration(
    group: str, *, root: Path | None = None
) -> dict[str, ChannelCalibration] | None:
    """Load the persisted v2 calibration for ``group``, or ``None`` if absent.

    The token encoders call this once before their encode loop; a ``None``
    result means "no calibration on disk" and the encoder keeps its existing
    (per-shot) behaviour, so the corpus is never silently mis-encoded.
    """
    path = calibration_path(group, root=root)
    if not path.exists():
        return None
    return load_signal_calibration(path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_shard(text: str | None) -> tuple[int, int] | None:
    if text is None:
        return None
    i_str, n_str = text.split("/", 1)
    i, n = int(i_str), int(n_str)
    if not (0 <= i < n):
        raise ValueError(f"shard must be 0 <= i < n, got {text!r}")
    return i, n


def main(argv: list[str] | None = None) -> int:
    """Compute corpus calibration for one or all signal groups and persist it.

    Resume-safe: a group whose v2 calibration JSON already exists is skipped
    unless ``--overwrite`` is passed.  Shardable for fan-out across CPU nodes;
    a sharded run writes ``{group}.shard-i-of-n.json`` so the orchestrator can
    reduce them with ``--reduce`` once every shard is done.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--group",
        default="all",
        help="signal group name, or 'all' for every known group",
    )
    parser.add_argument(
        "--shots",
        default="all",
        help="'all' (enumerate the source mirror) or comma-separated ids",
    )
    parser.add_argument(
        "--shard",
        default=None,
        help="i/n — keep only the i-th contiguous-stride shard (IO fan-out)",
    )
    parser.add_argument(
        "--out-root",
        default=None,
        help="calibration root override (default: CALIBRATION_V2_ROOT)",
    )
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="recompute even if the target JSON already exists",
    )
    parser.add_argument(
        "--reduce",
        action="store_true",
        help="merge all {group}.shard-*.json into {group}.json and exit",
    )
    args = parser.parse_args(argv)

    root = Path(args.out_root) if args.out_root else CALIBRATION_V2_ROOT
    groups = list(KNOWN_GROUPS) if args.group == "all" else [args.group]
    shard = _parse_shard(args.shard)

    if args.reduce:
        for group in groups:
            _reduce_shards(group, root=root)
        return 0

    if args.shots == "all":
        shot_ids = None
    else:
        shot_ids = [int(s) for s in args.shots.split(",") if s.strip()]

    for group in groups:
        if shard is not None:
            out_path = root / "signals" / f"{group}.shard-{shard[0]}-of-{shard[1]}.json"
        else:
            out_path = calibration_path(group, root=root)
        if out_path.exists() and not args.overwrite:
            logger.info("group %r: %s exists — skipping", group, out_path)
            continue
        try:
            cal = compute_group_calibration(
                group,
                shot_ids=shot_ids,
                shard=shard,
                max_workers=args.max_workers,
            )
        except Exception:
            logger.exception("group %r: calibration failed", group)
            continue
        save_calibration(cal, out_path)
        logger.info("group %r: wrote %d channels -> %s", group, len(cal), out_path)
    return 0


def _reduce_shards(group: str, *, root: Path) -> None:
    """Merge every ``{group}.shard-*.json`` under ``root`` into ``{group}.json``."""
    shard_dir = root / "signals"
    shards = sorted(shard_dir.glob(f"{group}.shard-*.json"))
    if not shards:
        logger.warning("group %r: no shard files to reduce in %s", group, shard_dir)
        return
    merged: dict[str, ChannelCalibration] = {}
    for sp in shards:
        merged = _merge(merged, load_signal_calibration(sp))
    out_path = calibration_path(group, root=root)
    save_calibration(merged, out_path)
    logger.info(
        "group %r: reduced %d shards -> %d channels -> %s",
        group,
        len(shards),
        len(merged),
        out_path,
    )


if __name__ == "__main__":
    raise SystemExit(main())
