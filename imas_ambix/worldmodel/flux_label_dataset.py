"""Pair pinned Nova flux-label sessions with native-cadence rbb tokens.

Only atomically completed shot sessions are discoverable.  Session geometry is
joined to the nearest recorded camera-frame time, while token histories remain
contiguous on that camera's native time axis.  The returned mappings deliberately
contain geometry, image tokens, identity, and provenance but no actuator values.
"""

from __future__ import annotations

import json
import re
from collections import OrderedDict
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray

from imas_ambix.camdyn.dataset import (
    DEFAULT_CAMERA,
    DEFAULT_VOCAB_VERSION,
    FRAME_GRID,
    frames_token_path,
    level1_shot_path,
)
from imas_ambix.data.paths import LEVEL1_DIR, TOKEN_ROOT
from imas_ambix.data.stream_encode import REGISTRY_OFFSET
from imas_ambix.worldmodel.flux_conditioning import (
    FluxGrid,
    geometry_vector,
    render_flux_conditioning,
)
from imas_ambix.worldmodel.spacetime_dataset import CAMERA_VOCAB

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping, Sequence

DEFAULT_SESSION_ROOT = Path(
    "/work/projects/imas_gpu/sophelio/labeller_sessions/76906a29"
)
DEFAULT_COHORT_REPORT = Path(
    "/home/ITER/mcintos/.config/reckon/crew/reports/"
    "physics-carried-playable-plasma/labeller-cohort-census.md"
)
EXPECTED_POLICY_DIGEST = (
    "b5f9ef456f665c593a40010b9756b16fb36abd4ebbd1fea4d83a3d628c77bc7d"
)
EXPECTED_CARRIER_IDENTITY = (
    "1d2c4a2b2f448ab8f1ae981031bbaf85fe4ee87f8ed9606fe6847d0fc9f1e994"
)
DEFAULT_HISTORY_FRAMES = 4
MAX_FRAME_DELTA_SECONDS = 0.0025
VALIDATION_INTERVAL = 20

DatasetSplit = Literal["train", "validation", "all"]
IntegerArray = NDArray[np.integer[Any]]

_SESSION_FIELDS = (
    "flux_surface_psi_norm",
    "flux_surface_r",
    "flux_surface_z",
    "magnetic_axis_r",
    "magnetic_axis_z",
    "x_point_r",
    "x_point_z",
    "finite_mask",
    "diverted",
    "elongation",
    "delta_upper",
    "delta_lower",
    "R_major",
    "a_minor",
)


@dataclass(frozen=True, slots=True)
class FluxLabelReference:
    """One admitted session slice and its joined camera-frame location."""

    shot_id: int
    rank: int
    split: str
    manifest_row: int
    session_index: int
    slice_time: float
    frame_index: int
    frame_time: float
    frame_delta_s: float
    conditioned: bool
    session_path: Path
    token_path: Path


def _read_integer_lines(path: Path) -> list[int]:
    values: list[int] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        if not line.isdigit():
            raise ValueError(f"{path}:{line_number} is not a shot number: {line!r}")
        values.append(int(line))
    return values


def _ranked_shot_positions(root: Path) -> dict[int, int]:
    """Return one-based shot ranks from the interleaved per-card walk.

    A resumed producer omits already completed shots from its card lists.  Its
    original contiguous shard inventory is then used to put those completed
    shots back at their stable positions, after proving that the card walk is
    an order-preserving subset of that inventory.
    """
    card_paths = sorted(
        (root / ".cards").glob("card-*.txt"),
        key=lambda path: int(path.stem.rsplit("-", 1)[1]),
    )
    if not card_paths:
        raise FileNotFoundError(f"no ranked card lists found under {root / '.cards'}")
    cards = [_read_integer_lines(path) for path in card_paths]
    ranked = [shot for row in zip_longest(*cards) for shot in row if shot is not None]
    if len(ranked) != len(set(ranked)):
        raise ValueError("ranked card lists repeat at least one shot")
    positions = {shot: index for index, shot in enumerate(ranked, start=1)}

    shard_paths = sorted((root / ".shards").glob("shard-*.txt"))
    if shard_paths:
        original = [shot for path in shard_paths for shot in _read_integer_lines(path)]
        if len(original) != len(set(original)):
            raise ValueError("ranked shard inventory repeats at least one shot")
        ranked_set = set(ranked)
        if [shot for shot in original if shot in ranked_set] != ranked:
            raise ValueError(
                "ranked card walk is not an order-preserving shard-inventory subset"
            )
        positions = {shot: index for index, shot in enumerate(original, start=1)}
    return positions


def read_labeller_cohort_shots(path: Path = DEFAULT_COHORT_REPORT) -> set[int]:
    """Read the four disjoint whole-shot partitions of the labeller census."""
    text = Path(path).read_text(encoding="utf-8")
    headings = (
        "Labeller train",
        "Labeller validation",
        "Clean same-campaign test",
        "Held-out campaign test",
    )
    result: set[int] = set()
    for heading in headings:
        match = re.search(
            rf"^### {re.escape(heading)}\b.*?(?=^### |\Z)",
            text,
            flags=re.MULTILINE | re.DOTALL,
        )
        if match is None:
            raise ValueError(f"cohort report is missing the {heading!r} section")
        shots = [int(value) for value in re.findall(r"\b(\d{5}):\d+\b", match.group())]
        if not shots:
            raise ValueError(f"cohort report contains no shots in {heading!r}")
        overlap = result.intersection(shots)
        if overlap:
            raise ValueError(
                f"cohort report repeats shots across partitions: {sorted(overlap)}"
            )
        result.update(shots)
    return result


def _nearest_frame_indices(
    frame_times: NDArray[np.float64], query_times: NDArray[np.float64]
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Join each query to its nearest native frame, preferring the later tie."""
    if frame_times.ndim != 1 or not frame_times.size:
        raise ValueError("camera frame times must be a non-empty one-dimensional array")
    if not np.isfinite(frame_times).all() or np.any(np.diff(frame_times) <= 0.0):
        raise ValueError("camera frame times must be finite and strictly increasing")
    if query_times.ndim != 1 or not np.isfinite(query_times).all():
        raise ValueError("slice times must be a finite one-dimensional array")
    high = np.clip(
        np.searchsorted(frame_times, query_times, side="left"),
        0,
        frame_times.size - 1,
    )
    low = np.clip(high - 1, 0, frame_times.size - 1)
    use_high = np.abs(frame_times[high] - query_times) <= np.abs(
        query_times - frame_times[low]
    )
    indices = np.where(use_high, high, low).astype(np.int64, copy=False)
    delta = (frame_times[indices] - query_times).astype(np.float64, copy=False)
    return indices, delta


def _load_companion(
    path: Path, slices: Sequence[Mapping[str, Any]]
) -> tuple[NDArray[np.int32], NDArray[np.float64], NDArray[np.bool_]]:
    written = [row for row in slices if bool(row.get("written", False))]
    expected_rows = np.asarray([int(row["row"]) for row in written], dtype=np.int32)
    expected_times = np.asarray(
        [float(row["time"]) for row in written], dtype=np.float64
    )
    with np.load(path, allow_pickle=False) as companion:
        missing = {"row", "time", "conditioned"}.difference(companion.files)
        if missing:
            raise ValueError(f"{path} is missing companion fields {sorted(missing)}")
        rows = np.asarray(companion["row"], dtype=np.int32).reshape(-1)
        times = np.asarray(companion["time"], dtype=np.float64).reshape(-1)
        conditioned = np.asarray(companion["conditioned"], dtype=bool).reshape(-1)
    if not (rows.shape == times.shape == conditioned.shape):
        raise ValueError(f"{path} companion arrays do not have identical lengths")
    if not np.array_equal(rows, expected_rows):
        raise ValueError(f"{path} rows are not aligned to the manifest's written rows")
    if not np.allclose(times, expected_times, rtol=0.0, atol=1.0e-9):
        raise ValueError(f"{path} times are not aligned to the manifest's written rows")
    return rows, times, conditioned


def _session_times(path: Path) -> NDArray[np.float64]:
    import xarray as xr  # noqa: PLC0415

    with xr.open_dataset(path, group="steering", engine="h5netcdf") as session:
        return np.asarray(session["time"], dtype=np.float64).reshape(-1)


def _camera_times(path: Path, camera: str) -> NDArray[np.float64]:
    import zarr  # noqa: PLC0415

    store = zarr.open_group(str(path), mode="r")
    if camera not in set(store.group_keys()) or "time" not in set(
        store[camera].array_keys()
    ):
        raise KeyError(f"{path} does not contain {camera}/time")
    return np.asarray(store[camera]["time"], dtype=np.float64).reshape(-1)


def _token_frame_count(path: Path) -> int:
    import zarr  # noqa: PLC0415

    store = zarr.open_group(str(path), mode="r")
    if "tokens" not in set(store.array_keys()):
        raise KeyError(f"{path} does not contain tokens")
    tokens = store["tokens"]
    if tuple(tokens.shape[1:]) != FRAME_GRID:
        raise ValueError(f"{path} token frames must have trailing shape {FRAME_GRID}")
    return int(tokens.shape[0])


def _split_for_rank(rank: int) -> str:
    return "validation" if rank % VALIDATION_INTERVAL == 0 else "train"


class FluxLabelDataset:
    """Map-style dataset of flux geometry, target tokens, and token history.

    Discovery performs all policy, carrier, cohort, slice-verdict, and time-join
    checks once.  NetCDF geometry and Zarr token payloads remain lazy, with a
    small per-process cache of loaded geometry sessions.
    """

    def __init__(
        self,
        session_root: Path = DEFAULT_SESSION_ROOT,
        *,
        split: DatasetSplit = "train",
        history_frames: int = DEFAULT_HISTORY_FRAMES,
        camera: str = DEFAULT_CAMERA,
        token_root: Path = TOKEN_ROOT,
        level1_root: Path = LEVEL1_DIR,
        cohort_report: Path = DEFAULT_COHORT_REPORT,
        cohort_shots: Collection[int] | None = None,
        shot_ids: Collection[int] | None = None,
        policy_digest: str = EXPECTED_POLICY_DIGEST,
        carrier_identity: str = EXPECTED_CARRIER_IDENTITY,
        max_frame_delta_s: float = MAX_FRAME_DELTA_SECONDS,
        grid: FluxGrid | None = None,
        session_cache_size: int = 2,
    ) -> None:
        if split not in {"train", "validation", "all"}:
            raise ValueError("split must be 'train', 'validation', or 'all'")
        if history_frames < 1:
            raise ValueError("history_frames must be positive")
        if max_frame_delta_s <= 0.0:
            raise ValueError("max_frame_delta_s must be positive")
        if session_cache_size < 1:
            raise ValueError("session_cache_size must be positive")

        self._root = Path(session_root)
        self._split = split
        self._history_frames = int(history_frames)
        self._camera = str(camera)
        self._token_root = Path(token_root)
        self._level1_root = Path(level1_root)
        self._grid = grid or FluxGrid()
        self._cache_size = int(session_cache_size)
        self._session_cache: OrderedDict[Path, Any] = OrderedDict()

        allowed_shots = None if shot_ids is None else {int(shot) for shot in shot_ids}
        excluded = (
            read_labeller_cohort_shots(cohort_report)
            if cohort_shots is None
            else {int(shot) for shot in cohort_shots}
        )
        ranks = _ranked_shot_positions(self._root)
        references: list[FluxLabelReference] = []
        dropped = {
            "unwritten": 0,
            "unconverged": 0,
            "cohort_shot": 0,
            "other_split": 0,
            "missing_token_store": 0,
            "missing_frame_times": 0,
            "outside_time_tolerance": 0,
            "insufficient_history": 0,
            "token_time_length_mismatch": 0,
        }
        counts = {
            "manifest_files": 0,
            "complete_sessions": 0,
            "ranked_shots": len(ranks),
            "cohort_shots": len(excluded),
            "cohort_shots_excluded": 0,
            "selected_sessions": 0,
            "train_shots": 0,
            "validation_shots": 0,
            "manifest_slices": 0,
            "written_slices": 0,
            "converged_slices": 0,
            "paired_slices": 0,
            "conditioned_slices": 0,
            "cohort_overlap": 0,
        }
        selected_shots: set[int] = set()
        train_shots: set[int] = set()
        validation_shots: set[int] = set()
        maximum_delta = 0.0

        for manifest_path in sorted(self._root.glob("*.manifest.json")):
            stem = manifest_path.name.removesuffix(".manifest.json")
            if not stem.isdigit():
                continue
            shot = int(stem)
            if allowed_shots is not None and shot not in allowed_shots:
                continue
            counts["manifest_files"] += 1
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("status") != "complete":
                continue
            counts["complete_sessions"] += 1
            if int(manifest.get("shot", -1)) != shot:
                raise ValueError(
                    f"{manifest_path} shot identity does not match its name"
                )
            observed_policy = str(manifest.get("policy_digest", ""))
            observed_carrier = str(manifest.get("carrier_identity", ""))
            if observed_policy != policy_digest:
                raise ValueError(
                    f"{manifest_path} policy_digest {observed_policy!r} does not match "
                    f"the pinned digest {policy_digest!r}"
                )
            if observed_carrier != carrier_identity:
                raise ValueError(
                    f"{manifest_path} carrier_identity {observed_carrier!r} does not "
                    f"match the pinned identity {carrier_identity!r}"
                )
            if shot not in ranks:
                raise ValueError(
                    f"complete session shot {shot} is absent from ranked lists"
                )

            slices = manifest.get("slices")
            if not isinstance(slices, list):
                raise ValueError(f"{manifest_path} has no slice-row list")
            counts["manifest_slices"] += len(slices)
            written_rows = [row for row in slices if bool(row.get("written", False))]
            converged_rows = [
                row for row in written_rows if bool(row.get("converged", False))
            ]
            counts["written_slices"] += len(written_rows)
            counts["converged_slices"] += len(converged_rows)
            dropped["unwritten"] += len(slices) - len(written_rows)
            dropped["unconverged"] += len(written_rows) - len(converged_rows)

            if shot in excluded:
                counts["cohort_shots_excluded"] += 1
                dropped["cohort_shot"] += len(converged_rows)
                continue
            rank = ranks[shot]
            shot_split = _split_for_rank(rank)
            if split != "all" and shot_split != split:
                dropped["other_split"] += len(converged_rows)
                continue

            companion_path = self._root / f"{shot}.npz"
            session_path = self._root / f"{shot}.nc"
            if not companion_path.is_file() or not session_path.is_file():
                raise FileNotFoundError(
                    f"complete session {shot} is missing its netCDF or NPZ companion"
                )
            rows, companion_times, conditioned = _load_companion(companion_path, slices)
            session_times = _session_times(session_path)
            if session_times.shape != companion_times.shape or not np.allclose(
                session_times, companion_times, rtol=0.0, atol=1.0e-9
            ):
                raise ValueError(
                    f"{session_path} times are not aligned to its companion"
                )
            row_to_session = {int(row): index for index, row in enumerate(rows)}

            token_path = frames_token_path(
                shot,
                self._camera,
                DEFAULT_VOCAB_VERSION,
                token_root=self._token_root,
            )
            eligible_count = len(converged_rows)
            if not token_path.exists():
                dropped["missing_token_store"] += eligible_count
                continue
            level1_path = level1_shot_path(shot, level1_dir=self._level1_root)
            if not level1_path.exists():
                dropped["missing_frame_times"] += eligible_count
                continue
            try:
                frame_times = _camera_times(level1_path, self._camera)
            except KeyError, ValueError:
                dropped["missing_frame_times"] += eligible_count
                continue
            token_count = _token_frame_count(token_path)
            query_times = np.asarray(
                [float(row["time"]) for row in converged_rows], dtype=np.float64
            )
            frame_indices, deltas = _nearest_frame_indices(frame_times, query_times)
            for row, frame_index, delta in zip(
                converged_rows, frame_indices, deltas, strict=True
            ):
                frame = int(frame_index)
                if frame >= token_count:
                    dropped["token_time_length_mismatch"] += 1
                    continue
                if abs(float(delta)) > max_frame_delta_s + np.finfo(np.float64).eps:
                    dropped["outside_time_tolerance"] += 1
                    continue
                if frame < self._history_frames:
                    dropped["insufficient_history"] += 1
                    continue
                manifest_row = int(row["row"])
                session_index = row_to_session[manifest_row]
                references.append(
                    FluxLabelReference(
                        shot_id=shot,
                        rank=rank,
                        split=shot_split,
                        manifest_row=manifest_row,
                        session_index=session_index,
                        slice_time=float(row["time"]),
                        frame_index=frame,
                        frame_time=float(frame_times[frame]),
                        frame_delta_s=float(delta),
                        conditioned=bool(conditioned[session_index]),
                        session_path=session_path,
                        token_path=token_path,
                    )
                )
                selected_shots.add(shot)
                if shot_split == "train":
                    train_shots.add(shot)
                else:
                    validation_shots.add(shot)
                maximum_delta = max(maximum_delta, abs(float(delta)))

        references.sort(
            key=lambda item: (item.rank, item.slice_time, item.manifest_row)
        )
        counts["selected_sessions"] = len(selected_shots)
        counts["train_shots"] = len(train_shots)
        counts["validation_shots"] = len(validation_shots)
        counts["paired_slices"] = len(references)
        counts["conditioned_slices"] = sum(item.conditioned for item in references)
        counts["cohort_overlap"] = len(selected_shots.intersection(excluded))
        self._references = references
        self.receipt: dict[str, object] = {
            "grid": self._grid.receipt(),
            "pins": {
                "policy_digest": policy_digest,
                "carrier_identity": carrier_identity,
            },
            "split": split,
            "history_frames": self._history_frames,
            "maximum_frame_delta_s": float(max_frame_delta_s),
            "max_abs_delta_t_s": maximum_delta,
            "token_id_space": "local",
            "counts": counts,
            "dropped_slices": dropped,
        }

    def __len__(self) -> int:
        return len(self._references)

    @property
    def references(self) -> tuple[FluxLabelReference, ...]:
        """Return the immutable discovery and pairing index."""
        return tuple(self._references)

    @property
    def split(self) -> DatasetSplit:
        return self._split

    @property
    def history_frames(self) -> int:
        return self._history_frames

    def _load_session(self, path: Path) -> Any:
        if path in self._session_cache:
            session = self._session_cache.pop(path)
            self._session_cache[path] = session
            return session
        import xarray as xr  # noqa: PLC0415

        with xr.open_dataset(path, group="steering", engine="h5netcdf") as source:
            missing = set(_SESSION_FIELDS).difference(source.variables)
            if missing:
                raise ValueError(
                    f"{path} is missing conditioning fields {sorted(missing)}"
                )
            session = source[list(_SESSION_FIELDS)].load()
        self._session_cache[path] = session
        while len(self._session_cache) > self._cache_size:
            self._session_cache.popitem(last=False)
        return session

    @staticmethod
    def _local_tokens(path: Path, start: int, stop: int) -> IntegerArray:
        import zarr  # noqa: PLC0415

        store = zarr.open_group(str(path), mode="r")
        values = np.asarray(store["tokens"][start:stop], dtype=np.int64)
        local = values - REGISTRY_OFFSET
        if np.any(local < 0) or np.any(local >= CAMERA_VOCAB):
            raise ValueError(
                f"{path} contains camera token ids outside the registry range"
            )
        return local

    def __getitem__(self, index: int) -> dict[str, object]:
        reference = self._references[index]
        session = self._load_session(reference.session_path)
        fields = session.isel(time=reference.session_index)
        history = self._local_tokens(
            reference.token_path,
            reference.frame_index - self._history_frames,
            reference.frame_index,
        )
        target = self._local_tokens(
            reference.token_path,
            reference.frame_index,
            reference.frame_index + 1,
        )[0]
        if history.shape != (self._history_frames, *FRAME_GRID):
            raise RuntimeError("token history does not have the discovered frame count")
        if target.shape != FRAME_GRID:
            raise RuntimeError("target token frame does not have the camera grid shape")
        return {
            "conditioning": render_flux_conditioning(fields, self._grid),
            "geometry": geometry_vector(fields),
            "target_tokens": target,
            "history_tokens": history,
            "conditioned": reference.conditioned,
            "shot_id": reference.shot_id,
            "rank": reference.rank,
            "slice_time": reference.slice_time,
            "frame_time": reference.frame_time,
            "frame_delta_s": reference.frame_delta_s,
            "manifest_row": reference.manifest_row,
            "split": reference.split,
        }


__all__ = [
    "DEFAULT_COHORT_REPORT",
    "DEFAULT_HISTORY_FRAMES",
    "DEFAULT_SESSION_ROOT",
    "EXPECTED_CARRIER_IDENTITY",
    "EXPECTED_POLICY_DIGEST",
    "FluxLabelDataset",
    "FluxLabelReference",
    "MAX_FRAME_DELTA_SECONDS",
    "VALIDATION_INTERVAL",
    "read_labeller_cohort_shots",
]
