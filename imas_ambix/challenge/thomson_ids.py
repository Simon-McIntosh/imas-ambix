"""Publish challenge Thomson measurements through their native IMAS IDS."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from math import pi
from pathlib import Path
from typing import TYPE_CHECKING, Any

import imas
import numpy as np
from imas.dd_zip import dd_xml_versions
from imas.ids_struct_array import IDSStructArray
from imas.ids_structure import IDSStructure

from imas_ambix.challenge.download import REVISION
from imas_ambix.challenge.loader import ChallengeShot, load_shot
from imas_ambix.thomson.bank import _profile_geometry

if TYPE_CHECKING:
    from collections.abc import Iterable

    from numpy.typing import NDArray

DATA_DICTIONARY_MAJOR = 4
DATA_DICTIONARY_VERSION = "4.1.1"
DIIID_THOMSON_TOROIDAL_ANGLE_RAD = 2.0 * pi / 3.0
DIIID_THOMSON_LOCATION_SOURCE = "https://www.osti.gov/servlets/purl/1351087"
GATE_SHOTS = (
    "d3d_shot_00000c4a7b",
    "d3d_shot_0003ff34e7",
    "d3d_shot_001554e054",
    "d3d_shot_002495e835",
    "d3d_shot_0040ca9bdc",
)


class ThomsonIdsError(ValueError):
    """A challenge measurement cannot satisfy the Thomson IDS contract."""


@dataclass(frozen=True)
class ThomsonChannel:
    """One detached consumer channel in the units declared by the dictionary."""

    name: str
    position_r_m: float
    position_z_m: float
    position_phi_rad: float
    time_s: NDArray[np.float64]
    density_m3: NDArray[np.float64]
    temperature_ev: NDArray[np.float64]


def _publication_version(versions: Iterable[str] | None = None) -> str:
    """Require the configured pin to be an installed publication dictionary."""

    available = set(dd_xml_versions() if versions is None else versions)
    if DATA_DICTIONARY_VERSION not in available:
        raise ThomsonIdsError(
            f"required Data Dictionary {DATA_DICTIONARY_VERSION} is unavailable"
        )
    major = int(DATA_DICTIONARY_VERSION.partition(".")[0])
    if major != DATA_DICTIONARY_MAJOR:
        raise ThomsonIdsError(
            f"configured Data Dictionary major is {major}, "
            f"expected {DATA_DICTIONARY_MAJOR}"
        )
    return DATA_DICTIONARY_VERSION


def _profile_indices(shot: ChallengeShot, profile_name: str) -> NDArray[np.int64]:
    names = np.asarray(shot.chord_geometry["thomson_chord_name"], dtype=str)
    if profile_name == "core":
        prefix = "TS_core"
    elif profile_name == "edge":
        prefix = "TS_tangential"
    else:
        raise ThomsonIdsError(f"unsupported Thomson profile {profile_name!r}")
    return np.flatnonzero(np.char.startswith(names, prefix))


def _source_comment(source_path: Path) -> str:
    return (
        f"DIII-D challenge source shot {source_path.name}; corpus revision "
        f"{REVISION}; electron density, electron temperature, and scattering-volume "
        "positions were read through imas_ambix.challenge.loader; channel toroidal "
        "position is the published DIII-D Thomson diagnostic location at 120 degrees. "
        "The corpus supplies no measurement uncertainty, laser start or end point, "
        "or channel collection line-of-sight endpoints; those fields are absent."
    )


def build_thomson_ids(
    source_path: str | Path,
    *,
    dd_version: str | None = None,
) -> Any:
    """Build one heterogeneous-time Thomson IDS from a challenge shot."""

    source = Path(source_path)
    version = _publication_version()
    if dd_version is not None and dd_version != version:
        raise ThomsonIdsError(
            f"writer is pinned to Data Dictionary {version}, got {dd_version}"
        )
    shot = load_shot(source)
    if shot.source != "DIII-D":
        raise ThomsonIdsError(f"expected a DIII-D challenge shot, found {shot.source}")

    ids = imas.IDSFactory(version).new("thomson_scattering")
    ids.ids_properties.homogeneous_time = 0
    ids.ids_properties.comment = _source_comment(source)

    channel_records: list[
        tuple[str, float, float, np.ndarray, np.ndarray, np.ndarray]
    ] = []
    all_names = np.asarray(shot.chord_geometry["thomson_chord_name"], dtype=str)
    for profile_name in ("core", "edge"):
        profile = shot.thomson[profile_name]
        indices = _profile_indices(shot, profile_name)
        radius, height = _profile_geometry(shot, profile_name)
        if len(indices) != len(radius):
            raise ThomsonIdsError(
                f"{profile_name} channel and geometry counts disagree"
            )
        time_s = np.asarray(profile.time_ms, dtype=np.float64) * 1.0e-3
        for offset, source_index in enumerate(indices):
            channel_records.append(
                (
                    str(all_names[source_index]),
                    float(radius[offset]),
                    float(height[offset]),
                    time_s,
                    np.asarray(profile.density_m3[:, offset], dtype=np.float64),
                    np.asarray(profile.temperature_ev[:, offset], dtype=np.float64),
                )
            )

    ids.channel.resize(len(channel_records))
    for target, record in zip(ids.channel, channel_records, strict=True):
        name, radius, height, time_s, density, temperature = record
        target.name = name
        target.position.r = radius
        target.position.z = height
        target.position.phi = DIIID_THOMSON_TOROIDAL_ANGLE_RAD
        target.n_e.time = time_s
        target.n_e.data = density
        target.t_e.time = time_s
        target.t_e.data = temperature
    ids.validate()
    return ids


def write_thomson_ids(
    source_path: str | Path,
    output_path: str | Path,
    *,
    dd_version: str | None = None,
) -> Path:
    """Write one shot as occurrence zero and reopen it at the exact pin."""

    version = _publication_version()
    ids = build_thomson_ids(source_path, dd_version=dd_version)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with imas.DBEntry(output, "x", dd_version=version) as entry:
        entry.put(ids)
    with imas.DBEntry(output, "r", dd_version=version) as entry:
        if entry.list_all_occurrences("thomson_scattering") != [0]:
            raise ThomsonIdsError("written entry must contain exactly occurrence zero")
        reopened = entry.get("thomson_scattering", 0, autoconvert=False)
        reopened.validate()
        written = str(reopened.ids_properties.version_put.data_dictionary)
        if written != version:
            raise ThomsonIdsError(
                f"written Data Dictionary is {written}, expected {version}"
            )
    return output


def read_thomson_channels(
    entry_path: str | Path,
    *,
    dd_version: str | None = None,
) -> tuple[ThomsonChannel, ...]:
    """Read and detach the measured channels for one challenge shot."""

    version = _publication_version()
    if dd_version is not None and dd_version != version:
        raise ThomsonIdsError(
            f"reader is pinned to Data Dictionary {version}, got {dd_version}"
        )
    with imas.DBEntry(Path(entry_path), "r", dd_version=version) as entry:
        ids = entry.get("thomson_scattering", 0, autoconvert=False)
        ids.validate()
        written = str(ids.ids_properties.version_put.data_dictionary)
        if written != version:
            raise ThomsonIdsError(f"entry carries Data Dictionary {written}")
        if int(ids.ids_properties.homogeneous_time) != 0:
            raise ThomsonIdsError("per-family time bases require heterogeneous time")
        channels = []
        for channel in ids.channel:
            density_time = np.asarray(channel.n_e.time, dtype=np.float64)
            temperature_time = np.asarray(channel.t_e.time, dtype=np.float64)
            if not np.array_equal(density_time, temperature_time):
                raise ThomsonIdsError(f"{channel.name} signal time bases disagree")
            channels.append(
                ThomsonChannel(
                    name=str(channel.name),
                    position_r_m=float(channel.position.r),
                    position_z_m=float(channel.position.z),
                    position_phi_rad=float(channel.position.phi),
                    time_s=density_time.copy(),
                    density_m3=np.asarray(channel.n_e.data, dtype=np.float64).copy(),
                    temperature_ev=np.asarray(
                        channel.t_e.data, dtype=np.float64
                    ).copy(),
                )
            )
    names = [channel.name for channel in channels]
    if not names or len(names) != len(set(names)):
        raise ThomsonIdsError("channel names must be nonempty and unique")
    return tuple(channels)


def _populated_paths(node: Any) -> set[str]:
    if isinstance(node, IDSStructure):
        return set().union(*(_populated_paths(child) for child in node))
    if isinstance(node, IDSStructArray):
        return set().union(*(_populated_paths(child) for child in node))
    if node.has_value:
        return {str(node.metadata.path_string)}
    return set()


def _assert_exact_source_round_trip(
    source_path: Path, channels: tuple[ThomsonChannel, ...]
) -> None:
    shot = load_shot(source_path)
    expected_names: list[str] = []
    expected_r: list[float] = []
    expected_z: list[float] = []
    expected_times: list[np.ndarray] = []
    expected_density: list[np.ndarray] = []
    expected_temperature: list[np.ndarray] = []
    names = np.asarray(shot.chord_geometry["thomson_chord_name"], dtype=str)
    for profile_name in ("core", "edge"):
        profile = shot.thomson[profile_name]
        indices = _profile_indices(shot, profile_name)
        radius, height = _profile_geometry(shot, profile_name)
        for offset, source_index in enumerate(indices):
            expected_names.append(str(names[source_index]))
            expected_r.append(float(radius[offset]))
            expected_z.append(float(height[offset]))
            expected_times.append(np.asarray(profile.time_ms, dtype=float) * 1.0e-3)
            expected_density.append(
                np.asarray(profile.density_m3[:, offset], dtype=float)
            )
            expected_temperature.append(
                np.asarray(profile.temperature_ev[:, offset], dtype=float)
            )
    if [channel.name for channel in channels] != expected_names:
        raise ThomsonIdsError("round-trip channel names differ from the source")
    np.testing.assert_array_equal(
        [channel.position_r_m for channel in channels], expected_r
    )
    np.testing.assert_array_equal(
        [channel.position_z_m for channel in channels], expected_z
    )
    for channel, time_s, density, temperature in zip(
        channels,
        expected_times,
        expected_density,
        expected_temperature,
        strict=True,
    ):
        np.testing.assert_array_equal(channel.time_s, time_s)
        np.testing.assert_array_equal(channel.density_m3, density)
        np.testing.assert_array_equal(channel.temperature_ev, temperature)


def _artifact_record(source_path: Path, artifact_path: Path) -> dict[str, Any]:
    version = _publication_version()
    channels = read_thomson_channels(artifact_path)
    _assert_exact_source_round_trip(source_path, channels)
    with imas.DBEntry(artifact_path, "r", dd_version=version) as entry:
        ids = entry.get("thomson_scattering", 0, autoconvert=False)
        paths = sorted(_populated_paths(ids))
    core = [channel for channel in channels if channel.name.startswith("TS_core")]
    tangential = [
        channel for channel in channels if channel.name.startswith("TS_tangential")
    ]
    return {
        "artifact": artifact_path.name,
        "source_shot": source_path.name,
        "sha256": hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
        "size_bytes": artifact_path.stat().st_size,
        "data_dictionary": version,
        "data_dictionary_major": int(version.partition(".")[0]),
        "homogeneous_time": 0,
        "channel_count": len(channels),
        "family_channel_counts": {
            "core": len(core),
            "tangential": len(tangential),
        },
        "time_samples": {
            "core": len(core[0].time_s),
            "tangential": len(tangential[0].time_s),
        },
        "written_dd_paths": paths,
        "round_trip_exact": True,
    }


def vendor_gate_shot_artifacts(
    corpus_root: str | Path,
    output_directory: str | Path,
) -> dict[str, Any]:
    """Write, reopen, and report the fixed five-shot artifact census."""

    corpus = Path(corpus_root)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for shot in GATE_SHOTS:
        source_path = corpus / f"{shot}.parquet"
        if not source_path.is_file():
            raise ThomsonIdsError(f"challenge source is unavailable: {source_path}")
        artifact_path = output / f"{shot}.nc"
        write_thomson_ids(source_path, artifact_path)
        records.append(_artifact_record(source_path, artifact_path))
    report = {
        "corpus": "Sophelio/fusion-equilibrium-challenge",
        "corpus_revision": REVISION,
        "data_dictionary": _publication_version(),
        "data_dictionary_major": DATA_DICTIONARY_MAJOR,
        "toroidal_position": {
            "value_rad": DIIID_THOMSON_TOROIDAL_ANGLE_RAD,
            "value_deg": 120.0,
            "source": DIIID_THOMSON_LOCATION_SOURCE,
        },
        "artifacts": records,
        "declared_absent": [
            {
                "quantity": "channel measurement uncertainty",
                "reason": "no density or temperature uncertainties occur in the corpus",
            },
            {
                "quantity": "channel line_of_sight first_point and second_point",
                "reason": (
                    "the scattering volumes do not fix each collection optic endpoint"
                ),
            },
            {
                "quantity": "laser start_point and end_point",
                "reason": (
                    "the corpus and cited machine source do not give exact endpoints"
                ),
            },
            {
                "quantity": "scattering-volume position uncertainty",
                "reason": "the corpus supplies positions without uncertainty",
            },
            {
                "quantity": "plasma-edge marker",
                "reason": "no edge or separatrix marker occurs in the Thomson columns",
            },
        ],
    }
    report_path = output / "validation.json"
    with report_path.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return report
