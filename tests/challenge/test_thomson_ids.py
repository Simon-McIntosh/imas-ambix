from __future__ import annotations

import json
import os
from math import isclose, pi
from pathlib import Path

import imas
import numpy as np
import pytest

from imas_ambix.challenge.download import REVISION
from imas_ambix.challenge.loader import load_shot
from imas_ambix.challenge.thomson_ids import (
    DATA_DICTIONARY_MAJOR,
    DATA_DICTIONARY_VERSION,
    DIIID_THOMSON_TOROIDAL_ANGLE_RAD,
    GATE_SHOTS,
    _profile_indices,
    _publication_version,
    read_thomson_channels,
    write_thomson_ids,
)
from imas_ambix.thomson.bank import _profile_geometry

CORPUS_ROOT = Path(
    os.environ.get(
        "SOPHELIO_DIIID_TRAIN",
        "/work/projects/imas_gpu/sophelio/raw/data/diii_d_train",
    )
)
ARTIFACT_ROOT = (
    Path(__file__).parents[2]
    / "imas_ambix"
    / "challenge"
    / "artifacts"
    / "thomson_scattering"
)
EXPECTED_TIME_SAMPLES = {
    "d3d_shot_00000c4a7b.nc": {"core": 1352, "tangential": 419},
    "d3d_shot_0003ff34e7.nc": {"core": 1763, "tangential": 422},
    "d3d_shot_001554e054.nc": {"core": 1345, "tangential": 422},
    "d3d_shot_002495e835.nc": {"core": 1560, "tangential": 209},
    "d3d_shot_0040ca9bdc.nc": {"core": 1456, "tangential": 423},
}


def test_data_dictionary_pin_uses_publication_major() -> None:
    assert _publication_version(["3.42.2", "4.0.0", "4.1.1"]) == "4.1.1"
    assert DATA_DICTIONARY_VERSION.split(".")[0] == str(DATA_DICTIONARY_MAJOR)


@pytest.mark.parametrize("shot_name", GATE_SHOTS)
def test_writer_round_trips_source_arrays_exactly(
    tmp_path: Path, shot_name: str
) -> None:
    source = CORPUS_ROOT / f"{shot_name}.parquet"
    if not source.is_file():
        pytest.skip("the pinned DIII-D challenge corpus is unavailable")
    output = tmp_path / f"{shot_name}.nc"
    write_thomson_ids(source, output)
    channels = read_thomson_channels(output)
    shot = load_shot(source)
    expected_names = []
    expected_positions = []
    expected_signals = []
    names = np.asarray(shot.chord_geometry["thomson_chord_name"], dtype=str)
    for profile_name in ("core", "edge"):
        profile = shot.thomson[profile_name]
        indices = _profile_indices(shot, profile_name)
        radius, height = _profile_geometry(shot, profile_name)
        for offset, source_index in enumerate(indices):
            expected_names.append(str(names[source_index]))
            expected_positions.append((radius[offset], height[offset]))
            expected_signals.append(
                (
                    profile.time_ms * 1.0e-3,
                    profile.density_m3[:, offset],
                    profile.temperature_ev[:, offset],
                )
            )
    assert [channel.name for channel in channels] == expected_names
    np.testing.assert_array_equal(
        [(channel.position_r_m, channel.position_z_m) for channel in channels],
        expected_positions,
    )
    for channel, (time_s, density, temperature) in zip(
        channels, expected_signals, strict=True
    ):
        assert channel.time_s.dtype == np.dtype(np.float64)
        assert channel.density_m3.dtype == np.dtype(np.float64)
        assert channel.temperature_ev.dtype == np.dtype(np.float64)
        np.testing.assert_array_equal(channel.time_s, time_s)
        np.testing.assert_array_equal(channel.density_m3, density)
        np.testing.assert_array_equal(channel.temperature_ev, temperature)
        assert channel.position_phi_rad == DIIID_THOMSON_TOROIDAL_ANGLE_RAD


def test_vendored_artifacts_are_native_valid_entries() -> None:
    report = json.loads((ARTIFACT_ROOT / "validation.json").read_text())
    assert report["data_dictionary"] == DATA_DICTIONARY_VERSION
    assert report["data_dictionary_major"] == DATA_DICTIONARY_MAJOR
    assert isclose(report["toroidal_position"]["value_rad"], 2.0 * pi / 3.0)
    assert [record["artifact"] for record in report["artifacts"]] == [
        f"{shot}.nc" for shot in GATE_SHOTS
    ]
    expected_paths = {
        "ids_properties/comment",
        "ids_properties/homogeneous_time",
        "ids_properties/version_put/data_dictionary",
        "ids_properties/version_put/access_layer",
        "ids_properties/version_put/access_layer_language",
        "channel/name",
        "channel/position/r",
        "channel/position/z",
        "channel/position/phi",
        "channel/n_e/data",
        "channel/n_e/time",
        "channel/t_e/data",
        "channel/t_e/time",
    }
    for record in report["artifacts"]:
        assert record["data_dictionary_major"] == DATA_DICTIONARY_MAJOR
        assert record["homogeneous_time"] == 0
        assert record["channel_count"] == 54
        assert record["family_channel_counts"] == {"core": 44, "tangential": 10}
        assert record["time_samples"] == EXPECTED_TIME_SAMPLES[record["artifact"]]
        assert record["round_trip_exact"] is True
        assert set(record["written_dd_paths"]) == expected_paths
        artifact = ARTIFACT_ROOT / record["artifact"]
        with imas.DBEntry(artifact, "r", dd_version=DATA_DICTIONARY_VERSION) as entry:
            assert entry.list_all_occurrences("thomson_scattering") == [0]
            ids = entry.get("thomson_scattering", 0, autoconvert=False)
            ids.validate()
            assert str(ids.ids_properties.version_put.data_dictionary) == (
                DATA_DICTIONARY_VERSION
            )
            assert record["source_shot"] in str(ids.ids_properties.comment)
            assert REVISION in str(ids.ids_properties.comment)
            assert len(ids.channel) == 54
            assert len(ids.laser) == 0
            assert not ids.channel[0].line_of_sight.has_value
            assert not ids.channel[0].position.r_error_lower.has_value
            assert not ids.channel[0].position.r_error_upper.has_value


def test_vendored_report_declares_unsupplied_geometry_absent() -> None:
    report = json.loads((ARTIFACT_ROOT / "validation.json").read_text())
    absent = {item["quantity"] for item in report["declared_absent"]}
    assert "channel line_of_sight first_point and second_point" in absent
    assert "laser start_point and end_point" in absent
    assert "scattering-volume position uncertainty" in absent
