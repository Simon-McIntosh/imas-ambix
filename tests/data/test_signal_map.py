"""Distilled signal-map validation, versioning, and compilation."""

from __future__ import annotations

import json

import numpy as np
import pytest

from imas_ambix.data.signal_map import (
    MAP_SCHEMA_VERSION,
    BlockedSignal,
    CalibrationRule,
    SignalMap,
    SignalMapError,
    SignalRule,
    load_packaged_signal_map,
    load_signal_map,
)


def _signal(**overrides) -> SignalRule:
    values = {
        "semantic_id": "plasma_current",
        "source_group": "amc",
        "source_array": "plasma_current",
        "source_unit": "kA",
        "target_path": "magnetics/ip/data",
        "target_unit": "A",
        "target_index": 0,
        "transformation": "ip_like",
        "source_cocos": 3,
        "unit_factor": 1000.0,
        "channel_factor": 1.0,
        "standard_name": None,
        "evidence": "imas-codex receipt sha256:source",
    }
    values.update(overrides)
    return SignalRule(**values)


def _map(
    *,
    signals=(),
    calibrations=(),
    blocked=(),
    target_dd_version="4.1.1",
    target_cocos=17,
) -> SignalMap:
    return SignalMap.create(
        schema_version=MAP_SCHEMA_VERSION,
        set_version="0.1.0",
        machine="mast",
        system="magnetics",
        source_dataset="fair-mast-level1",
        target_dd_version=target_dd_version,
        target_cocos=target_cocos,
        discovery_producer="imas-codex",
        discovery_receipt="sha256:discovery",
        signals=signals or (_signal(),),
        calibrations=calibrations,
        blocked=blocked,
    )


def test_standard_name_is_optional_while_dd_target_identity_is_required():
    source_map = _map()
    assert source_map.signals[0].standard_name is None
    assert source_map.signals[0].target_key == ("magnetics/ip/data", 0)


def test_map_has_a_semantic_release_and_an_exact_content_digest():
    first = _map(
        signals=(
            _signal(semantic_id="z", source_array="z", target_index=1),
            _signal(semantic_id="a", source_array="a", target_index=0),
        )
    )
    second = _map(signals=tuple(reversed(first.signals)))
    assert first.set_version == "0.1.0"
    assert first.digest == second.digest
    assert len(first.digest) == 64


def test_compilation_fuses_unit_cocos_and_shot_calibration():
    calibration = CalibrationRule(
        semantic_id="plasma_current",
        first_shot=20000,
        last_shot=20999,
        scale=0.5,
        offset=2.0,
        evidence="measured acquisition range",
    )
    compiled = _map(calibrations=(calibration,)).compile(20500)
    assert compiled["plasma_current"].scale == 500.0
    assert compiled.apply("plasma_current", np.array([1.0, 2.0])).tolist() == [
        502.0,
        1002.0,
    ]

    outside = _map(calibrations=(calibration,)).compile(21000)
    assert outside["plasma_current"].scale == 1000.0
    assert outside["plasma_current"].offset == 0.0


def test_packed_columns_use_one_vectorized_affine_operation():
    source_map = _map(
        signals=(
            _signal(semantic_id="ip", source_array="ip", target_index=0),
            _signal(
                semantic_id="psi",
                source_array="psi",
                source_unit="Wb/rad",
                target_path="equilibrium/time_slice/profiles_2d/psi",
                target_unit="Wb",
                target_index=None,
                transformation="psi_like",
                unit_factor=1.0,
            ),
        )
    ).compile(30420)
    values = np.array([[1.0, 1.0], [2.0, 2.0]])
    converted = source_map.apply_columns(values, ("ip", "psi"))
    assert converted[:, 0] == pytest.approx([1000.0, 2000.0])
    assert converted[:, 1] == pytest.approx([2.0 * np.pi, 4.0 * np.pi])


def test_overlapping_calibration_intervals_are_rejected():
    rows = (
        CalibrationRule("plasma_current", 100, 200, 1.0, 0.0, "first"),
        CalibrationRule("plasma_current", 200, 300, 2.0, 0.0, "second"),
    )
    with pytest.raises(SignalMapError, match="overlap"):
        _map(calibrations=rows)


def test_a_source_array_cannot_be_served_and_blocked():
    blocked = BlockedSignal(
        source_group="amc",
        source_array="plasma_current",
        reason="ambiguous scale",
        unmet="reviewed calibration evidence",
    )
    with pytest.raises(SignalMapError, match="both served and blocked"):
        _map(blocked=(blocked,))


@pytest.mark.parametrize(
    ("dd_version", "cocos"),
    [("3.40.0", 17), ("4.1.1", 11)],
)
def test_noncanonical_targets_are_rejected(dd_version: str, cocos: int):
    with pytest.raises(SignalMapError):
        _map(target_dd_version=dd_version, target_cocos=cocos)


def test_canonical_json_round_trip_loads_without_reading_source_data(tmp_path):
    source_map = _map()
    path = tmp_path / "magnetics.json"
    path.write_bytes(source_map.canonical_bytes())
    loaded = load_signal_map(path)
    assert loaded.digest == source_map.digest
    assert json.loads(path.read_text())["discovery"]["producer"] == "imas-codex"


def test_packaged_mast_angle_map_serves_ddv4_radians_without_a_standard_name():
    source_map = load_packaged_signal_map("mast", "magnetics")
    signal = source_map.compile(30420)["poloidal_field_probe_directed_angle"]

    assert source_map.set_version == "0.1.0"
    assert signal.rule.standard_name is None
    assert signal.rule.target_key == (
        "magnetics/b_field_pol_probe/poloidal_angle",
        None,
    )
    assert signal.apply(np.array([0.0, 90.0])).tolist() == pytest.approx(
        [0.0, -np.pi / 2.0]
    )
