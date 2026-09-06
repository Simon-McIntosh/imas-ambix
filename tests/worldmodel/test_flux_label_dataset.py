from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import zarr

from imas_ambix.data.stream_encode import REGISTRY_OFFSET
from imas_ambix.worldmodel.flux_label_dataset import (
    DEFAULT_COHORT_REPORT,
    DEFAULT_SESSION_ROOT,
    EXPECTED_CARRIER_IDENTITY,
    EXPECTED_POLICY_DIGEST,
    FluxLabelDataset,
)

SHOT = 12345


def _surface_geometry(times: np.ndarray) -> xr.Dataset:
    levels = np.linspace(0.0, 1.0, 11)
    angles = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    radii = np.sqrt(levels)[:, None, None]
    phase = angles[None, :, None]
    centres = np.linspace(0.78, 0.82, times.size)[None, None, :]
    surface_r = centres + 0.34 * radii * np.cos(phase)
    surface_z = np.broadcast_to(
        0.02 + 0.68 * radii * np.sin(phase), surface_r.shape
    ).copy()
    return xr.Dataset(
        data_vars={
            "flux_surface_psi_norm": (("surface",), levels),
            "flux_surface_r": (("surface", "angle", "time"), surface_r),
            "flux_surface_z": (("surface", "angle", "time"), surface_z),
            "magnetic_axis_r": (("time",), centres.reshape(-1)),
            "magnetic_axis_z": (("time",), np.full(times.size, 0.02)),
            "x_point_r": (
                ("x_slot", "time"),
                np.repeat([[0.72], [0.91]], times.size, axis=1),
            ),
            "x_point_z": (
                ("x_slot", "time"),
                np.repeat([[-0.62], [0.61]], times.size, axis=1),
            ),
            "finite_mask": (
                ("component", "time"),
                np.repeat(
                    np.asarray([[True], [True], [True], [False], [False], [True]]),
                    times.size,
                    axis=1,
                ),
            ),
            "diverted": (("time",), np.ones(times.size, dtype=bool)),
            "elongation": (("time",), np.full(times.size, 1.9)),
            "delta_upper": (("time",), np.full(times.size, 0.23)),
            "delta_lower": (("time",), np.full(times.size, 0.19)),
            "R_major": (("time",), centres.reshape(-1)),
            "a_minor": (("time",), np.full(times.size, 0.34)),
        },
        coords={"time": times},
    )


def _write_synthetic_session(tmp_path: Path) -> tuple[Path, Path, Path]:
    session_root = tmp_path / "sessions"
    token_root = tmp_path / "tokens"
    level1_root = tmp_path / "level1"
    (session_root / ".cards").mkdir(parents=True)
    level1_root.mkdir()

    ranked = [*range(12001, 12020), SHOT]
    for card, card_shots in enumerate((ranked[0::3], ranked[1::3], ranked[2::3])):
        (session_root / ".cards" / f"card-{card}.txt").write_text(
            "".join(f"{shot}\n" for shot in card_shots), encoding="utf-8"
        )

    slice_times = np.asarray([0.041, 0.060, 0.105], dtype=np.float64)
    _surface_geometry(slice_times).to_netcdf(
        session_root / f"{SHOT}.nc", group="steering", engine="h5netcdf"
    )
    np.savez(
        session_root / f"{SHOT}.npz",
        row=np.arange(3, dtype=np.int32),
        time=slice_times,
        conditioned=np.asarray([True, False, False]),
    )
    manifest = {
        "schema": "nova-forward-labeller-shot",
        "shot": SHOT,
        "status": "complete",
        "policy_digest": EXPECTED_POLICY_DIGEST,
        "carrier_identity": EXPECTED_CARRIER_IDENTITY,
        "nova_revision": "moving-writer-revision-is-not-a-corpus-pin",
        "slices": [
            {
                "row": 0,
                "time": float(slice_times[0]),
                "written": True,
                "converged": True,
                "qualified": False,
                "conditioned": False,
            },
            {
                "row": 1,
                "time": float(slice_times[1]),
                "written": True,
                "converged": False,
                "qualified": True,
                "conditioned": True,
            },
            {
                "row": 2,
                "time": float(slice_times[2]),
                "written": True,
                "converged": True,
                "qualified": False,
                "conditioned": True,
            },
        ],
    }
    (session_root / f"{SHOT}.manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    (session_root / "99999.manifest.json").write_text(
        json.dumps(
            {
                "shot": 99999,
                "status": "writing",
                "policy_digest": "not-yet-authoritative",
                "carrier_identity": "not-yet-authoritative",
            }
        ),
        encoding="utf-8",
    )

    token_path = token_root / "v1" / "frames" / str(SHOT) / "rbb.zarr"
    token_store = zarr.open_group(str(token_path), mode="w")
    tokens = np.empty((10, 16, 16), dtype=np.int32)
    for frame in range(tokens.shape[0]):
        tokens[frame].fill(REGISTRY_OFFSET + frame)
    token_store.create_array("tokens", data=tokens)

    frame_times = np.arange(10, dtype=np.float64) * 0.01
    level1_store = zarr.open_group(str(level1_root / f"{SHOT}.zarr"), mode="w")
    camera = level1_store.create_group("rbb")
    camera.create_array("time", data=frame_times)
    return session_root, token_root, level1_root


def test_complete_session_pairs_geometry_and_native_token_history(
    tmp_path: Path,
) -> None:
    session_root, token_root, level1_root = _write_synthetic_session(tmp_path)
    dataset = FluxLabelDataset(
        session_root,
        split="validation",
        token_root=token_root,
        level1_root=level1_root,
        cohort_shots=set(),
    )

    assert len(dataset) == 1
    item = dataset[0]
    assert item["shot_id"] == SHOT
    assert item["rank"] == 20
    assert item["split"] == "validation"
    assert item["conditioned"] is True
    assert item["frame_delta_s"] == pytest.approx(-0.001)
    assert item["conditioning"].shape == (6, 64, 64)
    assert np.isfinite(item["conditioning"]).all()
    assert item["geometry"].shape == (12,)
    assert np.isfinite(item["geometry"]).all()
    assert item["target_tokens"].shape == (16, 16)
    assert np.issubdtype(item["target_tokens"].dtype, np.integer)
    assert np.all(item["target_tokens"] == 4)
    assert item["history_tokens"].shape == (4, 16, 16)
    np.testing.assert_array_equal(item["history_tokens"][:, 0, 0], np.arange(4))
    assert not any("current" in key for key in item)

    receipt = dataset.receipt
    assert receipt["counts"]["manifest_files"] == 2
    assert receipt["counts"]["complete_sessions"] == 1
    assert receipt["counts"]["paired_slices"] == 1
    assert receipt["counts"]["conditioned_slices"] == 1
    assert receipt["counts"]["cohort_overlap"] == 0
    assert receipt["dropped_slices"]["unconverged"] == 1
    assert receipt["dropped_slices"]["outside_time_tolerance"] == 1
    assert receipt["max_abs_delta_t_s"] == pytest.approx(0.001)
    assert receipt["pins"] == {
        "policy_digest": EXPECTED_POLICY_DIGEST,
        "carrier_identity": EXPECTED_CARRIER_IDENTITY,
    }


def test_corpus_pins_refuse_mismatch_and_cohort_filter_is_whole_shot(
    tmp_path: Path,
) -> None:
    session_root, token_root, level1_root = _write_synthetic_session(tmp_path)
    excluded = FluxLabelDataset(
        session_root,
        split="all",
        token_root=token_root,
        level1_root=level1_root,
        cohort_shots={SHOT},
    )
    assert len(excluded) == 0
    assert excluded.receipt["counts"]["cohort_shots_excluded"] == 1
    assert excluded.receipt["dropped_slices"]["cohort_shot"] == 2
    assert excluded.receipt["counts"]["cohort_overlap"] == 0

    manifest_path = session_root / f"{SHOT}.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["policy_digest"] = "different"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="policy_digest"):
        FluxLabelDataset(
            session_root,
            split="all",
            token_root=token_root,
            level1_root=level1_root,
            cohort_shots=set(),
        )


def test_real_shot_pairing_receipt_and_item_contract() -> None:
    required = (
        DEFAULT_SESSION_ROOT / "21858.manifest.json",
        DEFAULT_SESSION_ROOT / "21858.nc",
        DEFAULT_SESSION_ROOT / "21858.npz",
        DEFAULT_COHORT_REPORT,
    )
    if not all(path.exists() for path in required):
        pytest.skip("real nova session evidence is unavailable on this host")

    dataset = FluxLabelDataset(shot_ids=[21858], split="train")
    receipt = dataset.receipt
    dropped = receipt["dropped_slices"]
    print(
        "real shot 21858: "
        f"pairs={receipt['counts']['paired_slices']} "
        f"dropped={sum(dropped.values())} "
        f"max_abs_delta_t_s={receipt['max_abs_delta_t_s']:.9g}"
    )

    assert len(dataset) == 85
    assert receipt["counts"]["cohort_overlap"] == 0
    assert receipt["counts"]["conditioned_slices"] == 16
    assert dropped["outside_time_tolerance"] == 16
    assert receipt["max_abs_delta_t_s"] <= 0.0025

    item = dataset[0]
    assert item["conditioning"].shape == (6, 64, 64)
    assert np.isfinite(item["conditioning"]).all()
    assert item["geometry"].shape == (12,)
    assert item["target_tokens"].shape == (16, 16)
    assert np.issubdtype(item["target_tokens"].dtype, np.integer)
    assert item["history_tokens"].shape == (4, 16, 16)
    assert not any("current" in key for key in item)
